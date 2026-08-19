// miner-pool.js — Multi-worker nonce search.
//
// # Reasoning
//
// Mining here is a VDF: one nonce is a 1,000,000-iteration sequential BLAKE3
// chain costing ~130 ms, and no amount of batching or scheduling improves a
// single thread. The work is embarrassingly parallel though — nonces are
// independent — so cores are the only lever, and an 8-core machine really is
// ~8× a one-core one.
//
// The partition must satisfy two constraints at once. `search_nonces` tests four
// SIMD lanes from a contiguous base, so a worker's own nonces must stay
// contiguous within a call; and no two workers may ever test the same value, or
// the extra cores buy nothing. Striding by `iterations × 4 × workerCount`
// satisfies both: worker `k` takes the k-th block of four and then skips the
// blocks belonging to its peers.
//
// This class deliberately exposes the same `searchFn`-shaped surface the
// single-threaded `Miner` already consumes, so pool/solo logic, template
// lifecycle and submission are unchanged by turning it on.

import { Worker } from 'node:worker_threads';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import os from 'node:os';

const WORKER_URL = new URL('./miner-worker.js', import.meta.url);

/**
 * A pool of nonce-search workers.
 *
 * @example
 * const pool = new MinerPool({ workers: 8 });
 * await pool.start();
 * const miner = new Miner(wallet, client, { searchFn: pool.searchFn });
 */
export class MinerPool {
    /**
     * @param {Object} [opts]
     * @param {number} [opts.workers]      Worker count. Defaults to cores − 1,
     *   leaving one for the coordinator and the event loop.
     * @param {number} [opts.iterations=1] SIMD batches per search call (×4 nonces).
     * @param {Uint8Array} [opts.wasmBytes] The .wasm module. Read from `pkg/` if
     *   omitted. Passed explicitly because a worker has no `fetch` relative to a
     *   module URL under Node.
     */
    constructor(opts = {}) {
        this.workerCount = Math.max(1, opts.workers ?? defaultWorkerCount());
        this.iterations = opts.iterations ?? 1;
        this.wasmBytes = opts.wasmBytes ?? null;
        this.workers = [];
        this.running = false;

        this._found = [];
        this._noncesTested = 0;
        this._onFound = null;
        this._onError = null;
        this._current = null;
    }

    onFound(cb) { this._onFound = cb; return this; }
    onError(cb) { this._onError = cb; return this; }

    /** Spawn and initialise the workers. */
    async start() {
        if (this.running) return;
        if (!this.wasmBytes) {
            const wasmPath = fileURLToPath(new URL('../pkg/wasm_wallet_bg.wasm', import.meta.url));
            this.wasmBytes = await readFile(wasmPath);
        }

        const ready = [];
        for (let i = 0; i < this.workerCount; i++) {
            const w = new Worker(WORKER_URL, { workerData: { wasmBytes: this.wasmBytes } });
            w.on('message', (m) => this._onMessage(i, m));
            w.on('error', (e) => { if (this._onError) this._onError(e); });
            this.workers.push(w);
            ready.push(new Promise((resolve, reject) => {
                const t = setTimeout(() => reject(new Error(`worker ${i} did not initialise`)), 60_000);
                const h = (m) => {
                    if (m.type === 'READY') { clearTimeout(t); w.off('message', h); resolve(); }
                    else if (m.type === 'ERROR') { clearTimeout(t); w.off('message', h); reject(new Error(m.error)); }
                };
                w.on('message', h);
            }));
            w.postMessage({ type: 'INIT' });
        }
        await Promise.all(ready);
        this.running = true;
    }

    _onMessage(index, m) {
        if (m.type === 'FOUND') {
            this._found.push(BigInt(m.nonce));
            if (this._onFound) this._onFound(BigInt(m.nonce));
        } else if (m.type === 'PROGRESS') {
            this._noncesTested += m.nonces;
        } else if (m.type === 'ERROR') {
            if (this._onError) this._onError(new Error(`worker ${index}: ${m.error}`));
        }
    }

    /** Point every worker at a (midstate, target), from a fresh random base. */
    searchTemplate(midstate, target) {
        if (!this.running) throw new Error('MinerPool is not started.');
        this._found = [];
        const base = Math.floor(Math.random() * (Number.MAX_SAFE_INTEGER - 1_000_000_000));
        this._current = { midstate, target, base };
        for (let i = 0; i < this.workers.length; i++) {
            this.workers[i].postMessage({
                type: 'START',
                midstate, target, base,
                index: i, count: this.workers.length,
                iterations: this.iterations,
            });
        }
    }

    /** Halt the search without tearing down the workers. */
    stopSearch() {
        for (const w of this.workers) w.postMessage({ type: 'STOP' });
        this._current = null;
    }

    /**
     * A `searchFn` for {@link Miner}.
     *
     * The single-threaded interface is synchronous and range-based; a worker
     * pool is neither. The adaptation is to treat the pool as a background
     * producer: the first call for a given (midstate, target) starts every
     * worker, and each call drains whatever has been found since. `Miner`'s
     * `nonce` argument is ignored because the pool owns the partition — letting
     * the caller also advance a cursor would double-count and create overlap.
     */
    get searchFn() {
        return (midstate, target) => {
            const c = this._current;
            if (!c || c.midstate !== midstate || c.target !== target) {
                this.stopSearch();
                this.searchTemplate(midstate, target);
                return undefined;
            }
            return this._found.length > 0 ? this._found.shift() : undefined;
        };
    }

    /** Nonces tested across all workers since start. */
    get noncesTested() { return this._noncesTested; }

    /** Terminate every worker. */
    async stop() {
        this.running = false;
        this.stopSearch();
        await Promise.all(this.workers.map((w) => w.terminate().catch(() => {})));
        this.workers = [];
    }
}

function defaultWorkerCount() {
    try {
        if (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) {
            return Math.max(1, navigator.hardwareConcurrency - 1);
        }
    } catch { /* fall through */ }
    try {
        // eslint-disable-next-line no-undef
        return Math.max(1, os.cpus().length - 1);
    } catch { /* fall through */ }
    return 1;
}
