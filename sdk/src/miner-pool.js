// miner-pool.js — Multi-worker nonce search.
//
// # Reasoning
//
// Each nonce is a fixed amount of sequential work — a 1,000,000-iteration BLAKE3
// chain (~130 ms) below height 420,000, one Midstate Walk (a few ms) from then
// on — and no amount of batching or scheduling improves a single thread. The
// work is embarrassingly parallel though — nonces are independent — so cores
// are the only lever. Workers are told which proof of work to run with each
// template (`pow`), and a change of `pow` restarts them like a change of
// midstate does.
//
// The partition must satisfy two constraints at once. A search call tests four
// lanes from a contiguous base, so a worker's own nonces must stay
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

import { BLAKE3_CHAIN } from './pow.js';

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
        /** Numbers each searchTemplate; workers echo it on every find. */
        this._job = 0;
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
            // A find posted before the workers saw a template change belongs to
            // a template nobody is mining any more.
            if (!this._current || m.job !== this._current.job) return;
            this._found.push(BigInt(m.nonce));
            if (this._onFound) this._onFound(BigInt(m.nonce));
        } else if (m.type === 'PROGRESS') {
            this._noncesTested += m.nonces;
        } else if (m.type === 'ERROR') {
            if (this._onError) this._onError(new Error(`worker ${index}: ${m.error}`));
        }
    }

    /** Point every worker at a (midstate, target, pow), from a fresh random base. */
    searchTemplate(midstate, target, pow = BLAKE3_CHAIN) {
        if (!this.running) throw new Error('MinerPool is not started.');
        this._found = [];
        const base = Math.floor(Math.random() * (Number.MAX_SAFE_INTEGER - 1_000_000_000));
        const job = ++this._job;
        this._current = { midstate, target, pow, base, job };
        for (let i = 0; i < this.workers.length; i++) {
            this.workers[i].postMessage({
                type: 'START',
                midstate, target, pow, base, job,
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
     * `nonce` and `iterations` arguments are ignored because the pool owns the
     * partition — letting the caller also advance a cursor would double-count
     * and create overlap. Its fifth argument, `pow`, is honoured.
     */
    get searchFn() {
        return (midstate, target, _nonce, _iterations, pow = BLAKE3_CHAIN) => {
            const c = this._current;
            if (!c || c.midstate !== midstate || c.target !== target || c.pow !== pow) {
                this.stopSearch();
                this.searchTemplate(midstate, target, pow);
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
