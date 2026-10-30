// miner-worker.js — One nonce-search worker.
//
// Runs under Node `worker_threads` and, unchanged, under a browser `Worker`.
// The only environment-specific part is how messages arrive, which is resolved
// once at load.
//
// # Reasoning
//
// The proof of work is the whole cost: one nonce is a 1,000,000-iteration
// sequential BLAKE3 chain (~130 ms) below height 420,000, or one Midstate Walk
// (a few ms) from then on, and nothing about batching or scheduling changes
// that. Parallelism is therefore the only lever, and it is embarrassingly
// parallel — each nonce is independent — so the design is simply "partition the
// nonce space and never let two workers test the same value". START names the
// proof of work (`pow`); a START without one means the BLAKE3 chain.
//
// Each worker is handed a disjoint stride: worker `k` of `n` tests
// `base + k*4, base + (k+n)*4, …`, so the four SIMD lanes inside one
// `search_nonces` call stay contiguous (which the SIMD implementation requires)
// while the workers never overlap. Random per-session bases mean two independent
// miners on the same template do not retrace each other either.
//
// The loop yields between search calls so a STOP is honoured promptly rather
// than after a full batch, which matters because a batch is ~half a second and a
// template rotation should not wait on it.

import init, * as wasm from '../pkg/wasm_wallet.js';
import { BLAKE3_CHAIN, searchFor } from './pow.js';

let mining = false;
let ready = false;

/**
 * Bumped by every START. A loop runs only while its own generation is current.
 * The coordinator changes template with STOP and an immediate START, and both
 * are handled while the old loop is parked between search calls; with `mining`
 * alone, the old loop woke to find it true again and carried on beside the new
 * one, so every template change left one more loop grinding a dead template.
 */
let generation = 0;

/** Resolved once: node:worker_threads or a browser Worker global. */
let post, listen;

async function bindTransport() {
    if (typeof self !== 'undefined' && typeof self.postMessage === 'function' && typeof window === 'undefined') {
        post = (m) => self.postMessage(m);
        listen = (cb) => { self.onmessage = (e) => cb(e.data); };
        return null;
    }
    const { parentPort, workerData } = await import('node:worker_threads');
    post = (m) => parentPort.postMessage(m);
    listen = (cb) => parentPort.on('message', cb);
    return workerData;
}

const workerData = await bindTransport();

/**
 * Initialise WASM.
 *
 * The bytes are passed in rather than fetched: a Node worker has no `fetch`
 * relative to a module URL, and requiring each worker to locate the .wasm itself
 * is how the browser miner ended up failing at load with an opaque
 * `[object Event]`.
 */
async function ensureInit(wasmBytes) {
    if (ready) return;
    await init(wasmBytes ? { module_or_path: wasmBytes } : undefined);
    ready = true;
}

async function run({ midstate, target, base, index, count, iterations, pow = BLAKE3_CHAIN, job }, gen) {
    const search = searchFor(wasm, pow);
    const perCall = BigInt(iterations * 4);
    const stride = perCall * BigInt(count);
    let nonce = BigInt(base) + perCall * BigInt(index);

    let since = 0;
    let reportAt = Date.now();

    while (mining && gen === generation) {
        const found = search(midstate, target, nonce, iterations);
        nonce += stride;
        since += iterations * 4;

        if (found !== undefined && found !== null) {
            // `job` lets the coordinator drop a find that crossed a template
            // change in flight.
            post({ type: 'FOUND', nonce: found.toString(), job });
            // Whether to stop is the coordinator's decision: a solo hit ends the
            // job, a pool share does not. It answers by sending STOP or not.
        }

        const elapsed = Date.now() - reportAt;
        if (elapsed >= 1000) {
            post({ type: 'PROGRESS', nonces: since, ms: elapsed, job });
            since = 0;
            reportAt = Date.now();
        }

        // Yield so STOP is seen between batches rather than after one.
        await new Promise((r) => setTimeout(r, 0));
    }
    post({ type: 'STOPPED', job });
}

listen(async (msg) => {
    try {
        if (msg.type === 'INIT') {
            await ensureInit(msg.wasmBytes ?? workerData?.wasmBytes);
            post({ type: 'READY' });
        } else if (msg.type === 'START') {
            if (mining) return;
            await ensureInit(msg.wasmBytes ?? workerData?.wasmBytes);
            mining = true;
            const gen = ++generation;
            run(msg, gen).catch((e) => {
                // An unknown or unsupported proof of work ends this search; the
                // worker stays alive for the next START.
                if (gen === generation) mining = false;
                post({ type: 'ERROR', error: String(e?.message || e), job: msg.job });
            });
        } else if (msg.type === 'STOP') {
            mining = false;
        }
    } catch (e) {
        post({ type: 'ERROR', error: String(e?.message || e) });
    }
});
