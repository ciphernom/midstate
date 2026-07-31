// miner.js — Dedicated mining worker (one instance per CPU core)
//
// Stateless: receives a template (midstate + target), searches nonces
// in a tight WASM SIMD loop, posts back hashrate and winning nonces.
// The main thread coordinates all miners and handles block submission
// via the wallet worker.

import init, { search_nonces } from './pkg/wasm_wallet.js';
import { GpuMiner } from './gpu_miner.js';

let mining = false;
let loopRunning = false; // guard against re-entrant START
let throttleMs = 400; // ms to sleep between search_nonces calls (default: Balanced)

// ─── GPU backend (optional) ──────────────────────────────────────────────────
//
// `navigator.gpu` is exposed in workers, so the GPU miner runs here beside the
// SIMD loop rather than being marshalled across postMessage. It replaces the
// SIMD search in this worker when — and only when — it has proven itself
// bit-identical to the WASM reference; otherwise this worker behaves exactly as
// before. That mirrors `Backend::Auto` on the native side.
let gpu = null;

// ─── Reference hashes for the GPU self-test ──────────────────────────────────
//
// `build_solo_extension` is a METHOD on the `WebWallet` instance, not a free
// wasm export — importing it by name here produced a module that could not link,
// which killed every miner worker at load with an opaque `[object Event]`.
//
// The instance lives in worker.js, so the reference hash is requested over
// postMessage: miner → index.html → worker.js → back. Slow (~200ms per nonce)
// but it runs 8 times, once per session, and it keeps the oracle as the exact
// implementation this wallet submits shares with rather than a second copy.

let _verifyNextId = 1;
const _verifyPending = new Map();

/** Resolves a pending GPU_VERIFY_RESULT relayed from the wallet worker. */
function _verifyReceive(id, hashes, error) {
    const p = _verifyPending.get(id);
    if (!p) return;
    _verifyPending.delete(id);
    if (error) p.reject(new Error(error));
    else p.resolve(hashes);
}

/**
 * Asks the wallet worker for the reference final hashes of `count` nonces
 * starting at 0, for the given midstate.
 */
function requestReferenceHashes(midstateHex, count) {
    return new Promise((resolve, reject) => {
        const id = _verifyNextId++;
        _verifyPending.set(id, { resolve, reject });
        self.postMessage({ type: 'GPU_VERIFY_REQUEST', payload: { id, midstate: midstateHex, count } });
        setTimeout(() => {
            if (_verifyPending.has(id)) {
                _verifyPending.delete(id);
                reject(new Error('reference hash request timed out'));
            }
        }, 60_000);
    });
}



self.onmessage = async (e) => {
    const { type, payload } = e.data;

    if (type === 'INIT') {
        try {
            await init();
            // GPU is opportunistic: only this worker (index 0) attempts it, since
            // a GPUDevice is not shareable and ten workers each grabbing one would
            // contend for the same card while starving the CPU search.
            if (payload?.use_gpu) {
                const log = (m) => self.postMessage({ type: 'LOG', payload: `[gpu] ${m}` });
                const g = new GpuMiner();
                if (await g.init('./pow.wgsl', log)) {
                    // Hard gate: no mining on an unverified kernel.
                    const refs = await requestReferenceHashes('a5'.repeat(32), 8)
                        .catch(e => { log(`reference hashes unavailable (${e.message}); GPU disabled`); return null; });
                    if (refs && await g.selfTest(async (_m, i) => refs[i], log)) {
                        gpu = g;
                        self.postMessage({ type: 'GPU_READY', payload: { adapter: g.adapterName } });
                    } else {
                        log('GPU excluded from mining (self-test failed); using CPU SIMD');
                        g.destroy();
                    }
                }
            }
            self.postMessage({ type: 'READY' });
        } catch (err) {
            self.postMessage({ type: 'ERROR', payload: `WASM init failed: ${err}` });
        }
    }

    else if (type === 'GPU_VERIFY_RESULT') {
        _verifyReceive(payload.id, payload.hashes, payload.error);
    }

    else if (type === 'START') {
        // Stop any existing loop before starting a new one
        mining = false;
        // Wait for the old loop to finish its current search_nonces() call.
        // Each call takes ~800ms at CHUNK=1, so poll until it exits.
        while (loopRunning) {
            await new Promise(r => setTimeout(r, 100));
        }

        mining = true;
        loopRunning = true;
        const { midstate, target } = payload;
        if (payload.throttle_ms !== undefined) throttleMs = payload.throttle_ms;

        // Pool mode: a hit against the pool's share target is a *share*, not a
        // block, and the template stays valid afterwards — so the worker must
        // keep grinding the same midstate from the next nonce. Halting (the solo
        // behaviour) is fatal here: the midstate doesn't change when a share is
        // found, so the main thread's "restart on new midstate" path never fires
        // and this worker would sit idle until the next network block.
        const continueOnFound = !!payload.continue_on_found;

        self.postMessage({ type: 'LOG', payload: `START received. midstate=${midstate?.substring(0,16)}… throttle=${throttleMs}ms` });

        if (!midstate || !target) {
            self.postMessage({ type: 'ERROR', payload: 'START missing midstate or target' });
            mining = false;
            loopRunning = false;
            return;
        }

// Each worker starts at a random nonce offset to avoid overlap.
        // Capped to JS MAX_SAFE_INTEGER to prevent JSON.parse corruption over the network.
        const MAX_SAFE = 9007199254740991; 
        const startRange = Math.floor(Math.random() * (MAX_SAFE - 1000000000));
        let nonce = BigInt(startRange);


        // CHUNK controls how many SIMD iterations per search_nonces() call.
        // Each iteration = 4 SIMD lanes = 4 nonces. This chain uses expensive
        // iterated hashing (EXTENSION_ITERATIONS), so each nonce takes ~200ms.
        // CHUNK=1 → 4 nonces → ~0.8s per call. This keeps the worker responsive
        // to STOP/THROTTLE messages and allows frequent hashrate reports.
        // (The original single-threaded miner used CHUNK=5.)
        const CHUNK = 1;
        const NONCES_PER_CHUNK = CHUNK * 4;

        let chunkCount = 0;
        let reportStart = Date.now();

        // ── GPU path ──
        //
        // Takes over this worker entirely when a self-tested device is present:
        // running both would have the SIMD loop competing with the GPU for the
        // same event loop that has to service its dispatch checkpoints.
        //
        // Candidates from the kernel are re-hashed here with the WASM reference
        // before being reported, so a driver that returns garbage costs a batch
        // rather than producing a rejected share. This is the browser counterpart
        // of the `create_extension` re-check in `mine_gpu`.
        if (gpu && gpu.verified) {
            try {
                while (mining) {
                    const candidates = await gpu.searchBatch({
                        midstateHex: midstate,
                        targetHex: target,
                        poolHex: null,
                        shouldStop: () => !mining,
                        onProgress: (steps, lanes) => {
                            // One "nonce" is a full EXTENSION_ITERATIONS chain, so
                            // progress is fractional: (steps/1e6) chains per lane.
                            const now = Date.now();
                            if (now - reportStart >= 1000) {
                                const nps = Math.floor((steps / 1_000_000) * lanes / ((now - reportStart) / 1000));
                                self.postMessage({ type: 'HASHRATE', payload: { nps, nonces: 0 } });
                                reportStart = now;
                            }
                        },
                    });

                    // Candidates are reported as-is. Re-verification still happens,
                    // just one layer up where the WebWallet instance lives: solo
                    // submission rebuilds the extension with
                    // `wallet.build_solo_extension` before sending, and the pool
                    // recomputes `create_extension` server-side before crediting.
                    // So a diverging device costs a rejected submission, never an
                    // invalid block — and `selfTest` has already gated gross
                    // breakage at startup.
                    let stop = false;
                    for (const nonceStr of candidates) {
                        self.postMessage({ type: 'FOUND', payload: { nonce: nonceStr } });
                        if (!continueOnFound) { stop = true; break; }
                    }
                    if (stop) break;
                }
            } catch (err) {
                self.postMessage({ type: 'LOG', payload: `[gpu] search failed, falling back to CPU: ${err.message}` });
                gpu.destroy();
                gpu = null;
                // fall through to the SIMD loop below
            }
            if (!gpu) {
                self.postMessage({ type: 'LOG', payload: '[gpu] reverted to CPU SIMD for this worker' });
            } else {
                mining = false;
                loopRunning = false;
                return;
            }
        }

        try {
            while (mining) {
                const result = search_nonces(midstate, target, nonce, CHUNK);

                chunkCount++;
                nonce += BigInt(NONCES_PER_CHUNK);

                if (result !== undefined && result !== null) {
                    self.postMessage({ type: 'FOUND', payload: { nonce: result.toString() } });
                    if (!continueOnFound) {
                        mining = false;
                        loopRunning = false;
                        return;
                    }
                    // `nonce` was already advanced past this chunk above, so the
                    // search resumes cleanly without re-finding the same share.
                }

                // Report hashrate every ~1 second
                const now = Date.now();
                const elapsed = now - reportStart;
                if (elapsed >= 1000) {
                    const totalNonces = chunkCount * NONCES_PER_CHUNK;
                    const nps = Math.floor(totalNonces / (elapsed / 1000));
                    self.postMessage({ type: 'HASHRATE', payload: { nps, nonces: totalNonces } });
                    chunkCount = 0;
                    reportStart = now;
                }

                // Yield after every call to stay responsive to STOP/THROTTLE.
                // Each search_nonces(1) already takes ~0.8s, so the throttle
                // sleep on top of that is the total pause between chunks.
                await new Promise(r => setTimeout(r, throttleMs));
            }
        } catch (err) {
            self.postMessage({ type: 'ERROR', payload: `Mining loop error: ${err}` });
        }

        loopRunning = false;
    }

    else if (type === 'STOP') {
        mining = false;
    }

    else if (type === 'THROTTLE') {
        if (payload && payload.throttle_ms !== undefined) {
            throttleMs = payload.throttle_ms;
        }
    }
};
