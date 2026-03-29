// miner.js — Dedicated mining worker (one instance per CPU core)
//
// Stateless: receives a template (midstate + target), searches nonces
// in a tight WASM SIMD loop, posts back hashrate and winning nonces.
// The main thread coordinates all miners and handles block submission
// via the wallet worker.

import init, { search_nonces } from './pkg/wasm_wallet.js';

let mining = false;
let loopRunning = false; // guard against re-entrant START
let throttleMs = 400; // ms to sleep between search_nonces calls (default: Balanced)

self.onmessage = async (e) => {
    const { type, payload } = e.data;

    if (type === 'INIT') {
        try {
            await init();
            self.postMessage({ type: 'READY' });
        } catch (err) {
            self.postMessage({ type: 'ERROR', payload: `WASM init failed: ${err}` });
        }
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

        try {
            while (mining) {
                const result = search_nonces(midstate, target, nonce, CHUNK);

                chunkCount++;
                nonce += BigInt(NONCES_PER_CHUNK);

                if (result !== undefined && result !== null) {
                    self.postMessage({ type: 'FOUND', payload: { nonce: result.toString() } });
                    mining = false;
                    loopRunning = false;
                    return;
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
