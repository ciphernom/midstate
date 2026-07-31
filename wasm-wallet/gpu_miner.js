// gpu_miner.js — WebGPU nonce search for the browser, running the *same* kernel
// as the native miner.
//
// # Reasoning
//
// The CPU path (`miner.js` → `search_nonces`) gets 4 SIMD lanes under wasm32
// simd128 against 8 on native AVX2/NEON, and each nonce is a 1,000,000-round
// sequential BLAKE3 chain that no amount of lane width shortens. A GPU cannot
// shorten the chain either — but it can run thousands of chains at once, which is
// why `core/gpu_mining.rs` exists natively and why it is worth having here.
//
// The kernel is fetched from `pow.wgsl`, the identical file `gpu_mining.rs` pulls
// in with `include_str!`. That file's own header declares itself
// CONSENSUS-CRITICAL and requires bit-identical output to `create_extension`.
// Hand-transcribing it into a JS string would be the obvious way to build this and
// the wrong one: the failure mode is plausible-looking hashes that every peer
// rejects, which is close to un-debuggable from the outside.
//
// # Why WebGPU and not WebGL
// WebGL2 has no compute shaders at all. There is no WebGL route to this.
//
// # Why this can live in a Worker
// `navigator.gpu` IS exposed in `WorkerGlobalScope`, unlike `RTCPeerConnection`
// (which is what forced the pool client onto the main thread). So the GPU miner
// runs beside the mining loop rather than being marshalled across postMessage.
//
// # Safety / Invariants
//
//   - **The kernel is never trusted.** It only *surfaces candidate nonces*. Every
//     candidate is re-hashed with the WASM `build_solo_extension` and re-compared
//     against the target before it leaves this module — the same safety net
//     `mine_gpu` applies on the native side. A wrong driver can cost throughput;
//     it cannot produce an invalid share.
//   - **`selfTest()` is a hard gate.** Mining refuses to start until the GPU has
//     reproduced the WASM reference bit-for-bit over a full 1e6-iteration chain.
//   - **Dispatches are checkpointed.** Each nonce's 32-byte chaining state lives
//     in a GPU buffer and advances `itersPerDispatch` steps per dispatch. A kernel
//     that ran a whole chain in one launch would trip the browser's GPU watchdog
//     and be killed — the same constraint TDR imposes natively.

const IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
            0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

/** Mirrors `MAX_WINNERS` / `WINNERS_BYTES` in gpu_mining.rs. */
const MAX_WINNERS   = 256;
const WINNERS_BYTES = 16 + MAX_WINNERS * 4 * 3;
/** Mirrors `Params` (std430, 128 bytes): 8+8+8 words + 8 scalars. */
const PARAMS_WORDS  = 32;
const SELFTEST_N    = 8;
/** Mirrors `EXTENSION_ITERATIONS`. */
const EXTENSION_ITERATIONS = 1_000_000;

/** 32 hex chars → 8 u32 little-endian words (matches `words_le` in Rust). */
function wordsLE(hex) {
    const w = new Uint32Array(8);
    for (let i = 0; i < 8; i++) {
        const b = hex.slice(i * 8, i * 8 + 8);
        w[i] = (parseInt(b.slice(6, 8), 16) << 24 | parseInt(b.slice(4, 6), 16) << 16 |
                parseInt(b.slice(2, 4), 16) << 8  | parseInt(b.slice(0, 2), 16)) >>> 0;
    }
    return w;
}

/** 32 hex chars → 8 u32 big-endian words (matches `words_be` in Rust). */
function wordsBE(hex) {
    const w = new Uint32Array(8);
    for (let i = 0; i < 8; i++) w[i] = parseInt(hex.slice(i * 8, i * 8 + 8), 16) >>> 0;
    return w;
}

export class GpuMiner {
    constructor() {
        this.device = null;
        this.adapterName = 'unknown';
        this.batchNonces = 1 << 14;      // 16384 chains in flight
        this.itersPerDispatch = 2000;    // mirrors DEFAULT_ITERS_PER_DISPATCH
        this.verified = false;
    }

    /**
     * Probe support without initialising anything.
     *
     * Note `navigator.gpu` is only present in a *secure context* — HTTPS or
     * localhost. A wallet served over plain HTTP from a real host has no WebGPU
     * at all, and this returns false rather than throwing.
     */
    static available() {
        return typeof navigator !== 'undefined' && !!navigator.gpu;
    }

    /**
     * Acquire a device, compile the shared kernel, and self-test it.
     *
     * Returns false (never throws) on any failure, so the caller falls back to
     * the SIMD path exactly as the native miner falls back to `mine_extension`.
     */
    async init(wgslUrl = './pow.wgsl', log = () => {}) {
        if (!GpuMiner.available()) { log('WebGPU unavailable (needs HTTPS or localhost)'); return false; }
        try {
            const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
            if (!adapter) { log('No WebGPU adapter'); return false; }
            this.adapterName = adapter.info?.description || adapter.info?.vendor || 'WebGPU device';

            this.device = await adapter.requestDevice();
            // Surface validation errors instead of letting them tear the device
            // down silently mid-dispatch.
            this.device.addEventListener?.('uncapturederror', (e) =>
                log(`WebGPU error: ${e.error?.message || e}`));

            const wgsl = await (await fetch(wgslUrl, { cache: 'no-store' })).text();
            if (!wgsl.includes('fn k_step')) throw new Error(`${wgslUrl} does not look like the mining kernel`);

            const module = this.device.createShaderModule({ code: wgsl });
            const info = await module.getCompilationInfo?.();
            const errs = (info?.messages || []).filter(m => m.type === 'error');
            if (errs.length) throw new Error(`WGSL: ${errs.map(m => m.message).join('; ')}`);

            // A workgroup is 64 invocations; clamp so one dispatch never exceeds
            // the browser's per-dimension workgroup limit (usually 65535).
            const maxGroups = this.device.limits.maxComputeWorkgroupsPerDimension || 65535;
            this.batchNonces = Math.min(this.batchNonces, maxGroups * 64);

            const mk = (size, usage) => this.device.createBuffer({ size, usage });
            const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
            this.paramsBuf   = mk(PARAMS_WORDS * 4, S | CD);
            this.stateBuf    = mk(this.batchNonces * 32, S | CS);
            this.winnersBuf  = mk(WINNERS_BYTES, S | CD | CS);
            this.readbackBuf = this.device.createBuffer({
                size: WINNERS_BYTES, usage: GPUBufferUsage.MAP_READ | CD,
            });

            const layout = this.device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                ],
            });
            this.bindGroup = this.device.createBindGroup({
                layout,
                entries: [
                    { binding: 0, resource: { buffer: this.paramsBuf } },
                    { binding: 1, resource: { buffer: this.stateBuf } },
                    { binding: 2, resource: { buffer: this.winnersBuf } },
                ],
            });
            const pl = this.device.createPipelineLayout({ bindGroupLayouts: [layout] });
            const pipe = (entryPoint) => this.device.createComputePipeline({
                layout: pl, compute: { module, entryPoint },
            });
            this.pipeInit = pipe('k_init');
            this.pipeStep = pipe('k_step');
            this.pipeTest = pipe('k_test');

            log(`WebGPU device: ${this.adapterName} (${this.batchNonces} chains/batch)`);
            return true;
        } catch (e) {
            log(`WebGPU init failed: ${e.message}`);
            this.device = null;
            return false;
        }
    }

    _writeParams({ midstateHex, targetHex, poolHex, baseLo, baseHi, nNonces, iters }) {
        const p = new Uint32Array(PARAMS_WORDS);
        p.set(wordsLE(midstateHex), 0);
        p.set(wordsBE(targetHex), 8);
        if (poolHex) p.set(wordsBE(poolHex), 16);
        p[24] = baseLo >>> 0;
        p[25] = baseHi >>> 0;
        p[26] = nNonces >>> 0;
        p[27] = iters >>> 0;
        p[28] = poolHex ? 1 : 0;
        this.device.queue.writeBuffer(this.paramsBuf, 0, p);
    }

    _dispatch(pipeline, nNonces) {
        const enc = this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(nNonces / 64));
        pass.end();
        this.device.queue.submit([enc.finish()]);
    }

    async _readback(src, bytes) {
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(src, 0, this.readbackBuf, 0, bytes);
        this.device.queue.submit([enc.finish()]);
        await this.readbackBuf.mapAsync(GPUMapMode.READ, 0, bytes);
        const copy = new Uint8Array(this.readbackBuf.getMappedRange(0, bytes).slice(0));
        this.readbackBuf.unmap();
        return copy;
    }

    /**
     * Prove the GPU reproduces the WASM reference bit-for-bit over the full
     * 1,000,000-iteration chain.
     *
     * # Formal Specification
     * ```text
     * Pre:  init() succeeded; blake3Ref(midstateHex, nonce) computes the
     *       reference final hash (WASM build_solo_extension)
     * Post: returns true  ⇒ ∀ i ∈ 0..SELFTEST_N • gpu_state[i] = blake3Ref(m, base+i)
     *       returns false ⇒ this.verified stays false and mining must not start
     * ```
     *
     * # Safety / Invariants
     * - **This is the only defence against a driver that computes plausible but
     *   wrong hashes.** Such a driver would otherwise mine continuously and have
     *   every single share rejected, with nothing in the logs pointing at the GPU.
     * - Costs one full chain of latency (~one nonce), paid once per session.
     */
    async selfTest(blake3Ref, log = () => {}) {
        this.verified = false;
        if (!this.device) return false;
        try {
            const midstateHex = 'a5'.repeat(32);
            const params = {
                midstateHex,
                targetHex: '00'.repeat(32),   // unreachable: we want state, not winners
                poolHex: null, baseLo: 0, baseHi: 0,
                nNonces: SELFTEST_N, iters: 0,
            };
            this._writeParams(params);
            this._dispatch(this.pipeInit, SELFTEST_N);

            let done = 0;
            while (done < EXTENSION_ITERATIONS) {
                const step = Math.min(this.itersPerDispatch, EXTENSION_ITERATIONS - done);
                this._writeParams({ ...params, iters: step });
                this._dispatch(this.pipeStep, SELFTEST_N);
                done += step;
            }

            const raw = await this._readback(this.stateBuf, SELFTEST_N * 32);
            for (let i = 0; i < SELFTEST_N; i++) {
                const got = Array.from(raw.slice(i * 32, i * 32 + 32))
                    .map(b => b.toString(16).padStart(2, '0')).join('');
                const want = await blake3Ref(midstateHex, i);
                if (!want) throw new Error('reference hash unavailable');
                if (got !== want.toLowerCase()) {
                    log(`GPU self-test FAILED at nonce ${i}: gpu=${got.slice(0, 16)}… ref=${want.slice(0, 16)}…`);
                    return false;
                }
            }
            this.verified = true;
            log(`GPU self-test passed on ${this.adapterName} (${SELFTEST_N} nonces)`);
            return true;
        } catch (e) {
            log(`GPU self-test error: ${e.message}`);
            return false;
        }
    }

    /**
     * Search one batch. Resolves to an array of candidate nonces (as strings).
     *
     * Candidates are NOT verified here — the caller must re-hash each with the
     * WASM reference before treating it as a share. `shouldStop()` is polled
     * between dispatches, so a job rotation abandons the batch within roughly
     * `itersPerDispatch` steps rather than a full chain.
     */
    async searchBatch({ midstateHex, targetHex, poolHex, shouldStop, onProgress }) {
        if (!this.device || !this.verified) throw new Error('GPU miner not self-tested');

        const baseLo = (Math.random() * 0x100000000) >>> 0;
        const baseHi = (Math.random() * 0x100000000) >>> 0;
        const n = this.batchNonces;
        const params = { midstateHex, targetHex, poolHex, baseLo, baseHi, nNonces: n, iters: 0 };

        this.device.queue.writeBuffer(this.winnersBuf, 0, new Uint32Array([0, MAX_WINNERS, 0, 0]));
        this._writeParams(params);
        this._dispatch(this.pipeInit, n);

        let done = 0;
        while (done < EXTENSION_ITERATIONS) {
            if (shouldStop && shouldStop()) return [];
            const step = Math.min(this.itersPerDispatch, EXTENSION_ITERATIONS - done);
            this._writeParams({ ...params, iters: step });
            this._dispatch(this.pipeStep, n);
            // Yield so the event loop stays responsive and the watchdog sees gaps.
            await this.device.queue.onSubmittedWorkDone();
            done += step;
            if (onProgress) onProgress(step, n);
        }

        this._writeParams({ ...params, iters: 0 });
        this._dispatch(this.pipeTest, n);

        const raw = await this._readback(this.winnersBuf, WINNERS_BYTES);
        const dv = new DataView(raw.buffer);
        const count = Math.min(dv.getUint32(0, true), MAX_WINNERS);
        const out = [];
        for (let i = 0; i < count; i++) {
            const lo = BigInt(dv.getUint32(16 + i * 4, true));
            const hi = BigInt(dv.getUint32(16 + MAX_WINNERS * 4 + i * 4, true));
            out.push(((hi << 32n) | lo).toString());
        }
        return out;
    }

    destroy() {
        try { this.device?.destroy?.(); } catch (_) {}
        this.device = null;
        this.verified = false;
    }
}
