// miner.js — Solo block mining for the Midstate SDK.
//
// ═══════════════════════════════════════════════════════════════════════════
//  Reasoning (module level)
// ═══════════════════════════════════════════════════════════════════════════
//
// Mining on this chain is a Verifiable Delay Function, not a throughput race.
// `create_extension` hashes a (midstate ⌢ nonce) seed and then iterates BLAKE3
// EXTENSION_ITERATIONS (1,000,000) times sequentially. One nonce costs ~130 ms
// of unavoidable serial work; SIMD gives four lanes at once, so a single worker
// lands around 15–30 nonces/second and no amount of batching changes that.
//
// Three properties of the template follow from the node's own code, and the
// browser wallet gets at least the first two wrong in ways that fail silently —
// blocks are found and then rejected, which looks exactly like "mining doesn't
// work".
//
// ── 1. Grind on `mining_midstate`, never on `post_tx_midstate` ──
//
// `finish_template` returns `mining_midstate = compute_header_hash(header)`.
// The node's comment is explicit that grinding on `post_tx_midstate` instead is
// "the bug that silently rejected every block the web wallet ever mined over
// WebRTC". `verify_extension` recomputes from the header hash on receipt, so a
// nonce found against anything else simply does not validate. This module only
// ever passes `template.mining_midstate` to the search.
//
// ── 2. The template expires; the timestamp is locked inside the hash ──
//
// `finish_template` locks `actual_timestamp` into the header BEFORE computing
// `mining_midstate`, precisely so a miner "cannot bump it post-grind without
// invalidating the header hash they searched on". The submitted batch must
// therefore carry the template's timestamp verbatim.
//
// But `validate_timestamp` requires a new block's timestamp to exceed the median
// of the last MEDIAN_TIME_PAST_WINDOW (11) blocks. At TARGET_BLOCK_TIME = 60s
// that median trails the tip by roughly five blocks, so a template older than
// about five minutes produces a block the node rejects on arrival — and the
// miner cannot fix it by touching the timestamp, because that invalidates the
// PoW. The only correct response is to abandon the template and fetch a fresh
// one. This module refreshes on every tip change and on a hard age cap.
//
// ── 3. The coinbase total must equal block_reward + total_fees ──
//
// The node rejects any other total with a `CoinbaseTotalMismatch` carrying
// `expected_total`. Fees are only knowable from a template, and a template needs
// a coinbase, so the first request of a session is a guess. The browser wallet
// guesses `block_reward` alone, which mismatches on every non-empty mempool and
// costs a round trip each time. This module carries `expected_total` forward
// between templates, so it guesses correctly after the first block and treats
// the retry as the exception rather than the norm.

/** Locked-in timestamp goes stale against MEDIAN_TIME_PAST_WINDOW; refresh well before. */
export const TEMPLATE_MAX_AGE_MS = 120_000;

/** Nonces tested per `search_nonces` call: `iterations × 4` SIMD lanes. */
const DEFAULT_ITERATIONS = 1;

/**
 * A nonce range large enough that two workers starting at random offsets will
 * not collide within any realistic session. Kept under 2^53 so the value
 * survives JSON transport to a pool without precision loss.
 */
const NONCE_SPACE = Number.MAX_SAFE_INTEGER - 1_000_000_000;

import { search_nonces } from '../pkg/wasm_wallet.js';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Solo miner.
 *
 * Drives the full lifecycle: build a coinbase, request a template, grind
 * nonces against `mining_midstate`, splice the winning extension into the
 * template's batch, and submit. Refreshes the template whenever the tip moves
 * or the timestamp risks going stale.
 *
 * @example
 * const miner = new Miner(wallet, client);
 * miner.on('found',    (b) => console.log('block at', b.height));
 * miner.on('rejected', (b) => console.log('rejected:', b.reason));
 * await miner.start();
 */
export class Miner {
    /**
     * @param {Object} wallet  A `Wallet` (needs `.inner` for the WASM methods).
     * @param {Object} client  A connected `MidstateClient`.
     * @param {Object} [opts]
     * @param {number} [opts.iterations=1]      SIMD batches per search call (×4 nonces).
     * @param {number} [opts.throttleMs=0]      Sleep between search calls.
     * @param {number} [opts.tipPollMs=15000]   How often to re-check the tip.
     * @param {number} [opts.templateMaxAgeMs]  Force a refresh after this long.
     * @param {string} [opts.payoutMssAddress]  Pay coinbase to this MSS address
     *   instead of consuming a fresh WOTS index per block.
     * @param {Object} [opts.pool]              A connected `PoolClient`. Switches
     *   to pool mining: grind the pool's share target, submit shares, and keep
     *   going on a hit (see below).
     * @param {string} [opts.poolAddress]       Payout address registered with the pool.
     * @param {string} [opts.workerName='sdk']  Worker label reported on shares.
     * @param {Function} [opts.searchFn]        Override the nonce search. Defaults
     *   to the WASM SIMD `search_nonces`; injectable so a GPU backend or a test
     *   can substitute one without touching the loop.
     */
    constructor(wallet, client, opts = {}) {
        this.wallet = wallet;
        this.client = client;
        this.iterations = opts.iterations ?? DEFAULT_ITERATIONS;
        this.throttleMs = opts.throttleMs ?? 0;
        this.tipPollMs = opts.tipPollMs ?? 15_000;
        this.templateMaxAgeMs = opts.templateMaxAgeMs ?? TEMPLATE_MAX_AGE_MS;
        this.payoutMssAddress = opts.payoutMssAddress ?? null;
        this.searchFn = opts.searchFn ?? search_nonces;

        // ── Pool mode ──
        //
        // A share is not a block. Finding one leaves `mining_midstate`
        // unchanged, so the template stays valid and the search must continue
        // from the next nonce. Halting — the correct solo behaviour, where a hit
        // means the tip moved — leaves a pool miner idle until the next network
        // block, because the "restart on new midstate" path never fires.
        this.pool = opts.pool ?? null;
        this.poolAddress = opts.poolAddress ?? null;
        this.workerName = opts.workerName ?? 'sdk';
        this.continueOnFound = !!this.pool;
        this._poolJobDirty = false;

        this.mining = false;
        this.template = null;
        this.templateAt = 0;
        this.stats = {
            nonces: 0, found: 0, accepted: 0, rejected: 0, templates: 0,
            shares: 0, sharesAccepted: 0, startedAt: 0,
        };

        /** Carried between templates so the coinbase total is right first try. */
        this._expectedTotal = null;
        this._listeners = {};
    }

    /** Subscribe to `template`, `progress`, `found`, `accepted`, `rejected`, `error`. */
    on(event, cb) {
        (this._listeners[event] = this._listeners[event] || []).push(cb);
        return this;
    }

    _emit(event, payload) {
        for (const cb of this._listeners[event] || []) {
            try { cb(payload); } catch { /* a listener must not kill the loop */ }
        }
    }

    /**
     * Build a coinbase and fetch a template for it.
     *
     * # Reasoning
     *
     * The coinbase total must equal `block_reward + total_fees` exactly. Fees
     * are only visible in a template response, and a template needs a coinbase,
     * so the first attempt of a session is necessarily a guess. `expected_total`
     * from a previous mismatch (or a previous template) is carried forward, so
     * steady-state mining guesses right and the retry is exceptional.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - client? is connected
     *
     * Post (success):
     *   template!.mining_midstate = compute_header_hash(candidate_header)
     *   ∧ Σ template!.batch_template.coinbase[].value
     *       = template!.block_reward + template!.total_fees
     *   ∧ template!.mining_addrs ≠ ∅
     *
     * Post (failure): raises after `retries` mismatches; wallet state unchanged
     *   (mining addresses are only committed on acceptance).
     * ```
     *
     * # Safety / Invariants
     *
     * - Does NOT advance `nextWotsIndex`. The indices in `mining_addrs` are
     *   reserved, not consumed: an unaccepted block must not burn key material,
     *   and coinbase addresses are only ever paid to by a block that lands.
     * - Returns the template verbatim. Nothing here may alter `timestamp`,
     *   which is bound into `mining_midstate`.
     *
     * @param {number} [retries=4]
     * @returns {Promise<Object>} The template, augmented with mining metadata.
     */
    async fetchTemplate(retries = 4) {
        // Pool mode: the pool owns the block. It builds the coinbase (paying the
        // registered address), rebuilds the block from its own copy on a winning
        // share, and submits it. There is nothing for this client to negotiate,
        // so the whole coinbase-total dance below is skipped.
        if (this.pool) {
            if (!this.poolAddress) throw new Error('Pool mining requires opts.poolAddress.');
            const job = await this.pool.getTemplate(this.poolAddress);
            if (!job || !job.mining_midstate) throw new Error('Pool returned no job.');
            this.stats.templates++;
            this._poolJobDirty = false;
            return {
                mining_midstate: job.mining_midstate,
                // The SHARE target, not the network target. Grinding the network
                // target here would find one share per several million years.
                target: job.target,
                network_target: job.network_target ?? null,
                job_id: job.job_id,
                chain_height: job.height,
                total_fees: 0,
                block_reward: 0,
                mining_addrs: [],
                batch_template: job.batch_template ?? { transactions: [] },
                isPoolJob: true,
            };
        }

        const state = await this.client.getState();
        let total = this._expectedTotal ?? BigInt(state.block_reward ?? 0);

        for (let attempt = 0; attempt < retries; attempt++) {
            const cbStr = this.payoutMssAddress
                ? this.wallet.inner.build_coinbase_to_mss(BigInt(total), this.payoutMssAddress)
                : this.wallet.inner.build_coinbase(BigInt(total), this.wallet.nextWotsIndex);
            if (!cbStr) throw new Error('build_coinbase returned nothing.');
            const cb = JSON.parse(cbStr);

            const resp = await this.client.getBlockTemplate(cb.coinbase);
            if (resp.ok) {
                const tmpl = await resp.json();
                tmpl.mining_addrs = cb.mining_addrs || [];
                tmpl.next_wots_index = cb.next_wots_index ?? this.wallet.nextWotsIndex;
                tmpl.chain_height = state.height;
                // Remember the correct total for the next template.
                this._expectedTotal = BigInt(tmpl.block_reward ?? 0) + BigInt(tmpl.total_fees ?? 0);
                this.stats.templates++;
                return tmpl;
            }

            // Fee mismatch: the node hands back the total it wants.
            //
            // Read the body defensively. The P2P client returns { ok, status,
            // json } with no `text()`, so calling resp.text() here — as the
            // browser wallet does on the unknown-error path — throws a
            // TypeError that masks the real error.
            let body = null;
            try { body = await resp.json(); } catch { /* not JSON */ }
            if (typeof body === 'string') { try { body = JSON.parse(body); } catch { /* leave */ } }

            if (body && body.expected_total !== undefined) {
                total = BigInt(body.expected_total);
                this._expectedTotal = total;
                continue;
            }
            throw new Error(`Block template rejected: ${body?.error || JSON.stringify(body) || resp.status}`);
        }
        throw new Error(`Could not agree a coinbase total after ${retries} attempts (fees kept moving).`);
    }

    /**
     * Assemble and submit a block for a winning nonce.
     *
     * # Reasoning
     *
     * The extension is recomputed here rather than trusted from the search.
     * `search_nonces` returns only the nonce, and a GPU or remote searcher can
     * return a nonce whose hash does not actually reproduce — re-deriving it
     * with `build_solo_extension` costs one VDF evaluation (~130 ms) and turns a
     * would-be invalid submission into a local no-op.
     *
     * The batch is cloned and ONLY `extension` is written. In particular
     * `timestamp` is left exactly as the template set it: it is folded into
     * `mining_midstate`, so changing it would invalidate the very PoW being
     * submitted.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - nonce? was found against template.mining_midstate
     *
     * Post (accepted):
     *   batch.extension = create_extension(mining_midstate, nonce?)
     *   ∧ batch.timestamp = template.batch_template.timestamp   (unchanged)
     *   ∧ ∀ a ∈ template.mining_addrs • a.address ∈ dom wots_addrs'
     *   ∧ next_wots_index' = template.next_wots_index
     *
     * Post (rejected):
     *   wots_addrs' = wots_addrs ∧ next_wots_index' = next_wots_index
     *   (a rejected block burns no key material)
     * ```
     *
     * # Safety / Invariants
     *
     * - Key material is committed only on acceptance. Registering coinbase
     *   addresses for a rejected block would advance `next_wots_index` past
     *   indices that were never paid to, wasting them for no benefit.
     * - Never mutates `timestamp`; see above.
     *
     * @param {Object} template
     * @param {bigint|string|number} nonce
     * @returns {Promise<Object>} Submission outcome.
     */
    async submitBlock(template, nonce) {
        const n = BigInt(nonce);

        // Pool mode: submit a share, not a block. The pool re-derives the hash
        // from (job_id, nonce) against its own copy of the template, so there is
        // no extension to build and no batch to assemble here.
        if (template.isPoolJob) {
            this.stats.shares++;
            try {
                const res = await this.pool.submitShare(this.poolAddress, template.job_id, n, this.workerName);
                this.stats.sharesAccepted++;
                return {
                    accepted: true, isShare: true, nonce: n.toString(),
                    jobId: template.job_id, height: template.chain_height,
                    blockFound: !!(res && (res.block_found || res.is_block)),
                    reason: null,
                };
            } catch (e) {
                // A stale share means the job rotated under us — refresh rather
                // than keep grinding a superseded midstate.
                if (/stale|unknown job|job_id/i.test(e.message || '')) this._poolJobDirty = true;
                return { accepted: false, isShare: true, nonce: n.toString(), reason: e.message };
            }
        }

        const extStr = this.wallet.inner.build_solo_extension(template.mining_midstate, n);
        if (!extStr) {
            return { accepted: false, reason: 'extension did not reproduce for this nonce', nonce: n.toString() };
        }

        const batch = JSON.parse(JSON.stringify(template.batch_template));
        batch.extension = JSON.parse(extStr);

        const resp = await this.client.submitBatch(batch);
        const accepted = !!(resp && resp.ok);

        if (accepted) {
            // Commit the reserved coinbase addresses now that a block paid them.
            for (const entry of template.mining_addrs || []) {
                this.wallet.wotsAddrs[entry.address] = entry.index;
            }
            if (typeof template.next_wots_index === 'number') {
                this.wallet.nextWotsIndex = Math.max(this.wallet.nextWotsIndex, template.next_wots_index);
            }
            this.wallet._setWatchlist();
            await this.wallet.save();
            this.stats.accepted++;
        } else {
            this.stats.rejected++;
        }

        const ext = batch.extension;
        const finalHash = Array.isArray(ext.final_hash)
            ? ext.final_hash.map((b) => b.toString(16).padStart(2, '0')).join('')
            : String(ext.final_hash);

        return {
            accepted,
            reason: accepted ? null : (resp?.body || resp?.error || 'rejected'),
            nonce: n.toString(),
            finalHash,
            height: template.chain_height,
            reward: BigInt(template.block_reward ?? 0) + BigInt(template.total_fees ?? 0),
            fees: BigInt(template.total_fees ?? 0),
            txCount: batch.transactions?.length ?? 0,
        };
    }

    /** True when the current template must be abandoned. */
    _templateStale(tipHeight) {
        if (!this.template) return true;
        if (this.template.isPoolJob) {
            // The pool tells us when a job rotates; height is its own signal.
            if (this._poolJobDirty) return true;
            if (typeof tipHeight === 'number' && tipHeight !== this.template.chain_height) return true;
            return Date.now() - this.templateAt > this.templateMaxAgeMs;
        }
        if (Date.now() - this.templateAt > this.templateMaxAgeMs) return true;
        if (typeof tipHeight === 'number' && tipHeight !== this.template.chain_height) return true;
        return false;
    }

    /**
     * Run the mining loop until {@link Miner#stop}.
     *
     * # Reasoning
     *
     * The loop is structured around template freshness rather than hash count,
     * because on this chain a stale template is the dominant failure mode: the
     * PoW is expensive enough that a naive miner spends most of its time
     * grinding a header the network has already moved past, and every block it
     * finds is rejected for building on a stale parent or carrying a timestamp
     * below the median of the last 11 blocks.
     *
     * Each worker starts at a random nonce offset so that several instances
     * against the same template do not retrace each other's work.
     *
     * # Safety / Invariants
     *
     * - Only ever searches against `template.mining_midstate`.
     * - Never mutates the template's `timestamp` or `transactions`.
     * - A failed submission does not stop mining; it refreshes and continues.
     *
     * @returns {Promise<Object>} Final stats, once stopped.
     */
    async start() {
        if (this.mining) throw new Error('Miner is already running.');
        this.mining = true;
        this.stats.startedAt = Date.now();

        let nonce = BigInt(Math.floor(Math.random() * NONCE_SPACE));
        let lastTipCheck = 0;
        let tipHeight = null;
        let reportAt = Date.now();
        let noncesSinceReport = 0;

        try {
            while (this.mining) {
                // Re-check the tip on a timer; a moved tip invalidates the template.
                if (Date.now() - lastTipCheck > this.tipPollMs) {
                    lastTipCheck = Date.now();
                    try {
                        const st = await this.client.getState();
                        tipHeight = st.height;
                    } catch (e) {
                        this._emit('error', e);
                    }
                }

                if (this._templateStale(tipHeight)) {
                    try {
                        this.template = await this.fetchTemplate();
                        this.templateAt = Date.now();
                        nonce = BigInt(Math.floor(Math.random() * NONCE_SPACE));
                        this._emit('template', {
                            height: this.template.chain_height,
                            target: this.template.target,
                            txCount: this.template.batch_template?.transactions?.length ?? 0,
                            fees: BigInt(this.template.total_fees ?? 0),
                        });
                    } catch (e) {
                        this._emit('error', e);
                        await sleep(5_000);
                        continue;
                    }
                }

                const found = this._search(this.template, nonce);

                const perCall = this.iterations * 4;
                nonce += BigInt(perCall);
                this.stats.nonces += perCall;
                noncesSinceReport += perCall;

                if (found !== undefined && found !== null) {
                    this.stats.found++;
                    this._emit('found', { nonce: found.toString(), height: this.template.chain_height });
                    let result;
                    try {
                        result = await this.submitBlock(this.template, found);
                    } catch (e) {
                        this._emit('error', e);
                        result = { accepted: false, reason: e.message };
                    }
                    this._emit(result.isShare ? 'share' : (result.accepted ? 'accepted' : 'rejected'), result);

                    // A share leaves the job valid, so keep grinding from the
                    // next nonce (already advanced above). Dropping the template
                    // here would idle this miner until the next network block.
                    if (!this.continueOnFound) {
                        this.template = null;
                    }
                    continue;
                }

                const elapsed = Date.now() - reportAt;
                if (elapsed >= 1000) {
                    this._emit('progress', {
                        nonces: this.stats.nonces,
                        rate: noncesSinceReport / (elapsed / 1000),
                        height: this.template?.chain_height ?? null,
                    });
                    noncesSinceReport = 0;
                    reportAt = Date.now();
                }

                // Always yield, even at zero throttle.
                //
                // `search_nonces` is a synchronous WASM call that blocks for
                // ~250 ms, and with no await in the loop the event loop never
                // runs — so a timer calling stop(), an inbound push, or any
                // other I/O is starved until some other branch happens to
                // await. Observed live: a 1-second mining run kept going for
                // 15.75 s, because the only await on the path was the 15 s tip
                // poll. A zero-delay yield is free against a 250 ms VDF and
                // makes stop() take effect within one search call.
                await sleep(this.throttleMs);
            }
        } finally {
            this.mining = false;
        }
        return this.getStats();
    }

    /** Isolated so tests can substitute a search without a real VDF. */
    _search(template, nonce) {
        return this.searchFn(template.mining_midstate, template.target, nonce, this.iterations);
    }

    /** Stop the loop after the current search call returns. */
    stop() { this.mining = false; }

    /** @returns {Object} Cumulative statistics. */
    getStats() {
        const secs = this.stats.startedAt ? (Date.now() - this.stats.startedAt) / 1000 : 0;
        return {
            ...this.stats,
            elapsedSecs: secs,
            noncesPerSec: secs > 0 ? this.stats.nonces / secs : 0,
        };
    }
}
