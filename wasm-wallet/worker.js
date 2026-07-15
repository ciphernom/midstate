/**
 * @fileoverview Midstate Web Wallet — Worker Thread
 *
 * This Web Worker manages all wallet operations: key derivation, chain scanning,
 * transaction building/signing, and solo mining coordination. It communicates
 * with the main thread (index.html) via postMessage for UI updates and RPC
 * calls (the main thread owns the WebRTC LightClient since RTCPeerConnection
 * is unavailable in workers).
 *
 * ## MSS Tree Storage (IndexedDB)
 *
 * MSS Merkle trees (~64 KB each at height 10) are stored as compact binary
 * blobs in IndexedDB rather than in the encrypted localStorage wallet JSON.
 * This eliminates the 5 MB localStorage limit concern and keeps the wallet
 * JSON lightweight (metadata only). Trees are loaded into the WASM wallet
 * on login via `loadMssCaches()`.
 *
 * ## Key Lifecycle
 *
 * 1. CREATE: Generate mnemonic → derive WOTS keys → generate MSS tree → save
 *    to IndexedDB + localStorage.
 * 2. LOGIN: Decrypt localStorage → construct WebWallet → load MSS trees from
 *    IndexedDB → ready.
 * 3. SEND: prepare_spend (coin selection) → commit → PoW → reveal → confirm.
 * 4. SCAN: Fetch block filters → check for wallet relevance → process matches.
 *
 * @module worker
 */

import init, { WebWallet, generate_phrase, compute_coin_id_hex, compute_commitment_hex, decrypt_cli_wallet, mine_commitment_pow, blake3_hash_hex, build_multisig_2of2_address, build_channel_state, build_channel_reveal, verify_mss_sig_wasm, mine_chat_pow_v2_wasm, build_htlc_bytecode_hex, build_covenant_htlc_bytecode_hex, build_limit_order_covenant_bytecode_hex, compute_p2pk_address_hex, qbolt_channel_address, qbolt_channel_bytecode_hex, qbolt_build_state, qbolt_build_refund_state, qbolt_build_close_reveal, qbolt_build_refund_reveal, qbolt_build_legacy_close_state, qbolt_build_legacy_close_reveal } from './pkg/wasm_wallet.js';

/** @type {WebWallet|null} The WASM wallet instance. Null until CREATE or LOGIN. */
let wallet = null;

/** @type {string|null} The user's password, held in memory for encrypting state saves. */
let password = null;

/** @type {boolean} Guard against concurrent send operations. */
let isSending = false;
// FIFO waiters for the send lock. acquireSendLock() resolves when the caller may proceed;
// releaseSendLock() hands the lock to the next waiter (or clears it). This turns the old
// "throw if busy" guard into "wait your turn", while keeping the strict one-at-a-time
// serialization that WOTS/MSS key-index safety depends on. Handler bodies are unchanged:
// they just `await acquireSendLock()` instead of `if (isSending) throw`, and call
// releaseSendLock() where they used to set `isSending = false`.
const _sendWaiters = [];
function acquireSendLock(label) {
    self.postMessage({ type: 'TX_QUEUE', payload: { running: isSending ? undefined : (label || 'transaction'), waiting: _sendWaiters.length + (isSending ? 1 : 0) } });
    if (!isSending) { isSending = true; self.postMessage({ type: 'TX_QUEUE', payload: { running: label || 'transaction', waiting: _sendWaiters.length } }); return Promise.resolve(); }
    return new Promise((resolve) => { _sendWaiters.push({ resolve, label: label || 'transaction' }); });
}
function releaseSendLock() {
    const next = _sendWaiters.shift();
    if (next) {
        // Keep isSending true — lock passes directly to the next waiter (no gap where a
        // fresh request could jump the queue).
        self.postMessage({ type: 'TX_QUEUE', payload: { running: next.label, waiting: _sendWaiters.length } });
        next.resolve();
    } else {
        isSending = false;
        self.postMessage({ type: 'TX_QUEUE', payload: { running: null, waiting: 0 } });
    }
}
// Covenant swaps: the MDS-side over-funds the HTLC by this many sats so the
// DELIVERY fee is paid out of the LOCKED value, not from a separate coin. That
// is what lets a buyer who holds zero MDS still self-deliver as a fallback.
// Unused budget (FEE_BUDGET − actual fee) returns to whoever broadcasts the
// delivery as change. Generous on purpose; the real cost is only the network fee.
const COVENANT_FEE_BUDGET = 4096;
// What a buyer requires the lock to be over-funded by before locking ETH, so the
// delivery is guaranteed to be affordable from the locked value.
const COVENANT_MIN_FEE_RESERVE = 1024;

// ── Contract tracking (MidstateConnect) ─────────────────────────────────────
let watchedContracts = new Set();   // hex contract addresses we are tracking
let contractCoins = {};             // coinId -> { address, value, salt, state|null, coin_id }

/** @type {boolean} Guard against concurrent block submissions. */
let isSubmitting = false;

/** @type {boolean} Guard against concurrent chain scans. */
let isScanning = false;

/** @type {boolean} Set when a scan is requested while one is already running.
 *  The wrapper loops once more when the current iteration finishes. */
let scanRequested = false;

/** @type {number} Bumped to cancel an in-flight scan. The inner loop checks
 *  this and exits without committing partial state. */
let scanGeneration = 0;

/** @type {boolean} When true, the next scan iteration wipes wallet state
 *  before scanning. Set by RESCAN; consumed atomically inside the wrapper. */
let scanResetPending = false;

/** @type {boolean} Guard against concurrent template requests to prevent WebRTC stream exhaustion. */
let isFetchingTemplate = false;

/** @type {number} The current network height fetched from the node. */
let networkHeight = 0;

/** @type {number} The current mempool size fetched from the node. */
let mempoolSize = 0;

/** @type {Array<Function>} Resolvers awaiting the next block push event. */
let nextBlockResolvers = [];
/** @type {Object|null} Tracks an outgoing Lightning channel open intent */
let pendingChannelOpen = null;
let miningMode = 'solo'; // 'solo' | 'pool'
let poolUrl = '';
let payoutAddress = '';

/**
 * Suspend execution until the next block arrives via WebRTC push, 
 * or fallback to resolving after a maximum timeout.
 * @param {number} timeoutMs - Maximum wait time before continuing anyway
 * @returns {Promise<boolean>} True if resolved by block, False if timeout.
 */
function waitForNextBlock(timeoutMs = 15000) {
    return new Promise(resolve => {
        let timer = setTimeout(() => {
            nextBlockResolvers = nextBlockResolvers.filter(r => r !== resolve);
            resolve(false);
        }, timeoutMs);
        nextBlockResolvers.push(() => {
            clearTimeout(timer);
            resolve(true);
        });
    });
}

// ── confirm-wait helpers ─────────────────────────────────────────────────────
//
// These replace the old `while (true) { checkCommitment/checkCoin … }` loops,
// which could freeze a UI card forever: a blind `catch {}` swallowed every
// network error, nothing bounded the wait, and a commit evicted from the
// mempool (node restart, WebRTC flap mid-broadcast, replacement) would simply
// never be mined while the loop spun on.

/** Hard wall-clock ceiling for confirmation waits (~60 s blocks → ≈20 blocks).
 *  On expiry the flow fails with a clear, actionable error instead of hanging. */
const CONFIRM_WAIT_MS = 20 * 60_000;
/** Consecutive failed status checks tolerated before declaring the network
 *  unreachable. Iterations are ≥15 s apart, so 12 ≈ 3 min of solid failure. */
const CONFIRM_MAX_NET_FAILS = 12;

/**
 * Wait until `commitmentHex` is mined into state, defending against the ways
 * the old loop could hang:
 *
 *  • Polls once per block push (15 s fallback cadence when pushes flap).
 *  • While unmined, probes the mempool every other iteration; if the commit
 *    is absent from BOTH state and mempool on two consecutive probes it was
 *    evicted — re-mine the anti-spam PoW against fresh state (the PoW anchors
 *    a recent height, so replaying the original nonce can be rejected) and
 *    rebroadcast, up to 3 times, then abort.
 *  • Counts consecutive check failures instead of swallowing them; aborts
 *    with the underlying reason after CONFIRM_MAX_NET_FAILS.
 *  • Aborts after CONFIRM_WAIT_MS regardless.
 *
 * @param {string} commitmentHex - The commitment (hex) already broadcast.
 * @param {(msg: string) => void} [onStatus] - Optional phase reporter (e.g. dexPhase).
 * @returns {Promise<Object>} The final checkCommitment response (has `.height` when the node provides it).
 */
async function awaitCommitmentMined(commitmentHex, onStatus) {
    const started = Date.now();
    const say = (m) => { try { if (onStatus) onStatus(m); } catch (_) {} };

    // The commitment appears in mempool JSON either as a hex string or as a
    // 32-entry byte array depending on serde config. Probe the stringified
    // pool for both forms so we don't depend on the exact Transaction shape.
    const hexLower = String(commitmentHex).toLowerCase().replace(/^0x/, '');
    let byteArrJson = null;
    try {
        const bytes = [];
        for (let i = 0; i < hexLower.length; i += 2) bytes.push(parseInt(hexLower.slice(i, i + 2), 16));
        if (bytes.length === 32 && bytes.every(Number.isFinite)) byteArrJson = JSON.stringify(bytes);
    } catch (_) {}

    let netFails = 0;
    let absentProbes = 0;   // consecutive "not in state AND not in mempool"
    let rebroadcasts = 0;
    let iter = 0;

    while (true) {
        if (Date.now() - started > CONFIRM_WAIT_MS) {
            throw new Error(`Timed out after ${Math.round(CONFIRM_WAIT_MS / 60000)} min waiting for the commit to be mined. It may still confirm later — sync the wallet before retrying.`);
        }

        let minedCheckOk = false;
        try {
            const c = await rpc.checkCommitment(commitmentHex);
            netFails = 0;
            if (c && c.exists) return c;
            minedCheckOk = true;
        } catch (e) {
            netFails++;
            if (e && (e.code === 'RATE_LIMITED' || e.code === 'GATEWAY_UNAVAILABLE')) {
                say('Network is rate-limiting — waiting it out…');
            }
            if (netFails >= CONFIRM_MAX_NET_FAILS) {
                throw new Error(`Lost contact with the network while waiting for the commit (${(e && e.message) || e}). It may still confirm — sync the wallet before retrying.`);
            }
        }

        // Eviction probe — only when the state check itself succeeded, only
        // every other iteration to keep the extra RPC load negligible, and
        // never on the first iteration (give the broadcast time to propagate).
        if (minedCheckOk && (iter++ % 2 === 1)) {
            let present = null;   // null = probe failed → unknown, don't count
            try {
                const mp = await rpc.getMempool();
                const hay = JSON.stringify((mp && mp.transactions) || []).toLowerCase();
                present = hay.includes(hexLower) || (byteArrJson !== null && hay.includes(byteArrJson));
            } catch (_) {}

            if (present === true) {
                absentProbes = 0;
            } else if (present === false && ++absentProbes >= 2) {
                absentProbes = 0;
                if (rebroadcasts >= 3) {
                    throw new Error('The commit keeps disappearing from the mempool (evicted). Aborting — sync the wallet and try the operation again.');
                }
                rebroadcasts++;
                say(`Commit was dropped from the mempool — rebroadcasting (${rebroadcasts}/3)…`);
                let ok = false, body = '';
                try {
                    const st = await rpc.getState();
                    const nonce = Number(mine_commitment_pow(commitmentHex, st.required_pow || 24, BigInt(st.height), st.header_hash));
                    const r = await rpc.commit(commitmentHex, nonce);
                    ok = !!(r && r.ok);
                    body = String((r && (r.body || r.error)) || '');
                } catch (e) {
                    ok = false;
                    body = (e && e.message) || String(e);
                }
                // "already known / exists / duplicate" means it's back in the pool
                // or got mined between probes — both fine. Transient network noise
                // is retried next cycle. A real consensus rejection aborts.
                if (!ok && !/already|exist|duplicate|RATE_LIMITED|GATEWAY_UNAVAILABLE|timeout/i.test(body)) {
                    throw new Error(`Commit rebroadcast rejected by the node: ${body}`);
                }
            }
        }

        await waitForNextBlock(15000);
    }
}

/**
 * Wait until `coinId` disappears from state — i.e. the reveal spending it was
 * mined. Same failure discipline as awaitCommitmentMined (hard deadline +
 * counted network failures instead of a blind catch), but deliberately NO
 * rebroadcast: re-sending a reveal is flow-specific and can burn one-time key
 * material, so on failure the error tells the user to SYNC FIRST rather than
 * retry blind.
 *
 * @param {string} coinId - Input coin id consumed by the reveal.
 * @param {(msg: string) => void} [onStatus] - Optional phase reporter.
 * @returns {Promise<Object>} The final checkCoin response (may carry `.spentHeight`).
 */
/**
 * Wait until a broadcast reveal is mined, rebroadcasting it if the network
 * loses it.
 *
 * `awaitCoinSpent` only polls — it never rebroadcasts. That asymmetry is a real
 * hole: `awaitCommitmentMined` defends phase 1 against a mempool eviction or a
 * transport flap mid-broadcast, but phase 2 had no such defence, so a reveal
 * dropped on a dying WebRTC connection would spin for the full timeout and be
 * lost. The commitment is already mined and single-shot at that point, so the
 * tx cannot simply be rebuilt — but the signed reveal can be re-sent verbatim.
 *
 * Rebroadcasting the identical payload is always safe: same one-time leaf over
 * the same message, so it carries no key-reuse risk. The node either accepts it
 * or reports the inputs already spent (which means we are done).
 *
 * @param {string} inputCoinId  - An input the reveal spends; its disappearance means "mined".
 * @param {string} revealStr    - The exact signed reveal payload to re-send.
 * @param {string} commitmentHex- Used to spot the reveal sitting in the mempool.
 * @param {(msg: string) => void} [onStatus]
 * @returns {Promise<Object>} The final checkCoin response.
 */
async function awaitRevealMined(inputCoinId, revealStr, commitmentHex, onStatus, allInputIds) {
    const started = Date.now();
    const say = (m) => { try { if (onStatus) onStatus(m); } catch (_) {} };
    const inputIds = (allInputIds && allInputIds.length) ? allInputIds : [inputCoinId];

    // The node evicts a reveal for exactly three reasons (mempool prune_invalid):
    // an input coin missing from state, the commitment missing/expired, or —
    // only at capacity — being outbid in the fee market. Repeated eviction of a
    // freshly-committed tx therefore almost always means a DEAD INPUT: admission
    // verifies signatures but not coin existence, so the reveal is accepted and
    // then pruned on the very next block, forever. Diagnose instead of guessing.
    const diagnoseAndHeal = async () => {
        const missing = [];
        for (const id of inputIds) {
            try { const c = await rpc.checkCoin(id); if (c && !c.exists) missing.push(id); } catch (_) {}
        }
        let commitAlive = null;
        try { const cm = await rpc.checkCommitment(commitmentHex); commitAlive = !!(cm && cm.exists); } catch (_) {}

        if (missing.length) {
            // This transaction can NEVER confirm — every rebroadcast will be
            // pruned. Heal: drop the phantom coins from the wallet's UTXO set
            // (a rescan re-credits anything that genuinely exists), release the
            // pending record, and say exactly which coin is dead.
            for (const id of missing) { if (wState.utxos && wState.utxos[id]) delete wState.utxos[id]; }
            pendingSends = pendingSends.filter(tx => !(tx.inputs || []).some(i => missing.includes(i)));
            if (wState.pending_tx && wState.pending_tx.commitment === commitmentHex) delete wState.pending_tx;
            await saveState();
            performScan().catch(() => {});
            throw new Error(`This transaction can never confirm: input coin(s) ${missing.map(m => m.substring(0, 12) + '…').join(', ')} do not exist on-chain. The wallet's records were out of sync — the stale coin(s) have been removed and a rescan started. Please re-send the transaction.`);
        }
        if (commitAlive === false) {
            if (wState.pending_tx && wState.pending_tx.commitment === commitmentHex) delete wState.pending_tx;
            await saveState();
            throw new Error('The commitment is no longer in chain state (expired or reorged out), so this reveal is dead. The coins were never spent and remain yours — please re-send the transaction.');
        }

        // Inputs exist and the commitment is live, yet the node keeps evicting.
        // The remaining eviction cause is a BURNED ONE-TIME LEAF: this reveal is
        // signed with an MSS/WOTS leaf that a previously-confirmed transaction
        // already spent. Admission only compares against the live mempool, so it
        // accepts the reveal; prune_on_new_block then evicts it against the
        // chain's burned-leaf accumulator, forever. Reconcile every key against
        // the node and, if that moved any counter, heal so the re-send signs
        // with a fresh leaf.
        const before = JSON.stringify(Object.fromEntries(Object.entries(wState.mssAddrs || {}).map(([a, m]) => [a, m.next_leaf])));
        await reconcileMssLeavesWithNode().catch(() => {});
        const after = JSON.stringify(Object.fromEntries(Object.entries(wState.mssAddrs || {}).map(([a, m]) => [a, m.next_leaf])));
        if (before !== after) {
            if (wState.pending_tx && wState.pending_tx.commitment === commitmentHex) delete wState.pending_tx;
            await saveState();
            throw new Error('This reveal was signed with a one-time signature leaf that a previous transaction already used, so the node evicts it every block. The leaf counters have now been re-synced with the chain — please re-send the transaction; it will sign with a fresh leaf.');
        }
        return null;                                          // still inconclusive → keep trying
    };

    const hexLower = String(commitmentHex || '').toLowerCase().replace(/^0x/, '');
    let byteArrJson = null;
    try {
        const bytes = [];
        for (let i = 0; i < hexLower.length; i += 2) bytes.push(parseInt(hexLower.slice(i, i + 2), 16));
        if (bytes.length === 32 && bytes.every(Number.isFinite)) byteArrJson = JSON.stringify(bytes);
    } catch (_) {}

    let netFails = 0, absentProbes = 0, rebroadcasts = 0, iter = 0;

    while (true) {
        if (Date.now() - started > CONFIRM_WAIT_MS) {
            throw new Error(`Timed out after ${Math.round(CONFIRM_WAIT_MS / 60000)} min waiting for the reveal to be mined. The wallet saved it and will retry on the next unlock — do NOT rebuild the transaction, or you risk spending the same coins twice.`);
        }

        // Mined? The input coin is gone from the UTXO set.
        try {
            const inp = await rpc.checkCoin(inputCoinId);
            netFails = 0;
            if (inp && !inp.exists) return inp;
        } catch (e) {
            netFails++;
            if (netFails >= CONFIRM_MAX_NET_FAILS) {
                throw new Error(`Network unreachable while confirming the reveal: ${e && e.message || e}`);
            }
        }

        // Every other iteration, make sure it's still queued somewhere.
        if (iter % 2 === 1) {
            let present = null;
            try {
                const mp = await rpc.getMempool();
                const hay = JSON.stringify(mp || '').toLowerCase();
                present = hay.includes(hexLower) || (byteArrJson !== null && hay.includes(byteArrJson));
            } catch (_) { present = null; }             // unknown → don't count it against us

            if (present === false) {
                absentProbes++;
                if (absentProbes >= 2) {
                    absentProbes = 0;
                    // The first eviction can be a fluke (race with a block); the
                    // second means something structural — diagnose before wasting
                    // further rebroadcasts. This throws with the exact cause when
                    // the tx is provably dead, and heals the wallet state.
                    if (rebroadcasts >= 1) await diagnoseAndHeal();
                    if (rebroadcasts >= 3) {
                        // Inputs exist and the commitment is live, so by the node's
                        // own prune_invalid the only remaining cause is a BURNED
                        // ONE-TIME LEAF. We cannot prove it from here: the exact
                        // authority is the chain's burned_wots accumulator, which
                        // has no RPC, and getMssState is unreliable — its HTTP
                        // handler scans only the last ~2000 blocks and answers 0
                        // for older keys, so a reconcile that moves nothing proves
                        // nothing. Say what's actually known.
                        throw new Error(
                            'The node keeps evicting this reveal even though its inputs exist and its commitment is live. '
                            + 'That leaves one cause: it is signed with a one-time signature leaf that an earlier transaction already burned — '
                            + 'the mempool accepts it, then prunes it against the chain on every block. '
                            + 'The wallet cannot confirm this itself: the authoritative burned-leaf set has no RPC, and /mss_state over HTTP '
                            + 'only scans the last ~2000 blocks (it answers 0 for older keys, so its silence means nothing). '
                            + 'Fix the node\'s get_mss_state to use storage.query_mss_leaf_index, then reload — the leaf counters will re-sync and a fresh send will use an unburned leaf. '
                            + 'These coins were never spent and remain yours.'
                        );
                    }
                    rebroadcasts++;
                    say(`Reveal was dropped from the mempool — rebroadcasting (${rebroadcasts}/3)…`);
                    try {
                        const rr = await rpc.send(revealStr);
                        if (!rr.ok) {
                            const body = String(rr.body || rr.error || '');
                            // "not found" = the inputs are already spent, i.e. it landed after all.
                            if (/not found|already spent|spent/i.test(body)) {
                                say('The reveal already landed — confirming…');
                            } else if (/expired/i.test(body)) {
                                throw new Error(`The commitment expired before the reveal was mined: ${body}`);
                            }
                        }
                    } catch (e) {
                        if (/expired/i.test(String(e && e.message))) throw e;
                        say(`Rebroadcast failed, will retry: ${e && e.message || e}`);
                    }
                }
            } else if (present === true) {
                absentProbes = 0;
            }
        }

        iter++;
        await waitForNextBlock(15000);
    }
}

/**
 * Resume a transaction that was in flight when the wallet last stopped.
 *
 * A send is two network phases (commit → reveal) separated by minutes of block
 * time. A reload, crash, or transport flap in between used to lose it silently:
 * the reveal was never re-sent and the mined commitment quietly aged out of its
 * TTL. The reveal is already signed and idempotent, so the honest recovery is
 * to re-send it verbatim.
 *
 * This NEVER rebuilds. Once the commitment is mined it is single-shot on-chain,
 * and re-signing would spend another one-time MSS/WOTS leaf.
 */
async function recoverPendingTx() {
    const p = wState.pending_tx;
    if (!p || !p.commitment) return;
    const say = (m) => self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: `[recovery] ${m}` } });
    const done = async (msg, level) => {
        delete wState.pending_tx;
        await saveState();
        self.postMessage({ type: 'L2_EVENT', payload: { level: level || 'info', msg, channelId: null } });
        performScan().catch(() => {});
    };

    try {
        // Did it land while we were away?
        if (p.inputCoinId) {
            const inp = await rpc.checkCoin(p.inputCoinId);
            if (inp && !inp.exists) return done('A transaction that was interrupted has since confirmed.');
        }

        // COMMITMENT_TTL is 1000 blocks: past that a mined commitment can no
        // longer be revealed, and the coins simply become spendable again.
        if (p.commitHeight != null && networkHeight - p.commitHeight > 1000) {
            return done('An interrupted transaction expired before it could be mined. Its coins are spendable again — please re-send it.', 'warn');
        }

        const cm = await rpc.checkCommitment(p.commitment).catch(() => null);
        const committed = !!(cm && cm.exists);

        if (!committed && p.stage === 'commit') {
            // The commit never landed. Re-mine the anti-spam PoW against fresh
            // state (it anchors a recent height, so the old nonce may be stale).
            say('Re-broadcasting an interrupted commit…');
            const st = await rpc.getState();
            const nonce = Number(mine_commitment_pow(p.commitment, st.required_pow || 24, BigInt(st.height), st.header_hash));
            const cr = await rpc.commit(p.commitment, nonce);
            if (!cr.ok && !/already|exist|duplicate/i.test(String(cr.body || cr.error || ''))) {
                self.postMessage({ type: 'LOG', payload: `[recovery] commit retry rejected: ${cr.body || cr.error}` });
                return;                                       // keep it pending; retry next block
            }
        }

        if (!p.revealPayload) return;
        self.postMessage({ type: 'L2_EVENT', payload: { level: 'info', msg: 'Resuming a transaction that was interrupted — re-broadcasting it.', channelId: null } });
        const rr = await rpc.send(p.revealPayload);
        if (!rr.ok) {
            const body = String(rr.body || rr.error || '');
            if (/not found|already spent|spent/i.test(body)) return done('An interrupted transaction had already confirmed.');
            if (/expired/i.test(body)) {
                return done('An interrupted transaction expired before it could be mined. Its coins are spendable again — please re-send it.', 'warn');
            }
            self.postMessage({ type: 'LOG', payload: `[recovery] reveal retry rejected: ${body}` });
            return;                                           // keep it; retried on the next block
        }
        await awaitRevealMined(p.inputCoinId, p.revealPayload, p.commitment, say, p.inputCoinIds);
        await done('A transaction that was interrupted has now confirmed.');
    } catch (e) {
        // Keep the record. A failure here is nearly always transient, and
        // discarding it is precisely what loses the transaction.
        self.postMessage({ type: 'LOG', payload: `[recovery] deferred, will retry: ${e && e.message || e}` });
    }
}

/** Poll until `coinId` leaves the UTXO set. Used where the caller has no reveal
 *  payload to rebroadcast (DEX legs, sweeps); the send path uses
 *  awaitRevealMined instead, which can recover a dropped reveal. */
async function awaitCoinSpent(coinId, onStatus) {
    const started = Date.now();
    const say = (m) => { try { if (onStatus) onStatus(m); } catch (_) {} };
    let netFails = 0;

    while (true) {
        if (Date.now() - started > CONFIRM_WAIT_MS) {
            throw new Error(`Timed out after ${Math.round(CONFIRM_WAIT_MS / 60000)} min waiting for the transaction to confirm. Do NOT retry immediately — sync the wallet first; it may already have confirmed.`);
        }
        try {
            const inp = await rpc.checkCoin(coinId);
            netFails = 0;
            if (inp && !inp.exists) return inp;
        } catch (e) {
            netFails++;
            if (e && (e.code === 'RATE_LIMITED' || e.code === 'GATEWAY_UNAVAILABLE')) {
                say('Network is rate-limiting — waiting it out…');
            }
            if (netFails >= CONFIRM_MAX_NET_FAILS) {
                throw new Error(`Lost contact with the network while waiting for confirmation (${(e && e.message) || e}). Sync the wallet before retrying — it may already have confirmed.`);
            }
        }
        await waitForNextBlock(15000);
    }
}
// ── end confirm-wait helpers ─────────────────────────────────────────────────

/**
 * @type {Array<Object>} Pending send transactions displayed in the UI.
 * Cleared when the send completes or fails.
 */
let pendingSends = [];

/**
 * @const {number} Number of WOTS addresses to pre-derive ahead of the
 * highest known index. Prevents missed coins if the user receives to
 * addresses beyond the currently synced range.
 */
const GAP_LIMIT = 100;




/**
 * @typedef {Object} WalletState
 * @property {string|null} phrase - BIP39 mnemonic (null for imported CLI wallets).
 * @property {number} nextWotsIndex - Next unused WOTS HD derivation index.
 * @property {number} nextMssIndex - Next unused MSS HD derivation index.
 * @property {Object<string, number>} wotsAddrs - Map of hex address → derivation index.
 * @property {Object<string, MssMetadata>} mssAddrs - Map of hex address → MSS metadata.
 * @property {Object<string, UtxoEntry>} utxos - Map of coin_id → UTXO data.
 * @property {Array<HistoryEntry>} history - Transaction history entries.
 * @property {number} lastScannedHeight - Last fully scanned block height.
 */

/**
 * @typedef {Object} MssMetadata
 * @property {number} index - MSS HD derivation index.
 * @property {number} height - Merkle tree height.
 * @property {number} next_leaf - Next unused leaf counter.
 */

/**
 * @typedef {Object} DexSecretRecord   — one entry per limit-order UNIT this wallet posted.
 * @property {string} secret        64-hex BLAKE3 preimage. THE asset: whoever knows it can
 *                                  claim the taker's Base ETH (maker) or spend the covenant
 *                                  once public (anyone). Lives ONLY here, AES-GCM-encrypted.
 * @property {string} secretHash    64-hex H = BLAKE3(secret); the public hashlock.
 * @property {string} covAddr       64-hex covenant address = BLAKE3(covenant bytecode).
 * @property {number} value         Unit size in MDS (power of two).
 * @property {string} salt          64-hex coin salt (needed to rebuild coin_id).
 * @property {number} timeoutHeight Absolute refund height baked into the covenant.
 * @property {string} makerMdsPk    Refund pubkey baked into the covenant (ours).
 * @property {string} makerEvmAddr  Base address paid on fill.
 * @property {string} weiAmount     Ask price in wei (string — may exceed 2^53).
 * @property {string} offerId       UI offer id for this unit.
 * @property {string|null} groupId  Funding-bundle group id (null for legacy single path).
 * @property {'funding'|'live'|'limit_cancelled'} status
 *                                  funding: written before the funding broadcast, not yet
 *                                           confirmed (adjudicated by DEX_RECOVER_BUNDLES);
 *                                  live:    confirmed on-chain and offered on the book;
 *                                  limit_cancelled: maker withdrew the order locally.
 * @property {number} createdAt     ms epoch.
 *
 * INVARIANTS (wState.dex_secrets):
 *  I1. Keys are normalized (lowercase, un-prefixed) 64-hex coin ids.
 *  I2. A record is written and persisted (saveState inside reserveAndLock) BEFORE the
 *      funding commit for its coin is broadcast — a crash can orphan a record, never a coin.
 *  I3. The raw `secret` field never leaves the worker except via DEX_GET_SECRET, which
 *      serves only records present in this map (i.e. our own), keyed by coin_id.
 *  I4. Records are deleted only on: refund success, explicit DEX_SECRET_CONSUMED after the
 *      maker's Base claim (preimage is then public anyway), or DEX_RECOVER_BUNDLES
 *      concluding the funding never landed. Ambiguity always keeps the record.
 *
 * INVARIANTS (wState.dex_cancelled):
 *  C1. Keys are normalized coin ids; value = { at, offerId }.
 *  C2. Purely LOCAL policy — cancellation never touches the chain. The covenant coin stays
 *      locked until its timeout; the map only stops us advertising/serving/auto-filling it.
 *  C3. Entries are removed when their coin's refund succeeds (record is then moot).
 */

/** @type {WalletState} */
let wState = {
    phrase: null,
    nextWotsIndex: 0,
    nextMssIndex: 0,
    wotsAddrs: {},
    spentWots: {},   // CRITICAL FIX: Tracks completely dead WOTS addresses to prevent dusting attacks
    pendingSpends: {}, // coin_id → timestamp. Coins locked by an in-flight commit/reveal. Prevents WOTS reuse across crashes.
    mssAddrs: {},
    utxos: {},
    history: [],
    lastScannedHeight: 0,
    vaultUtxo: null,
    l2_channels: {},
    l2_secrets: {},  // Stores preimages for invoices we generate
    l2_routes: {},   // Stores multi-hop routing map for Hubs
    dex_secrets: {},         // coin_id → DexSecretRecord (see typedef + invariants above)
    dex_cancelled: {},       // coin_id → { at, offerId } — locally-withdrawn limit orders
    pendingLimitBundles: {}, // groupId → funding recovery record (units are SECRET-FREE; secrets live in dex_secrets)
    annFragPool: {}          // MDXF fragment reassembly pool (persists across scans)
};

function getPrimaryMssPk() {
    if (!wallet) return null;
    const mssList = Object.keys(wState.mssAddrs);
    if (mssList.length === 0) return null;
    // Pin the L2 identity to ONE tree once chosen.
    //
    // `mssList[mssList.length - 1]` is insertion-ordered, so the identity used
    // to be whichever MSS address was derived most recently — it would silently
    // change the moment the wallet made another one. Peers derive the funding
    // covenant from this key, so a shift makes every inbound open to the old
    // identity fail the channel-id check, and it is invisible to the user.
    // Pinning keeps a published identity stable for the life of the wallet.
    let addr = wState.l2_identity_addr;
    if (!addr || !wState.mssAddrs[addr]) {
        addr = mssList[mssList.length - 1];
        wState.l2_identity_addr = addr;      // persisted by the next saveState()
    }
    return wallet.get_mss_pubkey(addr);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  IndexedDB: Full MSS Tree Storage
// ═══════════════════════════════════════════════════════════════════════════════
//
// Each MSS tree at height 10 is ~64 KB as a compact binary blob.
// IndexedDB can hold hundreds of MB — far beyond what we'll ever need.
// Trees are keyed by "mss_<hex_address>" for fast lookup.

/** @const {string} IndexedDB database name. */
const IDB_NAME = 'midstate_wallet';

/** @const {number} IndexedDB schema version. */
const IDB_VERSION = 1;

/** @const {string} IndexedDB object store for MSS trees. */
const IDB_STORE = 'mss_trees';

/**
 * Open (or create) the IndexedDB database.
 * Creates the `mss_trees` object store on first run.
 *
 * @returns {Promise<IDBDatabase>}
 */
function openIDB() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(IDB_NAME, IDB_VERSION);
        req.onupgradeneeded = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains(IDB_STORE)) {
                db.createObjectStore(IDB_STORE);
            }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

/**
 * Store a value in IndexedDB.
 *
 * @param {string} key - The storage key (e.g., "mss_<address>").
 * @param {*} value - The value to store (typically a Uint8Array).
 * @returns {Promise<void>}
 */
async function idbPut(key, value) {
    const db = await openIDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(IDB_STORE, 'readwrite');
        tx.objectStore(IDB_STORE).put(value, key);
        tx.oncomplete = () => { db.close(); resolve(); };
        tx.onerror = () => { db.close(); reject(tx.error); };
    });
}

/**
 * Retrieve a value from IndexedDB.
 *
 * @param {string} key - The storage key.
 * @returns {Promise<*|undefined>} The stored value, or undefined if not found.
 */
async function idbGet(key) {
    const db = await openIDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(IDB_STORE, 'readonly');
        const req = tx.objectStore(IDB_STORE).get(key);
        req.onsuccess = () => { db.close(); resolve(req.result); };
        req.onerror = () => { db.close(); reject(req.error); };
    });
}

/**
 * Delete a value from IndexedDB.
 *
 * @param {string} key - The storage key.
 * @returns {Promise<void>}
 */
async function idbDelete(key) {
    const db = await openIDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(IDB_STORE, 'readwrite');
        tx.objectStore(IDB_STORE).delete(key);
        tx.oncomplete = () => { db.close(); resolve(); };
        tx.onerror = () => { db.close(); reject(tx.error); };
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MSS Cache Management
// ═══════════════════════════════════════════════════════════════════════════════
//
// On login/import, we load all MSS trees from IndexedDB into the WASM wallet.
// If a tree isn't in IndexedDB (first run, upgrade from old FractionalMss),
// we regenerate it once and persist it. After that, loading is instant (~1ms).

/**
 * @type {boolean} Whether all MSS caches have been loaded into the WASM wallet.
 * Guards against redundant loading and ensures caches are ready before signing.
 */
let mssCachesReady = false;
let mssCachesLoading = null;

/**
 * Load all MSS trees from IndexedDB into the WASM wallet's in-memory cache.
 *
 * This is the critical function that eliminates the 15-minute delay. It:
 * 1. Checks if the tree is already in WASM memory (skip).
 * 2. Tries to load the binary blob from IndexedDB (~1ms per tree).
 * 3. Falls back to full regeneration if not found (one-time migration cost).
 *
 * Called automatically on LOGIN, IMPORT_CLI, and at the start of SCAN/SEND.
 * Idempotent — safe to call multiple times (guarded by `mssCachesReady`).
 *
 * @returns {Promise<void>}
 */
async function loadMssCaches() {
    if (!wallet || mssCachesReady) return;
    
    // Prevent race conditions if Sync and Send are clicked simultaneously
    if (mssCachesLoading) {
        await mssCachesLoading;
        return;
    }

    mssCachesLoading = (async () => {
        const entries = Object.entries(wState.mssAddrs);
        if (entries.length === 0) { mssCachesReady = true; return; }

        self.postMessage({ type: 'LOG', payload: `Loading ${entries.length} MSS tree(s) from storage...` });

        for (const [addrHex, mss] of entries) {
            try {
                if (wallet.has_mss_cache(addrHex)) {
                    wallet.set_mss_leaf_index(addrHex, mss.next_leaf);
                    continue;
                }

                const treeBytes = await idbGet(`mss_${addrHex}`);
                if (treeBytes) {
                    wallet.import_mss_bytes(addrHex, new Uint8Array(treeBytes));
                    wallet.set_mss_leaf_index(addrHex, mss.next_leaf);
                    continue;
                }

                // Regenerate fallback
                self.postMessage({ type: 'MSS_PROGRESS', payload: { current: 0, total: 100, label: "Regenerating MSS tree (one-time)..." } });
                const addr = wallet.get_mss_address(mss.index, mss.height, (current, total) => {
                    const now = Date.now();
                    if (now - lastMssUpdate > 66 || current === total) {
                        lastMssUpdate = now;
                        self.postMessage({ type: 'MSS_PROGRESS', payload: { current, total, label: `Regenerating MSS tree (${current}/${total})...` } });
                    }
                });

                const exportedBytes = wallet.export_mss_bytes(addr);
                await idbPut(`mss_${addr}`, exportedBytes);
                wallet.set_mss_leaf_index(addrHex, mss.next_leaf);
            } catch (e) {
                self.postMessage({ type: 'LOG', payload: `Warning: MSS load failed for ${addrHex.substring(0,12)}…: ${e}` });
            }
        }
        mssCachesReady = true;
    })();

    await mssCachesLoading;
    mssCachesLoading = null;
    self.postMessage({ type: 'LOG', payload: "All MSS trees loaded." });

    // One reconciliation pass against the node. Forward-only (max), so it can
    // never cause reuse — it only ever catches a counter that is BEHIND what the
    // chain has witnessed (another device, a lost local view).
    //
    // Deliberately NOT treated as authoritative: the node's HTTP /mss_state
    // scans only the last ~2000 blocks and answers 0 for older keys, while its
    // light/WebRTC path answers exactly from MSS_LEAF_INDEX_TABLE. The transport
    // is chosen at random on a flaky link, so a low answer proves nothing.
    reconcileMssLeavesWithNode().catch(() => {});
}

/** Pull the node's high-water leaf index for each MSS key and advance our
 *  counters (and the WASM trees) to at least that. Forward-only. */
async function reconcileMssLeavesWithNode() {
    for (const [addrHex, mss] of Object.entries(wState.mssAddrs || {})) {
        let pk = null;
        try { pk = wallet.get_mss_pubkey(addrHex); } catch (_) {}
        if (!pk) continue;
        try {
            const st = await rpc.getMssState(pk);
            if (st && Number.isFinite(st.next_index) && st.next_index > (mss.next_leaf || 0)) {
                self.postMessage({ type: 'LOG', payload: `Reconciled MSS leaf for ${addrHex.substring(0, 12)}…: ${mss.next_leaf || 0} → ${st.next_index} (chain/mempool witnessed more).` });
                mss.next_leaf = st.next_index;
                wallet.set_mss_leaf_index(addrHex, mss.next_leaf);
                await idbPut(`mss_${addrHex}`, wallet.export_mss_bytes(addrHex));
            }
        } catch (_) { /* offline → per-operation reconcile still guards */ }
    }
    await saveState();
}


/**
 * CRITICAL: Safe MSS Signing Helper.
 *
 * MSS/WOTS leaves are ONE-TIME keys: signing twice with the same leaf on two
 * different messages is a catastrophic key-reuse fault, and the node silently
 * evicts any reveal that reuses a burned leaf (mempool prune step 3). Two
 * counters have to agree for that never to happen:
 *
 *   • the WASM tree's internal `kp.next_leaf` (persisted inside the IndexedDB
 *     blob), which is what actually selects the signing leaf, and
 *   • `wState.mssAddrs[addr].next_leaf` (persisted in wState).
 *
 * They drift whenever the two are written at different times — and a rescan, an
 * import, or signing on another device leaves BOTH behind what the chain has
 * already witnessed. So we cross-check against the node before signing and take
 * the MAX of our view and its `getMssState.next_index`, pin BOTH counters to it,
 * sign, then persist the post-increment value atomically. Taking the max can
 * only ever skip forward, never reuse.
 *
 * The node is a CROSS-CHECK, NOT a source of truth. Its two transports disagree:
 * the light/WebRTC path answers from MSS_LEAF_INDEX_TABLE (O(1), full history —
 * correct), while HTTP /mss_state still scans only the last 2000 blocks and
 * returns 0 for any key untouched for ~33 h. We cannot tell which answered, so a
 * LOW answer must never be trusted — hence max(), never assignment. The locally
 * persisted counter remains the primary record; the node can only ever push it
 * forward.
 */
async function signMssAndSync(pkHex, commitmentHex) {
    const addrHex = compute_p2pk_address_hex(pkHex);
    const rec = wState.mssAddrs[addrHex];
    if (!rec) {
        // No local metadata for this key — sign without leaf bookkeeping we
        // can't do safely, rather than silently guessing a leaf.
        return wallet.sign_mss_hex(pkHex, commitmentHex);
    }

    // 1. Reconcile against the node (authoritative: scans chain AND mempool).
    let authoritative = rec.next_leaf || 0;
    try {
        const st = await rpc.getMssState(pkHex);
        if (st && Number.isFinite(st.next_index)) {
            authoritative = Math.max(authoritative, st.next_index);
        }
    } catch (_) { /* offline / flake → fall back to the local counter */ }

    // 2. Pin BOTH counters to the reconciled value before signing.
    if (authoritative !== rec.next_leaf) {
        rec.next_leaf = authoritative;
    }
    wallet.set_mss_leaf_index(addrHex, rec.next_leaf);

    // 3. Capacity guard (the WASM signer would throw, but check with a clear msg).
    if (rec.height && rec.next_leaf >= (1 << rec.height)) {
        throw new Error(`MSS key exhausted: all ${1 << rec.height} one-time signatures for ${addrHex.substring(0, 12)}… are used. Generate a new receiving address.`);
    }

    // 4. Sign (advances the WASM counter internally) and persist atomically.
    const sigHex = wallet.sign_mss_hex(pkHex, commitmentHex);
    rec.next_leaf++;
    await idbPut(`mss_${addrHex}`, wallet.export_mss_bytes(addrHex));
    await saveState();
    return sigHex;
}


// ═══════════════════════════════════════════════════════════════════════════════
//  RPC Bridge
// ═══════════════════════════════════════════════════════════════════════════════
//
// All network calls are proxied to the main thread (index.html) which owns the
// LightClient. RTCPeerConnection is not available in Web Workers, so WebRTC
// must live on the main thread. Each call posts an RPC_REQUEST and awaits the
// corresponding RPC_RESPONSE matched by a unique request id.

/** @type {number} Auto-incrementing RPC request ID. */
let _rpcNextId = 1;

/** @type {Map<number, {resolve: Function, reject: Function}>} Pending RPC promises. */
const _rpcPending = new Map();

/**
 * Handle an incoming RPC_RESPONSE from the main thread.
 *
 * @param {number} id - The request ID to match.
 * @param {*} result - The successful result (undefined if error).
 * @param {string} [error] - Error message (undefined if success).
 */
function _rpcReceive(id, result, error, code) {
    const p = _rpcPending.get(id);
    if (!p) return;
    _rpcPending.delete(id);
    if (error !== undefined) {
        const err = new Error(error);
        // 'RATE_LIMITED' / 'GATEWAY_UNAVAILABLE' from the main thread's
        // gatewayFetch — lets chain scans pause/abort wholesale instead of
        // grinding block-by-block through a rate-limit window.
        if (code) err.code = code;
        p.reject(err);
    }
    else p.resolve(result);
}

/**
 * Send an RPC request to the main thread and await the response.
 *
 * @param {string} method - The RPC method name.
 * @param {Object} [params] - Method parameters.
 * @returns {Promise<*>} The RPC response.
 * @throws {Error} On timeout (120s) or RPC error.
 */
function rpcCall(method, params) {
    return new Promise((resolve, reject) => {
        const id = _rpcNextId++;
        _rpcPending.set(id, { resolve, reject });
        self.postMessage({ type: 'RPC_REQUEST', payload: { id, method, params } });
        setTimeout(() => {
            if (_rpcPending.has(id)) {
                _rpcPending.delete(id);
                reject(new Error(`RPC timeout: ${method}`));
            }
        }, 120_000);
    });
}

/** @type {Object} Thin wrappers matching the shapes callers expect. */
const rpc = {
    getState:       ()           => rpcCall('getState'),
    getMempool:     ()           => rpcCall('getMempool'),
    getBlock:       (height)     => rpcCall('getBlock', { height }),
    getFilters:     (s, e)       => rpcCall('getFilters', { startHeight: s, endHeight: e }),
    getMssState:    (pk)         => rpcCall('getMssState', { masterPkHex: pk }),
    submitBatch:    (batch)      => rpcCall('submitBatch', { batch }),
    commit:         (c, n)       => rpcCall('commit', { commitmentHex: c, spamNonce: n }),
    send:           (reveal)     => rpcCall('send', { revealPayload: reveal }),
    checkCommitment: (commitment) => rpcCall('checkCommitment', { commitmentHex: commitment }),
    checkCoin:      (coin)       => rpcCall('checkCoin', { coinHex: coin }),
    sendChat:       (words, replyTo, attachments) => rpcCall('sendChat', { words, replyTo, attachments }),
    getChat:        ()           => rpcCall('getChat'),
    submitChat:     (sender, timestamp, nonce, replyTo, words, attachments) => rpcCall('submitChat', { sender, timestamp, nonce, replyTo, words, attachments }),
    
    /**
     * Get a block template for solo mining.
     * @param {Array<Object>} coinbase - Coinbase output specifications.
     * @returns {Promise<{ok: boolean, status: number, json: Function, text: Function}>}
     */
    async getBlockTemplate(coinbase) {
        const r = await rpcCall('getBlockTemplate', { coinbase });
        return {
            ok:     r.ok,
            status: r.status,
            json:   () => Promise.resolve(r.body),
            text:   () => Promise.resolve(typeof r.body === 'string' ? r.body : JSON.stringify(r.body))
        };
    },
};

/**
 * Fetch an inclusive range of blocks for a chain scan, one rpc.getBlock per
 * height. The main thread serves already-seen heights from its immutable
 * block cache and throttles anything that must go to the HTTPS gateway, so
 * no extra pacing is needed here (the old 20 ms/block sleep is gone).
 *
 * A genuinely missing or transiently failing block yields null — but gateway
 * rate-limiting (err.code === 'RATE_LIMITED' / 'GATEWAY_UNAVAILABLE') is
 * rethrown so the caller can pause or abort the WHOLE scan rather than grind
 * block-by-block through thousands of doomed requests.
 *
 * @param {number} lo - First height (inclusive).
 * @param {number} hi - Last height (inclusive).
 * @returns {Promise<Array<Object|null>>} blocks[i] is the block at lo + i.
 */
async function fetchScanBatch(lo, hi) {
    const out = [];
    for (let k = lo; k <= hi; k++) {
        try {
            out.push(await rpc.getBlock(k));
        } catch (e) {
            if (e && (e.code === 'RATE_LIMITED' || e.code === 'GATEWAY_UNAVAILABLE')) throw e;
            out.push(null);
        }
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  WASM Client-Side PoW for Chat & L2
// ═══════════════════════════════════════════════════════════════════════════════

async function submitClientMinedChat(words, replyTo, attachments) {
    const sender = getPrimaryMssPk() || "0000000000000000000000000000000000000000000000000000000000000000";
    const timestamp = Math.floor(Date.now() / 1000);
    
    self.postMessage({ type: 'LOG', payload: "Mining PoW locally for state update..." });
    await new Promise(r => setTimeout(r, 10)); // Yield to UI
    
    const nonce = Number(mine_chat_pow_v2_wasm(
        sender,
        BigInt(timestamp),
        JSON.stringify(replyTo !== undefined ? replyTo : null),
        JSON.stringify(words),
        JSON.stringify(attachments)
    ));
    
    // Fixed: Passing 'words' correctly to the RPC bridge!
    return await rpc.submitChat(sender, timestamp, nonce, replyTo, words, attachments);
}
    
// ═══════════════════════════════════════════════════════════════════════════════
//  Hex / Crypto Utilities
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Normalize data to a lowercase hex string.
 *
 * Handles strings, Uint8Arrays, and regular arrays.
 *
 * @param {string|Uint8Array|Array<number>|null} data
 * @returns {string} Lowercase hex string, or empty string if input is falsy.
 */
function normalizeHex(data) {
    if (!data) return "";
    if (typeof data === 'string') return data.toLowerCase();
    if (Array.isArray(data) || data instanceof Uint8Array) {
        return Array.from(data).map(b => b.toString(16).padStart(2, '0')).join('').toLowerCase();
    }
    return "";
}

// ── On-chain DEX limit-order BUNDLE announcement codec ──────────────────────
// Published as a 0-value DataBurn so any taker can discover standing orders by
// scanning the chain (no chat dependency, survives node restarts, maker offline).
// Layout (big-endian): MAGIC(4) VER(1) makerEvmAddr(20) makerMdsPk(32)
//   timeoutHeight(8) groupId(6) unitCount(1)  then per unit: H(32) salt(32)
//   valueExp(1) weiAmount(16)  = 81 bytes/unit. Recovery of covAddr/coin_id is
//   recomputed by the reader from H+value+timeout+makerMdsPk, so it isn't stored.
const ANN_MAGIC = "4d445841"; // "MDXA"
const ANN_VER = 1;
function _annHb(h){ h=(h||"").replace(/^0x/,''); const a=new Uint8Array(h.length/2); for(let i=0;i<a.length;i++) a[i]=parseInt(h.substr(i*2,2),16); return a; }
function _annU(n,bytes){ const a=new Uint8Array(bytes); let v=BigInt(n); for(let i=bytes-1;i>=0;i--){a[i]=Number(v&0xffn);v>>=8n;} return a; }
function _annRd(a,o,n){ let v=0n; for(let i=0;i<n;i++) v=(v<<8n)|BigInt(a[o+i]); return v; }
function _annLog2(v){ let n=BigInt(v),e=0; if(n<=0n) throw new Error("value<=0"); while(n>1n){ if(n&1n) throw new Error("value not power of two"); n>>=1n; e++; } return e; }
// Node-side DataBurn payload cap (types.rs MAX_BURN_DATA_SIZE = 80, an OP_RETURN analog).
// CONSENSUS — DO NOT RAISE. A self-contained MDXA announcement is 72B header + 81B/unit, so
// it can NEVER fit one burn. Instead the wallet splits each announcement into MDXF fragments
// — 12B header (magic4 + groupId6 + idx1 + total1) + up to 68B of MDXA bytes — and ships ALL
// fragments as separate 0-value burns inside the SAME funding tx, so they land in one block
// and reassemble trivially. Multiple burns per tx are already legal; consensus is untouched.
const NODE_MAX_BURN_BYTES = 80;
const FRAG_MAGIC = "4d445846";      // "MDXF"
const FRAG_HEADER_BYTES = 12;       // magic4 + groupId6 + idx1 + total1
const FRAG_PAYLOAD_BYTES = NODE_MAX_BURN_BYTES - FRAG_HEADER_BYTES; // 68

// Split a full MDXA announcement (hex) into MDXF fragment burns (hex[]), each <= 80 bytes.
function fragmentAnnouncement(annHex, groupId) {
    const g6 = normalizeHex(_annHb((groupId || "").padStart(12, '0')).slice(0, 6)); // same slice as encodeAnnouncement
    const body = annHex.toLowerCase();
    const step = FRAG_PAYLOAD_BYTES * 2;
    const total = Math.max(1, Math.ceil(body.length / step));
    if (total > 255) throw new Error("Announcement too large: > 255 fragments");
    const frags = [];
    for (let i = 0; i < total; i++) {
        frags.push(FRAG_MAGIC + g6 + i.toString(16).padStart(2, '0') + total.toString(16).padStart(2, '0') + body.slice(i * step, (i + 1) * step));
    }
    return frags;
}

// Parse one MDXF fragment (hex) -> { key, idx, total, chunk } or null.
function tryParseFragment(hex) {
    if (typeof hex !== 'string') return null;
    hex = hex.replace(/^0x/, '').toLowerCase();
    if (hex.length < FRAG_HEADER_BYTES * 2 + 2 || hex.slice(0, 8) !== FRAG_MAGIC) return null;
    const groupId = hex.slice(8, 20);
    const idx = parseInt(hex.slice(20, 22), 16);
    const total = parseInt(hex.slice(22, 24), 16);
    if (!total || idx >= total) return null;
    return { key: groupId + ':' + total, idx, total, chunk: hex.slice(24) };
}

// Pull every DataBurn payload out of a block object, whatever shape serde gave it.
// LightRequest::GetBlock serializes core types with derive(Serialize): externally-tagged
// enums with Vec<u8> as JSON NUMBER ARRAYS — {"DataBurn":{"payload":[77,68,...],...}} —
// which no hex-run regex can ever match. So we walk the object tree and accept
// number-array OR hex-string payloads (normalizeHex handles both).
function extractBurnPayloadHexes(node, out) {
    if (!node || typeof node !== 'object') return out;
    if (node.DataBurn && node.DataBurn.payload !== undefined) out.push(normalizeHex(node.DataBurn.payload));
    if (node.type === 'data_burn' && node.payload !== undefined) out.push(normalizeHex(node.payload));
    for (const k in node) {
        const v = node[k];
        if (v && typeof v === 'object') extractBurnPayloadHexes(v, out);
    }
    return out;
}


function encodeAnnouncement({ makerEvmAddr, makerMdsPk, timeoutHeight, groupId, units }) {
    if (!units.length || units.length > 255) throw new Error("unit count out of range");
    const parts = [ _annHb(ANN_MAGIC), new Uint8Array([ANN_VER]), _annHb(makerEvmAddr), _annHb(makerMdsPk),
                    _annU(timeoutHeight,8), _annHb((groupId||"").padStart(12,'0')).slice(0,6), new Uint8Array([units.length]) ];
    for (const u of units) parts.push(_annHb(u.secretHash), _annHb(u.salt), new Uint8Array([_annLog2(u.value)]), _annU(u.weiAmount,16));
    const len = parts.reduce((s,p)=>s+p.length,0), out = new Uint8Array(len); let off=0;
    for (const p of parts){ out.set(p,off); off+=p.length; }
    return normalizeHex(out);
}

// ── TAKER-LOCK ANNOUNCEMENT (MDXT) ──────────────────────────────────────────
// Mirror of the maker MDXA announcement, but for the TAKER side of a swap (someone
// who locked MDS to fill a buy offer — the shape that had no on-chain breadcrumb and
// so couldn't be recovered after a data-clear, i.e. the Alice case). Publishing this
// makes a taker lock recoverable from seed alone, exactly like a maker order.
//
// SAFETY: carries secretHASH, never the secret preimage. The hash is already public
// (it's the Base contract hashlock and appears in the covenant script), and BLAKE3 is
// one-way, so an observer learns nothing that enables theft. The taker's REFUND path
// needs only their own signature + the timeout — never the secret. So everything here
// is safe to publish, and only the taker (holding the refund key) can act on it.
const ANN_T_MAGIC = "4d445854"; // "MDXT"
const ANN_T_VER = 1;
function encodeTakerAnnouncement({ takerMdsPk, secretHash, salt, receiverAddr, timeoutHeight, value, weiAmount }) {
    const parts = [
        _annHb(ANN_T_MAGIC), new Uint8Array([ANN_T_VER]),
        _annHb(takerMdsPk),           // 32 — refund key; how the taker recognises their own lock
        _annHb(secretHash),           // 32 — safe (already the public hashlock)
        _annHb(salt),                 // 32 — the coin salt (the thing localStorage loses)
        _annHb(receiverAddr),         // 32 — buyer's receiving address (needed to rebuild covAddr)
        _annU(timeoutHeight, 8),
        new Uint8Array([_annLog2(value)]),
        _annU(weiAmount, 16)
    ];
    const len = parts.reduce((s, p) => s + p.length, 0), out = new Uint8Array(len); let off = 0;
    for (const p of parts) { out.set(p, off); off += p.length; }
    return normalizeHex(out);
}
function tryDecodeTakerAnnouncement(hex) {
    if (typeof hex !== 'string') return null;
    hex = hex.replace(/^0x/, '').toLowerCase();
    if (!/^[0-9a-f]+$/.test(hex) || hex.slice(0, 8) !== ANN_T_MAGIC) return null;
    const a = _annHb(hex); let o = 4;
    if (a[o] !== ANN_T_VER) return null; o += 1;
    if (a.length < 4 + 1 + 32 + 32 + 32 + 32 + 8 + 1 + 16) return null;
    const takerMdsPk = normalizeHex(a.slice(o, o + 32)); o += 32;
    const secretHash = normalizeHex(a.slice(o, o + 32)); o += 32;
    const salt = normalizeHex(a.slice(o, o + 32)); o += 32;
    const receiverAddr = normalizeHex(a.slice(o, o + 32)); o += 32;
    const timeoutHeight = Number(_annRd(a, o, 8)); o += 8;
    const value = Number(1n << BigInt(a[o])); o += 1;
    const weiAmount = _annRd(a, o, 16).toString(); o += 16;
    return { takerMdsPk, secretHash, salt, receiverAddr, timeoutHeight, value, weiAmount };
}

function tryDecodeAnnouncement(hex) {
    if (typeof hex !== 'string') return null;
    hex = hex.replace(/^0x/,'').toLowerCase();
    if (!/^[0-9a-f]+$/.test(hex) || hex.length < 144 || hex.slice(0,8) !== ANN_MAGIC) return null;
    const a = _annHb(hex); let o = 4;
    if (a[o] !== ANN_VER) return null; o += 1;
    const makerEvmAddr = '0x'+normalizeHex(a.slice(o,o+20)); o += 20;
    const makerMdsPk = normalizeHex(a.slice(o,o+32)); o += 32;
    const timeoutHeight = Number(_annRd(a,o,8)); o += 8;
    const groupId = normalizeHex(a.slice(o,o+6)); o += 6;
    const count = a[o]; o += 1;
    const units = [];
    for (let i=0;i<count;i++){
        if (o + 81 > a.length) return null;
        const secretHash = normalizeHex(a.slice(o,o+32)); o += 32;
        const salt = normalizeHex(a.slice(o,o+32)); o += 32;
        const value = Number(1n << BigInt(a[o])); o += 1;
        const weiAmount = _annRd(a,o,16).toString(); o += 16;
        units.push({ secretHash, salt, value, weiAmount });
    }
    return { makerEvmAddr, makerMdsPk, timeoutHeight, groupId, units };
}

/**
 * Derive an AES-GCM-256 key from a password and salt using PBKDF2.
 *
 * @param {string} pwd - The password.
 * @param {Uint8Array} salt - 16-byte random salt.
 * @returns {Promise<CryptoKey>}
 * @throws {Error} If Web Crypto API is unavailable (non-HTTPS context).
 */
async function deriveCryptoKey(pwd, salt) {
    if (!self.crypto || !self.crypto.subtle) {
        throw new Error("Cryptography unavailable: This wallet requires a secure (HTTPS) connection.");
    }
    const enc = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey("raw", enc.encode(pwd), { name: "PBKDF2" }, false, ["deriveKey"]);
    return await crypto.subtle.deriveKey(
        { name: "PBKDF2", salt: salt, iterations: 100000, hash: "SHA-256" },
        keyMaterial, { name: "AES-GCM", length: 256 }, false, ["encrypt", "decrypt"]
    );
}

/**
 * Encrypt and save the wallet state to localStorage (via main thread).
 *
 * The state JSON is encrypted with AES-GCM-256, using a key derived from
 * the user's password via PBKDF2-SHA256 (100k iterations).
 *
 * Note: MSS trees are NOT included — they live in IndexedDB. Only lightweight
 * metadata (address, height, next_leaf) is saved here.
 *
 * @returns {Promise<void>}
 */
async function saveState() {
    if (!password) return;
    const salt = crypto.getRandomValues(new Uint8Array(16));
    const iv   = crypto.getRandomValues(new Uint8Array(12));
    const key  = await deriveCryptoKey(password, salt);
    const enc  = new TextEncoder();

    // Strip any legacy fractional_data before saving — trees live in IndexedDB now.
    // This ensures wallets upgraded from the FractionalMss era don't bloat localStorage.
    const cleanState = JSON.parse(JSON.stringify(wState));
    for (const addr of Object.keys(cleanState.mssAddrs)) {
        delete cleanState.mssAddrs[addr].fractional_data;
    }

    const encrypted = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, enc.encode(JSON.stringify(cleanState)));
    const bundle = {
        salt: normalizeHex(salt),
        iv:   normalizeHex(iv),
        data: normalizeHex(new Uint8Array(encrypted))
    };
    self.postMessage({ type: 'SAVE_WALLET', payload: JSON.stringify(bundle) });
}

// ── MSS leaf-reuse recovery ─────────────────────────────────────────────────
// A reveal signs its wallet inputs with one-time MSS leaves. The leaf floor we
// pick comes from getMssState, which can be stale: the queried peer may lag the
// tip, or rapid back-to-back txs can outrun the counter before it updates. When
// that happens the network rejects the reveal with "MSS leaf <pk> already spent".
//
// Crucially the leaf is a *witness*, not part of the commitment — the commitment
// is H(input_coin_ids, output_hashes, salt) and does not depend on which leaf
// signs it. So we can advance to a fresh leaf and re-sign against the SAME,
// already-mined commitment. No re-commit, no new PoW: just rebuild the reveal
// with a higher leaf and resend. Each retry re-queries getMssState (in case the
// peer has since caught up) and also steps forward locally (in case it hasn't).
//
// Works for both reveal builders: covenant/funding reveals (build_script_reveal,
// inputs under `wallet_inputs`) and ordinary sends (build_reveal, inputs under
// `selected_inputs`). Callers pass a `rebuild(ctxStr)` for the send path; funding
// callers can omit it and get build_script_reveal by default.
async function sendRevealWithMssLeafRetry(prebuiltPayload, ctxStrOrObj, commitment, txSalt, phase, rebuild) {
    const MAX_RETRIES = 6;
    const STEP = 4; // leaves to skip per retry when getMssState is still behind
    const ctxObj = (typeof ctxStrOrObj === 'string')
        ? JSON.parse(ctxStrOrObj)
        : JSON.parse(JSON.stringify(ctxStrOrObj));
    // Funding contexts name their inputs `wallet_inputs`; spend contexts use `selected_inputs`.
    const inputs = Array.isArray(ctxObj.wallet_inputs) ? ctxObj.wallet_inputs
        : Array.isArray(ctxObj.selected_inputs) ? ctxObj.selected_inputs : [];
    const mssAddrs = [...new Set(inputs.filter(i => i.is_mss).map(i => i.address))];
    // Default rebuild = the covenant/funding path; the send path passes build_reveal.
    const rebuildFn = rebuild || ((cs) => wallet.build_script_reveal(cs, commitment, txSalt));
    let payloadStr = prebuiltPayload;

    for (let attempt = 0; ; attempt++) {
        const res = await rpc.send(payloadStr);
        // Report the payload that was actually sent. A leaf-reuse retry re-signs
        // with a FRESH leaf, so the caller must persist this exact string — a
        // later rebroadcast of the original would carry a spent leaf and fail.
        res.revealPayload = payloadStr;
        if (res.ok) return res;

        const msg = String(res.body || res.error || '');
        const leafReuse = /leaf\s+[0-9a-fA-F]+\s+already spent/i.test(msg);
        // Only self-heal leaf reuse; anything else (or out of retries) surfaces as-is.
        if (!leafReuse || attempt >= MAX_RETRIES || mssAddrs.length === 0) return res;

        for (const addr of mssAddrs) {
            const cur = (wState.mssAddrs[addr] && wState.mssAddrs[addr].next_leaf) || 0;
            let useLeaf = cur + STEP;
            try {
                const st = await rpc.getMssState(addr);
                if (st && (st.next_index + STEP) > useLeaf) useLeaf = st.next_index + STEP;
            } catch (e) { /* keep the local step-forward */ }
            // Re-tag this address's inputs (one leaf per address per tx) and advance
            // the persistent counter past it.
            for (const inp of inputs) {
                if (inp.is_mss && inp.address === addr) inp.mss_leaf = useLeaf;
            }
            if (wState.mssAddrs[addr]) wState.mssAddrs[addr].next_leaf = useLeaf + 1;
            wallet.set_mss_leaf_index(addr, useLeaf + 1);
        }
        await saveState();
        if (phase) phase(`MSS leaf reuse detected — re-signing with a fresh key (retry ${attempt + 1}/${MAX_RETRIES})\u2026`);
        // Same commitment, fresh leaves: rebuild the reveal only.
        payloadStr = rebuildFn(JSON.stringify(ctxObj));
    }
}

/**
 * Decrypt wallet state from a localStorage bundle and initialize the wallet.
 *
 * After decrypting, this function:
 * 1. Migrates legacy state formats (array UTXOs, missing history).
 * 2. Restores embedded MSS trees from backup to IndexedDB (if present).
 * 3. Constructs a new WebWallet from the mnemonic.
 * 4. Loads all MSS trees from IndexedDB into WASM.
 * 5. Posts WALLET_LOADED to the UI.
 *
 * Backups created with EXPORT_BACKUP include full MSS trees in `_mss_trees`,
 * which are written to IndexedDB before `loadMssCaches()` runs. This means
 * importing a complete backup on a new browser is instant — no regeneration.
 * Old backups without `_mss_trees` fall back to one-time regeneration.
 *
 * @param {string} pwd - The user's password.
 * @param {string} bundleStr - The encrypted JSON bundle from localStorage.
 * @returns {Promise<void>}
 * @throws {Error} If the password is wrong or the data is corrupted.
 */
async function loadState(pwd, bundleStr) {
    if (!bundleStr) throw new Error("No wallet found");
    const bundle = JSON.parse(bundleStr);
    const parseHexArray = (h) => new Uint8Array((h || "").match(/.{1,2}/g)?.map(b => parseInt(b, 16)) || []);
    const salt = parseHexArray(bundle.salt);
    const iv   = parseHexArray(bundle.iv);
    const data = parseHexArray(bundle.data);
    const key  = await deriveCryptoKey(pwd, salt);
    try {
        const decrypted = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, data);
        const loadedState = JSON.parse(new TextDecoder().decode(decrypted));
        wState = loadedState;

        // Ensure L2 structures exist for upgraded wallets
        wState.l2_channels = wState.l2_channels || {};
        wState.l2_secrets = wState.l2_secrets || {};
        wState.l2_routes = wState.l2_routes || {};
        wState.dex_swaps = wState.dex_swaps || {};
        wState.spentWots = wState.spentWots || {};
        wState.pendingSpends = wState.pendingSpends || {};
        // DEX limit-order structures (see the DexSecretRecord typedef for invariants).
        wState.dex_secrets = wState.dex_secrets || {};
        wState.dex_cancelled = wState.dex_cancelled || {};
        wState.pendingLimitBundles = wState.pendingLimitBundles || {};
        wState.annFragPool = wState.annFragPool || {};
        
        // Migrate legacy array-format UTXOs to map format
        if (Array.isArray(wState.utxos)) {
            const utxoMap = {};
            for (const u of wState.utxos) utxoMap[u.coin_id] = u;
            wState.utxos = utxoMap;
        }

        // Migrate wallets that pre-date the history feature
        if (wState.history === undefined) {
            self.postMessage({ type: 'LOG', payload: "Legacy backup detected. Re-indexing chain to rebuild transaction history..." });
            wState.history = [];
            if (wState.lastScannedHeight > 0) { wState.lastScannedHeight = 0; wState.utxos = {}; }
        }

        // Restore embedded MSS trees to IndexedDB (from complete backups)
        if (wState._mss_trees) {
            for (const [addr, hexBytes] of Object.entries(wState._mss_trees)) {
                const bytes = new Uint8Array(hexBytes.match(/.{1,2}/g).map(b => parseInt(b, 16)));
                await idbPut(`mss_${addr}`, bytes);
                self.postMessage({ type: 'LOG', payload: `Restored MSS tree ${addr.substring(0,12)}… from backup.` });
            }
            delete wState._mss_trees; // Don't keep in runtime state or re-save to localStorage
        }

        password = pwd;
        if (wState.phrase) {
            wallet = new WebWallet(wState.phrase);
        } else if (wState.master_seed) {
            wallet = WebWallet.from_seed_hex(wState.master_seed);
        } else {
            throw new Error("Corrupted wallet: No seed phrase or master seed found.");
        }

        // Load MSS trees from IndexedDB into WASM (instant if previously cached or just restored)
        mssCachesReady = false;
        await loadMssCaches();

        self.postMessage({ type: 'WALLET_LOADED', payload: buildDashboardPayload() });
    } catch(e) {
        throw new Error("Incorrect password or corrupted wallet file");
    }
}



// ═══════════════════════════════════════════════════════════════════════════════
//  Mining
// ═══════════════════════════════════════════════════════════════════════════════



/**
 * Request and return a mining template from the network.
 *
 * Handles auto-syncing if the chain has advanced, mempool stats, and
 * coinbase construction. The returned template contains everything the
 * miner workers need: midstate, target, and the full batch to submit.
 *
 * @returns {Promise<Object|null>} The mining template, or null on failure.
 * @throws {Error} If the wallet is not initialized.
 */
async function handleGetTemplate() {
    if (!wallet) throw new Error("Wallet not initialized.");
    if (isFetchingTemplate) return null;
    isFetchingTemplate = true;

    try {
        // Update dashboard stats regardless of mode
        const stateObj = await rpc.getState().catch(() => ({ height: networkHeight, block_reward: 0 }));
        let mempoolTxs = 0, mempoolFees = 0;
        try {
            const mempool = await rpc.getMempool();
            mempoolTxs = mempool.size || 0;
            mempoolFees = (mempool.transactions || []).reduce((s, tx) => s + (tx.fee || 0), 0);
        } catch (e) {}

        networkHeight = stateObj.height || networkHeight;
        mempoolSize = mempoolTxs;
        self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });

        if (stateObj.height > wState.lastScannedHeight) {
            self.postMessage({ type: 'LOG', payload: "Chain advanced! Auto-syncing..." });
            await performScan();
        }

        // --- POOL MODE LOGIC ---
        if (miningMode === 'pool') {
            if (!poolUrl) throw new Error("Pool URL not set.");
            const cleanUrl = poolUrl.replace(/\/+$/, '');
            const res = await fetch(`${cleanUrl}/api/template`, { cache: 'no-store' });
            if (!res.ok) throw new Error(`Pool error: ${await res.text()}`);
            
            const tmpl = await res.json();
            const txCount = tmpl.batch_template.transactions?.length || 0;
            self.postMessage({ type: 'LOG', payload: `Pool Template | ${txCount} txs | Your Shares: ${tmpl.shares_recorded}` });

            return {
                mining_midstate: tmpl.mining_midstate,
                target:          tmpl.target,
                batch_template:  tmpl.batch_template,
                mining_addrs:    [], // Pool handles keys
                next_wots_index: wState.nextWotsIndex, // We don't advance our counter
                total_fees:      0, // Pool handles fees
                chainHeight:     networkHeight,
                blockReward:     stateObj.block_reward || 0,
                mempoolTxs,
                mempoolFees
            };
        }
        // --- END POOL MODE LOGIC ---

        // Solo Mode Logic (Original)
        const template = await buildMiningTemplate(stateObj);
        if (!template) return null;

        const txCount = template.batch_template.transactions?.length || 0;
        self.postMessage({ type: 'LOG', payload: `Solo Template at height ${stateObj.height} | ${txCount} txs | fees: ${template.total_fees}` });

        return {
            mining_midstate: template.mining_midstate,
            target:          template.target,
            batch_template:  template.batch_template,
            mining_addrs:    template.mining_addrs,
            next_wots_index: template.next_wots_index,
            total_fees:      template.total_fees,
            chainHeight:     stateObj.height,
            blockReward:     stateObj.block_reward || 0,
            mempoolTxs,
            mempoolFees
        };
    } finally {
        isFetchingTemplate = false;
    }
}

/**
 * Submit a mined block to the network.
 *
 * @param {Object} template - The mining template from handleGetTemplate().
 * @param {string} nonce - The winning nonce as a string (BigInt-compatible).
 * @returns {Promise<Object>} Submission result with `accepted`, `rejectReason`, etc.
 */
async function handleSubmitMinedBlock(template, nonce) {
    if (!wallet) throw new Error("Wallet not initialized.");
    if (isSubmitting) {
        self.postMessage({ type: 'LOG', payload: 'Duplicate block find ignored — submission already in progress.' });
        return { accepted: false, rejectReason: 'duplicate', reward: 0, height: template.chainHeight };
    }
    isSubmitting = true;
    try {
        const extStr = wallet.build_solo_extension(template.mining_midstate, BigInt(nonce));
        if (!extStr) throw new Error("Failed to recompute extension hash.");

        const batch = JSON.parse(JSON.stringify(template.batch_template));
        batch.extension = JSON.parse(extStr);

        let accepted = false;
        let rejectReason = null;

        // --- POOL MODE SUBMISSION ---
        if (miningMode === 'pool') {
            payoutAddress = getPrimaryMssPk() || "";
            if (!payoutAddress) {
                throw new Error("No L2/MSS Identity found! Generate a new address first to receive pool payouts.");
            }

            const cleanUrl = poolUrl.replace(/\/+$/, '');
            const res = await fetch(`${cleanUrl}/api/submit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ batch, payout_address: payoutAddress })
            });
            
            accepted = res.ok;
            if (accepted) {
                const respJson = await res.json();
                self.postMessage({ type: 'LOG', payload: `✅ Pool: ${respJson.message}` });
            } else {
                rejectReason = await res.text();
                self.postMessage({ type: 'LOG', payload: `❌ Pool Rejected: ${rejectReason}` });
            }
        } 
        // --- SOLO MODE SUBMISSION ---
        else {
            const submitReq = await rpc.submitBatch(batch);
            accepted = submitReq.ok;
            rejectReason = accepted ? null : (submitReq.body || 'rejected');

            if (accepted) {
                for (const entry of template.mining_addrs) wState.wotsAddrs[entry.address] = entry.index;
                wState.nextWotsIndex = template.next_wots_index;
                self.postMessage({ type: 'LOG', payload: `✅ Solo Block accepted! Height: ${template.chainHeight}` });
                await saveState();
                await performScan();
            } else {
                self.postMessage({ type: 'LOG', payload: `❌ Solo Block rejected: ${rejectReason}` });
            }
        }

        const finalHashHex = Array.from(batch.extension.final_hash).map(b => b.toString(16).padStart(2, '0')).join('');
        return {
            accepted, rejectReason,
            reward:    (template.blockReward || 0) + (template.total_fees || 0),
            height:    template.chainHeight,
            finalHash: finalHashHex,
            timestamp: batch.timestamp,
            txCount:   batch.transactions?.length || 0,
            fees:      template.total_fees || 0
        };
    } finally {
        isSubmitting = false;
    }
}

/**
 * Build a mining template with proper coinbase outputs.
 *
 * Retries up to 3 times if fees change between coinbase construction
 * and template validation (409 conflict).
 *
 * @param {Object} stateObj - Chain state from rpc.getState().
 * @returns {Promise<Object|null>} The template or null on failure.
 */
async function buildMiningTemplate(stateObj) {
    const MAX_RETRIES = 3;
    let totalValue = stateObj.block_reward;

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        const cbStr = wallet.build_coinbase(BigInt(totalValue), wState.nextWotsIndex);
        if (!cbStr) { self.postMessage({ type: 'ERROR', payload: "Failed to build coinbase outputs." }); return null; }
        const cbData = JSON.parse(cbStr);

        const resp = await rpc.getBlockTemplate(cbData.coinbase);

        if (resp.ok) {
            const tmpl = await resp.json();
            tmpl.mining_addrs    = cbData.mining_addrs;
            tmpl.next_wots_index = cbData.next_wots_index;
            return tmpl;
        }

        // Handle fee mismatch — retry with corrected total
        try {
            let err = await resp.json();
            if (typeof err === 'string') err = JSON.parse(err);
            if (err.expected_total) {
                self.postMessage({ type: 'LOG', payload: `Fees changed (${totalValue} → ${err.expected_total}). Rebuilding coinbase...` });
                totalValue = err.expected_total;
                continue;
            }
        } catch (e) {}

        // Unknown error — bail
        const errText = await resp.text();
        self.postMessage({ type: 'ERROR', payload: `Template error: ${errText}` });
        return null;
    }

    self.postMessage({ type: 'ERROR', payload: "Failed to build template after retries." });
    return null;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Message Handler
// ═══════════════════════════════════════════════════════════════════════════════

self.onmessage = async (e) => {
    const { type, payload } = e.data;
    try {
        if (type === 'INIT') {
            await init();
            self.postMessage({ type: 'INIT_DONE' });
        }

        else if (type === 'RPC_RESPONSE') {
            _rpcReceive(payload.id, payload.result, payload.error, payload.code);
        }

        else if (type === 'GENERATE') {
            self.postMessage({ type: 'PHRASE_GENERATED', payload: generate_phrase() });
        }
        else if (type === 'DEX_BROADCAST_OFFER') {
            // The worker is the single source of truth for our MDS identity. Inject it
            // here so the UI never has to call getPrimaryMssPk() (which lives only here).
            const full = { ...payload, makerMdsPk: getPrimaryMssPk() };
            const jsonBytes = new TextEncoder().encode(JSON.stringify(full));
            submitClientMinedChat([255, 200], null, [
                { kind: "signature", value: normalizeHex(jsonBytes) }
            ]).catch(()=>{});
        }
        else if (type === 'DEX_BROADCAST_BIDFILL') {
            const jsonBytes = new TextEncoder().encode(JSON.stringify(payload));
            submitClientMinedChat([255, 205], null, [
                { kind: "signature", value: normalizeHex(jsonBytes) }
            ]).catch(()=>{});
        }
        else if (type === 'DEX_BROADCAST_BIDSECRET') {
            const jsonBytes = new TextEncoder().encode(JSON.stringify(payload));
            submitClientMinedChat([255, 206], null, [
                { kind: "signature", value: normalizeHex(jsonBytes) }
            ]).catch(()=>{});
        }
        else if (type === 'DEX_BROADCAST_ACCEPT') {
            // Taker side. We send ONLY our identity + EVM address — never any secret or
            // secret hash. The maker generates the secret (see DEX_LOCK_MIDSTATE).
            const full = { offerId: payload.offerId, takerMdsPk: getPrimaryMssPk(), takerEvmAddr: payload.takerEvmAddr };
            const jsonBytes = new TextEncoder().encode(JSON.stringify(full));
            submitClientMinedChat([255, 201], payload.offerNonce ?? null, [
                { kind: "signature", value: normalizeHex(jsonBytes) }
            ]).catch(()=>{});
        }
        else if (type === 'DEX_BROADCAST_LOCKED') {
            // Maker → taker. Carries the hashlock H, the funded HTLC coin set, the
            // timeout height and the maker's pubkey. NEVER the secret.
            const jsonBytes = new TextEncoder().encode(JSON.stringify(payload));
            submitClientMinedChat([255, 202], payload.replyTo ?? null, [
                { kind: "signature", value: normalizeHex(jsonBytes) }
            ]).catch(()=>{});
        }
        else if (type === 'DEX_BROADCAST_LOCKING') {
            // Tiny notice (cmd 203) so the counterparty knows we've begun the on-chain
            // MDS lock. Carries only the offerId — no secret, no hash, no coins.
            const jsonBytes = new TextEncoder().encode(JSON.stringify({ offerId: payload.offerId }));
            submitClientMinedChat([255, 203], null, [
                { kind: "signature", value: normalizeHex(jsonBytes) }
            ]).catch(()=>{});
        }
        else if (type === 'DEX_BROADCAST_SUBMARINE') {
            // FEATURE 2, taker → maker (cmd 204): "I'm taking this limit unit via an L2
            // submarine swap — here is my L2 identity and the NEW invoice hashlock."
            // Keyed on the covenant coinId: the one identifier BOTH sides always share.
            // (Maker unit offerIds are random 8-byte hex; a taker who discovered the order
            // by chain scan only has 'chain:<coinId…>' — offerId alone would never match.)
            // Carries no secret — only its hash. The maker routes MDS against this hash
            // ONLY after seeing real ETH locked under it on Base, so spoofing this message
            // costs an attacker an actual ETH lock, which just completes the swap honestly.
            const full = {
                coinId: payload.coinId,
                offerId: payload.offerId,
                takerL2Pk: getPrimaryMssPk(),
                takerEvmAddr: payload.takerEvmAddr,
                submarineHash: payload.secretHash
            };
            const jsonBytes = new TextEncoder().encode(JSON.stringify(full));
            submitClientMinedChat([255, 204], null, [
                { kind: "signature", value: normalizeHex(jsonBytes) }
            ]).catch(()=>{});
        }
        else if (type === 'DEX_LOCK_MIDSTATE') {
            await acquireSendLock();
            try {
                const { offerId, expectedAmount, takerPk, swapMode } = payload;
                const covenant = (swapMode === 'covenant');

                // ── The MAKER generates the secret ──────────────────────────────
                // Both chains hash-lock on H = BLAKE3(secret). The maker reveals the
                // secret by claiming the ETH on Base; the taker then reads it and
                // claims the MDS here. Because both sides use the SAME BLAKE3 hash,
                // a single preimage unlocks both — no cross-hash theft vector.
                const secretBytes = crypto.getRandomValues(new Uint8Array(32));
                const rawSecret = Array.from(secretBytes).map(b => b.toString(16).padStart(2, '0')).join('');
                const secretHash = blake3_hash_hex(rawSecret); // BLAKE3 of the 32 secret bytes

                const myPk = getPrimaryMssPk();
                const timeoutHeight = networkHeight + 1440; // ~24h at 1-minute blocks (the LONG leg)

                let htlcScriptHex, htlcAddressHex, fundAmount;
                if (covenant) {
                    // COVENANT HTLC: the claim path needs NO signature — it instead forces
                    // the spend to pay >= minPayout to the BUYER's address. So anyone (the
                    // seller, or the buyer themselves) can deliver the MDS once the secret
                    // is public, and the buyer needs no MDS. receiver = the ETH-side's
                    // RECEIVING ADDRESS (= compute_p2pk_address of their pubkey).
                    const buyerAddr = compute_p2pk_address_hex(normalizeHex(takerPk));
                    htlcScriptHex = build_covenant_htlc_bytecode_hex(
                        secretHash, buyerAddr, BigInt(expectedAmount), BigInt(timeoutHeight), myPk
                    );
                    htlcAddressHex = blake3_hash_hex(htlcScriptHex);
                    // Over-fund by the fee budget so the DELIVERY fee comes from the locked
                    // value (keeps the buyer's self-deliver fallback affordable with 0 MDS).
                    fundAmount = Number(expectedAmount) + COVENANT_FEE_BUDGET;
                } else {
                    // Classic HTLC: receiver = taker (claims with the secret + their sig),
                    // refund = maker (after timeout). The taker pays the claim fee.
                    htlcScriptHex = build_htlc_bytecode_hex(secretHash, takerPk, BigInt(timeoutHeight), myPk);
                    htlcAddressHex = blake3_hash_hex(htlcScriptHex);
                    fundAmount = expectedAmount;
                }

                // Fund the HTLC. prepare_fund_tx splits the amount into power-of-two
                // coins, EACH with a random salt, so funding yields a SET of coins.
                // performContractTx now returns that set (salts are not on-chain, so
                // the taker can only spend them if we transmit {coin_id,value,salt}).
                let lockTxId = null;
                const fundRes = await performContractTx({
                    reqId: 999,
                    dexOfferId: offerId,   // routes commit/reveal phase progress to this swap card
                    kind: 'fund',
                    contractAddress: htlcAddressHex,
                    amount: fundAmount
                });
                const htlcCoins = (fundRes && fundRes.coins) || [];
                if (fundRes) lockTxId = fundRes.txid;

                // RECOVERY BREADCRUMB: publish a taker-lock announcement on-chain so this
                // lock can be rebuilt from the seed alone if local data is later cleared
                // (the gap that froze Alice's funds). One MDXT announcement per coin; each
                // carries the salt + params but NOT the secret, so it's safe. Best-effort:
                // a failure here doesn't fail the lock (the swap still works with local
                // data) — it only forgoes seed-only recoverability for this coin.
                try {
                    const myAddr = compute_p2pk_address_hex(myPk);
                    const recvAddr = covenant ? compute_p2pk_address_hex(normalizeHex(takerPk)) : normalizeHex(takerPk);
                    for (const c of htlcCoins) {
                        if (!c || !c.salt || !c.coin_id) continue;
                        const annHex = encodeTakerAnnouncement({
                            takerMdsPk: myPk, secretHash, salt: normalizeHex(c.salt),
                            receiverAddr: recvAddr, timeoutHeight, value: Number(c.value),
                            weiAmount: String(Math.max(1, Math.floor(Number(expectedAmount) / Math.max(1, htlcCoins.length))))
                        });
                        const frags = fragmentAnnouncement(annHex, (c.coin_id || '').slice(0, 12));
                        for (const f of frags) { try { await performSend(myAddr, 1, f, 0); } catch (_) {} }
                    }
                } catch (annErr) {
                    self.postMessage({ type: 'LOG', payload: 'Taker recovery announcement failed (lock is fine, seed-only recovery unavailable for it): ' + ((annErr && annErr.message) || annErr) });
                }

                self.postMessage({ type: 'DEX_MIDSTATE_LOCKED_SUCCESS', payload: {
                    offerId,
                    secret: rawSecret,        // for the MAKER only — needed to claim ETH on Base
                    secretHash,               // H = BLAKE3(secret); the Base hashlock is identical
                    htlcAddressHex,
                    htlcReceiverAddr: covenant ? compute_p2pk_address_hex(normalizeHex(takerPk)) : normalizeHex(takerPk), // needed to rebuild the script for reclaim
                    htlcCoins,                // [{coin_id, value, salt}] — the taker sweeps these
                    timeoutHeight,
                    makerMdsPk: myPk,
                    takerMdsPk: takerPk,
                    swapMode: covenant ? 'covenant' : 'htlc',
                    minPayout: covenant ? Number(expectedAmount) : undefined,
                    lockTxId 
                }});
            } catch (err) {
                // Clear the card's live phase on failure so it doesn't sit on a stale
                // commit/reveal step; re-throw so the outer handler still reports it.
                if (payload && payload.offerId) self.postMessage({ type: 'DEX_PHASE', payload: { offerId: payload.offerId, phase: null } });
                throw err;
            } finally {
                releaseSendLock();
            }
        }
        else if (type === 'DEX_CREATE_LIMIT_ORDER') {
            // FEATURE 1 (maker origination). Lock MDS into a limit-order covenant and post a
            // standing on-chain order. ONE order = ONE fresh secret = ONE covenant coin, filled
            // atomically. (A larger order is just N of these — call this N times, one secret
            // each — because a single hashlock can back only one trustless fill: the maker
            // reveals the secret on Base to get paid, after which that H is public.)
            await acquireSendLock();
            try {
                const { offerId, unitValue, weiAmount, makerEvmAddr } = payload;
                const v = Number(unitValue);
                // Power-of-two => exactly one covenant coin => one clean atomic fill unit.
                if (!Number.isInteger(v) || v < 1 || (v & (v - 1)) !== 0)
                    throw new Error("Unit size must be a power of two (one atomic covenant coin).");

                // The MAKER generates the secret (matches the Base contract's protocol).
                const secretBytes = crypto.getRandomValues(new Uint8Array(32));
                const rawSecret = Array.from(secretBytes).map(b => b.toString(16).padStart(2, '0')).join('');
                const secretHash = blake3_hash_hex(rawSecret);   // H = BLAKE3(secret), shared with Base

                const myPk = getPrimaryMssPk();
                const timeoutHeight = networkHeight + 1440;       // ~24h refund window (the LONG leg)

                // max_claim = the full unit value => the whole coin is claimed atomically, no
                // remainder, so the post-reveal remainder-drain can't apply to this order.
                const covScriptHex = build_limit_order_covenant_bytecode_hex(
                    secretHash, BigInt(v), BigInt(timeoutHeight), myPk
                );
                const covAddr = blake3_hash_hex(covScriptHex);    // = hash(script); the taker recomputes the same

                // Fund the covenant with exactly the unit value (one power-of-two coin).
                const fundRes = await performContractTx({
                    reqId: 998, dexOfferId: offerId, kind: 'fund', contractAddress: covAddr, amount: v
                });
                const coins = (fundRes && fundRes.coins) || [];
                if (!coins.length) throw new Error("Funding produced no covenant coin");

                // ── Encrypted secret storage (LEGACY PATH) ──────────────────────────────
                // The UI exclusively uses DEX_CREATE_LIMIT_BUNDLE, whose secrets persist BEFORE
                // broadcast (invariant I2). This handler is kept for external callers only.
                // KNOWN LIMITATION: performContractTx rolls the coin salt internally and returns
                // it post-confirmation, so persist-before-broadcast is impossible here — a crash
                // mid-funding loses salt AND secret. Anything that needs that guarantee must use
                // the bundle path (a bundle of one unit is fine). What we DO guarantee: the
                // preimage is written to the encrypted map and never crosses postMessage.
                wState.dex_secrets = wState.dex_secrets || {};
                for (const c of coins) {
                    wState.dex_secrets[normalizeHex(c.coin_id)] = {
                        secret: rawSecret, secretHash, covAddr,
                        value: c.value, salt: normalizeHex(c.salt),
                        timeoutHeight, makerMdsPk: myPk, makerEvmAddr,
                        weiAmount: String(weiAmount), offerId, groupId: null,
                        status: 'live', createdAt: Date.now()
                    };
                }
                await saveState();

                self.postMessage({ type: 'DEX_LIMIT_ORDER_CREATED', payload: {
                    offerId,
                    covAddr,
                    coins,                 // [{coin_id, value, salt}] — one coin for a power-of-two value
                    secretHash,            // advertised to takers; also the Base hashlock
                    maxClaim: v,           // (the preimage stays in wState.dex_secrets — DEX_GET_SECRET serves it)
                    timeoutHeight,
                    makerMdsPk: myPk,      // refund pk baked into the covenant
                    makerEvmAddr,
                    weiAmount
                }});
            } catch (err) {
                if (payload && payload.offerId) self.postMessage({ type: 'DEX_PHASE', payload: { offerId: payload.offerId, phase: null } });
                self.postMessage({ type: 'DEX_LIMIT_ORDER_FAILED', payload: { offerId: payload && payload.offerId, error: (err && err.message) || String(err) } });
            } finally {
                releaseSendLock();
            }
        }
        else if (type === 'DEX_CREATE_LIMIT_BUNDLE') {
            // FEATURE 1 (maker origination, batched). Posts a non-power-of-two amount as a BUNDLE
            // of independent single-fill units — one fresh secret/covenant per unit — but funds
            // ALL of them in ONE transaction (one ~2-block commit/reveal) via prepare_fund_many,
            // instead of N serial funds. The N independent secrets are still required (one hashlock
            // backs one trustless fill); only the funding is batched. All-or-nothing: if the single
            // funding tx fails, nothing is locked.
            await acquireSendLock();
            try {
                const { groupId, units, makerEvmAddr } = payload;
                if (!Array.isArray(units) || !units.length) throw new Error("No units supplied");
                const dexPhase = (p) => self.postMessage({ type: 'DEX_PHASE', payload: { offerId: groupId, phase: p } });

                const myPk = getPrimaryMssPk();
                const timeoutHeight = networkHeight + 1440;   // ~24h refund window

                // Per unit: fresh secret -> H -> covenant -> covAddr. Collect the fundings list.
                const built = units.map(u => {
                    const v = Number(u.unitValue);
                    if (!Number.isInteger(v) || v < 1 || (v & (v - 1)) !== 0) throw new Error(`Unit ${v} is not a power of two`);
                    const sb = crypto.getRandomValues(new Uint8Array(32));
                    const rawSecret = Array.from(sb).map(b => b.toString(16).padStart(2, '0')).join('');
                    const secretHash = blake3_hash_hex(rawSecret);
                    const covScriptHex = build_limit_order_covenant_bytecode_hex(secretHash, BigInt(v), BigInt(timeoutHeight), myPk);
                    const covAddr = blake3_hash_hex(covScriptHex);
                    // Pre-generate the unit's coin salt CLIENT-SIDE. The MDXA announcement must
                    // carry each salt (takers recompute coin_id = H(covAddr‖value‖salt) to bind
                    // the coin to the covenant), so choosing salts here — instead of letting
                    // prepare_fund_many roll them — is what lets the announcement be encoded
                    // BEFORE the funding tx and ride inside it as a burn (atomic announce).
                    const salt = Array.from(crypto.getRandomValues(new Uint8Array(32))).map(b => b.toString(16).padStart(2, '0')).join('');
                    return { v, rawSecret, secretHash, covAddr, salt, weiAmount: u.weiAmount };
                });
                const fundings = built.map(b => ({ address: normalizeHex(b.covAddr), amount: b.v, salt: b.salt }));

                if (!mssCachesReady) await loadMssCaches();
                // MSS safety fast-forward (identical policy to performContractTx), with retries.
                dexPhase("Verifying MSS safety indices\u2026");
                await verifyMssSafetyIndices(dexPhase);
                const utxoArray = getSpendableUtxos().map(u =>
                    (u.is_mss && wState.mssAddrs[u.address]) ? { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf } : u);

                // ATOMIC ANNOUNCE: encode ONE MDXA announcement for the whole bundle NOW
                // (we chose the salts above) and ship it as MDXF fragment burns riding IN
                // the funding tx itself — each fragment <= the 80-byte consensus DataBurn
                // cap, which stays untouched. One commit+reveal: covenants and every
                // fragment land in the same transaction/block, so discovery is atomic with
                // funding and the scanner reassembles them from a single block.
                const announcementHex = encodeAnnouncement({
                    makerEvmAddr, makerMdsPk: myPk, timeoutHeight, groupId,
                    units: built.map(b => ({ secretHash: b.secretHash, salt: b.salt, value: b.v, weiAmount: b.weiAmount }))
                });
                const annFrags = fragmentAnnouncement(announcementHex, groupId);
                const burnsJson = JSON.stringify(annFrags.map(h => ({ payload: h, value_burned: 0 })));

                // ONE transaction funding all N covenant addresses (+ the fragment burns).
                const ctxStr = wallet.prepare_fund_many(JSON.stringify(utxoArray), JSON.stringify(fundings), wState.nextWotsIndex, burnsJson);
                const ctx = JSON.parse(ctxStr);

                // FEATURE-DETECT the WASM build. A pkg from the current lib.rs honours the
                // per-funding salts and embeds EVERY fragment; older pkgs ignore the extra
                // arg and the unknown salt field and roll their own salts. Both must hold
                // together, else fall back to the separate-announce path with the ctx's
                // REAL salts.
                const burnPayloads = new Set((ctx.outputs || []).filter(o => o.type === 'data_burn').map(o => String(o.payload).toLowerCase()));
                const burnEmbedded = annFrags.every(h => burnPayloads.has(h.toLowerCase()));
                const saltsHonoured = built.every(b => {
                    const o = (ctx.outputs || []).find(x => x.type === 'standard' && normalizeHex(x.address) === normalizeHex(b.covAddr));
                    return o && normalizeHex(o.salt) === b.salt;
                });
                const atomicAnnounce = burnEmbedded && saltsHonoured;
                if (!atomicAnnounce) self.postMessage({ type: 'LOG', payload: 'prepare_fund_many: wasm build without atomic announce — using separate announce tx.' });

                // Map each unit to its funded coin NOW: ctx.outputs carries the random salts, and a
                // coin is unspendable without its salt (coin_id = H(addr||value||salt)). The salt is
                // NOT recoverable from chain, so we must capture it before the tx goes out.
                // NOTE the units below are SECRET-FREE: they cross the postMessage boundary to the
                // UI (and into localStorage via dexMySwaps), so the raw preimage must not be on them.
                // Preimages go into the encrypted wState.dex_secrets map instead (just below).
                const outUnits = built.map(b => {
                    const o = (ctx.outputs || []).find(o => o.type === "standard" && normalizeHex(o.address) === normalizeHex(b.covAddr));
                    if (!o) throw new Error(`Funded coin for unit ${b.v} not found in tx outputs`);
                    const coin_id = compute_coin_id_hex(normalizeHex(o.address), BigInt(o.value), normalizeHex(o.salt));
                    return {
                        offerId: Array.from(crypto.getRandomValues(new Uint8Array(8))).map(x => x.toString(16).padStart(2, '0')).join(''),
                        covAddr: b.covAddr,
                        coin: { coin_id, value: o.value, salt: normalizeHex(o.salt) },
                        secretHash: b.secretHash,
                        maxClaim: b.v, timeoutHeight, makerMdsPk: myPk, makerEvmAddr, weiAmount: b.weiAmount,
                        lockTxId: ctx.commitment // <--- Add this line
                    };
                });

                // ── Encrypted secret storage ────────────────────────────────────────────
                // REASONING: the preimage is the maker's claim on the taker's ETH. If it lives in
                //   localStorage (plaintext, sync-readable by any script in the origin, included in
                //   naive backups) it is one XSS or one shared backup away from theft; if it lives
                //   only in UI memory it dies with the tab and the maker can never claim a fill.
                //   The AES-GCM wallet state is the only store that is both persistent and
                //   encrypted under the user's password, so the preimage lives there and nowhere
                //   else, keyed by the one identifier every flow shares: the covenant coin id.
                // PRE:  `built` holds one fresh (secret, H, covAddr, salt) per unit; `outUnits`
                //       maps unit i to its funded coin id; no funding bytes have been broadcast.
                // POST: wState.dex_secrets[coin_id] exists (status 'funding') for every unit, and —
                //       via the saveState() inside reserveAndLock() below — is persisted to the
                //       encrypted bundle BEFORE rpc.commit() runs. Status flips to 'live' only
                //       after the reveal confirms.
                // SAFETY: a crash at ANY later point orphans at worst an encrypted record, never a
                //       funded coin without its preimage/salt. DEX_RECOVER_BUNDLES adjudicates
                //       orphans: confirmed funding → records flip 'live'; funding never landed →
                //       records are deleted (nothing was locked, so the preimage guards nothing).
                if (!wState.dex_secrets) wState.dex_secrets = {};
                outUnits.forEach((u, i) => {
                    wState.dex_secrets[normalizeHex(u.coin.coin_id)] = {
                        secret: built[i].rawSecret, secretHash: built[i].secretHash,
                        covAddr: u.covAddr, value: u.coin.value, salt: u.coin.salt,
                        timeoutHeight, makerMdsPk: myPk, makerEvmAddr,
                        weiAmount: String(u.weiAmount), offerId: u.offerId, groupId,
                        status: 'funding', createdAt: Date.now()
                    };
                });

                // CRASH-SAFETY: persist the recovery record (coins + params) BEFORE the commit is
                // broadcast. If we die after the tx confirms but before the UI registers these
                // units, DEX_RECOVER_BUNDLES rebuilds them from here; without it the locked MDS
                // would be unspendable (salts gone) and even un-refundable (no covenant params).
                // The units stored here are the same secret-free objects the UI receives — the
                // preimages are already in wState.dex_secrets above, so a re-emitted record leaks
                // nothing. The saveState() just below persists this; it is cleared on success.
                if (!wState.pendingLimitBundles) wState.pendingLimitBundles = {};
                wState.pendingLimitBundles[groupId] = {
                    groupId, units: outUnits, commitment: ctx.commitment,
                    announcedAtomically: atomicAnnounce,
                    firstCoinId: outUnits[0].coin.coin_id, createdAt: Date.now()
                };

                // Reserve key material once (mirrors performContractTx / prepare_fund_tx flow).
                await reserveAndLock(ctx, "Saving wallet state...");

                // Funding consumes no contract coin, so build_script_reveal signs the wallet
                // inputs internally — no separate covenant signature needed.
                const revealPayloadStr = wallet.build_script_reveal(ctxStr, ctx.commitment, ctx.tx_salt);

                dexPhase(`Mining PoW [Commit: ${ctx.commitment.substring(0,8)}...]`);
                const stateData = await rpc.getState();
                await new Promise(r => setTimeout(r, 50));
                const spamNonce = Number(mine_commitment_pow(ctx.commitment, stateData.required_pow || 24, BigInt(stateData.height), stateData.header_hash));
                const commitReq = await rpc.commit(ctx.commitment, spamNonce);
                if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);
                dexPhase(`Commit broadcast — waiting to be mined (1/2) [Commit: ${ctx.commitment.substring(0,8)}...]`);
                await awaitCommitmentMined(ctx.commitment, dexPhase);
                // Self-heal MSS leaf reuse: if the leaf floor was stale, advance and re-sign
                // against the SAME already-mined commitment (leaf is a witness, not committed to).
                const revealReq = await sendRevealWithMssLeafRetry(revealPayloadStr, ctxStr, ctx.commitment, ctx.tx_salt, dexPhase);
                if (!revealReq.ok) throw new Error(`Reveal rejected: ${revealReq.body || revealReq.error}`);
                dexPhase(`Reveal broadcast — waiting to be mined (2/2) [Commit: ${ctx.commitment.substring(0,8)}...]`);
                const firstInputId = ctx.input_coin_ids && ctx.input_coin_ids.length ? ctx.input_coin_ids[0] : null;
                if (firstInputId) { await awaitCoinSpent(firstInputId, dexPhase); }
                dexPhase("Confirmed \u2713 — syncing\u2026");
                await performScan();
                
                // Funded and confirmed — mark every unit's secret record live, hand the
                // (secret-free) units to the UI, then clear the recovery record.
                for (const u of outUnits) {
                    const rec = wState.dex_secrets && wState.dex_secrets[normalizeHex(u.coin.coin_id)];
                    if (rec) rec.status = 'live';
                }
                self.postMessage({ type: 'DEX_LIMIT_BUNDLE_CREATED', payload: { groupId, units: outUnits, announcedAtomically: atomicAnnounce } });
                if (wState.pendingLimitBundles) delete wState.pendingLimitBundles[groupId];
                await saveState();
                dexPhase(null);
            } catch (err) {
                let errMsg = (err && err.message) || String(err);
                // A wasm trap ("unreachable") leaves the wallet object POISONED: wasm-bindgen's
                // borrow flag stays set, so every later wallet.* call throws "recursive use of
                // an object … unsafe aliasing in rust". Only reloading re-instantiates the
                // module — say so, instead of letting in-place retries mislead.
                if (/unreachable|unsafe aliasing/i.test(errMsg)) errMsg += " (the WASM module hit a fatal trap — reload the page before retrying)";
                if (payload && payload.groupId) self.postMessage({ type: 'DEX_PHASE', payload: { offerId: payload.groupId, phase: null } });
                self.postMessage({ type: 'DEX_LIMIT_BUNDLE_FAILED', payload: { groupId: payload && payload.groupId, error: errMsg } });
            } finally {
                releaseSendLock();
            }
        }
        else if (type === 'DEX_RECOVER_BUNDLES') {
            // On reload: re-deliver any limit-order bundle whose funding tx confirmed but whose
            // units never got registered (page closed between confirmation and the UI handling
            // DEX_LIMIT_BUNDLE_CREATED). The UI dedupes by offerId, so re-emitting is idempotent.
            // A record whose coin still isn't on-chain after a grace window is treated as a funding
            // that never landed and is dropped (the wallet inputs return on the next scan).
            try {
                const pend = wState.pendingLimitBundles || {};
                const GRACE_MS = 10 * 60 * 1000;
                let changed = false;
                for (const groupId of Object.keys(pend)) {
                    const rec = pend[groupId];
                    if (!rec || !rec.firstCoinId || !Array.isArray(rec.units)) { delete pend[groupId]; changed = true; continue; }
                    // MIGRATION: records written by pre-dex_secrets builds carry the raw preimage
                    // on each unit. Move it into the encrypted map and strip it from the unit so
                    // the re-emit below (which crosses into UI/localStorage) is secret-free.
                    wState.dex_secrets = wState.dex_secrets || {};
                    for (const u of rec.units) {
                        if (u && u.secret && u.coin && u.coin.coin_id) {
                            const cid = normalizeHex(u.coin.coin_id);
                            if (!wState.dex_secrets[cid]) {
                                wState.dex_secrets[cid] = {
                                    secret: u.secret, secretHash: u.secretHash, covAddr: u.covAddr,
                                    value: u.coin.value, salt: u.coin.salt, timeoutHeight: u.timeoutHeight,
                                    makerMdsPk: u.makerMdsPk, makerEvmAddr: u.makerEvmAddr,
                                    weiAmount: String(u.weiAmount), offerId: u.offerId, groupId,
                                    status: 'funding', createdAt: rec.createdAt || Date.now()
                                };
                            }
                            delete u.secret; changed = true;
                        }
                    }
                    let funded = null;
                    try { const inp = await rpc.checkCoin(normalizeHex(rec.firstCoinId)); funded = !!(inp && inp.exists); }
                    catch (e) { continue; }   // RPC not ready — leave it; a later call retries
                    if (funded) {
                        // Funding landed: the units are real, so their preimages now guard money.
                        for (const u of rec.units) {
                            const sr = wState.dex_secrets[normalizeHex(u.coin.coin_id)];
                            if (sr && sr.status === 'funding') sr.status = 'live';
                        }
                        self.postMessage({ type: 'DEX_LIMIT_BUNDLE_CREATED', payload: { groupId, units: rec.units, recovered: true, announcedAtomically: !!rec.announcedAtomically } });
                        delete pend[groupId]; changed = true;
                    } else if (Date.now() - (rec.createdAt || 0) > GRACE_MS) {
                        // Funding never landed: no coin exists, so these preimages guard nothing —
                        // deleting them is the ONE unambiguous purge case (dex_secrets invariant I4).
                        for (const u of rec.units) delete wState.dex_secrets[normalizeHex(u.coin.coin_id)];
                        delete pend[groupId]; changed = true;
                        self.postMessage({ type: 'DEX_LIMIT_BUNDLE_FAILED', payload: { groupId, error: "Funding did not confirm; nothing was locked.", recovered: true } });
                    }
                }
                if (changed) await saveState();
            } catch (e) { /* best-effort recovery */ }
        }
        else if (type === 'DEX_ANNOUNCE_BUNDLE') {
            // MAKER FALLBACK: with a current wasm build the announcement fragments ride
            // INSIDE the funding tx (see DEX_CREATE_LIMIT_BUNDLE), so this handler only
            // serves older pkg builds and manual/auto re-announce retries. Old prepare_spend
            // carries ONE burn per tx, so fragments go out as consecutive small sends —
            // usually the same or adjacent blocks; the scanner's persistent fragment pool
            // reassembles across blocks either way. A retry after partial failure re-sends
            // all fragments (duplicates are benign: same key/idx overwrite, and orders
            // dedupe by coinId — it only costs fees).
            try {
                const announcementHex = encodeAnnouncement(payload);
                const frags = fragmentAnnouncement(announcementHex, payload.groupId);
                for (let i = 0; i < frags.length; i++) {
                    if (frags.length > 1) self.postMessage({ type: 'LOG', payload: `Announcing fragment ${i + 1}/${frags.length}…` });
                    await performSend(myAddr, 1, frags[i], 0);
                }
                self.postMessage({ type: 'DEX_ANNOUNCE_DONE', payload: { groupId: payload.groupId } });
            } catch (err) {
                self.postMessage({ type: 'DEX_ANNOUNCE_FAILED', payload: { groupId: payload && payload.groupId, error: (err && err.message) || String(err) } });
            }
        }
        else if (type === 'DEX_RECOVER_ORDERS') {
            // SEED-ONLY RECOVERY. Rebuilds the user's own stuck limit orders purely from
            // on-chain data — no localStorage needed. This is the real fix for the "cleared
            // my browser, lost my salt" failure: the MDXA announcement published on-chain
            // when an order was posted already carries {secretHash, salt, value, timeout}
            // for each unit, and the refund key is our own primary MSS pubkey. So after
            // restoring a seed we can: scan DataBurns → keep announcements whose makerMdsPk
            // is OURS → rebuild each covenant + coin id → check it's still unspent on-chain →
            // recreate the swap record so the reclaim button works. Filled/in-flight swaps
            // (which also need the random secret + counterparty address) are NOT recovered
            // here; only unfilled locked orders, which is the common stuck case.
            try {
                const myPk = normalizeHex(getPrimaryMssPk() || "");
                if (!myPk) throw new Error("Wallet not ready.");
                const SCAN_DEPTH = 1440 * 3;   // look back ~3 timeout windows
                const tip = networkHeight;
                let from = Number.isFinite(payload && payload.fromHeight) ? payload.fromHeight : Math.max(0, tip - SCAN_DEPTH);
                from = Math.max(0, Math.min(from, tip));
                const recovered = [];
                const seenCoin = new Set();
                const BATCH = 12;
                // On gateway rate-limiting, pause the WHOLE scan and retry the
                // batch; three strikes aborts (the main thread's block cache
                // means a rerun resumes cheaply). Grinding on block-by-block
                // is what used to cause the 429 storms.
                let rateLimitStrikes = 0;
                for (let h = from; h <= tip; h += BATCH) {
                    let blocks;
                    try {
                        blocks = await fetchScanBatch(h, Math.min(h + BATCH - 1, tip));
                        rateLimitStrikes = 0;
                    } catch (e) {
                        if (e && (e.code === 'RATE_LIMITED' || e.code === 'GATEWAY_UNAVAILABLE')) {
                            if (++rateLimitStrikes >= 3) {
                                throw new Error('RPC gateway is rate-limiting or unreachable — recovery scan aborted. Try again in a minute; it will resume from cache.');
                            }
                            await new Promise(r => setTimeout(r, 15000 * rateLimitStrikes));
                            h -= BATCH;   // retry this batch after the pause
                            continue;
                        }
                        throw e;
                    }
                    for (const blk of blocks) {
                        if (!blk) continue;
                        const payloads = extractBurnPayloadHexes(blk, []);
                        for (const m of payloads) {
                            // --- Maker limit orders (MDXA) ---
                            const ann = tryDecodeAnnouncement(m);
                            if (ann && normalizeHex(ann.makerMdsPk) === myPk) {
                                for (const u of ann.units) {
                                    try {
                                        const covScriptHex = build_limit_order_covenant_bytecode_hex(u.secretHash, BigInt(u.value), BigInt(ann.timeoutHeight), myPk);
                                        const covAddr = blake3_hash_hex(covScriptHex);
                                        const coinId = compute_coin_id_hex(covAddr, BigInt(u.value), normalizeHex(u.salt));
                                        if (seenCoin.has(coinId)) continue;
                                        seenCoin.add(coinId);
                                        let exists = false;
                                        try { const r = await rpc.checkCoin(coinId); exists = !!(r && r.exists); } catch (e) { exists = false; }
                                        if (!exists) continue;
                                        // A unit the maker cancelled recovers as cancelled: still
                                        // reclaimable after timeout, but never re-advertised.
                                        const wasCancelled = !!(wState.dex_cancelled && wState.dex_cancelled[normalizeHex(coinId)]) ||
                                            (wState.dex_secrets && wState.dex_secrets[normalizeHex(coinId)] &&
                                             wState.dex_secrets[normalizeHex(coinId)].status === 'limit_cancelled');
                                        recovered.push({
                                            offerId: 'chain:' + coinId.slice(0, 16),
                                            role: 'maker', side: 'mds', kind: 'limit', status: wasCancelled ? 'limit_cancelled' : 'limit_posted',
                                            mdsAmount: u.value, weiAmount: u.weiAmount, makerEvmAddr: ann.makerEvmAddr,
                                            secretHash: u.secretHash, recoveredFromChain: true,
                                            covenant: {
                                                coinId, value: u.value, salt: normalizeHex(u.salt), covAddr,
                                                secretHash: u.secretHash, maxClaim: u.value, timeoutHeight: ann.timeoutHeight,
                                                makerMdsPk: myPk, makerEvmAddr: ann.makerEvmAddr, weiAmount: u.weiAmount
                                            }
                                        });
                                    } catch (e) { /* skip bad unit */ }
                                }
                                continue;
                            }
                            // --- Taker locks (MDXT) — the Alice-shape, now recoverable ---
                            const tann = tryDecodeTakerAnnouncement(m);
                            if (tann && normalizeHex(tann.takerMdsPk) === myPk) {
                                try {
                                    // Taker covenant HTLC: rebuild with the receiver addr the
                                    // announcement carries (the buyer's address). Refund goes to us.
                                    const covScriptHex = build_covenant_htlc_bytecode_hex(
                                        tann.secretHash, tann.receiverAddr, BigInt(tann.value), BigInt(tann.timeoutHeight), myPk
                                    );
                                    const covAddr = blake3_hash_hex(covScriptHex);
                                    const coinId = compute_coin_id_hex(covAddr, BigInt(tann.value), normalizeHex(tann.salt));
                                    if (seenCoin.has(coinId)) continue;
                                    seenCoin.add(coinId);
                                    let exists = false;
                                    try { const r = await rpc.checkCoin(coinId); exists = !!(r && r.exists); } catch (e) { exists = false; }
                                    if (!exists) continue;
                                    recovered.push({
                                        offerId: 'chain:' + coinId.slice(0, 16),
                                        role: 'taker', side: 'mds', kind: 'swap', status: 'recovered_locked',
                                        mdsAmount: tann.value, weiAmount: tann.weiAmount, recoveredFromChain: true,
                                        secretHash: tann.secretHash, swapMode: 'covenant',
                                        timeoutHeight: tann.timeoutHeight,
                                        htlcReceiverAddr: tann.receiverAddr, makerMdsPk: myPk, minPayout: tann.value,
                                        htlcCoins: [{ coin_id: coinId, value: tann.value, salt: normalizeHex(tann.salt) }]
                                    });
                                } catch (e) { /* skip bad taker unit */ }
                                continue;
                            }
                        }
                    }
                    self.postMessage({ type: 'DEX_RECOVER_PROGRESS', payload: { at: Math.min(h + BATCH, tip), tip, from, found: recovered.length } });
                }
                self.postMessage({ type: 'DEX_RECOVER_DONE', payload: { recovered, scannedToHeight: tip } });
            } catch (err) {
                self.postMessage({ type: 'DEX_RECOVER_DONE', payload: { recovered: [], error: (err && err.message) || String(err) } });
            }
        }
        else if (type === 'DEX_SCAN_ANNOUNCEMENTS') {
            // TAKER: scan blocks for on-chain order announcements. Parsing is shape-agnostic — we
            // pull any magic-prefixed hex out of the block JSON, so we don't depend on the node's
            // exact OutputData serde. Each unit's covenant address + coin id is RECOMPUTED from the
            // announced fields and then verified on-chain, so a forged/garbage announcement can't
            // inject a fake order (the coin simply won't exist).
            // ── ORDERBOOK PURITY ────────────────────────────────────────────────────────
            // REASONING: an order is fillable iff its covenant coin exists AND its timeout is
            //   still ahead AND its maker hasn't withdrawn it. Every order's timeout is set to
            //   creation+1440, so any order that can still pass the timeout filter was
            //   announced within the last ANNOUNCE_SCAN_DEPTH(=1440) blocks — which means a
            //   full-window scan below is COMPLETE for live orders. That completeness is what
            //   lets the UI treat this result as authoritative and replace its stale on-chain
            //   asks with it (ghost orders disappear).
            // PRE:  networkHeight is current; wState.dex_cancelled / dex_secrets are loaded.
            // POST: `orders` contains exactly the units that are (a) reconstructible from an
            //   announcement, (b) alive on-chain, (c) unexpired, (d) not locally cancelled,
            //   and (e) — if ours — whose secret record is status 'live'.
            // SAFETY: filters only ever REMOVE entries relative to the raw chain data; nothing
            //   here can inject an order (coin existence is still verified per unit).
            try {
                const ANNOUNCE_SCAN_DEPTH = 1440;   // ~ the order timeout window
                const tip = networkHeight;
                // Always cover the FULL live window: a narrower incremental scan would be
                // incomplete and would break the UI's replace-stale-asks contract above. A
                // caller may pass fromHeight only to scan DEEPER (never shallower).
                let from = Math.max(0, tip - ANNOUNCE_SCAN_DEPTH);
                if (Number.isFinite(payload && payload.fromHeight)) from = Math.max(0, Math.min(payload.fromHeight, from));
                const orders = [];
                const seenCoin = new Set();
                const BATCH = 12;
                let fragPoolDirty = false;
                // Same rate-limit posture as the recover scan: pause the whole
                // scan on 429s, abort after three strikes. The error lands in
                // DEX_ANNOUNCED_ORDERS' error field, so the orderbook keeps its
                // existing entries (an errored scan is non-authoritative) and
                // the next 3-minute cycle retries from the main thread's cache.
                let rateLimitStrikes = 0;
                for (let h = from; h <= tip; h += BATCH) {
                    let blocks;
                    try {
                        blocks = await fetchScanBatch(h, Math.min(h + BATCH - 1, tip));
                        rateLimitStrikes = 0;
                    } catch (e) {
                        if (e && (e.code === 'RATE_LIMITED' || e.code === 'GATEWAY_UNAVAILABLE')) {
                            if (++rateLimitStrikes >= 3) {
                                throw new Error('RPC gateway is rate-limiting or unreachable — orderbook scan aborted; will retry next cycle.');
                            }
                            await new Promise(r => setTimeout(r, 15000 * rateLimitStrikes));
                            h -= BATCH;   // retry this batch after the pause
                            continue;
                        }
                        throw e;
                    }
                    for (const blk of blocks) {
                        if (!blk) continue;
                        // Burns arrive as serde number-arrays, not hex runs, so walk the
                        // block object instead of regexing its JSON.
                        const payloads = extractBurnPayloadHexes(blk, []);
                        const fullAnns = [];
                        for (const p of payloads) {
                            const frag = tryParseFragment(p);
                            if (frag) {
                                wState.annFragPool = wState.annFragPool || {};
                                const slot = wState.annFragPool[frag.key] || { total: frag.total, parts: {}, ts: Date.now() };
                                if (!slot.parts[frag.idx]) { slot.parts[frag.idx] = frag.chunk; slot.ts = Date.now(); fragPoolDirty = true; }
                                wState.annFragPool[frag.key] = slot;
                                if (Object.keys(slot.parts).length === slot.total) {
                                    let full = '';
                                    for (let i = 0; i < slot.total; i++) full += slot.parts[i];
                                    fullAnns.push(full);
                                }
                            } else if (p.slice(0, 8) === ANN_MAGIC) {
                                fullAnns.push(p);   // tolerate any full-size announcement shape
                            }
                        }
                        for (const m of fullAnns) {
                            const ann = tryDecodeAnnouncement(m);
                            if (!ann) continue;
                            for (const u of ann.units) {
                                try {
                                    // FILTER 1 (unexpired): past the timeout only the maker's refund
                                    // branch is spendable — an expired unit is not an order. Checked
                                    // FIRST because it's free (no RPC round-trip).
                                    if (!(Number(ann.timeoutHeight) > tip)) continue;
                                    const covScriptHex = build_limit_order_covenant_bytecode_hex(u.secretHash, BigInt(u.value), BigInt(ann.timeoutHeight), ann.makerMdsPk);
                                    const covAddr = blake3_hash_hex(covScriptHex);
                                    const coinId = compute_coin_id_hex(covAddr, BigInt(u.value), normalizeHex(u.salt));
                                    if (seenCoin.has(coinId)) continue;
                                    seenCoin.add(coinId);
                                    // FILTER 2 (not withdrawn): locally-cancelled units never re-enter
                                    // the book, even though their coin still exists until the refund.
                                    if (wState.dex_cancelled && wState.dex_cancelled[normalizeHex(coinId)]) continue;
                                    // FILTER 3 (own units): an order of OURS is offered only while its
                                    // secret record is 'live' — a 'funding' record isn't confirmed and a
                                    // 'limit_cancelled' one is withdrawn (belt-and-braces with FILTER 2).
                                    const own = wState.dex_secrets && wState.dex_secrets[normalizeHex(coinId)];
                                    if (own && own.status !== 'live') continue;
                                    // FILTER 4 (alive): verify the covenant coin actually exists
                                    // (not spent/filled, not forged).
                                    let exists = false;
                                    try { const r = await rpc.checkCoin(coinId); exists = !!(r && r.exists); } catch (e) { exists = false; }
                                    if (!exists) continue;
                                    orders.push({
                                        offerId: 'chain:' + coinId.slice(0, 16),
                                        kind: 'ask', mdsAmount: u.value, weiAmount: u.weiAmount,
                                        makerEvmAddr: ann.makerEvmAddr, onchain: true,
                                        covenant: {
                                            coinId, value: u.value, salt: normalizeHex(u.salt), covAddr,
                                            secretHash: u.secretHash, maxClaim: u.value, timeoutHeight: ann.timeoutHeight,
                                            makerMdsPk: ann.makerMdsPk, makerEvmAddr: ann.makerEvmAddr, weiAmount: u.weiAmount
                                        }
                                    });
                                } catch (e) { /* skip bad unit */ }
                            }
                        }
                    }
                    self.postMessage({ type: 'DEX_SCAN_PROGRESS', payload: { at: Math.min(h + BATCH, tip), tip, from } });
                }
                // Persist partial fragment sets so announcements whose fragments straddle
                // a scan boundary (possible only via the multi-tx FALLBACK announce path)
                // still assemble on a later scan; GC stale partials after ~3 days.
                if (wState.annFragPool) {
                    const cutoff = Date.now() - 72 * 3600 * 1000;
                    for (const k of Object.keys(wState.annFragPool)) {
                        if (wState.annFragPool[k].ts < cutoff) { delete wState.annFragPool[k]; fragPoolDirty = true; }
                    }
                }
                if (fragPoolDirty) { try { await saveState(); } catch (_) {} }
                self.postMessage({ type: 'DEX_ANNOUNCED_ORDERS', payload: { orders, scannedToHeight: tip } });
            } catch (err) {
                self.postMessage({ type: 'DEX_ANNOUNCED_ORDERS', payload: { orders: [], scannedToHeight: networkHeight, error: (err && err.message) || String(err) } });
            }
        }
        else if (type === 'DEX_VERIFY_HTLC') {
            // Taker-side safety check: before locking ETH, prove the maker's MDS HTLC
            // actually exists on-chain with the agreed hashlock, our pubkey as receiver,
            // and the full expected value. Without this a malicious maker could get the
            // taker to lock ETH against a fake or short Midstate lock.
            const { offerId, secretHash, timeoutHeight, makerMdsPk, htlcCoins, expectedTotal, baseRefundSecs, swapMode } = payload;
            const covenant = (swapMode === 'covenant');
            try {
                const myPk = getPrimaryMssPk();
                // Reconstruct the script ourselves — NEVER trust parameters off the wire.
                // For a covenant we bake in OUR OWN receiving address and minPayout =
                // the agreed amount, so a mismatch (e.g. a smaller minPayout) fails here.
                let scriptHex;
                if (covenant) {
                    const myAddr = compute_p2pk_address_hex(myPk);
                    scriptHex = build_covenant_htlc_bytecode_hex(
                        secretHash, myAddr, BigInt(expectedTotal), BigInt(timeoutHeight), makerMdsPk
                    );
                } else {
                    scriptHex = build_htlc_bytecode_hex(secretHash, myPk, BigInt(timeoutHeight), makerMdsPk);
                }
                const addrHex = blake3_hash_hex(scriptHex);
                let total = 0, ok = true, reason = "";

                // SAFETY: the Midstate HTLC timeout is ABSOLUTE (fixed when the maker
                // locked), but our Base refund is RELATIVE to when we lock. If the
                // Midstate leg expires too soon, a malicious maker could refund MDS
                // AND claim our ETH. Require the remaining Midstate window to exceed
                // the Base refund by a wide margin (~12h at 1-minute blocks).
                const refundBlocks = Math.ceil((Number(baseRefundSecs) || 21600) / 60);
                const minRemaining = refundBlocks + 720; // + ~12h margin
                const remaining = Number(timeoutHeight) - networkHeight;
                if (remaining < minRemaining) {
                    ok = false;
                    reason = `Midstate HTLC expires in ~${remaining} blocks; need ≥ ${minRemaining}. Unsafe to lock ETH.`;
                }

                for (const c of (ok ? (htlcCoins || []) : [])) {
                    const expectId = compute_coin_id_hex(normalizeHex(addrHex), BigInt(c.value), normalizeHex(c.salt));
                    if (normalizeHex(expectId) !== normalizeHex(c.coin_id)) { ok = false; reason = "HTLC parameters do not match the advertised coins"; break; }
                    const chk = await rpc.checkCoin(normalizeHex(c.coin_id)).catch(() => null);
                    if (!chk || !chk.exists) { ok = false; reason = "HTLC coin not found on-chain yet"; break; }
                    total += Number(c.value);
                }
                if (ok && expectedTotal != null) {
                    if (covenant) {
                        // The maker OVER-funds a covenant by the fee budget so the delivery
                        // fee comes from the locked value. Require enough headroom that the
                        // delivery (which pays the buyer the full amount) can afford its fee;
                        // otherwise the MDS could get stranded after we've locked ETH.
                        const need = Number(expectedTotal) + COVENANT_MIN_FEE_RESERVE;
                        if (total < need) {
                            ok = false;
                            reason = `Covenant holds ${total} MDS; needs ≥ ${need} (amount + fee reserve) to be deliverable`;
                        }
                    } else if (total !== Number(expectedTotal)) {
                        ok = false; reason = `HTLC locks ${total} MDS but the offer was ${expectedTotal} MDS`;
                    }
                }
                self.postMessage({ type: 'DEX_HTLC_VERIFIED', payload: { offerId, ok, reason, htlcAddressHex: addrHex } });
            } catch (e) {
                const detail = (e && e.message) ? e.message
                             : (typeof e === 'string') ? e
                             : (e && typeof e.toString === 'function' && e.toString() !== '[object Object]') ? e.toString()
                             : JSON.stringify(e);
                self.postMessage({ type: 'DEX_HTLC_VERIFIED', payload: { offerId, ok: false, reason: detail } });
            }
        }
        else if (type === 'DEX_CLAIM_MIDSTATE') {
            await acquireSendLock();
            try {
                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Claiming HTLC..." } });
                const { swapIdx, rawSecret, htlcCoins, makerMdsPk, secretHash, timeoutHeight, offerId } = payload;
                const dexPhase = (phase) => { if (offerId) self.postMessage({ type: 'DEX_PHASE', payload: { offerId, phase } }); };
                dexPhase("Building claim transaction\u2026");

                if (!Array.isArray(htlcCoins) || htlcCoins.length === 0) throw new Error("No HTLC coins to claim");

                // We are the receiver baked into the HTLC, so reconstruct the script
                // with OUR pubkey (never trust a pubkey passed across the wire for this).
                const myPk = getPrimaryMssPk();
                const htlcScriptHex = build_htlc_bytecode_hex(secretHash, myPk, BigInt(timeoutHeight), makerMdsPk);

                if (!mssCachesReady) await loadMssCaches();
                const utxoArray = getSpendableUtxos().map(u => {
                    if (u.is_mss && wState.mssAddrs[u.address]) return { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf };
                    return u;
                });

                // Receive the full HTLC value to our primary address.
                const takerAddressHex = Object.keys(wState.mssAddrs)[0];
                const totalValue = htlcCoins.reduce((a, c) => a + Number(c.value), 0);

                // CONSENSUS: every standard output must be a NONZERO power of two
                // (apply_transaction rejects anything else). prepare_script_spend
                // splits its OWN fee-change via decompose_value, but it does NOT
                // split the caller's outputs — so we must hand it power-of-two coins.
                // Decompose the swept total into one coin per set bit (the same shape
                // the node uses). The receive side has no index-based covenant, so
                // ordering is irrelevant; the wallet rediscovers each coin (and its
                // salt) from the on-chain reveal on the next scan.
                const pow2Parts = [];
                { let n = BigInt(totalValue), bit = 0n;
                  while (n > 0n) { if (n & 1n) pow2Parts.push(Number(1n << bit)); n >>= 1n; bit += 1n; } }

                const outputsJson = JSON.stringify(pow2Parts.map(v => ({
                    out_type: "standard",
                    address: takerAddressHex,
                    value: v,
                    salt: null // prepare_script_spend generates a salt; it lands in the reveal
                })));

                // One contract input PER funded HTLC coin, all sharing the one bytecode.
                const contractInputsJson = JSON.stringify(htlcCoins.map(c => ({
                    coin_id: normalizeHex(c.coin_id),
                    witness: "",            // filled after we have the commitment
                    value: Number(c.value),
                    salt: normalizeHex(c.salt),
                    state: null
                })));

                // 1. Build the spend context across all inputs.
                let ctx = JSON.parse(wallet.prepare_script_spend(
                    JSON.stringify(utxoArray),
                    htlcScriptHex,
                    contractInputsJson,
                    outputsJson,
                    wState.nextWotsIndex
                ));

                // 2. Sign the (single) commitment with our key. The same signature
                //    satisfies CHECKSIGVERIFY for every input (same pk, same commitment).
                const sigHex = await signMssAndSync(myPk, ctx.commitment);

                // 3. Inject the claim witness [Signature, Secret, 0x01] into each input.
                for (let i = 0; i < ctx.contract_inputs.length; i++) {
                    ctx.contract_inputs[i].witness = `${sigHex},${rawSecret},01`;
                }

                // 4. Build the reveal.
                const revealPayloadStr = wallet.build_script_reveal(JSON.stringify(ctx), ctx.commitment, ctx.tx_salt);

                // 5. Advance key material exactly once.
                await reserveAndLock(ctx, "Saving wallet state...");

                // 6. Mine, commit, wait, reveal, wait.
                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Mining PoW..." } });
                dexPhase(`Mining PoW [Commit: ${ctx.commitment.substring(0,8)}...]`);

                const stateData = await rpc.getState();
                await new Promise(r => setTimeout(r, 50));
                const spamNonce = Number(mine_commitment_pow(ctx.commitment, stateData.required_pow || 24, BigInt(stateData.height), stateData.header_hash));

                const commitReq = await rpc.commit(ctx.commitment, spamNonce);
                if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Waiting for Block Confirmation..." } });
                dexPhase(`Commit broadcast — waiting to be mined (1/2) [Commit: ${ctx.commitment.substring(0,8)}...]`);
                await awaitCommitmentMined(ctx.commitment, dexPhase);

                const revealReq = await rpc.send(revealPayloadStr);
                if (!revealReq.ok) throw new Error(`Reveal rejected: ${revealReq.body || revealReq.error}`);

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Broadcasting claim..." } });
                dexPhase(`Reveal broadcast — waiting to be mined (2/2) [Commit: ${ctx.commitment.substring(0,8)}...]`);
                const firstCoin = normalizeHex(htlcCoins[0].coin_id);
                await awaitCoinSpent(firstCoin, dexPhase);

                dexPhase("Confirmed \u2713 \u2014 syncing wallet\u2026");
                await performScan();
                self.postMessage({ type: 'DEX_CLAIM_SUCCESS', payload: { swapIdx, offerId, claimTxId: ctx.commitment } 
            });
            } catch (err) {
                // Clear the card's live phase so a failed claim doesn't sit on a stale
                // "waiting to be mined" line (payload is in scope here; dexPhase isn't).
                if (payload && payload.offerId) self.postMessage({ type: 'DEX_PHASE', payload: { offerId: payload.offerId, phase: null } });
                // wasm-bindgen throws Err(JsValue::from_str(...)) as a bare JS STRING,
                // which has no .message — so reading err.message swallowed the real
                // Rust reason. Extract it from whatever shape the throw actually is.
                const detail = (err && err.message) ? err.message
                             : (typeof err === 'string') ? err
                             : (err && typeof err.toString === 'function' && err.toString() !== '[object Object]') ? err.toString()
                             : JSON.stringify(err);
                throw new Error(`Claim failed: ${detail}`);
            } finally {
                releaseSendLock();
            }
        }
        else if (type === 'DEX_SETTLE_COVENANT') {
            // COVENANT DELIVERY. Spend the covenant-HTLC coins so the MDS lands on the
            // BUYER's address. The covenant's claim path has NO signature — it only
            // checks the secret AND forces >= minPayout to the buyer — so the witness is
            // just [secret, 0x01] and ANYONE can broadcast it. Two callers use this:
            //   • the SELLER (happy path), right after claiming the ETH, with the secret
            //     it generated. Its change (the over-funded fee budget minus the real
            //     fee) comes back to the seller.
            //   • the BUYER (fallback), if the seller never delivers, using the secret it
            //     read from the Base `Claimed` event. A buyer holding ZERO MDS can still
            //     do this because the fee is paid out of the locked value, not a fee coin.
            await acquireSendLock();
            try {
                const { offerId, swapIdx, rawSecret, buyerPk, minPayout, timeoutHeight, makerMdsPk, htlcCoins, role } = payload;
                const dexPhase = (phase) => { if (offerId) self.postMessage({ type: 'DEX_PHASE', payload: { offerId, phase } }); };
                dexPhase(role === 'buyer' ? "Self-delivering your MDS\u2026" : "Delivering MDS to the buyer\u2026");

                if (!Array.isArray(htlcCoins) || htlcCoins.length === 0) throw new Error("No covenant coins to deliver");

                // Reconstruct the EXACT covenant bytecode used at lock time, so the coin
                // ids match. secretHash is derived from the secret (guarantees the witness
                // preimage actually satisfies the hashlock). The receiver address baked in
                // is the buyer's P2PK address — derived from the buyer's pubkey, never
                // trusted off the wire for our own side.
                const secretHash = blake3_hash_hex(rawSecret);
                // The receiver baked into the covenant is the ETH-side's P2PK address. When
                // WE are that buyer (self-deliver fallback), use our own pubkey — never trust
                // one off the wire for our own side. When we're the seller delivering, use the
                // buyer's pubkey we were given.
                const buyerPkResolved = (role === 'buyer') ? getPrimaryMssPk() : normalizeHex(buyerPk);
                const buyerAddr = compute_p2pk_address_hex(buyerPkResolved);
                const covScriptHex = build_covenant_htlc_bytecode_hex(
                    secretHash, buyerAddr, BigInt(minPayout), BigInt(timeoutHeight), makerMdsPk
                );

                if (!mssCachesReady) await loadMssCaches();
                const utxoArray = getSpendableUtxos().map(u => {
                    if (u.is_mss && wState.mssAddrs[u.address]) return { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf };
                    return u;
                });

                // Force exactly `minPayout` (the agreed amount) to the buyer, decomposed
                // into power-of-two coins (consensus). The fee comes from the over-funded
                // surplus the maker locked, so we add NO wallet inputs of our own; the
                // surplus-minus-fee returns to whoever is spending as change.
                const payout = Number(minPayout);
                const pow2Parts = [];
                { let n = BigInt(payout), bit = 0n;
                  while (n > 0n) { if (n & 1n) pow2Parts.push(Number(1n << bit)); n >>= 1n; bit += 1n; } }

                const outputsJson = JSON.stringify(pow2Parts.map(v => ({
                    out_type: "standard",
                    address: buyerAddr,
                    value: v,
                    salt: null // generated into the reveal; the buyer rediscovers it on scan
                })));

                // One contract input per covenant coin. Witness is COMPLETE up front —
                // [secret, 0x01], no signature — because the covenant claim path has no
                // CHECKSIG. build_script_reveal passes contract witnesses through verbatim.
                const contractInputsJson = JSON.stringify(htlcCoins.map(c => ({
                    coin_id: normalizeHex(c.coin_id),
                    witness: `${rawSecret},01`,
                    value: Number(c.value),
                    salt: normalizeHex(c.salt),
                    state: null
                })));

                let ctx = JSON.parse(wallet.prepare_script_spend(
                    JSON.stringify(utxoArray),
                    covScriptHex,
                    contractInputsJson,
                    outputsJson,
                    wState.nextWotsIndex
                ));

                // No contract-input signing needed. Any wallet fee inputs (there should be
                // none for a covenant delivery) are signed inside build_script_reveal.
                const revealPayloadStr = wallet.build_script_reveal(JSON.stringify(ctx), ctx.commitment, ctx.tx_salt);

                await reserveAndLock(ctx, "Saving wallet state...");

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Mining PoW..." } });
                dexPhase(`Mining PoW [Commit: ${ctx.commitment.substring(0,8)}...]`);

                const stateData = await rpc.getState();
                await new Promise(r => setTimeout(r, 50));
                const spamNonce = Number(mine_commitment_pow(ctx.commitment, stateData.required_pow || 24, BigInt(stateData.height), stateData.header_hash));

                const commitReq = await rpc.commit(ctx.commitment, spamNonce);
                if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Waiting for Block Confirmation..." } });
                dexPhase(`Commit broadcast — waiting to be mined (1/2) [Commit: ${ctx.commitment.substring(0,8)}...]`);
                await awaitCommitmentMined(ctx.commitment, dexPhase);

                const revealReq = await rpc.send(revealPayloadStr);
                if (!revealReq.ok) throw new Error(`Reveal rejected: ${revealReq.body || revealReq.error}`);

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Broadcasting delivery..." } });
                dexPhase(`Reveal broadcast — waiting to be mined (2/2) [Commit: ${ctx.commitment.substring(0,8)}...]`);
                const firstCoin = normalizeHex(htlcCoins[0].coin_id);
                await awaitCoinSpent(firstCoin, dexPhase);

                dexPhase("Confirmed \u2713 \u2014 syncing wallet\u2026");
                // The buyer scans to pick up the freshly delivered coins; the seller scans
                // to pick up the surplus change. Either way a scan is correct here.
                await performScan();
                self.postMessage({ type: 'DEX_SETTLE_SUCCESS', payload: { offerId, swapIdx, role, settleTxId: ctx.commitment } });

            } catch (err) {
                if (payload && payload.offerId) self.postMessage({ type: 'DEX_PHASE', payload: { offerId: payload.offerId, phase: null } });
                const detail = (err && err.message) ? err.message
                             : (typeof err === 'string') ? err
                             : (err && typeof err.toString === 'function' && err.toString() !== '[object Object]') ? err.toString()
                             : JSON.stringify(err);
                self.postMessage({ type: 'DEX_SETTLE_FAILED', payload: { offerId: payload && payload.offerId, swapIdx: payload && payload.swapIdx, role: payload && payload.role, error: detail } });
            } finally {
                releaseSendLock();
            }
        }
        else if (type === 'DEX_FILL_LIMIT') {
            // FEATURE 1 — taker-side fill of a maker's limit-order covenant. Spends EXACTLY
            // ONE covenant coin (single-input safety — see the multi-coin caveat in
            // compile_limit_order_covenant) and pays the FULL unit to the taker. Mirrors the
            // proven DEX_SETTLE_COVENANT covenant-spend template.
            //
            // ── PRODUCTION POLICY: full-unit fills ONLY ─────────────────────────────────
            // REASONING: a partial fill would route `coinValue - claimed` back to the same
            //   covenant address — a remainder coin locked under the SAME BLAKE3 hashlock H.
            //   But this fill's spend witness publishes the preimage of H on-chain, and the
            //   covenant's claim branch is gated on nothing else. From that block onward,
            //   ANYONE can spend the remainder and route it to themselves: the "remainder"
            //   is not a smaller live order, it is a donation to whoever sees the preimage
            //   first. One secret backs exactly one trustless fill; smaller fills exist as
            //   smaller UNITS (each with a fresh secret), never as partial claims.
            // PRE:  payload.claimed === coin.value (rejected otherwise, below).
            // POST: outputs pay the full coin value to the taker; nothing returns to covAddr;
            //       `remainder` is identically 0 on every path.
            // SAFETY: enforcing this in the WORKER (not just the UI) means no future UI bug,
            //   devtools call, or external client can reintroduce the drain.
            await acquireSendLock();
            try {
                const { offerId, rawSecret, coin, claimed, maxClaim, timeoutHeight, makerMdsPk } = payload;
                const dexPhase = (p) => { if (offerId) self.postMessage({ type: 'DEX_PHASE', payload: { offerId, phase: p } }); };

                if (!coin || !coin.coin_id) throw new Error("No covenant coin supplied");
                const coinValue = Number(coin.value);
                const claimAmt  = Number(claimed);
                if (claimAmt !== coinValue) throw new Error(
                    `Production policy: limit-order units fill atomically in full. ` +
                    `claimed=${claimAmt} != unit value ${coinValue}. A partial fill's remainder ` +
                    `would share this unit's revealed BLAKE3 preimage and be drainable by anyone.`);
                if (claimAmt > Number(maxClaim)) throw new Error(`claimed ${claimAmt} exceeds covenant max_claim ${maxClaim}`);
                const remainder = 0;   // full-unit policy: no remainder path exists (see block above)

                // Reconstruct the EXACT covenant bytecode the maker locked, so coin ids match.
                const secretHash = blake3_hash_hex(rawSecret);
                const covScriptHex = build_limit_order_covenant_bytecode_hex(
                    secretHash, BigInt(maxClaim), BigInt(timeoutHeight), makerMdsPk
                );
                const covAddr = blake3_hash_hex(covScriptHex);   // = hash(script); matches lock-time address

                // sanity: the coin we're about to spend really belongs to this covenant
                const expectId = compute_coin_id_hex(covAddr, BigInt(coinValue), normalizeHex(coin.salt));
                if (normalizeHex(expectId) !== normalizeHex(coin.coin_id)) {
                    throw new Error("Coin does not match the reconstructed covenant address/params");
                }

                const myPk = getPrimaryMssPk();
                const buyerAddr = compute_p2pk_address_hex(myPk);

                const pow2 = (n) => { const out = []; let v = BigInt(n), bit = 0n; while (v > 0n) { if (v & 1n) out.push(Number(1n << bit)); v >>= 1n; bit += 1n; } return out; };

                // outputs: the FULL unit -> taker. No remainder ever returns to covAddr (see the
                // production-policy block above: a post-reveal remainder is drainable by anyone).
                const outputs = pow2(claimAmt).map(v => ({ out_type: "standard", address: buyerAddr, value: v, salt: null }));
                const outputsJson = JSON.stringify(outputs);

                // single covenant input; witness complete up front: [secret, 0x01] (no signature)
                const contractInputsJson = JSON.stringify([{
                    coin_id: normalizeHex(coin.coin_id),
                    witness: `${rawSecret},01`,
                    value: coinValue,
                    salt: normalizeHex(coin.salt),
                    state: null
                }]);

                if (!mssCachesReady) await loadMssCaches();
                const utxoArray = getSpendableUtxos().map(u =>
                    (u.is_mss && wState.mssAddrs[u.address]) ? { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf } : u);

                let ctx = JSON.parse(wallet.prepare_script_spend(
                    JSON.stringify(utxoArray), covScriptHex, contractInputsJson, outputsJson, wState.nextWotsIndex
                ));
                const revealPayloadStr = wallet.build_script_reveal(JSON.stringify(ctx), ctx.commitment, ctx.tx_salt);

                await reserveAndLock(ctx, "Saving wallet state...");

                dexPhase(`Mining PoW [Commit: ${ctx.commitment.substring(0,8)}...]`);
                const stateData = await rpc.getState();
                await new Promise(r => setTimeout(r, 50));
                const spamNonce = Number(mine_commitment_pow(ctx.commitment, stateData.required_pow || 24, BigInt(stateData.height), stateData.header_hash));
                const commitReq = await rpc.commit(ctx.commitment, spamNonce);
                if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);

                dexPhase(`Commit broadcast — waiting to be mined (1/2) [Commit: ${ctx.commitment.substring(0,8)}...]`);
                await awaitCommitmentMined(ctx.commitment, dexPhase);

                const revealReq = await rpc.send(revealPayloadStr);
                if (!revealReq.ok) throw new Error(`Reveal rejected: ${revealReq.body || revealReq.error}`);

                dexPhase(`Reveal broadcast — waiting to be mined (2/2) [Commit: ${ctx.commitment.substring(0,8)}...]`);
                const firstCoin = normalizeHex(coin.coin_id);
                await awaitCoinSpent(firstCoin, dexPhase);

                dexPhase("Confirmed ✓ — syncing…");
                await performScan();
                // Revealing `rawSecret` in the spend witness lets the maker harvest it and claim the
                // taker's ETH on Base. The unit is consumed in full — nothing remains at covAddr
                // (remainder is 0 by production policy), so the order simply leaves the book.
                self.postMessage({ type: 'DEX_FILL_SUCCESS', payload: { offerId, claimed: claimAmt, remainder, covAddr, fillTxId: ctx.commitment } });
            } catch (err) {
                const detail = (err && err.message) ? err.message : String(err);
                self.postMessage({ type: 'DEX_FILL_FAILED', payload: { offerId: payload && payload.offerId, error: detail } });
            } finally {
                releaseSendLock();
            }
        }
        else if (type === 'DEX_REFUND_LIMIT_ORDER') {
            // FEATURE 1 — maker reclaims an UNFILLED limit-order unit after its timeout.
            // Spends the covenant coin via the ELSE (refund) branch: the VM enforces
            // height >= timeout (CHECKTIMEVERIFY) and a maker signature (CHECKSIGVERIFY),
            // then we route the full coin value back to the maker. Witness = [sig, dummy, 0x00].
            await acquireSendLock();
            try {
                const { offerId, coin, secretHash, maxClaim, timeoutHeight, makerMdsPk } = payload;
                const dexPhase = (p) => { if (offerId) self.postMessage({ type: 'DEX_PHASE', payload: { offerId, phase: p } }); };
                if (!coin || !coin.coin_id) throw new Error("No covenant coin supplied");
                if (networkHeight < Number(timeoutHeight))
                    throw new Error(`Too early — this order's refund unlocks at height ${timeoutHeight} (now ${networkHeight}).`);

                const coinValue = Number(coin.value);
                const refundPk = makerMdsPk || getPrimaryMssPk();

                // Reconstruct the EXACT covenant bytecode so the coin id + address match what was locked.
                const covScriptHex = build_limit_order_covenant_bytecode_hex(
                    secretHash, BigInt(maxClaim), BigInt(timeoutHeight), refundPk
                );
                const covAddr = blake3_hash_hex(covScriptHex);
                const expectId = compute_coin_id_hex(covAddr, BigInt(coinValue), normalizeHex(coin.salt));
                if (normalizeHex(expectId) !== normalizeHex(coin.coin_id))
                    throw new Error("Coin does not match the reconstructed covenant address/params");

                const myAddr = compute_p2pk_address_hex(refundPk);
                // coinValue is a power of two => one coin back to the maker; the fee comes from wallet inputs.
                const pow2 = (n) => { const out = []; let v = BigInt(n), bit = 0n; while (v > 0n) { if (v & 1n) out.push(Number(1n << bit)); v >>= 1n; bit += 1n; } return out; };
                const outputsJson = JSON.stringify(pow2(coinValue).map(v => ({ out_type: "standard", address: myAddr, value: v, salt: null })));

                // Witness filled AFTER the commitment is known (it carries our signature).
                const contractInputsJson = JSON.stringify([{
                    coin_id: normalizeHex(coin.coin_id), witness: "", value: coinValue, salt: normalizeHex(coin.salt), state: null
                }]);

                if (!mssCachesReady) await loadMssCaches();
                const utxoArray = getSpendableUtxos().map(u =>
                    (u.is_mss && wState.mssAddrs[u.address]) ? { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf } : u);

                let ctx = JSON.parse(wallet.prepare_script_spend(
                    JSON.stringify(utxoArray), covScriptHex, contractInputsJson, outputsJson, wState.nextWotsIndex
                ));
                // Sign the spend with the refund key, then inject the ELSE-branch witness [sig, dummy(32B), 0x00].
                const sigHex = await signMssAndSync(refundPk, ctx.commitment);
                const dummy = "00".repeat(32);
                for (let i = 0; i < ctx.contract_inputs.length; i++) ctx.contract_inputs[i].witness = `${sigHex},${dummy},00`;

                const revealPayloadStr = wallet.build_script_reveal(JSON.stringify(ctx), ctx.commitment, ctx.tx_salt);

                await reserveAndLock(ctx, "Saving wallet state...");

                dexPhase("Mining spam-proof (PoW)\u2026");
                const stateData = await rpc.getState();
                await new Promise(r => setTimeout(r, 50));
                const spamNonce = Number(mine_commitment_pow(ctx.commitment, stateData.required_pow || 24, BigInt(stateData.height), stateData.header_hash));
                const commitReq = await rpc.commit(ctx.commitment, spamNonce);
                if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);
                dexPhase("Commit broadcast — waiting to be mined (1/2)\u2026");
                await awaitCommitmentMined(ctx.commitment, dexPhase);
                const revealReq = await rpc.send(revealPayloadStr);
                if (!revealReq.ok) throw new Error(`Reveal rejected: ${revealReq.body || revealReq.error}`);
                dexPhase("Reveal broadcast — waiting to be mined (2/2)\u2026");
                const firstCoin = normalizeHex(coin.coin_id);
                await awaitCoinSpent(firstCoin, dexPhase);
                dexPhase("Confirmed \u2713 — syncing\u2026");
                await performScan();
                // CLEANUP (dex_secrets invariant I4): the covenant coin no longer exists, so its
                // preimage guards nothing and its cancellation marker is moot — purge both.
                {
                    const cid = normalizeHex(coin.coin_id);
                    let dirty = false;
                    if (wState.dex_secrets && wState.dex_secrets[cid]) { delete wState.dex_secrets[cid]; dirty = true; }
                    if (wState.dex_cancelled && wState.dex_cancelled[cid]) { delete wState.dex_cancelled[cid]; dirty = true; }
                    if (dirty) await saveState();
                }
                self.postMessage({ type: 'DEX_REFUND_SUCCESS', payload: { offerId, reclaimed: coinValue } });
            } catch (err) {
                const detail = (err && err.message) ? err.message : String(err);
                self.postMessage({ type: 'DEX_REFUND_FAILED', payload: { offerId: payload && payload.offerId, error: detail } });
            } finally {
                releaseSendLock();
            }
        }
        else if (type === 'DEX_CANCEL_LIMIT') {
            // FEATURE 4 — maker withdraws a live limit-order unit from the book.
            // REASONING: the covenant has exactly two spend branches — secret-claim (a fill)
            //   and post-timeout refund. There is no third "cancel" branch, so cancellation
            //   cannot be an on-chain act before the timeout. What the maker CAN do is stop
            //   offering: stop advertising, stop serving the order to scanners, and stop
            //   auto-claiming any late taker lock. That is a purely LOCAL state change, and
            //   because the scan filter and the maker watch both read it from the encrypted
            //   wallet state, it survives reloads and devices that share the wallet bundle.
            // PRE:  payload.coinId names a covenant coin (ideally one of ours; marking a
            //       foreign id is harmless — it only hides that order from OUR book).
            // POST: wState.dex_cancelled[coinId] set; if we own the secret record its status
            //       is 'limit_cancelled'; state persisted; DEX_CANCEL_DONE emitted.
            // SAFETY: no chain interaction, no key material touched, nothing broadcast. The
            //       MDS stays locked until the timeout, when the normal refund path (which
            //       purges both records on success) reclaims it. A taker who saw the order
            //       pre-cancel can still fill it until the timeout — cancellation is a
            //       policy statement, not a revocation; the UI copy says exactly this.
            try {
                const { offerId, coinId } = payload || {};
                if (!coinId) throw new Error("No coinId supplied");
                const cid = normalizeHex(coinId);
                wState.dex_cancelled = wState.dex_cancelled || {};
                wState.dex_cancelled[cid] = { at: Date.now(), offerId: offerId || null };
                if (wState.dex_secrets && wState.dex_secrets[cid]) {
                    wState.dex_secrets[cid].status = 'limit_cancelled';
                }
                await saveState();
                self.postMessage({ type: 'DEX_CANCEL_DONE', payload: { offerId: offerId || null, coinId: cid } });
            } catch (err) {
                self.postMessage({ type: 'DEX_CANCEL_FAILED', payload: { offerId: payload && payload.offerId, error: (err && err.message) || String(err) } });
            }
        }
        else if (type === 'DEX_GET_SECRET') {
            // FEATURE 5 — serve a preimage from the encrypted store to the UI, on demand.
            // REASONING: the maker's Base claim (startMakerLimitWatch) is the ONE moment the
            //   UI legitimately needs a preimage. Since secrets no longer ride on swap
            //   records, the UI asks for exactly one, exactly when claiming — the secret is
            //   in UI memory for the duration of a contract call instead of at rest forever.
            // PRE:  payload.coinId is a 64-hex coin id.
            // POST: DEX_SECRET_RESULT carries { coinId, secret, secretHash, status } if — and
            //       only if — wState.dex_secrets holds a record for it (i.e. it is OURS);
            //       otherwise { coinId, error }. Lookup is by coin_id ONLY: no enumeration,
            //       no lookup by hash, offerId, or address.
            // SAFETY: dex_secrets contains exclusively self-generated records, so this can
            //       never serve another party's preimage. Nothing is logged.
            try {
                const cid = normalizeHex((payload && payload.coinId) || "");
                const rec = (cid && wState.dex_secrets) ? wState.dex_secrets[cid] : null;
                if (!rec || !rec.secret) {
                    self.postMessage({ type: 'DEX_SECRET_RESULT', payload: { coinId: cid, error: "No secret held for this coin." } });
                } else {
                    self.postMessage({ type: 'DEX_SECRET_RESULT', payload: { coinId: cid, secret: rec.secret, secretHash: rec.secretHash, status: rec.status } });
                }
            } catch (err) {
                self.postMessage({ type: 'DEX_SECRET_RESULT', payload: { coinId: payload && payload.coinId, error: (err && err.message) || String(err) } });
            }
        }
        else if (type === 'DEX_SECRET_CONSUMED') {
            // FEATURE 1 (cleanup on fill) — the UI reports that the maker's Base claim tx
            // CONFIRMED. Claiming publishes the preimage on Base, so from this moment the
            // secret has zero confidentiality value and the record only adds surface — purge
            // it (dex_secrets invariant I4). Called strictly AFTER tx.wait(): a record must
            // never be deleted for a claim that might still fail.
            try {
                const cid = normalizeHex((payload && payload.coinId) || "");
                let dirty = false;
                if (cid && wState.dex_secrets && wState.dex_secrets[cid]) { delete wState.dex_secrets[cid]; dirty = true; }
                if (cid && wState.dex_cancelled && wState.dex_cancelled[cid]) { delete wState.dex_cancelled[cid]; dirty = true; }
                if (dirty) await saveState();
            } catch (_) { /* best-effort cleanup; an orphan record is harmless */ }
        }
        else if (type === 'DEX_IMPORT_SECRETS') {
            // FEATURE 1 (one-time migration) — absorb legacy plaintext limit-order secrets
            // that older builds stored on localStorage swap records, so the UI can strip them.
            // PRE:  wallet unlocked (password set — saveState() would otherwise no-op and the
            //       UI would strip secrets that were never persisted: refuse instead).
            // POST: every well-formed entry exists in wState.dex_secrets (existing records are
            //       NEVER overwritten — the encrypted store is senior to localStorage); state
            //       persisted; DEX_SECRETS_IMPORTED lists the coinIds now safe to strip.
            // SAFETY: import direction is localStorage → encrypted store only. The reply
            //       carries coin ids, never secret material.
            try {
                if (!password) throw new Error("Wallet is locked — cannot persist migrated secrets.");
                const entries = (payload && Array.isArray(payload.entries)) ? payload.entries : [];
                wState.dex_secrets = wState.dex_secrets || {};
                const imported = [];
                for (const e of entries) {
                    if (!e || !e.coinId || !e.secret) continue;
                    const cid = normalizeHex(e.coinId);
                    if (!wState.dex_secrets[cid]) {
                        wState.dex_secrets[cid] = {
                            secret: e.secret, secretHash: e.secretHash || blake3_hash_hex(e.secret),
                            covAddr: e.covAddr || null, value: Number(e.value) || 0, salt: e.salt || null,
                            timeoutHeight: Number(e.timeoutHeight) || 0,
                            makerMdsPk: e.makerMdsPk || null, makerEvmAddr: e.makerEvmAddr || null,
                            weiAmount: e.weiAmount != null ? String(e.weiAmount) : null,
                            offerId: e.offerId || null, groupId: e.groupId || null,
                            status: 'live', createdAt: Date.now(), migrated: true
                        };
                    }
                    imported.push(cid);   // already-present ids are also safe to strip
                }
                if (imported.length) await saveState();
                self.postMessage({ type: 'DEX_SECRETS_IMPORTED', payload: { coinIds: imported } });
            } catch (err) {
                self.postMessage({ type: 'DEX_SECRETS_IMPORTED', payload: { coinIds: [], error: (err && err.message) || String(err) } });
            }
        }
        else if (type === 'DEX_RECLAIM_HTLC') {
            // Reclaim a STUCK taker-side (or any) HTLC lock after its timeout. This is the
            // safety net for a half-completed swap: the MDS was locked in an HTLC covenant
            // and the swap stalled, so we spend the refund branch back to ourselves. Handles
            // BOTH lock shapes — covenant HTLC (build_covenant_htlc_bytecode_hex, refund =
            // refund_pk after timeout) and classic HTLC (build_htlc_bytecode_hex). A lock
            // funds a SET of power-of-two coins, so we reclaim each in turn.
            await acquireSendLock();
            try {
                const { offerId, htlcCoins, secretHash, timeoutHeight, refundPk, receiverAddr, minPayout, swapMode } = payload;
                const dexPhase = (p) => { if (offerId) self.postMessage({ type: 'DEX_PHASE', payload: { offerId, phase: p } }); };
                if (!Array.isArray(htlcCoins) || htlcCoins.length === 0) throw new Error("No locked coins to reclaim");
                if (networkHeight < Number(timeoutHeight))
                    throw new Error(`Too early — refund unlocks at height ${timeoutHeight} (now ${networkHeight}).`);

                const myPk = refundPk || getPrimaryMssPk();
                const myAddr = compute_p2pk_address_hex(myPk);
                const covenant = (swapMode === 'covenant');

                // Reconstruct the EXACT script that was locked, so coin ids/address match.
                let scriptHex;
                if (covenant) {
                    if (!receiverAddr) throw new Error("Missing receiver address for covenant HTLC reclaim");
                    scriptHex = build_covenant_htlc_bytecode_hex(
                        secretHash, receiverAddr, BigInt(minPayout || 0), BigInt(timeoutHeight), myPk
                    );
                } else {
                    scriptHex = build_htlc_bytecode_hex(secretHash, receiverAddr, BigInt(timeoutHeight), myPk);
                }
                const scriptAddr = blake3_hash_hex(scriptHex);

                const pow2 = (n) => { const out = []; let v = BigInt(n), bit = 0n; while (v > 0n) { if (v & 1n) out.push(Number(1n << bit)); v >>= 1n; bit += 1n; } return out; };
                let totalReclaimed = 0, reclaimedCoins = 0;

                for (const coin of htlcCoins) {
                    if (!coin || !coin.coin_id || !coin.salt) continue;
                    const coinValue = Number(coin.value);
                    // Verify the coin actually belongs to this reconstructed script.
                    const expectId = compute_coin_id_hex(scriptAddr, BigInt(coinValue), normalizeHex(coin.salt));
                    if (normalizeHex(expectId) !== normalizeHex(coin.coin_id)) {
                        self.postMessage({ type: 'LOG', payload: `Reclaim: coin ${coin.coin_id.slice(0,12)}… doesn't match reconstructed HTLC — skipping.` });
                        continue;
                    }
                    // Skip coins already spent (e.g. a partially-completed swap).
                    try { const chk = await rpc.checkCoin(normalizeHex(coin.coin_id)); if (chk && !chk.exists) { continue; } } catch (_) {}

                    const outputsJson = JSON.stringify(pow2(coinValue).map(v => ({ out_type: "standard", address: myAddr, value: v, salt: null })));
                    const contractInputsJson = JSON.stringify([{ coin_id: normalizeHex(coin.coin_id), witness: "", value: coinValue, salt: normalizeHex(coin.salt), state: null }]);

                    if (!mssCachesReady) await loadMssCaches();
                    const utxoArray = getSpendableUtxos().map(u =>
                        (u.is_mss && wState.mssAddrs[u.address]) ? { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf } : u);

                    let ctx = JSON.parse(wallet.prepare_script_spend(
                        JSON.stringify(utxoArray), scriptHex, contractInputsJson, outputsJson, wState.nextWotsIndex
                    ));
                    // Refund branch witness: [sig, dummy(32B), 0x00] selects the ELSE/timeout path.
                    const sigHex = await signMssAndSync(myPk, ctx.commitment);
                    const dummy = "00".repeat(32);
                    for (let i = 0; i < ctx.contract_inputs.length; i++) ctx.contract_inputs[i].witness = `${sigHex},${dummy},00`;

                    const revealPayloadStr = wallet.build_script_reveal(JSON.stringify(ctx), ctx.commitment, ctx.tx_salt);
                    await reserveAndLock(ctx, "Saving wallet state...");

                    dexPhase(`Reclaiming coin ${reclaimedCoins + 1}/${htlcCoins.length} — mining PoW…`);
                    const stateData = await rpc.getState();
                    await new Promise(r => setTimeout(r, 50));
                    const spamNonce = Number(mine_commitment_pow(ctx.commitment, stateData.required_pow || 24, BigInt(stateData.height), stateData.header_hash));
                    const commitReq = await rpc.commit(ctx.commitment, spamNonce);
                    if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);
                    await awaitCommitmentMined(ctx.commitment, dexPhase);
                    const revealReq = await rpc.send(revealPayloadStr);
                    if (!revealReq.ok) throw new Error(`Reveal rejected: ${revealReq.body || revealReq.error}`);
                    await awaitCoinSpent(normalizeHex(coin.coin_id), dexPhase);

                    totalReclaimed += coinValue; reclaimedCoins++;
                }

                dexPhase("Confirmed ✓ — syncing…");
                await performScan();
                if (reclaimedCoins === 0) throw new Error("Nothing to reclaim — coins already spent or not found on-chain.");
                self.postMessage({ type: 'DEX_RECLAIM_SUCCESS', payload: { offerId, reclaimed: totalReclaimed, coins: reclaimedCoins } });
            } catch (err) {
                const detail = (err && err.message) ? err.message : String(err);
                self.postMessage({ type: 'DEX_RECLAIM_FAILED', payload: { offerId: payload && payload.offerId, error: detail } });
            } finally {
                releaseSendLock();
            }
        }
        else if (type === 'DEX_LOCK_L2') {
            // FEATURE 2 — submarine-swap intercept (MAKER side). Called by the EVM watcher
            // when the taker's ETH lock for an L2-settled offer is seen. Adds an HTLC over
            // an open Q-Bolt v2 channel to the taker with the SAME hashlock instead of a
            // DEX_LOCK_MIDSTATE on-chain lock.
            //
            // v2: we must be the SENDER on a channel to the taker (only the sender can add
            // HTLCs), the channel must be acked and active with enough sender-side balance,
            // and it must have enough life left to outlive the HTLC.
            const { offerId, takerL2Pk, mdsAmount, secretHash, baseRefundSecs } = payload;
            const takerPk = String(takerL2Pk || '').substring(0, 64);
            const amt = Number(mdsAmount);

            // Tie the L2 HTLC timeout under the Base refund window. htlcTimeout is a MIDSTATE
            // height (~60 s blocks), so convert seconds with /60. Half the window leaves the
            // maker ample Base-lock time to sweep after the latest possible L2 claim reveals
            // the preimage. Floor it so a very short Base window still yields a sane HTLC.
            const htlcTimeout = networkHeight + Math.max(QB.HTLC_MIN_HEADROOM + QB.HTLC_HOP_DELTA, Math.floor(Number(baseRefundSecs) / 60 / 2));

            let chan = null;
            for (const c of Object.values(wState.l2_channels || {})) {
                if (c.v !== 2 || c.role !== 'sender' || c.status !== 'active' || !c.open_acked) continue;
                if (c.peer_pk !== takerPk) continue;
                if (c.latest.sender_amt < amt) continue;
                if (networkHeight >= c.expiry - QB.PAY_CUTOFF) continue;
                if (htlcTimeout > c.expiry + QB.HTLC_MAX_PAST_EXPIRY) continue;
                chan = c; break;
            }
            if (!chan) { self.postMessage({ type: 'DEX_LOCK_L2_FAILED', payload: { offerId, error: "No outbound Q-Bolt channel to the taker with enough capacity and lifetime." } }); return; }

            try {
                await qbSenderAdvance(chan, (next) => {
                    next.sender_amt -= amt;
                    next.htlcs.push({ amount: amt, timeout: htlcTimeout, secret_hash: secretHash });
                }, QB.CMD_HTLC_ADD, [{ kind: "address", value: takerPk }]);
                self.postMessage({ type: 'DEX_LOCK_L2_SENT', payload: { offerId, chanId: chan.id, secretHash } });
            } catch (e) {
                self.postMessage({ type: 'DEX_LOCK_L2_FAILED', payload: { offerId, error: String(e && e.message || e) } });
            }
        }

        else if (type === 'DEX_CHECK_L2_CHANNEL') {
            // FEATURE 2 pre-flight (TAKER side): verify — BEFORE any ETH is locked — that the
            // maker will be able to route. The maker adds the HTLC as SENDER on a channel to
            // us, so what we check is: does a channel exist where the MAKER is the sender toward
            // us (we are the receiver), active/acked, with enough maker-side balance and life?
            // Mirrors DEX_LOCK_L2's selection from the other side.
            const { offerId, peerPk, amount } = payload;
            const makerPk = String(peerPk || '').substring(0, 64);
            const amt = Number(amount);
            let ok = false, reason = "No Q-Bolt channel from the maker to you. Ask the maker to open one toward your node.";
            for (const c of Object.values(wState.l2_channels || {})) {
                if (c.v !== 2) continue;
                if (c.role !== 'receiver' || c.peer_pk !== makerPk) continue;
                if (c.status !== 'active' || !c.open_acked) { reason = "Your channel from the maker isn't active yet — retry in a moment."; continue; }
                if (networkHeight >= c.expiry - QB.PAY_CUTOFF) { reason = "The maker's channel to you is too close to expiry to route safely."; continue; }
                if (c.latest.sender_amt < amt) { reason = `The maker's channel balance (${c.latest.sender_amt} MDS) can't cover ${amt} MDS.`; continue; }
                ok = true; reason = null; break;
            }
            self.postMessage({ type: 'DEX_L2_CHANNEL_STATUS', payload: { offerId, ok, reason } });
        }

        else if (type === 'DEX_SUBMARINE_STATUS') {
            // FEATURE 2 read-only poll, used by BOTH submarine roles (a poll — unlike a push —
            // survives page reloads mid-swap):
            //   observedSecret — preimage harvested from a cmd-43 claim seen on the chat bus
            //                    (lets the MAKER sweep the Base ETH);
            //   claimedAmount  — an inbound HTLC our own wallet auto-claimed as destination
            //                    (lets the TAKER mark the swap complete);
            //   htlcPending    — the hash still sits in a live channel state (the maker's
            //                    re-route guard: re-issuing DEX_LOCK_L2 is only safe when the
            //                    HTLC is provably absent, otherwise it would double-pay).
            const { offerId, secretHash } = payload;
            const observedSecret = (wState.l2_observed_secrets && wState.l2_observed_secrets[secretHash]) || null;
            const claimedAmount = (wState.l2_claimed && wState.l2_claimed[secretHash] != null) ? wState.l2_claimed[secretHash] : null;
            let htlcPending = false;
            for (const c of Object.values(wState.l2_channels || {})) {
                const htlcs = (c.v === 2) ? (c.latest && c.latest.htlcs) : (c.latest_state && c.latest_state.htlcs);
                if ((htlcs || []).some(h => h.secret_hash === secretHash)) { htlcPending = true; break; }
            }
            self.postMessage({ type: 'DEX_SUBMARINE_STATUS_RESULT', payload: { offerId, secretHash, observedSecret, claimedAmount, htlcPending } });
        }

        else if (type === 'DEX_CHECK_SETTLED') {
            const { offerId, coinId } = payload;
            let settled = false;
            let success = false;
            try { 
                const chk = await rpc.checkCoin(normalizeHex(coinId)); 
                if (chk) {
                    settled = !chk.exists; 
                    success = true;
                }
            } catch (e) {}
            self.postMessage({ type: 'DEX_SETTLED_STATUS', payload: { offerId, settled, success } });
        }
        // ═══════════ ESCROWED-BID FILLS (Part 2) ═══════════
        else if (type === 'DEX_BIDFILL_LOCK') {
            // SELLER: lock MDS for a reserved Base bid. Same shape as the covenant
            // branch of DEX_LOCK_MIDSTATE, but the hashlock comes from the BID (the
            // bidder generated the secret) and the receiver is the bid's makerMdsAddr
            // verbatim — an address, not a pubkey to derive.
            await acquireSendLock();
            try {
                const { bidId, hashlock, receiverAddr, minPayout, timeoutHeight } = payload;
                const sh = normalizeHex(String(hashlock ?? '').replace(/^0x/i, '')), ra = normalizeHex(String(receiverAddr ?? '').replace(/^0x/i, ''));
                if (!/^[0-9a-f]{64}$/.test(sh) || !/^[0-9a-f]{64}$/.test(ra)) throw new Error("Bad hashlock/receiver");
                const myPk = getPrimaryMssPk();
                const covScriptHex = build_covenant_htlc_bytecode_hex(sh, ra, BigInt(minPayout), BigInt(timeoutHeight), myPk);
                const covAddr = blake3_hash_hex(covScriptHex);
                // Over-fund by the fee budget: the bidder's claim fee comes out of the
                // locked value, so a bidder holding zero MDS can still claim.
                const fundRes = await performContractTx({
                    reqId: 4200, kind: 'fund',
                    contractAddress: covAddr,
                    amount: Number(minPayout) + COVENANT_FEE_BUDGET
                });
                const coins = (fundRes && fundRes.coins) || [];
                if (!coins.length) throw new Error("Funding produced no coins");
                self.postMessage({ type: 'DEX_BIDFILL_LOCKED', payload: {
                    bidId, coins, secretHash: sh, receiverAddr: ra,
                    minPayout: Number(minPayout), timeoutHeight: Number(timeoutHeight),
                    refundPk: myPk, lockedAtHeight: networkHeight
                }});
            } catch (err) {
                self.postMessage({ type: 'DEX_BIDFILL_LOCK_FAILED', payload: { bidId: payload && payload.bidId, error: (err && err.message) || String(err) } });
            } finally { releaseSendLock(); }
        }
        else if (type === 'DEX_BIDFILL_VERIFY') {
            // BIDDER: trustlessly verify a seller's fill-announce. The bytecode is
            // rebuilt LOCALLY from the announce params + our own expectations; each
            // coin id must recompute from (covAddr‖value‖salt) and exist on-chain.
            try {
                const { bidId, coins, secretHash, receiverAddr, minPayout, timeoutHeight, refundPk } = payload;
                const fail = (reason) => self.postMessage({ type: 'DEX_BIDFILL_VERIFY_RESULT', payload: { bidId, ok: false, reason } });
                if (!Array.isArray(coins) || !coins.length) return fail("no coins");
                if (Number(timeoutHeight) < networkHeight + 45) return fail("covenant timeout too soon to claim safely");
                if (Number(timeoutHeight) > networkHeight + 5000) return fail("covenant timeout implausibly far");
                const covScriptHex = build_covenant_htlc_bytecode_hex(
                    normalizeHex(String(secretHash ?? '').replace(/^0x/i, '')), normalizeHex(String(receiverAddr ?? '').replace(/^0x/i, '')),
                    BigInt(minPayout), BigInt(timeoutHeight), normalizeHex(String(refundPk ?? '').replace(/^0x/i, ''))
                );
                const covAddr = blake3_hash_hex(covScriptHex);
                let total = 0;
                for (const c of coins) {
                    const expect = compute_coin_id_hex(covAddr, BigInt(c.value), normalizeHex(String(c.salt ?? '').replace(/^0x/i, '')));
                    if (normalizeHex(String(expect ?? '').replace(/^0x/i, '')) !== normalizeHex(String(c.coin_id ?? '').replace(/^0x/i, ''))) return fail("coin id does not match covenant params");
                    const chk = await rpc.checkCoin(normalizeHex(String(c.coin_id ?? '').replace(/^0x/i, '')));
                    if (!chk || !chk.exists) return fail("covenant coin not found on-chain (yet?)");
                    total += Number(c.value);
                }
                if (total < Number(minPayout) + 1024) return fail("locked value too small to pay out + claim fee");

                self.postMessage({ type: 'DEX_BIDFILL_VERIFY_RESULT', payload: { bidId, ok: true } });
            } catch (err) {
                self.postMessage({ type: 'DEX_BIDFILL_VERIFY_RESULT', payload: { bidId: payload && payload.bidId, ok: false, reason: (err && err.message) || String(err) } });
            }
        }
        else if (type === 'DEX_BIDFILL_CLAIM') {
            // BIDDER: claim the seller's covenant with OUR OWN secret — this is the
            // reveal that lets the seller collect the ETH on Base. Mirrors
            // DEX_SETTLE_COVENANT, except the receiver address is taken verbatim
            // (it's the bid's makerMdsAddr — ours) instead of derived from a pubkey.
            await acquireSendLock();
            try {
                const { bidId, rawSecret, receiverAddr, minPayout, timeoutHeight, refundPk, htlcCoins } = payload;
                if (!Array.isArray(htlcCoins) || htlcCoins.length === 0) throw new Error("No covenant coins to claim");
                const secretHash = blake3_hash_hex(normalizeHex(String(rawSecret ?? '').replace(/^0x/i, '')));
                const ra = normalizeHex(String(receiverAddr ?? '').replace(/^0x/i, ''));
                const covScriptHex = build_covenant_htlc_bytecode_hex(
                    secretHash, ra, BigInt(minPayout), BigInt(timeoutHeight), normalizeHex(String(refundPk ?? '').replace(/^0x/i, ''))
                );

                if (!mssCachesReady) await loadMssCaches();
                const utxoArray = getSpendableUtxos().map(u => {
                    if (u.is_mss && wState.mssAddrs[u.address]) return { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf };
                    return u;
                });

                const payout = Number(minPayout);
                const pow2Parts = [];
                { let n = BigInt(payout), bit = 0n;
                  while (n > 0n) { if (n & 1n) pow2Parts.push(Number(1n << bit)); n >>= 1n; bit += 1n; } }
                const outputsJson = JSON.stringify(pow2Parts.map(v => ({
                    out_type: "standard", address: ra, value: v, salt: null
                })));
                const contractInputsJson = JSON.stringify(htlcCoins.map(c => ({
                    coin_id: normalizeHex(String(c.coin_id ?? '').replace(/^0x/i, '')),
                    witness: `${normalizeHex(String(rawSecret ?? '').replace(/^0x/i, ''))},01`,
                    value: Number(c.value),
                    salt: normalizeHex(String(c.salt ?? '').replace(/^0x/i, '')),
                    state: null
                })));

                let ctx = JSON.parse(wallet.prepare_script_spend(
                    JSON.stringify(utxoArray), covScriptHex, contractInputsJson, outputsJson, wState.nextWotsIndex
                ));
                const revealPayloadStr = wallet.build_script_reveal(JSON.stringify(ctx), ctx.commitment, ctx.tx_salt);
                await reserveAndLock(ctx, "Saving wallet state...");

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Mining PoW..." } });
                const stateData = await rpc.getState();
                await new Promise(r => setTimeout(r, 50));
                const spamNonce = Number(mine_commitment_pow(ctx.commitment, stateData.required_pow || 24, BigInt(stateData.height), stateData.header_hash));
                const commitReq = await rpc.commit(ctx.commitment, spamNonce);
                if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);
                await awaitCommitmentMined(ctx.commitment, (m) => self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: m } }));
                const revealReq = await rpc.send(revealPayloadStr);
                if (!revealReq.ok) throw new Error(`Reveal rejected: ${revealReq.body || revealReq.error}`);
                const firstCoin = normalizeHex(String(htlcCoins[0].coin_id ?? '').replace(/^0x/i, ''));
                await awaitCoinSpent(firstCoin, (m) => self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: m } }));
                await performScan();
                self.postMessage({ type: 'DEX_BIDFILL_CLAIMED', payload: { bidId } });
            } catch (err) {
                self.postMessage({ type: 'DEX_BIDFILL_CLAIM_FAILED', payload: { bidId: payload && payload.bidId, error: (err && err.message) || String(err) } });
            } finally { releaseSendLock(); }
        }
        else if (type === 'DEX_BIDFILL_WATCH') {
            // SELLER: has the covenant coin been spent, and if so what preimage was
            // revealed? checkCoin answers the first; for the second we scan recent
            // block JSON for a 32-byte hex whose BLAKE3 equals the hashlock — the
            // witness [secret, 0x01] serializes into the batch, so it's in there.
            try {
                const { bidId, coinId, hashlock, sinceHeight } = payload;
                const chk = await rpc.checkCoin(normalizeHex(String(coinId ?? '').replace(/^0x/i, '')));
                if (chk && chk.exists) {
                    self.postMessage({ type: 'DEX_BIDFILL_WATCH_RESULT', payload: { bidId, spent: false, height: networkHeight } });
                } else {
                    let preimage = null;
                    const want = normalizeHex(String(hashlock ?? '').replace(/^0x/i, ''));
                    const from = Math.max(1, Math.max(Number(sinceHeight) || 0, networkHeight - 180));
                    for (let h = networkHeight; h >= from && !preimage; h--) {
                        let blk;
                        try { blk = await rpc.getBlock(h); }
                        catch (e) {
                            // On gateway rate-limiting, stop the backward walk —
                            // it would just be up to 180 more doomed requests.
                            // preimage stays null, which the result handler
                            // already tolerates; the next watch retries.
                            if (e && (e.code === 'RATE_LIMITED' || e.code === 'GATEWAY_UNAVAILABLE')) break;
                            continue;
                        }
                        if (!blk) continue;
                        const seen = new Set();
                        for (const cand of (JSON.stringify(blk).match(/[0-9a-fA-F]{64}/g) || [])) {
                            const c = cand.toLowerCase();
                            if (seen.has(c) || c === want) continue;
                            seen.add(c);
                            try { if (blake3_hash_hex(c) === want) { preimage = c; break; } } catch (e) {}
                        }
                    }
                    self.postMessage({ type: 'DEX_BIDFILL_WATCH_RESULT', payload: { bidId, spent: true, preimage, height: networkHeight } });
                }
            } catch (err) {
                self.postMessage({ type: 'DEX_BIDFILL_WATCH_RESULT', payload: { bidId: payload && payload.bidId, spent: false, error: (err && err.message) || String(err), height: networkHeight } });
            }
        }

        else if (type === 'PUSH_NEW_BLOCK') {
            if (payload.ChatMessage) {
                if (payload.ChatMessage.words && payload.ChatMessage.words[0] === 255) {
                    handleL2Chat(payload.ChatMessage).catch(()=>{});
                } else {
                    self.postMessage({ type: 'CHAT_MESSAGE', payload: payload.ChatMessage });
                }
                return;
            }
            const notif = payload.NewBlockTip;
            if (!notif) return;
            
            networkHeight = notif.height;
            self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
            self.postMessage({ type: 'LOG', payload: `⚡ Network Push: New block found at height ${notif.height}!` });
            
            // Awake any pending transaction operations instantly!
            const resolvers = nextBlockResolvers;
            nextBlockResolvers = [];
            resolvers.forEach(r => r());

            // Q-Bolt heartbeat: resume closes, auto-close/refund on schedule,
            // reconcile external closes, resolve on-chain HTLCs.
            if (wallet) qbWatchTick(notif.height).catch(e => qbLog(`watch: ${e && e.message || e}`));
            
            // 1. Instant Miner Update: Stop wasting hashes, get the new template!
            if (isMiningActive) {
                handleGetTemplate().then(tmpl => {
                    if (tmpl) self.postMessage({ type: 'TEMPLATE_READY', payload: tmpl });
                }).catch(()=>{});
            }
            
            // 2. Instant Sync: Check the block's filter for incoming funds!
            if (notif.filter_hex && notif.block_hash && notif.element_count > 0) {
                const matched = wallet.check_filter(notif.filter_hex, notif.block_hash, notif.element_count);
                if (matched) {
                    self.postMessage({ type: 'LOG', payload: `Incoming funds detected! Auto-scanning...` });
                    performScan().catch(()=>{});
                } else if (
                    !isScanning &&
                    notif.height === wState.lastScannedHeight + 1
                ) {
                    // Caught up and the new block is exactly the next one — safe fast-path.
                    wState.lastScannedHeight = notif.height;
                    saveState();
                    self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
                } else if (notif.height > wState.lastScannedHeight + 1) {
                    // We're behind by more than one block — there's actual history to
                    // scan. Don't fast-forward the marker; let performScan handle it.
                    performScan().catch(()=>{});
                }
            }
        }
        
        else if (type === 'CREATE') {
            scanGeneration++;
            while (isScanning) {
                await new Promise(r => setTimeout(r, 50));
            }
            if (wallet) wallet.free();
            password = payload.password;
            wState = {
                phrase: payload.phrase,
                nextWotsIndex: 0, nextMssIndex: 0,
                wotsAddrs: {}, spentWots: {},pendingSpends: {}, mssAddrs: {}, utxos: {}, history: [],
                lastScannedHeight: 0,
                l2_channels: {}, l2_secrets: {}, l2_routes: {},
                l2_invoices: {}, l2_inv_reqs: {}, l2_pay_pending: {}, qb_fwd: {},
                qb_pending_opens: {}, qb_open_intent: null, pending_tx: null,
                l2_observed_secrets: {}, l2_claimed: {},
                dex_secrets: {}, dex_cancelled: {}, pendingLimitBundles: {}, annFragPool: {}
            };
            wallet = new WebWallet(payload.phrase);
            mssCachesReady = false;

            // Pre-derive GAP_LIMIT WOTS addresses for chain scanning
            for (let i = 0; i < GAP_LIMIT; i++) {
                deriveNextWots();
                if (i % 10 === 0) {
                    self.postMessage({ type: 'MSS_PROGRESS', payload: { current: i, total: GAP_LIMIT, label: `Deriving base keys (${i}/${GAP_LIMIT})...` } });
                    await new Promise(r => setTimeout(r, 0));
                }
            }

            // Generate the first MSS (reusable) address
            self.postMessage({ type: 'MSS_PROGRESS', payload: { current: 0, total: 100, label: "Generating Post-Quantum MSS Address..." } });
            await new Promise(r => setTimeout(r, 10));
            await deriveNextMss(10);
            mssCachesReady = true;

            await saveState();
            self.postMessage({ type: 'WALLET_LOADED', payload: buildDashboardPayload() });
            self.postMessage({ type: 'AUTO_CONNECT_WEBRTC' });
        }

        else if (type === 'LOGIN') {
            await loadState(payload.password, payload.bundleStr);
            self.postMessage({ type: 'AUTO_CONNECT_WEBRTC' });
            // Re-broadcast anything that was mid-flight when we last stopped,
            // before Q-Bolt reasons about channels that may depend on it.
            recoverPendingTx()
                .catch(e => self.postMessage({ type: 'LOG', payload: `[recovery] ${e && e.message || e}` }))
                .finally(() => qbBoot().catch(e => qbLog(`boot: ${e && e.message || e}`)));
        }

        else if (type === 'SCAN') {
            await performScan();
        }

        else if (type === 'RESCAN') {
            // Cancel any in-flight scan (its generation check will bail without
            // committing partial state) and ask the wrapper to do an atomic
            // wipe-then-scan in its next iteration.
            scanGeneration++;
            scanResetPending = true;
            await performScan();
        }

        else if (type === 'SEND') {
            await acquireSendLock();
            try { await performSend(payload.toAddress, payload.amount, null, 0, !!payload.sendAll); }
            catch (err) {
                // Surface send failures as a distinct, persistent error the UI can pin —
                // never let a send just silently stop. Include the message verbatim.
                self.postMessage({ type: 'SEND_ERROR', payload: { error: (err && err.message) ? err.message : String(err) } });
            }
            finally { releaseSendLock(); }
        }
        else if (type === 'DEFRAG_UTXOS') {
            await acquireSendLock();
            try {
                let utxos = getSpendableUtxos();
                if (utxos.length < 2) throw new Error("Not enough UTXOs to defragment.");

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Preparing defragmentation..." } });

                if (!mssCachesReady) await loadMssCaches();
                await verifyMssSafetyIndices((m) => self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: m } }));

                utxos = utxos.map(u => {
                    if (u.is_mss && wState.mssAddrs[u.address]) return { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf };
                    return u;
                });

                let destMssPkHex = getPrimaryMssPk();
                if (!destMssPkHex) {
                    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Generating MSS destination address..." } });
                    await deriveNextMss(10);
                    destMssPkHex = getPrimaryMssPk();
                }
                const destAddrHex = compute_p2pk_address_hex(destMssPkHex);

                const spendContextStr = wallet.prepare_defrag(
                    JSON.stringify(utxos),
                    destAddrHex,
                    250, // max inputs
                    wState.nextWotsIndex
                );

                const ctx = JSON.parse(spendContextStr);

                await reserveAndLock(ctx, "Saving wallet state...");

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Mining PoW..." } });
                const stateData = await rpc.getState();
                await new Promise(r => setTimeout(r, 50));
                const spamNonce = Number(mine_commitment_pow(ctx.commitment, stateData.required_pow || 24, BigInt(stateData.height), stateData.header_hash));

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Broadcasting commit..." } });
                const commitReq = await rpc.commit(ctx.commitment, spamNonce);
                if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Waiting for commit to be mined (1/2)..." } });
                await awaitCommitmentMined(ctx.commitment, (m) => self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: m } }));

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Broadcasting defrag reveal..." } });
                
                // standard reveal, because it's a cross-address sweep, NOT a single-address consolidate!
                const revealPayloadStr = wallet.build_reveal(spendContextStr, ctx.commitment, ctx.tx_salt);
                
                const revealReq = await rpc.send(revealPayloadStr);
                if (!revealReq.ok) throw new Error(`Defrag rejected: ${revealReq.body || revealReq.error}`);

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Waiting for defrag to be mined (2/2)..." } });
                const firstCoinId = ctx.input_coin_ids[0];
                await awaitCoinSpent(firstCoinId, (m) => self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: m } }));

                pendingSends = []; // Clear locks
                await performScan();
                self.postMessage({ type: 'SEND_COMPLETE', payload: buildDashboardPayload() });
                self.postMessage({ type: 'LOG', payload: `Successfully defragmented ${ctx.selected_inputs.length} UTXOs into a single MSS address.` });
            } catch (err) {
                self.postMessage({ type: 'SEND_ERROR', payload: { error: err.message || String(err) } });
            } finally {
                releaseSendLock();
            }
        }
        else if (type === 'CONSOLIDATE_UTXOS') {
            await acquireSendLock();
            try {
                const targetAddr = payload.address;
                let utxosAtAddress = getSpendableUtxos().filter(u => u.address === targetAddr);
                
                if (utxosAtAddress.length < 2) throw new Error("Need at least 2 UTXOs to consolidate.");
                // Hard consensus cap for a single transaction
                if (utxosAtAddress.length > 5000) utxosAtAddress = utxosAtAddress.slice(0, 5000);


                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: `Preparing consolidation of ${utxosAtAddress.length} coins...` } });

                if (!mssCachesReady) await loadMssCaches();
                await verifyMssSafetyIndices((m) => self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: m } }));

                // SAFETY: Ensure local MSS indices are attached for signing
                utxosAtAddress = utxosAtAddress.map(u => {
                    if (u.is_mss && wState.mssAddrs[u.address]) return { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf };
                    return u;
                });

                let destMssPkHex = getPrimaryMssPk();
                if (!destMssPkHex) {
                    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Generating MSS address for consolidation output..." } });
                    await deriveNextMss(10);
                    destMssPkHex = getPrimaryMssPk();
                }
                let destAddrHex = compute_p2pk_address_hex(destMssPkHex);

                // --- THIS IS THE MAGIC: One clean call to the new Rust method ---
                let spendContextStr = wallet.prepare_consolidate(
                    JSON.stringify(utxosAtAddress), 
                    destAddrHex, 
                    wState.nextWotsIndex
                );
                
                let ctx = JSON.parse(spendContextStr);

                // Advance key material indices safely
                await reserveAndLock(ctx, "Saving wallet state...");

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Mining PoW..." } });
                const stateData = await rpc.getState();
                await new Promise(r => setTimeout(r, 50));
                const spamNonce = Number(mine_commitment_pow(ctx.commitment, stateData.required_pow || 24, BigInt(stateData.height), stateData.header_hash));

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Broadcasting commit..." } });
                const commitReq = await rpc.commit(ctx.commitment, spamNonce);
                if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Waiting for commit to be mined (1/2)..." } });
                await awaitCommitmentMined(ctx.commitment, (m) => self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: m } }));

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Broadcasting consolidate reveal..." } });
                
                // CALL THE NEW RUST METHOD! (Lightning fast, only 1 signature)
                let revealPayloadStr = wallet.build_consolidate_reveal(spendContextStr);
                let revealPayload = JSON.parse(revealPayloadStr);

                // Add the routing flag for the node
                revealPayload.is_consolidate = true;

                const revealReq = await rpc.send(JSON.stringify(revealPayload));
                if (!revealReq.ok) throw new Error(`Sweep rejected: ${revealReq.body || revealReq.error}`);

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Waiting for sweep to be mined (2/2)..." } });
                const firstCoinId = ctx.input_coin_ids[0];
                await awaitCoinSpent(firstCoinId, (m) => self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: m } }));

                await performScan();
                self.postMessage({ type: 'SEND_COMPLETE', payload: buildDashboardPayload() });
                self.postMessage({ type: 'LOG', payload: `Successfully consolidated ${utxosAtAddress.length} UTXOs into a single coin.` });
            } catch (err) {
                self.postMessage({ type: 'SEND_ERROR', payload: { error: err.message || String(err) } });
            } finally {
                releaseSendLock();
            }
        }
else if (type === 'L2_OPEN_CHANNEL') {
            // Q-Bolt v2: we are the SENDER. Fund the timeout covenant, then
            // create the channel record once funding confirms (via qb_open_intent).
            await qbOpenOutbound(payload.peerPk, payload.amount, payload.lifetime);
        }
        else if (type === 'L2_PAY') {
            const { channelId, amount } = payload;
            const channel = qbChan(channelId);
            if (!channel) throw new Error("Channel not found (or it is a frozen legacy channel).");
            const amt = Number(amount);
            if (!Number.isFinite(amt) || amt <= 0) throw new Error("Enter a positive amount.");
            await qbSenderAdvance(channel, (next) => {
                if (next.sender_amt < amt) throw new Error(`Insufficient channel balance (${next.sender_amt} MDS spendable).`);
                next.sender_amt -= amt;
                next.receiver_amt += amt;
            }, QB.CMD_UPDATE);
            self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
        }
        else if (type === 'L2_SYNC') {
            const channel = qbChan(payload.channelId);
            if (!channel) throw new Error("Channel not found.");
            let sent;
            if (channel.role === 'sender') {
                if (!channel.open_acked) sent = await qbBroadcastOpen(channel);
                else sent = await qbSendMsg(QB.CMD_UPDATE, channel.id, qbPackState(channel.latest, channel.latest.sender_sig));
            } else {
                // Receiver re-ACKs the current state so a sender who missed the
                // ACK stops rebroadcasting.
                sent = await qbSendMsg(QB.CMD_ACK, channel.id, qbPackU32(0, channel.latest.nonce));
            }
            // Report the truth: qbSendMsg returns false when the node rejected it.
            if (sent) qbEvent('info', 'Re-sent to peer. If they stay silent, check that their wallet has synced its L2 identity.', channel.id);
        }
        else if (type === 'L2_CLOSE') {
            const channelId = payload.channelId;
            const channel = qbChan(channelId);
            if (!channel) {
                // Might be a frozen legacy channel — route to the legacy flow.
                const legacy = (wState.l2_channels || {})[channelId];
                if (legacy && legacy.v !== 2) { await qbLegacyClose(channelId); return; }
                throw new Error("Channel not found.");
            }
            if (channel.role === 'receiver') {
                await performChannelClose(channelId, 'close');
            } else {
                // Sender can't settle a balance unilaterally. Ask the receiver to
                // close; if past expiry, take the refund.
                if (networkHeight >= channel.expiry) await performChannelClose(channelId, 'refund');
                else {
                    // Flag it rather than setting status='closing'. The watcher
                    // skips non-active channels entirely, so a status change here
                    // would disable the expiry refund — locking the funds if the
                    // peer never settles. The flag only silences the update
                    // rebroadcast, which is what provokes the peer's rejects.
                    channel.close_requested = { at: networkHeight, nonce: channel.latest.nonce };
                    await saveState();
                    await qbSendMsg(QB.CMD_CLOSE_REQ, channelId, qbPackU32(0, channel.latest.nonce));
                    self.postMessage({ type: 'L2_EVENT', payload: { level: 'info', channelId, msg: `Close requested. If the peer is offline, your funds unlock via refund at block ${channel.expiry}.` } });
                    self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
                }
            }
        }
        else if (type === 'L2_LEGACY_CLOSE') {
            await qbLegacyClose(payload.channelId);
        }
        else if (type === 'L2_REFUND') {
            const channel = qbChan(payload.channelId);
            if (!channel) throw new Error("Channel not found.");
            await performChannelClose(payload.channelId, 'refund');
        }
        else if (type === 'L2_CREATE_INVOICE') {
            const inv = await qbMintInvoice(payload.amount);
            self.postMessage({ type: 'L2_INVOICE_CREATED', payload: inv.text });
        }
        else if (type === 'L2_PAY_INVOICE') {
            await qbPayParsed(qbParseInvoice(payload.invoice));
        }
        else if (type === 'L2_PAY_TO') {
            // Static-address payment: ask the payee's wallet for a signed invoice
            // over the bus, then pay whatever it returns (see qbOnInvoice).
            const destPk = String(payload.destPk || '').replace(/^0x/i, '').substring(0, 64);
            const amount = Number(payload.amount);
            if (!/^[0-9a-fA-F]{64}$/.test(destPk)) throw new Error("Payee identity must be 64 hex characters.");
            if (destPk === getPrimaryMssPk()) throw new Error("You cannot pay yourself.");
            if (!Number.isFinite(amount) || amount <= 0) throw new Error("Enter a positive amount.");
            const reqId = qbHex(crypto.getRandomValues(new Uint8Array(32)));
            wState.l2_inv_reqs = wState.l2_inv_reqs || {};
            wState.l2_inv_reqs[reqId] = { destPk, amount, height: networkHeight || 0 };
            await saveState();
            const extra = new Uint8Array(8);
            new DataView(extra.buffer).setBigUint64(0, BigInt(amount), true);
            await qbSendMsg(QB.CMD_INVOICE_REQ, reqId, qbPackU32(0, 0, extra), [{ kind: "address", value: destPk }]);
            qbEvent('info', `Requested an invoice for ${amount} MDS from ${destPk.substring(0, 12)}…`);
        }
        else if (type === 'WATCH_CONTRACT') {
            // payload: { address } — start tracking a contract's coins.
            const addr = normalizeHex(payload.address);
            if (!watchedContracts.has(addr)) {
                watchedContracts.add(addr);
                updateWasmWatchlist();
                // Rescan so we discover coins that already exist at this address.
                scanResetPending = true;
                scanGeneration++;
                performScan().catch(() => {});
            }
            self.postMessage({ type: 'CONTRACT_WATCHED', payload: { address: addr } });
        }

        else if (type === 'FUND_CONTRACT') {
            await acquireSendLock();
            try { await performContractTx({ kind: 'fund', ...payload }); }
            finally { releaseSendLock(); }
        }

        else if (type === 'SPEND_CONTRACT') {
            await acquireSendLock();
            try { await performContractTx({ kind: 'spend', ...payload }); }
            finally { releaseSendLock(); }
        }
        else if (type === 'SIGN_CHANNEL') {
            await acquireSendLock();
            try {
                self.postMessage({ type: 'CONTRACT_TX_PROGRESS', payload: { reqId: payload.reqId, msg: "Signing L2 Channel State..." } });
                
                const sigHex = await wallet.signChannelState(payload);
                
                // We reuse the CONTRACT_TX_COMPLETE bridge event, stuffing the signature into the `txid` field
                self.postMessage({ type: 'CONTRACT_TX_COMPLETE', payload: { reqId: payload.reqId, txid: sigHex } });
            } catch (err) {
                // Return the error back across the dApp bridge
                self.postMessage({ type: 'ERROR', payload: { reqId: payload.reqId, msg: err.toString() } });
            } finally {
                releaseSendLock();
            }
        }
        else if (type === 'NEW_ADDRESS') {
            self.postMessage({ type: 'LOG', payload: "Deriving new receiving address..." });
            await deriveNextMss(10);
            await saveState();
            self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
            self.postMessage({ type: 'LOG', payload: "New address generated successfully." });
        }

        else if (type === 'REVEAL_SEED') {
            if (wState.phrase) self.postMessage({ type: 'SEED_REVEALED', payload: wState.phrase });
            else self.postMessage({ type: 'ERROR', payload: "Seed phrase not found in memory." });
        }
        else if (type === 'GET_CHAT') {
            try {
                const res = await rpc.getChat();
                if (res.messages) {
                    const normalMessages = [];
                    for (const msg of res.messages) {
                        if (msg.words && msg.words[0] === 255) {
                            handleL2Chat(msg).catch(()=>{});
                        } else {
                            normalMessages.push(msg);
                        }
                    }
                    res.messages = normalMessages;
                }
                self.postMessage({ type: 'CHAT_HISTORY', payload: res });
            } catch (e) {
                self.postMessage({ type: 'ERROR', payload: `Chat sync failed: ${e}` });
            }
        }
        else if (type === 'SEND_CHAT') {
            try {
                const attachments = payload.attachments || [];
                const res = await submitClientMinedChat(payload.words, payload.replyTo, attachments);
                if (res.ok) {
                    self.postMessage({ type: 'CHAT_SENT' });
                } else {
                    self.postMessage({ type: 'ERROR', payload: res.body || "Chat rejected" });
                }
            } catch (e) {
                self.postMessage({ type: 'ERROR', payload: `Send chat failed: ${e}` });
            }
        }
        else if (type === 'GET_TEMPLATE') {
            try {
                const result = await handleGetTemplate();
                self.postMessage({ type: 'TEMPLATE_READY', payload: result });
            } catch (e) {
                self.postMessage({ type: 'TEMPLATE_ERROR', payload: e.toString() });
            }
        }

        else if (type === 'SUBMIT_MINED_BLOCK') {
            try {
                const result = await handleSubmitMinedBlock(payload.template, payload.nonce);
                self.postMessage({ type: 'BLOCK_SUBMITTED', payload: result });
            } catch (e) {
                self.postMessage({ type: 'ERROR', payload: e.toString() });
            }
        }

        else if (type === 'EXPORT_BACKUP') {
            // Build a complete backup with MSS trees included
            const exportState = JSON.parse(JSON.stringify(wState));
            
            // Strip legacy fractional_data
            for (const addr of Object.keys(exportState.mssAddrs)) {
                delete exportState.mssAddrs[addr].fractional_data;
            }
            
            // Pull full trees from IndexedDB and embed as hex
            const mssTreesBackup = {};
            for (const addr of Object.keys(exportState.mssAddrs)) {
                const treeBytes = await idbGet(`mss_${addr}`);
                if (treeBytes) {
                    mssTreesBackup[addr] = normalizeHex(new Uint8Array(treeBytes));
                }
            }
            exportState._mss_trees = mssTreesBackup;
            
            // Encrypt the whole thing
            const salt = crypto.getRandomValues(new Uint8Array(16));
            const iv   = crypto.getRandomValues(new Uint8Array(12));
            const key  = await deriveCryptoKey(password, salt);
            const encrypted = await crypto.subtle.encrypt(
                { name: "AES-GCM", iv }, key,
                new TextEncoder().encode(JSON.stringify(exportState))
            );
            const bundle = {
                salt: normalizeHex(salt),
                iv:   normalizeHex(iv),
                data: normalizeHex(new Uint8Array(encrypted))
            };
            self.postMessage({ type: 'BACKUP_READY', payload: JSON.stringify(bundle) });
        }

        // ═══════════════════════════════════════════════════════════════════════════
        //  IMPORT_CLI — full replacement for the existing handler in worker.js
        //
        //  Supersedes PR #28. That PR fixed the map KEY (pk → derived address) but left
        //  imported wallets unable to sign: the WASM mss_cache stayed empty, leaf
        //  counters never synced (set_mss_leaf_index silently no-ops on a cache miss),
        //  every MSS entry kept index:0 (so loadMssCaches' regeneration fallback would
        //  rebuild key 0 for ALL entries), wotsAddrs stayed empty (historical change
        //  addresses unwatched), and every coin carried index:0 / mss_height:10.
        //
        //  This version:
        //   1. Keys mssAddrs by the derived P2PK address (same fix as the PR) — but
        //      gets the address from get_mss_address(), which ALSO rebuilds the tree
        //      into the WASM signing cache.
        //   2. Self-verifies: asserts the index-i derivation reproduces the CLI's
        //      master_pk, so a derivation-scheme mismatch aborts loudly instead of
        //      importing a wallet that scans but cannot sign.
        //   3. Syncs next_leaf BEFORE exporting, then persists the tree to IndexedDB
        //      under mss_<address>, exactly mirroring deriveNextMss().
        //   4. Re-derives all historically used WOTS addresses + GAP_LIMIT lookahead,
        //      restoring the native invariant (wotsAddrs = indices 0..nextWotsIndex-1)
        //      that performSend and addUtxo's gap extension rely on.
        //   5. Recovers true per-coin metadata (WOTS index via the address map;
        //      per-key MSS height/leaf) instead of hardcoding.
        //   6. Builds everything into locals and commits wState/wallet atomically at
        //      the end — a mid-import failure leaves the previous wallet untouched.
        //   7. Surfaces the real error message (decrypt failure vs derivation
        //      mismatch need different user actions) and kicks a scan on success.
        // ═══════════════════════════════════════════════════════════════════════════

        else if (type === 'IMPORT_CLI') {
            try {
                scanGeneration++;
                while (isScanning) {
                    await new Promise(r => setTimeout(r, 50));
                }
                const cliJsonStr = decrypt_cli_wallet(payload.fileBytes, payload.password);
                const cliData    = JSON.parse(cliJsonStr);
                if (!cliData.master_seed) throw new Error("Legacy (non-HD) wallets not supported in Web.");

                // Build on a LOCAL instance; commit to the `wallet` global only on success.
                const importWallet = WebWallet.from_seed_hex(normalizeHex(cliData.master_seed));

                // ── MSS keys: regenerate by derivation index, verified against the CLI ──
                const mssKeys = cliData.mss_keys || [];
                let newMssAddrs = {};
                for (let i = 0; i < mssKeys.length; i++) {
                    const mss = mssKeys[i];
                    const masterPkHex = normalizeHex(mss.master_pk);
                    self.postMessage({ type: 'MSS_PROGRESS', payload: { current: 0, total: 100, label: `Rebuilding MSS key ${i + 1}/${mssKeys.length}…` } });

                    // Rebuilds the full tree into the WASM signing cache and returns the
                    // canonical address: hex(compute_address(master_pk)) — identical to
                    // what deriveNextMss() stores for native keys.
                    const addr = importWallet.get_mss_address(i, mss.height, (current, total) => {
                        const now = Date.now();
                        if (now - lastMssUpdate > 66 || current === total) {
                            lastMssUpdate = now;
                            self.postMessage({ type: 'MSS_PROGRESS', payload: { current, total, label: `Rebuilding MSS key ${i + 1}/${mssKeys.length} (${current}/${total})…` } });
                        }
                    });

                    // Self-verifying import. VERIFIED against the CLI source (wallet/mod.rs,
                    // wallet/hd.rs): generate_mss → allocate_next_mss_seed is the ONLY writer
                    // to mss_keys (append-only, sequential next_mss_index), and the WASM
                    // imports the identical midstate::wallet::hd::derive_mss_seed. For HD
                    // wallets this check therefore always passes. The one real failure case
                    // is a legacy wallet later upgraded to HD: allocate_next_mss_seed used
                    // rand::random() before the master_seed existed, so early keys are not
                    // reproducible from the seed. Abort loudly — never import a wallet that
                    // scans but cannot sign.
                    if (importWallet.get_mss_pubkey(addr) !== masterPkHex) {
                        throw new Error(
                            `MSS key ${i} was not derived from this wallet's seed (legacy key ` +
                            `from a pre-HD wallet upgrade). Import aborted; your existing web ` +
                            `wallet is untouched. Sweep this key's funds to a fresh address ` +
                            `using the CLI, then re-export and import.`
                        );
                    }

                    // Sync the leaf counter BEFORE export so the persisted blob carries it
                    // ([height:4][seed:32][next_leaf:8][master_pk:32][tree] layout).
                    importWallet.set_mss_leaf_index(addr, mss.next_leaf);
                    await idbPut(`mss_${addr}`, importWallet.export_mss_bytes(addr));

                    newMssAddrs[addr] = { index: i, height: mss.height, next_leaf: mss.next_leaf };
                }

                // ── WOTS addresses: all historically used indices + gap-limit lookahead ──
                // Restores the native invariant wotsAddrs = {0 .. nextWotsIndex-1}, so
                // historical change addresses are watched and coin indices are recoverable.
                const cliNextWots = cliData.next_wots_index || 0;
                const wotsTarget  = cliNextWots + GAP_LIMIT;
                let newWotsAddrs  = {};
                for (let i = 0; i < wotsTarget; i++) {
                    newWotsAddrs[importWallet.get_wots_address(i)] = i;
                    if (i % 25 === 0) {
                        self.postMessage({ type: 'MSS_PROGRESS', payload: { current: i, total: wotsTarget, label: `Deriving WOTS addresses (${i}/${wotsTarget})…` } });
                        await new Promise(r => setTimeout(r, 0));
                    }
                }

                // ── Coins: classify by the rebuilt maps; recover true metadata ──
                // WalletCoin in the CLI also carries an optional `commitment` (confidential /
                // state-thread coins) — pass it through, or spends of such coins break.
                let newUtxos = {};
                for (const coin of (cliData.coins || [])) {
                    const addrHex = normalizeHex(coin.address);
                    const coinId  = normalizeHex(coin.coin_id);
                    const saltHex = normalizeHex(coin.salt);
                    const commit  = coin.commitment ? normalizeHex(coin.commitment) : null;
                    const mssMeta = newMssAddrs[addrHex];

                    if (mssMeta) {
                        newUtxos[coinId] = {
                            index: mssMeta.index, is_mss: true,
                            mss_height: mssMeta.height, mss_leaf: mssMeta.next_leaf,
                            address: addrHex, value: coin.value, salt: saltHex,
                            coin_id: coinId, commitment: commit
                        };
                    } else if (newWotsAddrs[addrHex] !== undefined) {
                        newUtxos[coinId] = {
                            index: newWotsAddrs[addrHex], is_mss: false,
                            mss_height: 0, mss_leaf: 0,
                            address: addrHex, value: coin.value, salt: saltHex,
                            coin_id: coinId, commitment: commit
                        };
                    } else {
                        // Not derivable from this seed. Two known sources (verified in
                        // wallet/mod.rs): import_coin() coins carrying foreign WOTS seeds,
                        // and script-address coins. The CLI export includes the raw seed
                        // for the former, but the web wallet has no sign-with-raw-seed
                        // path — keep the coin visible as watch-only and direct the user
                        // to sweep it with the CLI.
                        self.postMessage({ type: 'LOG', payload: `Warning: coin ${coinId.substring(0, 12)}… is at an address this seed cannot derive; imported as watch-only (sweep it with the CLI to make it spendable here).` });
                        newUtxos[coinId] = {
                            index: 0, is_mss: false, mss_height: 0, mss_leaf: 0,
                            address: addrHex, value: coin.value, salt: saltHex,
                            coin_id: coinId, commitment: commit, watch_only: true 
                        };
                    }
                }

                // ── Atomic commit ──
                if (wallet) wallet.free();
                wallet   = importWallet;
                password = payload.password;
                wState = {
                    phrase: null,
                    master_seed: normalizeHex(cliData.master_seed),
                    nextWotsIndex: wotsTarget,   
                    nextMssIndex:  Math.max(cliData.next_mss_index || 0, mssKeys.length),
                    wotsAddrs: newWotsAddrs,
                    spentWots: {},
                    pendingSpends: {},
                    mssAddrs:  newMssAddrs,
                    utxos:     newUtxos,
                    history:   cliData.history || [],
                    lastScannedHeight: cliData.last_scan_height || 0,
                    l2_channels: {}, l2_secrets: {}, l2_routes: {},
                    l2_invoices: {}, l2_inv_reqs: {}, l2_pay_pending: {}, qb_fwd: {},
                    dex_secrets: {}, dex_cancelled: {}, pendingLimitBundles: {}, annFragPool: {}
                };
                mssCachesReady = true;           // populated above; loadMssCaches would short-circuit anyway
                await saveState();
                self.postMessage({ type: 'WALLET_LOADED', payload: buildDashboardPayload() });

                // Catch up from the CLI's last scanned height (picks up anything received
                // at the — now correctly keyed — MSS address since the CLI last ran).
                performScan().catch(() => {});
            } catch (err) {
                // Decrypt failure and derivation mismatch require different user actions —
                // surface the real reason instead of a generic message.
                self.postMessage({ type: 'ERROR', payload: `Failed to import CLI wallet: ${err && err.message ? err.message : err}` });
            }
        }

        // ─── Self-Test Harness ──────────────────────────────────────────
        //
        // Triggered by: worker.postMessage({ type: 'RUN_TESTS' })
        // Reports results via: { type: 'TEST_RESULTS', payload: { passed, failed, results } }
        //
        // These tests exercise the IndexedDB + WASM integration layer that
        // Rust unit tests cannot cover (they require a browser environment).

        else if (type === 'RUN_TESTS') {
            const results = await runSelfTests();
            self.postMessage({ type: 'TEST_RESULTS', payload: results });
        }

    } catch (err) {
        if (pendingSends.length > 0) {
            pendingSends = [];
            self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
        }
        let errMsg = err.toString();
        if (errMsg.startsWith("Error: ")) errMsg = errMsg.substring(7);
        self.postMessage({ type: 'ERROR', payload: errMsg });
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Key Derivation
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Derive the next WOTS address and add it to the watchlist.
 * Increments `wState.nextWotsIndex`.
 */
function deriveNextWots() {
    const addr = wallet.get_wots_address(wState.nextWotsIndex);
    wState.wotsAddrs[addr] = wState.nextWotsIndex;
    wState.nextWotsIndex++;
}

/** Safe list of spendable UTXOs: excludes any coin already locked by an in-flight send. */
function getSpendableUtxos() {
    wState.pendingSpends = wState.pendingSpends || {};
    const now = Date.now();
    
    // Auto-unlock UTXOs that have been stuck for >30 minutes (1800000 ms)
    for (const [cid, timestamp] of Object.entries(wState.pendingSpends)) {
        if (now - timestamp > 1800000) {
            delete wState.pendingSpends[cid];
        }
    }
    
    // ---FILTER OUT watch_only COINS ---
    return Object.values(wState.utxos).filter(u => !wState.pendingSpends[u.coin_id] && !u.watch_only);
}


/**
 * Crash-safe reservation of key material + lock of selected inputs.
 * Call this immediately after prepare_* returns and BEFORE any network broadcast.
 * Advances nextWotsIndex / MSS leaves, marks the input coin_ids as pending,
 * and persists everything so a crash cannot cause WOTS key reuse.
 */
async function reserveAndLock(ctx, progressMsg) {
    // 1. Advance HD counters for any new change addresses
    while (wState.nextWotsIndex < (ctx.next_wots_index || 0)) deriveNextWots();

    // 2. Advance MSS leaf counters for any MSS inputs that will be signed
    const usedMss = new Set();
    const inputs = ctx.selected_inputs || ctx.wallet_inputs || [];
    for (const inp of inputs) {
        if (inp.is_mss) usedMss.add(inp.address);
    }
    for (const addr of usedMss) {
        if (wState.mssAddrs[addr]) wState.mssAddrs[addr].next_leaf++;
    }

    // 3. Lock the selected input coins so they cannot be selected again
    wState.pendingSpends = wState.pendingSpends || {};
    const now = Date.now();
    for (const inp of inputs) {
        if (inp.coin_id) wState.pendingSpends[inp.coin_id] = now;
    }

    if (progressMsg) self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: progressMsg } });
    await saveState();
}

/** @type {number} Timestamp of last MSS progress UI update (throttle to 15fps). */
let lastMssUpdate = 0;

/**
 * Generate a new MSS address, persist the full tree to IndexedDB,
 * and add the metadata to wallet state.
 *
 * @param {number} height - Merkle tree height (10 = 1024 signatures).
 * @returns {Promise<string>} The hex-encoded MSS address.
 */
async function deriveNextMss(height) {
    const progressCallback = (current, total) => {
        const now = Date.now();
        if (now - lastMssUpdate > 66 || current === total) {
            lastMssUpdate = now;
            self.postMessage({ type: 'MSS_PROGRESS', payload: { current, total, label: `Hashing tree leaves (${current}/${total})...` } });
        }
    };

    // Generate the full MSS tree in WASM — returns the hex address
    const addr = wallet.get_mss_address(wState.nextMssIndex, height, progressCallback);

    // Export the full tree (~64 KB binary) and persist to IndexedDB
    const treeBytes = wallet.export_mss_bytes(addr);
    await idbPut(`mss_${addr}`, treeBytes);

    wState.mssAddrs[addr] = {
        index: wState.nextMssIndex,
        height: height,
        next_leaf: 0
    };
    wState.nextMssIndex++;
    return addr;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Dashboard
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Build the dashboard payload for the UI.
 *
 * Computes the safe balance (total UTXOs minus pending sends), merges
 * pending sends into the history, and returns the primary address.
 *
 * @returns {{primaryAddress: string, balance: number, utxos: Array, history: Array}}
 */
function buildDashboardPayload() {
    const mssList    = Object.keys(wState.mssAddrs);
    const utxoArray  = Object.values(wState.utxos);
    const totalUtxoValue   = utxoArray.reduce((s, u) => s + Number(u.value), 0);
    const pendingDeduction = pendingSends.reduce((s, tx) => s + tx.value + tx.fee, 0);
    const safeBalance      = Math.max(0, totalUtxoValue - pendingDeduction);
    const sortedHistory    = [...pendingSends, ...wState.history].sort((a, b) => b.timestamp - a.timestamp);
    return {
        primaryAddress: mssList.length > 0 ? mssList[mssList.length - 1] : "None",
        // The Q-Bolt identity is the MSS *public key*, NOT the receiving address.
        // `primaryAddress` above is a key of wState.mssAddrs — i.e. an address —
        // while the covenant, every signature check, and the channel id are all
        // derived from get_mss_pubkey(address). Showing the address as the "L2
        // identity" makes a peer build a different covenant, so the channel id
        // never matches and every open is rejected. Keep the two distinct.
        primaryL2Pk: (typeof getPrimaryMssPk === 'function' && getPrimaryMssPk()) || "None",
        balance: safeBalance,
        utxos:   utxoArray,
        history: sortedHistory,
        lastScannedHeight: wState.lastScannedHeight || 0,
        networkHeight: networkHeight || 0,
        mempoolSize: mempoolSize || 0,
        l2LeafBudget: (typeof qbLeafBudget === 'function') ? qbLeafBudget() : null,
        l2Channels: Object.entries(wState.l2_channels || {}).map(([id, c]) => {
            if (c.v !== 2) {
                // Frozen legacy channel — expose just enough for the UI to offer rescue.
                const ls = c.latest_state || {};
                return {
                    id, version: 1, legacy: true, status: c.status || 'frozen-legacy',
                    peer_pk: (getPrimaryMssPk && getPrimaryMssPk() === c.alice_pk) ? c.bob_pk : c.alice_pk,
                    aliceAmt: ls.alice_amt || 0, bobAmt: ls.bob_amt || 0,
                    isAlice: (getPrimaryMssPk && getPrimaryMssPk() === c.alice_pk),
                };
            }
            const htlcLocked = (c.latest.htlcs || []).reduce((s, h) => s + h.amount, 0);
            const blocksLeft = c.expiry - (networkHeight || 0);
            return {
                id, version: 2, legacy: false,
                role: c.role, direction: c.role === 'sender' ? 'outbound' : 'inbound',
                peer_pk: c.peer_pk, status: c.status,
                capacity: c.capacity,
                spendable: c.role === 'sender' ? c.latest.sender_amt : c.latest.receiver_amt,
                receivable: c.role === 'sender' ? c.latest.receiver_amt : c.latest.sender_amt,
                senderAmt: c.latest.sender_amt, receiverAmt: c.latest.receiver_amt,
                htlcCount: (c.latest.htlcs || []).length, htlcLocked,
                nonce: c.latest.nonce, expiry: c.expiry, blocksLeft,
                openAcked: !!c.open_acked, closeStage: c.close ? c.close.stage : null,
                onchainHtlcs: (c.onchain_htlcs || []).filter(h => !h.swept).length,
                warn: blocksLeft <= QB.WARN_MARGIN,
            };
        })
    };
}

async function refreshNetworkStats() {
    if (!wallet) return;
    try {
        const state = await rpc.getState();
        networkHeight = state.height;
        const mempool = await rpc.getMempool();
        mempoolSize = mempool.size || 0;
        self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });

        // Auto-sync if we're behind
        if (networkHeight > wState.lastScannedHeight + 1) {
            performScan().catch(() => {});
        }
    } catch(e) {}
}

setInterval(refreshNetworkStats, 20000);

/**
 * Update the WASM-side watchlist from current wallet state.
 * Includes all known WOTS addresses, MSS addresses, and UTXO coin IDs.
 */
function updateWasmWatchlist() {
    const watchList = [
        ...Object.keys(wState.wotsAddrs),
        ...Object.keys(wState.mssAddrs),
        ...Object.keys(wState.utxos),
        ...watchedContracts,                  // contract addresses
        ...Object.keys(contractCoins),        // known contract coin ids
    ];
    wallet.set_watchlist(JSON.stringify(watchList));
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Chain Scanning
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Public entry point for chain scanning.
 *
 * Guarantees:
 *  • At most one scan iteration runs at a time (mutex on isScanning).
 *  • A scan request that arrives mid-flight is coalesced into a single trailing
 *    pass after the current one finishes.
 *  • RESCAN-style state resets happen atomically just before the scan that
 *    consumes them, so a queued in-flight scan can't run with mixed state.
 *  • Any thrown error inside the scan body is reported and the UI is unstuck;
 *    the wrapper does NOT loop on a thrown error to avoid hot-loop crashes.
 *  • A bumped scanGeneration cancels the in-flight scan; partial state from
 *    the cancelled scan is never committed (lastScannedHeight + saveState are
 *    only written when the generation still matches at the end of the loop).
 */
async function performScan() {
    if (isScanning) {
        // Coalesce: ask the running wrapper to do another pass, then wait for
        // both the current and the queued pass to complete before returning.
        scanRequested = true;
        while (isScanning) {
            await new Promise(r => setTimeout(r, 50));
        }
        return;
    }

    isScanning = true;
    try {
        do {
            // Apply any pending state reset atomically before scanning.
            // (RESCAN sets scanResetPending; doing the wipe here closes the
            // race where the 20s timer could otherwise fire between the
            // wipe and the rescan call.)
            if (scanResetPending) {
                scanResetPending = false;
                wState.lastScannedHeight = 0;
                wState.utxos = {};
                wState.history = [];
                await saveState();
            }

            scanRequested = false;
            const myGen = scanGeneration;
            try {
                await _performScanInner(myGen);
            } catch (err) {
                self.postMessage({
                    type: 'ERROR',
                    payload: `Scan failed: ${err && err.message ? err.message : err}`
                });
                break; // don't hot-loop on a persistent error
            }
        } while (scanRequested);
    } finally {
        isScanning = false;
    }
}

/**
 * Inner scan body. Must be called only from performScan() so the mutex,
 * coalescing, generation, and reset semantics above are honoured.
 *
 * @param {number} myGen - Generation token at the start of this iteration.
 *   If scanGeneration moves past this value, the inner exits without
 *   committing lastScannedHeight or sending SCAN_COMPLETE.
 */
async function _performScanInner(myGen) {
    if (!mssCachesReady) await loadMssCaches();
    if (myGen !== scanGeneration) return;

    self.postMessage({ type: 'LOG', payload: "Fetching chain state..." });
    const state       = await rpc.getState();
    const chainHeight = state.height;
    const heightAdvanced = chainHeight > networkHeight;
    networkHeight     = chainHeight;

    // Everything that waits on a block (commit/reveal confirmation, the Q-Bolt
    // watcher) used to be woken ONLY by the WebRTC NewBlockTip push. When that
    // transport flaps — as it does on a bad link — those pushes are missed and
    // the whole reality layer stops advancing even though polling knows the
    // height moved. Drive them from here too; both paths are idempotent.
    if (heightAdvanced) {
        const resolvers = nextBlockResolvers;
        nextBlockResolvers = [];
        resolvers.forEach(r => { try { r(); } catch (_) {} });
        if (wallet) qbWatchTick(chainHeight).catch(() => {});
    }
    self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });

    if (myGen !== scanGeneration) return;

    if (chainHeight <= wState.lastScannedHeight) {
        for (const [addr, mss] of Object.entries(wState.mssAddrs)) {
            wallet.set_mss_leaf_index(addr, mss.next_leaf);
        }
        self.postMessage({ type: 'SCAN_COMPLETE', payload: buildDashboardPayload() });
        return;
    }

    self.postMessage({
        type: 'LOG',
        payload: `Scanning blocks ${wState.lastScannedHeight} to ${chainHeight}...`
    });

    let currentHeight = wState.lastScannedHeight;
    updateWasmWatchlist();

    while (currentHeight < chainHeight) {
        if (myGen !== scanGeneration) return; // cancelled — drop partial work

        const end        = Math.min(currentHeight + 1000, chainHeight);
        const filterData = await rpc.getFilters(currentHeight, end);
        const numFilters = filterData.filters ? filterData.filters.length : 0;

        await new Promise(r => setTimeout(r, 15));

        for (let i = 0; i < numFilters; i++) {
            if (myGen !== scanGeneration) return;

            const height = filterData.start_height + i;
            if (height % 100 === 0) {
                self.postMessage({ type: 'SCAN_PROGRESS', payload: { height, max: chainHeight } });
            }

            const n         = filterData.element_counts ? filterData.element_counts[i] : 0;
            if (n === 0) continue;
            const blockHash = filterData.block_hashes ? filterData.block_hashes[i] : undefined;

            if (!blockHash) {
                const mutated = await processFullBlock(height);
                if (mutated) updateWasmWatchlist();
                continue;
            }
            if (wallet.check_filter(filterData.filters[i], blockHash, n)) {
                const mutated = await processFullBlock(height);
                if (mutated) updateWasmWatchlist();
            }
        }

        currentHeight += numFilters;
        if (currentHeight < end) {
            while (currentHeight < end) {
                if (myGen !== scanGeneration) return;
                const mutated = await processFullBlock(currentHeight);
                if (mutated) updateWasmWatchlist();
                currentHeight++;
                if (currentHeight % 100 === 0) {
                    self.postMessage({ type: 'SCAN_PROGRESS', payload: { height: currentHeight, max: chainHeight } });
                }
            }
        }

        // Save progress periodically to prevent restarting from scratch on error
        if (myGen === scanGeneration) {
            wState.lastScannedHeight = currentHeight;
            await saveState();
        }
    }

    // Final commit guarded by the generation: a cancelled scan never writes
    // a stale lastScannedHeight or stale UTXO state to disk.
    if (myGen !== scanGeneration) return;

    for (const [addr, mss] of Object.entries(wState.mssAddrs)) {
        wallet.set_mss_leaf_index(addr, mss.next_leaf);
    }

    wState.lastScannedHeight = chainHeight;
    await saveState();
    self.postMessage({ type: 'SCAN_COMPLETE', payload: buildDashboardPayload() });
}

function compute_confidential_coin_id(addrHex, commitmentHex, saltHex) {
    return blake3_hash_hex("434f4e464944454e5449414c" + addrHex + commitmentHex + saltHex);
}

/**
 * Process a single block for wallet-relevant transactions.
 *
 * Checks coinbase outputs and transaction reveals for addresses/salts
 * we own. Updates UTXOs and history accordingly.
 *
 * @param {number} height - Block height to process.
 * @returns {Promise<boolean>} `true` if any wallet-relevant activity was found.
 */
async function processFullBlock(height) {
    const block = await rpc.getBlock(height);
    if (!block) throw new Error(`Network failed to fetch block at height ${height}.`);

    let matchFound = false;
    const ourSalts = new Map();
    for (const [cid, u] of Object.entries(wState.utxos)) ourSalts.set(u.salt, cid);

    let coinbaseReceives = [];
    if (block.coinbase) {
        for (const cb of block.coinbase) {
            const addrHex = normalizeHex(cb.address);
            const saltHex = normalizeHex(cb.salt);
            if (wState.wotsAddrs[addrHex] !== undefined || wState.mssAddrs[addrHex] !== undefined) {
                const coinId = compute_coin_id_hex(addrHex, BigInt(cb.value), saltHex);
                if (addUtxo(addrHex, Number(cb.value), saltHex, coinId)) coinbaseReceives.push({ id: coinId, val: Number(cb.value) });
                matchFound = true;
            }
        }
    }

    if (coinbaseReceives.length > 0) {
        const alreadyRecorded = wState.history.some(h => h.outputs.some(out => coinbaseReceives.map(c=>c.id).includes(out)));
        if (!alreadyRecorded) {
            wState.history.push({
                kind: 'coinbase', timestamp: block.timestamp || Math.floor(Date.now() / 1000),
                fee: 0, inputs: [], outputs: coinbaseReceives.map(c => c.id),
                value: coinbaseReceives.reduce((s, c) => s + c.val, 0)
            });
        }
    }

    if (block.transactions) {
        for (const tx of block.transactions) {
            // FIX: Parse both Reveal AND Consolidate transactions for UTXO tracking
            const action = tx.Reveal || tx.reveal || tx.Consolidate || tx.consolidate;
            if (!action) continue;

            let spentIds = [], spentValue = 0, createdOutputs = [];
            
            // Extract Sender Identity 
            let senderAddrHex = "";
            let txId = "";
            if (action.inputs && action.inputs.length > 0) {
                const bytecode = action.inputs[0].predicate?.Script?.bytecode || action.inputs[0].bytecode;
                if (bytecode) senderAddrHex = blake3_hash_hex(normalizeHex(bytecode));
                const saltHex = normalizeHex(action.inputs[0].salt);
                txId = compute_coin_id_hex(senderAddrHex, BigInt(action.inputs[0].value), saltHex);
            }

            if (action.inputs) {
                for (const inp of action.inputs) {
                    const saltHex = normalizeHex(inp.salt);
                    const bytecode = inp.predicate?.Script?.bytecode || inp.bytecode;
                    
                    let cid = null;
                    if (bytecode) {
                        // Recompute the exact Coin ID from the blockchain data
                        const addrHex = blake3_hash_hex(normalizeHex(bytecode));
                        if (inp.commitment) {
                            cid = compute_confidential_coin_id(addrHex, normalizeHex(inp.commitment), saltHex);
                        } else {
                            cid = compute_coin_id_hex(addrHex, BigInt(inp.value), saltHex);
                        }
                    }

                    // Fallback to the salt map only if the block data is severely malformed
                    if (!cid) cid = ourSalts.get(saltHex);

                    // Safely delete the accurately identified coin
                    if (cid && wState.utxos[cid]) {
                        // CRITICAL FIX: Track spent WOTS addresses and purge siblings to prevent Key Reuse!
                        const spentCoin = wState.utxos[cid];
                        if (!spentCoin.is_mss) {
                            wState.spentWots = wState.spentWots || {};
                            wState.spentWots[spentCoin.address] = true;
                            // Purge any siblings to ensure absolute safety
                            for (const key in wState.utxos) {
                                if (wState.utxos[key].address === spentCoin.address) {
                                    delete wState.utxos[key];
                                    if (wState.pendingSpends) delete wState.pendingSpends[key]; // also free any lock
                                }
                            }
                        }

                        delete wState.utxos[cid];
                        if (wState.pendingSpends) delete wState.pendingSpends[cid]; // free the lock for this coin
                        if (ourSalts.get(saltHex) === cid) ourSalts.delete(saltHex);
                        spentIds.push(cid);
                        spentValue += Number(inp.value);
                        matchFound = true;
                    }

                    // ── Remove spent contract coins (MidstateConnect) ──
                    {
                        const ibc = inp.predicate?.Script?.bytecode || inp.bytecode;
                        if (ibc) {
                            const iAddr = blake3_hash_hex(normalizeHex(ibc));
                            if (watchedContracts.has(iAddr)) {
                                const iSalt = normalizeHex(inp.salt);
                                const iComm = inp.commitment ? normalizeHex(inp.commitment) : null;
                                const id = iComm
                                    ? compute_confidential_coin_id(iAddr, iComm, iSalt)
                                    : compute_coin_id_hex(iAddr, BigInt(inp.value), iSalt);
                                if (contractCoins[id]) { delete contractCoins[id]; matchFound = true; }
                            }
                        }
                    }
                }
            }

            if (action.outputs) {
                for (const out of action.outputs) {
                    const outData = out.Standard || out.standard;
                    if (outData) {
                        const addrHex = normalizeHex(outData.address);
                        const saltHex = normalizeHex(outData.salt);
                        if (wState.wotsAddrs[addrHex] !== undefined || wState.mssAddrs[addrHex] !== undefined) {
                            const coinId = compute_coin_id_hex(addrHex, BigInt(outData.value), saltHex);
                            if (addUtxo(addrHex, Number(outData.value), saltHex, coinId)) {
                                createdOutputs.push({ id: coinId, val: Number(outData.value) });
                                ourSalts.set(saltHex, coinId);
                            }
                            matchFound = true;
                        }
                    }
                    const burnData = out.DataBurn || out.data_burn;
                    if (burnData) {
                        const payloadHex = normalizeHex(burnData.payload);
                    }

                    // ── Contract coins at a watched address (MidstateConnect) ──
                    {
                        const stdC = out.Standard || out.standard;
                        if (stdC) {
                            const cAddr = normalizeHex(stdC.address);
                            if (watchedContracts.has(cAddr)) {
                                const cSalt = normalizeHex(stdC.salt);
                                const cId = compute_coin_id_hex(cAddr, BigInt(stdC.value), cSalt);
                                if (!contractCoins[cId]) {
                                    contractCoins[cId] = { address: cAddr, value: Number(stdC.value), salt: cSalt, state: null, coin_id: cId };
                                }
                                matchFound = true;
                            }
                        }
                        const conf = out.Confidential || out.confidential;
                        if (conf) {
                            const cAddr = normalizeHex(conf.address);
                            if (watchedContracts.has(cAddr)) {
                                const cSalt = normalizeHex(conf.salt);
                                const cState = normalizeHex(conf.commitment);
                                const cId = compute_confidential_coin_id(cAddr, cState, cSalt);
                                contractCoins[cId] = { address: cAddr, value: 0, salt: cSalt, state: cState, coin_id: cId };
                                matchFound = true;
                            }
                        }
                    }
                }
            }

            if (spentIds.length > 0) {
                const alreadyRecorded = wState.history.some(h => (h.kind === 'sent' || h.kind === 'mixed') && h.inputs.some(inp => spentIds.includes(inp)));
                if (!alreadyRecorded) {
                    let totalTxIn = 0, totalTxOut = 0;
                    if (action.inputs)  action.inputs.forEach(i  => totalTxIn  += Number(i.value));
                    if (action.outputs) action.outputs.forEach(o => { let od = o.Standard || o.standard; if (od) totalTxOut += Number(od.value); });
                    let actualFee = totalTxIn - totalTxOut;
                    let netSent   = Math.max(0, spentValue - createdOutputs.reduce((s,c) => s+c.val, 0) - actualFee);
                    wState.history.push({
                        kind: 'sent', timestamp: block.timestamp || Math.floor(Date.now() / 1000),
                        fee: actualFee, inputs: spentIds, outputs: createdOutputs.map(c => c.id), value: netSent
                    });
                }
            } else if (createdOutputs.length > 0) {
                const alreadyRecorded = wState.history.some(h => h.outputs.some(out => createdOutputs.map(c=>c.id).includes(out)));
                if (!alreadyRecorded) {
                    wState.history.push({
                        kind: 'received', timestamp: block.timestamp || Math.floor(Date.now() / 1000),
                        fee: 0, inputs: [], outputs: createdOutputs.map(c => c.id),
                        value: createdOutputs.reduce((s, c) => s + c.val, 0)
                    });
                }
            }
        }
    }
    return matchFound;
}

/**
 * Add a UTXO to the wallet state.
 *
 * Determines whether the UTXO is WOTS or MSS-backed based on address
 * ownership. Extends the gap limit if needed.
 *
 * @param {string} address - Hex-encoded owner address.
 * @param {number} value - Coin value.
 * @param {string} salt - Hex-encoded salt.
 * @param {string} coinId - Hex-encoded coin ID.
 * @returns {boolean} `true` if the UTXO was new (not a duplicate).
 */
function addUtxo(address, value, salt, coinId, commitment = null) {
    // CRITICAL FIX: Do not import dust to an already-spent WOTS address!
    if (wState.spentWots && wState.spentWots[address]) {
        self.postMessage({ type: 'LOG', payload: `⚠️ SECURITY: Ignored dust payment to already-spent WOTS address ${address.substring(0,8)}...`});
        return false;
    }

    let index = 0, is_mss = false, mss_height = 0, mss_leaf = 0;
    if (wState.wotsAddrs[address] !== undefined) {
        index = wState.wotsAddrs[address];
        while (wState.nextWotsIndex <= index + GAP_LIMIT) deriveNextWots();
    } else {
        const mss = wState.mssAddrs[address];
        index = mss.index; is_mss = true; mss_height = mss.height; mss_leaf = mss.next_leaf;
    }
    if (!wState.utxos[coinId]) {
        wState.utxos[coinId] = { index, is_mss, mss_height, mss_leaf, address, value, salt, coin_id: coinId, commitment };
        return true;
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Send
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Execute a full send transaction lifecycle.
 *
 * Steps:
 * 1. Coin selection and transaction building (WASM).
 * 2. Spam-proof PoW mining.
 * 3. Commit submission and confirmation wait.
 * 4. Reveal submission and confirmation wait.
 * 5. State update and persistence.
 *
 * @param {string} toAddress - Recipient hex address.
 * @param {number} amount - Amount to send.
 * @returns {Promise<void>}
 * @throws {Error} On any failure (insufficient funds, network errors, timeouts).
 */
/**
 * MSS safety fast-forward, shared by performSend / performContractTx / the DEX
 * bundle funding. Verifies the local MSS leaf index against the node for every
 * tracked MSS address so a used leaf can never be re-signed.
 *
 * RESILIENCE FIX: each getMssState now gets 3 attempts with backoff before we
 * abort. The old code took ONE un-retried shot per address over WebRTC — whose
 * stream layer throws on a momentary disconnect flag, a stale-connection
 * newStream, or a single 15s read timeout — so any transient hiccup surfaced as
 * the scary "Safety Check Failed / key reuse" abort. It hit the DEX announce
 * hardest: that is the first send fired immediately after the bundle-funding
 * flow has spent minutes hammering the connection (PoW, block polling, then a
 * full rescan), exactly when the channel is most likely mid-renegotiation.
 * Aborting when the state is truly unverifiable is still the right policy;
 * giving up on the first blip was the bug.
 */
async function verifyMssSafetyIndices(onProgress) {
    const ATTEMPTS = 3, BACKOFF_MS = [0, 1500, 4000];
    for (const [addr, mss] of Object.entries(wState.mssAddrs)) {
        let mssState = null, lastErr = null, got = false;
        for (let i = 0; i < ATTEMPTS; i++) {
            if (BACKOFF_MS[i]) await new Promise(r => setTimeout(r, BACKOFF_MS[i]));
            try { mssState = await rpc.getMssState(addr); got = true; break; }
            catch (e) {
                lastErr = e;
                if (onProgress && i < ATTEMPTS - 1) onProgress(`Network hiccup verifying MSS state \u2014 retry ${i + 1}/${ATTEMPTS - 1}\u2026`);
            }
        }
        if (!got) {
            throw new Error("Safety Check Failed. Aborting to prevent key reuse. " +
                `(Could not verify MSS key state after ${ATTEMPTS} attempts: ` +
                `${(lastErr && lastErr.message) || lastErr}. Nothing was broadcast \u2014 safe to retry.)`);
        }
        if (mssState && Number.isFinite(mssState.next_index) && mssState.next_index > mss.next_leaf) {
            // Forward-only reconcile to exactly the node's high-water mark. The
            // old code added +20 as a fudge against counter races, but that
            // silently burned 20 one-time leaves PER send — exhausting a 1024-leaf
            // tree in ~50 sends and forcing constant new-address generation. The
            // node's next_index already accounts for chain + mempool, so matching
            // it exactly is both correct and leaf-frugal. sendRevealWithMssLeafRetry
            // still handles the rare genuine race by re-signing on an explicit
            // "already spent" rejection.
            mss.next_leaf = mssState.next_index;
            self.postMessage({ type: 'LOG', payload: `Reconciled MSS index to the network's high-water mark (${mss.next_leaf}).` });
        }
        wallet.set_mss_leaf_index(addr, mss.next_leaf);
    }
}

async function performSend(toAddress, amount, burnDataHex = null, burnValue = 0, sendAll = false) {
    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Selecting coins and building transaction..." } });
    await new Promise(r => setTimeout(r, 10));

    if (!mssCachesReady) await loadMssCaches();

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Verifying MSS safety indices..." } });
    await verifyMssSafetyIndices((m) => self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: m } }));

    const utxoArray = getSpendableUtxos().map(u => {
        if (u.is_mss && wState.mssAddrs[u.address]) return { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf };
        return u;
    });

    // "Send all": spend the whole balance with no change. The fee is size-based, so instead
    // of guessing it we ask the wallet (prepare_spend returns the exact fee) and back it out.
    // The fee depends on the output count, which depends on (total - fee), so iterate to a
    // fixed point — it settles in 1-2 rounds. prepare_spend has no side effects (it only
    // builds a context), so probing it here is safe.
    if (sendAll) {
        let total = 0n;
        for (const u of utxoArray) total += BigInt(u.value);
        if (total <= 0n) throw new Error("Nothing to send — balance is zero.");
        let amt = total > 3000n ? total - 3000n : total;   // start under a safe fee overestimate
        let fee = 3000n, converged = false;
        for (let i = 0; i < 10; i++) {
            let est;
            try {
                est = JSON.parse(wallet.prepare_spend(JSON.stringify(utxoArray), toAddress, amt, wState.nextWotsIndex, null, null));
            } catch (e) {
                amt = amt > 500n ? amt - 500n : 0n;      // fee didn't fit yet; step down and retry
                if (amt <= 0n) throw new Error("Balance is too small to cover the network fee.");
                continue;
            }
            fee = BigInt(est.fee || 0);
            const target = total - fee;                   // spend everything minus the fee (zero change)
            if (target === amt) { converged = true; break; }
            amt = target > 0n ? target : 1n;
        }
        // If the fee wobbled by an output between rounds, leave a 2-MDS buffer so we can
        // never tip into over-spend; when it converged exactly, send the whole balance.
        if (!converged) amt = total - fee - 2n;
        if (amt <= 0n) throw new Error("Balance is too small to cover the network fee.");
        amount = amt.toString();
        self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: 'Sending your full balance minus fee\u2026' } });
    }

    let spendContextStr;
    try {
        spendContextStr = wallet.prepare_spend(
            JSON.stringify(utxoArray), 
            toAddress, 
            BigInt(amount), 
            wState.nextWotsIndex,
            burnDataHex,
            burnDataHex ? BigInt(burnValue) : null
        );
    } catch (e) {
        throw new Error(`Failed to prepare transaction: ${e.toString()}`);
    }

    const ctx = JSON.parse(spendContextStr);
    self.postMessage({ type: 'SEND_STEP', payload: {
        step: 'built', title: 'Transaction built',
        detail: `${ctx.selected_inputs.length} input(s) → ${(ctx.outputs || []).length} output(s)`,
        sub: `fee ${ctx.fee} MDS`
    }});

    // PRE-FLIGHT: verify every selected input actually exists on-chain BEFORE
    // mining the commit PoW. A phantom UTXO (stale wallet record, missed spend,
    // reorg) produces a tx the node accepts at admission but prunes on every
    // block — an unbreakable eviction loop that costs minutes of PoW and burns
    // a one-time signature leaf per attempt. A handful of checkCoin calls here
    // kills that whole failure class at its cheapest point. Network errors
    // don't block the send; only a definitive "does not exist" does.
    {
        const phantom = [];
        for (const inp of ctx.selected_inputs) {
            try { const c = await rpc.checkCoin(inp.coin_id); if (c && !c.exists) phantom.push(inp.coin_id); } catch (_) {}
        }
        if (phantom.length) {
            for (const id of phantom) { if (wState.utxos && wState.utxos[id]) delete wState.utxos[id]; }
            await saveState();
            performScan().catch(() => {});
            throw new Error(`Aborted before signing: the wallet selected coin(s) that do not exist on-chain (${phantom.map(m => m.substring(0, 12) + '…').join(', ')}). The stale record(s) were removed and a rescan started — please retry the send.`);
        }
    }

    // Intercept L2 Open Intents — capture EVERY output at the channel address.
    // prepare_spend splits value into power-of-2 denominations, so a channel is
    // funded by one coin per set bit of capacity, not a single coin.
    if (pendingChannelOpen && pendingChannelOpen.isQbolt) {
        const fundingOuts = (ctx.outputs || []).filter(o => o.address === pendingChannelOpen.channelAddr && o.type !== 'data_burn');
        if (fundingOuts.length) {
            pendingChannelOpen.fundingCoins = fundingOuts.map(o => ({
                value: Number(o.value), salt: o.salt,
                coin_id: compute_coin_id_hex(o.address, BigInt(o.value), o.salt),
            }));
            // Channel id = lexicographically smallest funding coin id (matches the
            // WASM builder's input ordering and the receiver's independent check).
            pendingChannelOpen.channelId = [...pendingChannelOpen.fundingCoins].map(c => c.coin_id).sort()[0];
        }
    }

    pendingSends.push({ kind: 'pending', timestamp: Math.floor(Date.now() / 1000), fee: ctx.fee, inputs: ctx.selected_inputs.map(i => i.coin_id), outputs: [], value: Number(amount) });
    self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });

    await reserveAndLock(ctx, "Saving wallet state...");

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Fetching network difficulty..." } });
    const stateData   = await rpc.getState();
    const requiredPow = stateData.required_pow || 24;

    // Verbose step log: the commitment is the tx's binding hash; report it plus the PoW
    // parameters so the user can see exactly what's being mined and later match it on-chain.
    self.postMessage({ type: 'SEND_STEP', payload: {
        step: 'pow', title: 'Mining proof-of-work for the commitment',
        detail: `commitment ${ctx.commitment}`, sub: `difficulty ${requiredPow} bits · anchored to block ${stateData.height}`
    }});
    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: `Mining PoW (difficulty ${requiredPow})…` } });
    await new Promise(r => setTimeout(r, 50));
    const _powT0 = Date.now();
    const spamNonce = Number(mine_commitment_pow(ctx.commitment, requiredPow, BigInt(stateData.height), stateData.header_hash));
    self.postMessage({ type: 'SEND_STEP', payload: {
        step: 'pow_done', title: 'Proof-of-work found',
        detail: `nonce ${spamNonce}`, sub: `${((Date.now() - _powT0) / 1000).toFixed(1)}s`
    }});

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Broadcasting commit…" } });
    const commitReq = await rpc.commit(ctx.commitment, spamNonce);
    if (!commitReq.ok) throw new Error(`Commit rejected by node: ${commitReq.body || commitReq.error}`);
    self.postMessage({ type: 'SEND_STEP', payload: {
        step: 'commit_sent', title: 'Commit broadcast to the network',
        detail: `commitment ${ctx.commitment}`, sub: 'waiting for a block to include it…'
    }});

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Waiting for the commit to be mined (phase 1 of 2)…" } });
    const revealPayloadStr = wallet.build_reveal(spendContextStr, ctx.commitment, ctx.tx_salt);

    // Persist the whole in-flight transaction BEFORE we wait on the network.
    // The reveal is already signed here, so a reload/crash/transport flap can
    // re-send it verbatim instead of losing it. Rebuilding is NOT an option once
    // the commitment is mined: it is single-shot on-chain, and re-signing would
    // burn another one-time leaf.
    wState.pending_tx = {
        commitment: ctx.commitment,
        revealPayload: revealPayloadStr,
        inputCoinId: ctx.selected_inputs[0].coin_id,
        inputCoinIds: ctx.selected_inputs.map(i => i.coin_id),   // full set, for dead-input diagnosis
        stage: 'commit',
        createdAt: Date.now(),
        startedHeight: networkHeight,
    };
    await saveState();

    const _commitResp = await awaitCommitmentMined(ctx.commitment, (m) => self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: m } }));
    const _commitHeight = _commitResp.height ?? _commitResp.block_height ?? null;
    if (wState.pending_tx && wState.pending_tx.commitment === ctx.commitment) {
        wState.pending_tx.stage = 'reveal';
        wState.pending_tx.commitHeight = _commitHeight;
        await saveState();
    }
    self.postMessage({ type: 'SEND_STEP', payload: {
        step: 'commit_mined', title: 'Commit confirmed on-chain',
        detail: _commitHeight != null ? `included in block ${_commitHeight}` : 'commit is now on-chain',
        sub: `commitment ${ctx.commitment}`
    }});

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Commit confirmed — broadcasting reveal…" } });
    // Self-heal MSS leaf reuse on the send/announce path too: re-sign against the same
    // already-mined commitment with a fresh leaf (the leaf is a witness, not committed to).
    const sendLeafPhase = (p) => self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: p } });
    const revealReq = await sendRevealWithMssLeafRetry(
        revealPayloadStr, spendContextStr, ctx.commitment, ctx.tx_salt, sendLeafPhase,
        (cs) => wallet.build_reveal(cs, ctx.commitment, ctx.tx_salt)
    );
    if (!revealReq.ok) throw new Error(`Reveal rejected by node: ${revealReq.body || revealReq.error}`);
    self.postMessage({ type: 'SEND_STEP', payload: {
        step: 'reveal_sent', title: 'Reveal broadcast to the network',
        detail: `links to commitment ${ctx.commitment}`, sub: 'waiting for a block to include it…'
    }});

    // Store the payload actually accepted by the node — the MSS-leaf retry may
    // have re-signed it with a fresh leaf, and a rebroadcast must match.
    if (wState.pending_tx && wState.pending_tx.commitment === ctx.commitment) {
        if (revealReq.revealPayload) wState.pending_tx.revealPayload = revealReq.revealPayload;
        await saveState();
    }

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Waiting for the reveal to be mined (phase 2 of 2)…" } });
    const inputCoinToCheck = ctx.selected_inputs[0].coin_id;

    const _revealResp = await awaitRevealMined(
        inputCoinToCheck,
        (wState.pending_tx && wState.pending_tx.revealPayload) || revealPayloadStr,
        ctx.commitment,
        (m) => self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: m } }),
        ctx.selected_inputs.map(i => i.coin_id)
    );
    delete wState.pending_tx;
    await saveState();
    const _revealHeight = _revealResp.spentHeight ?? _revealResp.height ?? null;
    self.postMessage({ type: 'SEND_STEP', payload: {
        step: 'reveal_mined', title: 'Reveal confirmed — transaction complete',
        detail: _revealHeight != null ? `settled in block ${_revealHeight}` : 'transaction is now settled on-chain',
        sub: `commit ${_commitHeight != null ? 'block ' + _commitHeight : ''} → reveal ${_revealHeight != null ? 'block ' + _revealHeight : ''}`.trim()
    }});

    pendingSends = [];
    // Do NOT eagerly delete UTXOs here! Let performScan() discover the spend naturally
    // so it can properly register the history entry.
    // Finalize L2 Open (Q-Bolt v2). The funding tx just confirmed; create the
    // sender-side channel, sign state 0, and broadcast OPEN2. qbFinalizeOutboundOpen
    // is idempotent and also runs from qbBoot if we crash right here.
    if (pendingChannelOpen && pendingChannelOpen.isQbolt && pendingChannelOpen.channelId && pendingChannelOpen.fundingCoins) {
        // Persist the resolved coins into the crash-recovery intent first.
        wState.qb_open_intent = {
            channelAddr: pendingChannelOpen.channelAddr,
            senderPk: pendingChannelOpen.senderPk,
            receiverPk: pendingChannelOpen.receiverPk,
            expiry: pendingChannelOpen.expiry,
            amount: pendingChannelOpen.amount,
            channelId: pendingChannelOpen.channelId,
            fundingCoins: pendingChannelOpen.fundingCoins,
            createdAt: Date.now(),
        };
        await saveState();
        const intent = wState.qb_open_intent;
        pendingChannelOpen = null;
        try { await qbFinalizeOutboundOpen(intent); }
        catch (e) { qbLog(`Channel finalize deferred to watcher: ${e && e.message || e}`); }
    } else {
        pendingChannelOpen = null;
    }
    // Scan locally rather than blindly accepting outputs to prevent mismatches
    await performScan();

    self.postMessage({ type: 'SEND_COMPLETE', payload: buildDashboardPayload() });
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Contract Funding & Execution (MidstateConnect)
// ═══════════════════════════════════════════════════════════════════════════════

async function performContractTx(req) {
    const prog = (msg) => {
        self.postMessage({ type: 'CONTRACT_TX_PROGRESS', payload: { reqId: req.reqId, msg } });
        // If this tx belongs to a DEX swap, also surface the phase on that swap's card.
        if (req.dexOfferId) self.postMessage({ type: 'DEX_PHASE', payload: { offerId: req.dexOfferId, phase: msg } });
    };

    if (!mssCachesReady) await loadMssCaches();

    // MSS safety fast-forward (identical policy to performSend).
    prog("Verifying MSS safety indices...");
    await verifyMssSafetyIndices(prog);

    const utxoArray = getSpendableUtxos().map(u => {
        if (u.is_mss && wState.mssAddrs[u.address]) return { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf };
        return u;
    });

    // ── Build the spend context (phase 1) ───────────────────────────────────
    prog("Building transaction...");
    let ctxStr;
    try {
        if (req.kind === 'fund') {
            ctxStr = wallet.prepare_fund_tx(
                JSON.stringify(utxoArray),
                normalizeHex(req.contractAddress),
                BigInt(req.amount || 0),
                req.state ? normalizeHex(req.state) : null,
                wState.nextWotsIndex
            );
        } else {
            // SPEND. Resolve the contract's on-chain coins from our discovered
            // bucket and assemble the inputs array prepare_script_spend expects.
            const contractAddr = normalizeHex(req.contractAddress || blake3_hash_hex(normalizeHex(req.bytecode)));
            const inputsArg = buildContractInputs(req, contractAddr);
            ctxStr = wallet.prepare_script_spend(
                JSON.stringify(utxoArray),
                normalizeHex(req.bytecode),
                JSON.stringify(inputsArg),
                JSON.stringify(req.outputs || []),
                wState.nextWotsIndex
            );
        }
    } catch (e) {
        throw new Error(`Failed to prepare transaction: ${e.toString()}`);
    }

    const ctx = JSON.parse(ctxStr);

    // Reserve wallet key material exactly like performSend (so a concurrent
    // scan/derive can't reuse a WOTS index or MSS leaf we just committed to).
    pendingSends.push({ kind: 'pending', timestamp: Math.floor(Date.now() / 1000), fee: ctx.fee, inputs: (ctx.wallet_inputs || []).map(i => i.coin_id), outputs: [], value: 0 });
    self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
        await reserveAndLock(ctx, "Saving wallet state...");

    // ── Commit → PoW → wait → reveal → wait ───────
    prog("Fetching network difficulty...");
    const stateData   = await rpc.getState();
    const requiredPow = stateData.required_pow || 24;

    prog(`Mining PoW [Commit: ${ctx.commitment.substring(0,8)}...]`);
    await new Promise(r => setTimeout(r, 50));
    const spamNonce = Number(mine_commitment_pow(ctx.commitment, requiredPow, BigInt(stateData.height), stateData.header_hash));

    prog("Submitting commit...");
    const commitReq = await rpc.commit(ctx.commitment, spamNonce);
    if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);

    prog(`Commit broadcast — waiting to be mined (1/2) [Commit: ${ctx.commitment.substring(0,8)}...]`);
    const revealPayloadStr = wallet.build_script_reveal(ctxStr, ctx.commitment, ctx.tx_salt);

    await awaitCommitmentMined(ctx.commitment, prog);

    prog("Commit mined ✓ — submitting reveal…");
    const revealReq = await rpc.send(revealPayloadStr);
    if (!revealReq.ok) throw new Error(`Reveal rejected: ${revealReq.body || revealReq.error}`);

    prog("Reveal broadcast — waiting to be mined (step 2 of 2)…");
    // Use the first input coin id (contract or wallet) to detect inclusion.
    const firstInputId = ctx.input_coin_ids && ctx.input_coin_ids.length ? ctx.input_coin_ids[0] : null;
    if (firstInputId) {
        await awaitCoinSpent(firstInputId, prog);
    }

    pendingSends = [];
    await performScan();

    // The reveal's commitment doubles as a stable transaction identifier here.
    self.postMessage({ type: 'CONTRACT_TX_COMPLETE', payload: { reqId: req.reqId, txid: ctx.commitment } });

    // For funds, surface the freshly created coins at the contract address. Their
    // salts are random and NOT recoverable from chain, so callers (e.g. the DEX
    // maker) must capture them here to let a counterparty spend the contract later.
    let createdCoins = [];
    if (req.kind === 'fund') {
        const cAddr = normalizeHex(req.contractAddress || (req.bytecode ? blake3_hash_hex(normalizeHex(req.bytecode)) : ""));
        for (const o of (ctx.outputs || [])) {
            if (normalizeHex(o.address) === cAddr) {
                if (o.type === 'confidential') {
                    createdCoins.push({
                        coin_id: compute_confidential_coin_id(normalizeHex(o.address), normalizeHex(o.commitment), normalizeHex(o.salt)),
                        value: 0,
                        salt: normalizeHex(o.salt),
                        state: normalizeHex(o.commitment)
                    });
                } else {
                    createdCoins.push({
                        coin_id: compute_coin_id_hex(normalizeHex(o.address), BigInt(o.value), normalizeHex(o.salt)),
                        value: o.value,
                        salt: normalizeHex(o.salt)
                    });
                }
            }
        }
    }
    return { txid: ctx.commitment, coins: createdCoins };
}

/**
 * Assemble the inputs[] array for prepare_script_spend from a dApp request.
 *
 * Accepts EITHER an explicit `req.inputs` (advanced dApps that already know the
 * coin ids/witnesses) OR the IDE's high-level shape: a value coin + an optional
 * state coin at the contract address, with witnesses supplied per role.
 *
 * IDE-style fields:
 *   req.valueWitness  — witness stack for the value coin (e.g. "addr,01,<preimage>")
 *   req.stateWitness  — witness stack for the state coin (if the contract has one)
 *   (coin ids/salts/states are resolved from the discovered contractCoins bucket)
 */
function buildContractInputs(req, contractAddr) {
    if (Array.isArray(req.inputs) && req.inputs.length) {
        // Advanced path: trust the dApp's explicit inputs, but backfill
        // value/salt/state from our bucket when only a coinId was given.
        return req.inputs.map(i => {
            const known = i.coinId ? contractCoins[normalizeHex(i.coinId)] : null;
            return {
                coin_id: i.coinId ? normalizeHex(i.coinId) : (known ? known.coin_id : ""),
                witness: i.witness || "",
                value:   i.value != null ? Number(i.value) : (known ? known.value : 0),
                salt:    i.salt ? normalizeHex(i.salt) : (known ? known.salt : ""),
                state:   i.inputState ? normalizeHex(i.inputState) : (known ? known.state : null),
            };
        });
    }

    // IDE high-level path: pick coins at the contract address from our bucket.
    // Mirror the IDE's CLI convention EXACTLY (see updateCliInstructions):
    //   - when a state coin exists, the user's witness (#ctxWitness, e.g. "BB,01")
    //     drives the STATE coin (with its input-state), and the VALUE coin takes
    //     the fixed routing witness "02" with no state;
    //   - when there is no state coin, the user's witness drives the value coin.
    const coins = Object.values(contractCoins).filter(c => c.address === contractAddr);
    if (!coins.length) {
        throw new Error("No on-chain coins found for this contract. Fund it first, then Sync.");
    }
    const stateCoin = coins.find(c => c.state);                 // the confidential state thread
    const valueCoin = coins.find(c => !c.state && c.value > 0)  // a fundable value coin
                   || coins.find(c => !c.state);

    const userWitness = req.valueWitness || req.witness || "";  // the IDE's #ctxWitness
    const inputs = [];

    if (stateCoin) {
        // State present: user witness → state coin; value coin → fixed "02".
        inputs.push({
            coin_id: stateCoin.coin_id,
            witness: userWitness,
            value:   0,
            salt:    stateCoin.salt,
            state:   stateCoin.state,
        });
        if (valueCoin) {
            inputs.push({
                coin_id: valueCoin.coin_id,
                witness: req.fixedValueWitness || "02",
                value:   valueCoin.value,
                salt:    valueCoin.salt,
                state:   null,
            });
        }
    } else if (valueCoin) {
        // No state: user witness drives the value coin.
        inputs.push({
            coin_id: valueCoin.coin_id,
            witness: userWitness,
            value:   valueCoin.value,
            salt:    valueCoin.salt,
            state:   null,
        });
    }
    return inputs;
}


// ═══════════════════════════════════════════════════════════════════════════════
//  Q-BOLT v2 — Spilman channel engine (the "reality" layer)
//
//  Direction: every channel has exactly one SENDER (funds it, signs every
//  balance state) and one RECEIVER (validates + stores states, signs nothing
//  until close). The funding covenant is:
//      IF   2-of-2(sender, receiver)                 → close (receiver-driven)
//      ELSE height ≥ expiry + sender signature       → refund (sender-driven)
//
//  Safety invariants enforced here:
//    · The receiver's floor (their balance) only ever rises — enforced on
//      every inbound state. Replaying an old state can only pay them less.
//    · The sender never holds a receiver signature, so the sender cannot
//      broadcast ANY balance state — their only unilateral exit is the
//      time-locked refund.
//    · A commitment is single-shot on-chain and expires COMMITMENT_TTL
//      blocks after being mined. We therefore NEVER commit a state until a
//      close is actually in flight, and every retry that needs a fresh
//      commitment bumps `attempt` (which changes the tx salt).
//    · Every state transition is persisted BEFORE the network action that
//      depends on it, and every in-flight close is resumable from disk.
// ═══════════════════════════════════════════════════════════════════════════════

const QB = {
    VERSION: 2,
    CLOSE_FEE: 2000,                    // mirrors QBOLT_CLOSE_FEE in wasm lib.rs
    MSS_LEAVES: 1024,                   // MSS tree height 10 → 2^10 one-time leaves
    LEAF_RESERVE: 8,                    // always keep leaves for closes/sweeps
    MIN_CAPACITY: 4096,                 // capacity (amount + fee) sanity floor
    DEFAULT_LIFETIME: 4320,             // ~3 days of 60 s blocks
    MIN_LIFETIME: 360,                  // ~6 h
    MAX_LIFETIME: 43200,                // ~30 d
    CLOSE_MARGIN: 60,                   // receiver auto-closes at expiry − 60
    WARN_MARGIN: 240,                   // UI warning at expiry − 240
    PAY_CUTOFF: 90,                     // sender stops creating states at expiry − 90
    MIN_LIFE_AT_ACCEPT: 180,            // receiver rejects channels expiring sooner
    HTLC_MIN_HEADROOM: 60,              // htlc timeout ≥ now + 60 (stall-close + sweep room)
    HTLC_MAX_PAST_EXPIRY: 2880,         // htlc timeout ≤ expiry + 2 days
    HTLC_HOP_DELTA: 30,                 // hub claims upstream ≥30 blocks before its own deadline
                                        // (must cover: force-close + commit PoW + reveal + sweep)
    MAX_HTLCS: 8,
    HOP_FEE: 50,                        // flat fee a hub keeps per forward (pays its 2 MSS leaves)
    INVOICE_TTL: 720,                   // invoices expire ~12 h after minting
    JIT_OPEN: true,                     // hubs auto-open a channel to unknown FINAL destinations
    JIT_MARGIN: 15,                     // extra timeout blocks required before attempting a JIT open
    CLAIM_STALL_BLOCKS: 20,             // preimage sent, no credited state → force close
    OPEN_VERIFY_BLOCKS: 30,             // blocks a pending inbound OPEN waits for funding
    OPEN_REBROADCAST_EVERY: 10,         // sender re-sends OPEN2 until first ACK
    UPDATE_REBROADCAST_EVERY: 5,        // sender re-sends an unacked latest state
    REBROADCAST_MAX: 20,
    RECONCILE_EVERY: 5,                 // blocks between funding-existence sweeps
    EXTERNAL_PROBE_DEPTH: 20,           // nonce look-back when reconciling a foreign close
    // HTLC sweep, zero-balance fallback: reserves tried in order when the wallet
    // has no UTXOs to pay the fee, so the fee comes out of the HTLC coin itself.
    // The wallet targets ~100/KB (10x the consensus floor) and an MSS witness is
    // ~940 bytes, so a single-input sweep needs ~190-510 depending on how many
    // power-of-2 outputs the value decomposes into. Over-reserving is free: the
    // surplus returns as change to our own address.
    SWEEP_FEE_LADDER: [600, 1200, 2400],
    // wire commands (words[0] = 255, words[1] = cmd)
    CMD_OPEN: 110, CMD_UPDATE: 50, CMD_ACK: 51, CMD_HTLC_ADD: 52,
    CMD_HTLC_CLAIM: 53, CMD_CLOSE_REQ: 54, CMD_REJECT: 55,
    CMD_RESIGN_REQ: 56, CMD_RESIGN: 57,
    CMD_LEGACY_CLOSE_REQ: 58, CMD_LEGACY_CLOSE_SIG: 59,
    // Announce "I settled this channel on-chain". Without it the counterparty's
    // only way to notice is the funding-existence sweep — up to RECONCILE_EVERY
    // blocks of showing a live channel that no longer exists, during which it
    // would happily sign a state against spent funding.
    CMD_CLOSED: 60,
    // Routed payments: fast failure back-propagation + invoice request/reply
    // (the "pay anyone by pk" flow — the payee's pk is the static address).
    CMD_HTLC_FAIL: 61, CMD_INVOICE_REQ: 62, CMD_INVOICE: 63,
};
const QB_CMDS = new Set([QB.CMD_OPEN, QB.CMD_UPDATE, QB.CMD_ACK, QB.CMD_HTLC_ADD,
    QB.CMD_HTLC_CLAIM, QB.CMD_CLOSE_REQ, QB.CMD_REJECT, QB.CMD_RESIGN_REQ,
    QB.CMD_RESIGN, QB.CMD_LEGACY_CLOSE_REQ, QB.CMD_LEGACY_CLOSE_SIG,
    QB.CMD_CLOSED, QB.CMD_HTLC_FAIL, QB.CMD_INVOICE_REQ, QB.CMD_INVOICE]);

const QB_REJECT_REASONS = {
    1: "unknown channel", 2: "bad signature", 3: "conservation violated",
    4: "receiver balance decreased", 5: "HTLC rules violated", 6: "stale nonce",
    7: "channel is closing", 8: "rejected",
};

/** Per-channel re-entrancy guard for close/refund/sweep operations. */
const qbInFlight = new Set();

/** Opens already diagnosed and rejected this session.
 *  `getChat` returns the node's ENTIRE chat history on every poll, so without
 *  this an unusable open is re-parsed, re-derived and re-logged every 30 s
 *  forever — burning CPU and drowning the log that has to explain it. */
const qbRejectedOpens = new Set();
function qbMarkRejected(channelId) {
    if (qbRejectedOpens.size > 200) qbRejectedOpens.clear();   // bounded
    qbRejectedOpens.add(channelId);
}

function qbLog(msg) { self.postMessage({ type: 'LOG', payload: `[Q-Bolt] ${msg}` }); }
function qbEvent(level, msg, channelId) {
    self.postMessage({ type: 'L2_EVENT', payload: { level, msg, channelId: channelId || null } });
    qbLog(msg);
}

// ── byte/hex helpers ────────────────────────────────────────────────────────
function qbHex(bytes) { return Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join(''); }
function qbBytes(hex) { return new Uint8Array((hex.match(/.{1,2}/g) || []).map(b => parseInt(b, 16))); }

// ── wire codecs ─────────────────────────────────────────────────────────────
// OPEN2: [ver u8][expiry u64][count u8][{value u64, salt 32}×n][sig0 …]
function qbPackOpen(expiry, fundingCoins, sig0Hex) {
    const sig = qbBytes(sig0Hex);
    const bin = new Uint8Array(10 + fundingCoins.length * 40 + sig.length);
    const v = new DataView(bin.buffer);
    bin[0] = QB.VERSION;
    v.setBigUint64(1, BigInt(expiry), true);
    bin[9] = fundingCoins.length;
    let o = 10;
    for (const c of fundingCoins) {
        v.setBigUint64(o, BigInt(c.value), true);
        bin.set(qbBytes(c.salt), o + 8);
        o += 40;
    }
    bin.set(sig, o);
    return bin;
}
function qbUnpackOpen(hex) {
    const bin = qbBytes(hex);
    if (bin.length < 10 || bin[0] !== QB.VERSION) return null;
    const v = new DataView(bin.buffer);
    const expiry = Number(v.getBigUint64(1, true));
    const n = bin[9];
    if (bin.length < 10 + n * 40) return null;
    const funding = [];
    let o = 10;
    for (let i = 0; i < n; i++) {
        funding.push({ value: Number(v.getBigUint64(o, true)), salt: qbHex(bin.slice(o + 8, o + 40)) });
        o += 40;
    }
    const sig0 = qbHex(bin.slice(o));
    return { expiry, funding, sig0 };
}

// STATE (UPDATE2 / HTLC_ADD2 / RESIGN2 tail):
// [ver u8][nonce u32][sender_amt u64][receiver_amt u64][hcount u8]
// [{amount u64, timeout u64, hash 32}×h][sender_sig …]
function qbPackState(st, sigHex) {
    const sig = qbBytes(sigHex);
    const h = st.htlcs || [];
    const bin = new Uint8Array(22 + h.length * 48 + sig.length);
    const v = new DataView(bin.buffer);
    bin[0] = QB.VERSION;
    v.setUint32(1, st.nonce, true);
    v.setBigUint64(5, BigInt(st.sender_amt), true);
    v.setBigUint64(13, BigInt(st.receiver_amt), true);
    bin[21] = h.length;
    let o = 22;
    for (const x of h) {
        v.setBigUint64(o, BigInt(x.amount), true);
        v.setBigUint64(o + 8, BigInt(x.timeout), true);
        bin.set(qbBytes(x.secret_hash), o + 16);
        o += 48;
    }
    bin.set(sig, o);
    return bin;
}
function qbUnpackState(hex) {
    const bin = qbBytes(hex);
    if (bin.length < 22 || bin[0] !== QB.VERSION) return null;
    const v = new DataView(bin.buffer);
    const nonce = v.getUint32(1, true);
    const sender_amt = Number(v.getBigUint64(5, true));
    const receiver_amt = Number(v.getBigUint64(13, true));
    const n = bin[21];
    if (n > 12 || bin.length < 22 + n * 48) return null;
    const htlcs = [];
    let o = 22;
    for (let i = 0; i < n; i++) {
        htlcs.push({
            amount: Number(v.getBigUint64(o, true)),
            timeout: Number(v.getBigUint64(o + 8, true)),
            secret_hash: qbHex(bin.slice(o + 16, o + 48)),
        });
        o += 48;
    }
    return { nonce, sender_amt, receiver_amt, htlcs, sig: qbHex(bin.slice(o)) };
}

function qbPackU32(tag, n, extra) { // [ver][u32][extra bytes…]
    const ex = extra || new Uint8Array(0);
    const bin = new Uint8Array(5 + ex.length);
    bin[0] = QB.VERSION;
    new DataView(bin.buffer).setUint32(1, n >>> 0, true);
    bin.set(ex, 5);
    return bin;
}
function qbUnpackU32(hex) {
    const bin = qbBytes(hex);
    if (bin.length < 5 || bin[0] !== QB.VERSION) return null;
    return { n: new DataView(bin.buffer).getUint32(1, true), extra: bin.slice(5) };
}

/** Human-readable command names — silent numeric codes are useless in a log. */
const QB_CMD_NAMES = {
    110: 'OPEN', 50: 'UPDATE', 51: 'ACK', 52: 'HTLC_ADD', 53: 'HTLC_CLAIM',
    54: 'CLOSE_REQ', 55: 'REJECT', 56: 'RESIGN_REQ', 57: 'RESIGN',
    58: 'LEGACY_CLOSE_REQ', 59: 'LEGACY_CLOSE_SIG', 60: 'CLOSED',
    61: 'HTLC_FAIL', 62: 'INVOICE_REQ', 63: 'INVOICE',
};

const QB_FAIL_REASONS = {
    1: "unknown payment hash", 2: "amount below the invoice", 3: "amount does not cover the routing fee",
    4: "no route to the destination", 5: "insufficient hub capacity", 6: "timeout too tight to route",
    7: "route failed downstream",
};

/** Post a Q-Bolt message to the chat bus. Returns true only if the node
 *  actually accepted it.
 *
 *  CRITICAL: `rpc.submitChat` RESOLVES with `{ok:false, body:"..."}` when the
 *  node rejects a message — it does NOT throw. A bare try/catch here swallows
 *  every rejection and makes the sender believe it broadcast. That bug made
 *  channel opens fail completely silently, so the failure is now surfaced.
 */
async function qbSendMsg(cmd, channelId, sigBin, extraAtts) {
    const name = QB_CMD_NAMES[cmd] || `cmd ${cmd}`;
    const atts = [
        { kind: "coin_id", value: channelId },
        { kind: "signature", value: normalizeHex(qbHex(sigBin)) },
        ...(extraAtts || []),
    ];
    // The bus caps attachments at 4 and the payload is length-prefixed into the
    // PoW preimage; catch an over-long frame here rather than on the wire.
    if (atts.length > 4) {
        qbEvent('error', `Cannot send ${name}: ${atts.length} attachments (bus allows 4).`, channelId);
        return false;
    }
    try {
        const r = await submitClientMinedChat([255, cmd], null, atts);
        if (r && r.ok === false) {
            qbEvent('error', `Bus rejected ${name}: ${r.body || r.error || 'rejected'} (payload ${sigBin.length}B, ${atts.length} attachments)`, channelId);
            return false;
        }
        qbLog(`sent ${name} for ${String(channelId).substring(0, 12)}… (${sigBin.length}B)`);
        return true;
    } catch (e) {
        qbEvent('error', `Bus send failed (${name}): ${e && e.message || e}`, channelId);
        return false;
    }
}

// ── channel accessors ───────────────────────────────────────────────────────
function qbChan(channelId) {
    const c = (wState.l2_channels || {})[channelId];
    return (c && c.v === 2) ? c : null;
}
function qbFundingJson(channel) {
    return JSON.stringify(channel.funding.map(f => ({ value: f.value, salt: f.salt })));
}
function qbLeafBudget() {
    const pk = getPrimaryMssPk();
    if (!pk) return { used: 0, total: QB.MSS_LEAVES, remaining: 0 };
    const addr = compute_p2pk_address_hex(pk);
    const used = (wState.mssAddrs[addr] && wState.mssAddrs[addr].next_leaf) || 0;
    return { used, total: QB.MSS_LEAVES, remaining: Math.max(0, QB.MSS_LEAVES - used) };
}
function qbSpendable(channel) { return channel.role === 'sender' ? channel.latest.sender_amt : channel.latest.receiver_amt; }

/** Push the state we're about to supersede into a bounded ring buffer.
 *  Purely for reconciliation accounting: the receiver holds the sender's
 *  signature for EVERY state ever sent, so any of them could be the one that
 *  lands on-chain. This lets qbReconcileExternalClose identify which did. */
function qbPushHistory(channel) {
    channel.state_history = channel.state_history || [];
    channel.state_history.push({
        nonce: channel.latest.nonce,
        sender_amt: channel.latest.sender_amt,
        receiver_amt: channel.latest.receiver_amt,
        htlcs: (channel.latest.htlcs || []).map(h => ({ ...h })),
    });
    if (channel.state_history.length > QB.EXTERNAL_PROBE_DEPTH) {
        channel.state_history.splice(0, channel.state_history.length - QB.EXTERNAL_PROBE_DEPTH);
    }
}

/** Build the canonical close state for arbitrary balances on this channel. */
function qbBuildState(channel, senderAmt, receiverAmt, nonce, htlcs, attempt) {
    return JSON.parse(qbolt_build_state(
        channel.id, channel.sender_pk, channel.receiver_pk, BigInt(channel.expiry),
        qbFundingJson(channel), BigInt(senderAmt), BigInt(receiverAmt),
        nonce, JSON.stringify(htlcs || []), attempt >>> 0
    ));
}

// ── inbound state validation (the receiver's gauntlet) ─────────────────────
// Returns { ok:true, commitment } or { ok:false, code, reason }.
function qbValidateInbound(channel, st, height) {
    if (channel.status !== 'active') return { ok: false, code: 7, reason: "channel is not active" };
    if (channel.role !== 'receiver') return { ok: false, code: 8, reason: "not the receiver on this channel" };
    const cur = channel.latest;
    if (st.nonce <= cur.nonce) {
        // Exact re-delivery of the current state is idempotent — re-ACK it.
        if (st.nonce === cur.nonce && st.sender_amt === cur.sender_amt &&
            st.receiver_amt === cur.receiver_amt && st.sig === cur.sender_sig) {
            return { ok: true, idempotent: true };
        }
        return { ok: false, code: 6, reason: `stale nonce ${st.nonce} (have ${cur.nonce})` };
    }
    if ((st.htlcs || []).length > QB.MAX_HTLCS) return { ok: false, code: 5, reason: "too many HTLCs" };

    // The WASM builder enforces exact conservation (sender+receiver+htlcs =
    // capacity − fee) and rebuilds the commitment the sender must have signed.
    let built;
    try { built = qbBuildState(channel, st.sender_amt, st.receiver_amt, st.nonce, st.htlcs, 0); }
    catch (e) { return { ok: false, code: 3, reason: String(e && e.message || e) }; }

    if (!verify_mss_sig_wasm(st.sig, built.commitment, channel.sender_pk)) {
        return { ok: false, code: 2, reason: "sender signature does not verify" };
    }
    // Monotonic floor: my guaranteed balance can never go down.
    if (st.receiver_amt < cur.receiver_amt) {
        return { ok: false, code: 4, reason: `receiver balance decreased ${cur.receiver_amt} → ${st.receiver_amt}` };
    }
    // HTLC delta rules. New HTLCs need sane timeouts; removed HTLCs must be
    // either credited to me (claim) or provably expired (cancel).
    const prevByHash = new Map((cur.htlcs || []).map(h => [h.secret_hash, h]));
    const newByHash = new Map((st.htlcs || []).map(h => [h.secret_hash, h]));
    for (const h of st.htlcs || []) {
        if (!(h.amount > 0)) return { ok: false, code: 5, reason: "zero-value HTLC" };
        if (!prevByHash.has(h.secret_hash)) {
            if (h.timeout < height + QB.HTLC_MIN_HEADROOM)
                return { ok: false, code: 5, reason: `HTLC timeout ${h.timeout} too soon (need ≥ ${height + QB.HTLC_MIN_HEADROOM})` };
            if (h.timeout > channel.expiry + QB.HTLC_MAX_PAST_EXPIRY)
                return { ok: false, code: 5, reason: "HTLC timeout too far past channel expiry" };
        } else {
            const p = prevByHash.get(h.secret_hash);
            if (p.amount !== h.amount || p.timeout !== h.timeout)
                return { ok: false, code: 5, reason: "existing HTLC was mutated" };
        }
    }
    let mustCredit = 0;
    for (const [hash, p] of prevByHash) {
        if (!newByHash.has(hash)) {
            if (height > p.timeout) continue;               // expired → cancel back to sender is legal
            if ((channel.failed_htlcs || {})[hash] !== undefined) continue; // we FAILed it — consented to uncredited removal
            mustCredit += p.amount;                          // otherwise removal must pay me
        }
    }
    if (st.receiver_amt - cur.receiver_amt < mustCredit) {
        return { ok: false, code: 4, reason: "unexpired HTLC removed without crediting the receiver" };
    }
    return { ok: true, commitment: built.commitment };
}

// ── sender-side state advancement ───────────────────────────────────────────
// `mutate(next)` edits { sender_amt, receiver_amt, htlcs } in place.
async function qbSenderAdvance(channel, mutate, wireCmd, extraAtts) {
    if (channel.role !== 'sender') throw new Error("Only the channel sender can create states. Ask the peer to open a channel toward you for the reverse direction.");
    if (channel.status !== 'active') throw new Error(`Channel is ${channel.status}.`);
    if (!channel.open_acked) throw new Error("The peer hasn't acknowledged this channel yet — use Sync, or wait a moment.");
    if (channel.close_requested) throw new Error("A close has already been requested on this channel — no further payments can be made.");
    if (networkHeight >= channel.expiry - QB.PAY_CUTOFF) throw new Error(`Channel is inside its closing window (expires at block ${channel.expiry}). Open a fresh channel.`);
    const budget = qbLeafBudget();
    if (budget.remaining <= QB.LEAF_RESERVE) throw new Error(`MSS key exhausted (${budget.remaining} one-time leaves left; ${QB.LEAF_RESERVE} are reserved for closing). Close this channel and derive a new address.`);

    const next = {
        sender_amt: channel.latest.sender_amt,
        receiver_amt: channel.latest.receiver_amt,
        htlcs: (channel.latest.htlcs || []).map(h => ({ ...h })),
    };
    mutate(next);
    if (next.sender_amt < 0 || next.receiver_amt < 0) throw new Error("Insufficient channel balance");
    if (next.htlcs.length > QB.MAX_HTLCS) throw new Error(`Too many pending HTLCs (max ${QB.MAX_HTLCS}).`);

    const nonce = channel.latest.nonce + 1;
    const built = qbBuildState(channel, next.sender_amt, next.receiver_amt, nonce, next.htlcs, 0);
    const sig = await signMssAndSync(channel.sender_pk, built.commitment);
    // Remember the state we're superseding. The receiver holds our signature for
    // EVERY state we ever sent, so any of them could be the one that lands
    // on-chain — this bounded ring buffer lets qbReconcileExternalClose work out
    // which one actually did. (Accounting only: older states pay us more.)
    qbPushHistory(channel);
    channel.latest = { nonce, sender_amt: next.sender_amt, receiver_amt: next.receiver_amt, htlcs: next.htlcs, sender_sig: sig };
    channel.sig_attempt = 0;
    channel.leaf_spent = (channel.leaf_spent || 0) + 1;
    channel.last_send_height = networkHeight;
    channel.send_tries = 0;
    await saveState();                                       // persist BEFORE broadcast

    await qbSendMsg(wireCmd, channel.id, qbPackState(channel.latest, sig), extraAtts);
    self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
    return channel.latest;
}

// ═══ inbound protocol handlers ══════════════════════════════════════════════

/** Q-Bolt messages already processed this session.
 *
 *  `getChat` returns the node's ENTIRE history on every 30 s poll, so without
 *  this every message is re-handled on every poll. That is not merely wasteful:
 *  a rejected state got re-rejected forever, each REJECT costing a fresh chat
 *  PoW, and the resulting flood filled the node's 100-message chat ring — which
 *  evicts the very ACKs and CLOSED notices the protocol depends on, and made
 *  getChat itself time out. `sender+timestamp+nonce` is a unique identity: the
 *  PoW nonce is bound to the exact message contents. */
const qbSeenMsgs = new Set();
function qbMsgId(msg) {
    return `${msg.sender || ''}:${msg.timestamp || 0}:${msg.nonce || 0}`;
}

async function handleQbChat(msg) {
    // Replay guard FIRST — before any parsing, validation or reply.
    const mid = qbMsgId(msg);
    if (qbSeenMsgs.has(mid)) return;
    if (qbSeenMsgs.size > 5000) qbSeenMsgs.clear();          // bounded; a re-process is harmless
    qbSeenMsgs.add(mid);

    const cmd = msg.words[1];
    const att = (kind) => msg.attachments.find(a => a.kind === kind)?.value;
    const channelId = att("coin_id");
    if (!channelId) return;
    const myPk = getPrimaryMssPk();
    if (!myPk) {
        // During boot the MSS caches haven't loaded yet, so this is expected and
        // transient — the 30 s chat poll redelivers everything once they have.
        // Only warn when the caches ARE ready and there is genuinely no identity.
        if (!mssCachesReady) return;
        if (!qbInFlight.has('warn:no-identity')) {
            qbInFlight.add('warn:no-identity');
            qbEvent('error', 'Q-Bolt traffic is arriving but this wallet has no L2 identity yet. Run Network Sync to initialize it, then ask the peer to Resend.');
        }
        return;
    }

    if (cmd === QB.CMD_OPEN) return qbOnOpen(channelId, att, myPk);
    // Invoice request/reply are not channel-scoped: channelId is an opaque
    // request id minted by the requester.
    if (cmd === QB.CMD_INVOICE_REQ) return qbOnInvoiceReq(channelId, att, myPk);
    if (cmd === QB.CMD_INVOICE) return qbOnInvoice(channelId, att, myPk);

    const channel = qbChan(channelId);
    if (!channel) {
        // Unknown channel: tell the counterparty (bounded — once per channel id per session).
        if ((cmd === QB.CMD_UPDATE || cmd === QB.CMD_HTLC_ADD) && !qbInFlight.has(`rej:${channelId}`)) {
            qbInFlight.add(`rej:${channelId}`);
            await qbSendMsg(QB.CMD_REJECT, channelId, qbPackU32(0, 0, new Uint8Array([1])));
        }
        return;
    }

    if (cmd === QB.CMD_UPDATE || cmd === QB.CMD_HTLC_ADD) return qbOnUpdate(channel, cmd, att, myPk, msg);
    if (cmd === QB.CMD_ACK) return qbOnAck(channel, att);
    if (cmd === QB.CMD_HTLC_CLAIM) return qbOnClaim(channel, att);
    if (cmd === QB.CMD_HTLC_FAIL) return qbOnFail(channel, att);
    if (cmd === QB.CMD_CLOSE_REQ) return qbOnCloseReq(channel);
    if (cmd === QB.CMD_CLOSED) return qbOnClosed(channel, att);
    if (cmd === QB.CMD_REJECT) return qbOnReject(channel, att);
    if (cmd === QB.CMD_RESIGN_REQ) return qbOnResignReq(channel, att);
    if (cmd === QB.CMD_RESIGN) return qbOnResign(channel, att);
    if (cmd === QB.CMD_LEGACY_CLOSE_REQ) return qbOnLegacyCloseReq(channelId, att, myPk);
    if (cmd === QB.CMD_LEGACY_CLOSE_SIG) return qbOnLegacyCloseSig(channelId, att);
}

async function qbOnOpen(channelId, att, myPk) {
    const senderPkRaw = att("address");
    const payload = att("signature");
    if (!senderPkRaw || !payload) {
        qbLog(`ignored an OPEN with missing attachments (address=${!!senderPkRaw}, payload=${!!payload})`);
        return;
    }
    // Address attachments come back from the node with an 8-char checksum.
    const senderPk = senderPkRaw.substring(0, 64);
    if (senderPk === myPk) return;                           // our own broadcast echo
    if (qbChan(channelId)) {                                  // duplicate OPEN → re-ACK for the sender's sake
        const c = qbChan(channelId);
        if (c.role === 'receiver') await qbSendMsg(QB.CMD_ACK, channelId, qbPackU32(0, 0));
        return;
    }

    if (qbRejectedOpens.has(channelId)) return;               // already diagnosed this session

    const open = qbUnpackOpen(payload);
    if (!open || !open.funding.length) {
        qbEvent('warn', `Received an unreadable channel open from ${senderPk.substring(0, 12)}… (${payload.length / 2}B payload). Version mismatch — both wallets must run the same build.`, channelId);
        return;
    }
    qbLog(`inbound OPEN from ${senderPk.substring(0, 12)}…: ${open.funding.length} funding coin(s), expiry ${open.expiry}`);

    wState.qb_pending_opens = wState.qb_pending_opens || {};
    wState.qb_pending_opens[channelId] = {
        senderPk, expiry: open.expiry, funding: open.funding, sig0: open.sig0,
        first_seen: networkHeight, tries: 0,
    };
    await saveState();
    await qbTryFinalizeInboundOpen(channelId);
}

/** Verify a pending inbound OPEN against the chain; called on receipt and
 *  re-tried by the watcher until funding confirms or the window lapses. */
async function qbTryFinalizeInboundOpen(channelId) {
    const p = (wState.qb_pending_opens || {})[channelId];
    if (!p) return;
    const myPk = getPrimaryMssPk();
    const short = String(channelId).substring(0, 12);
    // Every `return` below used to be silent, which made a dropped open
    // indistinguishable from one that never arrived. They all speak now.
    const drop = async (why) => {
        delete wState.qb_pending_opens[channelId];
        qbMarkRejected(channelId);                            // don't re-diagnose on every chat poll
        await saveState();
        qbEvent('warn', `Discarded inbound channel ${short}…: ${why}`, channelId);
    };
    if (!myPk) {
        qbEvent('error', 'Received a channel open but this wallet has no L2 identity yet — run Network Sync, then ask the peer to Resend.', channelId);
        return;                                              // keep it pending; retry once we have an identity
    }
    p.tries++;

    let addr, ids;
    try {
        addr = qbolt_channel_address(p.senderPk, myPk, BigInt(p.expiry));
        ids = p.funding.map(f => compute_coin_id_hex(addr, BigInt(f.value), f.salt));
    } catch (e) { return drop(`could not derive the covenant address (${e && e.message || e})`); }

    // Never trust the wire: the channel id must be the smallest funding coin id.
    const sortedIds = [...ids].sort();
    if (sortedIds[0] !== channelId) {
        return drop(`channel id mismatch — the peer must open toward this wallet's exact L2 identity `
            + `(${myPk.substring(0, 12)}…). Derived ${sortedIds[0].substring(0, 12)}…, message claims ${short}…`);
    }

    const capacity = p.funding.reduce((a, f) => a + f.value, 0);
    if (capacity < QB.MIN_CAPACITY) return drop(`capacity ${capacity} is below the ${QB.MIN_CAPACITY} minimum`);
    if (p.expiry <= networkHeight + QB.MIN_LIFE_AT_ACCEPT) {
        return drop(`expires at block ${p.expiry}, which is less than ${QB.MIN_LIFE_AT_ACCEPT} blocks away (now ${networkHeight})`);
    }

    // Every claimed funding coin must actually exist on-chain.
    for (const id of ids) {
        let chk = null;
        try { chk = await rpc.checkCoin(id); }
        catch (e) { qbLog(`open ${short}…: checkCoin failed, retrying next block (${e && e.message || e})`); return; }
        if (!chk || !chk.exists) {
            if (networkHeight - p.first_seen > QB.OPEN_VERIFY_BLOCKS) {
                return drop(`funding coin ${id.substring(0, 12)}… never confirmed within ${QB.OPEN_VERIFY_BLOCKS} blocks`);
            }
            qbLog(`open ${short}…: waiting for funding to confirm (${networkHeight - p.first_seen}/${QB.OPEN_VERIFY_BLOCKS} blocks)`);
            return;                                          // not confirmed yet → retry on next block
        }
    }

    // Verify the sender's signature over state 0 (all funds on their side).
    const fundingJson = JSON.stringify(p.funding);
    let built0;
    try {
        built0 = JSON.parse(qbolt_build_state(channelId, p.senderPk, myPk, BigInt(p.expiry),
            fundingJson, BigInt(capacity - QB.CLOSE_FEE), 0n, 0, "[]", 0));
    } catch (e) { return drop(`could not rebuild the opening state (${e && e.message || e})`); }
    if (!p.sig0) return drop('the open carried no opening signature');
    if (!verify_mss_sig_wasm(p.sig0, built0.commitment, p.senderPk)) {
        return drop(`the opening signature does not verify against sender ${p.senderPk.substring(0, 12)}…`);
    }

    wState.l2_channels = wState.l2_channels || {};
    wState.l2_channels[channelId] = {
        v: 2, id: channelId, role: 'receiver',
        sender_pk: p.senderPk, receiver_pk: myPk, peer_pk: p.senderPk,
        expiry: p.expiry, funding: p.funding.map((f, i) => ({ ...f, coin_id: ids[i] })),
        capacity, channel_addr: addr, status: 'active',
        latest: { nonce: 0, sender_amt: capacity - QB.CLOSE_FEE, receiver_amt: 0, htlcs: [], sender_sig: p.sig0 },
        sig_attempt: 0, acked_nonce: 0, open_acked: true, created_height: networkHeight,
        leaf_spent: 0, close: null, closed: null, pending_claims: {}, onchain_htlcs: [],
    };
    delete wState.qb_pending_opens[channelId];
    if (typeof watchedContracts !== 'undefined') { watchedContracts.add(addr); updateWasmWatchlist(); }
    await saveState();
    await qbSendMsg(QB.CMD_ACK, channelId, qbPackU32(0, 0));
    qbEvent('info', `Inbound channel open: ${capacity - QB.CLOSE_FEE} MDS capacity, expires at block ${p.expiry}.`, channelId);
    self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
}

async function qbOnUpdate(channel, cmd, att, myPk, msg) {
    const payload = att("signature");
    if (!payload) return;
    const st = qbUnpackState(payload);
    if (!st) return;
    if (channel.role === 'sender') return;                    // our own echo / misdirected

    const verdict = qbValidateInbound(channel, st, networkHeight);
    if (!verdict.ok) {
        qbLog(`Rejected state ${st.nonce} on ${channel.id.substring(0, 12)}…: ${verdict.reason}`);
        // A REJECT costs a chat PoW and a slot in the node's 100-message history
        // ring. Once we're closing/closed the peer's states are moot and there is
        // nothing they can usefully do about it — replying to each one just
        // floods the bus and evicts messages that matter. Tell them once.
        const terminal = channel.status !== 'active';
        const key = `rejsent:${channel.id}`;
        if (terminal) {
            if (qbInFlight.has(key)) return;
            qbInFlight.add(key);
        }
        await qbSendMsg(QB.CMD_REJECT, channel.id, qbPackU32(0, st.nonce, new Uint8Array([verdict.code])));
        return;
    }
    if (!verdict.idempotent) {
        const prevHtlcs = channel.latest.htlcs || [];
        qbPushHistory(channel);   // lets us identify a close made from another device
        channel.latest = { nonce: st.nonce, sender_amt: st.sender_amt, receiver_amt: st.receiver_amt, htlcs: st.htlcs, sender_sig: st.sig };
        channel.sig_attempt = 0;
        await saveState();                                    // persist BEFORE the ACK leaves

        // Anything that was pending a claim-credit and is now credited: clear the stall timer.
        for (const hash of Object.keys(channel.pending_claims || {})) {
            const still = (st.htlcs || []).some(h => h.secret_hash === hash);
            if (!still) delete channel.pending_claims[hash];
        }
        // FAIL consents are single-use: once the hash left the state, forget it.
        for (const hash of Object.keys(channel.failed_htlcs || {})) {
            if (!(st.htlcs || []).some(h => h.secret_hash === hash)) delete channel.failed_htlcs[hash];
        }
        await saveState();
        self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });

        if (cmd === QB.CMD_HTLC_ADD) {
            const added = (st.htlcs || []).filter(h => !prevHtlcs.some(p => p.secret_hash === h.secret_hash));
            for (const h of added) await qbOnHtlcAdded(channel, h, msg, myPk);
        }
    }
    await qbSendMsg(QB.CMD_ACK, channel.id, qbPackU32(0, st.nonce));
}

/** A new HTLC landed on a channel where we are the receiver. Either we are
 *  the destination (claim it) or we are a hub (forward toward the next hop).
 *
 *  The route is the ordered `address` attachments: the remaining path, ending
 *  at the final destination. Empty (or just us) means we are the destination.
 *  Attachment budget (4, minus coin_id + signature) caps routes at 2 hops:
 *  payer → hubA → hubB → payee. */
async function qbOnHtlcAdded(channel, htlc, msg, myPk) {
    const route = (msg.attachments || []).filter(a => a.kind === 'address')
        .map(a => String(a.value).substring(0, 64));
    const iAmDest = route.length === 0 || (route.length === 1 && route[0] === myPk);

    if (iAmDest) {
        const secret = (wState.l2_secrets || {})[htlc.secret_hash];
        if (!secret) return;                                  // not ours / another device holds it — never FAIL here
        const inv = (wState.l2_invoices || {})[htlc.secret_hash];
        if (inv && htlc.amount < inv.amount) {
            // NEVER reveal the preimage below the invoice amount: a hub could
            // underpay us and use the preimage to collect the full upstream HTLC.
            await qbFail(channel, htlc.secret_hash, 2);
            return;
        }
        channel.pending_claims = channel.pending_claims || {};
        channel.pending_claims[htlc.secret_hash] = { sent_height: networkHeight, amount: htlc.amount };
        await saveState();
        await qbSendMsg(QB.CMD_HTLC_CLAIM, channel.id,
            qbPackU32(0, channel.latest.nonce, qbBytes(htlc.secret_hash)),
            [{ kind: "midstate", value: secret }]);
        wState.l2_claimed = wState.l2_claimed || {};
        wState.l2_claimed[htlc.secret_hash] = htlc.amount;
        await saveState();
        self.postMessage({ type: 'L2_HTLC_CLAIMED', payload: { secretHash: htlc.secret_hash, amount: htlc.amount } });
        return;
    }

    // HUB: keep QB.HOP_FEE, forward the rest toward the next hop.
    const skipSelf = route[0] === myPk && route.length > 1;   // tolerate a route that names us first
    const nextPk = skipSelf ? route[1] : route[0];
    const remaining = route.slice(skipSelf ? 2 : 1);
    const outAmt = htlc.amount - QB.HOP_FEE;
    const downstreamTimeout = htlc.timeout - QB.HTLC_HOP_DELTA;
    if (outAmt <= 0) return qbFail(channel, htlc.secret_hash, 3);
    if (downstreamTimeout < networkHeight + QB.HTLC_MIN_HEADROOM) return qbFail(channel, htlc.secret_hash, 6);

    let fwd = null;
    for (const c of Object.values(wState.l2_channels || {})) {
        if (c.v !== 2 || c.id === channel.id) continue;
        if (c.role !== 'sender' || c.status !== 'active' || !c.open_acked) continue;
        if (c.peer_pk !== nextPk) continue;
        if (c.latest.sender_amt < outAmt) continue;
        if (networkHeight >= c.expiry - QB.PAY_CUTOFF) continue;
        if (downstreamTimeout > c.expiry + QB.HTLC_MAX_PAST_EXPIRY) continue;
        fwd = c; break;
    }

    if (!fwd) {
        // JIT open: no channel to a FINAL destination → fund one on-chain now,
        // park the forward, and let the watcher deliver once the peer ACKs.
        const existing = Object.values(wState.l2_channels || {}).some(c => c.v === 2
            && c.role === 'sender' && c.peer_pk === nextPk && c.status === 'active');
        const canJit = QB.JIT_OPEN && !existing && remaining.length === 0
            && !(wState.qb_fwd || {})[htlc.secret_hash]
            && !wState.qb_open_intent && !pendingChannelOpen
            && htlc.timeout - networkHeight >= QB.HTLC_MIN_HEADROOM + QB.HTLC_HOP_DELTA + QB.JIT_MARGIN;
        if (!canJit) return qbFail(channel, htlc.secret_hash, existing ? 5 : 4);

        wState.qb_fwd = wState.qb_fwd || {};
        wState.qb_fwd[htlc.secret_hash] = {
            nextPk, amount: outAmt, timeout: downstreamTimeout,
            upId: channel.id, inAmount: htlc.amount, created: networkHeight,
        };
        await saveState();
        qbEvent('info', `No channel to ${nextPk.substring(0, 12)}… — opening one just-in-time to route ${outAmt} MDS.`, channel.id);
        qbOpenOutbound(nextPk, Math.max(QB.MIN_CAPACITY - QB.CLOSE_FEE, outAmt), QB.DEFAULT_LIFETIME)
            .catch(async (e) => {
                qbLog(`JIT open failed: ${e && e.message || e}`);
                delete wState.qb_fwd[htlc.secret_hash];
                await saveState();
                await qbFail(channel, htlc.secret_hash, 5);
            });
        return;
    }

    try {
        await qbSenderAdvance(fwd, (next) => {
            next.sender_amt -= outAmt;
            next.htlcs.push({ amount: outAmt, timeout: downstreamTimeout, secret_hash: htlc.secret_hash });
        }, QB.CMD_HTLC_ADD, remaining.map(pk => ({ kind: "address", value: pk })));
        wState.l2_routes = wState.l2_routes || {};
        wState.l2_routes[htlc.secret_hash] = { fromCoinId: channel.id, amount: htlc.amount };
        await saveState();
    } catch (e) {
        qbLog(`HTLC forward failed: ${e && e.message || e}`);
        await qbFail(channel, htlc.secret_hash, 5);
    }
}

async function qbOnAck(channel, att) {
    if (channel.role !== 'sender') return;
    const p = qbUnpackU32(att("signature") || "");
    if (!p) return;
    channel.open_acked = true;
    channel.acked_nonce = Math.max(channel.acked_nonce || 0, p.n);
    channel.send_tries = 0;
    await saveState();
    self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
}

async function qbOnReject(channel, att) {
    if (channel.role !== 'sender') return;
    const p = qbUnpackU32(att("signature") || "");
    if (!p) return;
    const code = p.extra && p.extra.length ? p.extra[0] : 8;
    const prev = channel.last_reject;
    const isNew = !prev || prev.nonce !== p.n || prev.code !== code;
    channel.last_reject = { nonce: p.n, code, reason: QB_REJECT_REASONS[code] || "rejected", at: networkHeight };
    if (!isNew) return;                                       // duplicate — don't re-persist or re-toast
    await saveState();
    qbEvent('warn', `Peer rejected state ${p.n}: ${channel.last_reject.reason}.`, channel.id);
}

/** Receiver revealed a preimage → sender credits it in the next state.
 *  If we're a hub, the same preimage unlocks our upstream claim. */
async function qbOnClaim(channel, att) {
    const p = qbUnpackU32(att("signature") || "");
    const secretRaw = att("midstate");
    if (!p || !secretRaw || p.extra.length < 32) return;
    const secret = secretRaw.substring(0, 64);
    const hash = qbHex(p.extra.slice(0, 32));
    if (blake3_hash_hex(secret) !== hash) return;

    // Publish/persist the observed preimage (submarine-swap makers watch this).
    wState.l2_observed_secrets = wState.l2_observed_secrets || {};
    if (!wState.l2_observed_secrets[hash]) {
        wState.l2_observed_secrets[hash] = secret;
        wState.l2_secrets = wState.l2_secrets || {};
        if (!wState.l2_secrets[hash]) wState.l2_secrets[hash] = secret;   // lets a hub claim upstream on-chain if forced
        await saveState();
        self.postMessage({ type: 'DEX_SUBMARINE_SECRET', payload: { secretHash: hash, secret } });
    }

    if (channel.role === 'sender') {
        const h = (channel.latest.htlcs || []).find(x => x.secret_hash === hash);
        if (h) {
            try {
                await qbSenderAdvance(channel, (next) => {
                    next.htlcs = next.htlcs.filter(x => x.secret_hash !== hash);
                    next.receiver_amt += h.amount;
                }, QB.CMD_UPDATE);
                const pay = (wState.l2_pay_pending || {})[hash];
                if (pay) {
                    delete wState.l2_pay_pending[hash];
                    await saveState();
                    qbEvent('info', `Payment of ${pay.amount} MDS to ${pay.destPk.substring(0, 12)}… completed — preimage ${secret.substring(0, 12)}… is your receipt.`, channel.id);
                }
            } catch (e) { qbLog(`Claim credit failed: ${e && e.message || e}`); }
        }
    }
    // Hub: pull the preimage upstream (we are the RECEIVER on the upstream channel).
    const route = (wState.l2_routes || {})[hash];
    if (route && route.fromCoinId !== channel.id) {
        const up = qbChan(route.fromCoinId);
        if (up && up.role === 'receiver' && up.status === 'active') {
            up.pending_claims = up.pending_claims || {};
            up.pending_claims[hash] = { sent_height: networkHeight, amount: route.amount };
            await saveState();
            await qbSendMsg(QB.CMD_HTLC_CLAIM, up.id, qbPackU32(0, up.latest.nonce, qbBytes(hash)),
                [{ kind: "midstate", value: secret }]);
        }
        delete wState.l2_routes[hash];
        await saveState();
    }
}

async function qbOnCloseReq(channel) {
    if (channel.role !== 'receiver' || channel.status !== 'active') return;
    qbEvent('info', 'Peer requested a close — settling the latest state on-chain.', channel.id);
    performChannelClose(channel.id, 'close').catch(e => qbEvent('error', `Close failed: ${e && e.message || e}`, channel.id));
}

// ═══ routed payments: failure propagation, invoices, pay-by-identity ═════════

/** Receiver-side: refuse an inbound HTLC. Marks consent so the sender's
 *  uncredited removal of this exact hash passes qbValidateInbound. */
async function qbFail(channel, hash, code) {
    channel.failed_htlcs = channel.failed_htlcs || {};
    channel.failed_htlcs[hash] = networkHeight;
    await saveState();
    const extra = new Uint8Array(33);
    extra.set(qbBytes(hash), 0); extra[32] = code;
    await qbSendMsg(QB.CMD_HTLC_FAIL, channel.id, qbPackU32(0, 0, extra));
    qbLog(`HTLC ${hash.substring(0, 12)}… failed: ${QB_FAIL_REASONS[code] || code}`);
}

/** An HTLC we sent was refused. Cancel it immediately (the peer consented to
 *  the uncredited removal), then propagate the failure upstream if we are a
 *  hub, or surface it if we initiated the payment. */
async function qbOnFail(channel, att) {
    const p = qbUnpackU32(att("signature") || "");
    if (!p || p.extra.length < 33) return;
    const hash = qbHex(p.extra.slice(0, 32));
    const code = p.extra[32];

    if (channel.role === 'sender') {
        const h = (channel.latest.htlcs || []).find(x => x.secret_hash === hash);
        if (h && !qbInFlight.has(`fail:${channel.id}:${hash}`)) {
            qbInFlight.add(`fail:${channel.id}:${hash}`);
            try {
                await qbSenderAdvance(channel, (next) => {
                    next.htlcs = next.htlcs.filter(x => x.secret_hash !== hash);
                    next.sender_amt += h.amount;
                }, QB.CMD_UPDATE);
            } catch (e) { qbLog(`FAIL cancel deferred: ${e && e.message || e}`); }
            finally { qbInFlight.delete(`fail:${channel.id}:${hash}`); }
        }
    }
    if ((wState.qb_fwd || {})[hash]) { delete wState.qb_fwd[hash]; await saveState(); }

    // Hub: pass the failure up to whoever sent it to us.
    const route = (wState.l2_routes || {})[hash];
    if (route && route.fromCoinId !== channel.id) {
        const up = qbChan(route.fromCoinId);
        delete wState.l2_routes[hash];
        await saveState();
        if (up && up.role === 'receiver' && up.status === 'active') await qbFail(up, hash, 7);
    }

    const pay = (wState.l2_pay_pending || {})[hash];
    if (pay) {
        delete wState.l2_pay_pending[hash];
        await saveState();
        qbEvent('warn', `Payment of ${pay.amount} MDS failed (${QB_FAIL_REASONS[code] || 'refused'}) — the balance was returned.`, channel.id);
    }
}

// INVOICE reply wire: [ver u8][amount u64][expiry u64][hash 32][hcount u8][hints 32×n][mss_sig …]
function qbPackInvoice(hashHex, amount, expiry, hints, sigHex) {
    const sig = qbBytes(sigHex);
    const bin = new Uint8Array(50 + hints.length * 32 + sig.length);
    const v = new DataView(bin.buffer);
    bin[0] = QB.VERSION;
    v.setBigUint64(1, BigInt(amount), true);
    v.setBigUint64(9, BigInt(expiry), true);
    bin.set(qbBytes(hashHex), 17);
    bin[49] = hints.length;
    let o = 50;
    for (const h of hints) { bin.set(qbBytes(h), o); o += 32; }
    bin.set(sig, o);
    return bin;
}
function qbUnpackInvoice(hex) {
    const bin = qbBytes(hex);
    if (bin.length < 50 || bin[0] !== QB.VERSION) return null;
    const v = new DataView(bin.buffer);
    const amount = Number(v.getBigUint64(1, true));
    const expiry = Number(v.getBigUint64(9, true));
    const hash = qbHex(bin.slice(17, 49));
    const n = bin[49];
    if (n > 2 || bin.length < 50 + n * 32) return null;
    const hints = [];
    let o = 50;
    for (let i = 0; i < n; i++) { hints.push(qbHex(bin.slice(o, o + 32))); o += 32; }
    return { amount, expiry, hash, hints, sig: qbHex(bin.slice(o)) };
}

/** What the payee's MSS signature over a bus-delivered invoice binds:
 *  its own pk + the hash, amount, expiry and hints. The payer verifies this
 *  against the pk it addressed, so nobody on the public bus can race a fake
 *  invoice (own hash, own hints) at an open request. */
function qbInvoiceCommit(payeePk, hashHex, amount, expiry, hints) {
    const head = new Uint8Array(87 + hints.length * 32);
    head.set([0x71, 0x62, 0x69, 0x6e, 0x76, 0x31], 0);        // "qbinv1"
    head.set(qbBytes(payeePk), 6);
    head.set(qbBytes(hashHex), 38);
    const v = new DataView(head.buffer);
    v.setBigUint64(70, BigInt(amount), true);
    v.setBigUint64(78, BigInt(expiry), true);
    head[86] = hints.length;
    let o = 87;
    for (const h of hints) { head.set(qbBytes(h), o); o += 32; }
    return blake3_hash_hex(qbHex(head));
}

/** Mint an invoice: fresh secret, expected amount recorded (underpay guard),
 *  route hints = hubs holding an open sender-side channel toward us with
 *  enough balance and life to route it, best-funded first. */
async function qbMintInvoice(amount, opts) {
    const amt = Number(amount);
    if (!Number.isFinite(amt) || amt <= 0) throw new Error("Invoice amount must be positive.");
    const myPk = getPrimaryMssPk();
    if (!myPk) throw new Error("Network Sync required first to initialize your MSS L2 identity.");
    const secret = qbHex(crypto.getRandomValues(new Uint8Array(32)));
    const hash = blake3_hash_hex(secret);
    const expiry = (networkHeight || 0) + QB.INVOICE_TTL;
    const hints = Object.values(wState.l2_channels || {})
        .filter(c => c.v === 2 && c.role === 'receiver' && c.status === 'active'
            && c.latest.sender_amt >= amt
            && (networkHeight || 0) < c.expiry - QB.PAY_CUTOFF - QB.HTLC_MIN_HEADROOM)
        .sort((a, b) => b.latest.sender_amt - a.latest.sender_amt)
        .slice(0, 2).map(c => c.peer_pk);
    wState.l2_secrets = wState.l2_secrets || {};
    wState.l2_invoices = wState.l2_invoices || {};
    wState.l2_secrets[hash] = secret;
    wState.l2_invoices[hash] = { amount: amt, expiry };
    await saveState();
    let sig = '';
    if (opts && opts.sign) sig = await signMssAndSync(myPk, qbInvoiceCommit(myPk, hash, amt, expiry, hints));
    return { destPk: myPk, hash, amount: amt, expiry, hints, sig,
             text: `l2inv1:${myPk}:${hash}:${amt}:${expiry}:${hints.join(',')}` };
}

function qbParseInvoice(s) {
    const p = String(s || '').trim().split(':');
    if (p[0] === 'l2inv' && p.length === 4) {
        return { destPk: p[1], hash: p[2], amount: Number(p[3]), expiry: 0, hints: [] };
    }
    if (p[0] === 'l2inv1' && p.length >= 6) {
        return { destPk: p[1], hash: p[2], amount: Number(p[3]), expiry: Number(p[4]),
                 hints: p[5] ? p[5].split(',').filter(x => /^[0-9a-fA-F]{64}$/.test(x)) : [] };
    }
    throw new Error("Invalid invoice format");
}

/** Route and send a parsed invoice. Cheapest viable path wins:
 *  direct channel → via a hinted hub → via our best hub into a hinted hub
 *  (that entry hub may not reach the hint; HTLC_FAIL refunds within seconds). */
async function qbPayParsed(inv) {
    const amount = Number(inv.amount);
    const destPk = String(inv.destPk || '').substring(0, 64);
    if (!/^[0-9a-fA-F]{64}$/.test(destPk)) throw new Error("Invalid destination identity.");
    if (!/^[0-9a-fA-F]{64}$/.test(String(inv.hash || ''))) throw new Error("Invalid payment hash.");
    if (!Number.isFinite(amount) || amount <= 0) throw new Error("Invalid invoice amount.");
    if (inv.expiry && networkHeight >= inv.expiry) throw new Error("Invoice has expired — ask the payee for a fresh one.");
    if ((wState.l2_invoices || {})[inv.hash]) throw new Error("That is this wallet's own invoice.");

    const live = (c) => c.v === 2 && c.role === 'sender' && c.status === 'active' && c.open_acked
        && networkHeight < c.expiry - QB.PAY_CUTOFF;
    const chans = Object.values(wState.l2_channels || {}).filter(live);

    let chan = null, route = [], hops = 0;
    const direct = chans.find(c => c.peer_pk === destPk && c.latest.sender_amt >= amount);
    if (direct) { chan = direct; }
    if (!chan) {
        for (const h of inv.hints) {
            const c = chans.find(x => x.peer_pk === h && x.latest.sender_amt >= amount + QB.HOP_FEE);
            if (c) { chan = c; route = [destPk]; hops = 1; break; }
        }
    }
    if (!chan && inv.hints.length) {
        const c = chans.filter(x => x.peer_pk !== destPk && !inv.hints.includes(x.peer_pk))
            .sort((a, b) => b.latest.sender_amt - a.latest.sender_amt)[0];
        if (c && c.latest.sender_amt >= amount + 2 * QB.HOP_FEE) { chan = c; route = [inv.hints[0], destPk]; hops = 2; }
    }
    if (!chan && !inv.hints.length) {
        // Legacy invoice with no hints: best-funded hub; FAIL (or the hub's
        // JIT open) sorts out whether the last mile exists.
        const c = chans.filter(x => x.peer_pk !== destPk)
            .sort((a, b) => b.latest.sender_amt - a.latest.sender_amt)[0];
        if (c && c.latest.sender_amt >= amount + QB.HOP_FEE) { chan = c; route = [destPk]; hops = 1; }
    }
    if (!chan) throw new Error("No outbound channel can reach this payee. Open a channel to them directly"
        + (inv.hints.length ? ` or to one of their hubs (${inv.hints.map(h => h.substring(0, 12) + '…').join(', ')}).` : "."));

    const total = amount + hops * QB.HOP_FEE;
    const timeout = networkHeight + QB.HTLC_MIN_HEADROOM + (hops + 1) * QB.HTLC_HOP_DELTA;
    if (timeout > chan.expiry + QB.HTLC_MAX_PAST_EXPIRY) throw new Error("Channel is too close to expiry to route this payment — open a fresh one.");

    wState.l2_pay_pending = wState.l2_pay_pending || {};
    wState.l2_pay_pending[inv.hash] = { total, amount, destPk, timeout, at: networkHeight };
    await saveState();
    try {
        await qbSenderAdvance(chan, (next) => {
            if (next.sender_amt < total) throw new Error(`Insufficient channel balance (${next.sender_amt} MDS spendable, need ${total} incl. ${hops * QB.HOP_FEE} routing fee).`);
            next.sender_amt -= total;
            next.htlcs.push({ amount: total, timeout, secret_hash: inv.hash });
        }, QB.CMD_HTLC_ADD, route.map(pk => ({ kind: "address", value: pk })));
    } catch (e) {
        delete wState.l2_pay_pending[inv.hash];
        await saveState();
        throw e;
    }
    qbEvent('info', `Payment of ${amount} MDS in flight to ${destPk.substring(0, 12)}… (${hops} hop${hops === 1 ? '' : 's'}, fee ${hops * QB.HOP_FEE}).`, chan.id);
    self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
}

/** Someone asked us for an invoice over the bus (channelId = their request id). */
async function qbOnInvoiceReq(reqId, att, myPk) {
    const target = String(att("address") || '').substring(0, 64);
    if (target !== myPk) return;                              // not addressed to us / our own echo
    // Replay guard must be PERSISTENT: the node replays its whole chat ring on
    // every poll, and each re-answer would burn an MSS leaf on the signature.
    wState.qb_answered_reqs = wState.qb_answered_reqs || {};
    if (wState.qb_answered_reqs[reqId]) return;
    const p = qbUnpackU32(att("signature") || "");
    if (!p || p.extra.length < 8) return;
    const amount = Number(new DataView(p.extra.buffer, p.extra.byteOffset).getBigUint64(0, true));
    if (!Number.isFinite(amount) || amount <= 0) return;
    // Bounded auto-mint: prune expired, refuse if too many are outstanding.
    wState.l2_invoices = wState.l2_invoices || {};
    for (const [h, inv] of Object.entries(wState.l2_invoices)) {
        if (inv.expiry && networkHeight > inv.expiry) delete wState.l2_invoices[h];
    }
    if (Object.keys(wState.l2_invoices).length > 64) { qbLog('Ignored an invoice request — too many outstanding invoices.'); return; }
    if (qbLeafBudget().remaining <= QB.LEAF_RESERVE + 1) { qbLog('Ignored an invoice request — MSS key nearly exhausted.'); return; }
    if (Object.keys(wState.qb_answered_reqs).length > 200) wState.qb_answered_reqs = {};  // bounded
    wState.qb_answered_reqs[reqId] = networkHeight || 1;
    await saveState();
    const inv = await qbMintInvoice(amount, { sign: true });  // 1 MSS leaf: binds hash+amount to our pk
    await qbSendMsg(QB.CMD_INVOICE, reqId, qbPackInvoice(inv.hash, inv.amount, inv.expiry, inv.hints, inv.sig));
    qbEvent('info', `Issued an invoice for ${amount} MDS on request.`);
}

/** An invoice arrived for one of our pending pay-to requests. Verify the
 *  payee's signature (the request is on a public bus — anyone could race a
 *  fake invoice at it otherwise), then pay it. */
async function qbOnInvoice(reqId, att, myPk) {
    const req = (wState.l2_inv_reqs || {})[reqId];
    if (!req) return;                                         // not ours / already handled / echo
    const inv = qbUnpackInvoice(att("signature") || "");
    if (!inv) return;
    if (inv.amount !== req.amount) { qbLog('Invoice reply amount mismatch — ignored.'); return; }
    if (!inv.sig || !verify_mss_sig_wasm(inv.sig, qbInvoiceCommit(req.destPk, inv.hash, inv.amount, inv.expiry, inv.hints), req.destPk)) {
        qbLog(`Rejected an unsigned/forged invoice reply for request ${reqId.substring(0, 12)}…`);
        return;
    }
    delete wState.l2_inv_reqs[reqId];
    await saveState();
    try {
        await qbPayParsed({ destPk: req.destPk, hash: inv.hash, amount: inv.amount, expiry: inv.expiry, hints: inv.hints });
    } catch (e) {
        qbEvent('error', `Pay-to failed: ${e && e.message || e}`);
    }
}

// ── commitment-TTL recovery: the receiver's stored sender signature binds one
//    exact commitment (attempt N). If that commitment was mined but its TTL
//    lapsed before the reveal landed, only a FRESH sender signature (attempt
//    N+1 → new salt → new commitment) can save the state. Cooperative only —
//    which is exactly why the auto-closer runs with a fat margin.
async function qbOnResignReq(channel, att) {
    if (channel.role !== 'sender') return;
    const p = qbUnpackU32(att("signature") || "");
    if (!p || p.extra.length < 4) return;
    const attempt = new DataView(p.extra.buffer, p.extra.byteOffset).getUint32(0, true);
    if (p.n !== channel.latest.nonce) return;                 // only ever re-sign the exact latest state
    if (attempt > (channel.sig_attempt || 0) + 8) return;     // bound leaf burn from a hostile peer
    const budget = qbLeafBudget();
    if (budget.remaining <= 2) return;
    try {
        const built = qbBuildState(channel, channel.latest.sender_amt, channel.latest.receiver_amt,
            channel.latest.nonce, channel.latest.htlcs, attempt);
        const sig = await signMssAndSync(channel.sender_pk, built.commitment);
        channel.latest.sender_sig = sig;
        channel.sig_attempt = attempt;
        await saveState();
        const extra = new Uint8Array(4);
        new DataView(extra.buffer).setUint32(0, attempt, true);
        await qbSendMsg(QB.CMD_RESIGN, channel.id,
            qbPackU32(0, channel.latest.nonce, new Uint8Array([...extra, ...qbBytes(sig)])));
        qbLog(`Re-signed state ${channel.latest.nonce} at attempt ${attempt} on peer request.`);
    } catch (e) { qbLog(`Re-sign failed: ${e && e.message || e}`); }
}

async function qbOnResign(channel, att) {
    if (channel.role !== 'receiver') return;
    const p = qbUnpackU32(att("signature") || "");
    if (!p || p.extra.length < 4 + 32) return;
    const attempt = new DataView(p.extra.buffer, p.extra.byteOffset).getUint32(0, true);
    const sig = qbHex(p.extra.slice(4));
    if (p.n !== channel.latest.nonce) return;
    let built;
    try {
        built = qbBuildState(channel, channel.latest.sender_amt, channel.latest.receiver_amt,
            channel.latest.nonce, channel.latest.htlcs, attempt);
    } catch (_) { return; }
    if (!verify_mss_sig_wasm(sig, built.commitment, channel.sender_pk)) return;
    channel.latest.sender_sig = sig;
    channel.sig_attempt = attempt;
    if (channel.close && channel.close.stage === 'need-resign') {
        channel.close = null;                                 // engine restarts cleanly at the new attempt
    }
    await saveState();
    qbEvent('info', `Received a fresh close signature (attempt ${attempt}) — retrying settlement.`, channel.id);
    performChannelClose(channel.id, 'close').catch(() => {});
}

// ═══ close / refund engine — persisted and resumable ════════════════════════

async function performChannelClose(channelId, kind) {
    const channel = qbChan(channelId);
    if (!channel) throw new Error("Channel not found");
    if (qbInFlight.has(channelId)) return;
    qbInFlight.add(channelId);
    try { await qbCloseInner(channel, kind); }
    finally { qbInFlight.delete(channelId); }
}

async function qbCloseInner(channel, kind) {
    if (channel.status === 'closed') return;
    const say = (m) => { qbLog(m); self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: `[Q-Bolt] ${m}` } }); };

    // ── resume or start ────────────────────────────────────────────────────
    if (!channel.close || channel.close.done) {
        if (kind === 'close') {
            if (channel.role !== 'receiver') throw new Error("Only the receiver settles balances. As the sender you can Request Close, or claim the refund once block " + channel.expiry + " passes.");
            if (!channel.latest.sender_sig) throw new Error("No signed state to settle.");
        } else if (kind === 'refund') {
            if (channel.role !== 'sender') throw new Error("Only the sender can claim the refund.");
            if (networkHeight < channel.expiry) throw new Error(`Refund unlocks at block ${channel.expiry} (now ${networkHeight}).`);
        }
        channel.close = { kind, stage: 'build', attempt: kind === 'close' ? (channel.sig_attempt || 0) : 0, started_height: networkHeight, started_at: Date.now() };
        channel.status = kind === 'refund' ? 'refunding' : 'closing';
        await saveState();
    }
    const rec = channel.close;
    kind = rec.kind;

    // ── build (deterministic, safe to redo) ────────────────────────────────
    if (rec.stage === 'build' || !rec.state_json) {
        let built, mySig, peerSig = null;
        if (kind === 'refund') {
            built = JSON.parse(qbolt_build_refund_state(channel.id, channel.sender_pk, channel.receiver_pk,
                BigInt(channel.expiry), qbFundingJson(channel), rec.attempt >>> 0));
            say(`Signing refund (attempt ${rec.attempt})…`);
            mySig = await signMssAndSync(channel.sender_pk, built.commitment);
        } else {
            built = qbBuildState(channel, channel.latest.sender_amt, channel.latest.receiver_amt,
                channel.latest.nonce, channel.latest.htlcs, rec.attempt);
            peerSig = channel.latest.sender_sig;
            say(`Co-signing state ${channel.latest.nonce}…`);
            mySig = await signMssAndSync(channel.receiver_pk, built.commitment);
        }
        rec.state_json = JSON.stringify(built);
        rec.commitment = built.commitment;
        rec.my_sig = mySig;
        rec.peer_sig = peerSig;
        rec.stage = 'commit';
        await saveState();
    }

    // ── commit (idempotent: check the chain before mining) ─────────────────
    if (rec.stage === 'commit') {
        let already = false;
        try { const c = await rpc.checkCommitment(rec.commitment); already = !!(c && c.exists); } catch (_) {}
        if (!already) {
            say("Fetching network difficulty…");
            const st = await rpc.getState();
            say(`Mining commit PoW (difficulty ${st.required_pow || 24})…`);
            await new Promise(r => setTimeout(r, 30));
            const nonce = Number(mine_commitment_pow(rec.commitment, st.required_pow || 24, BigInt(st.height), st.header_hash));
            const cr = await rpc.commit(rec.commitment, nonce);
            const body = String((cr && (cr.body || cr.error)) || '');
            if (!cr.ok && !/already|exist|duplicate/i.test(body)) throw new Error(`Commit rejected: ${body}`);
        }
        say("Waiting for the commit to be mined (1/2)…");
        await awaitCommitmentMined(rec.commitment, say);
        rec.stage = 'reveal';
        rec.commit_seen_height = networkHeight;
        await saveState();
    }

    // ── reveal (retryable; failure classification is everything here) ──────
    if (rec.stage === 'reveal') {
        const built = JSON.parse(rec.state_json);
        const revealStr = (kind === 'refund')
            ? qbolt_build_refund_reveal(channel.sender_pk, channel.receiver_pk, BigInt(channel.expiry),
                qbFundingJson(channel), rec.state_json, rec.my_sig)
            : qbolt_build_close_reveal(channel.sender_pk, channel.receiver_pk, BigInt(channel.expiry),
                qbFundingJson(channel), rec.state_json, rec.peer_sig, rec.my_sig);
        say("Broadcasting settlement (2/2)…");
        const rr = await rpc.send(revealStr);
        if (!rr.ok) {
            const body = String(rr.body || rr.error || '');
            if (/not found/i.test(body)) {                    // funding already spent → someone beat us
                rec.done = true; await saveState();
                return qbReconcileExternalClose(channel.id);
            }
            if (/expired/i.test(body)) {                      // commitment TTL lapsed
                if (kind === 'refund') {
                    rec.attempt++; rec.stage = 'build'; rec.state_json = null; await saveState();
                    say(`Commitment expired — rebuilding refund at attempt ${rec.attempt}…`);
                    return qbCloseInner(channel, kind);       // sender self-signs: retry freely
                }
                rec.stage = 'need-resign'; await saveState();
                const extra = new Uint8Array(4);
                new DataView(extra.buffer).setUint32(0, (rec.attempt || 0) + 1, true);
                await qbSendMsg(QB.CMD_RESIGN_REQ, channel.id, qbPackU32(0, built.nonce, extra));
                qbEvent('warn', 'Settlement window lapsed — asked the peer for a fresh signature. Retrying automatically when it arrives.', channel.id);
                return;
            }
            throw new Error(`Settlement rejected: ${body}`); // transient/unknown → watcher retries this stage
        }
        rec.stage = 'confirm';
        await saveState();
    }

    if (rec.stage === 'need-resign') return;                 // parked until CMD_RESIGN arrives

    // ── confirm: funding gone AND our outputs exist ─────────────────────────
    if (rec.stage === 'confirm') {
        say("Waiting for the settlement to be mined…");
        await awaitCoinSpent(channel.funding[0].coin_id, say);
        const built = JSON.parse(rec.state_json);
        const probe = built.outputs && built.outputs.length ?
            compute_coin_id_hex(built.outputs[0].address, BigInt(built.outputs[0].value), built.outputs[0].salt) : null;
        let ours = false;
        if (probe) { try { const c = await rpc.checkCoin(probe); ours = !!(c && c.exists); } catch (_) { ours = true; } }
        rec.done = true;
        if (!ours) { await saveState(); return qbReconcileExternalClose(channel.id); }

        channel.status = 'closed';
        channel.closed = { height: networkHeight, kind, external: false };
        qbRegisterHtlcCoins(channel, built);
        await saveState();
        qbEvent('info', kind === 'refund'
            ? `Refund settled — ${built.capacity - QB.CLOSE_FEE} MDS returned on-chain.`
            : `Channel settled on-chain at state ${channel.latest.nonce}.`, channel.id);

        // Tell the peer immediately. Their funding-existence sweep would find
        // this eventually, but only every RECONCILE_EVERY blocks — until then
        // their UI shows a live channel and they could sign a state against
        // funding that no longer exists. Best-effort: the sweep is the backstop.
        qbSendMsg(QB.CMD_CLOSED, channel.id, qbPackU32(0, built.nonce != null ? built.nonce : (channel.latest.nonce || 0))).catch(() => {});

        performScan().catch(() => {});
        self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
    }
}

/** The peer says they settled this channel on-chain.
 *
 *  Never trust the wire: an unverified "closed" claim from a peer would be a
 *  free way to make us abandon a live channel. Confirm the funding really is
 *  spent before doing anything; if it isn't, ignore the message and let the
 *  sweep decide later. */
async function qbOnClosed(channel, att) {
    if (channel.status === 'closed') return;
    let spent = false;
    try {
        const chk = await rpc.checkCoin(channel.funding[0].coin_id);
        spent = !!(chk && !chk.exists);
    } catch (_) { return; }                                   // network flake → sweep will retry
    if (!spent) {
        qbLog(`Peer announced a close on ${channel.id.substring(0, 12)}… but its funding is still unspent — ignoring.`);
        return;
    }
    qbLog(`Peer announced a close on ${channel.id.substring(0, 12)}… — verified funding is spent, reconciling.`);
    await qbReconcileExternalClose(channel.id);
}

/** Track HTLC coins materialized by a close so the watcher can resolve them
 *  on-chain: receiver claims with the preimage before `timeout`; the sender
 *  refunds after. */
function qbRegisterHtlcCoins(channel, builtState) {
    const coins = builtState.htlc_coins || [];
    if (!coins.length) return;
    channel.onchain_htlcs = channel.onchain_htlcs || [];
    for (const c of coins) {
        channel.onchain_htlcs.push({ ...c, swept: false });
        if (typeof watchedContracts !== 'undefined') watchedContracts.add(c.address);
    }
    if (typeof updateWasmWatchlist === 'function') updateWasmWatchlist();
    qbEvent('warn', `${coins.length} HTLC coin(s) settled on-chain and await resolution — the wallet will handle them automatically.`, channel.id);
}

/** The funding was spent but not by us. Work out which state landed (bounded
 *  probe over recent nonces — output salts are deterministic per nonce), mark
 *  the channel closed, and register any HTLC coins from that state. */
async function qbReconcileExternalClose(channelId) {
    const channel = qbChan(channelId);
    if (!channel || channel.status === 'closed') return;

    // A never-acknowledged sender channel whose funding is "missing" was NOT
    // settled by the peer — the peer never even created the channel, so they
    // hold no state to settle and cannot spend the covenant before expiry.
    // Missing funding here means the funding record itself is wrong or the
    // funding tx never actually landed. Report that truth per-coin instead of
    // inventing a peer action, and void the channel rather than "closing" it.
    if (channel.role === 'sender' && !channel.open_acked) {
        const missing = [], alive = [];
        for (const f of channel.funding) {
            try {
                const c = await rpc.checkCoin(f.coin_id);
                (c && c.exists ? alive : missing).push(f.coin_id);
            } catch (_) { alive.push(f.coin_id); }            // unknown ≠ missing
        }
        if (!missing.length) return;                          // transient flake — funding is intact
        channel.status = 'closed';
        channel.closed = { height: networkHeight, kind: 'void-open', external: false };
        await saveState();
        qbEvent('warn',
            `Channel open failed: ${missing.length} of ${channel.funding.length} recorded funding coin(s) `
            + `(${missing.map(m => m.substring(0, 12) + '…').join(', ')}) do not exist on-chain, and the peer never `
            + `acknowledged the channel. The funding transaction either never confirmed or the wallet's record of it is wrong. `
            + `Any value that actually left the wallet will be re-credited by the scan.`, channelId);
        performScan().catch(() => {});
        self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
        return;
    }

    let landed = null;
    const tryNonce = async (nonce, senderAmt, receiverAmt, htlcs) => {
        let b;
        try { b = qbBuildState(channel, senderAmt, receiverAmt, nonce, htlcs, 0); } catch (_) { return null; }
        if (!b.outputs.length) return null;
        const o = b.outputs[0];
        try {
            const c = await rpc.checkCoin(compute_coin_id_hex(o.address, BigInt(o.value), o.salt));
            return (c && c.exists) ? b : null;
        } catch (_) { return null; }
    };

    // Latest first (overwhelmingly the common case), then the refund shape,
    // then a bounded walk back through past states we can reconstruct.
    landed = await tryNonce(channel.latest.nonce, channel.latest.sender_amt, channel.latest.receiver_amt, channel.latest.htlcs || []);
    if (!landed) {
        try {
            const rb = JSON.parse(qbolt_build_refund_state(channel.id, channel.sender_pk, channel.receiver_pk,
                BigInt(channel.expiry), qbFundingJson(channel), 0));
            const o = rb.outputs[0];
            const c = await rpc.checkCoin(compute_coin_id_hex(o.address, BigInt(o.value), o.salt)).catch(() => null);
            if (c && c.exists) landed = rb;
        } catch (_) {}
    }
    const history = channel.state_history || [];
    for (let i = history.length - 1; i >= 0 && !landed && history.length - i <= QB.EXTERNAL_PROBE_DEPTH; i--) {
        const s = history[i];
        landed = await tryNonce(s.nonce, s.sender_amt, s.receiver_amt, s.htlcs || []);
    }

    channel.status = 'closed';
    channel.closed = { height: networkHeight, kind: 'external', external: true };
    if (landed) qbRegisterHtlcCoins(channel, landed);
    await saveState();

    if (channel.role === 'sender') {
        // Any receiver-broadcast state pays the sender AT LEAST the latest
        // sender balance (older states favor the sender) — never less.
        qbEvent('info', `Channel was settled by the peer${landed ? '' : ' (with an earlier state — that only ever pays you more)'}. Funds arrive with the next scan.`, channelId);
    } else {
        qbEvent(landed ? 'info' : 'warn', landed
            ? 'Channel was settled on-chain (possibly by this wallet on another device). Funds arrive with the next scan.'
            : 'Channel was closed externally. If the sender reclaimed it via the expiry refund, off-chain earnings on it were forfeited — this wallet auto-closes well before expiry precisely to prevent that; it can happen only if the wallet was offline through the entire closing window.', channelId);
    }
    performScan().catch(() => {});
    self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
}

// ═══ on-chain HTLC resolution after a close ═════════════════════════════════

async function qbSweepHtlc(channel, entry) {
    const key = `${channel.id}:${entry.coin_id}`;
    if (qbInFlight.has(key)) return;
    qbInFlight.add(key);
    try {
        const iClaim = channel.role === 'receiver';
        const secret = (wState.l2_secrets || {})[entry.secret_hash] || (wState.l2_observed_secrets || {})[entry.secret_hash];
        if (iClaim && !secret) return;                        // nothing to do without the preimage
        if (iClaim && networkHeight >= entry.timeout) return; // too late — the refund path owns it now
        if (!iClaim && networkHeight < entry.timeout) return; // sender must wait out the timeout

        let chk = null;
        try { chk = await rpc.checkCoin(entry.coin_id); } catch (_) { return; }
        if (!chk || !chk.exists) { entry.swept = true; await saveState(); return; }

        await acquireSendLock();
        try {
            if (!mssCachesReady) await loadMssCaches();
            const myPk = getPrimaryMssPk();
            const myAddr = compute_p2pk_address_hex(myPk);
            const utxoArray = getSpendableUtxos().map(u =>
                (u.is_mss && wState.mssAddrs[u.address]) ? { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf } : u);

            const buildOutputs = (val) => JSON.stringify((() => {
                const parts = []; let n = BigInt(val), bit = 0n;
                while (n > 0n) { if (n & 1n) parts.push(Number(1n << bit)); n >>= 1n; bit += 1n; }
                return parts.map(v => ({ out_type: "standard", address: myAddr, value: v, salt: null }));
            })());
            const inputsJson = JSON.stringify([{ coin_id: entry.coin_id, witness: "", value: entry.value, salt: entry.salt, state: null }]);

            // Fee strategy. prepare_script_spend only reaches for wallet UTXOs when
            // (outputs + fee) exceeds the contract input's own value; otherwise it
            // needs none and returns the surplus as change to us. So: first try to
            // keep the FULL value (the wallet pays the fee and we net 100%), then
            // walk a ladder of self-funded reserves for the zero-balance case.
            // Over-reserving is free — the unused remainder comes back as change.
            let ctx = null, lastErr = null;
            for (const reserve of [0, ...QB.SWEEP_FEE_LADDER]) {
                if (reserve >= entry.value) break;             // can't self-fund from a coin this small
                try {
                    ctx = JSON.parse(wallet.prepare_script_spend(
                        JSON.stringify(utxoArray), entry.bytecode, inputsJson,
                        buildOutputs(entry.value - reserve), wState.nextWotsIndex));
                    break;
                } catch (e) { lastErr = e; }
            }
            if (!ctx) {
                // Every rung failed: the coin is too small to pay its own way and the
                // wallet has nothing to contribute. Leave it for a later block — a
                // channel close or any incoming payment makes this succeed.
                qbLog(`HTLC ${entry.coin_id.substring(0, 12)}… not sweepable yet (${entry.value} MDS, empty wallet): ${lastErr && lastErr.message || lastErr}`);
                return;
            }

            const sig = await signMssAndSync(myPk, ctx.commitment);
            for (let i = 0; i < ctx.contract_inputs.length; i++) {
                ctx.contract_inputs[i].witness = iClaim ? `${sig},${secret},01` : `${sig},00,00`;
            }
            const revealStr = wallet.build_script_reveal(JSON.stringify(ctx), ctx.commitment, ctx.tx_salt);
            await reserveAndLock(ctx, "Saving wallet state…");

            const st = await rpc.getState();
            const nonce = Number(mine_commitment_pow(ctx.commitment, st.required_pow || 24, BigInt(st.height), st.header_hash));
            const cr = await rpc.commit(ctx.commitment, nonce);
            if (!cr.ok) throw new Error(`Commit rejected: ${cr.body || cr.error}`);
            await awaitCommitmentMined(ctx.commitment, (m) => qbLog(m));
            const rr = await rpc.send(revealStr);
            if (!rr.ok) {
                if (/not found/i.test(String(rr.body || rr.error || ''))) { entry.swept = true; await saveState(); return; }
                throw new Error(`Reveal rejected: ${rr.body || rr.error}`);
            }
            await awaitCoinSpent(entry.coin_id, (m) => qbLog(m));
            entry.swept = true;
            await saveState();
            qbEvent('info', `${iClaim ? 'Claimed' : 'Refunded'} on-chain HTLC (${entry.value} MDS).`, channel.id);
            performScan().catch(() => {});
        } finally { releaseSendLock(); }
    } catch (e) {
        qbLog(`HTLC sweep deferred (${entry.coin_id.substring(0, 12)}…): ${e && e.message || e}`);
    } finally { qbInFlight.delete(key); }
}

// ═══ the watcher — reality's heartbeat ══════════════════════════════════════

let qbTickRunning = false;
let qbLastReconcileHeight = 0;

async function qbWatchTick(height) {
    if (qbTickRunning || !wallet || !height) return;
    qbTickRunning = true;
    try {
        // A transaction interrupted mid-flight is retried here every block, not
        // just at login — a reveal dropped by a flapping transport should not
        // wait for the user to reload.
        if (wState.pending_tx && !qbInFlight.has('pending_tx')) {
            qbInFlight.add('pending_tx');
            recoverPendingTx().catch(() => {}).finally(() => qbInFlight.delete('pending_tx'));
        }
        // Pending inbound opens: funding may have just confirmed.
        for (const id of Object.keys(wState.qb_pending_opens || {})) {
            await qbTryFinalizeInboundOpen(id).catch(() => {});
        }

        // JIT-parked forwards: deliver once the freshly opened channel ACKs,
        // or fail upstream if the timeout budget runs out first.
        for (const [hash, f] of Object.entries(wState.qb_fwd || {})) {
            if (height + QB.HTLC_MIN_HEADROOM >= f.timeout) {
                delete wState.qb_fwd[hash]; await saveState();
                const up = qbChan(f.upId);
                try { if (up && up.role === 'receiver' && up.status === 'active') await qbFail(up, hash, 6); } catch (_) {}
                continue;
            }
            const c = Object.values(wState.l2_channels || {}).find(x => x.v === 2 && x.role === 'sender'
                && x.status === 'active' && x.open_acked && x.peer_pk === f.nextPk
                && x.latest.sender_amt >= f.amount && height < x.expiry - QB.PAY_CUTOFF);
            if (!c || qbInFlight.has(`fwd:${hash}`)) continue;
            qbInFlight.add(`fwd:${hash}`);
            try {
                await qbSenderAdvance(c, (next) => {
                    next.sender_amt -= f.amount;
                    next.htlcs.push({ amount: f.amount, timeout: f.timeout, secret_hash: hash });
                }, QB.CMD_HTLC_ADD, []);
                wState.l2_routes = wState.l2_routes || {};
                wState.l2_routes[hash] = { fromCoinId: f.upId, amount: f.inAmount };
                delete wState.qb_fwd[hash];
                await saveState();
                qbEvent('info', `JIT channel ready — forwarded ${f.amount} MDS to ${f.nextPk.substring(0, 12)}….`, c.id);
            } catch (e) { qbLog(`JIT forward deferred: ${e && e.message || e}`); }
            finally { qbInFlight.delete(`fwd:${hash}`); }
        }
        // Invoice requests that never got a reply.
        for (const [rid, r] of Object.entries(wState.l2_inv_reqs || {})) {
            if (height - r.height > 3) {
                delete wState.l2_inv_reqs[rid]; await saveState();
                qbEvent('warn', `No invoice reply from ${r.destPk.substring(0, 12)}… — the payee's wallet must be online to be paid.`);
            }
        }
        // In-flight payments whose HTLC timed out (the expired-HTLC cancel below refunds them).
        for (const [hash, p] of Object.entries(wState.l2_pay_pending || {})) {
            if (height > p.timeout) {
                delete wState.l2_pay_pending[hash]; await saveState();
                qbEvent('warn', `Payment of ${p.amount} MDS to ${p.destPk.substring(0, 12)}… timed out — the balance returns automatically.`);
            }
        }
        // Invoices long past expiry: forget the amounts (secrets stay — harmless).
        for (const [hash, inv] of Object.entries(wState.l2_invoices || {})) {
            if (inv.expiry && height > inv.expiry + QB.HTLC_MAX_PAST_EXPIRY) delete wState.l2_invoices[hash];
        }

        for (const [id, c] of Object.entries(wState.l2_channels || {})) {
            if (c.v !== 2) continue;

            // Resume any parked close/refund.
            if (c.close && !c.close.done && c.close.stage !== 'need-resign' && !qbInFlight.has(id)) {
                performChannelClose(id, c.close.kind).catch(e => qbLog(`Close resume deferred: ${e && e.message || e}`));
            }
            if (c.status !== 'active') {
                // Post-close: resolve materialized HTLC coins.
                for (const entry of (c.onchain_htlcs || [])) {
                    if (!entry.swept) qbSweepHtlc(c, entry).catch(() => {});
                }
                continue;
            }

            // Receiver: warn, then auto-close before the sender's refund unlocks.
            if (c.role === 'receiver') {
                if (height >= c.expiry - QB.WARN_MARGIN && !c.warned) {
                    c.warned = true; await saveState();
                    qbEvent('warn', `Channel enters its settlement window soon (auto-settles at block ${c.expiry - QB.CLOSE_MARGIN}).`, id);
                }
                if (height >= c.expiry - QB.CLOSE_MARGIN) {
                    qbEvent('info', 'Settlement window reached — settling the channel now.', id);
                    performChannelClose(id, 'close').catch(e => qbEvent('error', `Auto-close failed (will retry): ${e && e.message || e}`, id));
                    continue;
                }
                // Claim stall: we revealed a preimage and the sender never credited it.
                for (const [hash, pc] of Object.entries(c.pending_claims || {})) {
                    if (height - pc.sent_height > QB.CLAIM_STALL_BLOCKS) {
                        qbEvent('warn', 'Sender did not honor a revealed preimage — force-settling with the HTLC on-chain.', id);
                        performChannelClose(id, 'close').catch(() => {});
                        break;
                    }
                }
            }

            // Sender: refund the instant it unlocks; nudge handshake/updates.
            if (c.role === 'sender') {
                if (height >= c.expiry) {
                    qbEvent('info', 'Channel expired — claiming the refund.', id);
                    performChannelClose(id, 'refund').catch(e => qbEvent('error', `Refund failed (will retry): ${e && e.message || e}`, id));
                    continue;
                }
                const tries = c.send_tries || 0;
                if (!c.open_acked && tries < QB.REBROADCAST_MAX &&
                    height - (c.last_send_height || 0) >= QB.OPEN_REBROADCAST_EVERY) {
                    c.send_tries = tries + 1; c.last_send_height = height; await saveState();
                    await qbBroadcastOpen(c);
                } else if (c.open_acked && !c.close_requested && (c.acked_nonce || 0) < c.latest.nonce && tries < QB.REBROADCAST_MAX &&
                    height - (c.last_send_height || 0) >= QB.UPDATE_REBROADCAST_EVERY) {
                    c.send_tries = tries + 1; c.last_send_height = height; await saveState();
                    await qbSendMsg(QB.CMD_UPDATE, id, qbPackState(c.latest, c.latest.sender_sig));
                }
                // Housekeeping: cancel HTLCs that expired unclaimed (frees balance).
                const expired = (c.latest.htlcs || []).filter(h => height > h.timeout);
                if (expired.length && !qbInFlight.has(`cancel:${id}`)) {
                    qbInFlight.add(`cancel:${id}`);
                    try {
                        await qbSenderAdvance(c, (next) => {
                            for (const h of expired) {
                                next.htlcs = next.htlcs.filter(x => x.secret_hash !== h.secret_hash);
                                next.sender_amt += h.amount;
                            }
                        }, QB.CMD_UPDATE);
                        qbLog(`Cancelled ${expired.length} expired HTLC(s).`);
                    } catch (_) {} finally { qbInFlight.delete(`cancel:${id}`); }
                }
            }
        }

        // Funding-existence sweep: detects closes we didn't perform.
        if (height - qbLastReconcileHeight >= QB.RECONCILE_EVERY) {
            qbLastReconcileHeight = height;
            for (const [id, c] of Object.entries(wState.l2_channels || {})) {
                if (c.v !== 2 || (c.status !== 'active' && c.status !== 'closing' && c.status !== 'refunding')) continue;
                if (qbInFlight.has(id)) continue;             // our own settlement is spending it right now
                try {
                    const chk = await rpc.checkCoin(c.funding[0].coin_id);
                    if (chk && !chk.exists) await qbReconcileExternalClose(id);
                } catch (_) {}
            }
        }
    } finally { qbTickRunning = false; }
}

/** Fund a new outbound (sender-side) channel. Shared by the UI open and the
 *  hub's JIT open; the record is created once funding confirms on-chain
 *  (performSend captures the funding coins via pendingChannelOpen). */
async function qbOpenOutbound(peerPk, amount, lifetime) {
    const myPk = getPrimaryMssPk();
    if (!myPk) throw new Error("Network Sync required first to initialize your MSS L2 identity.");
    const rawPeer = String(peerPk || '').replace(/^0x/i, '').substring(0, 64);
    if (!/^[0-9a-fA-F]{64}$/.test(rawPeer)) throw new Error("Peer identity must be 64 hex characters.");
    if (rawPeer === myPk) throw new Error("You cannot open a channel to yourself.");
    const amt = Number(amount);
    if (!Number.isFinite(amt) || amt <= 0) throw new Error("Enter a positive amount to fund.");
    const capacity = amt + QB.CLOSE_FEE;
    if (capacity < QB.MIN_CAPACITY) throw new Error(`Minimum channel size is ${QB.MIN_CAPACITY - QB.CLOSE_FEE} MDS.`);

    const budget = qbLeafBudget();
    if (budget.remaining <= QB.LEAF_RESERVE + 2) throw new Error(`MSS key nearly exhausted (${budget.remaining} one-time signatures left). Generate a new receiving address before opening a channel.`);

    let life = Number(lifetime) || QB.DEFAULT_LIFETIME;
    life = Math.max(QB.MIN_LIFETIME, Math.min(QB.MAX_LIFETIME, life));
    const expiry = (networkHeight || 0) + life;
    if (!networkHeight) throw new Error("Waiting for the current block height — Sync first.");

    const channelAddr = qbolt_channel_address(myPk, rawPeer, BigInt(expiry));

    // Stash the open intent so a crash between funding and record-creation
    // is recoverable (qbBoot re-checks and finalizes).
    wState.qb_open_intent = {
        channelAddr, senderPk: myPk, receiverPk: rawPeer, expiry,
        amount: amt, createdAt: Date.now(), channelId: null, fundingCoins: null,
    };
    await saveState();

    pendingChannelOpen = { channelAddr, senderPk: myPk, receiverPk: rawPeer, expiry, amount: amt, isQbolt: true };
    await acquireSendLock();
    try { await performSend(channelAddr, capacity); }
    finally { releaseSendLock(); }
}

async function qbBroadcastOpen(channel) {
    return await qbSendMsg(QB.CMD_OPEN, channel.id,
        qbPackOpen(channel.expiry, channel.funding, channel.state0_sig),
        [{ kind: "address", value: channel.sender_pk }]);
}

/** Boot pass: legacy migration, crashed-open recovery, close resumption. */
async function qbBoot() {
    let dirty = false;
    for (const [id, c] of Object.entries(wState.l2_channels || {})) {
        if (c.v !== 2 && c.status !== 'frozen-legacy') {
            c.status = 'frozen-legacy';
            dirty = true;
            qbEvent('warn', 'A channel from the old Q-Bolt protocol was frozen. Its 2-of-2 funding has no timeout escape — use "Legacy Settle" (both parties must be online on the new wallet) to recover the funds.', id);
        }
        if (c.v === 2 && c.channel_addr && typeof watchedContracts !== 'undefined') {
            watchedContracts.add(c.channel_addr);
        }
    }
    // A crash between funding confirmation and channel creation leaves an intent.
    const intent = wState.qb_open_intent;
    if (intent) {
        const stale = Date.now() - (intent.createdAt || 0) > 2 * 3600 * 1000;
        if (intent.channelId && intent.fundingCoins && intent.fundingCoins.length) {
            // Funding outputs were captured before the interruption — check the
            // chain and either finalize the channel or age the intent out.
            try {
                const chk = await rpc.checkCoin(intent.fundingCoins[0].coin_id);
                if (chk && chk.exists) { await qbFinalizeOutboundOpen(intent); dirty = false; }
                else if (stale) { delete wState.qb_open_intent; dirty = true; qbLog('Dropped a stale channel-open intent — its funding never confirmed.'); }
            } catch (_) {}
        } else if (stale) {
            // The send failed before any funding output existed; nothing to
            // recover. (recoverPendingTx handles the tx itself, if one is saved.)
            delete wState.qb_open_intent; dirty = true;
            qbLog('Dropped a stale channel-open intent — the funding transaction never completed.');
        }
    }
    if (typeof updateWasmWatchlist === 'function') updateWasmWatchlist();
    if (dirty) await saveState();
    qbWatchTick(networkHeight).catch(() => {});
}

/** Create the sender-side channel record once funding is confirmed on-chain,
 *  sign state 0 (so the receiver can settle a zero-payment channel any time),
 *  and broadcast OPEN2. Idempotent. */
async function qbFinalizeOutboundOpen(intent) {
    if (qbChan(intent.channelId)) { delete wState.qb_open_intent; await saveState(); return; }
    const capacity = intent.fundingCoins.reduce((a, f) => a + f.value, 0);

    const built0 = JSON.parse(qbolt_build_state(intent.channelId, intent.senderPk, intent.receiverPk,
        BigInt(intent.expiry), JSON.stringify(intent.fundingCoins.map(f => ({ value: f.value, salt: f.salt }))),
        BigInt(capacity - QB.CLOSE_FEE), 0n, 0, "[]", 0));
    const sig0 = await signMssAndSync(intent.senderPk, built0.commitment);

    wState.l2_channels = wState.l2_channels || {};
    wState.l2_channels[intent.channelId] = {
        v: 2, id: intent.channelId, role: 'sender',
        sender_pk: intent.senderPk, receiver_pk: intent.receiverPk, peer_pk: intent.receiverPk,
        expiry: intent.expiry, funding: intent.fundingCoins, capacity,
        channel_addr: intent.channelAddr, status: 'active',
        latest: { nonce: 0, sender_amt: capacity - QB.CLOSE_FEE, receiver_amt: 0, htlcs: [], sender_sig: sig0 },
        state0_sig: sig0, sig_attempt: 0, acked_nonce: -1, open_acked: false,
        created_height: networkHeight, leaf_spent: 1, close: null, closed: null,
        pending_claims: {}, onchain_htlcs: [], last_send_height: networkHeight, send_tries: 1,
    };
    delete wState.qb_open_intent;
    if (typeof watchedContracts !== 'undefined') { watchedContracts.add(intent.channelAddr); updateWasmWatchlist(); }
    await saveState();
    await qbBroadcastOpen(wState.l2_channels[intent.channelId]);
    qbEvent('info', `Channel funded and open: ${capacity - QB.CLOSE_FEE} MDS spendable, expires at block ${intent.expiry} (~${Math.round((intent.expiry - networkHeight) / 60)} h).`, intent.channelId);
    self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
}

// ═══ legacy (v1) cooperative rescue ═════════════════════════════════════════
// v1 channels sit at a bare 2-of-2 whose "close" never actually worked (it
// skipped the commit phase and assumed a single funding coin). This path
// builds a CORRECT multi-coin close and gathers both fresh signatures over
// the bus. There is no unilateral option — a bare 2-of-2 has none.

async function qbLegacyClose(channelId) {
    const c = (wState.l2_channels || {})[channelId];
    if (!c || c.v === 2) throw new Error("Not a legacy channel");
    const myPk = getPrimaryMssPk();
    const addr = build_multisig_2of2_address(c.alice_pk, c.bob_pk);

    // Discover ALL coins still sitting at the 2-of-2 (the v1 open split the
    // funding into power-of-2 denominations; v1 only recorded one of them).
    if (typeof watchedContracts !== 'undefined' && !watchedContracts.has(addr)) {
        watchedContracts.add(addr); updateWasmWatchlist();
        await performScan().catch(() => {});
    }
    const coins = Object.values(contractCoins || {}).filter(x => x.address === addr && !x.state);
    if (!coins.length) throw new Error("No coins found at the legacy channel address — Sync first (or the channel was already settled).");
    const capacity = coins.reduce((a, x) => a + Number(x.value), 0);
    if (capacity <= QB.CLOSE_FEE) throw new Error("Legacy channel value does not cover the settlement fee.");

    const dist = capacity - QB.CLOSE_FEE;
    let aliceAmt = Math.min(Number(c.latest_state.alice_amt || 0), dist);
    let bobAmt = dist - aliceAmt;                             // conservation, fee-adjusted

    const funding = coins.map(x => ({ value: Number(x.value), salt: x.salt }));
    const stateJson = qbolt_build_legacy_close_state(channelId, c.alice_pk, c.bob_pk,
        JSON.stringify(funding), BigInt(aliceAmt), BigInt(bobAmt), 0);
    const built = JSON.parse(stateJson);
    const iAmAlice = myPk === c.alice_pk;
    const mySig = await signMssAndSync(myPk, built.commitment);

    c.legacy_close = { state_json: stateJson, funding, alice_amt: aliceAmt, bob_amt: bobAmt, my_sig: mySig, i_am_alice: iAmAlice, attempt: 0 };
    await saveState();

    // Ship the proposal: [ver][attempt u32][aliceAmt u64][bobAmt u64][count u8][{value,salt}×n][sig…]
    const coinsBin = new Uint8Array(funding.length * 40);
    { const v = new DataView(coinsBin.buffer); let o = 0;
      for (const f of funding) { v.setBigUint64(o, BigInt(f.value), true); coinsBin.set(qbBytes(f.salt), o + 8); o += 40; } }
    const head = new Uint8Array(21);
    { const v = new DataView(head.buffer); v.setUint32(0, 0, true);
      v.setBigUint64(4, BigInt(aliceAmt), true); v.setBigUint64(12, BigInt(bobAmt), true); head[20] = funding.length; }
    await qbSendMsg(QB.CMD_LEGACY_CLOSE_REQ, channelId,
        qbPackU32(0, 0, new Uint8Array([...head, ...coinsBin, ...qbBytes(mySig)])));
    qbEvent('info', 'Legacy settlement proposed — waiting for the peer to co-sign (they must be online on the new wallet).', channelId);
}

async function qbOnLegacyCloseReq(channelId, att, myPk) {
    const c = (wState.l2_channels || {})[channelId];
    if (!c || c.v === 2) return;
    const p = qbUnpackU32(att("signature") || "");
    if (!p || p.extra.length < 21) return;
    const v = new DataView(p.extra.buffer, p.extra.byteOffset);
    const attempt = v.getUint32(0, true);
    const aliceAmt = Number(v.getBigUint64(4, true));
    const bobAmt = Number(v.getBigUint64(12, true));
    const n = p.extra[20];
    if (p.extra.length < 21 + n * 40) return;
    const funding = [];
    for (let i = 0; i < n; i++) {
        const o = 21 + i * 40;
        funding.push({ value: Number(v.getBigUint64(o, true)), salt: qbHex(p.extra.slice(o + 8, o + 40)) });
    }
    const theirSig = qbHex(p.extra.slice(21 + n * 40));

    // Verify their sig over the exact proposed state, and sanity-check the
    // split against our own v1 record before co-signing.
    let built;
    try {
        built = JSON.parse(qbolt_build_legacy_close_state(channelId, c.alice_pk, c.bob_pk,
            JSON.stringify(funding), BigInt(aliceAmt), BigInt(bobAmt), attempt));
    } catch (_) { return; }
    const iAmAlice = myPk === c.alice_pk;
    const peerPk = iAmAlice ? c.bob_pk : c.alice_pk;
    if (!verify_mss_sig_wasm(theirSig, built.commitment, peerPk)) return;
    const myRecorded = iAmAlice ? Number(c.latest_state.alice_amt || 0) : Number(c.latest_state.bob_amt || 0);
    const myProposed = iAmAlice ? aliceAmt : bobAmt;
    if (myProposed + 64 < Math.min(myRecorded, aliceAmt + bobAmt)) {   // small tolerance for the fee haircut
        qbEvent('warn', `Legacy settlement proposal pays this wallet ${myProposed} MDS vs ${myRecorded} recorded — refusing to co-sign.`, channelId);
        return;
    }
    const mySig = await signMssAndSync(myPk, built.commitment);
    await qbSendMsg(QB.CMD_LEGACY_CLOSE_SIG, channelId,
        qbPackU32(0, attempt, qbBytes(mySig)));
    qbEvent('info', `Co-signed legacy settlement (receiving ${myProposed} MDS) — the peer will broadcast it.`, channelId);
}

async function qbOnLegacyCloseSig(channelId, att) {
    const c = (wState.l2_channels || {})[channelId];
    if (!c || c.v === 2 || !c.legacy_close) return;
    const p = qbUnpackU32(att("signature") || "");
    if (!p || !p.extra.length) return;
    const rec = c.legacy_close;
    const built = JSON.parse(rec.state_json);
    const peerPk = rec.i_am_alice ? c.bob_pk : c.alice_pk;
    const theirSig = qbHex(p.extra);
    if (!verify_mss_sig_wasm(theirSig, built.commitment, peerPk)) return;

    const aliceSig = rec.i_am_alice ? rec.my_sig : theirSig;
    const bobSig = rec.i_am_alice ? theirSig : rec.my_sig;
    qbEvent('info', 'Peer co-signed — broadcasting the legacy settlement…', channelId);
    try {
        const st = await rpc.getState();
        const nonce = Number(mine_commitment_pow(built.commitment, st.required_pow || 24, BigInt(st.height), st.header_hash));
        const cr = await rpc.commit(built.commitment, nonce);
        if (!cr.ok && !/already|exist|duplicate/i.test(String(cr.body || cr.error || ''))) throw new Error(`Commit rejected: ${cr.body || cr.error}`);
        await awaitCommitmentMined(built.commitment, (m) => qbLog(m));
        const revealStr = qbolt_build_legacy_close_reveal(c.alice_pk, c.bob_pk,
            JSON.stringify(rec.funding), rec.state_json, aliceSig, bobSig);
        const rr = await rpc.send(revealStr);
        if (!rr.ok) throw new Error(`Reveal rejected: ${rr.body || rr.error}`);
        delete wState.l2_channels[channelId];
        await saveState();
        qbEvent('info', 'Legacy channel settled on-chain. Funds arrive with the next scan.', channelId);
        performScan().catch(() => {});
        self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
    } catch (e) {
        qbEvent('error', `Legacy settlement failed: ${e && e.message || e}`, channelId);
    }
}


async function handleL2Chat(msg) {
    const cmd = msg.words[1];

    // Q-Bolt v2 protocol traffic.
    if (QB_CMDS.has(cmd)) { await handleQbChat(msg); return; }

    // The legacy v1 channel handlers (cmds 100 / 40-43) are intentionally
    // GONE: they auto-co-signed ANY counterparty state carrying a fresh
    // nonce - the exact behavior that made balance theft possible. v1
    // channels are frozen at boot and recovered via the legacy settlement
    // flow (qbLegacyClose) instead.

    // ── L2 DEX ROUTING ──
    if (cmd >= 200 && cmd <= 206) {
        const sigAtt = msg.attachments.find(a => a.kind === "signature")?.value;
        if (!sigAtt) return;

        try {
            // Decode the arbitrary JSON payload we tunneled through the Signature attachment
            const hexPairs = sigAtt.match(/.{1,2}/g) || [];
            const jsonStr = new TextDecoder().decode(new Uint8Array(hexPairs.map(b => parseInt(b, 16))));
            const payload = JSON.parse(jsonStr);
            payload.nonce = msg.nonce;

            // Route to the UI Thread
            if (cmd === 200) {
                self.postMessage({ type: 'DEX_OFFER_RECEIVED', payload });
            } else if (cmd === 201) {
                self.postMessage({ type: 'DEX_ACCEPT_RECEIVED', payload });
            } else if (cmd === 202) {
                self.postMessage({ type: 'DEX_LOCKED_RECEIVED', payload });
            } else if (cmd === 203) {
                self.postMessage({ type: 'DEX_LOCKING_RECEIVED', payload });
            } else if (cmd === 204) {
                // Feature 2: a taker's submarine (L2) intent for one of our limit units.
                self.postMessage({ type: 'DEX_SUBMARINE_RECEIVED', payload });
            } else if (cmd === 205) {
                self.postMessage({ type: 'DEX_BIDFILL_RECEIVED', payload });
            } else if (cmd === 206) {
                self.postMessage({ type: 'DEX_BIDSECRET_RECEIVED', payload });
            }
        } catch (e) {
            console.error("Failed to parse DEX L2 payload", e);
        }
    }
}



// ═══════════════════════════════════════════════════════════════════════════════
//  Self-Test Harness
// ═══════════════════════════════════════════════════════════════════════════════
//
// Run via: worker.postMessage({ type: 'RUN_TESTS' })
// Results: worker.onmessage → { type: 'TEST_RESULTS', payload: { passed, failed, results } }
//
// These tests exercise the IndexedDB ↔ WASM integration layer that the
// Rust unit tests cannot cover (they require a browser environment with
// IndexedDB, Web Crypto API, and the WASM runtime).

/**
 * Run all self-tests and return results.
 * @returns {Promise<{passed: number, failed: number, results: Array<{name: string, ok: boolean, error?: string}>}>}
 */
async function runSelfTests() {
    const results = [];

    /**
     * @param {string} name
     * @param {Function} fn
     */
    async function test(name, fn) {
        try {
            await fn();
            results.push({ name, ok: true });
        } catch (e) {
            results.push({ name, ok: false, error: e.toString() });
        }
    }

    function assert(condition, msg) {
        if (!condition) throw new Error(`Assertion failed: ${msg}`);
    }

    function assertEqual(a, b, msg) {
        if (a !== b) throw new Error(`${msg}: expected ${b}, got ${a}`);
    }

    // ── IndexedDB round-trip ────────────────────────────────────────────

    await test('idb_put_get_roundtrip', async () => {
        const testData = new Uint8Array([1, 2, 3, 4, 5]);
        await idbPut('__test_key', testData);
        const retrieved = await idbGet('__test_key');
        assert(retrieved instanceof Uint8Array || retrieved instanceof ArrayBuffer, 'Should return typed data');
        const arr = new Uint8Array(retrieved);
        assertEqual(arr.length, 5, 'Length');
        assertEqual(arr[0], 1, 'First byte');
        assertEqual(arr[4], 5, 'Last byte');
        await idbDelete('__test_key');
    });

    await test('idb_get_missing_key_returns_undefined', async () => {
        const val = await idbGet('__nonexistent_key_' + Date.now());
        assertEqual(val, undefined, 'Missing key');
    });

    await test('idb_overwrite', async () => {
        await idbPut('__test_overwrite', new Uint8Array([10]));
        await idbPut('__test_overwrite', new Uint8Array([20]));
        const arr = new Uint8Array(await idbGet('__test_overwrite'));
        assertEqual(arr[0], 20, 'Should be overwritten');
        await idbDelete('__test_overwrite');
    });

    await test('idb_delete', async () => {
        await idbPut('__test_delete', new Uint8Array([1]));
        await idbDelete('__test_delete');
        const val = await idbGet('__test_delete');
        assertEqual(val, undefined, 'Should be deleted');
    });

    await test('idb_large_blob', async () => {
        const size = 65_616; // ~64 KB (size of a height-10 MSS tree)
        const blob = new Uint8Array(size);
        blob[0] = 0xAA;
        blob[size - 1] = 0xBB;
        await idbPut('__test_large', blob);
        const retrieved = new Uint8Array(await idbGet('__test_large'));
        assertEqual(retrieved.length, size, 'Size');
        assertEqual(retrieved[0], 0xAA, 'First byte');
        assertEqual(retrieved[size - 1], 0xBB, 'Last byte');
        await idbDelete('__test_large');
    });

    // ── normalizeHex ────────────────────────────────────────────────────

    await test('normalizeHex_string', async () => {
        assertEqual(normalizeHex('AABB'), 'aabb', 'Lowercase');
    });

    await test('normalizeHex_uint8array', async () => {
        assertEqual(normalizeHex(new Uint8Array([0x0A, 0xFF])), '0aff', 'Uint8Array');
    });

    await test('normalizeHex_null', async () => {
        assertEqual(normalizeHex(null), '', 'Null');
        assertEqual(normalizeHex(undefined), '', 'Undefined');
    });

    await test('normalizeHex_array', async () => {
        assertEqual(normalizeHex([0, 255]), '00ff', 'Array');
    });

    // ── Dashboard ───────────────────────────────────────────────────────

    await test('buildDashboardPayload_empty', async () => {
        const saved = { ...wState };
        wState = { phrase: null, nextWotsIndex: 0, nextMssIndex: 0, wotsAddrs: {}, spentWots: {}, pendingSpends: {}, mssAddrs: {}, utxos: {}, history: [], lastScannedHeight: 0 };
        pendingSends = [];
        const p = buildDashboardPayload();
        assertEqual(p.primaryAddress, 'None', 'No MSS = None');
        assertEqual(p.balance, 0, 'Zero balance');
        assertEqual(p.utxos.length, 0, 'No UTXOs');
        assertEqual(p.history.length, 0, 'No history');
        wState = saved;
    });

    await test('buildDashboardPayload_with_pending_deduction', async () => {
        const saved = { ...wState };
        const savedPending = [...pendingSends];
        wState = { phrase: null, nextWotsIndex: 0, nextMssIndex: 0, wotsAddrs: {}, mssAddrs: {}, utxos: { 'abc': { value: 100, coin_id: 'abc' } }, history: [], lastScannedHeight: 0 };
        pendingSends = [{ kind: 'pending', value: 30, fee: 5, timestamp: 0, inputs: [], outputs: [] }];
        const p = buildDashboardPayload();
        assertEqual(p.balance, 65, 'Balance should deduct pending (100 - 30 - 5)');
        wState = saved;
        pendingSends = savedPending;
    });

    // ── addUtxo ─────────────────────────────────────────────────────────

    await test('addUtxo_deduplicates', async () => {
        const saved = JSON.parse(JSON.stringify(wState));
        wState.wotsAddrs = { 'aabbcc': 0 };
        wState.utxos = {};
        wState.nextWotsIndex = 1;

        const added1 = addUtxo('aabbcc', 8, 'salt1', 'coin1');
        assert(added1, 'First add should return true');

        const added2 = addUtxo('aabbcc', 8, 'salt1', 'coin1');
        assert(!added2, 'Duplicate should return false');

        assertEqual(Object.keys(wState.utxos).length, 1, 'Should have 1 UTXO');
        wState = JSON.parse(JSON.stringify(saved));
    });

    // ── Q-Bolt v2: wire codecs ──────────────────────────────────────────

    await test('qbolt_open_codec_roundtrip', async () => {
        const funding = [
            { value: 4096, salt: 'aa'.repeat(32) },
            { value: 512,  salt: 'bb'.repeat(32) },
        ];
        const sig0 = 'cd'.repeat(936);
        const packed = qbPackOpen(123456, funding, sig0);
        const out = qbUnpackOpen(qbHex(packed));
        assert(out, 'unpack returned null');
        assertEqual(out.expiry, 123456, 'expiry');
        assertEqual(out.funding.length, 2, 'funding count');
        assertEqual(out.funding[0].value, 4096, 'coin 0 value');
        assertEqual(out.funding[1].salt, 'bb'.repeat(32), 'coin 1 salt');
        assertEqual(out.sig0, sig0, 'sig0');
    });

    await test('qbolt_state_codec_roundtrip', async () => {
        const st = {
            nonce: 7, sender_amt: 900, receiver_amt: 1100,
            htlcs: [
                { amount: 250, timeout: 900100, secret_hash: '11'.repeat(32) },
                { amount: 50,  timeout: 900200, secret_hash: '22'.repeat(32) },
            ],
        };
        const sig = 'ef'.repeat(936);
        const out = qbUnpackState(qbHex(qbPackState(st, sig)));
        assert(out, 'unpack returned null');
        assertEqual(out.nonce, 7, 'nonce');
        assertEqual(out.sender_amt, 900, 'sender_amt');
        assertEqual(out.receiver_amt, 1100, 'receiver_amt');
        assertEqual(out.htlcs.length, 2, 'htlc count');
        assertEqual(out.htlcs[0].timeout, 900100, 'htlc 0 timeout');
        assertEqual(out.htlcs[1].secret_hash, '22'.repeat(32), 'htlc 1 hash');
        assertEqual(out.sig, sig, 'sig');
    });

    await test('qbolt_state_codec_rejects_wrong_version', async () => {
        const bin = qbPackState({ nonce: 1, sender_amt: 1, receiver_amt: 1, htlcs: [] }, 'aa');
        bin[0] = 9;                       // wrong protocol version byte
        assertEqual(qbUnpackState(qbHex(bin)), null, 'must reject foreign version');
        assertEqual(qbUnpackState('00'), null, 'must reject a truncated frame');
    });

    await test('qbolt_u32_codec_roundtrip', async () => {
        const out = qbUnpackU32(qbHex(qbPackU32(0, 4294967295, new Uint8Array([1, 2, 3]))));
        assert(out, 'unpack returned null');
        assertEqual(out.n, 4294967295, 'u32 max round-trips');
        assertEqual(out.extra.length, 3, 'extra length');
        assertEqual(out.extra[2], 3, 'extra content');
    });

    // ── Q-Bolt v2: the receiver's validation gauntlet ────────────────────
    // Exercises every rejection branch that does NOT need a real signature,
    // plus the ordering guarantee that a stale nonce is refused before any
    // expensive rebuild happens.

    await test('qbolt_validate_rejects_stale_and_inactive', async () => {
        const chan = {
            v: 2, id: 'ff'.repeat(32), role: 'receiver', status: 'active',
            sender_pk: 'a'.repeat(64), receiver_pk: 'b'.repeat(64),
            expiry: 1000, capacity: 10000, funding: [{ value: 8192, salt: '00'.repeat(32), coin_id: 'ff'.repeat(32) }],
            latest: { nonce: 5, sender_amt: 5000, receiver_amt: 3000, htlcs: [], sender_sig: 'aa' },
        };
        const mk = (over) => Object.assign({ nonce: 6, sender_amt: 4000, receiver_amt: 4000, htlcs: [], sig: 'aa' }, over);

        let v = qbValidateInbound(chan, mk({ nonce: 5 }), 500);
        assert(!v.ok && v.code === 6, `stale nonce must be code 6 (got ${v.code})`);

        v = qbValidateInbound(chan, mk({ nonce: 4 }), 500);
        assert(!v.ok && v.code === 6, 'older nonce must be code 6');

        // Exact re-delivery of the CURRENT state is idempotent, not a rejection.
        v = qbValidateInbound(chan, { nonce: 5, sender_amt: 5000, receiver_amt: 3000, htlcs: [], sig: 'aa' }, 500);
        assert(v.ok && v.idempotent, 're-delivery of the current state must be idempotent');

        const closing = { ...chan, status: 'closing' };
        v = qbValidateInbound(closing, mk({}), 500);
        assert(!v.ok && v.code === 7, 'inactive channel must be code 7');

        const asSender = { ...chan, role: 'sender' };
        v = qbValidateInbound(asSender, mk({}), 500);
        assert(!v.ok && v.code === 8, 'sender must not accept inbound states');

        const many = mk({ htlcs: new Array(QB.MAX_HTLCS + 1).fill(0).map((_, i) => ({ amount: 1, timeout: 900, secret_hash: String(i).padStart(64, '0') })) });
        v = qbValidateInbound(chan, many, 500);
        assert(!v.ok && v.code === 5, 'HTLC count cap must be code 5');
    });


    const failed = results.filter(r => !r.ok).length;
    return { passed, failed, results };
}
