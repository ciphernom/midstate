// reorg.js — Reuse-safe chain-reorganisation handling for the Midstate SDK.
//
// ═══════════════════════════════════════════════════════════════════════════
//  Notation (per the Midstate Coding & Documentation Standard)
// ═══════════════════════════════════════════════════════════════════════════
//
//   Hash          ≜ BLAKE3 output ([u8; 32]), carried here as 64-char hex
//   Height        ≜ ℕ
//   ℙ T           ≜ power set of T
//   A ⇸ B         ≜ partial function (a JS object used as a map)
//   dom f, ran f  ≜ domain / range of f
//   #s            ≜ cardinality
//   x'            ≜ value of x AFTER the operation
//   x?            ≜ input parameter, y! ≜ output
//   𝔹             ≜ {true, false}
//   ⊕             ≜ functional override (map update)
//   ◁, ⩤          ≜ domain restriction / domain anti-restriction
//
// State space shared by every schema in this module:
//
//   Coin ≜ (coin_id : Hash, address : Hash, value : ℕ, salt : Hash,
//           is_mss : 𝔹, created_at : Height, reuse_locked : 𝔹)
//
//   ContractCoin ≜ (coin_id : Hash, address : Hash, value : ℕ, salt : Hash,
//                   state : Hash ∪ {null}, created_at : Height)
//     — script-locked, held at a CONTRACT address. `state ≠ null` marks a state
//       thread, which consensus requires to have value 0.
//
//     WalletState
//     ─────────────────────────────────────────────────────────────
//     utxos        : ℙ Coin
//     spent_log    : Height ⇸ ℙ Coin
//     block_hashes : Height ⇸ Hash
//     spent_wots   : Hash ⇸ (height : Height, commitment : Hash ∪ {null})
//     sent_reveals : Hash ⇸ (commitment : Hash, payload : String,
//                            coin_ids : ℙ Hash, sent_at : Height)
//     reuse_locked : Hash ⇸ (address : Hash, prior_commitment : Hash ∪ {null})
//     contract_coins : Hash ⇸ ContractCoin
//     scanned      : Height
//
//     INV-1  ∀ c ∈ utxos • ¬c.is_mss ∧ c.address ∈ dom spent_wots
//                        ⇒ c.reuse_locked
//            (a coin at a published one-time key is never freely spendable)
//     INV-2  dom reuse_locked ⊆ { c.coin_id | c ∈ utxos }
//     INV-3  ∀ h ∈ dom spent_log • h ≤ scanned
//     INV-4  #(dom block_hashes) ≤ REORG_DEPTH + RETAIN_SLACK
//     INV-5  dom contract_coins ∩ { c.coin_id | c ∈ utxos } = ∅
//            (a script-locked coin is never in the wallet-key spend set)
//     INV-6  ∀ c ∈ ran contract_coins • c.state ≠ null ⇒ c.value = 0
//
// ═══════════════════════════════════════════════════════════════════════════
//  Reasoning (module level)
// ═══════════════════════════════════════════════════════════════════════════
//
// Before this module the wallet applied blocks forward-only and irreversibly:
// a spend deleted its input coins, purged WOTS siblings, and set
// spent_wots[addr] permanently. An orphaned block therefore
//
//   (a) left phantom credits for outputs that only ever existed on the dead
//       branch, and — far worse —
//   (b) stranded real coins. A spend that got reorged out had already had its
//       inputs deleted and its address poisoned, and addUtxo refuses a poisoned
//       address forever, so not even a full rescan could re-credit the coin.
//
// The naive repair — clear spent_wots on rollback and restore the coin — is
// UNSAFE. A WOTS key signs exactly one message. When a spend is orphaned its
// signature, over that transaction's commitment, is already public and may
// still confirm on the winning branch or be replayed by anyone. If the address
// were simply un-poisoned and the restored coin spent into ANY different
// transaction, the wallet would produce a SECOND signature over a DIFFERENT
// commitment from the same key. That is textbook one-time-key reuse: it leaks
// the private key, and the node ships a Key Reuse Punishment Burn Protocol
// that lets any observer burn the funds.
//
// So a restored coin must be spendable ONLY by re-sending the identical
// original signature (same commitment), never by re-signing. Every mutation is
// height-tagged; on reorg we roll back to the fork and resolve each restored
// spend three ways:
//
//   CASE 1  The verbatim signed reveal is still held (within COMMITMENT_TTL):
//           clear the poison and re-arm the pending transaction so the SAME
//           signature is re-broadcast. Reuse-safe and automatic.
//   CASE 2  The reveal is no longer held but the address is poisoned: restore
//           the coin REUSE-LOCKED. It counts toward a separate locked balance,
//           is never offered to coin selection, and is surfaced to the caller.
//   CASE 3  MSS inputs: leaf counters are reconciled against the node and each
//           leaf is distinct, so there is no per-key reuse cliff. Restored
//           normally.
//
// ═══════════════════════════════════════════════════════════════════════════
//  Fork detection against this node
// ═══════════════════════════════════════════════════════════════════════════
//
// `LightNotification::NewBlockTip` carries height, target, filter_hex,
// block_hash and element_count — but NO parent hash. Parent-linkage detection
// is therefore unavailable to a light client on protocol /midstate/light/2.0.0,
// and any check written against `prev_hash` is dead code against this node.
//
// What IS available is a per-height block identity from two independent
// sources that agree: `get_filters` returns `block_hashes[i]` =
// `batch.extension.final_hash`, and `get_block(h)` returns the batch whose
// `extension.final_hash` is the same value. Detection therefore keys on hash
// CONFLICT at a height we have already recorded, confirmed by descending until
// a recorded hash still matches the node's.
//
// `get_filters` is used for the descent rather than `get_block`: it is a far
// smaller response, and a single-height range costs one request either way.
// The descent is bounded by REORG_DEPTH so a pathological peer cannot walk us
// to genesis one request at a time.

/**
 * Depth of hash history retained, and the maximum reorg this module can heal
 * precisely. Deeper divergence falls back to a rescan from the floor.
 */
export const REORG_DEPTH = 200;

/** Slack above REORG_DEPTH before pruning runs, to avoid pruning every block. */
const RETAIN_SLACK = 8;

/**
 * Blocks a commitment survives in chain state (`COMMITMENT_TTL` in
 * core/types.rs). A retained reveal older than this can no longer be revealed,
 * so it is pruned and its address falls to CASE 2.
 */
export const COMMITMENT_TTL = 1000;

const normalizeHex = (h) => (typeof h === 'string' ? h.replace(/^0x/, '').toLowerCase() : '');

// ═══════════════════════════════════════════════════════════════════════════

/**
 * Record the canonical block identity observed at a height.
 *
 * # Reasoning
 *
 * Reorg detection is only possible if the wallet remembers what it believed the
 * chain looked like. This is the sole writer of that memory. It is deliberately
 * cheap and total: it is called for every scanned height, including heights
 * with no wallet-relevant activity, because a fork can occur at a block that
 * paid us nothing and we still need to notice it.
 *
 * Heights whose hash the node could not supply (the undecidable placeholder,
 * an empty string) are skipped rather than stored as a sentinel. Storing a
 * sentinel would make the descent in {@link findForkHeight} treat that height
 * as a confirmed match and stop early at a block we never actually verified.
 *
 * # Formal Specification
 *
 * ```text
 * Pre:
 *   - height? ∈ Height
 *
 * Post:
 *   hash? ≠ ""  ⇒  block_hashes' = block_hashes ⊕ {height? ↦ hash?}
 *   hash? = ""  ⇒  block_hashes' = block_hashes
 *   utxos' = utxos ∧ spent_log' = spent_log ∧ spent_wots' = spent_wots
 * ```
 *
 * ```zed
 *     RecordBlockHash
 *     ─────────────────────────────
 *     ΔWalletState
 *     height? : Height
 *     hash?   : Hash ∪ {""}
 *
 *     post hash? ≠ "" ⇒ block_hashes' = block_hashes ⊕ {height? ↦ hash?}
 *     post hash? = "" ⇒ block_hashes' = block_hashes
 *     post utxos' = utxos
 *     post spent_log' = spent_log
 *     post spent_wots' = spent_wots
 *     post scanned' = scanned
 * ```
 *
 * # Safety / Invariants
 *
 * - Maintains INV-4 by delegating to {@link pruneHistory}, which the caller
 *   invokes once per batch rather than once per height.
 * - Never widens spendability: touches no coin state.
 *
 * @param {Object} w      Wallet instance.
 * @param {number} height Block height.
 * @param {string} hash   `extension.final_hash`, or "" if unavailable.
 */
export function recordBlockHash(w, height, hash) {
    const h = normalizeHex(hash);
    if (!h) return;
    w.blockHashes[height] = h;
}

/**
 * Drop history that can no longer participate in a heal.
 *
 * # Reasoning
 *
 * Three structures grow without bound as the chain advances, and each has a
 * different, principled cutoff:
 *
 *  - `block_hashes` and `spent_log` are bounded by REORG_DEPTH. Below that
 *    depth a reorg is not healable precisely anyway, so retaining the data
 *    buys nothing; a spend that deep is final.
 *  - `sent_reveals` is bounded by COMMITMENT_TTL instead, because its purpose
 *    is different: it exists so CASE 1 can re-broadcast a verbatim signature.
 *    Once the commitment has expired from chain state that reveal is dead —
 *    the node would reject it — so retaining it would only mislead the CASE 1
 *    test into promising a re-send that cannot succeed.
 *
 * Pruning `sent_reveals` on the REORG_DEPTH clock instead would be a
 * correctness bug in the unsafe direction: it would discard signatures that are
 * still re-sendable and demote recoverable coins to reuse-locked.
 *
 * # Formal Specification
 *
 * ```text
 * Pre:
 *   - tip? ∈ Height   (current network height)
 *
 * Post:
 *   let cutoff = max(dom block_hashes) − REORG_DEPTH
 *   #(dom block_hashes) > REORG_DEPTH + RETAIN_SLACK ⇒
 *     block_hashes' = { h ↦ v ∈ block_hashes | h ≥ cutoff }
 *     spent_log'    = { h ↦ v ∈ spent_log    | h ≥ cutoff }
 *   sent_reveals' = { a ↦ r ∈ sent_reveals | tip? − r.sent_at ≤ COMMITMENT_TTL }
 *   utxos' = utxos
 * ```
 *
 * ```zed
 *     PruneHistory
 *     ─────────────────────────────
 *     ΔWalletState
 *     tip? : Height
 *
 *     pre  tip? ≥ 0
 *
 *     post ∀ h ∈ dom block_hashes' • h ≥ max(dom block_hashes) − REORG_DEPTH
 *     post dom spent_log' ⊆ dom spent_log
 *     post ∀ h ∈ dom spent_log' • h ≥ max(dom block_hashes) − REORG_DEPTH
 *     post ∀ a ∈ dom sent_reveals' • tip? − sent_reveals'(a).sent_at ≤ COMMITMENT_TTL
 *     post utxos' = utxos
 *     post reuse_locked' = reuse_locked
 * ```
 *
 * # Safety / Invariants
 *
 * - Establishes INV-4.
 * - Does NOT prune `spent_wots`. A published one-time key stays published
 *   forever; forgetting it would re-open the address to admission by addUtxo
 *   and hand a reuse opportunity back to coin selection.
 *
 * @param {Object} w    Wallet instance.
 * @param {number} tip  Current network height.
 */
export function pruneHistory(w, tip) {
    const heights = Object.keys(w.blockHashes).map(Number);
    if (heights.length > REORG_DEPTH + RETAIN_SLACK) {
        const cutoff = Math.max(...heights) - REORG_DEPTH;
        for (const h of heights) if (h < cutoff) delete w.blockHashes[h];
        for (const h of Object.keys(w.spentLog).map(Number)) {
            if (h < cutoff) delete w.spentLog[h];   // buried spends are final
        }
    }
    for (const addr of Object.keys(w.sentReveals)) {
        const r = w.sentReveals[addr];
        if (r && typeof r.sentAtHeight === 'number' && tip - r.sentAtHeight > COMMITMENT_TTL) {
            delete w.sentReveals[addr];
        }
    }
}

/**
 * Locate the highest height at which our recorded chain still agrees with the
 * node's.
 *
 * # Reasoning
 *
 * A reorg is only actionable if we can say precisely where the branches
 * diverge: rolling back too far needlessly discards confirmed state, and
 * rolling back too little leaves dead-branch coins credited. This descends from
 * the tip comparing recorded hashes against the node's until one matches.
 *
 * The descent is bounded by REORG_DEPTH. Without a bound a peer that answers
 * every probe with a mismatch would walk the client to genesis one request per
 * height — a self-inflicted rate-limit violation driven entirely by a remote
 * party, which is precisely the amplification shape the node's own rate-limiter
 * comments warn about. On exhaustion we return the floor and let the caller
 * rescan that window; correctness is preserved because a rescan re-derives
 * everything above the floor from the winning branch.
 *
 * Heights we never recorded are skipped, not treated as mismatches: absence of
 * evidence is not divergence.
 *
 * # Formal Specification
 *
 * ```text
 * Pre:
 *   - tip? ≤ scanned
 *   - client? answers getFilters(h, h+1) with the canonical hash at h, or ""
 *
 * Post:
 *   floor = max(0, tip? − REORG_DEPTH)
 *   fork! = max { h ∈ [floor, tip?] | h ∈ dom block_hashes
 *                                   ∧ node_hash(h) = block_hashes(h) }
 *   if that set is empty then fork! = floor ∧ exhausted! = true
 *   0 ≤ fork! ≤ tip?
 *   state unchanged (this is a query, not a transition)
 * ```
 *
 * ```zed
 *     FindForkHeight
 *     ─────────────────────────────
 *     ΞWalletState
 *     tip?       : Height
 *     fork!      : Height
 *     exhausted! : 𝔹
 *
 *     pre  tip? ≤ scanned
 *
 *     post floor = max(0, tip? − REORG_DEPTH)
 *     post floor ≤ fork! ≤ tip?
 *     post ¬exhausted! ⇒ block_hashes(fork!) = node_hash(fork!)
 *     post exhausted!  ⇒ fork! = floor
 * ```
 *
 * # Safety / Invariants
 *
 * - Ξ (no state change): this function must never mutate wallet state, because
 *   it runs before the caller has decided whether a rollback is warranted.
 * - Bounded request count: at most REORG_DEPTH + 1 getFilters calls.
 *
 * @param {Object} w      Wallet instance.
 * @param {Object} client Connected MidstateClient.
 * @param {number} tip    Height to start descending from.
 * @returns {Promise<{fork:number, exhausted:boolean, probes:number}>}
 */
export async function findForkHeight(w, client, tip) {
    const floor = Math.max(0, tip - REORG_DEPTH);
    let probes = 0;

    for (let h = tip; h >= floor; h--) {
        const recorded = w.blockHashes[h];
        if (!recorded) continue;               // never observed — not evidence of divergence
        let nodeHash = null;
        try {
            const fd = await client.getFilters(h, h + 1);
            probes++;
            const raw = fd?.block_hashes?.[0];
            nodeHash = raw ? normalizeHex(raw) : null;
        } catch {
            continue;                          // transient failure — keep descending
        }
        if (nodeHash && nodeHash === recorded) {
            return { fork: h, exhausted: false, probes };
        }
    }
    return { fork: floor, exhausted: true, probes };
}

/**
 * Roll wallet state back to `forkHeight` — whose own effects are RETAINED —
 * undoing every mutation above it, restoring orphaned spends reuse-safely.
 *
 * # Reasoning
 *
 * This is the heart of the module and the only place where a coin at a
 * published one-time key can be returned to the UTXO set. See the module
 * header for why the obvious implementation leaks keys.
 *
 * The ordering below is not incidental. Dead-branch credits are removed BEFORE
 * spends are restored, so that a coin which was both created and spent above
 * the fork is not resurrected: it never existed on the winning branch. The
 * `created_at ≤ forkHeight` guard in step 2 enforces the same thing from the
 * other side.
 *
 * Classification in step 3 is per ADDRESS, not per coin, because the reuse
 * cliff is a property of the key: if one coin at an address is re-sendable they
 * all are, and if the signature is gone they are all locked together.
 *
 * # Formal Specification
 *
 * ```text
 * Pre:
 *   - 0 ≤ fork? ≤ scanned
 *   - tip? ≥ fork?
 *
 * Post (all simultaneous):
 *   -- 1. dead-branch credits removed
 *   utxos₁ = { c ∈ utxos | c.created_at ≤ fork? }
 *
 *   contract_coins₁ = { id ↦ c ∈ contract_coins | c.created_at ≤ fork? }
 *
 *   -- 2. orphaned spends restored, partitioned by lock type
 *   R = { c ∈ ran(spent_log ▷ heights > fork?) | c.created_at ≤ fork?
 *                                              ∧ c.coin_id ∉ live_ids }
 *   R_w = { c ∈ R | ¬c.is_contract }      R_c = { c ∈ R | c.is_contract }
 *   utxos'          = utxos₁ ∪ R_w
 *   contract_coins' = contract_coins₁ ∪ R_c
 *   spent_log' = { h ↦ v ∈ spent_log | h ≤ fork? }
 *
 *   -- 3. per-address reuse resolution, for each a ∈ addresses(R)
 *   --    with spent_wots(a).height > fork?
 *   CASE 1  a ∈ dom sent_reveals ∧ fresh(sent_reveals(a))
 *             spent_wots' = {a} ⩤ spent_wots
 *             pending'    = sent_reveals(a)          (verbatim re-send armed)
 *   CASE 2  otherwise
 *             spent_wots'(a) = spent_wots(a) ⊕ {reuse_locked ↦ true}
 *             ∀ c ∈ R with c.address = a • c.reuse_locked' = true
 *             reuse_locked' = reuse_locked ∪ { c.coin_id ↦ (a, prior) }
 *   CASE 3  a ∉ dom spent_wots  (MSS)  — no action; already restored by step 2
 *
 *   -- 4..6 bookkeeping
 *   block_hashes' = { h ↦ v ∈ block_hashes | h ≤ fork? }
 *   scanned'      = min(scanned, fork?)
 *
 * Post (error): this operation does not fail; it is total over its precondition.
 * ```
 *
 * ```zed
 *     RollbackTo
 *     ─────────────────────────────
 *     ΔWalletState
 *     fork? : Height
 *     tip?  : Height
 *     restored!, removed!, resendable!, locked! : ℕ
 *
 *     pre  0 ≤ fork? ≤ scanned
 *     pre  tip? ≥ fork?
 *
 *     post ∀ c ∈ utxos' • c.created_at ≤ fork?
 *     post ∀ c ∈ ran contract_coins' • c.created_at ≤ fork?
 *     post dom contract_coins' ∩ { c.coin_id | c ∈ utxos' } = ∅
 *     post dom spent_log' = { h ∈ dom spent_log | h ≤ fork? }
 *     post dom block_hashes' = { h ∈ dom block_hashes | h ≤ fork? }
 *     post scanned' = min(scanned, fork?)
 *     post ∀ c ∈ utxos' • ¬c.is_mss ∧ c.address ∈ dom spent_wots'
 *                       ⇒ c.reuse_locked
 *     post removed! + restored! ≥ 0
 * ```
 *
 * # Safety / Invariants
 *
 * - **Re-establishes INV-1.** This is the critical obligation. Every restored
 *   WOTS coin leaves this function either (CASE 1) at an address whose poison
 *   was cleared because the only way it can be spent is a verbatim re-send of
 *   the existing signature, or (CASE 2) flagged `reuse_locked`. There is no
 *   third outcome for a poisoned address.
 * - **Establishes INV-2**: every id added to `reuse_locked` is added to `utxos`
 *   in the same step.
 * - **Maintains INV-5**: restored coins are routed by `is_contract`, so a
 *   script-locked coin can never land in `utxos` and be offered to
 *   `prepare_spend`, and a wallet coin can never land in `contract_coins`.
 * - **State-thread convergence.** Chained advances resolve correctly without
 *   special handling: for S₀→S₁ at h₁ and S₁→S₂ at h₂ (both above the fork),
 *   step 1 discards S₁ and S₂ as dead-branch, and step 2 restores only S₀
 *   because S₁'s `created_at` is itself above the fork. The wallet is left
 *   holding exactly the state the winning branch has.
 * - **Never clears `spent_wots` without a held signature.** CASE 1 is gated on
 *   possessing the reveal AND its commitment matching AND it still being within
 *   COMMITMENT_TTL. Weakening any of the three re-opens key reuse.
 * - `scanned` is rewound so the caller re-applies the winning branch.
 *
 * @param {Object} w     Wallet instance.
 * @param {number} fork  Fork height (its effects are kept).
 * @param {number} tip   Current network height, for reveal freshness.
 * @returns {{restored:number, removed:number, resendable:number, locked:number, alerts:Object[]}}
 */
export function rollbackTo(w, fork, tip) {
    let restored = 0, removed = 0, resendable = 0, locked = 0, restoredContract = 0;
    const alerts = [];

    // 1. Remove coins credited above the fork — they exist only on the dead
    //    branch. Done first so step 2 cannot resurrect a coin that was both
    //    created and spent above the fork.
    const survivors = [];
    for (const c of w.utxos) {
        if ((c.createdAtHeight || 0) > fork) { removed++; continue; }
        survivors.push(c);
    }
    w.utxos = survivors;

    // Contract coins are height-tagged the same way and die on the same branch.
    // A state thread advanced above the fork must be discarded along with the
    // execution that advanced it, or the wallet keeps a state the winning chain
    // never had and every subsequent execution is rejected as stale.
    for (const [id, c] of Object.entries(w.contractCoins)) {
        if ((c.createdAtHeight || 0) > fork) { delete w.contractCoins[id]; removed++; }
    }

    const liveIds = new Set([
        ...w.utxos.map((c) => c.coin_id),
        ...Object.keys(w.contractCoins),
    ]);

    // 2. Restore coins SPENT above the fork, grouped by address so each key is
    //    classified once.
    const restoredByAddr = new Map();
    for (const hStr of Object.keys(w.spentLog)) {
        const h = Number(hStr);
        if (h <= fork) continue;
        for (const coin of w.spentLog[h]) {
            // A coin created above the fork is dead-branch, not ours to revive.
            if ((coin.createdAtHeight || 0) > fork) continue;
            if (liveIds.has(coin.coin_id)) continue;
            const revived = { ...coin, value: BigInt(coin.value) };

            // Contract coins go back to `contractCoins`, never to `utxos`, and
            // never enter the per-address reuse classification below: they are
            // script-locked, so no one-time key was published by spending them
            // and there is nothing to lock. Restoring one to `utxos` would offer
            // a script-locked coin to prepare_spend.
            if (revived.isContract) {
                w.contractCoins[revived.coin_id] = revived;
                liveIds.add(revived.coin_id);
                restoredContract++;
                restored++;
                continue;
            }

            w.utxos.push(revived);
            liveIds.add(revived.coin_id);
            if (!restoredByAddr.has(revived.address)) restoredByAddr.set(revived.address, []);
            restoredByAddr.get(revived.address).push(revived);
            restored++;
        }
        delete w.spentLog[h];
    }

    // 3. Resolve each address that had an orphaned spend.
    for (const [addr, coins] of restoredByAddr) {
        const poison = w.spentWots[addr];
        const poisonedAboveFork =
            poison && typeof poison.height === 'number' && poison.height > fork;

        // CASE 3 — MSS addresses are never in spentWots. Each spend consumes a
        // distinct leaf and the counter is reconciled with the node, so there is
        // no reuse cliff. Already restored by step 2; nothing further to do.
        if (!poisonedAboveFork) continue;

        const reveal = w.sentReveals[addr];
        const freshEnough = reveal && (tip - (reveal.sentAtHeight || 0) <= COMMITMENT_TTL);
        const commitmentMatches =
            reveal && (!poison.commitment || reveal.commitment === poison.commitment);

        if (reveal && freshEnough && commitmentMatches) {
            // CASE 1 — re-sendable. Clearing the poison is safe ONLY because the
            // coin can now leave exclusively via the identical signature we
            // still hold; no new signature is ever produced over a different
            // commitment.
            delete w.spentWots[addr];
            for (const c of coins) delete c.reuseLocked;
            w.pendingResend = w.pendingResend || {
                commitment: reveal.commitment,
                revealPayload: reveal.revealPayload,
                inputCoinIds: reveal.inputCoinIds,
                armedAtHeight: tip,
            };
            resendable++;
            alerts.push({
                kind: 'reorg_resend',
                address: addr,
                commitment: reveal.commitment,
                message:
                    `Reorg: re-sending the original signed transaction for ${addr.slice(0, 8)}… ` +
                    `(same signature; no key reuse).`,
            });
        } else {
            // CASE 2 — the signature is gone and the key is published. Restore
            // the coins REUSE-LOCKED: excluded from spendable balance and from
            // coin selection. The poison is demoted, not cleared, so addUtxo
            // still refuses arbitrary new dust at this address.
            for (const c of coins) {
                c.reuseLocked = true;
                w.reuseLocked[c.coin_id] = {
                    address: addr,
                    priorCommitment: poison.commitment || null,
                };
            }
            w.spentWots[addr] = { ...poison, reuseLocked: true };
            locked++;
            alerts.push({
                kind: 'reuse_lock',
                address: addr,
                coinIds: coins.map((c) => c.coin_id),
                priorCommitment: poison.commitment || null,
                message:
                    `A confirmed transaction was reversed by a chain reorganization, but its ` +
                    `signature is no longer held by this wallet. The coin(s) at ${addr.slice(0, 10)}… ` +
                    `are back, but this one-time key already signed once, so they are locked to ` +
                    `prevent key reuse. Do not attempt to move them manually.`,
            });
        }
    }

    // 4. Drop per-height block hashes above the fork.
    for (const hStr of Object.keys(w.blockHashes)) {
        if (Number(hStr) > fork) delete w.blockHashes[hStr];
    }

    // 5. Rewind the scan marker so the winning branch is re-applied.
    w.lastScannedHeight = Math.min(w.lastScannedHeight, fork);

    return { restored, removed, resendable, locked, restoredContract, alerts };
}

/**
 * Decide whether the chain has diverged from what the wallet recorded and, if
 * so, roll back to the fork.
 *
 * # Reasoning
 *
 * Detection runs before each sync extends the chain, and on demand when a push
 * notification arrives. The trigger is a hash CONFLICT at a height we recorded:
 * the node reports a different `extension.final_hash` than we stored. As noted
 * in the module header, parent-linkage detection is impossible here because
 * `NewBlockTip` carries no parent hash, so conflict is the only signal that
 * this protocol actually affords.
 *
 * A wallet with no recorded history cannot detect anything and must not guess —
 * it returns "no reorg" rather than rolling back to zero, which would be a
 * spurious full rescan on every fresh wallet.
 *
 * # Formal Specification
 *
 * ```text
 * Pre:
 *   - client? is connected
 *
 * Post:
 *   probe = max(dom block_hashes) ∩ [0, scanned]
 *   probe undefined  ⇒  result! = ⊥ ∧ state unchanged
 *   node_hash(probe) = block_hashes(probe)  ⇒  result! = ⊥ ∧ state unchanged
 *   node_hash(probe) ≠ block_hashes(probe)  ⇒
 *     result! = RollbackTo(FindForkHeight(probe).fork)
 * ```
 *
 * ```zed
 *     MaybeHandleReorg
 *     ─────────────────────────────
 *     ΔWalletState
 *     client? : Client
 *     tip?    : Height
 *     result! : RollbackReport ∪ {⊥}
 *
 *     post result! = ⊥ ⇒ θWalletState' = θWalletState
 *     post result! ≠ ⊥ ⇒ scanned' ≤ scanned
 *     post result! ≠ ⊥ ⇒ ∀ c ∈ utxos' • ¬c.is_mss ∧ c.address ∈ dom spent_wots'
 *                                     ⇒ c.reuse_locked
 * ```
 *
 * # Safety / Invariants
 *
 * - On the no-reorg path this is Ξ: nothing is written, so a routine sync pays
 *   one extra cheap request and no state churn.
 * - Preserves INV-1 by delegating every mutation to {@link rollbackTo}.
 *
 * @param {Object} w      Wallet instance.
 * @param {Object} client Connected MidstateClient.
 * @param {number} tip    Current network height.
 * @returns {Promise<Object|null>} Rollback report, or null if the chain agrees.
 */
export async function maybeHandleReorg(w, client, tip) {
    const recorded = Object.keys(w.blockHashes).map(Number).filter((h) => h <= w.lastScannedHeight);
    if (recorded.length === 0) return null;      // nothing to compare against

    // Probe from the highest recorded height downwards until one is DECIDABLE.
    //
    // A single probe at the tip is not enough. The node caps `end_height` at its
    // own height, so if it has reorged to a SHORTER chain the height we last
    // recorded is no longer servable and comes back as an undecidable
    // placeholder — indistinguishable, to a single-probe check, from a chain
    // that agrees. That is exactly the case where a rollback is most needed, so
    // giving up there would leave the wallet permanently ahead of the network.
    //
    // The walk is capped: this is a cheap conflict check, not the fork descent,
    // and an unreachable peer should cost a handful of requests rather than a
    // scan of our whole history.
    const PROBE_LIMIT = 8;
    const descending = recorded.sort((a, b) => b - a).slice(0, PROBE_LIMIT);

    let probe = null, nodeHash = null;
    for (const h of descending) {
        try {
            const fd = await client.getFilters(h, h + 1);
            const raw = fd?.block_hashes?.[0];
            const hash = raw ? normalizeHex(raw) : null;
            if (hash) { probe = h; nodeHash = hash; break; }
        } catch {
            return null;                         // cannot verify — do not guess
        }
    }
    if (probe === null || !nodeHash) return null;   // wholly undecidable
    if (nodeHash === w.blockHashes[probe]) return null;

    const { fork, exhausted, probes } = await findForkHeight(w, client, probe);
    const report = rollbackTo(w, fork, tip);
    report.fork = fork;
    report.exhausted = exhausted;
    report.probes = probes + 1;
    if (exhausted) {
        report.alerts.push({
            kind: 'deep_reorg',
            message:
                `Reorg deeper than ${REORG_DEPTH} blocks; rescanning from height ${fork}. ` +
                `Coins spent below that depth are treated as final.`,
        });
    }
    return report;
}
