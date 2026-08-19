import initWasm, {
    WebWallet,
    generate_phrase,
    mine_commitment_pow,
    compute_coin_id_hex,
    compute_commitment_hex,
    blake3_hash_hex,
} from '../pkg/wasm_wallet.js';
import { MemoryStorage } from './storage.js';
import {
    recordBlockHash, pruneHistory, maybeHandleReorg, rollbackTo,
    REORG_DEPTH, COMMITMENT_TTL,
} from './reorg.js';

/** Max inputs in a single Consolidate, from MAX_CONSOLIDATE_INPUTS in the node. */
const MAX_CONSOLIDATE_INPUTS = 8192;

// ── Hex helper ────────────────────────────────────────────────────────────────
// Block fields (addresses, salts) may arrive as hex strings OR as byte arrays
// depending on how the node serialized them, so normalize both to lowercase hex.
function normalizeHex(data) {
    if (!data) return '';
    if (typeof data === 'string') return data.toLowerCase();
    if (Array.isArray(data) || data instanceof Uint8Array) {
        return Array.from(data).map((b) => b.toString(16).padStart(2, '0')).join('').toLowerCase();
    }
    return '';
}

// Coin ID for a state-thread coin.
//
// ── On the name `Confidential` ──
//
// The enum variant is called `OutputData::Confidential` and the domain tag is
// the literal ASCII string "CONFIDENTIAL", but **nothing here is confidential**.
// The name is vestigial: an earlier design had zk-STARK confidential
// transactions, those were removed, and state threads were retconned into the
// same variant. The node's own source calls the name "a trap" and notes it has
// already caused one incorrect fix, with a regression test
// (`state_thread_output_contributes_no_value_to_the_fee`) replacing an older one
// written on the assumption that the value was hidden.
//
// So the correct model is:
//
//   value(Confidential{..}) = 0        — known to be zero, NOT unknown
//
// It carries a 32-byte data commitment that scripts read through
// `OP_READ_INPUT_STATE`. It carries data, not money. A covenant spending 100
// into `[Standard 96, Confidential(state)]` really does pay a fee of 4 — the
// state thread does not absorb the missing 4 into a hidden amount.
//
// Treating the value as *hidden* rather than *zero* would be wrong in the
// opposite direction to the bug this module already fixed: it would make the
// wallet skip these coins from conservation instead of counting them as zero.
//
// `InputReveal::coin_id()` hashes that tag followed by address, commitment and
// salt — distinct from the standard (address, value, salt) form, because a state
// thread has no value to hash.
const STATE_THREAD_TAG_HEX = '434f4e464944454e5449414c';   // ASCII "CONFIDENTIAL"
function computeStateThreadCoinId(addrHex, commitmentHex, saltHex) {
    return blake3_hash_hex(STATE_THREAD_TAG_HEX + addrHex + commitmentHex + saltHex);
}

export class Wallet {
    static async init(wasmPathOrBuffer) {
        // Initializes the WASM runtime.
        // Passing it as an object fixes the wasm-bindgen deprecation warning.
        await initWasm({ module_or_path: wasmPathOrBuffer });
    }

    /**
     * Create a wallet, optionally from an existing recovery phrase.
     *
     * Without a phrase a fresh one is generated. Passing one is how you open an
     * existing wallet from seed — previously impossible through this entry
     * point, which meant any tool built on it could only ever look at a brand
     * new, empty wallet.
     *
     * @param {Object} [storageProvider]
     * @param {string} [phrase] Existing 24-word recovery phrase.
     */
    static async create(storageProvider = new MemoryStorage(), phrase = null) {
        const wallet = new Wallet(phrase || generate_phrase(), storageProvider);
        await wallet.save();
        return wallet;
    }

    static async restore(storageProvider) {
        const metadataStr = await storageProvider.loadMetadata();
        if (!metadataStr) throw new Error("No wallet metadata found in storage");

        const data = JSON.parse(metadataStr);
        const wallet = new Wallet(data.phrase, storageProvider);
        wallet.nextWotsIndex = data.nextWotsIndex;
        wallet.nextMssIndex = data.nextMssIndex;
        wallet.utxos = (data.utxos || []).map(u => ({ ...u, value: BigInt(u.value) }));
        wallet.wotsAddrs = data.wotsAddrs;
        wallet.mssAddrs = data.mssAddrs;
        wallet.lastScannedHeight = data.lastScannedHeight || 0;
        wallet.spentWots = data.spentWots || {};
        wallet.blockHashes = data.blockHashes || {};
        wallet.spentLog = Object.fromEntries(Object.entries(data.spentLog || {}).map(
            ([h, coins]) => [h, coins.map(c => ({ ...c, value: BigInt(c.value) }))]));
        wallet.sentReveals = data.sentReveals || {};
        wallet.reuseLocked = data.reuseLocked || {};
        wallet.pendingResend = data.pendingResend || null;
        wallet.fullScanCompleted = !!data.fullScanCompleted;
        wallet.watchedContracts = data.watchedContracts || {};
        wallet.contractCoins = Object.fromEntries(Object.entries(data.contractCoins || {}).map(
            ([id, c]) => [id, { ...c, value: BigInt(c.value) }]));

        // --- HEAL MISSING CHANGE ADDRESSES ---
        // If previous versions forgot to add change addresses to the watchlist,
        // this loop regenerates them and adds them back.
        for (let i = 0; i < wallet.nextWotsIndex; i++) {
            const addr = wallet.inner.get_wots_address(i);
            if (wallet.wotsAddrs[addr] === undefined) {
                wallet.wotsAddrs[addr] = i;
            }
        }

        for (const addr of Object.keys(wallet.mssAddrs)) {
            const treeBytes = await storageProvider.loadMssTree(addr);
            if (treeBytes) {
                wallet.inner.import_mss_bytes(addr, new Uint8Array(treeBytes));
                wallet.inner.set_mss_leaf_index(addr, wallet.mssAddrs[addr].next_leaf);
            }
        }
        return wallet;
    }

    constructor(phrase, storageProvider = new MemoryStorage()) {
        this.phrase = phrase;
        this.inner = new WebWallet(phrase);
        this.storage = storageProvider;
        this.nextWotsIndex = 0;
        this.nextMssIndex = 0;
        this.utxos = [];
        this.wotsAddrs = {};
        this.mssAddrs = {};
        // Highest block height fully scanned by sync(). Persisted so subsequent
        // syncs are incremental instead of re-scanning the whole chain.
        this.lastScannedHeight = 0;
        // WOTS addresses whose one-time key has been published by a spend.
        // Coins arriving at these addresses are unspendable without reusing the
        // key, so they are never admitted to the UTXO set.
        this.spentWots = {};

        // ── Reorg support (see reorg.js) ────────────────────────────────────
        // Every mutation below the tip is height-tagged so a rollback can undo
        // it precisely. Without this the wallet applies blocks irreversibly and
        // an orphaned spend strands its inputs permanently.
        /** Height ⇸ Hash — canonical `extension.final_hash` per scanned height. */
        this.blockHashes = {};
        /** Height ⇸ Coin[] — coins removed at that height, for restoration. */
        this.spentLog = {};
        /** Address ⇸ verbatim signed reveal, retained while re-sendable. */
        this.sentReveals = {};
        /** CoinId ⇸ lock record — restored coins pinned by one-time-key reuse. */
        this.reuseLocked = {};
        /** A verbatim re-send armed by a rollback, or null. */
        this.pendingResend = null;
        /** True once a scan has run from genesis; gates gap extension. */
        this.fullScanCompleted = false;

        // ── Contract / state-thread tracking ────────────────────────────────
        // State threads live at CONTRACT addresses, not wallet addresses, and
        // are consumed through prepare_script_spend's contract_inputs argument
        // rather than the wallet UTXO array. They are therefore held separately:
        // putting them in `utxos` would offer them to prepare_spend, which would
        // try to sign a script-locked coin with a wallet key.
        /** Address ⇸ metadata for contracts this wallet follows. */
        this.watchedContracts = {};
        /** CoinId ⇸ contract coin (state thread when `state` is non-null). */
        this.contractCoins = {};
    }

    /**
     * Persist the full wallet state, including all key-lifecycle counters.
     *
     * # Reasoning
     *
     * Persistence is a key-safety mechanism, not merely a convenience. Three
     * classes of state here are load-bearing:
     *
     *  - **HD counters** (`next_wots_index`, per-address `next_leaf`). Losing or
     *    rewinding these re-derives a key that may already have signed. They are
     *    written before a broadcast, not after, so a crash leaves them ahead of
     *    reality rather than behind it.
     *  - **`spent_wots`**, the published-key set. Losing it re-opens those
     *    addresses to {@link Wallet#addUtxo} and hands a reuse opportunity back
     *    to coin selection.
     *  - **Reorg state** (`spent_log`, `block_hashes`, `sent_reveals`,
     *    `reuse_locked`). Losing `sent_reveals` silently demotes recoverable
     *    coins to reuse-locked on the next rollback; losing `reuse_locked`
     *    silently *promotes* locked coins back to spendable, which is the unsafe
     *    direction.
     *
     * BigInt is not JSON-serializable, so every currency value — in `utxos` and
     * in the coin records inside `spent_log` — is written as a decimal string
     * and rehydrated by {@link Wallet.restore}. Letting either degrade to Number
     * would put the money path back on a lossy representation.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - storage is writable
     *
     * Post:
     *   ∃ D • D = encode(θWalletState)
     *     ∧ storage' = storage ⊕ { metadata ↦ D }
     *     ∧ decode(D) = θWalletState        (round-trip fidelity)
     *   θWalletState' = θWalletState        (Δ on storage only)
     * ```
     *
     * ```zed
     *     Save
     *     ─────────────────────────────
     *     ΞWalletState
     *     ΔStorage
     *
     *     post decode(storage'.metadata) = θWalletState
     *     post ∀ c ∈ utxos • decode(storage'.metadata).utxos(c.coin_id).value = c.value
     *     post ∀ h ∈ dom spent_log • ∀ c ∈ spent_log(h) •
     *            decode(storage'.metadata).spent_log(h)(c.coin_id).value = c.value
     * ```
     *
     * # Safety / Invariants
     *
     * - **Round-trip fidelity**: `restore ∘ save = id` over every field that
     *   gates spendability — `utxos`, `spent_wots`, `reuse_locked`,
     *   `sent_reveals`, `spent_log`, `mss_addrs` and the HD counters. The
     *   value-as-string treatment exists to make that hold for currency.
     * - **Exception, by design**: `restore` is *monotonic* on `wots_addrs`
     *   rather than exact. It re-derives indices `[0, next_wots_index)` and
     *   backfills any missing entry, repairing wallets written before change
     *   addresses were added to the watchlist. So the guarantee is
     *   `wots_addrs ⊆ (restore ∘ save)(wots_addrs)`, never `⊋`. Widening the
     *   watchlist can only cause the wallet to *notice* more of its own coins,
     *   so this direction is safe; narrowing it would hide funds.
     * - Never writes a narrower view than it holds. Omitting a field here is
     *   equivalent to deleting it on the next restore.
     */
    async save() {
        await this.storage.saveMetadata(JSON.stringify({
            phrase: this.phrase,
            nextWotsIndex: this.nextWotsIndex,
            nextMssIndex: this.nextMssIndex,
            // BigInt is not JSON-serializable; persist values as decimal strings
            // and rehydrate them to BigInt in restore() so the money path never
            // silently degrades to Number across a reload.
            utxos: this.utxos.map(u => ({ ...u, value: u.value.toString() })),
            wotsAddrs: this.wotsAddrs,
            mssAddrs: this.mssAddrs,
            lastScannedHeight: this.lastScannedHeight,
            spentWots: this.spentWots,
            // Reorg state. spentLog holds full coin records, so its BigInt
            // values need the same decimal-string treatment as utxos.
            blockHashes: this.blockHashes,
            spentLog: Object.fromEntries(Object.entries(this.spentLog).map(
                ([h, coins]) => [h, coins.map(c => ({ ...c, value: c.value.toString() }))])),
            sentReveals: this.sentReveals,
            reuseLocked: this.reuseLocked,
            pendingResend: this.pendingResend,
            fullScanCompleted: this.fullScanCompleted,
            watchedContracts: this.watchedContracts,
            contractCoins: Object.fromEntries(Object.entries(this.contractCoins).map(
                ([id, c]) => [id, { ...c, value: c.value.toString() }])),
        }));
    }

    async getNewAddress() {
        const addr = this.inner.get_wots_address(this.nextWotsIndex);
        this.wotsAddrs[addr] = this.nextWotsIndex++;
        await this.save();
        return addr;
    }

    /**
     * Derive a new reusable (MSS) address and persist its signing tree.
     *
     * # Reasoning
     *
     * An MSS address is a Merkle tree over 2^height one-time WOTS keypairs. It
     * "tolerates reuse" only in the sense that each spend consumes a *different*
     * leaf; the tree is a finite budget of 2^height signatures and every leaf is
     * still one-time. Two things follow.
     *
     * First, the tree must be exported and persisted here, at creation. It
     * cannot be cheaply regenerated on demand, and without it the address can
     * receive but never spend.
     *
     * Second, `next_leaf` starts at 0 and is the authority for signing — see
     * {@link Wallet#_utxosForWasm}, which re-reads it at spend time rather than
     * trusting the value stamped onto a UTXO at discovery.
     *
     * Generation is the slowest operation in the wallet (2^height keypairs), so
     * `get_mss_address` takes a progress callback; it is surfaced rather than
     * hidden because a height-20 tree is ~1M keypairs.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - height? ≥ 1
     *
     * Post:
     *   addr! = mss_address(next_mss_index, height?)
     *   mss_addrs' = mss_addrs ⊕ { addr! ↦ (index ↦ next_mss_index,
     *                                       height ↦ height?, next_leaf ↦ 0) }
     *   next_mss_index' = next_mss_index + 1
     *   storage'.mss_tree(addr!) = export_mss_bytes(addr!)
     *   utxos' = utxos ∧ spent_wots' = spent_wots
     * ```
     *
     * ```zed
     *     GetNewReusableAddress
     *     ─────────────────────────────
     *     ΔWalletState
     *     ΔStorage
     *     height? : ℕ
     *     addr!   : Hash
     *
     *     pre  height? ≥ 1
     *     pre  addr! ∉ dom mss_addrs
     *
     *     post next_mss_index' = next_mss_index + 1
     *     post mss_addrs'(addr!).next_leaf = 0
     *     post mss_addrs'(addr!).height = height?
     *     post addr! ∈ dom storage'.mss_trees
     *     post utxos' = utxos
     * ```
     *
     * # Safety / Invariants
     *
     * - `next_leaf` starts at 0 and only ever increases; it is decremented
     *   nowhere in this module. A rewind would re-sign with a consumed leaf.
     * - The tree is persisted before the address is recorded as usable, so a
     *   crash cannot leave a receivable address whose tree is unrecoverable.
     * - MSS addresses are never entered into `spent_wots`; their reuse safety
     *   comes from leaf distinctness, not from address retirement.
     *
     * @param {number}   [height=10]  Tree height; the address can sign 2^height times.
     * @param {Function} [onProgress] Called as (current, total) during generation.
     * @returns {Promise<string>} Hex-encoded address.
     */
    async getNewReusableAddress(height = 10, onProgress = null) {
        const addr = this.inner.get_mss_address(this.nextMssIndex, height, onProgress ?? undefined);
        const treeBytes = this.inner.export_mss_bytes(addr);
        await this.storage.saveMssTree(addr, treeBytes);
        this.mssAddrs[addr] = { index: this.nextMssIndex++, height, next_leaf: 0 };
        await this.save();
        return addr;
    }

    /**
     * Add a UTXO to local wallet state.
     * @returns {boolean} true if newly added, false if it was a duplicate.
     */
    /**
     * Admit a coin to the UTXO set if it belongs to this wallet and its key is
     * still safe to use.
     *
     * # Reasoning
     *
     * This is the only entry point for credits, so it is where the wallet
     * decides what it is willing to call spendable. The decision is not simply
     * "is this address mine": a WOTS address whose one-time key has already been
     * published is mine and can still receive coins, but redeeming anything
     * there would produce a second signature from a burned key. The node ships a
     * Key Reuse Punishment Burn Protocol that lets any observer exploit exactly
     * that, so such a coin is refused rather than silently risked.
     *
     * Refusal is the conservative direction: the coin is real and remains on
     * chain, it is simply not offered. Admitting it and hoping coin selection
     * avoids it would put the guarantee in the wrong place.
     *
     * Every credit is height-stamped so a later rollback can distinguish a coin
     * that exists on the winning branch from one that only ever existed on a
     * branch being discarded.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - address? ∈ dom wots_addrs ∪ dom mss_addrs      (else raises)
     *
     * Post (admitted):
     *   address? ∈ dom wots_addrs ∧ address? ∉ dom spent_wots
     *     ∨ address? ∈ dom mss_addrs
     *   ∧ coin_id? ∉ ids(utxos)
     *   ⇒ utxos' = utxos ∪ { (coin_id?, address?, value?, salt?,
     *                         is_mss, created_at ↦ height?) }
     *     ∧ result! = true
     *
     * Post (refused):
     *   address? ∈ dom spent_wots  ∨  coin_id? ∈ ids(utxos)
     *   ⇒ utxos' = utxos ∧ result! = false
     *
     * Post (error):
     *   address? ∉ dom wots_addrs ∪ dom mss_addrs ⇒ raises, utxos' = utxos
     * ```
     *
     * ```zed
     *     AddUtxo
     *     ─────────────────────────────
     *     ΔWalletState
     *     address?, salt?, coin_id? : Hash
     *     value?   : ℕ
     *     height?  : Height
     *     result!  : 𝔹
     *
     *     pre  address? ∈ dom wots_addrs ∪ dom mss_addrs
     *
     *     post address? ∈ dom spent_wots ⇒ utxos' = utxos ∧ ¬result!
     *     post result! ⇒ #utxos' = #utxos + 1
     *     post ¬result! ⇒ utxos' = utxos
     *     post ∀ c ∈ utxos' • ¬c.is_mss ∧ c.address ∈ dom spent_wots ⇒ c.reuse_locked
     * ```
     *
     * # Safety / Invariants
     *
     * - **Maintains INV-1.** A free (unlocked) coin is never admitted at an
     *   address in `spent_wots`. The only path by which a coin at a published
     *   key enters `utxos` is {@link rollbackTo}, which flags it `reuse_locked`.
     * - Idempotent on `coin_id`: re-scanning a block cannot double-credit.
     * - `mss_leaf` stamped here is a discovery-time snapshot and is NOT
     *   authoritative for signing; see {@link Wallet#_utxosForWasm}.
     *
     * @param {string} address
     * @param {bigint|number|string} value
     * @param {string} salt
     * @param {string} coinId
     * @param {number|null} [createdAtHeight] Height of the crediting block.
     * @returns {boolean} true if the coin was admitted.
     */
    addUtxo(address, value, salt, coinId, createdAtHeight = null) {
        let is_mss = false, index = 0, mss_height = 0, mss_leaf = 0;
        if (this.wotsAddrs[address] !== undefined) {
            // A WOTS address whose key has already been published cannot safely
            // sign again. Admitting a coin that landed here after the spend
            // would offer it to coin selection as spendable, and redeeming it
            // would reveal enough of the one-time key for anyone to forge
            // against it — the node ships a Key Reuse Punishment Burn Protocol
            // that does exactly that. The coin is real but unspendable; it is
            // excluded from the balance rather than silently risked.
            if (this.spentWots[address] !== undefined) return false;
            index = this.wotsAddrs[address];
        } else if (this.mssAddrs[address] !== undefined) {
            is_mss = true;
            index = this.mssAddrs[address].index;
            // WasmUtxo.mss_height is a required u64 with no serde default, so it
            // must be present on every UTXO handed to prepare_spend or the WASM
            // rejects the whole JSON. WOTS coins carry 0; MSS coins carry their
            // tree height. (Mirrors worker.js addUtxo in the browser wallet.)
            mss_height = this.mssAddrs[address].height;
            mss_leaf = this.mssAddrs[address].next_leaf;
        } else {
            throw new Error("Address does not belong to this wallet");
        }

        if (!this.utxos.find(u => u.coin_id === coinId)) {
            // Currency values are held as BigInt end-to-end. Coin values are
            // bounded powers of 2 (safe individually), but balances are summed
            // and can exceed Number's 2^53 safe-integer ceiling, so the whole
            // money path uses BigInt to avoid silent precision loss.
            this.utxos.push({
                address, value: BigInt(value), salt, coin_id: coinId,
                index, is_mss, mss_height, mss_leaf,
                // Height-tag every credit so a rollback can tell a coin that
                // exists on the winning branch from one that only ever existed
                // on the branch being discarded.
                createdAtHeight: createdAtHeight ?? this._reorgHeight ?? 0,
            });
            return true;
        }
        return false;
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Chain scanning (compact block-filter sync)
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Push the current set of watched items (WOTS + MSS addresses, and held
     * coin IDs) into the WASM wallet so check_filter() can test block filters
     * against them. Must be refreshed whenever the watch set changes.
     * @private
     */
    /**
     * Serialize the UTXO set into the shape `WasmUtxo` expects, refreshing the
     * MSS leaf index from live wallet state and withholding reuse-locked coins.
     *
     * # Reasoning
     *
     * This is the single choke point where wallet UTXOs reach the signer, which
     * makes it the right place to enforce two distinct one-time-key properties.
     *
     * **Leaf freshness.** `build_reveal` signs with `kp.next_leaf =
     * inp.mss_leaf` — the leaf comes from the UTXO record handed to it, NOT from
     * a counter inside WASM. Passing `utxos` through untouched meant every coin
     * carried the `mss_leaf` stamped onto it by `addUtxo` at discovery time. Two
     * coins received at the same MSS address before either was spent both carry
     * leaf 0, so spending them in separate transactions signs twice with leaf 0.
     * An MSS leaf is a one-time WOTS key inside the Merkle tree: "MSS tolerates
     * reuse" refers to the *address*, since each spend consumes a different
     * leaf. Reusing a *leaf* is exactly as fatal as reusing a bare WOTS key.
     *
     * **Lock enforcement.** A coin carrying `reuse_locked` was restored by a
     * rollback at an address whose key already signed a transaction we no longer
     * hold (reorg.js CASE 2). Offering it to coin selection would let a spend
     * produce a second signature over a different commitment from that key.
     * Filtering here means no spend path can reach one by accident.
     *
     * **On `Number(value)`.** `WasmUtxo.value` is a u64 that serde parses from a
     * JSON number, so the BigInt must be narrowed. That is safe rather than
     * lucky: coin values are always powers of two and total supply is bounded
     * near 2^50 (`INITIAL_REWARD` = 2^30, halving yearly at
     * `TARGET_BLOCK_TIME` = 60s), while JSON round-trips integers exactly
     * through 2^53. The guard fails loudly rather than corrupting a value if
     * that ever stops holding.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - ∀ c ∈ utxos • c.value ≤ MAX_SAFE_INTEGER   (else Err)
     *
     * Post:
     *   out! = { serialize(c) | c ∈ utxos ∧ ¬c.reuse_locked }
     *   ∀ c ∈ out! • c.is_mss ⇒ c.mss_leaf   = mss_addrs(c.address).next_leaf
     *                        ∧ c.mss_height = mss_addrs(c.address).height
     *   state unchanged  (Ξ — this is a projection, not a transition)
     *
     * Post (error):
     *   ∃ c ∈ utxos • c.value > MAX_SAFE_INTEGER ⇒ raises, state unchanged
     * ```
     *
     * ```zed
     *     UtxosForWasm
     *     ─────────────────────────────
     *     ΞWalletState
     *     out! : seq WasmUtxo
     *
     *     pre  ∀ c ∈ utxos • c.value ≤ 2^53 − 1
     *
     *     post #out! = #{ c ∈ utxos | ¬c.reuse_locked }
     *     post ∀ c ∈ out! • ¬c.reuse_locked
     *     post ∀ c ∈ out! • c.is_mss ⇒ c.mss_leaf = mss_addrs(c.address).next_leaf
     *     post θWalletState' = θWalletState
     * ```
     *
     * # Safety / Invariants
     *
     * - **Enforces INV-1 at the boundary.** Even if a reuse-locked coin somehow
     *   reached `utxos` unflagged elsewhere, this filter is the last line before
     *   a signature is produced.
     * - Ξ: must never mutate. Callers invoke it speculatively (to size a spend)
     *   as well as to execute one, and a mutation here would advance key
     *   material for a transaction that never happens.
     * - The leaf refresh must read `mss_addrs`, never the UTXO's stored value;
     *   reading the stored value is the bug this function exists to prevent.
     *
     * @private
     * @returns {Object[]} UTXOs in `WasmUtxo` shape, locked coins excluded.
     */
    _utxosForWasm() {
        // Reuse-locked coins are withheld from the signer entirely.
        //
        // A coin carrying `reuseLocked` was restored by a rollback at an
        // address whose one-time key has already signed a transaction we no
        // longer hold. Handing it to prepare_spend would let coin selection
        // pick it, and signing it would produce a SECOND signature over a
        // DIFFERENT commitment from that key — the exact condition the node's
        // Key Reuse Punishment Burn Protocol exists to punish. Filtering here,
        // at the single choke point where UTXOs reach WASM, means no spend path
        // can reach them by accident.
        return this.utxos.filter((u) => !u.reuseLocked).map((u) => {
            const value = BigInt(u.value);
            if (value > Number.MAX_SAFE_INTEGER) {
                throw new Error(
                    `Coin value ${value} exceeds the safe JSON integer range; ` +
                    `refusing to narrow it and risk a corrupted spend.`
                );
            }
            const out = { ...u, value: Number(value) };
            if (u.is_mss) {
                const meta = this.mssAddrs[u.address];
                if (meta) {
                    out.mss_leaf = meta.next_leaf;
                    out.mss_height = meta.height;
                }
            }
            return out;
        });
    }

    /**
     * Record coins being removed from the UTXO set, bucketed by the height at
     * which the removal happened.
     *
     * # Reasoning
     *
     * A rollback can only restore what was written down. This is the sole
     * writer of that record, and it stores the FULL coin object rather than an
     * id: after a reorg the coin must be re-credited with its original address,
     * value, salt, key index and MSS leaf intact, and none of that is
     * recoverable from an id alone once the coin has left `utxos`.
     *
     * Coins are cloned on the way in. Storing a live reference would let a
     * later mutation of the UTXO object silently rewrite history, which is
     * exactly the kind of aliasing bug that turns a rollback into a
     * key-reuse incident.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - coins? ⊆ utxos
     *   - _reorgHeight ∈ Height
     *
     * Post:
     *   spent_log' = spent_log ⊕ { _reorgHeight ↦
     *                              spent_log(_reorgHeight) ∪ copy(coins?) }
     *   utxos' = utxos            (removal is the caller's responsibility)
     * ```
     *
     * ```zed
     *     LogSpent
     *     ─────────────────────────────
     *     ΔWalletState
     *     coins? : ℙ Coin
     *     h      : Height = _reorgHeight
     *
     *     pre  coins? ⊆ utxos
     *
     *     post spent_log'(h) = spent_log(h) ∪ coins?
     *     post ∀ k ∈ dom spent_log \ {h} • spent_log'(k) = spent_log(k)
     *     post utxos' = utxos
     * ```
     *
     * # Safety / Invariants
     *
     * - Maintains INV-3 (`∀ h ∈ dom spent_log • h ≤ scanned`) because
     *   `_reorgHeight` is always the height currently being applied.
     * - Writes no key state; poisoning is the caller's job so that the spend
     *   log stays a pure record of removals.
     *
     * @private
     * @param {Object[]} coins Coins about to be removed.
     */
    _logSpent(coins) {
        if (!coins || coins.length === 0) return;
        const h = this._reorgHeight ?? 0;
        if (!this.spentLog[h]) this.spentLog[h] = [];
        for (const c of coins) this.spentLog[h].push({ ...c });
    }

    /**
     * Spendable balance: the sum of every coin coin selection may legally use.
     *
     * # Reasoning
     *
     * After a reorg the UTXO set can contain coins that are genuinely ours and
     * genuinely unspent, yet must never be spent, because their one-time key
     * already signed a transaction whose signature we no longer hold. Counting
     * them in the headline balance would invite exactly the spend that leaks
     * the key. They are reported separately by {@link getLockedBalance} so the
     * money is visible without being offered.
     *
     * # Formal Specification
     *
     * ```text
     * Post:
     *   balance! = Σ { c.value | c ∈ utxos ∧ ¬c.reuse_locked }
     *   state unchanged
     * ```
     *
     * @returns {bigint} Spendable balance.
     */
    getBalance() {
        return this.utxos.reduce((a, u) => (u.reuseLocked ? a : a + BigInt(u.value)), 0n);
    }

    /**
     * Balance held in coins pinned by one-time-key reuse risk.
     *
     * These are recoverable only by re-sending the original signature, which
     * the wallet no longer holds. They are surfaced so the total is auditable;
     * they are never offered to coin selection.
     *
     * @returns {bigint} Reuse-locked balance.
     */
    getLockedBalance() {
        return this.utxos.reduce((a, u) => (u.reuseLocked ? a + BigInt(u.value) : a), 0n);
    }

    /**
     * Push the set of addresses and coin ids this wallet cares about into the
     * WASM compact-filter matcher.
     *
     * Watched contract addresses are included because `CompactFilter::items_in`
     * indexes BOTH coin ids and addresses for every input and output. Omitting
     * them meant a block that advanced a followed state thread produced no
     * filter hit, so the block was never fetched and the state update was never
     * seen — leaving the wallet holding a stale state that the node will reject
     * on the next contract execution.
     */
    /**
     * Follow a contract so its coins and state thread are tracked by sync.
     *
     * # Reasoning
     *
     * A contract's state thread is not addressed to any wallet key, so nothing
     * in the ordinary output-matching path would ever notice it. Following the
     * contract explicitly is what puts its address into the compact-filter
     * watchlist and its coins into `contractCoins`.
     *
     * This matters because `prepare_script_spend` validates the caller's
     * `coin_id` against the canonical derivation from
     * `(address, value, salt, state)` and rejects a mismatch outright. A wallet
     * that does not track the thread cannot supply the current state, and every
     * execution it builds is rejected as stale.
     *
     * @param {string} contractAddrHex 32-byte hex contract address.
     * @param {Object} [meta] Optional caller metadata (label, bytecode, …).
     * @returns {boolean} true if newly followed.
     */
    watchContract(contractAddrHex, meta = {}) {
        const addr = normalizeHex(contractAddrHex);
        if (this.watchedContracts[addr] !== undefined) return false;
        this.watchedContracts[addr] = { ...meta };
        this._setWatchlist();
        return true;
    }

    /**
     * Stop following a contract and forget its coins.
     *
     * The coins are dropped rather than retained: they were never spendable by
     * this wallet's keys, so there is nothing to protect, and keeping them would
     * leave stale state that a later re-follow would have to reconcile.
     *
     * @param {string} contractAddrHex
     * @returns {boolean} true if it had been followed.
     */
    unwatchContract(contractAddrHex) {
        const addr = normalizeHex(contractAddrHex);
        if (this.watchedContracts[addr] === undefined) return false;
        delete this.watchedContracts[addr];
        for (const [id, c] of Object.entries(this.contractCoins)) {
            if (c.address === addr) delete this.contractCoins[id];
        }
        this._setWatchlist();
        return true;
    }

    /**
     * Admit a contract coin (state thread or contract-owned funds).
     *
     * # Reasoning
     *
     * Held separately from `utxos` on purpose. A contract coin is script-locked;
     * placing it in the wallet UTXO set would offer it to
     * {@link Wallet#_utxosForWasm} and thence to `prepare_spend`, which would
     * try to sign it with a wallet key it cannot satisfy.
     *
     * Consensus requires a state-thread output to have value exactly 0
     * (`apply_transaction` bails on `commitment.is_some() && value != 0`), so
     * that is asserted here rather than discovered later inside WASM.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - address? ∈ dom watched_contracts
     *   - state? ≠ null ⇒ value? = 0
     *
     * Post:
     *   coin_id? ∉ dom contract_coins ⇒
     *     contract_coins' = contract_coins ⊕
     *       { coin_id? ↦ (address?, value?, salt?, state?, created_at ↦ height) }
     *     ∧ result! = true
     *   coin_id? ∈ dom contract_coins ⇒ contract_coins' = contract_coins ∧ ¬result!
     *   utxos' = utxos
     * ```
     *
     * ```zed
     *     AddContractCoin
     *     ─────────────────────────────
     *     ΔWalletState
     *     address?, salt?, coin_id? : Hash
     *     state?  : Hash ∪ {null}
     *     value?  : ℕ
     *     result! : 𝔹
     *
     *     pre  address? ∈ dom watched_contracts
     *     pre  state? ≠ null ⇒ value? = 0
     *
     *     post result! ⇒ coin_id? ∈ dom contract_coins'
     *     post ¬result! ⇒ contract_coins' = contract_coins
     *     post utxos' = utxos
     *     post spent_wots' = spent_wots
     * ```
     *
     * # Safety / Invariants
     *
     * - Contract coins never enter `utxos`, so INV-1 is untouched by this path.
     * - Never writes `spent_wots`: contract coins carry no one-time key.
     * - Height-stamped like wallet coins so rollback can discard dead-branch
     *   state.
     *
     * @private
     */
    _addContractCoin(address, value, salt, state, coinId) {
        if (state !== null && BigInt(value) !== 0n) {
            throw new Error(
                `State-thread coin at ${address.slice(0, 10)}… must have value 0 (got ${value}); ` +
                `contract funds belong in a separate standard coin.`
            );
        }
        if (this.contractCoins[coinId]) return false;
        this.contractCoins[coinId] = {
            address, value: BigInt(value), salt, state, coin_id: coinId,
            createdAtHeight: this._reorgHeight ?? 0,
            isContract: true,
        };
        return true;
    }

    /**
     * Current state thread for a followed contract.
     *
     * A contract has at most one live state thread; if scanning ever produced
     * more, the newest by height wins, since an older one can only be a record
     * this wallet failed to retire.
     *
     * @param {string} contractAddrHex
     * @returns {Object|null} The state coin, or null if none is tracked.
     */
    getStateThread(contractAddrHex) {
        const addr = normalizeHex(contractAddrHex);
        const threads = Object.values(this.contractCoins)
            .filter((c) => c.address === addr && c.state !== null)
            .sort((a, b) => (b.createdAtHeight || 0) - (a.createdAtHeight || 0));
        return threads[0] ?? null;
    }

    /**
     * All tracked coins for a followed contract (state thread plus funds).
     *
     * @param {string} contractAddrHex
     * @returns {Object[]}
     */
    getContractCoins(contractAddrHex) {
        const addr = normalizeHex(contractAddrHex);
        return Object.values(this.contractCoins).filter((c) => c.address === addr);
    }

    /**
     * Build the `contract_inputs` argument for `prepare_script_spend` from
     * tracked state.
     *
     * Supplies `coin_id`, `value`, `salt` and `state` from what sync actually
     * observed, so the caller cannot accidentally pass a stale state that the
     * node rejects. The witness is caller-supplied since only the caller knows
     * what satisfies the script.
     *
     * @param {string} contractAddrHex
     * @param {string} [witness=''] Comma-separated hex witness stack.
     * @param {boolean} [includeFunds=true] Include the contract's standard coin.
     * @returns {Object[]} ScriptInputArg-shaped entries.
     */
    buildContractInputs(contractAddrHex, witness = '', includeFunds = true) {
        const coins = this.getContractCoins(contractAddrHex);
        const chosen = includeFunds ? coins : coins.filter((c) => c.state !== null);
        if (chosen.length === 0) {
            throw new Error(
                `No tracked coins for contract ${contractAddrHex.slice(0, 10)}…. ` +
                `Call watchContract() and sync before building an execution.`
            );
        }
        // State thread first: the script reads it via OP_READ_INPUT_STATE and
        // the emulator's input ordering is what witnesses are written against.
        const ordered = [
            ...chosen.filter((c) => c.state !== null),
            ...chosen.filter((c) => c.state === null),
        ];
        return ordered.map((c) => ({
            coin_id: c.coin_id,
            witness,
            value: Number(c.value),
            salt: c.salt,
            ...(c.state !== null ? { state: c.state } : {}),
        }));
    }

    /**
     * Complete a verbatim re-send armed by a rollback.
     *
     * # Reasoning
     *
     * {@link rollbackTo} CASE 1 clears an address's poison and stages the
     * original signed reveal in `pendingResend`. Clearing the poison is only
     * defensible because the coin can leave exclusively via that identical
     * signature — so something must actually send it. Until this ran, a CASE 1
     * recovery left the coins spendable-in-principle and stuck in practice.
     *
     * The dead-commitment branch is where this deliberately diverges from the
     * browser wallet. There, a dead commitment means a transaction that never
     * confirmed, and the correct advice is "the coins are still yours, re-send".
     * Here the transaction *did* confirm once — that is why the key was poisoned
     * — and was then orphaned. The signature is public. Re-sending would mean
     * re-signing over a fresh commitment with a key that has already signed, so
     * the coins are demoted to CASE 2 (reuse-locked) instead. Telling the caller
     * to "just re-send" would be an instruction to leak the key.
     *
     * A missing input coin is treated the same way: the reveal can never
     * confirm, so the verbatim path is exhausted and the lock applies.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - client? is connected
     *
     * Post (no armed re-send):
     *   state unchanged ∧ status! = 'idle'
     *
     * Post (commitment live, inputs live):
     *   node receives pending_resend.payload verbatim
     *   ∧ pending_resend' = null ∧ status! = 'resent'
     *   ∧ no new signature was produced
     *
     * Post (commitment dead ∨ an input coin absent):
     *   ∀ a ∈ addresses(pending_resend.input_coin_ids) •
     *     spent_wots'(a) = (…, reuse_locked ↦ true)
     *   ∧ ∀ c ∈ utxos' with c.address = a • c.reuse_locked
     *   ∧ pending_resend' = null ∧ status! = 'dead'
     * ```
     *
     * ```zed
     *     RecoverPendingTx
     *     ─────────────────────────────
     *     ΔWalletState
     *     client?  : Client
     *     status!  : {idle, resent, dead, retry}
     *
     *     post status! = resent ⇒ pending_resend' = null
     *     post status! = dead   ⇒ pending_resend' = null
     *          ∧ ∀ c ∈ utxos' • c.coin_id ∈ pending_resend.input_coin_ids
     *                         ∧ ¬c.is_mss ⇒ c.reuse_locked
     *     post status! = idle   ⇒ θWalletState' = θWalletState
     *     post ¬∃ new signature produced by this operation
     * ```
     *
     * # Safety / Invariants
     *
     * - **Produces no signature.** It transmits a payload built earlier and
     *   nothing else. Any change that made this call a signing path would
     *   reintroduce the reuse it exists to avoid.
     * - **Re-establishes INV-1 on the dead path.** Coins whose verbatim route is
     *   gone are locked rather than left free at a published key.
     * - Idempotent: clearing `pendingResend` on both terminal outcomes means a
     *   repeated call cannot double-send or double-lock.
     *
     * @param {MidstateClient} client
     * @returns {Promise<{status:string, alerts?:Object[], error?:string}>}
     */
    async recoverPendingTx(client) {
        const pending = this.pendingResend;
        if (!pending) return { status: 'idle' };

        const dead = async (reason) => {
            const alerts = [];
            const ids = new Set(pending.inputCoinIds || []);
            const addrs = new Set(
                this.utxos.filter((c) => ids.has(c.coin_id) && !c.is_mss).map((c) => c.address)
            );
            for (const addr of addrs) {
                const prior = this.spentWots[addr] || {};
                this.spentWots[addr] = {
                    height: prior.height ?? null,
                    commitment: pending.commitment,
                    reuseLocked: true,
                };
                for (const c of this.utxos) {
                    if (c.address !== addr) continue;
                    c.reuseLocked = true;
                    this.reuseLocked[c.coin_id] = { address: addr, priorCommitment: pending.commitment };
                }
                alerts.push({
                    kind: 'reuse_lock',
                    address: addr,
                    priorCommitment: pending.commitment,
                    message:
                        `The original signed transaction can no longer be re-sent (${reason}). ` +
                        `The coin(s) at ${addr.slice(0, 10)}… are locked: this one-time key has ` +
                        `already signed, so re-sending would require a new signature and leak it. ` +
                        `Do not attempt to move them manually.`,
                });
            }
            this.pendingResend = null;
            await this.save();
            return { status: 'dead', reason, alerts };
        };

        // The commitment must still be in chain state or the reveal is dead.
        let commitAlive = null;
        try {
            const r = await client.checkCommitment(pending.commitment);
            commitAlive = !!(r && r.exists);
        } catch {
            return { status: 'retry', error: 'could not reach the node to verify the commitment' };
        }
        if (commitAlive === false) return dead('its commitment has expired from chain state');

        // Every input must still exist on-chain, or the node will prune the
        // reveal on every rebroadcast.
        for (const id of pending.inputCoinIds || []) {
            try {
                const c = await client.checkCoin(id);
                if (c && c.exists === false) return dead(`input coin ${id.slice(0, 12)}… no longer exists`);
            } catch {
                return { status: 'retry', error: 'could not reach the node to verify inputs' };
            }
        }

        const resp = await client.send(pending.revealPayload);
        if (!resp || resp.ok === false) {
            return { status: 'retry', error: resp?.body || resp?.error || 'node rejected the re-send' };
        }
        this.pendingResend = null;
        await this.save();
        return { status: 'resent', commitment: pending.commitment };
    }

    /**
     * Reconcile wallet state against a `NewBlockTip` push notification.
     *
     * # Reasoning
     *
     * Detection previously only ran inside {@link Wallet#sync}, so a wallet
     * sitting idle on an open push subscription would not notice a fork until
     * something else prompted a scan. `NewBlockTip` carries `height` and
     * `block_hash`, which is exactly the pair needed to spot a conflict against
     * recorded history — no extra request required for the common case.
     *
     * Note the notification carries no parent hash, so this cannot detect a fork
     * at a height it has not already recorded; it compares the announced hash
     * against our own record at that height and defers to
     * {@link maybeHandleReorg} for the descent when they disagree.
     *
     * @param {MidstateClient} client
     * @param {{height:number, block_hash?:string}} notif
     * @returns {Promise<Object|null>} Rollback report, or null if consistent.
     */
    async handleBlockTip(client, notif) {
        if (!notif || typeof notif.height !== 'number') return null;
        const h = notif.height;
        const announced = normalizeHex(notif.block_hash || '');
        const recorded = this.blockHashes[h];

        // Consistent, or nothing recorded at that height to compare against.
        if (!announced || !recorded || announced === recorded) return null;

        const report = await maybeHandleReorg(this, client, Math.max(h, this.lastScannedHeight));
        if (report) {
            this._setWatchlist();
            await this.save();
        }
        return report;
    }

    /** @returns {bigint} Total value held in tracked coins of a followed contract. */
    getContractBalance(contractAddrHex) {
        return this.getContractCoins(contractAddrHex)
            .reduce((a, c) => a + BigInt(c.value), 0n);
    }

    /**
     * Consolidate every coin at a single address into one output.
     *
     * # Reasoning
     *
     * `Transaction::Consolidate` requires all inputs to share one predicate
     * address — the node bails on a mismatch — which is the WOTS co-spend rule
     * made explicit at the transaction level. Callers therefore need to pick an
     * address, not just an amount, so this takes the source address and gathers
     * its coins rather than running general coin selection.
     *
     * It is still a two-phase transaction: the node checks the commitment is in
     * chain state exactly as it does for a Reveal. Only the reveal encoding
     * differs, carrying a single signature for the shared address.
     *
     * # Safety / Invariants
     *
     * - Consuming every coin at a WOTS address is exactly what the one-time key
     *   permits: one signature, all coins. {@link Wallet#_broadcastTwoPhaseTx}
     *   retires the address afterwards.
     * - Reuse-locked coins are excluded upstream by
     *   {@link Wallet#_utxosForWasm}, so a locked coin cannot be swept in here.
     *
     * @param {MidstateClient} client
     * @param {string} sourceAddrHex Address whose coins are consolidated.
     * @param {string} destAddrHex   Destination for the consolidated output.
     */
    async consolidate(client, sourceAddrHex, destAddrHex) {
        const src = normalizeHex(sourceAddrHex);
        const coins = this._utxosForWasm().filter((u) => u.address === src);
        if (coins.length < 2) {
            throw new Error(
                `Consolidate needs at least two coins at ${src.slice(0, 10)}…; found ${coins.length}.`
            );
        }
        if (coins.length > MAX_CONSOLIDATE_INPUTS) {
            throw new Error(
                `Consolidate accepts at most ${MAX_CONSOLIDATE_INPUTS} inputs; ` +
                `${src.slice(0, 10)}… has ${coins.length}. Consolidate in batches.`
            );
        }
        const ctxStr = this.inner.prepare_consolidate(
            JSON.stringify(coins), normalizeHex(destAddrHex), this.nextWotsIndex
        );
        return this._broadcastTwoPhaseTx(client, ctxStr, 'consolidate');
    }

    /**
     * Sweep fragmented WOTS coins into a fresh reusable (MSS) address.
     *
     * Unlike {@link Wallet#consolidate} this spans addresses: it is an ordinary
     * Reveal that happens to have many inputs, so the co-spend constraint is
     * handled by WASM's sibling bundling rather than by a single-address rule.
     *
     * @param {MidstateClient} client
     * @param {string} destAddrHex  Destination address (typically a new MSS one).
     * @param {number} [maxInputs=50] Cap on coins swept in one transaction.
     */
    async defrag(client, destAddrHex, maxInputs = 50) {
        const ctxStr = this.inner.prepare_defrag(
            JSON.stringify(this._utxosForWasm()),
            normalizeHex(destAddrHex),
            maxInputs,
            this.nextWotsIndex
        );
        return this._broadcastTwoPhaseTx(client, ctxStr, 'standard');
    }

    /**
     * Fund several addresses in one commit/reveal.
     *
     * Each funding's amount is split into power-of-two coins paid to its
     * address, with wallet inputs covering the sum plus a size-scaled fee. The
     * point is latency: funding N covenants becomes one ~2-block round trip
     * instead of N.
     *
     * @param {MidstateClient} client
     * @param {Array<{address:string, amount:bigint|number|string}>} fundings
     * @param {string|null} [databurnsJson] Optional data-burn outputs.
     */
    async fundMany(client, fundings, databurnsJson = null) {
        if (!Array.isArray(fundings) || fundings.length === 0) {
            throw new Error('fundMany requires at least one { address, amount } entry.');
        }
        const payload = fundings.map((f) => {
            const amount = BigInt(f.amount);
            if (amount > Number.MAX_SAFE_INTEGER) {
                throw new Error(`Funding amount ${amount} exceeds the safe JSON integer range.`);
            }
            return { address: normalizeHex(f.address), amount: Number(amount) };
        });
        const ctxStr = this.inner.prepare_fund_many(
            JSON.stringify(this._utxosForWasm()),
            JSON.stringify(payload),
            this.nextWotsIndex,
            databurnsJson
        );
        return this._broadcastTwoPhaseTx(client, ctxStr, 'script');
    }

    /**
     * Advance a followed contract's state thread.
     *
     * # Reasoning
     *
     * `build_state_thread_tx` needs the thread's current state, coin id and
     * salt, and rejects a mismatch against the canonical derivation. Requiring
     * the caller to supply those by hand is exactly how stale state creeps in —
     * particularly after a reorg, where a caller working from its own notes has
     * no way to know the thread was rolled back. This reads them from what sync
     * actually observed.
     *
     * Passing `null` for the current thread creates one, which is the correct
     * shape for a contract's first state transition.
     *
     * # Safety / Invariants
     *
     * - Reads the thread through {@link Wallet#getStateThread}, so a rollback is
     *   automatically reflected and a superseded state can never be submitted.
     * - Maintains INV-6: the new state output is zero-value by construction in
     *   WASM; this never fabricates a value-bearing thread.
     *
     * @param {MidstateClient} client
     * @param {string} contractBytecodeHex
     * @param {string} newStateHex 32-byte hex commitment for the new state.
     * @param {Object[]} [extraOutputs=[]] Additional outputs (e.g. covenant payments).
     */
    async advanceStateThread(client, contractBytecodeHex, newStateHex, extraOutputs = []) {
        const contractAddr = blake3_hash_hex(normalizeHex(contractBytecodeHex));
        if (this.watchedContracts[contractAddr] === undefined) {
            throw new Error(
                `Contract ${contractAddr.slice(0, 10)}… is not followed. ` +
                `Call watchContract() and sync so the current state is known.`
            );
        }
        const cur = this.getStateThread(contractAddr);
        const ctxStr = this.inner.build_state_thread_tx(
            JSON.stringify(this._utxosForWasm()),
            normalizeHex(contractBytecodeHex),
            cur ? cur.state : null,
            cur ? cur.coin_id : null,
            cur ? cur.salt : null,
            normalizeHex(newStateHex),
            JSON.stringify(extraOutputs),
            this.nextWotsIndex
        );
        return this._broadcastTwoPhaseTx(client, ctxStr, 'script');
    }

    /**
     * Subscribe to the node's push stream and keep wallet state reconciled.
     *
     * # Reasoning
     *
     * The pieces for live reconciliation all existed but nothing joined them:
     * `onPushEvent` delivered `NewBlockTip`, {@link Wallet#handleBlockTip} could
     * act on it, and {@link Wallet#recoverPendingTx} could finish an armed
     * re-send — but a caller had to wire all three by hand, and a caller who
     * didn't would sit on a stale or forked view indefinitely.
     *
     * Recovery is attempted after a rollback because that is exactly when a
     * CASE 1 re-send gets armed; running it there closes the loop in one pass
     * instead of waiting for the next manual call.
     *
     * Errors in a push handler are reported through `onError` rather than
     * thrown, because throwing inside the subscription would tear down the
     * stream and leave the wallet worse off than the fork it was reacting to.
     *
     * @param {MidstateClient} client
     * @param {Object}   [handlers]
     * @param {Function} [handlers.onReorg]  Rollback report, after it is applied.
     * @param {Function} [handlers.onRecover] Result of an attempted re-send.
     * @param {Function} [handlers.onChat]   Chat pushes, passed through.
     * @param {Function} [handlers.onTip]    Every NewBlockTip, after reconciliation.
     * @param {Function} [handlers.onError]  Any error raised while handling a push.
     * @returns {Function} Unsubscribe.
     */
    autoReconcile(client, handlers = {}) {
        const { onReorg, onRecover, onChat, onTip, onError } = handlers;
        let stopped = false;

        client.onPushEvent(async (event) => {
            if (stopped || !event) return;
            try {
                if (event.ChatMessage && onChat) { onChat(event.ChatMessage); return; }

                const tip = event.NewBlockTip || event.newBlockTip;
                if (!tip) return;

                const report = await this.handleBlockTip(client, tip);
                if (report) {
                    if (onReorg) onReorg(report);
                    // A rollback is the moment a verbatim re-send gets armed.
                    if (this.pendingResend) {
                        const rec = await this.recoverPendingTx(client);
                        if (onRecover) onRecover(rec);
                    }
                }
                if (onTip) onTip(tip, report);
            } catch (e) {
                if (onError) onError(e);
            }
        });

        return () => { stopped = true; };
    }

    /**
     * Reconcile local MSS leaf counters against the chain.
     *
     * # Reasoning
     *
     * `next_leaf` is a local guess at how much of a Merkle signature tree has
     * been consumed. It can fall *behind* reality — a spend confirmed from
     * another device or another install of the same seed, or a local state file
     * restored from an older backup. When it does, the wallet signs with a leaf
     * the chain has already seen.
     *
     * That failure is invisible at admission and permanent afterwards. The
     * mempool only compares a new reveal against the live mempool, so the node
     * accepts it; `prune_on_new_block` then evicts it against the chain's burned
     * leaf accumulator, every block, forever. From the caller's side the
     * transaction simply never confirms and re-sending never helps, because each
     * attempt signs with the same exhausted leaf.
     *
     * `mss_state` answers this in one O(1) lookup, and deliberately returns
     * `max(chain, mempool)` so an in-flight unmined spend is counted too.
     *
     * Counters only ever move FORWARD here. Trusting a lower remote value would
     * rewind onto leaves this wallet has already used — the exact reuse this
     * exists to prevent — so a node reporting a smaller index is ignored.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - client? is connected
     *
     * Post:
     *   ∀ a ∈ dom mss_addrs •
     *     mss_addrs'(a).next_leaf = max(mss_addrs(a).next_leaf, node_next_index(a))
     *   ∧ mss_addrs'(a).next_leaf ≥ mss_addrs(a).next_leaf     (monotonic)
     *   moved! = #{ a | mss_addrs'(a).next_leaf > mss_addrs(a).next_leaf }
     *   utxos' = utxos
     * ```
     *
     * ```zed
     *     ReconcileMssLeaves
     *     ─────────────────────────────
     *     ΔWalletState
     *     client? : Client
     *     moved!  : ℕ
     *
     *     post ∀ a ∈ dom mss_addrs • mss_addrs'(a).next_leaf ≥ mss_addrs(a).next_leaf
     *     post ∀ a ∈ dom mss_addrs • mss_addrs'(a).next_leaf ≥ node_next_index(a)
     *     post utxos' = utxos
     *     post spent_wots' = spent_wots
     * ```
     *
     * # Safety / Invariants
     *
     * - **Monotonic by construction.** `max` is the whole mechanism; any path
     *   that could lower a counter would re-open leaf reuse.
     * - Mirrors each advance into the WASM cache via `set_mss_leaf_index`, so
     *   the two views cannot drift.
     * - An address whose public key is unavailable (tree not loaded) is skipped
     *   rather than reset — absence of evidence is not evidence of zero.
     *
     * @param {MidstateClient} client
     * @returns {Promise<{checked:number, moved:number, details:Object[]}>}
     */
    async reconcileMssLeaves(client) {
        const details = [];
        let checked = 0, moved = 0;

        for (const [addr, meta] of Object.entries(this.mssAddrs)) {
            let pk;
            try { pk = this.inner.get_mss_pubkey(addr); } catch { pk = null; }
            if (!pk) continue;                       // tree not resident — cannot ask
            checked++;

            let remote;
            try {
                const st = await client.getMssState(pk);
                remote = Number(st?.next_index ?? 0);
            } catch {
                continue;                            // transient; leave the counter alone
            }

            if (Number.isFinite(remote) && remote > meta.next_leaf) {
                const from = meta.next_leaf;
                meta.next_leaf = remote;
                this.inner.set_mss_leaf_index(addr, remote);
                moved++;
                details.push({ address: addr, from, to: remote });
            }
        }

        if (moved > 0) await this.save();
        return { checked, moved, details };
    }

    /**
     * Ensure at least `gap` unused WOTS addresses sit above the highest index
     * that has ever held a coin.
     *
     * # Reasoning
     *
     * A wallet does not use its key indices contiguously. Change addresses,
     * multi-output spends and coinbase payouts all consume indices in bursts, so
     * a real history leaves coins scattered across a range far wider than the
     * number of coins. Observed on a live wallet: 320 coins spread over WOTS
     * indices 0..1728.
     *
     * A FIXED derivation window cannot cope with that. Whatever ceiling is
     * chosen, a wallet whose coins reach past it is scanned to completion and
     * reports a confident, wrong, too-small balance — the worst failure shape
     * available, because nothing signals that anything was missed. The previous
     * implementation pre-derived `gapLimit` addresses once and never grew.
     *
     * So the window is anchored to evidence rather than to a guess: it always
     * extends `gap` indices past the highest index actually seen holding a coin.
     * The node's own CLI does the same thing, reporting each step as
     * `Extended to WOTS index N (last coin by index N)`.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - gap? > 0
     *
     * Post:
     *   let used = max({ wots_addrs(c.address) | c ∈ utxos ∧ ¬c.is_mss } ∪ {-1})
     *   next_wots_index' ≥ used + 1 + gap?
     *   next_wots_index' ≥ next_wots_index            (monotonic; never rewinds)
     *   ∀ i ∈ [0, next_wots_index') • wots_address(i) ∈ dom wots_addrs'
     *   utxos' = utxos                                (derivation credits nothing)
     *   added! = next_wots_index' − next_wots_index
     * ```
     *
     * ```zed
     *     ExtendGap
     *     ─────────────────────────────
     *     ΔWalletState
     *     gap?   : ℕ₁
     *     added! : ℕ
     *
     *     pre  gap? > 0
     *
     *     post next_wots_index' ≥ next_wots_index
     *     post next_wots_index' ≥ highest_used_index + 1 + gap?
     *     post added! = next_wots_index' − next_wots_index
     *     post utxos' = utxos
     *     post spent_wots' = spent_wots
     * ```
     *
     * # Safety / Invariants
     *
     * - **Monotonic.** `next_wots_index` only ever increases. Rewinding it would
     *   re-issue an index that may already have signed, which is the one-time
     *   key failure the whole wallet is built to avoid.
     * - Derivation alone credits nothing and touches no key state: it widens
     *   what the wallet can *notice*, never what it can spend.
     * - Includes burned addresses in the "used" set so a retired index still
     *   pushes the window forward; a gap measured only from spendable coins
     *   would shrink after a sweep.
     *
     * @private
     * @param {number} gap Unused addresses to keep above the highest used index.
     * @returns {Promise<number>} How many addresses were newly derived.
     */
    async _extendGap(gap) {
        let highestUsed = -1;
        for (const c of this.utxos) {
            if (c.is_mss) continue;
            const idx = this.wotsAddrs[c.address];
            if (typeof idx === 'number' && idx > highestUsed) highestUsed = idx;
        }
        // A burned address is still evidence the wallet reached that index.
        for (const addr of Object.keys(this.spentWots)) {
            const idx = this.wotsAddrs[addr];
            if (typeof idx === 'number' && idx > highestUsed) highestUsed = idx;
        }

        const target = Math.max(this.nextWotsIndex, highestUsed + 1 + gap);
        const before = this.nextWotsIndex;
        while (this.nextWotsIndex < target) {
            const addr = this.inner.get_wots_address(this.nextWotsIndex);
            this.wotsAddrs[addr] = this.nextWotsIndex++;
        }
        if (this.nextWotsIndex !== before) await this.save();
        return this.nextWotsIndex - before;
    }

    /** @returns {number} Highest WOTS index known to have held a coin, or -1. */
    highestUsedWotsIndex() {
        let hi = -1;
        for (const c of this.utxos) {
            if (c.is_mss) continue;
            const idx = this.wotsAddrs[c.address];
            if (typeof idx === 'number' && idx > hi) hi = idx;
        }
        for (const addr of Object.keys(this.spentWots)) {
            const idx = this.wotsAddrs[addr];
            if (typeof idx === 'number' && idx > hi) hi = idx;
        }
        return hi;
    }

    _setWatchlist() {
        const watch = [
            ...Object.keys(this.wotsAddrs),
            ...Object.keys(this.mssAddrs),
            ...Object.keys(this.watchedContracts),
            ...this.utxos.map(u => u.coin_id),
            ...Object.keys(this.contractCoins),
        ];
        this.inner.set_watchlist(JSON.stringify(watch));
    }

    /**
     * Scan one fetched block for wallet-relevant activity, mutating this.utxos.
     * Adds coinbase outputs and reveal outputs paid to our addresses; removes
     * coins of ours that appear as reveal inputs (i.e. were spent).
     * @private
     * @returns {boolean} true if the block touched our wallet.
     */
    /**
     * Apply one block to wallet state: credit outputs we own, retire inputs we
     * spent, and poison any WOTS key the block published.
     *
     * # Reasoning
     *
     * Two things here were wrong before and both lost money silently.
     *
     * **Only `Reveal` was scanned.** `Transaction` also has a `Consolidate`
     * variant carrying its own inputs and outputs, used for dust sweeps. Every
     * spend and receipt through that path was invisible.
     *
     * **Spends were matched on salt alone.** `InputReveal` has no `address`
     * field: it carries a `predicate`, and `Predicate::address()` is
     * `BLAKE3(bytecode)` — every address is pay-to-script-hash. A coin's
     * identity is `(address, value, salt)`, or the domain-separated
     * `"CONFIDENTIAL" ⌢ address ⌢ commitment ⌢ salt` form for state threads
     * (the tag is a vestigial name; nothing about them is confidential).
     * Salt matching happened to work because salts are random, but it cannot
     * distinguish a confidential coin from a standard one and relied on a
     * collision-free accident. The coin id is now recomputed from the predicate,
     * with salt lookup demoted to a fallback for malformed block data.
     *
     * **WOTS sibling purge.** Spending any coin at a WOTS address publishes that
     * one-time key, so every sibling at the same address is burned with it. The
     * node enforces co-spend, so in a well-formed block siblings appear as
     * inputs anyway — but purging by address rather than by matched input also
     * covers coins that arrived at the address after the spend. MSS addresses
     * tolerate reuse (each spend consumes a distinct leaf) and are retired
     * singly.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - height? ∈ Height
     *   - block? is a Batch as returned by get_block (witnesses stripped)
     *
     * Post:
     *   let O = coinbase outputs ∪ { o ∈ tx.outputs | tx ∈ Reveal ∪ Consolidate }
     *   let I = { i ∈ tx.inputs  | tx ∈ Reveal ∪ Consolidate }
     *
     *   -- credits
     *   ∀ o ∈ O • o.address ∈ dom wots_addrs ∪ dom mss_addrs ⇒
     *       AddUtxo(o.address, o.value, o.salt, coin_id(o), height?)
     *
     *   -- spends
     *   ∀ i ∈ I • coin_id(i) ∈ ids(utxos) ⇒
     *     ¬is_mss(i) ⇒ spent_wots'(i.address) = (height ↦ height?)
     *                ∧ utxos' = utxos \ { c | c.address = i.address }
     *                ∧ spent_log'(height?) ⊇ { c | c.address = i.address }
     *     is_mss(i)  ⇒ utxos' = utxos \ { c | c.coin_id = coin_id(i) }
     *                ∧ spent_log'(height?) ⊇ { c | c.coin_id = coin_id(i) }
     *
     *   result! = true ⟺ utxos' ≠ utxos ∨ spent_wots' ≠ spent_wots
     * ```
     *
     * ```zed
     *     ProcessBlock
     *     ─────────────────────────────
     *     ΔWalletState
     *     block?  : Batch
     *     height? : Height
     *     result! : 𝔹
     *
     *     pre  height? ≥ 0
     *
     *     post ∀ c ∈ utxos' \ utxos • c.created_at = height?
     *     post ∀ i ∈ spent_inputs(block?) • ¬is_mss(i) ∧ matched(i) ⇒
     *            i.address ∈ dom spent_wots'
     *          ∧ ¬∃ c ∈ utxos' • c.address = i.address ∧ ¬c.reuse_locked
     *     post dom spent_log' ⊆ dom spent_log ∪ {height?}
     * ```
     *
     * # Safety / Invariants
     *
     * - **Maintains INV-1**: an address is added to `spent_wots` and all its
     *   free coins removed in the same step.
     * - **Maintains INV-3**: removals are logged under `height?`, which is the
     *   height being applied and therefore ≤ `scanned` once the batch commits.
     * - Every removal is recorded in `spent_log` before it happens, so a
     *   rollback can restore it. A removal that skips the log is unrecoverable.
     *
     * @private
     * @param {Object} block
     * @param {number|null} [height]
     * @returns {boolean} true if the block touched wallet state.
     */
    _processBlock(block, height = null) {
        // Height under which this block's mutations are recorded. Falls back to
        // the block's own field so callers that already have the batch need not
        // pass it twice.
        this._reorgHeight = height ?? block?.height ?? 0;
        if (!block) throw new Error('block fetch returned null');
        let matched = false;

        // salt → coin_id for coins we currently hold (used to detect spends).
        const ourSalts = new Map();
        for (const u of this.utxos) ourSalts.set(normalizeHex(u.salt), u.coin_id);

        // Coinbase outputs paid to us.
        if (Array.isArray(block.coinbase)) {
            for (const cb of block.coinbase) {
                const addrHex = normalizeHex(cb.address);
                const saltHex = normalizeHex(cb.salt);
                if (this.wotsAddrs[addrHex] !== undefined || this.mssAddrs[addrHex] !== undefined) {
                    const coinId = compute_coin_id_hex(addrHex, BigInt(cb.value), saltHex);
                    if (this.addUtxo(addrHex, cb.value, saltHex, coinId)) ourSalts.set(saltHex, coinId);
                    matched = true;
                }
            }
        }

        // Spends and receipts. Both Reveal and Consolidate carry inputs and
        // outputs and both move our coins, so both must be scanned — reading
        // only `Reveal` made every dust-sweep invisible to the wallet.
        if (Array.isArray(block.transactions)) {
            for (const tx of block.transactions) {
                const action = tx.Reveal || tx.reveal || tx.Consolidate || tx.consolidate;
                if (!action) continue;

                if (Array.isArray(action.inputs)) {
                    for (const inp of action.inputs) {
                        const saltHex = normalizeHex(inp.salt);

                        // Recompute the coin ID from the block data rather than
                        // trusting a salt lookup.
                        //
                        // InputReveal has no `address` field: it carries a
                        // `predicate`, and Predicate::address() is
                        // BLAKE3(bytecode) — every address is pay-to-script-hash.
                        // A coin's identity is (address, value, salt), or the
                        // domain-separated CONFIDENTIAL form when the input
                        // carries a state commitment. Matching on salt alone
                        // happened to work because salts are random, but it
                        // cannot distinguish a state-thread coin from a standard
                        // one and silently relies on a collision-free accident.
                        // Salt lookup stays only as a fallback for malformed
                        // block data.
                        const bytecode = inp.predicate?.Script?.bytecode ?? inp.bytecode;
                        let cid = null;
                        if (bytecode) {
                            const addrHex = blake3_hash_hex(normalizeHex(bytecode));
                            cid = inp.commitment
                                ? computeStateThreadCoinId(addrHex, normalizeHex(inp.commitment), saltHex)
                                : compute_coin_id_hex(addrHex, BigInt(inp.value), saltHex);
                        }
                        if (!cid) cid = ourSalts.get(saltHex);
                        if (!cid) continue;

                        // A contract coin is consumed by satisfying its script,
                        // not by a one-time signature, so it carries no key-reuse
                        // risk and skips the WOTS poisoning below entirely. It
                        // still MUST be logged and removed: a state thread that
                        // stays in `contractCoins` after being spent is a stale
                        // state, and the node rejects the next execution built on
                        // it ("coin_id does not match (address,value,salt,state)").
                        const contractCoin = this.contractCoins[cid];
                        if (contractCoin) {
                            this._logSpent([{ ...contractCoin, isContract: true }]);
                            delete this.contractCoins[cid];
                            matched = true;
                            continue;
                        }

                        const spent = this.utxos.find(u => u.coin_id === cid);
                        if (!spent) continue;

                        // Spending any coin at a one-time WOTS address publishes
                        // that key, so every sibling coin at the same address is
                        // burned with it. The node enforces the co-spend rule, so
                        // in a well-formed block the siblings appear as inputs
                        // here too and would be removed individually — but a coin
                        // that arrived at the address *after* the spend never
                        // will be, and leaving it in the UTXO set offers it to
                        // the next coin selection as if it were spendable.
                        // Redeeming it would reuse the one-time key, which the
                        // node punishes with the Key Reuse Burn Protocol. MSS
                        // addresses tolerate reuse and are removed singly.
                        if (!spent.is_mss) {
                            this.spentWots[spent.address] = { height: this._reorgHeight };
                            this._logSpent(this.utxos.filter(u => u.address === spent.address));
                            this.utxos = this.utxos.filter(u => u.address !== spent.address);
                            for (const [s, id] of ourSalts) {
                                if (!this.utxos.some(u => u.coin_id === id)) ourSalts.delete(s);
                            }
                        } else {
                            this._logSpent([spent]);
                            this.utxos = this.utxos.filter(u => u.coin_id !== cid);
                            ourSalts.delete(saltHex);
                        }
                        matched = true;
                    }
                }

                if (Array.isArray(action.outputs)) {
                    for (const out of action.outputs) {
                        // ── Standard output ──
                        const outData = out.Standard || out.standard;
                        if (outData) {
                            const addrHex = normalizeHex(outData.address);
                            const saltHex = normalizeHex(outData.salt);
                            if (this.wotsAddrs[addrHex] !== undefined || this.mssAddrs[addrHex] !== undefined) {
                                const coinId = compute_coin_id_hex(addrHex, BigInt(outData.value), saltHex);
                                if (this.addUtxo(addrHex, outData.value, saltHex, coinId)) ourSalts.set(saltHex, coinId);
                                matched = true;
                            } else if (this.watchedContracts[addrHex] !== undefined) {
                                // A contract's spendable funds live in a standard
                                // coin at the contract address, separate from its
                                // state thread (which must have value 0).
                                const coinId = compute_coin_id_hex(addrHex, BigInt(outData.value), saltHex);
                                if (this._addContractCoin(addrHex, outData.value, saltHex, null, coinId)) matched = true;
                            }
                            continue;
                        }

                        // ── State thread (the `Confidential` variant) ──
                        //
                        // Zero-value by consensus, carrying a 32-byte commitment
                        // that scripts read via OP_READ_INPUT_STATE. Its coin id
                        // uses the domain-separated CONFIDENTIAL form, not the
                        // standard (address, value, salt) hash. These were
                        // ignored entirely before, so a followed contract's state
                        // was never recorded and every subsequent execution
                        // supplied a stale state that the node rejected.
                        const conf = out.Confidential || out.confidential;
                        if (conf) {
                            const cAddr = normalizeHex(conf.address);
                            if (this.watchedContracts[cAddr] !== undefined) {
                                const cSalt = normalizeHex(conf.salt);
                                const cState = normalizeHex(conf.commitment);
                                const cId = computeStateThreadCoinId(cAddr, cState, cSalt);
                                if (this._addContractCoin(cAddr, 0, cSalt, cState, cId)) matched = true;
                            }
                        }
                    }
                }
            }
        }

        return matched;
    }

    /**
     * Sync wallet UTXOs from the chain using compact block filters.
     *
     * For each block from lastScannedHeight to the current tip, the node's
     * compact filter is tested locally against our watchlist; only blocks that
     * match (plus probabilistic false positives, plus any block the node served
     * without a filter) are fully downloaded and scanned. Discovered coins are
     * added and spent coins removed, then the result is persisted.
     *
     * Only addresses already known to the wallet (via getNewAddress /
     * getNewReusableAddress) are detected. If you expect to receive on
     * addresses you haven't generated yet, pass `gapLimit` to pre-derive that
     * many WOTS addresses ahead before scanning.
     *
     * @param {MidstateClient} client
     * @param {Object}   [opts]
     * @param {number}   [opts.gapLimit=0]    Pre-derive this many extra WOTS addresses first.
     * @param {number}   [opts.batchSize=1000] Blocks per getFilters request.
     * @param {boolean}  [opts.rescan=false]  Wipe UTXOs and rescan from genesis.
     * @param {Function} [opts.onProgress]    Called as ({height, chainHeight, balance, note?}).
     * @param {number}   [opts.filterIntervalMs=550] Minimum spacing between getFilters
     *   requests. The node rate-limits "expensive" requests (get_filters and
     *   block_template share one budget) to ~120 per 60s window for a fresh
     *   peer — `current_expensive_limit()` is `20 + 200 * P(honest)`, and a new
     *   peer sits at P=0.5 under the Beta(1,1) prior. Each rejection counts as
     *   a violation; LIGHT_BAN_THRESHOLD (20) violations triggers a
     *   LIGHT_BAN_DURATION_SECS (500s) ban. Pacing at >500ms keeps a full-chain
     *   scan inside that budget. Set to 0 to disable pacing (only safe for
     *   short scans).
     * @param {number}   [opts.rateLimitCooldownMs=62000] If the limiter still fires
     *   (e.g. reduced reputation), wait roughly one window before retrying.
     * @param {number}   [opts.maxRateLimitRetries=6] Cap on consecutive cooldowns
     *   per batch (stays well under the 20-violation ban threshold).
     * @param {number}   [opts.maxStalledBatches=20] Give up after this many
     *   consecutive batches in which the peer returned no filters at all.
     * @param {Function} [opts.onReorg] Called with the rollback report if a
     *   chain reorganisation is detected and healed.
     * @param {boolean}  [opts.skipReorgCheck=false] Skip divergence detection.
     * @returns {Promise<{height:number, found:number, balance:bigint,
     *   lockedBalance:bigint, utxos:number, reorg:Object|null}>}
     *
     * # Reasoning
     *
     * Sync is the wallet's only source of truth about the chain, so it carries
     * three obligations that are easy to get subtly wrong.
     *
     * **Undecidable filters must be fetched, not skipped.** The node emits
     * `element_count = 0` with an empty filter to mean "I could not build a
     * filter for this height — fetch it yourself", never "this block is empty":
     * every block has coinbase outputs, so a genuine filter is never empty.
     * Treating zero as "nothing here" silently skipped any block whose filter
     * failed to load, losing coins paid to us in it.
     *
     * **Short batches must not be walked block-by-block.** The node now emits
     * placeholders instead of truncating, so a short batch means the peer cannot
     * serve the range. Grinding the remainder turns one request into up to 1000
     * and is indistinguishable from deliberate rate-limit evasion.
     *
     * **The reorg check runs before the early return.** A reorg that replaces
     * blocks without advancing the tip leaves `chainHeight` unchanged, so
     * checking after the "nothing to do" return misses exactly the case where
     * our view is wrong but our height is right.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - client? is connected
     *
     * Post:
     *   -- reorg reconciliation first
     *   diverged ⇒ RollbackTo(fork) applied before any extension
     *
     *   -- then forward scan
     *   ∀ h ∈ [scanned, chain_height) •
     *     h ∈ dom block_hashes'                      (recorded unconditionally)
     *     ∧ (undecidable(h) ∨ filter_hit(h) ⇒ ProcessBlock(h) applied)
     *   scanned' = chain_height
     *   ∀ c ∈ utxos' • ¬c.is_mss ∧ c.address ∈ dom spent_wots' ⇒ c.reuse_locked
     *
     * Post (peer cannot serve the range):
     *   raises after maxStalledBatches consecutive empty batches; scanned' is
     *   checkpointed at the last fully-processed height, so a re-run resumes
     *   rather than rescanning from genesis.
     * ```
     *
     * # Safety / Invariants
     *
     * - Maintains INV-1 and INV-3 by delegating every mutation to
     *   {@link Wallet#_processBlock} and {@link rollbackTo}.
     * - Maintains INV-4 via {@link pruneHistory}, called once per batch.
     * - Checkpoints `scanned` only after a batch is fully applied, so an
     *   interruption never marks unscanned heights as done.
     */
    async sync(client, {
        gapLimit = 0,
        mssGap = 0,
        mssHeight = 10,
        maxGapPasses = 12,
        cacheFilters = true,
        maxCachedBatches = 4096,
        allowPartialGapScan = false,
        batchSize = 1000,
        rescan = false,
        onProgress,
        filterIntervalMs = 550,
        rateLimitCooldownMs = 62_000,
        maxRateLimitRetries = 6,
        maxStalledBatches = 20,
        skipReorgCheck = false,
        deepRescan = false,
        onReorg,
    } = {}) {
        if (rescan) {
            // Chain-derived state is discarded and rebuilt from blocks.
            //
            // `spentWots`, `reuseLocked` and `sentReveals` are deliberately NOT
            // cleared. They record which one-time keys this wallet has published
            // — a fact about our own signing history, not about any particular
            // chain. A rescan re-derives the published-key set only for spends
            // that appear on the branch being scanned; a key burned by a spend
            // that was later orphaned appears nowhere, so clearing these would
            // silently re-admit its coins as spendable and hand back the exact
            // reuse the burn protocol punishes.
            this.utxos = [];
            this.contractCoins = {};
            this.blockHashes = {};
            this.spentLog = {};
            this.lastScannedHeight = 0;
        }

        // Ensure a gap window exists above the highest index we already watch.
        // Extension after each pass is what makes this adaptive; see
        // {@link Wallet#_extendGap}.
        if (gapLimit > 0) await this._extendGap(gapLimit);
        if (mssGap > 0) {
            while (this.nextMssIndex < mssGap) await this.getNewReusableAddress(mssHeight);
        }

        const state = await client.getState();
        const chainHeight = state.height;

        // ── Reorg check, before anything else ───────────────────────────────
        //
        // This runs BEFORE the "nothing to do" early return on purpose. A reorg
        // that replaces blocks without advancing the tip leaves chainHeight
        // unchanged, so checking after the early return would miss exactly the
        // case where our view is wrong but our height is right. See reorg.js for
        // why an undetected fork can strand coins permanently.
        let reorg = null;
        if (!rescan && !skipReorgCheck) {
            reorg = await maybeHandleReorg(this, client, chainHeight);
            if (reorg && reorg.exhausted && !deepRescan) {
                // The fork is deeper than REORG_DEPTH, so `spentLog` no longer
                // holds the records needed to restore what was spent down there
                // and the UTXO set cannot be repaired incrementally. Rebuild it
                // from chain data instead. Published-key state survives the
                // rebuild (see the rescan branch above), so this heals the coin
                // set without widening what is spendable.
                reorg.escalatedToRescan = true;
                this.utxos = [];
                this.contractCoins = {};
                this.blockHashes = {};
                this.spentLog = {};
                this.lastScannedHeight = 0;
            }
            if (reorg) {
                this._setWatchlist();
                await this.save();
                if (onReorg) onReorg(reorg);
                if (onProgress) {
                    onProgress({
                        height: this.lastScannedHeight, chainHeight,
                        balance: this.getBalance(), reorg,
                    });
                }
            }
        }

        if (chainHeight <= this.lastScannedHeight) {
            return {
                height: chainHeight, found: 0, balance: this.getBalance(),
                lockedBalance: this.getLockedBalance(), utxos: this.utxos.length, reorg,
            };
        }

        this._setWatchlist();

        // Pace getFilters under the node's expensive-request budget, and treat a
        // rate-limit response as a cooldown-and-retry (one violation per cooldown)
        // rather than a hard failure.
        const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
        const isRateLimit = (e) => /rate.?limit/i.test(e?.message ?? '');
        let lastFilterAt = 0;
        const fetchFilters = async (start, end) => {
            let attempt = 0;
            while (true) {
                if (filterIntervalMs > 0) {
                    const wait = filterIntervalMs - (Date.now() - lastFilterAt);
                    if (wait > 0) await sleep(wait);
                }
                lastFilterAt = Date.now();
                try {
                    return await client.getFilters(start, end);
                } catch (e) {
                    if (isRateLimit(e) && attempt < maxRateLimitRetries) {
                        attempt++;
                        if (onProgress) {
                            onProgress({
                                height: start, chainHeight, balance: this.getBalance(),
                                note: `rate-limited, cooling down ~${Math.round(rateLimitCooldownMs / 1000)}s (retry ${attempt}/${maxRateLimitRetries})`,
                            });
                        }
                        await sleep(rateLimitCooldownMs);
                        lastFilterAt = 0; // window reset after the long wait
                        continue;
                    }
                    throw e;
                }
            }
        };

        // ── Adaptive gap passes ──
        //
        // Widening the derivation window mid-scan is not enough on its own:
        // blocks already examined were filtered against the OLD watchlist, so a
        // coin paid to a newly derived address in one of them was never seen.
        // The window has to widen and the range be re-examined until a pass
        // discovers nothing new — which is why the node's CLI reports a run of
        // `Extended to WOTS index …` lines rather than one.
        //
        // Bounded by maxGapPasses so a pathological wallet cannot loop forever;
        // exhausting the bound is reported rather than silently accepted,
        // because the alternative is a confident, too-small balance.
        const scanFrom = this.lastScannedHeight;
        let gapPasses = 0;
        let gapExhausted = false;

        // ── Gap extension needs the whole history ──
        //
        // The walk chains from one discovered index to the next, so an index
        // whose only activity lies below the scan floor is invisible — and a
        // hole wider than `gapLimit` stops the walk early, producing a balance
        // that is too small and carries no indication of it.
        //
        // Observed live: a 200k-block window ending at 251,015 walked to index
        // 1412 and stopped, where a full-chain scan reached 1472.
        //
        // So a partial range is refused unless the wallet has already completed
        // a scan from genesis (the normal incremental case, where earlier
        // passes established the index space) or the caller opts out knowingly.
        if (gapLimit > 0 && scanFrom > 0 && !this.fullScanCompleted && !allowPartialGapScan) {
            throw new Error(
                `Gap extension needs a full-chain scan: this run starts at height ${scanFrom} and ` +
                `this wallet has never scanned from genesis, so indices used below that height are ` +
                `invisible and the walk can stop early with a too-small balance. Scan from 0, or ` +
                `pass allowPartialGapScan: true if an incomplete result is acceptable.`
            );
        }

        // Filters are immutable per height; only the watchlist changes between
        // passes. Bounded so a full-chain scan cannot exhaust memory.
        const filterCache = (gapLimit > 0 && cacheFilters) ? new Map() : null;
        let cacheHits = 0;

        let current = this.lastScannedHeight;
        let found = 0;
        // Consecutive batches where the peer returned zero filters. Bounded so
        // a node that cannot serve this range costs a handful of requests
        // rather than thousands.
        let stalledBatches = 0;

        scanPass:
        while (current < chainHeight) {
            const end = Math.min(current + batchSize, chainHeight);
            // Filters do not change between gap passes — only the watchlist
            // does — so they are fetched once and re-tested locally thereafter.
            // Without this a 200k-block scan with 10 extensions re-fetched every
            // filter eleven times: measured at 2087s against a live node, of
            // which all but the first pass was avoidable.
            let filterData;
            if (filterCache && filterCache.has(current)) {
                filterData = filterCache.get(current);
                cacheHits++;
            } else {
                filterData = await fetchFilters(current, end);
                if (filterCache && filterCache.size < maxCachedBatches) {
                    filterCache.set(current, filterData);
                }
            }
            const filters = filterData.filters || [];
            const counts = filterData.element_counts || [];
            const hashes = filterData.block_hashes || [];
            const startH = filterData.start_height ?? current;
            const numFilters = filters.length;

            for (let i = 0; i < numFilters; i++) {
                const height = startH + i;
                const n = counts[i] || 0;
                const blockHash = hashes[i];

                // `element_count === 0` does NOT mean "nothing here".
                //
                // The node builds every filter from `CompactFilter::items_in`,
                // which always inserts each coinbase output — so a real block
                // ALWAYS has a non-zero count. Zero is emitted by exactly two
                // branches of the node's GetFilters handler, both meaning
                // "undecidable, fetch it yourself":
                //   - the block loaded but its filter did not (hash present,
                //     filter string empty), and
                //   - the height was unreadable (hash and filter both empty).
                //
                // Treating zero as "empty block, skip" — and testing it BEFORE
                // the missing-hash fallback, which made that fallback dead code
                // — silently skipped every such block. Any coin paid to us in a
                // block whose filter failed to load was never seen. That is a
                // funds-visibility bug, not a performance one.
                //
                // So: an absent hash, an empty filter, or a zero count all mean
                // "must fetch". Only a genuine, non-empty filter may rule a
                // block out.
                const filterHex = filters[i];
                let fetch;
                if (!blockHash || !filterHex || n === 0) {
                    fetch = true;                        // undecidable → fetch
                } else {
                    fetch = this.inner.check_filter(filterHex, blockHash, n); // may false-positive
                }

                // Record this height's canonical identity before applying it.
                // Recorded for EVERY height, not just ones that paid us: a fork
                // can occur at a block irrelevant to this wallet, and the
                // recorded hash is the only thing that later reveals it.
                recordBlockHash(this, height, blockHash);

                if (fetch) {
                    const mutated = this._processBlock(await client.getBlock(height), height);
                    if (mutated) { found++; this._setWatchlist(); }
                }

                if (onProgress && height % 100 === 0) {
                    onProgress({ height, chainHeight, balance: this.getBalance() });
                }
            }

            current += numFilters;

            // ── Short batch handling ──
            //
            // This used to be `while (current < end) getBlock(current)`, walking
            // the whole remainder of the batch one block at a time. That is a
            // 1000x request amplification: a single get_filters covering 1000
            // blocks becomes up to 1000 get_block calls. The node budgets a
            // scan at "120 requests * 1000 blocks = 120,000 blocks synced per
            // minute" — the filter path IS the sync path — and an unknown peer
            // gets ~275 general requests per minute. Grinding the tail blew
            // that budget continuously, and the node's own comment notes the
            // pattern is indistinguishable from deliberate rate-limit evasion.
            // At LIGHT_BAN_THRESHOLD (20) violations it earns a
            // LIGHT_BAN_DURATION_SECS (500s) ban.
            //
            // The current node no longer truncates on an unreadable height: it
            // emits an undecidable placeholder and keeps going, so all four
            // arrays span the full request and numFilters === end - start. A
            // short batch therefore means the peer could not serve this range
            // at all. Step over one height directly and re-request from there;
            // if the peer keeps returning nothing, surface it so the caller can
            // pick another node rather than hammering this one.
            if (numFilters === 0) {
                if (++stalledBatches > maxStalledBatches) {
                    throw new Error(
                        `Scan stalled at height ${current}: peer returned no filters ` +
                        `${stalledBatches} times. Try a different node.`
                    );
                }
                const mutated = this._processBlock(await client.getBlock(current), current);
                if (mutated) { found++; this._setWatchlist(); }
                current++;
            } else {
                stalledBatches = 0;
            }

            // Checkpoint after each fully-processed batch so progress survives an
            // interruption and a re-run resumes instead of rescanning from genesis.
            this.lastScannedHeight = current;
            // Bound the reorg structures once per batch rather than per block.
            pruneHistory(this, chainHeight);
            await this.save();

            // End of range: widen the window against what we actually found and,
            // if that derived anything, re-examine the range with it.
            if (current >= chainHeight && gapLimit > 0) {
                const added = await this._extendGap(gapLimit);
                if (added > 0) {
                    if (++gapPasses >= maxGapPasses) {
                        gapExhausted = true;
                        break scanPass;
                    }
                    if (onProgress) {
                        onProgress({
                            height: current, chainHeight, balance: this.getBalance(),
                            note: `extended to WOTS index ${this.nextWotsIndex} ` +
                                  `(last coin by index ${this.highestUsedWotsIndex()}); rescanning`,
                        });
                    }
                    this._setWatchlist();
                    this.lastScannedHeight = scanFrom;
                    current = scanFrom;
                }
            }
        }

        if (onProgress) onProgress({ height: chainHeight, chainHeight, balance: this.getBalance() });

        if (scanFrom === 0) {
            // Records that the index space has been established from genesis,
            // which is what makes later incremental gap extension trustworthy.
            this.fullScanCompleted = true;
            await this.save();
        }

        if (gapExhausted) {
            throw new Error(
                `Gap extension did not settle after ${maxGapPasses} passes (now at WOTS index ` +
                `${this.nextWotsIndex}). The balance below this point is INCOMPLETE — raise ` +
                `maxGapPasses or gapLimit and re-run rather than trusting it.`
            );
        }

        return {
            height: chainHeight, found, balance: this.getBalance(),
            lockedBalance: this.getLockedBalance(), utxos: this.utxos.length, reorg,
            wotsDerived: this.nextWotsIndex, highestUsedWots: this.highestUsedWotsIndex(),
            gapPasses, filterCacheHits: cacheHits, fullScan: scanFrom === 0,
        };
    }
    
    /**
     * Compute and sign an off-chain Layer 2 Channel state update.
     * 
     * # Reasoning
     * Trustless Hub-and-Spoke channels require the user to sign a new balance distribution
     * between themselves and the hub (Bot). This function computes the exact commitment
     * hash the Midstate blockchain expects, signs it, and securely persists the incremented
     * MSS leaf index to prevent key reuse.
     * 
     * # Formal Specification
     * ```text
     * Pre:  params contains { channelId, botAddress, newBotAmount, userAddress, newUserAmount, salt }
     *       wState.mssAddrs[userAddress] exists
     * Post: wState.mssAddrs[userAddress].next_leaf' = wState.mssAddrs[userAddress].next_leaf + 1
     *       wallet.dat is saved to disk
     *       result is the hex-encoded MSS signature
     * ```
     * 
     * @param {Object} params - The channel state parameters
     * @returns {Promise<string>} The hex-encoded MSS signature
     */
    /**
     * Sign a Q-Bolt payment-channel state with an MSS key.
     *
     * # Reasoning
     *
     * This previously opened with `if (!mssCachesReady) await loadMssCaches();`
     * — neither identifier exists in this module, so it raised a ReferenceError
     * before reaching any logic. The intent was to guarantee the signing tree is
     * resident before signing, which the explicit checks below now do.
     *
     * The tree is loaded on demand rather than assumed present: a wallet that
     * generated the key in a previous process and never re-imported would
     * otherwise fail inside WASM with a far less actionable message.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - params.userAddress ∈ dom mss_addrs           (else raises)
     *   - has_mss_cache(userAddress) ∨ storage holds the tree  (else raises)
     *
     * Post:
     *   mss cache contains userAddress
     *   ∧ cache leaf index = mss_addrs(userAddress).next_leaf
     *   ∧ sig! = MSS_sign(userAddress, channel_state)
     *   mss_addrs' = mss_addrs                 (leaf advanced by the spend path,
     *                                           not by signing a channel state)
     * ```
     *
     * # Safety / Invariants
     *
     * - The leaf index is synced from `mss_addrs` on import, never left at the
     *   tree's serialized default. Importing a tree and signing without setting
     *   the index would re-sign from leaf 0.
     * - This does not itself advance `next_leaf`; channel-state signing and
     *   on-chain spending consume leaves through different paths and conflating
     *   them would either waste or reuse leaves.
     */
    async signChannelState(params) {
        // This previously opened with `if (!mssCachesReady) await loadMssCaches();`
        // — neither identifier exists in this module, so the function threw a
        // ReferenceError before reaching any of its logic. The intent was to
        // guarantee the MSS signing tree is resident before signing, which is
        // what the explicit check below does.
        if (this.mssAddrs[params.userAddress] === undefined) {
            throw new Error("User MSS address not found in wallet.");
        }

        // sign_mss_hex signs from the in-memory tree cache. restore() reloads
        // trees from storage, but a wallet that generated its key in a previous
        // process and never re-imported would fail inside WASM with a less
        // obvious message, so load it here on demand.
        if (!this.inner.has_mss_cache(params.userAddress)) {
            const treeBytes = await this.storage.loadMssTree(params.userAddress);
            if (!treeBytes) {
                throw new Error(
                    `MSS signing tree for ${params.userAddress} is not loaded and not present in storage.`
                );
            }
            this.inner.import_mss_bytes(params.userAddress, new Uint8Array(treeBytes));
            this.inner.set_mss_leaf_index(params.userAddress, this.mssAddrs[params.userAddress].next_leaf);
        }

        // 1. Reconstruct the output hashes for the new channel state
        const outBot = compute_coin_id_hex(params.botAddress, BigInt(params.newBotAmount), params.salt);
        const outUser = compute_coin_id_hex(params.userAddress, BigInt(params.newUserAmount), params.salt);
        
        // 2. Compute the exact commitment hash
        const commitmentHash = compute_commitment_hex(
            JSON.stringify([params.channelId]), 
            JSON.stringify([outBot, outUser]), 
            params.salt
        );

        // 3. Sign it with the user's MSS key
        const signatureHex = this.inner.sign_mss_hex(params.userAddress, commitmentHash);

        // 4. Update the leaf counter locally and save to disk to prevent key reuse
        this.mssAddrs[params.userAddress].next_leaf++;
        await this.save();

        return signatureHex;
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Transaction Execution
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Internal helper to execute the 2-phase Commit/Reveal protocol.
     * @private
     */
    /**
     * Execute the commit–reveal lifecycle for a prepared spend context.
     *
     * # Reasoning
     *
     * This function advances and persists HD key material (WOTS indices and MSS
     * leaf counters) and is therefore the wallet's most dangerous state
     * transition: every one-time key it consumes can be consumed exactly once,
     * for ever. Two orderings matter, and both are chosen to fail in the safe
     * direction.
     *
     * **Reservation before broadcast.** Counters are advanced and persisted
     * BEFORE the commit is sent. If the process dies mid-flight the counters
     * have already moved, so the next run derives fresh keys rather than
     * rewinding onto a key this attempt may have signed with. Skipping a key
     * costs nothing; reusing one leaks it.
     *
     * **Retirement before transmission.** The moment `build_reveal` returns,
     * real signatures exist over real one-time keys. Those keys are burned
     * whether or not the node accepts the transaction and whether or not this
     * process survives the next line. Retiring them first means a crash during
     * `send`, or a rejection followed by a user retry, cannot re-select the same
     * coins and sign a second time over a different commitment.
     *
     * The signed reveal is retained in `sent_reveals` keyed by input address.
     * This is what makes reorg CASE 1 possible: if the transaction is later
     * orphaned, the wallet can re-broadcast the IDENTICAL signature instead of
     * re-signing. Without this retention every orphaned WOTS spend would fall to
     * CASE 2 and its coins would be permanently reuse-locked.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - ctx.selected_inputs ⊆ utxos
     *   - ∀ i ∈ ctx.selected_inputs • ¬i.reuse_locked
     *   - ctx.next_wots_index ≥ next_wots_index
     *
     * Post (success):
     *   next_wots_index' = ctx.next_wots_index
     *   ∀ a ∈ mss_addrs(ctx.selected_inputs) •
     *       mss_addrs'(a).next_leaf = mss_addrs(a).next_leaf + 1
     *       (exactly +1 per ADDRESS, never per input)
     *   ∀ i ∈ ctx.selected_inputs • ¬i.is_mss ⇒
     *       spent_wots'(i.address) = (height ↦ tip, commitment ↦ ctx.commitment)
     *       sent_reveals'(i.address) = (ctx.commitment, payload, ids, tip)
     *   utxos' = utxos \ (ctx.selected_inputs ∪ siblings(wots inputs))
     *   spent_log'(tip) ⊇ removed coins
     *
     * Post (commit timeout):
     *   spent_wots' = spent_wots  ∧  utxos' = utxos
     *   (no reveal was built, so no key was touched; only the PoW is lost)
     *
     * Post (reveal rejected by node):
     *   identical to Post(success) — the signature exists regardless of the
     *   node's verdict, so the key must still be retired.
     * ```
     *
     * ```zed
     *     BroadcastTwoPhaseTx
     *     ─────────────────────────────
     *     ΔWalletState
     *     ΔHdCounters
     *     ctx?  : SpendContext
     *     tip?  : Height
     *     resp! : NodeResponse
     *
     *     pre  ctx?.selected_inputs ⊆ utxos
     *     pre  ∀ i ∈ ctx?.selected_inputs • ¬i.reuse_locked
     *
     *     post next_wots_index' = ctx?.next_wots_index
     *     post ∀ a ∈ { i.address | i ∈ ctx?.selected_inputs ∧ i.is_mss } •
     *            mss_addrs'(a).next_leaf = mss_addrs(a).next_leaf + 1
     *     post ∀ i ∈ ctx?.selected_inputs • ¬i.is_mss ⇒
     *            i.address ∈ dom spent_wots' ∧ i.address ∈ dom sent_reveals'
     *     post ∀ c ∈ utxos' • c.coin_id ∉ { i.coin_id | i ∈ ctx?.selected_inputs }
     *     post ∀ c ∈ utxos' • ¬c.is_mss ∧ c.address ∈ dom spent_wots' ⇒ c.reuse_locked
     * ```
     *
     * # Safety / Invariants
     *
     * - **Maintains INV-1.** Every WOTS input address is added to `spent_wots`
     *   and all its coins removed in the same step, so no free coin is left at
     *   a published key.
     * - **One leaf per MSS address per transaction.** `build_reveal` caches the
     *   signature per address and reuses it across that address's inputs — they
     *   all sign the same commitment — so incrementing per input would silently
     *   burn a multiple of the tree capacity actually consumed.
     * - **Retirement is unconditional on `send`'s verdict.** Making it
     *   conditional would reintroduce the retry-reuse path.
     * - The reveal retained in `sent_reveals` is the verbatim payload. It must
     *   never be regenerated on recovery; see reorg.js CASE 1.
     *
     * @private
     */
    async _broadcastTwoPhaseTx(client, spendCtxStr, revealKind = 'standard') {
        const ctx = JSON.parse(spendCtxStr);
        const inputs = ctx.selected_inputs || ctx.wallet_inputs || [];

        // ── Reserve key material BEFORE anything is broadcast ───────────────
        //
        // Derived change addresses and consumed MSS leaves are persisted up
        // front, on purpose. If the process dies here the counters have already
        // moved forward, so the next run derives fresh keys instead of rewinding
        // onto keys this attempt may have signed with. Skipping a key is free;
        // reusing one is not.
        while (this.nextWotsIndex < ctx.next_wots_index) {
            const addr = this.inner.get_wots_address(this.nextWotsIndex);
            this.wotsAddrs[addr] = this.nextWotsIndex;
            this.nextWotsIndex++;
        }

        // One MSS leaf is consumed per ADDRESS per transaction, not per input.
        // build_reveal caches the signature for an address and reuses it across
        // every input at that address, because they all sign the same
        // commitment. Incrementing once per input over-consumed the tree — a
        // height-10 address has only 1024 leaves, so a 3-input spend burned
        // three times the capacity it actually used. Dedupe by address.
        const mssAddrsUsed = new Set(
            inputs.filter((i) => i.is_mss && this.mssAddrs[i.address]).map((i) => i.address)
        );
        for (const addr of mssAddrsUsed) {
            const next = this.mssAddrs[addr].next_leaf + 1;
            this.mssAddrs[addr].next_leaf = next;
            // Keep the WASM cache's counter in step with ours. build_reveal sets
            // the leaf from the UTXO record, but other entry points read the
            // cache, and letting the two drift is how leaf reuse creeps back in.
            this.inner.set_mss_leaf_index(addr, next);
        }
        await this.save();

        const state = await client.getState();
        const requiredPow = state.required_pow || 24;

        // 1. Mine the anti-spam PoW for the commitment.
        const spamNonce = mine_commitment_pow(
            ctx.commitment,
            requiredPow,
            BigInt(state.height),
            state.header_hash
        );

        // 2. Commit (phase 1).
        const commitReq = await client.commit(ctx.commitment, Number(spamNonce));
        if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);

        // 3. Wait for the commitment to appear in chain state.
        //
        // Poll spacing is well inside the node's budget: check_commitment is a
        // cheap request (only block_template and get_filters count against the
        // expensive limit), and one call per 10s is 6/min against a ~275/min
        // allowance for a fresh peer. The commitment itself lives COMMITMENT_TTL
        // (1000 blocks, ~16.7h) once mined, so this window is generous.
        let mined = false;
        for (let i = 0; i < 120; i++) {
            await new Promise((r) => setTimeout(r, 10_000));
            const res = await client.checkCommitment(ctx.commitment).catch(() => ({}));
            if (res?.exists) { mined = true; break; }
        }
        if (!mined) {
            // Funds are not at risk: commits neither lock nor modify UTXOs, so
            // an unmined commitment costs the PoW effort and nothing else. The
            // reveal was never built, so no signing key was touched and the
            // coins remain spendable.
            throw new Error(
                "Timed out waiting for Commit to be mined. No funds were spent and no signing key was used; retry when the network catches up."
            );
        }

        // 4. Reveal (phase 2). Building the reveal is the point of no return:
        //    it produces real signatures over real one-time keys.
        //
        // All three kinds are two-phase: Consolidate also references a
        // commitment in chain state (`apply_transaction` bails on an unknown
        // one). Only the reveal encoding differs — a Consolidate carries a
        // single signature for the shared input address, which is why its
        // builder takes no commitment arguments and reads them from the context.
        let revealPayloadStr;
        if (revealKind === 'script') {
            revealPayloadStr = this.inner.build_script_reveal(spendCtxStr, ctx.commitment, ctx.tx_salt);
        } else if (revealKind === 'consolidate') {
            revealPayloadStr = this.inner.build_consolidate_reveal(spendCtxStr);
        } else {
            revealPayloadStr = this.inner.build_reveal(spendCtxStr, ctx.commitment, ctx.tx_salt);
        }

        // ── Retire the consumed keys BEFORE transmitting ────────────────────
        //
        // The signatures now exist. A WOTS key that has signed is burned whether
        // or not the node accepts the transaction, and whether or not this
        // process survives the next line. Recording that first means a crash
        // during send, or a rejected reveal followed by a user retry, cannot
        // re-select the same coins and sign a second time over a different
        // commitment — which is precisely what the node's Key Reuse Punishment
        // Burn Protocol exists to exploit.
        //
        // This deliberately runs before send() and regardless of its outcome.
        // Dropping a coin the network never accepted is recoverable with a
        // rescan; signing it twice is not.
        const spentIds = new Set(inputs.map((i) => i.coin_id));
        const wotsAddrsUsed = new Set();
        for (const inp of inputs) {
            if (!inp.is_mss) {
                wotsAddrsUsed.add(inp.address);
                // Record the commitment alongside the height. A rollback uses it
                // to confirm that a retained reveal belongs to the spend being
                // undone before it dares clear the poison (reorg.js CASE 1).
                this.spentWots[inp.address] = {
                    height: state.height,
                    commitment: ctx.commitment,
                };
                for (const u of this.utxos) {
                    if (u.address === inp.address) spentIds.add(u.coin_id);
                }
            }
        }

        // Retain the verbatim signed reveal so an orphaned spend can be
        // re-broadcast rather than re-signed. Keyed by address because the
        // reuse cliff is a property of the key, not of the individual coin.
        for (const addr of wotsAddrsUsed) {
            this.sentReveals[addr] = {
                commitment: ctx.commitment,
                revealPayload: revealPayloadStr,
                inputCoinIds: [...spentIds],
                sentAtHeight: state.height,
            };
        }

        // Log the removals under the current tip so a rollback can restore them.
        this._reorgHeight = state.height;
        this._logSpent(this.utxos.filter((u) => spentIds.has(u.coin_id)));
        this.utxos = this.utxos.filter((u) => !spentIds.has(u.coin_id));
        await this.save();

        return client.send(revealPayloadStr);
    }

    /**
     * Two-phase commit→reveal spend, with optional DataBurn.
     * @param {MidstateClient} client
     * @param {string} toAddressHex
     * @param {bigint|number|string} amountMDS 
     * @param {string|null} burnDataHex - Optional hex payload to permanently burn onto the chain
     * @param {bigint|number|string} burnValue - Amount of MDS to burn alongside the data (usually 0)
     */
    async send(client, toAddressHex, amountMDS, burnDataHex = null, burnValue = 0) {
        const amount = BigInt(amountMDS);
        const bValue = BigInt(burnValue);
        
        // Minor check, real check happens in WASM based on fees
        if (this.getBalance() <= (amount + bValue)) throw new Error("Insufficient funds (need extra for fees)");

        const utxosForWasm = this._utxosForWasm();
        const spendCtxStr = this.inner.prepare_spend(
            JSON.stringify(utxosForWasm),
            toAddressHex,
            amount,
            this.nextWotsIndex,
            burnDataHex,
            burnDataHex ? bValue : null
        );
        
        return this._broadcastTwoPhaseTx(client, spendCtxStr, 'standard');
    }

    /**
     * Fund a Smart Contract by sending value and/or creating a State Thread.
     * @param {MidstateClient} client
     * @param {string} contractAddrHex - 32-byte hex address of the contract
     * @param {bigint|number|string} amountMDS - Value to lock in the contract
     * @param {string|null} stateHex - Optional 32-byte initial state commitment
     */
    async fundContract(client, contractAddrHex, amountMDS, stateHex = null) {
        const amount = BigInt(amountMDS);
        const utxosForWasm = this._utxosForWasm();
        
        const ctxStr = this.inner.prepare_fund_tx(
            JSON.stringify(utxosForWasm),
            contractAddrHex,
            amount,
            stateHex,
            this.nextWotsIndex
        );

        return this._broadcastTwoPhaseTx(client, ctxStr, 'script');
    }

    /**
     * Execute a Smart Contract transaction.
     * @param {MidstateClient} client
     * @param {string} contractBytecodeHex - The raw compiled contract bytecode
     * @param {Array<Object>} contractInputsArray - Array of { coin_id, witness, value, salt, state }
     * @param {Array<Object>} outputsArray - Array of { out_type: "standard"|"confidential", address, value, state, salt }
     */
    async executeContract(client, contractBytecodeHex, contractInputsArray, outputsArray) {
        const utxosForWasm = this._utxosForWasm();
        
        const ctxStr = this.inner.prepare_script_spend(
            JSON.stringify(utxosForWasm),
            contractBytecodeHex,
            JSON.stringify(contractInputsArray),
            JSON.stringify(outputsArray),
            this.nextWotsIndex
        );

        return this._broadcastTwoPhaseTx(client, ctxStr, 'script');
    }
}
