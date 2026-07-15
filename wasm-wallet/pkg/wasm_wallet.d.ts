/* tslint:disable */
/* eslint-disable */

/**
 * The core wallet struct holding the master seed and cached MSS trees.
 *
 * All MSS Merkle trees are stored as complete `MssKeypair` structs — the
 * same type used by the native CLI wallet. This means signing is a simple
 * tree lookup (~10μs) rather than the expensive subtree recomputation that
 * the old `FractionalMss` design required.
 *
 * ## Lifecycle
 *
 * 1. Created via [`WebWallet::new`] (from mnemonic) or [`WebWallet::from_seed_hex`].
 * 2. MSS trees loaded from IndexedDB via [`WebWallet::import_mss_bytes`].
 * 3. Signing via [`WebWallet::build_reveal`] (uses cached trees).
 * 4. After generating new MSS keys, export via [`WebWallet::export_mss_bytes`]
 *    for IndexedDB persistence.
 */
export class WebWallet {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Build coinbase outputs for solo mining.
     *
     * Decomposes `total_value` (block reward + fees) into power-of-2
     * denominations, derives a fresh WOTS address for each, and returns
     * a JSON string containing:
     *
     * - `coinbase`: array of `{address, value, salt}` for the block template.
     * - `mining_addrs`: array of `{address, index}` for the wallet to track.
     * - `next_wots_index`: the updated derivation counter.
     *
     * # Returns
     *
     * `None` if CSPRNG fails (should never happen in a browser).
     */
    build_coinbase(total_value: bigint, next_wots_index: number): string | undefined;
    /**
     * Build coinbase outputs for web solo mining.
     *
     * Decomposes `total_value` into power-of-2 denominations and assigns
     * them directly to the user's reusable MSS address.
     */
    build_coinbase_to_mss(total_value: bigint, address_hex: string): string;
    /**
     * Assemble the Reveal payload for a Consolidate transaction.
     *
     * # Reasoning
     * Standard `build_reveal` generates a 1.5 KB signature for *every* input. For 5000+ dust
     * UTXOs, computing 5000 WOTS signatures requires billions of BLAKE3 hashes (freezing
     * the browser for 10+ seconds) and generates megabytes of useless signature data.
     * A Consolidate transaction strictly requires only ONE signature covering all inputs.
     * This function bypasses the redundant signing, keeping the browser lightning fast.
     *
     * # Formal Specification
     * ```text
     * Pre:
     *   - ctx_json is a valid SpendContext.
     *   - ctx_json.selected_inputs is not empty.
     *
     * Post:
     *   result = Ok(reveal_json) ⇒
     *     reveal_json.signatures contains EXACTLY ONE signature (the first input's signature).
     *     reveal_json.inputs contains all inputs without signatures.
     * ```
     */
    build_consolidate_reveal(ctx_json: string): string;
    /**
     * Build the reveal payload (inputs + signatures + outputs) for a committed transaction.
     *
     * # Safety Check
     *
     * Before returning, the function recomputes the commitment hash from the
     * generated reveals and verifies it matches the server commitment. This
     * catches any internal payload tracking errors that would cause the
     * transaction to be rejected on-chain.
     *
     * # MSS Signature Caching
     *
     * When multiple UTXOs share the same MSS address, only one MSS leaf is
     * consumed. The signature is cached and reused for all UTXOs at that
     * address within the same transaction (they all sign the same commitment,
     * so the signatures are identical).
     *
     * # Errors
     *
     * - `"MSS tree missing from cache."` — a required MSS tree wasn't loaded.
     * - `"Fatal Hash Mismatch!"` — internal consistency check failed.
     */
    build_reveal(spend_context_json: string, server_commitment_hex: string, server_salt_hex: string): string;
    /**
     * Phase 2: sign the wallet fee-inputs over the committed commitment and emit
     * the wire `reveal` payload. Mirrors `build_reveal` but (a) splices contract
     * witnesses verbatim, (b) hashes confidential outputs, (c) leaves contract
     * inputs unsigned. `commitment_hex` / `salt_hex` are the ctx values returned
     * by `prepare_script_spend` (pass ctx.commitment and ctx.tx_salt — there is
     * no server-side salt contribution in this protocol).
     */
    build_script_reveal(ctx_json: string, commitment_hex: string, salt_hex: string): string;
    /**
     * Recompute the block extension hash for a found nonce.
     *
     * Called after a mining worker finds a valid nonce. Produces the full
     * `Extension { nonce, final_hash }` JSON needed for block submission.
     *
     * # Returns
     *
     * `None` if `midstate_hex` is not valid 64-character hex.
     */
    build_solo_extension(midstate_hex: string, nonce: bigint): string | undefined;
    /**
     * Universal DeFi Transaction Builder
     * Constructs a transaction that transitions a State Thread while securely attaching
     * physical UTXOs to satisfy covenants (like paying a Treasury).
     * Uses dynamic fee calculation and greedy UTXO defragmentation.
     */
    build_state_thread_tx(available_utxos_json: string, contract_bytecode_hex: string, current_state_hex: string | null | undefined, current_coin_id_hex: string | null | undefined, current_salt_hex: string | null | undefined, new_state_hex: string, extra_outputs_json: string, next_wots_index: number): string;
    /**
     * Test a Golomb-coded compact block filter for wallet relevance.
     *
     * Returns `true` if any address in the watchlist matches the filter,
     * indicating the block should be fetched and fully processed.
     *
     * # Returns
     *
     * `false` if:
     * - The filter or block hash hex is invalid.
     * - The watchlist is empty.
     * - No watchlist address matches.
     */
    check_filter(filter_hex: string, block_hash_hex: string, n: number): boolean;
    /**
     * Export an MSS tree as a compact binary blob for IndexedDB storage.
     *
     * The format is documented at the module level. For height 10, the
     * output is ~64 KB — small enough for IndexedDB but too large for
     * localStorage's 5 MB limit when hex-encoded into the wallet JSON.
     *
     * # Layout
     *
     * ```text
     * [height:4][master_seed:32][next_leaf:8][master_pk:32][tree_len:4][tree:N×32]
     * ```
     *
     * All multi-byte integers are little-endian.
     *
     * # Errors
     *
     * Returns `Err` if the address is not in the WASM cache.
     */
    export_mss_bytes(address_hex: string): Uint8Array;
    /**
     * Create a wallet from a raw 32-byte master seed (hex-encoded).
     *
     * Used when importing a CLI wallet backup where the master seed is
     * available directly rather than via a mnemonic phrase.
     *
     * # Errors
     *
     * Returns `Err` if `seed_hex` is not exactly 64 valid hex characters.
     */
    static from_seed_hex(seed_hex: string): WebWallet;
    /**
     * Generate a full MSS keypair and cache it in memory.
     *
     * This is computationally expensive: height 10 requires generating
     * 1024 WOTS public keys (each: 18 chains × 65,535 BLAKE3 hashes).
     * With WASM SIMD128, this takes ~20–30 seconds on desktop, 1–3 minutes
     * on mobile.
     *
     * After generation, call [`export_mss_bytes`] to persist the tree to
     * IndexedDB so that subsequent logins are instant.
     *
     * # Arguments
     *
     * * `index` — MSS HD derivation index.
     * * `height` — Merkle tree height (10 = 1024 signatures).
     * * `progress_cb` — Optional JS callback `(current: u32, total: u32) => void`
     *   for UI progress reporting.
     *
     * # Returns
     *
     * The hex-encoded 32-byte MSS address (Merkle root hash).
     *
     * # Errors
     *
     * Returns `Err` if height is 0 or exceeds `MAX_HEIGHT` (20).
     */
    get_mss_address(index: number, height: number, progress_cb?: Function | null): string;
    get_mss_pubkey(address_hex: string): string | undefined;
    /**
     * Derive the WOTS address at a given HD index.
     *
     * Returns the hex-encoded 32-byte address. This is a pure computation
     * with no side effects — the address is not cached.
     */
    get_wots_address(index: number): string;
    /**
     * Check whether an MSS tree is loaded in the WASM-side cache.
     *
     * Returns `true` if the tree is ready for signing, `false` if it
     * needs to be loaded from IndexedDB or regenerated.
     */
    has_mss_cache(address_hex: string): boolean;
    /**
     * Import an MSS tree from a binary blob (previously exported via [`export_mss_bytes`]).
     *
     * After import, the tree is ready for signing — no recomputation needed.
     * Loading a 64 KB blob takes ~1ms.
     *
     * # Validation
     *
     * - Header must be at least 80 bytes.
     * - `tree_len` field must match the remaining data length.
     * - The `master_pk` in the blob is trusted (it was computed during generation).
     *
     * # Security Note
     *
     * The blob contains the MSS master seed. The caller is responsible for
     * encrypting the IndexedDB storage (the wallet uses AES-GCM via the
     * Web Crypto API before persisting).
     *
     * # Errors
     *
     * Returns `Err` if the data is too short or truncated.
     */
    import_mss_bytes(address_hex: string, data: Uint8Array): void;
    /**
     * Create a new wallet from a BIP39 mnemonic phrase.
     *
     * The master seed is derived via BLAKE3 from the mnemonic. The wallet
     * starts with an empty MSS cache — call [`import_mss_bytes`] to load
     * persisted trees, or [`get_mss_address`] to generate new ones.
     *
     * # Errors
     *
     * Returns `Err` if the mnemonic is invalid (wrong word count, unknown words,
     * or bad checksum).
     */
    constructor(phrase: string);
    /**
     * Prepare a Consolidate transaction (dust sweeping) for the Web Wallet.
     *
     * # Reasoning
     * Standard transactions (`prepare_spend`) budget for a 1.5 KB WOTS/MSS signature
     * *per input*. For dust sweeping (e.g., 100+ inputs), this overestimates the fee
     * massively, leading to false "Insufficient funds" errors. A `Consolidate`
     * transaction mathematically requires only *one* signature for the entire batch
     * of inputs (as long as they share the same address). This function applies
     * the heavily discounted single-signature fee calculation, enabling users to
     * sweep thousands of dust UTXOs affordably.
     *
     * # Formal Specification
     *
     * ```text
     * Pre:
     *   - available_utxos contains ≥ 2 UTXOs.
     *   - All UTXOs in available_utxos share the exact same address.
     *   - The sum of UTXO values > calculated_fee.
     *
     * Post:
     *   result = Ok(ctx_json) ⇒
     *     ctx_json.fee is calculated based on a 1-signature size budget.
     *     ctx_json.outputs contains power-of-2 denominations of (total - fee) at dest_address.
     *   result = Err(_) ⇒ state unchanged.
     * ```
     *
     * ```zed
     *     PrepareConsolidate
     *     ------------------
     *     ΔWebWallet
     *     available? : seq WasmUtxo
     *     dest_address? : String
     *     next_wots_index? : ℕ₃₂
     *     ctx! : String
     *
     *     pre  #available? ≥ 2
     *     pre  ∀ u, v ∈ available? • u.address = v.address
     *     let total = ∑ u ∈ available? • u.value
     *     let fee = (((600 + 3000 + 100 + #available? * 125) * 10) / 1024) + 20
     *     pre  total > fee
     *     post ctx! = JSON(SpendContext)
     * ```
     *
     * # Safety / Invariants
     * - Output values strictly conform to consensus power-of-2 requirements via `decompose_value`.
     * - Inputs are verified to share the same address to satisfy the `Transaction::Consolidate` rule.
     */
    prepare_consolidate(available_utxos_json: string, dest_address_hex: string, next_wots_index: number): string;
    /**
     * Plans a defragmentation batch: moves up to `max_inputs` fragmented
     * WOTS coins to a fresh MSS destination address (minus shape-dependent fee).
     */
    prepare_defrag(available_utxos_json: string, dest_address_hex: string, max_inputs: number, next_wots_index: number): string;
    /**
     * Fund MANY contract addresses in ONE transaction.
     *
     * Identical to [`prepare_fund_tx`] but takes a list of `{address, amount}`
     * fundings instead of a single address. Every funding's amount is split into
     * power-of-two coins paid to its address; wallet inputs cover the SUM plus a
     * size-scaled fee, with change returned to deterministic wallet addresses.
     * Used to fund a bundle of independent limit-order covenants (one fresh
     * secret/address each) in a single ~2-block commit/reveal rather than N of them.
     *
     * `fundings_json` — JSON array: `[{ "address": <64-hex>, "amount": <u64> }, ...]`.
     * Returns the same `ScriptSpendContext` JSON as `prepare_fund_tx`; the caller
     * recovers each covenant's coin by matching `outputs[].address`.
     */
    prepare_fund_many(available_utxos_json: string, fundings_json: string, next_wots_index: number, databurns_json?: string | null): string;
    /**
     * Phase 1 for FUNDING a contract. Pays `amount` to the contract address as
     * power-of-two "value" coins, optionally seeds a confidential "state" coin,
     * returns change to the wallet, and reuses `build_script_reveal` for phase 2
     * (its `contract_inputs` list is simply empty here — the wallet pays).
     *
     * Mirrors the CLI fund instruction:  `--to addr:amount` (+ `--to addr:0:state`).
     * `state_hex` = None for a plain value-only funding.
     */
    prepare_fund_tx(available_utxos_json: string, contract_addr_hex: string, amount: bigint, state_hex: string | null | undefined, next_wots_index: number): string;
    prepare_script_spend(available_utxos_json: string, contract_bytecode_hex: string, contract_inputs_json: string, outputs_json: string, next_wots_index: number): string;
    /**
     * This implements the full coin selection algorithm:
     *
     * 1. **Greedy selection**: picks largest coins first until the amount + fee is covered.
     * 2. **WOTS co-spend grouping**: pulls in all coins at the same WOTS address
     *    (required by the one-time signature security model).
     * 3. **Snowball merge**: opportunistically pulls in coins matching change
     *    denominations to consolidate the UTXO set.
     * 4. **Fee estimation**: iterates until the fee estimate stabilizes.
     *
     * # Arguments
     *
     * * `available_utxos_json` — JSON array of [`WasmUtxo`] objects.
     * * `to_address_hex` — recipient's 64-char or 72-char (checksummed) hex address.
     * * `send_amount` — total value to send.
     * * `next_wots_index` — current WOTS HD derivation counter.
     *
     * # Returns
     *
     * JSON string of [`SpendContext`] containing selected inputs, outputs,
     * commitment, and fee — everything needed for commit and reveal.
     *
     * # Errors
     *
     * - `"Insufficient funds."` — UTXO values don't cover amount + fee.
     * - `"MSS signing key not loaded."` — an MSS-backed UTXO's tree is missing.
     *   The user should run a Network Sync to trigger cache loading.
     * Selects coins and builds a transaction for a specified send amount.
     *
     * This is a complex state-machine loop that balances three competing goals:
     * 1. Security: Strictly enforcing the "One-Time Signature" (WOTS) co-spend rule.
     * 2. Efficiency: Consolidating fragmented UTXOs into larger denominations (Snowball Merge).
     * 3. Restorability: Ensuring all change coins are discoverable from the seed phrase.
     */
    prepare_spend(available_utxos_json: string, to_address_hex: string, send_amount: bigint, next_wots_index: number, databurn_hex?: string | null, databurn_value?: bigint | null): string;
    /**
     * Update the next-leaf counter for an MSS tree.
     *
     * Called by the JS layer after loading wallet state to synchronize
     * the WASM-side leaf counter with the persisted value.
     *
     * # No-op
     *
     * Silently does nothing if the address is not in the cache.
     */
    set_mss_leaf_index(address_hex: string, leaf_index: number): void;
    /**
     * Set the list of addresses the wallet watches during chain scanning.
     *
     * `addrs_json` is a JSON array of hex-encoded 32-byte addresses.
     * Replaces the entire watchlist. Invalid hex entries are silently skipped.
     */
    set_watchlist(addrs_json: string): void;
    /**
     * Sign a raw commitment hash using a cached MSS key for Layer 2 Payment Channels.
     *
     * # Reasoning
     * Payment channels require users to sign off-chain state updates (commitments)
     * without immediately broadcasting a `Reveal` transaction. This exposes the raw
     * MSS signature mechanism to JavaScript to facilitate trustless Hub-and-Spoke
     * L2 networks.
     *
     * # Formal Specification
     * ```text
     * Pre:  ∃ kp ∈ self.mss_cache.values() s.t. kp.master_pk == mss_pk_hex
     *       commitment_hex is a valid 64-character hex string (32 bytes)
     *       kp.remaining() > 0
     * Post: kp.next_leaf' = kp.next_leaf + 1
     *       result is Ok(signature_hex)
     * ```
     *
     * ```zed
     *     SignMssHex
     *     ----------
     *     ΔWebWallet
     *     mss_pk_hex? : String
     *     commitment_hex? : String
     *     sig! : String
     *
     *     let kp == (μ k ∈ ran(mss_cache) | hex(k.master_pk) = mss_pk_hex?)
     *
     *     pre  kp exists
     *     pre  kp.next_leaf < 2^{kp.height}
     *     post kp'.next_leaf = kp.next_leaf + 1
     *     post sig! = hex(sign(kp.master_seed, commitment))
     * ```
     */
    sign_mss_hex(mss_pk_hex: string, commitment_hex: string): string;
}

/**
 * Hash a hex-encoded byte string with BLAKE3.
 * Returns the 32-byte hash as a 64-character hex string.
 * Used by the IDE to generate P2SH addresses.
 */
export function blake3_hash_hex(hex_data: string): string;

export function build_channel_reveal(channel_value: bigint, channel_salt_hex: string, alice_pk_hex: string, bob_pk_hex: string, state_json: string, alice_sig_hex: string, bob_sig_hex: string): string;

export function build_channel_state(channel_coin_id_hex: string, alice_pk_hex: string, bob_pk_hex: string, alice_amount: bigint, bob_amount: bigint, nonce: number, htlcs_json: string): string;

export function build_covenant_htlc_bytecode_hex(secret_hash_hex: string, receiver_addr_hex: string, min_payout: bigint, timeout_height: bigint, refund_pk_hex: string): string;

/**
 * Builds the Midstate HTLC bytecode for cross-chain atomic swaps.
 */
export function build_htlc_bytecode_hex(secret_hash_hex: string, receiver_pk_hex: string, timeout_height: bigint, refund_pk_hex: string): string;

/**
 * Builds the limit-order covenant bytecode (Feature 1). See
 * `midstate::core::script::compile_limit_order_covenant` for the security notes.
 */
export function build_limit_order_covenant_bytecode_hex(secret_hash_hex: string, max_claim: bigint, timeout_height: bigint, refund_pk_hex: string): string;

export function build_multisig_2of2_address(pk1_hex: string, pk2_hex: string): string;

/**
 * Compute the coin ID (UTXO identifier) from an address, value, and salt.
 *
 * `coin_id = BLAKE3(address || value_le_bytes || salt)`
 *
 * All inputs and output are hex-encoded strings.
 *
 * # Returns
 *
 * A 64-character hex string representing the 32-byte coin ID.
 *
 * # Edge Cases
 *
 * If `address_hex` or `salt_hex` are invalid hex (wrong length, bad chars),
 * the corresponding bytes default to all zeros. This matches the CLI behavior
 * where malformed inputs produce deterministic (but useless) outputs rather
 * than panicking.
 */
export function compute_coin_id_hex(address_hex: string, value: bigint, salt_hex: string): string;

/**
 * Compute a transaction commitment hash directly from WASM.
 *
 * # Formal Specification
 * ```text
 * Pre:  input_ids_json and output_hashes_json are valid JSON arrays of 64-char hex strings.
 *       salt_hex is a valid 64-character hex string.
 * Post: result = BLAKE3(MAGIC || len(inputs) || inputs || len(outputs) || outputs || salt)
 * ```
 */
export function compute_commitment_hex(input_ids_json: string, output_hashes_json: string, salt_hex: string): string;

export function compute_p2pk_address_hex(owner_pk_hex: string): string;

/**
 * Decompose an amount into canonical power-of-2 denominations.
 *
 * The Midstate UTXO model requires all coin values to be exact powers of 2.
 * This function splits any amount into the minimal set of such denominations.
 *
 * # Example
 *
 * An input of `13` yields `[1, 4, 8]` (three coins: 2^0 + 2^2 + 2^3 = 13).
 */
export function decompose_amount(amount: bigint): BigUint64Array;

/**
 * Decrypt a native CLI wallet file (`.dat`) using its password.
 *
 * Returns the decrypted JSON string containing the full wallet data
 * (master_seed, coins, mss_keys, history, etc.).
 *
 * # Errors
 *
 * Returns `Err` if decryption fails (wrong password) or the decrypted
 * data is not valid UTF-8.
 */
export function decrypt_cli_wallet(data: Uint8Array, password: string): string;

/**
 * Generate a new BIP39 24-word mnemonic phrase.
 *
 * Returns the phrase as a space-separated string. The corresponding master
 * seed is derived when the phrase is passed to [`WebWallet::new`].
 *
 * # Panics
 *
 * Panics if the system CSPRNG fails (should never happen in a browser).
 */
export function generate_phrase(): string;

/**
 * Mine the Anti-Spam Proof of Work for a P2P Chat Message directly in the browser.
 *
 * # Reasoning
 * Pushing PoW to the client prevents node CPU exhaustion and enables true
 * decentralized P2P dApps (like L2 Lightning Hubs) over WebRTC without relying
 * on central RPC servers to mine on the user's behalf.
 *
 * # Formal Specification
 * ```text
 * Pre:  sender is a valid PeerId string
 *       words_json is a JSON array of u8 (0-255)
 *       attachments_json is a JSON array of valid ChatAttachment objects
 * Post: result = Ok(nonce) where verify_chat_pow_v2(..., nonce) == true
 * ```
 */
export function mine_chat_pow_v2_wasm(sender: string, timestamp: bigint, reply_to_json: string, words_json: string, attachments_json: string): bigint;

/**
 * Safely mines the Commitment PoW in the WebAssembly context.
 *
 * # Reasoning
 * Intercepts invalid or missing hex strings (e.g., from an out-of-sync RPC cache)
 * and handles them gracefully. Replacing `.unwrap()` with silent fallbacks prevents
 * the Web Worker from panicking and permanently hanging the UI on "Mining PoW...".
 *
 * # Formal Specification
 * ```text
 * Pre:  true
 * Post: result = mine_pow(commitment, required_pow, target_height, header_hash) if hex valid
 *       result = 0 if commitment_hex invalid
 * ```
 *
 * ```zed
 *     MineCommitmentPowWasm
 *     ---------------------
 *     commitment_hex? : String
 *     required_pow? : ℕ₃₂
 *     target_height? : ℕ₆₄
 *     header_hash_hex? : String
 *     nonce! : ℕ₆₄
 *
 *     pre  true
 *     post (isHex32(commitment_hex?) ⇒ nonce! = MinePow(...))
 *        ∧ (¬isHex32(commitment_hex?) ⇒ nonce! = 0)
 * ```
 */
export function mine_commitment_pow(commitment_hex: string, required_pow: number, target_height: bigint, header_hash_hex: string): bigint;

/**
 * Cooperative / unilateral-receiver close reveal.
 * Witness per input: [sender_sig, receiver_sig, 0x01].
 */
export function qbolt_build_close_reveal(sender_pk_hex: string, receiver_pk_hex: string, expiry: bigint, funding_json: string, state_json: string, sender_sig_hex: string, receiver_sig_hex: string): string;

export function qbolt_build_legacy_close_reveal(alice_pk_hex: string, bob_pk_hex: string, funding_json: string, state_json: string, alice_sig_hex: string, bob_sig_hex: string): string;

/**
 * LEGACY RESCUE — v1 channels funded a bare 2-of-2 with NO timeout branch,
 * and the v1 close never committed its commitment on-chain (so it could not
 * confirm) and assumed a single funding coin (funding was actually split
 * into power-of-2 denominations). These two builders produce a CORRECT
 * cooperative close for that legacy covenant: multi-coin aware, and meant
 * to be driven through the full commit → reveal engine. Both parties must
 * still cooperate — a bare 2-of-2 has no unilateral path, ever.
 */
export function qbolt_build_legacy_close_state(channel_id_hex: string, alice_pk_hex: string, bob_pk_hex: string, funding_json: string, alice_amt: bigint, bob_amt: bigint, attempt: number): string;

/**
 * Sender's post-expiry refund reveal.
 * Witness per input: [sender_sig, 0x00].
 */
export function qbolt_build_refund_reveal(sender_pk_hex: string, receiver_pk_hex: string, expiry: bigint, funding_json: string, state_json: string, sender_sig_hex: string): string;

/**
 * Build the sender's post-expiry refund state: everything (minus fee) back
 * to the sender. Uses nonce = u32::MAX so its salts can never collide with
 * a payment state.
 */
export function qbolt_build_refund_state(channel_id_hex: string, sender_pk_hex: string, receiver_pk_hex: string, expiry: bigint, funding_json: string, attempt: number): string;

/**
 * Build the canonical close state for a Q-Bolt v2 channel.
 * `channel_id_hex` is the channel's stable identifier (the lexicographically
 * smallest funding coin id) — used only for salt derivation.
 */
export function qbolt_build_state(channel_id_hex: string, sender_pk_hex: string, receiver_pk_hex: string, expiry: bigint, funding_json: string, sender_amt: bigint, receiver_amt: bigint, nonce: number, htlcs_json: string, attempt: number): string;

export function qbolt_channel_address(sender_pk_hex: string, receiver_pk_hex: string, expiry: bigint): string;

export function qbolt_channel_bytecode_hex(sender_pk_hex: string, receiver_pk_hex: string, expiry: bigint): string;

/**
 * Search a range of nonces for a valid block extension hash.
 *
 * This is the inner loop of the browser-based solo miner. Each call tests
 * `iterations × 4` nonces using SIMD 4-way parallelism (WASM SIMD128).
 *
 * # Arguments
 *
 * * `midstate_hex` — 64-char hex of the block header midstate.
 * * `target_hex` — 64-char hex of the difficulty target (big-endian).
 * * `start_nonce` — first nonce to test.
 * * `iterations` — number of SIMD batches (each batch = 4 nonces).
 *
 * # Returns
 *
 * `Some(winning_nonce)` if a hash below target was found, `None` otherwise.
 *
 * # Performance
 *
 * With WASM SIMD128 enabled, each call with `iterations=1` takes ~800ms
 * due to the expensive iterated hashing (EXTENSION_ITERATIONS = 1,000,000).
 */
export function search_nonces(midstate_hex: string, target_hex: string, start_nonce: bigint, iterations: number): bigint | undefined;

export function verify_mss_sig_wasm(sig_hex: string, msg_hex: string, pk_hex: string): boolean;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_webwallet_free: (a: number, b: number) => void;
    readonly blake3_hash_hex: (a: number, b: number, c: number) => void;
    readonly build_channel_reveal: (a: number, b: bigint, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => void;
    readonly build_channel_state: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: bigint, i: bigint, j: number, k: number, l: number) => void;
    readonly build_covenant_htlc_bytecode_hex: (a: number, b: number, c: number, d: number, e: number, f: bigint, g: bigint, h: number, i: number) => void;
    readonly build_htlc_bytecode_hex: (a: number, b: number, c: number, d: number, e: number, f: bigint, g: number, h: number) => void;
    readonly build_limit_order_covenant_bytecode_hex: (a: number, b: number, c: number, d: bigint, e: bigint, f: number, g: number) => void;
    readonly build_multisig_2of2_address: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly compute_coin_id_hex: (a: number, b: number, c: number, d: bigint, e: number, f: number) => void;
    readonly compute_commitment_hex: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly compute_p2pk_address_hex: (a: number, b: number, c: number) => void;
    readonly decrypt_cli_wallet: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly generate_phrase: (a: number) => void;
    readonly mine_chat_pow_v2_wasm: (a: number, b: number, c: number, d: bigint, e: number, f: number, g: number, h: number, i: number, j: number) => void;
    readonly mine_commitment_pow: (a: number, b: number, c: number, d: bigint, e: number, f: number) => bigint;
    readonly qbolt_build_close_reveal: (a: number, b: number, c: number, d: number, e: number, f: bigint, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => void;
    readonly qbolt_build_legacy_close_reveal: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => void;
    readonly qbolt_build_legacy_close_state: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: bigint, k: bigint, l: number) => void;
    readonly qbolt_build_refund_reveal: (a: number, b: number, c: number, d: number, e: number, f: bigint, g: number, h: number, i: number, j: number, k: number, l: number) => void;
    readonly qbolt_build_refund_state: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: bigint, i: number, j: number, k: number) => void;
    readonly qbolt_build_state: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: bigint, i: number, j: number, k: bigint, l: bigint, m: number, n: number, o: number, p: number) => void;
    readonly qbolt_channel_address: (a: number, b: number, c: number, d: number, e: number, f: bigint) => void;
    readonly qbolt_channel_bytecode_hex: (a: number, b: number, c: number, d: number, e: number, f: bigint) => void;
    readonly search_nonces: (a: number, b: number, c: number, d: number, e: number, f: bigint, g: number) => void;
    readonly verify_mss_sig_wasm: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly webwallet_build_coinbase: (a: number, b: number, c: bigint, d: number) => void;
    readonly webwallet_build_coinbase_to_mss: (a: number, b: number, c: bigint, d: number, e: number) => void;
    readonly webwallet_build_consolidate_reveal: (a: number, b: number, c: number, d: number) => void;
    readonly webwallet_build_reveal: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly webwallet_build_script_reveal: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly webwallet_build_solo_extension: (a: number, b: number, c: number, d: number, e: bigint) => void;
    readonly webwallet_build_state_thread_tx: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number) => void;
    readonly webwallet_check_filter: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly webwallet_export_mss_bytes: (a: number, b: number, c: number, d: number) => void;
    readonly webwallet_from_seed_hex: (a: number, b: number, c: number) => void;
    readonly webwallet_get_mss_address: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly webwallet_get_mss_pubkey: (a: number, b: number, c: number, d: number) => void;
    readonly webwallet_get_wots_address: (a: number, b: number, c: number) => void;
    readonly webwallet_has_mss_cache: (a: number, b: number, c: number) => number;
    readonly webwallet_import_mss_bytes: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly webwallet_new: (a: number, b: number, c: number) => void;
    readonly webwallet_prepare_consolidate: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly webwallet_prepare_defrag: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly webwallet_prepare_fund_many: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
    readonly webwallet_prepare_fund_tx: (a: number, b: number, c: number, d: number, e: number, f: number, g: bigint, h: number, i: number, j: number) => void;
    readonly webwallet_prepare_script_spend: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => void;
    readonly webwallet_prepare_spend: (a: number, b: number, c: number, d: number, e: number, f: number, g: bigint, h: number, i: number, j: number, k: number, l: bigint) => void;
    readonly webwallet_set_mss_leaf_index: (a: number, b: number, c: number, d: number) => void;
    readonly webwallet_set_watchlist: (a: number, b: number, c: number) => void;
    readonly webwallet_sign_mss_hex: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly decompose_amount: (a: bigint) => number;
    readonly __wbindgen_export: (a: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export2: (a: number, b: number) => number;
    readonly __wbindgen_export3: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
