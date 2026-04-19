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
     * Select coins and build a transaction for the given send amount.
     *
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
    prepare_spend(available_utxos_json: string, to_address_hex: string, send_amount: bigint, next_wots_index: number): string;
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
}

/**
 * Hash a hex-encoded byte string with BLAKE3.
 * Returns the 32-byte hash as a 64-character hex string.
 * Used by the IDE to generate P2SH addresses.
 */
export function blake3_hash_hex(hex_data: string): string;

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
 * Mine a spam-proof PoW nonce for a transaction commitment.
 *
 * Searches sequentially for a nonce `n` such that:
 *   `leading_zeros(BLAKE3(commitment || n_le_bytes)) >= required_pow`
 *
 * This is a CPU-bound loop that runs synchronously. At difficulty 24,
 * it typically takes 0.5–5 seconds in WASM SIMD.
 *
 * # Arguments
 *
 * * `commitment_hex` — 64-char hex string of the 32-byte commitment hash.
 * * `required_pow` — minimum number of leading zero bits required.
 *
 * # Panics
 *
 * Panics if `commitment_hex` is not exactly 64 valid hex characters.
 */
export function mine_commitment_pow(commitment_hex: string, required_pow: number, target_height: bigint): bigint;

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

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_webwallet_free: (a: number, b: number) => void;
    readonly blake3_hash_hex: (a: number, b: number, c: number) => void;
    readonly compute_coin_id_hex: (a: number, b: number, c: number, d: bigint, e: number, f: number) => void;
    readonly decrypt_cli_wallet: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly generate_phrase: (a: number) => void;
    readonly mine_commitment_pow: (a: number, b: number, c: number, d: bigint) => bigint;
    readonly search_nonces: (a: number, b: number, c: number, d: number, e: number, f: bigint, g: number) => void;
    readonly webwallet_build_coinbase: (a: number, b: number, c: bigint, d: number) => void;
    readonly webwallet_build_reveal: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly webwallet_build_solo_extension: (a: number, b: number, c: number, d: number, e: bigint) => void;
    readonly webwallet_build_state_thread_tx: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number) => void;
    readonly webwallet_check_filter: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly webwallet_export_mss_bytes: (a: number, b: number, c: number, d: number) => void;
    readonly webwallet_from_seed_hex: (a: number, b: number, c: number) => void;
    readonly webwallet_get_mss_address: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly webwallet_get_wots_address: (a: number, b: number, c: number) => void;
    readonly webwallet_has_mss_cache: (a: number, b: number, c: number) => number;
    readonly webwallet_import_mss_bytes: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly webwallet_new: (a: number, b: number, c: number) => void;
    readonly webwallet_prepare_spend: (a: number, b: number, c: number, d: number, e: number, f: number, g: bigint, h: number) => void;
    readonly webwallet_set_mss_leaf_index: (a: number, b: number, c: number, d: number) => void;
    readonly webwallet_set_watchlist: (a: number, b: number, c: number) => void;
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
