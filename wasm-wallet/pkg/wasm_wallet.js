/* @ts-self-types="./wasm_wallet.d.ts" */

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
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WebWallet.prototype);
        obj.__wbg_ptr = ptr;
        WebWalletFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WebWalletFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_webwallet_free(ptr, 0);
    }
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
     * @param {bigint} total_value
     * @param {number} next_wots_index
     * @returns {string | undefined}
     */
    build_coinbase(total_value, next_wots_index) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.webwallet_build_coinbase(retptr, this.__wbg_ptr, total_value, next_wots_index);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            let v1;
            if (r0 !== 0) {
                v1 = getStringFromWasm0(r0, r1).slice();
                wasm.__wbindgen_export4(r0, r1 * 1, 1);
            }
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
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
     * @param {string} spend_context_json
     * @param {string} server_commitment_hex
     * @param {string} server_salt_hex
     * @returns {string}
     */
    build_reveal(spend_context_json, server_commitment_hex, server_salt_hex) {
        let deferred5_0;
        let deferred5_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(spend_context_json, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passStringToWasm0(server_commitment_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
            const len1 = WASM_VECTOR_LEN;
            const ptr2 = passStringToWasm0(server_salt_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
            const len2 = WASM_VECTOR_LEN;
            wasm.webwallet_build_reveal(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            var ptr4 = r0;
            var len4 = r1;
            if (r3) {
                ptr4 = 0; len4 = 0;
                throw takeObject(r2);
            }
            deferred5_0 = ptr4;
            deferred5_1 = len4;
            return getStringFromWasm0(ptr4, len4);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export4(deferred5_0, deferred5_1, 1);
        }
    }
    /**
     * Recompute the block extension hash for a found nonce.
     *
     * Called after a mining worker finds a valid nonce. Produces the full
     * `Extension { nonce, final_hash }` JSON needed for block submission.
     *
     * # Returns
     *
     * `None` if `midstate_hex` is not valid 64-character hex.
     * @param {string} midstate_hex
     * @param {bigint} nonce
     * @returns {string | undefined}
     */
    build_solo_extension(midstate_hex, nonce) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(midstate_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            wasm.webwallet_build_solo_extension(retptr, this.__wbg_ptr, ptr0, len0, nonce);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            let v2;
            if (r0 !== 0) {
                v2 = getStringFromWasm0(r0, r1).slice();
                wasm.__wbindgen_export4(r0, r1 * 1, 1);
            }
            return v2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
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
     * @param {string} filter_hex
     * @param {string} block_hash_hex
     * @param {number} n
     * @returns {boolean}
     */
    check_filter(filter_hex, block_hash_hex, n) {
        const ptr0 = passStringToWasm0(filter_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(block_hash_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.webwallet_check_filter(this.__wbg_ptr, ptr0, len0, ptr1, len1, n);
        return ret !== 0;
    }
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
     * @param {string} address_hex
     * @returns {Uint8Array}
     */
    export_mss_bytes(address_hex) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(address_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            wasm.webwallet_export_mss_bytes(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            if (r3) {
                throw takeObject(r2);
            }
            var v2 = getArrayU8FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export4(r0, r1 * 1, 1);
            return v2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Create a wallet from a raw 32-byte master seed (hex-encoded).
     *
     * Used when importing a CLI wallet backup where the master seed is
     * available directly rather than via a mnemonic phrase.
     *
     * # Errors
     *
     * Returns `Err` if `seed_hex` is not exactly 64 valid hex characters.
     * @param {string} seed_hex
     * @returns {WebWallet}
     */
    static from_seed_hex(seed_hex) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(seed_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            wasm.webwallet_from_seed_hex(retptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return WebWallet.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
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
     * @param {number} index
     * @param {number} height
     * @param {Function | null} [progress_cb]
     * @returns {string}
     */
    get_mss_address(index, height, progress_cb) {
        let deferred2_0;
        let deferred2_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.webwallet_get_mss_address(retptr, this.__wbg_ptr, index, height, isLikeNone(progress_cb) ? 0 : addHeapObject(progress_cb));
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            var ptr1 = r0;
            var len1 = r1;
            if (r3) {
                ptr1 = 0; len1 = 0;
                throw takeObject(r2);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export4(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Derive the WOTS address at a given HD index.
     *
     * Returns the hex-encoded 32-byte address. This is a pure computation
     * with no side effects — the address is not cached.
     * @param {number} index
     * @returns {string}
     */
    get_wots_address(index) {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.webwallet_get_wots_address(retptr, this.__wbg_ptr, index);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export4(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Check whether an MSS tree is loaded in the WASM-side cache.
     *
     * Returns `true` if the tree is ready for signing, `false` if it
     * needs to be loaded from IndexedDB or regenerated.
     * @param {string} address_hex
     * @returns {boolean}
     */
    has_mss_cache(address_hex) {
        const ptr0 = passStringToWasm0(address_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.webwallet_has_mss_cache(this.__wbg_ptr, ptr0, len0);
        return ret !== 0;
    }
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
     * @param {string} address_hex
     * @param {Uint8Array} data
     */
    import_mss_bytes(address_hex, data) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(address_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArray8ToWasm0(data, wasm.__wbindgen_export2);
            const len1 = WASM_VECTOR_LEN;
            wasm.webwallet_import_mss_bytes(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
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
     * @param {string} phrase
     */
    constructor(phrase) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(phrase, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            wasm.webwallet_new(retptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            WebWalletFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
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
     * @param {string} available_utxos_json
     * @param {string} to_address_hex
     * @param {bigint} send_amount
     * @param {number} next_wots_index
     * @returns {string}
     */
    prepare_spend(available_utxos_json, to_address_hex, send_amount, next_wots_index) {
        let deferred4_0;
        let deferred4_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(available_utxos_json, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passStringToWasm0(to_address_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
            const len1 = WASM_VECTOR_LEN;
            wasm.webwallet_prepare_spend(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1, send_amount, next_wots_index);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            var ptr3 = r0;
            var len3 = r1;
            if (r3) {
                ptr3 = 0; len3 = 0;
                throw takeObject(r2);
            }
            deferred4_0 = ptr3;
            deferred4_1 = len3;
            return getStringFromWasm0(ptr3, len3);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export4(deferred4_0, deferred4_1, 1);
        }
    }
    /**
     * Update the next-leaf counter for an MSS tree.
     *
     * Called by the JS layer after loading wallet state to synchronize
     * the WASM-side leaf counter with the persisted value.
     *
     * # No-op
     *
     * Silently does nothing if the address is not in the cache.
     * @param {string} address_hex
     * @param {number} leaf_index
     */
    set_mss_leaf_index(address_hex, leaf_index) {
        const ptr0 = passStringToWasm0(address_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        wasm.webwallet_set_mss_leaf_index(this.__wbg_ptr, ptr0, len0, leaf_index);
    }
    /**
     * Set the list of addresses the wallet watches during chain scanning.
     *
     * `addrs_json` is a JSON array of hex-encoded 32-byte addresses.
     * Replaces the entire watchlist. Invalid hex entries are silently skipped.
     * @param {string} addrs_json
     */
    set_watchlist(addrs_json) {
        const ptr0 = passStringToWasm0(addrs_json, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        wasm.webwallet_set_watchlist(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) WebWallet.prototype[Symbol.dispose] = WebWallet.prototype.free;

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
 * @param {string} address_hex
 * @param {bigint} value
 * @param {string} salt_hex
 * @returns {string}
 */
export function compute_coin_id_hex(address_hex, value, salt_hex) {
    let deferred3_0;
    let deferred3_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passStringToWasm0(address_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(salt_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
        const len1 = WASM_VECTOR_LEN;
        wasm.compute_coin_id_hex(retptr, ptr0, len0, value, ptr1, len1);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred3_0 = r0;
        deferred3_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export4(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Decompose an amount into canonical power-of-2 denominations.
 *
 * The Midstate UTXO model requires all coin values to be exact powers of 2.
 * This function splits any amount into the minimal set of such denominations.
 *
 * # Example
 *
 * An input of `13` yields `[1, 4, 8]` (three coins: 2^0 + 2^2 + 2^3 = 13).
 * @param {bigint} amount
 * @returns {BigUint64Array}
 */
export function decompose_amount(amount) {
    const ret = wasm.decompose_amount(amount);
    return takeObject(ret);
}

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
 * @param {Uint8Array} data
 * @param {string} password
 * @returns {string}
 */
export function decrypt_cli_wallet(data, password) {
    let deferred4_0;
    let deferred4_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passArray8ToWasm0(data, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(password, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
        const len1 = WASM_VECTOR_LEN;
        wasm.decrypt_cli_wallet(retptr, ptr0, len0, ptr1, len1);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
        var ptr3 = r0;
        var len3 = r1;
        if (r3) {
            ptr3 = 0; len3 = 0;
            throw takeObject(r2);
        }
        deferred4_0 = ptr3;
        deferred4_1 = len3;
        return getStringFromWasm0(ptr3, len3);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export4(deferred4_0, deferred4_1, 1);
    }
}

/**
 * Generate a new BIP39 24-word mnemonic phrase.
 *
 * Returns the phrase as a space-separated string. The corresponding master
 * seed is derived when the phrase is passed to [`WebWallet::new`].
 *
 * # Panics
 *
 * Panics if the system CSPRNG fails (should never happen in a browser).
 * @returns {string}
 */
export function generate_phrase() {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.generate_phrase(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export4(deferred1_0, deferred1_1, 1);
    }
}

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
 * @param {string} commitment_hex
 * @param {number} required_pow
 * @returns {bigint}
 */
export function mine_commitment_pow(commitment_hex, required_pow) {
    const ptr0 = passStringToWasm0(commitment_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.mine_commitment_pow(ptr0, len0, required_pow);
    return BigInt.asUintN(64, ret);
}

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
 * @param {string} midstate_hex
 * @param {string} target_hex
 * @param {bigint} start_nonce
 * @param {number} iterations
 * @returns {bigint | undefined}
 */
export function search_nonces(midstate_hex, target_hex, start_nonce, iterations) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passStringToWasm0(midstate_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(target_hex, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
        const len1 = WASM_VECTOR_LEN;
        wasm.search_nonces(retptr, ptr0, len0, ptr1, len1, start_nonce, iterations);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r2 = getDataViewMemory0().getBigInt64(retptr + 8 * 1, true);
        return r0 === 0 ? undefined : BigInt.asUintN(64, r2);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_is_function_3c846841762788c1: function(arg0) {
            const ret = typeof(getObject(arg0)) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_object_781bc9f159099513: function(arg0) {
            const val = getObject(arg0);
            const ret = typeof(val) === 'object' && val !== null;
            return ret;
        },
        __wbg___wbindgen_is_string_7ef6b97b02428fae: function(arg0) {
            const ret = typeof(getObject(arg0)) === 'string';
            return ret;
        },
        __wbg___wbindgen_is_undefined_52709e72fb9f179c: function(arg0) {
            const ret = getObject(arg0) === undefined;
            return ret;
        },
        __wbg___wbindgen_throw_6ddd609b62940d55: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_call_2d781c1f4d5c0ef8: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = getObject(arg0).call(getObject(arg1), getObject(arg2));
            return addHeapObject(ret);
        }, arguments); },
        __wbg_call_dcc2662fa17a72cf: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            const ret = getObject(arg0).call(getObject(arg1), getObject(arg2), getObject(arg3));
            return addHeapObject(ret);
        }, arguments); },
        __wbg_crypto_38df2bab126b63dc: function(arg0) {
            const ret = getObject(arg0).crypto;
            return addHeapObject(ret);
        },
        __wbg_getRandomValues_c44a50d8cfdaebeb: function() { return handleError(function (arg0, arg1) {
            getObject(arg0).getRandomValues(getObject(arg1));
        }, arguments); },
        __wbg_length_ea16607d7b61445b: function(arg0) {
            const ret = getObject(arg0).length;
            return ret;
        },
        __wbg_msCrypto_bd5a034af96bcba6: function(arg0) {
            const ret = getObject(arg0).msCrypto;
            return addHeapObject(ret);
        },
        __wbg_new_from_slice_14158c9615ed2369: function(arg0, arg1) {
            const ret = new BigUint64Array(getArrayU64FromWasm0(arg0, arg1));
            return addHeapObject(ret);
        },
        __wbg_new_with_length_825018a1616e9e55: function(arg0) {
            const ret = new Uint8Array(arg0 >>> 0);
            return addHeapObject(ret);
        },
        __wbg_node_84ea875411254db1: function(arg0) {
            const ret = getObject(arg0).node;
            return addHeapObject(ret);
        },
        __wbg_process_44c7a14e11e9f69e: function(arg0) {
            const ret = getObject(arg0).process;
            return addHeapObject(ret);
        },
        __wbg_prototypesetcall_d62e5099504357e6: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), getObject(arg2));
        },
        __wbg_randomFillSync_6c25eac9869eb53c: function() { return handleError(function (arg0, arg1) {
            getObject(arg0).randomFillSync(takeObject(arg1));
        }, arguments); },
        __wbg_require_b4edbdcf3e2a1ef0: function() { return handleError(function () {
            const ret = module.require;
            return addHeapObject(ret);
        }, arguments); },
        __wbg_static_accessor_GLOBAL_8adb955bd33fac2f: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_ad356e0db91c7913: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_static_accessor_SELF_f207c857566db248: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_static_accessor_WINDOW_bb9f1ba69d61b386: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_subarray_a068d24e39478a8a: function(arg0, arg1, arg2) {
            const ret = getObject(arg0).subarray(arg1 >>> 0, arg2 >>> 0);
            return addHeapObject(ret);
        },
        __wbg_versions_276b2795b1c6a219: function(arg0) {
            const ret = getObject(arg0).versions;
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000001: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000003: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return addHeapObject(ret);
        },
        __wbindgen_object_clone_ref: function(arg0) {
            const ret = getObject(arg0);
            return addHeapObject(ret);
        },
        __wbindgen_object_drop_ref: function(arg0) {
            takeObject(arg0);
        },
    };
    return {
        __proto__: null,
        "./wasm_wallet_bg.js": import0,
    };
}

const WebWalletFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_webwallet_free(ptr >>> 0, 1));

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function dropObject(idx) {
    if (idx < 1028) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function getArrayU64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getBigUint64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedBigUint64ArrayMemory0 = null;
function getBigUint64ArrayMemory0() {
    if (cachedBigUint64ArrayMemory0 === null || cachedBigUint64ArrayMemory0.byteLength === 0) {
        cachedBigUint64ArrayMemory0 = new BigUint64Array(wasm.memory.buffer);
    }
    return cachedBigUint64ArrayMemory0;
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getObject(idx) { return heap[idx]; }

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        wasm.__wbindgen_export(addHeapObject(e));
    }
}

let heap = new Array(1024).fill(undefined);
heap.push(undefined, null, true, false);

let heap_next = heap.length;

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedBigUint64ArrayMemory0 = null;
    cachedDataViewMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('wasm_wallet_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
