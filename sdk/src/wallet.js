import initWasm, { WebWallet, generate_phrase, mine_commitment_pow, compute_coin_id_hex } from '../pkg/wasm_wallet.js';
import { MemoryStorage } from './storage.js';

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

export class Wallet {
    static async init(wasmPathOrBuffer) {
        // Initializes the WASM runtime.
        // Passing it as an object fixes the wasm-bindgen deprecation warning.
        await initWasm({ module_or_path: wasmPathOrBuffer });
    }

    static async create(storageProvider = new MemoryStorage()) {
        const phrase = generate_phrase();
        const wallet = new Wallet(phrase, storageProvider);
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
    }

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
            lastScannedHeight: this.lastScannedHeight
        }));
    }

    async getNewAddress() {
        const addr = this.inner.get_wots_address(this.nextWotsIndex);
        this.wotsAddrs[addr] = this.nextWotsIndex++;
        await this.save();
        return addr;
    }

    async getNewReusableAddress(height = 10) {
        const addr = this.inner.get_mss_address(this.nextMssIndex, height);
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
    addUtxo(address, value, salt, coinId) {
        let is_mss = false, index = 0, mss_height = 0, mss_leaf = 0;
        if (this.wotsAddrs[address] !== undefined) {
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
            this.utxos.push({ address, value: BigInt(value), salt, coin_id: coinId, index, is_mss, mss_height, mss_leaf });
            return true;
        }
        return false;
    }

    /** @returns {bigint} total spendable balance in base units (MDS). */
    getBalance() {
        return this.utxos.reduce((sum, u) => sum + BigInt(u.value), 0n);
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
    _setWatchlist() {
        const watch = [
            ...Object.keys(this.wotsAddrs),
            ...Object.keys(this.mssAddrs),
            ...this.utxos.map(u => u.coin_id),
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
    _processBlock(block) {
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

        // Transaction reveals: spent inputs first, then created outputs.
        if (Array.isArray(block.transactions)) {
            for (const tx of block.transactions) {
                const reveal = tx.Reveal || tx.reveal;
                if (!reveal) continue;

                if (Array.isArray(reveal.inputs)) {
                    for (const inp of reveal.inputs) {
                        const saltHex = normalizeHex(inp.salt);
                        const cid = ourSalts.get(saltHex);
                        if (cid) {
                            this.utxos = this.utxos.filter(u => u.coin_id !== cid);
                            ourSalts.delete(saltHex);
                            matched = true;
                        }
                    }
                }

                if (Array.isArray(reveal.outputs)) {
                    for (const out of reveal.outputs) {
                        const outData = out.Standard || out.standard;
                        if (!outData) continue;
                        const addrHex = normalizeHex(outData.address);
                        const saltHex = normalizeHex(outData.salt);
                        if (this.wotsAddrs[addrHex] !== undefined || this.mssAddrs[addrHex] !== undefined) {
                            const coinId = compute_coin_id_hex(addrHex, BigInt(outData.value), saltHex);
                            if (this.addUtxo(addrHex, outData.value, saltHex, coinId)) ourSalts.set(saltHex, coinId);
                            matched = true;
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
     *   block_template share one budget) to ~120/60s for a fresh peer, and each
     *   rejection counts as a violation — 10 violations triggers a 5-minute ban.
     *   Pacing at >500ms keeps a full-chain scan under that budget so it never
     *   trips the limiter. Set to 0 to disable pacing (only safe for short scans).
     * @param {number}   [opts.rateLimitCooldownMs=62000] If the limiter still fires
     *   (e.g. reduced reputation), wait roughly one window before retrying.
     * @param {number}   [opts.maxRateLimitRetries=6] Cap on consecutive cooldowns
     *   per batch (stays well under the 10-violation ban threshold).
     * @returns {Promise<{height:number, found:number, balance:number, utxos:number}>}
     */
    async sync(client, {
        gapLimit = 0,
        batchSize = 1000,
        rescan = false,
        onProgress,
        filterIntervalMs = 550,
        rateLimitCooldownMs = 62_000,
        maxRateLimitRetries = 6,
    } = {}) {
        if (rescan) {
            this.utxos = [];
            this.lastScannedHeight = 0;
        }

        if (gapLimit > 0) {
            const target = this.nextWotsIndex + gapLimit;
            while (this.nextWotsIndex < target) {
                const addr = this.inner.get_wots_address(this.nextWotsIndex);
                this.wotsAddrs[addr] = this.nextWotsIndex++;
            }
            await this.save();
        }

        const state = await client.getState();
        const chainHeight = state.height;

        if (chainHeight <= this.lastScannedHeight) {
            return { height: chainHeight, found: 0, balance: this.getBalance(), utxos: this.utxos.length };
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

        let current = this.lastScannedHeight;
        let found = 0;

        while (current < chainHeight) {
            const end = Math.min(current + batchSize, chainHeight);
            const filterData = await fetchFilters(current, end);
            const filters = filterData.filters || [];
            const counts = filterData.element_counts || [];
            const hashes = filterData.block_hashes || [];
            const startH = filterData.start_height ?? current;
            const numFilters = filters.length;

            for (let i = 0; i < numFilters; i++) {
                const height = startH + i;
                const n = counts[i] || 0;
                const blockHash = hashes[i];

                let fetch = false;
                if (n === 0) {
                    fetch = false;                       // empty block — nothing to match
                } else if (!blockHash) {
                    fetch = true;                        // node gave no hash → can't filter, fetch
                } else if (this.inner.check_filter(filters[i], blockHash, n)) {
                    fetch = true;                        // filter hit (may be a false positive)
                }

                if (fetch) {
                    const mutated = this._processBlock(await client.getBlock(height));
                    if (mutated) { found++; this._setWatchlist(); }
                }

                if (onProgress && height % 100 === 0) {
                    onProgress({ height, chainHeight, balance: this.getBalance() });
                }
            }

            current += numFilters;

            // If the node returned fewer filters than the requested span (some
            // blocks have no filter), fetch the remainder of this batch directly.
            while (current < end) {
                const mutated = this._processBlock(await client.getBlock(current));
                if (mutated) { found++; this._setWatchlist(); }
                current++;
            }

            // Checkpoint after each fully-processed batch so progress survives an
            // interruption and a re-run resumes instead of rescanning from genesis.
            this.lastScannedHeight = current;
            await this.save();
        }

        if (onProgress) onProgress({ height: chainHeight, chainHeight, balance: this.getBalance() });

        return { height: chainHeight, found, balance: this.getBalance(), utxos: this.utxos.length };
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Transaction Execution
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Internal helper to execute the 2-phase Commit/Reveal protocol.
     * @private
     */
    async _broadcastTwoPhaseTx(client, spendCtxStr, isScript = false) {
        const ctx = JSON.parse(spendCtxStr);
        
        // --- ADD CHANGE ADDRESSES TO WATCHLIST ---
        while (this.nextWotsIndex < ctx.next_wots_index) {
            const addr = this.inner.get_wots_address(this.nextWotsIndex);
            this.wotsAddrs[addr] = this.nextWotsIndex;
            this.nextWotsIndex++;
        }

        // Advance MSS indices for any used inputs
        const inputs = ctx.selected_inputs || ctx.wallet_inputs || [];
        for (const inp of inputs) {
            if (inp.is_mss && this.mssAddrs[inp.address]) {
                this.mssAddrs[inp.address].next_leaf++;
            }
        }
        await this.save();

        const state = await client.getState();
        const requiredPow = state.required_pow || 24;

        // 1. Mine PoW
        const spamNonce = mine_commitment_pow(
            ctx.commitment, 
            requiredPow, 
            BigInt(state.height), 
            state.header_hash
        );

        // 2. Commit
        const commitReq = await client.commit(ctx.commitment, Number(spamNonce));
        if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);

        // 3. Wait for confirmation
        let mined = false;
        for (let i = 0; i < 24; i++) {
            await new Promise(r => setTimeout(r, 5000));
            const res = await client.checkCommitment(ctx.commitment).catch(() => ({}));
            if (res?.exists) { mined = true; break; }
        }
        if (!mined) throw new Error("Timed out waiting for Commit to be mined.");

        // 4. Reveal
        let revealPayloadStr;
        if (isScript) {
            revealPayloadStr = this.inner.build_script_reveal(spendCtxStr, ctx.commitment, ctx.tx_salt);
        } else {
            revealPayloadStr = this.inner.build_reveal(spendCtxStr, ctx.commitment, ctx.tx_salt);
        }
        
        const response = await client.send(revealPayloadStr);

        // 5. Cleanup Local State
        const spentIds = inputs.map(i => i.coin_id);
        this.utxos = this.utxos.filter(u => !spentIds.includes(u.coin_id));
        await this.save();

        return response;
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

        const utxosForWasm = this.utxos.map(u => ({ ...u, value: Number(u.value) }));
        const spendCtxStr = this.inner.prepare_spend(
            JSON.stringify(utxosForWasm),
            toAddressHex,
            amount,
            this.nextWotsIndex,
            burnDataHex,
            burnDataHex ? bValue : null
        );
        
        return this._broadcastTwoPhaseTx(client, spendCtxStr, false);
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
        const utxosForWasm = this.utxos.map(u => ({ ...u, value: Number(u.value) }));
        
        const ctxStr = this.inner.prepare_fund_tx(
            JSON.stringify(utxosForWasm),
            contractAddrHex,
            amount,
            stateHex,
            this.nextWotsIndex
        );

        return this._broadcastTwoPhaseTx(client, ctxStr, true);
    }

    /**
     * Execute a Smart Contract transaction.
     * @param {MidstateClient} client
     * @param {string} contractBytecodeHex - The raw compiled contract bytecode
     * @param {Array<Object>} contractInputsArray - Array of { coin_id, witness, value, salt, state }
     * @param {Array<Object>} outputsArray - Array of { out_type: "standard"|"confidential", address, value, state, salt }
     */
    async executeContract(client, contractBytecodeHex, contractInputsArray, outputsArray) {
        const utxosForWasm = this.utxos.map(u => ({ ...u, value: Number(u.value) }));
        
        const ctxStr = this.inner.prepare_script_spend(
            JSON.stringify(utxosForWasm),
            contractBytecodeHex,
            JSON.stringify(contractInputsArray),
            JSON.stringify(outputsArray),
            this.nextWotsIndex
        );

        return this._broadcastTwoPhaseTx(client, ctxStr, true);
    }
}
