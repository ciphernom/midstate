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

import init, { WebWallet, generate_phrase, compute_coin_id_hex, decrypt_cli_wallet, mine_commitment_pow } from './pkg/wasm_wallet.js';

/** @type {WebWallet|null} The WASM wallet instance. Null until CREATE or LOGIN. */
let wallet = null;

/** @type {string|null} The user's password, held in memory for encrypting state saves. */
let password = null;

/** @type {boolean} Guard against concurrent send operations. */
let isSending = false;

/** @type {boolean} Guard against concurrent block submissions. */
let isSubmitting = false;

/** @type {boolean} Guard against concurrent template requests to prevent WebRTC stream exhaustion. */
let isFetchingTemplate = false;

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

/** @type {WalletState} */
let wState = {
    phrase: null,
    nextWotsIndex: 0,
    nextMssIndex: 0,
    wotsAddrs: {},
    mssAddrs: {},
    utxos: {},
    history: [],
    lastScannedHeight: 0
};

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

    const entries = Object.entries(wState.mssAddrs);
    if (entries.length === 0) { mssCachesReady = true; return; }

    self.postMessage({ type: 'LOG', payload: `Loading ${entries.length} MSS tree(s) from storage...` });

    for (const [addrHex, mss] of entries) {
        try {
            // Already in WASM memory (e.g., just generated during CREATE)
            if (wallet.has_mss_cache(addrHex)) {
                wallet.set_mss_leaf_index(addrHex, mss.next_leaf);
                continue;
            }

            // Try IndexedDB — instant load (~1ms for 64 KB)
            const treeBytes = await idbGet(`mss_${addrHex}`);
            if (treeBytes) {
                wallet.import_mss_bytes(addrHex, new Uint8Array(treeBytes));
                wallet.set_mss_leaf_index(addrHex, mss.next_leaf);
                self.postMessage({ type: 'LOG', payload: `MSS tree ${addrHex.substring(0,12)}… loaded from IndexedDB.` });
                continue;
            }

            // Not in IndexedDB — regenerate from seed (one-time migration cost)
            self.postMessage({ type: 'LOG', payload: `MSS tree ${addrHex.substring(0,12)}… not cached. Generating (one-time)...` });
            self.postMessage({ type: 'MSS_PROGRESS', payload: { current: 0, total: 100, label: "Regenerating MSS tree (one-time)..." } });

            const addr = wallet.get_mss_address(mss.index, mss.height, (current, total) => {
                const now = Date.now();
                if (now - lastMssUpdate > 66 || current === total) {
                    lastMssUpdate = now;
                    self.postMessage({ type: 'MSS_PROGRESS', payload: { current, total, label: `Regenerating MSS tree (${current}/${total})...` } });
                }
            });

            // Persist to IndexedDB so next load is instant
            const exportedBytes = wallet.export_mss_bytes(addr);
            await idbPut(`mss_${addr}`, exportedBytes);

            wallet.set_mss_leaf_index(addrHex, mss.next_leaf);
            self.postMessage({ type: 'LOG', payload: `MSS tree ${addrHex.substring(0,12)}… generated and cached.` });
        } catch (e) {
            console.warn("MSS load failed for", addrHex, e);
            self.postMessage({ type: 'LOG', payload: `Warning: MSS load failed for ${addrHex.substring(0,12)}…: ${e}` });
        }
    }

    mssCachesReady = true;
    self.postMessage({ type: 'LOG', payload: "All MSS trees loaded." });
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
function _rpcReceive(id, result, error) {
    const p = _rpcPending.get(id);
    if (!p) return;
    _rpcPending.delete(id);
    if (error !== undefined) p.reject(new Error(error));
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
    checkCoin:      (coin)       => rpcCall('checkCoin', { coinHex: coin }),

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
        wallet = new WebWallet(wState.phrase);

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
    
    // FIX: Prevent concurrent executions which would spam the node and trigger a peer ban
    if (isFetchingTemplate) return null;
    isFetchingTemplate = true;

    try {
        const stateObj = await rpc.getState();

        let mempoolTxs = 0, mempoolFees = 0;
        try {
            const mempool = await rpc.getMempool();
            mempoolTxs = mempool.size || 0;
            mempoolFees = (mempool.transactions || []).reduce((s, tx) => s + (tx.fee || 0), 0);
        } catch (e) {}

        if (stateObj.height > wState.lastScannedHeight) {
            self.postMessage({ type: 'LOG', payload: "Chain advanced! Auto-syncing..." });
            await performScan();
        }

        const template = await buildMiningTemplate(stateObj);
        if (!template) return null;

        const txCount = template.batch_template.transactions?.length || 0;
        self.postMessage({ type: 'LOG', payload: `Template at height ${stateObj.height} | ${txCount} txs | fees: ${template.total_fees}` });

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
        batch.timestamp = Math.floor(Date.now() / 1000);
        batch.extension = JSON.parse(extStr);

        for (const entry of template.mining_addrs) wState.wotsAddrs[entry.address] = entry.index;
        wState.nextWotsIndex = template.next_wots_index;

        const submitReq = await rpc.submitBatch(batch);
        const accepted = submitReq.ok;
        const rejectReason = accepted ? null : (submitReq.body || 'rejected');

        if (accepted) {
            self.postMessage({ type: 'LOG', payload: `✅ Block accepted! Height: ${template.chainHeight}` });
            await saveState();
            await performScan();
        } else {
            self.postMessage({ type: 'LOG', payload: `❌ Block rejected: ${rejectReason}` });
            await saveState();
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
            _rpcReceive(payload.id, payload.result, payload.error);
        }

        else if (type === 'GENERATE') {
            self.postMessage({ type: 'PHRASE_GENERATED', payload: generate_phrase() });
        }

        else if (type === 'CREATE') {
            if (wallet) wallet.free();
            password = payload.password;
            wState = {
                phrase: payload.phrase,
                nextWotsIndex: 0, nextMssIndex: 0,
                wotsAddrs: {}, mssAddrs: {}, utxos: {}, history: [],
                lastScannedHeight: 0
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
        }

        else if (type === 'SCAN') {
            await performScan();
        }

        else if (type === 'RESCAN') {
            wState.lastScannedHeight = 0;
            wState.utxos = {};
            wState.history = [];
            await saveState();
            await performScan();
        }

        else if (type === 'SEND') {
            if (isSending) throw new Error("A transaction is already in progress. Please wait for it to complete.");
            isSending = true;
            try { await performSend(payload.toAddress, payload.amount); }
            finally { isSending = false; }
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

        else if (type === 'IMPORT_CLI') {
            try {
                const cliJsonStr = decrypt_cli_wallet(payload.fileBytes, payload.password);
                const cliData    = JSON.parse(cliJsonStr);
                if (!cliData.master_seed) throw new Error("Legacy (non-HD) wallets not supported in Web.");

                let newUtxos = {};
                for (const coin of cliData.coins) {
                    newUtxos[normalizeHex(coin.coin_id)] = {
                        index: 0,
                        is_mss: cliData.mss_keys.some(k => normalizeHex(k.master_pk) === normalizeHex(coin.owner_pk)),
                        mss_height: 10, mss_leaf: 0,
                        address: normalizeHex(coin.address),
                        value:   coin.value,
                        salt:    normalizeHex(coin.salt),
                        coin_id: normalizeHex(coin.coin_id)
                    };
                }
                let newMssAddrs = {};
                for (const mss of cliData.mss_keys) {
                    newMssAddrs[normalizeHex(mss.master_pk)] = { index: 0, height: mss.height, next_leaf: mss.next_leaf };
                }
                wState = {
                    phrase: null,
                    nextWotsIndex: cliData.next_wots_index || 0,
                    nextMssIndex:  cliData.next_mss_index  || 0,
                    wotsAddrs: {}, mssAddrs: newMssAddrs, utxos: newUtxos,
                    history: cliData.history || [],
                    lastScannedHeight: cliData.last_scan_height || 0
                };
                wallet   = WebWallet.from_seed_hex(normalizeHex(cliData.master_seed));
                password = payload.password;
                mssCachesReady = false;
                await loadMssCaches();
                await saveState();
                self.postMessage({ type: 'WALLET_LOADED', payload: buildDashboardPayload() });
            } catch (err) {
                self.postMessage({ type: 'ERROR', payload: "Failed to import CLI wallet: Incorrect password or corrupt file." });
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
        balance: safeBalance,
        utxos:   utxoArray,
        history: sortedHistory
    };
}

/**
 * Update the WASM-side watchlist from current wallet state.
 * Includes all known WOTS addresses, MSS addresses, and UTXO coin IDs.
 */
function updateWasmWatchlist() {
    const watchList = [
        ...Object.keys(wState.wotsAddrs),
        ...Object.keys(wState.mssAddrs),
        ...Object.keys(wState.utxos)
    ];
    wallet.set_watchlist(JSON.stringify(watchList));
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Chain Scanning
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Scan the blockchain for wallet-relevant transactions.
 *
 * Uses compact block filters (Golomb-coded sets) to efficiently skip
 * irrelevant blocks. Only fetches full block data when a filter matches.
 *
 * @returns {Promise<void>}
 */
async function performScan() {
    // Ensure MSS caches are loaded before scanning (handles login edge cases)
    if (!mssCachesReady) await loadMssCaches();

    self.postMessage({ type: 'LOG', payload: "Fetching chain state..." });
    const state       = await rpc.getState();
    const chainHeight = state.height;

    if (chainHeight <= wState.lastScannedHeight) {
        // Sync leaf indices even when chain hasn't advanced
        for (const [addr, mss] of Object.entries(wState.mssAddrs)) {
            wallet.set_mss_leaf_index(addr, mss.next_leaf);
        }
        self.postMessage({ type: 'SCAN_COMPLETE', payload: buildDashboardPayload() });
        return;
    }

    self.postMessage({ type: 'LOG', payload: `Scanning blocks ${wState.lastScannedHeight} to ${chainHeight}...` });

    let currentHeight = wState.lastScannedHeight;
    updateWasmWatchlist();

    while (currentHeight < chainHeight) {
        const end        = Math.min(currentHeight + 1000, chainHeight);
        const filterData = await rpc.getFilters(currentHeight, end);
        const numFilters = filterData.filters ? filterData.filters.length : 0;

        for (let i = 0; i < numFilters; i++) {
            const height = filterData.start_height + i;
            if (height % 100 === 0) self.postMessage({ type: 'SCAN_PROGRESS', payload: { height, max: chainHeight } });

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
                const mutated = await processFullBlock(currentHeight);
                if (mutated) updateWasmWatchlist();
                currentHeight++;
                if (currentHeight % 100 === 0) self.postMessage({ type: 'SCAN_PROGRESS', payload: { height: currentHeight, max: chainHeight } });
            }
        }
    }

    // Sync leaf indices from wState into WASM cache
    for (const [addr, mss] of Object.entries(wState.mssAddrs)) {
        wallet.set_mss_leaf_index(addr, mss.next_leaf);
    }

    wState.lastScannedHeight = chainHeight;
    await saveState();
    self.postMessage({ type: 'SCAN_COMPLETE', payload: buildDashboardPayload() });
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
    if (!block) return false;

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
                kind: 'coinbase',
                timestamp: block.timestamp || Math.floor(Date.now() / 1000),
                fee: 0, inputs: [],
                outputs: coinbaseReceives.map(c => c.id),
                value:   coinbaseReceives.reduce((s, c) => s + c.val, 0)
            });
        }
    }

    if (block.transactions) {
        for (const tx of block.transactions) {
            const reveal = tx.Reveal || tx.reveal;
            if (!reveal) continue;

            let spentIds = [], spentValue = 0, createdOutputs = [];

            if (reveal.inputs) {
                for (const inp of reveal.inputs) {
                    const saltHex = normalizeHex(inp.salt);
                    const cid     = ourSalts.get(saltHex);
                    if (cid) {
                        delete wState.utxos[cid];
                        ourSalts.delete(saltHex);
                        spentIds.push(cid);
                        spentValue += Number(inp.value);
                        matchFound = true;
                    }
                }
            }

            if (reveal.outputs) {
                for (const out of reveal.outputs) {
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
                }
            }

            if (spentIds.length > 0) {
                const alreadyRecorded = wState.history.some(h =>
                    (h.kind === 'sent' || h.kind === 'mixed') && h.inputs.some(inp => spentIds.includes(inp))
                );
                if (!alreadyRecorded) {
                    let totalTxIn = 0, totalTxOut = 0;
                    if (reveal.inputs)  reveal.inputs.forEach(i  => totalTxIn  += Number(i.value));
                    if (reveal.outputs) reveal.outputs.forEach(o => { let od = o.Standard || o.standard; if (od) totalTxOut += Number(od.value); });
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
function addUtxo(address, value, salt, coinId) {
    let index = 0, is_mss = false, mss_height = 0, mss_leaf = 0;
    if (wState.wotsAddrs[address] !== undefined) {
        index = wState.wotsAddrs[address];
        while (wState.nextWotsIndex <= index + GAP_LIMIT) deriveNextWots();
    } else {
        const mss = wState.mssAddrs[address];
        index = mss.index; is_mss = true; mss_height = mss.height; mss_leaf = mss.next_leaf;
    }
    if (!wState.utxos[coinId]) {
        wState.utxos[coinId] = { index, is_mss, mss_height, mss_leaf, address, value, salt, coin_id: coinId };
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
async function performSend(toAddress, amount) {
    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Selecting coins and building transaction..." } });
    await new Promise(r => setTimeout(r, 10));

    // Ensure MSS caches are loaded
    if (!mssCachesReady) await loadMssCaches();

    // -----------------------------------------------------------------------
    // Verify MSS safety indices with the node before signing
    // -----------------------------------------------------------------------
    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Verifying MSS safety indices..." } });
    for (const [addr, mss] of Object.entries(wState.mssAddrs)) {
        try {
            const mssState = await rpc.getMssState(addr);
            // If the node has seen more signatures than we have locally, FAST FORWARD.
            if (mssState && mssState.next_index > mss.next_leaf) {
                const SAFETY_MARGIN = 20;
                mss.next_leaf = mssState.next_index + SAFETY_MARGIN;
                self.postMessage({ type: 'LOG', payload: `⚠️ Fast-forwarded MSS index for ${addr.substring(0,8)} to ${mss.next_leaf} for safety.` });
            }
        } catch (e) {
            throw new Error(`Safety Check Failed: Could not verify MSS state for ${addr.substring(0,8)}. Aborting to prevent key reuse. Run a Network Sync.`);
        }
    }
    // -----------------------------------------------------------------------

    // Sync leaf indices before spend
    for (const [addr, mss] of Object.entries(wState.mssAddrs)) {
        wallet.set_mss_leaf_index(addr, mss.next_leaf);
    }

    const utxoArray = Object.values(wState.utxos).map(u => {
        if (u.is_mss && wState.mssAddrs[u.address]) return { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf };
        return u;
    });

    let spendContextStr;
    try {
        spendContextStr = wallet.prepare_spend(JSON.stringify(utxoArray), toAddress, BigInt(amount), wState.nextWotsIndex);
    } catch (e) {
        throw new Error(`Failed to prepare transaction: ${e.toString()}.\n\nWhat to do: Ensure you have enough funds to cover the amount plus the network fee. Try running a Network Sync first.`);
    }

    const ctx = JSON.parse(spendContextStr);

    // Show pending in UI immediately
    pendingSends.push({ kind: 'pending', timestamp: Math.floor(Date.now() / 1000), fee: ctx.fee, inputs: ctx.selected_inputs.map(i => i.coin_id), outputs: [], value: Number(amount) });
    self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });

    // Advance WOTS counter for any change addresses derived during prepare_spend
    while (wState.nextWotsIndex < ctx.next_wots_index) deriveNextWots();

    // Advance MSS leaf counters for used addresses
    const usedMssAddrs = new Set();
    for (const inp of ctx.selected_inputs) if (inp.is_mss) usedMssAddrs.add(inp.address);
    for (const addr of usedMssAddrs) wState.mssAddrs[addr].next_leaf++;

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Encrypting and saving wallet state..." } });
    await new Promise(r => setTimeout(r, 10));
    await saveState();

    // Mine spam-proof PoW
    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Fetching network difficulty..." } });
    const stateData   = await rpc.getState();
    const requiredPow = stateData.required_pow || 24;

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: `Mining Proof-of-Work (difficulty: ${requiredPow})...` } });
    await new Promise(r => setTimeout(r, 50));
    const spamNonce = Number(mine_commitment_pow(ctx.commitment, requiredPow));

    // Submit commit
    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "PoW complete. Submitting commitment..." } });
    const commitReq = await rpc.commit(ctx.commitment, spamNonce);

    if (!commitReq.ok) {
        let errText = commitReq.body || 'rejected';

        try { errText = JSON.parse(errText).error || errText; } catch(e) {}
        throw new Error(`Commit rejected by network:\n${errText}\n\nWhat to do: The network might be congested, or your UTXOs might be out of sync. Your funds have not moved. Run a Network Sync and try again.`);
    }

    // Wait for commit to be mined, then submit reveal
    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Commitment accepted. Waiting for block confirmation..." } });
    const revealPayloadStr = wallet.build_reveal(spendContextStr, ctx.commitment, ctx.tx_salt);

    let mempoolAccepted = false;
    for (let attempts = 0; attempts < 1500; attempts++) {
        if (attempts > 0 && attempts % 15 === 0) self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: `Still waiting for commit block (${attempts * 2}s)...` } });

        const revealReq = await rpc.send(revealPayloadStr);
        if (revealReq.ok) { mempoolAccepted = true; break; }

        let errText = revealReq.body || 'rejected';

        try { errText = JSON.parse(errText).error || errText; } catch(e) {}
        if (errText.includes("No matching commitment found")) {
            await new Promise(r => setTimeout(r, 3000));
        } else {
            throw new Error(`Reveal rejected by network:\n${errText}\n\nWhat to do: A cryptographic error or double-spend occurred. Your funds are safe. Run a Network Sync and try again.`);
        }
    }

    if (!mempoolAccepted) throw new Error("Timed out waiting for Commit to be mined.\n\nWhat to do: Your funds are perfectly safe. The network dropped the transaction due to high traffic. Please try sending again in a few minutes.");

    // Wait for reveal to be mined
    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Commit confirmed! Broadcasting reveal..." } });

    const inputCoinToCheck = ctx.selected_inputs[0].coin_id;
    let revealMined = false;
    for (let attempts = 0; attempts < 150; attempts++) {
        if (attempts > 0 && attempts % 15 === 0) self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: `Waiting for reveal to be mined (${attempts * 2}s)...` } });

        const checkResp = await rpc.checkCoin(inputCoinToCheck);
        if (checkResp && !checkResp.exists) { revealMined = true; break; }
        await new Promise(r => setTimeout(r, 2000));
    }

    if (!revealMined) throw new Error("Timed out waiting for Reveal to be mined. Your transaction is likely stuck in the mempool.");

    // Update local state
    pendingSends = [];
    for (const inp of ctx.selected_inputs) delete wState.utxos[inp.coin_id];

    let outIds = [];
    for (const out of ctx.outputs) {
        const addrHex = normalizeHex(out.address);
        if (wState.wotsAddrs[addrHex] !== undefined || wState.mssAddrs[addrHex] !== undefined) {
            const saltHex = normalizeHex(out.salt);
            const coinId  = compute_coin_id_hex(addrHex, BigInt(out.value), saltHex);
            if (addUtxo(addrHex, Number(out.value), saltHex, coinId)) outIds.push(coinId);
        }
    }

    wState.history.push({ kind: 'sent', timestamp: Math.floor(Date.now() / 1000), fee: ctx.fee, inputs: ctx.selected_inputs.map(i => i.coin_id), outputs: outIds, value: Number(amount) });
    await saveState();
    self.postMessage({ type: 'SEND_COMPLETE', payload: buildDashboardPayload() });
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
        wState = { phrase: null, nextWotsIndex: 0, nextMssIndex: 0, wotsAddrs: {}, mssAddrs: {}, utxos: {}, history: [], lastScannedHeight: 0 };
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

    // ── Summary ─────────────────────────────────────────────────────────

    const passed = results.filter(r => r.ok).length;
    const failed = results.filter(r => !r.ok).length;
    return { passed, failed, results };
}
