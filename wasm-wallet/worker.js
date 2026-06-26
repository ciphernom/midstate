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

import init, { WebWallet, generate_phrase, compute_coin_id_hex, decrypt_cli_wallet, mine_commitment_pow, blake3_hash_hex, build_multisig_2of2_address, build_channel_state, build_channel_reveal, verify_mss_sig_wasm, mine_chat_pow_v2_wasm, build_htlc_bytecode_hex } from './pkg/wasm_wallet.js';

/** @type {WebWallet|null} The WASM wallet instance. Null until CREATE or LOGIN. */
let wallet = null;

/** @type {string|null} The user's password, held in memory for encrypting state saves. */
let password = null;

/** @type {boolean} Guard against concurrent send operations. */
let isSending = false;

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
    lastScannedHeight: 0,
    vaultUtxo: null,
    l2_channels: {},
    l2_secrets: {},  // Stores preimages for invoices we generate
    l2_routes: {}    // Stores multi-hop routing map for Hubs
};

function getPrimaryMssPk() {
    if (!wallet) return null;
    const mssList = Object.keys(wState.mssAddrs);
    if (mssList.length === 0) return null;
    return wallet.get_mss_pubkey(mssList[mssList.length - 1]);
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

        // Ensure L2 structures exist for upgraded wallets
        wState.l2_channels = wState.l2_channels || {};
        wState.l2_secrets = wState.l2_secrets || {};
        wState.l2_routes = wState.l2_routes || {};
        wState.dex_swaps = wState.dex_swaps || {};
        
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
            _rpcReceive(payload.id, payload.result, payload.error);
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
        else if (type === 'DEX_LOCK_MIDSTATE') {
            if (isSending) throw new Error("Wallet busy.");
            isSending = true;
            try {
                const { offerId, expectedAmount, takerPk } = payload;

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

                // HTLC: receiver = taker (claims with the secret), refund = maker (after timeout).
                const htlcScriptHex = build_htlc_bytecode_hex(secretHash, takerPk, BigInt(timeoutHeight), myPk);
                const htlcAddressHex = blake3_hash_hex(htlcScriptHex);

                // Fund the HTLC. prepare_fund_tx splits the amount into power-of-two
                // coins, EACH with a random salt, so funding yields a SET of coins.
                // performContractTx now returns that set (salts are not on-chain, so
                // the taker can only spend them if we transmit {coin_id,value,salt}).
                const fundRes = await performContractTx({
                    reqId: 999,
                    dexOfferId: offerId,   // routes commit/reveal phase progress to this swap card
                    kind: 'fund',
                    contractAddress: htlcAddressHex,
                    amount: expectedAmount
                });
                const htlcCoins = (fundRes && fundRes.coins) || [];

                self.postMessage({ type: 'DEX_MIDSTATE_LOCKED_SUCCESS', payload: {
                    offerId,
                    secret: rawSecret,        // for the MAKER only — needed to claim ETH on Base
                    secretHash,               // H = BLAKE3(secret); the Base hashlock is identical
                    htlcAddressHex,
                    htlcCoins,                // [{coin_id, value, salt}] — the taker sweeps these
                    timeoutHeight,
                    makerMdsPk: myPk,
                    takerMdsPk: takerPk
                }});
            } catch (err) {
                // Clear the card's live phase on failure so it doesn't sit on a stale
                // commit/reveal step; re-throw so the outer handler still reports it.
                if (payload && payload.offerId) self.postMessage({ type: 'DEX_PHASE', payload: { offerId: payload.offerId, phase: null } });
                throw err;
            } finally {
                isSending = false;
            }
        }
        else if (type === 'DEX_VERIFY_HTLC') {
            // Taker-side safety check: before locking ETH, prove the maker's MDS HTLC
            // actually exists on-chain with the agreed hashlock, our pubkey as receiver,
            // and the full expected value. Without this a malicious maker could get the
            // taker to lock ETH against a fake or short Midstate lock.
            const { offerId, secretHash, timeoutHeight, makerMdsPk, htlcCoins, expectedTotal, baseRefundSecs } = payload;
            try {
                const myPk = getPrimaryMssPk();
                const scriptHex = build_htlc_bytecode_hex(secretHash, myPk, BigInt(timeoutHeight), makerMdsPk);
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
                if (ok && expectedTotal != null && total !== Number(expectedTotal)) {
                    ok = false; reason = `HTLC locks ${total} MDS but the offer was ${expectedTotal} MDS`;
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
            if (isSending) throw new Error("Wallet busy.");
            isSending = true;
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
                const utxoArray = Object.values(wState.utxos).map(u => {
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
                const sigHex = wallet.sign_mss_hex(myPk, ctx.commitment);

                // 3. Inject the claim witness [Signature, Secret, 0x01] into each input.
                for (let i = 0; i < ctx.contract_inputs.length; i++) {
                    ctx.contract_inputs[i].witness = `${sigHex},${rawSecret},01`;
                }

                // 4. Build the reveal.
                const revealPayloadStr = wallet.build_script_reveal(JSON.stringify(ctx), ctx.commitment, ctx.tx_salt);

                // 5. Advance key material exactly once.
                while (wState.nextWotsIndex < ctx.next_wots_index) deriveNextWots();
                const usedMss = new Set();
                for (const inp of ctx.wallet_inputs) if (inp.is_mss) usedMss.add(inp.address);
                for (const addr of usedMss) wState.mssAddrs[addr].next_leaf++;
                await saveState();

                // 6. Mine, commit, wait, reveal, wait.
                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Mining PoW..." } });
                dexPhase("Mining spam-proof (PoW)\u2026");
                const stateData = await rpc.getState();
                await new Promise(r => setTimeout(r, 50));
                const spamNonce = Number(mine_commitment_pow(ctx.commitment, stateData.required_pow || 24, BigInt(stateData.height), stateData.header_hash));

                const commitReq = await rpc.commit(ctx.commitment, spamNonce);
                if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Waiting for Block Confirmation..." } });
                dexPhase("Commit broadcast \u2014 waiting to be mined (step 1 of 2)\u2026");
                while (true) {
                    try { const c = await rpc.checkCommitment(ctx.commitment); if (c && c.exists) break; } catch (e) {}
                    await waitForNextBlock(15000);
                }

                const revealReq = await rpc.send(revealPayloadStr);
                if (!revealReq.ok) throw new Error(`Reveal rejected: ${revealReq.body || revealReq.error}`);

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Broadcasting claim..." } });
                dexPhase("Reveal broadcast \u2014 waiting to be mined (step 2 of 2)\u2026");
                const firstCoin = normalizeHex(htlcCoins[0].coin_id);
                while (true) {
                    try { const inp = await rpc.checkCoin(firstCoin); if (inp && !inp.exists) break; } catch (e) {}
                    await waitForNextBlock(15000);
                }

                dexPhase("Confirmed \u2713 \u2014 syncing wallet\u2026");
                await performScan();
                self.postMessage({ type: 'DEX_CLAIM_SUCCESS', payload: { swapIdx } });

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
                isSending = false;
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
                wotsAddrs: {}, mssAddrs: {}, utxos: {}, history: [],
                lastScannedHeight: 0,
                l2_channels: {}, l2_secrets: {}, l2_routes: {}
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
            // Cancel any in-flight scan (its generation check will bail without
            // committing partial state) and ask the wrapper to do an atomic
            // wipe-then-scan in its next iteration.
            scanGeneration++;
            scanResetPending = true;
            await performScan();
        }

        else if (type === 'SEND') {
            if (isSending) throw new Error("A transaction is already in progress. Please wait for it to complete.");
            isSending = true;
            try { await performSend(payload.toAddress, payload.amount); }
            finally { isSending = false; }
        }
else if (type === 'L2_OPEN_CHANNEL') {
            if (isSending) throw new Error("Wallet busy.");
            const { peerPk, amount } = payload;
            const myPk = getPrimaryMssPk();
            if (!myPk) throw new Error("Network Sync required first to initialize your MSS L2 identity.");
            
            let aPk, bPk, isAlice;
            if (myPk < peerPk) { aPk = myPk; bPk = peerPk; isAlice = true; }
            else { aPk = peerPk; bPk = myPk; isAlice = false; }
            
            const channelAddr = build_multisig_2of2_address(aPk, bPk);
            pendingChannelOpen = { channelAddr, alicePk: aPk, bobPk: bPk, amount: Number(amount), isAlice };
            
            isSending = true;
            try { await performSend(channelAddr, Number(amount) + 100); } 
            finally { isSending = false; }
        }
        else if (type === 'L2_PAY') {
            const { channelId, amount } = payload;
            const channel = wState.l2_channels[channelId];
            if (!channel) throw new Error("Channel not found");
            
            let newAliceAmt = channel.latest_state.alice_amt;
            let newBobAmt = channel.latest_state.bob_amt;
            
            if (channel.is_alice) {
                if (newAliceAmt < amount) throw new Error("Insufficient channel balance");
                newAliceAmt -= amount; newBobAmt += amount;
            } else {
                if (newBobAmt < amount) throw new Error("Insufficient channel balance");
                newBobAmt -= amount; newAliceAmt += amount;
            }
            
            const newNonce = channel.latest_state.nonce + 1;
            const htlcs = channel.latest_state.htlcs || [];
            const stateJson = build_channel_state(channelId, channel.alice_pk, channel.bob_pk, BigInt(newAliceAmt), BigInt(newBobAmt), newNonce, JSON.stringify(htlcs));
            const parsedState = JSON.parse(stateJson);
            const myPk = channel.is_alice ? channel.alice_pk : channel.bob_pk;
            
            const sigHex = wallet.sign_mss_hex(myPk, parsedState.commitment);
            
            channel.latest_state = {
                nonce: newNonce, alice_amt: newAliceAmt, bob_amt: newBobAmt, htlcs,
                alice_sig: channel.is_alice ? sigHex : null,
                bob_sig: channel.is_alice ? null : sigHex,
                is_fully_signed: false
            };
            await saveState();
            
            const binPayload = packChannelState(newNonce, newAliceAmt, newBobAmt, htlcs, sigHex);
            
            submitClientMinedChat([255, 40], null, [
                { kind: "coin_id", value: channelId },
                { kind: "signature", value: normalizeHex(binPayload) }
            ]).catch(()=>{});
            self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
        }
        else if (type === 'L2_SYNC') {
            const { channelId } = payload;
            const channel = wState.l2_channels[channelId];
            if (!channel) return;
            
            const myPk = getPrimaryMssPk();
            
            // 1. Resend OPEN message
            submitClientMinedChat([255, 100], null, [
                { kind: "coin_id", value: channelId },
                { kind: "address", value: myPk }, 
                { kind: "midstate", value: channel.channel_salt }
            ]).catch(()=>{});
            
            // 2. If we have a pending UPDATE that the peer missed, resend it after a short delay
            if (channel.latest_state.nonce > 0 && !channel.latest_state.is_fully_signed) {
                const binPayload = packChannelState(
                    channel.latest_state.nonce, 
                    channel.latest_state.alice_amt, 
                    channel.latest_state.bob_amt, 
                    channel.latest_state.htlcs || [], 
                    channel.is_alice ? channel.latest_state.alice_sig : channel.latest_state.bob_sig
                );
                
                setTimeout(() => {
                    submitClientMinedChat([255, 40], null, [
                        { kind: "coin_id", value: channelId },
                        { kind: "signature", value: normalizeHex(binPayload) }
                    ]).catch(()=>{});
                }, 2000);
            }
        }
        else if (type === 'L2_CLOSE') {
            const { channelId } = payload;
            const channel = wState.l2_channels[channelId];
            if (!channel || !channel.latest_state.is_fully_signed) throw new Error("Channel cannot be cooperatively closed (missing signature).");
            
            const htlcs = channel.latest_state.htlcs || [];
            const stateJson = build_channel_state(channelId, channel.alice_pk, channel.bob_pk, BigInt(channel.latest_state.alice_amt), BigInt(channel.latest_state.bob_amt), channel.latest_state.nonce, JSON.stringify(htlcs));
            const revealPayloadStr = build_channel_reveal(BigInt(channel.channel_value), channel.channel_salt, channel.alice_pk, channel.bob_pk, stateJson, channel.latest_state.alice_sig, channel.latest_state.bob_sig);
            
            self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Broadcasting Cooperative Close..." } });
            const revealReq = await rpc.send(revealPayloadStr);
            if (!revealReq.ok) throw new Error(`Close rejected: ${revealReq.body || revealReq.error}`);
            
            delete wState.l2_channels[channelId];
            await saveState();
            self.postMessage({ type: 'SEND_COMPLETE', payload: buildDashboardPayload() });
        }
        else if (type === 'L2_CREATE_INVOICE') {
            const { amount } = payload;
            const secretHex = Array.from(crypto.getRandomValues(new Uint8Array(32))).map(b=>b.toString(16).padStart(2,'0')).join('');
            const secretHash = blake3_hash_hex(secretHex);
            wState.l2_secrets[secretHash] = secretHex;
            await saveState();
            
            const myPk = getPrimaryMssPk();
            const invoice = `l2inv:${myPk}:${secretHash}:${amount}`;
            self.postMessage({ type: 'L2_INVOICE_CREATED', payload: invoice });
        }
        else if (type === 'L2_PAY_INVOICE') {
            const { invoice } = payload;
            const parts = invoice.split(':');
            if (parts.length !== 4 || parts[0] !== 'l2inv') throw new Error("Invalid invoice format");
            
            const destPk = parts[1];
            const secretHash = parts[2];
            const amount = Number(parts[3]);
            
            // Find a channel with enough balance to act as the first hop Hub
            let hubChannelId = null;
            let hubChannel = null;
            for (const [cid, c] of Object.entries(wState.l2_channels)) {
                const myBal = c.is_alice ? c.latest_state.alice_amt : c.latest_state.bob_amt;
                if (c.latest_state.is_fully_signed && myBal >= amount) {
                    hubChannelId = cid;
                    hubChannel = c;
                    break;
                }
            }
            if (!hubChannel) throw new Error("No active channel with sufficient capacity to route this payment.");
            
            let newAliceAmt = hubChannel.latest_state.alice_amt;
            let newBobAmt = hubChannel.latest_state.bob_amt;
            if (hubChannel.is_alice) newAliceAmt -= amount; else newBobAmt -= amount;
            
            const newNonce = hubChannel.latest_state.nonce + 1;
            const htlcs = [...(hubChannel.latest_state.htlcs || [])];
            htlcs.push({ amount, timeout: networkHeight + 100, receiver_is_alice: !hubChannel.is_alice, secret_hash: secretHash });
            
            const stateJson = build_channel_state(hubChannelId, hubChannel.alice_pk, hubChannel.bob_pk, BigInt(newAliceAmt), BigInt(newBobAmt), newNonce, JSON.stringify(htlcs));
            const parsedState = JSON.parse(stateJson);
            const myPk = hubChannel.is_alice ? hubChannel.alice_pk : hubChannel.bob_pk;
            const sigHex = wallet.sign_mss_hex(myPk, parsedState.commitment);
            
            hubChannel.latest_state = {
                nonce: newNonce, alice_amt: newAliceAmt, bob_amt: newBobAmt, htlcs,
                alice_sig: hubChannel.is_alice ? sigHex : null,
                bob_sig: hubChannel.is_alice ? null : sigHex,
                is_fully_signed: false
            };
            await saveState();
            
            const binPayload = packChannelState(newNonce, newAliceAmt, newBobAmt, htlcs, sigHex);
            
            // Send ADD_HTLC (42) to the Hub, attaching the DestPk so it knows where to route it
            submitClientMinedChat([255, 42], null, [
                { kind: "coin_id", value: hubChannelId },
                { kind: "signature", value: normalizeHex(binPayload) },
                { kind: "address", value: destPk }
            ]).catch(()=>{});
            
            self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
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
            if (isSending) throw new Error("A transaction is already in progress.");
            isSending = true;
            try { await performContractTx({ kind: 'fund', ...payload }); }
            finally { isSending = false; }
        }

        else if (type === 'SPEND_CONTRACT') {
            if (isSending) throw new Error("A transaction is already in progress.");
            isSending = true;
            try { await performContractTx({ kind: 'spend', ...payload }); }
            finally { isSending = false; }
        }
        else if (type === 'SIGN_CHANNEL') {
            if (isSending) throw new Error("A transaction is already in progress.");
            isSending = true;
            try {
                self.postMessage({ type: 'CONTRACT_TX_PROGRESS', payload: { reqId: payload.reqId, msg: "Signing L2 Channel State..." } });
                
                const sigHex = await wallet.signChannelState(payload);
                
                // We reuse the CONTRACT_TX_COMPLETE bridge event, stuffing the signature into the `txid` field
                self.postMessage({ type: 'CONTRACT_TX_COMPLETE', payload: { reqId: payload.reqId, txid: sigHex } });
            } catch (err) {
                // Return the error back across the dApp bridge
                self.postMessage({ type: 'ERROR', payload: { reqId: payload.reqId, msg: err.toString() } });
            } finally {
                isSending = false;
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

        else if (type === 'IMPORT_CLI') {
            try {
                scanGeneration++;
                while (isScanning) {
                    await new Promise(r => setTimeout(r, 50));
                }
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
                    master_seed: normalizeHex(cliData.master_seed),
                    nextWotsIndex: cliData.next_wots_index || 0,
                    nextMssIndex:  cliData.next_mss_index  || 0,
                    wotsAddrs: {}, mssAddrs: newMssAddrs, utxos: newUtxos,
                    history: cliData.history || [],
                    lastScannedHeight: cliData.last_scan_height || 0,
                    l2_channels: {}, l2_secrets: {}, l2_routes: {}
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
        history: sortedHistory,
        lastScannedHeight: wState.lastScannedHeight || 0,
        networkHeight: networkHeight || 0,
        mempoolSize: mempoolSize || 0,
        l2Channels: Object.entries(wState.l2_channels || {}).map(([id, c]) => ({ id, ...c }))
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
    networkHeight     = chainHeight;
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
            const reveal = tx.Reveal || tx.reveal;
            if (!reveal) continue;

            let spentIds = [], spentValue = 0, createdOutputs = [];
            
            // Extract Sender Identity 
            let senderAddrHex = "";
            let txId = "";
            if (reveal.inputs && reveal.inputs.length > 0) {
                const bytecode = reveal.inputs[0].predicate?.Script?.bytecode || reveal.inputs[0].bytecode;
                if (bytecode) senderAddrHex = blake3_hash_hex(normalizeHex(bytecode));
                const saltHex = normalizeHex(reveal.inputs[0].salt);
                txId = compute_coin_id_hex(senderAddrHex, BigInt(reveal.inputs[0].value), saltHex);
            }

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
                    const burnData = out.DataBurn || out.data_burn;
                    if (burnData) {
                        const payloadHex = normalizeHex(burnData.payload);
                        //could do something with the burn data here i guess
                    }

                    // ── Contract coins at a watched address (MidstateConnect) ──
                    // Standard = value coin, Confidential = state coin. coin_id /
                    // value / salt / state are all present in-block.
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
function addUtxo(address, value, salt, coinId, commitment = null) {
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
async function performSend(toAddress, amount, burnDataHex = null, burnValue = 0) {
    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Selecting coins and building transaction..." } });
    await new Promise(r => setTimeout(r, 10));

    if (!mssCachesReady) await loadMssCaches();

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Verifying MSS safety indices..." } });
    for (const [addr, mss] of Object.entries(wState.mssAddrs)) {
        try {
            const mssState = await rpc.getMssState(addr);
            if (mssState && mssState.next_index > mss.next_leaf) {
                mss.next_leaf = mssState.next_index + 20;
                self.postMessage({ type: 'LOG', payload: `⚠️ Fast-forwarded MSS index for safety.` });
            }
        } catch (e) {
            throw new Error(`Safety Check Failed. Aborting to prevent key reuse.`);
        }
        wallet.set_mss_leaf_index(addr, mss.next_leaf);
    }

    const utxoArray = Object.values(wState.utxos).map(u => {
        if (u.is_mss && wState.mssAddrs[u.address]) return { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf };
        return u;
    });

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

    // Intercept L2 Open Intents
    if (pendingChannelOpen) {
        const outObj = ctx.outputs.find(o => o.address === pendingChannelOpen.channelAddr);
        if (outObj) {
            pendingChannelOpen.channelSalt = outObj.salt;
            pendingChannelOpen.channelCoinId = compute_coin_id_hex(outObj.address, BigInt(outObj.value), outObj.salt);
        }
    }

    pendingSends.push({ kind: 'pending', timestamp: Math.floor(Date.now() / 1000), fee: ctx.fee, inputs: ctx.selected_inputs.map(i => i.coin_id), outputs: [], value: Number(amount) });
    self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });

    while (wState.nextWotsIndex < ctx.next_wots_index) deriveNextWots();
    const usedMssAddrs = new Set();
    for (const inp of ctx.selected_inputs) if (inp.is_mss) usedMssAddrs.add(inp.address);
    for (const addr of usedMssAddrs) wState.mssAddrs[addr].next_leaf++;

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Saving wallet state..." } });
    await saveState();

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Fetching network difficulty..." } });
    const stateData   = await rpc.getState();
    const requiredPow = stateData.required_pow || 24;

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: `Mining PoW...` } });
    await new Promise(r => setTimeout(r, 50));
    const spamNonce = Number(mine_commitment_pow(ctx.commitment, requiredPow, BigInt(stateData.height), stateData.header_hash));

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Submitting commit..." } });
    const commitReq = await rpc.commit(ctx.commitment, spamNonce);
    if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Waiting for Block Confirmation (Phase 1)..." } });
    const revealPayloadStr = wallet.build_reveal(spendContextStr, ctx.commitment, ctx.tx_salt);

    while (true) {
        try {
            const checkResp = await rpc.checkCommitment(ctx.commitment);
            if (checkResp && checkResp.exists) break;
        } catch (e) {}
        await waitForNextBlock(15000);
    }

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Commit confirmed! Submitting reveal..." } });
    const revealReq = await rpc.send(revealPayloadStr);
    if (!revealReq.ok) throw new Error(`Reveal rejected: ${revealReq.body || revealReq.error}`);

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Broadcasting reveal..." } });
    const inputCoinToCheck = ctx.selected_inputs[0].coin_id;

    while (true) {
        try {
            const checkInp = await rpc.checkCoin(inputCoinToCheck);
            if (checkInp && !checkInp.exists) break;
        } catch (e) {}
        await waitForNextBlock(15000);
    }

    pendingSends = [];
    // Do NOT eagerly delete UTXOs here! Let performScan() discover the spend naturally
    // so it can properly register the history entry.
    // Finalize L2 Open
    if (pendingChannelOpen && pendingChannelOpen.channelCoinId) {
        wState.l2_channels = wState.l2_channels || {};
        wState.l2_channels[pendingChannelOpen.channelCoinId] = {
            alice_pk: pendingChannelOpen.alicePk,
            bob_pk: pendingChannelOpen.bobPk,
            channel_value: pendingChannelOpen.amount + 100, 
            channel_salt: pendingChannelOpen.channelSalt,
            is_alice: pendingChannelOpen.isAlice,
            latest_state: {
                nonce: 0,
                alice_amt: pendingChannelOpen.isAlice ? pendingChannelOpen.amount : 0,
                bob_amt: pendingChannelOpen.isAlice ? 0 : pendingChannelOpen.amount,
                alice_sig: null, bob_sig: null, is_fully_signed: false
            }
        };
        await saveState();
        
        const myPk = getPrimaryMssPk();
        submitClientMinedChat([255, 100], null, [
            { kind: "coin_id", value: pendingChannelOpen.channelCoinId },
            { kind: "address", value: myPk }, 
            { kind: "midstate", value: pendingChannelOpen.channelSalt }
        ]).catch(()=>{});
        
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
    for (const [addr, mss] of Object.entries(wState.mssAddrs)) {
        try {
            const mssState = await rpc.getMssState(addr);
            if (mssState && mssState.next_index > mss.next_leaf) {
                mss.next_leaf = mssState.next_index + 20;
            }
        } catch (e) {
            throw new Error("Safety Check Failed. Aborting to prevent key reuse.");
        }
        wallet.set_mss_leaf_index(addr, mss.next_leaf);
    }

    const utxoArray = Object.values(wState.utxos).map(u => {
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
    while (wState.nextWotsIndex < ctx.next_wots_index) deriveNextWots();
    const usedMss = new Set();
    for (const inp of (ctx.wallet_inputs || [])) if (inp.is_mss) usedMss.add(inp.address);
    for (const addr of usedMss) wState.mssAddrs[addr].next_leaf++;
    await saveState();

    // ── Commit → PoW → wait → reveal → wait (identical to performSend) ───────
    prog("Fetching network difficulty...");
    const stateData   = await rpc.getState();
    const requiredPow = stateData.required_pow || 24;

    prog("Mining PoW...");
    await new Promise(r => setTimeout(r, 50));
    const spamNonce = Number(mine_commitment_pow(ctx.commitment, requiredPow, BigInt(stateData.height), stateData.header_hash));

    prog("Submitting commit...");
    const commitReq = await rpc.commit(ctx.commitment, spamNonce);
    if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);

    prog("Commit broadcast — waiting to be mined (step 1 of 2)…");
    const revealPayloadStr = wallet.build_script_reveal(ctxStr, ctx.commitment, ctx.tx_salt);

    while (true) {
        try {
            const c = await rpc.checkCommitment(ctx.commitment);
            if (c && c.exists) break;
        } catch (e) {}
        await waitForNextBlock(15000);
    }

    prog("Commit mined ✓ — submitting reveal…");
    const revealReq = await rpc.send(revealPayloadStr);
    if (!revealReq.ok) throw new Error(`Reveal rejected: ${revealReq.body || revealReq.error}`);

    prog("Reveal broadcast — waiting to be mined (step 2 of 2)…");
    // Use the first input coin id (contract or wallet) to detect inclusion.
    const firstInputId = ctx.input_coin_ids && ctx.input_coin_ids.length ? ctx.input_coin_ids[0] : null;
    if (firstInputId) {
        while (true) {
            try {
                const inp = await rpc.checkCoin(firstInputId);
                if (inp && !inp.exists) break;
            } catch (e) {}
            await waitForNextBlock(15000);
        }
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
                createdCoins.push({
                    coin_id: compute_coin_id_hex(normalizeHex(o.address), BigInt(o.value), normalizeHex(o.salt)),
                    value: o.value,
                    salt: normalizeHex(o.salt)
                });
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


function packChannelState(nonce, aliceAmt, bobAmt, htlcs, sigHex) {
    const sigBytes = new Uint8Array(sigHex.match(/.{1,2}/g).map(b => parseInt(b, 16)));
    const bin = new Uint8Array(21 + (htlcs.length * 49) + sigBytes.length);
    const view = new DataView(bin.buffer);
    view.setUint32(0, nonce, true);
    view.setBigUint64(4, BigInt(aliceAmt), true);
    view.setBigUint64(12, BigInt(bobAmt), true);
    view.setUint8(20, htlcs.length);
    let offset = 21;
    for (const h of htlcs) {
        view.setBigUint64(offset, BigInt(h.amount), true);
        view.setBigUint64(offset+8, BigInt(h.timeout), true);
        view.setUint8(offset+16, h.receiver_is_alice ? 1 : 0);
        bin.set(new Uint8Array(h.secret_hash.match(/.{1,2}/g).map(b=>parseInt(b, 16))), offset+17);
        offset += 49;
    }
    bin.set(sigBytes, offset);
    return bin;
}

function unpackChannelState(binPayload) {
    const bin = new Uint8Array(binPayload.match(/.{1,2}/g).map(b => parseInt(b, 16)));
    const view = new DataView(bin.buffer);
    const nonce = view.getUint32(0, true);
    const aliceAmt = Number(view.getBigUint64(4, true));
    const bobAmt = Number(view.getBigUint64(12, true));
    const numHtlcs = view.getUint8(20);
    const htlcs = [];
    let offset = 21;
    for (let i = 0; i < numHtlcs; i++) {
        const amount = Number(view.getBigUint64(offset, true));
        const timeout = Number(view.getBigUint64(offset+8, true));
        const receiver_is_alice = view.getUint8(offset+16) === 1;
        const secret_hash = Array.from(bin.slice(offset+17, offset+49)).map(b=>b.toString(16).padStart(2,'0')).join('');
        htlcs.push({ amount, timeout, receiver_is_alice, secret_hash });
        offset += 49;
    }
    const sigHex = Array.from(bin.slice(offset)).map(b=>b.toString(16).padStart(2,'0')).join('');
    return { nonce, aliceAmt, bobAmt, htlcs, sigHex };
}

async function handleL2Chat(msg) {
    const cmd = msg.words[1];
    
    if (cmd === 100) { // OPEN
        const coinId = msg.attachments.find(a => a.kind === "coin_id")?.value;
        const peerPkRaw = msg.attachments.find(a => a.kind === "address")?.value;
        const sigAtt = msg.attachments.find(a => a.kind === "midstate")?.value;

        if (!coinId || !peerPkRaw || !sigAtt) return;

        // Strip the 8-character checksum the node adds to address attachments
        const peerPk = peerPkRaw.substring(0, 64);
        
        const myPk = getPrimaryMssPk();
        if (!myPk) return;
        
        let aPk, bPk, isAlice;
        if (peerPk < myPk) { 
            aPk = peerPk; bPk = myPk; isAlice = false; // Peer is smaller, Peer is Alice
        } else { 
            aPk = myPk; bPk = peerPk; isAlice = true;  // I am smaller, I am Alice
        }

        wState.l2_channels = wState.l2_channels || {};
        if (wState.l2_channels[coinId]) return;

        wState.l2_channels[coinId] = {
            alice_pk: aPk, bob_pk: bPk, channel_value: 0, channel_salt: sigAtt,
            is_alice: isAlice,
            latest_state: { nonce: 0, alice_amt: 0, bob_amt: 0, htlcs: [], alice_sig: null, bob_sig: null, is_fully_signed: false }
        };
        await saveState();
        self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
    }
    else if (cmd === 40 || cmd === 41 || cmd === 42 || cmd === 43) { 
        // 40=UPDATE, 41=CONFIRM, 42=ADD_HTLC, 43=CLAIM_HTLC
        const coinId = msg.attachments.find(a => a.kind === "coin_id")?.value;
        const sigAtt = msg.attachments.find(a => a.kind === "signature")?.value;
        if (!coinId || !sigAtt) return;

        const channel = wState.l2_channels[coinId];
        if (!channel) return;

        const { nonce, aliceAmt, bobAmt, htlcs, sigHex: counterpartySig } = unpackChannelState(sigAtt);

        if (nonce <= channel.latest_state.nonce && channel.latest_state.is_fully_signed) return;

        let stateJson;
        try {
            stateJson = build_channel_state(coinId, channel.alice_pk, channel.bob_pk, BigInt(aliceAmt), BigInt(bobAmt), nonce, JSON.stringify(htlcs));
        } catch (e) {
            console.error("WASM build_channel_state failed:", e);
            return;
        }
        
        const parsedState = JSON.parse(stateJson);
        const counterpartyPk = channel.is_alice ? channel.bob_pk : channel.alice_pk;
        
        if (!verify_mss_sig_wasm(counterpartySig, parsedState.commitment, counterpartyPk)) return;

        // ── ROUTING LOGIC ──
        if (cmd === 42) { // ADD_HTLC received
            const destPkRaw = msg.attachments.find(a => a.kind === "address")?.value;
            const destPk = destPkRaw ? destPkRaw.substring(0, 64) : null;
            const newHtlc = htlcs[htlcs.length - 1]; // Assume latest added
            
            if (destPk) {
                // WE ARE THE HUB. Forward to Dest.
                let forwardChannelId = null;
                for (const [cid, c] of Object.entries(wState.l2_channels)) {
                    if ((c.alice_pk === destPk || c.bob_pk === destPk) && c.latest_state.is_fully_signed) {
                        forwardChannelId = cid; break;
                    }
                }
                if (forwardChannelId) {
                    const fC = wState.l2_channels[forwardChannelId];
                    let nA = fC.latest_state.alice_amt; let nB = fC.latest_state.bob_amt;
                    if (fC.is_alice) nA -= newHtlc.amount; else nB -= newHtlc.amount;
                    
                    const fHtlcs = [...(fC.latest_state.htlcs || [])];
                    fHtlcs.push({ amount: newHtlc.amount, timeout: newHtlc.timeout - 10, receiver_is_alice: !fC.is_alice, secret_hash: newHtlc.secret_hash });
                    
                    const fNonce = fC.latest_state.nonce + 1;
                    const fStateJson = build_channel_state(forwardChannelId, fC.alice_pk, fC.bob_pk, BigInt(nA), BigInt(nB), fNonce, JSON.stringify(fHtlcs));
                    const fSig = wallet.sign_mss_hex(fC.is_alice ? fC.alice_pk : fC.bob_pk, JSON.parse(fStateJson).commitment);
                    
                    fC.latest_state = { nonce: fNonce, alice_amt: nA, bob_amt: nB, htlcs: fHtlcs, alice_sig: fC.is_alice ? fSig : null, bob_sig: fC.is_alice ? null : fSig, is_fully_signed: false };
                    
                    wState.l2_routes = wState.l2_routes || {};
                    wState.l2_routes[newHtlc.secret_hash] = { fromCoinId: coinId, amount: newHtlc.amount };
                    
                    const fBin = packChannelState(fNonce, nA, nB, fHtlcs, fSig);
                    submitClientMinedChat([255, 42], null, [{ kind: "coin_id", value: forwardChannelId }, { kind: "signature", value: normalizeHex(fBin) }, { kind: "address", value: destPk }]).catch(()=>{});
                }
            } else {
                // WE ARE THE DESTINATION.
                const secret = wState.l2_secrets ? wState.l2_secrets[newHtlc.secret_hash] : null;
                if (secret) {
                    // We know the secret! Claim it immediately.
                    const cHtlcs = htlcs.filter(h => h.secret_hash !== newHtlc.secret_hash);
                    let nA = aliceAmt; let nB = bobAmt;
                    if (channel.is_alice) nA += newHtlc.amount; else nB += newHtlc.amount;
                    
                    const cNonce = nonce + 1;
                    const cStateJson = build_channel_state(coinId, channel.alice_pk, channel.bob_pk, BigInt(nA), BigInt(nB), cNonce, JSON.stringify(cHtlcs));
                    const cSig = wallet.sign_mss_hex(channel.is_alice ? channel.alice_pk : channel.bob_pk, JSON.parse(cStateJson).commitment);
                    
                    channel.latest_state = { nonce: cNonce, alice_amt: nA, bob_amt: nB, htlcs: cHtlcs, alice_sig: channel.is_alice ? cSig : null, bob_sig: channel.is_alice ? null : cSig, is_fully_signed: false };
                    
                    const cBin = packChannelState(cNonce, nA, nB, cHtlcs, cSig);
                    submitClientMinedChat([255, 43], null, [{ kind: "coin_id", value: coinId }, { kind: "signature", value: normalizeHex(cBin) }, { kind: "midstate", value: secret }]).catch(()=>{});
                }
            }
        }
        else if (cmd === 43) { // CLAIM_HTLC received
            const secret = msg.attachments.find(a => a.kind === "midstate")?.value;

            if (secret) {
                const secretHash = blake3_hash_hex(secret);
                // We are HUB. Pull funds from original sender.
                if (wState.l2_routes && wState.l2_routes[secretHash]) {
                    const route = wState.l2_routes[secretHash];
                    const pC = wState.l2_channels[route.fromCoinId];
                    if (pC) {
                        let pA = pC.latest_state.alice_amt; let pB = pC.latest_state.bob_amt;
                        if (pC.is_alice) pA += route.amount; else pB += route.amount;
                        const pHtlcs = (pC.latest_state.htlcs || []).filter(h => h.secret_hash !== secretHash);
                        const pNonce = pC.latest_state.nonce + 1;
                        
                        const pStateJson = build_channel_state(route.fromCoinId, pC.alice_pk, pC.bob_pk, BigInt(pA), BigInt(pB), pNonce, JSON.stringify(pHtlcs));
                        const pSig = wallet.sign_mss_hex(pC.is_alice ? pC.alice_pk : pC.bob_pk, JSON.parse(pStateJson).commitment);
                        
                        pC.latest_state = { nonce: pNonce, alice_amt: pA, bob_amt: pB, htlcs: pHtlcs, alice_sig: pC.is_alice ? pSig : null, bob_sig: pC.is_alice ? null : pSig, is_fully_signed: false };
                        
                        const pBin = packChannelState(pNonce, pA, pB, pHtlcs, pSig);
                        submitClientMinedChat([255, 43], null, [{ kind: "coin_id", value: route.fromCoinId }, { kind: "signature", value: normalizeHex(pBin) }, { kind: "midstate", value: secret }]).catch(()=>{});
                    }
                }
            }
        }
        
        // Apply state locally
        const myPk = channel.is_alice ? channel.alice_pk : channel.bob_pk;
        const mySig = wallet.sign_mss_hex(myPk, parsedState.commitment);

        channel.latest_state = {
            nonce, alice_amt: aliceAmt, bob_amt: bobAmt, htlcs,
            alice_sig: channel.is_alice ? mySig : counterpartySig,
            bob_sig: channel.is_alice ? counterpartySig : mySig,
            is_fully_signed: true
        };
        
        if (channel.channel_value === 0) {
            channel.channel_value = aliceAmt + bobAmt + 100;
        }
        await saveState();

        if (cmd === 40 || cmd === 42) { // If UPDATE or ADD_HTLC, reply CONFIRM
            const binPayload = packChannelState(nonce, aliceAmt, bobAmt, htlcs, mySig);
            submitClientMinedChat([255, 41], null, [
                { kind: "coin_id", value: coinId },
                { kind: "signature", value: normalizeHex(binPayload) }
            ]).catch(()=>{});
        }
        self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
    }
    // ── L2 DEX ROUTING ──
    else if (cmd >= 200 && cmd <= 202) {
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
