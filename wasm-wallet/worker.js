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

import init, { WebWallet, generate_phrase, compute_coin_id_hex, decrypt_cli_wallet, mine_commitment_pow, blake3_hash_hex, build_multisig_2of2_address, build_channel_state, build_channel_reveal, verify_mss_sig_wasm, mine_chat_pow_v2_wasm, build_htlc_bytecode_hex, build_covenant_htlc_bytecode_hex, build_limit_order_covenant_bytecode_hex, compute_p2pk_address_hex } from './pkg/wasm_wallet.js';

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
const COVENANT_FEE_BUDGET = 1024;
// What a buyer requires the lock to be over-funded by before locking ETH, so the
// delivery is guaranteed to be affordable from the locked value.
const COVENANT_MIN_FEE_RESERVE = 256;

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
                const fundRes = await performContractTx({
                    reqId: 999,
                    dexOfferId: offerId,   // routes commit/reveal phase progress to this swap card
                    kind: 'fund',
                    contractAddress: htlcAddressHex,
                    amount: fundAmount
                });
                const htlcCoins = (fundRes && fundRes.coins) || [];

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
                    minPayout: covenant ? Number(expectedAmount) : undefined
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

                self.postMessage({ type: 'DEX_LIMIT_ORDER_CREATED', payload: {
                    offerId,
                    covAddr,
                    coins,                 // [{coin_id, value, salt}] — one coin for a power-of-two value
                    secret: rawSecret,     // MAKER ONLY — reveal on Base (claim) to get paid; NEVER broadcast
                    secretHash,            // advertised to takers; also the Base hashlock
                    maxClaim: v,
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
                const utxoArray = Object.values(wState.utxos).map(u =>
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
                const outUnits = built.map(b => {
                    const o = (ctx.outputs || []).find(o => o.type === "standard" && normalizeHex(o.address) === normalizeHex(b.covAddr));
                    if (!o) throw new Error(`Funded coin for unit ${b.v} not found in tx outputs`);
                    const coin_id = compute_coin_id_hex(normalizeHex(o.address), BigInt(o.value), normalizeHex(o.salt));
                    return {
                        offerId: Array.from(crypto.getRandomValues(new Uint8Array(8))).map(x => x.toString(16).padStart(2, '0')).join(''),
                        covAddr: b.covAddr,
                        coin: { coin_id, value: o.value, salt: normalizeHex(o.salt) },
                        secret: b.rawSecret, secretHash: b.secretHash,
                        maxClaim: b.v, timeoutHeight, makerMdsPk: myPk, makerEvmAddr, weiAmount: b.weiAmount
                    };
                });

                // CRASH-SAFETY: persist the recovery record (secrets + coins + params) BEFORE the
                // commit is broadcast. If we die after the tx confirms but before the UI registers
                // these units, DEX_RECOVER_BUNDLES rebuilds them from here; without it the locked MDS
                // would be unspendable (salts gone) and even un-refundable (no covenant params). The
                // saveState() just below persists this; it is cleared on success.
                if (!wState.pendingLimitBundles) wState.pendingLimitBundles = {};
                wState.pendingLimitBundles[groupId] = {
                    groupId, units: outUnits, commitment: ctx.commitment,
                    announcedAtomically: atomicAnnounce,
                    firstCoinId: outUnits[0].coin.coin_id, createdAt: Date.now()
                };

                // Reserve key material once (mirrors performContractTx / prepare_fund_tx flow).
                while (wState.nextWotsIndex < ctx.next_wots_index) deriveNextWots();
                const usedMss = new Set();
                for (const inp of (ctx.wallet_inputs || [])) if (inp.is_mss) usedMss.add(inp.address);
                for (const addr of usedMss) wState.mssAddrs[addr].next_leaf++;
                await saveState();

                // Funding consumes no contract coin, so build_script_reveal signs the wallet
                // inputs internally — no separate covenant signature needed.
                const revealPayloadStr = wallet.build_script_reveal(ctxStr, ctx.commitment, ctx.tx_salt);

                dexPhase("Mining spam-proof (PoW)\u2026");
                const stateData = await rpc.getState();
                await new Promise(r => setTimeout(r, 50));
                const spamNonce = Number(mine_commitment_pow(ctx.commitment, stateData.required_pow || 24, BigInt(stateData.height), stateData.header_hash));
                const commitReq = await rpc.commit(ctx.commitment, spamNonce);
                if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);
                dexPhase("Commit broadcast — waiting to be mined (1/2)\u2026");
                while (true) { try { const c = await rpc.checkCommitment(ctx.commitment); if (c && c.exists) break; } catch (e) {} await waitForNextBlock(15000); }
                // Self-heal MSS leaf reuse: if the leaf floor was stale, advance and re-sign
                // against the SAME already-mined commitment (leaf is a witness, not committed to).
                const revealReq = await sendRevealWithMssLeafRetry(revealPayloadStr, ctxStr, ctx.commitment, ctx.tx_salt, dexPhase);
                if (!revealReq.ok) throw new Error(`Reveal rejected: ${revealReq.body || revealReq.error}`);
                dexPhase("Reveal broadcast — waiting to be mined (2/2)\u2026");
                const firstInputId = ctx.input_coin_ids && ctx.input_coin_ids.length ? ctx.input_coin_ids[0] : null;
                if (firstInputId) { while (true) { try { const inp = await rpc.checkCoin(firstInputId); if (inp && !inp.exists) break; } catch (e) {} await waitForNextBlock(15000); } }
                dexPhase("Confirmed \u2713 — syncing\u2026");
                await performScan();

                // Funded and confirmed — hand the units to the UI, then clear the recovery record.
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
                    let funded = null;
                    try { const inp = await rpc.checkCoin(normalizeHex(rec.firstCoinId)); funded = !!(inp && inp.exists); }
                    catch (e) { continue; }   // RPC not ready — leave it; a later call retries
                    if (funded) {
                        self.postMessage({ type: 'DEX_LIMIT_BUNDLE_CREATED', payload: { groupId, units: rec.units, recovered: true, announcedAtomically: !!rec.announcedAtomically } });
                        delete pend[groupId]; changed = true;
                    } else if (Date.now() - (rec.createdAt || 0) > GRACE_MS) {
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
                for (let h = from; h <= tip; h += BATCH) {
                    const heights = [];
                    for (let k = h; k < h + BATCH && k <= tip; k++) heights.push(k);
                    const blocks = await Promise.all(heights.map(ht => rpc.getBlock(ht).then(b => b).catch(() => null)));
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
                                        recovered.push({
                                            offerId: 'chain:' + coinId.slice(0, 16),
                                            role: 'maker', side: 'mds', kind: 'limit', status: 'limit_posted',
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
            try {
                const ANNOUNCE_SCAN_DEPTH = 1440;   // ~ the order timeout window
                const tip = networkHeight;
                let from = Number.isFinite(payload && payload.fromHeight) ? payload.fromHeight : Math.max(0, tip - ANNOUNCE_SCAN_DEPTH);
                from = Math.max(0, Math.min(from, tip));
                const orders = [];
                const seenCoin = new Set();
                const BATCH = 12;
                let fragPoolDirty = false;
                for (let h = from; h <= tip; h += BATCH) {
                    const heights = [];
                    for (let k = h; k < h + BATCH && k <= tip; k++) heights.push(k);
                    const blocks = await Promise.all(heights.map(ht => rpc.getBlock(ht).then(b => b).catch(() => null)));
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
                                    const covScriptHex = build_limit_order_covenant_bytecode_hex(u.secretHash, BigInt(u.value), BigInt(ann.timeoutHeight), ann.makerMdsPk);
                                    const covAddr = blake3_hash_hex(covScriptHex);
                                    const coinId = compute_coin_id_hex(covAddr, BigInt(u.value), normalizeHex(u.salt));
                                    if (seenCoin.has(coinId)) continue;
                                    seenCoin.add(coinId);
                                    // Verify the covenant coin actually exists (not spent/filled, not forged).
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
                self.postMessage({ type: 'DEX_CLAIM_SUCCESS', payload: { swapIdx, offerId } });

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
                const utxoArray = Object.values(wState.utxos).map(u => {
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

                while (wState.nextWotsIndex < ctx.next_wots_index) deriveNextWots();
                const usedMss = new Set();
                for (const inp of ctx.wallet_inputs) if (inp.is_mss) usedMss.add(inp.address);
                for (const addr of usedMss) wState.mssAddrs[addr].next_leaf++;
                await saveState();

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

                self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Broadcasting delivery..." } });
                dexPhase("Reveal broadcast \u2014 waiting to be mined (step 2 of 2)\u2026");
                const firstCoin = normalizeHex(htlcCoins[0].coin_id);
                while (true) {
                    try { const inp = await rpc.checkCoin(firstCoin); if (inp && !inp.exists) break; } catch (e) {}
                    await waitForNextBlock(15000);
                }

                dexPhase("Confirmed \u2713 \u2014 syncing wallet\u2026");
                // The buyer scans to pick up the freshly delivered coins; the seller scans
                // to pick up the surplus change. Either way a scan is correct here.
                await performScan();
                self.postMessage({ type: 'DEX_SETTLE_SUCCESS', payload: { offerId, swapIdx, role } });

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
            // FEATURE 1 — taker-side partial fill of a maker's limit-order covenant.
            // Spends EXACTLY ONE covenant coin (single-input safety — see the multi-coin
            // caveat in compile_limit_order_covenant), pays `claimed` MDS to the taker and
            // routes `coinValue - claimed` back to the covenant address. Mirrors the proven
            // DEX_SETTLE_COVENANT covenant-spend template.
            await acquireSendLock();
            try {
                const { offerId, rawSecret, coin, claimed, maxClaim, timeoutHeight, makerMdsPk } = payload;
                const dexPhase = (p) => { if (offerId) self.postMessage({ type: 'DEX_PHASE', payload: { offerId, phase: p } }); };

                if (!coin || !coin.coin_id) throw new Error("No covenant coin supplied");
                const coinValue = Number(coin.value);
                const claimAmt  = Number(claimed);
                if (claimAmt <= 0 || claimAmt > coinValue) throw new Error("claimed must be within (0, coinValue]");
                if (claimAmt > Number(maxClaim)) throw new Error(`claimed ${claimAmt} exceeds covenant max_claim ${maxClaim}`);
                const remainder = coinValue - claimAmt;

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

                // outputs: `claimed` -> taker, `remainder` -> back into the covenant (keeps order live)
                const outputs = [
                    ...pow2(claimAmt).map(v => ({ out_type: "standard", address: buyerAddr, value: v, salt: null })),
                    ...pow2(remainder).map(v => ({ out_type: "standard", address: covAddr,   value: v, salt: null })),
                ];
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
                const utxoArray = Object.values(wState.utxos).map(u =>
                    (u.is_mss && wState.mssAddrs[u.address]) ? { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf } : u);

                let ctx = JSON.parse(wallet.prepare_script_spend(
                    JSON.stringify(utxoArray), covScriptHex, contractInputsJson, outputsJson, wState.nextWotsIndex
                ));
                const revealPayloadStr = wallet.build_script_reveal(JSON.stringify(ctx), ctx.commitment, ctx.tx_salt);

                while (wState.nextWotsIndex < ctx.next_wots_index) deriveNextWots();
                const usedMss = new Set();
                for (const inp of ctx.wallet_inputs) if (inp.is_mss) usedMss.add(inp.address);
                for (const addr of usedMss) wState.mssAddrs[addr].next_leaf++;
                await saveState();

                dexPhase("Mining spam-proof (PoW)…");
                const stateData = await rpc.getState();
                await new Promise(r => setTimeout(r, 50));
                const spamNonce = Number(mine_commitment_pow(ctx.commitment, stateData.required_pow || 24, BigInt(stateData.height), stateData.header_hash));
                const commitReq = await rpc.commit(ctx.commitment, spamNonce);
                if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);

                dexPhase("Commit broadcast — waiting to be mined (1/2)…");
                while (true) { try { const c = await rpc.checkCommitment(ctx.commitment); if (c && c.exists) break; } catch (e) {} await waitForNextBlock(15000); }

                const revealReq = await rpc.send(revealPayloadStr);
                if (!revealReq.ok) throw new Error(`Reveal rejected: ${revealReq.body || revealReq.error}`);

                dexPhase("Reveal broadcast — waiting to be mined (2/2)…");
                const firstCoin = normalizeHex(coin.coin_id);
                while (true) { try { const inp = await rpc.checkCoin(firstCoin); if (inp && !inp.exists) break; } catch (e) {} await waitForNextBlock(15000); }

                dexPhase("Confirmed ✓ — syncing…");
                await performScan();
                // Revealing `rawSecret` in the spend witness lets the maker harvest it and claim the
                // taker's ETH on Base. The NEW remainder coin (at covAddr) stays on the book.
                self.postMessage({ type: 'DEX_FILL_SUCCESS', payload: { offerId, claimed: claimAmt, remainder, covAddr } });
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
                const utxoArray = Object.values(wState.utxos).map(u =>
                    (u.is_mss && wState.mssAddrs[u.address]) ? { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf } : u);

                let ctx = JSON.parse(wallet.prepare_script_spend(
                    JSON.stringify(utxoArray), covScriptHex, contractInputsJson, outputsJson, wState.nextWotsIndex
                ));
                // Sign the spend with the refund key, then inject the ELSE-branch witness [sig, dummy(32B), 0x00].
                const sigHex = wallet.sign_mss_hex(refundPk, ctx.commitment);
                const dummy = "00".repeat(32);
                for (let i = 0; i < ctx.contract_inputs.length; i++) ctx.contract_inputs[i].witness = `${sigHex},${dummy},00`;

                const revealPayloadStr = wallet.build_script_reveal(JSON.stringify(ctx), ctx.commitment, ctx.tx_salt);

                while (wState.nextWotsIndex < ctx.next_wots_index) deriveNextWots();
                const usedMss = new Set();
                for (const inp of ctx.wallet_inputs) if (inp.is_mss) usedMss.add(inp.address);
                for (const addr of usedMss) wState.mssAddrs[addr].next_leaf++;
                await saveState();

                dexPhase("Mining spam-proof (PoW)\u2026");
                const stateData = await rpc.getState();
                await new Promise(r => setTimeout(r, 50));
                const spamNonce = Number(mine_commitment_pow(ctx.commitment, stateData.required_pow || 24, BigInt(stateData.height), stateData.header_hash));
                const commitReq = await rpc.commit(ctx.commitment, spamNonce);
                if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);
                dexPhase("Commit broadcast — waiting to be mined (1/2)\u2026");
                while (true) { try { const c = await rpc.checkCommitment(ctx.commitment); if (c && c.exists) break; } catch (e) {} await waitForNextBlock(15000); }
                const revealReq = await rpc.send(revealPayloadStr);
                if (!revealReq.ok) throw new Error(`Reveal rejected: ${revealReq.body || revealReq.error}`);
                dexPhase("Reveal broadcast — waiting to be mined (2/2)\u2026");
                const firstCoin = normalizeHex(coin.coin_id);
                while (true) { try { const inp = await rpc.checkCoin(firstCoin); if (inp && !inp.exists) break; } catch (e) {} await waitForNextBlock(15000); }
                dexPhase("Confirmed \u2713 — syncing\u2026");
                await performScan();
                self.postMessage({ type: 'DEX_REFUND_SUCCESS', payload: { offerId, reclaimed: coinValue } });
            } catch (err) {
                const detail = (err && err.message) ? err.message : String(err);
                self.postMessage({ type: 'DEX_REFUND_FAILED', payload: { offerId: payload && payload.offerId, error: detail } });
            } finally {
                releaseSendLock();
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
                    const utxoArray = Object.values(wState.utxos).map(u =>
                        (u.is_mss && wState.mssAddrs[u.address]) ? { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf } : u);

                    let ctx = JSON.parse(wallet.prepare_script_spend(
                        JSON.stringify(utxoArray), scriptHex, contractInputsJson, outputsJson, wState.nextWotsIndex
                    ));
                    // Refund branch witness: [sig, dummy(32B), 0x00] selects the ELSE/timeout path.
                    const sigHex = wallet.sign_mss_hex(myPk, ctx.commitment);
                    const dummy = "00".repeat(32);
                    for (let i = 0; i < ctx.contract_inputs.length; i++) ctx.contract_inputs[i].witness = `${sigHex},${dummy},00`;

                    const revealPayloadStr = wallet.build_script_reveal(JSON.stringify(ctx), ctx.commitment, ctx.tx_salt);
                    while (wState.nextWotsIndex < ctx.next_wots_index) deriveNextWots();
                    const usedMss = new Set();
                    for (const inp of ctx.wallet_inputs) if (inp.is_mss) usedMss.add(inp.address);
                    for (const addr of usedMss) wState.mssAddrs[addr].next_leaf++;
                    await saveState();

                    dexPhase(`Reclaiming coin ${reclaimedCoins + 1}/${htlcCoins.length} — mining PoW…`);
                    const stateData = await rpc.getState();
                    await new Promise(r => setTimeout(r, 50));
                    const spamNonce = Number(mine_commitment_pow(ctx.commitment, stateData.required_pow || 24, BigInt(stateData.height), stateData.header_hash));
                    const commitReq = await rpc.commit(ctx.commitment, spamNonce);
                    if (!commitReq.ok) throw new Error(`Commit rejected: ${commitReq.body || commitReq.error}`);
                    while (true) { try { const c = await rpc.checkCommitment(ctx.commitment); if (c && c.exists) break; } catch (e) {} await waitForNextBlock(15000); }
                    const revealReq = await rpc.send(revealPayloadStr);
                    if (!revealReq.ok) throw new Error(`Reveal rejected: ${revealReq.body || revealReq.error}`);
                    while (true) { try { const inp = await rpc.checkCoin(normalizeHex(coin.coin_id)); if (inp && !inp.exists) break; } catch (e) {} await waitForNextBlock(15000); }

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
            // when the taker's ETH lock for an L2-settled offer is seen. Routes ADD_HTLC over
            // an open L2 channel to the taker with the SAME hashlock instead of DEX_LOCK_MIDSTATE.
            // RELIES on the L2 HTLC fixes above (Bugs A/B/C) being in place.
            const { offerId, takerL2Pk, mdsAmount, secretHash, baseRefundSecs } = payload;

            let chanId = null, chan = null;
            for (const [cid, c] of Object.entries(wState.l2_channels)) {
                const peer = c.is_alice ? c.bob_pk : c.alice_pk;
                const myBal = c.is_alice ? c.latest_state.alice_amt : c.latest_state.bob_amt;
                if (peer === takerL2Pk && c.latest_state.is_fully_signed && myBal >= Number(mdsAmount)) { chanId = cid; chan = c; break; }
            }
            if (!chan) { self.postMessage({ type: 'DEX_LOCK_L2_FAILED', payload: { offerId, error: "No L2 channel to taker with capacity" } }); return; }

            // Tie the L2 HTLC timeout under the Base refund window so the maker can always sweep
            // ETH before refunding and the L2 leg can't outlive the Base leg unsafely.
            // UNITS FIX: htlcTimeout is a MIDSTATE height (~60s blocks), so convert seconds
            // with /60, not /12 (Base's block time). The old math made the L2 leg ~5× LONGER
            // than the Base refund window (900 blocks ≈ 15h vs 6h), letting a taker refund
            // their ETH on Base and STILL claim the L2 HTLC afterwards — taking both legs.
            // Half the window (6h → ~3h of Midstate blocks) leaves the maker ~3h of Base
            // lock left to sweep after the latest possible L2 claim reveals the preimage.
            const htlcTimeout = networkHeight + Math.max(20, Math.floor(Number(baseRefundSecs) / 60 / 2));

            let nA = chan.latest_state.alice_amt, nB = chan.latest_state.bob_amt;
            if (chan.is_alice) nA -= Number(mdsAmount); else nB -= Number(mdsAmount);
            const htlcs = [...(chan.latest_state.htlcs || []), {
                amount: Number(mdsAmount), timeout: htlcTimeout,
                receiver_is_alice: !chan.is_alice, secret_hash: secretHash
            }];
            const newNonce = chan.latest_state.nonce + 1;
            const stateJson = build_channel_state(chanId, chan.alice_pk, chan.bob_pk, BigInt(nA), BigInt(nB), newNonce, JSON.stringify(htlcs));
            const sigHex = wallet.sign_mss_hex(chan.is_alice ? chan.alice_pk : chan.bob_pk, JSON.parse(stateJson).commitment);
            chan.latest_state = { nonce: newNonce, alice_amt: nA, bob_amt: nB, htlcs, alice_sig: chan.is_alice ? sigHex : null, bob_sig: chan.is_alice ? null : sigHex, is_fully_signed: false };
            await saveState();

            // Send ADD_HTLC straight to the taker (they ARE the destination → destPk = their pk).
            const bin = packChannelState(newNonce, nA, nB, htlcs, sigHex);
            submitClientMinedChat([255, 42], null, [
                { kind: "coin_id", value: chanId },
                { kind: "signature", value: normalizeHex(bin) },
                { kind: "address", value: takerL2Pk }
            ]).catch(() => {});

            self.postMessage({ type: 'DEX_LOCK_L2_SENT', payload: { offerId, chanId, secretHash } });
        }

        else if (type === 'DEX_CHECK_L2_CHANNEL') {
            // FEATURE 2 pre-flight (TAKER side): verify — BEFORE any ETH is locked — that a
            // fully-signed channel to the maker exists whose MAKER-side balance can cover the
            // trade. Mirrors DEX_LOCK_L2's own channel selection from the other side, so a
            // passing check here means the maker's route will actually find a channel.
            const { offerId, peerPk, amount } = payload;
            let ok = false, reason = "No open L2 channel to the maker. Open one on the Lightning tab first.";
            for (const c of Object.values(wState.l2_channels || {})) {
                const peer = c.is_alice ? c.bob_pk : c.alice_pk;
                if (peer !== peerPk) continue;
                if (!c.latest_state.is_fully_signed) { reason = "Your channel to the maker has an unconfirmed state — retry in a moment."; continue; }
                const peerBal = c.is_alice ? c.latest_state.bob_amt : c.latest_state.alice_amt;
                if (peerBal < Number(amount)) { reason = `The maker's channel balance (${peerBal} MDS) can't cover ${amount} MDS.`; continue; }
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
                if ((c.latest_state.htlcs || []).some(h => h.secret_hash === secretHash)) { htlcPending = true; break; }
            }
            self.postMessage({ type: 'DEX_SUBMARINE_STATUS_RESULT', payload: { offerId, secretHash, observedSecret, claimedAmount, htlcPending } });
        }

        else if (type === 'DEX_CHECK_SETTLED') {
            // Lightweight read-only poll used by the buyer while waiting for delivery:
            // once the covenant lock coin is spent, the MDS has been delivered to them.
            const { offerId, coinId } = payload;
            let settled = false;
            try { const chk = await rpc.checkCoin(normalizeHex(coinId)); settled = !!(chk && !chk.exists); } catch (e) {}
            self.postMessage({ type: 'DEX_SETTLED_STATUS', payload: { offerId, settled } });
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
            await acquireSendLock();
            try { await performSend(payload.toAddress, payload.amount, null, 0, !!payload.sendAll); }
            catch (err) {
                // Surface send failures as a distinct, persistent error the UI can pin —
                // never let a send just silently stop. Include the message verbatim.
                self.postMessage({ type: 'SEND_ERROR', payload: { error: (err && err.message) ? err.message : String(err) } });
            }
            finally { releaseSendLock(); }
        }
else if (type === 'L2_OPEN_CHANNEL') {
            const { peerPk, amount } = payload;
            const myPk = getPrimaryMssPk();
            if (!myPk) throw new Error("Network Sync required first to initialize your MSS L2 identity.");
            
            let aPk, bPk, isAlice;
            if (myPk < peerPk) { aPk = myPk; bPk = peerPk; isAlice = true; }
            else { aPk = peerPk; bPk = myPk; isAlice = false; }
            
            const channelAddr = build_multisig_2of2_address(aPk, bPk);
            pendingChannelOpen = { channelAddr, alicePk: aPk, bobPk: bPk, amount: Number(amount), isAlice };
            
            await acquireSendLock();
            try { await performSend(channelAddr, Number(amount) + 100); } 
            finally { releaseSendLock(); }
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
        if (mssState && mssState.next_index > mss.next_leaf) {
            mss.next_leaf = mssState.next_index + 20;
            self.postMessage({ type: 'LOG', payload: `\u26a0\ufe0f Fast-forwarded MSS index for safety.` });
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

    const utxoArray = Object.values(wState.utxos).map(u => {
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
        let amt = total > 300n ? total - 300n : total;   // start just under a safe fee overestimate
        let fee = 300n, converged = false;
        for (let i = 0; i < 6; i++) {
            let est;
            try {
                est = JSON.parse(wallet.prepare_spend(JSON.stringify(utxoArray), toAddress, amt, wState.nextWotsIndex, null, null));
            } catch (e) {
                amt = amt > 100n ? amt - 100n : 0n;      // fee didn't fit yet; step down and retry
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

    let _commitHeight = null;
    while (true) {
        try {
            const checkResp = await rpc.checkCommitment(ctx.commitment);
            if (checkResp && checkResp.exists) { _commitHeight = checkResp.height ?? checkResp.block_height ?? null; break; }
        } catch (e) {}
        await waitForNextBlock(15000);
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

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Waiting for the reveal to be mined (phase 2 of 2)…" } });
    const inputCoinToCheck = ctx.selected_inputs[0].coin_id;

    let _revealHeight = null;
    while (true) {
        try {
            const checkInp = await rpc.checkCoin(inputCoinToCheck);
            if (checkInp && !checkInp.exists) { _revealHeight = checkInp.spentHeight ?? checkInp.height ?? null; break; }
        } catch (e) {}
        await waitForNextBlock(15000);
    }
    self.postMessage({ type: 'SEND_STEP', payload: {
        step: 'reveal_mined', title: 'Reveal confirmed — transaction complete',
        detail: _revealHeight != null ? `settled in block ${_revealHeight}` : 'transaction is now settled on-chain',
        sub: `commit ${_commitHeight != null ? 'block ' + _commitHeight : ''} → reveal ${_revealHeight != null ? 'block ' + _revealHeight : ''}`.trim()
    }});

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
    await verifyMssSafetyIndices(prog);

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
        } catch (e) { console.error("WASM build_channel_state failed:", e); return; }

        const parsedState = JSON.parse(stateJson);
        const counterpartyPk = channel.is_alice ? channel.bob_pk : channel.alice_pk;
        if (!verify_mss_sig_wasm(counterpartySig, parsedState.commitment, counterpartyPk)) return;

        const myPk = channel.is_alice ? channel.alice_pk : channel.bob_pk;

        // ── STEP 1: commit the incoming state (co-sign + store + CONFIRM) ──
        // This makes an HTLC-add irrevocable BEFORE we act on it. Doing this FIRST
        // (rather than after the routing logic) is the fix for Bug B: the old code
        // built the claim state then let this tail clobber it back a nonce.
        const mySig = wallet.sign_mss_hex(myPk, parsedState.commitment);
        channel.latest_state = {
            nonce, alice_amt: aliceAmt, bob_amt: bobAmt, htlcs,
            alice_sig: channel.is_alice ? mySig : counterpartySig,
            bob_sig:   channel.is_alice ? counterpartySig : mySig,
            is_fully_signed: true
        };
        if (channel.channel_value === 0) channel.channel_value = aliceAmt + bobAmt + 100;
        await saveState();

        if (cmd === 40 || cmd === 42) { // reply CONFIRM for UPDATE / ADD_HTLC
            const confirmBin = packChannelState(nonce, aliceAmt, bobAmt, htlcs, mySig);
            submitClientMinedChat([255, 41], null, [
                { kind: "coin_id", value: coinId },
                { kind: "signature", value: normalizeHex(confirmBin) }
            ]).catch(() => {});
        }

        // ── STEP 2: act on the now-committed state (advances to nonce+1) ──
        if (cmd === 42) { // ADD_HTLC
            const destPkRaw = msg.attachments.find(a => a.kind === "address")?.value;
            const destPk = destPkRaw ? destPkRaw.substring(0, 64) : null;
            const iAmDestination = !destPk || destPk === myPk;            // Bug C fix
            const newHtlc = htlcs[htlcs.length - 1];

            if (!iAmDestination) {
                // WE ARE A HUB — forward to dest on another fully-signed channel.
                let fwdId = null;
                for (const [cid, c] of Object.entries(wState.l2_channels)) {
                    if (cid === coinId) continue;                          // never forward onto the incoming channel
                    if ((c.alice_pk === destPk || c.bob_pk === destPk) && c.latest_state.is_fully_signed) { fwdId = cid; break; }
                }
                if (fwdId) {
                    const fC = wState.l2_channels[fwdId];
                    let nA = fC.latest_state.alice_amt, nB = fC.latest_state.bob_amt;
                    if (fC.is_alice) nA -= newHtlc.amount; else nB -= newHtlc.amount;
                    const fHtlcs = [...(fC.latest_state.htlcs || []), { amount: newHtlc.amount, timeout: newHtlc.timeout - 10, receiver_is_alice: !fC.is_alice, secret_hash: newHtlc.secret_hash }];
                    const fNonce = fC.latest_state.nonce + 1;
                    const fState = build_channel_state(fwdId, fC.alice_pk, fC.bob_pk, BigInt(nA), BigInt(nB), fNonce, JSON.stringify(fHtlcs));
                    const fSig = wallet.sign_mss_hex(fC.is_alice ? fC.alice_pk : fC.bob_pk, JSON.parse(fState).commitment);
                    fC.latest_state = { nonce: fNonce, alice_amt: nA, bob_amt: nB, htlcs: fHtlcs, alice_sig: fC.is_alice ? fSig : null, bob_sig: fC.is_alice ? null : fSig, is_fully_signed: false };
                    wState.l2_routes = wState.l2_routes || {};
                    wState.l2_routes[newHtlc.secret_hash] = { fromCoinId: coinId, amount: newHtlc.amount };
                    await saveState();
                    const fBin = packChannelState(fNonce, nA, nB, fHtlcs, fSig);
                    submitClientMinedChat([255, 42], null, [{ kind: "coin_id", value: fwdId }, { kind: "signature", value: normalizeHex(fBin) }, { kind: "address", value: destPk }]).catch(() => {});
                }
            } else {
                // WE ARE THE DESTINATION — if we know the preimage, claim by advancing nonce+1.
                const secret = wState.l2_secrets ? wState.l2_secrets[newHtlc.secret_hash] : null;
                if (secret) {
                    const cHtlcs = (channel.latest_state.htlcs || []).filter(h => h.secret_hash !== newHtlc.secret_hash);
                    let nA = channel.latest_state.alice_amt, nB = channel.latest_state.bob_amt;
                    if (channel.is_alice) nA += newHtlc.amount; else nB += newHtlc.amount;
                    const cNonce = channel.latest_state.nonce + 1;
                    const cState = build_channel_state(coinId, channel.alice_pk, channel.bob_pk, BigInt(nA), BigInt(nB), cNonce, JSON.stringify(cHtlcs));
                    const cSig = wallet.sign_mss_hex(myPk, JSON.parse(cState).commitment);
                    channel.latest_state = { nonce: cNonce, alice_amt: nA, bob_amt: nB, htlcs: cHtlcs, alice_sig: channel.is_alice ? cSig : null, bob_sig: channel.is_alice ? null : cSig, is_fully_signed: false };
                    // FEATURE 2 (taker side): record the settled claim so the UI (live event
                    // below, or the DEX_SUBMARINE_STATUS poll after a reload) can complete the
                    // submarine swap card.
                    wState.l2_claimed = wState.l2_claimed || {};
                    wState.l2_claimed[newHtlc.secret_hash] = newHtlc.amount;
                    await saveState();
                    const cBin = packChannelState(cNonce, nA, nB, cHtlcs, cSig);
                    submitClientMinedChat([255, 43], null, [{ kind: "coin_id", value: coinId }, { kind: "signature", value: normalizeHex(cBin) }, { kind: "midstate", value: secret }]).catch(() => {});
                    // Feature 2 hook: broadcasting cmd 43 above publishes `secret` on the bus —
                    // a maker fulfilling a submarine swap harvests it there (see cmd 43 below)
                    // and sweeps the Base ETH with it.
                    self.postMessage({ type: 'L2_HTLC_CLAIMED', payload: { secretHash: newHtlc.secret_hash, amount: newHtlc.amount } });
                }
            }
        } else if (cmd === 43) { // CLAIM_HTLC — a hub pulls funds from the upstream sender
            const secret = msg.attachments.find(a => a.kind === "midstate")?.value;
            if (secret) {
                const secretHash = blake3_hash_hex(secret);
                // FEATURE 2 (maker side): every cmd-43 claim publishes its preimage on the bus.
                // Persist it keyed by hash so a maker mid-submarine-swap can sweep the Base ETH
                // — via the live event below, or via the DEX_SUBMARINE_STATUS poll after a
                // reload. (Previously this secret was dropped on the floor: it never reached
                // l2_secrets, which only ever holds preimages for invoices WE generated.)
                wState.l2_observed_secrets = wState.l2_observed_secrets || {};
                if (!wState.l2_observed_secrets[secretHash]) {
                    wState.l2_observed_secrets[secretHash] = secret;
                    await saveState();
                    self.postMessage({ type: 'DEX_SUBMARINE_SECRET', payload: { secretHash, secret } });
                }
                const route = wState.l2_routes && wState.l2_routes[secretHash];
                if (route) {
                    const pC = wState.l2_channels[route.fromCoinId];
                    if (pC) {
                        let pA = pC.latest_state.alice_amt, pB = pC.latest_state.bob_amt;
                        if (pC.is_alice) pA += route.amount; else pB += route.amount;
                        const pHtlcs = (pC.latest_state.htlcs || []).filter(h => h.secret_hash !== secretHash);
                        const pNonce = pC.latest_state.nonce + 1;
                        const pState = build_channel_state(route.fromCoinId, pC.alice_pk, pC.bob_pk, BigInt(pA), BigInt(pB), pNonce, JSON.stringify(pHtlcs));
                        const pSig = wallet.sign_mss_hex(pC.is_alice ? pC.alice_pk : pC.bob_pk, JSON.parse(pState).commitment);
                        pC.latest_state = { nonce: pNonce, alice_amt: pA, bob_amt: pB, htlcs: pHtlcs, alice_sig: pC.is_alice ? pSig : null, bob_sig: pC.is_alice ? null : pSig, is_fully_signed: false };
                        delete wState.l2_routes[secretHash];
                        await saveState();
                        const pBin = packChannelState(pNonce, pA, pB, pHtlcs, pSig);
                        submitClientMinedChat([255, 43], null, [{ kind: "coin_id", value: route.fromCoinId }, { kind: "signature", value: normalizeHex(pBin) }, { kind: "midstate", value: secret }]).catch(() => {});
                    }
                }
            }
        }

        self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
    }
    // ── L2 DEX ROUTING ──
    else if (cmd >= 200 && cmd <= 204) {
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
