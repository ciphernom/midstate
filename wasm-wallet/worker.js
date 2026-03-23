import init, { WebWallet, generate_phrase, compute_coin_id_hex, decrypt_cli_wallet, mine_commitment_pow } from './pkg/wasm_wallet.js';

let wallet = null;
let password = null;
let isSending = false;
let isSubmitting = false;
let pendingSends = [];

const GAP_LIMIT = 100;
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

// ─── RPC Bridge ───────────────────────────────────────────────────────────────
//
// All network calls are proxied to the main thread (index.html) which owns the
// LightClient. RTCPeerConnection is not available in Web Workers, so WebRTC
// must live on the main thread. Each call posts an RPC_REQUEST and awaits the
// corresponding RPC_RESPONSE matched by a unique request id.

let _rpcNextId = 1;
const _rpcPending = new Map(); // id -> { resolve, reject }

// Called by the message handler below when an RPC_RESPONSE arrives.
function _rpcReceive(id, result, error) {
    const p = _rpcPending.get(id);
    if (!p) return;
    _rpcPending.delete(id);
    if (error !== undefined) p.reject(new Error(error));
    else p.resolve(result);
}

function rpcCall(method, params) {
    return new Promise((resolve, reject) => {
        const id = _rpcNextId++;
        _rpcPending.set(id, { resolve, reject });
        self.postMessage({ type: 'RPC_REQUEST', payload: { id, method, params } });
        // Timeout after 30s to avoid hanging forever on a lost message
        setTimeout(() => {
            if (_rpcPending.has(id)) {
                _rpcPending.delete(id);
                reject(new Error(`RPC timeout: ${method}`));
            }
        }, 30_000);
    });
}

// Thin wrappers matching the shapes callers expect.
// Methods that return raw response objects (ok/status/json/text) wrap
// the result so the existing callers don't need to change.
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

    // getBlockTemplate returns a response-like object { ok, status, json(), text() }
    // The main thread returns { ok, status, body } and we reconstruct it here.
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

// ─── Hex / Crypto Utilities ───────────────────────────────────────────────────

function normalizeHex(data) {
    if (!data) return "";
    if (typeof data === 'string') return data.toLowerCase();
    if (Array.isArray(data) || data instanceof Uint8Array) {
        return Array.from(data).map(b => b.toString(16).padStart(2, '0')).join('').toLowerCase();
    }
    return "";
}

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

async function saveState() {
    if (!password) return;
    const salt = crypto.getRandomValues(new Uint8Array(16));
    const iv   = crypto.getRandomValues(new Uint8Array(12));
    const key  = await deriveCryptoKey(password, salt);
    const enc  = new TextEncoder();
    const encrypted = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, enc.encode(JSON.stringify(wState)));
    const bundle = {
        salt: normalizeHex(salt),
        iv:   normalizeHex(iv),
        data: normalizeHex(new Uint8Array(encrypted))
    };
    self.postMessage({ type: 'SAVE_WALLET', payload: JSON.stringify(bundle) });
}

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
        if (Array.isArray(wState.utxos)) {
            const utxoMap = {};
            for (const u of wState.utxos) utxoMap[u.coin_id] = u;
            wState.utxos = utxoMap;
        }
        if (wState.history === undefined) {
            self.postMessage({ type: 'LOG', payload: "Legacy backup detected. Re-indexing chain to rebuild transaction history..." });
            wState.history = [];
            if (wState.lastScannedHeight > 0) { wState.lastScannedHeight = 0; wState.utxos = {}; }
        }
        password = pwd;
        wallet = new WebWallet(wState.phrase);
        self.postMessage({ type: 'WALLET_LOADED', payload: buildDashboardPayload() });
    } catch(e) {
        throw new Error("Incorrect password or corrupted wallet file");
    }
}

// ─── Mining ───────────────────────────────────────────────────────────────────

async function handleGetTemplate() {
    if (!wallet) throw new Error("Wallet not initialized.");

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
}

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
        const rejectReason = accepted ? null : await submitReq.text();

        if (accepted) {
            self.postMessage({ type: 'LOG', payload: `✅ Block accepted! Height: ${template.chainHeight}` });
            await saveState();
            await performScan();
        } else {
            self.postMessage({ type: 'LOG', payload: `❌ Block rejected: ${rejectReason}` });
            await saveState();
        }

        return {
            accepted, rejectReason,
            reward:    (template.blockReward || 0) + (template.total_fees || 0),
            height:    template.chainHeight,
            finalHash: batch.extension.final_hash,
            timestamp: batch.timestamp,
            txCount:   batch.transactions?.length || 0,
            fees:      template.total_fees || 0
        };
    } finally {
        isSubmitting = false;
    }
}

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

        if (resp.status === 409) {
            try {
                const err = await resp.json();
                if (err.expected_total) {
                    self.postMessage({ type: 'LOG', payload: `Fees changed (${totalValue} → ${err.expected_total}). Rebuilding coinbase...` });
                    totalValue = err.expected_total;
                    continue;
                }
            } catch (e) {}
        }

        const errText = await resp.text();
        self.postMessage({ type: 'ERROR', payload: `Template error: ${errText}` });
        return null;
    }

    self.postMessage({ type: 'ERROR', payload: "Failed to build template after retries." });
    return null;
}

// ─── Message Handler ──────────────────────────────────────────────────────────

self.onmessage = async (e) => {
    const { type, payload } = e.data;
    try {
        if (type === 'INIT') {
            await init();
            self.postMessage({ type: 'INIT_DONE' });
        }

        else if (type === 'RPC_RESPONSE') {
            // Response from the main thread for an earlier RPC_REQUEST
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
            for (let i = 0; i < GAP_LIMIT; i++) {
                deriveNextWots();
                if (i % 10 === 0) {
                    self.postMessage({ type: 'MSS_PROGRESS', payload: { current: i, total: GAP_LIMIT, label: `Deriving base keys (${i}/${GAP_LIMIT})...` } });
                    await new Promise(r => setTimeout(r, 0));
                }
            }
            self.postMessage({ type: 'MSS_PROGRESS', payload: { current: 0, total: 100, label: "Generating Post-Quantum MSS Address..." } });
            await new Promise(r => setTimeout(r, 10));
            deriveNextMss(10);
            await saveState();
            self.postMessage({ type: 'WALLET_LOADED', payload: buildDashboardPayload() });
            // Signal main thread to attempt WebRTC auto-connect
            self.postMessage({ type: 'AUTO_CONNECT_WEBRTC' });
        }

        else if (type === 'LOGIN') {
            await loadState(payload.password, payload.bundleStr);
            // Signal main thread to attempt WebRTC auto-connect
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
            deriveNextMss(10);
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
                await saveState();
                self.postMessage({ type: 'WALLET_LOADED', payload: buildDashboardPayload() });
            } catch (err) {
                self.postMessage({ type: 'ERROR', payload: "Failed to import CLI wallet: Incorrect password or corrupt file." });
            }
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

// ─── Key Derivation ───────────────────────────────────────────────────────────

function deriveNextWots() {
    const addr = wallet.get_wots_address(wState.nextWotsIndex);
    wState.wotsAddrs[addr] = wState.nextWotsIndex;
    wState.nextWotsIndex++;
}

let lastMssUpdate = 0;
function deriveNextMss(height) {
    const progressCallback = (current, total) => {
        const now = Date.now();
        if (now - lastMssUpdate > 66 || current === total) {
            lastMssUpdate = now;
            self.postMessage({ type: 'MSS_PROGRESS', payload: { current, total, label: `Hashing tree leaves (${current}/${total})...` } });
        }
    };
    const addr = wallet.get_mss_address(wState.nextMssIndex, height, progressCallback);
    wState.mssAddrs[addr] = { index: wState.nextMssIndex, height, next_leaf: 0 };
    wState.nextMssIndex++;
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

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

function updateWasmWatchlist() {
    const watchList = [
        ...Object.keys(wState.wotsAddrs),
        ...Object.keys(wState.mssAddrs),
        ...Object.keys(wState.utxos)
    ];
    wallet.set_watchlist(JSON.stringify(watchList));
}

// ─── Chain Scanning ───────────────────────────────────────────────────────────

async function performScan() {
    self.postMessage({ type: 'LOG', payload: "Fetching chain state..." });
    const state       = await rpc.getState();
    const chainHeight = state.height;

    if (chainHeight <= wState.lastScannedHeight) {
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

    for (const [addrHex, mss] of Object.entries(wState.mssAddrs)) {
        try {
            const res = await rpc.getMssState(addrHex);
            if (res.next_index > mss.next_leaf) mss.next_leaf = res.next_index;
        } catch(e) {}
    }

    wState.lastScannedHeight = chainHeight;
    await saveState();
    self.postMessage({ type: 'SCAN_COMPLETE', payload: buildDashboardPayload() });
}

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

// ─── Send ─────────────────────────────────────────────────────────────────────

async function performSend(toAddress, amount) {
    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Selecting coins and building transaction..." } });
    await new Promise(r => setTimeout(r, 10));

    for (const [addr, mss] of Object.entries(wState.mssAddrs)) wallet.set_mss_leaf_index(addr, mss.next_leaf);

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

    pendingSends.push({ kind: 'pending', timestamp: Math.floor(Date.now() / 1000), fee: ctx.fee, inputs: ctx.selected_inputs.map(i => i.coin_id), outputs: [], value: Number(amount) });
    self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });

    while (wState.nextWotsIndex < ctx.next_wots_index) deriveNextWots();

    const usedMssAddrs = new Set();
    for (const inp of ctx.selected_inputs) if (inp.is_mss) usedMssAddrs.add(inp.address);
    for (const addr of usedMssAddrs) wState.mssAddrs[addr].next_leaf++;

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Encrypting and saving wallet state..." } });
    await new Promise(r => setTimeout(r, 10));
    await saveState();

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Fetching network difficulty..." } });
    const stateData   = await rpc.getState();
    const requiredPow = stateData.required_pow || 24;

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: `Mining Proof-of-Work (difficulty: ${requiredPow})...` } });
    await new Promise(r => setTimeout(r, 50));
    const spamNonce = Number(mine_commitment_pow(ctx.commitment, requiredPow));

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "PoW complete. Submitting commitment..." } });
    const commitReq = await rpc.commit(ctx.commitment, spamNonce);

    if (!commitReq.ok) {
        let errText = await commitReq.text();
        try { errText = JSON.parse(errText).error || errText; } catch(e) {}
        throw new Error(`Commit rejected by network:\n${errText}\n\nWhat to do: The network might be congested, or your UTXOs might be out of sync. Your funds have not moved. Run a Network Sync and try again.`);
    }

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Commitment accepted. Waiting for block confirmation..." } });

    const revealPayloadStr = wallet.build_reveal(spendContextStr, ctx.commitment, ctx.tx_salt);

    let mempoolAccepted = false;
    for (let attempts = 0; attempts < 150; attempts++) {
        if (attempts > 0 && attempts % 15 === 0) self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: `Still waiting for commit block (${attempts * 2}s)...` } });

        const revealReq = await rpc.send(revealPayloadStr);
        if (revealReq.ok) { mempoolAccepted = true; break; }

        let errText = await revealReq.text();
        try { errText = JSON.parse(errText).error || errText; } catch(e) {}
        if (errText.includes("No matching commitment found")) {
            await new Promise(r => setTimeout(r, 2000));
        } else {
            throw new Error(`Reveal rejected by network:\n${errText}\n\nWhat to do: A cryptographic error or double-spend occurred. Your funds are safe. Run a Network Sync and try again.`);
        }
    }

    if (!mempoolAccepted) throw new Error("Timed out waiting for Commit to be mined.\n\nWhat to do: Your funds are perfectly safe. The network dropped the transaction due to high traffic. Please try sending again in a few minutes.");

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
