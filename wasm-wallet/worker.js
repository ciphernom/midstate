import init, { WebWallet, generate_phrase, compute_coin_id_hex, decrypt_cli_wallet, mine_commitment_pow } from './pkg/wasm_wallet.js';

let wallet = null;
let rpcUrl = "https://rpc.cypherpunk.gold";
let password = null;
let isSending = false;
let pendingSends = [];

// Increased from 20 to 100. Change outputs consume multiple WOTS indices. 
// A higher gap limit ensures we find all coins when restoring from a seed phrase.
const GAP_LIMIT = 100;
let wState = {
    phrase: null,
    nextWotsIndex: 0,
    nextMssIndex: 0,
    wotsAddrs: {}, // hex -> index
    mssAddrs: {},  // hex -> { index, height, next_leaf }
    utxos: {},     // RIGID MAP: coinIdHex -> UTXO Object
    history: [],   // Transaction history array
    lastScannedHeight: 0
};

// Extremely robust hex normalizer
function normalizeHex(data) {
    if (!data) return "";
    if (typeof data === 'string') return data.toLowerCase();
    if (Array.isArray(data) || data instanceof Uint8Array) {
        return Array.from(data).map(b => b.toString(16).padStart(2, '0')).join('').toLowerCase();
    }
    return "";
}

async function deriveCryptoKey(pwd, salt) {
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
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const key = await deriveCryptoKey(password, salt);
    
    const enc = new TextEncoder();
    const encrypted = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, enc.encode(JSON.stringify(wState)));
    
    const bundle = {
        salt: normalizeHex(salt),
        iv: normalizeHex(iv),
        data: normalizeHex(new Uint8Array(encrypted))
    };
    self.postMessage({ type: 'SAVE_WALLET', payload: JSON.stringify(bundle) });
}

async function loadState(pwd, bundleStr) {
    if (!bundleStr) throw new Error("No wallet found");
    const bundle = JSON.parse(bundleStr);
    
    const parseHexArray = (hexStr) => new Uint8Array((hexStr || "").match(/.{1,2}/g)?.map(b => parseInt(b, 16)) || []);
    
    const salt = parseHexArray(bundle.salt);
    const iv = parseHexArray(bundle.iv);
    const data = parseHexArray(bundle.data);
    
    const key = await deriveCryptoKey(pwd, salt);
    try {
        const decrypted = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, data);
        const dec = new TextDecoder();
        const loadedState = JSON.parse(dec.decode(decrypted));
        
        wState = loadedState;
        
        if (Array.isArray(wState.utxos)) {
            const utxoMap = {};
            for (const u of wState.utxos) utxoMap[u.coin_id] = u;
            wState.utxos = utxoMap;
        }

        if (wState.history === undefined) {
            self.postMessage({ type: 'LOG', payload: "Legacy backup detected. Re-indexing chain to rebuild transaction history..." });
            wState.history = [];
            if (wState.lastScannedHeight > 0) {
                wState.lastScannedHeight = 0;
                wState.utxos = {};
            }
        }

        password = pwd;
        wallet = new WebWallet(wState.phrase);
        self.postMessage({ type: 'WALLET_LOADED', payload: buildDashboardPayload() });
    } catch(e) {
        throw new Error("Incorrect password or corrupted wallet file");
    }
}

self.onmessage = async (e) => {
    const { type, payload } = e.data;
    try {
        if (type === 'INIT') {
            await init();
            self.postMessage({ type: 'INIT_DONE' });
        } 
        else if (type === 'GENERATE') {
            self.postMessage({ type: 'PHRASE_GENERATED', payload: generate_phrase() });
        }
        else if (type === 'CREATE') {
                    rpcUrl = payload.rpcUrl;
                    password = payload.password;
                    
                    // FIX: Explicitly free the old Wasm instance to prevent memory leaks
                    if (wallet) {
                        wallet.free(); 
                    }
                    
                    // FIX: Completely reset volatile state on subsequent creation attempts
                    wState = {
                        phrase: payload.phrase,
                        nextWotsIndex: 0,
                        nextMssIndex: 0,
                        wotsAddrs: {},
                        mssAddrs: {},
                        utxos: {},
                        history: [],
                        lastScannedHeight: 0
                    };

                    wallet = new WebWallet(payload.phrase);
                    
                    for (let i = 0; i < GAP_LIMIT; i++) {
                        deriveNextWots();
                        if (i % 10 === 0) {
                            self.postMessage({ 
                                type: 'MSS_PROGRESS', 
                                payload: { current: i, total: GAP_LIMIT, label: `Deriving base keys (${i}/${GAP_LIMIT})...` } 
                            });
                            await new Promise(r => setTimeout(r, 0));
                        }
                    }
                    
                    self.postMessage({ type: 'MSS_PROGRESS', payload: { current: 0, total: 100, label: "Generating Post-Quantum MSS Address..." } });
                    await new Promise(r => setTimeout(r, 10));
                    
                    deriveNextMss(10); 

                    await saveState();
                    self.postMessage({ type: 'WALLET_LOADED', payload: buildDashboardPayload() });
        }
        else if (type === 'LOGIN') {
            rpcUrl = payload.rpcUrl;
            await loadState(payload.password, payload.bundleStr);
        }
        else if (type === 'SCAN') {
            await performScan();
        }
        else if (type === 'RESCAN') {
            // Clears local volatile state and forces a blockchain rebuild
            wState.lastScannedHeight = 0;
            wState.utxos = {};
            wState.history = [];
            await saveState();
            await performScan();
        }
        else if (type === 'SEND') {
            if (isSending) throw new Error("A transaction is already in progress. Please wait for it to complete.");
            isSending = true;
            try {
                await performSend(payload.toAddress, payload.amount);
            } finally {
                isSending = false;
            }
        }
        else if (type === 'NEW_ADDRESS') {
            self.postMessage({ type: 'LOG', payload: "Deriving new receiving address..." });
            deriveNextMss(10);
            await saveState();
            self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
            self.postMessage({ type: 'LOG', payload: "New address generated successfully." });
        } 
        else if (type === 'REVEAL_SEED') {
            if (wState.phrase) {
                self.postMessage({ type: 'SEED_REVEALED', payload: wState.phrase });
            } else {
                self.postMessage({ type: 'ERROR', payload: "Seed phrase not found in memory." });
            }
        }
        else if (type === 'IMPORT_CLI') {
            try {
                // 1. Decrypt the binary .dat file using Rust/Wasm (Argon2)
                const cliJsonStr = decrypt_cli_wallet(payload.fileBytes, payload.password);
                const cliData = JSON.parse(cliJsonStr);

                if (!cliData.master_seed) throw new Error("Legacy (non-HD) wallets not supported in Web.");

                // 2. Map the CLI WalletData into the Web wState
                let newUtxos = {};
                for (const coin of cliData.coins) {
                    newUtxos[normalizeHex(coin.coin_id)] = {
                        index: 0, // CLI doesn't track index per coin, but web handles it gracefully
                        is_mss: cliData.mss_keys.some(k => normalizeHex(k.master_pk) === normalizeHex(coin.owner_pk)),
                        mss_height: 10,
                        mss_leaf: 0,
                        address: normalizeHex(coin.address),
                        value: coin.value,
                        salt: normalizeHex(coin.salt),
                        coin_id: normalizeHex(coin.coin_id)
                    };
                }

                let newMssAddrs = {};
                for (const mss of cliData.mss_keys) {
                    // Note: In CLI, the address is the hash of the master_pk. 
                    // Wasm will rebuild this properly during the first spend.
                    newMssAddrs[normalizeHex(mss.master_pk)] = {
                        index: 0, 
                        height: mss.height,
                        next_leaf: mss.next_leaf
                    };
                }

                wState = {
                    phrase: null, // We don't have the 24 words!
                    nextWotsIndex: cliData.next_wots_index || 0,
                    nextMssIndex: cliData.next_mss_index || 0,
                    wotsAddrs: {}, 
                    mssAddrs: newMssAddrs,
                    utxos: newUtxos,
                    history: cliData.history || [],
                    lastScannedHeight: cliData.last_scan_height || 0
                };

                // 3. Boot the Wasm wallet using the raw master seed!
                const seedHex = normalizeHex(cliData.master_seed);
                wallet = WebWallet.from_seed_hex(seedHex);
                password = payload.password; // Keep the password to save into web storage

                // 4. Save to the browser's local storage in the Web format
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
        
        // Strip the "Error: " prefix if present to keep UI clean
        let errMsg = err.toString();
        if (errMsg.startsWith("Error: ")) errMsg = errMsg.substring(7);
        self.postMessage({ type: 'ERROR', payload: errMsg });
    }
};

function deriveNextWots() {
    const addr = wallet.get_wots_address(wState.nextWotsIndex);
    wState.wotsAddrs[addr] = wState.nextWotsIndex;
    wState.nextWotsIndex++;
}
let lastMssUpdate = 0;
function deriveNextMss(height) {
    const progressCallback = (current, total) => {
        const now = Date.now();
        // FIX: Throttle UI updates to roughly 66ms (~15 FPS)
        // Always ensure the final 100% update gets sent
        if (now - lastMssUpdate > 66 || current === total) {
            lastMssUpdate = now;
            self.postMessage({ 
                type: 'MSS_PROGRESS', 
                payload: { current, total, label: `Hashing tree leaves (${current}/${total})...` } 
            });
        }
    };
    const addr = wallet.get_mss_address(wState.nextMssIndex, height, progressCallback);
    wState.mssAddrs[addr] = { index: wState.nextMssIndex, height, next_leaf: 0 };
    wState.nextMssIndex++;
}

function buildDashboardPayload() {
    const mssList = Object.keys(wState.mssAddrs);
    const utxoArray = Object.values(wState.utxos);
    
    const totalUtxoValue = utxoArray.reduce((s, u) => s + Number(u.value), 0);
    const pendingDeduction = pendingSends.reduce((s, tx) => s + tx.value + tx.fee, 0);
    const safeBalance = Math.max(0, totalUtxoValue - pendingDeduction);
    
    const combinedHistory = [...pendingSends, ...wState.history];
    const sortedHistory = combinedHistory.sort((a, b) => b.timestamp - a.timestamp);
    
    return {
        primaryAddress: mssList.length > 0 ? mssList[mssList.length - 1] : "None",
        balance: safeBalance,
        utxos: utxoArray,
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

async function performScan() {
    self.postMessage({ type: 'LOG', payload: "Fetching chain state..." });
    
    // FIX: Add cache: 'no-store'
    const stateResp = await fetch(`${rpcUrl}/state`, { cache: 'no-store' });
    const state = await stateResp.json();
    const chainHeight = state.height;

    if (chainHeight <= wState.lastScannedHeight) {
        self.postMessage({ type: 'SCAN_COMPLETE', payload: buildDashboardPayload() });
        return;
    }

    self.postMessage({ type: 'LOG', payload: `Scanning blocks ${wState.lastScannedHeight} to ${chainHeight}...` });
    
    let currentHeight = wState.lastScannedHeight;
    const BATCH_SIZE = 1000;

    updateWasmWatchlist();

    while (currentHeight < chainHeight) {
        const end = Math.min(currentHeight + BATCH_SIZE, chainHeight);
        const filterReq = await fetch(`${rpcUrl}/filters`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ start_height: currentHeight, end_height: end }),
            cache: 'no-store' 
        });
        
        if (!filterReq.ok) throw new Error("Failed to fetch filters.");
        const filterData = await filterReq.json();
        const numFilters = filterData.filters ? filterData.filters.length : 0;

        for (let i = 0; i < numFilters; i++) {
            const height = filterData.start_height + i;
            if (height % 100 === 0) self.postMessage({ type: 'SCAN_PROGRESS', payload: { height, max: chainHeight } });

            const n = filterData.element_counts ? filterData.element_counts[i] : 0;
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
            const req = await fetch(`${rpcUrl}/mss_state`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ master_pk: addrHex })
            });
            const res = await req.json();
            if (res.next_index > mss.next_leaf) {
                mss.next_leaf = res.next_index; 
            }
        } catch(e) {}
    }

    wState.lastScannedHeight = chainHeight;
    await saveState();
    self.postMessage({ type: 'SCAN_COMPLETE', payload: buildDashboardPayload() });
}

async function processFullBlock(height) {
// FIX: Add cache: 'no-store' and keepalive to speed up sequential requests
    const blockResp = await fetch(`${rpcUrl}/block/${height}`, { 
        cache: 'no-store',
        keepalive: true 
    });
    if (!blockResp.ok) return false;
    const block = await blockResp.json();
    
    let matchFound = false;

    const ourSalts = new Map();
    for (const [cid, u] of Object.entries(wState.utxos)) {
        ourSalts.set(u.salt, cid);
    }

    let coinbaseReceives = [];
    if (block.coinbase) {
        for (const cb of block.coinbase) {
            const addrHex = normalizeHex(cb.address);
            const saltHex = normalizeHex(cb.salt);
            if (wState.wotsAddrs[addrHex] !== undefined || wState.mssAddrs[addrHex] !== undefined) {
                const coinId = compute_coin_id_hex(addrHex, BigInt(cb.value), saltHex);
                if (addUtxo(addrHex, Number(cb.value), saltHex, coinId)) {
                    coinbaseReceives.push({ id: coinId, val: Number(cb.value) });
                }
                matchFound = true;
            }
        }
    }
    
    if (coinbaseReceives.length > 0) {
        const alreadyRecorded = wState.history.some(h => 
            h.outputs.some(out => coinbaseReceives.map(c=>c.id).includes(out))
        );
        if (!alreadyRecorded) {
            wState.history.push({
                kind: 'coinbase',
                timestamp: block.timestamp || Math.floor(Date.now() / 1000),
                fee: 0,
                inputs: [],
                outputs: coinbaseReceives.map(c => c.id),
                value: coinbaseReceives.reduce((sum, c) => sum + c.val, 0)
            });
        }
    }

    if (block.transactions) {
        for (const tx of block.transactions) {
            const reveal = tx.Reveal || tx.reveal;
            if (!reveal) continue;

            let spentIds = [];
            let spentValue = 0;
            let createdOutputs = [];

            if (reveal.inputs) {
                for (const inp of reveal.inputs) {
                    const saltHex = normalizeHex(inp.salt);
                    const cid = ourSalts.get(saltHex);
                    
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
                    const createdValue = createdOutputs.reduce((sum, c) => sum + c.val, 0);
                    
                    let totalTxIn = 0;
                    let totalTxOut = 0;
                    if (reveal.inputs) reveal.inputs.forEach(i => totalTxIn += Number(i.value));
                    if (reveal.outputs) reveal.outputs.forEach(o => {
                        let od = o.Standard || o.standard;
                        if (od) totalTxOut += Number(od.value);
                    });
                    let actualFee = totalTxIn - totalTxOut;
                    
                    let netSent = spentValue - createdValue - actualFee;
                    if (netSent < 0) netSent = 0; 
                    
                    wState.history.push({
                        kind: 'sent',
                        timestamp: block.timestamp || Math.floor(Date.now() / 1000),
                        fee: actualFee, 
                        inputs: spentIds,
                        outputs: createdOutputs.map(c => c.id),
                        value: netSent
                    });
                }
                
            } else if (createdOutputs.length > 0) {
                const alreadyRecorded = wState.history.some(h => 
                    h.outputs.some(out => createdOutputs.map(c=>c.id).includes(out))
                );
                
                if (!alreadyRecorded) {
                    const receivedValue = createdOutputs.reduce((sum, c) => sum + c.val, 0);
                    wState.history.push({
                        kind: 'received',
                        timestamp: block.timestamp || Math.floor(Date.now() / 1000),
                        fee: 0,
                        inputs: [],
                        outputs: createdOutputs.map(c => c.id),
                        value: receivedValue
                    });
                }
            }
        }
    }
    return matchFound;
}

function addUtxo(address, value, salt, coinId) {
    let index = 0;
    let is_mss = false;
    let mss_height = 0;
    let mss_leaf = 0;

    if (wState.wotsAddrs[address] !== undefined) {
        index = wState.wotsAddrs[address];
        while (wState.nextWotsIndex <= index + GAP_LIMIT) deriveNextWots(); 
    } else {
        const mss = wState.mssAddrs[address];
        index = mss.index;
        is_mss = true;
        mss_height = mss.height;
        mss_leaf = mss.next_leaf;
    }
    
    if (!wState.utxos[coinId]) {
        wState.utxos[coinId] = { index, is_mss, mss_height, mss_leaf, address, value, salt, coin_id: coinId };
        return true;
    }
    return false;
}

async function performSend(toAddress, amount) {
    
    for (const [addr, mss] of Object.entries(wState.mssAddrs)) {
        wallet.set_mss_leaf_index(addr, mss.next_leaf);
    }
    
    const utxoArray = Object.values(wState.utxos).map(u => {
        if (u.is_mss && wState.mssAddrs[u.address]) {
            return { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf };
        }
        return u;
    });
    
    let spendContextStr;
    try {
        spendContextStr = wallet.prepare_spend(JSON.stringify(utxoArray), toAddress, BigInt(amount), wState.nextWotsIndex);
    } catch (e) {
        throw new Error(`Failed to prepare transaction: ${e.toString()}.\n\nWhat to do: Ensure you have enough funds to cover the amount plus the network fee. Try running a Network Sync first.`);
    }

    const ctx = JSON.parse(spendContextStr);
    
    // Optimistic UI: instantly show pending transaction on the dashboard
    pendingSends.push({
        kind: 'pending',
        timestamp: Math.floor(Date.now() / 1000),
        fee: ctx.fee,
        inputs: ctx.selected_inputs.map(i => i.coin_id),
        outputs: [],
        value: Number(amount)
    });
    self.postMessage({ type: 'REFRESH_DASHBOARD', payload: buildDashboardPayload() });
    
    while (wState.nextWotsIndex < ctx.next_wots_index) {
        deriveNextWots();
    }
    
    const usedMssAddrs = new Set();
    for (const inp of ctx.selected_inputs) {
        if (inp.is_mss) usedMssAddrs.add(inp.address);
    }
    for (const addr of usedMssAddrs) {
        wState.mssAddrs[addr].next_leaf++;
    }
    
    await saveState();

    // Fetch current difficulty from the node
    const stateResp = await fetch(`${rpcUrl}/state`, { cache: 'no-store' });
    const stateData = await stateResp.json();
    const requiredPow = stateData.required_pow || 24;

    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Mining Proof-of-Work locally..." } });
    await new Promise(r => setTimeout(r, 50)); // Yield so toast renders before Wasm locks the thread

    // Mine the PoW nonce client-side in Wasm
    const spamNonce = Number(mine_commitment_pow(ctx.commitment, requiredPow));


    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Submitting commitment to network..." } });

    const commitReq = await fetch(`${rpcUrl}/commit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            commitment: ctx.commitment,
            spam_nonce: spamNonce
        })
    });

    if (!commitReq.ok) {
        let errText = await commitReq.text();
        try { errText = JSON.parse(errText).error || errText; } catch(e) {}
        throw new Error(`Commit rejected by network:\n${errText}\n\nWhat to do: The network might be congested, or your UTXOs might be out of sync. Your funds have not moved. Run a Network Sync and try again.`);
    }

    // Build reveal using the CLIENT's own commitment and salt (server no longer provides these)
    const revealPayloadStr = wallet.build_reveal(spendContextStr, ctx.commitment, ctx.tx_salt);

    // --- WAIT FOR COMMIT TO BE MINED ---
    let mempoolAccepted = false;
    for (let attempts = 0; attempts < 150; attempts++) {

        const revealReq = await fetch(`${rpcUrl}/send`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: revealPayloadStr
        });

        if (revealReq.ok) {
            mempoolAccepted = true;
            break;
        } else {
            let errText = await revealReq.text();
            try { errText = JSON.parse(errText).error || errText; } catch(e) {}
            
            // "No matching commitment" means the Commit block isn't mined yet
            if (errText.includes("No matching commitment found")) {
                await new Promise(r => setTimeout(r, 2000));
            } else {
                throw new Error(`Reveal rejected by network:\n${errText}\n\nWhat to do: A cryptographic error or double-spend occurred. Your funds are safe. Run a Network Sync and try again.`); 
            }
        }
    }

    if (!mempoolAccepted) throw new Error("Timed out waiting for Commit to be mined.\n\nWhat to do: Your funds are perfectly safe. The network dropped the transaction due to high traffic. Please try sending again in a few minutes.");

    // --- TRANSITION TO PHASE 2 ---
    self.postMessage({ type: 'SEND_PROGRESS', payload: { msg: "Commitment mined! Broadcasting transaction..." } });

    // --- WAIT FOR REVEAL TO BE MINED ---
    // We verify the Reveal is mined by checking if our input coin has been successfully spent (removed from the UTXO set).
    const inputCoinToCheck = ctx.selected_inputs[0].coin_id;
    let revealMined = false;
    for (let attempts = 0; attempts < 150; attempts++) {
        const checkReq = await fetch(`${rpcUrl}/check`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ coin: inputCoinToCheck })
        });
        
        if (checkReq.ok) {
            const checkResp = await checkReq.json();
            if (!checkResp.exists) {
                revealMined = true; // The coin is gone from the state. Reveal is mined!
                break;
            }
        }
        await new Promise(r => setTimeout(r, 2000));
    }

    if (!revealMined) throw new Error("Timed out waiting for Reveal to be mined. Your transaction is likely stuck in the mempool.");

    // --- FINALIZE LOCAL STATE ---
    pendingSends = [];
    
    for (const inp of ctx.selected_inputs) {
        delete wState.utxos[inp.coin_id];
    }

    let outIds = [];
    for (const out of ctx.outputs) {
        const addrHex = normalizeHex(out.address);
        if (wState.wotsAddrs[addrHex] !== undefined || wState.mssAddrs[addrHex] !== undefined) {
            const saltHex = normalizeHex(out.salt);
            const coinId = compute_coin_id_hex(addrHex, BigInt(out.value), saltHex);
            if (addUtxo(addrHex, Number(out.value), saltHex, coinId)) {
                outIds.push(coinId);
            }
        }
    }

    wState.history.push({
        kind: 'sent',
        timestamp: Math.floor(Date.now() / 1000),
        fee: ctx.fee,
        inputs: ctx.selected_inputs.map(i => i.coin_id),
        outputs: outIds,
        value: Number(amount)
    });

    await saveState();
    
    self.postMessage({ type: 'SEND_COMPLETE', payload: buildDashboardPayload() });
}
