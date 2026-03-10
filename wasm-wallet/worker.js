import init, { WebWallet, generate_phrase, compute_coin_id_hex } from './pkg/wasm_wallet.js';

let wallet = null;
let rpcUrl = "https://rpc.cypherpunk.gold";
let password = null;

const GAP_LIMIT = 20;
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
    
    // Bulletproof parsing to prevent TypeError / SyntaxError crashes
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
        
        // Backwards compatibility migrations
        if (Array.isArray(wState.utxos)) {
            const utxoMap = {};
            for (const u of wState.utxos) utxoMap[u.coin_id] = u;
            wState.utxos = utxoMap;
        }
        if (!wState.history) {
            wState.history = [];
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
            wallet = new WebWallet(payload.phrase);
            wState.phrase = payload.phrase;
            
            self.postMessage({ type: 'LOG', payload: "Deriving 20 WOTS lookahead addresses..." });
            
            for (let i = 0; i < GAP_LIMIT; i++) {
                deriveNextWots();
                if (i % 5 === 0) {
                    self.postMessage({ type: 'LOG', payload: `Derived ${i}/${GAP_LIMIT} addresses...` });
                    await new Promise(r => setTimeout(r, 0));
                }
            }
            
            self.postMessage({ type: 'LOG', payload: "Generating Reusable MSS Address..." });
            await new Promise(r => setTimeout(r, 10));
            
            deriveNextMss(5); 

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
        else if (type === 'SEND') {
            await performSend(payload.toAddress, payload.amount);
        }
        else if (type === 'NEW_ADDRESS') {
            self.postMessage({ type: 'LOG', payload: "Deriving new receiving address..." });
            deriveNextMss(5);
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
    } catch (err) {
        self.postMessage({ type: 'ERROR', payload: err.toString() });
    }
};

function deriveNextWots() {
    const addr = wallet.get_wots_address(wState.nextWotsIndex);
    wState.wotsAddrs[addr] = wState.nextWotsIndex;
    wState.nextWotsIndex++;
}

function deriveNextMss(height) {
    const progressCallback = (current, total) => {
        self.postMessage({ type: 'LOG', payload: `Generating Reusable Address: Hashed ${current}/${total} leaves...` });
    };
    const addr = wallet.get_mss_address(wState.nextMssIndex, height, progressCallback);
    wState.mssAddrs[addr] = { index: wState.nextMssIndex, height, next_leaf: 0 };
    wState.nextMssIndex++;
}

function buildDashboardPayload() {
    const mssList = Object.keys(wState.mssAddrs);
    const utxoArray = Object.values(wState.utxos);
    const safeBalance = utxoArray.reduce((s, u) => s + Number(u.value), 0);
    
    return {
        primaryAddress: mssList.length > 0 ? mssList[mssList.length - 1] : "None",
        balance: safeBalance,
        utxos: utxoArray,
        history: wState.history.slice().reverse() // Newest first
    };
}

async function performScan() {
    self.postMessage({ type: 'LOG', payload: "Fetching chain state..." });
    const stateResp = await fetch(`${rpcUrl}/state`);
    const state = await stateResp.json();
    const chainHeight = state.height;

    if (chainHeight <= wState.lastScannedHeight) {
        self.postMessage({ type: 'SCAN_COMPLETE', payload: buildDashboardPayload() });
        return;
    }

    self.postMessage({ type: 'LOG', payload: `Scanning blocks ${wState.lastScannedHeight} to ${chainHeight}...` });
    
    let currentHeight = wState.lastScannedHeight;
    const BATCH_SIZE = 1000;

    while (currentHeight < chainHeight) {
        const end = Math.min(currentHeight + BATCH_SIZE, chainHeight);
        const filterReq = await fetch(`${rpcUrl}/filters`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ start_height: currentHeight, end_height: end })
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
                await processFullBlock(height);
                continue;
            }

            const watchList = [
                ...Object.keys(wState.wotsAddrs), 
                ...Object.keys(wState.mssAddrs),
                ...Object.keys(wState.utxos) 
            ];
            
            if (wallet.check_filter(filterData.filters[i], blockHash, n, JSON.stringify(watchList))) {
                await processFullBlock(height);
            }
        }

        currentHeight += numFilters;
        if (currentHeight < end) {
            while (currentHeight < end) {
                await processFullBlock(currentHeight);
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
    const blockResp = await fetch(`${rpcUrl}/block/${height}`);
    if (!blockResp.ok) return;
    const block = await blockResp.json();
    
    let matchFound = false;
    let receivedInBlock = [];

    if (block.transactions) {
        for (const tx of block.transactions) {
            const reveal = tx.Reveal || tx.reveal;
            if (reveal && reveal.inputs) {
                let spentOurCoin = false;
                for (const inp of reveal.inputs) {
                    const saltHex = normalizeHex(inp.salt);
                    for (const [cid, u] of Object.entries(wState.utxos)) {
                        if (u.salt === saltHex) {
                            delete wState.utxos[cid];
                            self.postMessage({ type: 'LOG', payload: `Spent UTXO removed: ${cid.substring(0,8)}` });
                            spentOurCoin = true;
                            matchFound = true;
                        }
                    }
                }
                if (spentOurCoin) {
                    self.postMessage({ type: 'LOG', payload: `Spend detected! Advancing derivation window...` });
                    for (let i = 0; i < GAP_LIMIT; i++) deriveNextWots();
                }
            }
        }
    }

    if (block.coinbase) {
        for (const cb of block.coinbase) {
            const addrHex = normalizeHex(cb.address);
            const saltHex = normalizeHex(cb.salt);
            if (wState.wotsAddrs[addrHex] !== undefined || wState.mssAddrs[addrHex] !== undefined) {
                const coinId = compute_coin_id_hex(addrHex, BigInt(cb.value), saltHex);
                if (addUtxo(addrHex, Number(cb.value), saltHex, coinId)) {
                    receivedInBlock.push({ id: coinId, val: Number(cb.value) });
                }
                matchFound = true;
            }
        }
    }
    
    if (block.transactions) {
        for (const tx of block.transactions) {
            const reveal = tx.Reveal || tx.reveal;
            if (reveal && reveal.outputs) {
                for (const out of reveal.outputs) {
                    const outData = out.Standard || out.standard;
                    if (outData) {
                        const addrHex = normalizeHex(outData.address);
                        const saltHex = normalizeHex(outData.salt);
                        if (wState.wotsAddrs[addrHex] !== undefined || wState.mssAddrs[addrHex] !== undefined) {
                            const coinId = compute_coin_id_hex(addrHex, BigInt(outData.value), saltHex);
                            if (addUtxo(addrHex, Number(outData.value), saltHex, coinId)) {
                                receivedInBlock.push({ id: coinId, val: Number(outData.value) });
                            }
                            matchFound = true;
                        }
                    }
                }
            }
        }
    }

    // Write to transaction history
    if (receivedInBlock.length > 0) {
        wState.history.push({
            kind: 'received',
            timestamp: block.timestamp || Math.floor(Date.now() / 1000),
            fee: 0,
            inputs: [],
            outputs: receivedInBlock.map(c => c.id),
            value: receivedInBlock.reduce((sum, c) => sum + c.val, 0)
        });
    }

    if (matchFound) self.postMessage({ type: 'LOG', payload: `Processed match at block ${height}.` });
}

// Helper returns true if the UTXO was newly added
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
        self.postMessage({ type: 'LOG', payload: `>>> UTXO FOUND! Value: ${value}, ID: ${coinId.substring(0,8)}` });
        return true;
    }
    return false;
}

async function performSend(toAddress, amount) {
    self.postMessage({ type: 'LOG', payload: "Phase 1: Syncing Wallet State & Selecting Coins..." });
    
    for (const [addr, mss] of Object.entries(wState.mssAddrs)) {
        wallet.set_mss_leaf_index(addr, mss.next_leaf);
    }
    
    const utxoArray = Object.values(wState.utxos).map(u => {
        if (u.is_mss && wState.mssAddrs[u.address]) {
            return { ...u, mss_leaf: wState.mssAddrs[u.address].next_leaf };
        }
        return u;
    });
    
    const spendContextStr = wallet.prepare_spend(JSON.stringify(utxoArray), toAddress, BigInt(amount), wState.nextWotsIndex);
    const ctx = JSON.parse(spendContextStr);
    
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

    self.postMessage({ type: 'LOG', payload: `Selected ${ctx.selected_inputs.length} inputs. Exact fee calculated: ${ctx.fee}` });
    self.postMessage({ type: 'LOG', payload: "Submitting Commit to mempool..." });

    const commitReq = await fetch(`${rpcUrl}/commit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ctx.commit_payload)
    });
    
    if (!commitReq.ok) {
        let errText = await commitReq.text();
        try { errText = JSON.parse(errText).error || errText; } catch(e) {}
        throw new Error(errText);
    }
    const commitResp = await JSON.parse(await commitReq.text());

    self.postMessage({ type: 'LOG', payload: `Commit submitted (Salt: ${commitResp.salt.substring(0,8)}...). Generating Signatures...` });
    
    const revealPayloadStr = wallet.build_reveal(spendContextStr, commitResp.commitment, commitResp.salt);

    self.postMessage({ type: 'LOG', payload: "Waiting for Commit to hit the blockchain state..." });
    
    let revealed = false;
    for (let attempts = 0; attempts < 150; attempts++) {
        const revealReq = await fetch(`${rpcUrl}/send`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: revealPayloadStr
        });

        if (revealReq.ok) {
            revealed = true;
            break;
        } else {
            let errText = await revealReq.text();
            try { errText = JSON.parse(errText).error || errText; } catch(e) {}
            
            if (errText.includes("No matching commitment found")) {
                self.postMessage({ type: 'LOG', payload: `Commit not mined yet (Attempt ${attempts+1}/150). Waiting 2s...` });
                await new Promise(r => setTimeout(r, 2000));
            } else {
                throw new Error(errText); 
            }
        }
    }

    if (!revealed) throw new Error("Timed out waiting for Commit to be mined.");

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

    // Write to transaction history
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
