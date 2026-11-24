// net.js: how the wallet reaches the network, for the page. The wallet
// worker cannot open WebRTC connections (RTCPeerConnection does not exist in
// workers), so it asks the page: RPC_REQUEST → handleRpcRequest →
// RPC_RESPONSE, and the same for the mining pool (POOL_REQUEST).
//
// A request goes over WebRTC to a full node when a connection has proved it
// works (a successful ping), and otherwise to an HTTPS gateway, queued and
// rate-limited (gatewayFetch). Moved here from the old single-file page
// (classic.html) unchanged in what it does; only its hooks into the page are
// new: hooks.post (to the worker), hooks.log, hooks.change (status).

const hooks = {
    post: () => {},
    log: () => {},
    change: () => {},
};
function log(msg) { try { hooks.log(String(msg)); } catch (_) {} }

/** connecting | connected | disconnected | failed (WebRTC), and whether the
 *  HTTPS gateway answered last time it was asked. */
let webrtcStatus = 'disconnected';
let gatewayHealthy = null;
function noteGateway(ok) {
    if (gatewayHealthy !== ok) { gatewayHealthy = ok; hooks.change(); }
}

let lightClient = null;
let _rpcUrl = "https://rpc.cypherpunk.gold";

// ═══════════════════════════════════════════════════════════════
//  WebRTC Pool Client (main thread)
//  Same constraint as LightClient above: RTCPeerConnection does not
//  exist in a Web Worker, so the pool's libp2p connection lives here.
//  worker.js sends POOL_REQUEST messages; we execute them against
//  PoolClient and post back POOL_RESPONSE. Pushed jobs travel the other
//  way as POOL_NOTIFY.
//
//  Loaded from the ROLLUP BUNDLE, not the source: pool_client.js imports
//  bare specifiers ('libp2p', '@libp2p/webrtc') that browsers cannot
//  resolve natively — exactly why light_client.bundle.js exists.
//  Imported lazily so solo and HTTP-pool users never pay to download it.
// ═══════════════════════════════════════════════════════════════

let poolClient = null;


async function createPoolClient() {
    const { PoolClient } = await import('./pool_client.bundle.js');
    return new PoolClient();
}

async function handlePoolRequest(id, method, params) {
    // id 0 is a fire-and-forget teardown from the worker; it expects no reply.
    const reply = (result, error) => {
        if (id === 0) return;
        hooks.post({ type: 'POOL_RESPONSE', payload: { id, result, error } });
    };

    try {
        if (method === 'disconnect') {
            if (poolClient) { await poolClient.stop().catch(() => {}); poolClient = null; }
            return reply({ ok: true });
        }

        if (method === 'connect') {
            // Rebuild from scratch on every connect: the worker only
            // sends this when the configured address actually changed.
            if (poolClient) { await poolClient.stop().catch(() => {}); poolClient = null; }
            const client = await createPoolClient();
            client.onNotify((notif) => {
                if (notif?.method !== 'notify') return;
                hooks.post({ type: 'POOL_NOTIFY', payload: notif.params || {} });
            });
            client.onStatus((status, detail) => {
                log(`Pool transport ${status}${detail ? ` (${detail})` : ''}`);
            });
            const peer = await client.start(params.addr);
            poolClient = client;
            return reply({ peer });
        }

        if (!poolClient) throw new Error('Pool transport not connected');

        switch (method) {
            case 'getTemplate':
                return reply(await poolClient.getTemplate(params.address, params.midwimble_address));
            case 'getProof':
                return reply(await poolClient.getProof(params.address));
            case 'submitShare':
                return reply(await poolClient.submitShare(
                    params.address, params.job_id, params.nonce, params.worker, params.midwimble_address));
            default:
                throw new Error(`Unknown pool method: ${method}`);
        }
    } catch (e) {
        reply(undefined, e?.message || String(e));
    }
}

async function createLightClient() {
    const { LightClient } = await import('./light_client.bundle.js');
    return new LightClient();
}

function updateWebRTCStatus(status) {
    webrtcStatus = status;
    hooks.change();
}

async function findHeaviestNetworkState(rpcUrls) {
    log(`[light] Quorum check: Querying ${rpcUrls.length} nodes to find the heaviest chain...`);
    const promises = rpcUrls.map(async (url) => {
        try {
            const cleanUrl = url.trim().replace(/\/+$/, '');
            const resp = await fetch(`${cleanUrl}/state`, { signal: AbortSignal.timeout(3000) });
            if (!resp.ok) return null;
            const state = await resp.json();
            return { url: cleanUrl, state };
        } catch(e) { return null; }
    });
    
    const results = (await Promise.all(promises)).filter(r => r !== null);
    if (results.length === 0) throw new Error("Could not reach any nodes for quorum check.");
    
    // Sort descending by depth (using BigInt because depth is u128)
    results.sort((a, b) => {
        const dA = BigInt(a.state.depth);
        const dB = BigInt(b.state.depth);
        return dA < dB ? 1 : (dA > dB ? -1 : 0);
    });
    
    const bestNode = results[0];
    log(`[light] Consensus reached. Highest depth is ${bestNode.state.height} blocks on ${bestNode.url}.`);
    return bestNode;
}

async function connectWebRTC(customAddr) {
    updateWebRTCStatus('connecting');
    try {
        let dialList = [];

        if (customAddr) {
            // 1. MANUAL OVERRIDE MODE (Local Node)
            dialList.push(customAddr);
            log(`[light] Manual override. Dialing custom WebRTC peer: ${customAddr}`);
            
            const userRpc = (localStorage.getItem('customRpcUrl') || '').trim();
            if (userRpc) {
                _rpcUrl = userRpc;
                log(`[light] Locked RPC to custom URL: ${_rpcUrl}`);
            }
        } else {
            // 2. AUTO-DISCOVERY MODE (Public Network)
            const bootstrapUrls = [
                "https://rpc.cypherpunk.gold",      // Sydney (Default)
                "https://rpc.usa.cypherpunk.gold"   // USA
            ];

            try {
                // Try to connect to HTTPS API first to find the best chain
                let bestNode = await findHeaviestNetworkState(bootstrapUrls);
                
                const webrtcAddrs = (bestNode.state.webrtc_addrs || []).filter(a => a.includes('webrtc-direct') && a.includes('certhash'));
                if (webrtcAddrs.length > 0) {
                    const communityPeers = webrtcAddrs.slice(1).sort(() => Math.random() - 0.5);
                    dialList.push(...communityPeers, webrtcAddrs[0]); 
                }
                
                // Lock RPC URL to the trusted node
                _rpcUrl = bestNode.url; 
                
            } catch (httpError) {
                log("[light] HTTPS Bootstraps unreachable. Falling back to direct WebRTC...", "var(--accent)");
                
                // --- HARDCODED WEBRTC FALLBACKS ---
                dialList.push("/ip4/74.208.253.44/udp/9335/webrtc-direct/certhash/uEiCTNpt4FH1jR4lzQHWuIDnUTyWSr1N4bMniW5Prhh2K0w");
                dialList.push("/ip4/134.199.148.215/udp/9335/webrtc-direct/certhash/uEiDwFtheq3OCKVvl4CWkujRc-e3L_eGxgz0wuQFsLXS4tA");
            }
        }

        if (dialList.length === 0) throw new Error("No WebRTC addresses available to dial.");

        if (lightClient) {
            // node.stop() can stall tearing down a dead/zombie session;
            // don't let a hung teardown block the fresh dial. The old
            // client object is dropped either way.
            await Promise.race([
                lightClient.stop().catch(() => {}),
                new Promise(r => setTimeout(r, 5000)),
            ]);
            lightClient = null;
        }
        lightClient = await createLightClient();
        
        lightClient.onStatusChange((status) => {
            updateWebRTCStatus(status);
            if (status === 'connected') {
                log('[light] WebRTC connected securely.');
                lightClient.discoverPeers().catch(()=>{});
                _kaPing(true).catch(()=>{}); // Prove health immediately
            } else if (status === 'failed') {
                log('[light] WebRTC failed. Falling back to HTTPS.');
                _lcMarkBad('status changed to failed');
            }
        });

        lightClient.onPushEvent((payload) => {
            hooks.post({ type: 'PUSH_NEW_BLOCK', payload });
        });

        await lightClient.start(dialList);
        
    } catch (e) {
        log(`[light] Connect error: ${e.message}`, 'var(--red)');
        updateWebRTCStatus('failed');
        lightClient = null;
    }
}

async function autoConnectWebRTC() {
    try {
        // Check if user saved custom node settings previously
        const savedAddr = localStorage.getItem('customWebrtcAddr');
        const savedRpc = localStorage.getItem('customRpcUrl');
        
        if (savedRpc) {
            _rpcUrl = savedRpc;
        }
        
        await connectWebRTC(savedAddr || null);
    } catch (e) {
        log(`[light] Auto-connect error: ${e.message}`);
    }
}

async function disconnectWebRTC() {
    if (lightClient) {
        await Promise.race([
            lightClient.stop().catch(() => {}),
            new Promise(r => setTimeout(r, 5000)),
        ]);
        lightClient = null;
    }
    updateWebRTCStatus('disconnected');
}

// ── WebRTC keep-alive & health gating ───────────────────────────────────
// The transport can "connect" successfully but fail all requests (e.g. if
// the bundle's stream reader misinterprets protobuf framing). If we route
// traffic just because `isConnected` is true, sync dies.
// We now require the connection to PROVE it works (a successful ping) before
// it gets any traffic. If it fails repeatedly, we "park" it for 10 minutes
// instead of constantly tearing down and redialing, which lets HTTPS take
// over smoothly.
const _lcHealth = {
    ok: false,
    parkedUntil: 0,
    strikes: 0,
};

function _lcMarkOk() {
    _lcHealth.ok = true;
    _lcHealth.strikes = 0;
    _lcHealth.parkedUntil = 0;
}

function _lcMarkBad(reason) {
    _lcHealth.ok = false;
    console.warn(`[light] WebRTC health marked bad: ${reason}`);
}

const _ka = {
    EVERY_MS: 10_000,
    PING_TIMEOUT_MS: 8_000,
    inFlight: false,
    missed: 0,
    reconnecting: false,
    disconnectedSince: 0,
};

async function _kaForceReconnect(reason) {
    if (_ka.reconnecting) return;
    _ka.reconnecting = true;
    _lcMarkBad(reason);
    try {
        console.warn(`[light] keep-alive: ${reason} — dialing a fresh WebRTC connection`);
        let failsafe = null;
        try {
            await Promise.race([
                autoConnectWebRTC(),
                new Promise(r => { failsafe = setTimeout(r, 60_000); }),
            ]);
        } finally {
            if (failsafe) clearTimeout(failsafe);
        }
    } catch (e) {
        console.warn('[light] keep-alive reconnect failed:', e && e.message);
    } finally {
        _ka.reconnecting = false;
    }
}

async function _kaPing(isEventDriven = false) {
    if (!lightClient || !lightClient.isConnected || _ka.inFlight) return;
    _ka.inFlight = true;
    try {
        const st = await Promise.race([
            lightClient.getState(),
            new Promise((_, rej) => setTimeout(() => rej(new Error('keep-alive ping timed out')), _ka.PING_TIMEOUT_MS)),
        ]);
        _ka.missed = 0;
        _lcMarkOk();
        if (st) _blockCacheNoteTip(st.height);
    } catch (e) {
        _ka.missed++;
        if (!isEventDriven) {
            console.warn(`[light] keep-alive ping failed (${_ka.missed} in a row):`, e && e.message);
            if (_ka.missed >= 2) {
                _ka.missed = 0;
                _lcHealth.strikes++;
                if (_lcHealth.strikes >= 3) {
                    console.error('[light] WebRTC failed 3 reconnect cycles. Parking transport for 10 minutes to stabilize HTTPS fallback.');
                    _lcHealth.parkedUntil = Date.now() + 10 * 60_000;
                    _lcHealth.strikes = 0;
                    if (lightClient) lightClient.stop().catch(()=>{});
                    lightClient = null;
                } else {
                    if (lightClient && lightClient.isConnected) {
                        _kaForceReconnect('connection is a zombie (socket up, requests dead)');
                    }
                }
            }
        }
    } finally {
        _ka.inFlight = false;
    }
}

setInterval(async () => {
    if (Date.now() < _lcHealth.parkedUntil) return; // Shh.
    if (_ka.reconnecting) return;

    if (lightClient && !lightClient.isConnected) {
        if (!_ka.disconnectedSince) _ka.disconnectedSince = Date.now();
        if (Date.now() - _ka.disconnectedSince >= 60_000) {
            _ka.disconnectedSince = 0;
            _kaForceReconnect('still disconnected after 60 s (reconnect scheduler exhausted)');
        }
        return;
    }
    _ka.disconnectedSince = 0;
    await _kaPing();
}, _ka.EVERY_MS);

// Async wrapper for light client calls with an absolute timeout so a stalled
// stream doesn't hang the worker.
async function lcCall(promise, timeoutMs = 15000) {
    return Promise.race([
        promise,
        new Promise((_, rej) => setTimeout(() => rej(new Error('WebRTC request timed out')), timeoutMs))
    ]);
}



// ── Gateway request governor ────────────────────────────────────────────
// Every HTTPS request to the RPC gateway funnels through gatewayFetch():
//   • one queue, max 2 in flight, ≥150 ms between launches (~6 req/s cap),
//     so bulk chain scans can never stampede the gateway into 429s;
//   • 429-aware: honors Retry-After when the response is readable and
//     applies a global cooldown + exponential retry. A 429 whose response
//     lacks CORS headers surfaces here as TypeError('Failed to fetch'),
//     indistinguishable from being offline — so network-level failures
//     are retried the same way, with a capped backoff so a genuinely
//     unreachable gateway still fails in reasonable time;
//   • a priority lane for user-initiated traffic (commit/send/chat) so a
//     background scan never queues ahead of a payment;
//   • exhausted retries reject with err.code = 'RATE_LIMITED' or
//     'GATEWAY_UNAVAILABLE'. The code rides back to worker.js in the
//     RPC_RESPONSE, letting scan loops pause or abort as a whole instead
//     of grinding block-by-block through thousands of doomed requests.
const _gw = {
    queue: [],
    active: 0,
    MAX_CONCURRENT: 2,
    MIN_GAP_MS: 150,
    lastLaunch: 0,
    cooldownUntil: 0,   // epoch ms; global pause after a 429 / net failure
    pumpScheduled: false,
};

function _gwSetCooldown(ms) {
    const until = Date.now() + ms;
    if (until > _gw.cooldownUntil) _gw.cooldownUntil = until;
}

function _gwPump() {
    if (_gw.pumpScheduled) return;
    if (_gw.active >= _gw.MAX_CONCURRENT || _gw.queue.length === 0) return;
    const now = Date.now();
    const wait = Math.max(_gw.cooldownUntil - now, _gw.lastLaunch + _gw.MIN_GAP_MS - now, 0);
    if (wait > 0) {
        _gw.pumpScheduled = true;
        setTimeout(() => { _gw.pumpScheduled = false; _gwPump(); }, wait);
        return;
    }
    const job = _gw.queue.shift();
    _gw.active++;
    _gw.lastLaunch = now;
    job.run().catch(() => {}).finally(() => { _gw.active--; _gwPump(); });
    _gwPump(); // a second slot may be free
}

function gatewayFetch(input, init, opts = {}) {
    const { priority = false, retries = 3 } = opts;
    return new Promise((resolve, reject) => {
        const run = async () => {
            for (let attempt = 0; ; attempt++) {
                let resp = null, netErr = null;
                try { resp = await fetch(input, init); }
                catch (e) { netErr = e; }

                if (resp && resp.status !== 429) { noteGateway(true); resolve(resp); return; }

                // Rate-limited (readable 429) or blocked/offline (TypeError,
                // which is also what a CORS-header-less 429 looks like).
                let waitMs;
                if (resp) {
                    // Retry-After may be seconds or an HTTP-date; parseFloat
                    // handles the former, the exp fallback covers the rest.
                    const ra = parseFloat(resp.headers.get('Retry-After'));
                    waitMs = Number.isFinite(ra) ? Math.max(ra * 1000, 1000) : 2000 * Math.pow(2, attempt);
                } else {
                    waitMs = 1500 * Math.pow(2, attempt);
                }
                waitMs = Math.min(waitMs + Math.random() * 500, 30000);
                _gwSetCooldown(waitMs);

                if (attempt >= retries) {
                    const err = resp
                        ? new Error(`RATE_LIMITED: gateway returned 429 for ${input}`)
                        : new Error(`GATEWAY_UNAVAILABLE: ${(netErr && netErr.message) || 'fetch failed'} (gateway rate-limiting without CORS headers, or unreachable)`);
                    err.code = resp ? 'RATE_LIMITED' : 'GATEWAY_UNAVAILABLE';
                    if (!resp) noteGateway(false);
                    reject(err);
                    return;
                }
                await new Promise(r => setTimeout(r, waitMs));
            }
        };
        if (priority) _gw.queue.unshift({ run }); else _gw.queue.push({ run });
        _gwPump();
    });
}

// ── Immutable block cache ───────────────────────────────────────────────
// Blocks at least CONFIRM_DEPTH below the highest height we've seen can
// never change, so the 3-minute full-window orderbook rescan should not
// re-download them — before this cache it re-fetched ~1,440 identical
// blocks per cycle, which is what kept tripping the gateway limiter.
// Map preserves insertion order → cheap LRU. Session-scoped (in memory).
const _blockCache = {
    map: new Map(),                // height -> { block, bytes }
    bytes: 0,
    MAX_ENTRIES: 4600,             // deepest scan window (3×1440) + slack
    MAX_BYTES: 48 * 1024 * 1024,
    CONFIRM_DEPTH: 12,
    maxSeen: 0,
};

function _blockCacheNoteTip(h) {
    h = Number(h);
    if (Number.isFinite(h) && h > _blockCache.maxSeen) _blockCache.maxSeen = h;
}

function _blockCacheGet(h) {
    const e = _blockCache.map.get(h);
    if (!e) return null;
    _blockCache.map.delete(h); _blockCache.map.set(h, e);   // LRU touch
    return e.block;
}

function _blockCachePut(h, block, bytesHint) {
    if (!block) return;
    if (h > _blockCache.maxSeen - _blockCache.CONFIRM_DEPTH) return;  // near tip: may reorg
    if (_blockCache.map.has(h)) return;
    let bytes = bytesHint;
    if (!Number.isFinite(bytes)) {
        try { bytes = JSON.stringify(block).length; } catch (_) { bytes = 16384; }
    }
    _blockCache.map.set(h, { block, bytes });
    _blockCache.bytes += bytes;
    while (_blockCache.map.size > 0 &&
           (_blockCache.map.size > _blockCache.MAX_ENTRIES || _blockCache.bytes > _blockCache.MAX_BYTES)) {
        const [oldH, oldE] = _blockCache.map.entries().next().value;
        _blockCache.map.delete(oldH);
        _blockCache.bytes -= oldE.bytes;
    }
}

// Unified single-block fetch: cache → WebRTC (loudly, not silently,
// falling back) → throttled HTTPS gateway. Returns null for a block the
// gateway says doesn't exist, matching the old semantics.
async function getBlockUnified(lc, url, height) {
    _blockCacheNoteTip(height);
    const hit = _blockCacheGet(height);
    if (hit) return hit;

    if (lc) {
        try {
            const b = await lcCall(lc.getBlock(height));
            _blockCachePut(height, b);
            _lcMarkOk();
            return b;
        } catch (e) {
            _lcMarkBad(e.message);
            console.warn(`[rpc] WebRTC getBlock(${height}) failed — falling back to the HTTPS gateway:`, e && e.message);
        }
    }

    const r = await gatewayFetch(`${url}/block/${height}`, { cache: 'no-store', keepalive: true });
    if (!r.ok) return null;
    const text = await r.text();
    let block = null;
    try { block = JSON.parse(text); } catch (_) { return null; }
    _blockCachePut(height, block, text.length);
    return block;
}

// ── RPC Bridge ──────────────────────────────────────────────────────────
// Handles RPC_REQUEST messages from worker.js. Executes the network call
// (via LightClient if connected, else plain HTTPS fetch) and posts back
// RPC_RESPONSE so the worker's awaiting promise resolves.

async function handleRpcRequest(id, method, params) {
    const respond = (result, error, code) => hooks.post({ type: 'RPC_RESPONSE', payload: { id, result, error, code } });
    try {
        // HEALTH GATING: ONLY route traffic if a ping has successfully proven the
        // stream reader can actually unframe responses.
        const lc  = (lightClient?.isConnected && _lcHealth.ok) ? lightClient : null;
        const url = _rpcUrl.replace(/\/+$/, '');
        let result;

        if (method === 'getState') {
            let success = false;
            if (lc) {
                try { result = await lcCall(lc.getState()); _lcMarkOk(); success = true; }
                catch (e) { _lcMarkBad(e.message); console.warn("WebRTC getState failed, falling back to HTTPS:", e.message); }
            }
            if (!success) {
                result = await gatewayFetch(`${url}/state`, { cache: 'no-store' }).then(r => r.json());
            }
            if (result) _blockCacheNoteTip(result.height);

        } else if (method === 'getMempool') {
            let success = false;
            if (lc) {
                try { result = await lcCall(lc.getMempool()); _lcMarkOk(); success = true; }
                catch (e) { _lcMarkBad(e.message); console.warn("WebRTC getMempool failed, falling back to HTTPS:", e.message); }
            }
            if (!success) {
                const r = await gatewayFetch(`${url}/mempool`, { cache: 'no-store' }); 
                result = r.ok ? await r.json() : { size: 0, transactions: [] }; 
            }

        } else if (method === 'getBlock') {
            // Cache → WebRTC (with a warning on failure) → throttled gateway.
            result = await getBlockUnified(lc, url, params.height);

        } else if (method === 'getFilters') {
            let success = false;
            if (lc) { 
                try { 
                    result = await lcCall(lc.getFilters(params.startHeight, params.endHeight)); 
                    _lcMarkOk();
                    success = true;
                } catch(e) { _lcMarkBad(e.message); console.warn("WebRTC filter fetch failed, falling back to HTTPS"); }
            }
            if (!success) {
                const r = await gatewayFetch(`${url}/filters`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ start_height: params.startHeight, end_height: params.endHeight }), cache: 'no-store' });
                if (!r.ok) throw new Error('Failed to fetch filters.');
                result = await r.json();
            }

        } else if (method === 'getMssState') {
            let success = false;
            if (lc) {
                try { result = await lcCall(lc.mssState(params.masterPkHex)); _lcMarkOk(); success = true; }
                catch (e) { _lcMarkBad(e.message); console.warn("WebRTC mss_state failed, falling back to HTTPS:", e.message); }
            }
            if (!success) {
                const r = await gatewayFetch(`${url}/mss_state`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ master_pk: params.masterPkHex }), cache: 'no-store' }, { priority: true });
                if (!r.ok) throw new Error('mss_state fetch failed: HTTP ' + r.status);
                result = await r.json();
            }

        } else if (method === 'getBlockTemplate') {
            if (lightClient && lightClient.isConnected) { // Template ignores health gate (mining is WebRTC only)
                try {
                    const resp = await lcCall(lightClient.getBlockTemplate(params.coinbase));
                    result = { ok: resp.ok, status: resp.status ?? (resp.ok ? 200 : 500), body: resp.ok ? await resp.json() : await resp.text() };
                } catch (e) {
                    result = { ok: false, status: 500, body: e.message };
                }
            } else {
                result = { ok: false, status: 403, body: 'Mining is only available over WebRTC. Connect to a full node via a bootstrap multiaddr to mine.' };
            }

        } else if (method === 'submitBatch') {
            if (lightClient && lightClient.isConnected) {
                try {
                    const resp = await lcCall(lightClient.submitBatch(params.batch));
                    result = { ok: resp.ok, body: resp.ok ? null : (resp.error || 'rejected') };
                    _lcMarkOk();
                } catch (e) {
                    _lcMarkBad(e.message);
                    console.warn('[rpc] submitBatch via WebRTC failed, attempting WebRTC reconnect:', e.message);
                    if (lightClient && !lightClient.isConnected) {
                        for (const addr of lightClient.knownMultiaddrs) {
                            try { await lightClient.connectTo(addr); break; } catch(_) {}
                        }
                    }
                    if (lightClient?.isConnected) {
                        const resp = await lcCall(lightClient.submitBatch(params.batch));
                        result = { ok: resp.ok, body: resp.ok ? null : (resp.error || 'rejected') };
                    } else {
                        throw new Error('Block submission failed: WebRTC connection lost and mining does not fall back to HTTPS.');
                    }
                }
            } else {
                result = { ok: false, body: 'Block submission is only available over WebRTC. Connect to a full node to submit mined blocks.' };
            }

        } else if (method === 'commit') {
            let submitted = false;
            if (lc) {
                try {
                    const resp = await lcCall(lc.commit(params.commitmentHex, params.spamNonce));
                    result = { ok: resp.ok, body: resp.ok ? null : (resp.error || 'rejected') };
                    _lcMarkOk();
                    submitted = true;
                } catch (e) {
                    _lcMarkBad(e.message);
                    console.warn('[rpc] commit via WebRTC failed, trying HTTPS:', e.message);
                }
            }
            if (!submitted) {
                const r = await gatewayFetch(`${url}/commit`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ commitment: params.commitmentHex, spam_nonce: params.spamNonce }) }, { priority: true });
                result = { ok: r.ok, body: r.ok ? null : await r.text() };
            }

        } else if (method === 'send') {
            let submitted = false;
            if (lc) {
                try {
                    const parsed = typeof params.revealPayload === 'string' ? JSON.parse(params.revealPayload) : params.revealPayload;
                    const resp = await lcCall(lc.send(parsed));
                    result = { ok: resp.ok, body: resp.ok ? null : (resp.error || 'rejected') };
                    _lcMarkOk();
                    submitted = true;
                } catch (e) {
                    _lcMarkBad(e.message);
                    console.warn('[rpc] send via WebRTC failed, trying HTTPS:', e.message);
                }
            }
            if (!submitted) {
                const r = await gatewayFetch(`${url}/send`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: params.revealPayload }, { priority: true });
                result = { ok: r.ok, body: r.ok ? null : await r.text() };
            }

        } else if (method === 'checkCoin') {
            let success = false;
            if (lc) {
                try {
                    result = await lcCall(lc.checkCoin(params.coinHex));
                    _lcMarkOk();
                    success = true;
                } catch (e) { _lcMarkBad(e.message); console.warn('[rpc] checkCoin via WebRTC failed:', e.message); }
            }
            if (!success) { 
                const r = await gatewayFetch(`${url}/check`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ coin: params.coinHex }) }); 
                result = r.ok ? await r.json() : null; 
            }

        } else if (method === 'checkCommitment') {
            let success = false;
            if (lc) {
                try {
                    result = await lcCall(lc.checkCommitment(params.commitmentHex));
                    _lcMarkOk();
                    success = true;
                } catch (e) { _lcMarkBad(e.message); console.warn('[rpc] checkCommitment via WebRTC failed:', e.message); }
            }
            if (!success) { 
                const r = await gatewayFetch(`${url}/check_commitment`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ commitment: params.commitmentHex }) }, { priority: true }); 
                result = r.ok ? await r.json() : null; 
            }
        } else if (method === 'sendChat') {
            const attachments = params.attachments || [];
            let submitted = false;
            if (lc) {
                try {
                    const resp = await lcCall(lc.sendChat(params.words, params.replyTo, attachments));
                    result = { ok: resp.ok, body: resp.ok ? null : (resp.error || 'rejected') };
                    _lcMarkOk();
                    submitted = true;
                } catch(e) {
                    _lcMarkBad(e.message);
                    console.warn('[rpc] sendChat WebRTC failed, trying HTTPS:', e.message);
                }
            } 
            if (!submitted) {
                const r = await gatewayFetch(`${url}/api/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        words: params.words,
                        reply_to: params.replyTo,
                        attachments,
                    }),
                }, { priority: true });
                result = { ok: r.ok, body: r.ok ? null : await r.text() };
            }
        } else if (method === 'submitChat') {
            const attachments = params.attachments || [];
            let submitted = false;
            if (lc) {
                try {
                    const resp = await lcCall(lc.submitChat(params.sender, params.timestamp, params.nonce, params.replyTo, params.words, attachments));
                    result = { ok: resp.ok, body: resp.ok ? null : (resp.error || 'rejected') };
                    _lcMarkOk();
                    submitted = true;
                } catch(e) {
                    _lcMarkBad(e.message);
                    console.warn('[rpc] submitChat WebRTC failed, trying HTTPS:', e.message);
                }
            } 
            if (!submitted) {
                const r = await gatewayFetch(`${url}/api/chat/submit`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sender: params.sender,
                        timestamp: params.timestamp,
                        nonce: params.nonce,
                        reply_to: params.replyTo,
                        words: params.words,
                        attachments,
                    }),
                }, { priority: true });
                result = { ok: r.ok, body: r.ok ? null : await r.text() };
            }
            
        } else if (method === 'getNameProof') {
            // A name's leaf, records and Merkle proof (names.js checks them).
            // The light protocol has no request for it, so HTTPS always.
            const r = await gatewayFetch(`${url}/name/${encodeURIComponent(params.name)}/proof`, { cache: 'no-store' }, { priority: true });
            const text = await r.text();
            try { result = JSON.parse(text); }
            catch (_) { throw new Error(r.ok ? 'The node’s answer about the name is not JSON' : `The node could not answer about the name (HTTP ${r.status})`); }

        } else if (method === 'getChat') {
            const r = await gatewayFetch(`${url}/api/chat`, { cache: 'no-store' });
            result = r.ok ? await r.json() : { messages: [], dictionary: [] };
        } else {
            throw new Error(`Unknown RPC method: ${method}`);
        }

        respond(result);
    } catch (e) {
        respond(undefined, e.message, e && e.code);
    }
}

// ── What the page uses ──────────────────────────────────────────────────

/** Wire the network to the page: `post` sends to the wallet worker. */
export function attach(h) { Object.assign(hooks, h); }

export { handleRpcRequest, handlePoolRequest, connectWebRTC, disconnectWebRTC };

/** Connect, to a node of the user's own if one is set. */
export const connect = () => autoConnectWebRTC();

/** Use a node of the user's own: a WebRTC multiaddr, and the HTTPS RPC that
 *  goes with it. Both empty: back to the public nodes. */
export function useNode(addr, rpc) {
    const a = String(addr || '').trim();
    const r = String(rpc || '').trim().replace(/\/+$/, '');
    try {
        if (a) localStorage.setItem('customWebrtcAddr', a); else localStorage.removeItem('customWebrtcAddr');
        if (r) localStorage.setItem('customRpcUrl', r); else localStorage.removeItem('customRpcUrl');
    } catch (_) {}
    _rpcUrl = r || 'https://rpc.cypherpunk.gold';
    return a ? connectWebRTC(a) : disconnectWebRTC().then(() => autoConnectWebRTC());
}

export const customNode = () => ({
    addr: (() => { try { return localStorage.getItem('customWebrtcAddr') || ''; } catch (_) { return ''; } })(),
    rpc: (() => { try { return localStorage.getItem('customRpcUrl') || ''; } catch (_) { return ''; } })(),
});

export const isWebRTC = () => !!(lightClient && lightClient.isConnected);
export const rpcUrl = () => _rpcUrl;

/** How the wallet reaches the network, in two words and a class. */
export function summary() {
    if (typeof navigator !== 'undefined' && navigator.onLine === false) return { cls: 'bad', text: 'Offline' };
    if (lightClient && lightClient.isConnected && _lcHealth.ok) return { cls: 'ok', text: 'Connected' };
    if (gatewayHealthy === true) return { cls: 'ok', text: 'Connected' };
    if (gatewayHealthy === false && webrtcStatus !== 'connecting') return { cls: 'bad', text: 'Offline' };
    return { cls: 'busy', text: 'Connecting' };
}

/** For Settings → Network. */
export function details() {
    return {
        webrtc: webrtcStatus,
        healthy: !!_lcHealth.ok,
        gateway: gatewayHealthy,
        rpc: _rpcUrl,
        peers: lightClient && lightClient.knownMultiaddrs ? [...lightClient.knownMultiaddrs].length : 0,
    };
}
