#!/usr/bin/env node
// hub.mjs — headless Q-Bolt hub. Runs the UNMODIFIED worker.js protocol engine
// under Node (>=20) by replacing the browser's main thread: HTTP RPC to a full
// node, file-backed wallet + MSS-tree storage, and the block/chat poll loops.
// Zero npm dependencies. Place next to worker.js and pkg/.
//
//   MDS_NODE_URL=http://127.0.0.1:8332 HUB_PASSWORD=secret node hub.mjs create
//   MDS_NODE_URL=... HUB_PASSWORD=... node hub.mjs import "word1 word2 ... word24"
//   MDS_NODE_URL=... HUB_PASSWORD=... node hub.mjs run          # (default)
//   MDS_NODE_URL=... HUB_SMOKE=1 node hub.mjs run               # plumbing check, no wallet needed
//
// Optional env: HUB_DIR (default ./hub-data), HUB_STATE_POLL_MS (10000),
// HUB_CHAT_POLL_MS (30000), HUB_STATUS_MS (60000), HUB_VERBOSE=1
//
// 24/7 via systemd (auto-restart on crash):
//   [Service]
//   WorkingDirectory=/path/to/wallet-dir
//   Environment=MDS_NODE_URL=http://127.0.0.1:8332 HUB_PASSWORD=...
//   ExecStart=/usr/bin/node hub.mjs run
//   Restart=always
//   RestartSec=5
//   [Install]
//   WantedBy=multi-user.target

import fs from 'node:fs';
import path from 'node:path';
import url from 'node:url';

const [MAJOR] = process.versions.node.split('.').map(Number);
if (MAJOR < 20) { console.error(`Node >= 20 required (found ${process.version}).`); process.exit(1); }

const HERE      = path.dirname(url.fileURLToPath(import.meta.url));
const CMD       = process.argv[2] || 'run';
const NODE_URL  = (process.env.MDS_NODE_URL || '').replace(/\/+$/, '');
const PASSWORD  = process.env.HUB_PASSWORD || '';
const HUB_DIR   = path.resolve(process.env.HUB_DIR || path.join(HERE, 'hub-data'));
const IDB_DIR   = path.join(HUB_DIR, 'idb');
const WALLET_F  = path.join(HUB_DIR, 'wallet.json');
const POLL_MS   = Number(process.env.HUB_STATE_POLL_MS || 10000);
const CHAT_MS   = Number(process.env.HUB_CHAT_POLL_MS || 30000);
const STATUS_MS = Number(process.env.HUB_STATUS_MS || 60000);
const VERBOSE   = process.env.HUB_VERBOSE === '1';
const SMOKE     = process.env.HUB_SMOKE === '1';

if (!NODE_URL) { console.error('Set MDS_NODE_URL to your full node RPC (e.g. http://127.0.0.1:8332).'); process.exit(1); }
fs.mkdirSync(IDB_DIR, { recursive: true });

const ts  = () => new Date().toISOString().replace('T', ' ').slice(0, 19);
const log = (tag, msg) => console.log(`${ts()} [${tag}] ${msg}`);

// ═══ browser shims (must exist BEFORE worker.js is imported) ═════════════════

globalThis.self = globalThis;

// fetch: the wasm-bindgen web glue loads wasm_wallet_bg.wasm via fetch(file://).
const realFetch = globalThis.fetch.bind(globalThis);
globalThis.fetch = async (input, opts) => {
    const s = typeof input === 'string' ? input : (input && input.url) || String(input);
    if (s.startsWith('file://')) {
        const bytes = await fs.promises.readFile(url.fileURLToPath(s));
        return new Response(bytes, { status: 200, headers: { 'content-type': 'application/wasm' } });
    }
    return realFetch(input, opts);
};

// indexedDB: worker.js stores MSS tree blobs (Uint8Array) via open/put/get/delete
// on a single object store. File-backed emulation of exactly that surface.
const idbFile = (key) => path.join(IDB_DIR, encodeURIComponent(key));
function idbRequest(fill) {
    const req = { onsuccess: null, onerror: null, onupgradeneeded: null, result: undefined, error: null };
    queueMicrotask(async () => {
        try { req.result = await fill(req); req.onsuccess && req.onsuccess({ target: req }); }
        catch (e) { req.error = e; req.onerror ? req.onerror({ target: req }) : log('idb', `error: ${e.message}`); }
    });
    return req;
}
globalThis.indexedDB = {
    open() {
        return idbRequest(async (req) => {
            const db = {
                objectStoreNames: { contains: () => true },
                createObjectStore: () => ({}),
                close: () => {},
                transaction: () => {
                    const tx = { oncomplete: null, onerror: null, error: null };
                    let pending = 0, done = false;
                    const settle = () => { if (done && pending === 0 && tx.oncomplete) queueMicrotask(() => tx.oncomplete({})); };
                    queueMicrotask(() => { done = true; settle(); });
                    const track = (p, req2) => {
                        pending++;
                        p.then((v) => { req2.result = v; req2.onsuccess && req2.onsuccess({ target: req2 }); })
                         .catch((e) => { req2.error = e; tx.error = e; (req2.onerror || tx.onerror || (() => {}))({ target: req2 }); })
                         .finally(() => { pending--; settle(); });
                    };
                    return {
                        ...tx,
                        set oncomplete(f) { tx.oncomplete = f; settle(); },
                        get oncomplete() { return tx.oncomplete; },
                        set onerror(f) { tx.onerror = f; },
                        get onerror() { return tx.onerror; },
                        get error() { return tx.error; },
                        objectStore: () => ({
                            put: (value, key) => { const r = { onsuccess: null, onerror: null };
                                track((async () => {
                                    if (value instanceof Uint8Array || Buffer.isBuffer(value)) {
                                        await fs.promises.writeFile(idbFile(key) + '.bin', value);
                                        await fs.promises.rm(idbFile(key) + '.json', { force: true });
                                    } else {
                                        await fs.promises.writeFile(idbFile(key) + '.json', JSON.stringify(value));
                                        await fs.promises.rm(idbFile(key) + '.bin', { force: true });
                                    }
                                })(), r); return r; },
                            get: (key) => { const r = { onsuccess: null, onerror: null };
                                track((async () => {
                                    try { return new Uint8Array(await fs.promises.readFile(idbFile(key) + '.bin')); }
                                    catch (_) { /* fall through */ }
                                    try { return JSON.parse(await fs.promises.readFile(idbFile(key) + '.json', 'utf8')); }
                                    catch (_) { return undefined; }
                                })(), r); return r; },
                            delete: (key) => { const r = { onsuccess: null, onerror: null };
                                track((async () => {
                                    await fs.promises.rm(idbFile(key) + '.bin', { force: true });
                                    await fs.promises.rm(idbFile(key) + '.json', { force: true });
                                })(), r); return r; },
                        }),
                    };
                },
            };
            // Signal first-run schema creation the way the worker expects.
            if (req.onupgradeneeded) { req.result = db; req.onupgradeneeded({ target: req }); }
            return db;
        });
    },
};

// ═══ main-thread replacement: message routing, RPC bridge, persistence ═══════

const waiters = new Map();               // message type -> [resolve, ...]
const waitFor = (type, timeoutMs = 0) => new Promise((resolve, reject) => {
    if (!waiters.has(type)) waiters.set(type, []);
    waiters.get(type).push(resolve);
    if (timeoutMs) setTimeout(() => reject(new Error(`timed out waiting for ${type}`)), timeoutMs).unref();
});
let lastDash = null;
let saving = Promise.resolve();

globalThis.self.postMessage = (msg) => {
    const { type, payload } = msg || {};
    const w = waiters.get(type);
    if (w && w.length) w.splice(0).forEach((r) => r(payload));

    switch (type) {
        case 'RPC_REQUEST': rpcBridge(payload).catch((e) => log('rpc', `bridge crash: ${e.message}`)); break;
        case 'SAVE_WALLET':
            // Atomic write, serialized: a torn wallet.json is unrecoverable.
            saving = saving.then(async () => {
                const tmp = WALLET_F + '.tmp';
                await fs.promises.writeFile(tmp, payload);
                await fs.promises.rename(tmp, WALLET_F);
            }).catch((e) => log('wallet', `SAVE FAILED: ${e.message}`));
            break;
        case 'LOG':            if (VERBOSE) log('worker', String(payload)); break;
        case 'L2_EVENT':       log(`l2:${payload.level}`, `${payload.msg}${payload.channelId ? ` [${String(payload.channelId).slice(0, 12)}…]` : ''}`); break;
        case 'L2_HTLC_CLAIMED': log('l2:recv', `claimed ${payload.amount} MDS (${String(payload.secretHash).slice(0, 12)}…)`); break;
        case 'ERROR':
        case 'SEND_ERROR':     log('error', typeof payload === 'string' ? payload : (payload && payload.error) || JSON.stringify(payload)); break;
        case 'MSS_PROGRESS':   if (VERBOSE) log('mss', payload.label || `${payload.current}/${payload.total}`); break;
        case 'REFRESH_DASHBOARD': lastDash = payload; break;
        case 'SCAN_PROGRESS':  if (VERBOSE) log('scan', JSON.stringify(payload)); break;
        default:               if (VERBOSE) log('msg', type); break; // AUTO_CONNECT_WEBRTC, CHAT_HISTORY, etc.
    }
};

// Deliver a message TO the worker exactly as the browser would.
const send = (type, payload) => globalThis.self.onmessage({ data: { type, payload } });

// HTTP RPC — the exact endpoint mapping from index.html's handleRpcRequest,
// HTTPS paths only (no light client). Mining methods are intentionally absent.
async function http(pathname, opts = {}, raw = false) {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 60000);
    try { return await realFetch(`${NODE_URL}${pathname}`, { ...opts, signal: ctl.signal }); }
    finally { clearTimeout(t); }
}
const j = (o) => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(o) });

async function rpcBridge({ id, method, params }) {
    const respond = (result, error, code) => send('RPC_RESPONSE', { id, result, error, code });
    try {
        let result;
        if (method === 'getState')            result = await (await http('/state')).json();
        else if (method === 'getMempool')     { const r = await http('/mempool'); result = r.ok ? await r.json() : { size: 0, transactions: [] }; }
        else if (method === 'getBlock')       { const r = await http(`/block/${params.height}`); result = r.ok ? JSON.parse(await r.text()) : null; }
        else if (method === 'getFilters')     { const r = await http('/filters', j({ start_height: params.startHeight, end_height: params.endHeight })); if (!r.ok) throw new Error('Failed to fetch filters.'); result = await r.json(); }
        else if (method === 'getMssState')    { const r = await http('/mss_state', j({ master_pk: params.masterPkHex })); if (!r.ok) throw new Error('mss_state fetch failed: HTTP ' + r.status); result = await r.json(); }
        else if (method === 'commit')         { const r = await http('/commit', j({ commitment: params.commitmentHex, spam_nonce: params.spamNonce })); result = { ok: r.ok, body: r.ok ? null : await r.text() }; }
        else if (method === 'send')           { const r = await http('/send', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: params.revealPayload }); result = { ok: r.ok, body: r.ok ? null : await r.text() }; }
        else if (method === 'checkCoin')      { const r = await http('/check', j({ coin: params.coinHex })); result = r.ok ? await r.json() : null; }
        else if (method === 'checkCommitment'){ const r = await http('/check_commitment', j({ commitment: params.commitmentHex })); result = r.ok ? await r.json() : null; }
        else if (method === 'getChat')        { const r = await http('/api/chat'); result = r.ok ? await r.json() : { messages: [], dictionary: [] }; }
        else if (method === 'sendChat')       { const r = await http('/api/chat', j({ words: params.words, reply_to: params.replyTo, attachments: params.attachments || [] })); result = { ok: r.ok, body: r.ok ? null : await r.text() }; }
        else if (method === 'submitChat')     { const r = await http('/api/chat/submit', j({ sender: params.sender, timestamp: params.timestamp, nonce: params.nonce, reply_to: params.replyTo, words: params.words, attachments: params.attachments || [] })); result = { ok: r.ok, body: r.ok ? null : await r.text() }; }
        else if (method === 'getBlockTemplate' || method === 'submitBatch') result = { ok: false, status: 403, body: 'Mining is not supported on the headless hub.' };
        else throw new Error(`Unknown RPC method: ${method}`);
        respond(result);
    } catch (e) { respond(undefined, e.message, e && e.code); }
}

// ═══ ESM bootstrap: load the real worker.js unmodified (modulo module type) ══
// The app's package.json is CommonJS, so worker.js/pkg can't be imported as-is.
// Generate .mjs twins (byte-identical apart from one import specifier) whenever
// the sources are newer, then import the twin.

function freshen(src, dst, rewrite) {
    const s = fs.statSync(src);
    if (fs.existsSync(dst) && fs.statSync(dst).mtimeMs >= s.mtimeMs) return;
    let text = fs.readFileSync(src, 'utf8');
    if (rewrite) text = rewrite(text);
    fs.writeFileSync(dst, text);
    log('hub', `generated ${path.basename(dst)} from ${path.basename(src)}`);
}

async function loadWorker() {
    freshen(path.join(HERE, 'pkg', 'wasm_wallet.js'), path.join(HERE, 'pkg', 'wasm_wallet.mjs'));
    freshen(path.join(HERE, 'worker.js'), path.join(HERE, '.hub_worker.mjs'),
        (t) => t.replace("'./pkg/wasm_wallet.js'", "'./pkg/wasm_wallet.mjs'"));
    await import(url.pathToFileURL(path.join(HERE, '.hub_worker.mjs')).href);
    send('INIT');
    await waitFor('INIT_DONE', 120000);
    log('hub', 'WASM engine initialized');
}

// ═══ run loops ═══════════════════════════════════════════════════════════════

let lastHeight = 0;
const timers = [];
function startLoops() {
    timers.push(setInterval(async () => {
        try {
            const r = await http('/state');
            if (!r.ok) return;
            const st = await r.json();
            if (st.height > lastHeight) {
                lastHeight = st.height;
                // Same shape the browser pushes; drives networkHeight + qbWatchTick.
                send('PUSH_NEW_BLOCK', { NewBlockTip: { height: st.height, block_hash: st.header_hash } });
            }
        } catch (e) { log('poll', `node unreachable: ${e.message}`); }
    }, POLL_MS));
    timers.push(setInterval(() => send('GET_CHAT'), CHAT_MS));
    timers.push(setInterval(() => {
        if (!lastDash) return;
        const ch = Object.values(lastDash.l2Channels || lastDash.l2_channels || {});
        const chTxt = Array.isArray(ch) ? `${ch.length} channels` : '';
        log('status', `height=${lastHeight} balance=${lastDash.balance ?? lastDash.safeBalance ?? '?'} ${chTxt} id=${String(lastDash.l2Identity || lastDash.primaryAddress || '').slice(0, 12)}…`);
    }, STATUS_MS));
    send('GET_CHAT');
}

async function main() {
    log('hub', `node=${NODE_URL} data=${HUB_DIR} cmd=${CMD}`);

    if (SMOKE) {
        await loadWorker();
        send('GET_CHAT');
        const hist = await waitFor('CHAT_HISTORY', 30000);
        log('hub', `smoke OK — node reachable, ${(hist.messages || []).length} chat messages in ring`);
        process.exit(0);
    }

    if (!PASSWORD) { console.error('Set HUB_PASSWORD.'); process.exit(1); }
    await loadWorker();

    if (CMD === 'create' || CMD === 'import') {
        if (fs.existsSync(WALLET_F)) { console.error(`${WALLET_F} already exists — delete it first if you really mean to replace it.`); process.exit(1); }
        let phrase = process.argv[3];
        if (CMD === 'create') {
            send('GENERATE');
            phrase = await waitFor('PHRASE_GENERATED', 30000);
            console.log('\n================ SEED PHRASE — WRITE THIS DOWN ================\n');
            console.log(phrase);
            console.log('\n================================================================\n');
        }
        if (!phrase || phrase.trim().split(/\s+/).length < 12) { console.error('import requires the seed phrase as the next argument (quoted).'); process.exit(1); }
        send('CREATE', { phrase: phrase.trim(), password: PASSWORD });
        await waitFor('WALLET_LOADED', 30 * 60000);           // MSS derivation takes a while
        await saving;
        log('hub', `wallet ready at ${WALLET_F} — scanning chain…`);
        send(CMD === 'import' ? 'RESCAN' : 'SCAN');
    } else {
        if (!fs.existsSync(WALLET_F)) { console.error(`No wallet at ${WALLET_F} — run "node hub.mjs create" or "... import" first.`); process.exit(1); }
        send('LOGIN', { password: PASSWORD, rpcUrl: NODE_URL, bundleStr: fs.readFileSync(WALLET_F, 'utf8') });
        await waitFor('WALLET_LOADED', 10 * 60000);
        log('hub', 'wallet unlocked — scanning chain…');
        send('SCAN');
    }
    startLoops();
    log('hub', 'hub is live. Q-Bolt forwarding, JIT opens, invoice replies and the channel watcher are all running.');
}

// A hub must not die from one bad message; systemd handles true crashes.
process.on('unhandledRejection', (e) => log('fatal?', `unhandled rejection: ${e && e.message || e}`));
process.on('uncaughtException', (e) => { log('fatal', `${e && e.stack || e}`); process.exit(1); });
for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, async () => {
    log('hub', `${sig} — flushing wallet state…`);
    timers.forEach(clearInterval);
    try { await Promise.race([saving, new Promise((r) => setTimeout(r, 3000))]); } catch (_) {}
    process.exit(0);
});

main().catch((e) => { log('fatal', e && e.stack || String(e)); process.exit(1); });
