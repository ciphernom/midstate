// app.js: the Midstate web wallet's page. On top it is as simple as a wallet
// gets: a balance, Send, Receive, what happened, and mining. Amounts are in
// gMDS, as in the desktop app's simple view. A name.mds can be paid: the
// node's proof is checked here, against the registry, before anything is
// signed (names.js, in the worker).
//
// The engine is unchanged: worker.js holds the keys and builds, signs and
// sends transactions in WebAssembly; net.js carries its network requests;
// mining.js runs the miners. Open trades, Q-Bolt channels and mining bonds
// from before live on in the classic page (classic.html), which this page
// links to while this browser holds any.
//
// Everything from the chain or the user is drawn as text, never as HTML.

import {
    h, fill, errOf, sleep, icon, spinner, group, linkRow, actionRow, infoRow, controlRow, textRow, seg, toggle, bar,
    field, dialog, sheet, confirmBox, okBox, toast, copyText, download,
} from './ui.js';
import {
    UNIT, gmds, withUnit, gmdsExact, exactUnit, plainAmount, parseAmount, shortAddr, groupHex, dayLabel, clock, cleanAddress, cleanAddressList, sendStep,
    progressWords, speed, dateTime, cleanPhrase, priceLabel, about, timerText, lately, initial, PRICES,
} from './format.js';
import { isName, normalize as normalizeName } from './names.js';
import { DICTIONARY, MAX_WORDS, parseDraft, suggest, senderLabel, textOf, view as roomView } from './room.js';
import * as net from './net.js';
import { Mining, EFFORTS, MAX_THREADS, DEFAULT_POOL_URL } from './mining.js';
import { qrSvg } from './qr.js';
import { startBridge } from './bridge.js';

const WALLET_KEY = 'midstate_wallet';        // the encrypted wallet, as the worker saves it
const SWAPS_KEY = 'midstate_dex_swaps';      // the classic page's trades (their salts live only here)
const BOOK_KEY = 'midstate_dex_orderbook';
const THEME_KEY = 'mds-theme';
const NEW_HOME = 'https://midstate.cash/wallet/';
const SOURCE = 'https://github.com/ciphernom/midstate';

const store = {
    get(k) { try { return localStorage.getItem(k); } catch (_) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); return true; } catch (_) { return false; } },
};
const hasSavedWallet = () => store.get(WALLET_KEY) !== null;

const app = {
    phase: 'loading',  // loading | elsewhere | welcome | locked | opening | wallet | broken
    fatal: '',
    dash: null,        // the worker's dashboard: balance, history, heights…
    scan: null,        // { height, max } while the wallet reads the chain
    scanStart: null,
    stalled: false,
    op: null,          // the send or tidy-up under way, for its progress
    draft: { to: '', amount: '', all: false },
    review: null,      // the send being reviewed
    sendError: '',
    flow: null,        // a new wallet: { words, phrase, quiz, checked, mine }
    fresh: false,      // the open wallet was made in this session
    opening: null,     // { what, back } while a wallet is made, restored or imported
    openError: '',
    making: null,      // { current, total, label } while keys are derived
    unlocking: false,
    unlockError: '',
    newAddress: false,
    revealing: null,
    bonds: 0,          // mining bonds still holding coins (the classic page's)
    bondsAsked: false,
    reuse: null,       // coins held back after a reorganisation
    update: false,     // a new version of the page is ready
    log: [],
    after: null,       // where to go once the wallet is open
    mineOnOpen: false,
    dapp: new URLSearchParams(location.search).get('dapp') === '1',
    messages: null,    // the worker's sealed messages: conversations, requests, settings
    msgDraft: { to: '', text: '' },
    room: null,        // the public room: what the bus says, and this wallet's name in it
    roomDraft: '',
    names: null,       // the names this wallet holds, and what is being done to them
};

// ── the worker ──────────────────────────────────────────────────────────────

let worker = null;
let mining = null;
const listeners = new Set(); // the dApp bridge watches the worker too
const waiting = new Map();
let seq = 0;

const post = (msg) => { if (worker) worker.postMessage(msg); };

/** A request the worker answers with the same id. */
function ask(type, payload) {
    return new Promise((resolve, reject) => {
        const id = ++seq;
        waiting.set(id, { resolve, reject });
        post({ type, payload: { ...payload, id } });
    });
}
function answer(id, value, error) {
    const w = waiting.get(id);
    if (!w) return;
    waiting.delete(id);
    if (error != null && error !== '') w.reject(new Error(error));
    else w.resolve(value);
}

function addLog(text) {
    const t = new Date();
    const stamp = `${String(t.getHours()).padStart(2, '0')}:${String(t.getMinutes()).padStart(2, '0')}:${String(t.getSeconds()).padStart(2, '0')}`;
    app.log.push(`${stamp}  ${String(text)}`);
    if (app.log.length > 400) app.log.splice(0, app.log.length - 400);
    if (current && current.sc.onLog) current.sc.onLog();
}

/** One tab at a time: two pages signing from one wallet could reuse a
 *  one-time key. The classic page takes the same lock. */
function takeLock() {
    if (!navigator.locks || !navigator.locks.request) return Promise.resolve(true);
    return new Promise((resolve) => {
        let settled = false;
        const ctl = new AbortController();
        // A page that just closed (say, the classic one) may still hold it for
        // a moment; wait a little before saying it is open elsewhere.
        const timer = setTimeout(() => {
            if (settled) return;
            settled = true;
            ctl.abort();
            resolve(false);
        }, 2000);
        navigator.locks.request('midstate-wallet', { signal: ctl.signal }, () => {
            if (settled) return undefined;
            settled = true;
            clearTimeout(timer);
            resolve(true);
            return new Promise(() => {}); // held while this page is open
        }).catch(() => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            resolve(true); // no usable lock: go on without one, as before
        });
    });
}

function fatal(text) {
    app.phase = 'broken';
    app.fatal = text;
    render();
}

function startWorker() {
    try {
        worker = new Worker('./worker.js', { type: 'module' });
    } catch (e) {
        return fatal(`This browser can’t run the wallet (${errOf(e)}). Try a current Chrome, Firefox or Safari.`);
    }
    worker.onmessage = (e) => onWorker(e.data);
    worker.onerror = (e) => {
        if (e && e.preventDefault) e.preventDefault();
        fatal(`The wallet’s engine didn’t load${e && e.message ? ` (${e.message})` : ''}. Reload the page; your wallet is safe in this browser.`);
    };
    net.attach({ post, log: addLog, change: () => update() });
    mining = new Mining({ wallet: worker, isWebRTC: net.isWebRTC, log: addLog });
    mining.on((kind, data) => {
        if (kind === 'block') {
            toast(data && data.isShare ? 'Your pool found a block with your work in it.' : `You mined block ${Number(data && data.height || 0).toLocaleString()}.`, 6000);
        } else if (kind === 'error') {
            okBox('Mining stopped', String(data));
        }
        update();
    });
    post({ type: 'INIT' });
}

function onWorker(msg) {
    const { type, payload } = msg || {};
    for (const fn of listeners) { try { fn(msg); } catch (_) {} }
    if (mining && mining.handle(type, payload)) { update(); return; }
    switch (type) {
        case 'INIT_DONE':
            app.phase = hasSavedWallet() ? 'locked' : 'welcome';
            render();
            return;
        case 'RPC_REQUEST':
            net.handleRpcRequest(payload.id, payload.method, payload.params);
            return;
        case 'POOL_REQUEST':
            net.handlePoolRequest(payload.id, payload.method, payload.params);
            return;
        case 'AUTO_CONNECT_WEBRTC':
            net.connect().catch(() => {});
            return;
        case 'SAVE_WALLET':
            if (!store.set(WALLET_KEY, payload)) {
                toast('This browser refused to save the wallet. Free some space, or save a backup file now.', 9000);
            }
            return;
        case 'LOG':
            addLog(payload);
            return;
        case 'PHRASE_GENERATED':
            gotPhrase(String(payload || ''));
            return;
        case 'MSS_PROGRESS':
            app.making = payload;
            update();
            return;
        case 'WALLET_LOADED':
            walletOpened(payload);
            return;
        case 'REFRESH_DASHBOARD':
            setDash(payload);
            return;
        case 'SCAN_PROGRESS':
            if (!app.scan) app.scanStart = payload.height;
            app.scan = payload;
            app.stalled = false;
            update();
            return;
        case 'SCAN_COMPLETE':
            app.scan = null;
            app.stalled = false;
            setDash(payload, true);
            if (!app.bondsAsked) {
                app.bondsAsked = true;
                post({ type: 'BONDS_LIST' });
            }
            update();
            return;
        case 'SCAN_STALLED':
            app.scan = null;
            app.stalled = true;
            update();
            return;
        case 'SEND_PROGRESS': {
            const o = app.op;
            if (payload && payload.msg) addLog(payload.msg);
            if (o && (o.kind === 'tidy' || o.kind === 'merge') && !o.done && !o.error) {
                const [words, frac] = progressWords(payload && payload.msg);
                o.words = words;
                o.frac = Math.max(o.frac || 0, frac);
                update();
            }
            return;
        }
        case 'SEND_STEP': {
            const o = app.op;
            addLog(`${payload.title || payload.step}${payload.detail ? ` (${payload.detail})` : ''}`);
            if (o && o.kind === 'send' && !o.error && !o.done) {
                const [words, frac] = sendStep(payload.step);
                o.words = words;
                o.frac = Math.max(o.frac || 0, frac);
                if (payload.step === 'reveal_mined') o.done = true;
                update();
            }
            return;
        }
        case 'SEND_COMPLETE': {
            const o = app.op;
            setDash(payload, true);
            if (o && !o.error) {
                const first = !o.completed;
                o.done = true;
                o.completed = true;
                o.frac = 1;
                o.words = o.kind === 'send' ? 'Sent' : 'Done';
                if (first) toast(o.kind === 'tidy' ? 'Coins gathered.' : o.kind === 'merge' ? 'Coins merged.' : `Sent ${withUnit(o.amount)}.`);
            }
            update();
            return;
        }
        case 'SEND_ERROR': {
            const text = sendProblem(payload && payload.error);
            addLog(`Send failed: ${payload && payload.error}`);
            const o = app.op;
            if (o && !o.done && !o.error) o.error = text;
            else toast(text, 8000);
            update();
            return;
        }
        case 'NAME_RESOLVED':
            answer(payload.id, payload);
            return;
        case 'NAME_ERROR':
            answer(payload.id, null, payload.error || 'The name couldn’t be checked.');
            return;
        case 'ADDRESS_CHECKED':
            answer(payload.id, payload);
            return;
        case 'SEND_QUOTE':
        case 'CONSOLIDATE_QUOTE':
            answer(payload.id, payload, payload.error);
            return;
        case 'SEED_REVEALED':
            if (app.revealing) { const r = app.revealing; app.revealing = null; r.resolve(String(payload)); }
            return;
        case 'SEED_REFUSED':
            if (app.revealing) { const r = app.revealing; app.revealing = null; r.reject(new Error('That password isn’t right.')); }
            return;
        case 'BACKUP_READY':
            saveBackup(payload);
            return;
        case 'BONDS':
            app.bonds = ((payload && payload.bonds) || []).filter(b => ['locking', 'locked', 'unlocked', 'unbonding'].includes(b.status)).length;
            update();
            return;
        case 'REUSE_LOCK_ALERT':
            app.reuse = payload;
            update();
            return;
        case 'MSG_BOOK':
            app.messages = payload;
            update();
            return;
        case 'MSG_ANSWER':
            answer(payload.id, payload.result, payload.error);
            return;
        case 'MSG_NOTICE': {
            // Not while that conversation is on screen: it shows there.
            const n = payload || {};
            if (n.text && !(n.id && routePath() === 'messages/c/' + n.id)) toast(n.text, 6000);
            return;
        }
        case 'ROOM':
            app.room = payload;
            update();
            return;
        case 'NAMES':
            app.names = payload;
            update();
            return;
        case 'NAMES_ANSWER':
            answer(payload.id, payload.result, payload.error);
            return;
        case 'NAMES_NOTICE': {
            // Not while that name is on screen: it shows there.
            const n = payload || {};
            if (n.text && !(n.name && routePath() === 'names/n/' + n.name)) toast(n.text, 7000);
            return;
        }
        case 'L2_EVENT':
            // Without a channel: a send interrupted last time, finished now.
            if (payload && payload.msg) {
                addLog(payload.msg);
                if (!payload.channelId) toast(payload.msg, 7000);
            }
            return;
        case 'ERROR':
            workerError(String(payload));
            return;
        default:
    }
}

let lastToast = { text: '', at: 0 };
function workerError(text) {
    addLog(`ERROR: ${text}`);
    if (app.revealing) {
        const r = app.revealing;
        app.revealing = null;
        r.reject(new Error(/not found/i.test(text)
            ? 'This wallet came from a command-line wallet file, so it has no 24 words here. Keep a backup file instead.'
            : text));
        return;
    }
    if (app.phase === 'locked' && app.unlocking) {
        app.unlocking = false;
        app.making = null;
        app.unlockError = /incorrect password|corrupted/i.test(text) ? 'That password isn’t right.' : text;
        update();
        if (current && current.sc.failed) current.sc.failed();
        return;
    }
    if (app.phase === 'opening') return openFailed(text);
    // The network coming and going is the status pill's to say; a toast for
    // every failed request would only repeat it.
    if (/GATEWAY_UNAVAILABLE|RATE_LIMITED|failed to fetch|networkerror|timed out/i.test(text)) return;
    if (app.phase === 'wallet') {
        const now = Date.now();
        if (text === lastToast.text && now - lastToast.at < 30_000) return;
        lastToast = { text, at: now };
        toast(text.length > 220 ? `${text.slice(0, 220)}…` : text, 7000);
    }
}

function setDash(d, quiet) {
    if (!d) return;
    const before = app.dash && app.dash.primaryAddress;
    app.dash = d;
    if (app.newAddress && before && d.primaryAddress !== before) {
        app.newAddress = false;
        app.making = null;
        toast('Your new address is ready. The old one keeps working.');
    }
    if (!quiet) update();
}

function walletOpened(payload) {
    // A wallet made just now has no past to find: its first read of the chain
    // is catching up, not restoring, whatever the worker calls it.
    app.fresh = !!(app.opening && app.opening.what === 'create');
    app.phase = 'wallet';
    app.unlocking = false;
    app.unlockError = '';
    app.opening = null;
    app.openError = '';
    app.making = null;
    app.flow = null; // the words leave this page's memory
    restoreDraft = '';
    setDash(payload, true);
    post({ type: 'SCAN' });
    const after = app.after != null ? app.after : app.mineOnOpen ? 'mine' : '';
    app.after = null;
    if (app.mineOnOpen) {
        app.mineOnOpen = false;
        try { mining.start(); } catch (e) { toast(errOf(e), 8000); }
    }
    history.replaceState(null, '', '#/' + after);
    render(true);
}

/** The worker's send errors, for people. */
function sendProblem(raw) {
    const t = String(raw || '').replace(/^Error:\s*/, '').replace(/^Failed to prepare transaction:\s*/i, '');
    if (/insufficient funds/i.test(t)) return 'That’s more than you have once the network fee is added. Send a little less, or use Max.';
    if (/invalid address|checksum/i.test(t)) return 'That address has a typo.';
    if (/still scanning/i.test(t)) return 'The wallet is still looking through the chain for this phrase’s earlier coins. Try again once that’s done.';
    if (/too small to cover/i.test(t)) return 'Your balance is too small to cover the network fee.';
    if (/nothing to send/i.test(t)) return 'There’s nothing to send.';
    if (/not loaded|run a network sync/i.test(t)) return 'The wallet is still loading its keys. Try again in a moment.';
    if (/GATEWAY_UNAVAILABLE|RATE_LIMITED|failed to fetch|network ?error/i.test(t) && !/may still/i.test(t)) {
        return 'Couldn’t reach the network. Try again when you’re back online.';
    }
    return t || 'The payment didn’t go through.';
}

/** A name that couldn't be paid, in plain words. */
function nameProblem(raw) {
    const t = String(raw || '');
    if (/no one has the name/i.test(t)) return t.replace(/^.*?(No one has)/i, '$1') + '.';
    if (/takes no MDS/i.test(t)) return t + '.';
    if (/lapse/i.test(t)) return t + '.';
    if (/GATEWAY|fetch|network|timed? ?out|unavailable/i.test(t)) return 'Couldn’t reach the network to check that name. Try again in a moment.';
    return `That name couldn’t be checked: ${t}`;
}

// ── a wallet coming into being ──────────────────────────────────────────────

function pickQuiz(n) {
    const picks = new Set();
    const r = new Uint32Array(1);
    while (picks.size < Math.min(3, n)) {
        crypto.getRandomValues(r);
        picks.add(r[0] % n);
    }
    return [...picks].sort((a, b) => a - b);
}

function gotPhrase(phrase) {
    const words = phrase.trim().split(/\s+/);
    app.flow = { mine: !!(app.flow && app.flow.mine), phrase, words, quiz: pickQuiz(words.length), checked: false, error: '' };
    if (routePath().startsWith('create')) render();
}

function startOpening(what, msg) {
    app.opening = { what, back: routePath() };
    app.openError = '';
    app.making = null;
    if (what === 'create') app.mineOnOpen = !!(app.flow && app.flow.mine);
    app.phase = 'opening';
    post(msg);
    render();
}

function openFailed(text) {
    const back = app.opening ? app.opening.back : '';
    app.opening = null;
    app.making = null;
    app.mineOnOpen = false;
    app.phase = hasSavedWallet() ? 'locked' : 'welcome';
    app.openError = friendlyOpenError(text);
    history.replaceState(null, '', '#/' + back);
    render();
}

function friendlyOpenError(text) {
    const t = String(text || '');
    const m = /unknown word \(word (\d+)\)/i.exec(t);
    if (m) return `Word ${Number(m[1]) + 1} isn’t one of the recovery-phrase words. Check its spelling.`;
    if (/invalid checksum/i.test(t)) return 'Those words don’t add up: one is probably wrong, or two are swapped. Check them against your paper.';
    if (/word count/i.test(t)) return 'A recovery phrase has 24 words.';
    if (/wrong password/i.test(t)) return 'That password doesn’t open this wallet file.';
    return t;
}

/** Why the last attempt to open a wallet failed, said once. */
function takeOpenError() {
    const e = app.openError;
    app.openError = '';
    return e || '';
}

const passwordProblem = (a, b) => (a.length < 8 ? 'Use at least 8 characters.' : a !== b ? 'The passwords don’t match.' : '');

/** A backup file's text: the wallet goes into this browser, and any open
 *  trades it carries join the ones already here. */
function openBackupText(text) {
    let parsed;
    try { parsed = JSON.parse(text); } catch (_) { throw new Error('That file isn’t a Midstate backup.'); }
    let bundle = parsed;
    if (parsed && parsed._midstate_backup && parsed.wallet) {
        mergeTrades(parsed.dex || {});
        tradesSeen.at = 0;
        bundle = parsed.wallet;
    }
    if (!bundle || typeof bundle !== 'object' || !bundle.salt || !bundle.iv || !bundle.data) {
        throw new Error('That file isn’t a Midstate backup.');
    }
    if (!store.set(WALLET_KEY, JSON.stringify(bundle))) throw new Error('This browser refused to save the wallet (its storage is full or blocked).');
    app.phase = 'locked';
}

/** Trades from a backup, added to this browser's without replacing any: a
 *  trade's record holds the only copy of the salt that can reclaim it. */
function mergeTrades(dex) {
    const merge = (key, incoming, same) => {
        if (!Array.isArray(incoming) || !incoming.length) return;
        let have = [];
        try { have = JSON.parse(store.get(key) || '[]'); } catch (_) {}
        if (!Array.isArray(have)) have = [];
        for (const r of incoming) if (r && !have.some(s => s && same(s, r))) have.push(r);
        store.set(key, JSON.stringify(have));
    };
    const json = x => JSON.stringify(x);
    merge(SWAPS_KEY, dex.swaps, (a, b) => (a.offerId != null && a.offerId === b.offerId && a.role === b.role)
        || (a.covenant && b.covenant && a.covenant.coinId && a.covenant.coinId === b.covenant.coinId)
        || json(a) === json(b));
    merge(BOOK_KEY, dex.orderbook, (a, b) => (a.offerId != null && a.offerId === b.offerId) || json(a) === json(b));
}

function saveBackup(bundleText) {
    let text = bundleText;
    try {
        text = JSON.stringify({
            _midstate_backup: 2,
            wallet: JSON.parse(bundleText),
            dex: { swaps: JSON.parse(store.get(SWAPS_KEY) || '[]'), orderbook: JSON.parse(store.get(BOOK_KEY) || '[]') },
        });
    } catch (_) { /* the bare wallet, then: it still opens */ }
    download(`midstate-backup-${new Date().toISOString().slice(0, 10)}.json`, text);
    toast('Backup saved. Keep a copy somewhere other than this device.', 6000);
}

function revealWords(password) {
    return new Promise((resolve, reject) => {
        app.revealing = { resolve, reject };
        post({ type: 'REVEAL_SEED', payload: { password } });
    });
}

// ── what the page shows about the wallet ────────────────────────────────────

const KIND = { coinbase: 'Mined', received: 'Received', sent: 'Sent', pending: 'Sending', mixed: 'Payment' };
const kindLabel = k => KIND[k] || (k ? k[0].toUpperCase() + k.slice(1) : 'Payment');
const isOut = e => e.kind === 'sent' || e.kind === 'pending';
const entryKey = e => [e.kind, e.timestamp, (e.outputs && e.outputs[0]) || (e.inputs && e.inputs[0]) || ''].join('-');
const whenText = ts => `${dayLabel(ts)} ${clock(ts)}`;
const hostOf = (url) => { try { return new URL(url).host; } catch (_) { return String(url || ''); } };
const balanceOf = () => (app.dash ? Number(app.dash.balance) || 0 : 0);
const pendingOut = () => (app.dash ? app.dash.history.filter(e => e.kind === 'pending').reduce((s, e) => s + e.value + e.fee, 0) : 0);
const busyOp = () => !!(app.op && !app.op.done && !app.op.error);

/** Whether the wallet is still reading the chain, in words. */
function syncState() {
    const d = app.dash;
    if (!d) return { text: '' };
    const s = app.scan;
    const frac = s && s.max > (app.scanStart || 0)
        ? Math.max(0, Math.min(1, (s.height - (app.scanStart || 0)) / (s.max - (app.scanStart || 0)))) : null;
    const pct = frac != null ? ` ${Math.floor(frac * 100)}%` : '';
    if (d.restoring) {
        return app.fresh
            ? { text: `Syncing${pct}`, frac, long: `Getting up to date with the network${pct ? ':' + pct : '…'}` }
            : { text: `Restoring${pct}`, frac, long: 'Looking through the chain for this phrase’s money. Sending waits until it’s done.' };
    }
    const behind = (d.networkHeight || 0) - (d.lastScannedHeight || 0);
    if (behind > 2 && (s || !d.lastScannedHeight)) {
        return { text: `Syncing${pct}`, frac, long: `Catching up with the network${pct ? ':' + pct : '…'}` };
    }
    return { text: '' };
}

/** Why sending has to wait, or ''. */
function sendBlocked() {
    const d = app.dash;
    if (!d) return 'Loading…';
    if (d.restoring) return app.fresh ? 'Catching up with the network…' : 'Waiting for the restore to finish';
    if ((d.networkHeight || 0) - (d.lastScannedHeight || 0) > 2) return 'Catching up with the network…';
    if (busyOp()) return 'Wait for the last payment to finish';
    return '';
}

/** Addresses with coins to merge: the wallet's reusable (MSS) addresses
 *  holding more coins than they need to, most first. Mining pays out in many small
 *  coins; a consolidate transaction merges an address's coins with a single
 *  signature, so it costs little per coin. (Coins on a one-time address are
 *  always spent together anyway, so they're not listed.) */
const DUST_HINT = 50;
function mergeable() {
    const d = app.dash;
    if (!d) return [];
    const main = d.primaryAddress && d.primaryAddress !== 'None' ? d.primaryAddress.slice(0, 64) : '';
    const by = new Map();
    for (const u of d.utxos) {
        if (!u.is_mss || u.watch_only || u.reuseLocked) continue;
        const a = by.get(u.address) || { address: u.address, coins: 0, value: 0, main: u.address === main };
        a.coins += 1;
        a.value += Number(u.value) || 0;
        by.set(u.address, a);
    }
    // Coins come in powers of two, so an address already holds as few coins
    // as its value allows when it has one per bit; merging that again would
    // only cost a fee. List an address when merging saves two coins or more.
    const bits = (v) => { let n = 0; for (let x = BigInt(v); x > 0n; x >>= 1n) n += Number(x & 1n); return n; };
    return [...by.values()].filter(a => a.coins - bits(a.value) >= 2).sort((x, y) => y.coins - x.coins);
}
const mergeableCoins = () => mergeable().reduce((n, a) => n + a.coins, 0);

/** Signatures the receiving address has left. Each payment out of it uses
 *  one, and money on an address with none left can't be moved, so the page
 *  asks for a new address well before. */
const LOW_SIGNATURES = 64;
function signaturesLeft() {
    const b = app.dash && app.dash.l2LeafBudget;
    return b && b.total ? b.remaining : null;
}

const FINISHED_TRADES = new Set(['completed', 'swap_complete', 'limit_refunded', 'limit_cancelled', 'submarine_swept', 'submarine_refunded']);
let tradesSeen = { at: 0, n: 0 };
function openTrades() {
    if (Date.now() - tradesSeen.at < 5000) return tradesSeen.n;
    let swaps = [];
    try { swaps = JSON.parse(store.get(SWAPS_KEY) || '[]'); } catch (_) {}
    const n = Array.isArray(swaps) ? swaps.filter(s => s && !FINISHED_TRADES.has(s.status)).length : 0;
    tradesSeen = { at: Date.now(), n };
    return n;
}
/** Trades, channels and bonds this browser holds: the classic page runs them. */
function positions() {
    const d = app.dash;
    const channels = d && Array.isArray(d.l2Channels)
        ? d.l2Channels.filter(c => c.status !== 'closed' && (!c.legacy || (c.isAlice ? c.aliceAmt : c.bobAmt) > 0)).length : 0;
    const p = { trades: openTrades(), channels, bonds: app.bonds };
    p.total = p.trades + p.channels + p.bonds;
    return p;
}
function positionsText(p) {
    const parts = [];
    if (p.trades) parts.push(`${p.trades} trade${p.trades === 1 ? '' : 's'}`);
    if (p.channels) parts.push(`${p.channels} channel${p.channels === 1 ? '' : 's'}`);
    if (p.bonds) parts.push(`${p.bonds} bond${p.bonds === 1 ? '' : 's'}`);
    return parts.join(', ');
}

// ── the theme ───────────────────────────────────────────────────────────────

const currentTheme = () => (document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark');
function setTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    store.set(THEME_KEY, t);
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', t === 'light' ? '#ffffff' : '#0a0a0a');
    topKey = null;
    update();
}

async function lock() {
    if (busyOp()) {
        if (!await confirmBox('Lock during a payment?', 'Locking reloads the page. The payment finishes the next time you unlock.', 'Lock')) return;
    } else if (mining && mining.state.active) {
        if (!await confirmBox('Lock and stop mining?', 'Locking reloads the page, which stops mining.', 'Lock')) return;
    }
    history.replaceState(null, '', location.pathname + location.search);
    location.reload();
}

// ── the frame: a bar along the top, sections down the side, the screen ─────

const $app = document.getElementById('app');
const $top = h('header', { class: 'top' });
const $side = h('nav', { class: 'side', 'aria-label': 'Sections' });
const $content = h('main', { class: 'content', id: 'main' });
$app.append($top, h('div', { class: 'body' }, $side, $content));

const SECTIONS = [
    ['', 'Home', 'home'], ['activity', 'Activity', 'clock'], ['send', 'Send', 'up'],
    ['receive', 'Receive', 'down'], ['messages', 'Messages', 'chat'], ['names', 'Names', 'tag'], ['mine', 'Mine', 'pick'], ['settings', 'Settings', 'gear'],
];

function routePath() {
    const raw = (location.hash.match(/^#\/?(.*)$/) || [])[1] || '';
    let path = raw;
    try { path = decodeURI(raw); } catch (_) {}
    return path.replace(/\/+$/, '');
}

/** Goes to a screen; `replace` keeps it out of the back button's way. */
function go(path, replace) {
    const hash = '#/' + path;
    if (replace || location.hash === hash) {
        history.replaceState(null, '', hash);
        render(true);
    } else {
        location.hash = hash;
    }
}

let current = null; // { path, sc, el }
let topKey = null;
let sideKey = null;

function drawSide(path) {
    const top = path.split('/')[0];
    const unread = book().unread || 0;
    const k = `${top}:${unread}:${nbook().available !== false}`;
    if (sideKey === k) return;
    sideKey = k;
    fill($side, SECTIONS.filter(([p]) => p !== 'names' || nbook().available !== false).map(([p, label, ic]) =>
        h('a', { href: '#/' + p, class: top === p ? 'on' : null, 'aria-current': top === p ? 'page' : null }, icon(ic), label,
            p === 'messages' && unread ? h('span', { class: 'count', 'aria-label': `${unread} unread` }, String(unread)) : null)));
}

function drawTop() {
    if (app.phase !== 'wallet') return;
    const sum = net.summary();
    const sync = syncState();
    const mine = !!(mining && mining.state.active);
    const theme = currentTheme();
    const unread = book().unread || 0;
    const key = JSON.stringify([sum, sync.text, mine, theme, unread]);
    if (key === topKey) return;
    topKey = key;
    fill($top,
        h('a', { class: 'brand', href: '#/', 'aria-label': 'Midstate wallet: home' }, h('b', null, 'MIDSTATE', h('span', null, ' WALLET'))),
        h('div', { class: 'top-actions' },
            mine ? h('a', { class: 'pill', href: '#/mine', title: 'Mining' }, icon('pick'), h('span', { class: 'wide-only' }, 'Mining')) : null,
            h('a', { class: 'pill', href: '#/settings/network', title: 'Network' },
                h('span', { class: 'dot ' + (sync.text ? 'busy' : sum.cls) }), sync.text || sum.text),
            h('button', {
                class: 'icon-btn', type: 'button', title: theme === 'dark' ? 'Light theme' : 'Dark theme',
                'aria-label': theme === 'dark' ? 'Use the light theme' : 'Use the dark theme',
                onclick: () => setTheme(theme === 'dark' ? 'light' : 'dark'),
            }, icon(theme === 'dark' ? 'sun' : 'moon')),
            h('button', { class: 'icon-btn', type: 'button', title: 'Lock', 'aria-label': 'Lock', onclick: lock }, icon('lock')),
            unread ? h('a', { class: 'icon-btn phone-only has-dot', href: '#/messages', title: 'Messages', 'aria-label': `Messages: ${unread} unread` }, icon('chat')) : null,
            h('a', { class: 'icon-btn phone-only', href: '#/settings', title: 'Settings', 'aria-label': 'Settings' }, icon('gear'))));
}

function backLink([label, href], always) {
    return h('a', { class: 'back' + (always ? '' : ' phone-only'), href }, icon('left'), label);
}

function render(animate) {
    const path = routePath();
    let sc;
    try {
        sc = screenFor(path);
    } catch (e) {
        console.error(e);
        sc = { title: 'Something went wrong', els: [h('p', { class: 'muted' }, errOf(e)), h('a', { class: 'btn', href: '#/' }, 'Home')] };
    }
    if (current && current.sc.leave) { try { current.sc.leave(); } catch (_) {} }
    const bare = !!sc.bare;
    $app.classList.toggle('bare', bare);
    $top.hidden = bare;
    $side.hidden = bare || app.phase !== 'wallet';
    if (!$side.hidden) drawSide(path); else sideKey = null;
    if (bare) topKey = null;
    const el = bare
        ? h('div', { class: 'center' }, h('div', { class: 'center-card' + (sc.form ? ' form-card' : '') },
            sc.back ? h('div', { class: 'page-head' }, backLink(sc.back, true)) : null,
            sc.title ? h('h1', null, sc.title) : null, sc.els))
        : h('div', { class: 'screen' + (sc.wide ? ' wide' : '') },
            sc.back ? h('div', { class: 'page-head' }, backLink(sc.back, sc.backAlways)) : null,
            sc.title ? h('h1', null, sc.title) : null, sc.els);
    fill($content, el);
    current = { path, sc, el };
    update();
    if (animate) window.scrollTo(0, 0);
    if (sc.focus) setTimeout(() => { try { sc.focus(); } catch (_) {} }, 0);
}

function update() {
    drawTop();
    if (!$side.hidden && current) drawSide(current.path);
    if (current && current.sc.update) {
        try { current.sc.update(); } catch (e) { console.error(e); }
    }
}

window.addEventListener('hashchange', () => render(true));
window.addEventListener('online', () => update());
window.addEventListener('offline', () => update());

/** A screen that sends somewhere else. */
function redirect(path) {
    setTimeout(() => go(path, true), 0);
    return { bare: app.phase !== 'wallet', els: [] };
}

function screenFor(path) {
    switch (app.phase) {
        case 'loading': return loadingScreen();
        case 'elsewhere': return elsewhereScreen();
        case 'broken': return brokenScreen();
        case 'opening': return openingScreen();
        default:
    }
    if (app.phase !== 'wallet') {
        if (path === 'have') return haveScreen();
        if (path === 'restore') return restoreScreen();
        if (path === 'open') return openScreen();
        if (path === 'network') return networkScreen(false);
        if (app.phase === 'welcome') {
            if (path === 'mine') return mineStartScreen();
            if (path === 'create' || path.startsWith('create/')) return createScreen(path);
            return welcomeScreen();
        }
        // Locked: remember where the link pointed, for after unlocking.
        if (path && path !== 'create' && !path.startsWith('create/')) app.after = path;
        return lockedScreen();
    }
    let m;
    if (path === 'activity') return activityScreen();
    if ((m = /^activity\/(.+)$/.exec(path))) return activityItemScreen(m[1]);
    if (path === 'send') return sendScreen();
    if (path === 'send/review') return reviewScreen();
    if (path === 'send/progress') return progressScreen();
    if (path === 'receive') return receiveScreen();
    if (path === 'messages') return messagesScreen('inbox');
    if (path === 'messages/requests') return messagesScreen('requests');
    if (path === 'messages/new') return composeScreen();
    if (path === 'messages/settings') return messageSettingsScreen();
    if (path === 'messages/public') return publicRoomScreen();
    if ((m = /^messages\/c\/(.+)$/.exec(path))) return conversationScreen(m[1]);
    if (path === 'names') return namesScreen();
    if (path === 'names/about') return nameAboutScreen();
    if ((m = /^names\/get\/(.+)$/.exec(path))) return nameGetScreen(m[1]);
    if ((m = /^names\/n\/(.+)$/.exec(path))) return nameScreen(m[1]);
    if (path === 'mine') return mineScreen();
    if (path === 'mine/pool') return poolScreen();
    if (path === 'settings') return settingsScreen();
    if (path === 'settings/backup') return backupScreen();
    if (path === 'settings/network' || path === 'network') return networkScreen(true);
    if (path === 'settings/care') return careScreen();
    if (path === 'settings/advanced') return advancedScreen();
    if (path === 'settings/about') return aboutScreen();
    return homeScreen();
}

// ── before a wallet is open ─────────────────────────────────────────────────

function loadingScreen() {
    return { bare: true, els: [h('p', { class: 'muted' }, spinner(), ' Starting…')] };
}

function elsewhereScreen() {
    const btn = h('button', {
        class: 'btn primary wide', type: 'button',
        onclick: async () => {
            btn.disabled = true;
            btn.textContent = 'Checking…';
            if (await takeLock()) {
                app.phase = 'loading';
                render();
                startWorker();
            } else {
                btn.disabled = false;
                btn.textContent = 'Try again';
                toast('It’s still open somewhere else.');
            }
        },
    }, 'Try again');
    return {
        bare: true,
        els: [h('h1', null, 'Open in another tab'),
            h('p', { class: 'lead' }, 'This wallet is already open in another tab or window. Use that one, or close it and try again here.'),
            btn],
    };
}

function brokenScreen() {
    return {
        bare: true,
        els: [h('h1', null, 'The wallet can’t start'), h('p', { class: 'lead' }, app.fatal),
            h('button', { class: 'btn primary wide', type: 'button', onclick: () => location.reload() }, 'Reload')],
    };
}

/** For a site this wallet has moved from. */
function migrationNote() {
    if (!/\.github\.io$/i.test(location.hostname)) return null;
    try { if (sessionStorage.getItem('hideMigrationBanner')) return null; } catch (_) {}
    const el = h('div', { class: 'banner' },
        h('span', { class: 'grow' }, h('b', null, 'This wallet has moved to midstate.cash. '),
            'A wallet doesn’t move between sites by itself: save a backup file here (Settings, then Backup) and open it at ',
            h('a', { href: NEW_HOME }, 'midstate.cash'), '.'),
        h('button', {
            class: 'btn small', type: 'button',
            onclick: () => { try { sessionStorage.setItem('hideMigrationBanner', '1'); } catch (_) {} el.remove(); },
        }, 'Dismiss'));
    return el;
}

function welcomeScreen() {
    return {
        bare: true,
        els: [
            h('h1', null, 'Welcome to Midstate'),
            h('p', { class: 'lead' }, 'A wallet that lives in this browser. No account, no sign-up.'),
            migrationNote(),
            h('div', { class: 'stack' },
                h('button', { class: 'btn primary wide', type: 'button', onclick: () => { app.flow = null; go('create'); } }, 'Create a new wallet'),
                h('a', { class: 'btn wide', href: '#/have' }, 'I already have a wallet')),
            h('a', { class: 'mine-cta', href: '#/mine' }, icon('pick'),
                h('span', null, h('b', null, 'Mine in your browser'), h('small', null, 'Make a wallet and earn MDS with this device’s spare power.'))),
            h('div', { class: 'links' },
                h('a', { class: 'quiet-link', href: '../' }, 'What is Midstate?'),
                h('a', { class: 'quiet-link', href: '#/network' }, 'Network')),
        ],
    };
}

const replaceWarning = () => h('div', { class: 'warn-box' }, h('b', null, 'This replaces the wallet saved in this browser.'),
    'Make sure you have its 24 words or a backup file before you go on.');

function haveScreen() {
    const replacing = hasSavedWallet();
    return {
        bare: true, form: true, back: ['Back', '#/'],
        title: replacing ? 'Use a different wallet' : 'Open your wallet',
        els: [
            replacing ? replaceWarning() : null,
            group(
                linkRow('Type your 24 words', '#/restore', { iconName: 'key', sub: 'The recovery phrase you wrote down' }),
                linkRow('Open a backup file', '#/open', { iconName: 'file', sub: 'A .json backup from this wallet, or a .dat from the command-line wallet' })),
        ],
    };
}

function mineStartScreen() {
    return {
        bare: true, back: ['Back', '#/'],
        els: [
            h('h1', null, 'Mine in your browser'),
            h('p', { class: 'lead' }, 'You need a wallet to collect what you mine. Making one takes a minute: write down 24 words, choose a password, and mining starts.'),
            h('button', { class: 'btn primary wide', type: 'button', onclick: () => { app.flow = { mine: true }; go('create'); } }, icon('pick'), 'Make a wallet and start mining'),
            h('div', { class: 'links' }, h('a', { class: 'quiet-link', href: '#/have' }, 'I already have a wallet')),
        ],
    };
}

function createScreen(path) {
    const f = app.flow;
    if (!f || !f.words) {
        if (!f || !f.asked) {
            app.flow = { mine: !!(f && f.mine), asked: true };
            post({ type: 'GENERATE' });
        }
        return { bare: true, els: [h('p', { class: 'muted' }, spinner(), ' Making your words…')] };
    }
    if (path === 'create/check') return checkWordsScreen();
    if (path === 'create/password') return f.checked ? newPasswordScreen() : redirect('create/check');
    return wordsScreen();
}

function wordsScreen() {
    const f = app.flow;
    return {
        bare: true, form: true, back: ['Cancel', '#/'],
        title: 'Write down these 24 words',
        els: [
            f.error ? h('p', { class: 'err' }, f.error) : null,
            h('div', { class: 'warn-box' }, h('b', null, 'This is the only time they are shown.'),
                'They are the only way back into this wallet if this browser loses it. Anyone who has them can take your money. Keep them on paper, somewhere safe.'),
            h('div', { class: 'seed-grid', translate: 'no' }, f.words.map((w, i) => h('div', null, h('i', null, `${i + 1}.`), w))),
            h('a', { class: 'btn primary wide', href: '#/create/check' }, 'I wrote them down'),
            h('div', { class: 'links' },
                h('button', {
                    class: 'quiet-link', type: 'button',
                    onclick: () => { app.flow = { mine: f.mine, asked: true }; post({ type: 'GENERATE' }); render(); },
                }, 'Show different words')),
        ],
    };
}

function checkWordsScreen() {
    const f = app.flow;
    const inputs = f.quiz.map(i => h('input', { type: 'text', autocomplete: 'off', autocapitalize: 'none', spellcheck: 'false', 'aria-label': `Word ${i + 1}` }));
    const submit = (e) => {
        e.preventDefault();
        const bad = f.quiz.findIndex((wi, k) => inputs[k].value.trim().toLowerCase() !== f.words[wi]);
        if (bad >= 0) {
            // Back to the paper: a check that can be passed by guessing is no check.
            f.error = `Word ${f.quiz[bad] + 1} doesn’t match. Read the words again, then check again.`;
            f.quiz = pickQuiz(f.words.length);
            go('create');
            return;
        }
        f.error = '';
        f.checked = true;
        go('create/password');
    };
    return {
        bare: true, form: true, back: ['Show the words again', '#/create'],
        title: 'Check your copy',
        els: [
            h('p', { class: 'lead' }, 'Type these words from your paper.'),
            h('form', { onsubmit: submit }, f.quiz.map((wi, k) => field(`Word ${wi + 1}`, inputs[k])),
                h('button', { class: 'btn primary wide', type: 'submit' }, 'Check')),
        ],
        focus: () => inputs[0].focus(),
    };
}

function newPasswordScreen() {
    const f = app.flow;
    const pw = h('input', { type: 'password', autocomplete: 'new-password' });
    const pw2 = h('input', { type: 'password', autocomplete: 'new-password' });
    const err = h('div', { class: 'err', role: 'alert' }, takeOpenError());
    const submit = (e) => {
        e.preventDefault();
        const problem = passwordProblem(pw.value, pw2.value);
        if (problem) { err.textContent = problem; return; }
        startOpening('create', { type: 'CREATE', payload: { phrase: f.phrase, password: pw.value, rpcUrl: net.rpcUrl() } });
    };
    return {
        bare: true, form: true, back: ['Back', '#/create'],
        title: 'Choose a password',
        els: [
            h('p', { class: 'lead' }, 'It opens this wallet in this browser. Your 24 words are the backup, not this password.'),
            h('form', { onsubmit: submit },
                field('Password', pw), field('Password again', pw2), err,
                h('button', { class: 'btn primary wide', type: 'submit' }, f.mine ? 'Create wallet and start mining' : 'Create wallet')),
        ],
        focus: () => pw.focus(),
    };
}

/** The words typed on the restore form, kept (in memory only) while the
 *  form comes and goes, so a typo doesn't mean typing all 24 again. */
let restoreDraft = '';

function restoreScreen() {
    const words = h('textarea', { rows: 4, autocomplete: 'off', autocapitalize: 'none', spellcheck: 'false', 'aria-label': 'Your 24 words' });
    words.value = restoreDraft;
    const count = h('span', { class: 'muted small' });
    const pw = h('input', { type: 'password', autocomplete: 'new-password' });
    const pw2 = h('input', { type: 'password', autocomplete: 'new-password' });
    const err = h('div', { class: 'err', role: 'alert' }, takeOpenError());
    const recount = () => {
        const n = cleanPhrase(words.value).split(' ').filter(Boolean).length;
        count.textContent = n ? `${n} of 24 words` : '';
    };
    words.addEventListener('input', () => { restoreDraft = words.value; recount(); });
    recount();
    const submit = (e) => {
        e.preventDefault();
        const phrase = cleanPhrase(words.value);
        const n = phrase ? phrase.split(' ').length : 0;
        if (n !== 24) { err.textContent = n ? `A recovery phrase has 24 words. That’s ${n}.` : 'Type your 24 words.'; return; }
        const problem = passwordProblem(pw.value, pw2.value);
        if (problem) { err.textContent = problem; return; }
        app.flow = null;
        startOpening('restore', { type: 'CREATE', payload: { phrase, password: pw.value, rpcUrl: net.rpcUrl() } });
    };
    return {
        bare: true, form: true, back: ['Back', '#/have'],
        title: 'Restore your wallet',
        els: [
            hasSavedWallet() ? replaceWarning() : null,
            h('form', { onsubmit: submit },
                h('label', { class: 'field' }, h('div', { class: 'field-line' }, h('b', null, 'Your 24 words'), count), words),
                field('A password for this browser', pw), field('Password again', pw2), err,
                h('button', { class: 'btn primary wide', type: 'submit' }, 'Restore')),
            h('p', { class: 'note' }, 'Your money appears as the wallet finds it on the network. That can take a few minutes.'),
        ],
        focus: () => words.focus(),
    };
}

function openScreen() {
    const file = h('input', { type: 'file', accept: '.json,.dat,application/json', hidden: true });
    const chosen = h('div');
    const err = h('div', { class: 'err', role: 'alert' }, takeOpenError());
    const pw = h('input', { type: 'password', autocomplete: 'current-password' });
    let datBytes = null;
    const datForm = h('form', {
        hidden: true,
        onsubmit: (e) => {
            e.preventDefault();
            if (!pw.value) { err.textContent = 'Type the wallet file’s password.'; return; }
            startOpening('import', { type: 'IMPORT_CLI', payload: { fileBytes: datBytes, password: pw.value } });
        },
    }, field('The wallet file’s password', pw, h('p', { class: 'note' }, 'It becomes this wallet’s password here too.')),
    h('button', { class: 'btn primary wide', type: 'submit' }, 'Open'));
    file.addEventListener('change', async () => {
        const f = file.files && file.files[0];
        file.value = '';
        if (!f) return;
        err.textContent = '';
        if (/\.dat$/i.test(f.name)) {
            datBytes = new Uint8Array(await f.arrayBuffer());
            fill(chosen, h('p', null, 'Command-line wallet file: ', h('b', null, f.name)));
            datForm.hidden = false;
            pw.focus();
            return;
        }
        try {
            openBackupText(await f.text());
            app.openError = '';
            toast('Backup opened. Unlock it with its password.');
            go('', true);
        } catch (e) {
            err.textContent = errOf(e);
        }
    });
    return {
        bare: true, form: true, back: ['Back', '#/have'],
        title: 'Open a backup file',
        els: [
            h('p', { class: 'lead' }, 'A backup saved from this wallet (.json), or a wallet file from the command-line wallet (.dat).'),
            hasSavedWallet() ? replaceWarning() : null,
            chosen, err,
            h('button', { class: 'btn wide', type: 'button', onclick: () => file.click() }, icon('file'), 'Choose a file'),
            file, datForm,
        ],
    };
}

function lockedScreen() {
    const pw = h('input', { type: 'password', autocomplete: 'current-password', placeholder: 'Password', 'aria-label': 'Password' });
    const btn = h('button', { class: 'btn primary wide', type: 'submit' }, 'Unlock');
    const err = h('div', { class: 'err', role: 'alert' });
    const progress = h('div');
    let localErr = '';
    const submit = (e) => {
        e.preventDefault();
        if (app.unlocking) return;
        localErr = '';
        if (!pw.value) { localErr = 'Type your password.'; update(); pw.focus(); return; }
        const bundleStr = store.get(WALLET_KEY);
        if (!bundleStr) { app.phase = 'welcome'; render(); return; }
        app.unlocking = true;
        app.unlockError = '';
        post({ type: 'LOGIN', payload: { password: pw.value, rpcUrl: net.rpcUrl(), bundleStr } });
        update();
    };
    const trades = openTrades();
    let shown = null;
    return {
        bare: true,
        els: [
            h('h1', null, 'Welcome back'),
            h('p', { class: 'lead' }, app.dapp ? 'A site is asking to use your wallet. Unlock to see what it wants.' : 'Unlock your wallet to go on.'),
            migrationNote(),
            h('form', { class: 'stack', onsubmit: submit }, pw, err, btn, progress),
            h('div', { class: 'links' },
                h('a', { class: 'quiet-link', href: '#/have' }, 'Use a different wallet'),
                trades ? h('a', { class: 'quiet-link', href: 'classic.html' }, 'Classic wallet') : null),
        ],
        focus: () => pw.focus(),
        failed: () => { pw.select(); pw.focus(); },
        update() {
            btn.disabled = app.unlocking;
            btn.textContent = app.unlocking ? 'Unlocking…' : 'Unlock';
            err.textContent = app.unlockError || localErr;
            const m = app.unlocking ? app.making : null;
            const key = JSON.stringify(m);
            if (key !== shown) {
                shown = key;
                fill(progress, m ? [bar(m.total ? m.current / m.total : null), h('p', { class: 'note' }, plainMaking(m.label))] : null);
            }
        },
    };
}

/** The worker's key-making steps, for people. */
function plainMaking(label) {
    const l = String(label || '').toLowerCase();
    if (/tree|mss|address/.test(l)) return 'Making your address…';
    if (/wots|base keys/.test(l)) return 'Preparing keys…';
    return 'Working…';
}

function openingScreen() {
    const progress = h('div');
    const titles = { create: 'Making your wallet', restore: 'Restoring your wallet', import: 'Opening the wallet file' };
    let shown = null;
    return {
        bare: true,
        els: [h('h1', null, titles[app.opening && app.opening.what] || 'Opening your wallet'), progress,
            h('p', { class: 'note' }, 'This takes a moment. Keep this tab open.')],
        update() {
            const m = app.making;
            const key = JSON.stringify(m);
            if (key === shown) return;
            shown = key;
            fill(progress, bar(m && m.total ? m.current / m.total : null), h('p', { class: 'muted small' }, m ? plainMaking(m.label) : 'Starting…'));
        },
    };
}

// ── home ────────────────────────────────────────────────────────────────────

function changes() {
    const keys = {};
    return (name, value) => {
        const k = JSON.stringify(value);
        if (keys[name] === k) return false;
        keys[name] = k;
        return true;
    };
}

function homeScreen() {
    const notices = h('div');
    const balance = h('section', { class: 'card balance', 'aria-label': 'Balance' });
    const op = h('div');
    const rows = h('div');
    const recent = h('section', { 'aria-label': 'Recent activity' });
    const changed = changes();
    return {
        els: [notices, balance, op, rows, recent],
        update() {
            const d = app.dash;
            if (changed('notices', [app.update, app.stalled, app.reuse && app.reuse.address, (signaturesLeft() ?? Infinity) < LOW_SIGNATURES])) fill(notices, drawNotices());
            const sync = syncState();
            if (changed('balance', [d && d.balance, sync, pendingOut()])) drawBalance(balance, sync);
            if (changed('op', app.op && [app.op.words, app.op.frac, app.op.done, app.op.error, app.op.seen])) fill(op, opCard());
            const m = mining.state;
            const mv = book();
            const nv = nbook();
            if (changed('rows', [m.active, Math.round(m.nps / 100), positions(), mergeableCoins() >= DUST_HINT && mergeableCoins(), mv.available, mv.unread, mv.requests,
                nv.available, nv.names.map(n => [n.name, n.state, !!n.job])])) fill(rows, homeRows());
            if (changed('recent', d && [d.history.slice(0, 5), d.history.length, d.restoring, app.scan != null])) fill(recent, recentList());
        },
    };
}

function banner(text, action, onclick) {
    return h('div', { class: 'banner' }, h('span', { class: 'grow' }, text),
        action ? h('button', { class: 'btn small', type: 'button', onclick }, action) : null);
}

function drawNotices() {
    const out = [];
    if (app.update) {
        out.push(banner('A new version of the wallet is ready.', 'Reload', async () => {
            if (mining.state.active && !await confirmBox('Reload now?', 'Reloading stops mining for a moment.', 'Reload')) return;
            location.reload();
        }));
    }
    out.push(migrationNote());
    if (app.stalled) {
        out.push(banner('The node stopped answering the wallet’s questions about the chain.', 'Try another node', () => {
            app.stalled = false;
            update();
            net.connect().then(() => post({ type: 'SCAN' })).catch(() => {});
        }));
    }
    const left = signaturesLeft();
    if (left != null && left < LOW_SIGNATURES) {
        out.push(banner(`Your address has ${left} signature${left === 1 ? '' : 's'} left. Make a new one to receive with.`, 'Receive', () => go('receive')));
    }
    if (app.reuse) {
        out.push(h('div', { class: 'banner' }, h('span', { class: 'grow' }, h('b', null, 'Some coins are on hold. '),
            'A chain reorganisation brought back coins whose one-time key had already signed, so the wallet won’t spend them. ',
            h('details', { class: 'tech' }, h('summary', null, 'Details'), h('p', { class: 'small' }, app.reuse.msg)))));
    }
    return out;
}

function drawBalance(el, sync) {
    const d = app.dash;
    const out = pendingOut();
    fill(el,
        h('div', { class: 'label' }, 'Balance'),
        h('div', { class: 'big' }, h('b', null, d ? gmds(d.balance) : '—'), h('span', null, UNIT)),
        sync.text ? [bar(sync.frac), h('div', { class: 'note' }, sync.long)] : null,
        !sync.text && out ? h('div', { class: 'note' }, `${withUnit(out)} on its way out`) : null,
        h('div', { class: 'btns' },
            h('a', { class: 'btn primary', href: '#/send' }, icon('up'), 'Send'),
            h('a', { class: 'btn', href: '#/receive' }, icon('down'), 'Receive')));
}

function opTitle(o) {
    if (o.kind === 'tidy') return o.error ? 'Gathering coins didn’t finish' : o.done ? 'Coins gathered' : 'Gathering coins';
    if (o.kind === 'merge') return o.error ? 'Merging coins didn’t finish' : o.done ? 'Coins merged' : 'Merging coins';
    const to = o.name || shortAddr(o.to);
    return o.error ? `Payment to ${to} didn’t go through` : o.done ? `Sent ${withUnit(o.amount)} to ${to}` : `Sending ${withUnit(o.amount)} to ${to}`;
}

function opCard() {
    const o = app.op;
    if (!o || ((o.done || o.error) && o.seen)) return null;
    return h('a', { class: 'card op', href: '#/send/progress' },
        h('div', { class: 'under-way' }, h('b', null, opTitle(o)), h('span', null, o.error ? 'See why' : o.done ? 'Done' : o.words)),
        o.error || o.done ? null : bar(o.frac));
}

function homeRows() {
    const m = mining.state;
    const p = positions();
    const dust = mergeableCoins();
    const mv = book();
    const msgSub = mv.unread ? `${mv.unread} unread` : mv.requests ? `${mv.requests} message request${mv.requests === 1 ? '' : 's'}`
        : mv.available === false ? 'The public room' : 'Sealed: only you and who you write to can read them';
    const nv = nbook();
    const held = nv.names.filter(n => n.state !== 'lost' && (n.state !== 'new' || n.job));
    const due = held.filter(n => n.state === 'lapsing' || n.state === 'lapsed').length;
    const namesSub = due ? `${due === 1 ? `${held.find(n => n.state === 'lapsing' || n.state === 'lapsed').shown} needs` : `${due} names need`} renewing`
        : held.length ? held.map(n => n.shown).join(', ') : 'Get a name.mds: people pay you, and write to you, by it';
    return group(
        linkRow('Messages', '#/messages', { iconName: 'chat', sub: msgSub, value: mv.unread ? String(mv.unread) : null }),
        nv.available !== false ? linkRow('Names', '#/names', { iconName: 'tag', sub: namesSub, value: due ? 'Renew' : null }) : null,
        linkRow('Mining', '#/mine', { iconName: 'pick', sub: m.active ? null : 'Earn MDS with this device’s spare power', value: m.active ? speed(m.nps) : 'Off' }),
        dust >= DUST_HINT ? linkRow('Merge small coins', '#/settings/care', { iconName: 'coins', sub: `${dust.toLocaleString()} coins could be a few` }) : null,
        p.total ? linkRow('Trades, channels and bonds', 'classic.html', { iconName: 'box', sub: `Still open in the classic wallet: ${positionsText(p)}`, value: String(p.total) }) : null);
}

function activityItem(e, withDay = true) {
    const out = isOut(e);
    return h('a', { class: 'item', href: '#/activity/' + encodeURIComponent(entryKey(e)) },
        h('span', { class: 'what' }, e.label || kindLabel(e.kind),
            h('small', null, e.kind === 'pending' ? 'Waiting for the network' : withDay ? whenText(e.timestamp) : clock(e.timestamp))),
        // A name's change burns nothing: what it cost is its fee.
        h('span', { class: 'amt' + (out ? ' out' : '') }, (out ? '−' : '+') + (e.label && !e.value ? gmdsExact(e.fee || 0) : gmds(e.value))));
}

function recentList() {
    const d = app.dash;
    if (!d) return null;
    const items = d.history.slice(0, 5);
    return [
        h('div', { class: 'section-head' }, h('h2', null, 'Recent activity'),
            d.history.length ? h('a', { class: 'quiet-link', href: '#/activity' }, 'See all') : null),
        items.length
            ? h('div', { class: 'card flat list' }, items.map(e => activityItem(e)))
            : h('div', { class: 'card flat empty' }, h('b', null, 'Nothing yet'),
                d.restoring || app.scan ? 'Still looking through the chain.' : 'To get paid, choose Receive and share your address.'),
    ];
}

// ── activity ────────────────────────────────────────────────────────────────

function activityScreen() {
    const body = h('div');
    const changed = changes();
    return {
        title: 'Activity', back: ['Home', '#/'], els: [body],
        update() {
            const d = app.dash;
            if (!d || !changed('list', [d.history, d.restoring])) return;
            if (!d.history.length) {
                fill(body, h('div', { class: 'card flat empty' }, h('b', null, 'Nothing yet'),
                    d.restoring ? 'Still looking through the chain for this phrase’s payments.' : 'Payments in and out show here.'));
                return;
            }
            const days = [];
            for (const e of d.history) {
                const label = e.kind === 'pending' ? 'Under way' : dayLabel(e.timestamp);
                let g = days[days.length - 1];
                if (!g || g.label !== label) days.push(g = { label, items: [] });
                g.items.push(e);
            }
            fill(body, days.map(g => [h('div', { class: 'day' }, g.label), h('div', { class: 'list' }, g.items.map(e => activityItem(e, false)))]));
        },
    };
}

function activityItemScreen(rawKey) {
    let key = rawKey;
    try { key = decodeURIComponent(rawKey); } catch (_) {}
    const body = h('div');
    const changed = changes();
    const explorer = q => `${net.rpcUrl()}/?q=${encodeURIComponent(q)}`;
    const coinList = (title, ids) => (ids && ids.length ? [h('p', { class: 'muted small' }, title),
        ids.map(id => h('a', { class: 'mono small break block', href: explorer(id), target: '_blank', rel: 'noopener' }, id))] : null);
    return {
        title: 'Payment', back: ['Activity', '#/activity'], backAlways: true, els: [body],
        update() {
            const d = app.dash;
            if (!d) return;
            const e = d.history.find(x => entryKey(x) === key);
            if (!changed('e', [e, d.networkHeight])) return;
            if (!e) {
                fill(body, h('p', { class: 'muted' }, 'This payment isn’t in the list any more.'), h('a', { class: 'btn', href: '#/activity' }, 'All activity'));
                return;
            }
            const out = isOut(e);
            const confirmations = e.height != null && d.networkHeight ? Math.max(1, d.networkHeight - e.height + 1) : null;
            fill(body,
                h('div', { class: 'card balance' }, h('div', { class: 'label' }, e.label || kindLabel(e.kind)),
                    h('div', { class: 'big' }, h('b', null, (out ? '−' : '+') + gmdsExact(e.value)), h('span', null, UNIT))),
                group(
                    infoRow('When', dateTime(e.timestamp)),
                    e.kind === 'pending' ? infoRow('Status', 'Waiting for the network')
                        : e.height != null ? infoRow('In block', h('a', { href: explorer(e.height), target: '_blank', rel: 'noopener' }, e.height.toLocaleString()))
                            : null,
                    confirmations ? infoRow('Confirmations', confirmations.toLocaleString()) : null,
                    out && e.fee ? infoRow('Network fee', exactUnit(e.fee)) : null),
                h('details', { class: 'tech' }, h('summary', null, 'Technical details'),
                    coinList(out ? 'Coins spent' : 'Coins', out ? e.inputs : e.outputs),
                    out ? coinList('Change back to this wallet', e.outputs) : null));
        },
    };
}

// ── sending ─────────────────────────────────────────────────────────────────

function sendScreen() {
    const to = h('input', { type: 'text', value: app.draft.to, placeholder: 'name.mds or address', autocomplete: 'off', autocapitalize: 'none', spellcheck: 'false', 'aria-label': 'To' });
    const toLine = h('div', { class: 'field-note' });
    const amount = h('input', { type: 'text', inputmode: 'decimal', value: app.draft.amount, placeholder: '0', autocomplete: 'off', 'aria-label': `Amount in ${UNIT}` });
    const amountLine = h('div', { class: 'field-note' });
    const have = h('span', { class: 'muted small' });
    const review = h('button', { type: 'submit', class: 'btn primary wide' }, 'Review');
    const err = h('div', { class: 'err', role: 'alert' }, app.sendError || '');
    app.sendError = '';
    let target = { state: 'empty' }; // empty | partial | checking | ok | bad
    let gen = 0;
    let timer = null;
    let amountTouched = false;

    const showTarget = (loud) => {
        const t = target;
        if (t.state === 'checking') fill(toLine, h('span', { class: 'muted' }, spinner(), ' Checking…'));
        else if (t.state === 'ok') {
            fill(toLine, h('span', { class: 'ok-line' }, t.name || (t.count ? `${t.count} valid addresses` : 'Valid address')),
                t.name ? h('span', { class: 'muted small mono' }, ` pays ${shortAddr(t.address, 8)}`) : null);
        } else if (t.state === 'bad' || (t.state === 'partial' && loud)) fill(toLine, h('span', { class: 'err' }, t.why));
        else fill(toLine);
        check();
    };

    const examine = (loud) => {
        const raw = to.value.trim();
        app.draft.to = to.value;
        const my = ++gen;
        clearTimeout(timer);
        if (!raw) { target = { state: 'empty' }; showTarget(); return; }
        if (/^mw1/i.test(raw)) {
            target = { state: 'bad', why: 'That’s a midwimble address. Send it from a midwimble wallet.' };
            showTarget();
            return;
        }
        if (isName(raw)) {
            let name;
            try { name = normalizeName(raw); } catch (e) { target = { state: 'bad', why: errOf(e) }; showTarget(); return; }
            target = { state: 'checking' };
            showTarget();
            timer = setTimeout(async () => {
                try {
                    const r = await ask('RESOLVE_NAME', { name });
                    if (my !== gen) return;
                    target = { state: 'ok', name: r.name, address: r.address };
                } catch (e) {
                    if (my !== gen) return;
                    target = { state: 'bad', why: nameProblem(errOf(e)) };
                }
                showTarget();
            }, 400);
            return;
        }
        const list = raw.includes(',') ? cleanAddressList(raw) : null;
        if (list) {
            // A multi-address bundle: every address must carry its checksum.
            target = { state: 'checking' };
            showTarget();
            Promise.all(list.map(address => ask('CHECK_ADDRESS', { address }))).then((rs) => {
                if (my !== gen) return;
                target = rs.every(r => r.ok)
                    ? { state: 'ok', address: list.join(','), count: list.length }
                    : { state: 'bad', why: 'One of those addresses has a typo. Check them with the person you’re paying.' };
                showTarget();
            }, () => {});
            return;
        }
        if (raw.includes(',')) {
            const parts = raw.split(',').map(p => p.replace(/\s+/g, '').replace(/^0x/i, ''));
            const typing = /^[0-9a-f,\sx]*$/i.test(raw) && parts[parts.length - 1].length < 72;
            target = { state: typing ? 'partial' : 'bad', why: 'A list of addresses takes 2 to 16 full 72-character addresses, separated by commas, none repeated.' };
            showTarget(loud);
            return;
        }
        const addr = cleanAddress(raw);
        if (!addr) {
            const hex = raw.replace(/\s+/g, '').replace(/^0x/i, '');
            const hexOnly = /^[0-9a-f]+$/i.test(hex);
            let why = 'Not an address or a name.';
            if (hexOnly && hex.length === 64) why = 'That address has no checksum. Ask for the full 72-character address.';
            else if (hexOnly && hex.length >= 20 && hex.length < 72) why = 'That address is too short: an address has 72 characters.';
            else if (hexOnly && hex.length > 72) why = 'That address is too long: an address has 72 characters.';
            else if (/^[a-z0-9-]+$/i.test(raw)) why = `Not an address. If it’s a name, add .mds: ${raw.toLowerCase()}.mds`;
            // While it's still being typed, say nothing yet.
            const partial = hexOnly ? hex.length < 72 && hex.length !== 64 : /^[a-z0-9-]+(\.m?d?s?)?$/i.test(raw);
            target = { state: partial ? 'partial' : 'bad', why };
            showTarget(loud);
            return;
        }
        target = { state: 'checking' };
        showTarget();
        ask('CHECK_ADDRESS', { address: addr }).then((r) => {
            if (my !== gen) return;
            target = r.ok ? { state: 'ok', address: addr } : { state: 'bad', why: 'That address has a typo. Check it with the person you’re paying.' };
            showTarget();
        }, () => {});
    };

    const check = () => {
        const bal = balanceOf();
        have.textContent = `You have ${withUnit(bal)}`;
        let ok = false;
        if (app.draft.all) {
            ok = bal > 0;
            fill(amountLine, ok ? h('span', { class: 'muted' }, 'Everything, less the network fee') : h('span', { class: 'err' }, 'There’s nothing to send.'));
        } else {
            const a = parseAmount(amount.value);
            if (a.error != null) fill(amountLine, a.error && (amountTouched || a.error !== 'Enter an amount') ? h('span', { class: 'err' }, a.error) : null);
            else if (a.units > bal) fill(amountLine, h('span', { class: 'err' }, 'More than you have.'));
            else { ok = true; fill(amountLine); }
        }
        const blocked = sendBlocked();
        review.disabled = !(target.state === 'ok' && ok) || !!blocked;
        review.textContent = blocked || 'Review';
    };

    to.addEventListener('input', () => { err.textContent = ''; examine(false); });
    to.addEventListener('blur', () => { if (target.state === 'partial') showTarget(true); });
    amount.addEventListener('input', () => {
        err.textContent = '';
        app.draft.amount = amount.value;
        app.draft.all = false;
        check();
    });
    amount.addEventListener('blur', () => { amountTouched = true; check(); });
    const max = () => {
        app.draft.all = true;
        amount.value = plainAmount(balanceOf());
        app.draft.amount = amount.value;
        check();
    };
    const paste = async () => {
        try {
            const text = await navigator.clipboard.readText();
            to.value = String(text || '').trim();
            examine(true);
        } catch (_) {
            toast('The browser didn’t allow pasting here. Paste into the field instead.');
            to.focus();
        }
    };
    const scan = async () => {
        const text = await scanCode();
        if (text) { to.value = text; examine(true); amount.focus(); }
    };
    const submit = (e) => {
        e.preventDefault();
        if (review.disabled) { examine(true); amountTouched = true; check(); return; }
        const units = app.draft.all ? balanceOf() : parseAmount(amount.value).units;
        app.review = { typed: to.value.trim(), name: target.name || null, address: target.address, amount: units, all: !!app.draft.all, typedAmount: amount.value };
        go('send/review');
    };
    if (app.draft.to) examine(true);
    return {
        title: 'Send', back: ['Home', '#/'],
        els: [
            h('form', { class: 'card', onsubmit: submit, autocomplete: 'off' },
                h('label', { class: 'field' }, h('span', null, 'To'), to),
                h('div', { class: 'field-tools' },
                    navigator.clipboard && navigator.clipboard.readText ? h('button', { type: 'button', class: 'btn small', onclick: paste }, icon('paste'), 'Paste') : null,
                    h('button', { type: 'button', class: 'btn small', onclick: scan }, icon('scan'), 'Scan')),
                toLine,
                h('div', { class: 'field-line spaced' }, h('b', null, 'Amount'),
                    h('span', null, have, ' · ', h('button', { type: 'button', class: 'quiet-link', onclick: max }, 'Max'))),
                h('div', { class: 'input-unit' }, amount, h('span', null, UNIT)),
                amountLine,
                err,
                review),
        ],
        update() { check(); },
        focus: () => (app.draft.to ? amount : to).focus(),
    };
}

function reviewScreen() {
    const r = app.review;
    if (!r) return redirect('send');
    const rows = h('div');
    const err = h('div', { class: 'err', role: 'alert' });
    const sendBtn = h('button', { class: 'btn primary', type: 'button' }, 'Send');
    let quote = null;
    let sending = false;
    const draw = () => {
        fill(rows, group(
            h('div', { class: 'row stacked' }, h('span', { class: 'label muted small' }, 'To'),
                r.name ? h('b', null, r.name) : null,
                ...r.address.split(',').map(a => h('span', { class: 'mono small break' + (r.name ? ' muted' : '') }, groupHex(a)))),
            infoRow('Amount', quote ? exactUnit(quote.amount) : r.all ? 'Everything' : exactUnit(r.amount)),
            infoRow('Network fee', quote ? exactUnit(quote.fee) : '…'),
            infoRow('Total', quote ? h('b', null, exactUnit(quote.amount + quote.fee)) : '…')));
        sendBtn.disabled = !quote || sending;
        sendBtn.textContent = sending ? 'Sending…' : 'Send';
    };
    const runQuote = async () => {
        quote = null;
        draw();
        try {
            quote = await ask('QUOTE_SEND', { toAddress: r.address, amount: r.amount, sendAll: r.all });
        } catch (e) {
            app.sendError = sendProblem(errOf(e));
            if (current && current.sc === screen) go('send', true);
            return;
        }
        if (current && current.sc === screen) draw();
    };
    sendBtn.addEventListener('click', async () => {
        if (sending || !quote) return;
        sending = true;
        err.textContent = '';
        draw();
        if (r.name) {
            // A name can be pointed somewhere else at any time: look once more.
            try {
                const again = await ask('RESOLVE_NAME', { name: normalizeName(r.name) });
                if (again.address !== r.address) {
                    r.address = again.address;
                    sending = false;
                    err.textContent = `${r.name} now pays a different address. Check it, then send.`;
                    runQuote();
                    return;
                }
            } catch (e) {
                sending = false;
                err.textContent = nameProblem(errOf(e));
                draw();
                return;
            }
        }
        startSend(r, quote);
    });
    const screen = {
        title: 'Review', back: ['Edit', '#/send'], backAlways: true,
        els: [rows, err,
            h('div', { class: 'btns' }, h('a', { class: 'btn', href: '#/send' }, 'Edit'), sendBtn),
            h('p', { class: 'note' }, 'A payment can’t be undone. If you typed the address, check it with the person you’re paying.')],
    };
    runQuote();
    return screen;
}

function startSend(r, quote) {
    app.op = {
        kind: 'send', to: r.address, name: r.name, amount: quote.amount, fee: quote.fee, all: r.all,
        typed: r.typed, typedAmount: r.typedAmount, words: 'Preparing', frac: 0.05, done: false, error: '', seen: false,
    };
    post({ type: 'SEND', payload: { toAddress: r.address, amount: r.all ? quote.amount : r.amount, sendAll: r.all } });
    app.draft = { to: '', amount: '', all: false };
    app.review = null;
    go('send/progress', true);
}

function progressScreen() {
    if (!app.op) return redirect('');
    const body = h('div');
    const changed = changes();
    const retry = () => {
        const o = app.op;
        app.draft = { to: o.typed || o.to, amount: o.all ? '' : (o.typedAmount || plainAmount(o.amount)), all: !!o.all };
        app.op = null;
        go('send');
    };
    return {
        back: ['Home', '#/'], els: [body],
        update() {
            const o = app.op;
            if (!o || !changed('o', [o.words, o.frac, o.done, o.error])) return;
            if (o.done || o.error) o.seen = true;
            const title = o.kind === 'send' ? (o.error ? 'Not sent' : o.done ? 'Sent' : 'Sending') : opTitle(o);
            fill(body,
                h('h1', null, title),
                o.kind === 'send' ? h('div', { class: 'card balance' },
                    h('div', { class: 'big' }, h('b', null, gmdsExact(o.amount)), h('span', null, UNIT)),
                    h('div', { class: 'muted' }, 'to ', o.name ? h('b', null, o.name) : h('span', { class: 'mono' }, shortAddr(o.to, 10)))) : null,
                o.error
                    ? [h('p', { class: 'err' }, o.error),
                        h('div', { class: 'btns' }, h('a', { class: 'btn', href: '#/' }, 'Home'),
                            o.kind === 'send' ? h('button', { class: 'btn primary', type: 'button', onclick: retry }, 'Try again') : null)]
                    : o.done
                        ? [h('p', { class: 'ok-line' }, o.kind === 'tidy' ? 'Your coins are on your main address.'
                            : o.kind === 'merge' ? 'Your small coins are merged into a few on your main address.' : 'It’s on the chain.'),
                            h('div', { class: 'btns' }, h('a', { class: 'btn primary', href: '#/', onclick: () => { app.op = null; } }, 'Done'))]
                        : [bar(o.frac), h('div', { class: 'under-way' }, h('b', null, o.words), h('span', null, `${Math.round(o.frac * 100)}%`)),
                            h('p', { class: 'note' }, 'Keep this tab open until it’s done. It takes a couple of blocks, usually a few minutes.')]);
        },
    };
}

// ── scanning a QR code ──────────────────────────────────────────────────────

function cameraTrouble(e) {
    switch (e && e.name) {
        case 'NotAllowedError':
        case 'SecurityError':
            return 'Camera access is blocked. Allow it for this site in the browser’s settings, or choose a photo of the code.';
        case 'NotFoundError':
        case 'OverconstrainedError':
            return 'No camera found. Choose a photo of the code instead.';
        case 'NotReadableError':
        case 'AbortError':
            return 'Another app is using the camera. Close it, or choose a photo of the code.';
        default:
            return `The camera didn’t start (${errOf(e)}). Choose a photo of the code instead.`;
    }
}

/** What a QR code holds, if it's somewhere to send to: a name or an address. */
function codeText(text) {
    const t = String(text || '').trim().replace(/^midstate:/i, '').split('?')[0];
    if (isName(t)) return t.toLowerCase();
    return cleanAddress(t);
}

/** Reads a QR code with the camera or from a photo; resolves with an
 *  address or a name, or null. The camera is off again by then. */
async function scanCode() {
    let qr;
    try { qr = await import('./qr_scan.js'); } catch (e) { await okBox('Can’t scan', `The QR reader didn’t load (${errOf(e)}).`); return null; }
    const video = h('video', { playsinline: true, autoplay: true, muted: true });
    video.muted = true;
    video.playsInline = true;
    const view = h('div', { class: 'scanner', hidden: true }, video);
    const status = h('p', { class: 'muted small', 'aria-live': 'polite' });
    const file = h('input', { type: 'file', accept: 'image/*', hidden: true });
    const camera = new qr.Camera(video);
    let done = false;
    let close = null;
    let rejected = null;
    const consider = (text) => {
        if (text == null || done || text === rejected) return;
        const t = codeText(text);
        if (!t) { rejected = text; status.textContent = 'That code isn’t a Midstate address or name.'; return; }
        done = true;
        close(t);
    };
    const watch = async () => {
        while (!done && camera.live) {
            let text = null;
            try { text = await camera.read(); } catch (e) { status.textContent = `The reader failed (${errOf(e)}). Choose a photo instead.`; return; }
            consider(text);
            await sleep(120);
        }
    };
    file.addEventListener('change', async () => {
        const f = file.files && file.files[0];
        file.value = '';
        if (!f || done) return;
        status.textContent = 'Reading the picture…';
        let text = null;
        try { text = await qr.readImageFile(f); } catch (_) { status.textContent = 'That file isn’t a picture the browser can read.'; return; }
        if (text == null) { status.textContent = 'No QR code in that picture. Try a sharper or closer one.'; return; }
        rejected = null;
        consider(text);
    });
    const result = sheet('Scan a code', (c) => {
        close = c;
        return [view, status, file, h('div', { class: 'btns' },
            h('button', { class: 'btn', type: 'button', onclick: () => file.click() }, 'Choose a photo'),
            h('button', { class: 'btn', type: 'button', onclick: () => c(null) }, 'Cancel'))];
    });
    if (qr.cameraSupported()) {
        status.textContent = 'Starting the camera…';
        camera.start(null).then(() => {
            if (done) return;
            view.hidden = false;
            status.textContent = 'Point the camera at the QR code.';
            watch();
        }, (e) => { if (!done) status.textContent = cameraTrouble(e); });
    } else {
        status.textContent = 'This browser can’t use a camera here. Choose a photo of the code instead.';
    }
    const text = await result;
    done = true;
    camera.close();
    return text;
}

// ── receiving ───────────────────────────────────────────────────────────────

function receiveScreen() {
    const qrBox = h('div', { class: 'qr' });
    const addrEl = h('div', { class: 'addr', translate: 'no' });
    const making = h('div');
    const low = h('div');
    const changed = changes();
    const address = () => (app.dash && app.dash.primaryAddress && app.dash.primaryAddress !== 'None' ? app.dash.primaryAddress : '');
    const newAddress = async () => {
        if (app.newAddress) return;
        if (!await confirmBox('Make a new address?', 'Your current address keeps working. A new one takes a moment to make.', 'Make one')) return;
        app.newAddress = true;
        app.making = null;
        post({ type: 'NEW_ADDRESS' });
        update();
    };
    return {
        title: 'Receive', back: ['Home', '#/'],
        els: [
            h('div', { class: 'card' },
                h('div', { class: 'qr-block' }, qrBox,
                    h('div', { class: 'grow' },
                        h('div', { class: 'muted small' }, 'Your address'),
                        addrEl,
                        h('div', { class: 'btns' },
                            h('button', { class: 'btn primary', type: 'button', onclick: () => address() && copyText(address(), 'Address copied') }, icon('copy'), 'Copy address'),
                            navigator.share ? h('button', { class: 'btn', type: 'button', onclick: () => navigator.share({ text: address() }).catch(() => {}) }, icon('share'), 'Share') : null)))),
            low,
            making,
            h('div', { class: 'links left' }, h('button', { class: 'quiet-link', type: 'button', onclick: newAddress }, 'Make a new address')),
        ],
        update() {
            const a = address();
            if (changed('a', a)) {
                fill(qrBox, a ? qrSvg(a.toUpperCase(), 'M') : null);
                fill(addrEl, a ? groupHex(a, 4) : app.newAddress ? 'Making your address…' : 'This wallet has no address yet. Make one below.');
            }
            const left = signaturesLeft();
            if (changed('low', [left, app.newAddress])) {
                fill(low, left != null && left < LOW_SIGNATURES && !app.newAddress
                    ? h('div', { class: 'warn-box' }, h('b', null, `This address has ${left} signature${left === 1 ? '' : 's'} left.`),
                        'Each payment out of it uses one. Make a new address and give that out instead, so what you’re sent stays easy to spend.')
                    : h('p', { class: 'note' }, 'Use this address as often as you like. Anyone can pay you with the code or the address.'));
            }
            const m = app.making;
            if (changed('m', [app.newAddress, m])) {
                fill(making, app.newAddress ? [bar(m && m.total ? m.current / m.total : null), h('p', { class: 'note' }, 'Making a new address…')] : null);
            }
        },
    };
}

// ── mining ──────────────────────────────────────────────────────────────────

const stat = (label, value) => h('div', { class: 'stat' }, h('small', null, label), h('b', null, value));

function mineScreen() {
    const status = h('section', { class: 'card mine-status', 'aria-label': 'Mining' });
    const settingsBox = h('div');
    const changed = changes();
    return {
        title: 'Mine', back: ['Home', '#/'], els: [status, settingsBox],
        update() {
            const s = mining.state;
            if (changed('s', [s.active, Math.round(s.nps), s.poolScore, s.mwScore, s.blocksFound, s.note, s.backend, s.mode, s.threads, mining.template != null])) drawMineStatus(status);
            if (changed('k', [s.mode, s.threads, s.effort, s.pauseOnBattery, s.poolUrl, s.midwimble, mining.hasBattery, mergeableCoins()])) fill(settingsBox, mineSettings());
        },
    };
}

function drawMineStatus(el) {
    const s = mining.state;
    const per = mining.perDay();
    const pooled = s.mode === 'pool';
    fill(el,
        h('div', { class: 'mine-state' }, h('span', { class: 'dot ' + (s.active ? 'ok' : 'bad') }), s.active ? 'Mining' : 'Not mining'),
        h('div', { class: 'mine-speed' }, s.active ? speed(s.nps) : '—'),
        h('p', { class: 'muted' }, s.active
            ? (per != null ? `About ${withUnit(per)} a day at this speed${pooled ? ', paid out by the pool' : ''}.`
                : `Getting work from the ${pooled ? 'pool' : 'network'}…`)
            : 'Uses this device’s spare power. Stop any time.'),
        s.note ? h('p', { class: 'note' }, s.note) : null,
        s.active ? h('div', { class: 'stat-grid' },
            pooled ? stat('Your shares', s.poolScore >= 0 ? Number(s.poolScore).toLocaleString() : '—') : stat('Blocks found', String(s.blocksFound)),
            stat('Working on', s.backend === 'cpu' ? `${s.threads} thread${s.threads === 1 ? '' : 's'}` : `GPU and ${s.threads - 1 || 0} thread${s.threads === 2 ? '' : 's'}`),
            pooled && s.midwimble && s.mwScore != null ? stat('Midwimble shares', Number(s.mwScore).toLocaleString()) : null) : null,
        h('button', {
            class: 'btn wide' + (s.active ? '' : ' primary'), type: 'button',
            onclick: () => {
                if (s.active) { mining.stop(); return; }
                try { mining.start(); } catch (e) { okBox('Can’t start mining', errOf(e)); }
            },
        }, icon('pick'), s.active ? 'Stop mining' : 'Start mining'));
}

function mineSettings() {
    const s = mining.state;
    const threads = h('input', { type: 'range', min: 1, max: MAX_THREADS, step: 1, value: s.threads, 'aria-label': 'Threads' });
    const threadsLabel = h('span', { class: 'value' }, `${s.threads} of ${MAX_THREADS}`);
    threads.addEventListener('input', () => { threadsLabel.textContent = `${threads.value} of ${MAX_THREADS}`; });
    threads.addEventListener('change', () => mining.setThreads(Number(threads.value)));
    const pooled = s.mode === 'pool';
    return [
        h('h2', null, 'Settings'),
        group(
            controlRow('How', seg([['pool', 'Pool'], ['solo', 'Solo']], s.mode, v => mining.setMode(v), 'How to mine'),
                pooled ? 'Small payouts, often' : 'A whole block, rarely'),
            h('div', { class: 'row stacked' }, h('div', { class: 'field-line' }, h('span', { class: 'label' }, 'Threads'), threadsLabel), threads),
            controlRow('Effort', seg(EFFORTS.map((e, i) => [i, e.label]), s.effort, v => mining.setEffort(v), 'Effort')),
            mining.hasBattery ? controlRow('Pause on low battery', toggle(s.pauseOnBattery, v => mining.setPauseOnBattery(v), 'Pause on low battery'), 'Below 20% and off the charger') : null,
            pooled ? linkRow('Pool', '#/mine/pool', { value: hostOf(s.poolUrl), sub: s.midwimble ? 'Also earning midwimble' : null }) : null),
        h('p', { class: 'note' }, pooled
            ? 'The pool pays your share of each block it finds to this wallet.'
            : 'Solo mining needs a direct connection to a node. A block you find pays its whole reward to this wallet.'),
        mergeableCoins() > 1 ? group(linkRow('Merge small coins', '#/settings/care', { iconName: 'coins', sub: 'Mining pays out in many small coins', value: `${mergeableCoins().toLocaleString()} coins` })) : null,
        h('div', { class: 'links left' }, h('button', { class: 'quiet-link', type: 'button', onclick: () => copyText(mining.shareUrl(), 'Link copied') }, 'Copy a link that invites a friend to mine')),
    ];
}

function poolScreen() {
    const s = mining.state;
    const url = h('input', { type: 'text', value: s.poolUrl, autocomplete: 'off', spellcheck: 'false', autocapitalize: 'none' });
    const mw = h('input', { type: 'text', value: s.midwimble, placeholder: 'mw1…', autocomplete: 'off', spellcheck: 'false', autocapitalize: 'none' });
    const err = h('div', { class: 'err', role: 'alert' });
    const save = (e) => {
        e.preventDefault();
        try {
            mining.setMidwimble(mw.value);
            mining.setPoolUrl(url.value);
            toast('Saved.');
            go('mine');
        } catch (x) {
            err.textContent = errOf(x);
        }
    };
    return {
        title: 'Pool', back: ['Mine', '#/mine'], backAlways: true,
        els: [h('form', { class: 'card', onsubmit: save },
            field('Pool address', url, h('p', { class: 'note' }, `Unless you choose another: ${DEFAULT_POOL_URL}`)),
            field('Also earn midwimble (optional)', mw, h('p', { class: 'note' }, 'A midwimble address. The pool pays it midwimble for the same work, as well as MDS here.')),
            err,
            h('div', { class: 'btns' },
                h('button', { class: 'btn', type: 'button', onclick: () => { url.value = DEFAULT_POOL_URL; } }, 'Use the default pool'),
                h('button', { class: 'btn primary', type: 'submit' }, 'Save')))],
    };
}

// ── settings ────────────────────────────────────────────────────────────────

// ── messages ────────────────────────────────────────────────────────────────
// Sealed messages between names, as the desktop app's Messages page has them:
// conversations, requests from strangers, writing to a name, and the name
// this wallet writes as. The worker keeps them (messages.js); this draws them.

const TIMER_CHOICES = [[0, 'Off'], [3_600, '1 hour'], [86_400, '1 day'], [7 * 86_400, '1 week'], [28 * 86_400, '4 weeks']];
const STAMP_RATE_GUESS = 2e6; // stamp tries a second in a browser, until the worker has measured
const MAX_MESSAGE = 750;      // bytes (sealed::MAX_BODY)

const book = () => app.messages || { available: null, chats: [], unread: 0, requests: 0, blocked: [] };
const msgAsk = (op, args = {}) => ask('MSG', { op, ...args });
const work = bits => about(2 ** bits / (book().rate || STAMP_RATE_GUESS));
const chatTitle = c => c.peer || 'Anonymous';
const oneLine = t => String(t || '').split(/\s+/).filter(Boolean).join(' ');
const chatHref = c => '#/messages/c/' + encodeURIComponent(c.key);
const findChat = key => book().chats.find(c => c.key === key) || null;
const blocksName = c => c.verified && !!c.peer;
const bytesOf = text => new TextEncoder().encode(String(text || '').trim()).length;
const coarsePointer = () => !!(window.matchMedia && matchMedia('(pointer: coarse)').matches);

/** What a stamp on a message to `who` costs, said under the box. */
function stampNote(who, bits) {
    if (bits <= 24) return `Until ${who} writes back, each message carries a stamp: ${work(bits)} of this device’s work, so nobody can flood an inbox for free.`;
    return `${who} asks strangers more: until they write back, each message carries a stamp of ${work(bits)} of this device’s work (${priceLabel(bits).toLowerCase()}). It goes once the work is done; you can keep writing meanwhile.`;
}

function timerNote(c, l) {
    const who = l.mine ? 'You' : c.peer || 'The sender';
    return l.timer ? `${who} set messages to disappear after ${timerText(l.timer)}` : `${who} turned off disappearing messages`;
}

/** A conversation's last word, for the list. */
function preview(c) {
    const l = c.lines[c.lines.length - 1];
    if (!l) return c.ttl ? `Messages disappear after ${timerText(c.ttl)}` : '';
    if (l.timer != null) return timerNote(c, l);
    if (l.pending) return `Waiting for its key: ${oneLine(l.text)}`;
    if (l.stamping) return `Stamping: ${oneLine(l.text)}`;
    if (l.failed) return `Not sent: ${oneLine(l.text)}`;
    return l.mine ? `You: ${oneLine(l.text)}` : oneLine(l.text);
}

/** Why a conversation waits under Requests. */
function requestReason(c) {
    if (!c.peer) return 'the sender did not say who they are';
    if (!c.verified) return `the sender’s claim to be ${c.peer} did not check out`;
    return `${c.peer} is not in your inbox yet`;
}

/** The line under a conversation's name. */
function chatSubtitle(c) {
    if (!c.id && c.lines.some(l => l.stamping)) return `Stamping your first message to ${c.peer}: it goes, sealed, once the work is done.`;
    if (!c.id) return `Waiting for ${c.peer}’s wallet to hand over its key: your messages go as soon as it does, if it does within 15 minutes. Their wallet has to be online for that.`;
    if (!c.peer) return 'Whoever opened this did not say who they are. Only they read your replies.';
    if (c.verified) return `Sealed: only your wallet and ${c.peer}’s can read this.`;
    return `Claims to be ${c.peer}. Only whoever opened this reads your replies.`;
}

// The worker reads the chat bus every 15 seconds while Messages is on screen,
// every minute otherwise.
let watchingMessages = false;
let unwatchTimer = null;
function watchMessages() {
    clearTimeout(unwatchTimer);
    if (watchingMessages) return;
    watchingMessages = true;
    post({ type: 'MSG', payload: { op: 'watch', on: true } });
}
function unwatchMessages() {
    clearTimeout(unwatchTimer);
    unwatchTimer = setTimeout(() => {
        if (routePath().startsWith('messages')) return;
        watchingMessages = false;
        post({ type: 'MSG', payload: { op: 'watch', on: false } });
    }, 400);
}

function convoRow(c) {
    const fresh = c.unread > 0;
    return h('a', { class: 'row link convo' + (fresh ? ' unread' : ''), href: chatHref(c) },
        h('span', { class: 'tile', 'aria-hidden': 'true' }, initial(chatTitle(c))),
        h('span', { class: 'label' },
            h('span', { class: 'convo-top' }, h('b', null, chatTitle(c)), c.peer && c.id && !c.verified ? h('span', { class: 'tag' }, 'unverified') : null),
            h('small', null, preview(c) || ' ')),
        h('span', { class: 'convo-side' }, h('small', null, lately(c.last)),
            fresh ? h('span', { class: 'badge', 'aria-label': `${c.unread} unread` }, String(c.unread)) : null));
}

function notBuilt() {
    return h('div', { class: 'card' }, h('p', null, 'Sealed messages aren’t in this build of the wallet.'),
        h('p', { class: 'muted small' }, 'Its WebAssembly predates them: reload once the page updates.'));
}

function messagesScreen(tab) {
    watchMessages();
    const bar = h('div');
    const room = h('div');
    const list = h('div');
    const foot = h('div');
    const changed = changes();
    msgAsk('room').then((r) => { if (!app.room) { app.room = r; update(); } }, () => {});
    return {
        title: 'Messages', back: ['Home', '#/'], leave: unwatchMessages,
        els: [bar, room, list, foot],
        update() {
            const v = book();
            if (changed('bar', [tab, v.unread, v.requests, v.available])) {
                fill(bar, v.available === false ? null : h('div', { class: 'msg-bar' },
                    seg([['inbox', v.unread ? `Inbox ${v.unread}` : 'Inbox'], ['requests', v.requests ? `Requests ${v.requests}` : 'Requests']], tab,
                        t => go(t === 'inbox' ? 'messages' : 'messages/requests', true), 'Messages'),
                    h('a', { class: 'btn small primary', href: '#/messages/new' }, icon('plus'), 'New message')));
            }
            const r = roomState();
            const last = r.messages[r.messages.length - 1];
            if (changed('room', [tab, last && [last.nonce, last.sender], r.sender])) fill(room, tab === 'inbox' ? group(roomRow(r)) : null);
            const shown = v.chats.filter(c => (tab === 'requests') === !!c.request);
            if (changed('list', [tab, v.available, shown.map(c => [c.key, c.last, c.unread, c.peer, c.verified, preview(c)])])) {
                if (v.available === false) fill(list, notBuilt());
                else if (v.available == null) fill(list, h('p', { class: 'muted' }, spinner(), ' Opening your messages…'));
                else if (shown.length) fill(list, h('h2', { class: 'list-head' }, tab === 'requests' ? 'Requests' : 'Sealed'), group(shown.map(convoRow)));
                else if (tab === 'requests') {
                    fill(list, h('div', { class: 'card flat empty' }, h('b', null, 'No message requests.'),
                        'Messages from people who don’t prove a name wait here until you answer or accept them. They raise no notice.'));
                } else {
                    fill(list, h('div', { class: 'card flat empty' }, h('b', null, 'No sealed conversations yet.'),
                        'Write to anyone whose name.mds takes sealed messages. Only their wallet can read what you send.',
                        h('div', { class: 'btns' }, h('a', { class: 'btn', href: '#/messages/new' }, icon('plus'), 'New message'))));
                }
            }
            if (changed('foot', [v.available, v.asName])) {
                fill(foot, v.available === false ? null : group(linkRow('Message settings', '#/messages/settings',
                    { iconName: 'gear', sub: v.asName ? `Writing as ${v.asName}` : 'Write as your name, blocked names' })));
            }
        },
    };
}

/** One conversation: what was said, and the box to say more. */
function conversationScreen(raw) {
    watchMessages();
    let key = raw;
    try { key = decodeURIComponent(raw); } catch (_) {}
    const sub = h('p', { class: 'chat-sub' });
    const banners = h('div');
    const thread = h('div', { class: 'thread', role: 'log', 'aria-live': 'polite' });
    const box = h('textarea', { rows: 2, 'aria-label': 'Message' });
    const count = h('span', { class: 'muted small' });
    const send = h('button', { type: 'submit', class: 'btn primary' }, icon('up'), 'Send');
    const err = h('div', { class: 'err', role: 'alert' });
    const note = h('p', { class: 'note' });
    const tools = h('div');
    const changed = changes();
    let busy = false;
    let readAsked = -1;
    let seenLines = -1;
    const first = findChat(key);
    const title = first ? chatTitle(first) : key.startsWith('to:') ? `${key.slice(3)}.mds` : 'Conversation';

    const sendIt = async () => {
        const c = findChat(key);
        if (!c || busy) return;
        busy = true;
        send.disabled = true;
        err.textContent = '';
        try {
            if (c.id) await msgAsk('reply', { chat: c.id, text: box.value });
            else await msgAsk('send', { to: c.peer, text: box.value });
            box.value = '';
        } catch (e) {
            err.textContent = errOf(e);
        }
        busy = false;
        counted();
        box.focus();
    };
    const counted = () => {
        const n = bytesOf(box.value);
        count.textContent = n > MAX_MESSAGE ? `${n - MAX_MESSAGE} bytes too long` : n > MAX_MESSAGE - 100 ? `${MAX_MESSAGE - n} bytes left` : '';
        send.disabled = busy || !n || n > MAX_MESSAGE;
    };
    box.addEventListener('input', () => { err.textContent = ''; counted(); });
    box.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && !coarsePointer()) { e.preventDefault(); sendIt(); }
    });
    const form = h('form', { class: 'composer', onsubmit: (e) => { e.preventDefault(); sendIt(); } },
        box, h('div', { class: 'composer-bar' }, count, send), err, note);
    counted();

    const confirmThen = async (t, m, yes, op, after) => {
        const c = findChat(key);
        if (!c || !await confirmBox(t, m, yes)) return;
        try {
            const r = await msgAsk(op, { chat: c.id, name: c.peer });
            if (after) after(r);
        } catch (e) { toast(errOf(e), 6000); }
    };
    const blockIt = () => {
        const c = findChat(key);
        if (!c) return;
        const named = blocksName(c);
        confirmThen(named ? `Block ${c.peer}?` : 'Block this sender?',
            named ? `Its conversations are deleted, and nothing ${c.peer} sends reaches you until you unblock it in Message settings.`
                : 'This conversation is deleted and nothing more sent in it is read. A sender who does not prove a name can’t be told from other strangers: a new conversation from them would arrive as a new request.',
            'Block', 'block', (r) => { toast(r && r.blocked ? `${r.blocked} is blocked.` : 'Blocked.'); go('messages', true); });
    };

    const drawBanners = (c) => {
        const out = [];
        if (c.request) {
            out.push(h('div', { class: 'card' }, h('b', null, 'Message request'),
                h('p', { class: 'muted small' }, `This waits under Requests because ${requestReason(c)}. Accepting it, or replying, moves it to your inbox; they see that you read it only if you reply.`),
                h('div', { class: 'btns' },
                    h('button', { type: 'button', class: 'btn small primary', onclick: () => msgAsk('accept', { chat: c.id }).catch(e => toast(errOf(e))) }, 'Accept'),
                    h('button', { type: 'button', class: 'btn small', onclick: blockIt }, 'Block'),
                    h('button', { type: 'button', class: 'btn small', onclick: () => confirmThen('Delete this request?', 'Nothing more sent in it will be read here.', 'Delete', 'delete', () => go('messages/requests', true)) }, 'Delete'))));
        }
        if (c.peer && c.id && !c.verified) {
            out.push(h('div', { class: 'banner' }, h('span', { class: 'grow' }, `This sender claims to be ${c.peer}, but the claim did not check out against the registry. Your replies go to whoever opened this conversation, not to ${c.peer}.`),
                h('button', { type: 'button', class: 'btn small', onclick: () => { app.msgDraft = { to: c.peer, text: '' }; go('messages/new'); } }, `Write to the real ${c.peer}`)));
        }
        if (!c.id && c.note) out.push(h('div', { class: 'banner' }, h('span', { class: 'grow' }, c.note)));
        const stamping = c.lines.filter(l => l.stamping).length;
        if (stamping && c.id) {
            out.push(h('div', { class: 'banner' }, h('span', { class: 'grow' }, stamping === 1 ? `Stamping 1 message: ${work(c.stampBits)} on this device.` : `Stamping ${stamping} messages: ${work(c.stampBits)} each on this device.`),
                h('button', { type: 'button', class: 'btn small', onclick: () => msgAsk('unsend', { chat: c.id }).then(() => toast('Nothing of it went.'), e => toast(errOf(e))) }, 'Don’t send')));
        }
        if (c.lines.some(l => l.failed)) {
            out.push(h('div', { class: 'banner' }, h('span', { class: 'grow' }, 'A message didn’t go: the node wouldn’t take it, and it was tried for an hour.'),
                h('button', { type: 'button', class: 'btn small', onclick: () => msgAsk('retry', { chat: c.id }).catch(e => toast(errOf(e))) }, 'Try again')));
        }
        return out;
    };

    const drawLines = (c) => {
        const out = [];
        const now = Date.now() / 1000;
        let day = null;
        for (const l of c.lines) {
            const d = dayLabel(l.at);
            if (d !== day) { day = d; out.push(h('div', { class: 'day' }, d)); }
            if (l.timer != null) { out.push(h('div', { class: 'timer-note' }, timerNote(c, l))); continue; }
            const status = l.failed ? 'Not sent' : l.pending ? 'Waiting for their key' : l.stamping ? 'Stamping…' : l.sending ? 'Sending…'
                : l.expires ? `${clock(l.at)} · goes in ${about(Math.max(0, l.expires - now)).replace(/^about /, '')}` : clock(l.at);
            out.push(h('div', { class: 'bubble ' + (l.mine ? 'mine' : 'theirs') + (l.pending || l.stamping || l.sending ? ' waiting' : '') + (l.failed ? ' failed' : '') },
                h('div', { class: 'text' }, l.text), h('div', { class: 'meta' }, status)));
        }
        if (!c.lines.length) out.push(h('p', { class: 'muted small' }, 'Nothing here yet.'));
        return out;
    };

    const drawTools = (c) => {
        const rows = [];
        if (c.id) {
            rows.push(h('div', { class: 'row stacked' }, h('span', { class: 'label' }, 'Disappearing messages',
                h('small', null, 'What either of you sends from now on is deleted from both wallets once its time is up.')),
            seg(TIMER_CHOICES, c.ttl, ttl => msgAsk('timer', { chat: c.id, ttl }).catch(e => toast(errOf(e))), 'Disappearing messages')));
        }
        if (blocksName(c) && c.id) {
            rows.push(actionRow(`Pay ${c.peer}`, () => { app.draft = { to: c.peer, amount: '', all: false }; go('send'); }, { iconName: 'up', sub: 'Send MDS to their name' }));
        }
        if (!c.id) {
            rows.push(actionRow('Stop sending', () => confirmThen(`Stop sending to ${c.peer}?`, 'What waits for their key is dropped. Nothing of it has gone.', 'Stop sending', 'cancel', () => go('messages', true)), { iconName: 'left' }));
        } else {
            rows.push(actionRow(blocksName(c) ? `Block ${c.peer}` : 'Block this sender', blockIt, { iconName: 'shield' }));
            rows.push(actionRow('Delete conversation', () => confirmThen('Delete this conversation?', 'Nothing more sent in it will be read here, and what was said can’t be brought back.', 'Delete', 'delete', () => go('messages', true)), { iconName: 'box' }));
        }
        return group(rows);
    };

    return {
        title, back: ['Messages', '#/messages'], backAlways: true, leave: unwatchMessages,
        els: [sub, banners, thread, form, tools],
        update() {
            let c = findChat(key);
            if (!c && key.startsWith('to:')) {
                // The first message went: the conversation has its session now.
                const name = key.slice(3) + '.mds';
                const opened = book().chats.find(x => x.id && x.peer === name && x.verified && x.openedByMe);
                if (opened) { go('messages/c/' + encodeURIComponent(opened.key), true); return; }
            }
            if (!c) {
                if (changed('gone', [book().available])) {
                    fill(sub, book().available === false ? notBuilt() : book().available == null ? [spinner(), ' Opening your messages…'] : 'This conversation is gone.');
                    fill(banners); fill(thread); fill(tools);
                    form.hidden = true;
                }
                return;
            }
            form.hidden = false;
            const h1 = current && current.el.querySelector('h1');
            if (h1 && h1.textContent !== chatTitle(c)) h1.textContent = chatTitle(c);
            if (changed('sub', [chatSubtitle(c)])) sub.textContent = chatSubtitle(c);
            if (changed('banners', [c.request, c.verified, c.note, c.lines.filter(l => l.stamping).length, c.lines.some(l => l.failed), c.stampBits])) fill(banners, drawBanners(c));
            if (changed('lines', [c.lines, Math.floor(Date.now() / 60000)])) {
                fill(thread, drawLines(c));
                if (c.lines.length !== seenLines) {
                    seenLines = c.lines.length;
                    setTimeout(() => form.scrollIntoView({ block: 'end' }), 0);
                }
            }
            if (changed('tools', [c.id, c.ttl, c.peer, c.verified])) fill(tools, drawTools(c));
            box.placeholder = c.id ? (c.peer ? `Message ${c.peer}` : 'Reply to whoever opened this') : `Add to what waits for ${c.peer}`;
            note.textContent = c.stamping ? stampNote(c.peer || 'they', c.stampBits) : '';
            if (c.id && c.unread && readAsked !== c.lines.length) {
                readAsked = c.lines.length;
                msgAsk('read', { chat: c.id }).catch(() => {});
            }
        },
        focus: () => { if (!coarsePointer()) box.focus(); },
    };
}

/** A new conversation: who to write to, checked as it is typed. */
function composeScreen() {
    watchMessages();
    const d = app.msgDraft;
    const to = h('input', { type: 'text', value: d.to, placeholder: 'name.mds', autocomplete: 'off', autocapitalize: 'none', spellcheck: 'false', 'aria-label': 'To' });
    const toLine = h('div', { class: 'field-note' });
    const box = h('textarea', { rows: 5, placeholder: 'Message', 'aria-label': 'Message' });
    box.value = d.text || '';
    const count = h('span', { class: 'muted small' });
    const send = h('button', { type: 'submit', class: 'btn primary wide' }, 'Send');
    const err = h('div', { class: 'err', role: 'alert' });
    const note = h('p', { class: 'note' });
    let card = null;
    let gen = 0;
    let timer = null;
    let sending = false;

    const check = () => {
        const n = bytesOf(box.value);
        count.textContent = n > MAX_MESSAGE ? `${n - MAX_MESSAGE} bytes too long` : n > MAX_MESSAGE - 100 ? `${MAX_MESSAGE - n} bytes left` : '';
        send.disabled = sending || !(card && card.ok) || !n || n > MAX_MESSAGE;
        send.textContent = card && card.ok ? `Send to ${card.name}` : 'Send';
        const me = book().asName;
        note.textContent = card && card.ok
            ? `${stampNote(card.name, card.price)} ${me ? `Your first message is signed as ${me}, so they see who wrote.` : 'You write anonymously, so it may wait among their requests: write as your name in Message settings.'}`
            : 'Only their wallet can read what you send. Until they write back, they see at most three of your messages.';
    };
    const examine = () => {
        d.to = to.value;
        const raw = to.value.trim();
        const my = ++gen;
        clearTimeout(timer);
        card = null;
        if (!raw) { fill(toLine, h('span', { class: 'muted' }, 'Write to any name that takes sealed messages.')); check(); return; }
        let name;
        try { name = normalizeName(raw); } catch (e) { fill(toLine, h('span', { class: 'err' }, errOf(e))); check(); return; }
        fill(toLine, h('span', { class: 'muted' }, spinner(), ` Looking up ${name}.mds…`));
        check();
        timer = setTimeout(async () => {
            try {
                const r = await msgAsk('recipient', { name });
                if (my !== gen) return;
                card = r;
                if (!r.ok) {
                    fill(toLine, h('span', { class: 'err' }, r.says), r.blocked ? h('button', {
                        type: 'button', class: 'quiet-link', onclick: () => msgAsk('unblock', { name: r.name }).then(examine, e => toast(errOf(e))),
                    }, 'Unblock') : null);
                } else {
                    const open = book().chats.find(c => c.peer === r.name && c.verified && c.id);
                    fill(toLine, h('span', { class: 'ok-line' }, r.price > 24 ? `${r.name} takes sealed messages, and asks strangers more: ${priceLabel(r.price).toLowerCase()}.` : `${r.name} takes sealed messages.`),
                        open ? h('a', { class: 'quiet-link', href: chatHref(open) }, 'You already have a conversation with them') : null);
                }
            } catch (e) {
                if (my !== gen) return;
                fill(toLine, h('span', { class: 'err' }, nameProblem(errOf(e))));
            }
            check();
        }, 400);
    };
    to.addEventListener('input', () => { err.textContent = ''; examine(); });
    box.addEventListener('input', () => { d.text = box.value; err.textContent = ''; check(); });
    const form = h('form', {
        class: 'card', onsubmit: async (e) => {
            e.preventDefault();
            if (send.disabled || !card) return;
            sending = true;
            check();
            try {
                const r = await msgAsk('send', { to: card.name, text: box.value });
                app.msgDraft = { to: '', text: '' };
                go('messages/c/' + encodeURIComponent(r.key), true);
            } catch (e2) {
                err.textContent = errOf(e2);
            }
            sending = false;
            check();
        },
    },
    field('To', to, toLine),
    field('Message', box, h('div', { class: 'field-note' }, count)),
    note, err, send);
    examine();
    return {
        title: 'New message', back: ['Messages', '#/messages'], backAlways: true, leave: unwatchMessages,
        els: [book().available === false ? notBuilt() : form],
        focus: () => (d.to ? box : to).focus(),
        update: check,
    };
}

/** What strangers pay to write: the name's to set, so it is changed where
 *  the name is, here if this wallet holds it. */
function strangersRow(v, price) {
    const label = price === 24 ? 'Standard' : `${priceLabel(price)} (${price} bits)`;
    const held = v.asName ? findName(v.asName.replace(/\.mds$/, '')) : null;
    if (held && held.state !== 'lost') {
        return linkRow('Strangers pay', nameHref(held.name), { value: label, sub: `The stamp a stranger’s first messages must carry: what ${held.shown} asks. Change it there.` });
    }
    return infoRow('Strangers pay', label, { sub: v.asName ? `The stamp a stranger’s first messages must carry: what ${v.asName} asks, set by the wallet that holds it.` : 'The stamp a stranger’s first messages must carry. Write as a name to set your own price.' });
}

/** The name this wallet writes as, what strangers pay, who is blocked. */
function messageSettingsScreen() {
    watchMessages();
    const body = h('div');
    const changed = changes();
    const setName = async () => {
        let set = '';
        const v = book();
        const r = await dialog({
            title: 'Write as a name',
            message: 'A name of yours whose identity key is one of this wallet’s and whose kem record is this wallet’s message key. A name taken here, in Names, has both; so do names made in the desktop app with the same recovery phrase. Each new conversation signed as the name uses one of that key’s one-time signatures, and the desktop app keeps its own count of them: write as such a name from one of the two, not both.',
            input: { placeholder: 'name.mds', label: 'Your name' },
            buttons: [{ label: 'Cancel', value: false }, ...(v.asName ? [{ label: 'Write anonymously', value: 'clear' }] : []), { label: 'Write as it', value: true }],
            check: async (text) => {
                try { set = (await msgAsk('name', { name: text })).asName; } catch (e) { return errOf(e); }
                return '';
            },
        });
        if (r.value === 'clear') {
            await msgAsk('name', { name: '' }).then(() => toast('You write anonymously now.'), e => toast(errOf(e)));
        } else if (r.value === true && set) {
            toast(`You write as ${set} now.`);
        }
    };
    const draw = (v) => {
        const price = v.price || 24;
        return [
            group(
                actionRow('Write as', setName, {
                    iconName: 'key', value: v.asName || 'Anonymous',
                    sub: v.asName ? 'Your first message to someone is signed as this name, and your wallet hands its key to whoever asks the name.' : 'Write as your name, so people see who wrote.',
                }),
                strangersRow(v, price),
                controlRow('Drop anonymous senders', toggle(v.dropAnonymous, on => msgAsk('anonymous', { on }).catch(e => toast(errOf(e))), 'Drop anonymous senders'),
                    'Rather than keep their messages under Requests')),
            h('h2', null, 'Blocked'),
            v.blocked.length
                ? group(v.blocked.map(n => h('div', { class: 'row' }, h('span', { class: 'label' }, n),
                    h('button', { type: 'button', class: 'btn small', onclick: () => msgAsk('unblock', { name: n }).then(() => toast(`${n} is unblocked.`), e => toast(errOf(e))) }, 'Unblock'))))
                : h('p', { class: 'muted' }, 'No one is blocked.'),
            h('h2', null, 'How it works'),
            h('div', { class: 'card flat small muted' },
                h('p', null, 'A message is sealed to the key a name publishes (its kem record) with ML-KEM-512 and XChaCha20-Poly1305. Nodes see random-looking bytes from a fresh random sender each time, so nothing on the network says who wrote to whom.'),
                h('p', null, 'Writing to a name asks its wallet for that key first. Their wallet has to be online: your message waits up to 15 minutes for it.'),
                h('p', null, 'This wallet reads the network while it is open. Nodes keep the last 100 chats, so a message sent while this page is closed can be missed. The desktop app, running all the time, misses none.'),
                h('p', null, 'Disappearing messages are deleted from both wallets once their time is up. The sealed copy stays on the network as long as nodes keep it, readable by whoever holds the conversation: a timer is tidiness, not secrecy.')),
        ];
    };
    return {
        title: 'Message settings', back: ['Messages', '#/messages'], backAlways: true, leave: unwatchMessages,
        els: [body],
        update() {
            const v = book();
            if (v.available === false) { if (changed('body', 'none')) fill(body, notBuilt()); return; }
            if (changed('body', [v.asName, v.price, v.dropAnonymous, v.blocked, nbook().names.map(n => [n.name, n.state])])) fill(body, draw(v));
        },
    };
}

// ── the public room ─────────────────────────────────────────────────────────
// The chat bus as everyone on it sees it: up to ten of its 256 fixed words
// to a message, and addresses (room.js). The worker reads it along with
// sealed messages, and says things as a random name of this wallet's, tied
// to none of its keys or coins. Nicknames are this browser's own, kept
// where the classic page kept them.

const roomState = () => app.room || { messages: [], sender: null, read: 0, error: null };
const NICK_KEY = 'nick_';
const nickOf = sender => store.get(NICK_KEY + sender) || senderLabel(sender);

function roomRow(r) {
    const last = r.messages[r.messages.length - 1];
    const who = last ? (last.sender === r.sender ? 'You' : nickOf(last.sender)) : '';
    const said = last ? (textOf(last.words) || (last.attachments && last.attachments.length ? 'an address' : '')) : '';
    return h('a', { class: 'row link convo', href: '#/messages/public' },
        h('span', { class: 'tile', 'aria-hidden': 'true' }, icon('hash')),
        h('span', { class: 'label' },
            h('span', { class: 'convo-top' }, h('b', null, 'Public room'), h('span', { class: 'tag' }, 'everyone')),
            h('small', null, last ? `${who}: ${said}` : 'The whole network, in 256 words')),
        h('span', { class: 'convo-side' }, h('small', null, last ? lately(last.timestamp) : '')));
}

async function nameSender(sender) {
    const now = store.get(NICK_KEY + sender) || '';
    const r = await dialog({
        title: 'A name for this sender',
        message: `Only this browser sees it. Their id in the room is ${sender}.`,
        input: { placeholder: senderLabel(sender), label: 'Nickname', value: now, maxlength: 24 },
        buttons: [{ label: 'Cancel', value: false }, ...(now ? [{ label: 'Remove', value: 'clear' }] : []), { label: 'Save', value: true }],
    });
    if (r.value === 'clear' || (r.value === true && !String(r.text || '').trim())) {
        try { localStorage.removeItem(NICK_KEY + sender); } catch (_) {}
    } else if (r.value === true) {
        store.set(NICK_KEY + sender, String(r.text).trim().slice(0, 24));
    }
    update();
}

function publicRoomScreen() {
    watchMessages();
    const sub = h('p', { class: 'chat-sub' });
    const thread = h('div', { class: 'thread room', role: 'log', 'aria-live': 'polite' });
    const box = h('input', {
        type: 'text', value: app.roomDraft || '', autocomplete: 'off', autocapitalize: 'none', spellcheck: 'false',
        'aria-label': 'Message', placeholder: 'gm fren',
    });
    const picks = h('div', { class: 'picker', role: 'group', 'aria-label': 'Words that fit' });
    const said = h('div', { class: 'field-note' });
    const replyLine = h('div');
    const attachLine = h('div');
    const send = h('button', { type: 'submit', class: 'btn primary' }, icon('up'), 'Send');
    const err = h('div', { class: 'err', role: 'alert' });
    const changed = changes();
    let replyTo = null;     // { nonce, sender, text }
    let attach = false;     // this wallet's receive address goes with it
    let busy = false;
    let seen = -1;

    const myAddress = () => {
        const a = app.dash && app.dash.primaryAddress;
        return a && a !== 'None' ? String(a).toLowerCase() : null;
    };
    const lastPiece = () => { const m = /(\S*)$/.exec(box.value); return m ? m[1] : ''; };
    const put = (word) => {
        const v = box.value;
        const cut = v.length - lastPiece().length;
        box.value = v.slice(0, cut) + word + ' ';
        app.roomDraft = box.value;
        check();
        box.focus();
    };
    const wordBtn = w => h('button', { type: 'button', class: 'word', onclick: () => put(w) }, w);
    const check = () => {
        const d = parseDraft(box.value);
        const n = d.words.length;
        if (d.problem) fill(said, h('span', { class: 'err' }, d.problem));
        else fill(said, n ? `${n} of ${MAX_WORDS} words` : 'Type, or pick words below.');
        const piece = lastPiece();
        const fits = piece ? suggest(piece, 10) : [];
        fill(picks, fits.map(wordBtn));
        picks.hidden = !fits.length;
        send.disabled = busy || !!d.problem || (!n && !attach);
        fill(replyLine, replyTo ? h('div', { class: 'chip-line' },
            h('span', { class: 'grow' }, `Replying to ${replyTo.mine ? 'yourself' : nickOf(replyTo.sender)}${replyTo.text ? `: ${replyTo.text}` : ''}`),
            h('button', { type: 'button', class: 'quiet-link', onclick: () => { replyTo = null; check(); } }, 'Don’t reply')) : null);
        const addr = myAddress();
        fill(attachLine, addr ? h('label', { class: 'check-line' },
            h('input', { type: 'checkbox', checked: attach, onchange: (e) => { attach = e.target.checked; check(); } }),
            h('span', null, 'Add my receive address')) : null);
    };
    const sendIt = async () => {
        if (send.disabled) return;
        const d = parseDraft(box.value);
        busy = true;
        err.textContent = '';
        send.textContent = 'Working…';
        check();
        try {
            const addr = myAddress();
            await msgAsk('room-send', {
                words: d.words, replyTo: replyTo ? replyTo.nonce : null,
                attachments: attach && addr ? [{ kind: 'address', value: addr }] : [],
            });
            box.value = '';
            app.roomDraft = '';
            replyTo = null;
            attach = false;
        } catch (e) {
            err.textContent = errOf(e);
        }
        busy = false;
        fill(send, icon('up'), 'Send');
        check();
        if (!coarsePointer()) box.focus();
    };
    box.addEventListener('input', () => { app.roomDraft = box.value; err.textContent = ''; check(); });
    box.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); sendIt(); }
        else if (e.key === 'Tab' && !e.shiftKey && lastPiece() && !picks.hidden) {
            const first = picks.querySelector('button');
            if (first) { e.preventDefault(); put(first.textContent); }
        }
    });
    const all = h('details', { class: 'all-words' }, h('summary', null, 'All 256 words'),
        h('div', { class: 'picker all' }, DICTIONARY.map(wordBtn)));
    const form = h('form', { class: 'composer', onsubmit: (e) => { e.preventDefault(); sendIt(); } },
        replyLine,
        h('div', { class: 'composer-row' }, box, send),
        picks, said, attachLine, err, all);

    const attachmentEl = (a) => {
        if (a.kind !== 'address') {
            const v = String(a.value || '');
            return h('div', { class: 'attachment' }, h('small', null, a.kind.replace(/_/g, ' ')),
                h('span', { class: 'mono' }, v.length > 24 ? `${v.slice(0, 10)}…${v.slice(-10)}` : v));
        }
        return h('div', { class: 'attachment' }, h('small', null, 'Address'),
            h('span', { class: 'mono' }, groupHex(a.value)),
            h('span', { class: 'btns' },
                h('button', { type: 'button', class: 'btn small', onclick: () => copyText(a.value, 'Address copied') }, icon('copy'), 'Copy'),
                h('button', { type: 'button', class: 'btn small', onclick: () => { app.draft = { to: a.value, amount: '', all: false }; go('send'); } }, icon('up'), 'Pay')));
    };
    const drawThread = (list) => {
        const out = [];
        let day = null;
        for (const m of list) {
            const d = dayLabel(m.at);
            if (d !== day) { day = d; out.push(h('div', { class: 'day' }, d)); }
            const reply = m.replyTo ? h('div', { class: 'reply-to' },
                `↳ ${m.replyTo.sender ? (m.replyTo.mine ? 'you' : nickOf(m.replyTo.sender)) : 'an earlier message'}${m.replyTo.text ? `: ${m.replyTo.text}` : ''}`) : null;
            out.push(h('div', { class: 'bubble ' + (m.mine ? 'mine' : 'theirs') },
                m.mine ? null : h('button', { type: 'button', class: 'who', title: m.sender, onclick: () => nameSender(m.sender) }, nickOf(m.sender)),
                reply,
                m.text ? h('div', { class: 'text' }, m.text) : null,
                m.attachments.map(attachmentEl),
                h('div', { class: 'meta' }, clock(m.at), ' · ',
                    h('button', { type: 'button', class: 'quiet-link', onclick: () => { replyTo = { nonce: m.nonce, sender: m.sender, text: m.text, mine: m.mine }; check(); box.focus(); } }, 'Reply'))));
        }
        if (!list.length) out.push(h('p', { class: 'muted small' }, roomState().read ? 'Nothing said lately.' : [spinner(), ' Reading the room…']));
        return out;
    };
    check();
    msgAsk('room').then((r) => { app.room = r; update(); }, () => {});
    return {
        title: 'Public room', back: ['Messages', '#/messages'], backAlways: true, leave: unwatchMessages,
        els: [sub, thread, form,
            h('div', { class: 'card flat small muted' },
                h('p', null, 'Everyone on the network reads the room: nothing here is private. Wallets and nodes show it in the same 256 words, so no one can post text there is no word for.'),
                h('p', null, 'You speak here as a random id this wallet keeps, tied to none of your coins or names. Each message carries a little proof of work, done on this device, so the room can’t be flooded for free.'))],
        update() {
            const r = roomState();
            if (changed('sub', [r.error, !!r.read])) sub.textContent = r.error || 'Everyone on the network, in 256 fixed words, ten at most to a message.';
            const list = roomView(r.messages, r.sender);
            const nicks = list.map(m => store.get(NICK_KEY + m.sender) || '');
            if (changed('thread', [list.map(m => m.key), nicks, !!r.read, Math.floor(Date.now() / 60000)])) {
                fill(thread, drawThread(list));
                if (list.length !== seen) {
                    seen = list.length;
                    setTimeout(() => form.scrollIntoView({ block: 'end' }), 0);
                }
            }
            if (changed('attach', [myAddress()])) check();
        },
        focus: () => { if (!coarsePointer()) box.focus(); },
    };
}

// ── names ───────────────────────────────────────────────────────────────────
// A name.mds this wallet holds itself, with no desktop app: registered (or
// claimed, once lapsed), renewed and changed here, held by a key only this
// wallet has (registrar.js, in the worker). What a name costs is burned:
// it goes to no one, and doesn't come back.

const nbook = () => app.names || { available: null, names: [], busy: null, sealed: false };
const namesAsk = (op, args = {}) => ask('NAMES', { op, ...args });
const TIER_NAMES = ['Platinum', 'Gold', 'Silver', 'Bronze'];
const RATES = [2 ** 36, 2 ** 33, 2 ** 30, 2 ** 27];   // names::RATES: a year of each tier
const YEAR_CHOICES = [1, 2, 3, 5, 10];
const yearsSeg = (current, onpick) => h('div', { class: 'seg-line' },
    seg(YEAR_CHOICES.map(y => [y, String(y)]), current, onpick, 'Years'), h('span', { class: 'muted small' }, current === 1 ? 'year' : 'years'));
const findName = n => nbook().names.find(x => x.name === n) || null;
const nameHref = n => '#/names/n/' + encodeURIComponent(n);
const yearsText = y => `${y} year${y === 1 ? '' : 's'}`;
const longDate = d => d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
/** When block `at` comes, from block `tip`, at a block a minute. */
const blockDate = (at, tip) => new Date(Date.now() + (at - tip) * 60_000);
const nameBusy = n => !!(n && n.job && n.job.phase !== 'failed');
/** A labelled block for buttons (a label would pass its clicks to them). */
const block = (label, el, hint) => h('div', { class: 'field' }, h('span', null, label), el, hint || null);

/** What is so of a name of this wallet's, in a line. */
function nameLine(n) {
    const j = n.job;
    if (j && j.phase === 'failed') return `Didn’t go through: ${j.error}`;
    if (j) return j.say || 'Waiting to start…';
    return standingLine(n);
}

/** How long a name holds, whatever is being done to it. */
function standingLine(n) {
    if (n.state === 'new') return 'Not yours yet';
    if (n.state === 'lost') return 'Someone else holds it now';
    const when = longDate(blockDate(n.expires, n.height));
    if (n.state === 'lapsed') return `Lapsed on ${when}: renew it before someone else takes it`;
    if (n.state === 'lapsing') return `Lapses on ${when}: renew it`;
    return `Yours until ${when}`;
}

function namesNotBuilt() {
    return h('div', { class: 'card' }, h('p', null, 'Names aren’t in this build of the wallet.'),
        h('p', { class: 'muted small' }, 'Its WebAssembly predates them: reload once the page updates.'));
}

async function adoptName(name) {
    toast(`Looking for the key that holds ${name}.mds…`);
    try {
        const r = await namesAsk('adopt', { name });
        toast(`${r.name} is back in this wallet.`);
        go('names/n/' + encodeURIComponent(name));
    } catch (e) {
        toast(errOf(e), 7000);
    }
}

async function adoptPrompt() {
    const r = await dialog({
        title: 'Find a name this wallet holds',
        message: 'A name taken in the desktop app, or before this wallet was restored from its 24 words, is held by one of the keys those words make. The wallet looks for it among its keys, and makes the next few if need be: about a minute each in a browser.',
        input: { placeholder: 'name.mds', label: 'Name' },
        buttons: [{ label: 'Cancel', value: false }, { label: 'Look for it', value: true }],
        check: (text) => { try { normalizeName(text); return ''; } catch (e) { return errOf(e); } },
    });
    if (r.value === true) adoptName(normalizeName(r.text));
}

/** What a look-up found, and what can be done about it. */
function lookLine(r) {
    const take = label => h('a', { class: 'btn small primary', href: '#/names/get/' + encodeURIComponent(r.name) }, label);
    const price = `${TIER_NAMES[r.tier]}: ${withUnit(r.rate)} a year`;
    if (r.mine) return [h('span', { class: 'ok-line' }, `${r.shown} is yours.`), ' ', h('a', { class: 'quiet-link', href: nameHref(r.name) }, 'Open it')];
    if (r.state === 'available') return [h('span', { class: 'ok-line' }, `${r.shown} is free. ${price}.`), h('div', { class: 'btns' }, take('Take it'))];
    if (r.state === 'unclaimed') return [h('span', { class: 'ok-line' }, `${r.shown} is a premium name no one has claimed. ${price}.`), h('div', { class: 'btns' }, take('Claim it'))];
    if (r.state === 'lapsed') return [h('span', { class: 'ok-line' }, `${r.shown} lapsed, so anyone can take it. ${price}.`), h('div', { class: 'btns' }, take('Take it'))];
    const until = longDate(blockDate(r.expires, r.height));
    return [h('span', { class: 'muted' }, `${r.shown} is taken until ${until}${r.deal === 'sale' ? ', and its holder is selling it' : ''}.`), ' ',
        h('button', { type: 'button', class: 'quiet-link', onclick: () => adoptName(r.name) }, 'It’s mine: look for its key')];
}

function namesScreen() {
    const busy = h('div');
    const list = h('div');
    const finder = h('div');
    const foot = h('div');
    const changed = changes();
    const input = h('input', { type: 'text', placeholder: 'yourname', autocomplete: 'off', autocapitalize: 'none', spellcheck: 'false', 'aria-label': 'Name', maxlength: 36 });
    const line = h('div', { class: 'field-note', 'aria-live': 'polite' });
    let gen = 0;
    let timer = null;
    const examine = () => {
        const raw = input.value.trim();
        const my = ++gen;
        clearTimeout(timer);
        if (!raw) { fill(line, h('span', { class: 'muted' }, 'Letters, numbers and -, up to 32. The shorter the name, the more a year of it costs.')); return; }
        let name;
        try { name = normalizeName(raw); } catch (e) { fill(line, h('span', { class: 'err' }, errOf(e))); return; }
        fill(line, h('span', { class: 'muted' }, spinner(), ` Looking up ${name}.mds…`));
        timer = setTimeout(async () => {
            try {
                const r = await namesAsk('look', { name });
                if (my === gen) fill(line, lookLine(r));
            } catch (e) {
                if (my === gen) fill(line, h('span', { class: 'err' }, nameProblem(errOf(e))));
            }
        }, 400);
    };
    input.addEventListener('input', examine);
    input.addEventListener('keydown', (e) => {
        if (e.key !== 'Enter') return;
        e.preventDefault();
        const a = line.querySelector('a.btn');
        if (a) a.click();
    });
    fill(finder, h('div', { class: 'card' }, h('h2', null, 'Find a name'),
        h('div', { class: 'input-unit' }, input, h('span', null, '.mds')), line));
    examine();
    return {
        title: 'Names', back: ['Home', '#/'],
        els: [busy, list, finder, foot],
        update() {
            const v = nbook();
            if (changed('busy', [v.busy && v.busy.say])) fill(busy, v.busy ? h('div', { class: 'banner' }, spinner(), h('span', { class: 'grow' }, v.busy.say)) : null);
            if (changed('list', [v.available, v.names.map(n => [n.name, nameLine(n), n.state, n.tier])])) {
                const mine = v.names.filter(n => n.state !== 'new' || n.job);
                if (v.available === false) fill(list, namesNotBuilt());
                else if (v.available == null) fill(list, h('p', { class: 'muted' }, spinner(), ' Opening your names…'));
                else if (mine.length) {
                    fill(list, h('h2', { class: 'list-head' }, 'Your names'), group(mine.map(n => linkRow(n.shown, nameHref(n.name), {
                        iconName: 'tag', sub: nameLine(n), value: nameBusy(n) ? '…' : n.state === 'lost' ? null : TIER_NAMES[n.tier],
                    }))));
                } else {
                    fill(list, h('div', { class: 'card flat empty' }, h('b', null, 'No names yet.'),
                        'A name.mds is yours for as long as you renew it. People pay you by it, and write to you with sealed messages, instead of copying a long address.'));
                }
            }
            if (changed('foot', [v.available])) {
                fill(foot, v.available ? group(
                    actionRow('Find a name this wallet holds', adoptPrompt, { iconName: 'key', sub: 'For a wallet restored from its 24 words' }),
                    linkRow('How names work', '#/names/about', { iconName: 'info' })) : null);
            }
            finder.hidden = v.available === false;
        },
        focus: () => { if (!nbook().names.length && !coarsePointer()) input.focus(); },
    };
}

function nameAboutScreen() {
    return {
        title: 'How names work', back: ['Names', '#/names'], backAlways: true,
        els: [h('div', { class: 'card flat' },
            h('p', null, 'The registry of names (MSN-1) is one coin on the Midstate chain, kept by a covenant every node runs. A name is held for the years paid, and can be renewed for more before it lapses; once it lapses, anyone can take it.'),
            h('p', null, 'A year of a name burns its tier’s price: 64 gMDS for one or two characters and the premium names, 8 for three, 1 for four, ⅛ for five or more. The burn goes to no one, and doesn’t come back if the name lapses. Each change also pays a small network fee.'),
            h('p', null, 'A name is held by one of this wallet’s keys, made from your 24 words as the desktop app makes its own, so the same words manage the same names in either app. Making a name’s key takes about a minute in a browser. A wallet restored from its words finds its names when asked (Find a name this wallet holds).'),
            h('p', null, 'What a name says is public: where payments to it go, its sealed-message key, a line of text. Anyone can read it.'),
            h('p', null, 'Each change is a registry spend in two blocks: a commit, then the change itself. If someone else changes the registry in between, the wallet tries again at the next block, by itself.'))],
    };
}

/** Taking a name: for how long, what it says, and what it costs. */
function nameGetScreen(raw) {
    let name;
    try { name = normalizeName(decodeURIComponent(raw)); } catch (e) {
        return { title: 'Take a name', back: ['Names', '#/names'], backAlways: true, els: [h('p', { class: 'err' }, errOf(e))] };
    }
    const shownName = `${name}.mds`;
    const status = h('div');
    const yearsEl = h('div');
    const priceEl = h('div');
    const text = h('input', { type: 'text', maxlength: 120, placeholder: 'Optional', 'aria-label': 'What it says', autocomplete: 'off' });
    const err = h('div', { class: 'err', role: 'alert' });
    const go_ = h('button', { type: 'submit', class: 'btn primary wide', disabled: true }, 'Take it');
    const sums = h('div', { class: 'note' });
    let look = null;
    let years = 1;
    let postage = 24;
    let sending = false;
    const cost = () => (look ? RATES[look.tier] * years : 0);
    const draw = () => {
        const v = nbook();
        fill(yearsEl, yearsSeg(years, (y) => { years = y; draw(); }));
        fill(priceEl, v.sealed ? seg(PRICES.map(([b, label]) => [b, label]), postage, (b) => { postage = b; draw(); }, 'Strangers’ messages') : null);
        if (!look) { go_.disabled = true; return; }
        const c = cost();
        const bal = app.dash ? app.dash.balance : null;
        const short = bal != null && bal < c;
        const until = longDate(blockDate(look.height + years * 525_600, look.height));
        fill(sums,
            h('p', null, `${withUnit(c)} for ${yearsText(years)}, held until about ${until}, plus a small network fee. The ${withUnit(c)} is burned: it goes to no one.`),
            short ? h('p', { class: 'err' }, `This wallet has ${withUnit(bal)}.`) : null);
        go_.disabled = sending || short || !['available', 'unclaimed', 'lapsed'].includes(look.state);
        go_.textContent = sending ? ((v.busy && v.busy.say) || 'Starting…') : `${look.state === 'available' ? 'Register' : 'Claim'} ${shownName} for ${withUnit(c)}`;
    };
    const ready = (r) => {
        look = r;
        if (r.mine) { go('names/n/' + encodeURIComponent(name), true); return; }
        const price = `${TIER_NAMES[r.tier]}: ${withUnit(r.rate)} a year.`;
        const says = {
            available: `${shownName} is free. ${price}`,
            unclaimed: `${shownName} is a premium name no one has claimed. ${price}`,
            lapsed: `${shownName} lapsed, so anyone can take it. ${price}`,
            held: `${shownName} is taken.`,
        }[r.state] || '';
        fill(status, h('p', { class: (r.state === 'held' ? 'err' : 'ok-line') + ' first-line' }, says));
        draw();
    };
    namesAsk('look', { name }).then(ready, e => fill(status, h('p', { class: 'err' }, nameProblem(errOf(e)))));
    fill(status, h('p', { class: 'muted first-line' }, spinner(), ` Looking up ${shownName}…`));
    const form = h('form', {
        class: 'card', onsubmit: async (e) => {
            e.preventDefault();
            if (go_.disabled || !look) return;
            if (new TextEncoder().encode(text.value.trim()).length > 255) { err.textContent = 'What it says is 255 bytes at most.'; return; }
            const c = cost();
            if (!await confirmBox(`Take ${shownName}?`,
                `${withUnit(c)} is burned for ${yearsText(years)}, and a small fee paid. It takes two blocks, a few minutes: you can leave this page meanwhile. If someone else takes the name first, nothing is burned.`,
                `${look.state === 'available' ? 'Register' : 'Claim'} it`)) return;
            sending = true;
            err.textContent = '';
            draw();
            try {
                await namesAsk('get', { name, years, text: text.value.trim(), postage });
                go('names/n/' + encodeURIComponent(name), true);
            } catch (e2) {
                err.textContent = errOf(e2);
            }
            sending = false;
            draw();
        },
    },
    status,
    block('For', yearsEl),
    field('What it says', text, h('div', { class: 'field-note' }, 'A line anyone who looks the name up sees. Leave it empty if you like.')),
    nbook().sealed ? block('What strangers’ messages cost', priceEl, h('div', { class: 'field-note' }, 'A stranger’s first sealed messages to you carry this much of their device’s work. Standard is a few seconds; raise it if you get spam.')) : null,
    h('p', { class: 'muted small' }, `Payments to ${shownName} come to this wallet’s receive address${nbook().sealed ? ', and sealed messages to it reach this wallet' : ''}.`),
    sums, err, go_);
    return {
        title: `Take ${shownName}`, back: ['Names', '#/names'], backAlways: true,
        els: [form],
        update: draw,
    };
}

/** A name of this wallet's: how long it holds, what it says, its changes. */
function nameScreen(raw) {
    let name = raw;
    try { name = normalizeName(decodeURIComponent(raw)); } catch (_) {}
    const shownName = `${name}.mds`;
    const head = h('div');
    const renewEl = h('div');
    const saysEl = h('div');
    const keyEl = h('div');
    const changed = changes();
    let years = 1;
    const doIt = async (op, args, done) => {
        try {
            await namesAsk(op, { name, ...args });
            if (done) toast(done);
        } catch (e) {
            toast(errOf(e), 7000);
        }
    };
    const editText = async (n) => {
        const r = await dialog({
            title: 'What it says',
            message: 'A line anyone who looks the name up sees. Changing it is a registry spend: a small fee, and two blocks.',
            input: { placeholder: 'Nothing', label: 'What it says', value: (n.records && n.records.text) || '', maxlength: 120 },
            buttons: [{ label: 'Cancel', value: false }, { label: 'Change it', value: true }],
            check: (t) => (new TextEncoder().encode(String(t || '').trim()).length > 255 ? 'That is over 255 bytes.' : ''),
        });
        if (r.value === true) doIt('edit', { text: String(r.text || '').trim() }, 'Changing what it says…');
    };
    const setPostage = async (bits) => {
        if (!await confirmBox(`${priceLabel(bits)} for strangers?`, `A stranger’s first sealed messages to ${shownName} will carry ${priceLabel(bits).toLowerCase()} work (${bits} bits). Changing it is a registry spend: a small fee, and two blocks.`, 'Change it')) {
            const n = findName(name);
            if (n) fill(saysEl, drawSays(n));
            return;
        }
        doIt('edit', { postage: bits }, 'Changing the price…');
    };
    const drawHead = (n) => {
        const j = n.job;
        const out = [h('p', { class: 'muted' }, `${TIER_NAMES[n.tier] || ''}${n.tier != null ? ' · ' : ''}${standingLine(n)}`)];
        if (j && j.phase !== 'failed') {
            out.push(h('div', { class: 'under-way' }, h('b', null, { register: 'Registering', claim: 'Claiming', renew: 'Renewing', edit: 'Changing it' }[j.op]),
                h('span', null, spinner(), ' ', j.say || 'Waiting to start…')));
            if (j.cancellable && ['queued', 'waiting'].includes(j.phase)) {
                out.push(h('div', { class: 'btns' }, h('button', { type: 'button', class: 'btn small', onclick: () => doIt('cancel', {}, 'Stopped. Nothing was signed.') }, 'Stop')));
            }
        } else if (j) {
            out.push(h('div', { class: 'btns' },
                h('button', { type: 'button', class: 'btn small primary', onclick: () => doIt('retry') }, 'Try again'),
                h('button', { type: 'button', class: 'btn small', onclick: () => doIt('cancel') }, 'Put it away')));
        }
        if (n.state === 'lost') {
            out.push(h('p', null, 'It lapsed, and someone else took it. This wallet’s key holds it no more.'),
                h('div', { class: 'btns' }, h('button', { type: 'button', class: 'btn small', onclick: () => doIt('forget').then(() => go('names', true)) }, 'Forget it')));
        }
        return h('div', { class: 'card' }, out);
    };
    const drawRenew = (n) => {
        if (n.state === 'new' || n.state === 'lost' || nameBusy(n)) return null;
        const c = RATES[n.tier] * years;
        const from = Math.max(n.expires, n.height);
        const until = longDate(blockDate(from + years * 525_600, n.height));
        return h('div', { class: 'card' }, h('h2', null, 'Renew'),
            yearsSeg(years, (y) => { years = y; fill(renewEl, drawRenew(findName(name))); }),
            h('p', { class: 'note' }, `Held until about ${until}. ${withUnit(c)} is burned, and a small fee paid.`),
            h('button', {
                type: 'button', class: 'btn primary', onclick: async () => {
                    if (!await confirmBox(`Renew ${shownName}?`, `${withUnit(c)} is burned for ${yearsText(years)} more, and a small fee paid. It takes two blocks.`, 'Renew')) return;
                    doIt('renew', { years }, 'Renewing…');
                },
            }, `Renew for ${withUnit(c)}`));
    };
    const drawSays = (n) => {
        const r = n.records;
        if (!r || n.state === 'lost') return null;
        const off = nameBusy(n) || n.state === 'new';
        const rows = [];
        const pays = r.pays;
        if (pays && pays.current) rows.push(infoRow('Payments', 'To this wallet', { sub: 'Its receive address' }));
        else {
            const what = !pays ? 'It takes no payments' : pays.mine ? 'To an older address of this wallet’s' : 'To an address of another wallet’s';
            rows.push(off ? infoRow('Payments', what) : actionRow('Payments', () => doIt('edit', { pay: true }, 'Pointing payments at this wallet…'), { value: what, sub: 'Point them at this wallet’s receive address' }));
        }
        if (nbook().sealed) {
            const here = r.kem && r.kem.mine && r.identity && r.identity.mine;
            if (here) rows.push(infoRow('Sealed messages', 'Reach this wallet'));
            else rows.push(off ? infoRow('Sealed messages', r.kem ? 'Go to another wallet' : 'Not taken') : actionRow('Sealed messages', () => doIt('edit', { messages: true }, 'Taking sealed messages here…'), { value: r.kem ? 'Go to another wallet' : 'Not taken', sub: 'Take them in this wallet' }));
            if (here) {
                rows.push(h('div', { class: 'row stacked' }, h('span', { class: 'label' }, 'What strangers’ messages cost',
                    h('small', null, 'The work a stranger’s first sealed messages carry. Raise it if you get spam.')),
                off ? h('span', { class: 'value' }, priceLabel(r.postage)) : seg(PRICES.map(([b, label]) => [b, label]), r.postage, setPostage, 'What strangers’ messages cost')));
                const asName = book().asName;
                rows.push(asName === shownName ? infoRow('Write as', 'This name', { sub: 'Your first message to someone is signed as it' })
                    : actionRow('Write as', () => msgAsk('name', { name }).then(() => toast(`You write as ${shownName} now.`), e => toast(errOf(e), 7000)), { value: asName || 'Anonymous', sub: `Sign new conversations as ${shownName}` }));
            }
        }
        rows.push(off ? infoRow('What it says', r.text || 'Nothing') : actionRow('What it says', () => editText(n), { value: r.text || 'Nothing' }));
        if (r.other) rows.push(textRow(h('span', { class: 'muted small' }, `${r.other} more record${r.other === 1 ? '' : 's'} this wallet doesn’t read, kept as ${r.other === 1 ? 'it is' : 'they are'}.`)));
        return [h('h2', null, 'What it says'), group(rows), h('p', { class: 'muted small' }, 'Anyone can read what a name says. Each change is a registry spend: a small fee, and two blocks.')];
    };
    const drawKey = (n) => {
        if (n.signatures == null || n.state === 'lost') return null;
        return group(infoRow('Its key', `${n.signatures} signatures left`, { sub: 'Each change of the name signs once. Near the last, the change hands the name to a new key of this wallet’s.' }));
    };
    return {
        title: shownName, back: ['Names', '#/names'], backAlways: true,
        els: [head, renewEl, saysEl, keyEl],
        update() {
            const n = findName(name);
            if (!n) {
                if (changed('gone', [nbook().available])) {
                    fill(head, nbook().available === false ? namesNotBuilt() : nbook().available == null ? h('p', { class: 'muted' }, spinner(), ' Opening your names…')
                        : h('div', { class: 'card' }, h('p', null, `This wallet doesn’t hold ${shownName}.`),
                            h('div', { class: 'btns' }, h('a', { class: 'btn small', href: '#/names/get/' + encodeURIComponent(name) }, 'Look it up'),
                                h('button', { type: 'button', class: 'btn small', onclick: () => adoptName(name) }, 'It’s mine: look for its key'))));
                    fill(renewEl); fill(saysEl); fill(keyEl);
                }
                return;
            }
            changed('gone', null);
            const busyKey = [n.job && [n.job.phase, n.job.say, n.job.error], n.state];
            if (changed('head', [busyKey, n.expires, n.tier, Math.floor(n.height / 1440)])) fill(head, drawHead(n));
            if (changed('renew', [busyKey, n.expires, n.tier])) fill(renewEl, drawRenew(n));
            if (changed('says', [busyKey, n.records, nbook().sealed, book().asName])) fill(saysEl, drawSays(n));
            if (changed('key', [n.signatures, n.state])) fill(keyEl, drawKey(n));
        },
    };
}

function settingsScreen() {
    const p = positions();
    return {
        title: 'Settings', back: ['Home', '#/'],
        els: [
            group(
                linkRow('Backup', '#/settings/backup', { iconName: 'shield', sub: 'Your 24 words and backup files' }),
                h('div', { class: 'row' }, icon('contrast'), h('span', { class: 'label' }, 'Theme'),
                    seg([['dark', 'Dark'], ['light', 'Light']], currentTheme(), setTheme, 'Theme')),
                linkRow('Network', '#/settings/network', { iconName: 'globe', value: net.summary().text }),
                book().available !== false ? linkRow('Messages', '#/messages/settings', { iconName: 'chat', sub: 'Write as your name, blocked names', value: book().asName || null }) : null,
                linkRow('Wallet care', '#/settings/care', { iconName: 'coins', sub: 'Merge small coins, gather coins, check the chain again' }),
                p.total ? linkRow('Classic wallet', 'classic.html', { iconName: 'box', sub: `${positionsText(p)} still open there`, value: String(p.total) }) : null,
                linkRow('Advanced', '#/settings/advanced', { iconName: 'list', sub: 'The wallet’s log' }),
                linkRow('About', '#/settings/about', { iconName: 'info' })),
            group(actionRow('Lock', lock, { iconName: 'lock' })),
        ],
    };
}

function backupScreen() {
    const words = h('div');
    const showWords = async () => {
        let phrase = null;
        const r = await dialog({
            title: 'Show your 24 words',
            message: 'Anyone who sees them can take your money. Make sure no one is watching your screen.',
            input: { type: 'password', placeholder: 'Password', label: 'Password', autocomplete: 'current-password' },
            buttons: [{ label: 'Cancel', value: false }, { label: 'Show', value: true }],
            check: async (text) => {
                if (!text) return 'Type your password.';
                try { phrase = await revealWords(text); } catch (e) { return errOf(e); }
                return '';
            },
        });
        if (!r.value || !phrase) return;
        const list = phrase.trim().split(/\s+/);
        fill(words,
            h('div', { class: 'seed-grid', translate: 'no' }, list.map((w, i) => h('div', null, h('i', null, `${i + 1}.`), w))),
            h('button', { class: 'btn small', type: 'button', onclick: () => fill(words) }, 'Hide them'));
    };
    return {
        title: 'Backup', back: ['Settings', '#/settings'], backAlways: true,
        els: [
            h('p', { class: 'lead' }, 'Two ways back into this wallet if this browser loses it.'),
            h('div', { class: 'card' },
                h('h2', null, 'Your 24 words'),
                h('p', { class: 'muted' }, 'The words you wrote down when you made this wallet. With them you can restore it in any browser, or in the desktop app.'),
                h('button', { class: 'btn', type: 'button', onclick: showWords }, icon('key'), 'Show my 24 words'),
                words),
            h('div', { class: 'card' },
                h('h2', null, 'A backup file'),
                h('p', { class: 'muted' }, 'An encrypted copy of this wallet, with any open trades. It opens with this wallet’s password.'),
                h('button', { class: 'btn', type: 'button', onclick: () => { post({ type: 'EXPORT_BACKUP' }); toast('Preparing the backup…'); } }, icon('download'), 'Save a backup file')),
        ],
        leave() { fill(words); },
    };
}

function networkScreen(open) {
    const info = h('div');
    const custom = net.customNode();
    const addr = h('input', { type: 'text', value: custom.addr, placeholder: '/ip4/…/udp/9335/webrtc-direct/certhash/…', autocomplete: 'off', spellcheck: 'false', autocapitalize: 'none' });
    const rpc = h('input', { type: 'text', value: custom.rpc, placeholder: 'https://…', autocomplete: 'off', spellcheck: 'false', autocapitalize: 'none' });
    const err = h('div', { class: 'err', role: 'alert' });
    const changed = changes();
    const apply = (a, r) => {
        if (a && !/^\/(ip4|ip6|dns4|dns6|dns)\/.+\/webrtc-direct\/certhash\//.test(a)) {
            err.textContent = 'That isn’t a WebRTC address. It looks like /ip4/…/udp/…/webrtc-direct/certhash/…';
            return;
        }
        if (r && !/^https?:\/\/\S+$/i.test(r)) { err.textContent = 'The RPC address starts with https://'; return; }
        err.textContent = '';
        toast(a ? 'Connecting to your node…' : 'Using the public nodes.');
        net.useNode(a, r).then(() => { if (open) post({ type: 'SCAN' }); }).catch(() => {});
        update();
    };
    const form = h('form', { onsubmit: (e) => { e.preventDefault(); apply(addr.value.trim(), rpc.value.trim()); } },
        field('WebRTC address', addr), field('RPC address (optional)', rpc), err,
        h('div', { class: 'btns' },
            h('button', { class: 'btn', type: 'button', onclick: () => { addr.value = ''; rpc.value = ''; apply('', ''); } }, 'Use the public nodes'),
            h('button', { class: 'btn primary', type: 'submit' }, 'Connect')));
    return {
        title: 'Network', bare: !open, form: !open,
        back: open ? ['Settings', '#/settings'] : ['Back', '#/'], backAlways: true,
        els: [
            info,
            h('h2', null, 'Your own node'),
            h('p', { class: 'muted' }, 'Connect to a node you run. Leave both empty to use the public nodes.'),
            form,
        ],
        update() {
            const d = net.details();
            const s = net.summary();
            const dash = app.dash;
            if (!changed('i', [d, s, open && dash && [dash.networkHeight, dash.lastScannedHeight, dash.mempoolSize]])) return;
            const route = d.webrtc === 'connected' && d.healthy ? 'Straight to a node (WebRTC)' : `Through ${hostOf(d.rpc)} (HTTPS)`;
            fill(info, open ? [
                group(
                    infoRow('Status', h('span', null, h('span', { class: 'dot ' + s.cls }), ' ', s.text)),
                    infoRow('Route', route),
                    d.peers ? infoRow('Nodes known', String(d.peers)) : null,
                    dash ? infoRow('Chain height', (dash.networkHeight || 0).toLocaleString()) : null,
                    dash ? infoRow('Wallet has read to', (dash.lastScannedHeight || 0).toLocaleString()) : null,
                    dash ? infoRow('Payments waiting for a block', (dash.mempoolSize || 0).toLocaleString()) : null),
                h('div', { class: 'btns' }, h('button', { class: 'btn small', type: 'button', onclick: () => { net.connect().then(() => post({ type: 'SCAN' })).catch(() => {}); toast('Reconnecting…'); } }, icon('rescan'), 'Reconnect')),
            ] : h('p', { class: 'lead' }, 'The wallet connects once it’s open. Choose your own node here first if you like.'));
        },
    };
}

function careScreen() {
    const merges = h('div');
    const coins = h('p', { class: 'small' });
    const tidyBtn = h('button', { class: 'btn', type: 'button' }, icon('coins'), 'Gather them');
    const rescanBtn = h('button', { class: 'btn', type: 'button' }, icon('rescan'), 'Check again');
    const spendable = () => (app.dash ? app.dash.utxos.filter(u => !u.watch_only && !u.reuseLocked) : []);
    const addresses = () => new Set(spendable().map(u => u.address)).size;
    const changed = changes();
    rescanBtn.addEventListener('click', async () => {
        if (!await confirmBox('Check the whole chain again?', 'The wallet reads every block again to rebuild your balance and activity. It can take a few minutes, and sending waits until it’s done.', 'Check again')) return;
        app.scan = null;
        post({ type: 'RESCAN' });
        go('');
    });
    tidyBtn.addEventListener('click', async () => {
        if (!await confirmBox('Gather your coins?', `This moves the coins on your ${addresses()} addresses onto your main one, with a payment to yourself, for a small network fee. It takes a couple of blocks.`, 'Gather them')) return;
        app.op = { kind: 'tidy', words: 'Preparing', frac: 0.05, done: false, error: '', seen: false };
        post({ type: 'DEFRAG_UTXOS' });
        go('send/progress');
    });
    // Merging one address's coins (a consolidate transaction): the worker
    // says what it would make of them, and the fee, before anything is signed.
    const merge = async (a, btn) => {
        btn.disabled = true;
        let q;
        try {
            q = await ask('QUOTE_CONSOLIDATE', { address: a.address });
        } catch (e) {
            btn.disabled = false;
            const t = errOf(e);
            okBox('Can’t merge these coins', /too low to pay/i.test(t) ? 'Together they’re worth less than the network fee to merge them.' : t);
            return;
        }
        btn.disabled = false;
        const where = a.main ? 'your address' : `an older address (${shortAddr(a.address)})`;
        const rest = q.left ? ` That’s the first ${q.coins.toLocaleString()}: merge again for the other ${q.left.toLocaleString()}.` : '';
        if (!await confirmBox('Merge these coins?',
            `The ${q.coins.toLocaleString()} coins on ${where} (${exactUnit(q.value)}) become ${q.outputs} on your main address. `
            + `The network fee is ${exactUnit(q.fee)}. It takes a couple of blocks.${rest}`, 'Merge them')) return;
        app.op = { kind: 'merge', words: 'Preparing', frac: 0.05, done: false, error: '', seen: false };
        post({ type: 'CONSOLIDATE_UTXOS', payload: { address: a.address } });
        go('send/progress');
    };
    return {
        title: 'Wallet care', back: ['Settings', '#/settings'], backAlways: true,
        els: [
            h('div', { class: 'card' },
                h('h2', null, 'Merge small coins ', h('span', { class: 'muted small' }, '(consolidate)')),
                h('p', { class: 'muted' }, 'Mining pays out in many small coins, and every coin a payment spends adds to its size and fee. Merging the coins on an address takes one signature for all of them, so it costs little, and it turns them into a few bigger coins.'),
                merges),
            h('div', { class: 'card' },
                h('h2', null, 'Gather your coins ', h('span', { class: 'muted small' }, '(defragment)')),
                h('p', { class: 'muted' }, 'Payments and change land on many one-time addresses, and a payment from them carries a signature for each. This moves everything onto your main address, where one signature covers it all, for a small network fee.'),
                coins, tidyBtn),
            h('div', { class: 'card' },
                h('h2', null, 'Look through the chain again'),
                h('p', { class: 'muted' }, 'If a payment you expect doesn’t show, the wallet can read the whole chain again. Your money stays where it is; sending waits until it’s done.'),
                rescanBtn),
        ],
        update() {
            const blockedNow = sendBlocked();
            const list = mergeable();
            if (changed('m', [list, blockedNow])) {
                fill(merges, list.length
                    ? group(list.map((a) => {
                        const btn = h('button', { class: 'btn small', type: 'button', disabled: !!blockedNow }, 'Merge');
                        btn.addEventListener('click', () => merge(a, btn));
                        return h('div', { class: 'row' },
                            h('span', { class: 'label' }, a.main ? 'Your address' : 'An older address',
                                h('small', null, `${a.coins.toLocaleString()} coins · ${withUnit(a.value)} · `, h('span', { class: 'mono' }, shortAddr(a.address)))),
                            btn);
                    }))
                    : h('p', { class: 'small' }, 'Nothing to merge: your coins are already as few as they can be.'));
            }
            const n = spendable().length;
            const a = addresses();
            coins.textContent = `This wallet holds ${n} coin${n === 1 ? '' : 's'} on ${a} address${a === 1 ? '' : 'es'}.`;
            tidyBtn.disabled = n < 2 || a < 2 || !!blockedNow;
            rescanBtn.disabled = busyOp();
        },
    };
}

function advancedScreen() {
    const pre = h('pre', { class: 'log' });
    const draw = () => {
        const atEnd = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
        pre.textContent = app.log.length ? app.log.join('\n') : 'Nothing yet.';
        if (atEnd) pre.scrollTop = pre.scrollHeight;
    };
    return {
        title: 'Advanced', back: ['Settings', '#/settings'], backAlways: true,
        els: [
            h('h2', null, 'The wallet’s log'),
            h('p', { class: 'muted' }, 'What the wallet has been doing, for when something needs explaining. It holds no keys.'),
            pre,
            h('div', { class: 'btns' }, h('button', { class: 'btn small', type: 'button', onclick: () => copyText(app.log.join('\n'), 'Log copied') }, icon('copy'), 'Copy the log')),
        ],
        update: draw,
        onLog: draw,
    };
}

function aboutScreen() {
    return {
        title: 'About', back: ['Settings', '#/settings'], backAlways: true,
        els: [
            group(
                infoRow('Your keys', 'Stay in this browser, locked with your password'),
                infoRow('Amounts', `In ${UNIT}: one ${UNIT} is 2³⁰ base units, the first block reward`),
                linkRow('Midstate', '../', { iconName: 'globe' }),
                linkRow('Block explorer', `${net.rpcUrl()}/`, { iconName: 'list', external: true }),
                linkRow('Source code', SOURCE, { iconName: 'file', external: true })),
        ],
    };
}

// ── start ───────────────────────────────────────────────────────────────────

function watchForUpdates() {
    if (!('serviceWorker' in navigator)) return;
    // The first install takes the page over quietly (the old page asked to
    // "update" on everyone's first visit); a takeover after that is a new
    // version, and the page that's open is the old one.
    let controlled = !!navigator.serviceWorker.controller;
    navigator.serviceWorker.register('./sw.js').catch(() => {});
    navigator.serviceWorker.addEventListener('controllerchange', () => {
        if (controlled) {
            app.update = true;
            update();
        }
        controlled = true;
    });
}

async function boot() {
    document.title = 'Midstate wallet';
    render();
    watchForUpdates();
    startBridge({
        post,
        isUnlocked: () => app.phase === 'wallet',
        primaryAddress: () => (app.dash && app.dash.primaryAddress !== 'None' ? app.dash.primaryAddress : null),
        onWorker: (fn) => listeners.add(fn),
    });
    if (!(await takeLock())) {
        app.phase = 'elsewhere';
        render();
        return;
    }
    startWorker();
}

boot();
