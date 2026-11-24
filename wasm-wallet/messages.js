// messages.js: sealed messages between names, as the desktop wallet sends
// and reads them (crates/walletd/src/service_identity.rs). A name's wallet is
// asked for its KEM key over the chat bus (wire commands 69 and 70), checked
// against the name's `kem` record; one sealed session carries a conversation
// (midstate::sealed, in WebAssembly through src/messages.rs); strangers pay
// to be heard with a stamp of the work the recipient's name asks; requests,
// blocking and disappearing messages work as they do there.
//
// The worker hands this the wallet's crypto, the chain and the bus (see
// Messenger's constructor) and keeps the book it writes in the wallet's saved
// state, encrypted with the rest of it.
//
// Where a browser makes it differ from the desktop wallet:
//  * The node keeps the last 100 chats, and they are read only while the
//    wallet is open: what came and went while it was closed is missed.
//  * Every chat's proof of work is done here and it goes to the node from a
//    fresh random sender, so nothing on the bus ties two of them together. A
//    chat the node does not take is tried again, for an hour, and then shown
//    as not sent until it is tried again by hand.
//  * The wallet writes as a name once told which ("Write as"), after checking
//    that the name's identity key is one of this wallet's and its kem record
//    this wallet's key. Then it also hands that key to whoever asks the
//    identity. Without a name it writes anonymously.
//  * What strangers pay is what that name asks (its postage record), or the
//    standard 24 bits. The name's holder sets it: Names, here, for a name
//    this wallet holds; the desktop app for one that holds it.
//  * Conversations keep at most MAX_LINES lines each and MAX_TEXT bytes in
//    all, oldest first, to fit the browser's storage.

import { normalize, kemHash, identityOf, postageOf, SUFFIX } from './names.js';

export const STAMP_BITS = 24;              // sealed::STAMP_BITS
export const MAX_BODY = 750;               // sealed::MAX_BODY, in bytes
export const CHUNK_HEX = 2 * 1536;         // chat::MAX_ENCRYPTED_MESSAGE_SIZE
export const MAX_ATTACHMENTS = 4;          // chat::MAX_CHAT_ATTACHMENTS
export const WINDOW = 16;                  // sealed::WINDOW
export const MARKER = 255;                 // core::channel::wire
export const CMD_KEM_REQ = 69;
export const CMD_KEM = 70;
export const KEM_REQ_PAYLOAD = '0200000000'; // wire::pack_u32(0, &[])
export const KEM_ANSWERS_PER_BLOCK = 8;
export const KEM_RETRY = 3;                // blocks
export const KEM_REQUEST_AGE = 30 * 60;    // a request older than this is not answered: its asker gave up
export const OUTBOX_PATIENCE = 15 * 60;    // seconds a message waits for a name's key
export const REQUEST_LINES = 3;
export const REQUESTS_MAX = 100;
export const REQUEST_AGE = 30 * 86_400;
export const TIMERS = [0, 3_600, 86_400, 7 * 86_400, 28 * 86_400]; // walletd::SEALED_TIMERS
export const POST_PATIENCE = 60 * 60;      // seconds a chat is tried before it shows as not sent
export const SEEN_MAX = 500;
export const MAX_LINES = 400;
export const MAX_TEXT = 300_000;
export const STAMP_SLICE = 1 << 16;
const NAME_CHECK_EVERY = 10 * 60;
const RECORDS_FRESH = 60;

// ── bytes ───────────────────────────────────────────────────────────────────

const enc = new TextEncoder();
const dec = new TextDecoder('utf-8', { fatal: false });
export const hexOf = bytes => Array.from(bytes, b => (b & 255).toString(16).padStart(2, '0')).join('');
export const bytesOf = hex => Uint8Array.from((String(hex).match(/../g) || []).map(b => parseInt(b, 16)));
export const textHex = text => hexOf(enc.encode(text));
/** A body's bytes as text: what is not UTF-8 is replaced, as from_utf8_lossy does. */
export const hexText = hex => dec.decode(bytesOf(hex));
const errText = e => (e && e.message) || String(e);
const shown = name => (name ? name + SUFFIX : '');
const pause = () => new Promise(r => setTimeout(r, 0));

/** A message's text, trimmed, if there is one and it fits. */
export function checkedText(text) {
    const t = String(text ?? '').trim();
    if (!t) throw new Error('Nothing to send.');
    const n = enc.encode(t).length;
    if (n > MAX_BODY) throw new Error(`A message holds at most ${MAX_BODY} bytes; this is ${n}.`);
    return t;
}

// ── the bus ─────────────────────────────────────────────────────────────────

/** A chat's sealed envelope (hex): its encrypted_message attachments joined
 *  in order, or null if it carries none. */
export function envelopeOf(attachments) {
    const parts = (attachments || []).filter(a => a && a.kind === 'encrypted_message' && typeof a.value === 'string');
    if (!parts.length) return null;
    const env = parts.map(a => a.value.toLowerCase()).join('');
    return /^([0-9a-f]{2})+$/.test(env) ? env : null;
}

/** The attachments that carry an envelope (an opening or a message). */
export function attachmentsFor(envHex) {
    const parts = String(envHex).match(new RegExp(`.{1,${CHUNK_HEX}}`, 'g')) || [];
    if (!parts.length || parts.length > MAX_ATTACHMENTS) throw new Error(`A sealed message takes one to ${MAX_ATTACHMENTS} attachments.`);
    return parts.map(value => ({ kind: 'encrypted_message', value }));
}

/** A wire frame (`words = [255, cmd]`), as walletd's channels::parse_frame
 *  reads one: the first coin id, signature and address. Addresses come
 *  checksummed (72 hex); the key is the first 64. */
export function parseFrame(m) {
    if (!m || !Array.isArray(m.words) || m.words.length < 2 || m.words[0] !== MARKER) return null;
    const f = { cmd: m.words[1], id: null, payload: null, address: null };
    for (const a of m.attachments || []) {
        if (!a || typeof a.value !== 'string') continue;
        if (a.kind === 'coin_id' && f.id == null) f.id = a.value.toLowerCase();
        else if (a.kind === 'signature' && f.payload == null) f.payload = a.value.toLowerCase();
        else if (a.kind === 'address' && f.address == null) f.address = a.value.toLowerCase().slice(0, 64);
    }
    return f;
}

/** Asking `identity` for its KEM key: the request id in the coin id, the
 *  identity asked in the address (checksummed with `check`). */
export function kemRequest(reqId, identity, check) {
    return {
        words: [MARKER, CMD_KEM_REQ],
        attachments: [{ kind: 'coin_id', value: reqId }, { kind: 'signature', value: KEM_REQ_PAYLOAD }, { kind: 'address', value: check(identity) }],
    };
}

/** The answer: the key itself in the signature attachment. */
export function kemReply(reqId, publicHex) {
    return { words: [MARKER, CMD_KEM], attachments: [{ kind: 'coin_id', value: reqId }, { kind: 'signature', value: publicHex }] };
}

export const seenKey = m => `${m.timestamp}:${m.nonce}:${String(m.sender || '').slice(0, 16)}`;

// ── the book ────────────────────────────────────────────────────────────────

const said = (mine, text, at, ttl) => ({ mine, text, at, ttl, timer: null, stamping: null, sending: null, failed: false });
const timerSet = (mine, ttl, at) => ({ mine, text: '', at, ttl: 0, timer: ttl, stamping: null, sending: null, failed: false });
/** A message waiting on its stamp has not been sent: its timer has not started. */
const expired = (l, now) => l.stamping == null && l.ttl > 0 && l.at + l.ttl <= now;
const sealedTtl = l => (l.timer != null ? l.timer : l.ttl);
const lastAt = c => (c.lines.length ? c.lines[c.lines.length - 1].at : c.started);

export function newBook() {
    return {
        v: 1, chats: [], outbox: [], kemReqs: {}, peerKems: {}, prices: {}, seen: [], blocked: [],
        dropAnonymous: false, openings: {}, posts: [], asName: null,
    };
}

/** Fills in what a book saved by an older page lacks, in place. */
export function loadBook(b) {
    const d = newBook();
    for (const k of Object.keys(d)) if (b[k] == null) b[k] = d[k];
    for (const c of b.chats) {
        c.lines = c.lines || [];
        c.read = Math.min(c.read ?? c.lines.length, c.lines.length);
        // The other side wrote first, or has written.
        c.heard = !!(c.heard || !c.session.opener || c.lines.some(l => !l.mine));
        for (const l of c.lines) {
            l.timer = l.timer ?? null; l.stamping = l.stamping ?? null; l.sending = l.sending ?? null; l.failed = !!l.failed;
        }
    }
    return b;
}

/** The other side's messages its user has not seen. */
export const unread = c => c.lines.slice(c.read).filter(l => !l.mine && l.timer == null).length;

/** Lines whose timer has run are deleted; the read mark keeps its place. */
export function prune(c, now) {
    const before = c.lines.length;
    let read = c.read;
    c.lines = c.lines.filter((l, i) => {
        const keep = !expired(l, now);
        if (!keep && i < c.read) read--;
        return keep;
    });
    c.read = read;
    return c.lines.length !== before;
}

/** A request: not accepted, opened by the other side, and not a name that
 *  checked out (the desktop's inbox, at its default bar). */
export const isRequest = c => !c.accepted && !c.session.opener && (!c.peer || !c.verified);

/** Whether a message the other side sent is kept: until this wallet answers
 *  or accepts, it must carry a stamp of `price`; a request keeps
 *  REQUEST_LINES of the sender's messages at most. */
export function admits(c, got, request, price) {
    if (!c.accepted && got.stamp_bits < price) return false;
    return !(request && c.lines.filter(l => !l.mine).length >= REQUEST_LINES);
}

/** The stamp this wallet's next message in `c` needs: what the other side's
 *  name asks until they have written; nothing after. */
export function stampFor(c, book) {
    if (c.heard) return 0;
    if (!c.peer) return STAMP_BITS;
    return book.prices[c.peer] ?? STAMP_BITS;
}

/** Room for one more request: those quiet REQUEST_AGE go, then the quietest
 *  while `max` are held. */
export function makeRoom(book, now, max = REQUESTS_MAX) {
    book.chats = book.chats.filter(c => !(isRequest(c) && lastAt(c) + REQUEST_AGE < now));
    for (;;) {
        const held = book.chats.filter(isRequest);
        if (held.length < max) return;
        const quietest = held.reduce((a, c) => (lastAt(c) < lastAt(a) ? c : a));
        book.chats = book.chats.filter(c => c !== quietest);
    }
}

/** Blocks `name`: its conversations go, and what waits for it. */
export function block(book, name) {
    if (!book.blocked.includes(name)) book.blocked.push(name);
    book.chats = book.chats.filter(c => c.peer !== name);
    book.outbox = book.outbox.filter(q => q.to !== name);
    delete book.openings[name];
}

/** Drops this wallet's messages in `c` still waiting on their stamps; how
 *  many. Their numbers are given back while nothing after them went, so the
 *  other side's window of numbers is not used up by messages that never came. */
export function unsend(c) {
    const dropped = new Set(c.lines.filter(l => l.stamping != null).map(l => l.stamping));
    const before = c.lines.length;
    let read = c.read;
    c.lines = c.lines.filter((l, i) => {
        const keep = l.stamping == null;
        if (!keep && i < c.read) read--;
        return keep;
    });
    c.read = read;
    while (c.session.sent > 1 && dropped.has(c.session.sent - 1)) c.session.sent--;
    return before - c.lines.length;
}

/** What waits for a name's key, by name, in the order first queued. */
export function waiting(book) {
    const out = new Map();
    for (const q of book.outbox) {
        if (!out.has(q.to)) out.set(q.to, []);
        out.get(q.to).push(q);
    }
    return [...out.entries()];
}

/** Keeps the book to what a browser's storage takes: MAX_LINES a
 *  conversation and MAX_TEXT bytes in all, the oldest said going first.
 *  What still waits to go is kept. */
export function trim(book) {
    let changed = false;
    const drop = (c, line) => {
        const i = c.lines.indexOf(line);
        if (i < 0) return;
        c.lines.splice(i, 1);
        if (i < c.read) c.read--;
        changed = true;
    };
    const settled = l => l.stamping == null && l.sending == null;
    for (const c of book.chats) {
        while (c.lines.length > MAX_LINES) {
            const old = c.lines.find(settled);
            if (!old) break;
            drop(c, old);
        }
    }
    let total = book.chats.reduce((s, c) => s + c.lines.reduce((t, l) => t + l.text.length, 0), 0);
    while (total > MAX_TEXT) {
        let oldest = null;
        for (const c of book.chats) {
            const l = c.lines.find(settled);
            if (l && (!oldest || l.at < oldest.l.at)) oldest = { c, l };
        }
        if (!oldest) break;
        total -= oldest.l.text.length;
        drop(oldest.c, oldest.l);
    }
    return changed;
}

/** The conversations as the page shows them, newest first. */
export function view(book, now, extra = {}) {
    const line = l => ({
        mine: l.mine, text: l.text, at: l.at, timer: l.timer, stamping: l.stamping != null,
        sending: l.sending != null, failed: !!l.failed, pending: false,
        expires: l.ttl > 0 && l.stamping == null ? l.at + l.ttl : 0,
    });
    const chats = book.chats.map(c => ({
        key: c.id, id: c.id, peer: shown(c.peer), verified: c.verified, openedByMe: !!c.session.opener,
        lines: c.lines.filter(l => !expired(l, now)).map(line),
        request: isRequest(c), unread: unread(c), ttl: c.ttl, stamping: !c.heard, stampBits: stampFor(c, book),
        last: lastAt(c), note: '',
    }));
    for (const [to, queued] of waiting(book)) {
        const stamped = !!book.openings[to];
        chats.push({
            key: 'to:' + to, id: '', peer: shown(to), verified: false, openedByMe: true,
            lines: queued.map(q => ({ mine: true, text: q.text, at: q.queued, timer: null, stamping: stamped, sending: false, failed: false, pending: !stamped, expires: 0 })),
            request: false, unread: 0, ttl: 0, stamping: true, stampBits: book.prices[to] ?? STAMP_BITS,
            last: queued[queued.length - 1].queued, note: stamped ? '' : (queued[0].note || ''),
        });
    }
    chats.sort((a, b) => b.last - a.last);
    return {
        available: true,
        chats,
        unread: chats.filter(c => !c.request).reduce((s, c) => s + c.unread, 0),
        requests: chats.filter(c => c.request).length,
        blocked: book.blocked.map(shown),
        dropAnonymous: !!book.dropAnonymous,
        asName: book.asName ? shown(book.asName.name) : '',
        ...extra,
    };
}

// ── the messenger ───────────────────────────────────────────────────────────

/**
 * deps:
 *   book       the saved book (wState.sealed), kept up to date in place
 *   crypto     kemPublic(), kemHash(), tryOpen(env), prepare(public, hash, asName) (async),
 *              preparedKey(p), preparedRecipient(p), finish(p, bodyHex, ttl, nonce),
 *              reserve(session), stampKey(session, n), seal(session, n, bodyHex, ttl, nonce),
 *              routes(session), open(session, n, env), stampSearch(key, bits, start, count),
 *              verifySender(name, identity, signature, transcript, identityOfName),
 *              checkPublic(public, hash) (throws), hash(hex), random(bytes), checksum(hex64)
 *   chain      height(), records(name) (async: records, null if no one holds it,
 *              throws if it could not be told), history() (async), post(words, attachments) (async)
 *   identities()  this wallet's MSS keys (hex)
 *   save()     persists the wallet (async)
 *   changed()  the conversations changed
 *   notice(text), log(text), now() (seconds)
 */
export class Messenger {
    constructor(deps) {
        this.d = deps;
        this.c = deps.crypto;
        this.book = loadBook(deps.book);
        this.jobs = new Map();
        this.stampsRunning = false;
        this.postsRunning = false;
        this.queue = Promise.resolve();
        this.routeCache = new Map();
        this.records_ = new Map();
        this.answered = { height: -1, count: 0 };
        this.rate = 0;
        this.nameChecked = 0;
        this.stopped = false;
        this.ticking = null;
        this.priceNow = null;
        this.myHash = this.c.kemHash();
        this.myPublic = null;
    }

    /** One at a time: what changes the book waits for what is changing it. */
    serial(fn) {
        const run = this.queue.then(fn, fn);
        this.queue = run.catch(() => {});
        return run;
    }

    stop() {
        this.stopped = true;
        this.jobs.clear();
    }

    state() {
        return view(this.book, this.d.now(), { rate: this.rate, kem: this.myHash, price: this.priceNow ?? STAMP_BITS });
    }

    async save() {
        await this.d.save();
        this.d.changed();
    }

    /** A name's records, kept a minute: null if no one holds it. */
    async records(name, fresh = false) {
        const now = this.d.now();
        const c = this.records_.get(name);
        if (c && !fresh && now - c.at < RECORDS_FRESH) return c.records;
        const records = await this.d.chain.records(name);
        this.records_.set(name, { at: now, records });
        return records;
    }

    /** What this wallet asks of a stranger's message: what the name it writes
     *  as asks, while that name takes this wallet's messages. */
    async price() {
        const me = this.book.asName;
        let bits = STAMP_BITS;
        if (me) {
            try {
                const r = await this.records(me.name);
                if (r && kemHash(r) === this.myHash) bits = postageOf(r);
            } catch (_) { /* as last known */ bits = this.priceNow ?? STAMP_BITS; }
        }
        this.priceNow = bits;
        return bits;
    }

    // ── reading the bus ─────────────────────────────────────────────────

    start() {
        this.serial(() => this.reconcile().then(ch => (ch ? this.save() : null))).catch(e => this.d.log(errText(e)));
        this.kickPosts();
    }

    /** Read the bus once (or `history`, read already), keep what opens for
     *  this wallet, let timers run, and retry or give up on what waits. */
    tick(history = null) {
        if (this.ticking) return this.ticking;
        this.ticking = (async () => {
            if (!history) history = await this.d.chain.history();
            await this.serial(() => this.ingestNow(history));
            await this.serial(() => this.housekeep());
        })().catch(e => this.d.log(`messages: ${errText(e)}`)).finally(() => { this.ticking = null; });
        return this.ticking;
    }

    /** Chats pushed by the node, as they come. */
    ingest(messages) {
        return this.serial(() => this.ingestNow(messages));
    }

    routeMap() {
        const map = new Map();
        for (const c of this.book.chats) {
            let r = this.routeCache.get(c.id);
            if (!r || r.received !== c.session.received) {
                r = { received: c.session.received, list: this.c.routes(c.session) };
                this.routeCache.set(c.id, r);
            }
            for (const [route, n] of r.list) map.set(route, { chat: c, n });
        }
        return map;
    }

    markSeen(key) {
        this.book.seen.push(key);
        if (this.book.seen.length > SEEN_MAX) this.book.seen.splice(0, this.book.seen.length - SEEN_MAX);
    }

    async ingestNow(messages) {
        const b = this.book;
        const price = await this.price();
        const tip = this.d.chain.height();
        let changed = false;
        let routes = null;
        for (const m of messages || []) {
            if (this.stopped) return;
            if (!m || !Array.isArray(m.attachments) || !m.attachments.length) continue;
            const key = seenKey(m);
            if (b.seen.includes(key)) continue;
            const f = parseFrame(m);
            if (f && (f.cmd === CMD_KEM_REQ || f.cmd === CMD_KEM)) {
                this.markSeen(key);
                changed = true;
                await this.frame(f, m, tip);
                continue;
            }
            const env = envelopeOf(m.attachments);
            if (!env) continue;
            routes = routes || this.routeMap();
            const routed = routes.get(env.slice(0, 32));
            if (routed) {
                this.markSeen(key);
                changed = true;
                const got = this.c.open(routed.chat.session, routed.n, env);
                if (got) {
                    routed.chat.session = got.session;
                    routes = null;
                    this.receive(routed.chat, got, Number(m.timestamp) || this.d.now(), price);
                }
                continue;
            }
            const opened = this.c.tryOpen(env);
            if (!opened) {
                this.markSeen(key);
                changed = true;
                continue;
            }
            if (b.chats.some(c => c.id === hexOf(opened.session.id))) {
                this.markSeen(key);
                continue;
            }
            // A name that could not be looked up now is tried again next time.
            if (await this.welcome(opened, Number(m.timestamp) || this.d.now(), price) === 'later') continue;
            this.markSeen(key);
            changed = true;
            routes = null;
        }
        if (changed) await this.save();
    }

    async frame(f, m, tip) {
        const b = this.book;
        if (f.cmd === CMD_KEM_REQ) {
            const me = b.asName;
            if (!me || !f.id || f.address !== me.identity) return;
            if (this.d.now() - Number(m.timestamp || 0) > KEM_REQUEST_AGE) return;
            if (this.answered.height !== tip) this.answered = { height: tip, count: 0 };
            if (this.answered.count >= KEM_ANSWERS_PER_BLOCK) return;
            this.answered.count++;
            this.myPublic = this.myPublic || this.c.kemPublic();
            this.queuePost(kemReply(f.id, this.myPublic), { kind: 'kem' });
            return;
        }
        const r = f.id && b.kemReqs[f.id];
        if (!r || !f.payload) return;
        try {
            this.c.checkPublic(f.payload, r.hash);
        } catch (e) {
            this.d.log(`a key from ${shown(r.name)}'s wallet is not the one the name lists`);
            return;
        }
        for (const [k, v] of Object.entries(b.kemReqs)) if (v.name === r.name) delete b.kemReqs[k];
        b.peerKems[r.name] = f.payload;
        await this.flush(r.name);
    }

    /** A message the other side sent in `c`, kept if it paid its way. Once
     *  they have written, what waits here on a stamp goes without one. */
    receive(c, got, at, price) {
        const request = isRequest(c);
        if (!admits(c, got, request, price)) return;
        if (!c.heard) {
            c.heard = true;
            this.sendStampedNow(c);
        }
        if (got.ttl !== c.ttl) {
            c.ttl = got.ttl;
            c.lines.push(timerSet(false, got.ttl, at));
        }
        if (got.body) {
            c.lines.push(said(false, hexText(got.body), at, got.ttl));
            // Requests wait quietly: a stranger cannot make this wallet speak up.
            if (!request) this.notify(c);
        }
    }

    /** An opening to this wallet's key: a new conversation, unless it did not
     *  pay `price`, or comes from a blocked name, or from someone anonymous
     *  when those are dropped. */
    async welcome(o, at, price) {
        const b = this.book;
        if (o.stamp_bits < price || !o.body) return 'no';
        // A claim that is not a name at all is no better than none.
        let peer = '';
        try { peer = o.name ? normalize(o.name) : ''; } catch (_) { peer = ''; }
        let verified = false;
        if (peer) {
            let recs;
            try { recs = await this.records(peer); } catch (_) { return 'later'; }
            verified = peer === o.name && this.c.verifySender(o.name, o.identity, o.signature, o.transcript, (recs && identityOf(recs)) || '');
        }
        if ((peer && b.blocked.includes(peer)) || (!peer && b.dropAnonymous)) return 'no';
        const now = this.d.now();
        const c = {
            id: hexOf(o.session.id), session: o.session, peer, verified, lines: [], started: now,
            read: 0, accepted: false, heard: true, ttl: o.ttl,
        };
        if (o.ttl > 0) c.lines.push(timerSet(false, o.ttl, at));
        c.lines.push(said(false, hexText(o.body), at, o.ttl));
        const request = isRequest(c);
        if (request) makeRoom(b, now);
        b.chats.push(c);
        if (!request) this.notify(c);
        return 'yes';
    }

    notify(c) {
        const who = c.peer ? `${shown(c.peer)}${c.verified ? '' : ' (unverified)'}` : 'someone anonymous';
        this.d.notice({ text: `New sealed message from ${who}.`, id: c.id });
    }

    // ── writing ─────────────────────────────────────────────────────────

    find(id) {
        const c = this.book.chats.find(x => x.id === id);
        if (!c) throw new Error('That conversation is gone.');
        return c;
    }

    /** Puts this wallet's `line` in `c` and sends it: now if it needs no
     *  stamp, or once its stamp is done. */
    writeLine(c, line) {
        const bits = stampFor(c, this.book);
        const r = this.c.reserve(c.session);
        c.session = r.session;
        if (bits === 0) {
            this.sendLine(c, r.n, line, 0n);
        } else {
            line.stamping = r.n;
            this.submitStamp({ kind: 'line', id: c.id, n: r.n }, r.key, bits);
        }
        c.lines.push(line);
    }

    sendLine(c, n, line, nonce) {
        const env = this.c.seal(c.session, n, textHex(line.text), sealedTtl(line), nonce);
        line.sending = this.queuePost({ words: [], attachments: attachmentsFor(env) }, { kind: 'line', id: c.id });
    }

    /** Writes `text` in `c`, under its timer. Writing is reading it, and
     *  takes it out of Requests. */
    sealIn(c, text) {
        this.writeLine(c, said(true, text, this.d.now(), c.ttl));
        c.read = c.lines.length;
        c.accepted = true;
    }

    /** What the compose box says of a name before anything is written. */
    async recipient(input) {
        const name = normalize(input);
        const s = shown(name);
        if (this.book.blocked.includes(name)) return { name: s, ok: false, blocked: true, says: `You blocked ${s}. Unblock it to write to it.` };
        const recs = await this.records(name, true);
        if (!recs) return { name: s, ok: false, says: `No one holds ${s}.` };
        const kem = kemHash(recs);
        if (!kem) return { name: s, ok: false, says: `${s} takes no sealed messages.` };
        if (kem === this.myHash) return { name: s, ok: false, says: `${s} takes this wallet’s own messages.` };
        const known = this.book.peerKems[name];
        const keyKnown = !!known && this.c.hash(known) === kem;
        if (!keyKnown && !identityOf(recs)) return { name: s, ok: false, says: `${s} lists no identity key, so its wallet can’t be asked for its key.` };
        const bits = postageOf(recs);
        this.book.prices[name] = bits;
        return { name: s, ok: true, price: bits, keyKnown, existing: this.book.chats.some(c => c.peer === name && c.verified) };
    }

    /** Writes `text` to a name: in the conversation with it, if one is
     *  open; otherwise it opens one, asking the name's wallet for its key
     *  first if need be. Says where it went. */
    send(to, text) {
        return this.serial(async () => {
            const name = normalize(to);
            if (this.book.blocked.includes(name)) throw new Error(`You blocked ${shown(name)}: unblock it to write to it.`);
            const t = checkedText(text);
            const open = [...this.book.chats].reverse().find(c => c.peer === name && c.verified);
            if (open) {
                this.sealIn(open, t);
                await this.save();
                return { key: open.id };
            }
            const card = await this.recipient(name);
            if (!card.ok) throw new Error(card.says);
            this.book.outbox.push({ to: name, text: t, queued: this.d.now(), note: '' });
            await this.save();
            await this.flush(name);
            return { key: 'to:' + name };
        });
    }

    reply(id, text) {
        return this.serial(async () => {
            this.sealIn(this.find(id), checkedText(text));
            await this.save();
            return { key: id };
        });
    }

    /** Sets conversation `id`'s timer: what either side sends from now on is
     *  deleted from both wallets `ttl` seconds after it went. */
    setTimer(id, ttl) {
        return this.serial(async () => {
            if (!TIMERS.includes(ttl)) throw new Error('A timer is off, an hour, a day, a week or four weeks.');
            const c = this.find(id);
            if (c.ttl === ttl) return {};
            this.writeLine(c, timerSet(true, ttl, this.d.now()));
            c.ttl = ttl;
            c.read = c.lines.length;
            c.accepted = true;
            await this.save();
            return {};
        });
    }

    read(id) {
        return this.serial(async () => {
            const c = this.book.chats.find(x => x.id === id);
            if (!c || c.read === c.lines.length) return {};
            c.read = c.lines.length;
            await this.save();
            return {};
        });
    }

    accept(id) {
        return this.serial(async () => {
            this.find(id).accepted = true;
            await this.save();
            return {};
        });
    }

    /** Forgets a conversation, its session too: what the other side sends
     *  in it later is never read. A new session from them still comes. */
    remove(id) {
        return this.serial(async () => {
            const c = this.find(id);
            this.book.chats = this.book.chats.filter(x => x !== c);
            this.dropStale();
            await this.save();
            return {};
        });
    }

    /** Blocks whoever is on the other side of `id`: a verified name by name;
     *  anyone else cannot be told from other strangers, so only the
     *  conversation goes. Says which name, if one. */
    blockChat(id) {
        return this.serial(async () => {
            const c = this.find(id);
            const name = c.verified && c.peer ? c.peer : null;
            if (name) block(this.book, name);
            else this.book.chats = this.book.chats.filter(x => x !== c);
            this.dropStale();
            await this.save();
            return { blocked: name ? shown(name) : '' };
        });
    }

    unblock(input) {
        return this.serial(async () => {
            const name = normalize(input);
            const before = this.book.blocked.length;
            this.book.blocked = this.book.blocked.filter(n => n !== name);
            if (this.book.blocked.length === before) throw new Error(`${shown(name)} is not blocked.`);
            await this.save();
            return {};
        });
    }

    /** Stops what waits to go to a name: for its key, or on the opening's stamp. */
    cancel(input) {
        return this.serial(async () => {
            const name = normalize(input);
            const before = this.book.outbox.length;
            this.book.outbox = this.book.outbox.filter(q => q.to !== name);
            delete this.book.openings[name];
            if (before === this.book.outbox.length) throw new Error(`Nothing is waiting to go to ${shown(name)}.`);
            this.dropStale();
            await this.save();
            return {};
        });
    }

    /** Stops what in `id` still waits on its stamp. */
    unsendChat(id) {
        return this.serial(async () => {
            if (unsend(this.find(id)) === 0) throw new Error('Nothing in that conversation is waiting to go.');
            this.dropStale();
            await this.save();
            return {};
        });
    }

    /** Sends again what was given up on. */
    retry(id) {
        return this.serial(async () => {
            const c = this.find(id);
            let n = 0;
            for (const l of c.lines) {
                const p = l.failed && this.book.posts.find(x => x.id === l.sending);
                if (!p) continue;
                Object.assign(p, { held: false, tries: 0, next: 0, made: this.d.now() });
                l.failed = false;
                n++;
            }
            if (!n) throw new Error('Nothing in that conversation is waiting to be sent again.');
            await this.save();
            this.kickPosts();
            return {};
        });
    }

    setDropAnonymous(on) {
        return this.serial(async () => {
            this.book.dropAnonymous = !!on;
            await this.save();
            return {};
        });
    }

    /** Writes as `input` from now on (null: anonymously), once its records
     *  show it is this wallet's: its identity key one of this wallet's, its
     *  kem record this wallet's key. */
    setName(input) {
        return this.serial(async () => {
            if (input == null || input === '') {
                this.book.asName = null;
                await this.save();
                return { asName: '' };
            }
            const name = normalize(input);
            const s = shown(name);
            const recs = await this.records(name, true);
            if (!recs) throw new Error(`No one holds ${s}.`);
            if (kemHash(recs) !== this.myHash) {
                throw new Error(kemHash(recs) ? `${s} lists another wallet’s message key, so what is sealed to it can’t be read here.` : `${s} takes no sealed messages yet: it needs this wallet’s message key (in Names, for a name this wallet holds).`);
            }
            const identity = identityOf(recs);
            if (!identity || !this.d.identities().includes(identity)) {
                throw new Error(`${s}’s identity key isn’t one of this wallet’s, so this wallet can’t sign as it or hand over its key.`);
            }
            this.book.asName = { name, identity };
            this.nameChecked = this.d.now();
            await this.save();
            return { asName: s };
        });
    }

    /** The page's requests, by name. */
    async handle(op, a = {}) {
        switch (op) {
            case 'state': return this.state();
            case 'recipient': return this.serial(() => this.recipient(a.name));
            case 'send': return this.send(a.to, a.text);
            case 'reply': return this.reply(a.chat, a.text);
            case 'timer': return this.setTimer(a.chat, Number(a.ttl));
            case 'read': return this.read(a.chat);
            case 'accept': return this.accept(a.chat);
            case 'delete': return this.remove(a.chat);
            case 'block': return this.blockChat(a.chat);
            case 'unblock': return this.unblock(a.name);
            case 'cancel': return this.cancel(a.name);
            case 'unsend': return this.unsendChat(a.chat);
            case 'retry': return this.retry(a.chat);
            case 'anonymous': return this.setDropAnonymous(a.on);
            case 'name': return this.setName(a.name);
            default: throw new Error(`Unknown request: ${op}`);
        }
    }

    // ── keys ────────────────────────────────────────────────────────────

    askKem(name, identity, hash, tip) {
        const b = this.book;
        if (Object.values(b.kemReqs).some(r => r.name === name && tip < r.at + KEM_RETRY)) return;
        for (const [k, v] of Object.entries(b.kemReqs)) if (v.name === name) delete b.kemReqs[k];
        const reqId = this.c.random(32);
        b.kemReqs[reqId] = { name, hash, at: tip };
        this.queuePost(kemRequest(reqId, identity, this.c.checksum), { kind: 'kem' });
    }

    /** Opens a session with `name` for what waits for it, if its key is known
     *  and still the one its record lists; asks for the key if not. */
    async flush(name) {
        const b = this.book;
        const queued = b.outbox.filter(q => q.to === name);
        if (!queued.length || b.openings[name]) return;
        const note = text => { for (const q of queued) q.note = text; };
        try {
            const s = shown(name);
            const recs = await this.records(name);
            if (!recs) throw new Error(`No one holds ${s} now.`);
            const hash = kemHash(recs);
            if (!hash) throw new Error(`${s} takes no sealed messages now.`);
            b.prices[name] = postageOf(recs);
            const known = b.peerKems[name];
            if (!known || this.c.hash(known) !== hash) {
                const identity = identityOf(recs);
                if (!identity) throw new Error(`${s} lists no identity to ask for its key.`);
                this.askKem(name, identity, hash, this.d.chain.height());
                note('');
                await this.save();
                return;
            }
            const prepared = await this.c.prepare(known, hash, b.asName);
            // Made up to the stamp now, signed once; sealed when it is done.
            b.openings[name] = prepared;
            note('');
            await this.save();
            this.submitStamp({ kind: 'opening', name }, this.c.preparedKey(prepared), b.prices[name]);
        } catch (e) {
            note(errText(e));
            this.d.log(`a message to ${shown(name)} is waiting: ${errText(e)}`);
        }
    }

    // ── stamps ──────────────────────────────────────────────────────────

    static jobKey(w) {
        return w.kind === 'line' ? `line:${w.id}:${w.n}` : `open:${w.name}`;
    }

    submitStamp(what, key, bits) {
        const start = BigInt(Math.floor(Math.random() * 2 ** 40)) << 16n;
        this.jobs.set(Messenger.jobKey(what), { what, key, bits, start });
        this.kickStamps();
    }

    reprice(what, bits) {
        const j = this.jobs.get(Messenger.jobKey(what));
        if (j) j.bits = bits;
        return !!j;
    }

    kickStamps() {
        if (this.stampsRunning || this.stopped) return;
        this.stampsRunning = true;
        this.stampLoop().catch(e => this.d.log(`stamps: ${errText(e)}`)).finally(() => { this.stampsRunning = false; });
    }

    /** The cheapest stamp first, a slice at a time. */
    async stampLoop() {
        while (this.jobs.size && !this.stopped) {
            let k = null;
            let job = null;
            for (const [jk, j] of this.jobs) if (!job || j.bits < job.bits) [k, job] = [jk, j];
            const t0 = Date.now();
            const nonce = this.c.stampSearch(job.key, job.bits, job.start, STAMP_SLICE);
            const ms = Date.now() - t0;
            if (ms > 0) this.rate = this.rate ? this.rate * 0.8 + 0.2 * (STAMP_SLICE * 1000 / ms) : STAMP_SLICE * 1000 / ms;
            job.start += BigInt(STAMP_SLICE);
            if (nonce !== undefined && nonce !== null && this.jobs.get(k) === job) {
                this.jobs.delete(k);
                await this.serial(() => this.stamped(job.what, job.bits, BigInt(nonce)));
            }
            await pause();
        }
    }

    /** Whether what `w` stamps still waits to go. */
    wanted(w) {
        const b = this.book;
        if (w.kind === 'opening') return !!b.openings[w.name] && b.outbox.some(q => q.to === w.name);
        return b.chats.some(c => c.id === w.id && c.lines.some(l => l.stamping === w.n));
    }

    dropStale() {
        for (const [k, j] of this.jobs) if (!this.wanted(j.what)) this.jobs.delete(k);
    }

    /** A stamp was done: send what it was for, unless that went meanwhile,
     *  or now asks more. */
    async stamped(w, bits, nonce) {
        try {
            if (w.kind === 'line') await this.stampedLine(w.id, w.n, bits, nonce);
            else await this.stampedOpening(w.name, bits, nonce);
        } catch (e) {
            this.d.log(`a stamped message did not go: ${errText(e)}`);
            if (await this.reconcile()) await this.save();
        }
    }

    async stampedLine(id, n, bits, nonce) {
        const c = this.book.chats.find(x => x.id === id);
        const line = c && c.lines.find(l => l.stamping === n);
        if (!line) return;
        const need = stampFor(c, this.book);
        if (need > bits) {
            // The price went up while it was worked: the work goes on.
            this.submitStamp({ kind: 'line', id, n }, this.c.stampKey(c.session, n), need);
            return;
        }
        this.sendLine(c, n, line, need === 0 ? 0n : nonce);
        line.stamping = null;
        // Its timer runs from when it went, as the other side's does.
        line.at = this.d.now();
        await this.save();
    }

    /** The opening to `name` was stamped: it goes with the first message
     *  that waited for it; the others follow in the session, each stamped
     *  until `name` answers. */
    async stampedOpening(name, bits, nonce) {
        const b = this.book;
        const prepared = b.openings[name];
        if (!prepared) return;
        const need = b.prices[name] ?? STAMP_BITS;
        if (need > bits) {
            this.submitStamp({ kind: 'opening', name }, this.c.preparedKey(prepared), need);
            return;
        }
        const queued = b.outbox.filter(q => q.to === name);
        if (!queued.length) {
            delete b.openings[name];
            await this.save();
            return;
        }
        const first = queued[0];
        const done = this.c.finish(prepared, textHex(first.text), 0, nonce);
        const now = this.d.now();
        const c = {
            id: hexOf(done.session.id), session: done.session, peer: name, verified: true, lines: [],
            started: now, read: 0, accepted: true, heard: false, ttl: 0,
        };
        const line = said(true, first.text, now, 0);
        line.sending = this.queuePost({ words: [], attachments: attachmentsFor(done.opening) }, { kind: 'line', id: c.id });
        c.lines.push(line);
        delete b.openings[name];
        b.outbox = b.outbox.filter(q => q.to !== name);
        b.chats.push(c);
        for (const q of queued.slice(1)) {
            try { this.writeLine(c, said(true, q.text, now, 0)); } catch (e) { this.d.log(`a message to ${shown(name)} did not go: ${errText(e)}`); }
        }
        c.read = c.lines.length;
        await this.save();
    }

    /** The other side of `c` has written: what waits on its stamp there goes
     *  now, unstamped. */
    sendStampedNow(c) {
        for (const l of c.lines) {
            if (l.stamping == null) continue;
            const n = l.stamping;
            try {
                this.sendLine(c, n, l, 0n);
            } catch (e) {
                this.d.log(`a sealed message did not go: ${errText(e)}`);
                continue;
            }
            l.stamping = null;
            l.at = this.d.now();
            this.jobs.delete(Messenger.jobKey({ kind: 'line', id: c.id, n }));
        }
    }

    /** What waits on a stamp is worked at the price asked now; work for what
     *  went stops; after a restart the work starts again. Whether the book
     *  changed. */
    async reconcile() {
        const b = this.book;
        let changed = false;
        this.dropStale();
        for (const c of b.chats) {
            const pending = c.lines.filter(l => l.stamping != null).map(l => l.stamping);
            if (!pending.length) continue;
            if (!c.heard && c.peer) {
                try {
                    const r = await this.records(c.peer);
                    if (r) b.prices[c.peer] = postageOf(r);
                } catch (_) { /* as last known */ }
            }
            const need = stampFor(c, b);
            if (need === 0) {
                this.sendStampedNow(c);
                changed = true;
                continue;
            }
            for (const n of pending) {
                const w = { kind: 'line', id: c.id, n };
                if (!this.reprice(w, need)) this.submitStamp(w, this.c.stampKey(c.session, n), need);
            }
        }
        for (const [name, prepared] of Object.entries(b.openings)) {
            let recs;
            try { recs = await this.records(name); } catch (_) { recs = undefined; }
            if (recs !== undefined && (!recs || kemHash(recs) !== this.c.preparedRecipient(prepared))) {
                // The name's key changed, or no one holds it now: open again,
                // to the key it lists now (asked for afresh).
                delete b.openings[name];
                this.dropStale();
                changed = true;
                continue;
            }
            if (recs) b.prices[name] = postageOf(recs);
            const need = b.prices[name] ?? STAMP_BITS;
            const w = { kind: 'opening', name };
            if (!this.reprice(w, need)) this.submitStamp(w, this.c.preparedKey(prepared), need);
        }
        return changed;
    }

    // ── the chats this wallet puts on the bus ───────────────────────────

    queuePost(frame, what) {
        const p = { id: this.c.random(8), words: frame.words, attachments: frame.attachments, made: this.d.now(), tries: 0, next: 0, held: false, what };
        this.book.posts.push(p);
        this.kickPosts();
        return p.id;
    }

    kickPosts() {
        if (this.postsRunning || this.stopped) return;
        this.postsRunning = true;
        this.postLoop().catch(e => this.d.log(`posting: ${errText(e)}`)).finally(() => { this.postsRunning = false; });
    }

    async postLoop() {
        await pause();
        while (!this.stopped) {
            const now = this.d.now();
            const p = this.book.posts.find(x => !x.held && (x.next || 0) <= now);
            if (!p) return;
            let error = null;
            try {
                await this.d.chain.post(p.words, p.attachments);
            } catch (e) {
                error = errText(e);
            }
            await this.serial(() => this.posted(p, error));
        }
    }

    async posted(p, error) {
        const b = this.book;
        if (!b.posts.includes(p)) return;
        const lines = b.chats.flatMap(c => c.lines.filter(l => l.sending === p.id));
        if (!error) {
            b.posts = b.posts.filter(x => x !== p);
            for (const l of lines) { l.sending = null; l.failed = false; }
        } else {
            const now = this.d.now();
            p.tries++;
            p.next = now + Math.min(300, 10 * 2 ** Math.min(p.tries, 5));
            this.d.log(`a chat did not go (${p.tries}): ${error}`);
            if (now - p.made > POST_PATIENCE) {
                if (!lines.length) {
                    b.posts = b.posts.filter(x => x !== p);
                } else {
                    p.held = true;
                    for (const l of lines) l.failed = true;
                    this.d.notice({ text: 'A sealed message couldn’t be sent. Open the conversation to try again.' });
                }
            }
        }
        await this.save();
    }

    // ── each tick ───────────────────────────────────────────────────────

    async housekeep() {
        const b = this.book;
        const now = this.d.now();
        const tip = this.d.chain.height();
        let changed = false;
        for (const c of b.chats) changed = prune(c, now) || changed;
        changed = trim(b) || changed;
        changed = (await this.reconcile()) || changed;
        // What waits for a key: asked again now and then, given up in the end.
        // An opening being stamped has its key, and waits as long as it takes.
        const givesUp = q => now - q.queued > OUTBOX_PATIENCE && !b.openings[q.to];
        const stale = [...new Set(b.outbox.filter(givesUp).map(q => q.to))];
        if (stale.length) {
            b.outbox = b.outbox.filter(q => !givesUp(q));
            for (const n of stale) this.d.notice({ text: `Gave up sending to ${shown(n)}: its wallet never handed over its key.` });
            changed = true;
        }
        for (const n of new Set(b.outbox.map(q => q.to))) {
            if (Object.values(b.kemReqs).every(r => r.name !== n || tip >= r.at + KEM_RETRY)) await this.flush(n);
        }
        for (const [k, r] of Object.entries(b.kemReqs)) {
            if (!b.outbox.some(q => q.to === r.name)) { delete b.kemReqs[k]; changed = true; }
        }
        // Posts nothing waits on that were given up.
        const before = b.posts.length;
        b.posts = b.posts.filter(p => !(p.held && !b.chats.some(c => c.lines.some(l => l.sending === p.id))));
        changed = changed || b.posts.length !== before;
        // The name this wallet writes as is still its own.
        if (b.asName && now - this.nameChecked > NAME_CHECK_EVERY) {
            try {
                const r = await this.records(b.asName.name, true);
                this.nameChecked = now;
                if (!r || kemHash(r) !== this.myHash || identityOf(r) !== b.asName.identity) {
                    this.d.notice({ text: `${shown(b.asName.name)} no longer lists this wallet’s keys, so messages go anonymously now.` });
                    b.asName = null;
                    changed = true;
                }
            } catch (_) { /* tried again next time */ }
        }
        if (changed) await this.save();
        this.kickPosts();
    }
}
