// registrar.js: names this wallet holds (MSN-1), registered, claimed, renewed
// and changed from the browser, with no desktop app.
//
// The worker hands it the chain (a node's checked answers, coins, commits and
// reveals), the wallet's keys and coins; this keeps the jobs, decides what
// each registry spend sets, and sees each one through: read the name's proof,
// plan the spend (WebAssembly: registry.rs), build and reserve it, commit, and
// once the commit is mined sign and reveal it, then check whose spend took
// the registry. Every name change spends the one registry coin, so another
// wallet's change can get in first; the job then starts over from a fresh
// proof at the next block.
//
// What is signed is signed late: the reveal (the wallet's fee coins, and the
// control key for an owner's change) only once the commit is mined and the
// registry coin is still there, so a spend that lost the race has usually
// signed nothing. If a reveal did go out and then can't land, its MSS coins
// sign again from a new leaf, which is safe; its WOTS coins can't (a second
// signature from a one-time key), so they are locked, as the wallet locks
// coins a reorganisation returns. Fees come from MSS coins when they cover
// them, which keeps that rare.
//
// A name is held as the desktop app holds one: by a fresh MSS key from the
// wallet's seed (its next HD index), and any name held by one of the
// wallet's MSS keys is its own, so the same 24 words manage the same names
// in either app. A key close to its last signature hands the name to a
// fresh key in the same spend.

import {
    normalize, SUFFIX, TAG, encodeRecords, standing, RATES, YEAR, MAX_YEARS, POSTAGE_BITS, POSTAGE_STANDARD,
    postageOf, identityOf, kemHash, MAX_RECORDS,
} from './names.js';

export const TRIES = 8;          // attempts that fail for a reason that may pass
export const SENDS = 3;          // times a reveal is offered before it is given up
export const HANDOVER = 4;       // a control key with this many signatures left hands the name on
export const REFRESH = 10 * 60;  // seconds between reads of a held name
export const TEXT_MAX = 255;     // bytes of a record
export const PATIENCE = 10;      // blocks this waits behind another change to the registry
export const YEAR_CHOICES = [1, 2, 3, 5, 10];

const BY = { register: 'register', claim: 'lapse', renew: 'owner', edit: 'owner' };
const NO_DEAL = '00';
const ACTIVE = new Set(['queued', 'reading', 'building', 'committing', 'committed', 'waiting', 'signing', 'revealing', 'confirming']);

const enc = new TextEncoder();
const dec = new TextDecoder();
const hexOf = bytes => Array.from(bytes, b => (b & 255).toString(16).padStart(2, '0')).join('');
const bytesOf = hex => Uint8Array.from((String(hex).match(/../g) || []).map(b => parseInt(b, 16)));
const errText = e => (e && e.message) || String(e);
export const shown = name => `${name}${SUFFIX}`;

/** A failure that trying again won't mend. */
export class Stop extends Error {}

// ── records ─────────────────────────────────────────────────────────────────

const valueHex = v => (typeof v === 'string' ? v.toLowerCase() : hexOf(v));
const rec = (tag, value) => ({ tag, value: valueHex(value) });

/** Sets one record of `tag` in `list`: the first in place, later ones
 *  dropped; `value` null drops them all. `first` puts a new one first. */
function setTag(list, tag, value, { first = false, tags = [tag] } = {}) {
    const at = list.findIndex(r => tags.includes(r.tag));
    const rest = list.filter(r => !tags.includes(r.tag));
    if (value == null) return rest;
    const r = rec(tag, value);
    if (at < 0) return first ? [r, ...rest] : [...rest, r];
    const out = [...rest];
    out.splice(Math.min(at, out.length), 0, r);
    return out;
}

/**
 * A name's records with `changes` made, every other record (tags this wallet
 * doesn't know among them) kept as it was. Changes, each optional:
 * `pay` { pk, height } (the MSS key payments go to: it replaces the name's
 * payment records, first among the records, as the first that pays wins),
 * `identity` (key hex), `kem` (hash hex), `text` (a line; '' drops it) and
 * `postage` (stamp bits; the standard drops the record). Returns hex.
 */
export function editRecords(list, changes = {}) {
    let out = (list || []).map(r => rec(r.tag, r.value));
    if (changes.pay !== undefined) {
        const p = changes.pay;
        out = setTag(out, TAG.MDS_KEY, p ? p.pk + p.height.toString(16).padStart(2, '0') : null, { first: true, tags: [TAG.MDS_KEY, TAG.MDS_ADDRESS] });
    }
    if (changes.identity !== undefined) out = setTag(out, TAG.IDENTITY, changes.identity);
    if (changes.kem !== undefined) out = setTag(out, TAG.KEM, changes.kem);
    if (changes.text !== undefined) out = setTag(out, TAG.TEXT, changes.text ? hexOf(enc.encode(changes.text)) : null);
    if (changes.postage !== undefined) {
        const bits = changes.postage;
        out = setTag(out, TAG.POSTAGE, bits == null || bits === POSTAGE_STANDARD ? null : bits.toString(16).padStart(2, '0'));
    }
    return encodeRecords(out);
}

/** What a name's records say, for the page, and which of it is this
 *  wallet's (`mine`: { pay: { pk }, identity, kem, keys: [pk] }). */
export function summarize(list, mine = {}) {
    const recs = list || [];
    const bytes = r => (typeof r.value === 'string' ? bytesOf(r.value) : Uint8Array.from(r.value));
    const parsed = recs.map(r => ({ tag: r.tag, value: bytes(r) }));
    const keys = new Set(mine.keys || []);
    const text = parsed.find(r => r.tag === TAG.TEXT);
    const first = parsed.find(r => (r.tag === TAG.MDS_KEY && r.value.length === 33) || (r.tag === TAG.MDS_ADDRESS && r.value.length === 32));
    let pays = null;
    if (first && first.tag === TAG.MDS_KEY) {
        const pk = hexOf(first.value.slice(0, 32));
        pays = { kind: 'key', key: pk, height: first.value[32], mine: keys.has(pk), current: !!(mine.pay && mine.pay.pk === pk) };
    } else if (first) {
        pays = { kind: 'address', address: hexOf(first.value), mine: false, current: false };
    }
    const id = identityOf(parsed);
    const kem = kemHash(parsed);
    const known = new Set([TAG.MDS_KEY, TAG.MDS_ADDRESS, TAG.IDENTITY, TAG.TEXT, TAG.KEM, TAG.POSTAGE]);
    return {
        text: text ? dec.decode(text.value) : '',
        postage: postageOf(parsed),
        pays,
        identity: id ? { key: id, mine: keys.has(id) } : null,
        kem: kem ? { hash: kem, mine: !!mine.kem && kem === mine.kem } : null,
        other: parsed.filter(r => !known.has(r.tag)).length,
    };
}

/** The records a name this wallet takes starts with: payments to its
 *  receive key, its identity and message keys, and what the page asked. */
export function freshRecords(mine, { text, postage } = {}) {
    return editRecords([], {
        pay: mine.pay || null,
        identity: mine.identity || null,
        kem: mine.kem || null,
        text: text || '',
        postage: postage ?? POSTAGE_STANDARD,
    });
}

/** When a spend sets a name to lapse: what `years` buy from the name's
 *  expiry, or from now if that has passed (MSN-1's `expires ≤ max(old
 *  expiry, height) + years × YEAR`); an edit leaves it. `height` is the
 *  node's, which the block the spend lands in is past. */
export function expiresFor(op, years, leaf, height) {
    if (op === 'edit') return leaf.expires;
    if (op === 'renew') return Math.max(leaf.expires, height) + years * YEAR;
    return height + years * YEAR;
}

/** Checks what the page asked for. */
function checkYears(years, least) {
    if (!Number.isInteger(years) || years < least || years > MAX_YEARS) throw new Stop(`A name is taken for ${least} to ${MAX_YEARS} years at a time.`);
}
function checkText(text) {
    if (text != null && enc.encode(String(text)).length > TEXT_MAX) throw new Stop(`What a name says is ${TEXT_MAX} bytes at most.`);
}
function checkPostage(bits) {
    if (bits != null && (!Number.isInteger(bits) || bits < POSTAGE_BITS[0] || bits > POSTAGE_BITS[1])) throw new Stop(`A price for strangers’ messages is ${POSTAGE_BITS[0]} to ${POSTAGE_BITS[1]} bits of work.`);
}

// ── the jobs ────────────────────────────────────────────────────────────────

/**
 * Deps (all required unless said):
 *   book                 { [name]: entry }, kept with the wallet
 *   answer(name)         a node's proof, checked (names.checkAnswer)
 *   coinExists(id)       whether a coin is unspent now
 *   mempoolSpends(coin, salt) whether a waiting reveal spends the registry coin
 *   plan(req)            registry.rs name_plan
 *   key(pk)              this wallet's control key with that public key, or null
 *   freshKey(name)       a new MSS key from the seed, kept: { address, pk, height, left }
 *   findKey(pk)          the wallet's MSS key with that public key, derived if need be, or null
 *   signControl(address, commitment) the key's signature (leaf picked and saved first)
 *   build(plan, label)   { ctx, ctxStr }: the wallet's coins added, nothing reserved
 *   reserve(ctx, label)  advance the key counters and lock the coins, saved
 *   commit(commitment)   proof of work done and the commit sent
 *   awaitCommit(c, say)  until the commit is mined
 *   buildReveal(ctxStr, commitment, salt) the signed reveal
 *   send(reveal)         { ok, body }
 *   awaitSpent(coin, say) until a coin is spent
 *   waitBlock()          until the next block
 *   lockExposed(ids, why) keep WOTS coins whose key signed a spend that can't land
 *   unlock(ids), touch(ids) release, or keep holding, coins for a spend
 *   mine()               { pay: {pk, height}, identity, kem, keys }
 *   withLock(label, fn)  one spend at a time, with the wallet's other spends
 *   save(), changed(), notice({ text, name }), log(text), now() (seconds)
 *   done(entry, job)     after a spend landed (rescan; optional)
 */
export class Registrar {
    constructor(d) {
        this.d = d;
        this.book = d.book;
        this.running = null;
        this.busy = null;       // { name, say } while a key is made or a name read
        this.stopped = false;
    }

    stop() { this.stopped = true; }

    entries() { return Object.values(this.book); }
    active(e) { return !!(e && e.job && ACTIVE.has(e.job.phase)); }

    async save() { await this.d.save(); this.d.changed(); }

    say(e, phase, text) {
        if (!e.job) return;
        e.job.phase = phase;
        e.job.say = text;
        this.d.changed();
    }

    // ── what the page reads ───────────────────────────────────────────────

    view() {
        const now = this.d.now();
        const names = this.entries().map((e) => {
            const tip = Math.max(e.height || 0, this.d.height() || 0);
            const left = e.expires - tip;
            const key = e.owner ? this.d.key(e.owner) : null;
            const j = e.job;
            return {
                name: e.name, shown: shown(e.name), tier: e.tier, expires: e.expires, height: tip,
                state: e.lost ? 'lost' : !e.held ? 'new' : left < 0 ? 'lapsed' : left < 30 * 1440 ? 'lapsing' : 'held',
                records: e.records ? summarize(e.records, this.d.mine()) : null,
                signatures: key ? key.left : null,
                job: j ? { op: j.op, phase: j.phase, say: j.say || '', error: j.error || '', years: j.years, burn: j.burn || null, sent: !!j.sent, made: j.made || null, cancellable: !j.reveal } : null,
                checked: e.checked || 0, fresh: now - (e.checked || 0) < REFRESH,
            };
        }).sort((a, b) => a.name.localeCompare(b.name));
        return { names, busy: this.busy, rates: RATES, years: YEAR_CHOICES };
    }

    /** What a name is now, for someone who might take it. */
    async look(input) {
        const name = normalize(input);
        const a = await this.d.answer(name);
        const st = standing(a);
        const ours = a.registered ? this.d.key(a.leaf.owner) : null;
        if (ours && !this.book[name]) {
            this.take(name, a, ours);
            await this.save();
        }
        return {
            name, shown: shown(name), state: st.state, tier: st.tier, rate: RATES[st.tier], expires: st.expires ?? null,
            height: a.height, deal: st.deal || null, mine: !!ours, held: !!this.book[name],
            records: a.registered ? summarize(a.records, this.d.mine()) : null,
        };
    }

    /** Keeps an entry for a name a key of this wallet holds. */
    take(name, a, key) {
        const e = this.book[name] || (this.book[name] = { name, job: null });
        Object.assign(e, this.fromAnswer(a), { owner: key.pk, held: true, lost: false });
        return e;
    }

    fromAnswer(a) {
        return {
            tier: a.leaf.tier, slot: a.slot, expires: a.leaf.expires, height: a.height,
            records: a.records.map(r => ({ tag: r.tag, value: valueHex(r.value) })),
            recordsHex: a.recordsHex, termsHex: a.termsHex, checked: this.d.now(),
        };
    }

    // ── what the page asks for ────────────────────────────────────────────

    /** Takes a name no one holds: registers one never registered, or claims
     *  one that lapsed (or a premium name no one has claimed yet). */
    async get(input, { years = 1, text = '', postage = POSTAGE_STANDARD } = {}) {
        const name = normalize(input);
        checkYears(years, 1);
        checkText(text);
        checkPostage(postage);
        const had = this.book[name];
        if (this.active(had)) throw new Stop(`${shown(name)} is already being worked on.`);
        const a = await this.d.answer(name);
        const st = standing(a);
        if (st.state === 'held') {
            if (this.d.key(a.leaf.owner)) throw new Stop(`This wallet already holds ${shown(name)}: renew it instead.`);
            throw new Stop(`Someone holds ${shown(name)} until block ${st.expires.toLocaleString('en-US')}.`);
        }
        const op = st.state === 'available' ? 'register' : 'claim';
        // A lapsed name of this wallet's is claimed back by its own key, while it has signatures to spare.
        let k = had && had.owner ? this.d.key(had.owner) : null;
        if (!k || k.left <= 2 * HANDOVER) {
            this.busy = { name, say: 'Making the name’s key…' };
            this.d.changed();
            try { k = await this.d.freshKey(name); } finally { this.busy = null; }
        }
        const records = freshRecords(this.d.mine(), { text, postage });
        const e = had || (this.book[name] = { name });
        // A lapsed name of this wallet's stays as it is until the claim lands.
        if (had && had.held && !had.lost) e.lost = false;
        else Object.assign(e, { tier: st.tier, held: false, lost: false, owner: null, expires: 0, height: a.height, records: null });
        e.job = this.newJob(op, { years, records, terms: NO_DEAL, owner: k.pk });
        await this.save();
        this.kick();
        return { name: shown(name), op };
    }

    /** Holds a name of this wallet's for `years` more. */
    async renew(input, years) {
        const e = this.mineOr(input);
        checkYears(years, 1);
        e.job = this.newJob('renew', { years, records: e.recordsHex, terms: e.termsHex || NO_DEAL, owner: e.owner });
        await this.save();
        this.kick();
        return { name: shown(e.name) };
    }

    /** Changes what a name of this wallet's says: `text`, `postage`, `pay`
     *  (true: payments to the wallet's receive key now), `messages` (true:
     *  this wallet's identity and message keys). */
    async edit(input, { text, postage, pay, messages } = {}) {
        const e = this.mineOr(input);
        checkText(text);
        checkPostage(postage);
        const mine = this.d.mine();
        const changes = {};
        if (text !== undefined) changes.text = text;
        if (postage !== undefined) changes.postage = postage;
        if (pay) {
            if (!mine.pay) throw new Stop('This wallet has no receive key for payments to go to.');
            changes.pay = mine.pay;
        }
        if (messages) {
            if (!mine.kem) throw new Stop('This build of the wallet has no sealed messages.');
            changes.identity = mine.identity;
            changes.kem = mine.kem;
        }
        const records = editRecords(e.records, changes);
        if (records.length / 2 > MAX_RECORDS) throw new Stop(`A name’s records hold ${MAX_RECORDS} bytes at most.`);
        if (records === e.recordsHex) throw new Stop('That is what it says already.');
        e.job = this.newJob('edit', { years: 0, records, terms: e.termsHex || NO_DEAL, owner: e.owner });
        e.job.changes = changes;
        await this.save();
        this.kick();
        return { name: shown(e.name) };
    }

    /** Finds a name one of this wallet's keys holds (a restored wallet's, or
     *  one taken in the desktop app), deriving the seed's next keys if need be. */
    async adopt(input) {
        const name = normalize(input);
        const a = await this.d.answer(name);
        if (!a.registered) throw new Stop(`No one holds ${shown(name)}.`);
        let k = this.d.key(a.leaf.owner);
        if (!k) {
            this.busy = { name, say: 'Looking through this wallet’s keys…' };
            this.d.changed();
            try { k = await this.d.findKey(a.leaf.owner); } finally { this.busy = null; }
        }
        this.d.changed();
        if (!k) throw new Stop(`None of this wallet’s keys holds ${shown(name)}.`);
        this.take(name, a, k);
        await this.save();
        return { name: shown(name) };
    }

    /** Drops a job that has signed nothing, between its tries. */
    async cancel(input) {
        const e = this.book[normalize(input)];
        if (!e || !e.job) return {};
        if (e.job.reveal) throw new Stop('It has been signed and sent, so it has to be seen through.');
        if (this.current === e.name) throw new Stop('It is being worked on this moment: stop it once it waits for the next block.');
        this.release(e.job);
        e.job = null;
        if (!e.held) delete this.book[e.name];
        await this.save();
        return {};
    }

    /** Tries a failed job again. */
    async retry(input) {
        const e = this.book[normalize(input)];
        if (!e || !e.job || e.job.phase !== 'failed') return {};
        Object.assign(e.job, { phase: 'queued', say: '', error: '', tries: 0 });
        await this.save();
        this.kick();
        return {};
    }

    /** Forgets a name this wallet no longer holds. */
    async forget(input) {
        const e = this.book[normalize(input)];
        if (!e) return {};
        if (this.active(e)) throw new Stop('It is being worked on.');
        delete this.book[e.name];
        await this.save();
        return {};
    }

    mineOr(input) {
        const e = this.book[normalize(input)];
        if (!e || !e.held) throw new Stop(`This wallet doesn’t hold ${shown(normalize(input))}.`);
        if (e.lost) throw new Stop(`${shown(e.name)} isn’t this wallet’s any more.`);
        if (this.active(e)) throw new Stop(`${shown(e.name)} is already being worked on.`);
        if (!this.d.key(e.owner)) throw new Stop(`This wallet doesn’t have the key that holds ${shown(e.name)}.`);
        return e;
    }

    newJob(op, { years, records, terms, owner }) {
        return { op, years, records, terms, owner, phase: 'queued', say: '', error: '', tries: 0, made: this.d.now() };
    }

    // ── seeing jobs through ───────────────────────────────────────────────

    /** Works on what waits, a job at a time; resolves when the pass is over.
     *  Asked again while a pass runs (a new job, a new block), the pass goes
     *  round once more. */
    kick() {
        if (this.stopped) return Promise.resolve();
        if (this.running) {
            this.again = true;
            return this.running;
        }
        this.running = (async () => {
            do {
                this.again = false;
                for (const e of this.entries()) {
                    if (this.stopped) return;
                    if (this.active(e)) await this.d.withLock(`${shown(e.name)}`, () => this.attempt(e));
                }
                for (const e of this.entries()) {
                    if (this.stopped) return;
                    if (e.held && !this.active(e) && this.d.now() - (e.checked || 0) > REFRESH) await this.refresh(e).catch(err => this.d.log(`${shown(e.name)}: ${errText(err)}`));
                }
            } while (this.again && !this.stopped);
        })().catch(err => this.d.log(`names: ${errText(err)}`)).finally(() => { this.running = null; });
        return this.running;
    }

    /** Whether a job waits on the chain. */
    waiting() { return this.entries().some(e => this.active(e)); }

    async refresh(e) {
        const a = await this.d.answer(e.name);
        e.checked = this.d.now();
        if (!a.registered || a.leaf.owner !== e.owner) {
            const k = a.registered ? this.d.key(a.leaf.owner) : null;
            if (k) Object.assign(e, this.fromAnswer(a), { owner: k.pk, lost: false });
            else if (!e.lost) {
                e.lost = true;
                this.d.notice({ name: e.name, text: `${shown(e.name)} isn’t this wallet’s any more: it lapsed, and someone else took it.` });
            }
        } else {
            Object.assign(e, this.fromAnswer(a), { lost: false });
        }
        await this.save();
    }

    async attempt(e) {
        const j = e.job;
        this.current = e.name;
        try {
            if (!j.commitment) await this.begin(e);
            if (e.job === j && j.commitment) await this.finish(e);
        } catch (err) {
            if (e.job !== j) return;
            if (err instanceof Stop) return this.fail(e, err.message);
            j.tries = (j.tries || 0) + 1;
            j.error = errText(err);
            this.d.log(`${shown(e.name)}: ${j.error}`);
            if (j.tries >= TRIES) return this.fail(e, j.error);
            this.say(e, 'waiting', 'Trying again at the next block.');
            await this.save();
        } finally {
            this.current = null;
        }
    }

    /** Whether the answer shows the job done: the leaf as the spend sets it. */
    landed(j, a) {
        return a.registered && j.expires != null && a.leaf.owner === j.owner && a.leaf.expires === j.expires
            && a.recordsHex === j.records && a.termsHex === j.terms;
    }

    /** Reads the name, plans the spend, builds and reserves it, and commits. */
    async begin(e) {
        const j = e.job;
        this.say(e, 'reading', 'Reading the registry…');
        const a = await this.d.answer(e.name);
        if (this.landed(j, a)) return this.complete(e, a);
        const st = standing(a);
        if (j.op === 'register' && a.registered) throw new Stop(`Someone registered ${shown(e.name)} first.`);
        if (j.op === 'claim' && st.state === 'held') throw new Stop(`Someone claimed ${shown(e.name)} first.`);
        if (j.op === 'claim' && st.state === 'available') throw new Stop(`${shown(e.name)} is not in the registry any more.`);
        let signer = null;
        if (j.op === 'renew' || j.op === 'edit') {
            if (!a.registered || a.leaf.owner !== e.owner) {
                e.lost = true;
                throw new Stop(`This wallet’s key no longer holds ${shown(e.name)}.`);
            }
            if (st.deal === 'rented') {
                const until = Number(new DataView(bytesOf(a.termsHex).buffer).getBigUint64(41, true));
                if (a.height < until) throw new Stop(`${shown(e.name)} is rented out until block ${until}: it can’t be changed before then.`);
            }
            signer = this.d.key(e.owner);
            if (!signer) throw new Stop(`This wallet doesn’t have the key that holds ${shown(e.name)}.`);
            if (signer.left < 1) throw new Stop(`The key that holds ${shown(e.name)} has no signatures left.`);
            // Close to its last signature: the name goes to the next key in
            // this same spend.
            if (j.handover == null) {
                j.handover = signer.left - 1 <= HANDOVER;
                if (j.handover) {
                    const next = await this.d.freshKey(e.name);
                    j.owner = next.pk;
                }
            }
        }
        if (!(await this.d.coinExists(a.coin))) throw new Error('The registry moved on while it was being read.');
        // A renewal keeps what the name says now, and an edit changes that.
        if (j.op === 'renew') j.records = a.recordsHex;
        if (j.op === 'edit' && j.changes) j.records = editRecords(a.records, j.changes);
        j.expires = expiresFor(j.op, j.years, a.leaf, a.height);
        const plan = this.d.plan({
            name: e.name, by: BY[j.op], slot: a.slot, leaf: a.leaf, proof: a.proof, root: a.root, terms: a.termsHex,
            thread: { coin: a.coin, salt: a.salt }, owner: j.owner, records: j.records, new_terms: j.terms,
            years: j.years, expires: j.expires, signer_height: signer ? signer.height : 0,
        });
        this.say(e, 'building', 'Building the transaction…');
        const label = `${{ register: 'Registering', claim: 'Claiming', renew: 'Renewing', edit: 'Changing' }[j.op]} ${shown(e.name)}`;
        const { ctx, ctxStr } = await this.d.build(plan, label);
        const ins = ctx.wallet_inputs || [];
        await this.d.reserve(ctx, label);
        Object.assign(j, {
            burn: plan.burn, fee: ctx.fee, tier: plan.tier, slot: plan.slot, thread: plan.thread_coin, threadSalt: plan.thread_salt,
            signAt: plan.sign_at, signer: signer ? signer.address : null,
            inputs: ins.map(i => i.coin_id), wots: ins.filter(i => !i.is_mss).map(i => i.coin_id),
            ctx: ctxStr, commitment: ctx.commitment, reveal: null, sent: false,
        });
        await this.save();
        this.say(e, 'committing', 'Doing the transaction’s proof of work…');
        try {
            await this.d.commit(ctx.commitment);
        } catch (err) {
            // Nothing is signed: the coins can go into the next try.
            this.release(j);
            Object.assign(j, { ctx: null, commitment: null });
            await this.save();
            throw err;
        }
    }

    /** From a commit sent: until the spend is in a block, and whose it was. */
    async finish(e) {
        const j = e.job;
        this.d.touch(j.inputs);
        this.say(e, 'committed', 'Waiting for a block (1 of 2)…');
        await this.d.awaitCommit(j.commitment, t => this.d.log(`${shown(e.name)}: ${t}`));
        let behind = 0;
        for (let k = 0; ; k++) {
            this.d.touch(j.inputs);
            if (!(await this.d.coinExists(j.thread))) return this.decide(e);
            if (!j.sent && await this.d.mempoolSpends(j.thread, j.threadSalt)) {
                // Another change to the registry waits for a block. If it
                // lands, this one starts over; nothing of it is signed yet.
                if (++behind > PATIENCE) throw new Error(`Another change to the registry has waited ${PATIENCE} blocks without landing.`);
                this.say(e, 'waiting', 'Another change to the registry is waiting for a block; this goes after it.');
                await this.d.waitBlock();
                k--;
                continue;
            }
            if (!j.reveal) {
                this.say(e, 'signing', 'Signing…');
                const ctx = JSON.parse(j.ctx);
                if (j.signAt != null) {
                    const sig = await this.d.signControl(j.signer, j.commitment);
                    const items = ctx.contract_inputs[0].witness.split(',');
                    if (sig.length !== items[j.signAt].length) throw new Stop('The control key’s signature is not the size set aside for it.');
                    items[j.signAt] = sig;
                    ctx.contract_inputs[0].witness = items.join(',');
                }
                j.reveal = this.d.buildReveal(JSON.stringify(ctx), j.commitment, ctx.tx_salt);
                await this.save();
            }
            // Counted as sent before it goes: a node may take it even if its
            // answer is lost on the way back.
            this.say(e, 'revealing', 'Sending…');
            j.sent = true;
            await this.save();
            const r = await this.d.send(j.reveal);
            const body = String((r && r.body) || '');
            if (r && (r.ok || /already|duplicate|exists/i.test(body))) break;
            this.d.log(`${shown(e.name)}: the node didn’t take the reveal: ${body}`);
            if (k + 1 >= SENDS) {
                if (!(await this.d.coinExists(j.thread))) return this.decide(e);
                throw new Stop(`The node won’t take the transaction: ${body || 'no reason given'}`);
            }
            await this.d.waitBlock();
        }
        this.say(e, 'confirming', 'Waiting for a block (2 of 2)…');
        await this.d.awaitSpent(j.thread, t => this.d.log(`${shown(e.name)}: ${t}`));
        return this.decide(e);
    }

    /** The registry coin is spent: by this spend (its fee coins are gone
     *  too), or by another, and this one starts over. */
    async decide(e) {
        const j = e.job;
        const ours = j.inputs.length ? !(await this.d.coinExists(j.inputs[0])) : false;
        if (ours) {
            let a = null;
            try { a = await this.d.answer(e.name); } catch (_) { /* the job says what it set */ }
            return this.complete(e, a && this.landed(j, a) ? a : null);
        }
        this.d.log(`${shown(e.name)}: another change to the registry got in first`);
        if (j.sent) this.exposed(j);
        this.release(j);
        Object.assign(j, { ctx: null, commitment: null, reveal: null, sent: false });
        j.tries = (j.tries || 0) + 1;
        if (j.tries >= TRIES) return this.fail(e, 'Other changes to the registry kept getting in first.');
        this.say(e, 'waiting', 'Another change to the registry got in first: trying again at the next block.');
        await this.save();
    }

    /** A reveal went out and can't land: its one-time keys signed once. */
    exposed(j) {
        const wots = (j.wots || []).filter(Boolean);
        if (wots.length) this.d.lockExposed(wots, { commitment: j.commitment });
        j.inputs = (j.inputs || []).filter(id => !wots.includes(id));
        j.wots = [];
    }

    release(j) {
        if (j.inputs && j.inputs.length) this.d.unlock(j.inputs);
        j.inputs = [];
        j.wots = [];
    }

    async complete(e, a) {
        const j = e.job;
        if (a) Object.assign(e, this.fromAnswer(a));
        else {
            // The node's index hasn't caught up: what the spend set, read again soon.
            Object.assign(e, {
                expires: j.expires, recordsHex: j.records, termsHex: j.terms, checked: 0,
                records: decodeList(j.records), tier: j.tier ?? e.tier, slot: j.slot ?? e.slot,
            });
        }
        Object.assign(e, { owner: j.owner, held: true, lost: false });
        e.last = { op: j.op, at: this.d.now(), burn: j.burn || 0 };
        e.job = null;
        await this.save();
        // The wallet rescans before the spent coins' locks go, so nothing
        // picks a coin this spend took.
        if (this.d.done) {
            try { await this.d.done(e, j); } catch (err) { this.d.log(`${shown(e.name)}: ${errText(err)}`); }
        }
        this.release(j);
        await this.save();
        const what = { register: 'is yours', claim: 'is yours', renew: 'is renewed', edit: 'is changed' }[j.op];
        this.d.notice({ name: e.name, text: `${shown(e.name)} ${what}.` });
    }

    async fail(e, why) {
        const j = e.job;
        // A reveal that went out may yet land, and then spends its coins
        // with the signature it carries; one that can't never will. Either
        // way its one-time keys have signed: they sign nothing else.
        if (j.sent) this.exposed(j);
        this.release(j);
        Object.assign(j, { phase: 'failed', error: why, say: '', ctx: null, commitment: null, reveal: null, sent: false });
        await this.save();
        this.d.notice({ name: e.name, text: `${shown(e.name)}: ${why}` });
    }
}

function decodeList(hex) {
    const b = bytesOf(hex);
    const out = [];
    for (let i = 0; i + 2 <= b.length;) {
        const len = b[i + 1];
        out.push({ tag: b[i], value: hexOf(b.slice(i + 2, i + 2 + len)) });
        i += 2 + len;
    }
    return out;
}

/** What taking a name costs for `years`, in base units. */
export const priceOf = (tier, years) => RATES[tier] * years;
