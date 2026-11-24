// node --test wasm-wallet/tests/
import test from 'node:test';
import assert from 'node:assert/strict';
import {
    Registrar, Stop, editRecords, summarize, freshRecords, expiresFor, HANDOVER, TRIES, SENDS,
} from '../registrar.js';
import { TAG, YEAR, RATES, encodeRecords, tierForLen } from '../names.js';

const hex = (b, n) => b.toString(16).padStart(2, '0').repeat(n);
const PAY = { pk: hex(0x11, 32), height: 10 };
const ID = hex(0x22, 32);
const KEM = hex(0x33, 32);
const MINE = { pay: PAY, identity: ID, kem: KEM, keys: [PAY.pk, ID] };
const decode = (h) => {
    const b = Buffer.from(h, 'hex');
    const out = [];
    for (let i = 0; i < b.length;) { out.push({ tag: b[i], value: new Uint8Array(b.subarray(i + 2, i + 2 + b[i + 1])) }); i += 2 + b[i + 1]; }
    return out;
};
const tick = () => new Promise(r => setImmediate(r));

/** A registry, a wallet's coins and a node, as the registrar sees them. */
class Chain {
    constructor({ height = 400_000, coins } = {}) {
        this.height = height;
        this.names = new Map();     // name → { leaf, recordsHex, termsHex }
        this.thread = 1;            // the registry coin is `t<thread>`
        this.coins = new Map((coins || [
            { coin_id: 'm1', is_mss: true, value: 2 ** 40 },
            { coin_id: 'w1', is_mss: false, value: 2 ** 40 },
        ]).map(c => [c.coin_id, { ...c, spent: false }]));
        this.plans = new Map();     // commitment → { req, ins }
        this.sent = [];
        this.signed = [];
        this.locked = new Set();
        this.exposed = [];
        this.commits = 0;
        this.reveals = 0;
        this.keys = new Map();      // address → key
        this.next = 0;              // the seed's next MSS index
        this.derived = 0;
        this.notices = [];
        this.before = null;         // (chain) => void, as a reveal goes
        this.reject = null;         // body to refuse reveals with
        this.mempool = false;
        this.preferMss = true;
    }
    coin() { return `t${this.thread}`; }
    hold(name, { owner, expires, tier = 3, records = '', terms = '00' }) {
        this.names.set(name, { leaf: { key: 'kk', next: 'nn', expires, tier, owner, data: 'dd', deal: 'ee' }, recordsHex: records, termsHex: terms });
    }
    answer(name) {
        const n = this.names.get(name);
        const base = { name, height: this.height, root: `r${this.thread}`, coin: this.coin(), salt: `s${this.thread}`, proof: 'pp', slot: 7 };
        if (!n) return { ...base, registered: false, records: [], recordsHex: '', termsHex: '00', expires: 0, leaf: { key: 'aa', next: 'zz', expires: 9, tier: 0, owner: '00'.repeat(32), data: 'dd', deal: 'ee' } };
        return { ...base, registered: true, records: decode(n.recordsHex), recordsHex: n.recordsHex, termsHex: n.termsHex, expires: n.leaf.expires, leaf: { ...n.leaf } };
    }
    /** Another wallet's change to the registry lands. */
    competitor() { this.thread++; }
    deps(book, extra = {}) {
        const c = this;
        return {
            book,
            answer: async name => c.answer(name),
            coinExists: async id => (id.startsWith('t') ? id === c.coin() : !!(c.coins.get(id) && !c.coins.get(id).spent)),
            mempoolSpends: async () => { const m = c.mempool; c.mempool = false; return m; },
            plan: (req) => {
                const tier = req.by === 'register' ? tierForLen(req.name.length) : c.names.get(req.name).leaf.tier;
                const burn = RATES[tier] * req.years;
                return {
                    items: ['aa', req.by === 'owner' ? '00'.repeat(4) : '00', 'bb'], sign_at: req.by === 'owner' ? 1 : null,
                    covenant: 'cc', registry: 'rr', thread_coin: req.thread.coin, thread_root: req.root, thread_salt: req.thread.salt,
                    root: 'new', tip_salt: 'ts', slot: 7, tier, burn, burn_payload: 'bp', burns: burn ? [burn] : [], req,
                };
            },
            key: pk => [...c.keys.values()].find(k => k.pk === pk) || null,
            // The seed's MSS keys: index i has public key `k<i>…`.
            freshKey: async () => {
                const i = c.next++;
                const k = { address: `a${i}`, pk: `k${i}`.padEnd(64, '0'), height: 10, left: 1024 };
                c.keys.set(k.address, k);
                return k;
            },
            findKey: async (pk) => {
                for (let i = c.next; i < c.next + 4; i++) {
                    c.derived++;
                    if (`k${i}`.padEnd(64, '0') === pk) {
                        const k = { address: `a${i}`, pk, height: 10, left: 1024 };
                        c.keys.set(k.address, k);
                        c.next = i + 1;
                        return k;
                    }
                }
                return null;
            },
            signControl: async (address, commitment) => {
                const k = c.keys.get(address);
                k.left--;
                c.signed.push({ address, commitment });
                return 'ff'.repeat(4);
            },
            build: async (plan) => {
                const free = [...c.coins.values()].filter(x => !x.spent && !c.locked.has(x.coin_id));
                const mss = free.filter(x => x.is_mss);
                const pool = c.preferMss && mss.length ? mss : free;
                if (!pool.length) throw new Stop('Not enough MDS.');
                const ins = [pool[0]];
                const commitment = `c${++c.commits}`;
                c.plans.set(commitment, { req: plan.req, ins: ins.map(i => i.coin_id), thread: plan.thread_coin });
                const ctx = { commitment, tx_salt: 'salt', fee: 100, wallet_inputs: ins.map(i => ({ coin_id: i.coin_id, is_mss: i.is_mss })), contract_inputs: [{ witness: plan.items.join(',') }] };
                return { ctx, ctxStr: JSON.stringify(ctx) };
            },
            reserve: async (ctx) => { for (const i of ctx.wallet_inputs) c.locked.add(i.coin_id); },
            commit: async () => {},
            awaitCommit: async () => {},
            buildReveal: (ctxStr, commitment) => { c.reveals++; return JSON.stringify({ commitment, witness: JSON.parse(ctxStr).contract_inputs[0].witness }); },
            send: async (reveal) => {
                const r = JSON.parse(reveal);
                c.sent.push(r);
                if (c.before) { const f = c.before; c.before = null; f(c); }
                if (c.reject) return { ok: false, body: c.reject };
                const p = c.plans.get(r.commitment);
                if (p.thread !== c.coin()) return { ok: false, body: 'Input coin not found (already spent)' };
                // It lands at once: the registry moves on and the coins go.
                const q = p.req;
                c.names.set(q.name, { leaf: { key: 'kk', next: 'nn', expires: q.expires, tier: (c.names.get(q.name) || { leaf: { tier: tierForLen(q.name.length) } }).leaf.tier, owner: q.owner, data: 'dd', deal: 'ee' }, recordsHex: q.records, termsHex: q.new_terms });
                c.thread++;
                for (const id of p.ins) c.coins.get(id).spent = true;
                return { ok: true };
            },
            awaitSpent: async () => {},
            waitBlock: async () => { c.height++; },
            lockExposed: (ids) => c.exposed.push(...ids),
            unlock: ids => ids.forEach(id => c.locked.delete(id)),
            touch: () => {},
            mine: () => MINE,
            withLock: async (label, fn) => fn(),
            save: async () => {},
            changed: () => {},
            notice: n => c.notices.push(n.text),
            log: () => {},
            now: () => 1_700_000_000,
            height: () => c.height,
            ...extra,
        };
    }
}

async function settle(r) {
    for (let i = 0; i < 20 && r.waiting(); i++) { await r.kick(); await tick(); }
}

test('records are changed tag by tag, and the ones this wallet doesn’t know are kept', () => {
    const list = decode(encodeRecords([
        { tag: TAG.MDS_ADDRESS, value: hex(0xaa, 32) },
        { tag: 77, value: 'beef' },
        { tag: TAG.TEXT, value: Buffer.from('old').toString('hex') },
        { tag: TAG.TEXT, value: Buffer.from('older').toString('hex') },
    ]));
    let out = decode(editRecords(list, { text: 'new', pay: PAY, postage: 28 }));
    assert.deepEqual(out.map(r => r.tag), [TAG.MDS_KEY, 77, TAG.TEXT, TAG.POSTAGE], 'payments first, the unknown tag kept, one text');
    assert.equal(Buffer.from(out[0].value).toString('hex'), PAY.pk + '0a');
    assert.equal(Buffer.from(out[2].value).toString(), 'new');
    assert.equal(out[3].value[0], 28);
    out = decode(editRecords(out, { text: '', postage: 24 }));
    assert.deepEqual(out.map(r => r.tag), [TAG.MDS_KEY, 77], 'no text, and the standard price needs no record');
    const fresh = summarize(decode(freshRecords(MINE, { text: 'hi', postage: 32 })), MINE);
    assert.equal(fresh.text, 'hi');
    assert.equal(fresh.postage, 32);
    assert.deepEqual(fresh.pays, { kind: 'key', key: PAY.pk, height: 10, mine: true, current: true });
    assert.deepEqual(fresh.identity, { key: ID, mine: true });
    assert.deepEqual(fresh.kem, { hash: KEM, mine: true });
    assert.equal(summarize(list, MINE).other, 1);
    assert.equal(summarize(list, MINE).pays.kind, 'address');
});

test('a spend holds a name for what its years buy, from its expiry or now', () => {
    const leaf = { expires: 500_000 };
    assert.equal(expiresFor('register', 2, leaf, 400_000), 400_000 + 2 * YEAR);
    assert.equal(expiresFor('claim', 1, { expires: 0 }, 400_000), 400_000 + YEAR);
    assert.equal(expiresFor('renew', 1, leaf, 400_000), 500_000 + YEAR, 'from its expiry while it holds');
    assert.equal(expiresFor('renew', 1, { expires: 300_000 }, 400_000), 400_000 + YEAR, 'from now once it lapsed');
    assert.equal(expiresFor('edit', 0, leaf, 400_000), 500_000);
});

test('a name never registered is registered, held by its own control key, and shows as this wallet’s', async () => {
    const chain = new Chain();
    const book = {};
    const r = new Registrar(chain.deps(book));
    const look = await r.look('Nick.mds');
    assert.equal(look.state, 'available');
    assert.equal(look.rate, RATES[2], 'four letters: silver');
    await r.get('nick', { years: 2, text: 'gm', postage: 28 });
    assert.equal(book.nick.job.op, 'register');
    await settle(r);
    const e = book.nick;
    assert.equal(e.job, null);
    assert.equal(e.held, true);
    assert.equal(e.owner, chain.keys.get('a0').pk, 'held by the seed’s next MSS key');
    assert.equal(e.expires, 400_000 + 2 * YEAR);
    const plan = chain.plans.get('c1').req;
    assert.equal(plan.by, 'register');
    assert.equal(plan.years, 2);
    assert.equal(plan.new_terms, '00');
    const recs = summarize(e.records, MINE);
    assert.equal(recs.text, 'gm');
    assert.equal(recs.postage, 28);
    assert.ok(recs.pays.current && recs.identity.mine && recs.kem.mine);
    assert.equal(chain.signed.length, 0, 'a registration needs no control signature');
    assert.equal(chain.exposed.length, 0);
    assert.equal(chain.coins.get('m1').spent, true, 'paid from the MSS coin');
    assert.match(chain.notices.at(-1), /nick\.mds is yours/);
    const v = r.view().names[0];
    assert.equal(v.state, 'held');
    assert.equal(v.signatures, 1024);
});

test('a race lost before anything is signed starts over, and locks nothing', async () => {
    const chain = new Chain();
    const book = {};
    const r = new Registrar(chain.deps(book, {
        awaitCommit: async () => { if (chain.commits === 1) chain.competitor(); },
    }));
    await r.get('nick', { years: 1 });
    await settle(r);
    assert.equal(book.nick.held, true);
    assert.equal(chain.commits, 2, 'built again from a fresh proof');
    assert.equal(chain.reveals, 1, 'and signed once');
    assert.equal(chain.exposed.length, 0);
    assert.equal(chain.plans.get('c2').req.thread.coin, 't2');
});

test('a race lost after the reveal went out locks its WOTS coins, not its MSS ones', async () => {
    const chain = new Chain({ coins: [{ coin_id: 'w1', is_mss: false, value: 2 ** 40 }, { coin_id: 'w2', is_mss: false, value: 2 ** 40 }] });
    chain.preferMss = false;
    const book = {};
    const r = new Registrar(chain.deps(book));
    chain.before = c => c.competitor();
    await r.get('nick', { years: 1 });
    await settle(r);
    assert.deepEqual(chain.exposed, ['w1'], 'w1 signed a spend that can’t land');
    assert.equal(book.nick.held, true, 'the next try landed');
    assert.deepEqual(chain.plans.get('c2').ins, ['w2'], 'and was paid from another coin');
});

test('a renewal is signed by the control key, in the room set aside for it', async () => {
    const chain = new Chain();
    const book = {};
    const r = new Registrar(chain.deps(book));
    await r.get('nick', { years: 1 });
    await settle(r);
    const was = book.nick.expires;
    chain.height += 100;
    await r.renew('nick', 3);
    await settle(r);
    assert.equal(book.nick.expires, was + 3 * YEAR);
    assert.equal(chain.signed.length, 1);
    const last = chain.sent.at(-1);
    assert.equal(last.witness.split(',')[1], 'ff'.repeat(4), 'the signature replaced its placeholder');
    assert.equal(chain.plans.get(last.commitment).req.by, 'owner');
    assert.equal(chain.plans.get(last.commitment).req.signer_height, 10);
});

test('an edit keeps the expiry and burns nothing; a key near its last signature hands the name on', async () => {
    const chain = new Chain({ coins: ['m1', 'm2', 'm3'].map(coin_id => ({ coin_id, is_mss: true, value: 2 ** 40 })) });
    const book = {};
    const r = new Registrar(chain.deps(book));
    await r.get('nick', { years: 1 });
    await settle(r);
    const was = book.nick.expires;
    await r.edit('nick', { text: 'new words', postage: 32 });
    await settle(r);
    assert.equal(book.nick.expires, was);
    assert.equal(summarize(book.nick.records, MINE).text, 'new words');
    assert.equal(chain.plans.get(chain.sent.at(-1).commitment).req.years, 0);
    await assert.rejects(r.edit('nick', { text: 'new words', postage: 32 }), /what it says already/);
    chain.keys.get('a0').left = HANDOVER + 1;
    await r.edit('nick', { text: 'handed on' });
    await settle(r);
    assert.equal(book.nick.job, null, JSON.stringify(book.nick.job));
    assert.equal(book.nick.owner, chain.keys.get('a1').pk, 'a fresh key holds it now');
    assert.equal(chain.signed.at(-1).address, 'a0', 'the old key signed the handover');
});

test('a lapsed name is claimed without a signature; a held one is refused', async () => {
    const chain = new Chain();
    chain.hold('gone', { owner: hex(0x99, 32), expires: 399_000, tier: 2 });
    chain.hold('kept', { owner: hex(0x99, 32), expires: 900_000 });
    const book = {};
    const r = new Registrar(chain.deps(book));
    assert.equal((await r.look('gone')).state, 'lapsed');
    await r.get('gone', { years: 1 });
    await settle(r);
    assert.equal(book.gone.held, true);
    const req = chain.plans.get('c1').req;
    assert.equal(req.by, 'lapse');
    assert.equal(req.expires, 400_000 + YEAR);
    assert.equal(chain.plans.get('c1').req.years, 1);
    await assert.rejects(r.get('kept', { years: 1 }), /Someone holds kept\.mds/);
    await assert.rejects(r.get('x', { years: 0 }), Stop);
    await assert.rejects(r.get('x', { years: 11 }), Stop);
    await assert.rejects(r.get('x', { text: 'é'.repeat(200) }), /255 bytes/);
});

test('someone else registering first ends the job, and says so', async () => {
    const chain = new Chain();
    const book = {};
    const r = new Registrar(chain.deps(book, {
        commit: async () => {},
        awaitCommit: async () => { chain.hold('nick', { owner: hex(0x99, 32), expires: 900_000 }); chain.competitor(); },
    }));
    await r.get('nick', { years: 1 });
    await settle(r);
    assert.equal(book.nick.job.phase, 'failed');
    assert.match(book.nick.job.error, /registered nick\.mds first/);
    assert.equal(chain.locked.size, 0, 'its coins are free again');
    await r.cancel('nick');
    assert.equal(book.nick, undefined);
});

test('a job committed before a reload is seen through without signing again', async () => {
    const chain = new Chain();
    const book = {};
    let r = new Registrar(chain.deps(book, { send: async () => { throw new Error('the page closed'); } }));
    await r.get('nick', { years: 1 });
    await r.kick();
    const j = book.nick.job;
    assert.ok(j.commitment && j.reveal, 'signed, and not sent');
    assert.equal(j.phase, 'waiting');
    const reveal = j.reveal;
    // The page opens again: the same book, a new registrar.
    r = new Registrar(chain.deps(book));
    await settle(r);
    assert.equal(book.nick.held, true);
    assert.equal(chain.reveals, 1, 'the reveal made before is the one sent');
    assert.equal(JSON.stringify(chain.sent.at(-1)), reveal);
});

test('a reveal the node keeps refusing ends the job after a few tries and locks its one-time keys', async () => {
    const chain = new Chain({ coins: [{ coin_id: 'w1', is_mss: false, value: 2 ** 40 }] });
    chain.preferMss = false;
    chain.reject = 'Script failed';
    const book = {};
    const r = new Registrar(chain.deps(book));
    await r.get('nick', { years: 1 });
    await settle(r);
    assert.equal(book.nick.job.phase, 'failed');
    assert.match(book.nick.job.error, /won’t take the transaction: Script failed/);
    assert.equal(chain.sent.length, SENDS);
    assert.deepEqual(chain.exposed, ['w1']);
    await r.cancel('nick');
    assert.equal(book.nick, undefined, 'a failed job can be put away');
});

test('transient failures are retried, then given up on', async () => {
    const chain = new Chain();
    const book = {};
    const r = new Registrar(chain.deps(book, { answer: async () => { throw new Error('no node answered'); } }));
    await assert.rejects(r.get('nick', { years: 1 }), /no node answered/);
    let calls = 0;
    const r2 = new Registrar(chain.deps(book, { answer: async name => { if (calls++) throw new Error('no node answered'); return chain.answer(name); } }));
    await r2.get('nick', { years: 1 });
    for (let i = 0; i < TRIES + 2; i++) await r2.kick();
    assert.equal(book.nick.job.phase, 'failed');
    assert.equal(book.nick.job.tries, TRIES);
    await r2.retry('nick');
    assert.notEqual(book.nick.job.phase, 'failed', 'it is tried again');
    assert.equal(book.nick.job.error, '');
});

test('a restored wallet, or the desktop app’s, finds the names its keys hold', async () => {
    const chain = new Chain();
    chain.hold('mine', { owner: 'k2'.padEnd(64, '0'), expires: 900_000 });
    chain.hold('theirs', { owner: hex(0x99, 32), expires: 900_000 });
    const book = {};
    const r = new Registrar(chain.deps(book));
    await r.adopt('mine');
    assert.equal(book.mine.owner, 'k2'.padEnd(64, '0'));
    assert.equal(book.mine.held, true);
    assert.ok(chain.keys.has('a2'), 'the key that holds it is kept');
    await assert.rejects(r.adopt('theirs'), /None of this wallet’s keys/);
    await assert.rejects(r.adopt('nobody'), /No one holds/);
    // A name looked up whose key the wallet already has is taken in.
    delete book.mine;
    const look = await r.look('mine');
    assert.equal(look.mine, true);
    assert.equal(book.mine.held, true);
});

test('a name someone else took after it lapsed shows as lost', async () => {
    const chain = new Chain();
    const book = {};
    const r = new Registrar(chain.deps(book));
    await r.get('nick', { years: 1 });
    await settle(r);
    chain.hold('nick', { owner: hex(0x99, 32), expires: 2_000_000 });
    book.nick.checked = 0;
    await r.kick();
    assert.equal(book.nick.lost, true);
    assert.equal(r.view().names[0].state, 'lost');
    await assert.rejects(r.renew('nick', 1), /isn’t this wallet’s/);
    await r.forget('nick');
    assert.equal(book.nick, undefined);
});

test('a job asked for while another is finishing starts in the same pass', async () => {
    const chain = new Chain({ coins: ['m1', 'm2'].map(coin_id => ({ coin_id, is_mss: true, value: 2 ** 40 })) });
    const book = {};
    let r;
    let asked = null;
    r = new Registrar(chain.deps(book, {
        // As the registration lands, the page asks for an edit.
        done: async () => { if (!asked) asked = r.edit('nick', { text: 'straight after' }).catch(e => e); },
    }));
    await r.get('nick', { years: 1 });
    await r.running;
    await asked;
    if (r.running) await r.running;
    assert.equal(book.nick.job, null, 'the edit ran without another kick');
    assert.equal(summarize(book.nick.records, MINE).text, 'straight after');
});

test('a renewal keeps what the name says on chain, and an edit changes that', async () => {
    const chain = new Chain({ coins: ['m1', 'm2', 'm3'].map(coin_id => ({ coin_id, is_mss: true, value: 2 ** 40 })) });
    const book = {};
    const r = new Registrar(chain.deps(book));
    await r.get('nick', { years: 1, text: 'first' });
    await settle(r);
    // The wallet's copy goes stale: the chain says something else.
    const onChain = editRecords(decode(chain.names.get('nick').recordsHex), { text: 'said elsewhere', postage: 32 });
    chain.names.get('nick').recordsHex = onChain;
    await r.renew('nick', 1);
    await settle(r);
    assert.equal(chain.plans.get(chain.sent.at(-1).commitment).req.records, onChain, 'renewed as it is');
    await r.edit('nick', { postage: 36 });
    await settle(r);
    const now = summarize(book.nick.records, MINE);
    assert.equal(now.text, 'said elsewhere', 'the edit changed only what it was asked to');
    assert.equal(now.postage, 36);
});

test('claiming a lapsed name of this wallet’s keeps it listed as this wallet’s meanwhile', async () => {
    const chain = new Chain();
    const book = {};
    const r = new Registrar(chain.deps(book));
    await r.get('nick', { years: 1 });
    await settle(r);
    chain.height += 2 * YEAR;
    chain.hold('nick', { ...chain.names.get('nick').leaf, records: chain.names.get('nick').recordsHex, expires: book.nick.expires });
    assert.equal((await r.look('nick')).state, 'lapsed');
    // Another spend holds the wallet: the claim waits its turn.
    const r2 = new Registrar(chain.deps(book, { withLock: () => new Promise(() => {}) }));
    await r2.get('nick', { years: 1 });
    assert.equal(book.nick.job.op, 'claim');
    assert.equal(book.nick.held, true);
    assert.equal(r2.view().names[0].state, 'lapsed');
    await r2.cancel('nick');
    assert.equal(book.nick.job, null);
    assert.equal(book.nick.held, true, 'putting the claim away keeps the name');
});
