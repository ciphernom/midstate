// The sealed-message engine (messages.js): wallets talking over a pretend
// chat bus, with stand-in crypto that keeps midstate::sealed's shape (an
// opening's key, a route per direction and number, a stamp bound to its key,
// a signature bound to the encapsulation). The real crypto, and the chat
// bus's proof of work, are tested in Rust (src/messages.rs); this tests what
// the page keeps and decides.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
    Messenger, newBook, loadBook, isRequest, admits, makeRoom, trim, unsend, prune, view, checkedText,
    envelopeOf, attachmentsFor, parseFrame, kemRequest, kemReply, textHex, hexText,
    CMD_KEM, CMD_KEM_REQ, REQUEST_LINES, OUTBOX_PATIENCE, POST_PATIENCE, MAX_LINES, KEM_ANSWERS_PER_BLOCK, CHUNK_HEX,
} from '../messages.js';
import { TAG } from '../names.js';

const sha = s => createHash('sha256').update(String(s)).digest('hex');
const jhex = o => Buffer.from(JSON.stringify(o)).toString('hex');
const unjhex = h => { try { return JSON.parse(Buffer.from(h, 'hex').toString()); } catch (_) { return null; } };
const idBytes = ct => Array.from(Buffer.from(sha('id' + ct), 'hex'));
const sleep0 = () => new Promise(r => setTimeout(r, 0));

/** The bus, the registry and the clock the wallets share. */
class World {
    constructor() {
        this.time = 1_800_000_000;
        this.height = 1000;
        this.bus = [];
        this.names = new Map();
        this.seq = 0;
        this.down = false;
    }
    register(name, recs) { this.names.set(name, recs); }
}

const rec = (tag, hex) => ({ tag, value: Uint8Array.from(Buffer.from(hex, 'hex')) });
const recBits = (tag, b) => ({ tag, value: Uint8Array.of(b) });

/** midstate::sealed, in miniature: hex JSON instead of ciphertext. */
function fakeCrypto(world, who) {
    const pub = Buffer.from(`kem public of ${who}`).toString('hex');
    const routeOf = (k, dir, n) => sha(`route:${k}:${dir}:${n}`).slice(0, 32);
    const stampKey = (k, fromOpener, n) => sha(`stamp:${k}:${fromOpener ? 'o' : 'r'}:${n}`);
    const bitsOf = (key, nonce) => {
        const v = BigInt(nonce);
        return (v >> 32n).toString(16).padStart(8, '0') === sha(key).slice(0, 8) ? Number(v & 0xffffffffn) : 0;
    };
    const fake = {
        slow: false,
        signed: 0,
        kemPublic: () => pub,
        kemHash: () => sha(pub),
        hash: h => sha(h),
        random: n => sha(`${who}:${world.seq++}`).slice(0, 2 * n),
        checksum: h => h + '00000000',
        checkPublic: (p, h) => { if (sha(p) !== h) throw new Error('not the key the name lists'); },
        async prepare(p, h, as) {
            if (sha(p) !== h) throw new Error('that KEM key is not the one the name publishes');
            const ct = fake.random(16);
            if (as) fake.signed++;
            return { ct, to: h, name: as ? as.name : '', identity: as ? as.identity : '', sig: as ? `sig:${as.name}:${as.identity}:${ct}` : '' };
        },
        preparedKey: p => sha(`stamp-open:${p.to}:${p.ct}`),
        preparedRecipient: p => p.to,
        finish(p, body, ttl, nonce) {
            const opening = jhex({ t: 'open', ct: p.ct, to: p.to, name: p.name, identity: p.identity, sig: p.sig, body, ttl, stamp: String(nonce) });
            return { opening, session: { id: idBytes(p.ct), k: p.ct, opener: true, sent: 1, received: 0, opened: [] } };
        },
        tryOpen(env) {
            const o = unjhex(env);
            if (!o || o.t !== 'open' || o.to !== sha(pub)) return null;
            return {
                session: { id: idBytes(o.ct), k: o.ct, opener: false, sent: 1, received: 0, opened: [] },
                name: o.name, identity: o.identity, signature: o.sig, transcript: o.ct,
                body: o.body, ttl: o.ttl, stamp_bits: bitsOf(fake.preparedKey(o), o.stamp),
            };
        },
        reserve: s => ({ session: { ...s, sent: s.sent + 1 }, n: s.sent, key: stampKey(s.k, s.opener, s.sent) }),
        stampKey: (s, n) => stampKey(s.k, s.opener, n),
        seal(s, n, body, ttl, nonce) {
            if (n === 0 || n >= s.sent) throw new Error(`message ${n} was not reserved`);
            return routeOf(s.k, s.opener ? 'to-recipient' : 'to-opener', n) + jhex({ t: 'msg', body, ttl, stamp: String(nonce) });
        },
        routes(s) {
            const dir = s.opener ? 'to-opener' : 'to-recipient';
            const out = [];
            for (let n = Math.max(1, s.received - 16); n <= s.received + 16; n++) out.push([routeOf(s.k, dir, n), n]);
            return out;
        },
        open(s, n, env) {
            const dir = s.opener ? 'to-opener' : 'to-recipient';
            if (env.slice(0, 32) !== routeOf(s.k, dir, n) || s.opened.includes(n)) return null;
            const o = unjhex(env.slice(32));
            if (!o || o.t !== 'msg') return null;
            return {
                session: { ...s, received: Math.max(s.received, n), opened: [...s.opened, n].slice(-33) },
                body: o.body, ttl: o.ttl, stamp_bits: bitsOf(stampKey(s.k, !s.opener, n), o.stamp),
            };
        },
        stampSearch(key, bits) {
            if (fake.slow) return undefined;
            return (BigInt('0x' + sha(key).slice(0, 8)) << 32n) | BigInt(bits);
        },
        verifySender: (name, identity, sig, transcript, ofName) => !!ofName && identity === ofName && sig === `sig:${name}:${identity}:${transcript}`,
    };
    return fake;
}

/** A wallet: its messenger, its book, what it was told. */
function wallet(world, who, { identity = null, book = newBook() } = {}) {
    const crypto = fakeCrypto(world, who);
    const w = { who, identity, book, crypto, notices: [], saves: 0 };
    w.m = new Messenger({
        book,
        crypto,
        chain: {
            height: () => world.height,
            records: async name => { if (world.offline) throw new Error('offline'); return world.names.get(name) || null; },
            history: async () => world.bus.slice(-100),
            post: async (words, attachments) => {
                if (world.down) throw new Error('the node is down');
                world.bus.push({ sender: sha(`sender${world.seq++}`), timestamp: world.time, nonce: world.seq++, reply_to: null, words, attachments });
            },
        },
        identities: () => (identity ? [identity] : []),
        save: async () => { w.saves++; },
        changed: () => {},
        notice: n => w.notices.push(n.text),
        log: () => {},
        now: () => world.time,
    });
    return w;
}

/** Lets a wallet finish what it is doing: its book, stamps and chats. */
async function idle(m) {
    for (let i = 0; i < 2000; i++) {
        await m.queue;
        await sleep0();
        // A stamp held up on purpose (crypto.slow) is left to run.
        const stamps = () => m.stampsRunning && !m.c.slow;
        if (!stamps() && !m.postsRunning && !m.ticking) {
            await m.queue;
            await sleep0();
            if (!stamps() && !m.postsRunning) return;
        }
    }
    throw new Error('never settled');
}

/** Rounds of every wallet reading the bus. */
async function run(...ws) {
    for (let round = 0; round < 6; round++) {
        for (const w of ws) {
            await w.m.tick();
            await idle(w.m);
        }
    }
}

/** alice.mds and bob.mds, both writing as their names; bob asks 28 bits. */
async function pair(world, { bobPostage = 28 } = {}) {
    const alice = wallet(world, 'alice', { identity: sha('alice id') });
    const bob = wallet(world, 'bob', { identity: sha('bob id') });
    world.register('alice', [rec(TAG.IDENTITY, alice.identity), rec(TAG.KEM, alice.crypto.kemHash())]);
    world.register('bob', [rec(TAG.IDENTITY, bob.identity), rec(TAG.KEM, bob.crypto.kemHash()), ...(bobPostage ? [recBits(TAG.POSTAGE, bobPostage)] : [])]);
    await alice.m.setName('alice.mds');
    await bob.m.setName('bob');
    return { alice, bob };
}

const texts = c => c.lines.filter(l => l.timer == null).map(l => `${l.mine ? '>' : '<'}${l.text}`);

test('a name writes to a name: the key is asked, the opening stamped at their price, both sides verified', async () => {
    const world = new World();
    const { alice, bob } = await pair(world);
    assert.deepEqual(await alice.m.send('Bob.mds', 'hello bob'), { key: 'to:bob' });
    await idle(alice.m);
    // Asked over the bus: bob's identity, the request id in the coin id.
    const ask = world.bus.find(m => m.words[1] === CMD_KEM_REQ);
    assert.equal(parseFrame(ask).address, bob.identity);
    assert.equal(Object.keys(alice.book.kemReqs).length, 1);
    assert.equal(alice.m.state().chats[0].key, 'to:bob', 'waiting for its key, shown as one conversation');
    assert.ok(alice.m.state().chats[0].lines[0].pending);

    await run(bob, alice, bob);
    assert.ok(world.bus.some(m => m.words[1] === CMD_KEM), 'bob’s wallet handed over its key');
    assert.equal(alice.crypto.signed, 1, 'one signature opens the session');
    const [b] = bob.book.chats;
    assert.equal(b.peer, 'alice');
    assert.ok(b.verified, 'alice’s claim checks out against the registry');
    assert.ok(!isRequest(b), 'a verified name reaches the inbox');
    assert.deepEqual(texts(b), ['<hello bob']);
    assert.deepEqual(bob.notices, ['New sealed message from alice.mds.']);
    const [a] = alice.book.chats;
    assert.ok(a.verified && !a.heard && a.peer === 'bob');
    assert.equal(alice.book.outbox.length, 0);

    // Until bob writes back, alice's messages carry bob's price.
    assert.equal(alice.m.state().chats[0].stampBits, 28);
    await alice.m.send('bob', 'are you there?');
    await run(alice, bob);
    assert.deepEqual(texts(bob.book.chats[0]), ['<hello bob', '<are you there?']);
    assert.equal(bob.m.state().unread, 2);
    await bob.m.read(b.id);
    assert.equal(bob.m.state().unread, 0);

    await bob.m.reply(b.id, 'hi alice');
    await run(bob, alice);
    const a2 = alice.book.chats[0];
    assert.ok(a2.heard, 'bob wrote back');
    assert.equal(alice.m.state().chats[0].stampBits, 0, 'no more stamps');
    await alice.m.send('bob', 'great');
    await run(alice, bob);
    assert.deepEqual(texts(bob.book.chats[0]), ['<hello bob', '<are you there?', '>hi alice', '<great']);
    assert.deepEqual(texts(alice.book.chats[0]), ['>hello bob', '>are you there?', '<hi alice', '>great']);
    // Nothing on the bus says who sent what.
    for (const m of world.bus) assert.ok(m.sender !== alice.identity && m.sender !== bob.identity);
    // Read twice, kept once.
    await bob.m.ingest(world.bus);
    await run(bob);
    assert.equal(bob.book.chats[0].lines.length, 4);
});

test('a stranger pays to be heard and waits under Requests, which hold three of its messages', async () => {
    const world = new World();
    const { bob } = await pair(world, { bobPostage: 0 });
    const carol = wallet(world, 'carol');
    await carol.m.send('bob', 'hi, it’s carol');
    await run(carol, bob, carol, bob);
    const [c] = bob.book.chats;
    assert.equal(c.peer, '', 'anonymous');
    assert.ok(isRequest(c));
    assert.deepEqual(bob.notices, [], 'a request raises no notice');
    for (const t of ['two', 'three', 'four']) await carol.m.send('bob', t);
    await run(carol, bob);
    assert.deepEqual(texts(bob.book.chats[0]), ['<hi, it’s carol', '<two', '<three'], 'three at most');
    assert.equal(bob.m.state().requests, 1);
    assert.equal(bob.m.state().unread, 0, 'requests are not counted as unread');

    await bob.m.accept(c.id);
    await carol.m.send('bob', 'five');
    await run(carol, bob);
    assert.ok(!isRequest(bob.book.chats[0]));
    assert.deepEqual(texts(bob.book.chats[0]).slice(-1), ['<five']);
});

test('the price: what a stranger’s message must show, and what is dropped without it', () => {
    const c = { accepted: false, session: { opener: false }, peer: '', verified: false, lines: [] };
    assert.ok(!admits(c, { stamp_bits: 23 }, true, 24));
    assert.ok(admits(c, { stamp_bits: 24 }, true, 24));
    c.lines = Array.from({ length: REQUEST_LINES }, () => ({ mine: false }));
    assert.ok(!admits(c, { stamp_bits: 30 }, true, 24), 'a request is full');
    c.accepted = true;
    assert.ok(admits(c, { stamp_bits: 0 }, false, 24), 'accepted: no stamp needed');
});

test('an opening without the stamp its recipient asks is dropped', async () => {
    const world = new World();
    const { alice, bob } = await pair(world, { bobPostage: 32 });
    await alice.m.send('bob', 'cheap');
    await idle(alice.m);
    // Bob raises his price; alice still reads the old one, and stamps short of it.
    const bobNow = [rec(TAG.IDENTITY, bob.identity), rec(TAG.KEM, bob.crypto.kemHash()), recBits(TAG.POSTAGE, 36)];
    world.register('bob', bobNow);
    bob.m.records_.clear();
    const realRecords = alice.m.records.bind(alice.m);
    const stale = [rec(TAG.IDENTITY, bob.identity), rec(TAG.KEM, bob.crypto.kemHash()), recBits(TAG.POSTAGE, 32)];
    alice.m.records = async (name, fresh) => (name === 'bob' ? stale : realRecords(name, fresh));
    await run(bob, alice, bob);
    assert.ok(alice.book.chats.length === 1 && alice.book.outbox.length === 0, 'alice sent it');
    assert.equal(bob.book.chats.length, 0, 'bob dropped it unread');
    // Once alice reads what bob asks now, a new conversation carries it.
    alice.m.records = realRecords;
    alice.m.records_.clear();
    await alice.m.remove(alice.book.chats[0].id);
    await alice.m.send('bob', 'dear');
    assert.equal(alice.book.prices.bob, 36);
    await run(alice, bob);
    assert.deepEqual(texts(bob.book.chats[0]), ['<dear']);
});

test('blocking a name drops it; anonymous senders can be dropped too', async () => {
    const world = new World();
    const { alice, bob } = await pair(world, { bobPostage: 0 });
    await alice.m.send('bob', 'one');
    await run(alice, bob, alice, bob);
    const id = bob.book.chats[0].id;
    assert.deepEqual(await bob.m.blockChat(id), { blocked: 'alice.mds' });
    assert.equal(bob.book.chats.length, 0);
    // Alice starts again: bob reads none of it.
    await alice.m.remove(alice.book.chats[0].id);
    await alice.m.send('bob', 'two');
    await run(alice, bob, alice, bob);
    assert.equal(bob.book.chats.length, 0, 'nothing from a blocked name');
    await assert.rejects(bob.m.unblock('carol'), /carol.mds is not blocked/);
    await bob.m.unblock('alice.mds');
    assert.deepEqual(bob.book.blocked, []);

    const carol = wallet(world, 'carol');
    await bob.m.setDropAnonymous(true);
    await carol.m.send('bob', 'anyone?');
    await run(carol, bob, carol, bob);
    assert.ok(!bob.book.chats.some(c => c.peer === ''), 'anonymous senders dropped');

    // Unblocked, alice is read again once she opens a new conversation.
    await alice.m.remove(alice.book.chats[0].id);
    await alice.m.send('bob', 'three');
    await run(alice, bob, alice, bob);
    const again = bob.book.chats.find(c => c.peer === 'alice');
    assert.deepEqual(texts(again), ['<three']);
    await bob.m.blockChat(again.id);
    await assert.rejects(bob.m.send('alice', 'hey'), /You blocked alice.mds/);
});

test('timers: set by either side, carried by every message, run out on both', async () => {
    const world = new World();
    const { alice, bob } = await pair(world, { bobPostage: 0 });
    await alice.m.send('bob', 'hello');
    await run(alice, bob, alice, bob);
    const a = alice.book.chats[0];
    await alice.m.setTimer(a.id, 3_600);
    await assert.rejects(alice.m.setTimer(a.id, 60), /A timer is off/);
    await alice.m.send('bob', 'gone in an hour');
    await run(alice, bob);
    const b = bob.book.chats[0];
    assert.equal(b.ttl, 3_600, 'bob’s wallet learnt the timer');
    assert.ok(b.lines.some(l => l.timer === 3_600 && !l.mine), 'and says who set it');
    world.time += 3_601;
    await run(alice, bob);
    assert.deepEqual(texts(bob.book.chats[0]), ['<hello']);
    assert.deepEqual(texts(alice.book.chats[0]), ['>hello']);
    // A message's own timer: bob replies under it, and turns it off.
    await bob.m.setTimer(b.id, 0);
    await run(bob, alice);
    assert.equal(alice.book.chats[0].ttl, 0);
});

test('what waits can be stopped: a message on its stamp, or one waiting for a key', async () => {
    const world = new World();
    const { alice, bob } = await pair(world);
    await alice.m.send('bob', 'first');
    await run(alice, bob, alice);
    const a = alice.book.chats[0];
    alice.crypto.slow = true;
    await alice.m.send('bob', 'second');
    await alice.m.send('bob', 'third');
    await idle(alice.m);
    assert.equal(alice.m.state().chats[0].lines.filter(l => l.stamping).length, 2);
    assert.equal(a.session.sent, 3);
    assert.equal(await alice.m.unsendChat(a.id).then(() => a.lines.length), 1);
    assert.equal(a.session.sent, 1, 'the numbers are given back');
    await assert.rejects(alice.m.unsendChat(a.id), /Nothing in that conversation/);
    alice.crypto.slow = false;

    // dave.mds's wallet never answers.
    const dave = sha('dave id');
    world.register('dave', [rec(TAG.IDENTITY, dave), rec(TAG.KEM, sha('dave kem'))]);
    await alice.m.send('dave', 'hi dave');
    await alice.m.cancel('dave.mds');
    assert.equal(alice.book.outbox.length, 0);
    await assert.rejects(alice.m.cancel('dave'), /Nothing is waiting/);
    await alice.m.send('dave', 'hi again');
    world.time += OUTBOX_PATIENCE + 1;
    await run(alice);
    assert.equal(alice.book.outbox.length, 0);
    assert.ok(alice.notices.some(t => /Gave up sending to dave.mds/.test(t)));
    // A name that takes no messages is refused before anything waits.
    world.register('erin', [rec(TAG.IDENTITY, sha('erin'))]);
    await assert.rejects(alice.m.send('erin', 'hi'), /takes no sealed messages/);
    await assert.rejects(alice.m.send('nobody', 'hi'), /No one holds nobody.mds/);
    await assert.rejects(alice.m.send('bob', ' '), /Nothing to send/);
});

test('the book is kept: a reload goes on where it was', async () => {
    const world = new World();
    const { alice, bob } = await pair(world, { bobPostage: 0 });
    await alice.m.send('bob', 'before');
    await run(alice, bob, alice, bob);
    alice.m.stop();
    const again = wallet(world, 'alice', { identity: alice.identity, book: loadBook(JSON.parse(JSON.stringify(alice.book))) });
    again.m.start();
    await bob.m.reply(bob.book.chats[0].id, 'after');
    await run(bob, again);
    assert.deepEqual(texts(again.book.chats[0]), ['>before', '<after']);
    // A message stamped across a reload goes once its stamp is done.
    const carol = wallet(world, 'carol');
    carol.crypto.slow = true;
    await carol.m.send('alice', 'slow one');
    await run(carol, again, carol);
    assert.ok(carol.book.openings.alice, 'made up to its stamp, and kept');
    const carol2 = wallet(world, 'carol', { book: loadBook(JSON.parse(JSON.stringify(carol.book))) });
    carol.m.stop();
    carol2.m.start();
    await run(carol2, again);
    assert.ok(again.book.chats.some(c => c.peer === '' && texts(c)[0] === '<slow one'));
});

test('a chat the node refuses is tried again, then shown as not sent until tried by hand', async () => {
    const world = new World();
    const { alice, bob } = await pair(world, { bobPostage: 0 });
    await alice.m.send('bob', 'hello');
    await run(alice, bob, alice, bob);
    world.down = true;
    await alice.m.send('bob', 'into the void');
    await run(alice);
    let l = alice.book.chats[0].lines.at(-1);
    assert.ok(l.sending && !l.failed);
    world.time += POST_PATIENCE + 1;
    await run(alice);
    l = alice.book.chats[0].lines.at(-1);
    assert.ok(l.failed, 'shown as not sent');
    world.down = false;
    world.time += 10_000;
    await run(alice);
    assert.ok(alice.book.chats[0].lines.at(-1).failed, 'not tried again by itself');
    await alice.m.retry(alice.book.chats[0].id);
    await run(alice, bob);
    assert.ok(!alice.book.chats[0].lines.at(-1).failed && !alice.book.chats[0].lines.at(-1).sending);
    assert.deepEqual(texts(bob.book.chats[0]).slice(-1), ['<into the void']);
});

test('writing as a name only if its records are this wallet’s; key requests answered for it alone', async () => {
    const world = new World();
    const w = wallet(world, 'w', { identity: sha('w id') });
    world.register('mine', [rec(TAG.IDENTITY, w.identity), rec(TAG.KEM, w.crypto.kemHash())]);
    world.register('theirs', [rec(TAG.IDENTITY, w.identity), rec(TAG.KEM, sha('other'))]);
    world.register('nokem', [rec(TAG.IDENTITY, w.identity)]);
    world.register('otherid', [rec(TAG.IDENTITY, sha('other id')), rec(TAG.KEM, w.crypto.kemHash())]);
    await assert.rejects(w.m.setName('theirs'), /another wallet’s message key/);
    await assert.rejects(w.m.setName('nokem'), /takes no sealed messages/);
    await assert.rejects(w.m.setName('otherid'), /isn’t one of this wallet’s/);
    await assert.rejects(w.m.setName('nobody'), /No one holds/);
    assert.deepEqual(await w.m.setName('mine.mds'), { asName: 'mine.mds' });

    // Key requests: to its identity, recent, a few a block.
    const req = (id, to, ts = world.time) => ({ sender: 'x', timestamp: ts, nonce: world.seq++, words: [255, CMD_KEM_REQ], attachments: kemRequest(id, to, h => h + '00000000').attachments });
    await w.m.ingest([req('aa'.repeat(32), sha('someone else'))]);
    await w.m.ingest([req('bb'.repeat(32), w.identity, world.time - 3_600)]);
    await idle(w.m);
    assert.equal(world.bus.length, 0, 'not ours, or asked long ago');
    await w.m.ingest(Array.from({ length: 12 }, (_, i) => req(String(i).padStart(64, '0'), w.identity)));
    await idle(w.m);
    const answers = world.bus.filter(m => m.words[1] === CMD_KEM);
    assert.equal(answers.length, KEM_ANSWERS_PER_BLOCK);
    assert.equal(parseFrame(answers[0]).payload, w.crypto.kemPublic());
    assert.deepEqual(await w.m.setName(null), { asName: '' });
    // A name that stops listing this wallet's keys is let go.
    await w.m.setName('mine');
    world.register('mine', [rec(TAG.IDENTITY, w.identity), rec(TAG.KEM, sha('rotated'))]);
    world.time += 3_600;
    await run(w);
    assert.equal(w.book.asName, null);
    assert.ok(w.notices.some(t => /no longer lists this wallet’s keys/.test(t)));
});

test('a key handed over is checked against the name’s record', async () => {
    const world = new World();
    const { alice, bob } = await pair(world, { bobPostage: 0 });
    await alice.m.send('bob', 'hi');
    await idle(alice.m);
    const [reqId] = Object.keys(alice.book.kemReqs);
    await alice.m.ingest([{ sender: 'm', timestamp: world.time, nonce: 1, words: [255, CMD_KEM], attachments: kemReply(reqId, Buffer.from('a forged key').toString('hex')).attachments }]);
    assert.equal(alice.book.peerKems.bob, undefined, 'a key that is not the one bob.mds lists');
    assert.equal(alice.book.outbox.length, 1);
    await run(bob, alice);
    assert.equal(alice.book.peerKems.bob, bob.crypto.kemPublic());
});

test('requests make room, and the book keeps to what a browser holds', () => {
    const book = newBook();
    const req = (i, at) => ({ id: 'r' + i, session: { opener: false }, peer: '', verified: false, accepted: false, started: at, lines: [{ at, mine: false, text: 'x', stamping: null, sending: null }], read: 0 });
    for (let i = 0; i < 5; i++) book.chats.push(req(i, 100 + i));
    makeRoom(book, 200, 3);
    assert.deepEqual(book.chats.map(c => c.id), ['r3', 'r4'], 'the quietest go');
    makeRoom(book, 104 + 30 * 86_400 + 1, 100);
    assert.deepEqual(book.chats.map(c => c.id), [], 'quiet a month');

    const c = { id: 'c', session: { opener: true, sent: 1 }, accepted: true, started: 0, read: MAX_LINES + 10, lines: [] };
    for (let i = 0; i < MAX_LINES + 10; i++) c.lines.push({ at: i, mine: i % 2 === 0, text: 'line ' + i, stamping: i === 0 ? 1 : null, sending: null, ttl: 0, timer: null });
    book.chats = [c];
    assert.ok(trim(book));
    assert.equal(c.lines.length, MAX_LINES);
    assert.equal(c.lines[0].text, 'line 0', 'what waits to go is kept');
    assert.equal(c.read, MAX_LINES);
});

test('the pieces: text, envelopes, frames, timers, giving numbers back', () => {
    assert.equal(checkedText('  hi  '), 'hi');
    assert.throws(() => checkedText('é'.repeat(376)), /at most 750 bytes; this is 752/);
    assert.equal(hexText(textHex('naïve ✓')), 'naïve ✓');
    assert.equal(hexText('ff'), '�', 'not UTF-8: replaced, as from_utf8_lossy does');

    const env = 'ab'.repeat(2000);
    const atts = attachmentsFor(env);
    assert.deepEqual(atts.map(a => a.value.length), [CHUNK_HEX, 4000 - CHUNK_HEX]);
    assert.equal(envelopeOf([{ kind: 'address', value: '00' }, ...atts]), env);
    assert.equal(envelopeOf([{ kind: 'signature', value: 'aa' }]), null);
    assert.throws(() => attachmentsFor('ab'.repeat(1536 * 4 + 1)), /one to 4 attachments/);

    const f = parseFrame({ words: [255, CMD_KEM_REQ], attachments: kemRequest('11'.repeat(32), '22'.repeat(32), h => h + 'deadbeef').attachments });
    assert.deepEqual(f, { cmd: CMD_KEM_REQ, id: '11'.repeat(32), payload: '0200000000', address: '22'.repeat(32) });
    assert.equal(parseFrame({ words: [1, 2], attachments: [] }), null);

    const c = { read: 2, lines: [{ ttl: 10, at: 0, stamping: null }, { ttl: 0, at: 0, stamping: null }, { ttl: 10, at: 5, stamping: 3 }] };
    assert.ok(prune(c, 100));
    assert.equal(c.lines.length, 2, 'what waits on its stamp has no timer running');
    assert.equal(c.read, 1);

    // 1 and 3 went; 2 and 4 wait on stamps. 4 is given back; 2 is not, as 3 went after it.
    const s = { session: { sent: 5 }, read: 3, lines: [{ stamping: null }, { stamping: 2 }, { stamping: null }, { stamping: 4 }] };
    assert.equal(unsend(s), 2);
    assert.equal(s.session.sent, 4);
    assert.equal(s.read, 2);
    const s2 = { session: { sent: 4 }, read: 0, lines: [{ stamping: null }, { stamping: 2 }, { stamping: 3 }] };
    unsend(s2);
    assert.equal(s2.session.sent, 2);

    const v = view(newBook(), 0);
    assert.deepEqual([v.chats, v.unread, v.requests], [[], 0, 0]);
});
