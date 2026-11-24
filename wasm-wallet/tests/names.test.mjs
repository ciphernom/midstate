// The name checks against the wallet's own WebAssembly (BLAKE3, addresses)
// and a real midstate node's answer (tests/fixtures/name_proof.json, the one
// midwimble's tests use).
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { initSync, blake3_hash_hex, compute_p2pk_address_hex } from '../pkg/wasm_wallet.js';
import { normalize, isName, checkProof, checkAnswer, standing, encodeRecords, costOf, tierForLen, RATES, payAddress, nameKey, parseRecords, kemHash, identityOf, postageOf, recordText, REGISTRY_ADDRESS, DEPTH, TAG } from '../names.js';

initSync({ module: fs.readFileSync(new URL('../pkg/wasm_wallet_bg.wasm', import.meta.url)) });
const blake3 = blake3_hash_hex;
const fixture = () => JSON.parse(fs.readFileSync(new URL('./fixtures/name_proof.json', import.meta.url)));
const hex = b => Buffer.from(b).toString('hex');
const H = bytes => Buffer.from(blake3(hex(bytes)), 'hex');

/** A one-leaf registry around `records`: the answer a node would give. The
 *  leaf is `name`'s, or (with `key`) a neighbour's that skips over it. */
function answer(name, records, expires, height, { key = null, owner = Buffer.alloc(32, 7), terms: termsIn = Buffer.from([0]) } = {}) {
    const k = key || nameKey(name, blake3);
    const next = Buffer.alloc(8, 0xff);
    const terms = termsIn;
    const data = H(records), deal = H(terms);
    const exp = Buffer.alloc(8); exp.writeBigUInt64LE(BigInt(expires));
    const leaf = Buffer.concat([k, next, exp, Buffer.from([3]), owner, data, deal]);
    let acc = H(leaf), empty = Buffer.alloc(32);
    const proof = [];
    for (let l = 0; l < DEPTH; l++) {
        const sib = l === 0 ? Buffer.alloc(32, 5) : empty;
        const side = l === 0 ? 1 : 0;
        proof.push(sib, Buffer.from([side]));
        acc = side ? H(Buffer.concat([sib, acc])) : H(Buffer.concat([acc, sib]));
        empty = H(Buffer.concat([empty, empty]));
    }
    const salt = Buffer.alloc(32, 9);
    const coin = H(Buffer.concat([Buffer.from('CONFIDENTIAL'), Buffer.from(REGISTRY_ADDRESS, 'hex'), acc, salt]));
    return {
        registered: !key, slot: 1,
        leaf: { key: hex(k), next: hex(next), expires, tier: 3, owner: hex(owner), data: hex(data), deal: hex(deal) },
        records: hex(records), terms: hex(terms), proof: hex(Buffer.concat(proof)),
        root: hex(acc), thread: { coin: hex(coin), salt: hex(salt) }, height,
    };
}
const rec = (tag, value) => Buffer.concat([Buffer.from([tag, value.length]), value]);

test('names are written one way', () => {
    assert.equal(normalize(' Nick.MDS '), 'nick');
    assert.equal(normalize('a-b9'), 'a-b9');
    assert.throws(() => normalize('-nick'), /start or end/);
    assert.throws(() => normalize('ni ck'), /only a-z/);
    assert.throws(() => normalize('x'.repeat(33)), /1 to 32/);
    assert.ok(isName('bob.mds') && !isName('ab'.repeat(36)));
});

test('a real node’s answer checks out', () => {
    const v = fixture();
    const r = checkProof('mdwname', v, blake3);
    assert.equal(r.height, 371503);
    assert.equal(r.coin, v.thread.coin);
    assert.deepEqual(r.records.map(x => [x.tag, recordText(x)]), [[TAG.MDW, 'mw1fixture']]);
});

test('a node that lies is caught', () => {
    const bad = (mut, re) => { const v = fixture(); mut(v); assert.throws(() => checkProof('mdwname', v, blake3), re); };
    bad(v => { v.records = hex(rec(TAG.MDW, Buffer.from('mw1evilxxx'))); }, /records are not the ones/);
    bad(v => { v.terms = '01'; }, /terms are not the ones/);
    bad(v => { v.leaf.expires += 1; }, /does not reach the registry root/);
    bad(v => { v.root = '00'.repeat(32); }, /does not reach the registry root/);
    bad(v => { v.thread.coin = '11'.repeat(32); }, /not the commitment of a registry coin/);
    bad(v => { v.proof = v.proof.slice(2); }, /not 24 levels|whole bytes/);
    // Another name's leaf.
    assert.throws(() => checkProof('othername', fixture(), blake3), /another name’s|not registered/);
    // An error from the node is passed on.
    assert.throws(() => checkProof('mdwname', { error: 'not ready' }, blake3), /The node says: not ready/);
});

test('a name no one holds, or that lapses, is refused', () => {
    const v = answer('nick', rec(TAG.MDS_ADDRESS, Buffer.alloc(32, 2)), 10_000, 5_000);
    // Not registered: the neighbour's link skips over the name's key, and the neighbour is proved.
    const unreg = answer('nick', Buffer.alloc(0), 2 ** 50, 5_000, { key: Buffer.alloc(8, 0) });
    assert.throws(() => checkProof('nick', unreg, blake3), /No one has the name nick.mds/);
    // A neighbour that is not proved is no answer at all.
    assert.throws(() => checkProof('nick', { ...unreg, leaf: { ...unreg.leaf, expires: 1 } }, blake3), /does not reach the registry root/);
    const lie = { ...v, registered: false };
    assert.throws(() => checkProof('nick', lie, blake3), /does not show that/);
    assert.throws(() => checkProof('nick', answer('nick', rec(TAG.MDS_ADDRESS, Buffer.alloc(32, 2)), 5_500, 5_000), blake3), /lapses within a day/);
    assert.throws(() => checkProof('nick', answer('nick', rec(TAG.MDS_ADDRESS, Buffer.alloc(32, 2)), 4_000, 5_000), blake3), /has lapsed/);
    // A name lapsing within a day is still held: it takes sealed messages.
    const soon = answer('nick', rec(TAG.KEM, Buffer.alloc(32, 8)), 5_500, 5_000);
    assert.equal(kemHash(checkProof('nick', soon, blake3, { lapsingOk: true }).records), '08'.repeat(32));
    assert.throws(() => checkProof('nick', answer('nick', rec(TAG.KEM, Buffer.alloc(32, 8)), 4_000, 5_000), blake3, { lapsingOk: true }), /has lapsed/);
});

test('where a name pays MDS', async () => {
    const key = Buffer.alloc(32, 4);
    const keyRec = rec(TAG.MDS_KEY, Buffer.concat([key, Buffer.from([10])]));   // 1,024 signatures
    const addrRec = rec(TAG.MDS_ADDRESS, Buffer.alloc(32, 2));
    const p2pk = compute_p2pk_address_hex;
    const used = n => async () => n;

    // An MSS key with signatures to spare pays its address.
    let r = checkProof('nick', answer('nick', Buffer.concat([keyRec, rec(TAG.TEXT, Buffer.from('hi'))]), 10_000, 5_000), blake3);
    assert.deepEqual(await payAddress(r.records, used(3), p2pk), { address: p2pk(hex(key)), kind: 'key', left: 1021 });
    // One about to run out pays nothing, unless an address follows it.
    assert.equal(await payAddress(r.records, used(1024 - 15), p2pk), null);
    r = checkProof('nick', answer('nick', Buffer.concat([keyRec, addrRec]), 10_000, 5_000), blake3);
    assert.deepEqual(await payAddress(r.records, used(1024 - 15), p2pk), { address: '02'.repeat(32), kind: 'address' });
    // The first record that pays wins.
    r = checkProof('nick', answer('nick', Buffer.concat([addrRec, keyRec]), 10_000, 5_000), blake3);
    assert.equal((await payAddress(r.records, used(0), p2pk)).kind, 'address');
    // MDW only: nothing for MDS.
    assert.equal(await payAddress(checkProof('mdwname', fixture(), blake3).records, used(0), p2pk), null);
});

test('records', () => {
    const kem = Buffer.alloc(32, 8);
    const id = Buffer.alloc(32, 5);
    const recs = parseRecords(Buffer.concat([rec(TAG.KEM, kem), rec(TAG.POSTAGE, Buffer.from([30])), rec(TAG.IDENTITY, id)]));
    assert.equal(kemHash(recs), hex(kem));
    assert.equal(identityOf(recs), hex(id));
    assert.equal(postageOf(recs), 30);
    assert.throws(() => parseRecords(Buffer.from([1, 5, 0])), /cut short/);
});

test('what a name asks of a stranger, as core::names reads it', () => {
    const only = (...rs) => parseRecords(Buffer.concat(rs));
    assert.equal(postageOf(only(rec(TAG.TEXT, Buffer.from('hi')))), 24, 'no record: the standard');
    assert.equal(postageOf(only(rec(TAG.POSTAGE, Buffer.from([16])))), 16);
    assert.equal(postageOf(only(rec(TAG.POSTAGE, Buffer.from([40])))), 40);
    // Out of range or not one byte: kept, and read as no price; a good one after it counts.
    assert.equal(postageOf(only(rec(TAG.POSTAGE, Buffer.from([41])))), 24);
    assert.equal(postageOf(only(rec(TAG.POSTAGE, Buffer.from([15])))), 24);
    assert.equal(postageOf(only(rec(TAG.POSTAGE, Buffer.from([30, 1])), rec(TAG.POSTAGE, Buffer.from([20])))), 20);
    // An identity or kem that is not 32 bytes is no identity or kem.
    assert.equal(identityOf(only(rec(TAG.IDENTITY, Buffer.alloc(31, 1)))), null);
    assert.equal(kemHash(only(rec(TAG.KEM, Buffer.alloc(33, 1)))), null);
});

test('any answer about a name, for one who wants it', () => {
    // Never registered: its neighbour is proved, and the name is available at its length's tier.
    const free = checkAnswer('nickname', answer('nickname', Buffer.alloc(0), 2 ** 50, 5_000, { key: Buffer.alloc(8, 0) }), blake3);
    assert.equal(free.registered, false);
    assert.equal(free.slot, 1);
    assert.equal(free.proof.length, DEPTH * 66);
    assert.deepEqual(standing(free), { state: 'available', tier: 3, height: 5_000 });
    // Held, lapsed, and a premium name no one has claimed (owner zero, lapsed since block 0).
    const held = checkAnswer('nick', answer('nick', rec(TAG.TEXT, Buffer.from('hi')), 10_000, 5_000), blake3);
    assert.equal(standing(held).state, 'held');
    assert.equal(standing(checkAnswer('nick', answer('nick', Buffer.alloc(0), 4_000, 5_000), blake3)).state, 'lapsed');
    assert.equal(standing(checkAnswer('nick', answer('nick', Buffer.alloc(0), 0, 5_000, { owner: Buffer.alloc(32) }), blake3)).state, 'unclaimed');
    // Offered for sale: its terms say so.
    const sale = Buffer.concat([Buffer.from([1]), Buffer.alloc(8, 1), Buffer.alloc(32, 2), Buffer.alloc(8)]);
    assert.equal(standing(checkAnswer('nick', answer('nick', Buffer.alloc(0), 10_000, 5_000, { terms: sale }), blake3)).deal, 'sale');
    // The registry's prices: 64, 8, 1 and an eighth of a gMDS a year.
    assert.deepEqual(RATES.map(r => r / 2 ** 30), [64, 8, 1, 0.125]);
    assert.deepEqual([1, 2, 3, 4, 5, 32].map(tierForLen), [0, 0, 1, 2, 3, 3]);
    assert.equal(costOf(3, 2), 2 ** 28);
});

test('records are written as the registry keeps them', () => {
    const hexOut = encodeRecords([{ tag: TAG.TEXT, value: Buffer.from('hi') }, { tag: TAG.POSTAGE, value: '1c' }]);
    assert.equal(hexOut, '060268690a011c');
    assert.deepEqual(parseRecords(Buffer.from(hexOut, 'hex')).map(r => r.tag), [TAG.TEXT, TAG.POSTAGE]);
    assert.throws(() => encodeRecords([{ tag: TAG.TEXT, value: Buffer.alloc(256) }]), /at most 255 bytes/);
    assert.throws(() => encodeRecords(Array.from({ length: 7 }, () => ({ tag: TAG.TEXT, value: Buffer.alloc(250) }))), /at most 1536 bytes/);
});
