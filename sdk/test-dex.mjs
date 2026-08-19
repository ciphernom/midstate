// test-dex.mjs — On-chain DEX announcement codec and covenant derivation.
//
// Vectors are built byte-by-byte from `core/dex.rs` rather than round-tripped
// against this implementation, because a symmetric bug round-trips perfectly and
// still produces announcements the node and the browser wallet cannot read. The
// wire is BIG-endian here and little-endian in `bincode.js`; that is the single
// easiest thing to get wrong and the tests target it directly.
//
// Run: node test-dex.mjs

import fs from 'fs/promises';
import initWasm, * as W from './pkg/wasm_wallet.js';
import * as DEX from './src/dex.js';

await initWasm({ module_or_path: await fs.readFile('./pkg/wasm_wallet_bg.wasm') });

let pass = 0, fail = 0;
async function t(name, fn) {
    try { await fn(); console.log(`  ✅ ${name}`); pass++; }
    catch (e) { console.log(`  ❌ ${name} — ${e.message}`); fail++; }
}
function assert(c, m) { if (!c) throw new Error(m || 'assertion failed'); }

const EVM = 'aa'.repeat(20);
const PK = 'bb'.repeat(32);
const GID = 'cc'.repeat(6);
const H1 = 'd1'.repeat(32), S1 = 'e1'.repeat(32);
const H2 = 'd2'.repeat(32), S2 = 'e2'.repeat(32);

const mkAnn = (units) => ({ makerEvmAddr: EVM, makerMdsPk: PK, timeoutHeight: 1440n, groupId: GID, units });
const ONE_ETH = 10n ** 18n;

console.log('\n━━━ MDXA byte layout ━━━');

await t('header is 72 bytes and each unit adds 81', () => {
    for (const n of [1, 2, 5]) {
        const units = Array.from({ length: n }, () => ({ secretHash: H1, salt: S1, value: 1024n, weiAmount: ONE_ETH }));
        const hex = DEX.encodeMakerAnnouncement(mkAnn(units));
        assert(hex.length / 2 === 72 + 81 * n, `n=${n}: ${hex.length / 2} bytes, expected ${72 + 81 * n}`);
    }
});

await t('field offsets match core/dex.rs exactly', () => {
    const hex = DEX.encodeMakerAnnouncement(mkAnn([{ secretHash: H1, salt: S1, value: 256n, weiAmount: 7n }]));
    assert(hex.slice(0, 8) === DEX.ANN_MAGIC, 'magic at 0..4');
    assert(hex.slice(8, 10) === '01', 'version at 4');
    assert(hex.slice(10, 50) === EVM, 'evm addr at 5..25');
    assert(hex.slice(50, 114) === PK, 'mds pk at 25..57');
    assert(hex.slice(114, 130) === '00000000000005a0', `timeout at 57..65 (big-endian 1440), got ${hex.slice(114, 130)}`);
    assert(hex.slice(130, 142) === GID, 'group id at 65..71');
    assert(hex.slice(142, 144) === '01', 'unit count at 71');
    // unit: secretHash(32) salt(32) exp(1) wei(16)
    assert(hex.slice(144, 208) === H1, 'unit secret hash');
    assert(hex.slice(208, 272) === S1, 'unit salt');
    assert(hex.slice(272, 274) === '08', `value exponent for 256 should be 08, got ${hex.slice(272, 274)}`);
    assert(hex.slice(274, 306) === '0'.repeat(31) + '7', `wei big-endian, got ${hex.slice(274, 306)}`);
});

await t('integers are BIG-endian, not little', () => {
    // The whole rest of the SDK's wire work is little-endian. Getting this
    // backwards decodes to plausible garbage rather than failing.
    const hex = DEX.encodeMakerAnnouncement(mkAnn([{ secretHash: H1, salt: S1, value: 2n, weiAmount: 1n }]));
    const timeout = hex.slice(114, 130);
    assert(timeout === '00000000000005a0', `timeout ${timeout} — little-endian would be a005000000000000`);
    assert(timeout !== 'a005000000000000', 'timeout is little-endian');
});

await t('round-trips every field', () => {
    const units = [
        { secretHash: H1, salt: S1, value: 1024n, weiAmount: ONE_ETH },
        { secretHash: H2, salt: S2, value: 4096n, weiAmount: 12345678901234567890n },
    ];
    const back = DEX.decodeMakerAnnouncement(DEX.encodeMakerAnnouncement(mkAnn(units)));
    assert(back !== null, 'decode returned null');
    assert(back.makerEvmAddr === EVM && back.makerMdsPk === PK, 'identity fields');
    assert(back.timeoutHeight === 1440n, `timeout ${back.timeoutHeight}`);
    assert(back.groupId === GID, 'group id');
    assert(back.units.length === 2, 'unit count');
    assert(back.units[1].value === 4096n, `value ${back.units[1].value}`);
    assert(back.units[1].weiAmount === 12345678901234567890n, `wei ${back.units[1].weiAmount}`);
});

await t('a u128 wei amount survives (no Number rounding)', () => {
    const wei = (1n << 100n) + 12345n;
    const back = DEX.decodeMakerAnnouncement(
        DEX.encodeMakerAnnouncement(mkAnn([{ secretHash: H1, salt: S1, value: 2n, weiAmount: wei }])));
    assert(back.units[0].weiAmount === wei, `wei ${back.units[0].weiAmount} != ${wei}`);
});

await t('non-power-of-two values are refused, not rounded', () => {
    // The exponent is one byte: a non-power-of-two cannot be represented, and a
    // silently rounded value produces an announcement whose coin does not exist.
    let threw = false;
    try { DEX.encodeMakerAnnouncement(mkAnn([{ secretHash: H1, salt: S1, value: 1000n, weiAmount: 1n }])); }
    catch (e) { threw = /power of two/.test(e.message); }
    assert(threw, 'accepted a non-power-of-two unit value');
});

await t('unit count bounds are enforced', () => {
    let a = false, b = false;
    try { DEX.encodeMakerAnnouncement(mkAnn([])); } catch { a = true; }
    try {
        DEX.encodeMakerAnnouncement(mkAnn(Array.from({ length: 256 }, () => ({ secretHash: H1, salt: S1, value: 2n, weiAmount: 1n }))));
    } catch { b = true; }
    assert(a, 'accepted zero units');
    assert(b, 'accepted 256 units');
});

await t('an out-of-range exponent is rejected on decode', () => {
    // exp >= 64 would overflow a u64 coin value.
    const hex = DEX.encodeMakerAnnouncement(mkAnn([{ secretHash: H1, salt: S1, value: 2n, weiAmount: 1n }]));
    const bad = hex.slice(0, 272) + '40' + hex.slice(274);   // exponent 64
    assert(DEX.decodeMakerAnnouncement(bad) === null, 'accepted exponent 64');
});

await t('foreign or truncated payloads decode to null, not garbage', () => {
    assert(DEX.decodeMakerAnnouncement('deadbeef') === null, 'foreign magic');
    assert(DEX.decodeMakerAnnouncement('') === null, 'empty');
    const hex = DEX.encodeMakerAnnouncement(mkAnn([{ secretHash: H1, salt: S1, value: 2n, weiAmount: 1n }]));
    assert(DEX.decodeMakerAnnouncement(hex.slice(0, 100)) === null, 'truncated');
    // Right magic, wrong version.
    assert(DEX.decodeMakerAnnouncement(DEX.ANN_MAGIC + '09' + hex.slice(10)) === null, 'wrong version');
});

console.log('\n━━━ MDXT taker locks ━━━');

const taker = {
    takerMdsPk: '11'.repeat(32), secretHash: H1, salt: S1,
    receiverAddr: '22'.repeat(32), timeoutHeight: 900n, value: 512n, weiAmount: ONE_ETH,
};

await t('taker announcement round-trips', () => {
    const back = DEX.decodeTakerAnnouncement(DEX.encodeTakerAnnouncement(taker));
    assert(back !== null, 'decode returned null');
    assert(back.takerMdsPk === taker.takerMdsPk && back.receiverAddr === taker.receiverAddr, 'identity');
    assert(back.value === 512n && back.weiAmount === ONE_ETH, 'amounts');
    assert(back.timeoutHeight === 900n, `timeout ${back.timeoutHeight}`);
});

await t('maker and taker formats do not decode as each other', () => {
    const m = DEX.encodeMakerAnnouncement(mkAnn([{ secretHash: H1, salt: S1, value: 2n, weiAmount: 1n }]));
    const tk = DEX.encodeTakerAnnouncement(taker);
    assert(DEX.decodeTakerAnnouncement(m) === null, 'maker decoded as taker');
    assert(DEX.decodeMakerAnnouncement(tk) === null, 'taker decoded as maker');
});

console.log('\n━━━ MDXF fragmentation ━━━');

await t('no fragment exceeds the consensus burn cap', () => {
    // MAX_BURN_DATA_SIZE = 80 is consensus. A fragment over it is simply
    // unpublishable, and the announcement is silently lost.
    for (const n of [1, 2, 5, 20]) {
        const units = Array.from({ length: n }, () => ({ secretHash: H1, salt: S1, value: 1024n, weiAmount: ONE_ETH }));
        const body = DEX.encodeMakerAnnouncement(mkAnn(units));
        for (const f of DEX.fragment(body, GID)) {
            assert(f.length / 2 <= DEX.MAX_BURN_DATA_SIZE,
                `n=${n}: fragment is ${f.length / 2} bytes, cap is ${DEX.MAX_BURN_DATA_SIZE}`);
        }
    }
});

await t('a single-unit announcement needs 3 fragments', () => {
    // 153 bytes over 68-byte payloads.
    const body = DEX.encodeMakerAnnouncement(mkAnn([{ secretHash: H1, salt: S1, value: 1024n, weiAmount: ONE_ETH }]));
    assert(DEX.fragment(body, GID).length === 3, `${DEX.fragment(body, GID).length} fragments for 153 bytes`);
});

await t('fragments reassemble to the original body', () => {
    const units = Array.from({ length: 4 }, (_, i) => ({ secretHash: H1, salt: S1, value: 1n << BigInt(i + 8), weiAmount: ONE_ETH }));
    const body = DEX.encodeMakerAnnouncement(mkAnn(units));
    const pool = new DEX.FragmentPool();
    let out = null;
    for (const f of DEX.fragment(body, GID)) out = pool.add(DEX.parseFragment(f)) || out;
    assert(out === body, 'reassembled body differs from the original');
    assert(DEX.decodeMakerAnnouncement(out).units.length === 4, 'reassembled body does not decode');
});

await t('fragments reassemble out of order', () => {
    const body = DEX.encodeMakerAnnouncement(mkAnn([{ secretHash: H1, salt: S1, value: 1024n, weiAmount: ONE_ETH }]));
    const frags = DEX.fragment(body, GID).reverse();
    const pool = new DEX.FragmentPool();
    let out = null;
    for (const f of frags) out = pool.add(DEX.parseFragment(f)) || out;
    assert(out === body, 'out-of-order reassembly failed');
});

await t('an incomplete group yields nothing and stays pending', () => {
    const body = DEX.encodeMakerAnnouncement(mkAnn([{ secretHash: H1, salt: S1, value: 1024n, weiAmount: ONE_ETH }]));
    const frags = DEX.fragment(body, GID);
    const pool = new DEX.FragmentPool();
    assert(pool.add(DEX.parseFragment(frags[0])) === null, 'completed on one fragment');
    assert(pool.add(DEX.parseFragment(frags[2])) === null, 'completed with a hole');
    assert(pool.pending === 1, `pending ${pool.pending}`);
});

await t('two interleaved groups do not contaminate each other', () => {
    const bodyA = DEX.encodeMakerAnnouncement(mkAnn([{ secretHash: H1, salt: S1, value: 1024n, weiAmount: 1n }]));
    const bodyB = DEX.encodeMakerAnnouncement({ ...mkAnn([{ secretHash: H2, salt: S2, value: 2048n, weiAmount: 2n }]), groupId: 'ff'.repeat(6) });
    const fa = DEX.fragment(bodyA, GID), fb = DEX.fragment(bodyB, 'ff'.repeat(6));
    const pool = new DEX.FragmentPool();
    const results = [];
    for (let i = 0; i < Math.max(fa.length, fb.length); i++) {
        if (fa[i]) { const r = pool.add(DEX.parseFragment(fa[i])); if (r) results.push(r); }
        if (fb[i]) { const r = pool.add(DEX.parseFragment(fb[i])); if (r) results.push(r); }
    }
    assert(results.includes(bodyA) && results.includes(bodyB), 'interleaved groups did not both reassemble');
});

await t('the pool is bounded against never-completing groups', () => {
    const pool = new DEX.FragmentPool(4);
    for (let i = 0; i < 50; i++) {
        pool.add({ groupId: i.toString(16).padStart(12, '0'), idx: 0, total: 9, chunk: 'aa' });
    }
    assert(pool.pending <= 4, `pool grew to ${pool.pending} incomplete groups`);
});

await t('a malformed fragment is rejected', () => {
    assert(DEX.parseFragment('deadbeef') === null, 'foreign magic');
    assert(DEX.parseFragment(DEX.FRAG_MAGIC + 'cc'.repeat(6) + '0500' + 'aa') === null, 'idx >= total');
    assert(DEX.parseFragment(DEX.FRAG_MAGIC + 'cc'.repeat(6) + '0000' + 'aa') === null, 'total = 0');
});

console.log('\n━━━ burn extraction and ingest ━━━');

await t('payloads are found as serde number arrays', () => {
    // GetBlock serializes Vec<u8> as a JSON number array; no hex regex matches it.
    const body = DEX.encodeMakerAnnouncement(mkAnn([{ secretHash: H1, salt: S1, value: 1024n, weiAmount: 1n }]));
    const frags = DEX.fragment(body, GID);
    const block = { transactions: [{ Reveal: { outputs: frags.map((f) => ({
        DataBurn: { payload: Array.from(Buffer.from(f, 'hex')), value_burned: 0 },
    })) } }] };
    const found = DEX.extractBurnPayloads(block);
    assert(found.length === frags.length, `found ${found.length} of ${frags.length} burns`);
    assert(found[0] === frags[0].toLowerCase(), 'payload bytes differ');
});

await t('ingest reassembles a fragmented order from a block', () => {
    const body = DEX.encodeMakerAnnouncement(mkAnn([
        { secretHash: H1, salt: S1, value: 1024n, weiAmount: ONE_ETH },
        { secretHash: H2, salt: S2, value: 2048n, weiAmount: 2n * ONE_ETH },
    ]));
    const block = { transactions: [{ Reveal: { outputs: DEX.fragment(body, GID).map((f) => ({
        DataBurn: { payload: Array.from(Buffer.from(f, 'hex')), value_burned: 0 },
    })) } }] };

    const pool = new DEX.FragmentPool();
    let ann = null;
    for (const p of DEX.extractBurnPayloads(block)) {
        const r = DEX.ingest(p, pool);
        if (r && r.kind === 'maker') ann = r.value;
    }
    assert(ann !== null, 'no maker order recovered from the block');
    assert(ann.units.length === 2, `${ann.units.length} units`);
});

await t('an unfragmented taker lock is ingested directly', () => {
    const r = DEX.ingest(DEX.encodeTakerAnnouncement(taker), new DEX.FragmentPool());
    assert(r && r.kind === 'taker', 'taker lock not recognised');
    assert(r.value.value === 512n, 'taker value wrong');
});

await t('unrelated burn data is ignored', () => {
    const pool = new DEX.FragmentPool();
    assert(DEX.ingest('00'.repeat(40), pool) === null, 'zero payload matched');
    assert(DEX.ingest('4f505f52455455524e', pool) === null, 'foreign payload matched');
});

console.log('\n━━━ covenant derivation ━━━');

await t('the covenant address is derived, not announced', () => {
    // A reader recomputes it from the announced terms, so a maker cannot
    // advertise one set of terms and fund a different covenant.
    const d = DEX.deriveCovenantAddress({ secretHash: H1, value: 1024n, timeoutHeight: 1440n, makerMdsPk: PK });
    assert(d.address.length === 64, `address length ${d.address.length}`);
    assert(d.address === W.blake3_hash_hex(d.script), 'address is not the hash of the script');
});

await t('derivation is deterministic and terms-sensitive', () => {
    const base = { secretHash: H1, value: 1024n, timeoutHeight: 1440n, makerMdsPk: PK };
    const a = DEX.deriveCovenantAddress(base).address;
    assert(a === DEX.deriveCovenantAddress(base).address, 'not deterministic');
    for (const [k, v] of [['secretHash', H2], ['value', 2048n], ['timeoutHeight', 1441n], ['makerMdsPk', '99'.repeat(32)]]) {
        const other = DEX.deriveCovenantAddress({ ...base, [k]: v }).address;
        assert(other !== a, `changing ${k} did not change the covenant address`);
    }
});

await t('expandOrders produces resolvable, priced orders', () => {
    const ann = DEX.decodeMakerAnnouncement(DEX.encodeMakerAnnouncement(mkAnn([
        { secretHash: H1, salt: S1, value: 1024n, weiAmount: ONE_ETH },
        { secretHash: H2, salt: S2, value: 2048n, weiAmount: ONE_ETH },
    ])));
    const orders = DEX.expandOrders(ann, (addr, val, salt) => W.compute_coin_id_hex(addr, BigInt(val), salt));
    assert(orders.length === 2, 'wrong order count');
    assert(orders[0].coinId && orders[0].coinId.length === 64, 'coin id not derived');
    assert(orders[0].covenantAddress !== orders[1].covenantAddress, 'units share a covenant address');
    // Same wei for twice the coin => half the unit price.
    assert(orders[1].priceWeiPerUnit < orders[0].priceWeiPerUnit, 'unit pricing is inverted');
});

await t('a derived coin id matches the announced salt and value', () => {
    const ann = DEX.decodeMakerAnnouncement(DEX.encodeMakerAnnouncement(mkAnn([
        { secretHash: H1, salt: S1, value: 1024n, weiAmount: ONE_ETH }])));
    const [o] = DEX.expandOrders(ann, (a, v, s) => W.compute_coin_id_hex(a, BigInt(v), s));
    assert(o.coinId === W.compute_coin_id_hex(o.covenantAddress, 1024n, S1),
        'coin id does not match (covenant address, value, announced salt)');
});

console.log('\n━━━ secrets ━━━');

await t('secrets are 32 bytes and hash with BLAKE3', () => {
    const { secret, secretHash } = DEX.newSecret();
    assert(secret.length === 64, `secret is ${secret.length / 2} bytes`);
    assert(secretHash === W.blake3_hash_hex(secret), 'hash is not BLAKE3(secret)');
});

await t('secrets are unique per call', () => {
    // One secret backs exactly one unit: reusing it would let the first fill
    // unlock every other unit in the order.
    const seen = new Set();
    for (let i = 0; i < 200; i++) seen.add(DEX.newSecret().secret);
    assert(seen.size === 200, `${200 - seen.size} duplicate secret(s) in 200 draws`);
});

await t('group ids are 6 bytes and unique', () => {
    const seen = new Set();
    for (let i = 0; i < 100; i++) {
        const g = DEX.newGroupId();
        assert(g.length === 12, `group id is ${g.length / 2} bytes`);
        seen.add(g);
    }
    assert(seen.size === 100, 'group id collisions');
});

console.log(`\n━━━ ${pass} passed, ${fail} failed ━━━\n`);
process.exit(fail ? 1 : 0);
