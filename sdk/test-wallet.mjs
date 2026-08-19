// test-wallet.mjs — Offline regression tests for wallet sync and block scanning.
//
// Uses a mock client that reproduces the node's GetFilters contract exactly:
// an unreadable filter is served as an empty filter string with element_count 0
// and (where the block itself loaded) a real block hash. Those are the cases the
// old scanner silently skipped, so they are the ones worth pinning down.
//
// Run: node test-wallet.mjs

import fs from 'fs/promises';
import initWasm, { compute_coin_id_hex, blake3_hash_hex } from './pkg/wasm_wallet.js';
import { Wallet } from './src/wallet.js';
import { MemoryStorage } from './src/storage.js';

await initWasm({ module_or_path: await fs.readFile('./pkg/wasm_wallet_bg.wasm') });

let pass = 0, fail = 0;
async function t(name, fn) {
    try { await fn(); console.log(`  ✅ ${name}`); pass++; }
    catch (e) { console.log(`  ❌ ${name} — ${e.message}`); fail++; }
}
function assert(c, m) { if (!c) throw new Error(m || 'assertion failed'); }

const SALT_A = 'aa'.repeat(32);
const SALT_B = 'bb'.repeat(32);
const SALT_C = 'cc'.repeat(32);

/**
 * Mock node. `blocks` maps height -> block. `filterMode` decides how each
 * height is served, mirroring the three branches of the node's handler.
 */
class MockClient {
    constructor({ height, blocks, filterMode = () => 'ok' }) {
        this.height = height; this.blocks = blocks; this.filterMode = filterMode;
        this.getBlockCalls = []; this.getFiltersCalls = [];
    }
    async getState() { return { height: this.height, required_pow: 24, header_hash: '00'.repeat(32) }; }
    async getBlock(h) { this.getBlockCalls.push(h); return this.blocks[h] ?? { coinbase: [], transactions: [] }; }
    async getFilters(start, end) {
        this.getFiltersCalls.push([start, end]);
        const filters = [], element_counts = [], block_hashes = [];
        for (let h = start; h < end; h++) {
            const mode = this.filterMode(h);
            if (mode === 'ok') {
                // A real filter that matches nothing. check_filter must rule it out.
                filters.push('00'.repeat(32)); block_hashes.push(blake3_hash_hex('%02x'.replace('%02x', h.toString(16).padStart(64, '0')))); element_counts.push(4);
            } else if (mode === 'nofilter') {
                filters.push(''); block_hashes.push('11'.repeat(32)); element_counts.push(0);
            } else if (mode === 'unreadable') {
                filters.push(''); block_hashes.push(''); element_counts.push(0);
            }
        }
        return { start_height: start, filters, element_counts, block_hashes };
    }
}

const noPace = { filterIntervalMs: 0, batchSize: 1000 };

console.log('\n━━━ undecidable filters must be fetched ━━━');

await t('a block with no filter (count 0, hash present) is scanned, not skipped', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();
    const coinId = compute_coin_id_hex(addr, 4096n, SALT_A);
    const blocks = { 5: { coinbase: [{ address: addr, value: 4096, salt: SALT_A }], transactions: [] } };
    const client = new MockClient({ height: 10, blocks, filterMode: (h) => (h === 5 ? 'nofilter' : 'ok') });

    const res = await w.sync(client, noPace);
    assert(client.getBlockCalls.includes(5), 'block 5 was never fetched — the funds-visibility bug');
    assert(w.getBalance() === 4096n, `balance ${w.getBalance()}, expected 4096`);
    assert(w.utxos[0].coin_id === coinId, 'coin id mismatch');
    assert(res.found === 1, `found ${res.found}`);
});

await t('a wholly unreadable height (no hash, no filter) is also fetched', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();
    const blocks = { 3: { coinbase: [{ address: addr, value: 1024, salt: SALT_B }], transactions: [] } };
    const client = new MockClient({ height: 8, blocks, filterMode: (h) => (h === 3 ? 'unreadable' : 'ok') });

    await w.sync(client, noPace);
    assert(client.getBlockCalls.includes(3), 'unreadable height was skipped');
    assert(w.getBalance() === 1024n, `balance ${w.getBalance()}`);
});

await t('genuine non-matching filters still rule blocks out (no over-fetching)', async () => {
    const w = await Wallet.create(new MemoryStorage());
    await w.getNewAddress();
    const client = new MockClient({ height: 50, blocks: {}, filterMode: () => 'ok' });
    await w.sync(client, noPace);
    assert(client.getBlockCalls.length === 0, `fetched ${client.getBlockCalls.length} blocks, expected 0`);
});

console.log('\n━━━ no request amplification ━━━');

await t('a short filter batch does not walk the remainder block-by-block', async () => {
    const w = await Wallet.create(new MemoryStorage());
    await w.getNewAddress();
    // Peer serves only 10 filters for a 1000-block request, then nothing.
    let served = 0;
    const client = new MockClient({ height: 1000, blocks: {} });
    client.getFilters = async (start, end) => {
        client.getFiltersCalls.push([start, end]);
        const n = served++ === 0 ? 10 : 0;
        return {
            start_height: start,
            filters: Array(n).fill('00'.repeat(32)),
            element_counts: Array(n).fill(4),
            block_hashes: Array(n).fill('22'.repeat(32)),
        };
    };
    let err = null;
    try { await w.sync(client, { ...noPace, maxStalledBatches: 3 }); } catch (e) { err = e; }
    assert(err && /stalled/i.test(err.message), `expected a stall error, got ${err && err.message}`);
    // The old code would have issued ~990 getBlock calls for the first batch
    // alone. Bounded stall handling costs one per stalled batch.
    assert(client.getBlockCalls.length <= 4, `issued ${client.getBlockCalls.length} getBlock calls — amplification`);
});

console.log('\n━━━ transaction scanning ━━━');

function revealTx({ inputs = [], outputs = [] }) { return { Reveal: { inputs, outputs, salt: SALT_C } }; }
function consolidateTx({ inputs = [], outputs = [] }) { return { Consolidate: { inputs, outputs, salt: SALT_C } }; }
// P2PK bytecode is opaque here; what matters is that address == blake3(bytecode).
function inputFor(bytecodeHex, value, salt) {
    return { predicate: { Script: { bytecode: bytecodeHex } }, value, salt };
}

await t('Consolidate transactions are scanned for receipts', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();
    const blocks = {
        2: { coinbase: [], transactions: [consolidateTx({ outputs: [{ Standard: { address: addr, value: 2048, salt: SALT_A } }] })] },
    };
    const client = new MockClient({ height: 5, blocks, filterMode: (h) => (h === 2 ? 'nofilter' : 'ok') });
    await w.sync(client, noPace);
    assert(w.getBalance() === 2048n, `Consolidate output missed — balance ${w.getBalance()}`);
});

await t('Consolidate transactions are scanned for spends', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();
    const bytecode = 'de'.repeat(20);
    const spendAddr = blake3_hash_hex(bytecode);
    // Put a coin at the script address into the wallet by pointing a watched
    // address at it, then spend it via Consolidate.
    w.wotsAddrs[spendAddr] = 1; w.nextWotsIndex = 2;
    const cid = compute_coin_id_hex(spendAddr, 512n, SALT_A);
    w.addUtxo(spendAddr, 512, SALT_A, cid);
    assert(w.getBalance() === 512n, 'setup failed');

    const blocks = { 2: { coinbase: [], transactions: [consolidateTx({ inputs: [inputFor(bytecode, 512, SALT_A)] })] } };
    const client = new MockClient({ height: 5, blocks, filterMode: (h) => (h === 2 ? 'nofilter' : 'ok') });
    await w.sync(client, noPace);
    assert(w.getBalance() === 0n, `Consolidate spend missed — balance ${w.getBalance()}`);
});

await t('spend is identified from the predicate, not just the salt', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const bytecode = 'ab'.repeat(20);
    const spendAddr = blake3_hash_hex(bytecode);
    w.wotsAddrs[spendAddr] = 0; w.nextWotsIndex = 1;
    const cid = compute_coin_id_hex(spendAddr, 256n, SALT_A);
    w.addUtxo(spendAddr, 256, SALT_A, cid);

    const blocks = { 1: { coinbase: [], transactions: [revealTx({ inputs: [inputFor(bytecode, 256, SALT_A)] })] } };
    const client = new MockClient({ height: 4, blocks, filterMode: (h) => (h === 1 ? 'nofilter' : 'ok') });
    await w.sync(client, noPace);
    assert(w.utxos.length === 0, 'coin not removed via predicate-derived id');
});

console.log('\n━━━ WOTS one-time key safety ━━━');

await t('spending a WOTS address purges its siblings', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const bytecode = 'cd'.repeat(20);
    const wotsAddr = blake3_hash_hex(bytecode);
    w.wotsAddrs[wotsAddr] = 0; w.nextWotsIndex = 1;
    w.addUtxo(wotsAddr, 128, SALT_A, compute_coin_id_hex(wotsAddr, 128n, SALT_A));
    w.addUtxo(wotsAddr, 256, SALT_B, compute_coin_id_hex(wotsAddr, 256n, SALT_B));
    assert(w.utxos.length === 2, 'setup failed');

    // Only ONE of the two siblings appears as an input.
    const blocks = { 1: { coinbase: [], transactions: [revealTx({ inputs: [inputFor(bytecode, 128, SALT_A)] })] } };
    const client = new MockClient({ height: 4, blocks, filterMode: (h) => (h === 1 ? 'nofilter' : 'ok') });
    await w.sync(client, noPace);
    assert(w.utxos.length === 0, `sibling at a burned WOTS address survived — ${w.utxos.length} left`);
});

await t('a coin arriving at a burned WOTS address is not admitted', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const bytecode = 'ef'.repeat(20);
    const wotsAddr = blake3_hash_hex(bytecode);
    w.wotsAddrs[wotsAddr] = 0; w.nextWotsIndex = 1;
    w.addUtxo(wotsAddr, 64, SALT_A, compute_coin_id_hex(wotsAddr, 64n, SALT_A));

    const blocks = {
        1: { coinbase: [], transactions: [revealTx({ inputs: [inputFor(bytecode, 64, SALT_A)] })] },
        2: { coinbase: [{ address: wotsAddr, value: 512, salt: SALT_C }], transactions: [] },
    };
    const client = new MockClient({ height: 5, blocks, filterMode: (h) => (h <= 2 ? 'nofilter' : 'ok') });
    await w.sync(client, noPace);
    assert(w.spentWots[wotsAddr] !== undefined, 'address not recorded as burned');
    assert(w.getBalance() === 0n, `unspendable coin admitted — balance ${w.getBalance()}`);
});

await t('MSS addresses are NOT purged (they tolerate reuse)', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const bytecode = '12'.repeat(20);
    const mssAddr = blake3_hash_hex(bytecode);
    w.mssAddrs[mssAddr] = { index: 0, height: 10, next_leaf: 0 }; w.nextMssIndex = 1;
    w.addUtxo(mssAddr, 128, SALT_A, compute_coin_id_hex(mssAddr, 128n, SALT_A));
    w.addUtxo(mssAddr, 256, SALT_B, compute_coin_id_hex(mssAddr, 256n, SALT_B));

    const blocks = { 1: { coinbase: [], transactions: [revealTx({ inputs: [inputFor(bytecode, 128, SALT_A)] })] } };
    const client = new MockClient({ height: 4, blocks, filterMode: (h) => (h === 1 ? 'nofilter' : 'ok') });
    await w.sync(client, noPace);
    assert(w.getBalance() === 256n, `MSS sibling wrongly purged — balance ${w.getBalance()}`);
});

console.log('\n━━━ persistence ━━━');

await t('spentWots survives save/restore', async () => {
    const store = new MemoryStorage();
    const w = await Wallet.create(store);
    const addr = await w.getNewAddress();
    w.spentWots[addr] = { height: 7 };
    await w.save();
    const w2 = await Wallet.restore(store);
    assert(w2.spentWots[addr] !== undefined, 'spentWots lost across restore');
});

await t('balances round-trip as BigInt', async () => {
    const store = new MemoryStorage();
    const w = await Wallet.create(store);
    const addr = await w.getNewAddress();
    w.addUtxo(addr, 4096, SALT_A, compute_coin_id_hex(addr, 4096n, SALT_A));
    await w.save();
    const w2 = await Wallet.restore(store);
    assert(typeof w2.getBalance() === 'bigint', 'balance is not BigInt after restore');
    assert(w2.getBalance() === 4096n, `balance ${w2.getBalance()}`);
});

console.log('\n━━━ spend path: key material accounting ━━━');

// A client that drives _broadcastTwoPhaseTx to completion without a network.
class SpendClient extends MockClient {
    constructor(opts) { super(opts); this.sent = []; this.committed = []; }
    async commit(c, n) { this.committed.push(c); return { ok: true, body: null }; }
    async checkCommitment() { return { exists: true }; }
    async send(payload) { this.sent.push(payload); return { ok: true, body: null }; }
}

await t('MSS leaf is read live at spend time, not stamped at discovery', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewReusableAddress(4);
    // Two coins arrive before either is spent. Both are stamped with the
    // next_leaf value current at discovery, which is 0 for both.
    w.addUtxo(addr, 4096, SALT_A, compute_coin_id_hex(addr, 4096n, SALT_A));
    w.addUtxo(addr, 4096, SALT_B, compute_coin_id_hex(addr, 4096n, SALT_B));
    assert(w.utxos[0].mss_leaf === 0 && w.utxos[1].mss_leaf === 0, 'setup: both should be stamped leaf 0');

    // Simulate one leaf having been consumed by an earlier transaction.
    w.mssAddrs[addr].next_leaf = 1;

    const forWasm = w._utxosForWasm();
    assert(forWasm.every((u) => u.mss_leaf === 1),
        `stale leaf leaked to the signer: ${JSON.stringify(forWasm.map((u) => u.mss_leaf))}`);
});

await t('_utxosForWasm refreshes mss_height too', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewReusableAddress(4);
    w.addUtxo(addr, 1024, SALT_A, compute_coin_id_hex(addr, 1024n, SALT_A));
    assert(w._utxosForWasm()[0].mss_height === 4, 'mss_height not carried');
});

await t('one MSS leaf is consumed per address, not per input', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewReusableAddress(4);
    for (const s of [SALT_A, SALT_B, SALT_C]) {
        w.addUtxo(addr, 1024, s, compute_coin_id_hex(addr, 1024n, s));
    }
    const before = w.mssAddrs[addr].next_leaf;

    // Three inputs, all at the same MSS address — build_reveal signs once and
    // reuses the signature, so exactly one leaf is consumed.
    const ctx = {
        next_wots_index: w.nextWotsIndex,
        commitment: 'aa'.repeat(32),
        tx_salt: 'bb'.repeat(32),
        selected_inputs: w.utxos.map((u) => ({ ...u, value: Number(u.value) })),
    };
    w.inner.build_reveal = () => '{}';   // stub: signing itself is WASM-tested
    const client = new SpendClient({ height: 100, blocks: {} });
    await w._broadcastTwoPhaseTx(client, JSON.stringify(ctx), false);

    const consumed = w.mssAddrs[addr].next_leaf - before;
    assert(consumed === 1, `consumed ${consumed} leaves for a 3-input spend, expected 1`);
});

await t('WOTS inputs are retired before the reveal is transmitted', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();
    const cid = compute_coin_id_hex(addr, 2048n, SALT_A);
    w.addUtxo(addr, 2048, SALT_A, cid);

    const ctx = {
        next_wots_index: w.nextWotsIndex,
        commitment: 'cc'.repeat(32),
        tx_salt: 'dd'.repeat(32),
        selected_inputs: [{ ...w.utxos[0], value: 2048 }],
    };
    w.inner.build_reveal = () => '{}';

    // Assert ordering: by the time send() runs, the key must already be retired.
    const client = new SpendClient({ height: 100, blocks: {} });
    let stateAtSend = null;
    client.send = async (p) => {
        stateAtSend = { burned: w.spentWots[addr] !== undefined, utxos: w.utxos.length };
        return { ok: true, body: null };
    };
    await w._broadcastTwoPhaseTx(client, JSON.stringify(ctx), false);

    assert(stateAtSend !== null, 'send was never called');
    assert(stateAtSend.burned, 'WOTS address not retired before transmit');
    assert(stateAtSend.utxos === 0, 'coin not removed before transmit');
});

await t('a rejected reveal still retires the key (no second signature)', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();
    w.addUtxo(addr, 512, SALT_A, compute_coin_id_hex(addr, 512n, SALT_A));

    const ctx = {
        next_wots_index: w.nextWotsIndex,
        commitment: 'ee'.repeat(32),
        tx_salt: 'ff'.repeat(32),
        selected_inputs: [{ ...w.utxos[0], value: 512 }],
    };
    w.inner.build_reveal = () => '{}';
    const client = new SpendClient({ height: 100, blocks: {} });
    client.send = async () => ({ ok: false, body: 'rejected by node' });

    await w._broadcastTwoPhaseTx(client, JSON.stringify(ctx), false);
    assert(w.spentWots[addr] !== undefined, 'key not retired after rejection');
    assert(w.getBalance() === 0n, 'coin still spendable after its key was signed with');
});

await t('a commit timeout touches no key material', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();
    w.addUtxo(addr, 256, SALT_A, compute_coin_id_hex(addr, 256n, SALT_A));

    const ctx = {
        next_wots_index: w.nextWotsIndex,
        commitment: '11'.repeat(32),
        tx_salt: '22'.repeat(32),
        selected_inputs: [{ ...w.utxos[0], value: 256 }],
    };
    const client = new SpendClient({ height: 100, blocks: {} });
    client.checkCommitment = async () => ({ exists: false });

    // Collapse the 120x10s poll so the test runs instantly.
    const realTimeout = globalThis.setTimeout;
    globalThis.setTimeout = (fn) => realTimeout(fn, 0);
    let err = null;
    try { await w._broadcastTwoPhaseTx(client, JSON.stringify(ctx), false); }
    catch (e) { err = e; }
    finally { globalThis.setTimeout = realTimeout; }

    assert(err && /Timed out/.test(err.message), `expected timeout, got ${err && err.message}`);
    assert(w.spentWots[addr] === undefined, 'key retired despite no reveal being built');
    assert(w.getBalance() === 256n, 'coin dropped despite nothing being signed');
});

await t('oversized coin value is refused rather than silently narrowed', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();
    w.utxos.push({ address: addr, value: (1n << 60n), salt: SALT_A, coin_id: 'ab'.repeat(32),
                   index: 0, is_mss: false, mss_height: 0, mss_leaf: 0 });
    let threw = false;
    try { w._utxosForWasm(); } catch (e) { threw = /safe JSON integer range/.test(e.message); }
    assert(threw, 'oversized value was narrowed instead of rejected');
});

console.log('\n━━━ adaptive gap extension ━━━');

/**
 * A chain paying the wallet at scattered WOTS indices, as a real history does.
 * Modelled on a live wallet: 320 coins spread over indices 0..1728, which no
 * fixed derivation window could have covered.
 */
async function chainPayingIndices(indices, tip = 60) {
    const w = await Wallet.create(new MemoryStorage());
    const blocks = {};
    const addrOf = (i) => w.inner.get_wots_address(i);
    indices.forEach((idx, n) => {
        const h = 5 + n;                       // one coin per block
        blocks[h] = { coinbase: [{ address: addrOf(idx), value: 1024, salt: (n + 1).toString(16).padStart(2, '0').repeat(32) }], transactions: [] };
    });
    const client = new MockClient({ height: tip, blocks, filterMode: () => 'nofilter' });
    return { w, client, addrOf };
}

await t('a coin beyond the initial window is still found', async () => {
    // The exact failure that under-reported a live wallet: with a fixed window
    // the scan completes and reports a confident, too-small balance.
    //
    // Indices are within `gap` of each other, which is what "gap limit" means —
    // a coin further than `gap` past the last one is legitimately out of reach
    // and needs a larger gap, not a bug fix.
    const { w, client } = await chainPayingIndices([0, 15, 35]);
    const res = await w.sync(client, { ...noPace, gapLimit: 20 });
    assert(w.getBalance() === 3072n, `balance ${w.getBalance()}, expected all three coins (3072)`);
    assert(res.gapPasses >= 1, 'the window never extended');
    assert(res.highestUsedWots === 35, `highest used reported as ${res.highestUsedWots}`);
});

await t('a coin further than `gap` past the last one is out of reach by design', async () => {
    // Honest about the limit: this is what `gap` MEANS. The remedy is a larger
    // gap, and the result reports how far derivation actually reached so a
    // caller can tell the difference between "settled" and "gave up".
    const { w, client } = await chainPayingIndices([0, 40]);
    const res = await w.sync(client, { ...noPace, gapLimit: 20 });
    assert(w.getBalance() === 1024n, 'a gap of 20 should not reach index 40');
    assert(res.wotsDerived < 40, `derived to ${res.wotsDerived}, which should not reach 40`);
    // And a gap wide enough does find it.
    const { w: w2, client: c2 } = await chainPayingIndices([0, 40]);
    await w2.sync(c2, { ...noPace, gapLimit: 60 });
    assert(w2.getBalance() === 2048n, `a gap of 60 missed index 40 (balance ${w2.getBalance()})`);
});

await t('the window chases a chain of coins outward', async () => {
    // Each discovery should push the ceiling further, as the CLI's repeated
    // "Extended to WOTS index N" lines do.
    const { w, client } = await chainPayingIndices([0, 15, 30, 45, 60, 75]);
    await w.sync(client, { ...noPace, gapLimit: 20 });
    assert(w.getBalance() === 6144n, `balance ${w.getBalance()}, expected 6 coins`);
    assert(w.nextWotsIndex >= 96, `derived to ${w.nextWotsIndex}; should reach past index 75 + gap`);
});

await t('a fixed window would have missed them (non-vacuity)', async () => {
    // Same chain, extension disabled: the scan succeeds and the balance is wrong.
    const { w, client } = await chainPayingIndices([0, 40]);
    for (let i = 0; i < 20; i++) await w.getNewAddress();
    await w.sync(client, { ...noPace, gapLimit: 0 });
    assert(w.getBalance() === 1024n,
        `fixture broken: a 20-address window should see only the first coin, saw ${w.getBalance()}`);
});

await t('extension is monotonic — the index never rewinds', async () => {
    // Rewinding would re-issue an index that may already have signed.
    const { w, client } = await chainPayingIndices([0, 30]);
    await w.sync(client, { ...noPace, gapLimit: 20 });
    const high = w.nextWotsIndex;
    await w.sync(client, { ...noPace, gapLimit: 5 });
    assert(w.nextWotsIndex >= high, `index rewound ${high} → ${w.nextWotsIndex}`);
});

await t('a burned address still pushes the window forward', async () => {
    // A gap measured only from spendable coins would shrink after a sweep.
    const w = await Wallet.create(new MemoryStorage());
    for (let i = 0; i < 50; i++) await w.getNewAddress();
    const addr = w.inner.get_wots_address(45);
    w.spentWots[addr] = { height: 10 };
    await w._extendGap(20);
    assert(w.nextWotsIndex >= 66, `derived to ${w.nextWotsIndex}; index 45 is burned so the window must clear 65`);
});

await t('a settled scan converges in a bounded number of passes', async () => {
    // One extra pass is inherent: finding a coin at index 2 moves the ceiling
    // from 20 to 23, and the range is re-examined once with the wider window.
    // What matters is that it converges rather than chasing its own tail.
    const { w, client } = await chainPayingIndices([0, 1, 2]);
    const res = await w.sync(client, { ...noPace, gapLimit: 20 });
    assert(res.gapPasses <= 2, `took ${res.gapPasses} passes for three adjacent coins`);
    assert(w.getBalance() === 3072n, `balance ${w.getBalance()}`);
});

await t('an unsettled scan fails loudly instead of under-reporting', async () => {
    const { w, client } = await chainPayingIndices(Array.from({ length: 12 }, (_, i) => i * 15));
    let msg = null;
    try { await w.sync(client, { ...noPace, gapLimit: 20, maxGapPasses: 2 }); }
    catch (e) { msg = e.message; }
    assert(msg && /INCOMPLETE/.test(msg),
        `expected a loud failure, got ${msg ? `"${msg}"` : 'a silent wrong balance'}`);
});

await t('progress reports each extension', async () => {
    const { w, client } = await chainPayingIndices([0, 40]);
    const notes = [];
    await w.sync(client, { ...noPace, gapLimit: 20, onProgress: (p) => { if (p.note) notes.push(p.note); } });
    assert(notes.some((n) => /extended to WOTS index/.test(n)), 'extension was not reported');
});

console.log('\n━━━ gap extension: scan range and filter cache ━━━');

await t('a partial range is refused rather than under-reporting', async () => {
    // The walk chains index to index, so activity below the floor is invisible
    // and a wide enough hole stops it early. Observed live: a 200k window
    // reached index 1412 where a full scan reached 1472.
    const { w, client } = await chainPayingIndices([0, 15]);
    w.lastScannedHeight = 30;                    // start part-way up the chain
    let msg = null;
    try { await w.sync(client, { ...noPace, gapLimit: 20 }); } catch (e) { msg = e.message; }
    assert(msg && /full-chain scan/.test(msg), `expected a refusal, got ${msg ? `"${msg}"` : 'silence'}`);
});

await t('a partial range is allowed when the caller opts in knowingly', async () => {
    const { w, client } = await chainPayingIndices([0, 15]);
    w.lastScannedHeight = 30;
    const res = await w.sync(client, { ...noPace, gapLimit: 20, allowPartialGapScan: true });
    assert(res.fullScan === false, 'should not be reported as a full scan');
});

await t('a completed full scan unlocks later incremental gap extension', async () => {
    // The normal operating case: once genesis-up has established the index
    // space, an incremental sync may extend without the guard firing.
    const { w, client } = await chainPayingIndices([0, 15]);
    const res = await w.sync(client, { ...noPace, gapLimit: 20 });
    assert(res.fullScan === true, 'a scan from 0 should report fullScan');
    assert(w.fullScanCompleted === true, 'the flag was not set');

    w.lastScannedHeight = 30;
    await w.sync(client, { ...noPace, gapLimit: 20 });   // must not throw
});

await t('the full-scan flag survives save/restore', async () => {
    const store = new MemoryStorage();
    const { w, client } = await chainPayingIndices([0, 15]);
    w.storage = store;
    await w.sync(client, { ...noPace, gapLimit: 20 });
    const w2 = await Wallet.restore(store);
    assert(w2.fullScanCompleted === true, 'flag lost across restore');
});

await t('filters are fetched once and re-tested locally across passes', async () => {
    // Filters are immutable per height; only the watchlist changes. Re-fetching
    // them each pass is what made a live 200k-block scan take 2087s.
    const { w, client } = await chainPayingIndices([0, 15, 35, 55]);
    let fetches = 0;
    const realGetFilters = client.getFilters.bind(client);
    client.getFilters = async (a, b) => { fetches++; return realGetFilters(a, b); };

    const res = await w.sync(client, { ...noPace, gapLimit: 20, batchSize: 10 });
    assert(res.gapPasses >= 2, `only ${res.gapPasses} passes — not exercising the cache`);
    assert(res.filterCacheHits > 0, 'the cache was never used');
    // Every height should have been fetched exactly once across all passes.
    const heights = 60;
    const batches = Math.ceil(heights / 10);
    assert(fetches <= batches + 2, `${fetches} fetches for ${batches} batches across ${res.gapPasses + 1} passes`);
});

await t('disabling the cache still produces the same balance', async () => {
    const a = await chainPayingIndices([0, 15, 35]);
    const b = await chainPayingIndices([0, 15, 35]);
    const ra = await a.w.sync(a.client, { ...noPace, gapLimit: 20 });
    const rb = await b.w.sync(b.client, { ...noPace, gapLimit: 20, cacheFilters: false });
    assert(a.w.getBalance() === b.w.getBalance(), `cached ${a.w.getBalance()} vs uncached ${b.w.getBalance()}`);
    assert(ra.filterCacheHits > 0 && rb.filterCacheHits === 0, 'cache flag had no effect');
});

console.log(`\n━━━ ${pass} passed, ${fail} failed ━━━\n`);
process.exit(fail ? 1 : 0);
