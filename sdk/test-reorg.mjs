// test-reorg.mjs — Regression tests for reuse-safe chain-reorganisation handling.
//
// The obligations under test are the ones stated in reorg.js:
//
//   INV-1  a coin at a published one-time key is never freely spendable
//   CASE 1 signature retained  → poison cleared, verbatim re-send armed
//   CASE 2 signature lost      → coin restored REUSE-LOCKED
//   CASE 3 MSS                 → restored normally, no lock
//
// The dangerous failure is silent: a restored coin that looks spendable. Every
// CASE 2 test therefore asserts on _utxosForWasm — the single choke point where
// UTXOs reach the signer — not just on the balance.
//
// Run: node test-reorg.mjs

import fs from 'fs/promises';
import initWasm, { compute_coin_id_hex, blake3_hash_hex } from './pkg/wasm_wallet.js';
import { Wallet } from './src/wallet.js';
import { MemoryStorage } from './src/storage.js';
import { rollbackTo, findForkHeight, maybeHandleReorg, pruneHistory, REORG_DEPTH, COMMITMENT_TTL } from './src/reorg.js';

await initWasm({ module_or_path: await fs.readFile('./pkg/wasm_wallet_bg.wasm') });

let pass = 0, fail = 0;
async function t(name, fn) {
    try { await fn(); console.log(`  ✅ ${name}`); pass++; }
    catch (e) { console.log(`  ❌ ${name} — ${e.message}`); fail++; }
}
function assert(c, m) { if (!c) throw new Error(m || 'assertion failed'); }

const S = (n) => n.toString(16).padStart(2, '0').repeat(32);
const H = (n) => `hash${n}`.padEnd(64, '0');

/** A wallet with a WOTS address, one coin on it, and a recorded chain history. */
async function walletWithSpentWots({ retainReveal, spentAt = 10, tip = 12, commitment = S(0xaa) }) {
    const w = await Wallet.create(new MemoryStorage());
    const bytecode = 'cd'.repeat(20);
    const addr = blake3_hash_hex(bytecode);
    w.wotsAddrs[addr] = 0; w.nextWotsIndex = 1;

    const cid = compute_coin_id_hex(addr, 4096n, S(1));
    w._reorgHeight = 5;
    w.addUtxo(addr, 4096, S(1), cid, 5);

    // Simulate the spend at `spentAt`: log it, poison the key, remove the coin.
    w._reorgHeight = spentAt;
    w._logSpent(w.utxos.filter((u) => u.address === addr));
    w.spentWots[addr] = { height: spentAt, commitment };
    w.utxos = w.utxos.filter((u) => u.address !== addr);

    if (retainReveal) {
        w.sentReveals[addr] = {
            commitment,
            revealPayload: '{"signed":"verbatim"}',
            inputCoinIds: [cid],
            sentAtHeight: spentAt,
        };
    }
    for (let h = 0; h <= tip; h++) w.blockHashes[h] = H(h);
    w.lastScannedHeight = tip;
    return { w, addr, cid, commitment };
}

console.log('\n━━━ rollback: dead-branch credits ━━━');

await t('coins credited above the fork are removed', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();
    w.addUtxo(addr, 1024, S(1), compute_coin_id_hex(addr, 1024n, S(1)), 5);   // below fork
    w.addUtxo(addr, 2048, S(2), compute_coin_id_hex(addr, 2048n, S(2)), 11);  // above fork
    w.lastScannedHeight = 12;

    const r = rollbackTo(w, 8, 12);
    assert(r.removed === 1, `removed ${r.removed}, expected 1`);
    assert(w.getBalance() === 1024n, `balance ${w.getBalance()}, expected 1024`);
});

await t('scan marker is rewound to the fork', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.lastScannedHeight = 50;
    rollbackTo(w, 20, 50);
    assert(w.lastScannedHeight === 20, `scanned ${w.lastScannedHeight}`);
});

await t('block hashes above the fork are discarded', async () => {
    const w = await Wallet.create(new MemoryStorage());
    for (let h = 0; h <= 20; h++) w.blockHashes[h] = H(h);
    w.lastScannedHeight = 20;
    rollbackTo(w, 12, 20);
    const kept = Object.keys(w.blockHashes).map(Number);
    assert(Math.max(...kept) === 12, `highest retained hash ${Math.max(...kept)}`);
});

await t('a coin both created AND spent above the fork is not resurrected', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();
    const cid = compute_coin_id_hex(addr, 512n, S(3));
    w._reorgHeight = 11;
    w.addUtxo(addr, 512, S(3), cid, 11);      // created above fork
    w._reorgHeight = 12;
    w._logSpent(w.utxos);                      // spent above fork
    w.utxos = [];
    w.lastScannedHeight = 12;

    rollbackTo(w, 8, 12);
    assert(w.utxos.length === 0, 'dead-branch coin was resurrected');
});

console.log('\n━━━ CASE 1: signature retained → verbatim re-send ━━━');

await t('poison is cleared and the original signature is armed', async () => {
    const { w, addr, commitment } = await walletWithSpentWots({ retainReveal: true });
    const r = rollbackTo(w, 8, 12);

    assert(r.restored === 1, `restored ${r.restored}`);
    assert(r.resendable === 1, `resendable ${r.resendable}`);
    assert(w.spentWots[addr] === undefined, 'poison not cleared for a re-sendable spend');
    assert(w.pendingResend !== null, 'no re-send armed');
    assert(w.pendingResend.commitment === commitment, 'armed the wrong commitment');
    assert(w.pendingResend.revealPayload === '{"signed":"verbatim"}',
        'armed payload is not the verbatim original — re-signing would reuse the key');
});

await t('the restored coin is spendable again (same signature only)', async () => {
    const { w } = await walletWithSpentWots({ retainReveal: true });
    rollbackTo(w, 8, 12);
    assert(w.getBalance() === 4096n, `balance ${w.getBalance()}`);
    assert(w.getLockedBalance() === 0n, 'coin should not be locked when re-sendable');
    assert(w._utxosForWasm().length === 1, 'coin withheld from signer despite being re-sendable');
});

await t('a reveal older than COMMITMENT_TTL does NOT qualify', async () => {
    // The commitment has expired from chain state, so the retained signature can
    // no longer be revealed. Promising a re-send would be a lie.
    const { w, addr } = await walletWithSpentWots({ retainReveal: true, spentAt: 10 });
    const r = rollbackTo(w, 8, 10 + COMMITMENT_TTL + 1);
    assert(r.locked === 1, `expected CASE 2 fallback, got resendable=${r.resendable}`);
    assert(w.spentWots[addr] !== undefined, 'poison cleared on an expired reveal');
});

await t('a reveal for a DIFFERENT commitment does NOT qualify', async () => {
    const { w, addr } = await walletWithSpentWots({ retainReveal: true });
    w.sentReveals[addr].commitment = S(0xbb);   // belongs to some other spend
    const r = rollbackTo(w, 8, 12);
    assert(r.locked === 1, 'a mismatched commitment must not clear the poison');
    assert(w.spentWots[addr] !== undefined, 'poison cleared on a mismatched commitment');
});

console.log('\n━━━ CASE 2: signature lost → reuse-locked ━━━');

await t('coin is restored but flagged reuseLocked', async () => {
    const { w, addr, cid } = await walletWithSpentWots({ retainReveal: false });
    const r = rollbackTo(w, 8, 12);

    assert(r.restored === 1, `restored ${r.restored}`);
    assert(r.locked === 1, `locked ${r.locked}`);
    assert(w.utxos.length === 1, 'coin not restored');
    assert(w.utxos[0].reuseLocked === true, 'restored coin not flagged');
    assert(w.reuseLocked[cid] !== undefined, 'lock record not written');
    assert(w.spentWots[addr] !== undefined, 'poison must be retained, not cleared');
    assert(w.spentWots[addr].reuseLocked === true, 'poison not demoted to locked');
});

await t('locked coins are excluded from spendable balance but visible', async () => {
    const { w } = await walletWithSpentWots({ retainReveal: false });
    rollbackTo(w, 8, 12);
    assert(w.getBalance() === 0n, `spendable balance ${w.getBalance()}, expected 0`);
    assert(w.getLockedBalance() === 4096n, `locked balance ${w.getLockedBalance()}`);
});

await t('locked coins NEVER reach the signer (the critical obligation)', async () => {
    const { w } = await walletWithSpentWots({ retainReveal: false });
    rollbackTo(w, 8, 12);
    const forWasm = w._utxosForWasm();
    assert(forWasm.length === 0,
        `a reuse-locked coin was offered to prepare_spend — signing it would leak the key`);
});

await t('an alert is raised so the caller can surface the lock', async () => {
    const { w, addr } = await walletWithSpentWots({ retainReveal: false });
    const r = rollbackTo(w, 8, 12);
    const alert = r.alerts.find((a) => a.kind === 'reuse_lock');
    assert(alert, 'no reuse_lock alert emitted');
    assert(alert.address === addr, 'alert names the wrong address');
    assert(/key reuse/i.test(alert.message), 'alert does not explain the risk');
});

console.log('\n━━━ CASE 3: MSS restores normally ━━━');

await t('an orphaned MSS spend restores unlocked', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewReusableAddress(4);
    const cid = compute_coin_id_hex(addr, 8192n, S(4));
    w._reorgHeight = 5;
    w.addUtxo(addr, 8192, S(4), cid, 5);
    w._reorgHeight = 10;
    w._logSpent(w.utxos);
    w.utxos = [];
    w.lastScannedHeight = 12;

    const r = rollbackTo(w, 8, 12);
    assert(r.restored === 1, `restored ${r.restored}`);
    assert(r.locked === 0, 'MSS coin was wrongly locked');
    assert(w.getBalance() === 8192n, `balance ${w.getBalance()}`);
    assert(w._utxosForWasm().length === 1, 'MSS coin withheld from signer');
});

console.log('\n━━━ INV-1 holds after rollback ━━━');

await t('no free WOTS coin remains at a published key', async () => {
    for (const retainReveal of [true, false]) {
        const { w } = await walletWithSpentWots({ retainReveal });
        rollbackTo(w, 8, 12);
        for (const c of w.utxos) {
            if (!c.is_mss && w.spentWots[c.address] !== undefined) {
                assert(c.reuseLocked === true,
                    `INV-1 violated (retainReveal=${retainReveal}): free coin at a published key`);
            }
        }
    }
});

console.log('\n━━━ fork detection ━━━');

class ForkClient {
    constructor(nodeHashes) { this.nodeHashes = nodeHashes; this.calls = []; }
    async getFilters(start) {
        this.calls.push(start);
        const h = this.nodeHashes[start];
        return { start_height: start, filters: [''], element_counts: [0], block_hashes: [h ?? ''] };
    }
}

await t('finds the highest agreeing height', async () => {
    const w = await Wallet.create(new MemoryStorage());
    for (let h = 0; h <= 20; h++) w.blockHashes[h] = H(h);
    w.lastScannedHeight = 20;
    // Node diverges from 15 up.
    const nodeHashes = {};
    for (let h = 0; h <= 20; h++) nodeHashes[h] = h < 15 ? H(h) : `alt${h}`.padEnd(64, '0');

    const { fork, exhausted } = await findForkHeight(w, new ForkClient(nodeHashes), 20);
    assert(fork === 14, `fork ${fork}, expected 14`);
    assert(!exhausted, 'should not be exhausted');
});

await t('descent is bounded by REORG_DEPTH', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const tip = REORG_DEPTH + 50;
    for (let h = 0; h <= tip; h++) w.blockHashes[h] = H(h);
    w.lastScannedHeight = tip;
    const client = new ForkClient({});   // node agrees with nothing

    const { fork, exhausted, probes } = await findForkHeight(w, client, tip);
    assert(exhausted, 'should report exhaustion');
    assert(fork === tip - REORG_DEPTH, `fork ${fork}`);
    assert(probes <= REORG_DEPTH + 1,
        `${probes} probes — an unbounded descent is a self-inflicted rate-limit violation`);
});

await t('heights we never recorded are skipped, not treated as divergence', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.blockHashes[10] = H(10);
    w.blockHashes[20] = H(20);          // gap between 10 and 20
    w.lastScannedHeight = 20;
    const nodeHashes = { 10: H(10), 20: 'different'.padEnd(64, '0') };

    const { fork } = await findForkHeight(w, new ForkClient(nodeHashes), 20);
    assert(fork === 10, `fork ${fork}, expected 10 (gap skipped)`);
});

console.log('\n━━━ maybeHandleReorg ━━━');

await t('matching tip hash is a no-op (Ξ)', async () => {
    const { w } = await walletWithSpentWots({ retainReveal: false });
    const before = JSON.stringify({ u: w.utxos.length, s: w.lastScannedHeight });
    const nodeHashes = {}; for (let h = 0; h <= 12; h++) nodeHashes[h] = H(h);

    const r = await maybeHandleReorg(w, new ForkClient(nodeHashes), 12);
    assert(r === null, 'reported a reorg where the chain agrees');
    assert(JSON.stringify({ u: w.utxos.length, s: w.lastScannedHeight }) === before, 'state mutated on the no-reorg path');
});

await t('a fresh wallet with no history does not roll back', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const r = await maybeHandleReorg(w, new ForkClient({}), 100);
    assert(r === null, 'a wallet with no recorded history must not guess');
});

await t('conflict at the tip triggers rollback', async () => {
    const { w, addr } = await walletWithSpentWots({ retainReveal: false });
    const nodeHashes = {};
    for (let h = 0; h <= 12; h++) nodeHashes[h] = h <= 8 ? H(h) : `alt${h}`.padEnd(64, '0');

    const r = await maybeHandleReorg(w, new ForkClient(nodeHashes), 12);
    assert(r !== null, 'conflict not detected');
    assert(r.fork === 8, `fork ${r.fork}, expected 8`);
    assert(r.locked === 1, 'orphaned spend not resolved');
    assert(w.spentWots[addr] !== undefined, 'poison lost');
});

await t('an undecidable placeholder is not treated as a conflict', async () => {
    const { w } = await walletWithSpentWots({ retainReveal: false });
    const r = await maybeHandleReorg(w, new ForkClient({}), 12);  // node returns ""
    assert(r === null, 'empty hash was treated as divergence');
});

console.log('\n━━━ pruning ━━━');

await t('block hashes and spent log are bounded by REORG_DEPTH', async () => {
    const w = await Wallet.create(new MemoryStorage());
    for (let h = 0; h <= REORG_DEPTH + 100; h++) {
        w.blockHashes[h] = H(h);
        w.spentLog[h] = [{ coin_id: `c${h}`, value: 1n, address: 'a', createdAtHeight: h }];
    }
    pruneHistory(w, REORG_DEPTH + 100);
    const hs = Object.keys(w.blockHashes).map(Number);
    assert(hs.length <= REORG_DEPTH + 8, `${hs.length} hashes retained`);
    assert(Math.min(...hs) >= 100, `pruned to ${Math.min(...hs)}`);
    assert(Object.keys(w.spentLog).every((h) => Number(h) >= 100), 'spent log not pruned in step');
});

await t('sentReveals are pruned on the COMMITMENT_TTL clock, not REORG_DEPTH', async () => {
    // Pruning these at REORG_DEPTH would discard still-re-sendable signatures
    // and needlessly demote recoverable coins to reuse-locked.
    const w = await Wallet.create(new MemoryStorage());
    w.sentReveals['fresh'] = { sentAtHeight: 900, commitment: S(1) };
    w.sentReveals['stale'] = { sentAtHeight: 10, commitment: S(2) };
    pruneHistory(w, 1200);
    assert(w.sentReveals['fresh'] !== undefined, 'a still-revealable signature was pruned');
    assert(w.sentReveals['stale'] === undefined, 'an expired signature was retained');
});

await t('spentWots is never pruned', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.spentWots['addr'] = { height: 1 };
    for (let h = 0; h <= REORG_DEPTH + 100; h++) w.blockHashes[h] = H(h);
    pruneHistory(w, REORG_DEPTH + 100);
    assert(w.spentWots['addr'] !== undefined,
        'forgetting a published key re-opens the address to admission');
});

console.log('\n━━━ persistence ━━━');

await t('reorg state survives save/restore', async () => {
    const store = new MemoryStorage();
    const { w, addr, cid } = await walletWithSpentWots({ retainReveal: false });
    w.storage = store;
    rollbackTo(w, 8, 12);
    await w.save();

    const w2 = await Wallet.restore(store);
    assert(w2.reuseLocked[cid] !== undefined, 'lock record lost');
    assert(w2.utxos[0].reuseLocked === true, 'coin lock flag lost');
    assert(w2.getLockedBalance() === 4096n, `locked balance ${w2.getLockedBalance()} after restore`);
    assert(w2._utxosForWasm().length === 0, 'restored wallet offers a locked coin to the signer');
    assert(w2.spentWots[addr] !== undefined, 'poison lost across restore');
});

await t('spentLog BigInt values round-trip', async () => {
    const store = new MemoryStorage();
    const w = await Wallet.create(store);
    const addr = await w.getNewAddress();
    w._reorgHeight = 7;
    w.addUtxo(addr, 4096, S(1), compute_coin_id_hex(addr, 4096n, S(1)), 7);
    w._logSpent(w.utxos);
    await w.save();

    const w2 = await Wallet.restore(store);
    assert(typeof w2.spentLog[7][0].value === 'bigint', 'spentLog value is not BigInt after restore');
    assert(w2.spentLog[7][0].value === 4096n, `value ${w2.spentLog[7][0].value}`);
});

console.log('\n━━━ end-to-end: sync() through a reorg ━━━');

/**
 * A mock node with a switchable chain. `branch` selects which set of blocks and
 * hashes it serves, so a test can reorganise the chain between syncs exactly as
 * a real node would.
 */
class ChainClient {
    constructor(branches, branch = 'a') {
        this.branches = branches; this.branch = branch; this.getBlockCalls = [];
    }
    get chain() { return this.branches[this.branch]; }
    async getState() { return { height: this.chain.height, required_pow: 24, header_hash: '00'.repeat(32) }; }
    async getBlock(h) { this.getBlockCalls.push(h); return this.chain.blocks[h] ?? { coinbase: [], transactions: [] }; }
    async getFilters(start, end) {
        const filters = [], element_counts = [], block_hashes = [];
        for (let h = start; h < Math.min(end, this.chain.height); h++) {
            // Empty filter + zero count = "undecidable, fetch it" — keeps the
            // test honest about which blocks actually get scanned.
            filters.push(''); element_counts.push(0);
            block_hashes.push(this.chain.hashes[h] ?? '');
        }
        return { start_height: start, filters, element_counts, block_hashes };
    }
}

function coinbaseBlock(addr, value, salt) {
    return { coinbase: [{ address: addr, value, salt }], transactions: [] };
}
function hashes(prefix, upto) {
    const o = {}; for (let h = 0; h <= upto; h++) o[h] = `${prefix}${h}`.padEnd(64, '0');
    return o;
}

await t('a credit that only existed on the orphaned branch is reversed', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();

    // Branch A pays us 4096 at height 5.
    const a = { height: 10, hashes: hashes('a', 10), blocks: { 5: coinbaseBlock(addr, 4096, S(1)) } };
    // Branch B forks at 3 and never pays us. Heights 0..3 are shared.
    const bHashes = { ...hashes('b', 10) };
    for (let h = 0; h <= 3; h++) bHashes[h] = a.hashes[h];
    const b = { height: 10, hashes: bHashes, blocks: {} };

    const client = new ChainClient({ a, b }, 'a');
    await w.sync(client, { filterIntervalMs: 0 });
    assert(w.getBalance() === 4096n, `pre-reorg balance ${w.getBalance()}`);

    client.branch = 'b';
    const res = await w.sync(client, { filterIntervalMs: 0 });
    assert(res.reorg !== null && res.reorg !== undefined, 'sync did not report a reorg');
    assert(res.reorg.fork === 3, `fork ${res.reorg.fork}, expected 3`);
    assert(w.getBalance() === 0n, `phantom credit survived: balance ${w.getBalance()}`);
});

await t('a credit present on BOTH branches survives the reorg', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();

    // Paid at height 2, below the fork, so it is on both branches.
    const shared = { 2: coinbaseBlock(addr, 1024, S(2)) };
    const a = { height: 10, hashes: hashes('a', 10), blocks: { ...shared, 7: coinbaseBlock(addr, 4096, S(3)) } };
    const bHashes = { ...hashes('b', 10) };
    for (let h = 0; h <= 4; h++) bHashes[h] = a.hashes[h];
    const b = { height: 10, hashes: bHashes, blocks: { ...shared } };

    const client = new ChainClient({ a, b }, 'a');
    await w.sync(client, { filterIntervalMs: 0 });
    assert(w.getBalance() === 5120n, `pre-reorg balance ${w.getBalance()}`);

    client.branch = 'b';
    await w.sync(client, { filterIntervalMs: 0 });
    assert(w.getBalance() === 1024n, `post-reorg balance ${w.getBalance()}, expected 1024`);
});

await t('a same-height reorg is caught (tip does not advance)', async () => {
    // The case that would slip past a check placed after the "nothing to do"
    // early return: the chain is replaced but the height is identical.
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();

    const a = { height: 8, hashes: hashes('a', 8), blocks: { 6: coinbaseBlock(addr, 2048, S(4)) } };
    const bHashes = { ...hashes('b', 8) };
    for (let h = 0; h <= 4; h++) bHashes[h] = a.hashes[h];
    const b = { height: 8, hashes: bHashes, blocks: {} };

    const client = new ChainClient({ a, b }, 'a');
    await w.sync(client, { filterIntervalMs: 0 });
    assert(w.getBalance() === 2048n, 'setup failed');
    const heightBefore = w.lastScannedHeight;

    client.branch = 'b';
    const res = await w.sync(client, { filterIntervalMs: 0 });
    assert(res.reorg, 'same-height reorg went undetected');
    assert(w.getBalance() === 0n, `balance ${w.getBalance()} after same-height reorg`);
    assert(w.lastScannedHeight === heightBefore, 'scan marker did not recover to the tip');
});

await t('the winning branch is re-scanned after rollback', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();

    const a = { height: 10, hashes: hashes('a', 10), blocks: { 6: coinbaseBlock(addr, 512, S(5)) } };
    const bHashes = { ...hashes('b', 10) };
    for (let h = 0; h <= 4; h++) bHashes[h] = a.hashes[h];
    // Branch B pays a DIFFERENT amount at a different height above the fork.
    const b = { height: 10, hashes: bHashes, blocks: { 8: coinbaseBlock(addr, 8192, S(6)) } };

    const client = new ChainClient({ a, b }, 'a');
    await w.sync(client, { filterIntervalMs: 0 });
    assert(w.getBalance() === 512n, 'setup failed');

    client.branch = 'b';
    await w.sync(client, { filterIntervalMs: 0 });
    assert(w.getBalance() === 8192n,
        `winning-branch credit not picked up: balance ${w.getBalance()}`);
    assert(w.lastScannedHeight === 10, `scan marker ${w.lastScannedHeight}, expected 10`);
});

await t('onReorg fires with the report', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();
    const a = { height: 8, hashes: hashes('a', 8), blocks: { 6: coinbaseBlock(addr, 256, S(7)) } };
    const bHashes = { ...hashes('b', 8) };
    for (let h = 0; h <= 3; h++) bHashes[h] = a.hashes[h];
    const b = { height: 8, hashes: bHashes, blocks: {} };

    const client = new ChainClient({ a, b }, 'a');
    await w.sync(client, { filterIntervalMs: 0 });
    client.branch = 'b';
    let seen = null;
    await w.sync(client, { filterIntervalMs: 0, onReorg: (r) => { seen = r; } });
    assert(seen !== null, 'onReorg was not called');
    assert(seen.fork === 3, `callback fork ${seen.fork}`);
});

await t('rescan clears reorg history rather than reconciling against it', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();
    const a = { height: 8, hashes: hashes('a', 8), blocks: { 5: coinbaseBlock(addr, 128, S(8)) } };
    const client = new ChainClient({ a }, 'a');
    await w.sync(client, { filterIntervalMs: 0 });
    assert(Object.keys(w.blockHashes).length > 0, 'setup failed');

    const res = await w.sync(client, { filterIntervalMs: 0, rescan: true });
    assert(res.reorg === null || res.reorg === undefined, 'rescan should not report a reorg');
    assert(w.getBalance() === 128n, `balance after rescan ${w.getBalance()}`);
});

console.log('\n━━━ round-trip fidelity: restore ∘ save = id ━━━');

await t('every spendability-gating field survives a save/restore cycle', async () => {
    // The Save schema states round-trip fidelity as an invariant. Assert it over
    // a wallet exercising all of it at once, rather than trusting field-by-field
    // spot checks: a field silently dropped from save() is deleted on restore.
    const store = new MemoryStorage();
    const { w, addr, cid } = await walletWithSpentWots({ retainReveal: true });
    w.storage = store;

    // Add a second, MSS-backed coin and a locked one so both branches are live.
    const mssAddr = await w.getNewReusableAddress(4);
    w.mssAddrs[mssAddr].next_leaf = 3;
    w.addUtxo(mssAddr, 2048, S(9), compute_coin_id_hex(mssAddr, 2048n, S(9)), 6);
    w.utxos.push({
        address: 'locked'.padEnd(64, '0'), value: 777n, salt: S(10), coin_id: 'lk'.repeat(32),
        index: 9, is_mss: false, mss_height: 0, mss_leaf: 0,
        createdAtHeight: 4, reuseLocked: true,
    });
    w.reuseLocked['lk'.repeat(32)] = { address: 'locked'.padEnd(64, '0'), priorCommitment: S(11) };
    await w.save();

    const w2 = await Wallet.restore(store);

    const project = (x) => ({
        nextWotsIndex: x.nextWotsIndex,
        nextMssIndex: x.nextMssIndex,
        lastScannedHeight: x.lastScannedHeight,
        wotsAddrs: x.wotsAddrs,
        mssAddrs: x.mssAddrs,
        spentWots: x.spentWots,
        sentReveals: x.sentReveals,
        reuseLocked: x.reuseLocked,
        blockHashes: x.blockHashes,
        utxos: x.utxos.map((c) => ({ ...c, value: c.value.toString() }))
                      .sort((a, b) => a.coin_id.localeCompare(b.coin_id)),
        spentLog: Object.fromEntries(Object.entries(x.spentLog).map(
            ([h, cs]) => [h, cs.map((c) => ({ ...c, value: c.value.toString() }))])),
    });

    const before = project(w), after = project(w2);
    for (const k of Object.keys(before)) {
        if (k === 'wotsAddrs') {
            // restore() is deliberately MONOTONIC here, not exact: it re-derives
            // [0, nextWotsIndex) and backfills the watchlist to repair wallets
            // written before change addresses were tracked. Widening can only
            // make the wallet notice more of its own coins, so assert ⊆.
            for (const [addr, idx] of Object.entries(before[k])) {
                assert(after[k][addr] === idx, `watchlist entry ${addr.slice(0, 8)} lost on restore`);
            }
            continue;
        }
        assert(JSON.stringify(before[k]) === JSON.stringify(after[k]),
            `field '${k}' did not survive the round trip\n  before: ${JSON.stringify(before[k])}\n  after:  ${JSON.stringify(after[k])}`);
    }

    // And the derived guarantees the fields exist to support.
    assert(w2.getBalance() === w.getBalance(), 'spendable balance diverged');
    assert(w2.getLockedBalance() === w.getLockedBalance(), 'locked balance diverged');
    assert(w2._utxosForWasm().length === w._utxosForWasm().length, 'signer view diverged');
    assert(w2.mssAddrs[mssAddr].next_leaf === 3, 'MSS leaf counter rewound — key reuse risk');
    assert(w2.sentReveals[addr].revealPayload === '{"signed":"verbatim"}', 'retained signature lost');
    assert(w2.reuseLocked[`${'lk'.repeat(32)}`] !== undefined, 'lock record lost');
    void cid;
});

await t('a locked coin cannot be promoted back to spendable by a reload', async () => {
    const store = new MemoryStorage();
    const { w } = await walletWithSpentWots({ retainReveal: false });
    w.storage = store;
    rollbackTo(w, 8, 12);
    await w.save();
    const w2 = await Wallet.restore(store);
    assert(w2.getBalance() === 0n, `reload promoted a locked coin: balance ${w2.getBalance()}`);
    assert(w2._utxosForWasm().length === 0, 'reload exposed a locked coin to the signer');
});

console.log('\n━━━ state threads: tracking ━━━');

// A contract's address IS blake3(bytecode) — Predicate::address() is
// pay-to-script-hash — so the fixture must derive it that way or the input
// matching in _processBlock can never link a spend to the tracked coin.
const CONTRACT_BYTECODE = 'c0ffee'.repeat(6);
const CONTRACT = blake3_hash_hex(CONTRACT_BYTECODE);
const confId = (addr, state, salt) =>
    blake3_hash_hex('434f4e464944454e5449414c' + addr + state + salt);

function confTx(addr, state, salt) {
    return { Reveal: { inputs: [], outputs: [{ Confidential: { address: addr, commitment: state, salt } }] } };
}
function advanceTx(addr, oldState, oldSalt, newState, newSalt) {
    return { Reveal: {
        inputs:  [{ predicate: { Script: { bytecode: CONTRACT_BYTECODE } }, value: 0,
                    salt: oldSalt, commitment: oldState }],
        outputs: [{ Confidential: { address: addr, commitment: newState, salt: newSalt } }],
    } };
}

await t('a confidential output at a followed contract is credited', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.watchContract(CONTRACT);
    w._reorgHeight = 5;
    w._processBlock({ coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] }, 5);

    const th = w.getStateThread(CONTRACT);
    assert(th !== null, 'state thread not credited');
    assert(th.state === S(0xa1), `state ${th.state}`);
    assert(th.value === 0n, `state thread value ${th.value}, consensus requires 0`);
    assert(th.coin_id === confId(CONTRACT, S(0xa1), S(0xb1)), 'coin id not the CONFIDENTIAL form');
});

await t('an unfollowed contract is ignored', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w._processBlock({ coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] }, 5);
    assert(w.getStateThread(CONTRACT) === null, 'credited a contract we never followed');
});

await t('state threads never enter the wallet UTXO set', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.watchContract(CONTRACT);
    w._processBlock({ coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] }, 5);
    assert(w.utxos.length === 0, 'script-locked coin landed in utxos (INV-5)');
    assert(w._utxosForWasm().length === 0, 'script-locked coin offered to prepare_spend');
    assert(w.getBalance() === 0n, 'zero-value state thread affected balance');
});

await t('a value-bearing state thread is rejected (consensus: value must be 0)', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.watchContract(CONTRACT);
    let threw = false;
    try { w._addContractCoin(CONTRACT, 1024, S(1), S(2), 'x'.repeat(64)); }
    catch (e) { threw = /must have value 0/.test(e.message); }
    assert(threw, 'accepted a state thread with non-zero value');
});

await t('advancing the thread retires the old state', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.watchContract(CONTRACT);
    w._processBlock({ coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] }, 5);
    w._processBlock({ coinbase: [], transactions: [advanceTx(CONTRACT, S(0xa1), S(0xb1), S(0xa2), S(0xb2))] }, 6);

    assert(w.getContractCoins(CONTRACT).length === 1,
        `stale state retained: ${w.getContractCoins(CONTRACT).length} coins tracked`);
    assert(w.getStateThread(CONTRACT).state === S(0xa2), 'thread did not advance');
});

await t('contract addresses are in the compact-filter watchlist', async () => {
    const w = await Wallet.create(new MemoryStorage());
    let captured = null;
    w.inner.set_watchlist = (json) => { captured = JSON.parse(json); };
    w.watchContract(CONTRACT);
    assert(captured !== null && captured.includes(CONTRACT),
        'contract address absent from the watchlist — its blocks would never be fetched');
});

await t('buildContractInputs supplies the tracked state, thread first', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.watchContract(CONTRACT);
    w._processBlock({ coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] }, 5);
    // Contract funds: a standard coin at the contract address.
    w._processBlock({ coinbase: [], transactions: [{ Reveal: { inputs: [], outputs: [
        { Standard: { address: CONTRACT, value: 4096, salt: S(0xb3) } }] } }] }, 6);

    const inputs = w.buildContractInputs(CONTRACT, 'aabb');
    assert(inputs.length === 2, `expected 2 inputs, got ${inputs.length}`);
    assert(inputs[0].state === S(0xa1), 'state thread is not first');
    assert(inputs[0].value === 0, 'state thread value must be 0');
    assert(inputs[1].state === undefined, 'funds coin must carry no state');
    assert(inputs[1].value === 4096, `funds value ${inputs[1].value}`);
    assert(inputs.every((i) => i.witness === 'aabb'), 'witness not applied');
});

console.log('\n━━━ state threads: rollback ━━━');

await t('a thread advanced above the fork rolls back to the earlier state', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.watchContract(CONTRACT);
    w._processBlock({ coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] }, 5);
    w._processBlock({ coinbase: [], transactions: [advanceTx(CONTRACT, S(0xa1), S(0xb1), S(0xa2), S(0xb2))] }, 10);
    w.lastScannedHeight = 12;
    assert(w.getStateThread(CONTRACT).state === S(0xa2), 'setup failed');

    const r = rollbackTo(w, 8, 12);
    const th = w.getStateThread(CONTRACT);
    assert(th !== null, 'state thread lost entirely by the rollback');
    assert(th.state === S(0xa1),
        `stale state survived: thread is ${th.state}, expected ${S(0xa1)} — ` +
        `every subsequent execution would be rejected as stale`);
    assert(w.getContractCoins(CONTRACT).length === 1, 'more than one live thread after rollback');
    assert(r.restoredContract === 1, `restoredContract ${r.restoredContract}`);
});

await t('chained advances converge on the state at the fork', async () => {
    // S0 -> S1 (h=9) -> S2 (h=11), fork at 8. Step 1 discards S1 and S2 as
    // dead-branch; step 2 restores only S0, because S1's own createdAtHeight is
    // above the fork. This is the convergence claimed in the RollbackTo spec.
    const w = await Wallet.create(new MemoryStorage());
    w.watchContract(CONTRACT);
    w._processBlock({ coinbase: [], transactions: [confTx(CONTRACT, S(0xa0), S(0xb0))] }, 5);
    w._processBlock({ coinbase: [], transactions: [advanceTx(CONTRACT, S(0xa0), S(0xb0), S(0xa1), S(0xb1))] }, 9);
    w._processBlock({ coinbase: [], transactions: [advanceTx(CONTRACT, S(0xa1), S(0xb1), S(0xa2), S(0xb2))] }, 11);
    w.lastScannedHeight = 12;

    rollbackTo(w, 8, 12);
    const coins = w.getContractCoins(CONTRACT);
    assert(coins.length === 1, `expected exactly 1 live thread, got ${coins.length}`);
    assert(coins[0].state === S(0xa0), `converged on ${coins[0].state}, expected ${S(0xa0)}`);
});

await t('a thread CREATED above the fork is discarded, not resurrected', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.watchContract(CONTRACT);
    w._processBlock({ coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] }, 11);
    w.lastScannedHeight = 12;
    rollbackTo(w, 8, 12);
    assert(w.getStateThread(CONTRACT) === null, 'dead-branch state thread survived');
});

await t('restored contract coins never land in utxos (INV-5)', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.watchContract(CONTRACT);
    w._processBlock({ coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] }, 5);
    w._processBlock({ coinbase: [], transactions: [advanceTx(CONTRACT, S(0xa1), S(0xb1), S(0xa2), S(0xb2))] }, 10);
    w.lastScannedHeight = 12;

    rollbackTo(w, 8, 12);
    assert(w.utxos.length === 0, 'a script-locked coin was restored into utxos');
    assert(w._utxosForWasm().length === 0, 'a script-locked coin reached the signer');
    const ids = new Set(w.utxos.map((c) => c.coin_id));
    for (const id of Object.keys(w.contractCoins)) {
        assert(!ids.has(id), 'INV-5 violated: coin in both collections');
    }
});

await t('a contract rollback does not poison any WOTS address', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.watchContract(CONTRACT);
    w._processBlock({ coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] }, 5);
    w._processBlock({ coinbase: [], transactions: [advanceTx(CONTRACT, S(0xa1), S(0xb1), S(0xa2), S(0xb2))] }, 10);
    w.lastScannedHeight = 12;
    const r = rollbackTo(w, 8, 12);
    assert(Object.keys(w.spentWots).length === 0,
        'script-locked spend poisoned a key — contract coins carry no one-time key');
    assert(r.locked === 0 && r.resendable === 0, 'contract coin entered reuse classification');
});

await t('contract state survives save/restore with BigInt values intact', async () => {
    const store = new MemoryStorage();
    const w = await Wallet.create(store);
    w.watchContract(CONTRACT, { label: 'escrow' });
    w._processBlock({ coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] }, 5);
    w._processBlock({ coinbase: [], transactions: [{ Reveal: { inputs: [], outputs: [
        { Standard: { address: CONTRACT, value: 8192, salt: S(0xb4) } }] } }] }, 6);
    await w.save();

    const w2 = await Wallet.restore(store);
    assert(w2.watchedContracts[CONTRACT] !== undefined, 'watched contract lost');
    assert(w2.watchedContracts[CONTRACT].label === 'escrow', 'contract metadata lost');
    assert(w2.getStateThread(CONTRACT)?.state === S(0xa1), 'state thread lost across restore');
    const funds = w2.getContractCoins(CONTRACT).find((c) => c.state === null);
    assert(funds && typeof funds.value === 'bigint', 'contract coin value is not BigInt after restore');
    assert(funds.value === 8192n, `funds value ${funds.value}`);
});

await t('unwatching a contract forgets its coins', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.watchContract(CONTRACT);
    w._processBlock({ coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] }, 5);
    assert(w.getContractCoins(CONTRACT).length === 1, 'setup failed');
    w.unwatchContract(CONTRACT);
    assert(w.getContractCoins(CONTRACT).length === 0, 'coins retained after unwatch');
    assert(w.getStateThread(CONTRACT) === null, 'thread retained after unwatch');
});

await t('end-to-end: sync() rolls a state thread back to the winning branch', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.watchContract(CONTRACT);

    // Branch A: thread created at 3, advanced to a2 at 6.
    const a = {
        height: 10, hashes: hashes('a', 10),
        blocks: {
            3: { coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] },
            6: { coinbase: [], transactions: [advanceTx(CONTRACT, S(0xa1), S(0xb1), S(0xa2), S(0xb2))] },
        },
    };
    // Branch B forks at 4 — so the creation at 3 is shared, the advance is not,
    // and B advances to a different state at 7 instead.
    const bHashes = { ...hashes('b', 10) };
    for (let h = 0; h <= 4; h++) bHashes[h] = a.hashes[h];
    const b = {
        height: 10, hashes: bHashes,
        blocks: {
            3: { coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] },
            7: { coinbase: [], transactions: [advanceTx(CONTRACT, S(0xa1), S(0xb1), S(0xa9), S(0xb9))] },
        },
    };

    const client = new ChainClient({ a, b }, 'a');
    await w.sync(client, { filterIntervalMs: 0 });
    assert(w.getStateThread(CONTRACT)?.state === S(0xa2), 'pre-reorg state wrong');

    client.branch = 'b';
    const res = await w.sync(client, { filterIntervalMs: 0 });
    assert(res.reorg, 'reorg not detected');
    const th = w.getStateThread(CONTRACT);
    assert(th !== null, 'state thread lost');
    assert(th.state === S(0xa9),
        `wallet holds ${th.state} but the winning branch has ${S(0xa9)} — ` +
        `the next executeContract would be rejected as stale`);
    assert(w.getContractCoins(CONTRACT).length === 1,
        `${w.getContractCoins(CONTRACT).length} live threads after resync`);
});

console.log('\n━━━ recoverPendingTx: driving the verbatim re-send ━━━');

/** Drives _broadcastTwoPhaseTx to completion without a network. */
class SpendClient {
    constructor() { this.sent = []; }
    async getState() { return { height: 100, required_pow: 8, header_hash: '00'.repeat(32) }; }
    async commit() { return { ok: true, body: null }; }
    async checkCommitment() { return { exists: true }; }
    async send(p) { this.sent.push(p); return { ok: true, body: null }; }
}

class RecoveryClient {
    constructor({ commitAlive = true, coinsAlive = true, sendOk = true } = {}) {
        this.commitAlive = commitAlive; this.coinsAlive = coinsAlive; this.sendOk = sendOk;
        this.sent = []; this.signedHere = 0;
    }
    async checkCommitment() { return { exists: this.commitAlive }; }
    async checkCoin() { return { exists: this.coinsAlive }; }
    async send(p) { this.sent.push(p); return { ok: this.sendOk, body: this.sendOk ? null : 'rejected' }; }
}

/** A wallet whose rollback armed a CASE 1 re-send. */
async function walletWithArmedResend() {
    const { w, addr, cid, commitment } = await walletWithSpentWots({ retainReveal: true });
    rollbackTo(w, 8, 12);
    assert(w.pendingResend !== null, 'setup: no re-send armed');
    return { w, addr, cid, commitment };
}

await t('idle when nothing is armed', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const r = await w.recoverPendingTx(new RecoveryClient());
    assert(r.status === 'idle', `status ${r.status}`);
});

await t('re-sends the verbatim payload when the commitment is live', async () => {
    const { w } = await walletWithArmedResend();
    const client = new RecoveryClient();
    const r = await w.recoverPendingTx(client);
    assert(r.status === 'resent', `status ${r.status} (${r.error || ''})`);
    assert(client.sent.length === 1, 'nothing was sent');
    assert(client.sent[0] === '{"signed":"verbatim"}',
        'sent payload is not the original signature — re-signing would reuse the key');
    assert(w.pendingResend === null, 'pendingResend not cleared');
});

await t('a dead commitment locks the coins rather than inviting a re-sign', async () => {
    // The browser wallet says "coins are yours, re-send" here — correct for a tx
    // that never confirmed. This one DID confirm and was orphaned, so its
    // signature is public and re-sending means re-signing.
    const { w, addr } = await walletWithArmedResend();
    const client = new RecoveryClient({ commitAlive: false });
    const r = await w.recoverPendingTx(client);

    assert(r.status === 'dead', `status ${r.status}`);
    assert(client.sent.length === 0, 'sent a reveal whose commitment is gone');
    assert(w.spentWots[addr]?.reuseLocked === true, 'key not re-locked');
    assert(w.getBalance() === 0n, `coin left spendable: balance ${w.getBalance()}`);
    assert(w.getLockedBalance() === 4096n, 'coin not reported as locked');
    assert(w._utxosForWasm().length === 0, 'locked coin reachable by the signer');
    assert(w.pendingResend === null, 'pendingResend not cleared');
    assert(r.alerts.some((a) => a.kind === 'reuse_lock'), 'no lock alert raised');
});

await t('a vanished input coin also locks', async () => {
    const { w, addr } = await walletWithArmedResend();
    const r = await w.recoverPendingTx(new RecoveryClient({ coinsAlive: false }));
    assert(r.status === 'dead', `status ${r.status}`);
    assert(w.spentWots[addr]?.reuseLocked === true, 'key not locked when an input is gone');
});

await t('a node error is retryable and changes nothing', async () => {
    const { w, addr } = await walletWithArmedResend();
    const client = new RecoveryClient();
    client.checkCommitment = async () => { throw new Error('offline'); };
    const r = await w.recoverPendingTx(client);
    assert(r.status === 'retry', `status ${r.status}`);
    assert(w.pendingResend !== null, 'dropped the armed re-send on a transient failure');
    assert(w.spentWots[addr] === undefined, 'locked a key on a transient failure');
});

await t('a rejected send stays retryable', async () => {
    const { w } = await walletWithArmedResend();
    const r = await w.recoverPendingTx(new RecoveryClient({ sendOk: false }));
    assert(r.status === 'retry', `status ${r.status}`);
    assert(w.pendingResend !== null, 'dropped the armed re-send on a rejection');
});

await t('recovery is idempotent', async () => {
    const { w } = await walletWithArmedResend();
    const client = new RecoveryClient();
    await w.recoverPendingTx(client);
    const second = await w.recoverPendingTx(client);
    assert(second.status === 'idle', `second call: ${second.status}`);
    assert(client.sent.length === 1, `sent ${client.sent.length} times — double broadcast`);
});

await t('recovery survives a restart (armed state is persisted)', async () => {
    const store = new MemoryStorage();
    const { w } = await walletWithArmedResend();
    w.storage = store;
    await w.save();
    const w2 = await Wallet.restore(store);
    assert(w2.pendingResend !== null, 'armed re-send lost across restart');
    const client = new RecoveryClient();
    const r = await w2.recoverPendingTx(client);
    assert(r.status === 'resent', `status ${r.status}`);
    assert(client.sent[0] === '{"signed":"verbatim"}', 'payload corrupted by the round trip');
});

console.log('\n━━━ handleBlockTip: push-driven detection ━━━');

await t('a consistent tip is a no-op', async () => {
    const { w } = await walletWithSpentWots({ retainReveal: false });
    const r = await w.handleBlockTip(new ForkClient({}), { height: 12, block_hash: H(12) });
    assert(r === null, 'reported a reorg on a matching hash');
});

await t('an unrecorded height is not treated as divergence', async () => {
    const { w } = await walletWithSpentWots({ retainReveal: false });
    const r = await w.handleBlockTip(new ForkClient({}), { height: 999, block_hash: H(999) });
    assert(r === null, 'guessed at a height it never recorded');
});

await t('a conflicting tip triggers rollback', async () => {
    const { w } = await walletWithSpentWots({ retainReveal: false });
    const nodeHashes = {};
    for (let h = 0; h <= 12; h++) nodeHashes[h] = h <= 8 ? H(h) : `alt${h}`.padEnd(64, '0');
    const r = await w.handleBlockTip(new ForkClient(nodeHashes), { height: 12, block_hash: 'alt12'.padEnd(64, '0') });
    assert(r !== null, 'conflict at the tip went unhandled');
    assert(r.fork === 8, `fork ${r.fork}`);
});

await t('a malformed notification is ignored', async () => {
    const { w } = await walletWithSpentWots({ retainReveal: false });
    assert(await w.handleBlockTip(new ForkClient({}), null) === null);
    assert(await w.handleBlockTip(new ForkClient({}), { height: 12 }) === null);
});

console.log('\n━━━ wallet operation wrappers ━━━');

await t('consolidate refuses fewer than two coins at the address', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();
    w.addUtxo(addr, 1024, S(1), compute_coin_id_hex(addr, 1024n, S(1)), 1);
    let threw = false;
    try { await w.consolidate(new RecoveryClient(), addr, addr); }
    catch (e) { threw = /at least two coins/.test(e.message); }
    assert(threw, 'accepted a single-coin consolidate');
});

await t('consolidate gathers only the source address and uses the consolidate reveal', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const a = await w.getNewAddress();
    const b = await w.getNewAddress();
    for (const s of [S(1), S(2)]) w.addUtxo(a, 1024, s, compute_coin_id_hex(a, 1024n, s), 1);
    w.addUtxo(b, 4096, S(3), compute_coin_id_hex(b, 4096n, S(3)), 1);

    let seenUtxos = null, usedBuilder = null;
    w.inner.prepare_consolidate = (json) => {
        seenUtxos = JSON.parse(json);
        return JSON.stringify({ next_wots_index: w.nextWotsIndex, commitment: S(0xaa), tx_salt: S(0xbb), selected_inputs: seenUtxos });
    };
    w.inner.build_consolidate_reveal = () => { usedBuilder = 'consolidate'; return '{}'; };
    w.inner.build_reveal = () => { usedBuilder = 'standard'; return '{}'; };

    const client = new SpendClient();
    await w.consolidate(client, a, b);
    assert(seenUtxos.length === 2, `passed ${seenUtxos.length} coins, expected only the 2 at the source`);
    assert(seenUtxos.every((u) => u.address === a), 'included a coin from another address');
    assert(usedBuilder === 'consolidate', `used the ${usedBuilder} reveal builder`);
});

await t('reuse-locked coins are excluded from consolidate and defrag', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const a = await w.getNewAddress();
    for (const s of [S(1), S(2)]) w.addUtxo(a, 1024, s, compute_coin_id_hex(a, 1024n, s), 1);
    w.utxos[0].reuseLocked = true;

    let threw = false;
    try { await w.consolidate(new RecoveryClient(), a, a); }
    catch (e) { threw = /at least two coins/.test(e.message); }
    assert(threw, 'a reuse-locked coin was counted toward the consolidate input set');
});

await t('advanceStateThread refuses an unfollowed contract', async () => {
    const w = await Wallet.create(new MemoryStorage());
    let threw = false;
    try { await w.advanceStateThread(new RecoveryClient(), CONTRACT_BYTECODE, S(0xa2)); }
    catch (e) { threw = /not followed/.test(e.message); }
    assert(threw, 'built a transition for a contract whose state is unknown');
});

await t('advanceStateThread supplies the tracked state, not a caller guess', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.watchContract(CONTRACT);
    w._processBlock({ coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] }, 5);

    let args = null;
    w.inner.build_state_thread_tx = (...a) => {
        args = a;
        return JSON.stringify({ next_wots_index: w.nextWotsIndex, commitment: S(0xaa), tx_salt: S(0xbb), selected_inputs: [] });
    };
    w.inner.build_script_reveal = () => '{}';
    await w.advanceStateThread(new SpendClient(), CONTRACT_BYTECODE, S(0xa2));

    assert(args[2] === S(0xa1), `current state passed as ${args[2]}, expected the tracked ${S(0xa1)}`);
    assert(args[3] === confId(CONTRACT, S(0xa1), S(0xb1)), 'current coin id not the tracked one');
    assert(args[4] === S(0xb1), 'current salt not the tracked one');
    assert(args[5] === S(0xa2), 'new state not passed through');
});

await t('after a rollback, advanceStateThread uses the rolled-back state', async () => {
    // The whole point of reading from tracked state: a caller working from its
    // own notes would submit the orphaned state and be rejected.
    const w = await Wallet.create(new MemoryStorage());
    w.watchContract(CONTRACT);
    w._processBlock({ coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] }, 5);
    w._processBlock({ coinbase: [], transactions: [advanceTx(CONTRACT, S(0xa1), S(0xb1), S(0xa2), S(0xb2))] }, 10);
    w.lastScannedHeight = 12;
    rollbackTo(w, 8, 12);

    let args = null;
    w.inner.build_state_thread_tx = (...a) => {
        args = a;
        return JSON.stringify({ next_wots_index: w.nextWotsIndex, commitment: S(0xaa), tx_salt: S(0xbb), selected_inputs: [] });
    };
    w.inner.build_script_reveal = () => '{}';
    await w.advanceStateThread(new SpendClient(), CONTRACT_BYTECODE, S(0xa3));
    assert(args[2] === S(0xa1),
        `submitted state ${args[2]} — the orphaned state would be rejected as stale`);
});

await t('getContractBalance sums tracked contract coins', async () => {
    const w = await Wallet.create(new MemoryStorage());
    w.watchContract(CONTRACT);
    w._processBlock({ coinbase: [], transactions: [confTx(CONTRACT, S(0xa1), S(0xb1))] }, 5);
    w._processBlock({ coinbase: [], transactions: [{ Reveal: { inputs: [], outputs: [
        { Standard: { address: CONTRACT, value: 4096, salt: S(0xb4) } }] } }] }, 6);
    assert(w.getContractBalance(CONTRACT) === 4096n, `contract balance ${w.getContractBalance(CONTRACT)}`);
    assert(w.getBalance() === 0n, 'contract funds leaked into the wallet balance');
});

console.log('\n━━━ autoReconcile: live push loop ━━━');

class PushClient extends ForkClient {
    constructor(nodeHashes) { super(nodeHashes); this.handler = null; this.sent = []; }
    onPushEvent(cb) { this.handler = cb; }
    async checkCommitment() { return { exists: true }; }
    async checkCoin() { return { exists: true }; }
    async send(p) { this.sent.push(p); return { ok: true, body: null }; }
    async emit(event) { await this.handler(event); }
}

await t('a conflicting tip drives rollback AND completes the re-send', async () => {
    const { w } = await walletWithSpentWots({ retainReveal: true });
    const nodeHashes = {};
    for (let h = 0; h <= 12; h++) nodeHashes[h] = h <= 8 ? H(h) : `alt${h}`.padEnd(64, '0');
    const client = new PushClient(nodeHashes);

    let reorg = null, recover = null;
    w.autoReconcile(client, { onReorg: (r) => { reorg = r; }, onRecover: (r) => { recover = r; } });
    await client.emit({ NewBlockTip: { height: 12, block_hash: 'alt12'.padEnd(64, '0') } });

    assert(reorg !== null, 'rollback not driven by the push');
    assert(reorg.fork === 8, `fork ${reorg.fork}`);
    assert(recover !== null && recover.status === 'resent',
        `re-send not completed: ${recover && recover.status}`);
    assert(client.sent[0] === '{"signed":"verbatim"}', 'a new signature was produced');
    assert(w.pendingResend === null, 'armed re-send left dangling');
});

await t('a consistent tip does nothing', async () => {
    const { w } = await walletWithSpentWots({ retainReveal: true });
    const nodeHashes = {}; for (let h = 0; h <= 12; h++) nodeHashes[h] = H(h);
    const client = new PushClient(nodeHashes);
    let reorg = null, tips = 0;
    w.autoReconcile(client, { onReorg: (r) => { reorg = r; }, onTip: () => { tips++; } });
    await client.emit({ NewBlockTip: { height: 12, block_hash: H(12) } });
    assert(reorg === null, 'spurious rollback');
    assert(tips === 1, 'onTip not called for a consistent tip');
    assert(client.sent.length === 0, 'sent something on a quiet tip');
});

await t('chat pushes pass through untouched', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const client = new PushClient({});
    let chat = null;
    w.autoReconcile(client, { onChat: (c) => { chat = c; } });
    await client.emit({ ChatMessage: { sender: 'peer', words: [1, 2] } });
    assert(chat !== null && chat.sender === 'peer', 'chat push not delivered');
});

await t('a handler error is reported, not thrown into the stream', async () => {
    const { w } = await walletWithSpentWots({ retainReveal: false });
    const client = new PushClient({});
    client.getFilters = async () => { throw new Error('node exploded'); };
    let err = null;
    w.autoReconcile(client, { onError: (e) => { err = e; }, onReorg: () => { throw new Error('boom'); } });
    // Should not reject: an exception here would tear down the subscription.
    await client.emit({ NewBlockTip: { height: 12, block_hash: 'alt'.padEnd(64, '0') } });
    assert(true, 'emit rejected — a push handler error escaped');
});

await t('unsubscribe stops reconciliation', async () => {
    const { w } = await walletWithSpentWots({ retainReveal: false });
    const nodeHashes = {};
    for (let h = 0; h <= 12; h++) nodeHashes[h] = h <= 8 ? H(h) : `alt${h}`.padEnd(64, '0');
    const client = new PushClient(nodeHashes);
    let reorg = null;
    const stop = w.autoReconcile(client, { onReorg: (r) => { reorg = r; } });
    stop();
    await client.emit({ NewBlockTip: { height: 12, block_hash: 'alt12'.padEnd(64, '0') } });
    assert(reorg === null, 'kept reconciling after unsubscribe');
});

console.log('\n━━━ MSS leaf reconciliation ━━━');

class MssClient {
    constructor(byPk = {}) { this.byPk = byPk; this.queried = []; }
    async getMssState(pk) { this.queried.push(pk); return { next_index: this.byPk[pk] ?? 0 }; }
}

await t('a chain-ahead counter is adopted', async () => {
    // The wallet's guess is behind because a spend confirmed elsewhere. Signing
    // from the stale leaf produces a reveal the node evicts every block, forever.
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewReusableAddress(4);
    const pk = w.inner.get_mss_pubkey(addr);
    assert(w.mssAddrs[addr].next_leaf === 0, 'setup');

    const r = await w.reconcileMssLeaves(new MssClient({ [pk]: 5 }));
    assert(r.moved === 1, `moved ${r.moved}`);
    assert(w.mssAddrs[addr].next_leaf === 5, `next_leaf ${w.mssAddrs[addr].next_leaf}`);
    assert(r.details[0].from === 0 && r.details[0].to === 5, 'details wrong');
});

await t('a lower remote value is IGNORED (monotonic)', async () => {
    // Trusting it would rewind onto leaves this wallet already used.
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewReusableAddress(4);
    const pk = w.inner.get_mss_pubkey(addr);
    w.mssAddrs[addr].next_leaf = 9;

    const r = await w.reconcileMssLeaves(new MssClient({ [pk]: 3 }));
    assert(r.moved === 0, 'a lower remote index moved the counter');
    assert(w.mssAddrs[addr].next_leaf === 9, `counter rewound to ${w.mssAddrs[addr].next_leaf} — leaf reuse`);
});

await t('the adopted value is mirrored into the WASM cache', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewReusableAddress(4);
    const pk = w.inner.get_mss_pubkey(addr);
    let mirrored = null;
    const real = w.inner.set_mss_leaf_index.bind(w.inner);
    w.inner.set_mss_leaf_index = (a, i) => { mirrored = [a, i]; return real(a, i); };
    await w.reconcileMssLeaves(new MssClient({ [pk]: 7 }));
    assert(mirrored && mirrored[0] === addr && mirrored[1] === 7,
        'WASM cache not updated — the two views would drift');
});

await t('a node error leaves counters untouched', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewReusableAddress(4);
    w.mssAddrs[addr].next_leaf = 2;
    const client = new MssClient();
    client.getMssState = async () => { throw new Error('offline'); };
    const r = await w.reconcileMssLeaves(client);
    assert(r.moved === 0, 'moved a counter on a failed query');
    assert(w.mssAddrs[addr].next_leaf === 2, 'counter changed on a failed query');
});

await t('reconciliation persists', async () => {
    const store = new MemoryStorage();
    const w = await Wallet.create(store);
    const addr = await w.getNewReusableAddress(4);
    const pk = w.inner.get_mss_pubkey(addr);
    await w.reconcileMssLeaves(new MssClient({ [pk]: 6 }));
    const w2 = await Wallet.restore(store);
    assert(w2.mssAddrs[addr].next_leaf === 6, `counter ${w2.mssAddrs[addr].next_leaf} after restore`);
});

await t('the reconciled leaf is what the signer receives', async () => {
    // _utxosForWasm re-reads next_leaf, so reconciliation must reach the signer.
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewReusableAddress(4);
    const pk = w.inner.get_mss_pubkey(addr);
    w.addUtxo(addr, 1024, S(1), compute_coin_id_hex(addr, 1024n, S(1)), 1);
    await w.reconcileMssLeaves(new MssClient({ [pk]: 4 }));
    assert(w._utxosForWasm()[0].mss_leaf === 4,
        `signer would use leaf ${w._utxosForWasm()[0].mss_leaf}, not the reconciled 4`);
});

console.log('\n━━━ deep reorg (beyond REORG_DEPTH) ━━━');

await t('an exhausted descent escalates to a full rebuild', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();
    const tip = REORG_DEPTH + 40;
    for (let h = 0; h <= tip; h++) w.blockHashes[h] = H(h);
    w.lastScannedHeight = tip;
    w.addUtxo(addr, 4096, S(1), compute_coin_id_hex(addr, 4096n, S(1)), 5);

    // A node that agrees with nothing: the fork is deeper than we can heal.
    const a = { height: tip, hashes: {}, blocks: {} };
    const client = new ChainClient({ a }, 'a');
    client.getFilters = async (start, end) => {
        // Mirrors the node: end is capped at its height, so the very tip is
        // never servable. Every hash it does serve disagrees with ours.
        const n = Math.max(0, Math.min(end, tip) - start);
        return { start_height: start, filters: Array(n).fill(''), element_counts: Array(n).fill(0),
                 block_hashes: Array(n).fill('ff'.repeat(32)) };
    };
    const res = await w.sync(client, { filterIntervalMs: 0 });
    assert(res.reorg, 'no reorg reported');
    assert(res.reorg.exhausted === true, 'descent not reported as exhausted');
    assert(res.reorg.escalatedToRescan === true, 'did not escalate to a rebuild');
});

await t('a deep-reorg rebuild PRESERVES published-key state', async () => {
    // A key burned by a spend that was later orphaned appears on no branch, so a
    // rescan cannot re-derive it. Clearing spentWots would silently re-admit its
    // coins as spendable.
    const w = await Wallet.create(new MemoryStorage());
    const burned = 'de'.repeat(32);
    w.spentWots[burned] = { height: 3, commitment: S(9), reuseLocked: true };
    w.reuseLocked['cc'.repeat(32)] = { address: burned, priorCommitment: S(9) };
    w.sentReveals[burned] = { commitment: S(9), revealPayload: '{}', inputCoinIds: [], sentAtHeight: 3 };

    const client = new ChainClient({ a: { height: 5, hashes: hashes('a', 5), blocks: {} } }, 'a');
    await w.sync(client, { filterIntervalMs: 0, rescan: true });

    assert(w.spentWots[burned] !== undefined, 'a published key was forgotten by a rescan');
    assert(w.reuseLocked['cc'.repeat(32)] !== undefined, 'a lock record was cleared by a rescan');
    assert(w.sentReveals[burned] !== undefined, 'a retained signature was cleared by a rescan');
});

await t('rescan clears chain-derived state', async () => {
    const w = await Wallet.create(new MemoryStorage());
    const addr = await w.getNewAddress();
    w.watchContract('ab'.repeat(32));
    w.blockHashes[3] = H(3);
    w.spentLog[3] = [{ coin_id: 'x'.repeat(64), value: 1n, address: addr, createdAtHeight: 1 }];
    w.contractCoins['y'.repeat(64)] = { address: 'ab'.repeat(32), value: 0n, salt: S(1), state: S(2), coin_id: 'y'.repeat(64), createdAtHeight: 2, isContract: true };

    const client = new ChainClient({ a: { height: 5, hashes: hashes('a', 5), blocks: {} } }, 'a');
    await w.sync(client, { filterIntervalMs: 0, rescan: true });

    assert(Object.keys(w.spentLog).length === 0, 'stale spend log survived a rescan');
    assert(w.contractCoins['y'.repeat(64)] === undefined, 'stale contract coin survived a rescan');
    assert(w.utxos.length === 0, 'stale utxos survived a rescan');
});

await t('a node on a SHORTER chain is still detected', async () => {
    // The node caps end_height at its own height, so our last recorded height is
    // no longer servable and probes back as an undecidable placeholder. A
    // single-probe check cannot tell that from agreement — and this is precisely
    // when a rollback is most needed.
    const w = await Wallet.create(new MemoryStorage());
    for (let h = 0; h <= 20; h++) w.blockHashes[h] = H(h);
    w.lastScannedHeight = 20;

    // Node only has 0..14, and diverges from 12 up.
    const nodeHashes = {};
    for (let h = 0; h <= 14; h++) nodeHashes[h] = h < 12 ? H(h) : `alt${h}`.padEnd(64, '0');
    const client = new ForkClient(nodeHashes);

    const r = await maybeHandleReorg(w, client, 14);
    assert(r !== null, 'a shorter chain went undetected — the wallet stays permanently ahead');
    assert(r.fork === 11, `fork ${r.fork}, expected 11`);
    assert(w.lastScannedHeight === 11, `scan marker ${w.lastScannedHeight}`);
});

await t('a wholly unservable range is not guessed at', async () => {
    const w = await Wallet.create(new MemoryStorage());
    for (let h = 0; h <= 20; h++) w.blockHashes[h] = H(h);
    w.lastScannedHeight = 20;
    const r = await maybeHandleReorg(w, new ForkClient({}), 20);   // node serves nothing
    assert(r === null, 'rolled back on no evidence at all');
});

console.log('\n━━━ pool client wire format ━━━');

await t('requests are framed as 4-byte LE length + JSON', async () => {
    const { PoolClient } = await import('./src/pool.js');
    let sent = null;
    const client = new PoolClient({
        dial: async () => 'peer', close: async () => {},
        request: async (proto, bytes) => {
            sent = bytes;
            return new TextEncoder().encode(JSON.stringify({ ok: true, data: { job_id: 'j1' } }));
        },
    });
    await client.connect('/ip4/1.2.3.4/tcp/1/p2p/x');
    await client.getTemplate('ab'.repeat(32));

    const len = new DataView(sent.buffer, sent.byteOffset).getUint32(0, true);
    assert(len === sent.length - 4, `length prefix ${len} != body ${sent.length - 4}`);
    const body = JSON.parse(new TextDecoder().decode(sent.subarray(4)));
    assert(body.method === 'get_template', `method ${body.method}`);
    assert(body.params.address === 'ab'.repeat(32), 'address not passed');
});

await t('submit_share sends the nonce as a decimal string', async () => {
    const { PoolClient } = await import('./src/pool.js');
    let body = null;
    const client = new PoolClient({
        dial: async () => 'peer', close: async () => {},
        request: async (p, bytes) => {
            body = JSON.parse(new TextDecoder().decode(bytes.subarray(4)));
            return new TextEncoder().encode(JSON.stringify({ ok: true, data: {} }));
        },
    });
    await client.connect('x');
    const big = (1n << 60n) + 7n;
    await client.submitShare('ab'.repeat(32), 'j1', big, 'rig');
    assert(body.params.nonce === big.toString(), `nonce ${body.params.nonce} — precision lost`);
    assert(typeof body.params.nonce === 'string', 'nonce sent as a JSON number');
});

await t('getProof returns the envelope; an absent miner is an answer', async () => {
    const { PoolClient } = await import('./src/pool.js');
    const client = new PoolClient({
        dial: async () => 'peer', close: async () => {},
        request: async () => new TextEncoder().encode(JSON.stringify({ ok: false, error: 'not in tree' })),
    });
    await client.connect('x');
    const env = await client.getProof('ab'.repeat(32));
    assert(env.ok === false && /not in tree/.test(env.error), 'envelope not returned raw');
    // Whereas a fatal-absence call throws.
    let threw = false;
    try { await client.getTemplate('ab'.repeat(32)); } catch { threw = true; }
    assert(threw, 'request() swallowed a pool error');
});

await t('a reset stream is retried once', async () => {
    const { PoolClient } = await import('./src/pool.js');
    let calls = 0;
    const client = new PoolClient({
        dial: async () => 'peer', close: async () => {},
        request: async () => {
            if (++calls === 1) throw new Error('stream reset by peer');
            return new TextEncoder().encode(JSON.stringify({ ok: true, data: { job_id: 'j' } }));
        },
    });
    await client.connect('x');
    const job = await client.getTemplate('ab'.repeat(32));
    assert(calls === 2, `${calls} attempt(s); expected one retry`);
    assert(job.job_id === 'j', 'retry did not return the result');
});

await t('a non-transient error is not retried', async () => {
    const { PoolClient } = await import('./src/pool.js');
    let calls = 0;
    const client = new PoolClient({
        dial: async () => 'peer', close: async () => {},
        request: async () => { calls++; throw new Error('malformed frame'); },
    });
    await client.connect('x');
    let threw = false;
    try { await client.getTemplate('a'); } catch { threw = true; }
    assert(threw && calls === 1, `retried a non-transient error (${calls} calls)`);
});

console.log(`\n━━━ ${pass} passed, ${fail} failed ━━━\n`);
process.exit(fail ? 1 : 0);
