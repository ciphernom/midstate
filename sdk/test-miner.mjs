// test-miner.mjs — Solo mining regression tests.
//
// The failure mode this guards against is silent: blocks get found and then
// rejected, which is indistinguishable from "mining doesn't work". The three
// causes are all template-lifecycle bugs — grinding the wrong midstate, letting
// the locked timestamp go stale, and getting the coinbase total wrong — so the
// tests assert on what is sent, not merely that something was.
//
// Includes one real end-to-end PoW round trip through the WASM VDF.
//
// Run: node test-miner.mjs

import fs from 'fs/promises';
import initWasm, * as W from './pkg/wasm_wallet.js';
import { Wallet } from './src/wallet.js';
import { MemoryStorage } from './src/storage.js';
import { Miner, TEMPLATE_MAX_AGE_MS } from './src/miner.js';

await initWasm({ module_or_path: await fs.readFile('./pkg/wasm_wallet_bg.wasm') });

let pass = 0, fail = 0;
async function t(name, fn) {
    try { await fn(); console.log(`  ✅ ${name}`); pass++; }
    catch (e) { console.log(`  ❌ ${name} — ${e.message}`); fail++; }
}
function assert(c, m) { if (!c) throw new Error(m || 'assertion failed'); }

const MIDSTATE = 'a5'.repeat(32);
const EASY = 'ff'.repeat(32);
const IMPOSSIBLE = '00'.repeat(31) + '01';

/**
 * A node that serves templates the way `finish_template` does: a locked
 * timestamp, a mining_midstate distinct from post_tx_midstate, and a coinbase
 * total it will insist on.
 */
class MockNode {
    constructor({ height = 100, reward = 1073741824, fees = 0, midstate = MIDSTATE, target = EASY } = {}) {
        this.height = height; this.reward = reward; this.fees = fees;
        this.midstate = midstate; this.target = target;
        this.templateRequests = []; this.submitted = []; this.acceptSubmissions = true;
        this.lockedTimestamp = 1_700_000_000;
    }
    async getState() {
        return { height: this.height, block_reward: this.reward, required_pow: 8, header_hash: '00'.repeat(32), target: this.target };
    }
    async getBlockTemplate(coinbase) {
        this.templateRequests.push(coinbase);
        const total = coinbase.reduce((a, c) => a + Number(c.value), 0);
        const expected = this.reward + this.fees;
        if (total !== expected) {
            return { ok: false, status: 409, json: async () => ({ error: 'Coinbase total mismatch', expected_total: expected, block_reward: this.reward, total_fees: this.fees }) };
        }
        return { ok: true, status: 200, json: async () => ({
            mining_midstate: this.midstate,
            target: this.target,
            total_fees: this.fees,
            block_reward: this.reward,
            batch_template: {
                prev_midstate: '11'.repeat(32), prev_header_hash: '22'.repeat(32),
                transactions: [], coinbase, timestamp: this.lockedTimestamp,
                target: this.target, state_root: '33'.repeat(32),
                extension: { nonce: 0, final_hash: new Array(32).fill(0) },
            },
        }) };
    }
    async submitBatch(batch) {
        this.submitted.push(JSON.parse(JSON.stringify(batch)));
        return this.acceptSubmissions ? { ok: true, body: null } : { ok: false, body: 'stale parent' };
    }
}

const mkWallet = () => Wallet.create(new MemoryStorage());

console.log('\n━━━ coinbase total negotiation ━━━');

await t('a zero-fee mempool needs one request', async () => {
    const w = await mkWallet(); const node = new MockNode({ fees: 0 });
    const m = new Miner(w, node);
    const tmpl = await m.fetchTemplate();
    assert(node.templateRequests.length === 1, `${node.templateRequests.length} requests for a trivial case`);
    assert(tmpl.mining_midstate === MIDSTATE, 'wrong midstate returned');
});

await t('fees are learned from the 409 and the retry succeeds', async () => {
    const w = await mkWallet(); const node = new MockNode({ fees: 5000 });
    const m = new Miner(w, node);
    const tmpl = await m.fetchTemplate();
    assert(node.templateRequests.length === 2, `expected one retry, saw ${node.templateRequests.length} requests`);
    const total = node.templateRequests[1].reduce((a, c) => a + Number(c.value), 0);
    assert(total === node.reward + node.fees, `retried with ${total}, expected ${node.reward + node.fees}`);
    assert(tmpl.total_fees === 5000, 'fees not carried through');
});

await t('the learned total is reused, so later templates need one request', async () => {
    // The browser wallet re-guesses block_reward every time and pays the round
    // trip on every block whenever the mempool is non-empty.
    const w = await mkWallet(); const node = new MockNode({ fees: 5000 });
    const m = new Miner(w, node);
    await m.fetchTemplate();
    const after = node.templateRequests.length;
    await m.fetchTemplate();
    assert(node.templateRequests.length - after === 1,
        `second template took ${node.templateRequests.length - after} requests; the total was not carried forward`);
});

await t('a non-mismatch error surfaces its message, not a TypeError', async () => {
    // The P2P client returns { ok, status, json } with no text(); calling
    // resp.text() here masks the real error behind a TypeError.
    const w = await mkWallet(); const node = new MockNode();
    node.getBlockTemplate = async () => ({ ok: false, status: 400, json: async () => ({ error: 'Invalid coinbase' }) });
    let msg = null;
    try { await new Miner(w, node).fetchTemplate(); } catch (e) { msg = e.message; }
    assert(msg && /Invalid coinbase/.test(msg), `got "${msg}" instead of the node's message`);
    assert(!/text is not a function/.test(msg || ''), 'error path called resp.text()');
});

await t('endlessly moving fees fail loudly rather than looping', async () => {
    const w = await mkWallet(); const node = new MockNode();
    let n = 0;
    node.getBlockTemplate = async () => ({ ok: false, status: 409, json: async () => ({ expected_total: 1000 + (++n) }) });
    let msg = null;
    try { await new Miner(w, node).fetchTemplate(3); } catch (e) { msg = e.message; }
    assert(msg && /fees kept moving/.test(msg), `got "${msg}"`);
});

console.log('\n━━━ template lifecycle ━━━');

await t('fetchTemplate does not advance nextWotsIndex', async () => {
    // Reserving is not consuming: an unaccepted block must not burn key material.
    const w = await mkWallet(); const node = new MockNode();
    const before = w.nextWotsIndex;
    await new Miner(w, node).fetchTemplate();
    assert(w.nextWotsIndex === before, `nextWotsIndex moved ${before} → ${w.nextWotsIndex} for an unmined template`);
});

await t('a template is stale once the tip moves', async () => {
    const w = await mkWallet(); const node = new MockNode({ height: 100 });
    const m = new Miner(w, node);
    m.template = await m.fetchTemplate();
    m.templateAt = Date.now();
    assert(m._templateStale(100) === false, 'fresh template reported stale');
    assert(m._templateStale(101) === true, 'template survived a tip change — it would build on a stale parent');
});

await t('a template is stale once it ages out', async () => {
    // validate_timestamp requires the block to beat the median of the last 11;
    // the locked timestamp cannot be bumped without invalidating the PoW.
    const w = await mkWallet(); const node = new MockNode();
    const m = new Miner(w, node);
    m.template = await m.fetchTemplate();
    m.templateAt = Date.now() - (TEMPLATE_MAX_AGE_MS + 1);
    assert(m._templateStale(100) === true, 'an aged template was kept; its timestamp would fail the median check');
});

console.log('\n━━━ submission ━━━');

await t('grinds and submits against mining_midstate, never post_tx_midstate', async () => {
    const w = await mkWallet();
    // post_tx_midstate differs from mining_midstate; grinding the wrong one is
    // the bug the node says "silently rejected every block the web wallet ever
    // mined over WebRTC".
    const node = new MockNode({ midstate: MIDSTATE });
    const m = new Miner(w, node);
    let searchedMidstate = null;
    m.searchFn = (ms) => { searchedMidstate = ms; return undefined; };
    m.template = await m.fetchTemplate();
    m._search(m.template, 0n);
    assert(searchedMidstate === MIDSTATE, `searched ${searchedMidstate}, expected the template's mining_midstate`);
});

await t('the submitted batch keeps the template timestamp verbatim', async () => {
    const w = await mkWallet(); const node = new MockNode();
    const m = new Miner(w, node);
    const tmpl = await m.fetchTemplate();
    await m.submitBlock(tmpl, 0n);
    assert(node.submitted.length === 1, 'nothing submitted');
    assert(node.submitted[0].timestamp === node.lockedTimestamp,
        `timestamp was rewritten (${node.submitted[0].timestamp} vs ${node.lockedTimestamp}) — that invalidates the PoW`);
});

await t('the submitted extension is recomputed, not trusted', async () => {
    const w = await mkWallet(); const node = new MockNode();
    const m = new Miner(w, node);
    const tmpl = await m.fetchTemplate();
    await m.submitBlock(tmpl, 7n);
    const ext = node.submitted[0].extension;
    const ref = JSON.parse(w.inner.build_solo_extension(MIDSTATE, 7n));
    assert(ext.nonce === 7, `nonce ${ext.nonce}`);
    assert(JSON.stringify(ext.final_hash) === JSON.stringify(ref.final_hash),
        'submitted extension does not match a local recomputation');
});

await t('acceptance commits the coinbase addresses and index', async () => {
    const w = await mkWallet(); const node = new MockNode();
    const m = new Miner(w, node);
    const tmpl = await m.fetchTemplate();
    const before = w.nextWotsIndex;
    const res = await m.submitBlock(tmpl, 1n);
    assert(res.accepted, 'submission not accepted');
    for (const a of tmpl.mining_addrs) {
        assert(w.wotsAddrs[a.address] === a.index, 'coinbase address not registered on acceptance');
    }
    assert(w.nextWotsIndex >= tmpl.next_wots_index, `nextWotsIndex ${w.nextWotsIndex} < ${tmpl.next_wots_index}`);
    assert(w.nextWotsIndex > before, 'index never advanced despite an accepted block');
});

await t('rejection burns no key material', async () => {
    const w = await mkWallet(); const node = new MockNode();
    node.acceptSubmissions = false;
    const m = new Miner(w, node);
    const tmpl = await m.fetchTemplate();
    const before = w.nextWotsIndex;
    const res = await m.submitBlock(tmpl, 1n);
    assert(!res.accepted, 'expected rejection');
    assert(res.reason === 'stale parent', `reason "${res.reason}"`);
    assert(w.nextWotsIndex === before, 'a rejected block advanced nextWotsIndex');
    assert(Object.keys(w.wotsAddrs).length === 0, 'a rejected block registered coinbase addresses');
});

await t('coinbase addresses become watchable so the reward is credited', async () => {
    const w = await mkWallet(); const node = new MockNode();
    const m = new Miner(w, node);
    const tmpl = await m.fetchTemplate();
    await m.submitBlock(tmpl, 1n);
    const addr = tmpl.mining_addrs[0].address;
    const cid = W.compute_coin_id_hex(addr, BigInt(node.reward), tmpl.batch_template.coinbase[0].salt);
    assert(w.addUtxo(addr, node.reward, tmpl.batch_template.coinbase[0].salt, cid),
        'the block reward could not be credited — the coinbase address is not in the wallet');
});

console.log('\n━━━ mining loop ━━━');

await t('the loop refreshes the template when the tip moves', async () => {
    const w = await mkWallet(); const node = new MockNode({ height: 100 });
    const m = new Miner(w, node, { tipPollMs: 0, throttleMs: 0 });
    m.searchFn = () => undefined;
    const heights = [];
    m.on('template', (t) => { heights.push(t.height); if (heights.length === 1) node.height = 101; if (heights.length >= 2) m.stop(); });
    await m.start();
    assert(heights.length >= 2, `only ${heights.length} template(s) fetched across a tip change`);
    assert(heights[1] === 101, `refreshed at height ${heights[1]}, expected 101`);
});

await t('a found nonce is submitted and reported', async () => {
    const w = await mkWallet(); const node = new MockNode();
    const m = new Miner(w, node, { tipPollMs: 999_999, throttleMs: 0 });
    let calls = 0;
    m.searchFn = () => (++calls === 1 ? 42n : undefined);
    let accepted = null;
    m.on('accepted', (r) => { accepted = r; m.stop(); });
    await m.start();
    assert(accepted !== null, 'no acceptance reported');
    assert(accepted.nonce === '42', `nonce ${accepted.nonce}`);
    assert(node.submitted[0].extension.nonce === 42, 'wrong nonce submitted');
});

await t('a rejected block does not stop mining', async () => {
    const w = await mkWallet(); const node = new MockNode();
    node.acceptSubmissions = false;
    const m = new Miner(w, node, { tipPollMs: 999_999, throttleMs: 0 });
    let calls = 0;
    m.searchFn = () => (calls++ < 2 ? 1n : undefined);
    let rejects = 0;
    m.on('rejected', () => { if (++rejects >= 2) m.stop(); });
    await m.start();
    assert(rejects >= 2, `stopped after ${rejects} rejection(s)`);
});

await t('a template error is reported and retried, not fatal', async () => {
    const w = await mkWallet(); const node = new MockNode();
    let failures = 0;
    const realGet = node.getBlockTemplate.bind(node);
    node.getBlockTemplate = async (cb) => {
        if (failures++ < 1) throw new Error('node offline');
        return realGet(cb);
    };
    const m = new Miner(w, node, { tipPollMs: 999_999, throttleMs: 0, templateMaxAgeMs: 999_999 });
    m.searchFn = () => undefined;
    let errored = false;
    m.on('error', () => { errored = true; });
    m.on('template', () => m.stop());
    // Collapse the 5s backoff.
    const realTimeout = globalThis.setTimeout;
    globalThis.setTimeout = (fn) => realTimeout(fn, 0);
    try { await m.start(); } finally { globalThis.setTimeout = realTimeout; }
    assert(errored, 'error not reported');
    assert(m.stats.templates === 1, 'never recovered to fetch a template');
});

await t('stats track nonces and rate', async () => {
    const w = await mkWallet(); const node = new MockNode();
    const m = new Miner(w, node, { tipPollMs: 999_999, throttleMs: 0, iterations: 2 });
    let n = 0;
    m.searchFn = () => { if (++n >= 5) m.stop(); return undefined; };
    const stats = await m.start();
    assert(stats.nonces === 5 * 2 * 4, `counted ${stats.nonces}, expected ${5 * 2 * 4} (iterations x 4 lanes)`);
    assert(stats.noncesPerSec > 0, 'no rate computed');
});

console.log('\n━━━ real PoW round trip (WASM VDF) ━━━');

await t('a nonce found by search_nonces reproduces via build_solo_extension', async () => {
    // The end-to-end invariant the node checks on receipt:
    //   verify_extension: final_hash < target ∧ create_extension(midstate, nonce) = final_hash
    // Uses the real VDF, so it is slow (~130ms per nonce) but proves the search
    // and the submission path agree.
    const w = await mkWallet();
    const found = W.search_nonces(MIDSTATE, EASY, 0n, 1);
    assert(found !== undefined, 'search found nothing against a maximal target');

    const ext = JSON.parse(w.inner.build_solo_extension(MIDSTATE, found));
    assert(ext.nonce === Number(found), `extension nonce ${ext.nonce} != searched ${found}`);

    const hashHex = ext.final_hash.map((b) => b.toString(16).padStart(2, '0')).join('');
    assert(hashHex < EASY, `final_hash ${hashHex.slice(0, 16)}… not below target`);

    // And the miner's submission path produces exactly this.
    const node = new MockNode();
    const m = new Miner(w, node);
    const tmpl = await m.fetchTemplate();
    await m.submitBlock(tmpl, found);
    assert(JSON.stringify(node.submitted[0].extension.final_hash) === JSON.stringify(ext.final_hash),
        'the submitted extension diverges from the verified one');
});

await t('an unreachable target yields nothing (comparison is not inverted)', async () => {
    const found = W.search_nonces(MIDSTATE, IMPOSSIBLE, 0n, 1);
    assert(found === undefined, `found nonce ${found} against a near-zero target — comparison is inverted`);
});

await t('the mining loop yields to the event loop every iteration', async () => {
    // search_nonces blocks synchronously, so without an await in the loop the
    // event loop never runs: timers, inbound pushes and stop() are all starved.
    // Observed live — a 1-second run continued for 15.75 s, ending only when the
    // 15 s tip poll happened to await.
    //
    // Tested by whether a 0 ms timer can fire at all, with a hard call budget so
    // a regression fails cleanly instead of hanging the suite. (A race against a
    // deadline cannot work here: under total starvation the deadline timer is
    // starved too.)
    const w = await mkWallet(); const node = new MockNode();
    const m = new Miner(w, node, { tipPollMs: 999_999, throttleMs: 0 });

    let timerFired = false;
    setTimeout(() => { timerFired = true; }, 0);

    let calls = 0;
    m.searchFn = () => {
        calls++;
        if (timerFired || calls >= 50) m.stop();
        return undefined;
    };
    await m.start();

    assert(timerFired, `the loop ran ${calls} iterations without yielding — a 0 ms timer never fired`);
    assert(calls < 50, `took ${calls} iterations before the timer could run`);
});

await t('a throttled miner also yields', async () => {
    const w = await mkWallet(); const node = new MockNode();
    const m = new Miner(w, node, { tipPollMs: 999_999, throttleMs: 5 });
    let timerFired = false;
    setTimeout(() => { timerFired = true; }, 0);
    let calls = 0;
    m.searchFn = () => { calls++; if (timerFired || calls >= 50) m.stop(); return undefined; };
    await m.start();
    assert(timerFired, `throttled loop ran ${calls} iterations without yielding`);
});

console.log('\n━━━ pool mining ━━━');

class MockPool {
    constructor({ shareTarget = EASY, jobId = 'job-1', height = 100 } = {}) {
        this.shareTarget = shareTarget; this.jobId = jobId; this.height = height;
        this.templateCalls = 0; this.shares = []; this.rejectAs = null;
    }
    async getTemplate(address) {
        this.templateCalls++;
        this.lastAddress = address;
        return {
            job_id: this.jobId, mining_midstate: MIDSTATE,
            target: this.shareTarget, network_target: '00' + 'ff'.repeat(31),
            height: this.height, batch_template: { transactions: [{}, {}] },
            merkle_root: 'aa'.repeat(32),
        };
    }
    async submitShare(address, jobId, nonce, worker) {
        if (this.rejectAs) throw new Error(this.rejectAs);
        this.shares.push({ address, jobId, nonce, worker });
        return { accepted: true };
    }
}

await t('pool mode grinds the SHARE target, not the network target', async () => {
    // The network target would yield one share per several million years.
    const w = await mkWallet(); const node = new MockNode();
    const pool = new MockPool({ shareTarget: EASY });
    const m = new Miner(w, node, { pool, poolAddress: 'ab'.repeat(32) });
    const tmpl = await m.fetchTemplate();
    assert(tmpl.target === EASY, `grinding ${tmpl.target}, expected the share target`);
    assert(tmpl.network_target !== tmpl.target, 'network target not distinguished');
    assert(tmpl.isPoolJob === true, 'job not flagged as a pool job');
});

await t('pool mode skips coinbase negotiation entirely', async () => {
    const w = await mkWallet(); const node = new MockNode({ fees: 5000 });
    const pool = new MockPool();
    const m = new Miner(w, node, { pool, poolAddress: 'ab'.repeat(32) });
    await m.fetchTemplate();
    assert(node.templateRequests.length === 0, 'asked the node for a template in pool mode');
    assert(pool.templateCalls === 1, 'did not ask the pool');
});

await t('a share is submitted as (address, job_id, nonce, worker)', async () => {
    const w = await mkWallet(); const node = new MockNode();
    const pool = new MockPool();
    const addr = 'ab'.repeat(32);
    const m = new Miner(w, node, { pool, poolAddress: addr, workerName: 'rig-1' });
    const tmpl = await m.fetchTemplate();
    const res = await m.submitBlock(tmpl, 12345n);
    assert(res.accepted && res.isShare, 'share not reported as accepted');
    assert(pool.shares.length === 1, 'no share submitted');
    const sh = pool.shares[0];
    assert(sh.address === addr && sh.jobId === 'job-1' && sh.worker === 'rig-1', 'share fields wrong');
    assert(sh.nonce === 12345n, 'nonce not passed through');
});

await t('the nonce reaches the pool as a decimal string (u64 safe)', async () => {
    const w = await mkWallet(); const node = new MockNode();
    let seen = null;
    const pool = new MockPool();
    pool.submitShare = async (a, j, n) => { seen = String(n); return { accepted: true }; };
    const m = new Miner(w, node, { pool, poolAddress: 'ab'.repeat(32) });
    const tmpl = await m.fetchTemplate();
    const big = (1n << 60n) + 12345n;   // beyond 2^53
    await m.submitBlock(tmpl, big);
    assert(seen === big.toString(), `nonce arrived as ${seen}, expected ${big} — JSON number would round it`);
});

await t('a share does NOT end the job (continue_on_found)', async () => {
    // Halting is correct solo behaviour but fatal here: the midstate does not
    // change on a share, so the restart-on-new-midstate path never fires and the
    // miner idles until the next network block.
    const w = await mkWallet(); const node = new MockNode();
    const pool = new MockPool();
    const m = new Miner(w, node, { pool, poolAddress: 'ab'.repeat(32), tipPollMs: 999_999, throttleMs: 0 });
    let calls = 0;
    m.searchFn = () => (calls < 3 ? 1n : undefined);
    let shares = 0;
    m.on('share', () => { calls++; if (++shares >= 3) m.stop(); });
    await m.start();
    assert(shares === 3, `stopped after ${shares} share(s) — the job was dropped on a hit`);
    assert(pool.templateCalls === 1, `refetched the job ${pool.templateCalls} times for one template`);
});

await t('a solo hit DOES end the job', async () => {
    const w = await mkWallet(); const node = new MockNode();
    const m = new Miner(w, node, { tipPollMs: 999_999, throttleMs: 0 });
    let calls = 0;
    m.searchFn = () => (++calls === 1 ? 5n : undefined);
    m.on('accepted', () => m.stop());
    await m.start();
    assert(m.template === null, 'solo template survived a block find');
});

await t('a stale-share rejection marks the job dirty', async () => {
    const w = await mkWallet(); const node = new MockNode();
    const pool = new MockPool();
    pool.rejectAs = 'stale share: unknown job_id';
    const m = new Miner(w, node, { pool, poolAddress: 'ab'.repeat(32) });
    const tmpl = await m.fetchTemplate();
    const res = await m.submitBlock(tmpl, 1n);
    assert(!res.accepted, 'stale share reported as accepted');
    assert(m._poolJobDirty === true, 'job not marked dirty after a stale rejection');
    m.template = tmpl; m.templateAt = Date.now();
    assert(m._templateStale(100) === true, 'dirty job not treated as stale');
});

await t('pool mining without an address fails loudly', async () => {
    const w = await mkWallet(); const node = new MockNode();
    let msg = null;
    try { await new Miner(w, node, { pool: new MockPool() }).fetchTemplate(); }
    catch (e) { msg = e.message; }
    assert(msg && /poolAddress/.test(msg), `got "${msg}"`);
});

await t('share stats are tracked separately from blocks', async () => {
    const w = await mkWallet(); const node = new MockNode();
    const pool = new MockPool();
    const m = new Miner(w, node, { pool, poolAddress: 'ab'.repeat(32) });
    const tmpl = await m.fetchTemplate();
    await m.submitBlock(tmpl, 1n);
    await m.submitBlock(tmpl, 2n);
    const st = m.getStats();
    assert(st.shares === 2 && st.sharesAccepted === 2, `shares ${st.shares}/${st.sharesAccepted}`);
    assert(st.accepted === 0, 'shares counted as blocks');
});

console.log('\n━━━ multi-worker search ━━━');

await t('the nonce partition is disjoint and gap-free', async () => {
    // Two constraints at once: SIMD needs each worker's four lanes contiguous,
    // and no two workers may test the same nonce or the cores buy nothing.
    const iterations = 1, perCall = iterations * 4;
    for (const count of [1, 2, 4, 8]) {
        const base = 1000n; const seen = new Set(); let collisions = 0;
        for (let k = 0; k < count; k++) {
            let n = base + BigInt(perCall * k);
            const stride = BigInt(perCall * count);
            for (let step = 0; step < 40; step++) {
                for (let lane = 0n; lane < BigInt(perCall); lane++) {
                    const v = n + lane;
                    if (seen.has(v)) collisions++;
                    seen.add(v);
                }
                n += stride;
            }
        }
        assert(collisions === 0, `workers=${count}: ${collisions} overlapping nonces`);
        const vals = [...seen].sort((a, b) => Number(a - b));
        assert(vals.every((v, i) => i === 0 || v === vals[i - 1] + 1n),
            `workers=${count}: partition leaves gaps`);
        assert(seen.size === count * 40 * perCall, `workers=${count}: wrong coverage`);
    }
});

await t('workers spawn, search, and report progress', async () => {
    const { MinerPool } = await import('./src/miner-pool.js');
    const pool = new MinerPool({ workers: 2 });
    await pool.start();
    try {
        pool.searchTemplate(MIDSTATE, EASY);
        await new Promise((r) => setTimeout(r, 2500));
        assert(pool.noncesTested > 0, 'no nonces reported across workers');
        assert(pool._found.length > 0, 'a maximal target produced no hits');
    } finally { await pool.stop(); }
});

await t('the pool exposes a Miner-compatible searchFn', async () => {
    const { MinerPool } = await import('./src/miner-pool.js');
    const pool = new MinerPool({ workers: 2 });
    await pool.start();
    try {
        const fn = pool.searchFn;
        assert(fn(MIDSTATE, EASY) === undefined, 'first call should start the search, not return');
        await new Promise((r) => setTimeout(r, 2500));
        const got = fn(MIDSTATE, EASY);
        assert(typeof got === 'bigint', `drained ${got}, expected a bigint nonce`);
    } finally { await pool.stop(); }
});

await t('changing the template restarts the workers', async () => {
    const { MinerPool } = await import('./src/miner-pool.js');
    const pool = new MinerPool({ workers: 1 });
    await pool.start();
    try {
        const fn = pool.searchFn;
        fn(MIDSTATE, EASY);
        const first = pool._current.base;
        fn('b6'.repeat(32), EASY);
        assert(pool._current.midstate === 'b6'.repeat(32), 'workers kept the old midstate');
        assert(pool._current.base !== first, 'base not re-randomised for the new template');
    } finally { await pool.stop(); }
});

console.log('\n━━━ pool precommitment audit ━━━');

const { verifyPoolProof, PoolClient } = await import('./src/pool.js');

/** Reference tree, built exactly as ShareMerkleTree::build does in the node. */
function buildTree(shares) {
    const sorted = [...shares].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    const leafHash = ([addr, score]) => {
        const le = new Uint8Array(8);
        new DataView(le.buffer).setBigUint64(0, BigInt(score), true);
        return W.blake3_hash_hex(addr + Array.from(le).map((b) => b.toString(16).padStart(2, '0')).join(''));
    };
    let layer = sorted.map(leafHash);
    const layers = [layer.slice()];
    while (layer.length > 1) {
        const next = [];
        for (let i = 0; i < layer.length; i += 2) {
            // A lone node is paired with ITSELF, not promoted.
            next.push(i + 1 < layer.length ? W.blake3_hash_hex(layer[i] + layer[i + 1])
                                           : W.blake3_hash_hex(layer[i] + layer[i]));
        }
        layers.push(next.slice());
        layer = next;
    }
    return { root: layer[0], layers, sorted };
}

function proofFor(tree, index) {
    const proof = [];
    let idx = index;
    for (const layer of tree.layers.slice(0, -1)) {
        const isRight = idx % 2 === 1;
        // generate_proof CLAMPS to the last element, so a lone node is its own sibling.
        const sib = isRight ? idx - 1 : Math.min(idx + 1, layer.length - 1);
        proof.push(layer[sib]);
        idx = Math.floor(idx / 2);
    }
    return proof;
}

const mkAddr = (i) => i.toString(16).padStart(2, '0').repeat(32);

await t('a valid proof verifies at every leaf, for many tree sizes', async () => {
    for (const n of [1, 2, 3, 4, 5, 7, 8, 9, 16]) {
        const shares = Array.from({ length: n }, (_, i) => [mkAddr(i + 1), (i + 1) * 10]);
        const tree = buildTree(shares);
        for (let idx = 0; idx < n; idx++) {
            const [addr, score] = tree.sorted[idx];
            const ok = verifyPoolProof(addr, score, idx, proofFor(tree, idx), tree.root);
            assert(ok, `n=${n} idx=${idx}: a valid proof was rejected`);
        }
    }
});

await t('the odd/lone-node case verifies (self-pairing, not promotion)', async () => {
    // generate_proof clamps the sibling index, so the right-most node of an odd
    // layer lists itself. A conventional "promote unchanged" rule fails exactly
    // the miners at the end of the tree.
    const shares = Array.from({ length: 5 }, (_, i) => [mkAddr(i + 1), 1]);
    const tree = buildTree(shares);
    const last = 4;
    const [addr, score] = tree.sorted[last];
    assert(verifyPoolProof(addr, score, last, proofFor(tree, last), tree.root),
        'the lone right-most leaf failed to verify');
});

await t('a tampered score is rejected', async () => {
    const shares = Array.from({ length: 8 }, (_, i) => [mkAddr(i + 1), (i + 1) * 10]);
    const tree = buildTree(shares);
    const [addr, score] = tree.sorted[3];
    assert(!verifyPoolProof(addr, Number(score) + 1, 3, proofFor(tree, 3), tree.root),
        'an inflated score still verified — a pool could under-credit undetected');
});

await t('a tampered sibling is rejected', async () => {
    const shares = Array.from({ length: 8 }, (_, i) => [mkAddr(i + 1), 5]);
    const tree = buildTree(shares);
    const proof = proofFor(tree, 2);
    proof[0] = 'ff'.repeat(32);
    assert(!verifyPoolProof(tree.sorted[2][0], 5, 2, proof, tree.root), 'a forged sibling verified');
});

await t('the wrong leaf index is rejected', async () => {
    const shares = Array.from({ length: 8 }, (_, i) => [mkAddr(i + 1), 5]);
    const tree = buildTree(shares);
    assert(!verifyPoolProof(tree.sorted[2][0], 5, 5, proofFor(tree, 2), tree.root),
        'a mismatched index verified');
});

await t('a proof against a different root is rejected', async () => {
    const shares = Array.from({ length: 4 }, (_, i) => [mkAddr(i + 1), 5]);
    const tree = buildTree(shares);
    assert(!verifyPoolProof(tree.sorted[0][0], 5, 0, proofFor(tree, 0), 'ab'.repeat(32)),
        'verified against an unrelated root');
});

await t('malformed input is rejected, not thrown on', async () => {
    assert(verifyPoolProof('short', 1, 0, [], 'ab'.repeat(32)) === false, 'short address');
    assert(verifyPoolProof(mkAddr(1), 1, 0, null, 'ab'.repeat(32)) === false, 'null proof');
    assert(verifyPoolProof(mkAddr(1), 1, 0, ['zz'], 'ab'.repeat(32)) === false, 'bad sibling hex');
});

await t('auditInclusion reports omission as an answer, not an error', async () => {
    const client = new PoolClient({
        dial: async () => 'p', close: async () => {},
        request: async () => new TextEncoder().encode(JSON.stringify({ ok: false, error: 'address not in tree' })),
    });
    await client.connect('x');
    const v = await client.auditInclusion(mkAddr(1), 'ab'.repeat(32));
    assert(v.included === false && v.verified === false, 'omission not reported cleanly');
    assert(/not in tree/.test(v.reason), `reason "${v.reason}"`);
});

await t('auditInclusion verifies a genuine proof', async () => {
    const shares = Array.from({ length: 6 }, (_, i) => [mkAddr(i + 1), (i + 1) * 3]);
    const tree = buildTree(shares);
    const [addr, score] = tree.sorted[2];
    const client = new PoolClient({
        dial: async () => 'p', close: async () => {},
        request: async () => new TextEncoder().encode(JSON.stringify({
            ok: true, data: { index: 2, score, proof: proofFor(tree, 2) },
        })),
    });
    await client.connect('x');
    const v = await client.auditInclusion(addr, tree.root);
    assert(v.included && v.verified, `audit failed: ${v.reason}`);
    assert(v.score === Number(score), `score ${v.score}`);
});

await t('auditInclusion flags a proof that does not reach the root', async () => {
    const shares = Array.from({ length: 6 }, (_, i) => [mkAddr(i + 1), 4]);
    const tree = buildTree(shares);
    const client = new PoolClient({
        dial: async () => 'p', close: async () => {},
        request: async () => new TextEncoder().encode(JSON.stringify({
            ok: true, data: { index: 2, score: 999, proof: proofFor(tree, 2) },
        })),
    });
    await client.connect('x');
    const v = await client.auditInclusion(tree.sorted[2][0], tree.root);
    assert(v.included === true && v.verified === false, 'a bogus score passed the audit');
    assert(/merkle_root/.test(v.reason), `reason "${v.reason}"`);
});

console.log(`\n━━━ ${pass} passed, ${fail} failed ━━━\n`);
process.exit(fail ? 1 : 0);
