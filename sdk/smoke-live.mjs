#!/usr/bin/env node
// smoke-live.mjs — Validate the SDK against a REAL node.
//
// Everything in `npm test` is offline: vectors derived from the Rust source and
// mocks built from reading it. That catches drift in what the code believes, but
// it cannot catch a belief that was wrong in the same way in both places. This
// script is the part only a real node can answer.
//
// It is deliberately READ-ONLY by default. Nothing here spends, and nothing
// touches key material, unless a flag explicitly asks for it.
//
//   node smoke-live.mjs --peer /ip4/1.2.3.4/tcp/9333/p2p/12D3Koo...
//   node smoke-live.mjs --rpc http://127.0.0.1:8080
//
// Optional, in increasing order of consequence:
//   --scan 5000     Scan the last N blocks and report what matched.
//
//                   Pair it with a real seed or it proves nothing: without one
//                   the wallet is freshly generated and empty, so no block is
//                   ever fetched or parsed. Supply the phrase by environment
//                   variable, NOT on the command line, where it would land in
//                   shell history and be visible to `ps`:
//
//                     MIDSTATE_PHRASE="word word ..." npm run smoke -- --peer … --scan 20000
//
//                   --phrase-file <path> is also accepted.
//   --addresses     Print the addresses derived from the seed and exit. Compare
//                   the first one against your wallet UI: if they differ, the
//                   SDK derives a different key tree and no scan can ever match,
//                   which is a far bigger problem than a scan window.
//   --mss N         Also watch N reusable (MSS) addresses when scanning. A scan
//                   watches only WOTS addresses by default, so a wallet that
//                   received to a reusable address matches nothing.
//   --contract <64-hex>   Follow a contract and report its tracked state thread.
//   --workers N     Mine across N worker threads. Defaults to 1 (single core).
//                   The VDF is ~130 ms/nonce and embarrassingly parallel, so
//                   expect roughly N x the single-core rate.
//   --mine 300      Solo-mine for N seconds against a REAL template. The pass
//                   condition is that no found block is REJECTED, so give it
//                   minutes — at ~8 nonces/s a short run proves nothing, and
//                   anything under a block interval never exercises tip-refresh.
//   --submit-probe  Submit ONE deliberately-invalid block and check the node
//                   rejects it for the PoW and nothing else. This is the only
//                   way to exercise the submission path without finding a real
//                   block, which at ~18 n/s will not happen. Safe: SubmitBatch
//                   rejection returns an error string and carries no ban or
//                   violation counter.
//   --pool /ip4/…/p2p/… --pool-payout <64-hex>   Connect to a pool, grind shares.
//
// Exit code is non-zero if any check fails, so this is usable in CI against a
// testnet node.

import fs from 'fs/promises';
import initWasm from './pkg/wasm_wallet.js';
import { MidstateClient } from './src/client.js';
import { Wallet } from './src/wallet.js';
import { MemoryStorage } from './src/storage.js';
import { Miner } from './src/miner.js';

const argv = process.argv.slice(2);
const arg = (name, dflt = null) => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 && argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : (argv.includes(`--${name}`) ? true : dflt);
};

/**
 * A numeric flag, validated.
 *
 * Passing the literal placeholder from the usage line (`--mine SECS`) used to
 * yield NaN, which `setTimeout` clamps to 1 ms — so the miner "ran" for a
 * millisecond and the check reported success having proved nothing. A test that
 * passes without testing is worse than one that fails, so this refuses rather
 * than coercing.
 */
const numArg = (name, dflt = null) => {
    const raw = arg(name, null);
    if (raw === null) return dflt;
    if (raw === true) {
        console.error(`\n--${name} needs a number, e.g. --${name} 300\n`);
        process.exit(2);
    }
    const n = Number(raw);
    if (!Number.isFinite(n) || n <= 0) {
        console.error(`\n--${name} must be a positive number (got "${raw}").` +
                      ` If you copied the usage line, replace the placeholder with a value.\n`);
        process.exit(2);
    }
    return n;
};

let pass = 0, fail = 0, skip = 0;
const ok   = (m, d) => { console.log(`  ✅ ${m}${d ? ` — ${d}` : ''}`); pass++; };
const bad  = (m, d) => { console.log(`  ❌ ${m}${d ? ` — ${d}` : ''}`); fail++; };
const note = (m)    => { console.log(`  ⊘  ${m} (skipped)`); skip++; };
async function check(name, fn) {
    try { const d = await fn(); ok(name, d); }
    catch (e) { bad(name, e.message); }
}

const peer = arg('peer');
const rpc = arg('rpc');
if (!peer && !rpc) {
    console.error('Usage: node smoke-live.mjs --peer <multiaddr> | --rpc <url>');
    console.error('  optional: --scan 5000  --mine 300  --workers 8  --submit-probe');
    console.error('            --phrase-file <path>  --contract <64-hex>');
    console.error('            --pool <multiaddr> --pool-payout <64-hex>');
    console.error('');
    console.error('Via npm, the -- separator is REQUIRED or npm keeps the flags itself:');
    console.error('  npm run smoke -- --peer /ip4/…/p2p/…  --scan 20000');
    console.error('              ^^ this');
    if (process.env.npm_lifecycle_event) {
        console.error('');
        console.error('(Ran through npm with no arguments — the missing -- is the usual cause.)');
    }
    if (process.env.MIDSTATE_PHRASE) {
        console.error('');
        console.error('MIDSTATE_PHRASE is set but no arguments arrived, so nothing ran.');
        console.error('Prefer --phrase-file <path> (chmod 600): an inline VAR=... command');
        console.error('is written to shell history and visible in `ps`.');
    }
    process.exit(2);
}

await initWasm({ module_or_path: await fs.readFile('./pkg/wasm_wallet_bg.wasm') });

console.log(`\nMidstate SDK live smoke test\n  target: ${peer || rpc}\n`);

// MidstateClient takes a positional target: a multiaddr (or array) for P2P,
// or an http(s) base URL. It infers the mode from the string.
const client = new MidstateClient(peer ? String(peer) : String(rpc));
if (client.connect) await client.connect();

// ── Chain state ────────────────────────────────────────────────────────────
console.log('━━━ chain state ━━━');

let state = null;
await check('get_state returns a usable tip', async () => {
    state = await client.getState();
    if (typeof state.height !== 'number') throw new Error(`height is ${typeof state.height}`);
    if (!state.header_hash) throw new Error('no header_hash');
    return `height ${state.height}, reward ${state.block_reward}`;
});

await check('get_filters honours the requested span', async () => {
    if (!state) throw new Error('no state');
    const start = Math.max(0, state.height - 10);
    const fd = await client.getFilters(start, start + 10);
    const n = fd.filters?.length ?? 0;
    // The node emits placeholders rather than truncating, so all four arrays
    // should span the request. A short response here means the SDK's
    // no-amplification assumption does not hold against this node.
    if (fd.element_counts?.length !== n || fd.block_hashes?.length !== n) {
        throw new Error(`array lengths disagree: ${n}/${fd.element_counts?.length}/${fd.block_hashes?.length}`);
    }
    return `${n} entries for a 10-block span`;
});

await check('block hashes match between get_filters and get_block', async () => {
    // This is the identity the whole reorg module keys on. If these ever
    // disagree, fork detection is comparing incomparable things.
    if (!state || state.height < 2) throw new Error('chain too short');
    const h = state.height - 2;
    const fd = await client.getFilters(h, h + 1);
    const fromFilters = (fd.block_hashes?.[0] || '').toLowerCase();
    if (!fromFilters) return 'height undecidable (no hash served) — inconclusive';
    const blk = await client.getBlock(h);
    const ext = blk?.extension?.final_hash;
    const fromBlock = Array.isArray(ext)
        ? ext.map((b) => b.toString(16).padStart(2, '0')).join('')
        : String(ext || '').toLowerCase();
    if (fromFilters !== fromBlock) {
        throw new Error(`filters say ${fromFilters.slice(0, 16)}…, block says ${fromBlock.slice(0, 16)}…`);
    }
    return `agree at height ${h}`;
});

await check('a zero element_count is never paired with a non-empty filter', async () => {
    // The undecidable contract: zero means "could not build a filter", so a
    // zero count alongside real filter bytes would invalidate the sync fix.
    if (!state) throw new Error('no state');
    const start = Math.max(0, state.height - 50);
    const fd = await client.getFilters(start, start + 50);
    let violations = 0;
    for (let i = 0; i < (fd.filters?.length ?? 0); i++) {
        if (fd.element_counts[i] === 0 && fd.filters[i] && fd.filters[i].length > 0) violations++;
    }
    if (violations) throw new Error(`${violations} block(s) had count 0 with a non-empty filter`);
    return 'contract holds across 50 blocks';
});

// ── Consensus activation gates ─────────────────────────────────────────────
console.log('\n━━━ activation gates ━━━');

await check('state threads are active at this height', () => {
    if (!state) throw new Error('no state');
    if (state.height < 65_000) throw new Error(`height ${state.height} < 65,000 — OP_READ_INPUT_STATE is InvalidOpcode`);
    return `height ${state.height}`;
});

await check('the pump curve avoids consensus-gated opcodes', async () => {
    // COVENANT_SUM_ACTIVATION_HEIGHT = 300,000. The curve derives its reserve
    // from supply precisely so it needs no sum_input_value() and can deploy
    // below that height; this confirms the shipped contract still does.
    const { compile } = await import('./src/compiler.js');
    const { pumpContractSource, MAX_DEPTH } = await import('./src/pump.js');
    const asm = compile(pumpContractSource(MAX_DEPTH)).asm.join(' ');
    if (/SUM_INPUT_VALUE/.test(asm) && state.height < 300_000) {
        throw new Error(`curve uses SUM_INPUT_VALUE but height is ${state.height} < 300,000`);
    }
    return /SUM_INPUT_VALUE/.test(asm)
        ? `uses SUM_INPUT_VALUE; active at height ${state.height}`
        : `no gated opcodes — deployable at height ${state.height}`;
});

// ── Mining template ────────────────────────────────────────────────────────
console.log('\n━━━ mining template ━━━');

// A recovery phrase, if one was supplied. Read from the environment or a file
// rather than argv: a seed on the command line is written to shell history and
// exposed to any process that can read /proc.
let phrase = process.env.MIDSTATE_PHRASE || null;
const phraseFile = arg('phrase-file');
if (phraseFile && phraseFile !== true) {
    phrase = (await fs.readFile(String(phraseFile), 'utf8')).trim();
}
if (argv.includes('--phrase')) {
    console.error('\nRefusing --phrase on the command line: it would be written to shell history.');
    console.error('Use MIDSTATE_PHRASE="..." or --phrase-file <path> instead.\n');
    process.exit(2);
}
const wordCount = phrase ? phrase.trim().split(/\s+/).length : 0;
if (phrase && wordCount !== 24) {
    console.error(`\nRecovery phrase has ${wordCount} words; expected 24.\n`);
    process.exit(2);
}

const wallet = await Wallet.create(new MemoryStorage(), phrase);
await wallet.getNewAddress();

// --addresses: derivation check, before anything touches the network.
//
// A scan that matches nothing has two very different causes: the coins are
// outside the window, or the SDK is deriving a different key tree than the
// wallet that holds them. Only the second is a bug, and comparing one address
// settles it in a second.
if (arg('addresses')) {
    const n = Number(arg('addresses')) || 10;
    console.log(`WOTS addresses derived from ${phrase ? 'the supplied seed' : 'a FRESH seed (pass a phrase!)'}:`);
    for (let i = 0; i < n; i++) console.log(`  ${String(i).padStart(3)}  ${wallet.inner.get_wots_address(i)}`);
    console.log('\nCompare index 0 against your wallet UI\'s first receive address.');
    console.log('If they differ, no scan can ever match and the derivation is the bug.');
    process.exit(0);
}
if (phrase) console.log('  using the supplied recovery phrase\n');
else console.log('  no phrase supplied — wallet is fresh and empty (scan will match nothing)\n');
let template = null;

await check('block_template negotiates a coinbase total', async () => {
    const miner = new Miner(wallet, client);
    template = await miner.fetchTemplate();
    const total = template.batch_template.coinbase.reduce((a, c) => a + Number(c.value), 0);
    const expected = Number(template.block_reward) + Number(template.total_fees);
    if (total !== expected) throw new Error(`coinbase ${total} != reward+fees ${expected}`);
    return `total ${total} (reward ${template.block_reward} + fees ${template.total_fees})`;
});

await check('mining_midstate is NOT post_tx_midstate', async () => {
    // The node's own note calls confusing these "the bug that silently rejected
    // every block the web wallet ever mined over WebRTC".
    if (!template) throw new Error('no template');
    const post = template.batch_template?.post_tx_midstate;
    if (post && post.toLowerCase() === template.mining_midstate.toLowerCase()) {
        throw new Error('template returns post_tx_midstate as mining_midstate');
    }
    return `midstate ${template.mining_midstate.slice(0, 16)}…`;
});

await check('the template timestamp is plausibly fresh', async () => {
    if (!template) throw new Error('no template');
    const ts = template.batch_template?.timestamp;
    if (typeof ts !== 'number') throw new Error('no timestamp in batch_template');
    const age = Math.floor(Date.now() / 1000) - ts;
    if (age > 300) throw new Error(`timestamp is ${age}s old; it will fail the median-time check`);
    return `${age}s old`;
});

await check('a locally computed extension reproduces', async () => {
    if (!template) throw new Error('no template');
    const ext = wallet.inner.build_solo_extension(template.mining_midstate, 1n);
    if (!ext) throw new Error('build_solo_extension returned nothing');
    const again = wallet.inner.build_solo_extension(template.mining_midstate, 1n);
    if (ext !== again) throw new Error('extension is not deterministic');
    return 'deterministic for the live midstate';
});

// ── Optional: chain scan ───────────────────────────────────────────────────
console.log('\n━━━ chain scan ━━━');

const scanN = numArg('scan');
if (!scanN) {
    note('scan (pass --scan N)');
} else {
    await check(`scan the last ${scanN} blocks`, async () => {
        const w = await Wallet.create(new MemoryStorage(), phrase);
        // Derive a gap of addresses so a real wallet's coins can actually match.
        // One address finds almost nothing even on a funded seed.
        // The gap is now ADAPTIVE: sync extends past the highest index it finds
        // a coin at and re-examines the range, so this is a starting window and
        // a per-step stride, not a ceiling.
        const gap = Number(arg('gap')) || 200;
        // MSS addresses are a separate derivation. A wallet that received to a
        // reusable address has no WOTS coin to find, so without this the scan
        // reports zero and looks like a filter bug.
        const mssN = Number(arg('mss')) || 0;
        if (mssN > 0) console.log(`     deriving ${mssN} MSS tree(s) — about 5s each…`);
        // Gap extension chains index to index, so a partial range can stop the
        // walk early and under-report. Scanning the whole chain is the only way
        // to trust the number; the SDK refuses a partial range otherwise.
        const from = Math.max(0, state.height - Number(scanN));
        const full = from === 0;
        console.log(`     gap ${gap} WOTS (adaptive) + ${mssN} MSS, from height ${from}` +
                    (full ? ' (FULL CHAIN)' : ' — partial: pass --scan ' + state.height + ' to trust the balance'));
        w.lastScannedHeight = from;
        const t0 = Date.now();
        const res = await w.sync(client, {
            gapLimit: gap, mssGap: mssN, allowPartialGapScan: !full,
            onProgress: (p) => { if (p.note) console.log(`     ${p.note}`); },
        });
        const secs = ((Date.now() - t0) / 1000).toFixed(1);
        if (res.height !== state.height && res.height < state.height) {
            throw new Error(`stopped at ${res.height} of ${state.height}`);
        }
        if (res.found === 0) {
            // The filter path ran, but no block was ever fetched or parsed. That
            // leaves the part most likely to be wrong — Consolidate handling,
            // predicate-derived coin ids, WOTS sibling purging — untouched.
            return phrase
                ? `${scanN} blocks in ${secs}s — 0 matches on the supplied seed. Check, in order: ` +
                  `(1) does --addresses index 0 match your wallet UI? (2) are the coins older than ` +
                  `${scanN} blocks — try --scan ${state.height}? (3) are they at reusable addresses — try --mss 5?`
                : `${scanN} blocks in ${secs}s — 0 matches, so BLOCK PARSING WAS NOT EXERCISED. ` +
                  `Set MIDSTATE_PHRASE to a funded seed to test it.`;
        }
        return `${scanN} blocks in ${secs}s, ${res.found} match(es), balance ${res.balance}` +
               ` (locked ${res.lockedBalance}) — derived to WOTS ${res.wotsDerived}, ` +
               `highest used ${res.highestUsedWots}, ${res.gapPasses} extension pass(es), ` +
               `${res.filterCacheHits} cached filter batch(es)` +
               (full ? '' : ' — PARTIAL RANGE, balance may be low');
    });

    await check('a re-sync is a no-op (idempotent)', async () => {
        const w = await Wallet.create(new MemoryStorage(), phrase);
        w.lastScannedHeight = Math.max(0, state.height - Number(scanN));
        await w.sync(client, { gapLimit: Number(arg('gap')) || 200, mssGap: Number(arg('mss')) || 0,
                               allowPartialGapScan: true });
        const before = w.utxos.length;
        const again = await w.sync(client);
        if (w.utxos.length !== before) throw new Error(`utxo count changed ${before} → ${w.utxos.length}`);
        if (again.reorg) throw new Error('spurious reorg reported on a clean re-sync');
        return 'no drift';
    });
}

// ── Optional: contract ─────────────────────────────────────────────────────
console.log('\n━━━ contract / state thread ━━━');

const contract = arg('contract');
if (!contract || contract === true) {
    note('contract (pass --contract <64-hex address>)');
} else {
    await check(`follow contract ${String(contract).slice(0, 10)}…`, async () => {
        const w = await Wallet.create(new MemoryStorage());
        w.watchContract(String(contract));
        w.lastScannedHeight = Math.max(0, state.height - Number(scanN || 2000));
        await w.sync(client);
        const th = w.getStateThread(String(contract));
        const coins = w.getContractCoins(String(contract));
        if (th && th.value !== 0n) throw new Error(`state thread has value ${th.value}; consensus requires 0`);
        return th
            ? `state ${th.state.slice(0, 16)}…, ${coins.length} coin(s), balance ${w.getContractBalance(String(contract))}`
            : `no state thread found in the scanned range (${coins.length} coin(s))`;
    });
}

// ── Optional: real mining ──────────────────────────────────────────────────
console.log('\n━━━ live mining ━━━');

const mineSecs = numArg('mine');
const poolAddr = arg('pool');

if (poolAddr && poolAddr !== true) {
    const payout = arg('pool-payout');
    if (!payout || payout === true) {
        bad('pool mining', 'pass --pool-payout <64-hex address>');
    } else {
        await check(`pool mine for ${mineSecs || 60}s`, async () => {
            const { PoolClient, libp2pPoolTransport } = await import('./src/pool.js');
            const pool = new PoolClient(libp2pPoolTransport(client.p2pClient));
            await pool.connect(String(poolAddr));
            const job = await pool.getTemplate(String(payout));
            if (!job?.mining_midstate) throw new Error('pool returned no job');

            // Audit inclusion BEFORE grinding: a miner absent from the
            // precommitment is working for nothing.
            if (job.merkle_root) {
                const audit = await pool.auditInclusion(String(payout), job.merkle_root);
                if (audit.included && !audit.verified) {
                    throw new Error(`pool proof does not reconstruct merkle_root: ${audit.reason}`);
                }
                console.log(`     inclusion: ${audit.included ? `verified, score ${audit.score}` : 'not yet recorded (normal on first connect)'}`);
            }

            const miner = new Miner(wallet, client, { pool, poolAddress: String(payout) });
            let shares = 0;
            miner.on('share', () => shares++);
            const stop = setTimeout(() => miner.stop(), Number(mineSecs || 60) * 1000);
            const stats = await miner.start();
            clearTimeout(stop);
            await pool.disconnect();
            return `job ${job.job_id}, ${stats.nonces} nonces, ${shares} share(s), ${stats.noncesPerSec.toFixed(1)} n/s`;
        });
    }
} else if (mineSecs) {
    const workers = numArg('workers', 1);
    await check(`solo mine for ${mineSecs}s across ${workers} worker(s)`, async () => {
        // One nonce is a 1,000,000-iteration sequential BLAKE3 chain, so a single
        // thread lands 15-30 n/s no matter what. Cores are the only lever.
        let pool = null;
        const opts = {};
        if (workers > 1) {
            const { MinerPool } = await import('./src/miner-pool.js');
            pool = new MinerPool({ workers });
            await pool.start();
            opts.searchFn = pool.searchFn;
        }
        const miner = new Miner(wallet, client, opts);
        let templates = 0, found = 0, rejected = null;
        miner.on('template', () => templates++);
        miner.on('found', () => found++);
        miner.on('rejected', (r) => { rejected = r.reason; });
        const stop = setTimeout(() => miner.stop(), Number(mineSecs) * 1000);
        const stats = await miner.start();
        clearTimeout(stop);
        // A worker pool reports its own totals; the coordinator only counts the
        // nonces it drained, which understates the real work by a large factor.
        const nonces = pool ? pool.noncesTested : stats.nonces;
        const rate = nonces / Math.max(1, stats.elapsedSecs);
        if (pool) await pool.stop();
        if (rejected) throw new Error(`a found block was REJECTED: ${rejected}`);
        if (templates === 0) throw new Error('never obtained a template');
        // The check that matters is "no rejection". A run too short to grind a
        // meaningful number of nonces cannot demonstrate that, so say so rather
        // than banking a green tick.
        if (nonces < 100) {
            return `${nonces} nonces in ${mineSecs}s — TOO SHORT TO PROVE ANYTHING. ` +
                   `Use --mine 300 or more; at ~16 n/s per core a real run needs minutes.`;
        }
        const note = templates > 1
            ? `${templates} templates (tip-refresh exercised)`
            : `${templates} template — run longer than a block interval to exercise tip-refresh`;
        const per = workers > 1 ? `, ${(rate / workers).toFixed(1)} n/s per worker` : '';
        return `${nonces} nonces, ${rate.toFixed(1)} n/s across ${workers} worker(s)${per}, ${note}, ${found} hit(s)`;
    });
} else {
    note('mining (pass --mine SECS, optionally --pool ADDR --pool-payout HEX)');
}

// ── Submission path ────────────────────────────────────────────────────────
console.log('\n━━━ block submission ━━━');

if (!arg('submit-probe')) {
    note('submit probe (pass --submit-probe)');
} else {
    await check('an invalid block is rejected for the PoW and nothing else', async () => {
        // A mining run that finds no block never exercises submitBlock: the
        // extension recompute, the verbatim timestamp copy, the batch assembly
        // and the serialization are all untested. Submitting one block whose
        // ONLY defect is the nonce tests all of them at once.
        //
        // Classifying the answer needs care. The node wraps validation in a 5s
        // timeout and returns "Block validation timed out" when its command loop
        // does not get to the submission in time — which happens when the node is
        // busy and says NOTHING about the batch. Its own comment calls validation
        // "comfortably sub-second", so a timeout is a liveness signal, not a
        // verdict. An earlier version of this check reported it as an SDK bug.
        //
        // Only "Block rejected: …" is a verdict, so only that is classified.
        const miner = new Miner(wallet, client);
        const attempts = [];

        for (let i = 0; i < 3; i++) {
            const tmpl = await miner.fetchTemplate();
            const res = await miner.submitBlock(tmpl, 1n);   // nonce 1 will not meet a real target
            if (res.accepted) {
                return 'the node ACCEPTED it — a block was found by accident. Check your wallet.';
            }
            const why = String(res.reason || '');
            attempts.push(why);

            if (/timed out|overloaded|dropped submission ack/i.test(why)) {
                await new Promise((r) => setTimeout(r, 3000));   // node busy; try again
                continue;
            }
            if (/difficulty|target|extension/i.test(why)) {
                return `rejected on PoW only ("${why.slice(0, 60)}") — batch shape, coinbase, ` +
                       `timestamp and serialization all accepted`;
            }
            throw new Error(
                `rejected for something other than the PoW: "${why}". The nonce was the only ` +
                `intentional defect, so this points at how the SDK assembles the batch.`
            );
        }

        // Never got a verdict. Inconclusive, and said so — the batch may be fine.
        return `INCONCLUSIVE after ${attempts.length} attempts: ${attempts.join(' | ')}. ` +
               `The node never returned a verdict, so the submission path is still untested. ` +
               `Retry when the node is quieter.`;
    });
}

// ── Chat codec ─────────────────────────────────────────────────────────────
console.log('\n━━━ chat ━━━');

await check('get_mempool responds', async () => {
    const mp = await client.getMempool();
    const n = Array.isArray(mp) ? mp.length : (mp?.transactions?.length ?? 0);
    return `${n} transaction(s)`;
});

if (client.disconnect) await client.disconnect();

console.log(`\n━━━ ${pass} passed, ${fail} failed, ${skip} skipped ━━━\n`);
if (fail > 0) {
    console.log('A failure here means the SDK and the node disagree about something');
    console.log('the offline tests could not check. Please report the line above.\n');
}
process.exit(fail ? 1 : 0);
