// test.js — Comprehensive Midstate SDK smoke test
//
// Coverage:
//   PHASE 1 (offline):   WASM init, wallet create/restore round-trip, WOTS + MSS
//                        address generation, balance + addUtxo, utils, storage
//                        (NodeFS + Memory) round-trips, WASM helper exports.
//   PHASE 2 (read RPCs): get_state, get_block, get_mempool, get_filters, check,
//                        check_commitment, mss_state, block_template.
//   PHASE 3 (push):      subscribe and observe a window of live push events.
//   PHASE 4 (writes):    send_chat, and a full commit→send transaction.
//                        GATED behind MIDSTATE_LIVE_TX=1 because they broadcast
//                        to the real network and (for tx) spend real funds.
//
// Every step is isolated: a failure is recorded and the run continues, with a
// PASS/FAIL/SKIP summary at the end. Exit code is nonzero if anything FAILED,
// so this is CI-friendly.
//
// Env flags:
//   MIDSTATE_P2P_DEBUG=1     verbose stream tracing (handled inside p2p.js)
//   MIDSTATE_LIVE_TX=1       enable network-mutating writes (chat + tx)
//   MIDSTATE_WALLET_DIR=...  persistent wallet dir for the live tx (default ./my_bot_wallet)
//   MIDSTATE_PUSH_WAIT_MS=N  how long to listen for push events (default 20000)
//   MIDSTATE_PEER=<multiaddr> override the bootstrap peer

import fs from 'fs/promises';
import path from 'path';
import os from 'os';
import {
    Wallet,
    MidstateClient,
    Storage,
    MidstateUtils,
    decompose_amount,
} from './src/index.js';
// PEX wire-format helpers, imported directly from the transport layer so the
// bincode GetAddr/Addr codec can be verified offline (no network needed).
import { encodeGetAddr, decodeAddr, encodeVarint } from './src/p2p.js';

// ── Config ───────────────────────────────────────────────────────────────────
const PEER = process.env.MIDSTATE_PEER
    || '/ip4/134.199.148.215/tcp/9333/p2p/12D3KooWPbR63SQg1UBLpAMiNngqrRHGM4LaMP8ieAJUxhfw7dxv';
const LIVE_TX = process.env.MIDSTATE_LIVE_TX === '1';
const PUSH_WAIT_MS = Number(process.env.MIDSTATE_PUSH_WAIT_MS || 20_000);
const TX_WALLET_DIR = process.env.MIDSTATE_WALLET_DIR || './my_bot_wallet';
const SCAN_WINDOW = Number(process.env.MIDSTATE_SCAN_WINDOW || 300);
const SCAN_PACE_MS = Number(process.env.MIDSTATE_SCAN_PACE_MS || 550);

// ── Tiny test runner ───────────────────────────────────────────────────────--
const results = [];
async function step(name, fn) {
    const t0 = Date.now();
    try {
        const detail = await fn();
        const ms = Date.now() - t0;
        results.push({ name, status: 'PASS', ms, detail: detail ?? '' });
        console.log(`  ✅ ${name}${detail ? ' — ' + detail : ''}  (${ms}ms)`);
    } catch (e) {
        const ms = Date.now() - t0;
        results.push({ name, status: 'FAIL', ms, detail: e.message });
        console.log(`  ❌ ${name} — ${e.message}  (${ms}ms)`);
    }
}
function skip(name, why) {
    results.push({ name, status: 'SKIP', ms: 0, detail: why });
    console.log(`  ⏭️  ${name} — SKIPPED (${why})`);
}
function assert(cond, msg) { if (!cond) throw new Error(msg || 'assertion failed'); }
function summarize(obj, max = 120) {
    let s;
    try { s = JSON.stringify(obj); } catch { s = String(obj); }
    if (s === undefined) s = String(obj);
    return s.length > max ? s.slice(0, max) + '…' : s;
}
function randHex(nBytes) {
    const b = new Uint8Array(nBytes);
    globalThis.crypto.getRandomValues(b);
    return [...b].map((x) => x.toString(16).padStart(2, '0')).join('');
}
const isHex = (s, len) => typeof s === 'string' && /^[0-9a-fA-F]+$/.test(s) && (len ? s.length === len : true);
// An Addr(Vec<String>) with zero entries: [disc 6][count 0]. Built from the same
// exported encodeVarint so it stays correct if the varint scheme ever changes.
function buildEmptyAddr() {
    const d = encodeVarint(6), c = encodeVarint(0);
    const out = new Uint8Array(d.length + c.length);
    out.set(d, 0); out.set(c, d.length);
    return out;
}

// ── Main ───────────────────────────────────────────────────────────────────--
async function run() {
    const tempDirs = [];
    let client;
    let connected = false;

    try {
        // ═══════════════════════════════════════════════════════════════════
        console.log('\n━━━ PHASE 1: Offline (WASM, wallet, utils, storage) ━━━');
        // ═══════════════════════════════════════════════════════════════════

        await step('WASM init', async () => {
            const buf = await fs.readFile('./pkg/wasm_wallet_bg.wasm');
            const mod = await WebAssembly.compile(buf);
            await Wallet.init(mod);
            return `${(buf.length / 1024).toFixed(0)} KiB`;
        });

        // Wallet mechanics on a throwaway temp dir (keeps runs clean/repeatable).
        const walletDir = await fs.mkdtemp(path.join(os.tmpdir(), 'midstate-wallet-'));
        tempDirs.push(walletDir);

        let wallet, wotsAddr, mssAddr;

        await step('Wallet.create', async () => {
            wallet = await Wallet.create(new Storage.NodeFSStorage(walletDir));
            assert(wallet && typeof wallet.phrase === 'string' && wallet.phrase.length > 0, 'no phrase');
            return `phrase words: ${wallet.phrase.split(/\s+/).length}`;
        });

        await step('getNewAddress (WOTS)', async () => {
            wotsAddr = await wallet.getNewAddress();
            assert(isHex(wotsAddr, 64), `bad WOTS addr: ${wotsAddr}`);
            return wotsAddr.slice(0, 16) + '…';
        });

        await step('getNewReusableAddress (MSS, h=2)', async () => {
            mssAddr = await wallet.getNewReusableAddress(2);
            assert(isHex(mssAddr, 64), `bad MSS addr: ${mssAddr}`);
            return mssAddr.slice(0, 16) + '…';
        });

        // Round-trip: restore from the same dir and confirm state matches. This
        // exercises import_mss_bytes + set_mss_leaf_index on the WASM side.
        await step('Wallet.restore round-trip', async () => {
            const restored = await Wallet.restore(new Storage.NodeFSStorage(walletDir));
            assert(restored.phrase === wallet.phrase, 'phrase mismatch after restore');
            assert(
                Object.keys(restored.wotsAddrs).length === Object.keys(wallet.wotsAddrs).length,
                'WOTS addr count mismatch',
            );
            assert(
                Object.keys(restored.mssAddrs).length === Object.keys(wallet.mssAddrs).length,
                'MSS addr count mismatch',
            );
            assert(restored.wotsAddrs[wotsAddr] !== undefined, 'restored wallet missing WOTS addr');
            assert(restored.mssAddrs[mssAddr] !== undefined, 'restored wallet missing MSS addr');
            return 'phrase + addrs intact';
        });

        await step('getBalance (fresh = 0) + addUtxo', async () => {
            assert(wallet.getBalance() === 0n, `fresh balance not 0: ${wallet.getBalance()}`);
            wallet.addUtxo(wotsAddr, 5000, randHex(32), randHex(32));
            assert(wallet.getBalance() === 5000n, `balance after addUtxo: ${wallet.getBalance()}`);
            // Duplicate coin_id must not double-count.
            const dupCoin = randHex(32);
            wallet.addUtxo(wotsAddr, 1000, randHex(32), dupCoin);
            wallet.addUtxo(wotsAddr, 1000, randHex(32), dupCoin);
            assert(wallet.getBalance() === 6000n, `dedup failed: ${wallet.getBalance()}`);
            return 'balance=6000, dedup ok';
        });

        await step('getBalance BigInt precision (> 2^53)', async () => {
            // Currency must not lose precision when summed past Number's safe
            // integer ceiling. Two coins just over 2^53 each would round if held
            // as Number; as BigInt the sum is exact.
            const w = await Wallet.create(new Storage.MemoryStorage());
            const a = await w.getNewAddress();
            const big = (1n << 53n) + 7n; // 9007199254740999, not Number-safe
            w.addUtxo(a, big, randHex(32), randHex(32));
            w.addUtxo(a, big, randHex(32), randHex(32));
            const expected = big * 2n; // 18014398509481998
            assert(typeof w.getBalance() === 'bigint', 'balance must be a BigInt');
            assert(w.getBalance() === expected, `precision lost: got ${w.getBalance()}, want ${expected}`);
            // A Number-based sum would land on 18014398509481996 (even rounding) — prove we don't.
            assert(w.getBalance() !== BigInt(Number(big) + Number(big)), 'balance matches the lossy Number sum — precision bug present');
            return `exact at ${expected} (Number would lose it)`;
        });

        await step('addUtxo rejects foreign address', async () => {
            let threw = false;
            try { wallet.addUtxo(randHex(32), 1, randHex(32), randHex(32)); }
            catch { threw = true; }
            assert(threw, 'expected addUtxo to reject an address not in the wallet');
            return 'rejected as expected';
        });

        // Every UTXO must carry the full field set WasmUtxo expects, because the
        // Rust struct has no serde(default) — a missing field (e.g. mss_height)
        // makes prepare_spend reject the whole JSON, breaking the first real
        // spend. Verify both a WOTS coin (mss_height 0) and an MSS coin.
        await step('UTXO shape matches WasmUtxo (incl. mss_height)', async () => {
            const w = await Wallet.create(new Storage.MemoryStorage());
            const required = ['index', 'is_mss', 'mss_height', 'mss_leaf', 'address', 'value', 'salt', 'coin_id'];

            const wotsA = await w.getNewAddress();
            w.addUtxo(wotsA, 1024, randHex(32), randHex(32));
            const wu = w.utxos[w.utxos.length - 1];
            for (const f of required) assert(f in wu, `WOTS UTXO missing field '${f}'`);
            assert(wu.is_mss === false && wu.mss_height === 0, 'WOTS coin should have is_mss=false, mss_height=0');

            const mssA = await w.getNewReusableAddress(8); // height 8
            w.addUtxo(mssA, 2048, randHex(32), randHex(32));
            const mu = w.utxos[w.utxos.length - 1];
            for (const f of required) assert(f in mu, `MSS UTXO missing field '${f}'`);
            assert(mu.is_mss === true && mu.mss_height === 8, `MSS coin should carry its tree height (got mss_height=${mu.mss_height})`);

            return 'WOTS + MSS UTXOs carry all 8 WasmUtxo fields';
        });

        // ── Utils ──
        await step('MidstateUtils.formatMDS', () => {
            assert(MidstateUtils.formatMDS(0).prefix === 'MDS', '0 → MDS');
            assert(MidstateUtils.formatMDS(5000).prefix === 'kMDS', '5000 → kMDS');
            assert(MidstateUtils.formatMDS(2_000_000).prefix === 'mMDS', '2M → mMDS');
            assert(MidstateUtils.formatMDS(3_000_000_000).prefix === 'gMDS', '3G → gMDS');
            return 'MDS/kMDS/mMDS/gMDS boundaries ok';
        });

        await step('MidstateUtils.computeCoinId', () => {
            const id = MidstateUtils.computeCoinId(wotsAddr, 1000, randHex(32));
            assert(isHex(id, 64), `coin id not 64-hex: ${id}`);
            return id.slice(0, 16) + '…';
        });

        await step('MidstateUtils.hash (blake3)', () => {
            const h = MidstateUtils.hash(wotsAddr);
            assert(isHex(h, 64), `hash not 64-hex: ${h}`);
            // Deterministic.
            assert(MidstateUtils.hash(wotsAddr) === h, 'hash not deterministic');
            return h.slice(0, 16) + '…';
        });

        await step('WASM decompose_amount', () => {
            let out;
            try { out = decompose_amount(12345); }
            catch { out = decompose_amount(BigInt(12345)); }
            assert(out != null, 'returned null');
            return summarize(out, 80);
        });

        // ── Storage providers (explicit round-trips) ──
        await step('Storage.MemoryStorage round-trip', async () => {
            const m = new Storage.MemoryStorage();
            await m.saveMetadata('{"hello":1}');
            assert((await m.loadMetadata()) === '{"hello":1}', 'metadata mismatch');
            await m.saveMssTree('addrX', new Uint8Array([1, 2, 3]));
            const t = await m.loadMssTree('addrX');
            assert(t && t.length === 3 && t[0] === 1, 'mss tree mismatch');
            assert((await m.loadMssTree('missing')) === undefined, 'missing key should be undefined');
            return 'metadata + mss tree ok';
        });

        await step('Storage.NodeFSStorage round-trip', async () => {
            const sdir = await fs.mkdtemp(path.join(os.tmpdir(), 'midstate-store-'));
            tempDirs.push(sdir);
            const s = new Storage.NodeFSStorage(sdir);
            await s.saveMetadata('{"x":42}');
            assert((await s.loadMetadata()) === '{"x":42}', 'metadata mismatch');
            await s.saveMssTree('aa', new Uint8Array([9, 8, 7]));
            const t = await s.loadMssTree('aa');
            assert(t && t.length === 3 && t[0] === 9, 'mss tree mismatch');
            // A provider pointed at a dir with no wallet returns null, not a throw.
            const empty = new Storage.NodeFSStorage(sdir + '-does-not-exist');
            assert((await empty.loadMetadata()) === null, 'missing metadata should be null');

            // BigInt UTXO values must survive a full save→restore as exact
            // BigInt — this is the reload path where a Number regression would
            // silently creep back in. Use a value above 2^53.
            const wdir = await fs.mkdtemp(path.join(os.tmpdir(), 'midstate-bal-'));
            tempDirs.push(wdir);
            const w = await Wallet.create(new Storage.NodeFSStorage(wdir));
            const a = await w.getNewAddress();
            const big = (1n << 53n) + 123n;
            w.addUtxo(a, big, randHex(32), randHex(32));
            await w.save();
            const rw = await Wallet.restore(new Storage.NodeFSStorage(wdir));
            assert(typeof rw.utxos[0].value === 'bigint', 'restored UTXO value is not BigInt');
            assert(rw.utxos[0].value === big, `restored value ${rw.utxos[0].value} != ${big}`);
            assert(rw.getBalance() === big, 'restored balance lost precision');

            return 'disk persistence ok (incl. BigInt value round-trip)';
        });

        // PEX wire format (offline). The binary /midstate/2.0.0 protocol is bincode
        // with DefaultOptions = little-endian varint. Pin the GetAddr/Addr codec so a
        // wire-format regression fails here, in CI, instead of silently degrading the
        // best-effort discovery path (whose errors are swallowed by design).
        await step('PEX codec (GetAddr/Addr bincode varint)', () => {
            // GetAddr is Message discriminant 5 → exactly the single byte 0x05.
            const ga = encodeGetAddr();
            assert(ga.length === 1 && ga[0] === 0x05, `GetAddr must be [0x05], got [${[...ga]}]`);

            // Build an Addr(Vec<String>) payload exactly as bincode would
            // (disc 6, varint count, then varint-len + utf8 per string) and
            // confirm decodeAddr round-trips it — including a >250-char entry
            // that exercises the u16 length marker (251 + LE u16).
            const sample = [
                '/ip4/134.199.148.215/tcp/9333/p2p/12D3KooWPbR63SQg1UBLpAMiNngqrRHGM4LaMP8ieAJUxhfw7dxv',
                '/ip4/203.0.113.7/tcp/9333/p2p/12D3KooWAlpha',
                '/ip4/198.51.100.2/tcp/9333/p2p/' + 'Q'.repeat(260),
            ];
            const parts = [encodeVarint(6), encodeVarint(sample.length)];
            for (const s of sample) {
                const sb = new TextEncoder().encode(s);
                parts.push(encodeVarint(sb.length), sb);
            }
            let total = 0; for (const p of parts) total += p.length;
            const payload = new Uint8Array(total);
            let off = 0; for (const p of parts) { payload.set(p, off); off += p.length; }

            const decoded = decodeAddr(payload);
            assert(JSON.stringify(decoded) === JSON.stringify(sample), 'Addr did not round-trip');
            assert(decodeAddr(buildEmptyAddr()).length === 0, 'empty Addr should decode to []');

            // A truncated payload must be rejected, not silently yield garbage.
            let threw = false;
            try { decodeAddr(Uint8Array.of(6, 1, 250)); } catch { threw = true; }
            assert(threw, 'truncated Addr payload must throw');

            return `GetAddr=0x05, Addr round-trips ${sample.length} addrs (incl. u16-len)`;
        });

        // ═══════════════════════════════════════════════════════════════════
        console.log('\n━━━ PHASE 2: Read RPCs (live P2P/TCP) ━━━');
        // ═══════════════════════════════════════════════════════════════════

        const pushEvents = [];
        client = new MidstateClient([PEER]);
        client.onStatusChange((s) => console.log(`     [status] ${s}`));
        client.onPushEvent((event) => {
            pushEvents.push(event);
            if (event.NewBlockTip) {
                console.log(`     🔔 push: NewBlockTip height=${event.NewBlockTip.height}`);
            } else if (event.ChatMessage) {
                console.log(`     💬 push: chat from ${event.ChatMessage.sender}: ${summarize(event.ChatMessage.words, 40)}`);
            } else {
                console.log(`     📨 push: ${summarize(event, 60)}`);
            }
        });

        await step('connect', async () => {
            await client.connect();
            connected = true;
            return 'P2P link up';
        });

        // PEX discovery is best-effort and fires in the background after connect,
        // so this is a SOFT observation, never a hard gate: if the bootstrap node
        // is the only reachable TCP peer right now, discovering zero new peers is
        // correct, not a failure. We give discovery a moment, then report what the
        // pool looks like. If the client doesn't surface its transport internals,
        // we skip cleanly rather than guess at field names.
        if (connected) {
            const p2p = (typeof client.getP2P === 'function' && client.getP2P())
                || client.p2pClient || client._p2p || client.p2p || client.transport || null;
            const pool = p2p && (p2p.knownMultiaddrs instanceof Set) ? p2p.knownMultiaddrs : null;

            if (!pool) {
                skip('PEX discovery (post-connect)', 'client does not expose the P2P peer pool; expose getP2P() or .knownMultiaddrs to enable this check');
            } else {
                await step('PEX discovery (post-connect)', async () => {
                    const before = pool.size;
                    await new Promise((r) => setTimeout(r, 4000)); // let GetAddr + spare dials settle
                    const after = pool.size;
                    // Sanity, not a count gate: the seed must still be there and the
                    // primary peer reference must not have been corrupted by extras.
                    assert(after >= 1, 'peer pool unexpectedly empty');
                    assert(p2p.connectedPeer, 'connectedPeer was lost after discovery');
                    const discovered = after - before;
                    const connCount = (typeof p2p.node?.getConnections === 'function')
                        ? p2p.node.getConnections().length : '?';
                    return discovered > 0
                        ? `discovered ${discovered} new peer(s); pool=${after}, live connections=${connCount}`
                        : `no new peers (bootstrap may be the only reachable TCP node); pool=${after}, live connections=${connCount}`;
                });
            }
        }

        let height = 0;
        if (connected) {
            await step('RPC get_state', async () => {
                const st = await client.getState();
                assert(st && typeof st.height === 'number', 'no numeric height');
                height = st.height;
                return `height=${st.height}`;
            });

            await step('RPC get_block (tip)', async () => {
                const b = await client.getBlock(Math.max(0, height - 1));
                assert(b != null, 'null block');
                return summarize(b, 90);
            });

            await step('RPC get_block (genesis #0)', async () => {
                const b = await client.getBlock(0);
                assert(b != null, 'null genesis block');
                return summarize(b, 90);
            });

            await step('RPC get_mempool', async () => {
                const mp = await client.getMempool();
                assert(mp != null, 'null mempool');
                return summarize(mp, 90);
            });

            await step('RPC get_filters', async () => {
                const f = await client.getFilters(Math.max(0, height - 3), height);
                assert(f != null, 'null filters');
                return summarize(f, 90);
            });

            await step('RPC check (random coin → not found)', async () => {
                const r = await client.checkCoin(randHex(32));
                assert(r != null, 'null check result');
                return summarize(r, 90);
            });

            await step('RPC check_commitment (random → not found)', async () => {
                const r = await client.checkCommitment(randHex(32));
                assert(r != null, 'null check_commitment result');
                return summarize(r, 90);
            });

            await step('RPC mss_state (fresh MSS addr)', async () => {
                const r = await client.getMssState(mssAddr);
                assert(r != null, 'null mss_state result');
                return summarize(r, 90);
            });

            await step('RPC block_template', async () => {
                const tmpl = await client.getBlockTemplate([
                    { address: wotsAddr, value: 1, salt: randHex(32) },
                ]);
                // Either a valid template (ok) or a structured fee-mismatch (409) is
                // proof the RPC round-trips correctly.
                const data = await tmpl.json();
                if (tmpl.ok) return 'template ok: ' + summarize(data, 70);
                return `server responded (status ${tmpl.status}): ` + summarize(data, 70);
            });

            // Exercise the compact block-filter scan against the live node over a
            // bounded recent window (cheap: ~1 getFilters call + any false-positive
            // block fetches). A fresh wallet finds nothing, which is correct — the
            // point is to prove set_watchlist + check_filter + getFilters + the
            // getBlock-on-match path all work end to end. Full-chain scan is what
            // the funded tx wallet does in Phase 4.
            await step(`wallet.sync (recent ${SCAN_WINDOW}-block window)`, async () => {
                const scanWallet = await Wallet.create(new Storage.MemoryStorage());
                await scanWallet.getNewAddress();
                // Bound the scan to the tail of the chain to keep the test fast.
                scanWallet.lastScannedHeight = Math.max(0, height - SCAN_WINDOW);
                const r = await scanWallet.sync(client);
                assert(r.height === height, `sync stopped at ${r.height}, expected tip ${height}`);
                return `scanned→${r.height}, blocks matched=${r.found}, balance=${r.balance}`;
            });

            // ═══════════════════════════════════════════════════════════════
            console.log(`\n━━━ PHASE 3: Push events (listening ${PUSH_WAIT_MS}ms) ━━━`);
            // ═══════════════════════════════════════════════════════════════
            await step(`push subscription (${PUSH_WAIT_MS}ms window)`, async () => {
                const before = pushEvents.length;
                await new Promise((r) => setTimeout(r, PUSH_WAIT_MS));
                const got = pushEvents.length - before;
                // 0 events is not a failure (block cadence varies); we only verify
                // the subscription path didn't throw and the window completed.
                return got > 0 ? `${got} event(s) received` : 'no events in window (ok — depends on block timing)';
            });

            // ═══════════════════════════════════════════════════════════════
            console.log('\n━━━ PHASE 4: Network writes ━━━');
            // ═══════════════════════════════════════════════════════════════
            if (LIVE_TX) {
                console.log('  (MIDSTATE_LIVE_TX=1 — these broadcast to the real network)');

                await step('WRITE send_chat (LIVE broadcast)', async () => {
                    const resp = await client.sendChat([1, 2, 3], null, []);
                    assert(resp.ok, `chat rejected: ${resp.body}`);
                    return 'chat broadcast accepted';
                });

                // Full transaction needs a funded wallet. Use a persistent dir so a
                // user who has funded one of its addresses can run it. We sync first
                // so coins sent to the wallet's addresses are discovered from-chain.
                let txWallet;
                try {
                    txWallet = await Wallet.restore(new Storage.NodeFSStorage(TX_WALLET_DIR));
                } catch {
                    txWallet = await Wallet.create(new Storage.NodeFSStorage(TX_WALLET_DIR));
                }

                await step('wallet.sync (full, discover funds)', async () => {
                    const r = await txWallet.sync(client, {
                        filterIntervalMs: SCAN_PACE_MS,
                        onProgress: ({ height: h, chainHeight, note }) => {
                            if (note) console.log(`     ${note}`);
                            else if (h % 20000 === 0) console.log(`     scanning ${h}/${chainHeight}…`);
                        },
                    });
                    return `height=${r.height}, balance=${r.balance}, utxos=${r.utxos}`;
                });

                const bal = txWallet.getBalance();
                if (bal > 1) {
                    await step('WRITE commit→send (LIVE, spends funds)', async () => {
                        const self = await txWallet.getNewAddress();
                        const resp = await txWallet.send(client, self, 1);
                        return summarize(resp, 100);
                    });
                } else {
                    skip('WRITE commit→send',
                        `wallet ${TX_WALLET_DIR} balance=${bal} after sync. Send funds to one of its getNewAddress() addresses, then re-run.`);
                }
            } else {
                skip('WRITE send_chat', 'set MIDSTATE_LIVE_TX=1 to broadcast a real chat');
                skip('WRITE commit→send', 'set MIDSTATE_LIVE_TX=1 (and fund a wallet) to broadcast a real tx');
            }
            // submit_batch is intentionally not exercised live: a valid batch requires
            // a fully mined block, and sending garbage would incur a server-side
            // reputation penalty. It travels the same _route() write path as the
            // writes above, so it's covered transitively.
            skip('WRITE submit_batch', 'requires a mined block; not safe to fuzz against a live node');
        } else {
            console.log('  Connection failed — skipping read RPCs, push, and writes.');
        }
    } finally {
        // ── Cleanup ──
        console.log('\n━━━ Cleanup ━━━');
        if (client && connected) {
            await client.disconnect();
            console.log('  ✅ disconnected');
        }
        for (const d of tempDirs) {
            await fs.rm(d, { recursive: true, force: true }).catch(() => {});
        }
        if (tempDirs.length) console.log(`  ✅ removed ${tempDirs.length} temp dir(s)`);
    }

    // ── Summary ──
    const pass = results.filter((r) => r.status === 'PASS').length;
    const fail = results.filter((r) => r.status === 'FAIL').length;
    const skipd = results.filter((r) => r.status === 'SKIP').length;

    console.log('\n══════════════════════ SUMMARY ══════════════════════');
    for (const r of results) {
        const icon = r.status === 'PASS' ? '✅' : r.status === 'FAIL' ? '❌' : '⏭️ ';
        console.log(`  ${icon} ${r.name.padEnd(40)} ${r.detail}`);
    }
    console.log('──────────────────────────────────────────────────────');
    console.log(`  ${pass} passed · ${fail} failed · ${skipd} skipped  (of ${results.length})`);
    console.log('══════════════════════════════════════════════════════');

    if (fail > 0) {
        console.log('\n❌ Some checks failed.');
        process.exitCode = 1;
    } else {
        console.log('\n🎉 All executed checks passed.');
        if (!LIVE_TX) console.log('   (Network writes were skipped — run with MIDSTATE_LIVE_TX=1 to include them.)');
    }
}

run().catch((e) => {
    console.error('\n💥 Fatal error outside test runner:', e);
    process.exitCode = 1;
});
