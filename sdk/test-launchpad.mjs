// test-launchpad.mjs — curve-then-pool launches, run in the VM.
//
// Every trade below runs the COMPILED contract once per reserve coin, the way
// consensus does, with the outputs the SDK builds. The attacks are the reason
// for each line of the script; the "without this check" cases prove the
// guards are load-bearing rather than decorative.
//
// The replay test then rebuilds the whole launch from the transactions alone,
// with witnesses stripped as every node serves them, and must arrive at the
// same ledger, state and coins.
//
// The SDK VM compares a signature with its key as strings. So every spend is
// also recorded with REAL MSS signatures over a fixed commitment, and
// `WRITE_VECTORS=1` writes them to test-vectors/launchpad.json, which the
// node's own VM replays in tests/launchpad_vectors.rs (`cargo test --test
// launchpad_vectors`). Same accept/refuse on both, or the SDK is wrong.
//
// Run: node test-launchpad.mjs

import fs from 'fs/promises';
import initWasm, * as W from './pkg/wasm_wallet.js';
import { compile, V6_ACTIVATION_HEIGHT } from './src/compiler.js';
import { execute } from './src/vm.js';
import * as LP from './src/launchpad.js';

await initWasm({ module_or_path: await fs.readFile('./pkg/wasm_wallet_bg.wasm') });

let pass = 0, fail = 0, CURRENT = '';
async function t(name, fn) {
    CURRENT = name;
    try { await fn(); console.log(`  ✅ ${name}`); pass++; }
    catch (e) { console.log(`  ❌ ${name} — ${e.message}`); fail++; }
}
function assert(c, m) { if (!c) throw new Error(m || 'assertion failed'); }
const H = LP.LAUNCHPAD_HEIGHT;
const le8 = LP.le8;

// Small enough to read, big enough for both phases: 20 tokens on the curve,
// 10 in the pool, 16 rows.
const params = { depth: 4, threshold: 20, price: 1000, seed: 1 };
// Pinned: if the compiler ever emits different code for this source, existing
// launches would no longer be found at their addresses. Change only on purpose.
const GOLDEN_CODE = 'b0a3278480d68b10', GOLDEN_GENESIS = 'd744f858d2f39233';
const SIG = 4;   // the VM compares signature and key as strings; real ones are MSS

// ── A tiny chain ────────────────────────────────────────────────────────────
//
// `Sim` holds the coins and the transactions accepted so far, and drives a
// Launch the way the page does: prepareTrade, then the wallet's part (inputs
// get the signature), then the VM, then applyTx on the accepted transaction.

const ann = LP.decodeAnnouncement(LP.encodeAnnouncement({ params, ticker: 'TEST', name: 'Test token', nonce: '0102030405060708' }));
const code = LP.compileLaunch(params, compile).code;
const baseAsm = compile(LP.contractSource(params), { height: H }).asm;
const asmFor = (st) => ['PUSH_HEX ' + st, ...baseAsm.slice(1)];

/** The page's view (`launch`) and the chain's accepted transactions. */
function freshSim() {
    const launch = new LP.Launch(ann, code);
    const out = launch.launchOutputs({ salt: '5a'.repeat(32) });
    const launchTx = { inputs: [], outputs: [
        ...out.fundings.map((f) => ({ type: 'standard', address: f.address, value: f.amount, salt: f.salt || '00'.repeat(32) })),
        ...out.burns.map((b) => ({ type: 'data_burn', payload: b, value_burned: 0 })),
    ] };
    const [found] = LP.launchesInTx(launchTx, () => code, 1);
    return { launch: found, txs: [launchTx] };
}

/** Run a spend through the VM for every reserve input; returns {ok, error}. */
function runSpend(spend, { inputsOverride = null } = {}) {
    const st = LP.splitBytecode(spend.bytecode).state;
    const addr = W.blake3_hash_hex(spend.bytecode);
    const inputs = inputsOverride || spend.inputs;
    const sum = inputs.reduce((a, i) => a + BigInt(i.value), 0n);
    const outputs = spend.outputs.map((o) => ({ address: o.address, value: BigInt(o.value), state: o.out_type === 'confidential' ? o.state : null }));
    for (const i of inputs) {
        const r = execute(asmFor(st), { height: H, witness: i.witness.split(','), thisAddress: addr, sumInputValue: sum, inputValue: BigInt(i.value), outputs });
        if (!r.ok) return r;
    }
    return { ok: true };
}

/** The wallet's job: put the signature in. Here the VM's signature is the key itself. */
function sign(spend, pk) {
    for (const i of spend.inputs) {
        const w = i.witness.split(',');
        assert(w[LP.SIG_ITEM] === '00'.repeat(SIG), 'placeholder missing');
        w[LP.SIG_ITEM] = pk;
        i.witness = w.join(',');
    }
    return spend;
}

/** What a block holds for an accepted spend: no witnesses. */
function blockTx(spend) {
    return { Reveal: {
        inputs: spend.inputs.map((i) => ({ predicate: { Script: { bytecode: spend.bytecode } }, value: i.value, salt: i.salt, commitment: null })),
        witnesses: [],
        outputs: spend.outputs.map((o) => o.out_type === 'confidential'
            ? { Confidential: { address: o.address, commitment: o.state, salt: o.salt } }
            : { Standard: { address: o.address, value: o.value, salt: o.salt } }),
        salt: '00'.repeat(32),
    } };
}

// Real MSS keys (height 2, from a fixed seed), so the recorded vectors carry
// signatures the node will actually verify.
const COMMITMENT = W.blake3_hash_hex('6c61756e63687061642074657374207665' + '63746f7273');   // "launchpad test vectors"
const keyWallet = W.WebWallet.from_seed_hex(W.blake3_hash_hex('6c61756e6368706164'));
const keys = {}, REAL = {};
for (const [i, who] of ['alice', 'bob', 'carol', 'dave', 'mallory'].entries()) {
    const pk = keyWallet.get_mss_pubkey(keyWallet.get_mss_address(i, 2, null));
    keys[who] = pk;
    REAL[pk] = keyWallet.sign_mss_hex(pk, COMMITMENT);
}

// ── Vectors for the node's VM ───────────────────────────────────────────────
const vectors = [];
function record(ok, spend, inputs) {
    // One witness per spend (every reserve input carries the same one); the
    // code and the signatures are stored once, at the top of the file.
    const ws = new Set(inputs.map((i) => i.witness));
    assert(ws.size === 1, 'inputs with different witnesses');
    const pks = Object.keys(REAL);
    const [ca, cb] = LP.splitCode(code);
    const witness = inputs[0].witness.split(',').map((x, k) =>
        (k === LP.SIG_ITEM && REAL[x]) ? `$sig${pks.indexOf(x)}` : (x === ca && k === 0 ? '$codeA' : (x === cb && k === 1 ? '$codeB' : x)));
    const n = vectors.filter((v) => v.name === CURRENT || v.name.startsWith(CURRENT + ' #')).length;
    vectors.push({
        name: n ? `${CURRENT} #${n + 1}` : CURRENT,
        ok,
        state: LP.splitBytecode(spend.bytecode).state,
        inputs: inputs.map((i) => Number(i.value)),
        witness,
        outputs: spend.outputs.map((o) => o.out_type === 'confidential'
            ? { confidential: { address: o.address, commitment: o.state, salt: o.salt } }
            : { standard: { address: o.address, value: Number(o.value), salt: o.salt } }),
    });
}

/** An honest trade; asserts it is accepted and applies it. */
function trade(sim, who, side, n, opts = {}) {
    const pk = keys[who];
    const p = sim.launch.prepareTrade({ side, n: BigInt(n), pubkey: pk, sigBytes: SIG, ...opts });
    sign(p.spend, pk);
    const r = runSpend(p.spend);
    record(true, p.spend, p.spend.inputs);
    assert(r.ok, `${who} ${side} ${n}: ${r.error}`);
    const tx = blockTx(p.spend);
    sim.txs.push(tx);
    const step = sim.launch.applyTx(tx, sim.txs.length);
    assert(step && step.slot === p.slot && step.newBal === p.newBal, 'replay disagrees with the trade');
    assert(sim.launch.state === p.next, 'state did not advance to the traded state');
    return { p, step };
}

/** A tampered trade; asserts the contract refuses it. */
function attack(sim, who, side, n, tamper, opts = {}) {
    const pk = keys[who];
    const p = sim.launch.prepareTrade({ side, n: BigInt(n), pubkey: pk, sigBytes: SIG, ...opts });
    sign(p.spend, pk);
    const w = p.spend.inputs[0].witness.split(',');
    const res = tamper({ w, spend: p.spend, p }) || {};
    const witness = (res.w || w).join(',');
    for (const i of p.spend.inputs) i.witness = witness;
    const r = runSpend(p.spend, res);
    record(false, p.spend, res.inputsOverride || p.spend.inputs);
    assert(!r.ok, 'the contract accepted it');
    return r.error;
}
// Witness positions.
const I = { codeA: 0, codeB: 1, st: 2, proof: 3, ownerOld: 4, ownerNew: 5, pk: LP.PK_ITEM, oldBal: 7, newBal: 8, s: 9, x: 10, y: 11, sig: LP.SIG_ITEM };
const resetReserve = (spend, x2, address) => {
    spend.outputs.splice(1, spend.outputs.length - 1, ...LP.pow2Split(x2 + 1n).map((v) => ({ out_type: 'standard', address, value: Number(v), salt: '11'.repeat(32) })));
};

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n━━━ the script ━━━');

await t('compiles under 1024 bytes with one signature check', () => {
    const c = compile(LP.contractSource(params), { height: H });
    assert(c.sizeBytes <= 1024 && c.sigops === 1, `${c.sizeBytes} bytes, ${c.sigops} sigops`);
    const big = compile(LP.contractSource({ depth: 16, threshold: 1_048_576, price: 3, seed: 1 << 20 }), { height: H });
    assert(big.sizeBytes <= 1024, `the largest launch is ${big.sizeBytes} bytes`);
});

await t('the code after PUSH(state) DROP is the same for every state', () => {
    const st = 'cd'.repeat(56);
    const again = compile(LP.contractSource(params, st), { height: H });
    assert(LP.bytecodeFor(code, st) === again.bytecode, 'splicing a state does not give the compiled script');
});

await t('no witness item but the signature can be read by a node as a one-time signature', () => {
    // 576 bytes reads as WOTS when first; 620+ can read as MSS. Every launch shape must stay clear.
    for (const depth of [1, 8, 16]) {
        for (const [T, P] of [[2, 1], [1_048_576, 3], [4_000_000_000, 1]]) {
            let c;
            try { c = LP.compileLaunch({ depth, threshold: T, price: P, seed: 1 << 20 }, compile).code; } catch { continue; }
            const [a, b] = LP.splitCode(c);
            const longest = Math.max(a.length, b.length, depth * 66) / 2;
            assert(longest <= LP.MAX_PLAIN_ITEM, `depth ${depth}, T ${T}: an item of ${longest} bytes`);
        }
    }
});

await t('the launch height is the v6 activation height', () => {
    assert(LP.LAUNCHPAD_HEIGHT === V6_ACTIVATION_HEIGHT);
});

await t('golden: a fixed launch compiles to a fixed address', () => {
    // A compiler change that alters this would orphan every launch made before it.
    const g = new LP.Launch(ann, code);
    assert(W.blake3_hash_hex(code).slice(0, 16) === GOLDEN_CODE, `code hash ${W.blake3_hash_hex(code).slice(0, 16)}`);
    assert(g.genesisAddress.slice(0, 16) === GOLDEN_GENESIS, `genesis ${g.genesisAddress.slice(0, 16)}`);
});

await t('parameters: odd thresholds, bad seeds and oversized pools are refused', () => {
    for (const bad of [{ ...params, threshold: 21 }, { ...params, seed: 3 }, { ...params, depth: 17 },
                       { depth: 8, threshold: 2_000_000, price: 1_000_000_000, seed: 1 }]) {
        let threw = false;
        try { LP.validateParams(bad); } catch { threw = true; }
        assert(threw, JSON.stringify(bad));
    }
});

await t('announcement round trip, and the tag binds every byte', () => {
    const hex = LP.encodeAnnouncement({ params, ticker: 'PEPE2', name: 'Pépé the second 🐸', nonce: 'aa'.repeat(8) });
    const a = LP.decodeAnnouncement(hex);
    assert(a && a.ticker === 'PEPE2' && a.name === 'Pépé the second 🐸' && a.params.price === 1000n, 'round trip');
    const b = LP.decodeAnnouncement(LP.encodeAnnouncement({ params, ticker: 'PEPE2', name: 'Pépé the third', nonce: 'aa'.repeat(8) }));
    assert(a.tag !== b.tag, 'renaming kept the tag');
    assert(new LP.Launch(a, code).genesisAddress !== new LP.Launch(b, code).genesisAddress, 'renaming kept the address');
    let threw = false;
    try { LP.encodeAnnouncement({ params, ticker: 'X', name: 'y'.repeat(60), nonce: '00'.repeat(8) }); } catch { threw = true; }
    assert(threw, 'an announcement over 80 bytes was encoded');
});

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n━━━ the curve ━━━');
const sim = freshSim();

await t('buys and a sell on the curve, each spending every reserve coin', () => {
    trade(sim, 'alice', 'buy', 5);
    trade(sim, 'bob', 'buy', 4);
    trade(sim, 'alice', 'buy', 2);
    const { step } = trade(sim, 'bob', 'sell', 1);
    assert(step.tokens === -1n && step.mds < 0n, 'sell recorded wrongly');
    const st = sim.launch.st;
    assert(st.supply === 10n && st.x === LP.curveReserve(1000, 10), `supply ${st.supply}, x ${st.x}`);
    assert(sim.launch.coins.length > 1, 'the reserve should be spread over several coins by now');
});

const attacks = [
    ['a different code copy', ({ w }) => { w[I.codeB] = w[I.codeB].slice(0, -2) + 'ff'; }],
    ['claiming another state', ({ w }) => { w[I.st] = LP.encodeState({ ...LP.decodeState(w[I.st]), supply: 0n }); }],
    ['a state copy that swallows the code up to its last DROP (reserve to an anyone-can-spend tail)', ({ w, spend, p }) => {
        const c0 = w[I.codeA] + w[I.codeB], j = c0.length / 2 - 5;
        assert(c0.slice(j * 2, j * 2 + 2) === '10', 'expected DROP before PUSH 1');
        w[I.st] = w[I.st] + '10' + c0.slice(0, j * 2);
        w[I.codeA] = c0.slice((j + 1) * 2);
        w[I.codeB] = '';
        const stolen = LP.addressFor(w[I.codeA], p.next);
        for (const o of spend.outputs) o.address = stolen;
    }],
    ['a 7-byte supply in the next state (reserve to a script nothing can spend)', ({ w, spend, p }) => {
        w[I.s] = w[I.s].slice(0, 14);
        const dead = W.blake3_hash_hex(LP.PREFIX + w[I.s] + w[I.x] + w[I.y] + p.next.slice(48) + LP.DROP + w[I.codeA] + w[I.codeB]);
        for (const o of spend.outputs) o.address = dead;
    }],
    ['a 7-byte new balance', ({ w }) => { w[I.newBal] = w[I.newBal].slice(0, 14); }],
    ['paying one unit less than the curve', ({ w }) => { w[I.x] = le8(LP.readLe8(w[I.x]) - 1n); }],
    ['crediting yourself an extra token', ({ w }) => { w[I.newBal] = le8(LP.readLe8(w[I.newBal]) + 1n); }],
    ['moving the pool side during the curve', ({ w }) => { w[I.y] = le8(LP.readLe8(w[I.y]) - 1n); w[I.newBal] = le8(LP.readLe8(w[I.newBal]) + 1n); }],
    ['leaving one reserve coin behind', ({ spend }) => ({ inputsOverride: spend.inputs.slice(1) })],
    ['one unit short at the next address', ({ spend }) => { spend.outputs[1].value -= 1; }],
    ['the reserve to another address', ({ spend }) => { for (const o of spend.outputs.slice(1)) o.address = 'dd'.repeat(32); }],
    ['no receipt', ({ spend }) => { spend.outputs.shift(); }],
    ['a receipt with another balance', ({ spend, p }) => { spend.outputs[0].state = p.owner + le8(p.newBal + 1n); }],
    ['a receipt at another address', ({ spend }) => { spend.outputs[0].address = 'dd'.repeat(32); }],
    ['a short proof', ({ w }) => { w[I.proof] = w[I.proof].slice(66); }],
    ['the old balance in 7 bytes', ({ w }) => { w[I.oldBal] = w[I.oldBal].slice(0, 14); }],
    ['a signature from another key', ({ w }) => { w[LP.SIG_ITEM] = keys.mallory; }],
];
for (const [what, fn] of attacks) {
    await t(`refused: ${what}`, () => { attack(sim, 'carol', 'buy', 3, fn); });
}

await t("refused: selling, or buying into, alice's row with mallory's key", () => {
    const slot = sim.launch.ledger.slotOf(LP.ownerOf(keys.alice));
    let threw = false;
    try { sim.launch.prepareTrade({ side: 'sell', n: 1n, pubkey: keys.mallory, sigBytes: SIG, slot }); } catch { threw = true; }
    assert(threw, 'the SDK built it');
    // Build it anyway, as an attacker would, by borrowing alice's trade.
    for (const side of ['sell', 'buy']) {
        attack(sim, 'alice', side, 1, ({ w }) => { w[LP.PK_ITEM] = keys.mallory; w[LP.SIG_ITEM] = keys.mallory; });
    }
});

await t("refused: re-reading alice's row as an empty one (0-byte balance)", () => {
    const slot = sim.launch.ledger.slotOf(LP.ownerOf(keys.alice));
    attack(sim, 'mallory', 'buy', 1, ({ w, spend, p }) => {
        w[I.oldBal] = '';
        w[I.ownerOld] = le8(sim.launch.ledger.balances[slot]) + sim.launch.ledger.owners[slot];
        w[I.newBal] = le8(1n);
        const root = LP.rootFromProof(LP.leafHash(1n, LP.ownerOf(keys.mallory)), w[I.proof]);
        const next = LP.encodeState({ ...LP.decodeState(p.next), root });
        const a = LP.addressFor(code, next);
        for (const o of spend.outputs) o.address = a;
        spend.outputs[0].state = LP.ownerOf(keys.mallory) + le8(1n);
    }, { slot: sim.launch.ledger.freeSlot() });
});

await t('refused: spending the reserve together with a donated copy of it', () => {
    attack(sim, 'carol', 'buy', 1, ({ spend }) => ({ inputsOverride: [...spend.inputs, ...spend.inputs.map((i) => ({ ...i }))] }));
});

await t('hand-over: alice gives her whole row to dave, who can then sell from it', () => {
    const slot = sim.launch.ledger.slotOf(LP.ownerOf(keys.alice));
    const { step } = trade(sim, 'alice', 'buy', 0, { to: LP.ownerOf(keys.dave) });
    assert(step.tokens === 0n && step.mds === 0n && step.owner === LP.ownerOf(keys.dave), 'hand-over recorded wrongly');
    assert(sim.launch.ledger.slotOf(LP.ownerOf(keys.alice)) === -1, 'alice still has a row');
    let threw = false;
    try { sim.launch.prepareTrade({ side: 'sell', n: 1n, pubkey: keys.alice, sigBytes: SIG, slot }); } catch { threw = true; }
    assert(threw, 'alice could still sell');
    trade(sim, 'dave', 'sell', 1);
});

await t('quotes refuse a buy past the threshold, and the contract refuses one built by hand', () => {
    const left = sim.launch.params.threshold - sim.launch.st.supply;
    let threw = false;
    try { sim.launch.prepareTrade({ side: 'buy', n: left + 1n, pubkey: keys.carol, sigBytes: SIG }); } catch { threw = true; }
    assert(threw, 'quoted past the threshold');
    attack(sim, 'carol', 'buy', left, ({ w, spend }) => {
        const s2 = sim.launch.params.threshold + 1n, x2 = LP.curveReserve(1000, s2);
        w[I.s] = le8(s2); w[I.x] = le8(x2); w[I.newBal] = le8(LP.readLe8(w[I.newBal]) + 1n);
        resetReserve(spend, x2, spend.outputs[0].address);
    });
});

await t('tokensFor matches quote on the curve', () => {
    const st = sim.launch.st;
    for (const mds of [0n, 999n, 10_000n, 55_555n, 10n ** 9n]) {
        const n = LP.tokensFor(params, st, mds);
        if (n > 0n) assert(LP.quote(params, st, 'buy', n).mds <= mds, 'over budget');
        if (n < params.threshold - Number(st.supply)) assert(LP.quote(params, st, 'buy', n + 1n).mds > mds, 'not the most');
    }
});

await t('the last curve buy graduates, and the pool opens at the last curve price', () => {
    const left = sim.launch.params.threshold - sim.launch.st.supply;
    const { step } = trade(sim, 'carol', 'buy', left);
    assert(step.graduated && sim.launch.graduated, 'did not graduate');
    const st = sim.launch.st, P = 1000n, T = 20n;
    assert(st.x === LP.curveReserve(P, T) && st.y === T / 2n, `pool x ${st.x}, y ${st.y}`);
    assert(st.x / st.y === P * (T - 1n), `pool price ${st.x / st.y} vs last curve price ${P * (T - 1n)}`);
});

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n━━━ the pool ━━━');

await t('buys and sells against the pool', () => {
    const k0 = sim.launch.st.x * sim.launch.st.y;
    trade(sim, 'bob', 'buy', 2);
    trade(sim, 'dave', 'sell', 4);
    trade(sim, 'carol', 'sell', 2);
    assert(sim.launch.st.x * sim.launch.st.y >= k0, 'x*y fell');
});

await t('refused: paying the pool one unit less than the quote', () => {
    attack(sim, 'carol', 'buy', 1, ({ w, spend }) => {
        const x2 = LP.readLe8(w[I.x]) - 1n; w[I.x] = le8(x2); resetReserve(spend, x2, spend.outputs[0].address);
    });
});
await t('refused: taking one unit more out than the quote', () => {
    attack(sim, 'carol', 'sell', 2, ({ w, spend }) => {
        const x2 = LP.readLe8(w[I.x]) - 1n; w[I.x] = le8(x2); resetReserve(spend, x2, spend.outputs[0].address);
    });
});
await t('refused: shrinking the supply to reopen the curve', () => {
    attack(sim, 'carol', 'buy', 1, ({ w }) => {
        const s2 = 19n, x2 = LP.curveReserve(1000, s2);
        w[I.s] = le8(s2); w[I.x] = le8(x2); w[I.y] = le8(10n);
    });
});

await t('everyone sells out: the pool keeps what is left, and never empties', () => {
    for (const who of ['alice', 'bob', 'carol', 'dave']) {
        const slot = sim.launch.ledger.slotOf(LP.ownerOf(keys[who]));
        if (slot >= 0) trade(sim, who, 'sell', sim.launch.ledger.balances[slot]);
    }
    const st = sim.launch.st;
    assert(sim.launch.ledger.total() === 0n && st.y === 30n && st.x > 0n, `x ${st.x}, y ${st.y}`);
});

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n━━━ reading it back from the chain ━━━');

await t('a fresh reader rebuilds the launch from witness-stripped transactions alone', () => {
    let reader = null;
    for (const [i, tx] of sim.txs.entries()) {
        if (!reader) { [reader] = LP.launchesInTx(tx, () => code, i + 1); continue; }
        assert(reader.applyTx(tx, i + 1), `tx ${i} did not apply`);
    }
    assert(reader.state === sim.launch.state, 'state differs');
    assert(reader.ledger.root === sim.launch.ledger.root, 'ledger differs');
    assert(reader.coins.map((c) => c.coinId).join() === sim.launch.coins.map((c) => c.coinId).join(), 'coins differ');
    assert(reader.trades.length === sim.txs.length - 1, 'trade count');
});

await t('a hand-over to a row nobody would guess is still found', () => {
    const s = freshSim();
    trade(s, 'alice', 'buy', 3);
    trade(s, 'bob', 'buy', 3, { slot: 13 });         // an unusual row
    trade(s, 'bob', 'buy', 0, { to: LP.ownerOf(keys.carol) });
    let reader = null;
    for (const tx of s.txs) { if (!reader) [reader] = LP.launchesInTx(tx, () => code); else reader.applyTx(tx); }
    assert(reader.state === s.launch.state && reader.ledger.owners[13] === LP.ownerOf(keys.carol), 'not rebuilt');
});

await t('a transaction that does not touch the launch is ignored', () => {
    const other = { Reveal: { inputs: [{ predicate: { Script: { bytecode: '0101000121' } }, value: 4, salt: '00'.repeat(32) }], witnesses: [], outputs: [], salt: '00'.repeat(32) } };
    assert(sim.launch.applyTx(other) === null);
});

await t('the registry script cannot be spent', () => {
    // PUSH tag DROP PUSH 0 VERIFY: VERIFY of 00 fails whatever the witness.
    const asm = ['PUSH_HEX ' + LP.REGISTRY_BYTECODE.slice(6, 6 + 42), 'DROP', 'PUSH_INT 0', 'VERIFY'];
    for (const witness of [[], ['01'], ['01', '01']]) assert(!execute(asm, { height: H, witness }).ok, 'spent');
    assert(LP.REGISTRY_BYTECODE.length / 2 === 3 + 21 + 1 + 4 + 1, 'registry bytecode layout');
});

if (process.env.WRITE_VECTORS) {
    await fs.mkdir('./test-vectors', { recursive: true });
    await fs.writeFile('./test-vectors/launchpad.json', JSON.stringify({
        about: 'Launchpad spends from test-launchpad.mjs, with real MSS signatures. Replayed by tests/launchpad_vectors.rs.',
        height: Number(H), commitment: COMMITMENT, code, code_split: LP.splitCode(code)[0].length / 2, sigs: Object.values(REAL), vectors,
    }, null, 1) + '\n');
    console.log(`  wrote ${vectors.length} vectors to test-vectors/launchpad.json`);
}

console.log(`\n━━━ ${pass} passed, ${fail} failed ━━━\n`);
process.exit(fail ? 1 : 0);
