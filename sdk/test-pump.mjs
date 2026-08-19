// test-pump.mjs — Bonding-curve launcher with a Merkle holder ledger.
//
// Every assertion here runs the COMPILED CONTRACT in the VM. Nothing is checked
// against the SDK's own arithmetic alone, because the failure that matters is
// the SDK and the contract disagreeing — a quote the contract rejects wastes a
// commit, and a trade the contract wrongly accepts drains the reserve.
//
// The attack cases are the point of the design. Two obvious token designs are
// unsound on this chain (see the reasoning in src/pump.js); the Merkle ledger is
// what makes "sell a balance you never bought" impossible, so that case is
// tested directly rather than argued about.
//
// Run: node test-pump.mjs

import fs from 'fs/promises';
import initWasm, * as W from './pkg/wasm_wallet.js';
import { compile } from './src/compiler.js';
import { execute } from './src/vm.js';
import * as P from './src/pump.js';
import { decodeState as L_decode_raw } from './src/launcher.js';
const L_decode = (hex) => L_decode_raw([['supply', 8, 'u64'], ['root', 24, 'bytes']], hex);
const DEX_MAGIC_SAMPLE = '4d445841' + '00'.repeat(60);

await initWasm({ module_or_path: await fs.readFile('./pkg/wasm_wallet_bg.wasm') });

let pass = 0, fail = 0;
async function t(name, fn) {
    try { await fn(); console.log(`  ✅ ${name}`); pass++; }
    catch (e) { console.log(`  ❌ ${name} — ${e.message}`); fail++; }
}
function assert(c, m) { if (!c) throw new Error(m || 'assertion failed'); }

// Above COVENANT_SUM_ACTIVATION_HEIGHT (300,000), because these contracts use
// sum_input_value(). Simulating below it would pass the arithmetic and hide the
// fact that a real node returns InvalidOpcode for the opcode entirely.
// Deliberately BELOW COVENANT_SUM_ACTIVATION_HEIGHT (300,000) and at roughly
// the live chain height, because the whole point of this variant is that it
// deploys today rather than in 36 days.
const TEST_HEIGHT = 248_041n;

const DEPTH = 4;                       // 16 slots — fast, same code path as 10
const contract = compile(P.pumpContractSource(DEPTH));
const ADDR = W.blake3_hash_hex(contract.bytecode);
// Identities are keypairs now. NOTE: the VM's CHECKSIG compares strings rather
// than verifying a post-quantum signature (see src/vm.js), so a "signature" here
// is just the pubkey echoed back. That is enough to prove the contract demands
// the right key and rejects the wrong one — it proves nothing about the
// cryptography, which only a real node can test.
const key = (seed) => {
    const pk = seed.repeat(32);
    return { pk, sig: pk, owner: P.ownerFromPubkey(pk) };
};
const A = key('a1'), B = key('b2'), M = key('cc');
const ALICE = A.owner, BOB = B.owner, MALLORY = M.owner;

/** A live launch: ledger plus on-chain supply and reserve. */
function launch() {
    const ledger = new P.PumpLedger(DEPTH);
    return { ledger, supply: 0n, reserve: 0n };
}

/** Run a trade through the contract. Returns the VM verdict and the plan. */
function run(state, plan, override = {}) {
    // Two outputs at the contract address: a zero-value state thread and a
    // separate standard coin holding the reserve. Consensus forbids combining
    // them, and the VM now enforces that before running a single opcode.
    const outState = override.newState ?? plan.newState;
    const outValue = override.outputValue ?? plan.newReserve;
    const outputs = override.outputs ?? [
        { address: ADDR, value: 0n, state: outState },
        { address: ADDR, value: outValue },
    ];
    return execute(contract.asm, {
        witness: override.witness ?? plan.witness,
        inputState: plan.currentState,
        inputValue: 0n,                    // the state thread itself carries none
        sumInputValue: plan.sumInputValue, // reserve arrives via the paired coin
        thisAddress: ADDR,
        height: TEST_HEIGHT,
        outputs,
    });
}

/** Trade and, if the contract accepts, advance the local state to match. */
function trade(state, slot, amount, side, who = A, newOwner = null) {
    const plan = P.buildTrade({ ledger: state.ledger, slot, amount, side,
        supply: state.supply, reserve: state.reserve, pubkey: who.pk, sig: who.sig, newOwner });
    const res = run(state, plan);
    if (res.ok) {
        state.ledger.owners[slot] = plan.owner;
        state.ledger.set(slot, plan.newBalance);
        state.supply = plan.newSupply;
        state.reserve = plan.newReserve;
    }
    return { res, plan };
}

console.log('\n━━━ contract ━━━');

await t('the contract source carries no template-literal metacharacters', () => {
    // PUMP_CONTRACT_TEMPLATE is embedded in a JS template literal. A backtick or
    // ${...} in a contract comment terminates it and breaks the whole module at
    // parse time — which has now happened twice while editing this contract.
    assert(!P.PUMP_CONTRACT_TEMPLATE.includes('`'), 'contract source contains a backtick');
    assert(!P.PUMP_CONTRACT_TEMPLATE.includes('${'), 'contract source contains ${');
});


await t('compiles within MAX_SCRIPT_SIZE at every usable depth', () => {
    for (let d = 1; d <= P.MAX_DEPTH; d++) {
        const r = compile(P.pumpContractSource(d));
        assert(r.sizeBytes <= 1024, `depth ${d}: ${r.sizeBytes} B`);
    }
});

await t('depth beyond MAX_DEPTH is refused rather than silently truncated', () => {
    let threw = false;
    try { P.pumpContractSource(P.MAX_DEPTH + 1); } catch { threw = true; }
    assert(threw, 'accepted an unbuildable depth');
});

await t('the address is deterministic for a depth', () => {
    assert(W.blake3_hash_hex(compile(P.pumpContractSource(DEPTH)).bytecode) === ADDR, 'address drifted');
});

console.log('\n━━━ ledger ━━━');

await t('an empty ledger has a well-defined root', () => {
    const l = new P.PumpLedger(DEPTH);
    assert(l.root.length === 64, `root length ${l.root.length}`);
    assert(l.size === 2 ** DEPTH, `size ${l.size}`);
    assert(new P.PumpLedger(DEPTH).root === l.root, 'empty root is not deterministic');
});

await t('proofs have exactly `depth` levels', () => {
    const l = new P.PumpLedger(DEPTH);
    for (const slot of [0, 1, 7, 15]) assert(l.proof(slot).length === DEPTH, `slot ${slot}`);
});

await t('a balance change moves the root', () => {
    const l = new P.PumpLedger(DEPTH);
    const before = l.root;
    l.claim(3, ALICE).set(3, 42n);
    assert(l.root !== before, 'root unchanged after a balance change');
});

await t('leaf encoding matches the VM number format', () => {
    // The contract builds leaves with the VM's minimal little-endian encoding.
    assert(P.minimalLE(0n) === '00', `zero encodes as ${P.minimalLE(0n)}`);
    assert(P.minimalLE(5n) === '05', `5 encodes as ${P.minimalLE(5n)}`);
    assert(P.minimalLE(256n) === '0001', `256 encodes as ${P.minimalLE(256n)}`);
    assert(P.leafHash(0n, ALICE) === W.blake3_hash_hex('00' + ALICE), 'leaf hash formula drifted');
});

console.log('\n━━━ the contract accepts honest trades ━━━');

await t('a first buy from an empty curve', () => {
    const st = launch();
    st.ledger.claim(0, ALICE);
    const { res, plan } = trade(st, 0, 5n, 'buy');
    assert(res.ok, `rejected: ${res.error}`);
    assert(plan.amountMds === 10n, `cost ${plan.amountMds}, expected 0+1+2+3+4 = 10`);
    assert(st.supply === 5n && st.reserve === 10n, `supply ${st.supply} reserve ${st.reserve}`);
});

await t('a second buyer pays the higher price', () => {
    const st = launch();
    st.ledger.claim(0, ALICE); st.ledger.claim(1, BOB);
    trade(st, 0, 5n, 'buy');
    const { res, plan } = trade(st, 1, 3n, 'buy', B);
    assert(res.ok, `rejected: ${res.error}`);
    assert(plan.amountMds === 18n, `cost ${plan.amountMds}, expected 5+6+7 = 18`);
    assert(st.supply === 8n && st.reserve === 28n, `supply ${st.supply} reserve ${st.reserve}`);
});

await t('a holder can sell back', () => {
    const st = launch();
    st.ledger.claim(0, ALICE); st.ledger.claim(1, BOB);
    trade(st, 0, 5n, 'buy'); trade(st, 1, 3n, 'buy', B);
    const { res, plan } = trade(st, 0, 3n, 'sell');
    assert(res.ok, `rejected: ${res.error}`);
    assert(plan.amountMds === 18n, `refund ${plan.amountMds}, expected 7+6+5 = 18`);
    assert(st.supply === 5n && st.reserve === 10n, `supply ${st.supply} reserve ${st.reserve}`);
    assert(st.ledger.balances[0] === 2n, `alice holds ${st.ledger.balances[0]}`);
});

await t('a full round trip is lossless', () => {
    // If the legs were not exact mirrors, repeated round trips would bleed the
    // reserve until honest holders could not exit.
    for (const [pre, n] of [[0n, 7n], [5n, 3n], [12n, 9n]]) {
        const st = launch();
        st.ledger.claim(0, ALICE); st.ledger.claim(1, BOB);
        if (pre > 0n) trade(st, 1, pre, 'buy', B);
        const r0 = st.reserve, s0 = st.supply;
        assert(trade(st, 0, n, 'buy').res.ok, `buy leg rejected (pre=${pre})`);
        assert(trade(st, 0, n, 'sell').res.ok, `sell leg rejected (pre=${pre})`);
        assert(st.reserve === r0, `reserve ${r0} → ${st.reserve}`);
        assert(st.supply === s0, `supply ${s0} → ${st.supply}`);
    }
});

await t('many holders trade independently', () => {
    const st = launch();
    for (let i = 0; i < 8; i++) st.ledger.claim(i, key(i.toString(16).padStart(2, '0')).owner);
    for (let i = 0; i < 8; i++) {
        const ki = key(i.toString(16).padStart(2, '0'));
        const { res } = trade(st, i, BigInt(i + 1), 'buy', ki, ki.owner);
        assert(res.ok, `holder ${i} could not buy`);
    }
    assert(st.supply === 36n, `supply ${st.supply}, expected 1+..+8 = 36`);
    // And each can exit.
    for (let i = 7; i >= 0; i--) {
        const { res } = trade(st, i, BigInt(i + 1), 'sell', key(i.toString(16).padStart(2, '0')));
        assert(res.ok, `holder ${i} could not exit`);
    }
    assert(st.supply === 0n && st.reserve === 0n, `residue: supply ${st.supply} reserve ${st.reserve}`);
});

console.log('\n━━━ the contract rejects attacks ━━━');

await t('cannot sell a balance that was never bought', () => {
    // The core soundness property. Without the Merkle ledger this drains the
    // reserve down to the real supply.
    const st = launch();
    st.ledger.claim(0, ALICE);
    trade(st, 0, 10n, 'buy', A);

    // Mallory asserts slot 5 held 6 units. Two independent defences now stand in
    // the way, and both are worth pinning: an unowned slot has no key that can
    // authorise a spend, and the leaf she claims is not in the tree.
    let sdkRefused = false;
    try {
        P.buildTrade({ ledger: st.ledger, slot: 5, amount: 6n, side: 'sell',
            supply: st.supply, reserve: st.reserve, pubkey: M.pk, sig: M.sig });
    } catch { sdkRefused = true; }
    assert(sdkRefused, 'the SDK built a spend against an unowned slot');

    // Now force it past the SDK, as an attacker would, with a tree that claims
    // she holds 6 — and present it against the REAL on-chain state.
    const forged = new P.PumpLedger(DEPTH, st.ledger.owners);
    forged.balances = st.ledger.balances.slice();
    forged.balances[5] = 6n;
    forged.owners[5] = MALLORY;
    forged._rebuild();
    const plan = P.buildTrade({ ledger: forged, slot: 5, amount: 6n, side: 'sell',
        supply: st.supply, reserve: st.reserve, pubkey: M.pk, sig: M.sig });
    const res = execute(contract.asm, {
        witness: plan.witness, inputState: P.encodePumpState(st.supply, st.ledger.root),
        inputValue: 0n, thisAddress: ADDR, height: TEST_HEIGHT,
        outputs: [{ address: ADDR, value: 0n, state: plan.newState },
                  { address: ADDR, value: Number(plan.newReserve) }],
    });
    assert(!res.ok, 'a fabricated balance was accepted — the reserve is drainable');
});


await t('cannot underpay a buy, even by one unit', () => {
    const st = launch();
    st.ledger.claim(0, ALICE);
    const plan = P.buildTrade({ ledger: st.ledger, slot: 0, amount: 5n, side: 'buy', supply: st.supply, reserve: st.reserve, pubkey: A.pk, sig: A.sig });
    assert(!run(st, plan, { outputValue: plan.newReserve - 1n }).ok, 'underpayment accepted');
});

await t('cannot over-refund a sell, even by one unit', () => {
    const st = launch();
    st.ledger.claim(0, ALICE);
    trade(st, 0, 5n, 'buy');
    const plan = P.buildTrade({ ledger: st.ledger, slot: 0, amount: 3n, side: 'sell', supply: st.supply, reserve: st.reserve, pubkey: A.pk, sig: A.sig });
    assert(!run(st, plan, { outputValue: plan.newReserve - 1n }).ok, 'over-refund accepted');
});

await t('cannot pay for a few and write many into the tree', () => {
    // The contract builds both leaves from (owner, old_balance, n) rather than
    // trusting witness leaves. This is what that binding prevents.
    const st = launch();
    st.ledger.claim(0, ALICE);
    const honest = P.buildTrade({ ledger: st.ledger, slot: 0, amount: 4n, side: 'buy', supply: st.supply, reserve: st.reserve, pubkey: A.pk, sig: A.sig });

    const inflated = new P.PumpLedger(DEPTH, st.ledger.owners);
    inflated.balances = st.ledger.balances.slice();
    inflated.balances[0] = 400n;
    inflated._rebuild();

    const res = run(st, honest, { newState: P.encodePumpState(honest.newSupply, inflated.root) });
    assert(!res.ok, 'free mint accepted — paid for 4, credited 400');
});

await t('cannot inflate supply without paying for it', () => {
    const st = launch();
    st.ledger.claim(0, ALICE);
    const plan = P.buildTrade({ ledger: st.ledger, slot: 0, amount: 5n, side: 'buy', supply: st.supply, reserve: st.reserve, pubkey: A.pk, sig: A.sig });
    const res = run(st, plan, { newState: P.encodePumpState(plan.newSupply + 100n, plan.newRoot) });
    assert(!res.ok, 'supply inflation accepted');
});

await t("cannot spend another holder's slot — the real attack", () => {
    // THE bug this contract exists to close. An earlier version compared the
    // owner field directly, so it was a password; publishing the ledger made
    // every password public and any observer could drain any slot.
    //
    // The earlier version of THIS TEST substituted a WRONG owner, which only
    // ever tested a typo: the leaf then fails the Merkle proof, so it passed
    // while the contract was fully drainable. Mallory here uses Alice's real,
    // published owner and only swaps the keypair.
    //
    // NOTE: the VM's CHECKSIG compares strings rather than verifying a
    // post-quantum signature (see src/vm.js). This proves the contract DEMANDS
    // the committed key and rejects another; it proves nothing about the
    // cryptography, which only a node can test.
    const st = launch();
    st.ledger.claim(0, ALICE);
    trade(st, 0, 5n, 'buy', A);

    const legit = P.buildTrade({ ledger: st.ledger, slot: 0, amount: 5n, side: 'sell',
        supply: st.supply, reserve: st.reserve, pubkey: A.pk, sig: A.sig });
    assert(legit.witness[0] === ALICE, 'fixture: owner_old should be Alice');

    // Substituted at the witness level, exactly as an attacker would — the SDK
    // guard is not the thing under test here.
    const wrongKey = { ...legit, witness: legit.witness.slice() };
    wrongKey.witness[2] = M.pk;
    wrongKey.witness[5] = M.sig;
    assert(!run(st, wrongKey).ok, 'a stranger drained a slot using the published owner');

    // And a forged signature against the correct key.
    const badSig = { ...legit, witness: legit.witness.slice() };
    badSig.witness[5] = M.sig;
    assert(!run(st, badSig).ok, 'the signature was not actually checked');

    // The owner alone is not a credential; the honest path still works.
    assert(run(st, legit).ok, 'the rightful owner was blocked');
});

await t('a funded slot\'s owner is immutable', () => {
    const st = launch();
    st.ledger.claim(0, ALICE);
    trade(st, 0, 5n, 'buy', A);
    let threw = false;
    try {
        P.buildTrade({ ledger: st.ledger, slot: 0, amount: 1n, side: 'sell',
            supply: st.supply, reserve: st.reserve, pubkey: A.pk, sig: A.sig, newOwner: MALLORY });
    } catch (e) { threw = /immutable/.test(e.message); }
    assert(threw, 'a funded slot was transferred to another owner');
});

await t('an emptied slot can be claimed by someone new', () => {
    // Selling out keeps the owner but leaves balance 0, and the claim branch
    // only requires old_bal == 0 — so the slot returns to the pool.
    const st = launch();
    st.ledger.claim(0, ALICE);
    trade(st, 0, 5n, 'buy', A);
    const { res } = trade(st, 0, 5n, 'sell', A);
    assert(res.ok, 'exit failed');
    assert(st.ledger.balances[0] === 0n, 'slot not emptied');
    const { res: res2 } = trade(st, 0, 3n, 'buy', M, MALLORY);
    assert(res2.ok, 'an emptied slot could not be reclaimed');
    assert(st.ledger.owners[0] === MALLORY, 'reclaim did not transfer ownership');
});

await t('a claim needs an owner or the coins would be unspendable', () => {
    const st = launch();
    let threw = false;
    try {
        P.buildTrade({ ledger: st.ledger, slot: 3, amount: 1n, side: 'buy',
            supply: 0n, reserve: 0n, pubkey: A.pk, sig: A.sig, newOwner: P.ZERO_OWNER });
    } catch (e) { threw = /owner/.test(e.message); }
    assert(threw, 'accepted a claim with a zero owner');
});

await t('a trade without a key is refused before any commit', () => {
    const st = launch();
    st.ledger.claim(0, ALICE);
    for (const missing of [{ sig: A.sig }, { pubkey: A.pk }]) {
        let threw = false;
        try {
            P.buildTrade({ ledger: st.ledger, slot: 0, amount: 1n, side: 'buy',
                supply: 0n, reserve: 0n, ...missing });
        } catch { threw = true; }
        assert(threw, 'built a plan with no key material');
    }
});


await t('cannot reuse a stale proof after someone else trades', () => {
    // Every trade moves the root, so a path captured earlier no longer verifies.
    // This is the ledger's concurrency behaviour, and it must fail closed.
    const st = launch();
    st.ledger.claim(0, ALICE); st.ledger.claim(1, BOB);
    const stale = P.buildTrade({ ledger: st.ledger, slot: 0, amount: 3n, side: 'buy', supply: st.supply, reserve: st.reserve, pubkey: A.pk, sig: A.sig });
    trade(st, 1, 4n, 'buy', B);       // Bob moves first; the root changes.
    const res = execute(contract.asm, {
        witness: stale.witness, inputState: P.encodePumpState(st.supply, st.ledger.root),
        inputValue: 0n, sumInputValue: st.reserve, thisAddress: ADDR, height: TEST_HEIGHT,
        outputs: [{ address: ADDR, value: 0n, state: stale.newState }, { address: ADDR, value: st.reserve + P.buyCost(st.supply, 3n) }],
    });
    assert(!res.ok, 'a stale Merkle proof was accepted');
});

await t('cannot divert the reserve to another address', () => {
    const st = launch();
    st.ledger.claim(0, ALICE);
    trade(st, 0, 5n, 'buy');
    const plan = P.buildTrade({ ledger: st.ledger, slot: 0, amount: 2n, side: 'sell', supply: st.supply, reserve: st.reserve, pubkey: A.pk, sig: A.sig });
    // Pay the remaining reserve somewhere that is not the contract.
    const res = execute(contract.asm, {
        witness: plan.witness, inputState: plan.currentState, inputValue: 0n, sumInputValue: plan.sumInputValue,
        thisAddress: ADDR, height: TEST_HEIGHT,
        outputs: [{ address: 'de'.repeat(32), value: 0n, state: plan.newState }, { address: ADDR, value: plan.newReserve }],
    });
    assert(!res.ok, 'the reserve was diverted to a foreign address');
});

await t('an unknown route is rejected', () => {
    const st = launch();
    st.ledger.claim(0, ALICE);
    const plan = P.buildTrade({ ledger: st.ledger, slot: 0, amount: 5n, side: 'buy', supply: st.supply, reserve: st.reserve, pubkey: A.pk, sig: A.sig });
    const w = plan.witness.slice();
    w[w.length - 1] = '07';
    assert(!run(st, plan, { witness: w }).ok, 'an undefined route was accepted');
});

await t('a state thread carrying value is rejected before any opcode runs', () => {
    // The bug this test exists for: an earlier version stored the reserve in the
    // state-carrying coin. It executed perfectly in the script VM and would have
    // been rejected by every node, because OutputData::Confidential has no value
    // field — "To send value AND state, create two outputs".
    const st = launch();
    st.ledger.claim(0, ALICE);
    const plan = P.buildTrade({ ledger: st.ledger, slot: 0, amount: 5n, side: 'buy', supply: st.supply, reserve: st.reserve, pubkey: A.pk, sig: A.sig });
    const res = run(st, plan, {
        outputs: [{ address: ADDR, value: plan.newReserve, state: plan.newState }],
    });
    assert(!res.ok, 'a value-bearing state thread was accepted');
    assert(/state threads must have value 0/.test(res.error), `wrong diagnosis: ${res.error}`);
});

await t('buildTrade emits a zero-value state thread and a separate reserve coin', () => {
    const st = launch();
    st.ledger.claim(0, ALICE);
    const plan = P.buildTrade({ ledger: st.ledger, slot: 0, amount: 5n, side: 'buy', supply: st.supply, reserve: st.reserve, pubkey: A.pk, sig: A.sig });
    assert(plan.outputs.length === 2, `${plan.outputs.length} outputs`);
    const [thread, reserve] = plan.outputs;
    assert(thread.kind === 'confidential' && thread.value === 0n, 'state thread is not zero-value');
    assert(thread.state === plan.newState, 'state thread does not carry the new state');
    assert(reserve.kind === 'standard' && reserve.state === undefined, 'reserve output carries state');
    assert(reserve.value === plan.newReserve, `reserve output holds ${reserve.value}`);
});

await t('the money rule is sum_to_addr == impliedReserve(new_supply)', () => {
    // The reserve is never read from an input. The only money assertion is that
    // what is left at this address equals the reserve the new supply implies —
    // so a buy must fund the difference and a sell may withdraw exactly it.
    const st = launch();
    st.ledger.claim(0, ALICE);
    trade(st, 0, 5n, 'buy');
    assert(st.reserve === P.impliedReserve(st.supply),
        `reserve ${st.reserve} != impliedReserve(${st.supply}) = ${P.impliedReserve(st.supply)}`);

    const plan = P.buildTrade({ ledger: st.ledger, slot: 0, amount: 2n, side: 'sell', supply: st.supply, reserve: st.reserve, pubkey: A.pk, sig: A.sig });
    // Withdraw one more than the new supply implies.
    const greedy = run(st, plan, {
        outputs: [{ address: ADDR, value: 0n, state: plan.newState },
                  { address: ADDR, value: P.impliedReserve(plan.newSupply) - 1n }],
    });
    assert(!greedy.ok, 'withdrew more than the new supply implies');
    // And leaving too much is equally rejected — the rule is equality.
    const stingy = run(st, plan, {
        outputs: [{ address: ADDR, value: 0n, state: plan.newState },
                  { address: ADDR, value: P.impliedReserve(plan.newSupply) + 1n }],
    });
    assert(!stingy.ok, 'accepted a reserve above what the supply implies');
});


await t('a state thread contributes ZERO value, not an unknown value', () => {
    // The `Confidential` name is vestigial — it predates the removal of zk-STARK
    // confidential transactions, and state threads were retconned into the same
    // variant. Its value is known to be zero, not hidden. Reading it as
    // "unknown" is the mirror-image bug to the one this module already fixed: it
    // would make a wallet skip these outputs from conservation rather than count
    // them as zero.
    const st = launch();
    st.ledger.claim(0, ALICE);
    const plan = P.buildTrade({ ledger: st.ledger, slot: 0, amount: 5n, side: 'buy', supply: st.supply, reserve: st.reserve, pubkey: A.pk, sig: A.sig });
    const [thread] = plan.outputs;
    assert(thread.value === 0n, `state thread value is ${thread.value}, must be exactly 0n`);
    assert(typeof thread.value === 'bigint', 'value is modelled as unknown rather than zero');

    // The contract agrees: sum_to_addr counts the thread as 0, so the paired
    // standard coin alone must cover the whole new reserve.
    const short = run(st, plan, {
        outputs: [{ address: ADDR, value: 0n, state: plan.newState },
                  { address: ADDR, value: plan.newReserve - 1n }],
    });
    assert(!short.ok, 'the state thread was credited with value it does not carry');
});

await t('the contract uses no consensus-gated opcode', () => {
    // sum_input_value() activates at 300,000. Depending on it would make the
    // curve undeployable until then, with its reserve stranded at an address
    // nobody could spend from.
    assert(!contract.asm.some((i) => /SUM_INPUT_VALUE/.test(i)),
        'the contract depends on SUM_INPUT_VALUE and cannot deploy below height 300,000');
});

await t('the reserve is implied by supply, never stored or read', () => {
    for (const s of [0n, 1n, 5n, 8n, 12n, 100n]) {
        let series = 0n;
        for (let i = 0n; i < s; i++) series += i;
        assert(P.impliedReserve(s) === series,
            `impliedReserve(${s}) = ${P.impliedReserve(s)}, series 0..${s - 1n} = ${series}`);
        if (s > 0n) assert(P.impliedReserve(s) === P.buyCost(0n, s), `disagrees with buyCost at ${s}`);
    }
    // And it must not underflow at zero supply.
    assert(P.impliedReserve(0n) === 0n, 'implied reserve underflows at supply 0');
});



console.log('\n━━━ transaction layer ━━━');

await t('deploy starts at zero reserve (the invariant is inductive)', () => {
    // r = impliedReserve(s) holds only if the launch begins empty. Seeding the
    // address locks its funds behind a money assertion that can never hold again.
    const d = P.deployPlan({ depth: DEPTH, saltHex: 'ab'.repeat(32), compileFn: compile });
    assert(d.outputs.length === 1, `deploy emitted ${d.outputs.length} outputs; only the state thread should exist`);
    assert(d.outputs[0].out_type === 'confidential' && d.outputs[0].value === 0, 'state thread is not zero-value');
    assert(d.address === ADDR, 'deploy address differs from the compiled contract address');
    const st = L_decode(d.initialState);
    assert(st.supply === 0n, `initial supply ${st.supply}`);
});

await t('deploy fixes the state salt so the coin can be spent later', () => {
    const d = P.deployPlan({ depth: DEPTH, saltHex: 'cd'.repeat(32), compileFn: compile });
    assert(d.outputs[0].salt === 'cd'.repeat(32), 'salt not carried into the output');
    let threw = false;
    try { P.deployPlan({ depth: DEPTH, compileFn: compile }); } catch { threw = true; }
    assert(threw, 'accepted a deploy with no explicit salt — the coin would be unspendable');
});

await t('trade outputs are ordered as the contract reads them', () => {
    // read_output_state(0) and output_address(0) == this_address() make the
    // order consensus-relevant: state thread first, reserve second.
    const st = launch();
    st.ledger.claim(0, ALICE);
    const plan = P.buildTrade({ ledger: st.ledger, slot: 0, amount: 5n, side: 'buy', supply: st.supply, reserve: st.reserve, pubkey: A.pk, sig: A.sig });
    const tx = P.tradeTx({ plan, address: ADDR, stateSalt: '11'.repeat(32), reserveSalt: '22'.repeat(32),
                           current: { stateCoinId: 'aa'.repeat(32), stateSalt: 'bb'.repeat(32) } });
    assert(tx.outputs[0].out_type === 'confidential', 'output 0 is not the state thread');
    assert(tx.outputs[0].address === ADDR, 'state thread is not at the contract address');
    assert(tx.outputs[1].out_type === 'standard', 'output 1 is not the reserve coin');
    assert(tx.outputs[1].value === Number(plan.newReserve), 'reserve output value wrong');
    assert(tx.outputs.every((o) => o.salt), 'an output has no explicit salt and would be unspendable');
});

await t('a first buy spends only the state thread (no reserve coin exists yet)', () => {
    const st = launch();
    st.ledger.claim(0, ALICE);
    const plan = P.buildTrade({ ledger: st.ledger, slot: 0, amount: 5n, side: 'buy', supply: 0n, reserve: 0n, pubkey: A.pk, sig: A.sig });
    const tx = P.tradeTx({ plan, address: ADDR, stateSalt: '11'.repeat(32), reserveSalt: '22'.repeat(32),
                           current: { stateCoinId: 'aa'.repeat(32), stateSalt: 'bb'.repeat(32) } });
    assert(tx.contractInputs.length === 1, `spent ${tx.contractInputs.length} inputs; there is no reserve coin yet`);
    assert(tx.contractInputs[0].value === 0, 'state thread input is not zero-value');
});

console.log('\n━━━ on-chain ledger publication ━━━');

await t('a ledger update fits one burn', () => {
    const hex = P.encodeLedgerUpdate({ contractAddr: ADDR, slot: 7, balance: 12345n, owner: ALICE });
    assert(hex.length / 2 === P.LEDGER_BURN_BYTES, `${hex.length / 2} bytes`);
    assert(hex.length / 2 <= 80, 'exceeds MAX_BURN_DATA_SIZE — would need fragmenting');
});

await t('a ledger update round-trips', () => {
    const back = P.decodeLedgerUpdate(P.encodeLedgerUpdate({ contractAddr: ADDR, slot: 300, balance: 1n << 40n, owner: BOB }));
    assert(back !== null, 'decode returned null');
    assert(back.slot === 300, `slot ${back.slot}`);
    assert(back.balance === (1n << 40n), `balance ${back.balance}`);
    assert(back.owner === BOB, 'owner lost');
    assert(back.contractId === P.contractShortId(ADDR), 'contract id lost');
});

await t('foreign burns decode to null', () => {
    assert(P.decodeLedgerUpdate('deadbeef') === null, 'foreign magic accepted');
    assert(P.decodeLedgerUpdate('') === null, 'empty accepted');
    assert(P.decodeLedgerUpdate(DEX_MAGIC_SAMPLE) === null, 'a DEX announcement decoded as a ledger update');
});

await t('replaying published updates rebuilds the exact tree', () => {
    // The property the whole launcher rests on: anyone with the chain can
    // reconstruct the tree, so a holder never depends on a service for a proof.
    const st = launch();
    const updates = [];
    for (let i = 0; i < 5; i++) st.ledger.claim(i, key(i.toString(16).padStart(2, '0')).owner);
    for (const [slot, amt, side] of [[0, 5n, 'buy'], [1, 3n, 'buy'], [2, 7n, 'buy'], [0, 2n, 'sell'], [3, 4n, 'buy']]) {
        const { res, plan } = trade(st, slot, amt, side, key(slot.toString(16).padStart(2, '0')));
        assert(res.ok, `trade rejected: ${res.error}`);
        updates.push(P.decodeLedgerUpdate(
            P.encodeLedgerUpdate({ contractAddr: ADDR, slot: plan.slot, balance: plan.newBalance, owner: plan.owner })));
    }

    const rebuilt = P.replayLedger({ depth: DEPTH, contractAddr: ADDR, updates, expectedRoot: st.ledger.root });
    assert(rebuilt.rootMatches, 'the replayed root does not match the on-chain root');
    assert(rebuilt.supply === st.supply, `replayed supply ${rebuilt.supply} != ${st.supply}`);
    assert(rebuilt.reserve === st.reserve, `replayed reserve ${rebuilt.reserve} != ${st.reserve}`);
    for (let i = 0; i < 5; i++) {
        assert(rebuilt.ledger.balances[i] === st.ledger.balances[i], `slot ${i} balance differs`);
    }
});

await t('a proof built from a replayed tree is accepted by the contract', () => {
    // End to end: rebuild from burns alone, then trade against the live state.
    const st = launch();
    const updates = [];
    st.ledger.claim(0, ALICE); st.ledger.claim(1, BOB);
    for (const [slot, amt] of [[0, 5n], [1, 4n]]) {
        const { plan } = trade(st, slot, amt, 'buy', slot === 0 ? A : B);
        updates.push(P.decodeLedgerUpdate(
            P.encodeLedgerUpdate({ contractAddr: ADDR, slot: plan.slot, balance: plan.newBalance, owner: plan.owner })));
    }
    const { ledger: rebuilt } = P.replayLedger({ depth: DEPTH, contractAddr: ADDR, updates });
    const plan = P.buildTrade({ ledger: rebuilt, slot: 0, amount: 2n, side: 'sell', supply: st.supply, reserve: st.reserve, pubkey: A.pk, sig: A.sig });
    const res = run(st, plan);
    assert(res.ok, `a proof from the replayed tree was rejected: ${res.error}`);
});

await t('an incomplete replay is reported, not absorbed', () => {
    const st = launch();
    const updates = [];
    st.ledger.claim(0, ALICE); st.ledger.claim(1, BOB);
    for (const [slot, amt] of [[0, 5n], [1, 4n]]) {
        const { plan } = trade(st, slot, amt, 'buy', slot === 0 ? A : B);
        updates.push(P.decodeLedgerUpdate(
            P.encodeLedgerUpdate({ contractAddr: ADDR, slot: plan.slot, balance: plan.newBalance, owner: plan.owner })));
    }
    updates.pop();   // a missed block
    const r = P.replayLedger({ depth: DEPTH, contractAddr: ADDR, updates, expectedRoot: st.ledger.root });
    assert(r.rootMatches === false, 'a missing update was silently absorbed — proofs would be rejected on chain');
});

await t("another launch's burns are ignored", () => {
    const st = launch();
    st.ledger.claim(0, ALICE);
    const { plan } = trade(st, 0, 5n, 'buy');
    const mine = P.decodeLedgerUpdate(P.encodeLedgerUpdate({ contractAddr: ADDR, slot: plan.slot, balance: plan.newBalance, owner: plan.owner }));
    const theirs = P.decodeLedgerUpdate(P.encodeLedgerUpdate({ contractAddr: 'ff'.repeat(32), slot: 0, balance: 999n, owner: BOB }));
    const r = P.replayLedger({ depth: DEPTH, contractAddr: ADDR, updates: [mine, theirs], expectedRoot: st.ledger.root });
    assert(r.applied === 1, `applied ${r.applied} updates; a foreign launch leaked in`);
    assert(r.rootMatches, 'foreign burn corrupted the tree');
});

console.log('\n━━━ SDK and contract agree (fuzz) ━━━');

await t('every plan the SDK produces is one the contract accepts', () => {
    let checked = 0;
    for (const seed of [1, 2, 3]) {
        const st = launch();
        for (let i = 0; i < 6; i++) st.ledger.claim(i, key(i.toString(16).padStart(2, '0')).owner);
        for (let step = 0; step < 12; step++) {
            const slot = (step * seed) % 6;
            const held = st.ledger.balances[slot];
            const side = (held > 0n && step % 3 === 2) ? 'sell' : 'buy';
            const n = side === 'sell' ? (held > 2n ? 2n : held) : BigInt((step % 4) + 1);
            if (n <= 0n) continue;
            const { res, plan } = trade(st, slot, n, side, key(slot.toString(16).padStart(2, '0')));
            assert(res.ok, `seed ${seed} step ${step}: ${side} ${n} rejected — ${res.error}`);
            assert(plan.newReserve >= 0n, 'negative reserve');
            checked++;
        }
        // Whatever the path, the reserve must equal the cost of the supply.
        assert(st.reserve === P.buyCost(0n, st.supply),
            `reserve ${st.reserve} does not match supply ${st.supply} (expected ${P.buyCost(0n, st.supply)})`);
    }
    assert(checked >= 25, `only ${checked} trades exercised`);
});

console.log(`\n━━━ ${pass} passed, ${fail} failed ━━━\n`);
process.exit(fail ? 1 : 0);
