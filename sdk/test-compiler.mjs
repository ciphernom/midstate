// test-compiler.mjs — Contract compiler and headless VM.
//
// Two independent ground truths are used, because a compiler validated only
// against its own VM proves nothing:
//
//   - All 17 IDE templates must compile. They are real contracts a human wrote
//     and the IDE accepted, so a language regression shows up here.
//   - The IDE's worked AMM example must EXECUTE to the same verdict the IDE's
//     own annotation claims (accept at Y=51, reject at Y=50). That ties the VM
//     to observed behaviour rather than to my reading of the opcode table.
//
// The curve contract is then verified by execution rather than by inspection:
// exact-cost trades accepted, underpayment and over-refund rejected, and a
// buy/sell round trip returning the reserve exactly.
//
// Run: node test-compiler.mjs

import fs from 'fs/promises';
import initWasm, * as W from './pkg/wasm_wallet.js';
import { compile, MAX_SCRIPT_SIZE, OPS } from './src/compiler.js';
import { execute } from './src/vm.js';
import * as L from './src/launcher.js';

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
const TEST_HEIGHT = 350_000n;

const AMM_ADDR = 'aa000000000000000000000000000000000000000000000000000000000000aa';

console.log('\n━━━ compiler ━━━');

await t('opcode table matches the node (core/script.rs)', () => {
    // A drifted table assembles cleanly, hashes to a plausible address, and
    // fails at execution — after funds are already locked there.
    const NODE = {
        PUSH_DATA: 0x01, DROP: 0x10, DUP: 0x11, SWAP: 0x12, OVER: 0x13, ROT: 0x14,
        SLICE: 0x15, CONCAT: 0x16, PICK: 0x17, EQUAL: 0x20, VERIFY: 0x21,
        EQUALVERIFY: 0x22, ADD: 0x23, GREATER_OR_EQUAL: 0x24, SUB: 0x25, MUL: 0x26,
        DIV: 0x27, MOD: 0x28, SIZE: 0x29, HASH: 0x30, CHECKSIG: 0x31,
        CHECKSIGVERIFY: 0x32, CHECKTIMEVERIFY: 0x33, IF: 0x40, ELSE: 0x41, ENDIF: 0x42,
        SUM_TO_ADDR: 0x50, READ_INPUT_STATE: 0x51, READ_OUTPUT_STATE: 0x52,
        INPUT_VALUE: 0x53, OUTPUT_ADDRESS: 0x54, THIS_ADDRESS: 0x55,
    };
    for (const [name, code] of Object.entries(NODE)) {
        assert(OPS[name] === code, `${name}: compiler has 0x${(OPS[name] ?? 0).toString(16)}, node has 0x${code.toString(16)}`);
    }
});

await t('a trivial contract compiles to the expected instructions', () => {
    const r = compile('assert(input_value() == 100); true;');
    assert(r.asm.join(' ') === 'INPUT_VALUE PUSH_INT 100 EQUAL VERIFY PUSH_INT 1', `asm: ${r.asm.join(' | ')}`);
    assert(r.sizeBytes === 11, `size ${r.sizeBytes}`);
});

await t('compilation is deterministic (same source, same address)', () => {
    const src = 'assert(input_value() >= 5); true;';
    const a = compile(src), b = compile(src);
    assert(a.bytecode === b.bytecode, 'bytecode differs between runs');
    assert(W.blake3_hash_hex(a.bytecode) === W.blake3_hash_hex(b.bytecode), 'address differs');
});

await t('the standard library is available without an import', () => {
    const r = compile('assert(min(3, 7) == 3); assert(max(3, 7) == 7); true;');
    assert(r.sizeBytes > 0, 'no output');
});

await t('a syntax error raises rather than emitting garbage', () => {
    let threw = false;
    try { compile('assert(input_value() ==== ;'); } catch { threw = true; }
    assert(threw, 'accepted malformed source');
});

await t('an oversized script is refused at MAX_SCRIPT_SIZE', () => {
    // Consensus caps a script at 1024 B; finding out here beats finding out
    // when the funding transaction is rejected.
    let threw = false;
    try { compile('assert(input_value() == 1);'.repeat(400) + 'true;'); }
    catch (e) { threw = /MAX_SCRIPT_SIZE/.test(e.message); }
    assert(threw, `did not enforce the ${MAX_SCRIPT_SIZE} B limit`);
});

await t('all 17 IDE templates compile', async () => {
    const html = await fs.readFile('../wasm-wallet/website/ide.html', 'utf8');
    const tpl = {};
    const re = /\n {2}(\w+)\s*:\s*\{\s*\n?\s*code\s*:\s*`(.*?)`,/gs;
    let m;
    while ((m = re.exec(html)) !== null) tpl[m[1]] = m[2];
    assert(Object.keys(tpl).length === 17, `found ${Object.keys(tpl).length} templates, expected 17`);
    for (const [name, code] of Object.entries(tpl)) {
        try { compile(code); } catch (e) { throw new Error(`${name}: ${e.message || e}`); }
    }
});

console.log('\n━━━ VM against the IDE worked example ━━━');

async function ammTemplate() {
    const html = await fs.readFile('../wasm-wallet/website/ide.html', 'utf8');
    const m = /\n {2}amm_uniswap\s*:\s*\{\s*\n?\s*code\s*:\s*`(.*?)`,/s.exec(html);
    return m[1];
}

await t("accepts the IDE's annotated trade (X 10→20, Y 100→51)", async () => {
    const r = compile(await ammTemplate());
    const res = execute(r.asm, {
        inputValue: 10n,
        inputState: L.encodeState(L.LAYOUTS.AMM, { reserveX: 10n, reserveY: 100n }),
        thisAddress: AMM_ADDR, height: TEST_HEIGHT, allowValueBearingState: true,
        outputs: [{ address: AMM_ADDR, value: 20n, state: L.encodeState(L.LAYOUTS.AMM, { reserveX: 20n, reserveY: 51n }) }],
    });
    assert(res.ok, `rejected a trade the IDE says is valid: ${res.error}`);
});

await t('rejects one unit more output than the IDE example', async () => {
    const r = compile(await ammTemplate());
    const res = execute(r.asm, {
        inputValue: 10n,
        inputState: L.encodeState(L.LAYOUTS.AMM, { reserveX: 10n, reserveY: 100n }),
        thisAddress: AMM_ADDR, height: TEST_HEIGHT, allowValueBearingState: true,
        outputs: [{ address: AMM_ADDR, value: 20n, state: L.encodeState(L.LAYOUTS.AMM, { reserveX: 20n, reserveY: 50n }) }],
    });
    assert(!res.ok, 'accepted an over-payout');
});

await t('the SDK quote and the contract agree, across many trades', async () => {
    // The property that matters for a launcher: a quote this SDK produces is
    // one the deployed contract actually accepts.
    const r = compile(await ammTemplate());
    let checked = 0;
    for (const [x, y] of [[10n, 100n], [1000n, 1000n], [50n, 7777n]]) {
        for (const dx of [1n, 3n, 10n, x, x * 2n]) {
            const out = L.getAmountOut(x, y, dx);
            if (out <= 0n) continue;
            const res = execute(r.asm, {
                inputValue: x,
                inputState: L.encodeState(L.LAYOUTS.AMM, { reserveX: x, reserveY: y }),
                thisAddress: AMM_ADDR, height: TEST_HEIGHT, allowValueBearingState: true,
                outputs: [{ address: AMM_ADDR, value: x + dx, state: L.encodeState(L.LAYOUTS.AMM, { reserveX: x + dx, reserveY: y - out }) }],
            });
            assert(res.ok, `contract rejected our quote: x=${x} y=${y} dx=${dx} out=${out} (${res.error})`);
            checked++;
        }
    }
    assert(checked >= 10, `only ${checked} trades exercised`);
});

console.log('\n━━━ VM semantics ━━━');

await t('state reads are refused below the activation height', () => {
    const r = compile('read_input_state(); 0; 8; slice(); 0; add(); 0; add(); true;');
    const res = execute(r.asm, { height: 64_999n, inputState: '00'.repeat(32) });
    assert(!res.ok && /65,?000|activate/i.test(res.error), `got: ${res.error}`);
});

await t('subtraction below zero is an error, not a wrap', () => {
    // Driven at the instruction level: the compiler constant-folds a literal
    // `5 - 10`, so the runtime behaviour has to be exercised directly.
    const res = execute(['PUSH_INT 5', 'PUSH_INT 10', 'SUB'], {});
    assert(!res.ok && /overflow|negative/i.test(res.error), `got: ${res.error}`);
});

await t('the compiler constant-folds a negative literal rather than emitting it', () => {
    let threw = false;
    try { compile('assert(5 - 10 == 0); true;'); } catch { threw = true; }
    assert(threw, 'emitted bytecode for an expression that underflows');
});

await t('math operands wider than 8 bytes are rejected', () => {
    const res = execute(['PUSH_HEX ' + 'ff'.repeat(9), 'PUSH_INT 1', 'ADD'], {});
    assert(!res.ok && /8 bytes/.test(res.error), `got: ${res.error}`);
});

await t('division by zero is caught', () => {
    const res = execute(['PUSH_INT 10', 'PUSH_INT 0', 'DIV'], {});
    assert(!res.ok && /Division by zero/.test(res.error), `got: ${res.error}`);
});

await t('SUM_TO_ADDR totals only matching outputs', () => {
    const A = 'aa'.repeat(32), B = 'bb'.repeat(32);
    const res = execute(['PUSH_HEX ' + A, 'SUM_TO_ADDR'], {
        outputs: [{ address: A, value: 30n }, { address: B, value: 500n }, { address: A, value: 12n }],
    });
    assert(res.ok, `failed: ${res.error}`);
    assert(res.stack[res.stack.length - 1] === '2a', `sum was ${res.stack.at(-1)}, expected 0x2a (42)`);
});

await t('the sigop limit is enforced', () => {
    const res = execute(Array(8).fill(null).flatMap(() => ['PUSH_HEX aa', 'PUSH_HEX aa', 'CHECKSIG', 'DROP']), {});
    assert(!res.ok && /SIGOPS/.test(res.error), `got: ${res.error}`);
});

await t('a runaway script hits the step limit', () => {
    const res = execute(Array(200).fill('PUSH_INT 1'), { maxSteps: 50 });
    assert(!res.ok && /Step limit/.test(res.error), `got: ${res.error}`);
});

await t('an empty final stack is a failure, not a pass', () => {
    assert(execute(['PUSH_INT 1', 'DROP'], {}).ok === false, 'empty stack reported ok');
});

await t('require_signed_by() binds its argument as a local — do not use it with a stack signature', () => {
    // A macro parameter is pushed as a LOCAL, so `require_signed_by(x)` leaves x
    // on the stack ABOVE the signature. CHECKSIGVERIFY then consumes the pubkey
    // twice, compares it against itself, passes vacuously, and the real
    // signature is dropped unverified at macro exit.
    //
    // This is not hypothetical: it silently disarmed the authorisation in the
    // pump contract, which is why that contract emits the two opcodes directly.
    // Any contract combining require_signed_by with a witness-supplied signature
    // has the same latent hole.
    const viaMacro = compile('require_signed_by(pick(1)); true;');
    const direct = compile('pick(1); CHECKSIGVERIFY; true;');

    // Witness: [payload, pk, sig]. Only the direct form should consume the sig.
    const wit = ['07', 'aa'.repeat(32), 'bb'.repeat(32)];
    const m = execute(viaMacro.asm, { witness: wit.slice(), height: TEST_HEIGHT });
    const d = execute(direct.asm, { witness: wit.slice(), height: TEST_HEIGHT });

    assert(!d.ok, 'the direct form accepted a signature that does not match the pubkey');
    assert(m.ok, 'macro form unexpectedly failed — the trap may have been fixed upstream');
});

await t('emitting CHECKSIGVERIFY directly does verify the signature', () => {
    const c = compile('pick(1); CHECKSIGVERIFY; true;');
    const good = execute(c.asm, { witness: ['07', 'aa'.repeat(32), 'aa'.repeat(32)], height: TEST_HEIGHT });
    const bad = execute(c.asm, { witness: ['07', 'aa'.repeat(32), 'bb'.repeat(32)], height: TEST_HEIGHT });
    assert(good.ok, `a matching signature was rejected: ${good.error}`);
    assert(!bad.ok, 'a mismatched signature was accepted');
});

console.log('\n━━━ bonding curve contract (executed) ━━━');

const curve = compile(L.CURVE_CONTRACT_SOURCE);
const CURVE_ADDR = W.blake3_hash_hex(curve.bytecode);
const curveState = (s, r) => L.encodeState(L.CURVE_STATE_LAYOUT, { supply: s, reserve: r });
// Two outputs at the contract address: a zero-value state thread and a separate
// standard coin holding the reserve. Consensus forbids combining them.
const runCurve = (s, r, ns, nr) => execute(curve.asm, {
    inputValue: 0n, sumInputValue: r, inputState: curveState(s, r),
    thisAddress: CURVE_ADDR, height: TEST_HEIGHT,
    outputs: [{ address: CURVE_ADDR, value: 0n, state: curveState(ns, nr) },
              { address: CURVE_ADDR, value: nr }],
});

await t('compiles within the script size limit', () => {
    assert(curve.sizeBytes <= MAX_SCRIPT_SIZE, `${curve.sizeBytes} B`);
    assert(curve.sizeBytes < 400, `${curve.sizeBytes} B is larger than expected`);
});

await t('accepts an exact-cost buy at every supply level', () => {
    for (const [s, n] of [[0n, 1n], [0n, 5n], [5n, 3n], [100n, 10n], [1000n, 1n]]) {
        const tr = L.curveTrade({ supply: s, reserve: L.curveBuyCost(0n, s), amount: n, side: 'buy' });
        const res = runCurve(s, tr.inputValue, tr.newSupply, tr.newReserve);
        assert(res.ok, `s=${s} n=${n} cost=${tr.cost}: ${res.error}`);
    }
});

await t('rejects underpayment by a single unit', () => {
    for (const [s, n] of [[0n, 5n], [5n, 3n], [100n, 10n]]) {
        const r0 = L.curveBuyCost(0n, s);
        const tr = L.curveTrade({ supply: s, reserve: r0, amount: n, side: 'buy' });
        assert(!runCurve(s, r0, tr.newSupply, tr.newReserve - 1n).ok, `s=${s} n=${n}: underpayment accepted`);
    }
});

await t('rejects claiming more supply than was paid for', () => {
    const tr = L.curveTrade({ supply: 5n, reserve: 10n, amount: 3n, side: 'buy' });
    assert(!runCurve(5n, 10n, tr.newSupply + 1n, tr.newReserve).ok, 'free mint accepted');
});

await t('accepts an exact sell', () => {
    for (const [s, n] of [[5n, 3n], [10n, 10n], [100n, 1n]]) {
        const r0 = L.curveBuyCost(0n, s);
        const tr = L.curveTrade({ supply: s, reserve: r0, amount: n, side: 'sell' });
        const res = runCurve(s, r0, tr.newSupply, tr.newReserve);
        assert(res.ok, `s=${s} n=${n} refund=${tr.refund}: ${res.error}`);
    }
});

await t('rejects an over-refund by a single unit', () => {
    const r0 = L.curveBuyCost(0n, 8n);
    const tr = L.curveTrade({ supply: 8n, reserve: r0, amount: 3n, side: 'sell' });
    assert(!runCurve(8n, r0, tr.newSupply, tr.newReserve - 1n).ok, 'over-refund accepted');
});

await t('the reserve cannot be drained by a round trip', () => {
    // Buy then sell the same amount must return the reserve exactly. If the
    // legs were not mirrors, repeated round trips would bleed the curve.
    for (const [s, n] of [[0n, 7n], [5n, 3n], [50n, 20n]]) {
        const r0 = L.curveBuyCost(0n, s);
        const buy = L.curveTrade({ supply: s, reserve: r0, amount: n, side: 'buy' });
        assert(runCurve(s, r0, buy.newSupply, buy.newReserve).ok, 'buy leg rejected');
        const sell = L.curveTrade({ supply: buy.newSupply, reserve: buy.newReserve, amount: n, side: 'sell' });
        assert(runCurve(buy.newSupply, buy.newReserve, sell.newSupply, sell.newReserve).ok, 'sell leg rejected');
        assert(sell.newReserve === r0, `round trip changed the reserve: ${r0} → ${sell.newReserve}`);
        assert(sell.newSupply === s, `round trip changed the supply: ${s} → ${sell.newSupply}`);
    }
});

await t('the state cannot lie about the MDS it holds', () => {
    // input_value() == reserve and sum_to_addr(this) == new_reserve are what
    // tie the claimed reserve to the coin's real value.
    const tr = L.curveTrade({ supply: 5n, reserve: 10n, amount: 3n, side: 'buy' });
    const wrongIn = execute(curve.asm, {
        inputValue: 0n, sumInputValue: 999n, inputState: curveState(5n, 10n),
        thisAddress: CURVE_ADDR, height: TEST_HEIGHT,
        outputs: [{ address: CURVE_ADDR, value: 0n, state: curveState(tr.newSupply, tr.newReserve) },
                  { address: CURVE_ADDR, value: tr.newReserve }],
    });
    assert(!wrongIn.ok, 'accepted a state whose reserve does not match the input coin');
    const wrongOut = execute(curve.asm, {
        inputValue: 0n, sumInputValue: 10n, inputState: curveState(5n, 10n),
        thisAddress: CURVE_ADDR, height: TEST_HEIGHT,
        outputs: [{ address: CURVE_ADDR, value: 0n, state: curveState(tr.newSupply, tr.newReserve) },
                  { address: CURVE_ADDR, value: 1n }],
    });
    assert(!wrongOut.ok, 'accepted an output whose value does not match the new reserve');
});

await t('SDK arithmetic and the contract never disagree (fuzz)', () => {
    let checked = 0;
    for (let s = 0n; s < 40n; s += 3n) {
        const r0 = L.curveBuyCost(0n, s);
        for (const n of [1n, 2n, 7n, 13n]) {
            const buy = L.curveTrade({ supply: s, reserve: r0, amount: n, side: 'buy' });
            assert(runCurve(s, r0, buy.newSupply, buy.newReserve).ok,
                `buy s=${s} n=${n} rejected by the contract`);
            if (n <= s) {
                const sell = L.curveTrade({ supply: s, reserve: r0, amount: n, side: 'sell' });
                assert(runCurve(s, r0, sell.newSupply, sell.newReserve).ok,
                    `sell s=${s} n=${n} rejected by the contract`);
            }
            checked++;
        }
    }
    assert(checked >= 40, `only ${checked} cases`);
});

console.log(`\n━━━ ${pass} passed, ${fail} failed ━━━\n`);
process.exit(fail ? 1 : 0);
