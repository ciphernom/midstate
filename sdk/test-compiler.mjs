// test-compiler.mjs — Contract compiler and headless VM.
//
// A compiler validated only against its own VM proves nothing, so the IDE's
// templates are the outside ground truth: real contracts a human wrote and the
// IDE accepted, all of which must compile. The VM is then held to the node's
// semantics opcode by opcode (core/script.rs).
//
// Run: node test-compiler.mjs

import fs from 'fs/promises';
import initWasm, * as W from './pkg/wasm_wallet.js';
import { compile, MAX_SCRIPT_SIZE, OPS } from './src/compiler.js';
import { execute } from './src/vm.js';

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

await t('all 18 IDE templates compile', async () => {
    const html = await fs.readFile('../wasm-wallet/website/ide.html', 'utf8');
    const tpl = {};
    const re = /\n {2}(\w+)\s*:\s*\{\s*\n?\s*code\s*:\s*`(.*?)`,/gs;
    let m;
    while ((m = re.exec(html)) !== null) tpl[m[1]] = m[2];
    assert(Object.keys(tpl).length === 18, `found ${Object.keys(tpl).length} templates, expected 18`);
    for (const [name, code] of Object.entries(tpl)) {
        try { compile(code); } catch (e) { throw new Error(`${name}: ${e.message || e}`); }
    }
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
    // An opcode probe: it finishes on the sum rather than on 0x01, which is
    // not a valid contract ending, so the consensus end-of-script rule is
    // relaxed for this one assertion.
    const res = execute(['PUSH_HEX ' + A, 'SUM_TO_ADDR'], {
        allowDirtyStack: true,
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

// OP_MERKLE_ROOT, as core/script.rs runs it: [leaf, proof] -> [root].
const V6 = 371_500n;
const LEAF = '11'.repeat(32), S1 = '22'.repeat(32), S2 = '33'.repeat(32);
const merkleProbe = (proof, root, height = V6) =>
    execute(['PUSH_HEX ' + LEAF, 'PUSH_HEX ' + proof, 'MERKLE_ROOT', 'PUSH_HEX ' + root, 'EQUAL'], { height });

await t('MERKLE_ROOT folds dir 00 as hash(acc‖sib) and 01 as hash(sib‖acc)', () => {
    const want = W.blake3_hash_hex(S2 + W.blake3_hash_hex(LEAF + S1));
    const res = merkleProbe(S1 + '00' + S2 + '01', want);
    assert(res.ok, `failed: ${res.error}`);
    // The other order is a different root.
    const swapped = W.blake3_hash_hex(W.blake3_hash_hex(S1 + LEAF) + S2);
    assert(!merkleProbe(S1 + '01' + S2 + '00', want).ok && merkleProbe(S1 + '01' + S2 + '00', swapped).ok,
        'direction bytes were not honoured');
});

await t('MERKLE_ROOT with an empty proof is the leaf itself', () => {
    const res = execute(['PUSH_HEX ' + LEAF, 'PUSH_HEX ', 'MERKLE_ROOT', 'PUSH_HEX ' + LEAF, 'EQUAL'], { height: V6 });
    assert(res.ok, `failed: ${res.error}`);
});

await t('MERKLE_ROOT is refused below the v6 activation height', () => {
    const res = merkleProbe(S1 + '00', W.blake3_hash_hex(LEAF + S1), V6 - 1n);
    assert(!res.ok && /InvalidOpcode/.test(res.error), `got: ${res.error}`);
});

await t('MERKLE_ROOT rejects malformed proofs as the node does', () => {
    const root = W.blake3_hash_hex(LEAF + S1);
    assert(/direction/.test(merkleProbe(S1 + '02', root).error), 'accepted direction byte 02');
    assert(/33-byte/.test(merkleProbe(S1, root).error), 'accepted a partial record');
    const short = execute(['PUSH_HEX 11', 'PUSH_HEX ' + S1 + '00', 'MERKLE_ROOT'], { height: V6, allowDirtyStack: true });
    assert(/32 bytes/.test(short.error), 'accepted a short leaf');
    assert(/exceeds/.test(merkleProbe((S1 + '00').repeat(33), root).error), 'walked 33 levels');
});

// Stack limits, as core/script.rs's stack_push applies them: every push,
// witness items included, may not exceed 1,536 bytes or a depth of 64.
await t('a witness item over 1,536 bytes is refused, and 1,536 is not', () => {
    const big = execute(['DROP', 'PUSH_INT 1'], { witness: ['00'.repeat(1537)] });
    assert(!big.ok && /ItemTooLarge/.test(big.error), `got: ${big.error}`);
    assert(execute(['DROP', 'PUSH_INT 1'], { witness: ['00'.repeat(1536)] }).ok, '1,536 bytes refused');
});

await t('the stack holds 64 items and no more', () => {
    const full = Array(64).fill('01');
    assert(execute(Array(63).fill('DROP'), { witness: full }).ok, '64 witness items refused');
    const over = execute(['PUSH_INT 1'], { witness: full, allowDirtyStack: true });
    assert(!over.ok && /StackOverflow/.test(over.error), `got: ${over.error}`);
    const tuck = execute(['TUCK'], { witness: full, height: V6, allowDirtyStack: true });
    assert(!tuck.ok && /StackOverflow/.test(tuck.error), `TUCK at 64: ${tuck.error}`);
});

await t('live IF nesting stops at 64; a dead branch nests freely', () => {
    const nest = (n, cond) => [...Array(n).fill(null).flatMap(() => [`PUSH_INT ${cond}`, 'IF']), ...Array(n).fill('ENDIF'), 'PUSH_INT 1'];
    assert(execute(nest(64, 1), {}).ok, '64 live IFs refused');
    const deep = execute(nest(65, 1), {});
    assert(!deep.ok && /UnbalancedConditional/.test(deep.error), `got: ${deep.error}`);
    // One false IF, then 70 more inside it: only the first pops, the rest are dead.
    const dead = ['PUSH_INT 0', 'IF', ...Array(70).fill('IF'), ...Array(71).fill('ENDIF'), 'PUSH_INT 1'];
    assert(execute(dead, {}).ok, `dead nesting refused: ${execute(dead, {}).error}`);
});

await t('an empty final stack is a failure, not a pass', () => {
    assert(execute(['PUSH_INT 1', 'DROP'], {}).ok === false, 'empty stack reported ok');
});

await t('require_signed_by() verifies the signature (was: bound its arg as a local)', () => {
    // HISTORY: require_signed_by was a STD_LIB macro whose body re-referenced
    // its own parameter, so the pubkey was pushed twice. CHECKSIGVERIFY then
    // compared the pubkey against itself, passed vacuously, and the real
    // signature was dropped unverified at macro exit. It silently disarmed
    // authorisation, which is why the pump contract emits the two opcodes
    // directly instead of using it.
    //
    // It is now a compiler builtin aliased to require_sig, so it behaves the
    // same as emitting CHECKSIGVERIFY by hand.
    const viaMacro = compile('require_signed_by(pick(1)); true;');
    const direct = compile('pick(1); CHECKSIGVERIFY; true;');

    const good = ['07', 'aa'.repeat(32), 'aa'.repeat(32)];
    const bad = ['07', 'aa'.repeat(32), 'bb'.repeat(32)];

    assert(!execute(viaMacro.asm, { allowDirtyStack: true, witness: bad.slice(), height: TEST_HEIGHT }).ok,
        'require_signed_by accepted a signature that does not match the pubkey');
    assert(execute(viaMacro.asm, { allowDirtyStack: true, witness: good.slice(), height: TEST_HEIGHT }).ok,
        'require_signed_by rejected a matching signature');
    assert(!execute(direct.asm, { allowDirtyStack: true, witness: bad.slice(), height: TEST_HEIGHT }).ok,
        'the direct form accepted a signature that does not match the pubkey');
});

await t('emitting CHECKSIGVERIFY directly does verify the signature', () => {
    const c = compile('pick(1); CHECKSIGVERIFY; true;');
    const good = execute(c.asm, { allowDirtyStack: true, witness: ['07', 'aa'.repeat(32), 'aa'.repeat(32)], height: TEST_HEIGHT });
    const bad = execute(c.asm, { allowDirtyStack: true, witness: ['07', 'aa'.repeat(32), 'bb'.repeat(32)], height: TEST_HEIGHT });
    assert(good.ok, `a matching signature was rejected: ${good.error}`);
    assert(!bad.ok, 'a mismatched signature was accepted');
});

console.log(`\n━━━ ${pass} passed, ${fail} failed ━━━\n`);
process.exit(fail ? 1 : 0);
