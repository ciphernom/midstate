// vm.js — Headless Midstate script VM.
//
// # Reasoning
//
// Ported from the IDE's step emulator with the DOM lookups replaced by an
// explicit context, so a contract can be executed from code instead of by hand.
//
// This exists because a launcher funds a *fresh* contract address for every
// token, and a contract that is wrong locks its funds at an address nobody can
// spend from. There is no second chance and no upgrade path. Being able to run a
// candidate transaction against the script before paying for a commit turns that
// class of mistake from "funds gone" into "assertion failed on line 12".
//
// # Fidelity, and where it deliberately stops
//
// The arithmetic, stack, slicing, state reads and value sums are faithful to
// `core/script.rs`. Two things are emulator conveniences inherited from the IDE
// and MUST NOT be relied on as security properties:
//
//   - `CHECKSIG` / `CHECKSIGVERIFY` compare strings. The real VM verifies a WOTS
//     or MSS signature. A contract that "passes" here on signature checks has
//     proved nothing about signatures.
//   - `HASH` uses BLAKE3, which is correct, but a non-hex operand is UTF-8
//     encoded first, matching the IDE rather than any on-chain behaviour.
//
// So: use this to prove *arithmetic and state* correctness, which is where
// bonding-curve and AMM bugs actually live. Do not use it to prove authorisation.

import { blake3_hash_hex } from '../pkg/wasm_wallet.js';

/** State threads activate at this height; reads below it are rejected. */
export const STATE_THREAD_ACTIVATION_HEIGHT = 65_000n;
/** The VM rejects a script that performs more than this many signature ops. */
export const MAX_SIGOPS_PER_SCRIPT = 3;

/**
 * `OP_SUM_INPUT_VALUE` is consensus-gated. Below this height a node returns
 * `InvalidOpcode`, so any contract using it is unspendable until activation.
 *
 * This is NOT the same as the state-thread activation height (65,000) — an easy
 * conflation, and one that makes a contract simulate cleanly at a height where
 * the real VM would refuse the opcode outright.
 */
export const COVENANT_SUM_ACTIVATION_HEIGHT = 300_000n;

const intToHexLE = (n) => {
    let v = typeof n === 'bigint' ? n : BigInt(n);
    if (v === 0n) return '00';
    let h = v.toString(16);
    if (h.length % 2) h = '0' + h;
    return h.match(/.{2}/g).reverse().join('');
};

const hexLEToInt = (h) => {
    if (!h) return 0n;
    // `to_u64` in the VM refuses operands wider than 8 bytes.
    if (h.length > 16) throw new Error('InvalidOpcode: math operands are limited to 8 bytes');
    return BigInt('0x' + h.match(/.{2}/g).reverse().join(''));
};

const isTrue = (h) => !!h && h.match(/.{2}/g).some((b) => parseInt(b, 16) !== 0);

/**
 * Execute a compiled contract.
 *
 * @param {Array} asm            Instruction list from `compile()`.
 * @param {Object} ctx
 * @param {string[]} [ctx.witness=[]]      Witness stack, pushed left→right.
 * @param {string}  [ctx.inputState]       64-hex state of the consumed thread.
 * @param {bigint}  [ctx.inputValue=0n]    Value of the coin being spent.
 * @param {bigint}  [ctx.sumInputValue]    Total value of all inputs sharing this
 *   predicate. Defaults to `inputValue` (the single-input case).
 * @param {boolean} [ctx.allowValueBearingState=false] Permit one output to carry
 *   BOTH a state commitment and value. Consensus forbids this; the flag exists
 *   only to reproduce the IDE emulator, whose `address:value:state` context
 *   format allows it. Never enable it to validate a contract you intend to
 *   deploy — that is precisely the mistake it is here to document.
 * @param {string}  [ctx.thisAddress]      64-hex address of this contract.
 * @param {bigint}  [ctx.height]           Block height, for timelocks.
 * @param {Array}   [ctx.outputs=[]]       `{address, value, state}` per output.
 * @param {number}  [ctx.maxSteps=100000]  Guard against a runaway script.
 * @returns {{ok:boolean, stack:string[], steps:number, error:string|null, trace:Array}}
 */
export function execute(asm, ctx = {}) {
    const {
        witness = [], inputState = null, inputValue = 0n,
        thisAddress = null, height = 100_000n, outputs = [], sumInputValue = null,
        allowValueBearingState = false,
        maxSteps = 100_000, trace: wantTrace = false,
    } = ctx;

    const stack = witness.map((w) => String(w).replace(/^0x/, '').toLowerCase());
    const execStack = [];
    const trace = [];
    let sigops = 0, steps = 0;

    const push = (v) => stack.push(String(v).toLowerCase());
    const pop = () => {
        if (!stack.length) throw new Error('Stack underflow');
        return stack.pop();
    };
    const requireState = () => {
        if (BigInt(height) < STATE_THREAD_ACTIVATION_HEIGHT) {
            throw new Error(`InvalidOpcode: state threads activate at height ${STATE_THREAD_ACTIVATION_HEIGHT}`);
        }
    };
    const outAt = (i) => {
        if (i < 0 || i >= outputs.length) throw new Error(`InvalidStateRead: output index ${i} out of bounds`);
        return outputs[i];
    };

    // ── Consensus precondition the script VM itself does not enforce ──
    //
    // `OutputData::Confidential` has NO value field: a state thread carries state
    // and nothing else, and `apply_transaction` rejects any attempt to give one a
    // value ("To send value AND state, create two outputs"). That rule lives in
    // transaction application, not in the script VM, so a contract that assumes a
    // state-carrying output can also hold a reserve executes perfectly here and is
    // rejected on chain. Checking it up front is the whole point of simulating.
    //
    // The variant name is vestigial and misleading. It predates the removal of
    // zk-STARK confidential transactions; state threads were retconned into the
    // same variant. Its value is **zero and known**, not hidden — the node's own
    // source calls the name "a trap" and carries a regression test for exactly
    // the wrong reading. Anyone tempted to treat these outputs as carrying an
    // unknown amount should read `value(Confidential{..}) = 0` in core/types.rs
    // first.
    for (let i = 0; i < outputs.length && !allowValueBearingState; i++) {
        const o = outputs[i];
        if (o && o.state && BigInt(o.value ?? 0) !== 0n) {
            return {
                ok: false, stack: [], steps: 0, trace: [],
                error: `output ${i} carries both a state commitment and value ${o.value}; ` +
                       'state threads must have value 0 — use a separate standard output for value',
            };
        }
    }

    try {
        for (let pc = 0; pc < asm.length; pc++) {
            if (++steps > maxSteps) throw new Error(`Step limit (${maxSteps}) exceeded`);
            const raw = asm[pc];
            const sp = String(raw).split(' ');
            const op = sp[0];
            const arg = sp.length > 1 ? sp.slice(1).join(' ') : undefined;
            const live = execStack.every((e) => e === true);

            if (wantTrace) trace.push({ pc, op, arg, live, stack: stack.slice() });

            // Branch bookkeeping runs even in a dead branch, or nesting breaks.
            if (op === 'IF') {
                execStack.push(live ? isTrue(pop()) : false);
                continue;
            }
            if (op === 'ELSE') {
                const parentLive = execStack.length <= 1 || execStack.slice(0, -1).every((e) => e === true);
                if (parentLive) execStack[execStack.length - 1] = !execStack[execStack.length - 1];
                continue;
            }
            if (op === 'ENDIF') { execStack.pop(); continue; }
            if (!live) continue;

            if (op === 'CHECKSIG' || op === 'CHECKSIGVERIFY') {
                if (++sigops > MAX_SIGOPS_PER_SCRIPT) {
                    throw new Error('VerifyFailed: MAX_SIGOPS_PER_SCRIPT exceeded');
                }
            }

            switch (op) {
                case 'PUSH_HEX': push(arg); break;
                case 'PUSH_INT': push(intToHexLE(BigInt(arg))); break;
                case 'DROP': pop(); break;
                case 'DUP': { const a = pop(); push(a); push(a); break; }
                case 'SWAP': { const a = pop(), b = pop(); push(a); push(b); break; }
                case 'OVER': {
                    if (stack.length < 2) throw new Error('Stack underflow (OVER)');
                    push(stack[stack.length - 2]); break;
                }
                case 'ROT': {
                    if (stack.length < 3) throw new Error('Stack underflow (ROT)');
                    push(stack.splice(stack.length - 3, 1)[0]); break;
                }
                case 'PICK': {
                    const n = Number(hexLEToInt(pop()));
                    if (n < 0 || n >= stack.length) throw new Error('Stack underflow (PICK)');
                    push(stack[stack.length - 1 - n]); break;
                }
                case 'SIZE': { const a = pop(); push(a); push(intToHexLE(BigInt(a.length / 2))); break; }
                case 'SLICE': {
                    const len = Number(hexLEToInt(pop()));
                    const off = Number(hexLEToInt(pop()));
                    const a = pop();
                    if (off * 2 + len * 2 > a.length) throw new Error('SLICE out of bounds');
                    push(a.slice(off * 2, off * 2 + len * 2)); break;
                }
                case 'CONCAT': { const b = pop(), a = pop(); push(a + b); break; }
                case 'EQUAL': { const b = pop(), a = pop(); push(a === b ? '01' : '00'); break; }
                case 'VERIFY': if (!isTrue(pop())) throw new Error('VERIFY failed'); break;
                case 'EQUALVERIFY': {
                    const b = pop(), a = pop();
                    if (a !== b) throw new Error(`EQUALVERIFY failed (${a} != ${b})`);
                    break;
                }
                case 'ADD': { const b = pop(), a = pop(); push(intToHexLE(hexLEToInt(a) + hexLEToInt(b))); break; }
                case 'SUB': {
                    const b = pop(), a = pop();
                    const d = hexLEToInt(a) - hexLEToInt(b);
                    // Values are unsigned on this VM; a negative result is a bug
                    // in the contract, not a wrapped number.
                    if (d < 0n) throw new Error('Math overflow (SUB went negative)');
                    push(intToHexLE(d)); break;
                }
                case 'MUL': { const b = pop(), a = pop(); push(intToHexLE(hexLEToInt(a) * hexLEToInt(b))); break; }
                case 'DIV': {
                    const b = pop(), a = pop(); const bv = hexLEToInt(b);
                    if (bv === 0n) throw new Error('Division by zero');
                    push(intToHexLE(hexLEToInt(a) / bv)); break;
                }
                case 'MOD': {
                    const b = pop(), a = pop(); const bv = hexLEToInt(b);
                    if (bv === 0n) throw new Error('Modulo by zero');
                    push(intToHexLE(hexLEToInt(a) % bv)); break;
                }
                case 'GREATER_OR_EQUAL': {
                    const b = pop(), a = pop();
                    push(hexLEToInt(a) >= hexLEToInt(b) ? '01' : '00'); break;
                }
                case 'HASH': {
                    const a = pop();
                    const hex = (/^[0-9a-f]*$/.test(a) && a.length % 2 === 0)
                        ? a
                        : Array.from(new TextEncoder().encode(a)).map((b) => b.toString(16).padStart(2, '0')).join('');
                    push(blake3_hash_hex(hex)); break;
                }
                // Emulator bypass — see the module header. Proves nothing.
                case 'CHECKSIG': { const pk = pop(), sig = pop(); push(pk === sig ? '01' : '00'); break; }
                case 'CHECKSIGVERIFY': {
                    const pk = pop(), sig = pop();
                    if (pk !== sig) throw new Error('CHECKSIGVERIFY failed');
                    break;
                }
                case 'CHECKTIMEVERIFY': {
                    const r = hexLEToInt(pop());
                    if (BigInt(height) < r) throw new Error(`CHECKTIMEVERIFY failed (height ${height} < ${r})`);
                    break;
                }
                case 'INPUT_VALUE': push(intToHexLE(BigInt(inputValue))); break;
                case 'SUM_INPUT_VALUE': {
                    if (BigInt(height) < COVENANT_SUM_ACTIVATION_HEIGHT) {
                        throw new Error('InvalidOpcode: OP_SUM_INPUT_VALUE is not active at this height');
                    }
                    push(intToHexLE(BigInt(sumInputValue ?? inputValue))); break;
                }
                case 'THIS_ADDRESS': {
                    if (!thisAddress || thisAddress.length !== 64) {
                        throw new Error('InvalidStateRead: this_address must be 32 bytes');
                    }
                    push(thisAddress); break;
                }
                case 'OUTPUT_ADDRESS': {
                    const o = outAt(Number(hexLEToInt(pop())));
                    const addr = String(o.address || '').toLowerCase();
                    if (addr.length !== 64) throw new Error('InvalidStateRead: output address must be 32 bytes');
                    push(addr); break;
                }
                case 'SUM_TO_ADDR': {
                    const addr = pop();
                    if (addr.length !== 64) throw new Error('VerifyFailed: SUM_TO_ADDR requires a 32-byte address');
                    let sum = 0n;
                    for (const o of outputs) {
                        if (String(o.address || '').toLowerCase() === addr) sum += BigInt(o.value ?? 0);
                    }
                    push(intToHexLE(sum)); break;
                }
                case 'READ_INPUT_STATE': {
                    requireState();
                    if (!inputState) throw new Error('InvalidStateRead: no input state provided');
                    if (inputState.length !== 64) throw new Error('InvalidStateRead: state must be 32 bytes');
                    push(inputState.toLowerCase()); break;
                }
                case 'READ_OUTPUT_STATE': {
                    requireState();
                    const o = outAt(Number(hexLEToInt(pop())));
                    const s = String(o.state || '');
                    if (s.length !== 64) throw new Error('InvalidStateRead: output state must be 32 bytes');
                    push(s.toLowerCase()); break;
                }
                default: throw new Error(`Unknown opcode: ${op}`);
            }
        }

        // A script succeeds on a truthy top of stack, as the node requires.
        const top = stack.length ? stack[stack.length - 1] : null;
        return {
            ok: top !== null && isTrue(top),
            stack: stack.slice(),
            steps,
            error: top === null ? 'script ended with an empty stack' : null,
            trace,
        };
    } catch (e) {
        return { ok: false, stack: stack.slice(), steps, error: e.message, trace };
    }
}

/**
 * Compile and execute in one call.
 *
 * The convenience form for "does this transaction satisfy this contract",
 * which is the question a launcher needs answered before it funds anything.
 */
export function simulate(compileFn, source, ctx) {
    const compiled = compileFn(source);
    const result = execute(compiled.asm, { ...ctx, thisAddress: ctx.thisAddress ?? blake3_hash_hex(compiled.bytecode) });
    return { ...result, compiled };
}
