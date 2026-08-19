// launcher.js — Token, AMM and bonding-curve state for Midstate contracts.
//
// ═══════════════════════════════════════════════════════════════════════════
//  Reasoning (module level)
// ═══════════════════════════════════════════════════════════════════════════
//
// Midstate has no native token type. A token is a **coloured coin**: a state
// thread whose 32-byte commitment carries `[balance u64][assetID 24]`, with a
// script enforcing conservation of mass across the outputs. An AMM is the same
// idea with `[reserveX u64][reserveY u64][padding 16]`. Both layouts, and the
// contracts that read them, come from the Midstate IDE templates.
//
// Two things make this easy to get catastrophically wrong, and both are the
// reason this module exists rather than leaving callers to build state by hand.
//
// ── 1. State fields are LITTLE-endian ──
//
// `OP_SLICE` hands bytes to `to_u64`, which is `u64::from_le_bytes`. So a
// balance of 10 is `0a00000000000000`, not `000000000000000a`. This is the exact
// opposite of the DEX announcement wire in `dex.js`, which is big-endian
// throughout. Getting it backwards does not fail loudly — it produces a state
// the contract reads as an astronomically large number, so a `conservation of
// mass` assert passes or fails for reasons that look nothing like the bug.
//
// ── 2. The SDK's arithmetic must match the contract's exactly ──
//
// The constant-product contract asserts, in integers, with no floats anywhere:
//
//     dx = max(0, new_x - x);  dy = max(0, new_y - y)
//     x_adj = new_x*1000 - dx*3
//     y_adj = new_y*1000 - dy*3
//     assert(x_adj * y_adj >= (x * y) * 1000000)
//
// A quote computed in floating point, or with the fee applied in a different
// order, lands one unit on the wrong side of that inequality and the trade is
// rejected on-chain after the user has paid for a commit. Every quote here is
// BigInt and is checked against a literal transcription of the contract's own
// assertion before being returned.
//
// ── What is NOT here ──
//
// No bytecode. The contracts are authored and compiled in the IDE; this module
// handles the state those contracts read and the arithmetic that keeps a
// proposed trade on the legal side of their asserts. `deployToken` and friends
// take a compiled `bytecode` argument for that reason.

import { blake3_hash_hex } from '../pkg/wasm_wallet.js';

/** Contract state is exactly 32 bytes — one BLAKE3-sized commitment. */
export const STATE_BYTES = 32;

/** Fee numerator/denominator for the IDE's constant-product template (0.3%). */
export const FEE_NUM = 997n;
export const FEE_DEN = 1000n;

// ── Layouts ─────────────────────────────────────────────────────────────────
//
// A layout is an ordered list of `[name, byteLength, kind]`. `u64` fields are
// little-endian numbers; `bytes` fields are raw and carried as hex.

export const LAYOUTS = {
    /** `state Token { balance: 8, assetID: 24 }` */
    TOKEN: [['balance', 8, 'u64'], ['assetID', 24, 'bytes']],
    /** `state AMM { reserveX: 8, reserveY: 8, padding: 16 }` */
    AMM: [['reserveX', 8, 'u64'], ['reserveY', 8, 'u64'], ['padding', 16, 'bytes']],
    /** `state Curve { supply: 8, padding: 24 }` */
    CURVE: [['supply', 8, 'u64'], ['padding', 24, 'bytes']],
};

const toHex = (b) => Array.from(b).map((x) => x.toString(16).padStart(2, '0')).join('');
const fromHex = (h) => {
    const s = String(h || '').replace(/^0x/, '').toLowerCase();
    const a = new Uint8Array(s.length / 2);
    for (let i = 0; i < a.length; i++) a[i] = parseInt(s.substr(i * 2, 2), 16);
    return a;
};

/**
 * Encode named fields into a 32-byte state commitment.
 *
 * `u64` fields are written little-endian to match `to_u64` in the VM. `bytes`
 * fields are right-padded with zeros, which is how the IDE's own examples encode
 * an ASCII asset id.
 *
 * @param {Array} layout One of {@link LAYOUTS}, or a compatible list.
 * @param {Object} values Field name → BigInt/number (u64) or hex/string (bytes).
 * @returns {string} 64-char hex.
 */
export function encodeState(layout, values) {
    const total = layout.reduce((s, [, len]) => s + len, 0);
    if (total !== STATE_BYTES) throw new Error(`launcher: layout is ${total} bytes, must be ${STATE_BYTES}`);

    const out = new Uint8Array(STATE_BYTES);
    let off = 0;
    for (const [name, len, kind] of layout) {
        const v = values[name];
        if (kind === 'u64') {
            let x = BigInt(v ?? 0);
            if (x < 0n) throw new Error(`launcher: ${name} must be non-negative`);
            if (x > (1n << 64n) - 1n) throw new Error(`launcher: ${name} exceeds u64`);
            // Little-endian: the VM reads these with u64::from_le_bytes.
            for (let i = 0; i < len; i++) { out[off + i] = Number(x & 0xffn); x >>= 8n; }
        } else {
            let bytes;
            if (v === undefined || v === null) bytes = new Uint8Array(0);
            else if (typeof v === 'string' && /^[0-9a-fA-F]*$/.test(v.replace(/^0x/, '')) && v.replace(/^0x/, '').length % 2 === 0) {
                bytes = fromHex(v);
            } else if (typeof v === 'string') {
                bytes = new TextEncoder().encode(v);      // ASCII asset ids, as in the IDE
            } else {
                bytes = Uint8Array.from(v);
            }
            if (bytes.length > len) throw new Error(`launcher: ${name} is ${bytes.length} bytes, field is ${len}`);
            out.set(bytes, off);                           // right-padded with zeros
        }
        off += len;
    }
    return toHex(out);
}

/**
 * Decode a 32-byte state commitment into named fields.
 *
 * `bytes` fields come back as hex and, when they decode cleanly as
 * zero-terminated ASCII, also as `<name>Ascii` — the IDE stores asset ids that
 * way and reading them back as hex alone makes them unrecognisable.
 */
export function decodeState(layout, stateHex) {
    const b = fromHex(stateHex);
    if (b.length !== STATE_BYTES) throw new Error(`launcher: state is ${b.length} bytes, expected ${STATE_BYTES}`);
    const out = {};
    let off = 0;
    for (const [name, len, kind] of layout) {
        if (kind === 'u64') {
            let v = 0n;
            for (let i = len - 1; i >= 0; i--) v = (v << 8n) | BigInt(b[off + i]);   // little-endian
            out[name] = v;
        } else {
            const raw = b.subarray(off, off + len);
            out[name] = toHex(raw);
            const trimmed = raw.subarray(0, raw.indexOf(0) === -1 ? raw.length : raw.indexOf(0));
            if (trimmed.length && trimmed.every((c) => c >= 0x20 && c < 0x7f)) {
                out[`${name}Ascii`] = new TextDecoder().decode(trimmed);
            }
        }
        off += len;
    }
    return out;
}

// ── Constant-product AMM ────────────────────────────────────────────────────

/**
 * Literal transcription of the IDE constant-product contract's assertion.
 *
 * Kept as its own function so every quote can be checked against the *contract's*
 * rule rather than against the formula used to produce it. A quote that satisfies
 * its own derivation but not this is a rejected transaction, and finding that out
 * here costs nothing while finding it out on-chain costs a commit.
 */
export function satisfiesConstantProduct(x, y, newX, newY) {
    const X = BigInt(x), Y = BigInt(y), NX = BigInt(newX), NY = BigInt(newY);
    const dx = NX > X ? NX - X : 0n;
    const dy = NY > Y ? NY - Y : 0n;
    const xAdj = NX * 1000n - dx * 3n;
    const yAdj = NY * 1000n - dy * 3n;
    return xAdj * yAdj >= X * Y * 1_000_000n;
}

/**
 * Output amount for a constant-product swap, after the 0.3% fee.
 *
 * Derived from the contract's own inequality rather than assumed: with `dy = 0`
 * the assert reduces to `new_y >= ceil(x*y*1000 / (x*1000 + dx*997))`, and the
 * largest legal payout is `y` minus that, which simplifies to the familiar
 * `floor(dx*997*y / (x*1000 + dx*997))`. The result is verified against
 * {@link satisfiesConstantProduct} before it is returned.
 *
 * @param {bigint|number} reserveIn   Reserve of the asset being paid in.
 * @param {bigint|number} reserveOut  Reserve of the asset being paid out.
 * @param {bigint|number} amountIn
 * @returns {bigint} Output amount, floored. Zero if the trade is not viable.
 */
export function getAmountOut(reserveIn, reserveOut, amountIn) {
    const x = BigInt(reserveIn), y = BigInt(reserveOut), dx = BigInt(amountIn);
    if (dx <= 0n) return 0n;
    if (x <= 0n || y <= 0n) throw new Error('launcher: pool has no liquidity');
    const num = dx * FEE_NUM * y;
    const den = x * FEE_DEN + dx * FEE_NUM;
    let out = num / den;                                  // floor
    if (out >= y) out = y - 1n;                           // never drain the pool
    if (out < 0n) return 0n;

    // Correct an off-by-one from the floor division.
    //
    // Bounded deliberately. The closed form is exact for this fee, so at most a
    // unit or two of slack is ever needed; if the loop runs longer than that,
    // the formula and the contract have genuinely diverged and grinding down one
    // unit at a time would spin for billions of iterations before admitting it.
    // Failing loudly is the only useful behaviour there.
    const SLACK = 4n;
    for (let i = 0n; i <= SLACK && out > 0n; i++) {
        if (satisfiesConstantProduct(x, y, x + dx, y - out)) return out;
        out -= 1n;
    }
    if (out <= 0n) return 0n;
    throw new Error(
        'launcher: quote does not satisfy the constant-product assert within tolerance; ' +
        'the fee constants here and in the contract have diverged'
    );
}

/**
 * Input required to receive exactly `amountOut`.
 *
 * Rounds UP and then confirms against the contract, because rounding down here
 * produces a quote that is one unit short and rejected on-chain.
 */
export function getAmountIn(reserveIn, reserveOut, amountOut) {
    const x = BigInt(reserveIn), y = BigInt(reserveOut), out = BigInt(amountOut);
    if (out <= 0n) return 0n;
    if (out >= y) throw new Error('launcher: output exceeds pool reserves');
    const num = x * out * FEE_DEN;
    const den = (y - out) * FEE_NUM;
    let dx = num / den + 1n;                              // ceil
    const SLACK = 4n;
    for (let i = 0n; i <= SLACK; i++) {
        if (satisfiesConstantProduct(x, y, x + dx, y - out)) return dx;
        dx += 1n;
    }
    throw new Error(
        'launcher: required input does not satisfy the constant-product assert within ' +
        'tolerance; the fee constants here and in the contract have diverged'
    );
}

/**
 * Quote a swap and produce the AMM state the contract expects afterwards.
 *
 * The contract also asserts `input_value() == reserveX` and
 * `sum_to_addr(CONTRACT) == new_x`, so `reserveX` is not bookkeeping — it must
 * equal the physical MDS held by the contract coin. `newReserveX` is therefore
 * what the funding output must actually carry.
 *
 * @param {Object} p
 * @param {bigint|number} p.reserveX  Native MDS held by the contract.
 * @param {bigint|number} p.reserveY  Token balance in the paired state.
 * @param {bigint|number} p.amountIn
 * @param {'x'|'y'} [p.direction='x']  Which side is paid in.
 * @param {string} [p.padding]         Preserved padding bytes.
 */
export function quoteSwap({ reserveX, reserveY, amountIn, direction = 'x', padding = '' }) {
    const x = BigInt(reserveX), y = BigInt(reserveY), dx = BigInt(amountIn);
    const payingX = direction === 'x';
    const out = payingX ? getAmountOut(x, y, dx) : getAmountOut(y, x, dx);
    if (out <= 0n) throw new Error('launcher: trade too small to yield any output');

    const newX = payingX ? x + dx : x - out;
    const newY = payingX ? y - out : y + dx;
    if (!satisfiesConstantProduct(x, y, newX, newY)) {
        throw new Error('launcher: computed trade violates the constant-product assert');
    }
    return {
        amountIn: dx,
        amountOut: out,
        newReserveX: newX,
        newReserveY: newY,
        // Effective price paid, for display only — never used in the arithmetic.
        effectivePrice: Number(dx) / Number(out),
        newState: encodeState(LAYOUTS.AMM, { reserveX: newX, reserveY: newY, padding }),
    };
}

// ── Linear bonding curve ────────────────────────────────────────────────────

/**
 * Cost of minting `count` units on the IDE's linear curve.
 *
 * The template asserts `sum_to_addr(TREASURY) == Curve.supply` and
 * `new_supply == supply + 1`: price equals current supply, and **one unit per
 * transaction**. Minting N units is therefore N transactions costing
 * `supply + (supply+1) + … + (supply+N-1)`, which is what this returns — the
 * closed form, so a caller can show a total without simulating each step.
 *
 * Note this is a mint price, not a swap: the payment goes to the treasury and no
 * reserve is held by the curve contract.
 */
export function bondingCurveCost(supply, count = 1) {
    const s = BigInt(supply), n = BigInt(count);
    if (n <= 0n) return 0n;
    // Σ from s to s+n-1  =  n*s + n(n-1)/2
    return n * s + (n * (n - 1n)) / 2n;
}

/** State after one mint on the linear curve. */
export function bondingCurveNextState(supply, padding = '') {
    const s = BigInt(supply);
    return {
        price: s,                                   // price == current supply
        newSupply: s + 1n,
        newState: encodeState(LAYOUTS.CURVE, { supply: s + 1n, padding }),
    };
}

// ── Token helpers ───────────────────────────────────────────────────────────

/**
 * Split a token balance across two output states.
 *
 * The IDE token contract asserts asset-id equality on BOTH outputs and
 * `balance == out0 + out1`. Producing states that do not sum exactly is the
 * single most common way to author an unspendable token coin, so the split is
 * computed here rather than left to the caller.
 */
export function splitTokenState({ balance, assetID, amount, padding }) {
    const bal = BigInt(balance), amt = BigInt(amount);
    if (amt <= 0n) throw new Error('launcher: split amount must be positive');
    if (amt > bal) throw new Error(`launcher: cannot split ${amt} from a balance of ${bal}`);
    const rest = bal - amt;
    void padding;
    return {
        sent: encodeState(LAYOUTS.TOKEN, { balance: amt, assetID }),
        change: encodeState(LAYOUTS.TOKEN, { balance: rest, assetID }),
        conserved: amt + rest === bal,
    };
}

/**
 * Asset id for a launched token.
 *
 * Derived from a name plus a caller-supplied nonce so two launches of the same
 * name do not collide — the token contract's forgery check is asset-id equality,
 * so two tokens sharing an id are mutually spendable.
 *
 * Truncated to 24 bytes because that is the field width; the collision domain is
 * 192 bits, which is ample.
 */
export function deriveAssetId(name, nonceHex = '') {
    const nameHex = toHex(new TextEncoder().encode(String(name)));
    return blake3_hash_hex(nameHex + String(nonceHex).replace(/^0x/, '')).slice(0, 48);
}

/**
 * Contract address for compiled bytecode.
 *
 * Every Midstate address is pay-to-script-hash: `Predicate::address()` is
 * `BLAKE3(bytecode)`. Compile in the IDE, pass the bytecode here.
 */
export function contractAddress(bytecodeHex) {
    return blake3_hash_hex(String(bytecodeHex).replace(/^0x/, ''));
}

// ── Contract templates ──────────────────────────────────────────────────────

/**
 * Bonding-curve contract source: batch buy/sell, contract holds its reserve.
 *
 * # Reasoning
 *
 * The IDE's `stateful` template asserts `new_supply == supply + 1`, so it mints
 * exactly one unit per transaction. At two blocks per commit/reveal that is
 * minutes per token, which is unusable for a launcher. It also keeps no reserve,
 * so there is nothing to sell back into — payment goes to a treasury and the
 * curve is buy-only.
 *
 * This version fixes both. It takes a batch delta and holds its own MDS, which
 * is what makes a sell possible:
 *
 *   buy  n:  cost   = n*s  + n*(n-1)/2      (linear price = supply)
 *   sell n:  refund = n*ns + n*(n-1)/2      (exact mirror, so a round trip is
 *                                            lossless and the reserve can never
 *                                            be drained below what was paid in)
 *
 * `input_value() == reserve` and `sum_to_addr(this_address()) == new_reserve`
 * are what stop the state lying about the MDS it claims to back: the reserve
 * field must equal the coin's real value both before and after.
 *
 * `this_address()` rather than a baked-in constant avoids the obvious
 * chicken-and-egg — the address is the hash of the bytecode, so bytecode cannot
 * contain it.
 *
 * Verified in `test-launcher.mjs` by executing it: exact-cost buys and sells are
 * accepted, underpayment and over-refund are rejected, and a buy/sell round trip
 * returns the reserve to its starting value.
 */
export const CURVE_CONTRACT_SOURCE = `// Pump-style bonding curve: batch buy/sell, contract holds its own reserve.\nstate Curve { supply: 8, reserve: 8, pad: 16 }\n\nmacro new_supply() { read_output_state(0); 0; 8; slice(); 0; add(); }\nmacro new_reserve() { read_output_state(0); 8; 8; slice(); 0; add(); }\n\n{\n    var s = Curve.supply;\n    var r = Curve.reserve;\n    var ns = new_supply();\n    var nr = new_reserve();\n\n    // The state cannot lie about the MDS it claims to hold.\n    // Reserve lives in a SEPARATE standard output at this address: consensus
    // gives OutputData::Confidential no value field at all, so a state thread
    // cannot also hold the MDS it describes. sum_input_value() rather than
    // input_value() because the script runs once per input and the state
    // thread's own value is 0.
    assert(sum_input_value() == r);\n    assert(sum_to_addr(this_address()) == nr);\n\n    if (ns > s) {\n        // BUY n: cost = n*s + n*(n-1)/2   (linear price = supply)\n        var n = ns - s;\n        var cost = n * s + (n * (n - 1)) / 2;\n        assert(nr == r + cost);\n    } else {\n        // SELL n: refund = n*ns + n*(n-1)/2  (mirror of the buy leg)\n        var n = s - ns;\n        var refund = n * ns + (n * (n - 1)) / 2;\n        assert(r == nr + refund);\n    }\n}\ntrue;\n`;

/** Layout the curve contract reads: `{supply, reserve, pad}`. */
export const CURVE_STATE_LAYOUT = [['supply', 8, 'u64'], ['reserve', 8, 'u64'], ['pad', 16, 'bytes']];

/**
 * Cost to buy `n` units starting from `supply`, on the batch curve.
 *
 * Identical to {@link bondingCurveCost} — kept as a named pair with
 * {@link curveSellRefund} so the two legs are obviously mirrors, which is the
 * property that makes a round trip lossless.
 */
export function curveBuyCost(supply, n) {
    const s = BigInt(supply), k = BigInt(n);
    if (k <= 0n) return 0n;
    return k * s + (k * (k - 1n)) / 2n;
}

/** Refund for selling `n` units from `supply`. Mirrors {@link curveBuyCost}. */
export function curveSellRefund(supply, n) {
    const s = BigInt(supply), k = BigInt(n);
    if (k <= 0n) return 0n;
    if (k > s) throw new Error(`launcher: cannot sell ${k} from a supply of ${s}`);
    const ns = s - k;
    return k * ns + (k * (k - 1n)) / 2n;
}

/**
 * Build the state pair for a curve buy or sell.
 *
 * @param {Object} p
 * @param {bigint|number} p.supply    Current supply.
 * @param {bigint|number} p.reserve   Current reserve (must equal the coin value).
 * @param {bigint|number} p.amount    Units to buy (positive) or sell.
 * @param {'buy'|'sell'} p.side
 */
export function curveTrade({ supply, reserve, amount, side = 'buy' }) {
    const s = BigInt(supply), r = BigInt(reserve), n = BigInt(amount);
    if (n <= 0n) throw new Error('launcher: trade amount must be positive');

    if (side === 'buy') {
        const cost = curveBuyCost(s, n);
        const ns = s + n, nr = r + cost;
        return {
            side, amount: n, cost, newSupply: ns, newReserve: nr,
            inputValue: r, outputValue: nr,
            newState: encodeState(CURVE_STATE_LAYOUT, { supply: ns, reserve: nr }),
            currentState: encodeState(CURVE_STATE_LAYOUT, { supply: s, reserve: r }),
        };
    }
    const refund = curveSellRefund(s, n);
    if (refund > r) throw new Error(`launcher: refund ${refund} exceeds reserve ${r}`);
    const ns = s - n, nr = r - refund;
    return {
        side, amount: n, refund, newSupply: ns, newReserve: nr,
        inputValue: r, outputValue: nr,
        newState: encodeState(CURVE_STATE_LAYOUT, { supply: ns, reserve: nr }),
        currentState: encodeState(CURVE_STATE_LAYOUT, { supply: s, reserve: r }),
    };
}
