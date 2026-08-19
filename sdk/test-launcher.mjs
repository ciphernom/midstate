// test-launcher.mjs — Token / AMM / bonding-curve state and math.
//
// The ground truth here is the Midstate IDE's own worked examples: their
// `inputState` and `outputs` strings are states a human authored and the VM
// accepted, so matching them byte-for-byte is stronger evidence than any
// round-trip against this implementation.
//
// Two failure modes are targeted specifically:
//
//   - Endianness. Contract state is LITTLE-endian (`to_u64` is
//     `u64::from_le_bytes`), the exact opposite of the DEX wire. Getting it
//     backwards produces a state the contract reads as an astronomical number,
//     so the resulting assert failure looks nothing like the bug.
//   - Off-by-one against the contract's assert. A quote that satisfies its own
//     derivation but lands one unit outside `x_adj*y_adj >= x*y*1e6` is a
//     transaction rejected on-chain after the user paid for a commit.
//
// Run: node test-launcher.mjs

import fs from 'fs/promises';
import initWasm from './pkg/wasm_wallet.js';
import * as L from './src/launcher.js';

await initWasm({ module_or_path: await fs.readFile('./pkg/wasm_wallet_bg.wasm') });

let pass = 0, fail = 0;
async function t(name, fn) {
    try { await fn(); console.log(`  ✅ ${name}`); pass++; }
    catch (e) { console.log(`  ❌ ${name} — ${e.message}`); fail++; }
}
function assert(c, m) { if (!c) throw new Error(m || 'assertion failed'); }

console.log('\n━━━ state encoding (little-endian) ━━━');

await t('matches the IDE AMM inputState exactly (X=10, Y=100)', () => {
    const IDE = '0a00000000000000640000000000000000000000000000000000000000000000';
    const got = L.encodeState(L.LAYOUTS.AMM, { reserveX: 10n, reserveY: 100n });
    assert(got === IDE, `\n    got ${got}\n    IDE ${IDE}`);
});

await t('matches the IDE AMM output state exactly (X=20, Y=51)', () => {
    const IDE = '1400000000000000330000000000000000000000000000000000000000000000';
    const got = L.encodeState(L.LAYOUTS.AMM, { reserveX: 20n, reserveY: 51n });
    assert(got === IDE, `\n    got ${got}\n    IDE ${IDE}`);
});

await t('matches the IDE token state exactly (balance 20, "MyGoldToken")', () => {
    const IDE = '14000000000000004d79476f6c64546f6b656e00000000000000000000000000';
    const got = L.encodeState(L.LAYOUTS.TOKEN, { balance: 20n, assetID: 'MyGoldToken' });
    assert(got === IDE, `\n    got ${got}\n    IDE ${IDE}`);
});

await t('matches the IDE bonding-curve state exactly (supply 10)', () => {
    const IDE = '0a00000000000000000000000000000000000000000000000000000000000000';
    const got = L.encodeState(L.LAYOUTS.CURVE, { supply: 10n });
    assert(got === IDE, `\n    got ${got}\n    IDE ${IDE}`);
});

await t('u64 fields are little-endian, not big', () => {
    // to_u64 is u64::from_le_bytes. Big-endian here reads as ~1.15e18.
    const got = L.encodeState(L.LAYOUTS.CURVE, { supply: 16n });
    assert(got.slice(0, 16) === '1000000000000000', `supply field is ${got.slice(0, 16)}`);
    assert(got.slice(0, 16) !== '0000000000000010', 'encoded big-endian');
});

await t('decodes the IDE states back to their annotated values', () => {
    const amm = L.decodeState(L.LAYOUTS.AMM, '0a00000000000000640000000000000000000000000000000000000000000000');
    assert(amm.reserveX === 10n && amm.reserveY === 100n, `got X=${amm.reserveX} Y=${amm.reserveY}`);
    const tok = L.decodeState(L.LAYOUTS.TOKEN, '14000000000000004d79476f6c64546f6b656e00000000000000000000000000');
    assert(tok.balance === 20n, `balance ${tok.balance}`);
    assert(tok.assetIDAscii === 'MyGoldToken', `assetID "${tok.assetIDAscii}"`);
});

await t('every layout is exactly 32 bytes', () => {
    for (const [name, layout] of Object.entries(L.LAYOUTS)) {
        const total = layout.reduce((s, [, len]) => s + len, 0);
        assert(total === 32, `${name} is ${total} bytes`);
        assert(L.encodeState(layout, {}).length === 64, `${name} encodes to the wrong width`);
    }
});

await t('oversized fields are refused, not truncated', () => {
    let a = false, b = false;
    try { L.encodeState(L.LAYOUTS.TOKEN, { balance: 1n << 70n, assetID: 'x' }); } catch { a = true; }
    try { L.encodeState(L.LAYOUTS.TOKEN, { balance: 1n, assetID: 'x'.repeat(30) }); } catch { b = true; }
    assert(a, 'accepted a balance beyond u64');
    assert(b, 'silently truncated an oversized assetID');
});

await t('a u64 at the boundary round-trips', () => {
    const max = (1n << 64n) - 1n;
    const s = L.encodeState(L.LAYOUTS.CURVE, { supply: max });
    assert(L.decodeState(L.LAYOUTS.CURVE, s).supply === max, 'u64 max did not survive');
});

console.log('\n━━━ constant-product AMM ━━━');

await t("reproduces the IDE's worked trade exactly", () => {
    // IDE comment: "Trade 10 X for Y. dx = 10. Output X = 20, Output Y = 51
    // (User withdrew 49 Y). x_adj = 19970, y_adj = 51000 → 1,018,470,000 >= 1e9."
    const out = L.getAmountOut(10n, 100n, 10n);
    assert(out === 49n, `got ${out} Y out, IDE says 49`);
    const q = L.quoteSwap({ reserveX: 10n, reserveY: 100n, amountIn: 10n });
    assert(q.newReserveX === 20n && q.newReserveY === 51n, `new reserves ${q.newReserveX}/${q.newReserveY}`);
    assert(q.newState === '1400000000000000330000000000000000000000000000000000000000000000',
        'new state does not match the IDE output');
});

await t("the IDE's own numbers satisfy the contract assert", () => {
    // x_adj * y_adj = 19970 * 51000 = 1,018,470,000 >= 1,000,000,000
    assert(L.satisfiesConstantProduct(10n, 100n, 20n, 51n), 'the IDE example was rejected');
    // One unit more output must fail.
    assert(!L.satisfiesConstantProduct(10n, 100n, 20n, 50n), 'an over-payout was accepted');
});

await t('every quote satisfies the contract assert (fuzz)', () => {
    // The check that matters: our arithmetic must never propose a trade the
    // contract rejects, at any reserve ratio or trade size.
    let checked = 0;
    for (const x of [1n, 7n, 100n, 4096n, 10n ** 9n]) {
        for (const y of [1n, 13n, 500n, 65536n, 10n ** 12n]) {
            for (const dx of [1n, 2n, 9n, x, x * 3n, x * 100n]) {
                let out;
                try { out = L.getAmountOut(x, y, dx); } catch { continue; }
                if (out <= 0n) continue;
                assert(L.satisfiesConstantProduct(x, y, x + dx, y - out),
                    `x=${x} y=${y} dx=${dx} out=${out} violates the assert`);
                checked++;
            }
        }
    }
    assert(checked > 50, `only ${checked} cases exercised`);
});

await t('a quote is maximal — one more unit out would be rejected', () => {
    for (const [x, y, dx] of [[10n, 100n, 10n], [1000n, 1000n, 7n], [12345n, 999n, 321n]]) {
        const out = L.getAmountOut(x, y, dx);
        assert(!L.satisfiesConstantProduct(x, y, x + dx, y - (out + 1n)),
            `x=${x} y=${y} dx=${dx}: out=${out} is not maximal, ${out + 1n} also passes`);
    }
});

await t('getAmountIn round-trips against getAmountOut', () => {
    for (const [x, y, want] of [[1000n, 1000n, 100n], [10n, 100n, 49n], [50000n, 777n, 13n]]) {
        const need = L.getAmountIn(x, y, want);
        assert(L.satisfiesConstantProduct(x, y, x + need, y - want),
            `x=${x} y=${y} want=${want}: computed input ${need} is rejected`);
        // And it should be minimal: one less must fail.
        assert(!L.satisfiesConstantProduct(x, y, x + need - 1n, y - want),
            `x=${x} y=${y}: input ${need} is not minimal`);
    }
});

await t('the fee is actually charged (0.3%)', () => {
    // Without a fee, a huge pool would return ~the input. With one, strictly less.
    const x = 10n ** 12n, y = 10n ** 12n, dx = 10n ** 6n;
    const out = L.getAmountOut(x, y, dx);
    assert(out < dx, `out ${out} >= in ${dx} — no fee charged`);
    assert(out > (dx * 995n) / 1000n, `out ${out} — fee looks larger than 0.3%`);
});

await t('the pool can never be fully drained', () => {
    const out = L.getAmountOut(1n, 100n, 10n ** 15n);
    assert(out < 100n, `drained the pool: ${out} of 100`);
    assert(out === 99n, `expected 99 (all but one), got ${out}`);
});

await t('degenerate inputs fail loudly', () => {
    assert(L.getAmountOut(10n, 10n, 0n) === 0n, 'zero input should quote zero');
    let threw = false;
    try { L.getAmountOut(0n, 10n, 5n); } catch { threw = true; }
    assert(threw, 'an empty pool did not raise');
    let threw2 = false;
    try { L.quoteSwap({ reserveX: 10n ** 12n, reserveY: 10n ** 12n, amountIn: 1n }); }
    catch { threw2 = true; }
    void threw2;   // a 1-unit trade into a huge pool may legitimately yield 0
});

await t('selling back moves reserves the other way', () => {
    const q = L.quoteSwap({ reserveX: 1000n, reserveY: 1000n, amountIn: 100n, direction: 'y' });
    assert(q.newReserveY === 1100n, `Y should rise: ${q.newReserveY}`);
    assert(q.newReserveX < 1000n, `X should fall: ${q.newReserveX}`);
    assert(L.satisfiesConstantProduct(1000n, 1000n, q.newReserveX, q.newReserveY), 'sell violates the assert');
});

console.log('\n━━━ linear bonding curve ━━━');

await t('price equals current supply', () => {
    const s = L.bondingCurveNextState(10n);
    assert(s.price === 10n, `price ${s.price}`);
    assert(s.newSupply === 11n, `new supply ${s.newSupply}`);
    assert(s.newState === L.encodeState(L.LAYOUTS.CURVE, { supply: 11n }), 'new state wrong');
});

await t('the IDE example: supply 10, pay 10, supply becomes 11', () => {
    // outputs: '...:10' to the treasury, and state 0b… = 11.
    const s = L.bondingCurveNextState(10n);
    assert(s.price === 10n, 'the IDE pays exactly the supply');
    assert(s.newState.slice(0, 16) === '0b00000000000000', `new state ${s.newState.slice(0, 16)}, IDE says 0b…`);
});

await t('multi-unit cost is the arithmetic series', () => {
    // The contract mints ONE unit per transaction, so N units is N txs.
    assert(L.bondingCurveCost(10n, 1n) === 10n, 'single unit');
    assert(L.bondingCurveCost(10n, 3n) === 33n, '10+11+12 = 33');
    assert(L.bondingCurveCost(0n, 5n) === 10n, '0+1+2+3+4 = 10');
    assert(L.bondingCurveCost(100n, 0n) === 0n, 'zero units');
    // Closed form must equal the simulated sum.
    let sum = 0n;
    for (let i = 0n; i < 50n; i++) sum += 7n + i;
    assert(L.bondingCurveCost(7n, 50n) === sum, 'closed form diverges from the simulation');
});

console.log('\n━━━ token helpers ━━━');

await t('a split conserves the balance exactly', () => {
    const r = L.splitTokenState({ balance: 20n, assetID: 'MyGoldToken', amount: 15n });
    assert(r.conserved, 'split did not conserve');
    const sent = L.decodeState(L.LAYOUTS.TOKEN, r.sent);
    const change = L.decodeState(L.LAYOUTS.TOKEN, r.change);
    assert(sent.balance + change.balance === 20n, `${sent.balance} + ${change.balance} != 20`);
    assert(sent.assetIDAscii === 'MyGoldToken' && change.assetIDAscii === 'MyGoldToken',
        'asset id not preserved on both outputs — the forgery check would fail');
});

await t('the IDE split example reproduces (20 → 15 + 5)', () => {
    const r = L.splitTokenState({ balance: 20n, assetID: 'MyGoldToken', amount: 15n });
    assert(r.sent === '0f000000000000004d79476f6c64546f6b656e00000000000000000000000000',
        `sent state ${r.sent}`);
    assert(r.change === '05000000000000004d79476f6c64546f6b656e00000000000000000000000000',
        `change state ${r.change}`);
});

await t('an over-split is refused', () => {
    let threw = false;
    try { L.splitTokenState({ balance: 5n, assetID: 'X', amount: 6n }); } catch { threw = true; }
    assert(threw, 'allowed a split larger than the balance');
});

await t('asset ids are 24 bytes and name-and-nonce dependent', () => {
    const a = L.deriveAssetId('PEPE', '00');
    assert(a.length === 48, `asset id is ${a.length / 2} bytes, field is 24`);
    assert(a !== L.deriveAssetId('PEPE', '01'), 'nonce does not affect the id');
    assert(a !== L.deriveAssetId('WOJAK', '00'), 'name does not affect the id');
    assert(a === L.deriveAssetId('PEPE', '00'), 'not deterministic');
    // And it fits the field without truncation.
    const st = L.encodeState(L.LAYOUTS.TOKEN, { balance: 1n, assetID: a });
    assert(L.decodeState(L.LAYOUTS.TOKEN, st).assetID === a, 'asset id did not survive the field');
});

await t('contract address is BLAKE3 of the bytecode', () => {
    const addr = L.contractAddress('deadbeef');
    assert(addr.length === 64, `address length ${addr.length}`);
    assert(addr === L.contractAddress('0xdeadbeef'), '0x prefix changed the address');
    assert(addr !== L.contractAddress('deadbeee'), 'different bytecode gave the same address');
});

console.log(`\n━━━ ${pass} passed, ${fail} failed ━━━\n`);
process.exit(fail ? 1 : 0);
