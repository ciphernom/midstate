// "Send all" against the wallet's own coin selection and fees (prepare_spend,
// in the WebAssembly), with coins at the test phrase's one-time addresses.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { initSync, WebWallet, compute_coin_id_hex, address_to_checksummed_hex } from '../pkg/wasm_wallet.js';
import { maxSendable, modelFee, candidates } from '../spend.js';

initSync({ module: fs.readFileSync(new URL('../pkg/wasm_wallet_bg.wasm', import.meta.url)) });
const wallet = new WebWallet(Array(23).fill('abandon').concat('art').join(' '));
const to = address_to_checksummed_hex('cd'.repeat(32));

/** Coins of these values, one per one-time address. */
function coins(values) {
    return values.map((value, index) => {
        const address = wallet.get_wots_address(index);
        const salt = index.toString(16).padStart(64, '0');
        return { index, is_mss: false, mss_height: 0, mss_leaf: 0, address, value, salt, coin_id: compute_coin_id_hex(address, BigInt(value), salt) };
    });
}
const prepare = (utxos, amount) => JSON.parse(wallet.prepare_spend(JSON.stringify(utxos), to, amount, 1000, null, null));
const sum = utxos => utxos.reduce((s, u) => s + BigInt(u.value), 0n);
/** The powers of two in a value: the coins it would be paid in. */
const split = (v) => { const out = []; for (let b = 0; b < 53; b++) if (Math.floor(v / 2 ** b) % 2) out.push(2 ** b); return out; };

/** maxSendable as the worker calls it, counting the WebAssembly's work. */
function sendAll(utxos) {
    let probes = 0;
    const amount = maxSendable(sum(utxos), (a) => {
        probes++;
        const ctx = prepare(utxos, a);
        return { fee: ctx.fee, inputs: ctx.selected_inputs.length, value: ctx.selected_inputs.reduce((s, u) => s + BigInt(u.value), 0n) };
    }, utxos.length);
    const ctx = prepare(utxos, amount);
    const change = ctx.outputs.filter(o => o.address !== to.slice(0, 64)).reduce((s, o) => s + BigInt(o.value), 0n);
    return { amount, fee: BigInt(ctx.fee), change, inputs: ctx.selected_inputs.length, probes };
}

test('the fee model matches the WebAssembly for sends that need every coin', () => {
    for (const values of [[2 ** 31], split(2 ** 31 - 1.25 * 2 ** 30 - 450), [2 ** 30, 2 ** 29, 2 ** 20]]) {
        const utxos = coins(values);
        const total = sum(utxos);
        for (const a of candidates(total, utxos.length)) {
            const ctx = prepare(utxos, a);
            // Coins too small to pay their share of the fee can be left out;
            // the model counts every coin, so it can only charge more.
            if (ctx.selected_inputs.length === utxos.length) assert.equal(BigInt(ctx.fee), modelFee(total, a, utxos.length), `${values.length} coins, ${total - a} short`);
            else assert.ok(BigInt(ctx.fee) < modelFee(total, a, utxos.length));
        }
    }
});

test('one coin: everything but the fee, nothing back, at the first try', () => {
    const utxos = coins([2 ** 31]);
    const r = sendAll(utxos);
    assert.equal(r.change, 0n);
    assert.equal(r.amount + r.fee, sum(utxos));
    assert.equal(r.probes, 1);
});

test('many small coins: all but crumbs sent, quickly', () => {
    // The change of a 1.25 gMDS send from a 2 gMDS coin, plus a 1.5 gMDS
    // payment: 27 coins. The old fixed point circled here and gave up.
    const utxos = coins([...split(2 ** 31 - 1.25 * 2 ** 30 - 450), 2 ** 30, 2 ** 29]);
    const r = sendAll(utxos);
    // What isn't sent or paid as the fee: change, and coins of a few units
    // each, worth less than their share of the fee, left where they are.
    const left = sum(utxos) - r.amount - r.fee;
    assert.ok(left < 1000n, `left ${left}`);
    assert.ok(r.inputs >= utxos.length - 6);
    assert.ok(r.probes <= 4, `${r.probes} probes`);
});

test('when the model misses, the slow way still finds an amount that fits', () => {
    const utxos = coins([2 ** 31]);
    const total = sum(utxos);
    // A fee rule the model doesn't know: a flat 1,000 units more.
    let probes = 0;
    const amount = maxSendable(total, (a) => {
        probes++;
        const f = BigInt(prepare(utxos, a).fee) + 1000n;
        if (a + f > total) throw new Error('Insufficient funds');
        return f;
    }, 1);
    assert.ok(amount + 1500n >= total - 1000n && amount < total - 1000n, `left ${total - amount}`);
    assert.ok(probes > 1);
});

test('send all looks again when the wallet leaves dust out', () => {
    // A stand-in for prepare_spend that, like the planner, spends only the
    // coins worth more than their own input's fee (here, all but 3 × 8).
    const big = [2 ** 20, 2 ** 18], dust = [8, 8, 8];
    const total = BigInt([...big, ...dust].reduce((s, v) => s + v, 0));
    const kept = BigInt(big[0] + big[1]);
    let probes = 0;
    const feeFor = (a) => {
        probes++;
        const fee = modelFee(kept, a, big.length);
        if (fee === null) throw new Error('Insufficient funds');
        return { fee, inputs: big.length, value: kept };
    };
    const a = maxSendable(total, feeFor, big.length + dust.length);
    assert.equal(a + modelFee(kept, a, big.length), kept, 'spends what it keeps, whole');
    assert.ok(probes <= 3, `${probes} probes`);
    // A plain fee (the old contract) still works, without the second look.
    const plain = maxSendable(kept, (x) => { const f = modelFee(kept, x, 2); if (f === null) throw new Error('no'); return f; }, 2);
    assert.equal(plain + modelFee(kept, plain, 2), kept);
});

test('too little to pay any fee', () => {
    // 16 units: less than the 27 a one-coin, one-output send costs.
    const utxos = coins([16]);
    assert.throws(() => maxSendable(sum(utxos), a => prepare(utxos, a).fee, 1), /too small/);
    assert.throws(() => maxSendable(0n, () => 0n), /balance is zero/);
});
