// spend.js: the one amount the wallet has to work out rather than be told:
// "send all". No WebAssembly here; the worker passes in what it needs, and
// tests/spend.test.mjs runs it against the real coin selection.

const popcount = (v) => { let n = 0; for (let x = v; x > 0n; x >>= 1n) n += Number(x & 1n); return n; };

/** The fee prepare_spend charges (plan.rs `fee_for_bytes`, the same in every
 *  wallet): 10 base units per KiB of an estimated size, plus 10. Used only
 *  to guess where to look: what the WebAssembly says is what counts. */
export function feeModel(inputs, outputs) {
    return BigInt(Math.floor((100 + inputs * 1636 + outputs * 100) * 10 / 1024) + 10);
}

/** The fee prepare_spend charges to send `amount` of `total` with `inputs`
 *  coins, or null if they can't cover it: the cheapest the model allows that
 *  pays for its own outputs, feeModel(inputs, payments + k) for the least k
 *  whose change fits in k outputs (plan.rs `settle_fee`). */
export function modelFee(total, amount, inputs) {
    const payOuts = popcount(amount);
    for (let k = 0; k <= 64; k++) {
        const fee = feeModel(inputs, payOuts + k);
        if (total < amount + fee) return null;
        if (popcount(total - amount - fee) <= k) return fee;
    }
    return null;
}

/** Where to look first, best first: amounts that spend everything with no
 *  change back, then the largest that fits with a little change. */
export function candidates(total, inputs) {
    const exact = [];
    let fits = null;
    // No fee is below the one-output fee for these inputs.
    for (let f = feeModel(inputs, 1); f <= 60000n && f < total; f++) {
        const a = total - f;
        const m = modelFee(total, a, inputs);
        if (m === null) continue;
        if (fits === null) fits = a;
        if (a + m === total) exact.push(a);
        if (exact.length >= 3 || a < fits - 4000n) break;
    }
    return [...exact, ...(fits === null ? [] : [fits])];
}

/**
 * The most that `total` base units of coins (`inputs` of them) can send with
 * the fee on top: a BigInt. `feeFor(amount)` builds the transaction (the
 * WebAssembly's prepare_spend, which only builds a context, so probing it is
 * safe) and returns its fee, or throws when the coins can't cover it. It may
 * instead return `{ fee, inputs, value }`: the fee and the coins the
 * transaction spends. The wallet leaves out coins worth less than their own
 * fee, and then an amount modelled on every coin leaves change behind; with
 * the coins it chose, the search looks again among amounts those spend whole.
 *
 * Each probe can be slow: every change coin needs a one-time key made, about
 * a tenth of a second each. So the fee model above picks the amounts to try,
 * and the first that the WebAssembly accepts is the answer: usually the
 * first, an amount that leaves no change at all.
 *
 * If the model misses (the WebAssembly's rule changed), the fixed point the
 * worker used before: everything but the fee, tried until the fee stops
 * moving. That alone could circle without landing when the fee moves with
 * the outputs (and its fallback, the last fee plus a 2-unit buffer, then
 * failed as "Insufficient funds"), so after it comes a slow but sure search.
 */
export function maxSendable(total, feeFor, inputs = 1) {
    if (total <= 0n) throw new Error("Nothing to send — balance is zero.");
    const probe = (amount) => {
        try {
            const r = feeFor(amount);
            if (r !== null && typeof r === 'object') {
                return { fee: BigInt(r.fee), inputs: r.inputs, value: r.value === undefined ? undefined : BigInt(r.value) };
            }
            return { fee: BigInt(r) };
        } catch (_) { return null; }
    };
    const fee = (amount) => { const p = probe(amount); return p === null ? null : p.fee; };

    for (const a of candidates(total, inputs)) {
        const p = probe(a);
        if (p === null) continue;
        // Coins were left out: the most is set by the ones chosen.
        if (p.value !== undefined && p.inputs !== undefined && p.inputs < inputs && p.value < total) {
            for (const b of candidates(p.value, p.inputs)) {
                if (b > a && fee(b) !== null) return b;
            }
        }
        return a;
    }

    let amount = total > 3000n ? total - 3000n : total;
    let last = null;
    for (let i = 0; i < 10; i++) {
        const f = fee(amount);
        if (f === null) {
            amount = amount > 500n ? amount - 500n : 0n;
            if (amount <= 0n) break;
            continue;
        }
        last = amount;
        const target = total - f;
        if (target === amount) return amount;
        amount = target > 0n ? target : 1n;
    }

    // The search: the largest amount that fits, between the best that did and
    // the total; from scratch if nothing has.
    let lo = last;
    if (lo === null) {
        if (fee(1n) === null) throw new Error("Balance is too small to cover the network fee.");
        lo = 1n;
    }
    let hi = total;
    while (hi - lo > 1n) {
        const mid = (lo + hi) / 2n;
        if (fee(mid) !== null) lo = mid; else hi = mid;
    }
    return lo;
}
