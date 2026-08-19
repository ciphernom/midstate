// pump.js — Bonding-curve token launcher with an unforgeable holder ledger.
//
// ═══════════════════════════════════════════════════════════════════════════
//  Reasoning (module level)
// ═══════════════════════════════════════════════════════════════════════════
//
// Midstate has no native token type and no minting policy, so a launcher has to
// build one out of scripts alone. Two designs look obvious and both are broken;
// understanding why is the whole reason this module is shaped the way it is.
//
// ── Why "token coins" do not work ──
//
// The natural eUTXO design gives each holder a coin whose state carries a
// balance, and has the curve verify balances on sale. It cannot be made sound:
//
//   1. A script only runs when ITS OWN coin is spent, and READ_INPUT_STATE
//      reads only its own input. On a sale the curve sees its own supply but
//      not the token coin's balance, and the token sees its balance but not the
//      curve's supply. Neither can verify the other's delta.
//   2. Creating an output at an address does NOT run that address's script.
//      So anyone can fabricate a coin at the token address claiming any balance,
//      then "sell" it and drain the reserve down to the real supply.
//
//  A same-transaction receipt fixes (1) — both scripts assert the delta against
//  their own knowledge, so they are forced to agree — but nothing fixes (2)
//  without a consensus-level minting rule, which is off the table.
//
// ── What does work: a Merkle ledger ──
//
// Balances live in a fixed-depth Merkle tree whose root is in the curve's own
// state. A trade must present a Merkle path proving its CURRENT balance against
// the CURRENT root, and the contract recomputes both the old and the new root in
// one pass (`merkle_update_step` from the IDE standard library). A balance that
// was never bought has no valid path, so it cannot be sold. Fabrication is
// impossible because there is nothing to fabricate — there are no token coins,
// only tree leaves.
//
// This is the same pattern as the IDE's `merkle_ledger` template, applied to a
// bonding curve.
//
// ── Contract invariants, all verified by execution in test-pump.mjs ──
//
//   reserve  = the contract coin's own MDS value (never stored, so it cannot lie)
//   buy  n:  cost   = n*s  + n*(n-1)/2
//   sell n:  refund = n*ns + n*(n-1)/2        (exact mirror; round trip lossless)
//   leaf     = BLAKE3(minimal_LE(balance) ⌢ owner24)
//
// The contract BUILDS both leaves from (owner, old_balance, n) rather than
// trusting leaves supplied in the witness. That binding is what stops a buyer
// paying for 4 and writing 400 into the tree.
//
// ── Known limits ──
//
//   * Depth 10 (1024 holder slots) is the largest that fits MAX_SCRIPT_SIZE.
//     Deeper trees need the language to emit a loop rather than unrolled steps.
//   * Every trade spends the single curve coin, so trades are serialised — and
//     each trade invalidates every other holder's path. A front end needs to
//     rebuild paths from the current tree between trades. This is inherent to
//     a single-UTXO curve, not to the ledger.
//   * Owner is a 24-byte truncated identifier. It is an identity slot, not an
//     authorisation check: this contract does not verify a signature, so slot
//     ownership must be enforced by whatever controls tree updates. Adding
//     `require_signed_by` costs ~40 B and one witness item.

import { blake3_hash_hex } from '../pkg/wasm_wallet.js';

/**
 * Largest tree depth that fits MAX_SCRIPT_SIZE (1024 B) — 512 holder slots.
 *
 * Signature verification costs about 235 bytes of script, which is two tree
 * levels. That is the price of the ledger being authorised rather than merely
 * proven, and it is not optional: without it the owner field is a password
 * published in plain sight and any observer can drain any slot.
 *
 * 512 slots is a per-LAUNCH limit, not a global one — each token is its own
 * contract. A launch expecting more holders needs the language to emit a loop
 * for the Merkle steps rather than unrolling them.
 */
export const MAX_DEPTH = 9;

/**
 * Contract source, with pick-depth placeholders.
 *
 * The placeholders are stack offsets that depend on tree depth, because the
 * witness carries `owner, old_balance, n` BELOW the sibling pairs — the Merkle
 * macro reaches down for its own siblings, so anything it must not disturb has
 * to sit under them.
 */
export const PUMP_CONTRACT_TEMPLATE = String.raw`
// PUMP CURVE — bonding curve with an unforgeable, AUTHORISED holder ledger.
//
// State: [supply 8][root 24].  Reserve is implied: (s*s - s) / 2.
// Witness (bottom -> top):
//   owner_old, owner_new, pk, old_bal, n, sig, sib_{D-1},dir_{D-1} ... sib_0,dir_0, route
//
// A leaf is BLAKE3(minimal_LE(balance) || owner24), where owner24 is the first
// 24 bytes of BLAKE3(pubkey). The Merkle proof shows WHAT a slot holds; the
// signature shows WHO may move it. An earlier version had only the proof, so
// the owner field was a password published in plain sight and any observer
// could drain any slot.
//
// Slot lifecycle, enforced below:  owner = 0  <=>  balance = 0
//   The signer ALWAYS proves owner_new's key, and owner_new may only differ from
//   owner_old when the slot was empty. So:
//     claim  (0 -> b) : anyone may take a free slot, naming themselves owner
//     spend  (a -> b) : owner_new == owner_old is forced, so the signer is
//                       proving the key the tree already commits to
//   Authorising against one owner rather than branching on two keeps the
//   signature check off the conditional path, where the stack discipline for
//   CHECKSIGVERIFY is fragile: 'var' claims stack slots without popping, so the
//   signature has to be the top of stack at exactly the right moment.
state Pump { supply: 8, root: 24 }

macro new_supply() { read_output_state(0); 0; 8; slice(); 0; add(); }
macro new_root24() { read_output_state(0); 8; 24; slice(); }

macro build_leaves() {
    pick({OLD});               // old_bal
    pick({N});                 // n
    add();                     // new_bal
    pick({OWN_NEW});           // owner_new
    concat(); hash();          // new_leaf
    pick({OLD_B});             // old_bal
    pick({OWN_OLD});           // owner_old
    concat(); hash();          // old_leaf
    swap();
}

macro build_leaves_sell() {
    pick({OLD});
    pick({N});
    sub();                     // new_bal = old_bal - n
    pick({OWN_NEW});
    concat(); hash();
    pick({OLD_B});
    pick({OWN_OLD});
    concat(); hash();
    swap();
}

route {
  case 0: {
    build_leaves();
    repeat({D}) { merkle_update_step(); }
    assert(slice(pop_hex(), 0, 24) == new_root24());
    assert(slice(pop_hex(), 0, 24) == Pump.root);
    {
        // AUTHORISE FIRST, on raw stack positions.
        //
        // 'var' claims a stack slot without popping it, so locals sit on the
        // stack and the deepest is declared first. CHECKSIGVERIFY needs the
        // signature on TOP, which is why sig is the last witness item and why
        // this runs before any local exists. Depths from the top here are
        // sig=0, n=1, old_bal=2, pk=3, owner_new=4, owner_old=5.
        // Raw ops, NOT require_signed_by(). That macro binds its argument as a
        // local, which leaves the argument itself on the stack above the
        // signature — so CHECKSIGVERIFY consumes the pubkey twice, compares it
        // against itself, and passes vacuously while the real signature is
        // dropped unverified. Emitting the two opcodes directly keeps the
        // signature on top where CHECKSIGVERIFY expects it.
        assert(slice(hash(pick(3)), 0, 24) == pick(5));
        pick(3);
        CHECKSIGVERIFY;

        var owner_old = pop_hex();
        var owner_new = pop_hex();
        var pk = pop_hex();
        var old_bal = pop_int();
        var n = pop_int();

        // A funded slot's owner is immutable, so proving owner_new is proving the
        // key the tree already commits to. Only an empty slot may name a new one.
        if (old_bal >= 1) { assert(owner_new == owner_old); }
        // Claiming an empty slot is open to anyone, so the signer proves the key
        // they are claiming WITH. Spending an occupied slot requires the key the
        // tree already commits to.
        var ns = new_supply();
        assert(ns == Pump.supply + n);
        assert(output_address(0) == this_address());
        assert(sum_to_addr(this_address()) == (ns * ns - ns) / 2);
    }
  }
  case 1: {
    build_leaves_sell();
    repeat({D}) { merkle_update_step(); }
    assert(slice(pop_hex(), 0, 24) == new_root24());
    assert(slice(pop_hex(), 0, 24) == Pump.root);
    {
        // AUTHORISE FIRST, on raw stack positions.
        //
        // 'var' claims a stack slot without popping it, so locals sit on the
        // stack and the deepest is declared first. CHECKSIGVERIFY needs the
        // signature on TOP, which is why sig is the last witness item and why
        // this runs before any local exists. Depths from the top here are
        // sig=0, n=1, old_bal=2, pk=3, owner_new=4, owner_old=5.
        // Raw ops, NOT require_signed_by(). That macro binds its argument as a
        // local, which leaves the argument itself on the stack above the
        // signature — so CHECKSIGVERIFY consumes the pubkey twice, compares it
        // against itself, and passes vacuously while the real signature is
        // dropped unverified. Emitting the two opcodes directly keeps the
        // signature on top where CHECKSIGVERIFY expects it.
        assert(slice(hash(pick(3)), 0, 24) == pick(5));
        pick(3);
        CHECKSIGVERIFY;

        var owner_old = pop_hex();
        var owner_new = pop_hex();
        var pk = pop_hex();
        var old_bal = pop_int();
        var n = pop_int();

        assert(owner_new == owner_old);
        var ns = new_supply();
        assert(Pump.supply == ns + n);
        assert(output_address(0) == this_address());
        assert(sum_to_addr(this_address()) == (ns * ns - ns) / 2);
    }
  }
  default: { fail(); }
}
true;
`;

export function pumpContractSource(depth = MAX_DEPTH) {
    const d = Number(depth);
    if (!Number.isInteger(d) || d < 1 || d > MAX_DEPTH) {
        throw new Error(`pump: depth must be 1..${MAX_DEPTH} (got ${depth})`);
    }
    // Stack offsets are computed from the witness LAYOUT, not written down.
    //
    // After the route byte is consumed the stack is the witness items with the
    // sibling pairs stacked above them, and every intermediate push shifts the
    // lot by one. A wrong offset does not fail to compile — it silently picks a
    // neighbouring slot. That has now caused two separate bugs here: one made
    // both Merkle leaves identical (passing every forgery test while rejecting
    // every honest trade), and one survived adding `sig` to the witness because
    // the constants were hand-written and nobody re-derived them.
    //
    // So the layout is declared once and the offsets fall out of it. Adding or
    // reordering a witness field now updates every pick automatically.
    const LAYOUT = ['owner_old', 'owner_new', 'pk', 'old_bal', 'n', 'sig']; // bottom → top
    const base = LAYOUT.length + 2 * d;          // stack depth once the route byte is gone
    const at = (field, pushed) => (base + pushed) - 1 - LAYOUT.indexOf(field);

    const subs = {
        // build_leaves, in order. `pushed` is how many temporaries are already
        // on top at that point in the sequence.
        '{OLD}': at('old_bal', 0),      // pick old_bal
        '{N}': at('n', 1),              // pick n, with old_bal on top
        '{OWN_NEW}': at('owner_new', 1),// pick owner_new, with new_bal on top
        '{OLD_B}': at('old_bal', 1),    // pick old_bal again, with new_leaf on top
        '{OWN_OLD}': at('owner_old', 2),// pick owner_old, with new_leaf + old_bal on top
        '{D}': d,
    };
    let out = PUMP_CONTRACT_TEMPLATE;
    for (const [k, v] of Object.entries(subs)) out = out.split(k).join(String(v));
    return out;
}

// ── Ledger ──────────────────────────────────────────────────────────────────

const H = (a, b) => blake3_hash_hex(a + b);

/** An unowned slot. `owner = 0 <=> balance = 0` is the canonical free form. */
export const ZERO_OWNER = '00'.repeat(24);

/**
 * Owner commitment for a public key: the first 24 bytes of BLAKE3(pk).
 *
 * # Reasoning
 *
 * The contract checks `slice(hash(pk), 0, 24) == owner` and then
 * CHECKSIGVERIFYs against that same pk, so the owner field is a COMMITMENT to a
 * key, not a secret. That distinction is the whole security model: the field is
 * published on chain in every ledger burn, and an earlier version treated it as
 * if publishing it were harmless because the contract compared it directly.
 * Anyone who read a burn could then spend the slot it described.
 *
 * 24 bytes is what the 32-byte state leaves after an 8-byte balance. 192 bits of
 * second-preimage resistance is ample; the truncation is a space constraint, not
 * a security choice.
 */
export function ownerFromPubkey(pubkeyHex) {
    return blake3_hash_hex(String(pubkeyHex).replace(/^0x/, '').toLowerCase()).slice(0, 48);
}

/** Minimal little-endian encoding, matching the VM's `from_u64`. */
export function minimalLE(v) {
    let x = BigInt(v);
    if (x === 0n) return '00';
    let h = x.toString(16);
    if (h.length % 2) h = '0' + h;
    return h.match(/.{2}/g).reverse().join('');
}

/** Fixed-width little-endian, for the 8-byte supply field. */
export function fixedLE(v, bytes = 8) {
    let x = BigInt(v), s = '';
    for (let i = 0; i < bytes; i++) { s += Number(x & 0xffn).toString(16).padStart(2, '0'); x >>= 8n; }
    return s;
}

/**
 * Leaf hash for a holder slot.
 *
 * `minimal_LE(balance) ⌢ owner24` is unambiguous despite the variable-width
 * balance: the owner is fixed at 24 bytes, so two different balances always
 * produce different total lengths or different leading bytes.
 */
export function leafHash(balance, owner24) {
    return blake3_hash_hex(minimalLE(balance) + String(owner24).replace(/^0x/, '').toLowerCase());
}

/**
 * A holder ledger: a fixed-depth Merkle tree over balance leaves.
 *
 * Fixed depth rather than sparse, because the contract unrolls exactly `depth`
 * verification steps — a shorter path simply would not verify.
 */
export class PumpLedger {
    /**
     * @param {number} depth
     * @param {string[]} owners One 24-byte hex id per slot; defaults to zeros.
     */
    constructor(depth = MAX_DEPTH, owners = []) {
        this.depth = depth;
        this.size = 2 ** depth;
        this.owners = Array.from({ length: this.size }, (_, i) => (owners[i] || '00'.repeat(24)).toLowerCase());
        this.balances = Array(this.size).fill(0n);
        this._rebuild();
    }

    _rebuild() {
        this.leaves = this.balances.map((b, i) => leafHash(b, this.owners[i]));
        let cur = this.leaves.slice();
        this.layers = [cur.slice()];
        while (cur.length > 1) {
            const next = [];
            for (let i = 0; i < cur.length; i += 2) next.push(H(cur[i], cur[i + 1]));
            this.layers.push(next.slice());
            cur = next;
        }
        this.root = cur[0];
    }

    /** Sibling/direction pairs from the leaf upward. `dir = '01'` means the sibling is on the left. */
    proof(slot) {
        const out = [];
        let i = slot;
        for (const layer of this.layers.slice(0, -1)) {
            const isRight = i % 2 === 1;
            out.push([layer[isRight ? i - 1 : i + 1], isRight ? '01' : '00']);
            i >>= 1;
        }
        return out;
    }

    /**
     * Record a slot's owner locally, ahead of the trade that commits it.
     *
     * # Reasoning
     *
     * This is bookkeeping, NOT a chain operation. On chain a slot's owner is
     * only ever written by a trade — the claim branch of the contract sets it as
     * part of the first buy — so calling this does not by itself change what the
     * network believes.
     *
     * An earlier version rebuilt the tree here, which made a locally claimed but
     * never-traded slot diverge from any tree reconstructed from chain data.
     * Every trade publishes its slot in a burn; a bare claim publishes nothing,
     * so there is nothing for a replay to apply. The tree is therefore left
     * alone and the owner is staged for {@link buildTrade} to use.
     *
     * @param {number} slot
     * @param {string} owner24 24-byte owner commitment, from {@link ownerFromPubkey}.
     */
    claim(slot, owner24) {
        if (this.balances[slot] !== 0n) throw new Error(`pump: slot ${slot} is already in use`);
        this.pendingOwners = this.pendingOwners || {};
        this.pendingOwners[slot] = String(owner24).replace(/^0x/, '').toLowerCase();
        return this;
    }

    /** Owner the next trade on `slot` will present: staged, else committed. */
    ownerFor(slot) {
        return (this.pendingOwners && this.pendingOwners[slot]) || this.owners[slot];
    }

    /** Apply a balance change and rebuild. Call only after the trade is accepted. */
    set(slot, balance) {
        this.balances[slot] = BigInt(balance);
        this._rebuild();
        return this;
    }

    /** Slot index for an owner, or -1. */
    slotOf(owner24) {
        return this.owners.indexOf(String(owner24).replace(/^0x/, '').toLowerCase());
    }

    /** First slot with no balance, no committed owner and no staged claim. */
    freeSlot() {
        return this.owners.findIndex((o, i) =>
            this.balances[i] === 0n && /^0+$/.test(o) && !(this.pendingOwners && this.pendingOwners[i]));
    }
}

// ── Curve arithmetic ────────────────────────────────────────────────────────

/**
 * Reserve implied by a supply.
 *
 * The contract stores no reserve and reads no input value. Every buy adds
 * exactly the curve cost and every sell removes exactly the mirror, so the MDS
 * held at the contract address is always `0 + 1 + ... + (s-1)`. Asserting
 * `sum_to_addr(this) == impliedReserve(new_supply)` on both legs is therefore
 * the whole money rule — and it needs no `sum_input_value()`, which is gated
 * behind COVENANT_SUM_ACTIVATION_HEIGHT (300,000).
 *
 * Written as `(s*s - s)/2` rather than `s*(s-1)/2` because the latter underflows
 * at `s = 0`, and SUB below zero is an error on this VM rather than a wrap.
 */
export function impliedReserve(supply) {
    const s = BigInt(supply);
    return (s * s - s) / 2n;
}

/** Cost to buy `n` units at supply `s`: the arithmetic series s..s+n-1. */
export function buyCost(supply, n) {
    const s = BigInt(supply), k = BigInt(n);
    if (k <= 0n) throw new Error('pump: amount must be positive');
    return k * s + (k * (k - 1n)) / 2n;
}

/** Refund for selling `n` units at supply `s`. Exact mirror of {@link buyCost}. */
export function sellRefund(supply, n) {
    const s = BigInt(supply), k = BigInt(n);
    if (k <= 0n) throw new Error('pump: amount must be positive');
    if (k > s) throw new Error(`pump: cannot sell ${k} of a supply of ${s}`);
    const ns = s - k;
    return k * ns + (k * (k - 1n)) / 2n;
}

/** Curve state commitment: `[supply 8][root 24]`. */
export function encodePumpState(supply, root32) {
    return fixedLE(supply, 8) + String(root32).replace(/^0x/, '').toLowerCase().slice(0, 48);
}

/**
 * Build the witness and the expected outputs for a trade.
 *
 * # Reasoning
 *
 * The witness order is not cosmetic. `merkle_update_step` reaches DOWN past the
 * two accumulators for its own sibling and direction, so the sibling pairs must
 * sit directly beneath the leaves — which means `owner`, `old_balance` and `n`
 * have to go at the very bottom, and the contract reaches back down for them
 * once the Merkle pass has consumed everything above.
 *
 * Siblings are pushed top-of-tree first so that the leaf-level pair ends up
 * closest to the accumulators, where the first step expects it.
 *
 * @param {Object} p
 * @param {PumpLedger} p.ledger
 * @param {number} p.slot
 * @param {bigint|number} p.amount
 * @param {'buy'|'sell'} p.side
 * @param {bigint|number} p.supply
 * @param {bigint|number} p.reserve
 */
export function buildTrade({ ledger, slot, amount, side, supply, reserve, pubkey, sig, newOwner = null }) {
    const n = BigInt(amount);
    const s = BigInt(supply), r = BigInt(reserve);
    const oldBal = ledger.balances[slot];
    const committedOwner = ledger.owners[slot];

    if (side !== 'buy' && side !== 'sell') throw new Error("pump: side must be 'buy' or 'sell'");
    if (side === 'sell' && n > oldBal) throw new Error(`pump: slot holds ${oldBal}, cannot sell ${n}`);
    if (!pubkey) throw new Error('pump: a trade needs the pubkey whose hash the slot commits to');
    if (!sig) throw new Error('pump: a trade needs a signature — the contract CHECKSIGVERIFYs it');

    const newBal = side === 'buy' ? oldBal + n : oldBal - n;
    const claiming = oldBal === 0n;

    // Resolve the owner the new leaf will carry.
    //
    // The contract asserts `owner_new == owner_old` whenever the slot is funded,
    // so a funded slot's owner is IMMUTABLE — including when selling out. A slot
    // emptied by a full sell keeps its owner and simply becomes reclaimable,
    // because the claim branch only requires `old_bal == 0`. An earlier version
    // tried to zero the owner on release and produced plans the contract
    // rejected on the honest path.
    let ownerNew;
    if (claiming) {
        ownerNew = String(newOwner || ledger.ownerFor(slot) || '').replace(/^0x/, '').toLowerCase();
        // A funded slot with a zero owner needs a pubkey hashing to zeros to
        // spend, which nobody can produce — the coins would be unspendable.
        // Enforced here rather than in the contract, where the check costs ~116
        // bytes of script (two whole tree levels) and only ever prevents a
        // caller locking THEIR OWN coins. Theft is prevented by the signature,
        // which stays on chain.
        if (!ownerNew || ownerNew === ZERO_OWNER) {
            throw new Error(
                `pump: claiming slot ${slot} needs an owner — pass one via claim() or newOwner, ` +
                `or its coins will be unspendable.`
            );
        }
    } else {
        if (newOwner && String(newOwner).replace(/^0x/, '').toLowerCase() !== committedOwner) {
            throw new Error(
                `pump: slot ${slot} is funded, so its owner is immutable. Sell out first; ` +
                `the slot can then be claimed afresh.`
            );
        }
        ownerNew = committedOwner;
    }

    // The owner the OLD leaf carries is whatever the tree already commits to.
    const ownerOld = committedOwner;

    // The contract always authorises against owner_new. On a funded slot that is
    // forced equal to owner_old, so the signer is proving the key the tree
    // already commits to; only an empty slot may name a new one. Checked here as
    // well so a mismatch fails before the caller pays for a commit.
    const expected = ownerNew;
    const derived = ownerFromPubkey(pubkey);
    if (derived !== expected) {
        throw new Error(
            `pump: pubkey hashes to ${derived.slice(0, 12)}… but the contract requires ` +
            `${expected.slice(0, 12)}… for this ${claiming ? 'claim' : 'spend'}.`
        );
    }

    const amountMds = side === 'buy' ? buyCost(s, n) : sellRefund(s, n);
    const newSupply = side === 'buy' ? s + n : s - n;
    const newReserve = impliedReserve(newSupply);
    if (newReserve < 0n) throw new Error('pump: refund exceeds the reserve');

    // Project the tree forward without touching the caller's ledger; it is only
    // advanced once the trade is actually accepted on chain.
    const projected = new PumpLedger(ledger.depth, ledger.owners);
    projected.balances = ledger.balances.slice();
    projected.balances[slot] = newBal;
    projected.owners[slot] = ownerNew;
    projected._rebuild();

    // Witness order is dictated by the Merkle macro, which reaches DOWN past the
    // two accumulators for its own sibling and direction. Anything it must not
    // disturb therefore sits beneath the sibling pairs, and the contract reaches
    // back down for it once the Merkle pass has consumed everything above.
    // Siblings go top-of-tree first so the leaf-level pair ends up closest to
    // the accumulators, where the first step expects it.
    const path = ledger.proof(slot);
    const witness = [
        ownerOld,
        ownerNew,
        String(pubkey).replace(/^0x/, ''),
        minimalLE(oldBal),
        minimalLE(n),
        // Last, so it is the top of stack when CHECKSIGVERIFY runs. A `var`
        // claims a stack slot without popping, so anything declared as a local
        // sits above the signature and puts it permanently out of reach.
        String(sig).replace(/^0x/, ''),
    ];
    for (let i = path.length - 1; i >= 0; i--) witness.push(path[i][0], path[i][1]);
    witness.push(side === 'buy' ? '00' : '01');

    const stateOutput = { index: 0, kind: 'confidential', address: null, value: 0n,
                          state: encodePumpState(newSupply, projected.root) };
    const reserveOutput = { index: 1, kind: 'standard', address: null, value: newReserve };

    return {
        slot, owner: ownerNew, ownerOld, claiming,
        side, amount: n, amountMds, oldBalance: oldBal, newBalance: newBal,
        newSupply, newReserve, witness,
        currentState: encodePumpState(s, ledger.root),
        newState: stateOutput.state,
        newRoot: projected.root,
        sumInputValue: r,
        outputs: [stateOutput, reserveOutput],
    };
}

// ═══════════════════════════════════════════════════════════════════════════
//  Transaction layer
// ═══════════════════════════════════════════════════════════════════════════
//
// ── Why the ledger is published on chain ──
//
// The contract commits only to the tree's ROOT. The balances themselves live
// nowhere on chain, and every trade moves the root, so each trade invalidates
// every other holder's Merkle proof. A holder who cannot rebuild a current proof
// cannot trade — and nobody else can build one for them either, because nobody
// has the tree.
//
// That is fatal for a public launcher: it makes coins unreachable after a lost
// cache, and it makes the whole thing depend on one privileged indexer.
//
// So every trade carries a zero-value `DataBurn` recording the slot it changed.
// Replaying those burns rebuilds the tree exactly, from chain data alone, by
// anyone. The burn is 47 bytes against a `MAX_BURN_DATA_SIZE` of 80, so it never
// needs the DEX's fragmentation.
//
// It does not disturb the contract: `DataBurn` carries no address, so it
// contributes nothing to `sum_to_addr`, and it sits after the two outputs the
// contract inspects by index.

export const LEDGER_MAGIC = '4d504d50';   // "MPMP"
export const LEDGER_VER = 1;
/** magic 4 + ver 1 + contract 8 + slot 2 + balance 8 + owner 24 */
export const LEDGER_BURN_BYTES = 47;

const beBytes = (v, n) => {
    const a = new Uint8Array(n);
    let x = BigInt(v);
    for (let i = n - 1; i >= 0; i--) { a[i] = Number(x & 0xffn); x >>= 8n; }
    return Array.from(a).map((b) => b.toString(16).padStart(2, '0')).join('');
};
const beRead = (hex, off, n) => BigInt('0x' + hex.slice(off * 2, (off + n) * 2));

/** Short, collision-resistant handle for a contract address (8 bytes). */
export function contractShortId(contractAddrHex) {
    return String(contractAddrHex).replace(/^0x/, '').toLowerCase().slice(0, 16);
}

/**
 * Encode one ledger update for publication in a burn.
 *
 * Big-endian, matching the DEX announcement wire rather than contract state —
 * these are announcements, read by scanners, not values read by scripts.
 */
export function encodeLedgerUpdate({ contractAddr, slot, balance, owner }) {
    return LEDGER_MAGIC
        + beBytes(LEDGER_VER, 1)
        + contractShortId(contractAddr)
        + beBytes(slot, 2)
        + beBytes(balance, 8)
        + String(owner).replace(/^0x/, '').toLowerCase().padStart(48, '0').slice(0, 48);
}

/** @returns {Object|null} Decoded update, or null if this burn isn't one. */
export function decodeLedgerUpdate(payloadHex) {
    const h = String(payloadHex || '').replace(/^0x/, '').toLowerCase();
    if (h.length < LEDGER_BURN_BYTES * 2) return null;
    if (h.slice(0, 8) !== LEDGER_MAGIC) return null;
    if (Number(beRead(h, 4, 1)) !== LEDGER_VER) return null;
    return {
        contractId: h.slice(10, 26),
        slot: Number(beRead(h, 13, 2)),
        balance: beRead(h, 15, 8),
        owner: h.slice(46, 94),
    };
}

/**
 * Rebuild a ledger by replaying published updates in chain order.
 *
 * # Reasoning
 *
 * This is what makes the launcher trustless to read. Anyone with the chain can
 * reconstruct the exact tree the contract is committed to, so a holder never
 * depends on a service to produce their proof.
 *
 * The rebuilt root is checked against the on-chain state where the caller
 * supplies it. A mismatch means the replay is incomplete — a missed block, a
 * burn from a fork, or an update this version cannot parse — and continuing on a
 * wrong tree would produce proofs the contract silently rejects. So it is
 * reported rather than absorbed.
 *
 * @param {Object} p
 * @param {number} p.depth
 * @param {string} p.contractAddr
 * @param {Object[]} p.updates Decoded updates, oldest first.
 * @param {string} [p.expectedRoot] On-chain root to verify against.
 */
export function replayLedger({ depth, contractAddr, updates, expectedRoot = null }) {
    const id = contractShortId(contractAddr);
    const ledger = new PumpLedger(depth);
    let applied = 0;
    for (const u of updates) {
        if (!u || u.contractId !== id) continue;      // another launch's burn
        if (u.slot >= ledger.size) continue;          // out of range for this depth
        ledger.owners[u.slot] = u.owner;
        ledger.balances[u.slot] = BigInt(u.balance);
        applied++;
    }
    ledger._rebuild();
    const supply = ledger.balances.reduce((a, b) => a + b, 0n);
    const rootMatches = expectedRoot === null
        ? null
        : ledger.root.slice(0, 48) === String(expectedRoot).replace(/^0x/, '').toLowerCase().slice(0, 48);
    return { ledger, applied, supply, reserve: impliedReserve(supply), rootMatches };
}

/**
 * Outputs that deploy a new curve.
 *
 * # Reasoning
 *
 * The reserve invariant `r = impliedReserve(s)` is inductive from `s = 0,
 * r = 0`, so the launch MUST start empty. Seeding the address with a balance
 * makes every later trade fail the money assertion, with the contract's funds
 * locked behind an assertion that can never again hold. The zero reserve is
 * therefore expressed here rather than left to the caller.
 *
 * The state thread's salt is fixed and returned, because the wallet has to spend
 * that exact coin on the first trade and cannot rediscover a random salt.
 *
 * @param {Object} p
 * @param {number} [p.depth]
 * @param {string} p.saltHex 32-byte salt for the state-thread coin.
 * @returns {{address, bytecode, outputs, initialState, ledger, depth}}
 */
export function deployPlan({ depth = MAX_DEPTH, saltHex, compileFn }) {
    if (!compileFn) throw new Error('pump: deployPlan needs a compile function (from src/compiler.js)');
    if (!saltHex || String(saltHex).replace(/^0x/, '').length !== 64) {
        throw new Error('pump: deployPlan needs a 32-byte hex salt for the state coin');
    }
    const source = pumpContractSource(depth);
    const compiled = compileFn(source);
    const address = blake3_hash_hex(compiled.bytecode);
    const ledger = new PumpLedger(depth);
    const initialState = encodePumpState(0n, ledger.root);

    return {
        address,
        bytecode: compiled.bytecode,
        depth,
        initialState,
        ledger,
        // One output only: the state thread. The reserve starts at zero, which
        // impliedReserve(0) requires, so there is no coin to create yet.
        outputs: [
            { out_type: 'confidential', address, value: 0, state: initialState, salt: String(saltHex).replace(/^0x/, '') },
        ],
    };
}

/**
 * Contract inputs and outputs for one trade, ready for `Wallet.executeContract`.
 *
 * # Reasoning
 *
 * Output ORDER is consensus-relevant here, not cosmetic. The contract reads
 * `read_output_state(0)` and asserts `output_address(0) == this_address()`, so
 * the state thread must be index 0 and the reserve coin index 1. The ledger burn
 * follows, where it cannot disturb either check.
 *
 * Salts are explicit for both contract-owned outputs so the wallet can spend
 * them on the next trade; a random salt would strand the coin.
 *
 * @param {Object} p
 * @param {Object} p.plan        From {@link buildTrade}.
 * @param {string} p.address     Contract address.
 * @param {string} p.stateSalt   Salt for the NEW state-thread coin.
 * @param {string} p.reserveSalt Salt for the NEW reserve coin.
 * @param {Object} p.current     `{ stateCoinId, stateSalt, reserveCoinId, reserveSalt }` of the coins being spent.
 * @param {string} p.witnessJoin Separator the caller's witness format expects.
 */
export function tradeTx({ plan, address, stateSalt, reserveSalt, current, witnessJoin = ',' }) {
    const addr = String(address).replace(/^0x/, '').toLowerCase();
    const witness = plan.witness.join(witnessJoin);

    // The state thread is always spent. The reserve coin only exists once the
    // curve holds something, so a first buy has no reserve input to consume.
    const contractInputs = [
        { coin_id: current.stateCoinId, witness, value: 0, salt: current.stateSalt, state: plan.currentState },
    ];
    if (current.reserveCoinId && plan.sumInputValue > 0n) {
        contractInputs.push({
            coin_id: current.reserveCoinId, witness,
            value: Number(plan.sumInputValue), salt: current.reserveSalt,
        });
    }

    const outputs = [
        { out_type: 'confidential', address: addr, value: 0, state: plan.newState, salt: String(stateSalt).replace(/^0x/, '') },
    ];
    if (plan.newReserve > 0n) {
        outputs.push({ out_type: 'standard', address: addr, value: Number(plan.newReserve), salt: String(reserveSalt).replace(/^0x/, '') });
    }

    return {
        contractInputs,
        outputs,
        ledgerBurn: encodeLedgerUpdate({
            contractAddr: addr, slot: plan.slot, balance: plan.newBalance, owner: plan.owner,
        }),
    };
}
