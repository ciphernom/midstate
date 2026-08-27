/**
 * Pump curve v6 — a bonding-curve launcher that graduates.
 *
 * Four routes. Before the threshold the curve is live and positions are rows in
 * a Merkle ledger. After it, the reserve moves to an AMM and each holder
 * converts their row into a transferable token UTXO.
 *
 * @param {Object} p
 * @param {number} p.depth      Ledger depth, 1..16 (the burn codec's slot field
 *                              is two bytes, so 65,536 slots is the real cap).
 * @param {bigint} p.threshold  Supply at which the curve freezes and graduates.
 * @param {string} p.ammAddr    32-byte address of the AMM contract.
 * @param {string} p.assetId    24-byte asset id of the graduated token.
 * @param {bigint} p.poolTokens Token side of the initial AMM reserve.
 */
export function pumpContractSourceV6({ depth, threshold, ammAddr, assetId, poolTokens }) {
  if (!Number.isInteger(depth) || depth < 1 || depth > 16) {
    throw new Error(`pump: depth must be 1..16 (got ${depth})`);
  }
  const proofBytes = depth * 33;
  const ZERO24 = '00'.repeat(24);
  return `
// PUMP CURVE v6 — bonding curve, Merkle holder ledger, graduation to an AMM.
//
// State: [supply 8][root 24]. Reserve is implied: (s*s - s) / 2.
//
// GRADUATION IS DERIVED, NOT STORED. All 32 bytes of state are spent, so there
// is no room for a flag. Instead the threshold is read off supply itself:
// buys may not push supply past it, sells are refused at or above it, so once
// reached the supply is frozen and 'supply >= THRESHOLD' is a one-way latch.
//
// The ledger walk is one OP_MERKLE_ROOT rather than an unrolled per-level loop,
// so the script does not grow with depth. Folding the SAME path with an updated
// leaf derives the next root, which proves the change was a single-leaf edit.
state Pump { supply: 8, root: 24 }

witness Trade {
    owner_old: 24,
    owner_new: 24,
    pk:        32,
    old_bal:   8,
    n:         8,
    sig:       0,
    proof:     ${proofBytes},
    route:     1
}

macro new_supply() { read_output_state(0); 0; 8; slice(); 0; add(); }
macro new_root24() { read_output_state(0); 8; 24; slice(); }
macro fold24()     { Trade.proof; MERKLE_ROOT; 0; 24; slice(); }
macro out1_bal()   { read_output_state(1); 0; 8; slice(); 0; add(); }
macro out1_asset() { read_output_state(1); 8; 24; slice(); }
macro amm_y()      { read_output_state(1); 8; 8; slice(); 0; add(); }

route {

  // ── 0 ── BUY from the curve ───────────────────────────────────────────
  case 0: {
    assert(size(Trade.proof) == ${proofBytes});
    Trade.sig; require_sig(Trade.pk);
    assert(slice(hash(Trade.pk), 0, 24) == Trade.owner_new);

    Trade.old_bal; Trade.owner_old; concat(); hash(); fold24();
    assert(pop_hex() == Pump.root);
    Trade.old_bal + Trade.n; Trade.owner_new; concat(); hash(); fold24();
    assert(pop_hex() == new_root24());

    if (Trade.old_bal >= 1) { assert(Trade.owner_new == Trade.owner_old); }

    var ns = new_supply();
    assert(ns == Pump.supply + Trade.n);
    // The curve stops taking money at the threshold. This is the latch.
    assert(${threshold} >= ns);
    assert(output_address(0) == this_address());
    assert(sum_to_addr(this_address()) == (ns * ns - ns) / 2);

    drop(); drop(); drop(); drop(); drop(); drop(); drop();
  }

  // ── 1 ── SELL back to the curve ───────────────────────────────────────
  case 1: {
    assert(size(Trade.proof) == ${proofBytes});
    Trade.sig; require_sig(Trade.pk);
    assert(slice(hash(Trade.pk), 0, 24) == Trade.owner_new);

    Trade.old_bal; Trade.owner_old; concat(); hash(); fold24();
    assert(pop_hex() == Pump.root);
    Trade.old_bal - Trade.n; Trade.owner_new; concat(); hash(); fold24();
    assert(pop_hex() == new_root24());

    assert(Trade.owner_new == Trade.owner_old);

    var ns = new_supply();
    assert(Pump.supply == ns + Trade.n);
    // Once graduated the curve is closed, so nothing can be sold back into it.
    assert(${threshold} >= Pump.supply + 1);
    assert(output_address(0) == this_address());
    assert(sum_to_addr(this_address()) == (ns * ns - ns) / 2);

    drop(); drop(); drop(); drop(); drop(); drop(); drop();
  }

  // ── 2 ── GRADUATE: move the reserve into the AMM ──────────────────────
  // Permissionless: no signature, because there is nothing to authorise. The
  // move is fully determined by the threshold and the curve, and anyone paying
  // the fee to trigger it is doing the holders a favour.
  case 2: {
    // Read supply once and derive the reserve once. Spelling the curve out at
    // each use cost ~20 bytes a time and pushed the script past 1024.
    var s = Pump.supply;
    var r = (s * s - s) / 2;

    assert(s >= ${threshold});

    // The ledger is untouched: same supply, same root.
    assert(new_supply() == s);
    assert(new_root24() == Pump.root);
    assert(output_address(0) == this_address());

    // Replay guard. The contract must still be HOLDING the reserve, which is
    // only true the first time: afterwards it holds nothing.
    assert(sum_input_value() == r);

    // The reserve leaves for the pool, in full...
    assert(sum_to_addr(this_address()) == 0);
    assert(sum_to_addr("${ammAddr}") == r);
    // ...and the pool opens with a book that matches what arrived.
    assert(out1_bal() == r);
    assert(amm_y() == ${poolTokens});

    drop(); drop(); drop(); drop(); drop(); drop(); drop();
  }

  // ── 3 ── REDEEM: ledger row -> transferable token UTXO ────────────────
  case 3: {
    assert(Pump.supply >= ${threshold});
    assert(size(Trade.proof) == ${proofBytes});

    Trade.sig; require_sig(Trade.pk);
    assert(slice(hash(Trade.pk), 0, 24) == Trade.owner_old);

    // Prove the row...
    Trade.old_bal; Trade.owner_old; concat(); hash(); fold24();
    assert(pop_hex() == Pump.root);
    // ...and burn it. owner = 0 <=> balance = 0, so both are cleared.
    0; "${ZERO24}"; concat(); hash(); fold24();
    assert(pop_hex() == new_root24());

    // Supply is a frozen historical marker after graduation.
    assert(new_supply() == Pump.supply);
    assert(output_address(0) == this_address());
    // The reserve has already gone to the pool; the curve holds nothing.
    assert(sum_to_addr(this_address()) == 0);

    // The row becomes a token coin the holder can send anywhere.
    assert(out1_bal() == Trade.old_bal);
    assert(out1_asset() == "${assetId}");

    drop(); drop(); drop(); drop(); drop(); drop(); drop();
  }

  default: { fail(); }
}
true;
`;
}
