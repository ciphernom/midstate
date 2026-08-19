#!/usr/bin/env node
/**
 * Midstate Pump.fun-style demo (offline)
 * Run from the SDK root:  node demo-pump.mjs
 */

import fs from 'fs/promises';
import initWasm, * as W from './pkg/wasm_wallet.js';
import { compile } from './src/compiler.js';
import { execute } from './src/vm.js';
import * as P from './src/pump.js';

const DEPTH = 4;                       // 16 slots – fast for demo (use 10 for real)
const TEST_HEIGHT = 310_000n;          // above COVENANT_SUM_ACTIVATION_HEIGHT

await initWasm({ module_or_path: await fs.readFile('./pkg/wasm_wallet_bg.wasm') });

// Simple fake users (VM only checks string equality for CHECKSIG)
const USERS = {
  alice: { pk: 'aa'.repeat(32), sig: 'aa'.repeat(32) },
  bob:   { pk: 'bb'.repeat(32), sig: 'bb'.repeat(32) },
  carol: { pk: 'cc'.repeat(32), sig: 'cc'.repeat(32) },
};

function ownerOf(user) {
  return P.ownerFromPubkey(USERS[user].pk);
}

function printState(label, state) {
  const holders = state.ledger.balances.filter(b => b > 0n).length;
  console.log(`\n── ${label} ──────────────────────────────`);
  console.log(`  Supply:   ${state.supply}`);
  console.log(`  Reserve:  ${state.reserve}`);
  console.log(`  Holders:  ${holders} / ${state.ledger.size}`);
  console.log(`  Root:     ${state.ledger.root.slice(0, 16)}…`);
}

function launch() {
  const compiled = compile(P.pumpContractSource(DEPTH));
  const address  = W.blake3_hash_hex(compiled.bytecode);
  const ledger   = new P.PumpLedger(DEPTH);

  const state = {
    address,
    ledger,
    supply:  0n,
    reserve: 0n,
    asm:     compiled.asm,
  };

  console.log('🚀 New Pump curve deployed');
  console.log(`   Address: ${address}`);
  printState('Initial', state);
  return state;
}

function trade(state, user, slot, amount, side) {
  const who = USERS[user];

  try {
    if (side === 'buy' && state.ledger.balances[slot] === 0n) {
      state.ledger.claim(slot, ownerOf(user));
    }

    const plan = P.buildTrade({
      ledger:  state.ledger,
      slot,
      amount:  BigInt(amount),
      side,
      supply:  state.supply,
      reserve: state.reserve,
      pubkey:  who.pk,
      sig:     who.sig,
    });

    const res = execute(state.asm, {
      witness:     plan.witness,
      inputState:  P.encodePumpState(state.supply, state.ledger.root),
      inputValue:  0n,
      thisAddress: state.address,
      height:      TEST_HEIGHT,
      outputs: [
        { address: state.address, value: 0n,                 state: plan.newState },
        { address: state.address, value: Number(plan.newReserve) },
      ],
    });

    if (!res.ok) {
      console.log(`❌ ${user} ${side} ${amount} → rejected by contract (${res.error || 'assert'})`);
      return false;
    }

    state.ledger.owners[slot] = plan.owner;
    state.ledger.set(slot, plan.newBalance);
    state.supply  = plan.newSupply;
    state.reserve = plan.newReserve;

    console.log(`✅ ${user} ${side}s ${amount}`);
    return true;

  } catch (e) {
    console.log(`❌ ${user} ${side} ${amount} → rejected by SDK (${e.message})`);
    return false;
  }
}

// ── Scenario ────────────────────────────────────────────────────────
const state = launch();

trade(state, 'alice', 0, 5, 'buy');
printState('After Alice buys 5', state);

trade(state, 'bob', 1, 10, 'buy');
printState('After Bob buys 10', state);

trade(state, 'carol', 2, 3, 'buy');
printState('After Carol buys 3', state);

trade(state, 'alice', 0, 2, 'sell');
printState('After Alice sells 2', state);

console.log('\n(Expecting rejection…)');
trade(state, 'bob', 1, 999, 'sell');

printState('Final state', state);
console.log('\nDone.');
