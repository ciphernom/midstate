export { MidstateClient } from './client.js';
export { Wallet } from './wallet.js';
export { MidstateUtils, CHAT_DICTIONARY } from './utils.js';
export * as Storage from './storage.js';

// Native P2P wire codec. Exported so consumers can decode chat pushes or build
// protocol frames without depending on the transport layer.
export * as Bincode from './bincode.js';

// Mining: solo and pool, single- or multi-threaded.
export { Miner, TEMPLATE_MAX_AGE_MS } from './miner.js';
export { PoolClient, verifyPoolProof, libp2pPoolTransport, POOL_PROTOCOL, POOL_PUSH_PROTOCOL } from './pool.js';
// MinerPool uses node:worker_threads; import it directly under Node.
// import { MinerPool } from '@midstate/sdk/miner-pool';

// On-chain DEX: order announcements, fragmentation, covenant derivation.
export * as Dex from './dex.js';

// Token / AMM / bonding-curve contract state and exact-integer curve math.
export * as Launcher from './launcher.js';

// Bonding-curve token launcher (pump-style): contract, ledger, trade builder.
export * as Pump from './pump.js';

// Contract compiler and headless VM — compile a contract per token launch and
// simulate the spend before funding an address you cannot recover from.
export { compile, MAX_SCRIPT_SIZE, OPS, STD_LIB } from './compiler.js';
export { execute as executeContract, simulate as simulateContract,
         STATE_THREAD_ACTIVATION_HEIGHT, MAX_SIGOPS_PER_SCRIPT } from './vm.js';

// Reorg handling. Exported so callers can drive rollback directly (e.g. from a
// push notification) and inspect the depth/TTL bounds the wallet operates under.
export {
    maybeHandleReorg, rollbackTo, findForkHeight, pruneHistory,
    REORG_DEPTH, COMMITMENT_TTL,
} from './reorg.js';

// ── WASM re-exports ─────────────────────────────────────────────────────────
//
// Kept in sync with pkg/wasm_wallet.d.ts, grouped by purpose so it stays
// obvious what the SDK surfaces.

export {
    // Mining / proof of work
    mine_commitment_pow,
    search_nonces,
    mine_chat_pow_v2_wasm,

    // Value and identity helpers
    decompose_amount,
    compute_coin_id_hex,
    compute_commitment_hex,
    compute_p2pk_address_hex,
    address_to_checksummed_hex,
    blake3_hash_hex,

    // Key material
    generate_phrase,
    decrypt_cli_wallet,
    verify_mss_sig_wasm,

    // Contracts / Layer 2
    build_multisig_2of2_address,
    build_htlc_bytecode_hex,
    build_covenant_htlc_bytecode_hex,
    build_limit_order_covenant_bytecode_hex,
    build_channel_state,
    build_channel_reveal,

    // Q-Bolt payment channels
    qbolt_channel_address,
    qbolt_channel_bytecode_hex,
    qbolt_build_state,
    qbolt_build_refund_state,
    qbolt_build_close_reveal,
    qbolt_build_refund_reveal,
    qbolt_build_legacy_close_state,
    qbolt_build_legacy_close_reveal,
} from '../pkg/wasm_wallet.js';

// The wasm-bindgen initializer. `Wallet.init()` wraps this for the common case;
// re-exported for callers who manage the WASM instance themselves.
export { default as initWasm, initSync as initWasmSync } from '../pkg/wasm_wallet.js';
