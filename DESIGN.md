# Midstate Protocol Design

## 1. Fundamentals

- **Hash Function**: BLAKE3 (32-byte output).
- **Architecture**: Sequential-time blockchain (Nakamoto consensus).
- **State Model**: UTXO Accumulator (Merkle-committed sorted vector).
- **Time**: Unix timestamp (seconds).
- **Storage**: `redb` for chain state and mining seed; flat bincode files for batch history.

## 2. Cryptography

### Coin IDs

A Coin ID is the hash of its properties. It is the only identifier stored in the state.

`CoinID = BLAKE3(address || value_le_bytes || salt)`

Where `address = BLAKE3(owner_pk)`.

### WOTS (Winternitz One-Time Signatures)

- **Parameter**: `w=16`.
- **Structure**: 16 message chains + 2 checksum chains = 18 chains total.
- **Max Digit**: 65,535.
- **Signature Size**: 576 bytes.
- **Security**: Post-quantum secure (stateful). One-time use only.

### MSS (Merkle Signature Scheme)

- **Structure**: Binary Merkle tree of WOTS keys.
- **Root**: Master Public Key (32 bytes).
- **Leaf**: WOTS public key.
- **Usage**: Allows `2^Height` signatures per public key.
- **Default Height**: 10 (1024 signatures).
- **Max Height**: 20 (~1M signatures).
- **Signature Size**: ~950 bytes at height 10.

## 3. Consensus & Mining

### Sequential Proof of Work

Mining requires sequential iteration to prove time has passed. Each mining attempt performs a full sequential hash chain that cannot be shortcut. 
However, miners can try multiple nonces in parallel across cores — the sequential property applies per-attempt, not across attempts. 
At low difficulty (loose target), most attempts succeed, making parallelism nearly useless. 
As difficulty increases, the advantage of parallel nonce search grows.

1. **Input**: `midstate` (after applying transactions and coinbase).
2. **Start**: `x₀ = BLAKE3(midstate || nonce)`.
3. **Work**: Iteratively hash `xᵢ = BLAKE3(xᵢ₋₁)` for `N` iterations, recording checkpoints every `C` steps.
4. **Result**: `final_hash = xₙ`.
5. **Target**: `final_hash` must be `< target`.
6. **Verification**: Verifiers spot-check random segments of the hash chain (segments chosen deterministically from `final_hash`) rather than recomputing the whole chain.

### Constants

- **Extension Iterations**: 1,000,000 (100 in fast-mining mode).
- **Checkpoint Interval**: 1,000 (10 in fast-mining mode).
- **Spot Checks**: 16 (3 in fast-mining mode).
- **Block Time**: 600 seconds (10 minutes).
- **Difficulty Adjustment**: Every 2,016 blocks (~2 weeks).
- **Adjustment Limit**: Max 4x increase or decrease per period.
- **Max Batch Size**: 100 transactions per block.
- **Commitment TTL**: 100 blocks.
- **Anchor**: Genesis anchored to Bitcoin block 935,897.

## 4. Economics

- **Unit**: Integer values only.
- **Denominations**: All output values must be powers of 2 (e.g., 1, 2, 4, 8...).
- **Block Reward**: Starts at 16.
- **Halving**: Reward halves every ~52,560 blocks (~1 year at 10-min blocks). Minimum reward: 1.
- **Fees**: `Sum(Inputs) - Sum(Outputs)`. Minimum fee for reveals: 1.

## 5. Transactions

Transactions use a two-phase Commit-Reveal scheme to separate intent from execution.

### Phase 1: Commit

- **Payload**: `commitment = BLAKE3(input_coin_ids... || output_coin_ids... || salt)`.
- **Action**: Commitment is added to the state and midstate is updated.
- **Cost**: 0 fee.
- **Expiry**: Commitments expire after 100 blocks (COMMITMENT_TTL).

### Phase 2: Reveal

- **Payload**:
    - **Inputs**: Preimages (Owner PK, Value, Salt) for spent coins.
    - **Signatures**: WOTS or MSS signatures over the commitment hash, verifying ownership.
    - **Outputs**: New coin definitions (Address, Value, Salt).
    - **Salt**: The blinding factor used in Phase 1.
- **Validation**:
    - Commitment must exist in state.
    - All inputs must exist in UTXO set.
    - No duplicate inputs.
    - Signatures must be valid against each input's `owner_pk`.
    - All output values must be powers of 2 and nonzero.
    - `Sum(Input Values) > Sum(Output Values)` (conservation of value + fee).
- **Limits**: Max 256 inputs, max 256 outputs per transaction.

## 6. Networking

- **Transport**: TCP.
- **Encryption**: Noise protocol (via libp2p).
- **Multiplexing**: Yamux.
- **Discovery**: Kademlia DHT + Identify protocol.
- **Protocol ID**: `/midstate/1.0.0`.
- **Serialization**: Bincode (length-prefixed, max 10 MB).
- **Sync**: Batch download (up to 100 per request) followed by full verification from genesis.
