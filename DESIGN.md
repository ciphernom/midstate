# Midstate Protocol Design

## 1. Fundamentals

- **Hash Function**: BLAKE3 (32-byte output).
- **Architecture**: Sequential-time blockchain.
- **State Model**: UTXO Accumulator (Sparse Merkle Tree / sorted vector hybrid).
- **Time**: Unix timestamp (seconds).
- **Storage**: Hybrid. `redb` (Key-Value) for hot chain state and mining seeds; flat `.bin` files chunked in directories for immutable batch history.

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

### Sequential Proof of Work (VDF-style)

Mining requires sequential iteration to prove time has passed. Each mining attempt performs a full sequential hash chain that cannot be shortcut. 
While miners can search for nonces in parallel, the computation of a single proof is strictly single-threaded, neutralizing ASIC advantages and enforcing "one CPU, one vote".

1. **Input**: `midstate` (after applying transactions and coinbase).
2. **Start**: `x₀ = BLAKE3(midstate || nonce)`.
3. **Work**: Iteratively hash `xᵢ = BLAKE3(xᵢ₋₁)` for `N` iterations, recording checkpoints every `C` steps.
4. **Result**: `final_hash = xₙ`.
5. **Target**: `final_hash` must be `< target`.
6. **Verification**: `O(1)`. Verifiers spot-check random segments of the hash chain (segments chosen deterministically from `final_hash`) rather than recomputing the whole chain.

### Fork Resolution & Bayesian Finality

- **Longest Chain Rule**: The chain with the highest cumulative sequential `depth` wins.
- **Reorgs**: Mempool seamlessly handles abandoned transactions from orphaned blocks.
- **Finality Estimator**: Nodes track network health using a Beta-Binomial probabilistic model. By observing honest extensions vs. adversarial orphans, nodes dynamically calculate the required block depth (`safe_depth`) to achieve a 1-in-a-million (`1e-6`) mathematical risk of a successful reorg.

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
- **Denominations**: All output values must be powers of 2 (e.g., 1, 2, 4, 8...). This naturally obfuscates exact transaction amounts.
- **Block Reward**: Starts at 16.
- **Halving**: Reward halves every ~52,560 blocks (~1 year at 10-min blocks). Minimum reward: 1.
- **Fees**: `Sum(Inputs) - Sum(Outputs)`. Minimum fee for reveals: 1.

## 5. Transactions

Transactions use a two-phase Commit-Reveal scheme to separate intent from execution, enhancing privacy and thwarting front-running.

### Phase 1: Commit

- **Payload**: `commitment = BLAKE3(input_coin_ids... || output_coin_ids... || salt)`.
- **Action**: Commitment is added to the state and midstate is updated. Requires a minor PoW spam nonce.
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

## 6. Networking & Synchronization

- **Transport**: TCP and QUIC via `libp2p`.
- **Encryption**: Noise protocol.
- **Multiplexing**: Yamux.
- **Discovery & PEX**: Kademlia DHT + Identify protocol + custom Peer Exchange (`GetAddr`/`Addr`).
- **NAT Traversal**: Built-in AutoNAT, DCUtR (Hole-punching), and Circuit Relay support. Nodes can sync and mine from behind strict home routers.
- **Serialization**: Bincode (length-prefixed).
- **Synchronization**: Asynchronous state-machine driven. 
  1. **Headers First**: Node downloads and verifies headers from genesis to find the deterministic fork point.
  2. **Batches**: Node downloads full blocks sequentially from the fork point, protecting against catch-up death spirals and "Frankenstein" chain corruption.
