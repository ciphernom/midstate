# Midstate Protocol Design

## 1. Fundamentals

- **Hash Function**: BLAKE3 (32-byte output).
- **Architecture**: Sequential-time blockchain.
- **State Model**: UTXO Accumulator via Sparse Merkle Tree (SMT).
- **Time**: Unix timestamp (seconds).
- **Storage**: Hybrid. `redb` (Key-Value) for hot chain state and mempool; flat `.bin` files chunked in directories for immutable batch history. Checkpoints are automatically pruned for deep history to save ~98% of disk space.

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
- **Security**: Post-quantum secure. Strictly one-time use.

### MSS (Merkle Signature Scheme)

- **Structure**: Binary Merkle tree of WOTS keys.
- **Root**: Master Public Key (32 bytes).
- **Usage**: Allows `2^Height` signatures per public key.
- **Safety**: Wallet statefully tracks the `next_leaf` index and actively syncs with the node mempool/chain before signing to prevent catastrophic key reuse.

## 3. Consensus & Mining

### Sequential Proof of Work (VDF-style)

Mining requires sequential iteration to prove time has passed. While miners can search for nonces in parallel, the computation of a single proof is strictly single-threaded, neutralizing ASIC advantages and enforcing "one CPU, one vote".

1. **Input**: `midstate` (after applying transactions and coinbase).
2. **Start**: `x₀ = BLAKE3(midstate || nonce)`.
3. **Work**: Iteratively hash `xᵢ = BLAKE3(xᵢ₋₁)` for `N` iterations, recording checkpoints every `C` steps.
4. **Result**: `final_hash = xₙ`.
5. **Target**: `final_hash` must be `< target`.
6. **Verification**: `O(1)`. Verifiers spot-check random segments of the hash chain.

### Difficulty Adjustment (ASERT)

Midstate uses the Absolutely Scheduled Exponentially Decaying (ASERT) algorithm.
- Adjusts continuously on every block based on absolute elapsed time since genesis.
- Completely eliminates sliding-window exploits (time-warp, hash-and-flee, echo effects).
- Uses integer-based Taylor polynomial approximations for determinism.
- **Half-life**: 4 hours.

### Fork Resolution & Bayesian Finality

- **Longest Chain Rule**: The chain with the highest cumulative sequential `depth` wins.
- **Finality Estimator**: Nodes track network health using a Beta-Binomial probabilistic model. By observing honest extensions vs. adversarial orphans, nodes dynamically calculate the required block depth (`safe_depth`) to achieve a 1-in-a-million (`1e-6`) mathematical risk of a successful reorg.

### Constants

- **Extension Iterations**: 1,000,000 (100 in fast-mining mode).
- **Checkpoint Interval**: 1,000 (10 in fast-mining mode).
- **Block Time**: 60 seconds (1 minute).
- **Max Batch Size**: 100 transactions per block.
- **Commitment TTL**: 100 blocks.

## 4. Economics

- **Unit**: Integer values only.
- **Denominations**: All on-chain output values must be strictly powers of 2 (e.g., 1, 2, 4, 8...). The wallet abstracts this away, allowing users to send arbitrary amounts which are automatically decomposed into multiple power-of-2 outputs. This naturally obfuscates exact transaction amounts.
- **Block Reward**: Starts at 16.
- **Halving**: Reward halves every 525,600 blocks (~1 year at 1-min blocks). Minimum reward: 1.
- **Fees**: `Sum(Inputs) - Sum(Outputs)`. Minimum fee for reveals: 1.

## 5. Transactions & Mempool

Transactions use a two-phase Commit-Reveal scheme to separate intent from execution, enhancing privacy and thwarting front-running. The mempool enforces Replace-By-Fee (RBF) for eviction when full.

### Phase 1: Commit
- **Payload**: `commitment = BLAKE3(input_coin_ids... || output_coin_ids... || salt)`.
- **Action**: Commitment is added to the SMT. Requires a minor PoW spam nonce.
- **Cost**: 0 fee.

### Phase 2: Reveal
- **Payload**: Preimages (Owner PK, Value, Salt) for spent coins, Signatures, and new Output definitions.
- **Validation**: Signatures must be valid against `owner_pk`. Output values must be powers of 2. `Sum(Input) > Sum(Output)`.

### Native CoinJoin Mixing
- The strict power-of-2 denominations enable perfect, subset-sum-resistant CoinJoins.
- Nodes negotiate mix sessions via P2P. A valid mix requires all inputs and outputs (except a single denomination-1 fee input) to share the exact same denomination.

## 6. Networking & Synchronization

- **Transport**: TCP and QUIC via `libp2p`.
- **Encryption & Muxing**: Noise protocol + Yamux.
- **Discovery**: Kademlia DHT + Identify + custom Peer Exchange.
- **NAT Traversal**: Built-in AutoNAT, DCUtR (Hole-punching), and Circuit Relays.
- **Header-First Sync**: Nodes download and verify lightweight `BatchHeader`s from genesis to find deterministic fork points *before* downloading full batches. This prevents "Frankenstein" chain corruptions and catch-up death spirals.
