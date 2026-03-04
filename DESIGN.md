# Midstate Protocol Design

## 1. Fundamentals

- **Hash Function**: BLAKE3 (32-byte output).
- **Architecture**: Sequential-time blockchain. Optimized for edge hardware (e.g., Raspberry Pi Zero 2 W).
- **State Model**: UTXO Accumulator via Sparse Merkle Tree (SMT).
- **Time**: Unix timestamp (seconds).
- **Storage**: Hybrid. `redb` (Key-Value) for hot chain state and mempool; flat `.bin` files chunked in directories for immutable batch history. Checkpoints are automatically pruned for deep history to save ~98% of disk space.

## 2. Cryptography

### Coin IDs & Addresses

A Coin ID is the hash of its properties. It is the only identifier stored in the state.

`CoinID = BLAKE3(address || value_le_bytes || salt)`

**Pay-to-Script-Hash (P2SH):** Every address in Midstate is the hash of a compiled MidstateScript bytecode payload.
`address = BLAKE3(script_bytecode)`

### HD Wallets (BIP39)
Midstate wallets are Hierarchical Deterministic (HD). A 24-word BIP39 mnemonic derives a master seed. From this master seed, infinite WOTS and MSS child seeds are derived using domain-separated BLAKE3 hashes (e.g., `BLAKE3("midstate/wots/v1" || master_seed || index)`). No elliptic curve math (BIP32) is required.

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
- **Safety**: The signature strictly binds the `leaf_index` to the cryptographic authentication path. The wallet statefully tracks the `next_leaf` index and actively syncs with the node to prevent catastrophic key reuse.

## 3. Consensus & Mining

### Sequential Proof of Work (VDF-style)

Mining requires sequential iteration to prove time has passed. While miners can search for nonces in parallel, the computation of a single proof is strictly single-threaded, neutralizing ASIC advantages and enforcing "one CPU, one vote".

### Decentralized Pool Mining
Midstate reverses traditional Stratum-style pools to eliminate censorship. 
1. The local node autonomously selects transactions from its own mempool and builds the block template. 
2. The node hardcodes the block reward to pay the **Pool's MSS Address**. 
3. The miner's personal payout address is cryptographically watermarked into the coinbase `salt`. 
4. The pool operator receives the share, verifies the watermark, and distributes rewards, but has zero power to censor network transactions.

### Difficulty Adjustment (ASERT)

Midstate uses the Absolutely Scheduled Exponentially Decaying (ASERT) algorithm.
- Adjusts continuously on every block based on absolute elapsed time since genesis.
- Completely eliminates sliding-window exploits (time-warp, hash-and-flee, echo effects).

### Fork Resolution & Bayesian Finality

- **Longest Chain Rule**: The chain with the highest cumulative sequential `depth` wins.
- **Finality Estimator**: Nodes track network health using a Beta-Binomial probabilistic model. By observing honest extensions vs. adversarial orphans, nodes dynamically calculate the required block depth (`safe_depth`) to achieve a 1-in-a-million (`1e-6`) mathematical risk of a successful reorg.

## 4. Economics

- **Unit**: Integer values only.
- **Denominations**: All on-chain output values must be strictly powers of 2 (e.g., 1, 2, 4, 8...). 
- **Block Reward**: Starts at 1,073,741,824 (2^30).
- **Halving**: Reward halves every 525,600 blocks (~1 year at 1-min blocks). Minimum reward: 1.
- **Fees**: `Sum(Inputs) - Sum(Outputs)`. Minimum fee for reveals: 1.

## 5. Transactions & Mempool

Transactions use a two-phase Commit-Reveal scheme to separate intent from execution. The mempool enforces Replace-By-Fee (RBF) for eviction when full, and strictly bounds memory capacity to allow node hosting on 512MB RAM devices.

### Phase 1: Commit
- **Payload**: `commitment = BLAKE3(input_coin_ids... || output_coin_ids... || salt)`.
- **Action**: Commitment is added to the SMT. Requires a minor PoW spam nonce.
- **Cost**: 0 fee.

### Phase 2: Reveal
- **Payload**: Preimages (Script Bytecode, Value, Salt) for spent coins, Stack Witnesses (e.g., Signatures, Preimages), and new Output definitions.

### Native CoinJoin Mixing
- The strict power-of-2 denominations enable perfect, subset-sum-resistant CoinJoins.
- Nodes negotiate mix sessions via P2P. A valid mix requires all inputs and outputs (except a single denomination-1 fee input) to share the exact same denomination.

## 6. Networking & Synchronization

- **Transport**: TCP and QUIC via `libp2p`.
- **Encryption & Muxing**: Noise protocol + Yamux.
- **Discovery**: Kademlia DHT + Identify + custom Peer Exchange.
- **Header-First Sync**: Nodes download and verify lightweight `BatchHeader`s from genesis to find deterministic fork points *before* downloading full batches.

## 7. MidstateScript Virtual Machine

Midstate abandons hardcoded transaction types in favor of a Turing-incomplete stack machine. 
- **Zero Gas**: No loops or backward jumps. 
- **Post-Quantum Native**: The VM's memory limits accommodate massive post-quantum signatures.
