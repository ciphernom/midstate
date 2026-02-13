# Midstate Protocol Design

## 1. Fundamentals

* **Hash Function**: BLAKE3 (32-byte output).
* **Architecture**: Sequential-time blockchain (effectively Nakamoto consensus).
* **State Model**: UTXO Accumulator (Merkle-committed sorted vector).
* **Time**: Unix timestamp (seconds).

## 2. Cryptography

### Coin IDs
A Coin ID is the hash of its properties. It is the only identifier stored in the state.
`CoinID = BLAKE3(owner_pk || value_le_bytes || salt)`

### WOTS (Winternitz One-Time Signatures)
* **Parameter**: `w=16`.
* **Structure**: 16 message chains + 2 checksum chains = 18 chains total.
* **Max Digit**: 65,535.
* **Signature Size**: 576 bytes.
* **Security**: Post-quantum secure (stateful). One-time use only.

### MSS (Merkle Signature Scheme)
* **Structure**: Binary Merkle tree of WOTS keys.
* **Root**: Master Public Key (32 bytes).
* **Leaf**: WOTS public key.
* **Usage**: Allows `2^Height` signatures per public key.
* **Default Height**: 10 (1024 signatures).

## 3. Consensus & Mining

### Sequential Proof of Work
Mining is not parallelizable. It requires sequential iteration to prove time has passed.

1.  **Input**: `prev_midstate` + `coinbase_hash`.
2.  **Work**: Iteratively hash `checkpoint[i-1]` -> `checkpoint[i]` for `N` iterations.
3.  **Result**: `final_hash`.
4.  **Target**: `final_hash` must be `< target`.
5.  **Verification**: Verifiers spot-check random segments of the hash chain rather than recomputing the whole chain.

### Constants
* **Block Time**: 600 seconds (10 minutes).
* **Difficulty Adjustment**: Every 2016 blocks (~2 weeks).
* **Adjustment Limit**: Max 4x increase or decrease per period.
* **Anchor**: Genesis anchored to Bitcoin block 935897.

## 4. Economics

* **Unit**: Integer values only.
* **Denominations**: All output values must be powers of 2 (e.g., 1, 2, 4, 8...).
* **Block Reward**: Starts at 16.
* **Halving**: Reward halves every 52,560 blocks (~1 year). Minimum reward 1.
* **Fees**: `Sum(Inputs) - Sum(Outputs)`.

## 5. Transactions

Transactions use a two-phase Commit-Reveal scheme to separate intent from execution.

### Phase 1: Commit
* **Payload**: `commitment = BLAKE3(input_ids... || output_ids... || salt)`.
* **Action**: Miner includes commitment in state.
* **Cost**: 0 fee.

### Phase 2: Reveal
* **Payload**:
    * **Inputs**: Preimages (Owner PK, Value, Salt) for spent coins.
    * **Signatures**: WOTS or MSS signatures verifying ownership.
    * **Outputs**: New Coin definitions (Owner PK, Value, Salt).
    * **Salt**: The blinding factor used in Phase 1.
* **Validation**:
    * Commitment must exist in state.
    * Inputs must exist in UTXO set.
    * Signatures must be valid.
    * `Input Value > Output Value` (conservation of value + fee).

## 6. Networking

* **Transport**: TCP.
* **Encryption**: Noise protocol (ChaCha20-Poly1305).
* **Multiplexing**: Yamux.
* **Discovery**: Kademlia DHT.
* **Protocol ID**: `/midstate/1.0.0`.
* **Sync**: Header-first (Batches), followed by full verification.
