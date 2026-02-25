# Midstate Operational Manual

This document details the operational mechanics, configuration, disaster recovery, and troubleshooting procedures for a Midstate node and wallet.

## 1. Directory Structure and Configuration

Midstate separates node data from wallet data.

### Node Directory (`./data` by default)

* `db/`: The Redb database containing the state accumulator (`state.redb`) and the `batches/` directory for historical block storage.
* `config.toml`: Node configuration file.
* `peer_key`: The libp2p identity key. This determines the node's `PeerId`.
* `coinbase_seeds.jsonl`: Append-only log of mined block rewards.
* `mining_seed`: The persistent seed used to generate deterministic coinbase outputs.

### Wallet Directory (`~/.midstate/` by default)

* `wallet.dat`: Encrypted JSON payload containing WOTS keys, MSS trees, unspent coin data, and transaction history.

### Static Peering (`config.toml`)

To persist connections across restarts, add peer multiaddrs to `config.toml`. The node will dial these addresses on boot.

```toml
bootstrap_peers = [
    "/ip4/203.0.113.10/tcp/9333/p2p/12D3KooW...",
    "/ip4/198.51.100.5/udp/9333/quic-v1/p2p/12D3KooW..."
]

```

## 2. Backup and Disaster Recovery

The `wallet.dat` file is encrypted using AES-256-GCM. The encryption key is derived from the password via Argon2id. Passwords cannot be recovered.

### Raw Data Extraction

If the wallet file is corrupted, or for offline paper backups, extract the raw coin data.

```bash
midstate wallet export --path wallet.dat --coin <COIN_ID_HEX>

```

Record the `Seed`, `Value`, and `Salt`. A coin can be reconstructed entirely from these three variables.

### Raw Data Restoration

To rebuild a coin from raw data:

```bash
midstate wallet import --path wallet.dat --seed <SEED_HEX> --value <INTEGER> --salt <SALT_HEX>

```

### MSS State Recovery

Merkle Signature Scheme (MSS) keys are stateful. Reusing a leaf index destroys the cryptographic security of the key. If you restore an old `wallet.dat` backup, the local leaf index will be outdated.

When you run `midstate wallet scan`, the wallet queries the node's `/mss_state` RPC endpoint. The node scans the chain and mempool for the highest used index for your MSS key. The wallet then fast-forwards its internal index plus a safety margin to prevent reuse.

## 3. Node Maintenance and Troubleshooting

### Sync States and Mining

A node cannot mine and sync simultaneously. If a peer broadcasts a longer chain, the node will abort the current mining task, download the headers, verify the PoW, calculate the fork point, and apply the new batches. Mining resumes automatically once the node reaches the chain tip.

### Thread Management

By default, the miner utilizes 100% of available CPU cores. This can starve the asynchronous network executor (Tokio), causing the node to drop peer connections.

Use the `--threads` flag to restrict the miner and reserve CPU time for network keep-alives.

```bash
midstate node --mine --threads 2

```

### Unrecoverable Coins (Address Reuse)

WOTS addresses are strictly single-use. If a sender transmits funds to a WOTS address that has already been spent from, the wallet will ignore the incoming transaction. Spending the second coin would reveal additional parts of the private key, compromising the funds. The second coin becomes "burnt" effectively being lost.

To verify if a specific coin exists in the state accumulator regardless of wallet recognition:

```bash
midstate balance --rpc-port 8545 --coin <COIN_ID_HEX>

```

## 4. Privacy Mechanics

### Smart Contract Auto-Solving

Because every address is a smart contract, the wallet performs Ahead-Of-Time (AOT) decompilation when you attempt to spend a coin.

If the wallet detects an `OP_CHECKSIGVERIFY` instruction preceded by a public key that matches a seed stored in your `wallet.dat`, it will automatically generate the required WOTS or MSS signature and inject it into the correct position on the VM execution stack.

### Private Sends

The standard `send` command aggregates inputs and produces a single transaction. The `--private` flag splits the transaction.

```bash
midstate wallet send --path wallet.dat --rpc-port 8545 --to <ADDRESS>:15 --private

```

If the wallet selects a 16-value coin to fund this, it must decompose the 15-value output into 8, 4, 2, and 1. The `--private` flag forces the wallet to execute four separate, independent Commit/Reveal cycles. This prevents chain observers from linking the exact outputs together, at the cost of higher network fees and extended execution time.

### CoinJoin P2P Timeouts

CoinJoin (`wallet mix`) operates in phases: `Collecting`, `Signing`, and `CommitSubmitted`.

If a peer disconnects or fails to provide a signature during the `Signing` phase, the session halts. The session will timeout after 300 seconds. Because the final `Reveal` transaction is never broadcast, the inputs remain unspent and can be used in a new transaction immediately.
