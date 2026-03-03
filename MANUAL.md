# Midstate Operational Manual

This document details the operational mechanics, configuration, disaster recovery, and troubleshooting procedures for a Midstate node and wallet.

## 1. Directory Structure and Configuration

Midstate separates node data from wallet data.

### Node Directory (`./data` by default)

* `db/`: The Redb database containing the state accumulator (`state.redb`) and the `batches/` directory for historical block storage.
* `config.toml`: Node P2P configuration file.
* `miner.toml`: Mining configuration file (Solo/Pool mode, pool URLs, payout addresses).
* `peer_key`: The libp2p identity key. This determines the node's `PeerId`.
* `coinbase_seeds.jsonl`: Append-only log of mined solo block rewards.
* `mining_seed`: The persistent seed used to generate deterministic solo coinbase outputs.

### Wallet Directory (`~/.midstate/` by default)

* `wallet.dat`: Encrypted JSON payload containing HD seeds, WOTS keys, MSS trees, unspent coin data, and transaction history.

## 2. Backup and Disaster Recovery

The `wallet.dat` file is encrypted using AES-256-GCM. The encryption key is derived from the password via Argon2id. Passwords cannot be recovered.

### Primary Backup: BIP39 Seed Phrase

By default, wallets are Hierarchical Deterministic (HD). The 24-word phrase generated at creation is your primary backup. **Do not lose it.**

To restore an HD wallet to a new machine:
```bash
midstate wallet restore --path wallet.dat

```

*Note: Because MSS keys are stateful, restoring a wallet requires scanning the blockchain. The node will automatically query the chain to fast-forward your MSS `leaf_index` to prevent cryptographic key reuse.*

### Legacy/Advanced: Raw Data Extraction

If dealing with a legacy (non-HD) wallet or for specific offline backups, extract the raw coin data:

```bash
midstate wallet export --path wallet.dat --coin <COIN_ID_HEX>

```

To rebuild a single coin from raw data:

```bash
midstate wallet import --path wallet.dat --seed <SEED_HEX> --value <INTEGER> --salt <SALT_HEX>

```

## 3. Node Maintenance and Troubleshooting

### Thread Management

By default, the solo miner utilizes all available CPU cores. This can starve the asynchronous network executor (Tokio) on low-power devices.
Use the `--threads` flag to restrict the miner and reserve CPU time for network keep-alives (e.g., on a 4-core Raspberry Pi, use 3 threads).

```bash
midstate node --mine --threads 3

```

### Unrecoverable Coins (Address Reuse)

WOTS addresses are strictly single-use. If a sender transmits funds to a WOTS address that has already been spent from, the wallet will ignore the incoming transaction to protect the key. The second coin becomes "burnt".

## 4. Privacy Mechanics

### Smart Contract Auto-Solving

Because every address is a smart contract, the wallet performs Ahead-Of-Time (AOT) decompilation when you attempt to spend a coin. If the wallet detects a matching public key, it automatically generates the required WOTS or MSS signature and injects it into the execution stack.

### Private Sends

The standard `send` command aggregates inputs. The `--private` flag splits the transaction into independent, denomination-specific transactions with decoy change outputs, thwarting chain-analysis linking.

```bash
midstate wallet send --to <ADDRESS>:15 --private

```

### CoinJoin P2P Timeouts

CoinJoin (`wallet mix`) operates in phases: `Collecting`, `Signing`, and `CommitSubmitted`. If a peer disconnects or fails to provide a signature, they are banned from the pool, and the session halts. Because the final `Reveal` transaction is never broadcast, your inputs remain unspent and safe.

## 5. Midstate Axe & Hardware Operations

Midstate is engineered to run on bare-metal ARM hardware (e.g., Raspberry Pi Zero 2 W).

When the node is running, you can access the local hardware dashboard by navigating a browser to:
`http://127.0.0.1:8545/axe` (or the IP of the device on your local network).

From the dashboard, you can:

1. Reconfigure network Wi-Fi settings (Captive Portal).
2. Toggle between Solo Mining and Decentralized Pool Mining.
3. Apply hardware-level overclocks. **Warning: Do not apply the 'Force Turbo' overclock without a physical copper/aluminum heatsink attached to the SoC, or thermal throttling will severely degrade node performance.**
