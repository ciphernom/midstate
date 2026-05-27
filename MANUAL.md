# Midstate Operational Manual

This document details the operational mechanics, configuration, disaster recovery, and troubleshooting procedures for a Midstate node and wallet.

## 1. Directory Structure and Configuration

Midstate separates node data from wallet data to ensure that running a public node does not expose private keys.

### Node Directory (`./data` by default)

* `db/`: The Redb database containing the state accumulator (`state.redb`), historical blocks (`batches/`), and the `snapshots/` directory used for instant deep-reorg recovery.
* `config.toml`: Node P2P configuration file (contains bootstrap peers and your persistent `peer_id`).
* `miner.toml`: Mining configuration file (Solo/Pool mode, pool URLs, payout addresses).
* `peer_key`: The libp2p identity key. This ensures your node maintains a static `PeerId` across restarts for Bayesian routing reputation.
* `coinbase_seeds.jsonl`: Append-only log of mined solo block rewards.
* `mining_seed.key`: The persistent master seed used to generate deterministic solo coinbase outputs.

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

### Mining Reward Recovery

If you solo-mine blocks, the rewards are tied to the node's `mining_seed.key`, *not* your wallet's HD seed. To sweep mined block rewards into your wallet, run:
```bash
midstate wallet import-rewards --coinbase-file ./data/coinbase_seeds.jsonl --data-dir ./data
```

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

By default, the solo miner utilizes all available CPU cores. This can starve the asynchronous network executor (Tokio) on low-power devices, causing peer disconnects. Use the `--threads` flag to restrict the miner and reserve CPU time for network keep-alives (e.g., on a 4-core Raspberry Pi, use 3 threads). You can also restrict verification threads to prevent CPU spikes during heavy chain syncs.

```bash
midstate node --mine --threads 3 --verify-threads 2
```

### Archival vs Pruned Mode

Midstate nodes can run in two modes regarding historical data:

- **Archival mode** (default): The node keeps the full history of all blocks. This is required for serving new peers that want to sync from genesis. At least a few archival nodes must exist on the network.

- **Pruned mode**: The node automatically deletes block data older than `PRUNE_DEPTH` (currently 1000 blocks). This dramatically reduces disk usage (roughly 98% savings on historical data) at the cost of not being able to serve old blocks to others.

**CLI flag:**

```bash
midstate node --prune
```

**Config file (`config.toml`):**

```toml
prune = true
```

When `--prune` is passed on the command line, it takes precedence over the value in `config.toml`.

**Startup messages:**

- Archival: `Running in archival mode (full history retained, PRUNE_DEPTH = 1000)`
- Pruned: `Running in pruned mode (retaining only the last 1000 blocks of history)`

**Recommendation:** Most operators should run in archival mode unless disk space is a serious constraint. The network relies on having enough archival nodes for bootstrapping new participants.

Pruning is safe from a consensus perspective because blocks older than `PRUNE_DEPTH` have their UTXO state fully represented in the rolling accumulator. However, pruned nodes cannot help new peers perform a full historical sync.

### WOTS Address Reuse & Co-spending (Consolidation)

WOTS addresses are strictly single-use. The protocol enforces this at the consensus level. 

* **Siblings:** If you receive multiple payments to the same WOTS address *before* spending from it, the wallet creates "Sibling" UTXOs. To prevent key reuse, the wallet enforces a **Co-spend Rule**: you must spend all siblings in the exact same transaction. 
* **Dust Sweeping:** If you accumulate too many siblings (exceeding the `MAX_TX_INPUTS` limit of 256), you must sweep them into a reusable MSS address using the consolidate command:
  ```bash
  midstate wallet consolidate --address <WOTS_ADDRESS>
  ```
* **Quarantine:** If a sender transmits funds to a WOTS address that has *already* been spent from, the wallet will quarantine the incoming transaction to protect your private key. The second coin is mathematically unspendable (burnt).

### Database Corruption & Self-Healing

Midstate employs an ultra-fast $O(\log N)$ periodic health check. If the node loses power abruptly and the `Redb` state accumulator falls out of sync with the block headers, the node will log a `CRITICAL` error and initiate a **Self-Healing Rollback**. It will automatically rewind the chain state to the last safe `snapshot` (taken every 100 blocks) and replay the blocks to repair the database without requiring a full resync from Genesis.

## 4. Privacy Mechanics

### Smart Contract Auto-Solving

Because every address is a smart contract, the wallet performs Ahead-Of-Time (AOT) decompilation when you attempt to spend a coin. If the wallet detects a matching public key, it automatically generates the required WOTS or MSS signature and injects it into the execution stack.

### Private Sends

The standard `send` command aggregates inputs. The `--private` flag splits the transaction into independent, denomination-specific transactions with decoy change outputs, thwarting chain-analysis linking.

```bash
midstate wallet send --to <ADDRESS>:15 --private
```

### Auto-Shatter & Drip Mixing (`automix`)

For maximum anonymity, Midstate includes an automated dark-pool routing command. `automix` takes a large UTXO, shatters it into standard power-of-2 denominations, and stochastically "drips" them into active P2P CoinJoin pools over time with random delays, breaking temporal heuristics.

```bash
midstate wallet automix --coin <COIN_ID>
```

### CoinJoin P2P Timeouts

CoinJoin (`wallet mix`) operates in phases: `Collecting`, `Signing`, and `CommitSubmitted`. If a peer disconnects or fails to provide a signature within 60 seconds, they are penalized via the node's Bayesian routing system and temporarily banned from the pool, halting the session. Because the final `Reveal` transaction is never broadcast, your inputs remain unspent and cryptographically safe.

## 5. Midstate Axe & Hardware Operations

Midstate is engineered to run on bare-metal ARM hardware (e.g., Raspberry Pi Zero 2 W).

When the node is running, you can access the local hardware dashboard by navigating a browser to:
`http://127.0.0.1:8545/axe` (or the IP of the device on your local network).

From the dashboard, you can:

1. Reconfigure network Wi-Fi settings (Captive Portal).
2. Toggle between Solo Mining and Decentralized Pool Mining.
3. Apply hardware-level overclocks. **Warning: Do not apply the 'Force Turbo' overclock without a physical copper/aluminum heatsink attached to the SoC, or thermal throttling will severely degrade node performance and potentially damage the board.**
