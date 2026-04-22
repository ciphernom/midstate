# Midstate

Midstate is a cryptocurrency implementing BLAKE3 sequential-time proof of work, post-quantum cryptography (WOTS and MSS), and strict power-of-2 coin denominations. Transactions operate on a two-phase commit-reveal protocol.

## 1. Build

Compilation requires the standard Rust toolchain.

```bash
cargo build --release

```

The binary is output to `./target/release/midstate`.

## 2. Node Operations

Nodes maintain the state accumulator, manage libp2p networking, and execute mining.

Start a node with mining enabled:

```bash
midstate node --data-dir ./data --port 9333 --rpc-port 8545 --mine --threads 3

```

**Native Batch Explorer**
Access the native explorer at `http://localhost:8545` to view detailed info - mempool, batches, height etc.

**Midstate Axe Dashboard:**
For hardware nodes (or local testing), access the web dashboard at `http://127.0.0.1:8545/axe` to configure Wi-Fi, view live telemetry, and set up pool mining.

## 3. Wallet Operations

The wallet communicates with the node via HTTP RPC. All wallet commands require a password, which is prompted interactively or read from the `MIDSTATE_PASSWORD` environment variable.

Create a new HD (BIP39) wallet:

```bash
midstate wallet create --path wallet.dat

```

Restore a wallet from a 24-word seed phrase:

```bash
midstate wallet restore --path wallet.dat

```

## 4. Addresses & Receiving

Midstate utilizes post-quantum signatures. Standard addresses are consumed upon spending.

**Generate a WOTS Address (Single-use):**

```bash
midstate wallet receive --path wallet.dat --label "payment1"

```

**Generate an MSS Address (Multi-use):**

```bash
midstate wallet generate-mss --path wallet.dat --height 10 --label "donation"

```

**Smart Contracts & Covenants:**
Compile a human-readable `.msc` assembly file into a Pay-to-Script-Hash (P2SH) address.

```bash
midstate wallet compile --file limit_order.msc

```

**Scan for Outputs:**
Incoming transactions must be scanned to update local balances:

```bash
midstate wallet scan --path wallet.dat --rpc-port 8545

```

## 5. Sending

Outputs must be powers of 2. The wallet automatically decomposes base-10 integers into power-of-2 denominations, computes change, and executes the required commit and reveal transactions.

**Standard Send:**

```bash
midstate wallet send --path wallet.dat --rpc-port 8545 --to <ADDRESS_HEX>:15

```

**Private Send:**

```bash
midstate wallet send --path wallet.dat --rpc-port 8545 --to <ADDRESS_HEX>:15 --private

```

The `--private` flag splits the payment into separate, independent transactions for each required denomination, preventing output linking.

## 6. P2P CoinJoin

Nodes coordinate uniform-denomination CoinJoin transactions over the p2p network to obfuscate UTXO lineage.

**Initiate a mix:**

```bash
midstate wallet mix --path wallet.dat --rpc-port 8545 --denomination 8

```

This outputs a `<MIX_ID>`.

**Join a mix:**

```bash
midstate wallet mix --path wallet.dat --rpc-port 8545 --denomination 8 --join <MIX_ID>

```

## 7. Mining Rewards

A solo mining node writes its deterministic coinbase seeds to a local log file. These must be imported to the wallet to be spent.

```bash
midstate wallet import-rewards --path wallet.dat --coinbase-file ./data/coinbase_seeds.jsonl

```

## CLI Reference

**Node**

* `node` - Start the node daemon.

**Wallet**

* `wallet create` - Initialize a new HD wallet.
* `wallet restore` - Restore an HD wallet from a 24-word seed phrase.
* `wallet receive` - Generate a WOTS address.
* `wallet generate-mss` - Generate an MSS address.
* `wallet compile` - Compile a `.msc` script.
* `wallet list` - Display controlled coins and unused keys.
* `wallet balance` - Display aggregate balance.
* `wallet scan` - Scan blockchain for incoming coins.
* `wallet send` - Construct and broadcast a transaction.
* `wallet mix` - Participate in a CoinJoin session.
* `wallet history` - Display past transactions.
* `wallet pending` - Display transactions awaiting block inclusion.
* `wallet import-rewards` - Import coinbase seeds from node logs.

**RPC (Debug)**

* `state` - Display chain height, depth, and midstate.
* `mempool` - Display pending transactions.
* `peers` - List active p2p connections.

---

# MidstateLabs


MidstateLabs will be the software company that continues to develop midstate. In addition to the ongoing development, it will be the company responsible for running the bootstraps, formal messaging the community, and selling the up-and-comping MidtateAxe hardware.  

By contributing to MidstateLabs startup fund, you will get yourself one of the first MidstateAxes.  

----
