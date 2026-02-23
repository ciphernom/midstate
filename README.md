# Midstate

Midstate is a cryptocurrency implementing BLAKE3 sequential-time proof of work, post-quantum cryptography (WOTS and MSS), and strict power-of-2 coin denominations. Transactions operate on a two-phase commit-reveal protocol.

## 1. Build

Compilation requires the standard Rust toolchain.

```bash
cargo build --release

```

The binary is output to `./target/release/midstate`.
To build with reduced hash iteration requirements for local testing, append `--features fast-mining`.

## 2. Node Operations

Nodes maintain the state accumulator, manage libp2p networking, and execute mining.

Start a node with mining enabled:

```bash
midstate node --data-dir ./data --port 9333 --rpc-port 8545 --mine --threads 2

```

* `--threads`: Restricts the number of CPU cores used by the miner. If omitted, all available cores are used.
* The node's multiaddr (e.g., `/ip4/127.0.0.1/tcp/9333/p2p/12D3...`) is printed to standard output on startup.

Connect a subsequent node:

```bash
midstate node --data-dir ./data2 --port 9334 --rpc-port 8546 --peer <PEER_MULTIADDR>

```

## 3. Wallet Operations

The wallet communicates with the node via HTTP RPC. All wallet commands require a password, which is prompted interactively or read from the `MIDSTATE_PASSWORD` environment variable.

Create a wallet:

```bash
midstate wallet create --path wallet.dat

```

Query local state:

```bash
midstate wallet list --path wallet.dat --rpc-port 8545
midstate wallet balance --path wallet.dat --rpc-port 8545

```

## 4. Addresses & Receiving

Midstate utilizes post-quantum signatures. Standard addresses are consumed upon spending.

**Generate a WOTS Address (Single-use):**

```bash
midstate wallet receive --path wallet.dat --label "payment1"

```

*Note: If multiple payments are sent to the same WOTS address, the wallet will only import the first one to prevent signature reuse. Subsequent coins sent to that address cannot be spent.*

**Generate an MSS Address (Multi-use):**

```bash
midstate wallet generate-mss --path wallet.dat --height 10 --label "donation"

```

The `--height` parameter defines tree depth. A height of 10 allows exactly 1,024 signatures before the address is exhausted.

**Scan for Outputs:**
Incoming transactions are not indexed automatically. You must scan the chain to update local balances:

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

The `--private` flag splits the payment into separate, independent transactions for each required denomination. This increases total network fees and processing time.

## 6. P2P CoinJoin

Nodes coordinate uniform-denomination CoinJoin transactions over the p2p network.

**Initiate a mix:**

```bash
midstate wallet mix --path wallet.dat --rpc-port 8545 --denomination 8

```

This outputs a `<MIX_ID>`.

**Join a mix:**

```bash
midstate wallet mix --path wallet.dat --rpc-port 8545 --denomination 8 --join <MIX_ID>

```

Exactly one participant in the session must supply the `--pay-fee` flag, which requires an available denomination-1 coin in their wallet to cover the network fee.

## 7. Mining Rewards

A mining node writes its deterministic coinbase seeds to a local log file. These must be imported to the wallet to be spent.

```bash
midstate wallet import-rewards --path wallet.dat --coinbase-file ./data/coinbase_seeds.jsonl

```

## CLI Reference

**Node**

* `node` - Start the node daemon.

**Wallet**

* `wallet create` - Initialize a new wallet file.
* `wallet receive` - Generate a WOTS address.
* `wallet generate-mss` - Generate an MSS address.
* `wallet list` - Display controlled coins and unused keys.
* `wallet balance` - Display aggregate balance.
* `wallet scan` - Scan blockchain for incoming coins.
* `wallet send` - Construct and broadcast a transaction.
* `wallet mix` - Participate in a CoinJoin session.
* `wallet history` - Display past transactions.
* `wallet pending` - Display transactions awaiting block inclusion.
* `wallet import-rewards` - Import coinbase seeds from node logs.
* `wallet export` - Display raw coin data (seed, salt, value).
* `wallet import` - Import raw coin data.

**RPC (Debug)**

* `state` - Display chain height, depth, and midstate.
* `mempool` - Display pending transactions.
* `peers` - List active p2p connections.
* `balance` - Check if a specific Coin ID exists in the UTXO set.
