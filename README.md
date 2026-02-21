# Midstate

Midstate is a minimal, post-quantum cryptocurrency. It uses a unique "sequential-time" proof of work and requires exact power-of-2 coin values under the hood (though the wallet handles the math for you). 

This guide covers everything you need to know to compile the software, run a node, mine, and use the wallet securely.

---

## 1. Installation & Building

You will need the [Rust toolchain](https://rustup.rs/) installed.

Clone the repository and build the project:
```bash
cargo build --release

```

The compiled binary will be located at `./target/release/midstate`.

*(Developer Note: If you want to test the network without waiting for the full 1-million hash iterations per block, build with `cargo build --release --features fast-mining`)*.

---

## 2. Running a Node

To interact with the network, you need to run a node. Nodes maintain the blockchain state, relay transactions, and handle mining.

### Starting your first node (Miner / Seed)

```bash
midstate node --data-dir ./node-data --port 9333 --rpc-port 8545 --mine

```

* `--data-dir`: Where the blockchain and node settings are saved.
* `--port`: The P2P network port (for connecting to other nodes).
* `--rpc-port`: The local port your wallet uses to talk to your node.
* `--mine`: Tells the node to continuously mine new blocks.

**Important:** When the node starts, look at the terminal output for your **Peer ID** (it looks like `/ip4/127.0.0.1/tcp/9333/p2p/12D3KooW...`). You will need this to connect other nodes to your network.

### Connecting a second node to the network

```bash
midstate node --data-dir ./node2-data --port 9334 --rpc-port 8546 --peer <PEER_ID_FROM_NODE_1>

```

---

## 3. Wallet Basics

Midstate uses post-quantum cryptography. This introduces a very important rule: **standard addresses can only be used to receive funds ONCE.** *(All wallet commands require a password. You will be prompted in the terminal, or you can set the `MIDSTATE_PASSWORD` environment variable).*

### Create a new wallet

```bash
midstate wallet create --path my_wallet.dat

```

### Checking your balance

Your wallet needs to ask your local node about the status of your coins. Make sure your node is running!

```bash
# List all individual coins and unused addresses
midstate wallet list --path my_wallet.dat --rpc-port 8545

# Show total spendable balance
midstate wallet balance --path my_wallet.dat --rpc-port 8545

```

---

## 4. Receiving Coins

Because standard addresses are strictly one-time use, Midstate offers two ways to receive funds:

### Option A: Standard One-Time Address (WOTS)

Generates a highly secure, single-use address.

```bash
midstate wallet receive --path my_wallet.dat --label "payment from Alice"

```

**Warning:** Never let someone send coins to a standard address twice. If you spend from it twice, the private key becomes mathematically compromised.

### Option B: Reusable Address (MSS)

If you need an address to post publicly (like on a website or profile), generate a Merkle Signature Scheme (MSS) address.

```bash
midstate wallet generate-mss --path my_wallet.dat --height 10 --label "donation address"

```

* `--height 10` means this address can safely sign exactly  (1,024) transactions before it is exhausted.
* Note: Generating high-capacity MSS addresses (height > 14) can take a few minutes.

### Scanning for Incoming Coins

Because Midstate is privacy-focused, your wallet doesn't automatically know when someone sends you money. You must scan the blockchain to find your incoming coins:

```bash
midstate wallet scan --path my_wallet.dat --rpc-port 8545

```

*Always run a scan before sending money to ensure your wallet's internal security indices are synced with the network.*

---

## 5. Sending Coins

Midstate natively uses a two-phase "Commit and Reveal" system to prevent network front-running, and it forces all coins to be exact powers of 2. **The wallet completely abstracts this for you.**

Send any integer amount to an address:

```bash
midstate wallet send --path my_wallet.dat --rpc-port 8545 --to <ADDRESS_HEX>:15

```

*The wallet will automatically break the `15` down into `8 + 4 + 2 + 1` behind the scenes, calculate change, submit the "Commit" transaction, wait for it to be mined, and then submit the final "Reveal" transaction.*

### Private Send

If you want to hide the link between your inputs and outputs, use the `--private` flag.

```bash
midstate wallet send --path my_wallet.dat --rpc-port 8545 --to <ADDRESS_HEX>:15 --private

```

This breaks your payment into completely independent, delayed transactions for each denomination. It costs slightly more in fees and takes longer to fully clear, but greatly enhances on-chain privacy.

---

## 6. Advanced Privacy: CoinJoin Mixing

Midstate has a built-in "Dark Pool" CoinJoin feature. It allows you to mix your coins with other users seamlessly over the P2P network, with no central coordinator.

Because Midstate coins are strictly powers of 2, these mixes are mathematically perfect—an observer cannot map inputs to outputs.

**Step 1: Announce a mix (User A)**
Decide on a power-of-2 denomination you want to mix (e.g., `8`).

```bash
midstate wallet mix --path my_wallet.dat --rpc-port 8545 --denomination 8

```

This will print a `mix_id`. Share this ID securely with the person you want to mix with.

**Step 2: Join a mix (User B)**

```bash
midstate wallet mix --path my_wallet.dat --rpc-port 8545 --denomination 8 --join <MIX_ID>

```

*Note: One user in the mix must append the `--pay-fee` flag to contribute a `1`-value coin to cover the network mining fee.*

The wallets will automatically negotiate, sign, and broadcast the completely anonymized transaction.

---

## 7. Claiming Mining Rewards

If you are running a node with the `--mine` flag, your node is saving your block rewards directly into a log file (`coinbase_seeds.jsonl`).

To make these coins spendable, import them into your wallet:

```bash
midstate wallet import-rewards --path my_wallet.dat --coinbase-file ./node-data/coinbase_seeds.jsonl

```

---

## CLI Command Cheat Sheet

### Node Commands

* `node` - Start the node daemon.

### Wallet Commands

* `wallet create` - Make a new wallet.
* `wallet receive` - Get a 1-time address.
* `wallet generate-mss` - Get a reusable address.
* `wallet list` - View your coins and unused keys.
* `wallet balance` - View aggregate balance.
* `wallet scan` - Find incoming payments on the blockchain.
* `wallet send` - Send funds to someone.
* `wallet mix` - Anonymize a coin via P2P CoinJoin.
* `wallet history` - View past transactions.
* `wallet pending` - View transactions waiting to be mined.
* `wallet import-rewards` - Claim mined coins.
* `wallet export` - Export raw coin data (seed, salt, value) for backups.
* `wallet import` - Manually import raw coin data.

### Low-Level RPC Commands (For debugging)

* `state` - View current blockchain height, difficulty, and midstate.
* `mempool` - View transactions waiting in the memory pool.
* `peers` - List connected network peers.
* `balance` - Check if a specific raw `Coin_ID` exists in the UTXO set.
