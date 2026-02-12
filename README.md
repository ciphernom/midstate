# Midstate

A minimal, post-quantum sequential-time cryptocurrency written in Rust.

## Features
* **Proof of Sequential Work:** Blake3-based sequential work.
* **Signatures:** Post-quantum WOTS (Winternitz One-Time Signatures) and MSS (Merkle Signature Scheme).
* **Consensus:** Nakamoto consensus with reorg handling.
* **State:** Merkle-based UTXO Accumulator.
* **Storage:** `redb` database.
* **Networking:** `libp2p` (noise encryption, yamux).

## Build

```bash
cargo build --release

```

## Running a Local Testnet

**Terminal 1: Miner**
Starts a node, mines blocks, and listens on port 9333.

```bash
./target/release/midstate node --data-dir ./node1 --port 9333 --rpc-port 8545 --mine

```

**Terminal 2: Peer**
Connects to the miner, syncs the chain, and listens on port 9334.

```bash
./target/release/midstate node --data-dir ./node2 --port 9334 --rpc-port 8546 --peer 127.0.0.1:9333

```

## Wallet Usage

All wallet commands require a password.

**1. Create a Wallet**

```bash
./target/release/midstate wallet create --path wallet.dat

```

**2. Generate Receiving Address**

```bash
./target/release/midstate wallet receive --path wallet.dat

```

**3. Check Balance & Status**
Checks the local wallet coins against the running node to see if they are live.

```bash
./target/release/midstate wallet list --path wallet.dat --rpc-port 8545

```

**4. Send Coins**
Send amount `4` to an address.
*Note: This is a privacy coin. The sender must communicate the resulting Coin details (Seed, Value, Salt) to the recipient off-chain, otherwise the recipient cannot detect the funds.*

```bash
./target/release/midstate wallet send --path wallet.dat --rpc-port 8545 --to <ADDRESS_HEX>:4

```

**5. Receive Incoming Coins**
Because the chain only stores hashed commitments (`CoinID`), you cannot scan the chain for payments. The sender must provide you with the **Seed**, **Value**, and **Salt**.

```bash
./target/release/midstate wallet import --path wallet.dat --seed <SEED_HEX> --value <AMOUNT> --salt <SALT_HEX>

```

**6. Import Mining Rewards**
If you ran with `--mine`, import your coinbase rewards from the miner's log.

```bash
./target/release/midstate wallet import-rewards --path wallet.dat --coinbase-file ./node1/coinbase_seeds.jsonl

```

## CLI Reference

* `node`: Run the full node.
* `wallet`: Manage keys and transactions.
* `peers`: List connected peers.
* `state`: Show chain height and difficulty.
* `mempool`: Show pending transactions.
