# Midstate

A minimal, post-quantum sequential-time cryptocurrency written completely in Rust.

## Features

- **Proof of Sequential Work:** BLAKE3-based sequential hash chain. Mining single proofs is inherently anti-ASIC and non-parallelizable.
- **Post-Quantum Signatures:** WOTS (Winternitz One-Time Signatures) and MSS (Merkle Signature Scheme) for reusable addresses.
- **Commit-Reveal Transactions:** Two-phase transaction model. Commitments are blinded hashes; reveals prove ownership and transfer value.
- **Power-of-2 Denominations:** All output values must be powers of 2 (1, 2, 4, 8, 16...).
- **Dynamic Finality:** Nakamoto-style longest-chain consensus enhanced by a Bayesian Beta-Binomial estimator to dynamically calculate safe confirmation depths.
- **Hostile-Network Ready:** Advanced `libp2p` networking featuring AutoNAT, DCUtR (hole-punching), and Circuit Relays to bypass strict firewalls.
- **State:** Merkle-committed UTXO Accumulator via a Sparse Merkle Tree (SMT).
- **Storage:** Hybrid — `redb` for hot chain state, flat files for immutable batch history.

## Build

```bash
cargo build --release

```

For faster mining during development/testing (reduces PoW iterations):

```bash
cargo build --release --features fast-mining

```

## Running a Local Testnet

**Terminal 1: Miner (Seed Node)**

Starts a node, mines blocks, and listens on port 9333.

```bash
./target/release/midstate node --data-dir ./node1 --port 9333 --rpc-port 8545 --mine

```

*Note:* Check `./node1/node.log` or your terminal output to find this node's `libp2p` Peer ID (e.g., `12D3KooW...`). You will need this to connect other peers.

**Terminal 2: Peer**

Connects to the miner, syncs the chain, and listens on port 9334. Replace `<PEER_ID>` with the ID from Terminal 1.

```bash
./target/release/midstate node --data-dir ./node2 --port 9334 --rpc-port 8546 --peer /ip4/127.0.0.1/tcp/9333/p2p/<PEER_ID>

```

*(Alternatively, you can place bootstrap multiaddrs inside `node2/config.toml`)*.

## Wallet Usage

All wallet commands require a password.

**1. Create a Wallet**

```bash
midstate wallet create --path wallet.dat

```

**2. Generate a Receiving Address (one-time WOTS key)**

```bash
midstate wallet receive --path wallet.dat

```

**3. Generate a Reusable Address (MSS Merkle tree)**

Creates an address that can sign up to `2^height` transactions (default height 10 = 1024 signatures).

```bash
midstate wallet generate-mss --path wallet.dat --height 10 --label "main"

```

**4. Check Balance & Coin Status**

Lists wallet coins and checks which are live on-chain via the node's RPC.

```bash
midstate wallet list --path wallet.dat --rpc-port 8545
midstate wallet balance --path wallet.dat --rpc-port 8545

```

**5. Send Coins**

Send value `4` to an address. Values must be powers of 2.

```bash
midstate wallet send --path wallet.dat --rpc-port 8545 --to <ADDRESS_HEX>:4

```

The wallet handles coin selection, change output creation, and the full commit-reveal flow automatically. For enhanced privacy (randomized timing, independent per-denomination transactions):

```bash
midstate wallet send --path wallet.dat --rpc-port 8545 --to <ADDRESS_HEX>:4 --private

```

**6. Receive Coins**

Share your address with the sender (from `wallet receive`). Once their transaction is mined, scan the chain:

```bash
midstate wallet scan --path wallet.dat --rpc-port 8545

```

The wallet automatically detects incoming coins by matching on-chain reveal outputs to your addresses.

Manual import is still available if needed (e.g., for offline wallets):

```bash
midstate wallet import --path wallet.dat --seed <SEED_HEX> --value <AMOUNT> --salt <SALT_HEX>

```

**7. Import Mining Rewards**

If you ran with `--mine`, import your coinbase rewards from the miner's log.

```bash
midstate wallet import-rewards --path wallet.dat --coinbase-file ./node1/coinbase_seeds.jsonl

```

**8. Other Wallet Commands**

```bash
midstate wallet generate --path wallet.dat --count 5     # Batch-generate receiving keys
midstate wallet export --path wallet.dat --coin <ID>     # Export coin details (seed, value, salt)
midstate wallet pending --path wallet.dat                # Show uncommitted reveals
midstate wallet reveal --path wallet.dat --rpc-port 8545 # Manually broadcast pending reveals
midstate wallet history --path wallet.dat --count 20     # Transaction history

```

## CLI Reference

| Command | Description |
| --- | --- |
| `node` | Run the full node (`--mine` to enable mining, `--peer` to connect) |
| `wallet create` | Create a new encrypted wallet |
| `wallet receive` | Generate a one-time WOTS receiving address |
| `wallet generate` | Batch-generate multiple receiving keys |
| `wallet generate-mss` | Generate a reusable MSS address |
| `wallet list` | List coins and keys (with on-chain status) |
| `wallet balance` | Show aggregate balance |
| `wallet send` | Send coins (handles commit-reveal automatically) |
| `wallet scan` | Scan chain for incoming coins to your addresses |
| `wallet import` | Import a coin from seed + value + salt |
| `wallet export` | Export coin details for off-chain transfer |
| `wallet import-rewards` | Import coinbase rewards from mining log |
| `wallet pending` | Show pending (uncommitted) reveals |
| `wallet reveal` | Manually broadcast pending reveals |
| `wallet history` | Show transaction history |
| `commit` | Submit a raw commitment (low-level) |
| `balance` | Check if a specific coin ID exists on-chain |
| `state` | Show chain height, depth, difficulty, and reward |
| `mempool` | Show pending transactions in the mempool |
| `peers` | List connected peers |
| `keygen` | Generate a standalone WOTS keypair |
| `sync` | Sync chain from genesis via a peer |
