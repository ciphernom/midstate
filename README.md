# Midstate: Complete Sequential-Time Cryptocurrency

A production-ready implementation of the Midstate protocol with full P2P networking, RPC API, metrics, and persistence.

## Features

✅ Pure SHA-256 consensus (quantum-safe)
✅ Sequential time-based mining
✅ Bearer asset model (no accounts)
✅ Complete RPC API
✅ Full P2P networking with handshakes
✅ Mempool persistence
✅ Automatic peer reconnection
✅ Ping/pong heartbeat
✅ Metrics and monitoring
✅ Comprehensive tests
✅ Multi-node support

## Quick Start

### Terminal 1: Mining node
```bash
cargo run --release -- node \
    --port 9333 \
    --rpc-port 8545 \
    --mine \
    --data-dir ./data1
```

### Terminal 2: Peer node
```bash
cargo run --release -- node \
    --port 9334 \
    --rpc-port 8546 \
    --peer 127.0.0.1:9333 \
    --data-dir ./data2
```

### Terminal 3: Another peer
```bash
cargo run --release -- node \
    --port 9335 \
    --rpc-port 8547 \
    --peer 127.0.0.1:9333 \
    --peer 127.0.0.1:9334 \
    --data-dir ./data3
```

## Using the System

### Generate a keypair
```bash
cargo run -- keygen
```

### Check state
```bash
cargo run -- state --rpc-port 8545
```

### Check mempool
```bash
cargo run -- mempool --rpc-port 8545
```

### Check if a coin exists
```bash
cargo run -- balance --coin <COIN_HEX> --rpc-port 8545
```

### Send a transaction
```bash
# Genesis secrets for testing:
# genesis_coin_1: 67656e657369735f636f696e5f31
# genesis_coin_2: 67656e657369735f636f696e5f32
# genesis_coin_3: 67656e657369735f636f696e5f33

# Generate destination
cargo run -- keygen

# Send
cargo run -- send \
    --secret 67656e657369735f636f696e5f31 \
    --dest <NEW_COIN_HEX> \
    --rpc-port 8545
```

## Testing
```bash
cargo test
```

## Architecture

- **Pure SHA-256** - No elliptic curves, quantum-safe
- **Sequential mining** - 1M iterations per extension (~1 second)
- **UTXO model** - No accounts, bearer assets
- **Difficulty lottery** - Like Bitcoin PoW
- **Longest depth** - Fork resolution

## License

GNU
