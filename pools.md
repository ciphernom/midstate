# Midstate Stratum Pool Guide

Midstate's Stratum implementation includes a Merkle precommitment in every block template. Miners automatically verify inclusion of their accumulated shares before hashing. This prevents pool operators from skimming rewards or stealing blocks.

## Critical Requirement: Use MSS Addresses

Midstate uses hash-based one-time signatures (WOTS) by default. Because mining pools issue many separate payouts over time, **you MUST use a reusable MSS address** to receive them. 

If you provide the pool with a standard WOTS address, your funds will become unspendable (burned) due to the network's one-time-use security rules.

Do not use `wallet receive`. Always use `wallet generate-mss` for pool configurations.

---

## Part 1: Running a Pool Server

Running a pool requires a synced Midstate backend node and the Stratum pool server.

### 1. Generate an MSS Pool Address
The pool takes a percentage fee from blocks found by its miners. Generate a reusable MSS address to receive these fees. Height 14 allows 16,384 payouts.
```bash
midstate wallet generate-mss --height 14 --label "Pool Fee Wallet"
```
*Save this address as `<POOL_ADDRESS>`.*

### 2. Start the Backend Node
Start your core node and expose the RPC interface. Do not use the `--mine` flag; the pool server manages mining coordination.
```bash
midstate node --rpc-bind 0.0.0.0 --rpc-port 8545
```
Wait for the node to fully sync to the network tip.

### 3. Start the Stratum Pool
In a new terminal, start the pool server. Point it at your backend node and supply your pool address.

**Standard startup (1% default fee):**
```bash
midstate pool --pool-address <POOL_ADDRESS>
```

**Custom fee and remote backend node:**
```bash
midstate pool --pool-address <POOL_ADDRESS> --fee 2.5 --rpc-host 10.0.0.5 --rpc-port 8545
```

**Port Binding:** 
The server will attempt to bind the Stratum TCP service to port `3333` and the HTTP Audit API to port `8081`. If these ports are in use, it will automatically increment the ports (e.g., `3334` and `8082`) until it finds an available pair. Check the startup logs to confirm the active ports. 

You must open/port-forward the Stratum TCP port (e.g., `3333`) on your firewall to accept external miners.

---

## Part 2: Mining on a Pool

You do not need to run a full node to mine. Midstate provides a standalone hashing client.

### 1. Generate an MSS Payout Address
Generate a reusable MSS address to receive your mining payouts. Height 10 allows 1,024 payouts.
```bash
midstate wallet generate-mss --height 10 --label "Mining Payouts"
```
*Save this address as `<PAYOUT_ADDRESS>`.*

### 2. Start the Miner
Run the `miner` command, providing the pool URL, your payout address, and the number of CPU threads to utilize.

```bash
midstate miner --pool-url stratum+tcp://<POOL_IP_ADDRESS>:3333 --payout-address <PAYOUT_ADDRESS> --threads 4
```
*(Omitting the `--threads` flag will utilize all available CPU cores).*

### 3. View the Dashboard
The miner runs silently by default. Press `[ENTER]` in the terminal at any time to print the live status dashboard:

```text
╔════════════ MINER STATUS ════════════╗
║ Hashrate:      301.20 nonces/s
║ Network:       2854.10 nonces/s
║ Your Share:    10.5532%
║ Solo ETA:      9m 28s
╠┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈╣
║ Your Shares:   142 acc / 0 rej
║ Expecting:     1 block per 4096 shares
║ Session Luck:  3.46% ⏳
╚══════════════════════════════════════╝
```

---

## Part 3: Manual Auditing

Your miner automatically audits the pool when a new block is found. To manually verify your current score, you can query the pool's HTTP Audit API. 

If the pool's Stratum port is `3333`, the corresponding Audit API runs on port `8081`.

```bash
curl "http://<POOL_IP_ADDRESS>:8081/api/proof?address=<PAYOUT_ADDRESS>"
```

**Response format:**
```json
{
  "index": 4,
  "proof": [
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
  ],
  "root": "bd4b0d014c99738090b8f2c3d100dcdeef457636e788bc53caee0e181829e2f9",
  "score": 145
}
```

---

## Troubleshooting

*   **`audit failed: template header hash mismatch`** or **`merkle root mismatch`**
    The pool operator modified the block template to redirect rewards or omitted your address from the payout structure. The miner automatically disconnected to prevent wasted work. Switch to a different pool.

*   **`share rejected: Duplicate share`** or **`Low difficulty`**
    Your miner submitted a share for a block that was just solved by the network (stale share), or network latency caused a duplicate submission. The miner will fetch the updated job and continue automatically.

*   **`fatal: could not find available stratum/api port pairs`**
    The pool server failed to bind to any ports in the `3333-3343` range. Check for lingering processes (e.g., `killall midstate`) or specify a custom bind address using `--bind-addr 0.0.0.0:<PORT>`.
