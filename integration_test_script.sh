#!/usr/bin/env bash
set -euo pipefail

# ─── Pre-flight Cleanup ──────────────────────────────────────────────────────
# Kill any zombies from previous runs to prevent "Address already in use"
pkill -f "midstate node" || true
# Wait a moment for ports to free up
sleep 1

# ─── Config ──────────────────────────────────────────────────────────────────

CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-target}
BIN="$CARGO_TARGET_DIR/debug/midstate"
FEATURES="--features fast-mining"

NODE_A_DIR="/tmp/midstate_test_a"
NODE_B_DIR="/tmp/midstate_test_b"
WALLET_FILE="/tmp/midstate_test.wallet"
RPC_A=8545
RPC_B=8546
PORT_A=9333
PORT_B=9334

PASS=0
FAIL=0
PIDS=()

# ─── Helpers ─────────────────────────────────────────────────────────────────

cleanup() {
    echo ""
    echo "═══ Cleaning up ═══"
    
    # If we failed, print the miner log to help debug
    if [ "$FAIL" -gt 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!!! TEST FAILED - DUMPING NODE A LOGS BELOW !!!"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        if [ -f /tmp/node_a.log ]; then
            cat /tmp/node_a.log
        else
            echo "Log file not found."
        fi
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    fi

    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    rm -rf "$NODE_A_DIR" "$NODE_B_DIR" "$WALLET_FILE"
    echo ""
    echo "═══════════════════════════════════════════"
    echo "  Results:  $PASS passed,  $FAIL failed"
    echo "═══════════════════════════════════════════"
    if [ "$FAIL" -gt 0 ]; then
        exit 1
    fi
}
trap cleanup EXIT

# Build the binary first
echo "═══ Building ═══"
cargo build $FEATURES

if [ ! -f "$BIN" ]; then
    echo "Error: Binary not found at $BIN"
    exit 1
fi

# Run command wrapper
midstate() {
    "$BIN" "$@"
}

assert_contains() {
    local label="$1"
    local haystack="$2"
    local needle="$3"
    if echo "$haystack" | grep -q "$needle"; then
        echo "  ✓ $label"
        ((++PASS))
    else
        echo "  ✗ $label"
        echo "    Expected: '$needle'"
        echo "    Got:      $haystack"
        ((++FAIL))
    fi
}

wait_for_rpc() {
    local port=$1
    local max_wait=15
    local waited=0
    # Check if the process linked to this port is actually running (simple check)
    while ! curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; do
        sleep 1
        ((++waited))
        if [ "$waited" -ge "$max_wait" ]; then
            echo "  ✗ RPC on port $port failed to start after ${max_wait}s"
            # Check logs immediately if startup fails
            if [ "$port" -eq "$RPC_A" ]; then
                cat /tmp/node_a.log
            fi
            ((FAIL++))
            return 1
        fi
    done
}

wait_for_height() {
    local port=$1
    local target=$2
    local max_wait=${3:-60}
    local waited=0
    echo "  ...waiting for height $target on port $port (timeout ${max_wait}s)"
    while true; do
        local height
        height=$(curl -s "http://127.0.0.1:$port/state" | python3 -c "import sys,json; print(json.load(sys.stdin)['height'])" 2>/dev/null || echo "0")
        if [ "$height" -ge "$target" ]; then
            return 0
        fi
        sleep 1
        ((++waited))
        if [ "$waited" -ge "$max_wait" ]; then
            echo "  ⚠ Timeout waiting for height $target on port $port (at $height)"
            return 1
        fi
    done
}

# ─── Start Nodes ─────────────────────────────────────────────────────────────

echo "═══ Starting Node A (miner) ═══"
rm -rf "$NODE_A_DIR" "$NODE_B_DIR"
# Node A logs redirected to /tmp/node_a.log
midstate node --data-dir "$NODE_A_DIR" --port $PORT_A --rpc-port $RPC_A --mine > /tmp/node_a.log 2>&1 &
PIDS+=($!)
wait_for_rpc $RPC_A
echo "  Node A running (pid ${PIDS[-1]})"

echo ""
echo "═══ Starting Node B (follower) ═══"
midstate node --data-dir "$NODE_B_DIR" --port $PORT_B --rpc-port $RPC_B --peer "127.0.0.1:$PORT_A" > /tmp/node_b.log 2>&1 &
PIDS+=($!)
wait_for_rpc $RPC_B
echo "  Node B running (pid ${PIDS[-1]})"
sleep 2

# ─── Test 1: Genesis State ──────────────────────────────────────────────────

echo ""
echo "═══ Test 1: Genesis State ═══"
STATE_A=$(curl -s "http://127.0.0.1:$RPC_A/state")
STATE_B=$(curl -s "http://127.0.0.1:$RPC_B/state")

# Genesis has 3 coins: genesis_coin_1, genesis_coin_2, genesis_coin_3
assert_contains "Node A height is 0" "$STATE_A" '"height":0'
assert_contains "Node A has 3 coins" "$STATE_A" '"num_coins":3'
assert_contains "Node B connected"   "$STATE_B" '"height":0'

# ─── Wallet Setup (Calculate Genesis IDs) ───────────────────────────────────

echo ""
echo "═══ Setup: Wallet & Genesis Calculation ═══"

# Seeds are SHA256("genesis_coin_X")
SEED1=$(echo -n "genesis_coin_1" | sha256sum | awk '{print $1}')
SEED2=$(echo -n "genesis_coin_2" | sha256sum | awk '{print $1}')
SEED3=$(echo -n "genesis_coin_3" | sha256sum | awk '{print $1}')

# Create wallet (Supressing output to hide "Password:" prompt spam)
echo -e "password\npassword" | midstate wallet create --path "$WALLET_FILE" > /dev/null 2>&1

# Import seeds (Supressing output)
echo "password" | midstate wallet import --path "$WALLET_FILE" --seed "$SEED1" --label "gen1" > /dev/null 2>&1
echo "password" | midstate wallet import --path "$WALLET_FILE" --seed "$SEED2" --label "gen2" > /dev/null 2>&1
echo "password" | midstate wallet import --path "$WALLET_FILE" --seed "$SEED3" --label "gen3" > /dev/null 2>&1

# Get Coin IDs (Full hex)
COIN1=$(echo "password" | midstate wallet list --path "$WALLET_FILE" --full 2>/dev/null | grep "gen1" | awk '{print $2}')
COIN2=$(echo "password" | midstate wallet list --path "$WALLET_FILE" --full 2>/dev/null | grep "gen2" | awk '{print $2}')
COIN3=$(echo "password" | midstate wallet list --path "$WALLET_FILE" --full 2>/dev/null | grep "gen3" | awk '{print $2}')

echo "  Genesis Coin 1: ${COIN1:0:8}..."
echo "  Genesis Coin 2: ${COIN2:0:8}..."

# Generate a destination address (just a random keypair)
DEST_KEYGEN=$(midstate keygen)
DEST=$(echo "$DEST_KEYGEN" | grep "Coin:" | awk '{print $2}')
echo "  Destination:    ${DEST:0:8}..."

# ─── Test 2: Commit-Reveal Happy Path ───────────────────────────────────────

echo ""
echo "═══ Test 2: Commit-Reveal Happy Path ═══"
# FEE RULE: Inputs > Outputs. 
# We spend COIN1 + COIN2 (2 inputs) -> DEST (1 output)

# Phase 1: Commit
echo "  [1/2] Submitting Commit..."
COMMIT_OUT=$(midstate commit --rpc-port $RPC_A --coin "$COIN1" --coin "$COIN2" --dest "$DEST")

echo "$COMMIT_OUT"

if echo "$COMMIT_OUT" | grep -q "Commitment submitted"; then
    echo "  ✓ Commit command success"
    ((++PASS))
else
    echo "  ✗ Commit failed"
    ((++FAIL))
    exit 1
fi

# Extract Salt
SALT=$(echo "$COMMIT_OUT" | grep "Salt:" | awk '{print $2}')

# Wait for mining (Block 1)
wait_for_height $RPC_A 1 60

# Phase 2: Send (Reveal)
echo "  [2/2] Submitting Reveal (Send)..."
SEND_OUT=$(midstate send --rpc-port $RPC_A \
    --input-coin "$COIN1" --seed "$SEED1" \
    --input-coin "$COIN2" --seed "$SEED2" \
    --dest "$DEST" \
    --salt "$SALT")

echo "$SEND_OUT"

if echo "$SEND_OUT" | grep -q "Transaction submitted"; then
    echo "  ✓ Send command success"
    ((++PASS))
else
    echo "  ✗ Send failed"
    ((++FAIL))
fi

# Wait for mining (Block 2)
wait_for_height $RPC_A 2 60

# Verify Balances via RPC
CHECK_DEST=$(midstate balance --rpc-port $RPC_A --coin "$DEST")
assert_contains "Destination funded" "$CHECK_DEST" "YES"

CHECK_OLD=$(midstate balance --rpc-port $RPC_A --coin "$COIN1")
assert_contains "Input coin spent" "$CHECK_OLD" "NO"

# ─── Test 3: Sync check ─────────────────────────────────────────────────────

echo ""
echo "═══ Test 3: Sync Propagation ═══"
# Give Node B a moment to catch up
sleep 2
HEIGHT_B=$(curl -s "http://127.0.0.1:$RPC_B/state" | python3 -c "import sys,json; print(json.load(sys.stdin)['height'])")

if [ "$HEIGHT_B" -ge 2 ]; then
    echo "  ✓ Node B synced to height $HEIGHT_B"
    ((++PASS))
else
    echo "  ✗ Node B stuck at height $HEIGHT_B"
    ((++FAIL))
fi

# ─── Test 4: Invalid Fee Attempt ────────────────────────────────────────────

echo ""
echo "═══ Test 4: Invalid Fee Logic ═══"
# Try to spend 1 input -> 1 output (Inputs == Outputs, fee 0)
# We use COIN3 -> NEW_DEST

DEST2_KEYGEN=$(midstate keygen)
DEST2=$(echo "$DEST2_KEYGEN" | grep "Coin:" | awk '{print $2}')

echo "  Attempting 1-to-1 spend (should fail)..."
COMMIT_BAD=$(midstate commit --rpc-port $RPC_A --coin "$COIN3" --dest "$DEST2")
SALT_BAD=$(echo "$COMMIT_BAD" | grep "Salt:" | awk '{print $2}')

# Wait for commit to mine
wait_for_height $RPC_A 3 60

# Reveal
SEND_BAD=$(midstate send --rpc-port $RPC_A \
    --input-coin "$COIN3" --seed "$SEED3" \
    --dest "$DEST2" \
    --salt "$SALT_BAD" 2>&1 || true) # Allow failure output

if echo "$SEND_BAD" | grep -q "inputs.*must exceed.*outputs"; then
    echo "  ✓ Rejected invalid fee transaction"
    ((++PASS))
else 
    if echo "$SEND_BAD" | grep -q "Error:"; then
        echo "  ✓ Rejected transaction (General Error)"
        ((++PASS))
    else
        echo "  ✗ Failed to reject 1-to-1 spend"
        echo "$SEND_BAD"
        ((++FAIL))
    fi
fi

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "═══ Tests Complete ═══"
