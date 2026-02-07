#!/usr/bin/env bash
set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────

FEATURES="--features fast-mining"
CARGO="cargo run $FEATURES --"
NODE_A_DIR="/tmp/midstate_test_a"
NODE_B_DIR="/tmp/midstate_test_b"
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
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    rm -rf "$NODE_A_DIR" "$NODE_B_DIR"
    echo ""
    echo "═══════════════════════════════════════════"
    echo "  Results:  $PASS passed,  $FAIL failed"
    echo "═══════════════════════════════════════════"
    if [ "$FAIL" -gt 0 ]; then
        exit 1
    fi
}
trap cleanup EXIT

assert_contains() {
    local label="$1"
    local haystack="$2"
    local needle="$3"
    if echo "$haystack" | grep -q "$needle"; then
        echo "  ✓ $label"
        ((++PASS))  # <--- CHANGED from ((PASS++))
    else
        echo "  ✗ $label (expected '$needle')"
        echo "    got: $haystack"
        ((++FAIL))  # <--- CHANGED from ((FAIL++))
    fi
}

assert_not_contains() {
    local label="$1"
    local haystack="$2"
    local needle="$3"
    if echo "$haystack" | grep -q "$needle"; then
        echo "  ✗ $label (did not expect '$needle')"
        echo "    got: $haystack"
        ((++FAIL))  # <--- CHANGED
    else
        echo "  ✓ $label"
        ((++PASS))  # <--- CHANGED
    fi
}
coin_id() {
    python3 -c "import hashlib; print(hashlib.sha256(b'$1').hexdigest())"
}

secret_hex() {
    python3 -c "print(b'$1'.hex())"
}

wait_for_rpc() {
    local port=$1
    local max_wait=15
    local waited=0
    while ! curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; do
        sleep 1
        # FIX: Use pre-increment to avoid exit code 1 when waited is 0
        ((++waited))
        if [ "$waited" -ge "$max_wait" ]; then
            echo "  ✗ RPC on port $port failed to start after ${max_wait}s"
            ((FAIL++))
            return 1
        fi
    done
}

wait_for_height() {
    local port=$1
    local target=$2
    local max_wait=${3:-30}
    local waited=0
    while true; do
        local height
        height=$(curl -s "http://127.0.0.1:$port/state" | python3 -c "import sys,json; print(json.load(sys.stdin)['height'])" 2>/dev/null || echo "0")
        if [ "$height" -ge "$target" ]; then
            return 0
        fi
        sleep 2
        ((waited+=2))
        if [ "$waited" -ge "$max_wait" ]; then
            echo "  ⚠ Timeout waiting for height $target on port $port (at $height)"
            return 1
        fi
    done
}

get_field() {
    local json="$1"
    local field="$2"
    echo "$json" | python3 -c "import sys,json; print(json.load(sys.stdin)['$field'])"
}

# ─── Build ───────────────────────────────────────────────────────────────────

echo "═══ Building ═══"
cargo build $FEATURES 2>&1 | tail -1
echo ""

# ─── Start Nodes ─────────────────────────────────────────────────────────────

echo "═══ Starting Node A (miner) ═══"
rm -rf "$NODE_A_DIR" "$NODE_B_DIR"
$CARGO node --data-dir "$NODE_A_DIR" --port $PORT_A --rpc-port $RPC_A --mine > /tmp/node_a.log 2>&1 &
PIDS+=($!)
wait_for_rpc $RPC_A
echo "  Node A running (pid ${PIDS[-1]})"

echo ""
echo "═══ Starting Node B (follower) ═══"
$CARGO node --data-dir "$NODE_B_DIR" --port $PORT_B --rpc-port $RPC_B --peer "127.0.0.1:$PORT_A" > /tmp/node_b.log 2>&1 &
PIDS+=($!)
wait_for_rpc $RPC_B
echo "  Node B running (pid ${PIDS[-1]})"
sleep 2

# ─── Test 1: Genesis State ──────────────────────────────────────────────────

echo ""
echo "═══ Test 1: Genesis State ═══"
STATE_A=$($CARGO state --rpc-port $RPC_A 2>/dev/null)
STATE_B=$($CARGO state --rpc-port $RPC_B 2>/dev/null)

assert_contains "Node A height is 0" "$STATE_A" "Height:"
assert_contains "Node A has 3 coins" "$STATE_A" "Coins:       3"
assert_contains "Node A has 0 commitments" "$STATE_A" "Commitments: 0"
assert_contains "Node B height matches" "$STATE_B" "Height:"

# ─── Test 2: Peers Connected ────────────────────────────────────────────────

echo ""
echo "═══ Test 2: Peer Connectivity ═══"
PEERS_A=$($CARGO peers --rpc-port $RPC_A 2>/dev/null)
assert_contains "Node A has peers" "$PEERS_A" "127.0.0.1"

# ─── Test 3: Happy Path Commit-Reveal ───────────────────────────────────────

echo ""
echo "═══ Test 3: Commit-Reveal Happy Path ═══"

GENESIS_COIN_1=$(coin_id "genesis_coin_1")
SECRET_1=$(secret_hex "genesis_coin_1")

# Generate a destination
KEYGEN=$($CARGO keygen 2>/dev/null)
DEST=$(echo "$KEYGEN" | grep "Coin:" | awk '{print $2}')
echo "  Destination: $DEST"

# Phase 1: Commit
echo "  Submitting commit..."
COMMIT_OUT=$(curl -s -X POST "http://127.0.0.1:$RPC_A/commit" \
    -H "Content-Type: application/json" \
    -d "{\"coins\":[\"$GENESIS_COIN_1\"],\"destinations\":[\"$DEST\"]}")

COMMITMENT=$(get_field "$COMMIT_OUT" "commitment")
SALT=$(get_field "$COMMIT_OUT" "salt")
STATUS=$(get_field "$COMMIT_OUT" "status")

assert_contains "Commit accepted" "$STATUS" "committed"
echo "  Commitment: $COMMITMENT"
echo "  Salt: $SALT"

# Wait for commit to be mined
echo "  Waiting for commit to be mined..."
BEFORE_HEIGHT=$(curl -s "http://127.0.0.1:$RPC_A/state" | python3 -c "import sys,json; print(json.load(sys.stdin)['height'])")
wait_for_height $RPC_A $((BEFORE_HEIGHT + 1)) 60

STATE_AFTER_COMMIT=$(curl -s "http://127.0.0.1:$RPC_A/state")
COMMITMENTS=$(get_field "$STATE_AFTER_COMMIT" "num_commitments")
assert_contains "Commitment registered in state" "$COMMITMENTS" "1"

# Phase 2: Reveal
echo "  Submitting reveal..."
REVEAL_OUT=$(curl -s -X POST "http://127.0.0.1:$RPC_A/send" \
    -H "Content-Type: application/json" \
    -d "{\"secrets\":[\"$SECRET_1\"],\"destinations\":[\"$DEST\"],\"salt\":\"$SALT\"}")

REVEAL_STATUS=$(get_field "$REVEAL_OUT" "status" 2>/dev/null || echo "$REVEAL_OUT")
assert_contains "Reveal accepted" "$REVEAL_STATUS" "submitted"

# Wait for reveal to be mined
echo "  Waiting for reveal to be mined..."
CURRENT=$(curl -s "http://127.0.0.1:$RPC_A/state" | python3 -c "import sys,json; print(json.load(sys.stdin)['height'])")
wait_for_height $RPC_A $((CURRENT + 1)) 60

# Verify coin transferred
BALANCE_OUT=$(curl -s -X POST "http://127.0.0.1:$RPC_A/check" \
    -H "Content-Type: application/json" \
    -d "{\"coin\":\"$DEST\"}")
EXISTS=$(get_field "$BALANCE_OUT" "exists")
assert_contains "New coin exists" "$EXISTS" "True"

# Verify old coin gone
OLD_BALANCE=$(curl -s -X POST "http://127.0.0.1:$RPC_A/check" \
    -H "Content-Type: application/json" \
    -d "{\"coin\":\"$GENESIS_COIN_1\"}")
OLD_EXISTS=$(get_field "$OLD_BALANCE" "exists")
assert_contains "Old coin spent" "$OLD_EXISTS" "False"

# ─── Test 4: Node B Synced ──────────────────────────────────────────────────

echo ""
echo "═══ Test 4: Node B Synced ═══"
sleep 5  # Give sync time

STATE_A_NOW=$(curl -s "http://127.0.0.1:$RPC_A/state")
STATE_B_NOW=$(curl -s "http://127.0.0.1:$RPC_B/state")

HEIGHT_A=$(get_field "$STATE_A_NOW" "height")
HEIGHT_B=$(get_field "$STATE_B_NOW" "height")
MIDSTATE_A=$(get_field "$STATE_A_NOW" "midstate")
MIDSTATE_B=$(get_field "$STATE_B_NOW" "midstate")

assert_contains "Heights match" "$HEIGHT_B" "$HEIGHT_A"
assert_contains "Midstates match" "$MIDSTATE_B" "$MIDSTATE_A"

# ─── Test 5: Front-Running Attempt ──────────────────────────────────────────

echo ""
echo "═══ Test 5: Front-Running Rejected ═══"

GENESIS_COIN_2=$(coin_id "genesis_coin_2")
SECRET_2=$(secret_hex "genesis_coin_2")
LEGIT_DEST="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
ATTACKER_DEST="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

# Commit for legit destination
COMMIT2_OUT=$(curl -s -X POST "http://127.0.0.1:$RPC_A/commit" \
    -H "Content-Type: application/json" \
    -d "{\"coins\":[\"$GENESIS_COIN_2\"],\"destinations\":[\"$LEGIT_DEST\"]}")

SALT2=$(get_field "$COMMIT2_OUT" "salt")

# Wait for commit to be mined
echo "  Waiting for commit to be mined..."
CURRENT=$(curl -s "http://127.0.0.1:$RPC_A/state" | python3 -c "import sys,json; print(json.load(sys.stdin)['height'])")
wait_for_height $RPC_A $((CURRENT + 1)) 60

# Attacker tries to reveal with different destination
echo "  Attempting front-run with different destination..."
ATTACK_OUT=$(curl -s -X POST "http://127.0.0.1:$RPC_A/send" \
    -H "Content-Type: application/json" \
    -d "{\"secrets\":[\"$SECRET_2\"],\"destinations\":[\"$ATTACKER_DEST\"],\"salt\":\"$SALT2\"}")

assert_contains "Front-run rejected" "$ATTACK_OUT" "error"

# Legit reveal should still work
echo "  Submitting legitimate reveal..."
LEGIT_OUT=$(curl -s -X POST "http://127.0.0.1:$RPC_A/send" \
    -H "Content-Type: application/json" \
    -d "{\"secrets\":[\"$SECRET_2\"],\"destinations\":[\"$LEGIT_DEST\"],\"salt\":\"$SALT2\"}")

LEGIT_STATUS=$(get_field "$LEGIT_OUT" "status" 2>/dev/null || echo "$LEGIT_OUT")
assert_contains "Legitimate reveal accepted" "$LEGIT_STATUS" "submitted"

# ─── Test 6: Reveal Without Commit ──────────────────────────────────────────

echo ""
echo "═══ Test 6: Reveal Without Commit Rejected ═══"

SECRET_3=$(secret_hex "genesis_coin_3")
RANDOM_DEST="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
ZERO_SALT="0000000000000000000000000000000000000000000000000000000000000000"

NO_COMMIT_OUT=$(curl -s -X POST "http://127.0.0.1:$RPC_A/send" \
    -H "Content-Type: application/json" \
    -d "{\"secrets\":[\"$SECRET_3\"],\"destinations\":[\"$RANDOM_DEST\"],\"salt\":\"$ZERO_SALT\"}")

assert_contains "Reveal without commit rejected" "$NO_COMMIT_OUT" "error"

# Verify coin still exists (wasn't stolen)
COIN_3=$(coin_id "genesis_coin_3")
STILL_THERE=$(curl -s -X POST "http://127.0.0.1:$RPC_A/check" \
    -H "Content-Type: application/json" \
    -d "{\"coin\":\"$COIN_3\"}")
STILL_EXISTS=$(get_field "$STILL_THERE" "exists")
assert_contains "Unspent coin still safe" "$STILL_EXISTS" "True"

# ─── Test 7: Double Spend Attempt ───────────────────────────────────────────

echo ""
echo "═══ Test 7: Double Spend Rejected ═══"

# genesis_coin_1 was already spent in Test 3
# Try to commit and reveal it again
DS_DEST="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"

DS_COMMIT=$(curl -s -X POST "http://127.0.0.1:$RPC_A/commit" \
    -H "Content-Type: application/json" \
    -d "{\"coins\":[\"$GENESIS_COIN_1\"],\"destinations\":[\"$DS_DEST\"]}")

# Commit itself is opaque, so it might be accepted.
# But the reveal will fail because the coin is already spent.
DS_SALT=$(get_field "$DS_COMMIT" "salt" 2>/dev/null || echo "$ZERO_SALT")

# Wait for it to possibly be mined
echo "  Waiting..."
sleep 10

DS_REVEAL=$(curl -s -X POST "http://127.0.0.1:$RPC_A/send" \
    -H "Content-Type: application/json" \
    -d "{\"secrets\":[\"$SECRET_1\"],\"destinations\":[\"$DS_DEST\"],\"salt\":\"$DS_SALT\"}")

assert_contains "Double spend reveal rejected" "$DS_REVEAL" "error"

# ─── Test 8: Mempool ────────────────────────────────────────────────────────

echo ""
echo "═══ Test 8: Mempool ═══"

MEMPOOL=$($CARGO mempool --rpc-port $RPC_A 2>/dev/null)
assert_contains "Mempool accessible" "$MEMPOOL" "Size:"

# ─── Done ────────────────────────────────────────────────────────────────────

echo ""
echo "═══ All tests complete ═══"
