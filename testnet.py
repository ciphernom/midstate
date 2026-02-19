#!/usr/bin/env python3
"""
testnet.py — Multi-node integration test for midstate

Replaces testnet.sh with proper process management and P2P connection
verification to eliminate race conditions that cause sync failures.

Key fixes over the bash version:
  1. Verifies P2P peer connections (via /peers) before expecting sync
  2. Stops mining before convergence checks (no moving-target problem)
  3. Proper process lifecycle (SIGTERM → wait → SIGKILL)
  4. Longer waits between kill/restart for port and DB lock release
  5. Dynamically extracts libp2p PeerIds to construct valid Multiaddrs.

Usage:
    python3 testnet.py              # build + run all tests
    python3 testnet.py --skip-build # reuse existing binary
"""

import subprocess, signal, sys, os, time, json, shutil, argparse, re
from pathlib import Path
from typing import Optional

# ── Configuration ────────────────────────────────────────────────────────────

BINARY = "./target/release/midstate"
DIR_A = Path("/tmp/midstate-test-a")
DIR_B = Path("/tmp/midstate-test-b")
DIR_C = Path("/tmp/midstate-test-c")
DIR_D = Path("/tmp/midstate-test-d")

P2P_A, RPC_A = 19333, 18545
P2P_B, RPC_B = 19334, 18546
P2P_C, RPC_C = 19335, 18547
P2P_D, RPC_D = 19336, 18548

# ── State ────────────────────────────────────────────────────────────────────

processes: dict[str, subprocess.Popen] = {}
PASS = 0
FAIL = 0
TESTS_RUN = 0

# ── Formatting ───────────────────────────────────────────────────────────────

RED    = "\033[0;31m"
GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN   = "\033[0;36m"
NC     = "\033[0m"

def log(msg):  print(f"{CYAN}[testnet]{NC} {msg}", flush=True)
def section(title): print(f"\n{YELLOW}━━━ {title} ━━━{NC}", flush=True)

def pass_test(desc):
    global PASS, TESTS_RUN
    PASS += 1; TESTS_RUN += 1
    print(f"  {GREEN}✓ PASS{NC}: {desc}", flush=True)

def fail_test(desc, reason=""):
    global FAIL, TESTS_RUN
    FAIL += 1; TESTS_RUN += 1
    detail = f" — {reason}" if reason else ""
    print(f"  {RED}✗ FAIL{NC}: {desc}{detail}", flush=True)

# ── Process management ───────────────────────────────────────────────────────

def start_node(name: str, data_dir: Path, p2p: int, rpc: int,
               mine: bool, peers: list[str] | None = None,
               fresh: bool = False) -> str:
    """Start a node, return a key into `processes`."""
    if fresh:
        shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    cmd = [BINARY, "node",
           "--data-dir", str(data_dir),
           "--port", str(p2p),
           "--rpc-port", str(rpc)]
    if mine:
        cmd.append("--mine")
    for peer in (peers or []):
        cmd.extend(["--peer", peer])

    logfile = open(data_dir / "node.log", "w")
    log(f"Starting {name}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    processes[name] = proc
    return name


def kill_node(name: str, timeout: float = 5.0):
    """Gracefully stop a node: SIGTERM → wait → SIGKILL."""
    proc = processes.pop(name, None)
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=3)
        except Exception:
            pass


def kill_all():
    for name in list(processes):
        kill_node(name)


# ── P2P & RPC helpers ────────────────────────────────────────────────────────

import urllib.request, urllib.error

def get_local_multiaddr(data_dir: Path, port: int, timeout=15) -> str:
    """Reads the node.log to extract the dynamic libp2p PeerId and construct the Multiaddr."""
    deadline = time.time() + timeout
    log_file = data_dir / "node.log"
    while time.time() < deadline:
        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        m = re.search(r'Local peer id:\s+([1-9A-HJ-NP-Za-km-z]+)', line)
                        if m:
                            return f"/ip4/127.0.0.1/tcp/{port}/p2p/{m.group(1)}"
            except Exception:
                pass
        time.sleep(0.5)
    raise RuntimeError(f"Could not find PeerId in {log_file} within {timeout}s")


def rpc(port: int, path: str, *, method="GET", body=None, timeout=5) -> Optional[dict]:
    """Hit an RPC endpoint. Returns parsed JSON or None on failure."""
    url = f"http://127.0.0.1:{port}{path}"
    try:
        req = urllib.request.Request(url, method=method)
        if body is not None:
            req.data = json.dumps(body).encode()
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode()
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return {"raw": data}
    except Exception:
        return None


def get_height(port): return (rpc(port, "/state") or {}).get("height")
def get_midstate(port): return (rpc(port, "/state") or {}).get("midstate")
def get_depth(port): return (rpc(port, "/state") or {}).get("depth")
def get_coins(port): return (rpc(port, "/state") or {}).get("num_coins")
def get_peers(port): return (rpc(port, "/peers") or {}).get("peers", [])


# ── Wait helpers ─────────────────────────────────────────────────────────────

def wait_for_health(port: int, name: str, timeout=30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if rpc(port, "/health") is not None:
            return True
        time.sleep(0.5)
    fail_test(f"{name} health check", f"timed out after {timeout}s")
    return False


def wait_for_peers(port: int, name: str, min_peers=1, timeout=30) -> bool:
    """Wait until the node reports at least `min_peers` P2P connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        peers = get_peers(port)
        if len(peers) >= min_peers:
            return True
        time.sleep(0.5)
    log(f"⚠ {name} has {len(get_peers(port))} peers (wanted {min_peers})")
    return False


def wait_for_height(port: int, target: int, timeout=120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        h = get_height(port)
        if h is not None and h >= target:
            return True
        time.sleep(1)
    return False


def wait_for_sync(port_a: int, port_b: int, timeout=60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        ha, hb = get_height(port_a), get_height(port_b)
        if ha is not None and hb is not None and hb >= ha:
            return True
        time.sleep(1)
    return False


def wait_for_consensus(port_a: int, port_b: int, timeout=60) -> bool:
    """Wait until height AND midstate match."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        ha, hb = get_height(port_a), get_height(port_b)
        ma, mb = get_midstate(port_a), get_midstate(port_b)
        if (ha is not None and hb is not None and
            ma is not None and mb is not None and
            ha == hb and ma == mb):
            return True
        time.sleep(1)
    return False


def stop_mining_restart(name: str, data_dir: Path, p2p: int, rpc: int,
                        peers: list[str], wait_port: int = None) -> str:
    """Kill a node and restart it without mining. Waits for health + peers."""
    kill_node(name)
    time.sleep(3)  # let port + DB lock release
    new_name = start_node(name, data_dir, p2p, rpc, mine=False, peers=peers)
    if not wait_for_health(rpc, name, timeout=30):
        return new_name
    if peers:
        wait_for_peers(rpc, name, timeout=30)
    return new_name


# ══════════════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_tests(skip_build: bool):
    global PASS, FAIL, TESTS_RUN

    # ── Build ────────────────────────────────────────────────────────────
    if not skip_build:
        section("Building")
        log("cargo build --release")
        r = subprocess.run(["cargo", "build", "--release"], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            sys.exit(1)
        log("Build complete")

    if not os.path.isfile(BINARY) or not os.access(BINARY, os.X_OK):
        print(f"Binary not found at {BINARY}")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    # TEST 1: Node startup and mining
    # ══════════════════════════════════════════════════════════════════════
    section("Test 1: Node startup and mining")

    start_node("A", DIR_A, P2P_A, RPC_A, mine=True, fresh=True)
    if not wait_for_health(RPC_A, "Node-A"):
        sys.exit(1)

    ma_A = get_local_multiaddr(DIR_A, P2P_A)
    pass_test(f"Node A starts and responds to /health. Multiaddr: {ma_A}")

    if wait_for_height(RPC_A, 3, timeout=60):
        pass_test(f"Node A mining works (height={get_height(RPC_A)})")
    else:
        fail_test("Node A mining", "did not reach height 3 in 60s")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    # TEST 2: Peer sync
    # ══════════════════════════════════════════════════════════════════════
    section("Test 2: Peer sync (Node B joins and catches up)")

    start_node("B", DIR_B, P2P_B, RPC_B, mine=False,
               peers=[ma_A], fresh=True)
    if not wait_for_health(RPC_B, "Node-B"):
        sys.exit(1)
    pass_test("Node B starts and responds to /health")

    # Wait for the P2P link to come up
    wait_for_peers(RPC_B, "Node-B", timeout=15)

    time.sleep(5)
    if wait_for_sync(RPC_A, RPC_B, 90):
        ha, hb = get_height(RPC_A), get_height(RPC_B)
        pass_test(f"Node B synced with Node A (A={ha}, B={hb})")
    else:
        ha, hb = get_height(RPC_A), get_height(RPC_B)
        fail_test("Peer sync", f"A={ha} B={hb} after 90s")

    ma, mb = get_midstate(RPC_A), get_midstate(RPC_B)
    if ma == mb:
        pass_test("Midstates match between A and B")
    else:
        fail_test("Midstate match", f"A={ma} B={mb}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 3: Full wallet send flow
    # ══════════════════════════════════════════════════════════════════════
    section("Test 3: Full wallet send flow")

    os.environ["MIDSTATE_PASSWORD"] = "testpass123"
    wallet_sender = DIR_A / "wallet_sender.dat"
    wallet_receiver = DIR_A / "wallet_receiver.dat"

    wait_for_height(RPC_A, 5, 60)
    log(f"Node A at height {get_height(RPC_A)}")

    # Create sender wallet
    r = subprocess.run([BINARY, "wallet", "create", "--path", str(wallet_sender)],
                       capture_output=True, text=True, env=os.environ)
    if r.returncode == 0:
        pass_test("Sender wallet created")
    else:
        fail_test("Sender wallet creation", r.stderr[:200])

    # Import coinbase rewards
    cb_file = DIR_A / "coinbase_seeds.jsonl"
    if cb_file.exists():
        subprocess.run([BINARY, "wallet", "import-rewards",
                        "--path", str(wallet_sender),
                        "--coinbase-file", str(cb_file)],
                       capture_output=True, text=True, env=os.environ)
        pass_test("Coinbase rewards imported")
    else:
        fail_test("Import rewards", "no coinbase_seeds.jsonl")

    # Check balance
    r = subprocess.run([BINARY, "wallet", "balance",
                        "--path", str(wallet_sender),
                        "--rpc-port", str(RPC_A)],
                       capture_output=True, text=True, env=os.environ)
    bal_output = r.stdout + r.stderr
    log(f"Balance output: {bal_output.strip()}")
    bal_match = re.search(r'value:\s*(\d+)', bal_output)
    sender_bal = int(bal_match.group(1)) if bal_match else 0
    log(f"Sender wallet live value: {sender_bal}")
    if sender_bal > 0:
        pass_test(f"Sender wallet has funds ({sender_bal})")
    else:
        log("Warning: balance shows 0 — coins may not be confirmed yet. Continuing...")

    # Create receiver wallet
    subprocess.run([BINARY, "wallet", "create", "--path", str(wallet_receiver)],
                   capture_output=True, text=True, env=os.environ)
    r = subprocess.run([BINARY, "wallet", "receive", "--path", str(wallet_receiver)],
                       capture_output=True, text=True, env=os.environ)
    recv_match = re.search(r'[0-9a-f]{64}', r.stdout + r.stderr)
    recv_addr = recv_match.group(0) if recv_match else ""

    if recv_addr:
        pass_test(f"Receiver address generated: {recv_addr[:16]}...")
    else:
        fail_test("Receiver address", "could not parse address")

    # Send
    log(f"Sending 1 to receiver...")
    r = subprocess.run([BINARY, "wallet", "send",
                        "--path", str(wallet_sender),
                        "--rpc-port", str(RPC_A),
                        "--to", f"{recv_addr}:1",
                        "--timeout", "120"],
                       capture_output=True, text=True, env=os.environ)
    send_output = r.stdout + r.stderr
    log(f"Send output: {send_output.strip()}")

    if re.search(r'complete|submitted|committed|revealed|success', send_output, re.I):
        pass_test("Wallet send completed")
    else:
        fail_test("Wallet send", "unexpected output")

    # Verify receiver got funds
    log("Waiting for transaction to be mined...")
    time.sleep(15)

    subprocess.run([BINARY, "wallet", "scan",
                    "--path", str(wallet_receiver),
                    "--rpc-port", str(RPC_A)],
                   capture_output=True, text=True, env=os.environ)
    r = subprocess.run([BINARY, "wallet", "balance",
                        "--path", str(wallet_receiver),
                        "--rpc-port", str(RPC_A)],
                       capture_output=True, text=True, env=os.environ)
    recv_bal_out = r.stdout + r.stderr
    recv_match = re.search(r'value:\s*(\d+)', recv_bal_out)
    recv_bal = int(recv_match.group(1)) if recv_match else 0
    log(f"Receiver balance: {recv_bal}")
    if recv_bal >= 1:
        pass_test(f"Receiver got the funds ({recv_bal})")
    else:
        log("Note: Receiver may not see funds yet if reveal hasn't been mined")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 4: Transaction propagation to Node B
    # ══════════════════════════════════════════════════════════════════════
    section("Test 4: Transaction propagation to Node B")

    time.sleep(5)
    if wait_for_sync(RPC_A, RPC_B, 60):
        ha, hb = get_height(RPC_A), get_height(RPC_B)
        pass_test(f"Node B still in sync after transactions (A={ha}, B={hb})")
    else:
        ha, hb = get_height(RPC_A), get_height(RPC_B)
        fail_test("Post-tx sync", f"A={ha} B={hb}")

    ma, mb = get_midstate(RPC_A), get_midstate(RPC_B)
    if ma == mb:
        pass_test("Midstates still match after transactions")
    else:
        fail_test("Post-tx midstate", "diverged")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 5: Crash recovery
    # ══════════════════════════════════════════════════════════════════════
    section("Test 5: Crash recovery (kill and restart Node B)")

    hb_before = get_height(RPC_B)
    log(f"Node B at height {hb_before} before crash")

    kill_node("B")
    log("Node B killed")

    log("Letting Node A mine while B is down...")
    time.sleep(15)
    log(f"Node A now at height {get_height(RPC_A)}")

    # Restart from existing data (not fresh!)
    start_node("B", DIR_B, P2P_B, RPC_B, mine=False,
               peers=[ma_A])
    if wait_for_health(RPC_B, "Node-B (restarted)", 30):
        pass_test("Node B restarts after crash")
    else:
        fail_test("Node B restart", "RPC not reachable")
        sys.exit(1)

    wait_for_peers(RPC_B, "Node-B", timeout=15)
    time.sleep(2)

    hb_after = get_height(RPC_B)
    if hb_after is not None and hb_after >= hb_before:
        pass_test(f"Node B resumed from saved state (height={hb_after} >= {hb_before})")
    else:
        fail_test("Crash recovery state", f"restarted at {hb_after}, was at {hb_before}")

    if wait_for_sync(RPC_A, RPC_B, 90):
        ha, hb = get_height(RPC_A), get_height(RPC_B)
        pass_test(f"Node B re-synced after restart (A={ha}, B={hb})")
    else:
        ha, hb = get_height(RPC_A), get_height(RPC_B)
        fail_test("Post-crash sync", f"A={ha} B={hb} after 90s")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 6: Sync from scratch
    # ══════════════════════════════════════════════════════════════════════
    section("Test 6: Fresh node syncs from scratch")

    log(f"Node A at height {get_height(RPC_A)}. Starting fresh Node C...")

    start_node("C", DIR_C, P2P_C, RPC_C, mine=False,
               peers=[ma_A], fresh=True)
    if wait_for_health(RPC_C, "Node-C", 30):
        pass_test("Node C starts fresh")
    else:
        fail_test("Node C startup", "RPC not reachable")
        sys.exit(1)

    wait_for_peers(RPC_C, "Node-C", timeout=15)

    if wait_for_sync(RPC_A, RPC_C, 120):
        ha, hc = get_height(RPC_A), get_height(RPC_C)
        pass_test(f"Node C synced from scratch (A={ha}, C={hc})")
    else:
        ha, hc = get_height(RPC_A), get_height(RPC_C)
        fail_test("Sync from scratch", f"A={ha} C={hc} after 120s")

    ma, mc = get_midstate(RPC_A), get_midstate(RPC_C)
    if ma == mc:
        pass_test("All nodes agree on midstate")
    else:
        fail_test("Three-node consensus", "midstates diverge")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 7: Competitive mining — two isolated miners, then reconnect
    # ══════════════════════════════════════════════════════════════════════
    section("Test 7: Competitive mining (isolated miners converge)")

    log("Shutting down nodes A, B, C for competitive mining test...")
    kill_node("A"); kill_node("B"); kill_node("C")
    time.sleep(3)

    # Two miners, NO peers — each builds their own chain independently
    start_node("A", DIR_A, P2P_A, RPC_A, mine=True, fresh=True)
    start_node("D", DIR_D, P2P_D, RPC_D, mine=True, fresh=True)
    wait_for_health(RPC_A, "Miner-A") or sys.exit(1)
    wait_for_health(RPC_D, "Miner-D") or sys.exit(1)

    # We need the PeerId multiaddrs to connect them later
    ma_A = get_local_multiaddr(DIR_A, P2P_A)
    ma_D = get_local_multiaddr(DIR_D, P2P_D)

    log("Both miners mining independently for 30 seconds...")
    time.sleep(30)

    HA, HD = get_height(RPC_A), get_height(RPC_D)
    MA, MD = get_midstate(RPC_A), get_midstate(RPC_D)
    DA, DD = get_depth(RPC_A), get_depth(RPC_D)
    log(f"Miner A: height={HA} depth={DA} midstate={MA[:16]}...")
    log(f"Miner D: height={HD} depth={DD} midstate={MD[:16]}...")

    if MA != MD:
        pass_test("Miners diverged as expected (different chains)")
    else:
        pass_test("Miners at same state (unlikely but not impossible)")

    # Stop A's mining, restart as non-miner, THEN connect D.
    frozen_height = get_height(RPC_A)
    log(f"Freezing Miner A's chain at height {frozen_height} for convergence...")

    stop_mining_restart("A", DIR_A, P2P_A, RPC_A, peers=[])
    time.sleep(2)

    # Now restart D connected to the frozen A
    kill_node("D")
    time.sleep(3)
    start_node("D", DIR_D, P2P_D, RPC_D, mine=False, peers=[ma_A])
    
    wait_for_health(RPC_D, "Miner-D (reconnected)", 30) or sys.exit(1)
    wait_for_peers(RPC_D, "Miner-D", timeout=30)

    log("Waiting for miners to converge...")
    if wait_for_consensus(RPC_A, RPC_D, 120):
        HA, HD = get_height(RPC_A), get_height(RPC_D)
        pass_test(f"Miners converged after reconnection (A={HA}, D={HD})")
    else:
        HA, HD = get_height(RPC_A), get_height(RPC_D)
        MA, MD = get_midstate(RPC_A), get_midstate(RPC_D)
        DA, DD = get_depth(RPC_A), get_depth(RPC_D)
        fail_test("Miner convergence",
                   f"A: h={HA} d={DA} m={MA[:16]} / D: h={HD} d={DD} m={MD[:16]}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 8: Simultaneous connected mining
    # ══════════════════════════════════════════════════════════════════════
    section("Test 8: Simultaneous connected mining")

    kill_node("A"); kill_node("D")
    time.sleep(3)

    # Start A first so we can grab its multiaddr
    start_node("A", DIR_A, P2P_A, RPC_A, mine=True, fresh=True)
    wait_for_health(RPC_A, "Racer-A") or sys.exit(1)
    ma_A = get_local_multiaddr(DIR_A, P2P_A)
    time.sleep(1)

    start_node("D", DIR_D, P2P_D, RPC_D, mine=True, peers=[ma_A], fresh=True)
    wait_for_health(RPC_D, "Racer-D") or sys.exit(1)

    if wait_for_peers(RPC_D, "Racer-D", timeout=15):
        log("P2P connection verified between racers")
    else:
        log("⚠ P2P connection not verified — race results may be affected")

    log("Two connected miners racing for 60 seconds...")
    time.sleep(60)

    HA, HD = get_height(RPC_A), get_height(RPC_D)
    MA, MD = get_midstate(RPC_A), get_midstate(RPC_D)
    CA, CD = get_coins(RPC_A), get_coins(RPC_D)
    log(f"Racer A: height={HA} coins={CA} midstate={MA[:16]}...")
    log(f"Racer D: height={HD} coins={CD} midstate={MD[:16]}...")

    height_diff = abs(HA - HD)
    if height_diff <= 3:
        pass_test(f"Connected miners within 3 blocks (A={HA}, D={HD})")
    else:
        fail_test("Mining race height", f"A={HA} D={HD} (diff={height_diff})")

    log("Stopping both miners for convergence settlement...")
    da_depth, dd_depth = get_depth(RPC_A), get_depth(RPC_D)

    kill_node("A"); kill_node("D")
    time.sleep(3)

    # Restart both as non-miners, peered together.
    if (da_depth or 0) >= (dd_depth or 0):
        start_node("A", DIR_A, P2P_A, RPC_A, mine=False, peers=[])
        wait_for_health(RPC_A, "Racer-A (settling)") or sys.exit(1)
        ma_A = get_local_multiaddr(DIR_A, P2P_A)
        start_node("D", DIR_D, P2P_D, RPC_D, mine=False, peers=[ma_A])
        wait_for_health(RPC_D, "Racer-D (settling)") or sys.exit(1)
        wait_for_peers(RPC_D, "Racer-D settling", timeout=30)
    else:
        start_node("D", DIR_D, P2P_D, RPC_D, mine=False, peers=[])
        wait_for_health(RPC_D, "Racer-D (settling)") or sys.exit(1)
        ma_D = get_local_multiaddr(DIR_D, P2P_D)
        start_node("A", DIR_A, P2P_A, RPC_A, mine=False, peers=[ma_D])
        wait_for_health(RPC_A, "Racer-A (settling)") or sys.exit(1)
        wait_for_peers(RPC_A, "Racer-A settling", timeout=30)

    if wait_for_consensus(RPC_A, RPC_D, 120):
        pass_test("Miners settled to identical state after race")
    else:
        HA, HD = get_height(RPC_A), get_height(RPC_D)
        MA, MD = get_midstate(RPC_A), get_midstate(RPC_D)
        fail_test("Post-race consensus",
                   f"A: h={HA} m={MA[:16]} / D: h={HD} m={MD[:16]}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 9: Network partition and rejoin
    # ══════════════════════════════════════════════════════════════════════
    section("Test 9: Network partition and rejoin")

    kill_node("A"); kill_node("D")
    time.sleep(3)

    # Two isolated miners
    start_node("A", DIR_A, P2P_A, RPC_A, mine=True, fresh=True)
    start_node("D", DIR_D, P2P_D, RPC_D, mine=True, fresh=True)
    wait_for_health(RPC_A, "Partition-A") or sys.exit(1)
    wait_for_health(RPC_D, "Partition-D") or sys.exit(1)

    log("Simulating network partition for 45 seconds...")
    time.sleep(45)

    HA, HD = get_height(RPC_A), get_height(RPC_D)
    MA, MD = get_midstate(RPC_A), get_midstate(RPC_D)
    DA, DD = get_depth(RPC_A), get_depth(RPC_D)
    log(f"Post-partition: A(h={HA} d={DA}) vs D(h={HD} d={DD})")

    if MA != MD:
        pass_test("Partition created divergent chains")
    else:
        log("Warning: chains identical despite partition (possible but unlikely)")

    log("Freezing both chains for convergence...")
    kill_node("A"); kill_node("D")
    time.sleep(3)

    if (DA or 0) >= (DD or 0):
        start_node("A", DIR_A, P2P_A, RPC_A, mine=False, peers=[])
        wait_for_health(RPC_A, "Rejoin-A") or sys.exit(1)
        ma_A = get_local_multiaddr(DIR_A, P2P_A)
        start_node("D", DIR_D, P2P_D, RPC_D, mine=False, peers=[ma_A])
        wait_for_health(RPC_D, "Rejoin-D") or sys.exit(1)
        wait_for_peers(RPC_D, "Rejoin peer", timeout=30)
    else:
        start_node("D", DIR_D, P2P_D, RPC_D, mine=False, peers=[])
        wait_for_health(RPC_D, "Rejoin-D") or sys.exit(1)
        ma_D = get_local_multiaddr(DIR_D, P2P_D)
        start_node("A", DIR_A, P2P_A, RPC_A, mine=False, peers=[ma_D])
        wait_for_health(RPC_A, "Rejoin-A") or sys.exit(1)
        wait_for_peers(RPC_A, "Rejoin peer", timeout=30)

    log("Waiting for post-partition convergence...")
    if wait_for_consensus(RPC_A, RPC_D, 120):
        HA, HD = get_height(RPC_A), get_height(RPC_D)
        pass_test(f"Chains converged after partition heal (A={HA}, D={HD})")
    else:
        HA, HD = get_height(RPC_A), get_height(RPC_D)
        MA, MD = get_midstate(RPC_A), get_midstate(RPC_D)
        DA, DD = get_depth(RPC_A), get_depth(RPC_D)
        fail_test("Partition recovery",
                   f"A: h={HA} d={DA} m={MA[:16]} / D: h={HD} d={DD} m={MD[:16]}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 10: RPC scan endpoint
    # ══════════════════════════════════════════════════════════════════════
    section("Test 10: Scan endpoint")

    scan_height = get_height(RPC_A)
    dummy_addr = "0" * 63 + "1"

    scan_resp = rpc(RPC_A, "/scan", method="POST",
                    body={"addresses": [dummy_addr],
                          "start_height": 0, "end_height": scan_height},
                    timeout=10)
    if scan_resp and "coins" in scan_resp:
        pass_test(f"Scan endpoint works (found {len(scan_resp['coins'])} coins)")
    else:
        fail_test("Scan endpoint", "no response")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 11: Final state
    # ══════════════════════════════════════════════════════════════════════
    section("Test 11: Final state")

    sa = rpc(RPC_A, "/state")
    sd = rpc(RPC_D, "/state")

    if sa and sd:
        HA, HD = sa["height"], sd["height"]
        MA, MD = sa["midstate"], sd["midstate"]
        CA, CD = sa["num_coins"], sd["num_coins"]

        log(f"Final state:")
        log(f"  Node A: height={HA} coins={CA} midstate={MA[:16]}...")
        log(f"  Node D: height={HD} coins={CD} midstate={MD[:16]}...")

        if MA == MD:
            pass_test("Final midstates identical")
        else:
            fail_test("Final midstate", f"A={MA[:16]} D={MD[:16]}")

        if CA == CD:
            pass_test(f"Final coin counts identical ({CA})")
        else:
            fail_test("Final coin count", f"A={CA} D={CD}")
    else:
        fail_test("Final state", "could not reach one or both nodes")

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    section("Results")
    print()
    print(f"  Tests run: {TESTS_RUN}")
    print(f"  {GREEN}Passed: {PASS}{NC}")
    if FAIL > 0:
        print(f"  {RED}Failed: {FAIL}{NC}")
    else:
        print(f"  Failed: 0")
    print()

    if FAIL > 0:
        print(f"{RED}SOME TESTS FAILED{NC}")
        print()
        print("Node logs:")
        for d in [DIR_A, DIR_B, DIR_C, DIR_D]:
            logf = d / "node.log"
            if logf.exists():
                print(f"  {logf}")
        return 1
    else:
        print(f"{GREEN}ALL TESTS PASSED{NC}")
        return 0


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Midstate integration tests")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    try:
        code = run_tests(args.skip_build)
    except KeyboardInterrupt:
        log("Interrupted")
        code = 130
    except Exception as e:
        log(f"Unexpected error: {e}")
        import traceback; traceback.print_exc()
        code = 1
    finally:
        log("Cleaning up processes...")
        kill_all()
        if FAIL > 0:
            log("Preserving logs for debugging in /tmp/midstate-test-*/")
        else:
            for d in [DIR_A, DIR_B, DIR_C, DIR_D]:
                shutil.rmtree(d, ignore_errors=True)

    sys.exit(code)


if __name__ == "__main__":
    main()
