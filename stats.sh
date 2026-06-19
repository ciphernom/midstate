#!/bin/bash

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Aborting."
    exit 1
fi

echo -e "\e[1;36mFetching statistics from local node (127.0.0.1:8545)...\e[0m"
echo -e "\e[33mPlease wait 3 seconds to measure local speed...\e[0m"
echo ""

python3 -c '
import urllib.request, json, time, sys

RPC_URL = "http://127.0.0.1:8545"
EXTENSION_ITERS = 1_000_000  # 1 Nonce = 1 Million Hashes in Midstate
BLOCK_TIME = 60              # ASERT targets 60 seconds per block

def format_hashrate(h):
    if h >= 1e12: return f"{h/1e12:.2f} TH/s"
    if h >= 1e9:  return f"{h/1e9:.2f} GH/s"
    if h >= 1e6:  return f"{h/1e6:.2f} MH/s"
    if h >= 1e3:  return f"{h/1e3:.2f} kH/s"
    return f"{h:.2f} H/s"

try:
    # 1. Fetch Global Network Target
    req = urllib.request.Request(f"{RPC_URL}/state", headers={"User-Agent": "Mozilla/5.0"})
    state = json.loads(urllib.request.urlopen(req, timeout=5).read())
    target_hex = state["target"]
    target_int = int(target_hex, 16)
    
    if target_int == 0: target_int = 1
        
    # Expected nonces to solve a block = (2^256) / Target
    expected_nonces = (1 << 256) // target_int
    network_nps = expected_nonces / BLOCK_TIME
    network_hps = network_nps * EXTENSION_ITERS
    
    # 2. Fetch Local Speed (Measure difference over 3 seconds)
    req1 = urllib.request.Request(f"{RPC_URL}/axe/stats", headers={"User-Agent": "Mozilla/5.0"})
    stats1 = json.loads(urllib.request.urlopen(req1, timeout=5).read())
    nonces_start = stats1["total_nonces"]
    
    time.sleep(3.0)
    
    req2 = urllib.request.Request(f"{RPC_URL}/axe/stats", headers={"User-Agent": "Mozilla/5.0"})
    stats2 = json.loads(urllib.request.urlopen(req2, timeout=5).read())
    nonces_end = stats2["total_nonces"]
    
    local_nps = (nonces_end - nonces_start) / 3.0
    local_hps = local_nps * EXTENSION_ITERS
    
    # Calculate network share
    share = (local_nps / network_nps * 100) if network_nps > 0 else 0

    # 3. Print Results
    print(f"\033[1;37m=== MIDSTATE MINING MONITOR ===\033[0m")
    print(f"\033[1;32mNetwork Speed:\033[0m   {network_nps:,.2f} N/s  \033[1;30m({format_hashrate(network_hps)} raw)\033[0m")
    print(f"\033[1;34mLocal Speed:  \033[0m   {local_nps:,.2f} N/s  \033[1;30m({format_hashrate(local_hps)} raw)\033[0m")
    print(f"\033[1;35mNetwork Share:\033[0m   {share:.5f}%")
    print(f"\033[1;37m===============================\033[0m")
    print(f"\033[3;37m* 1 Nonce (N) = 1,000,000 sequential BLAKE3 hashes\033[0m")

except urllib.error.URLError:
    print("\033[1;31mError: Could not connect to node.\033[0m")
    print("Make sure your Midstate node is running and the RPC is listening on 127.0.0.1:8545.")
except Exception as e:
    print(f"\033[1;31mError:\033[0m {e}")
'
