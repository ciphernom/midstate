#!/bin/bash

# 1. Prompt for password securely ONCE and export it so the CLI doesn't ask again
read -s -p "Enter Midstate Wallet Password: " MIDSTATE_PASSWORD
echo
export MIDSTATE_PASSWORD

# Path to your midstate binary
CLI="./target/release/midstate"

echo "Starting unattended wallet defragmentation..."

while true; do
    echo "============================================================"
    
    # 2. Safety Sweep: Reveal any commits that timed out in the last loop
    # Running 'reveal' without a specific commitment tells the wallet to 
    # find ALL pending commits and attempt to reveal them.
    echo "[1/2] Checking for pending commits..."
    $CLI wallet reveal

    # 3. Execute the next defrag batch (up to 256 inputs)
    echo "[2/2] Running defrag batch..."
    OUTPUT=$($CLI wallet defrag 2>&1)
    
    # Print the output to the screen so you can monitor progress
    echo "$OUTPUT"

    # 4. Check exit conditions
    if echo "$OUTPUT" | grep -q "No defragmentation needed"; then
        echo "✅ Defragmentation fully complete!"
        break
    fi

    if echo "$OUTPUT" | grep -q "Could not construct an economical defrag batch"; then
        echo "✅ Remaining fragmented dust is too small to cover network fees. Done!"
        break
    fi

    # 5. Short breather for network/mempool settling
    echo "⏳ Waiting 15 seconds before building the next batch..."
    sleep 15
done
