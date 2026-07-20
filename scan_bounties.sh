#!/bin/bash

# Configuration
RPC_URL="http://127.0.0.1:8545"
START_HEIGHT=167000

# 1. Fetch the current tip height from the node
STATE=$(curl -s -f "$RPC_URL/state")
if [ $? -ne 0 ]; then
    echo "Error: Could not connect to local node at $RPC_URL"
    exit 1
fi

END_HEIGHT=$(echo "$STATE" | jq -r '.height')

if [ -z "$END_HEIGHT" ] || [ "$END_HEIGHT" == "null" ]; then
    echo "Error: Could not determine current chain height."
    exit 1
fi

echo "Scanning blocks $START_HEIGHT to $END_HEIGHT for Key Reuse Bounty transactions..."

# 2. Iterate through all blocks
for (( HEIGHT=START_HEIGHT; HEIGHT<=END_HEIGHT; HEIGHT++ )); do
    # Print progress on the same line so it doesn't flood the terminal
    if (( HEIGHT % 10 == 0 )); then
        echo -ne "Scanning block $HEIGHT / $END_HEIGHT...\r"
    fi

    # Fetch the block data (silently)
    BLOCK_JSON=$(curl -s -f "$RPC_URL/batch/$HEIGHT")
    if [ $? -ne 0 ]; then
        continue # Block might not exist yet or node is busy
    fi

    # 3. Use jq to parse the block and find the specific DataBurn payload.
    # The `any()` function safely checks both Reveal and Consolidate outputs without crashing on Commits.
    MATCH=$(echo "$BLOCK_JSON" | jq -c '
        .transactions[]? |
        select(
            any(
                .Reveal.outputs[]?, .Consolidate.outputs[]?; 
                .DataBurn.payload? | type == "array" and implode == "PUNISHED FOR KEY REUSE"
            )
        )
    ')

    # 4. If a match is found, print it out nicely
    if [ -n "$MATCH" ]; then
        # Clear the progress line
        echo -e "\n\n🚨 Found Bounty Hunter Burn at Block $HEIGHT! 🚨"
        
        # Pretty-print the matched transaction
        echo "$MATCH" | jq .
        
        echo "--------------------------------------------------------"
    fi
done

echo -e "\nScan complete!"
