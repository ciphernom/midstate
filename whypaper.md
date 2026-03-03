# Midstate: A Minimal Peer-to-Peer Post-Quantum Electronic Cash System

### 1. Introduction

Modern blockchain networks have come to rely almost exclusively on massive data centers and specialized hardware to maintain consensus and process transactions. While these systems successfully operate without trusted third parties, their architectural choices have recreated the very centralization they sought to eliminate. As state databases bloat indefinitely, the cost of running a fully validating node has priced out the average user. Simultaneously, parallelized Proof-of-Work has concentrated network control into the hands of a few industrial mining farms, while Stratum pool operators hold unchecked power to censor transactions.

Furthermore, the fundamental cryptographic primitives underpinning these networks—specifically Elliptic Curve Cryptography (secp256k1)—are highly vulnerable to the looming advent of quantum computing. Finally, the transparent nature of modern mempools has given rise to Miner Extractable Value (MEV), allowing block producers to front-run users and extract wealth.

We propose a solution to these centralization and security vulnerabilities. Midstate is a purely peer-to-peer electronic cash system engineered from the ground up to run on extreme edge hardware. By abandoning elliptic curve mathematics entirely, enforcing sequential computational work, and separating transaction commitments from their execution, we create a network that is post-quantum secure, naturally resistant to MEV, and strictly bounds the hardware floor to a $15 edge device.

### 2. Transactions & State

We define an electronic coin as a cryptographically bound payload of value. In existing systems, transactions broadcast the spender's intentions in cleartext, allowing miners to reorder or inject transactions for their own profit (MEV). We propose a two-phase Commit-Reveal protocol to prevent front-running:

* **Phase 1 (Commit):** The sender broadcasts a cryptographic hash binding the inputs and outputs, along with a minor Proof-of-Work spam nonce to prevent network flooding. This carries zero fee and obscures the transaction details.
* **Phase 2 (Reveal):** Once the commitment is mined into the chain, the sender opens the commitment, providing the necessary signatures, executing scripts, and paying the network fee.

To govern these coins, we utilize **MidstateScript**, a Turing-incomplete stack machine requiring zero gas fees. Execution time is strictly bounded to $O(N)$. To guarantee determinism and simplicity, there are no loops or backward jumps (`LOOP` or `JUMP`). MidstateScript supports advanced covenants via the `SUM_TO_ADDR` opcode, allowing users to build un-censorable on-chain limit orders and atomic swaps.

To resolve the problem of unbounded global state bloat, we abandon the traditional UTXO database model. Instead, we accumulate UTXOs into a Sparse Merkle Tree (SMT). This mathematically bounds the memory usage of the active state. Historic block batches are pruned, requiring validating nodes to retain only lightweight headers and state snapshots for rapid fast-forward syncing.

### 3. Post-Quantum Cryptography

Future quantum computers running Shor's algorithm will trivially break the discrete logarithm problem underlying Elliptic Curve Cryptography. To ensure permanent security, Midstate uses exactly zero elliptic curve mathematics. The BLAKE3 hash function (producing a 32-byte output) is the sole cryptographic primitive used across the entire protocol.

Every address in Midstate is a Pay-to-Script-Hash (P2SH), where a coin is defined as:
$CoinID = BLAKE3(address \parallel value\_le\_bytes \parallel salt)$

We propose two tiers of hash-based signatures:

* **Signatures (Single-Use):** We implement Winternitz One-Time Signatures (WOTS) utilizing the parameter $w=16$. This generates highly compact 576-byte post-quantum signatures suitable for standard transactions.
* **Signatures (Reusable):** For entities requiring static receiving addresses, we implement a Merkle Signature Scheme (MSS) utilizing a binary tree of WOTS keys. This allows exactly $2^H$ signatures per master public key. To prevent catastrophic key reuse, the protocol verifier strictly enforces the `leaf_index` against the cryptographic authentication path, ensuring a WOTS leaf can never be consumed twice.

### 4. Sequential Proof-of-Work

The original vision of "one CPU, one vote" has been subverted by ASICs and parallel computing, which evaluate millions of independent hashes concurrently.

To defeat parallelized mining and return consensus to general-purpose hardware, we propose a Sequential Proof-of-Work mechanism. Mining a valid block extension requires 1,000,000 strictly sequential BLAKE3 hashes, simulating a Verifiable Delay Function (VDF):
$x_i = BLAKE3(x_{i-1})$

While miners can search for valid starting nonces in parallel across multiple cores, calculating a single proof is fundamentally single-threaded. It is impossible to divide the calculation of $x_{1000000}$ across multiple processors. This negates the advantage of highly parallelized ASIC architectures and re-democratizes the mining process.

### 5. Network Consensus

Traditional networks adjust their difficulty in discrete windows, creating vulnerabilities to time-warp and sliding-window exploits where miners manipulate timestamps to artificially lower difficulty.

We solve this using the ASERT (Absolutely Scheduled Exponentially Decaying) difficulty adjustment algorithm. Programmed with a 4-hour half-life, the difficulty adjusts continuously on every single block based on the absolute time elapsed since the genesis block.

Furthermore, we abandon arbitrary and unscientific confirmation counts (e.g., "wait 6 blocks for finality"). Instead, Midstate utilizes a Bayesian Finality Estimator. Nodes locally observe network behavior and use a Beta-Binomial model to continuously calculate a dynamic $safe\_depth$. This depth dictates exactly how many blocks must be chained to guarantee a 1-in-a-million ($1 \times 10^{-6}$) risk of a successful chain reorganization by a malicious actor.

### 6. Decentralized Pool Mining

Mining pools are a necessary reality to reduce variance in payouts, but traditional Stratum-based pools require the pool operator to construct the block template. This gives the central operator absolute power to censor transactions and dictate network rules.

We propose flipping the pool model entirely. In Midstate, the local miner autonomously selects transactions from their own local mempool and builds the block template. The block reward is hardcoded to pay the Pool's MSS address. To prove they performed the work, the miner cryptographically watermarks their personal payout address into the Coinbase $salt$. The pool operator acts purely as an automated ingestion and payout engine, effectively stripped of all censorship and block-construction power.

### 7. Privacy

Privacy is rarely achieved when transaction amounts are highly specific, as graph analysis can easily map change outputs back to original senders.

To break these heuristics, all UTXOs in Midstate must possess values that are exact powers of 2. Because of this strict uniformity, the node software includes a native P2P CoinJoin coordinator. Participants can seamlessly mix identical denominations in a trustless environment. To prevent Sybil attackers from griefing or stalling mix sessions, a minor Proof-of-Work challenge must be solved to join a mix.

At the network layer, transaction propagation privacy is achieved via Dandelion++ routing. Transactions originate in a "stem" phase, where they are routed linearly to single peers. After a randomized delay, the transaction enters a "fluff" phase where it is probabilistically broadcast to the global network. This obfuscates the originator's IP address from network observers.

### 8. Reclaiming the Edge

The overarching design philosophy of Midstate is to bound hardware requirements so strictly that the network remains perfectly decentralized. Modern blockchains require vast resources to sync and validate the chain state.

Midstate is explicitly engineered to run a full validating node and active miner on a $15 Raspberry Pi Zero 2 W with only 512MB of RAM. By combining the SMT UTXO accumulator, pruned historical block batches, and lightweight headers, the protocol drastically reduces disk I/O, bandwidth, and memory footprints. Fast-forward syncing allows edge nodes to join the network, verify the Bayesian safe depth, and achieve consensus without needing a datacenter.

### 9. Conclusion

We have proposed a post-quantum system for electronic transactions that fundamentally rejects hardware centralization and operator censorship. By relying strictly on the BLAKE3 hash function, we ensure long-term cryptographic security. Through Sequential Proof-of-Work, ASERT difficulty, and a flipped pool-mining architecture, we return block construction and consensus to the individual edge node. By implementing power-of-2 denominations with native CoinJoin, Dandelion++ routing, and a Commit-Reveal execution environment, we protect the user from both network surveillance and mempool extraction. The result is a robust, mathematically bounded protocol capable of validating global state on a 512MB device.
