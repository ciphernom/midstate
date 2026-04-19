# Midstate Formal Specification

## 1. Conventions and Notation

The Midstate protocol is defined as a deterministic state transition system. Let $\sigma_t$ represent the global state at block height $t$. The block transition function $\Upsilon$ maps a state and a valid block $B$ to a new state $\sigma_{t+1} \equiv \Upsilon(\sigma_t, B)$. Transactions within a block are evaluated by the transaction transition function $\Pi$.

Let $\mathbb{B}$ denote the set of byte arrays. $\mathbb{B}_{32}$ denotes a byte array of exactly 32 bytes.
Let $\mathcal{H}: \mathbb{B} \rightarrow \mathbb{B}_{32}$ denote the BLAKE3 cryptographic hash function.
Concatenation of byte arrays $X$ and $Y$ is denoted as $X \parallel Y$.

## 2. Cryptographic Primitives

The protocol relies exclusively on symmetric-key primitives to ensure post-quantum security guarantees.

### 2.1. Winternitz One-Time Signatures (WOTS)

Midstate utilizes a Winternitz One-Time Signature scheme with the Winternitz parameter $w = 16$. A message $M \in \mathbb{B}_{32}$ is partitioned into sixteen 16-bit digits $m_0, m_1, ..., m_{15}$. 

A checksum $C$ is calculated to prevent existential forgery:
$$C = \sum_{i=0}^{15} (65535 - m_i)$$

The maximum value of $C$ is $16 \times 65535 = 1,048,560$, which fits within 20 bits. $C$ is encoded as two 16-bit digits $c_0, c_1$, appending to the message digits to form an 18-element array $D$.

Let $S_{seed} \in \mathbb{B}_{32}$ be the private seed. The private key array $sk$ of length 18 is derived as:
$$sk_i = \mathcal{H}(S_{seed} \parallel i_{u32})$$

The public key $pk \in \mathbb{B}_{32}$ is derived by hashing each $sk_i$ exactly 65535 times, and hashing the concatenation of the endpoints:
$$pk = \mathcal{H}(\mathcal{H}^{65535}(sk_0) \parallel ... \parallel \mathcal{H}^{65535}(sk_{17}))$$

A signature $\Sigma$ is an 18-element array of 32-byte values, where:
$$\Sigma_i = \mathcal{H}^{D_i}(sk_i)$$

Verification asserts the following equality:
$$\mathcal{H}(\mathcal{H}^{65535 - D_0}(\Sigma_0) \parallel ... \parallel \mathcal{H}^{65535 - D_{17}}(\Sigma_{17})) \equiv pk$$

### 2.2. Merkle Signature Scheme (MSS)

To facilitate address reuse, WOTS is encapsulated within a Merkle Signature Scheme. An MSS keypair of height $H \in [1, 20]$ bounds $2^H$ WOTS public keys.

Leaf generation for index $k$:
$$seed_k = \mathcal{H}(S_{master} \parallel k_{u64})$$
$$pk_k = WOTS_{keygen}(seed_k)$$

The Master Public Key $PK_{MSS}$ is the root of the binary Merkle tree constructed from all $2^H$ WOTS public keys. An MSS signature requires $576 + 8 + 32 + 4 + (33 \times H)$ bytes, encapsulating the leaf index $k$, the WOTS signature, the leaf public key, and the authentication path to $PK_{MSS}$.

## 3. Transaction Model

The system mitigates front-running through a strict two-phase commit and reveal scheme.

### 3.1. Phase 1: Commitment

A `Commit` transaction declares intent to spend without revealing inputs or outputs. It is defined as a tuple $(C, N)$, where $C \in \mathbb{B}_{32}$ is the commitment hash and $N \in \mathbb{N}$ is a spam prevention nonce.

The commitment hash $C$ is strictly formed as:
$$C = \mathcal{H}(\text{"MIDSTATE\_MAINNET\_V1"} \parallel len(I) \parallel I \parallel len(O) \parallel O \parallel salt)$$
Where $I$ is the concatenation of input CoinIDs, $O$ is the concatenation of output commitment hashes, and $salt \in \mathbb{B}_{32}$.

The nonce $N$ must satisfy a dynamic Proof-of-Work threshold evaluated at block height $h$:
$$\text{LeadingZeros}(\mathcal{H}(h_{u64} \parallel C \parallel N_{u64})) \ge \text{Threshold}$$

To ensure strict cross-architecture determinism and avoid floating-point consensus failures, the threshold evaluates mempool congestion ($|\mathcal{M}_{commit}|$) using deterministic integer arithmetic via a piecewise step function:
$$\text{Threshold} = 24 + \min\left(6, \left\lfloor \frac{\max(0, |\mathcal{M}_{commit}| - 500)}{50} \right\rfloor \right)$$

For blocks $h \ge 80,000$, a `Commit` transaction is valid if the PoW threshold is met for *any* evaluation height $h'$ in the sliding window $[h - 1000, h]$. This "Wiggle Room" ensures high-difficulty commits mined during periods of mempool congestion are not invalidated by rapid block generation before inclusion.

A valid commitment is recorded in the global state and remains valid for a Time-to-Live (TTL) of 1000 blocks.

### 3.2. Phase 2: Reveal

A `Reveal` transaction executes the state transition authorized by a prior commitment. It is a tuple $(I_{reveal}, W, O_{data}, salt)$.

#### 3.2.1. Inputs
An input reveal $I_r$ specifies a Predicate $P$, a value $v \in \mathbb{N}$, a salt $s \in \mathbb{B}_{32}$, and an optional state commitment $c_{state}$.
The associated $CoinID$ is derived based on the output class that created it:
*   Standard: $CoinID = \mathcal{H}(\mathcal{H}(P) \parallel v_{u64} \parallel s)$
*   Confidential: $CoinID = \mathcal{H}(\text{"CONFIDENTIAL"} \parallel \mathcal{H}(P) \parallel c_{state} \parallel s)$

#### 3.2.2. Outputs
An output $O_d$ belongs to one of three classes:
1. **Standard:** Defined by $(addr \in \mathbb{B}_{32}, v \in \mathbb{N}_{>0}, s \in \mathbb{B}_{32})$.
2. **Confidential:** Defines a zero-value state thread $(addr \in \mathbb{B}_{32}, c_{state} \in \mathbb{B}_{32}, s \in \mathbb{B}_{32})$.
3. **DataBurn:** A provably unspendable payload up to 80 bytes.

### 3.3. Structural Bounds and Conservation

Let a Reveal transaction $tx$ be bounded by consensus parameters to prevent computational exhaustion:
*   $|I_{reveal}| \in [1, 256]$ and $|O_{data}| \in [1, 256]$.
*   The total serialized size of all witnesses $W$ must not exceed $1536 \times |I_{reveal}|$ bytes.
*   If $o \in O_{data}$ is of class `DataBurn`, its payload length is strictly bounded to $L_{burn} \le 80$ bytes.
*   Every $o.v \in O_{data}$ (except `Confidential` outputs and `DataBurn`) must be a power of two ($o.v \ \& \ (o.v - 1) == 0$) and strictly $> 0$.

The network fee $\phi$ is implicitly defined by value conservation:
$$\phi = \sum_{i \in I} i.v - \sum_{o \in O} o.v$$
A transaction is strictly invalid if $\phi < 0$.

### 3.4. Consensus Commitment Locks & WOTS Dusting Mitigation

The consensus layer strictly enforces a **Commitment Hash Lock** to guarantee the one-time-use property of WOTS keys. 

If any transaction $tx$ consumes an input $i \in U$ located at address $A$, the global state persistently maps address $A$ to the transaction's commitment hash $C_{tx}$. Any subsequent or concurrent transaction attempting to spend from $A$ must produce the exact same commitment hash $C_{tx}$, or it is rejected by consensus.

**Dusting Attack Mitigation:** An adversary cannot invalidate a victim's in-flight transaction by dusting address $A$. The victim's transaction remains mathematically valid and will be mined, successfully locking address $A$ to the victim's $C_{tx}$. The attacker's dust coin arrives at $A$, but because $A$ is now locked to $C_{tx}$ (which dictates specific outputs the attacker cannot satisfy), the attacker's dust becomes permanently unspendable. The consensus rule protects the honest user while punishing the attacker. 

*(Note: The wallet layer implements automatic co-spending of all known UTXOs at an address to prevent the user from stranding their own funds, detailed in Section 11.3).*

## 4. Virtual Machine Specification

The Midstate Virtual Machine evaluates a stateless script $P_{script}$ against a cryptographic witness $W$. The VM is a stack-based processor limited to 1024 bytes of bytecode.

### 4.1. Execution Context

The execution context $\Omega$ is defined as $(C, h, O_{data}, v_{in}, c_{state})$, providing the script with introspection capabilities over the current transaction commitment $C$, block height $h$, sibling outputs $O_{data}$, the specific input value $v_{in}$, and input state $c_{state}$.

#### 4.1.1. Execution Limits and the Clean Stack Rule
The machine restricts the evaluation stack $S$ to a maximum depth of 64 elements. No single element $e \in S$ may exceed $L_{max} = 1536$ bytes. Arithmetic operations (`OP_ADD`, `OP_SUB`, `OP_MUL`, `OP_DIV`) treat byte arrays up to 8 bytes as 64-bit unsigned little-endian integers.

The script evaluates to `True` if and only if execution halts without error and the terminal stack state satisfies the **Clean Stack Rule**. This requires a strict, unpadded, single-byte array equality:
$$|S| = 1 \land S[0] \equiv [0x01]$$
Zero-padded arrays (e.g., `[0x01, 0x00]`) or multi-element stacks are evaluated as strictly invalid.

### 4.2. Instruction Set

Bytecode execution maps opcodes to stack mutations. Notation $S \rightarrow S'$ denotes the stack transformation.

*   `0x01` (`OP_PUSH_DATA`): Followed by a 2-byte little-endian length $L$ and $L$ bytes of data. Pushes data to stack.
*   `0x10` (`OP_DROP`): $a \rightarrow \emptyset$
*   `0x11` (`OP_DUP`): $a \rightarrow a, a$
*   `0x12` (`OP_SWAP`): $a, b \rightarrow b, a$
*   `0x13` (`OP_OVER`): $a, b \rightarrow a, b, a$
*   `0x14` (`OP_ROT`): $a, b, c \rightarrow b, c, a$
*   `0x15` (`OP_SLICE`): $data, offset, len \rightarrow data[offset .. offset+len]$. Halts on bounds violation.
*   `0x16` (`OP_CONCAT`): $a, b \rightarrow a \parallel b$. Halts if length exceeds 1536.
*   `0x20` (`OP_EQUAL`): $a, b \rightarrow 1$ if $a \equiv b$, else $0$.
*   `0x21` (`OP_VERIFY`): $a \rightarrow \emptyset$. Halts if $a$ evaluates to false (all bytes zero).
*   `0x22` (`OP_EQUALVERIFY`): $a, b \rightarrow \emptyset$. Halts if $a \neq b$.
*   `0x23` (`OP_ADD`): $a, b \rightarrow a + b$. Evaluates operands as 64-bit little-endian integers.
*   `0x25` (`OP_SUB`): $a, b \rightarrow a - b$.
*   `0x26` (`OP_MUL`): $a, b \rightarrow a \times b$.
*   `0x27` (`OP_DIV`): $a, b \rightarrow \lfloor a / b \rfloor$. Halts on division by zero.
*   `0x24` (`OP_GREATER_OR_EQUAL`): $a, b \rightarrow 1$ if $a \ge b$, else $0$.
*   `0x30` (`OP_HASH`): $a \rightarrow \mathcal{H}(a)$.
*   `0x31` (`OP_CHECKSIG`): $sig, pk \rightarrow 1$ if signature is valid for context $C$, else $0$. Supports both WOTS and MSS signatures, differentiated by byte length.
*   `0x32` (`OP_CHECKSIGVERIFY`): $sig, pk \rightarrow \emptyset$. Halts on invalid signature. Max 3 sigops per script.
*   `0x33` (`OP_CHECKTIMEVERIFY`): $t \rightarrow \emptyset$. Halts if $t > \Omega.h$.
*   `0x50` (`OP_SUM_TO_ADDR`): $addr \rightarrow \sum_{o \in \Omega.O_{data}, o.addr \equiv addr} o.v$.
*   `0x51` (`OP_READ_INPUT_STATE`): $\emptyset \rightarrow \Omega.c_{state}$. Halts if undefined.
*   `0x52` (`OP_READ_OUTPUT_STATE`): $idx \rightarrow \Omega.O_{data}[idx].c_{state}$. Halts if undefined or out of bounds.
*   `0x40..0x42` (`OP_IF`, `OP_ELSE`, `OP_ENDIF`): Control flow mapped to a boolean execution stack.

**Note on VM Upgrades:** Operations `0x13` (`OP_OVER`), `0x14` (`OP_ROT`), `0x15` (`OP_SLICE`), `0x16` (`OP_CONCAT`), `0x25` (`OP_SUB`), `0x26` (`OP_MUL`), and `0x27` (`OP_DIV`) are gated behind the V3 VM Upgrade at block height $h_{v3} = 60,000$. Execution of these opcodes where $h < h_{v3}$ yields an immediate halt condition (`InvalidOpcode`).

## 5. Global State and Accumulators

The global state $\sigma_t$ comprises several cryptographic accumulators.

### 5.1. Bounded Sparse Merkle Tree (SMT)

To achieve $O(1)$ memory overhead while maintaining a depth of 256, the UTXO and Commitment accumulators are implemented as hybrid memory-bounded Sparse Merkle Trees. 

The tree is split at depth $H_{cache} = 240$. The top 16 levels ($2^{16}$ nodes maximum) are cached in memory. The remaining 240 levels are computed dynamically via binary search over lexicographically sorted 32-byte elements.

Let $P$ be the 16-bit prefix of an element. Elements sharing $P$ belong to bucket $B_P$. The root of bucket $B_P$ at depth 240 is evaluated dynamically:

$$
Root(B_P, h) = \begin{cases} 
EmptyHash_h & \text{if } |B_P| = 0 \\ 
B_P[0] & \text{if } h = 0 \\ 
\mathcal{H}(Root(B_{Left}, h-1) \parallel Root(B_{Right}, h-1)) & \text{otherwise} 
\end{cases}
$$

Where $EmptyHash_h$ is defined recursively as $EmptyHash_0 = [0]_{32}$ and $EmptyHash_k = \mathcal{H}(EmptyHash_{k-1} \parallel EmptyHash_{k-1})$.

### 5.2. Merkle Mountain Range (MMR) Peak Bagging

Block final hashes are appended to a Merkle Mountain Range (MMR). The MMR allows $O(\log N)$ append operations and inclusion proofs. 

For $N$ block hashes, the MMR decomposes the forest into a series of perfectly balanced binary trees (peaks). The peak positions for $N$ leaves are derived deterministically. Let $S = 2N - popcount(N)$ be the total node count. Peaks are identified by finding the largest $2^h - 1 \le S$, recording it, and subtracting it from $S$ iteratively.

The canonical MMR root is computed by "bagging" the peaks from right to left (smallest to largest):
$$Root_{MMR} = \mathcal{H}(P_{n-1} \parallel \dots \mathcal{H}(P_2 \parallel \mathcal{H}(P_1 \parallel P_0)))$$

## 6. Consensus and Block Generation

### 6.1. Sequential Proof of Work

Blocks are finalized through a linear, non-parallelizable Proof-of-Work. Given a target threshold $T \in \mathbb{B}_{32}$, a mining midstate $\mu_{mine} \in \mathbb{B}_{32}$, and a nonce $N$, the algorithm computes an iterative hash chain of length $1,000,000$.

Let $x_0 = \mathcal{H}(\mu_{mine} \parallel N_{u64})$.
Let $x_k = \mathcal{H}(x_{k-1})$ for $k \in [1, 1000000]$.
The block is valid if and only if $x_{1000000} < T$.

This mechanism ensures verification takes exactly $1,000,000$ operations (completed in $\sim 1$ ms), precluding any subset-grinding attacks.

### 6.2. Difficulty Adjustment (ASERT)

The target $T$ is recalculated at every block to target a mean block interval of 60 seconds, using the Absolute Schedule Error Relative Target algorithm with a half-life of 14,400 seconds.

Let $t_{actual} = \text{block timestamp} - \text{genesis timestamp}$.
Let $t_{ideal} = \text{block height} \times 60$.
Drift $\Delta t = t_{actual} - t_{ideal}$.

The exponent scaling factor is computed in 16.16 fixed-point arithmetic:
$$E_{raw} = \frac{\Delta t \times 65536}{14400}$$

The fractional part $x = E_{raw} \pmod{65536}$ is expanded using a third-order Taylor polynomial to approximate $2^x$:
$$f = 65536 + \frac{x \times 45426}{65536} + \frac{x^2 \times 15746}{4294967296} + \frac{x^3 \times 3643}{281474976710656}$$

The new target is calculated by scaling the genesis target by $f$, adjusted by bitwise shifts corresponding to $\lfloor E_{raw} / 65536 \rfloor$.

### 6.3. Block Transition Function

The function $\Upsilon(\sigma_{prev}, B)$ computes the new state $\sigma_{next}$.

1. **Midstate Chaining:** The base midstate $\mu_0$ equals $\sigma_{prev}.midstate$.
2. **Transaction Application:** For each $t_i \in B_{tx}$, the commitment or reveal hash is incorporated: $\mu_{i+1} = \mathcal{H}(\mu_i \parallel \mathcal{H}(t_i))$.
3. **Coinbase Application:** Coinbase outputs are generated and validated against emission schedules. The resulting IDs are incorporated into $\mu$.
4. **State Root Integration:** The SMT root of coins, SMT root of commitments, and MMR root of blocks are concatenated and hashed to form the $state\_root$, which is appended to $\mu$ yielding $\mu_{mine}$.
5. **Validation:** The sequential PoW is verified against $\mu_{mine}$, $B.extension.nonce$, and $B.target$.
6. **Key Reuse Enforcement:** The protocol asserts that no WOTS address or MSS leaf index present in $B$ was previously mapped to a differing commitment hash in the historical global state.

### 6.4. Block Structural Limits

To bound propagation latency and prevent computational denial-of-service, a valid block $B$ is strictly capped:
*   Maximum Phase 1 `Commit` transactions: $C_{max} = 2000$
*   Maximum Phase 2 `Reveal` transactions: $R_{max} = 500$
*   Total inputs across all transactions: $I_{max} = 1024$
*   Total outputs across all transactions: $O_{max} = 10000$

### 6.5. Emission Schedule and Coinbase Constraints

The block reward $R_h$ at height $h$ is subject to a discrete halving schedule every $Y = 525,600$ blocks (approximately one year at 60-second intervals). The initial reward is $V_{init} = 1,073,741,824$ ($2^{30}$).
$$R_h = \max\left(1, V_{init} \gg \min\left(\left\lfloor \frac{h}{Y} \right\rfloor, 30\right)\right)$$

The total coinbase value $V_{cb}$ must exactly equal $R_h + \sum \phi$ (the block reward plus all transaction fees). Furthermore, to comply with the uniform denomination constraint, $V_{cb}$ must be decomposed perfectly into its binary power-of-two components. Every $o \in B_{coinbase}$ must satisfy $o.v = 2^k$ for some integer $k$.

## 7. Network and Privacy Layer

### 7.1. Dandelion++ Routing

To protect transaction originator privacy, network broadcasting utilizes the Dandelion++ protocol.
*   **Stem Phase:** A node originating a Commit transaction forwards it to exactly one randomly selected peer.
*   **Fluff Phase:** The receiving peer evaluates a dynamic probability condition, $P_{fluff} = \max(2, \min(50, 100 / N_{outbound}))$. If the condition is met, the transaction enters the public mempool and is broadcast symmetrically. Otherwise, the stem phase continues to a subsequent peer.
*   **Fail-safe:** Transactions held in a local stem pool exceeding 30 seconds are autonomously fluffed to prevent network censorship.

### 7.2. Compact Block Filters

The protocol supports light-client validation via Golomb-Rice encoded sets. 
The filter is configured with a False Positive Rate (FPR) of $1$ in $1,000,000$, yielding a Golomb-Rice parameter $P = 20$ ($2^{20} \approx 10^6$). 

For block $B$, a set $K$ of all relevant $\mathbb{B}_{32}$ identifiers (Commitments, CoinIDs, Addresses) is constructed and deduplicated. Let $N = |K|$. The hash modulus is defined as $M = N \times 1,000,000$. Each element $k$ maps to $h_k = (\text{BLAKE3}(B.final\_hash \parallel k)_{u64\_le}) \pmod M$.

The sorted values $h_k$ are difference-encoded. Each delta $\Delta_i = h_i - h_{i-1}$ is split into a quotient $q = \lfloor \Delta_i / 2^{20} \rfloor$ and remainder $r = \Delta_i \pmod{2^{20}}$. The quotient is encoded in unary ( $q$ bits of `1` followed by a `0`), and the remainder in 20 bits of binary.

To prevent memory exhaustion attacks from malicious peers, clients cap the maximum decodable filter size to $|bytes| \times 8$ elements.

### 7.3. Bayesian Finality Estimation

A localized Bayesian model estimates the safety of the chain tip.
Let $\alpha$ represent the count of sequential block extensions (honest behavior) and $\beta$ represent the count of observed reorganizations or orphaned blocks (adversarial behavior).

The expected probability of honest network dominance is $p = \frac{\alpha}{\alpha + \beta}$.
For a given reorganization depth $z$, the probability of a malicious shadow chain overtaking the canonical chain is given by the catchup probability formula $P_{catchup} = \left(\frac{1-p}{p}\right)^z$ for $p > 0.5$, and $1.0$ otherwise.

The node calculates the minimum safe depth $z$ such that integrating $P_{catchup}$ across the Beta probability density function yields an expected risk strictly bounded below $10^{-6}$.

## 8. Transaction Pool Dynamics

The unconfirmed transaction memory pool $\mathcal{M}$ is bifurcated into two strictly isolated sets. $\mathcal{M}_{commit}$ holds Phase 1 commitments and $\mathcal{M}_{reveal}$ holds Phase 2 reveal operations. The protocol enforces deterministic eviction and admission policies to prevent memory exhaustion and prioritize network throughput.

### 8.1. Dynamic Proof of Work Scaling

To prevent denial of service via commitment flooding, $\mathcal{M}_{commit}$ enforces a dynamic Proof of Work threshold. Let $|\mathcal{M}_{commit}|$ denote the scalar count of pending commitments. The required leading zero bits $T_{commit}$ is evaluated using a deterministic, step-wise integer function based on congestion:

$$T_{commit}(|\mathcal{M}_{commit}|) = 24 + \min\left(6, \left\lfloor \frac{\max(0, |\mathcal{M}_{commit}| - 500)}{50} \right\rfloor \right)$$

This function yields a base threshold of 24 bits under normal conditions, scaling smoothly up to 30 bits as the pool approaches its strict upper bound of 1000 elements. Incoming commitments failing to meet $T_{commit}$ are immediately rejected.

### 8.2. Admission and Replace-By-Fee (RBF)

The reveal pool $\mathcal{M}_{reveal}$ is bounded by a maximum capacity of 9000 transactions and a byte limit of 100,000,000 bytes (100 MB). Transactions are prioritized by their fee rate $\rho$.

To maintain determinism without floating-point arithmetic, the fee rate $\rho$ is scaled by a constant factor $S_{fee} = 1024$:
$$\rho = \left\lfloor \frac{fee \times 1024}{size\_bytes} \right\rfloor$$
The mempool enforces a strict floor admission threshold of $\rho_{min} = 10$.

Replace-By-Fee (RBF) evaluates competing transactions over identical inputs. If $tx_{new}$ conflicts with a subset of transactions $T_{old} \subset \mathcal{M}_{reveal}$, $tx_{new}$ is admitted if and only if both conditions are met:
1. $\rho(tx_{new}) > \max_{tx \in T_{old}} \rho(tx)$
2. $fee(tx_{new}) > \sum_{tx \in T_{old}} fee(tx)$

If $\mathcal{M}_{reveal}$ reaches maximum capacity or byte bounds, the transaction with the lowest global $\rho$ is evicted.

## 9. CoinJoin Protocol Specification

Midstate integrates a coordinator-free, denomination-uniform CoinJoin mechanism directly into the network layer. This breaks deterministic transaction linkage through subset-sum obfuscation.

### 9.1. Protocol Parameters

A mix session $S_{mix}$ is defined by a 32-byte identifier $ID_{mix}$ and a power-of-2 denomination $v_d$. The session requires a minimum participant count $P_{min} \ge 2$ and is strictly capped at 16 participants.

Let $I_{mix}$ be the set of participant inputs and $O_{mix}$ be the set of requested outputs. The protocol mandates strict value uniformity.
$$\forall i \in I_{mix}, i.v = v_d$$
$$\forall o \in O_{mix}, o.v = v_d$$

A single auxiliary input $I_{fee}$ is required to satisfy the network fee, where $I_{fee}.v = 1$. Total value is strictly conserved as $\sum I_{mix} + I_{fee} = \sum O_{mix} + 1$.

### 9.2. Anti-Sybil Identity Binding

To prevent zero-cost Sybil floods where an attacker fills mix sessions with invalid signatures to execute Denial of Service, `MixJoin` network messages require a localized Proof of Work bound to the sender's network identity.

The sender must mine a nonce $N_{join}$ satisfying:
$$\text{LeadingZeros}(\mathcal{H}(ID_{mix} \parallel CoinID \parallel PeerID \parallel N_{join})) \ge 20$$

This requirement bounds the PoW to approximately $10^6$ operations (compute-friendly for mobile and low-power devices, resolving in $<10$ms). Instead of relying on a draconian computational barrier, the protocol ensures robust Sybil resistance by combining this lightweight PoW with the Bayesian subnet limits and reputation tracking defined in Section 10. Binding the $PeerID$ ensures an attacker cannot precompute nonces across rotating network identities.

### 9.3. State Machine Transitions

The mix session advances through a strict state machine.
1. **Collecting:** The session accepts `MixJoin` and `MixFee` payloads.
2. **Signing:** Once $|I_{mix}| \ge P_{min}$ and $I_{fee}$ is present, the node builds a deterministic proposal. To guarantee all participants construct identical commitment hashes without trusted coordination, a strict canonical sort is applied:
   * The input array $I$ is formed by sorting $I_{mix}$ lexicographically by their respective 32-byte `CoinID`s, and strictly appending $I_{fee}$ as the final element.
   * The output array $O$ is formed by sorting $O_{mix}$ lexicographically by their `hash_for_commitment()`.
   The canonical $Commitment$ is computed over these deterministically ordered arrays, combined with a shared $salt$. The session transitions to `Signing` and awaits network broadcasts of individual WOTS/MSS signatures.
3. **CommitSubmitted:** Upon receiving all valid signatures, the node constructs the final transaction, mines the Phase 1 $Commit$ PoW, and broadcasts it.
4. **Complete:** The node detects the $Commit$ in the global state and subsequently broadcasts the assembled $Reveal$ payload.

## 10. Network Security and Eclipse Defense

To preserve consensus integrity under adversarial network conditions, nodes implement localized subnet heuristics, Bayesian reputation scoring, and strict relay throttling.

### 10.1. Subnet Connection Limits

The node maintains a hash map of active connections organized by network prefix. IP addresses are masked to extract the routing prefix (IPv4 is masked to `/24`, IPv6 is masked to `/32`). The node forcefully terminates any incoming TCP connection if the corresponding subnet prefix already has 4 active connections (expanded to 50 for ephemeral WebRTC direct connections). This bounds the impact of attackers controlling specific Autonomous System Numbers (ASNs).

### 10.2. Bayesian Peer Exchange (PEX) Scoring

Nodes execute the Peer Exchange protocol to discover network topology. A given PeerID is tracked utilizing a Beta distribution model defined by successful operations $\alpha$ and failed operations $\beta$.

The probability of a peer acting honestly is the expected value of the Beta distribution:
$$P(honest) = \frac{\alpha}{\alpha + \beta}$$

*   Successful connections and valid cryptographic proofs increment $\alpha$.
*   Timeouts, invalid PoW, and malformed headers increment $\beta$ (weighted at $10\times$ penalty).

When $P(honest) < 0.1$, the corresponding PeerID is permanently purged from the PEX routing table. If the routing table reaches maximum capacity (1000 entries), the peer with the lowest $P(honest)$ is evicted to accommodate new entries.

### 10.3. Relay and Orphan Throttling

To prevent asymmetric resource exhaustion, nodes constrain bandwidth allocated to libp2p Circuit Relays. A node permits a maximum of 16 concurrent relayed circuits, capped at 2 circuits per PeerID, a maximum duration of 120 seconds, and a strict data limit of 1 MB per circuit.

Orphan blocks (valid blocks whose parent `prev_midstate` is not currently in the local chain) are held in memory pending resolution. The orphan pool is restricted to 256 entries total, and a maximum of 4 competing forks per specific `prev_midstate`. 

**Critical Security Property:** A block is *only* admitted to the orphan pool if it possesses a mathematically valid `EXTENSION_ITERATIONS` sequential hash chain. Thus, an attacker attempting to exhaust the orphan pool must physically compute $1,000,000$ sequential hashes for *each* orphan submitted, neutralizing costless flooding attacks. If limits are exceeded, orphans are evicted via FIFO.

## 11. Wallet and Key Management Algorithms

The wallet software manages UTXOs and derives keys deterministically while enforcing the single-use constraints of the WOTS cryptography.

### 11.1. Hierarchical Deterministic (HD) Derivation

All private key material is derived from a 256-bit BIP39 root entropy. The 64-byte BIP39 seed is compressed via BLAKE3 to a 32-byte master seed $S_{master}$.
Child seeds are generated using strict domain separation strings to prevent index collisions between different key types.

For standard receiving and change keys (WOTS) at index $k$:
$$S_{wots, k} = \mathcal{H}(\text{"midstate/wots/v1"} \parallel S_{master} \parallel k_{u64})$$

For reusable Merkle Signature Scheme (MSS) trees at index $k$:
$$S_{mss, k} = \mathcal{H}(\text{"midstate/mss/v1"} \parallel S_{master} \parallel k_{u64})$$

### 11.2. The Greedy Snowball Coin Selection

When constructing a transaction, the wallet must select a subset of live UTXOs to meet a target value $V_{target}$. To prevent UTXO fragmentation over time, the wallet implements an aggressive defragmentation heuristic.

1.  Live coins are sorted by value in descending order.
2.  Coins are selected iteratively until the sum $V_{sum} \ge V_{target}$.
3.  The optimal change value $C = V_{sum} - V_{target}$ is decomposed into its binary power-of-2 denominations.
4.  For each denomination $d$ in the binary decomposition of $C$, the wallet scans its unselected UTXOs. If an unselected coin matching $d$ exists, it is added to the input set. The target sum is re-evaluated, and the loop repeats until no further matching denominations exist.

This algorithm actively consumes fragmented dust coins by merging identical denominations into the next sequential power of 2.

### 11.3. Co-Spend Enforcement

While the consensus layer only enforces that an address binds to a singular Commitment Hash, the wallet layer must proactively protect the user from stranding their own funds. If a single WOTS address receives multiple distinct UTXOs, spending them independently would require signing different transaction commitments, violating the one-time-use constraint and leaking private key data.

The wallet resolves this via mandatory co-spending. Let $A$ be a target WOTS address. If any UTXO residing at $A$ is selected for a transaction, the wallet software automatically locates all other UTXOs residing at $A$ and appends them to the input array. Because all inputs in a transaction share the exact same global transaction commitment, the resulting WOTS signatures are mathematically identical. This maintains cryptographic security while allowing concurrent spends.

## 12. Storage and Database Architecture

The persistence layer guarantees ACID (Atomicity, Consistency, Isolation, Durability) properties without relying on complex file system manipulations.

### 12.1. Copy-on-Write (CoW) B-Trees

All blocks, headers, compact filters, and the global state are stored within a single embedded Database utilizing Copy-on-Write B-Trees. Data pages are never modified in place. When a new block is appended or a state transition occurs, new pages are written, and the root pointer is updated atomically. 

This mechanism provides immunity to partial writes during unexpected power loss. If the host hardware fails mid-transaction, the root pointer simply remains at the previous valid state.

### 12.2. Deterministic State Rebuilding and Checkpointing

To bound memory usage during deep synchronizations, the node flushes accumulated state modifications to disk in 500-block intervals. 
If a deep reorganization occurs, the node locates the most recent state snapshot preceding the fork point. The node then replays the historical block data from the snapshot height up to the fork point, evaluating $\Upsilon(\sigma_{prev}, B)$ sequentially. This allows the node to establish the precise state required to validate the new alternative chain without retaining the entire state history in active memory.

### 12.3. Autonomous State Corruption Recovery (Self-Healing)

The protocol implements an autonomous self-healing routine to recover from localized database corruption, bit rot, or incomplete disk I/O. This mechanism is triggered if the node encounters a missing or cryptographically invalid block payload during a background state rebuild operation.

Let $h_{corrupt}$ denote the block height where local data retrieval or state transition evaluation $\Upsilon(\sigma_{h-1}, B_h)$ deterministically fails. Upon detecting this local fault, the node halts the synchronization session and initiates a localized chain rollback.

The algorithm identifies the optimal safe recovery height $h_{snap}$ strictly below the point of corruption. Snapshots are generated at intervals of 100 blocks. The initial candidate height is calculated as:

$$h_{snap} = \begin{cases} \lfloor \frac{h_{corrupt}}{100} \rfloor \times 100 - 100 & \text{if } \lfloor \frac{h_{corrupt}}{100} \rfloor \times 100 = h_{corrupt} \\ \lfloor \frac{h_{corrupt}}{100} \rfloor \times 100 & \text{otherwise} \end{cases}$$

The node attempts to deserialize the state snapshot at $h_{snap}$. If the snapshot file itself is missing or corrupted, the node iteratively decrements $h_{snap}$ by 100 until a valid state representation $\sigma_{snap}$ is successfully loaded into memory. If $h_{snap}$ reaches 0 without locating a valid snapshot, the node defaults to the hardcoded genesis state.

Once $\sigma_{snap}$ is verified and loaded into the active state vector, the node executes a destructive truncation function $\Phi(h_{snap})$. This function iterates through the Copy-on-Write B-Trees and deterministically purges all stored blocks, block headers, compact filters, and subsequent state snapshots for all heights $h \ge h_{snap}$.

To ensure continuity of the Median-Time-Past (MTP) timestamp validation sequence, the node reconstructs its internal sliding window of recent timestamps. It performs linear disk reads for blocks in the range $[h_{snap} - 60, h_{snap} - 1]$, appending the timestamp of each block to the MTP validation array. 

Following the completion of the truncation and MTP reconstruction, the node resets its synchronization cursor to $h_{snap}$. It then resumes standard peer-to-peer network operations, effectively re-downloading the purged segment of the blockchain from honest peers to seamlessly overwrite the corrupted history.

## 13. SIMD Variable-Length Hash Chain Execution

To optimize the generation and verification of WOTS signatures, the protocol evaluates up to 8 independent hash chains simultaneously within a single SIMD register. Because WOTS digits dictate variable chain lengths, the protocol utilizes an execution masking and pre-sorting algorithm.

### 13.1. Batch Pre-Sorting
Let $K$ be a set of $N$ hash chains to be processed, where each chain $k_i$ requires $L_i$ iterations of the BLAKE3 compression function. To minimize execution divergence, the set $K$ is sorted in descending order based on the required length $L$:
$$K_{sorted} = \{k_x, k_y, ..., k_z\} \text{ where } L_x \ge L_y \ge ... \ge L_z$$

The sorted chains are then partitioned into batches of size $S$, where $S$ corresponds to the hardware register lane width (8 for AVX2, 4 for NEON and WASM). This grouping ensures that chains with similar iteration counts execute in the same hardware batch, minimizing wasted operations.

### 13.2. SIMD Masking Extraction (Ghost Hashing)
Let a batch contain $S$ chains with required iterations $L = \{L_0, L_1, ..., L_{S-1}\}$. The execution loop runs for exactly $\max(L)$ steps. 

Let $hw_t$ represent the $S$-wide SIMD register holding the intermediate hash states at step $t$. At each step $t \in [0, \max(L))$, the node applies the BLAKE3 compression function simultaneously across all $S$ lanes:
$$hw_{t+1} = \text{Compress}(hw_t)$$

To prevent chains with shorter target lengths from being over-hashed (which would invalidate the cryptographic signature), the algorithm performs conditional extraction. For each lane $j \in [0, S)$:
$$\text{If } t = L_j - 1, \text{ then } \text{Result}_j = \text{ExtractLane}(hw_{t+1}, j)$$

Lanes that have already reached their target length continue to undergo hashing (ghost hashing) until $t = \max(L)$ is reached, but their completed values have already been safely extracted and stored in the result vector.

## 14. Address Checksum Algorithm

To mitigate manual transcription errors and typo-squatting, the protocol enforces a 4-byte checksum on serialized addresses.

Let $A \in \mathbb{B}_{32}$ be the raw 32-byte address derived from a script predicate. The checksum $C \in \mathbb{B}_4$ is computed as the first 4 bytes of the hash of the address:
$$C = \mathcal{H}(A)[0..4]$$

The payload to be encoded is the 36-byte concatenation:
$$P_{payload} = A \parallel C$$

The final serialized address is represented as a 72-character hexadecimal string of $P_{payload}$. The parser function $\Psi_{parse}$ evaluates any 72-character input by extracting $A'$ from the first 64 characters and $C'$ from the final 8 characters. It yields success if and only if $\mathcal{H}(A')[0..4] \equiv C'$.

## 15. Hard Fork Activation Schedule

Consensus rules are bound to strict block height activation gates. Blocks mined prior to these heights are grandfathered to maintain historical continuity of the active ledger.

1.  **WOTS Address Reuse Ban ($h = 18,000$):** Enforces that a standard WOTS address cannot be spent from more than once with differing transaction commitments.
2.  **MSS Leaf Reuse Ban ($h = 25,000$):** Extends the single-use enforcement to the individual WOTS keys at the leaves of an MSS tree.
3.  **Mandatory State Root ($h = 30,000$):** The `state_root` field in block batches transitions from optional to mandatory. Blocks failing to carry the state root or carrying an incorrect state root are rejected.
4.  **Virtual Machine V3 Upgrade ($h = 60,000$):** Activates logic and arithmetic opcodes `OP_OVER`, `OP_ROT`, `OP_SLICE`, `OP_CONCAT`, `OP_SUB`, `OP_MUL`, and `OP_DIV`.
5.  **State Thread Activation ($h = 65,000$):** Repurposes zero-value outputs into Confidential State Threads and activates the `OP_READ_INPUT_STATE` and `OP_READ_OUTPUT_STATE` opcodes in the virtual machine.
6.  **Strict Median-Time-Past ($h = 70,000$):** Enforces strict block timestamp monotonicity against the 11-block median.
7.  **Recent PoW Wiggle Room ($h = 80,000$):** Modifies the Commit PoW check. Instead of validating against the exact current block height, the PoW is considered valid if it meets the difficulty threshold when evaluated against any valid block height in the range $[h - 1000, h]$.
8.  **Strict Intra-block Reuse Ban ($h = 85,000$):** Enforces that no WOTS key may be reused even within the same block or sync chunk.
