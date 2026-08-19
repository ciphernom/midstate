# Midstate SDK — update to match node `midstate-20260816`

Version 1.0.8 → 2.3.0.

The SDK was built against an older web wallet. This pass realigns it with the
current node and WASM API. Every claim below was checked against the node source
or the `pkg/` type definitions, and the behavioural fixes have regression tests
that were confirmed to fail before the fix and pass after.

Test suites: `npm test` — 22 codec + 19 wallet + 89 reorg/recovery/contract + 44 miner/pool + 30 DEX + 26 launcher + 28 compiler/VM + 27 pump launcher = **285 tests**, all offline.
Plus `npm run smoke` — a live-network script (see below) that only a real node can answer.

---

## Correctness fixes

### 1. Blocks with unreadable filters were silently skipped — missed funds

`sync()` treated `element_count === 0` as "empty block, nothing to match".

The node's `GetFilters` handler (`src/node.rs`) emits a zero count in exactly two
situations, both meaning *undecidable*:

- the block loaded but its filter did not — block hash present, filter string empty;
- the height was unreadable — hash and filter both empty.

Its own comment is explicit: *"Never treat a zero count as 'nothing of interest
here': every block has coinbase outputs, so a genuine filter is never empty."*
Every filter is built from `CompactFilter::items_in`, which always inserts each
coinbase output, so a real filter cannot be empty.

Because the zero-count test ran *before* the missing-hash fallback, that fallback
was dead code. Any coin paid to an address of ours in a block whose filter failed
to load was never seen. The empty-filter-string case was not checked at all.

Now an absent hash, an empty filter, or a zero count all force a fetch. Only a
genuine non-empty filter may rule a block out.

### 2. Short-batch handling could earn a node ban

After the filter loop, the SDK ran `while (current < end) getBlock(current)` to
walk the remainder of a short batch. That is up to 1000 `get_block` calls in
place of one `get_filters`.

The node budgets a scan at "120 requests × 1000 blocks = 120,000 blocks synced
per minute" — the filter path *is* the sync path — and grants an unknown peer
~275 general requests per minute (`current_rate_limit()` = `50 + 450 × P(honest)`,
`P = 0.5` under the Beta(1,1) prior). The node's comment calls the pattern
"indistinguishable from deliberate rate-limit evasion".

The current node no longer truncates on an unreadable height; it emits an
undecidable placeholder and continues, so all four arrays span the full request.
A short batch now means the peer cannot serve the range at all. Replaced with
bounded stall handling (`maxStalledBatches`, default 20) that steps over one
height and re-requests, then surfaces a clear error so the caller can pick
another node.

Measured: a stalled peer now costs ≤4 block fetches where the old loop issued
~990 for the first batch alone.

### 3. `Consolidate` transactions were invisible

`_processBlock` only read `tx.Reveal`. `Transaction` (`core/types.rs`) also has a
`Consolidate` variant with its own `inputs` and `outputs`, used for dust sweeps.
Both spends and receipts through that path were missed. Both variants are now
scanned.

### 4. Spends were matched on salt alone

`InputReveal` has no `address` field. It carries `predicate: Predicate`, and
`Predicate::address()` is `BLAKE3(bytecode)` — every address is
pay-to-script-hash. A coin's identity is `(address, value, salt)`, or the
domain-separated `"CONFIDENTIAL" ⌢ address ⌢ commitment ⌢ salt` form when the
input carries a state commitment.

Matching on salt alone worked by accident (salts are random) but could not
distinguish a confidential coin from a standard one. Coin IDs are now recomputed
from the predicate bytecode, with salt lookup demoted to a fallback for
malformed block data — mirroring the browser wallet.

### 5. MSS leaf reuse — stale leaf index at signing time

`build_reveal` signs with `kp.next_leaf = inp.mss_leaf`: the leaf comes from the
UTXO record handed to it, **not** from a counter inside WASM.

The SDK passed `this.utxos` through untouched, so every coin carried the
`mss_leaf` stamped onto it by `addUtxo` at discovery. Two coins received at the
same MSS address before either was spent both carry leaf 0 — spending them in
separate transactions signs twice with leaf 0.

An MSS leaf is a one-time WOTS key inside the Merkle tree. "MSS tolerates reuse"
refers to the *address* (you may receive at it repeatedly, since each spend
consumes a different leaf); reusing a *leaf* is exactly as fatal as reusing a
bare WOTS key.

New `_utxosForWasm()` re-reads `next_leaf` (and `mss_height`) from live wallet
state at every spend, which is what the browser wallet does at all ~10 of its
call sites.

### 6. MSS leaves were over-consumed

`_broadcastTwoPhaseTx` incremented `next_leaf` once per *input*. `build_reveal`
caches the signature per address and reuses it across every input at that address
— they all sign the same commitment — so exactly one leaf is consumed per address
per transaction.

A three-input spend from one address burned three leaves instead of one. A
height-10 address has 1024 leaves, so this cut usable capacity threefold.
Now deduped by address, and `set_mss_leaf_index` keeps the WASM cache in step.

### 7. WOTS key-reuse exposure

Spending any coin at a WOTS address publishes that one-time key. The node
enforces co-spend, so in a well-formed block the siblings appear as inputs and
were removed individually — but two gaps remained:

- a coin arriving at the address *after* the spend was admitted to the UTXO set
  as if spendable;
- a reveal that was built and transmitted but rejected left the coins in place,
  so a retry would re-select them and sign a *second* time over a different
  commitment.

Either is what the node's Key Reuse Punishment Burn Protocol (`core/wots.rs`)
exists to exploit. Now:

- spending purges all siblings at the address and records it in a persisted
  `spentWots` map;
- `addUtxo` refuses coins at a retired address;
- `_broadcastTwoPhaseTx` retires the keys and drops the coins **before**
  transmitting, and regardless of whether the node accepts the reveal.

MSS addresses are deliberately exempt from the purge.

Ordering note: key material is reserved *before* the commit and retired
*immediately after the reveal is built*. Skipping a key is free; reusing one is
not. A commit timeout touches no key material at all, and per `core/state.rs`
commits neither lock nor modify UTXOs — an expired commitment costs the PoW
effort and nothing else.

### 8. Chat attachment decoder — three bugs, each corrupting the rest of the message

Read against the hand-written `impl Serialize for ChatAttachment` in
`src/chat.rs` (the `BincodeHelper` branch):

| tag | variant | payload |
|-----|---------|---------|
| 0–6 | address, coin_id, mix_id, commitment, block_hash, midstate, data_hash | `[u8; 32]` |
| 7 | license_challenge | `[u8;32] ⌢ varint(u64) ⌢ [u8;32]` |
| 8 | signature | `varint(len) ⌢ len bytes` |

- `Signature` was read at **tag 9**. It is the ninth variant but the eighth
  index, so real signatures fell through the catch-all and were dropped.
- Its length was read as a fixed **4-byte LE u32**. bincode's `DefaultOptions`
  varint-encodes every integer including collection lengths, so a 576-byte WOTS
  signature is `251 40 02`.
- The catch-all advanced a flat **32 bytes**, but `license_challenge` is
  32 + varint + 32.

Attachments decode in sequence, so any one of these corrupted every attachment
after it. A ground-truth vector (address + 576-byte signature) desynced at byte
132 of 679 under the old decoder; it now round-trips exactly.

Unknown future tags are genuinely unskippable — their width is unknowable — so
decoding stops cleanly and reports `truncated` rather than inventing a width.

### 9. `signChannelState` threw on its first line

Opened with `if (!mssCachesReady) await loadMssCaches();`. Neither identifier
exists in the module, so the function raised a `ReferenceError` before reaching
any logic. Replaced with a real check that loads the MSS tree from storage on
demand and syncs the leaf index.

### 10. `decodeAddr` accepted truncated payloads

`subarray()` clamps rather than throwing, so a peer could send a length
overrunning the buffer and get a quietly-wrong multiaddr back — which then gets
dialled. Now bounds-checked, with the node's 1000-entry cap enforced.

---

## Reorg handling (new — `src/reorg.js`)

The largest gap from the previous pass. The wallet applied blocks forward-only
and irreversibly, so an orphaned block left phantom credits and — far worse —
**stranded real coins**: a spend that got reorged out had already had its inputs
deleted and its address poisoned, and `addUtxo` refuses a poisoned address
forever, so not even a full rescan could re-credit the coin.

### Why the obvious fix is unsafe

Clearing `spentWots` on rollback and restoring the coin **leaks keys**. A WOTS
key signs exactly one message. When a spend is orphaned its signature — over
that transaction's commitment — is already public and may still confirm on the
winning branch or be replayed by anyone. Un-poisoning the address and spending
the restored coin into any *different* transaction produces a second signature
over a *different* commitment from the same key. That is the exact condition the
node's Key Reuse Punishment Burn Protocol exists to exploit.

So a restored coin must be spendable **only** by re-sending the identical
original signature, never by re-signing. Every mutation is height-tagged, and
each restored spend resolves three ways:

| | Condition | Outcome |
|---|---|---|
| **CASE 1** | Verbatim reveal still held, commitment matches, within `COMMITMENT_TTL` | Poison cleared, original signature re-armed for verbatim re-send |
| **CASE 2** | Signature lost or expired | Coin restored **reuse-locked** — excluded from spendable balance and from the signer, surfaced via alert |
| **CASE 3** | MSS input | Restored normally; each leaf is distinct, so no per-key cliff |

All three gates on CASE 1 are load-bearing and independently tested: an expired
reveal and a mismatched commitment both correctly fall through to CASE 2.

### Detection

`LightNotification::NewBlockTip` carries height, target, filter_hex, block_hash
and element_count — but **no parent hash**. Parent-linkage detection is therefore
impossible on `/midstate/light/2.0.0`, and any check written against `prev_hash`
is dead code against this node. (The browser wallet has such a branch; it never
fires.)

Detection instead keys on hash *conflict* at a recorded height. Per-height
identity is `batch.extension.final_hash`, available identically from
`get_filters` (`block_hashes[i]`) and `get_block`. `get_filters` is used for the
descent since it is a far smaller response for the same request cost.

The descent is bounded by `REORG_DEPTH` (200). Unbounded, a peer answering every
probe with a mismatch would walk the client to genesis one request per height —
a self-inflicted rate-limit violation driven entirely by a remote party, the same
amplification shape fixed in item 2 above.

### Hook placement

The check runs **before** `sync()`'s "nothing to do" early return, not after. A
reorg that replaces blocks without advancing the tip leaves `chainHeight`
unchanged, so checking after the early return misses precisely the case where
our view is wrong but our height is right. Verified: gating the check on tip
advancement leaves a phantom credit standing in the integration test.

### Pruning cutoffs

Three structures are bounded on two different clocks, deliberately:

- `blockHashes` and `spentLog` → `REORG_DEPTH`. Below that depth a reorg is not
  precisely healable anyway; the spend is final.
- `sentReveals` → `COMMITMENT_TTL`. Its purpose is CASE 1 re-broadcast, and once
  the commitment has expired from chain state the reveal is dead. Pruning these
  on the `REORG_DEPTH` clock would be a bug in the *unsafe* direction: it would
  discard still-re-sendable signatures and needlessly demote recoverable coins
  to reuse-locked.
- `spentWots` is **never** pruned. A published one-time key stays published.

### Supporting changes

- `getBalance()` now excludes reuse-locked coins; `getLockedBalance()` reports
  them separately so the money stays auditable without being offered.
- `_utxosForWasm()` filters locked coins — the single choke point where UTXOs
  reach the signer, so no spend path can reach them by accident.
- `_broadcastTwoPhaseTx` retains the verbatim signed reveal keyed by input
  address, and records the commitment alongside the poison height. Without this
  retention every orphaned WOTS spend would fall to CASE 2.
- `addUtxo` height-stamps every credit (`createdAtHeight`).
- New `sync()` options: `onReorg`, `skipReorgCheck`. Return value gains `reorg`
  and `lockedBalance`.
- All reorg state persists across save/restore, with `spentLog` BigInt values
  given the same decimal-string treatment as `utxos`.

### Documentation

`reorg.js` and the Tier 1 `_broadcastTwoPhaseTx` follow the Midstate Coding &
Documentation Standard: Reasoning → Formal Specification (textual pre/post plus
Z schema) → Safety/Invariants, with a module-level state space and four numbered
invariants. `RollbackTo`'s stated obligation — re-establishing INV-1, that no
free WOTS coin remains at a published key — is asserted directly as a test over
both CASE 1 and CASE 2 outcomes.


---

## State threads / confidential coins (new)

State threads were invisible to the SDK end to end. `_processBlock` read only
`out.Standard`, so `OutputData::Confidential` — a zero-value output carrying the
32-byte commitment scripts read via `OP_READ_INPUT_STATE` — was never credited,
never retired when advanced, and therefore could not participate in a rollback.

This is not cosmetic. `prepare_script_spend` validates the caller's `coin_id`
against the canonical derivation from `(address, value, salt, state)` and rejects
any mismatch outright. A wallet holding a stale state builds executions the node
refuses, and after a reorg it holds a state the winning branch never had. The
symptom is a contract that silently stops working.

### Model

State threads live at **contract** addresses, not wallet addresses, and are
consumed through `prepare_script_spend`'s `contract_inputs` argument rather than
the wallet UTXO array. They are therefore held in a separate `contractCoins` map,
with two new invariants:

- **INV-5** `dom contract_coins ∩ ids(utxos) = ∅` — a script-locked coin never
  enters the wallet-key spend set. Putting one in `utxos` would offer it to
  `prepare_spend`, which would try to sign a script-locked coin with a wallet
  key.
- **INV-6** `state ≠ null ⇒ value = 0` — consensus requires it
  (`apply_transaction` bails on `commitment.is_some() && value != 0`), so it is
  asserted at admission rather than discovered inside WASM.

### Changes

- `watchContract()` / `unwatchContract()` — follow a contract. Its address joins
  the compact-filter watchlist, which matters because `CompactFilter::items_in`
  indexes **addresses as well as coin ids**: without it, a block advancing a
  followed thread produces no filter hit and is never fetched.
- `Confidential` outputs credited via the domain-separated
  `"CONFIDENTIAL" ⌢ address ⌢ commitment ⌢ salt` coin id; `Standard` outputs at
  a watched contract credited as the contract's funds coin.
- Contract inputs detected and retired on spend, so an advanced thread no longer
  leaves the superseded state tracked.
- Contract coins skip the WOTS reuse classification entirely — they are consumed
  by satisfying a script, not by a one-time signature, so no key is published and
  there is nothing to lock or re-send.
- `getStateThread()`, `getContractCoins()`, and `buildContractInputs()` — the
  last builds the `ScriptInputArg` array from observed state with the thread
  ordered first, so a caller cannot accidentally supply a stale state.
- `rollbackTo` routes restored coins by `isContract`, and discards dead-branch
  contract coins alongside wallet ones.

### Chained advances converge without special handling

For `S₀ → S₁` at h₁ and `S₁ → S₂` at h₂, both above the fork: step 1 discards S₁
and S₂ as dead-branch, and step 2 restores only S₀, because S₁'s own
`created_at` is above the fork. The wallet lands on exactly the state the winning
branch has. Verified directly, and verified non-vacuous — removing the
dead-branch discard leaves the wallet on `a2` when the fork requires `a1`.


---

## Recovery, live reconciliation, and wallet operations (new)

### `recoverPendingTx()` — driving the verbatim re-send

`rollbackTo` CASE 1 clears an address's poison and stages the original signed
reveal, but nothing sent it, so a CASE 1 recovery left coins spendable in
principle and stuck in practice. Clearing the poison is only defensible because
the coin can leave exclusively via that identical signature — so something has to
actually send it.

**This deliberately diverges from the browser wallet on the dead-commitment
branch.** There, a dead commitment means a transaction that never confirmed, and
the correct advice is "the coins are still yours, re-send". Here the transaction
*did* confirm — that is why the key was poisoned — and was then orphaned, so its
signature is public. Re-sending would mean re-signing over a fresh commitment
with a key that has already signed. The coins are demoted to CASE 2
(reuse-locked) instead. Telling the caller to "just re-send" would be an
instruction to leak the key.

The operation produces **no signature at all** — it transmits a payload built
earlier. It is idempotent, persists across restart, and treats node errors as
retryable without touching key state.

### `autoReconcile()` — live push loop

The pieces for live reconciliation existed but nothing joined them:
`onPushEvent` delivered `NewBlockTip`, `handleBlockTip` could act on it, and
`recoverPendingTx` could finish an armed re-send — but a caller had to wire all
three by hand, and one who didn't sat on a stale or forked view indefinitely.

`autoReconcile(client, handlers)` subscribes once and joins them: a conflicting
tip triggers rollback, and because a rollback is exactly when a re-send gets
armed, recovery runs in the same pass. Errors are reported through `onError`
rather than thrown, since throwing inside the subscription would tear down the
stream and leave the wallet worse off than the fork it was reacting to.

`handleBlockTip()` is also exposed directly for callers driving their own loop.

### Wallet operations

Four WASM entry points that had no wrapper:

- **`consolidate(client, sourceAddr, destAddr)`** — `Transaction::Consolidate`
  requires all inputs to share one predicate address (the WOTS co-spend rule at
  the transaction level), so this takes a source *address* rather than an amount.
  Still two-phase: the node checks the commitment in chain state exactly as for a
  Reveal; only the reveal encoding differs, carrying one signature for the shared
  address. `_broadcastTwoPhaseTx` now takes a `revealKind` rather than an
  `isScript` boolean to accommodate it.
- **`defrag(client, destAddr, maxInputs)`** — sweeps fragmented WOTS coins into a
  fresh MSS address. Unlike consolidate this spans addresses, so sibling bundling
  is left to WASM.
- **`fundMany(client, fundings)`** — funds N addresses in one ~2-block round trip
  instead of N.
- **`advanceStateThread(client, bytecode, newState, extraOutputs)`** — reads the
  current state, coin id and salt from tracked state rather than caller-supplied
  values. This is the point: after a reorg, a caller working from its own notes
  would submit the orphaned state and be rejected. Tested directly — following a
  rollback, the wrapper submits the rolled-back state.
- **`getContractBalance()`** — closes the reported gap in contract accounting.

Reuse-locked coins are excluded from all of these, since they all draw through
`_utxosForWasm`.


---

## Solo mining (new — `src/miner.js`)

The web wallet's mining path has three template-lifecycle problems. All of them
fail the same way — a block is found and then rejected — which is
indistinguishable from "mining doesn't work".

Worth stating up front: this chain's PoW is a **VDF, not a throughput race**.
`create_extension` iterates BLAKE3 `EXTENSION_ITERATIONS` (1,000,000) times
sequentially per nonce. Measured here: ~130 ms per nonce, ~230 ms per 4-lane SIMD
batch. A worker does 15–30 nonces/second and no batching changes that. Everything
below follows from the fact that grinding is slow relative to block time.

### 1. Grind on `mining_midstate`, never `post_tx_midstate`

`finish_template` returns `mining_midstate = compute_header_hash(header)`, and
the node's own comment calls grinding `post_tx_midstate` instead *"the bug that
silently rejected every block the web wallet ever mined over WebRTC"*.
`verify_extension` recomputes from the header hash on receipt. The SDK miner only
ever passes `template.mining_midstate` to the search, asserted by test.

### 2. The template expires, and the timestamp cannot be refreshed

`finish_template` locks `actual_timestamp` into the header **before** computing
`mining_midstate`, specifically so a miner "cannot bump it post-grind without
invalidating the header hash they searched on".

Meanwhile `validate_timestamp` requires a block to beat the median of the last
`MEDIAN_TIME_PAST_WINDOW` (11) blocks. At `TARGET_BLOCK_TIME` = 60s that median
trails the tip by roughly five blocks — so a template older than about five
minutes yields a block the node rejects, and the miner **cannot fix it** by
touching the timestamp, because that invalidates the PoW it just found.

The only correct response is to abandon the template. The SDK miner refreshes on
every tip change and on a hard age cap (`TEMPLATE_MAX_AGE_MS`, 120s), and copies
the template's timestamp verbatim into the submitted batch. Verified
non-vacuously: rewriting the timestamp on submission fails the test with the
exact symptom.

### 3. Coinbase total starts at `block_reward`, ignoring fees

The node requires `coinbase_total == block_reward + total_fees` and returns
`expected_total` on a mismatch. The web wallet guesses `block_reward` alone every
time, so with any fee-paying mempool **every** template costs a wasted round trip,
and with `MAX_RETRIES = 3` a busy mempool can exhaust the retries outright.

Fees are only knowable from a template and a template needs a coinbase, so the
first request of a session is necessarily a guess — but only the first. The SDK
miner carries `expected_total` forward, so steady-state mining is one request per
template.

### Also fixed: `resp.text()` on the error path

`buildMiningTemplate`'s unknown-error branch calls `await resp.text()`. The P2P
client returns `{ ok, status, json }` with **no `text` method**, so a genuine
template error throws a `TypeError` that masks the node's actual message. The SDK
reads the body defensively and surfaces the real error.

### What the miner does

`new Miner(wallet, client, opts)` runs the full lifecycle: build coinbase →
template → grind → recompute extension → submit. Events: `template`, `progress`,
`found`, `accepted`, `rejected`, `error`.

- Key material is **reserved, not consumed**, at template time: a rejected block
  advances no index and registers no address, since coinbase addresses are only
  paid to by a block that lands.
- The extension is **recomputed** via `build_solo_extension` before submission
  rather than trusted from the search, so a diverging GPU or remote searcher
  costs a local no-op instead of an invalid submission.
- `searchFn` is injectable, so a GPU backend can replace the SIMD search without
  touching the loop.
- `payoutMssAddress` pays coinbase to a reusable address instead of burning a
  fresh WOTS index per block.

Tests include a **real PoW round trip** through the WASM VDF: a nonce found by
`search_nonces` reproduces through `build_solo_extension`, lands below target,
and matches what the submission path sends — the exact conjunction
`verify_extension` checks. Plus a check that the target comparison is not
inverted.

---

## MSS leaf reconciliation (new)

`next_leaf` is a local guess and can fall *behind* reality — a spend confirmed on
another device with the same seed, or a state file restored from an older backup.
When it does, the wallet signs with a leaf the chain has already burned.

That failure is invisible then permanent: the mempool only compares against the
live mempool so admission succeeds, and `prune_on_new_block` then evicts the
reveal against the chain's burned-leaf accumulator every block, forever. The
transaction simply never confirms and re-sending never helps, because each
attempt signs the same exhausted leaf.

`reconcileMssLeaves(client)` asks `mss_state` (one O(1) lookup, returning
`max(chain, mempool)` so in-flight spends count) and advances counters to match.

**Counters only ever move forward.** A node reporting a *lower* index is ignored —
adopting it would rewind onto leaves this wallet has already used, which is the
exact reuse this exists to prevent. Verified non-vacuously: relaxing `>` to `!==`
fails the monotonicity test.


---

## Pool mining, multi-worker search, deep reorgs (new)

### Pool mining (`src/pool.js`, `Miner` pool mode)

Solo mining on this chain is a lottery a light client cannot win: 15–30
nonces/second against a network target set for the whole hashrate. Pool mining is
how such a client actually earns.

Three semantics matter, and the middle one is the reason the browser miner needed
its `continue_on_found` flag:

- **A share is not a block, and the job survives it.** Finding a share leaves
  `mining_midstate` unchanged, so a miner that halts on a hit — correct *solo*
  behaviour, where a hit means the tip moved — sits idle until the next network
  block, because the restart-on-new-midstate path never fires. `Miner` sets
  `continueOnFound` automatically in pool mode. Both directions are tested.
- **The pool rebuilds the block.** A share is just
  `(address, job_id, nonce, worker)`; `job_id` is what lets the pool reject work
  mined against a superseded template. The nonce goes as a decimal **string** —
  it is a u64 and a JSON number silently rounds above 2^53.
- **"The pool said no" ≠ "the request failed."** `requestRaw` returns the
  envelope; a miner absent from the precommitment tree is an *answer* to
  `get_proof`, not an exception. Collapsing both is what lets a dropped stream
  masquerade as pool omission and halt mining for the wrong reason.

Pool mode grinds `share_target`, not `network_target`, and skips coinbase
negotiation entirely since the pool owns the block. The transport is injected, so
the protocol is testable without libp2p and works over both TCP (Node) and
WebRTC-direct (browser).

### Multi-worker search (`src/miner-pool.js`, `src/miner-worker.js`)

The VDF is irreducible — ~130 ms per nonce, sequential — so cores are the only
lever, and the work is embarrassingly parallel. The partition has to satisfy two
constraints at once: `search_nonces` needs each worker's four SIMD lanes
contiguous, and no two workers may test the same nonce or the extra cores buy
nothing. Striding by `iterations × 4 × workerCount` satisfies both. Verified
disjoint and gap-free at 1, 2, 4 and 8 workers.

`MinerPool` exposes a `searchFn` with the same shape `Miner` already consumes, so
turning it on changes nothing about template lifecycle, submission, or pool/solo
logic. Runs on `node:worker_threads`; the worker file is written to work
unchanged as a browser `Worker`. WASM bytes are passed in rather than fetched,
because a Node worker has no `fetch` relative to a module URL — the failure that
killed the browser miners at load with an opaque `[object Event]`.

### Pool precommitment auditing

`get_proof` is only useful if something checks it. Without verification a pool can
hand back a plausible proof, or omit a miner from the payout tree entirely, and
the miner keeps grinding shares it will never be paid for.

`verifyPoolProof()` reconstructs the root exactly as `ShareMerkleTree::build`
does: `leaf = BLAKE3(address ⌢ le64(score))`, `parent = BLAKE3(left ⌢ right)`,
and — the subtle part — **a lone node is paired with itself**, not promoted
unchanged. `generate_proof` clamps the sibling index to the last element, so the
right-most node of an odd layer lists *itself* as its sibling. A conventional
"promote the odd node" rule fails exactly the miners at the end of the tree;
verified non-vacuously, that variant rejects valid proofs at n=3 and n=5.

`PoolClient.auditInclusion()` returns a structured verdict rather than throwing,
because "the pool has not recorded you yet" is normal for a freshly connected
miner — distinguishing that from "the pool is lying" is the entire point. The
smoke test audits inclusion before grinding.

### Deep reorgs

Beyond `REORG_DEPTH` the `spentLog` no longer holds what is needed to restore
deep spends, so the UTXO set cannot be repaired incrementally. An exhausted
descent now escalates to a full rebuild from chain data
(`reorg.escalatedToRescan`).

Critically, the rebuild **preserves published-key state**. `spentWots`,
`reuseLocked` and `sentReveals` record which one-time keys *this wallet* has
published — a fact about our own signing history, not about any chain. A rescan
re-derives the published-key set only for spends visible on the branch being
scanned, so a key burned by a later-orphaned spend appears nowhere; clearing
these would silently re-admit its coins as spendable.

**Bug found while doing this:** an earlier patch meant to clear `blockHashes` and
`spentLog` on `rescan` had silently failed to apply — the anchor never matched.
Fixed, with the assertion strengthened to actually check.

### Shorter-chain detection

Also found here: detection probed only the single highest recorded height. The
node caps `end_height` at its own height, so if it has reorged to a **shorter**
chain our last recorded height is unservable and comes back as an undecidable
placeholder — indistinguishable, to a single probe, from a chain that agrees.
That is exactly when a rollback is most needed. Detection now walks down to 8
recorded heights until one is decidable. Verified non-vacuously: restricting it
back to one probe fails with "the wallet stays permanently ahead".

---

## Live smoke test (`npm run smoke`)

Everything in `npm test` is offline — vectors and mocks derived from the Rust
source. That catches drift in what the code believes, but not a belief that was
wrong in the same way in both places. `smoke-live.mjs` is the part only a real
node can answer:

```
node smoke-live.mjs --peer /ip4/…/p2p/12D3Koo…
node smoke-live.mjs --rpc http://127.0.0.1:8080 --scan 5000
node smoke-live.mjs --peer … --mine 300
node smoke-live.mjs --peer … --pool /ip4/…/p2p/… --pool-payout <hex>
```

Read-only by default; nothing spends or touches key material without an explicit
flag. It checks the assumptions the offline suite had to assume: that
`get_filters` really returns full-span arrays, that a zero `element_count` never
accompanies a non-empty filter, that the block hash from `get_filters` matches
`extension.final_hash` from `get_block` (the identity the whole reorg module keys
on), that `mining_midstate` is not `post_tx_midstate`, and that a live template's
timestamp is fresh enough to survive the median-time check.


---

## DEX order announcements (new — `src/dex.js`)

Ported from `core/dex.rs`, the node's canonical port of the browser wallet, so
all three read and write the same order book.

**Every integer here is big-endian**, unlike `bincode.js` and the Q-Bolt channel
wire, which are little-endian. That is the single easiest thing to get wrong and
it fails silently — a little-endian implementation round-trips against itself
perfectly while producing announcements the node cannot read. The tests therefore
assert byte offsets against vectors built from the Rust source rather than
round-tripping. Verified non-vacuously: flipping to little-endian passes the
round-trip test and fails the offset test with `a005000000000000`.

### What announcements are for

A swap's coin salt is the one piece of state a wallet cannot re-derive from its
seed. Lose it and the funds are stranded even though the key is intact.
Publishing the salt — never the preimage — in a zero-value `DataBurn` makes every
order recoverable from seed alone. This is a **recovery channel first**, an order
book second.

### What's implemented

- **MDXA maker orders** and **MDXT taker locks**: encode/decode, exact byte
  layout (72-byte header + 81 per unit), `u128` wei amounts held as BigInt.
- **MDXF fragmentation.** Consensus caps a burn at `MAX_BURN_DATA_SIZE` = 80
  bytes and a one-unit MDXA is 153, so it can never fit — it ships as 3 fragments
  in the *same* transaction. `FragmentPool` reassembles out of order, tolerates
  interleaved groups, and is bounded so a long scan can't grow without limit on
  fragments whose siblings never arrive.
- **Burn extraction.** `GetBlock` serializes `Vec<u8>` as a JSON **number
  array** — `{"DataBurn":{"payload":[77,68,...]}}` — which no hex regex matches,
  so this walks the object tree and accepts either shape.
- **Covenant derivation.** `deriveCovenantAddress` recomputes the address from
  the announced terms, so a maker cannot advertise one set of terms and fund a
  different covenant. `expandOrders` turns an announcement into resolvable,
  priced orders with coin ids derived locally.
- **Secrets.** One secret backs exactly one unit — a single hashlock backs one
  trustless fill, and once the maker reveals the preimage `H` is public. Reusing
  a secret across units would let the first fill unlock all of them.

### Not implemented

The **swap state machine** is not ported. The browser wallet has ~30 DEX message
types covering the full cross-chain lifecycle (offer → accept → lock → claim →
refund), EVM-side coordination, submarine swaps, bundle recovery, and encrypted
secret custody with its own stated invariants. `src/dex.js` is the wire format and
covenant math those flows are built on, not the flows themselves.


---

## Token / AMM / bonding-curve state (new — `src/launcher.js`)

Midstate has no native token type. A token is a **coloured coin**: a state thread
whose 32-byte commitment carries `[balance u64][assetID 24]`, with a script
enforcing conservation of mass across the outputs. An AMM is the same idea with
`[reserveX u64][reserveY u64][padding 16]`. Both layouts, and the contracts that
read them, are taken from the Midstate IDE templates.

**Ground truth is the IDE's own worked examples.** Their `inputState` and
`outputs` strings are states a human authored and the VM accepted, so matching
them byte-for-byte is stronger evidence than any round-trip. Every one
reproduces exactly, including the constant-product trade the IDE annotates as
`X=10, Y=100 → trade 10 X → out 49 Y, x_adj·y_adj = 1,018,470,000 ≥ 1e9`.

### The two traps this module exists to close

**1. State fields are LITTLE-endian.** `OP_SLICE` feeds `to_u64`, which is
`u64::from_le_bytes` — so a balance of 10 is `0a00000000000000`. This is the exact
opposite of the DEX announcement wire in `dex.js`. Getting it backwards doesn't
fail loudly; it produces a state the contract reads as ~1.15e18, so the resulting
assert failure looks nothing like the bug. Verified non-vacuously.

**2. SDK arithmetic must match the contract's exactly.** The constant-product
contract asserts, in integers:

```
dx = max(0, new_x - x);   dy = max(0, new_y - y)
x_adj = new_x*1000 - dx*3;  y_adj = new_y*1000 - dy*3
assert(x_adj * y_adj >= (x * y) * 1000000)
```

A quote computed in floating point, or with the fee applied in a different order,
lands one unit on the wrong side and is rejected on-chain *after* the user paid
for a commit. So `satisfiesConstantProduct` is a literal transcription of that
assert, and every quote is checked against it before being returned — not against
the formula that produced it. Fuzzed across reserve ratios from 1:1 to 1e9:1e12,
and each quote is confirmed maximal (one more unit out would be rejected).

### What's included

- `encodeState` / `decodeState` for the three IDE layouts, with u64 bounds
  enforced rather than truncated.
- `getAmountOut` / `getAmountIn` / `quoteSwap` — exact BigInt integer math,
  0.3% fee, pool never fully drained.
- `bondingCurveCost` / `bondingCurveNextState` — the linear curve where price
  equals current supply. Note the template mints **one unit per transaction**,
  so N units is N transactions; the closed-form series is provided so a caller
  can show a total without simulating each step.
- `splitTokenState` — conservation computed here, since producing outputs that
  don't sum exactly is the commonest way to author an unspendable token coin.
- `deriveAssetId`, `contractAddress`.

**A bug found while testing this:** the off-by-one correction loops were
unbounded. Breaking the fee constant made them grind down billions of units one
at a time instead of failing. Now bounded to a slack of 4 with a diagnostic
naming the actual cause.

### What's not included

No bytecode. Contracts are authored and compiled in the IDE; this module handles
the state they read and the arithmetic that keeps a proposed trade legal.
`contractAddress(bytecode)` bridges the two.


---

## Contract compiler and headless VM (new — `src/compiler.js`, `src/vm.js`)

Lifted from the Midstate IDE with no language changes, so a contract compiled
here produces the same bytecode the IDE shows and therefore the same P2SH
address.

### Why the SDK needs a compiler

A launcher cannot pre-compile its contracts. Every token needs its own curve with
that token's parameters baked in, which means a distinct script and a distinct
address per launch. Compiling by hand works for one contract, not a thousand.

**Validation:** all 17 IDE templates compile, and the opcode table is asserted
against `core/script.rs` rather than trusted — a drifted table assembles
cleanly, hashes to a plausible address, and fails at execution, *after* funds are
locked there. (The node also defines `OP_SUM_INPUT_VALUE = 0x56`, which this
language does not expose.)

### Why the SDK needs a VM

A launcher funds a *fresh* address per token. A wrong contract locks its funds
where nobody can spend from — no second chance, no upgrade path. Simulating a
candidate transaction before paying for a commit turns that from "funds gone"
into "assertion failed".

The VM is validated against the IDE's own annotated AMM example: it accepts the
trade the IDE says is valid (`X 10→20, Y 100→51`) and rejects one unit more
output. That ties it to observed behaviour rather than to my reading of the
opcode table. There's also a cross-check that every quote `getAmountOut` produces
is one the compiled contract actually accepts.

**Two deliberate limits.** `CHECKSIG`/`CHECKSIGVERIFY` compare strings (inherited
from the IDE emulator; the real VM verifies WOTS/MSS), and `HASH` UTF-8 encodes a
non-hex operand. So the VM proves *arithmetic and state* correctness — where
curve and AMM bugs actually live — and proves nothing about authorisation.

---

## Bonding curve contract (new)

The IDE's `stateful` template asserts `new_supply == supply + 1`: one unit per
transaction, minutes per token at two blocks per commit/reveal, and no reserve —
payment goes to a treasury, so the curve is buy-only and nothing can be sold back.

`CURVE_CONTRACT_SOURCE` fixes both. Batch delta, and the contract holds its own
MDS:

```
buy  n:  cost   = n*s  + n*(n-1)/2
sell n:  refund = n*ns + n*(n-1)/2      (exact mirror)
```

`input_value() == reserve` and `sum_to_addr(this_address()) == new_reserve` stop
the state lying about the MDS it claims to back. `this_address()` avoids the
chicken-and-egg of baking an address into the bytecode that determines it.

**Verified by execution, not inspection:** exact-cost buys and sells accepted at
every supply level, underpayment and over-refund rejected by a single unit, free
minting rejected, a mismatched reserve rejected on both the input and output
side, and a buy/sell round trip returning the reserve *exactly* — the property
that stops repeated round trips bleeding the curve. Fuzzed against the SDK's own
arithmetic so a quote and the deployed contract never disagree.


---

## Pump-style token launcher (new — `src/pump.js`)

A bonding curve with an **unforgeable holder ledger**, buildable today with no
consensus change. Every assertion in `test-pump.mjs` runs the compiled contract
in the VM — nothing is checked against the SDK's arithmetic alone.

### Why the obvious designs don't work

Midstate has no native token type and no minting policy, so a launcher has to
build one from scripts. Two designs look right and both are broken:

1. **A script only runs when its own coin is spent**, and `READ_INPUT_STATE`
   reads only its own input. On a sale the curve sees its supply but not the
   token coin's balance; the token sees its balance but not the curve's supply.
   Neither can verify the other's delta.
2. **Creating an output does not run that address's script.** So anyone can
   fabricate a coin at the token address claiming any balance, then "sell" it and
   drain the reserve down to the real supply.

A same-transaction receipt fixes (1) — both scripts assert the delta against
their own knowledge, forcing agreement — but nothing fixes (2) without a
consensus-level minting rule.

### What does work

Balances live in a fixed-depth Merkle tree whose root sits in the curve's own
state. A trade must present a path proving its **current** balance against the
**current** root, and the contract recomputes old and new roots in one pass
(`merkle_update_step`, from the IDE standard library). A balance never bought has
no valid path. Fabrication is impossible because there is nothing to fabricate —
there are no token coins, only leaves.

```
reserve  = the contract coin's own MDS value (never stored, so it cannot lie)
buy  n:  cost   = n*s  + n*(n-1)/2
sell n:  refund = n*ns + n*(n-1)/2      (exact mirror; round trip lossless)
leaf     = BLAKE3(minimal_LE(balance) ⌢ owner24)
```

The contract **builds both leaves itself** from `(owner, old_balance, n)` rather
than trusting witness leaves. That binding is what stops a buyer paying for 4 and
writing 400 into the tree.

### Verified by execution

Honest: first buy, rising price for the second buyer, sell-back, lossless round
trip at several starting supplies, eight holders trading independently and all
exiting to zero residue.

Attacks, all rejected: selling a balance never bought; underpaying a buy by one
unit; over-refunding a sell by one; paying for 4 and crediting 400; inflating
supply without paying; spending another holder's slot; reusing a stale proof
after someone else trades; diverting the reserve to a foreign address; an unknown
route. Plus a fuzz run asserting the reserve always equals the cost of the
supply, whatever the trade sequence.

### Limits worth knowing before building on it

- **Depth 10 (1024 holder slots)** is the largest that fits `MAX_SCRIPT_SIZE`.
  Deeper needs the language to emit a loop rather than unrolled steps.
- **Trades serialise.** Every trade spends the single curve coin, and each one
  moves the root, so a front end must rebuild paths between trades. Tested: a
  stale proof fails closed. This is inherent to a single-UTXO curve.
- **Owner is an identity slot, not an authorisation check.** The contract does
  not verify a signature; slot ownership must be enforced by whatever controls
  tree updates. Adding `require_signed_by` costs ~40 B and one witness item.
- **No graduation to an AMM yet.** The constant-product template in
  `launcher.js` has no pool-creation path, so migrating a filled curve into a
  pool is unimplemented.

### A bug worth recording

An early version placed `PICK_OLD` and `PICK_OLD_B` as substitution
placeholders where one was a prefix of the other. The mangled offset picked a
neighbouring stack slot, making both Merkle leaves identical — which **passed
every forgery test while rejecting every honest trade**. Only running the
contract in the VM and tracing the stack found it. The shipped module derives
offsets from depth in code, with unambiguous placeholders.


---

## Correction: state threads cannot carry value

A review question — *"did you use state threads / eUTXOs?"* — surfaced a real bug
in what had already been shipped, and the fix touches the IDE templates too.

`OutputData::Confidential` has **no value field**. A state thread carries state
and nothing else, and `apply_transaction` rejects any attempt to give one a
value: *"State threads (Confidential outputs) must have value exactly 0. To send
value AND state, create two outputs."*

Both curve contracts stored the reserve in the state-carrying coin and read it
with `input_value()`. They executed perfectly in the script VM and **would have
been rejected by every node on the network** — because that rule lives in
transaction application, not in the script VM, so simulating the script alone
could never catch it.

### The fix

Every trade now produces **two** outputs at the contract address:

```
output 0 : Confidential state thread  [supply 8][root 24]   value 0
output 1 : Standard reserve coin                            value = new reserve
```

and the contract reads the total with `sum_input_value()` rather than
`input_value()`, because the script runs once per input and the state thread's
own value is zero. `OP_SUM_INPUT_VALUE` (0x56) exists in the node — *"sum of the
values of every input in this transaction that shares the currently-executing
predicate"* — but the IDE language did not expose it, which is very likely why
the templates were written the way they were. It is now emitted by the compiler
and modelled in the VM, including its `COVENANT_SUM_ACTIVATION_HEIGHT` gate.

### The VM now enforces it

The zero-value rule is checked before a single opcode runs, so this class of bug
fails loudly in simulation instead of on chain. `allowValueBearingState: true` is
available strictly to reproduce the IDE emulator, whose `address:value:state`
context format permits the combination — it should never be used to validate a
contract intended for deployment.

### Correction: activation heights

`COVENANT_SUM_ACTIVATION_HEIGHT = COMMIT_WEIGHT_CAP_ACTIVATION_HEIGHT = 300_000`,
**not** the state-thread height of 65,000. The VM had the wrong constant, so the
pump tests were simulating at height 100,000 — below activation, where a real
node returns `InvalidOpcode` for `sum_input_value()` and the contract cannot run
at all.

Corrected, test heights raised above the gate, and a test now pins that the
contract IS rejected below it. **`npm run smoke` checks both gates against the
live chain before anything else**, because deploying the curve below 300,000
would strand its reserve at an address nobody can spend from until activation.

### On the name `Confidential`

The variant is called `OutputData::Confidential` and the coin-id domain tag is
the ASCII string `"CONFIDENTIAL"`, but **nothing about these outputs is
confidential**. The name is vestigial: it predates the removal of zk-STARK
confidential transactions, and state threads were retconned into the same
variant.

The node's own source is blunt about it — *"The name is a trap and has already
caused one incorrect fix"* — and carries a regression test,
`state_thread_output_contributes_no_value_to_the_fee`, that replaced an older one
written on the assumption the value was hidden.

The correct model, and the one this SDK implements:

```
value(Confidential{..}) = 0        — known to be zero, NOT unknown
```

A covenant spending 100 into `[Standard 96, Confidential(state)]` really does pay
a fee of 4; the state thread does not absorb the missing 4 into a hidden amount.
Reading the value as *hidden* rather than *zero* is the mirror image of the bug
fixed above: instead of over-crediting a state thread, it would make a wallet
skip these outputs from conservation entirely.

The SDK now names the helper `computeStateThreadCoinId`, keeps `"CONFIDENTIAL"`
only where it is literally the wire tag, and documents the trap at the two places
a reader would otherwise re-derive the wrong model — the coin-id helper and the
VM's zero-value check. A test pins the semantics.

### This affects the IDE's own templates

`amm_uniswap` asserts `input_value() == x` while `read_output_state(0)` expects
the state on the same output. With the correct on-chain shape, `input_value()`
reads 0 for the state thread and the assertion fails. **The template is not
deployable as written.** Its arithmetic is sound — the IDE's worked example still
verifies, and is still tested here under the legacy flag — but it needs
`sum_input_value()` and a split output to run on a real node. The same applies to
any contract modelled on it.


---

## API surface

- **`submitChat()` added** to `MidstateClient`. `LightRequest::SubmitChat` and
  the `/api/chat/submit` route both existed; the SDK only had `send_chat`. The
  difference matters: `send_chat` asks the node to mine PoW and broadcast under
  its own peer id, while `submit_chat` carries a nonce you mined yourself (via
  `mine_chat_pow_v2_wasm`) so the message goes out under your identity. Route,
  field names and types verified against `SubmitChatRequest` in `rpc/types.rs`.

- **`getNewReusableAddress(height, onProgress)`** — `get_mss_address` gained a
  third `progress_cb` parameter. Tree generation is 2^height keypairs, so this
  is the slowest wallet operation and worth reporting.

- **`src/reorg.js` added** — exports `maybeHandleReorg`, `rollbackTo`,
  `findForkHeight`, `pruneHistory`, `REORG_DEPTH`, `COMMITMENT_TTL`.

- **`src/bincode.js` extracted.** The wire codec had been embedded in `p2p.js`,
  which meant testing the byte format required a full libp2p install. It is now
  a dependency-free module, re-exported from `p2p.js` so existing imports keep
  working. Adds `decodeChatV2`, `encodePong`, and named discriminant constants.

- **`src/index.js` rebuilt** against `pkg/wasm_wallet.d.ts` — 33 exports grouped
  by purpose, including the Q-Bolt channel builders, HTLC/covenant bytecode
  builders, and `initWasm`/`initWasmSync`.

- **Rate-limit documentation corrected**: 20 violations → 500-second ban
  (`LIGHT_BAN_THRESHOLD`, `LIGHT_BAN_DURATION_SECS`), not the documented
  "10 violations → 5 minutes".

---

## Checked and deliberately not changed

**`Number(u.value)` when serializing UTXOs.** I flagged this as a possible
precision hazard and it is not one. `WasmUtxo.value` is a u64 that serde parses
from a JSON number, so the BigInt must be narrowed. That is safe rather than
lucky: coin values are always powers of two, and total supply is bounded near
2^50 (`INITIAL_REWARD` = 2^30, halving yearly at `TARGET_BLOCK_TIME` = 60s),
while JSON round-trips integers exactly through 2^53. Divergence only begins at
2^56. A guard now throws if a value ever exceeds the safe range instead of
narrowing silently.

**Commit-confirmation poll pacing.** Also flagged, also not a problem.
`check_commitment` is a cheap request — only `BlockTemplate` and `GetFilters`
count against the expensive budget — and one call per 10s is 6/min against
~275/min.

---

## Not done

- **`test.js`** (the live-network suite) is unchanged. It still passes: all 16
  offline steps green, including the PEX codec step that the `decodeAddr` bounds
  fix repaired. It has no coverage of the new paths.
- **Load-aware peer placement** from `light_client.js` is not ported. This node's
  `get_state` returns no `light_load`, `light_connections` or `webrtc_addrs`, so
  it would be inert today.
- **No live-network validation.** All testing is offline against vectors derived
  from the Rust source. The commit→reveal path in particular has been
  restructured and unit-tested with a mock client, but never run against a real
  node.
- `decrypt_cli_wallet` is now re-exported from the index for CLI wallet import,
  but has no `Wallet.importFromCli()` convenience wrapper.
- **GPU mining.** `searchFn` is injectable and `MinerPool` shows the shape a
  backend takes, but `pow.wgsl` and its self-test harness are not ported. The
  self-test gate matters: the browser miner refuses to mine on a kernel that is
  not bit-identical to the WASM reference, and any GPU backend added here should
  keep that property.
- **Fee-coin root planting.** `verifyPoolProof` checks the path against the job's
  `merkle_root`, but nothing yet confirms that root is actually planted in the
  salt of the coinbase output paying the pool fee address. That closes the last
  gap between "the pool told me a root" and "the chain committed to that root".
