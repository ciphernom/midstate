# Launchpad

`launch.html` on the site runs pump.fun-style launches on Midstate. Each launch
starts as a bonding curve and becomes a constant-product pool. One script does
everything. There is no operator and no admin key. The script holds the MDS,
checks every trade, and keeps the holder ledger. Launches need the v6 opcodes,
so they exist from block **371,500**.

The code is in `sdk/src/launchpad.js` (SDK 2.11.0, `Launchpad`). Its tests are
in `sdk/test-launchpad.mjs`, and `tests/launchpad_vectors.rs` replays them on
the node's VM.

## Economics

| | |
|---|---|
| Curve | The k-th token sold costs (k−1)·P MDS. The curve sells T tokens (T even). |
| Raise | P·(T²−T)/2 MDS when the curve has sold out. |
| Pool | Opens with the raise and T/2 reserved tokens. Its price is x/y = P·(T−1), the curve's last price. |
| Pool fee | 0.3%. It stays in the pool. |
| Supply | 1.5·T. No more can ever exist. |
| Holders | 2^depth ledger rows (depth 12, 14 or 16 on the page). |

Graduation is one-way. The curve can't sell past T, and the pool can't change
the supply.

`validateParams` keeps the pool's x·y below 2^60, so the u64 checks have room
for fee growth. In practice the raise is limited to about 2^61 / T MDS.

## Why the state is in the address

Consensus runs a script once per input, and each input sees only its own state
commitment. A state coin carries no value, and a value coin carries no state.
So a "state thread plus reserve coin" contract can't spend its reserve: the
reserve coin's run has no state to read. `pump.js` and `launcher.js` were built
that way and stop after their first buy.

Here the bytecode is `PUSH_DATA(state, 56) DROP code`, so every coin at an
address carries the full state.

- **State:** `supply u64 | x u64 (MDS) | y u64 (pool tokens) | root 32`, little-endian.
- **Self-check:** the witness carries a copy of `code`, and the script checks
  `hash(01 3800 ‖ state ‖ 10 ‖ code) == this_address()`.
- **Next address:** the script builds the next state from the witness, hashes
  it the same way, and requires all of `x' + seed` to go there.
- **No forks:** `sum_input_value() == x + seed`, so every trade spends every
  reserve coin. To fork a state, someone has to pay a full second copy of its
  reserve to that address, and that copy only funds the fork.

## A trade

**Witness, in push order (13 items):**

| # | Item | Size (bytes) |
|---|---|---|
| 0 | `code_a` | about half the code |
| 1 | `code_b` | the rest of the code |
| 2 | `state` | 56 |
| 3 | `proof` | 33·depth |
| 4 | `owner_old` | 24 |
| 5 | `owner_new` | 24 |
| 6 | `pk` | 32 |
| 7 | `old_bal` | 8 |
| 8 | `new_bal` | 8 |
| 9 | `s'` | 8 |
| 10 | `x'` | 8 |
| 11 | `y'` | 8 |
| 12 | `sig` | MSS signature |

**What the script requires:**

- **Signature:** `sig` is a signature by `pk` over the transaction's
  commitment. If the row holds tokens, `hash(pk)[:24] == owner_old`. An owner
  can give the whole row to a new owner, which is how a position moves to
  another key or person.
- **Proofs:** the old row is in the ledger (`OP_MERKLE_ROOT`), and the new row
  replaces it at the same position, using the same proof.
- **Conservation:** `new_bal + y' + s == old_bal + y + s'`.
- **Curve phase (s < T):** `y' == y` and `x' == P·(s'²−s')/2`, with `s' ≤ T`.
- **Pool phase (s == T):** `s' == s`, and x·y doesn't fall after the 0.3% fee
  on the side paid in.
- **Receipt:** output 0 is a zero-value state coin at the next address whose
  32 bytes are `owner_new ‖ le8(new_bal)`.
- **Reserve:** outputs to the next address total `x' + seed`.

**Size checks.** Each check on state, balance and next-state field widths
stops a real attack, and each attack is in the tests:

- **State size:** a longer state copy would re-split the code, letting the
  reserve go to an anyone-can-spend tail of the script.
- **Next-state widths:** a short field would send the reserve to a script
  nothing can ever spend.
- **Balance widths:** a 0-byte balance would let a full row be read as an empty
  one and taken.

**Why the code travels in two halves.** Nodes record the "leaf" of any witness
item that parses as an MSS signature: 620 bytes or more, with a zero auth
length accepted whatever follows. Nodes also read a 576-byte first item as a
WOTS signature. A whole code copy is about 600–700 bytes. If one were read as a
signature, the first trade would mark it spent, and every later trade would be
refused as key reuse. So no item except `sig` may exceed 575 bytes, which also
keeps the ledger at depth 16 or less.

## Reading launches from the chain

Nodes serve blocks with their witnesses stripped. A stripped block still has
everything a reader needs:

- **The state being spent:** it's in the input's bytecode.
- **The next address and the new row:** both come from the receipt.
- **The new reserve:** the outputs to that address.

`Launch.applyTx` finds the row by trying the trader's existing row, then the
first empty row, then rows with the same balance (a hand-over), then all rows.
It accepts the row whose resulting state hashes to the receipt's address.

**Finding launches.** A launch transaction carries:

- the seed (1 MDS) at the genesis address;
- 1 MDS to the registry. The registry is the script
  `PUSH "midstate/launchpad/v1" DROP PUSH 0 VERIFY`, which nothing can spend.
- one DataBurn: `"MLP1" | ver 1 | depth 1 | T u32 | P u64 | log2(seed) 1 | nonce 8 | ticker ‖ 00 ‖ name` (big-endian, 80 bytes at most).

The launch's tag is the first 24 bytes of the burn's hash. Untouched ledger rows
carry the tag as their owner, so the name, the ticker and every parameter are
bound into the genesis root and into every address after it. The page watches
the registry and each launch's current address with compact filters, and
fetches only the blocks that match.

## The wallet's part

A row's owner signs every trade on that row, so the key can't be one-time. The
wallet makes an MSS key per site origin, height 8 (256 signatures), derived from
the recovery phrase at index `0xC0000000 + H("midstate/site-key/v1" ‖ origin ‖ generation) mod 2^30`.
When fewer than 16 signatures are left, the next generation becomes current.
`connect` lists the older generations, so the page can move positions off them.

`Midstate.connect({ siteKey: true })` returns `sitePk`, `siteSigBytes`,
`siteKeyLeft` and `sitePks`.

`Midstate.spendContract({ …, siteSig: { item, pk } })` works like this:

1. The page leaves a zero placeholder of `siteSigBytes` bytes at witness
   position `item` of every contract input.
2. The wallet builds the transaction, paying any shortfall and returning change.
3. The wallet signs the commitment, with the leaf taken as the larger of its own
   record and the node's.
4. The wallet puts the signature in every contract input.

The origin comes from the browser, never from the page.

`Midstate.fundContract({ fundings, burns })` pays several outputs, with optional
salts, and carries up to 80-byte public data outputs, all in one transaction.

## Limits

- **One trade per state.** Commit–reveal means two traders racing for the same
  state can't both win. The loser's reveal is refused, and the page re-quotes.
- **Trade size.** Coins are powers of two, so a reserve of x MDS sits in
  popcount(x) coins, and each one carries the full witness (about 3 KB). A trade
  is roughly 3 KB times the coin count, and fees are 10 units per KB.
- **No partial transfers.** Tokens move by buying, selling, or handing over a
  whole row.
