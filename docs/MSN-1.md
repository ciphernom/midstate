# MSN-1: names

`nick.mds` resolves to typed records: where to pay it in MDS and MDW, its Base
account, its identity key, a line of text. Names are bought by the year,
burning a price set by tier, and can be sold and rented out. Built from script
features the chain already has (the v6 opcodes); no fork. Code:
`src/core/names.rs` (covenant, tree, tiers, deals, records),
`src/storage/name_index.rs` (follower), `crates/walletd/src/service_names.rs`
(wallet).

## The registry

One state thread at `A = H(covenant)` (987 bytes): a zero-value coin whose
commitment is the root of a depth-24 Merkle tree (2^24 slots). The leaves are a
sorted linked list:

```
leaf = H(key ‖ next ‖ expires ‖ tier ‖ owner ‖ data ‖ deal)   8+8+8+1+32+32+32 bytes
key  = H("MSN1" ‖ name)[0..8], little-endian u64
data = H(records),  deal = H(terms)
```

- `owner` is the **control key**. It signs changes and nothing else.
- A name is held until `expires`. Every year of holding burns its tier's rate
  (below), so renewals cost too.
- **Genesis:** slot 0 holds a sentinel. Slots from 1 hold every premium name,
  unclaimed (owner zero, lapsed since block 0), and every reserved name, held
  by its key until block 1,000,000 (`RESERVED` in `names.rs`: the project's
  own — ciphernom). The genesis coin is
  `Confidential { A, root, H("MSN1 genesis") }`. **The registry is the first
  genesis coin at or after height 371,500**, then output 0 of each spend of the
  current thread. Look-alike threads at `A` are ignored. A new thread's salt is
  `H("MSN1 tip" ‖ spent coin id)` by convention.

## Tiers

A year of holding burns `DataBurn` outputs worth at least the name's rate. A
name keeps its tier for good.

| tier | per year | names |
|---|---|---|
| platinum | 64 gMDS | 1–2 characters, and listed names (bitcoin, wallet, google, …) |
| gold | 8 gMDS | 3 characters, and listed names (love, casino, netflix, …) |
| silver | 1 gMDS | 4 characters, and listed names (common first names and words) |
| bronze | ⅛ gMDS | 5 characters or more |

The list (`PREMIUM` in `names.rs`, about 230 names) is fixed by the genesis
root. A spend buys 1 to 10 years (`expires ≤ max(old expiry, height) +
years × YEAR`, YEAR = 525,600 blocks).

## Deals

`terms` are `[0]` (nothing), or `kind ‖ price (8) ‖ payee (32) ‖ time (8)`:

| kind | deal | |
|---|---|---|
| 1 | for sale | anyone takes it by paying `price` to `payee` |
| 2 | for rent | anyone takes its records for `time` blocks by paying `price` to `payee` |
| 3 | rented | the renter's records until `time`; the owner cannot change the name before then |

## Spends

The selector is exactly 0–4. A proof is 24 × (sibling 32 ‖ side 1).

| spend | witness, bottom to top | checks |
|---|---|---|
| 0 register | `P2, P1, ltail, lnext, lkey, years, expires, owner, records, terms, 0, name` | `lkey < key < lnext`; P1 proves the neighbour; P2 an empty slot once the neighbour points at `key`; tier from length; years ≥ 1 |
| 1 update | `P, sig, oterms, oowner, odata, otier, oexpires, next, years, expires, owner, records, terms, 1, name` | `sig` by `oowner`; if rented, height ≥ its end |
| 2 reclaim | same, selector 2 | height > `oexpires`; years ≥ 1 |
| 3 buy | same, selector 3 | `oterms` a sale, its price paid to its payee |
| 4 rent | same, selector 4 | `oterms` a rent offer, paid; new terms rented at the same price and payee, until ≤ height + span and ≥ 30 days before expiry; owner and expiry unchanged |

Every spend checks years ≤ 10 and the expiry rule, burns `rate × years`, and
requires output 0 to be at `A`. Sizes are checked wherever a split could move a
field.

## Records

`tag ‖ length (1 byte) ‖ value`, at most 1,536 bytes. Unknown tags are kept and
ignored.

| tag | record | value |
|---|---|---|
| 1 | mds-key | MSS key ‖ tree height: pay `compute_address(key)` |
| 2 | mds-address | any midstate address, paid as is |
| 3 | mdw | midwimble stealth address, as text |
| 4 | evm | Base account, 20 bytes |
| 5 | identity | identity key: ask it for a fresh address, an invoice, a channel |
| 6 | text | a line of UTF-8 |

An MDS payment goes to the first mds-address, or mds-key with at least 16
signatures left. Senders refuse a name that has lapsed or lapses within a day.

## Software

- **Node:** follows the registry inside each block's write, with an undo log,
  so reorgs unwind exactly. Always on.
  - `GET /name/nick`: holder, tier, `since`, `expires`, deal, records, and
    where MDS pays now.
  - `GET /names/market`: names offered for sale or rent.
- **Explorer:** search `nick.mds`.
- **Wallet (the Names page):**
  - **Find a name:** one box shows what any name is (available, unclaimed,
    lapsed, held, for sale, for rent, rented, yours), what it costs, and the
    one thing to do with it. Every spend asks for a confirm that says what it
    burns, pays and holds.
  - **Register** for 1–10 years. Premium and lapsed names are claimed the same
    way.
  - **Watch** a name: claim it the block it comes free, or buy it if it is
    offered for at most a price.
  - **Give…** a name to another wallet, by its name or control key.
  - The wallet finds names its keys hold — reserved at genesis, given to it,
    or held before a restore from its phrase — through the follower's owner
    index, and sets up any without records.
  - **Edit records**, **Renew a year**, and **Renew automatically** (on by
    default: a year at a time when 60 days are left).
  - **Sell…** or **Rent out…**, priced in gMDS; payment goes to a fresh address
    of the wallet's. **Stop offering** takes the offer down.
  - **Market:** every offer, searchable, filtered by kind and sorted by price
    or by how soon it lapses; **Buy…** and **Rent…** open it in Find. Recent
    sales and rentals are listed below it.
  - **Upkeep, once a block, unprompted:**
    - swap payment keys with 64 or fewer signatures left, and sweep coins that
      still reach the old ones;
    - hand control to a fresh key when the control key has 16 or fewer;
    - put the name's own records back when a rental ends.
  - Send to `nick.mds` in MDS or MDW.
  - A change that loses the race for the thread is tried again the next block.
    The chain's first registration also makes the genesis coin.

## What the opcodes cannot do

- No opcode reads an output's salt or counts outputs, so look-alike threads
  cannot be stopped. Which thread is the registry is a matter of lineage, known
  only to a node that has seen every block since 371,500.
- There is no "height ≤ x" check, only `OP_CHECKTIMEVERIFY` (height ≥ x).
  - A rental cannot end itself. It locks the owner out until its end, and the
    owner's wallet then takes the records back.
  - Selling a lapsed name cannot be refused, but buying one gains nothing that
    a reclaim would not.
- No loops: the canonical-name rule and the record format are checked by
  clients.
- 64-bit keys (`OP_LESS_THAN` compares u64s). A held name cannot be taken, and
  a name whose key matches a held one's can never be registered (finding one
  takes about 2^64 hashes).
- Prices are fixed in units, in the covenant. Changing them means a new
  registry.
- One spend per thread: changes queue, and wallets retry.
