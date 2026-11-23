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
| 7 | — | held for a Nostr public key; not built yet |
| 8 | kem | `H(key)` of an ML-KEM-512 public key, for sealed messages |
| 9 | script | part of the script behind an mds-address; the parts in order are the script |
| 10 | postage | one byte, 16–40: the stamp bits a stranger's sealed message to the name must show |

Tags 8–10 are client-side: a client that predates them keeps and ignores them,
as it does any tag (and as clients do tag 7 today). A `postage` outside 16–40,
or not one byte, is kept and read as no price: the standard, 24.

An MDS payment goes to the first mds-address, or mds-key with at least 16
signatures left. Senders refuse a name that has lapsed or lapses within a day.
A sender that can reach the name's identity key asks it for a fresh one-time
address first, and pays the record only if no answer comes.

- **What kind of address.** An mds-address is the hash of a script, so the
  address alone says nothing. When the records publish the script (`script`
  parts that hash to the address), wallets, the explorer and `GET /name` show
  what it is: a single key, an m-of-n multisig, a vault, an inheritance plan, a
  payment stream, a covenant. A name that publishes nothing shows as a script
  address of unpublished kind.
- **Sealed messages.** The `kem` record lets anyone open a sealed chat session
  with the name; see `src/sealed.rs`. The `postage` record is what the holder
  charges a stranger to be written to: each step of 4 bits is sixteen times
  the work. A wallet holding several names asks the lowest of their prices,
  so a stranger who paid what the name they wrote to asks is heard.

## Software

- **Node:** follows the registry inside each block's write, with an undo log,
  so reorgs unwind exactly. Always on.
  - `GET /name/nick`: holder, tier, `since`, `expires`, deal, records, and
    where MDS pays now, with what kind of address that is (`pays_kind`), the
    index's `height`, the identity, kem and published script, the MSB-1
    `backing` burned for it, and `registered`, the height its current
    registration began: backing counts from there, so a name another key
    claims after it lapsed starts from none (`docs/MSB-1.md`).
  - `GET /name/nick/proof`: the name's leaf, the records and terms it commits
    to, and its Merkle proof to the root of the registry's current thread (the
    predecessor's leaf, for a name never registered). Clients check a node's
    answer with it rather than trust it (midwimble's resolver does).
  - `GET /names/market`: names offered for sale or rent.
- **Explorer:** search `nick.mds`: what it pays and what kind of address that
  is, its wallet's handshake and sealed messages, its MSB-1 backing, and a
  link to its proof.
- **Pool:** a miner may give `nick.mds` as its payout; each job resolves it
  by `pays`, and refuses one that has lapsed or lapses within a day
  (`pools.md`).
- **Midwimble:** `midwimble wallet send --to nick.mds`, `midwimble name nick`,
  and the explorer resolve a name's MDW record through a Midstate node
  (`--midstate-rpc`, your own by default) and check it against the registry
  root with the proof. Then they have the node prove the registry coin
  unspent (`GET /utxo_proof/:coin`) and check that proof against the state
  root of the midstate header midwimble consensus relayed at that height
  (`GET /relay/{height}` on the midwimble node). A node whose state is not the
  relayed chain's is refused; when the relay has not reached the node's tip
  yet, the answer says the state is the node's word.
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
    - put the name's own records back when a rental ends;
    - keep a name pointed at one of the wallet's inheritance plans on the
      plan's next deadline (the address a check-in moves its coins to), so
      payments never land where the deadline has passed and heirs could
      claim them at once; a name on a plan the wallet closed goes back to
      its own key.
  - Send to `nick.mds` in MDS or MDW. A name with an `identity` record is
    asked for a fresh one-time address first (signed by that identity); if its
    wallet does not answer within 20 seconds, the record is paid as before. A
    name that points at a shared address (a published vault, multisig or plan
    script) is paid there, never around it.
  - **Edit records** also points a name at a vault, multisig, plan or shared
    wallet (this wallet's plans are offered, scripts published so the kind
    shows), and takes sealed messages (`kem`). Payments to a plan's address
    carry their payers' salts, so the owner's and the heirs' wallets watch
    every address of the plan in their block scan and keep what arrives as
    the plan's coins: check-ins carry it along, and claims pay it out. New names get
    `identity` and `kem` records by default.
  - **Sealed messages** (the Messages page): write to `nick.mds`; only its
    wallet reads it. One identity signature opens a session; the session key
    does the rest. The page is a messenger: conversations down the left with
    unread counts, the open one beside them, replies within it (to whoever
    opened it, verified or not), and the public room pinned at the top.
    Senders under the inbox's bar wait under Requests until accepted or
    answered; a conversation can be deleted, and messages still waiting for
    a name's key can be stopped.
  - **Strangers pay to be heard, at the recipient's price.** Until a wallet
    answers or accepts a conversation, every message in it must carry a
    stamp: proof of work bound to that one message, inside the seal so only
    the recipient can check it, of the bits the recipient's names ask
    (`postage`; 24 bits, about a second of one core, where they say none).
    Wallets stamp until the other side writes back. A wallet under attack
    raises its price from Messages (Strangers pay: standard, raised 28,
    high 32, very high 36, maximum 40 bits), which changes the `postage`
    record on each of its names that takes sealed messages; it applies once
    that confirms, and senders read it from the chain. Stamps are worked on
    a thread of their own, cheapest first, so a dear one holds nothing up: a
    message being stamped shows as such, can be called off, goes when its
    stamp is done, and its work starts again after a restart (an opening
    being stamped is kept made up to its stamp, so it is signed once). A
    price that changes while a stamp is worked lands on the work in hand.
    A request keeps at most three of its sender's
    messages, raises no notice, and goes after 30 days unanswered; at most
    100 are kept. A verified name can be blocked (its conversations go and
    nothing it sends is read); anyone else is blocked by forgetting the
    conversation, and anonymous senders can be dropped altogether.
  - **Disappearing messages.** A conversation can have a timer (an hour, a
    day, a week, four weeks): every message carries it, and both wallets
    delete a message once it has run. The sealed copy stays on the chat bus
    as long as nodes keep it, readable by whoever holds the session (see
    `src/sealed.rs`): a timer is tidiness, not forward secrecy.
  - **Swap market:** a seller's or buyer's name shows on its order or trade
    when its Base account signed the claim and the name's `evm` record is that
    account. The account stays what the trade uses.
  - **MDW market:** a maker's name shows on its order when the claim
    (`mwb_wire::NameClaim`, chat kind 91, which older wallets skip) is signed
    by the spend key of the MDW address the name's `mdw` record gives.
  - **Backing (MSB-1):** the Send page shows what has been burned to back a
    name being paid, and flags one with none; Find, Your names and the market
    show it too, with a way to add to it.
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
