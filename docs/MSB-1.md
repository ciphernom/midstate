# MSB-1: burns that back things

**Status: final, version 1.** The payload, the purposes 1–7, the weight, the
tiers and how a name's backing is counted are fixed: see [Frozen](#frozen) for
what may still be added, and how.

MSB-1 gives people reasons to burn MDS that last as long as the chain does:
burning *backs* a name, a key, a Base account, a mining pool or a payment-channel
hub, and wallets show and rank by that backing; or it *notarizes* a document. It
is client-side only. An MSB-1 burn is an ordinary `DataBurn` output, checked by
consensus as any burn is (a payload of at most 80 bytes, a value that is zero or
a power of two). Nodes index MSB-1 burns, and wallets and the explorer read the
index.

The chain's aim is to burn a little less than it issues each year. MSN-1 names
burn fixed prices, so what they burn is a growing share of an issuance that
halves every year. MSB-1 is weighed in block rewards instead, so the share of
issuance it burns does not change from one halving to the next. It moves only
with how much backing people want to hold.

## Payload

```text
payload = "MSB1" ‖ purpose (1) ‖ subject (32) ‖ label (0–43 bytes, UTF-8)
```

| purpose | backs | subject |
|---|---|---|
| 1 `name` | a `.mds` name | the canonical name, zero-padded |
| 2 `key` | any key or address | its 32 bytes |
| 3 `evm` | a Base (Ethereum) account | its 20 bytes, zero-padded |
| 4 `pool` | a mining pool | the pool's fee address |
| 5 `hub` | a payment-channel hub | the hub's channel identity key |
| 6 `notary` | a document, timestamped | its BLAKE3 hash |
| 7 `notary-sha256` | the same | its SHA-256 hash |

- A name subject must be a canonical name (`a-z`, `0-9`, `-`, as MSN-1 has
  it) followed by zeros; an `evm` subject's last 12 bytes must be zero. A
  payload that breaks either is not MSB-1 (it counts as an "other" burn).
- Purpose 0 is not MSB-1; purposes above 7 are kept and indexed, shown as
  `purpose-N`, for uses not written yet.
- The label is free text: a note, a pool's or a document's name. Readers show
  the most recent non-empty label a subject's burns carry. It is the burner's
  claim, not checked.
- A burn of more than one power of two is several `DataBurn` outputs with one
  payload, in one transaction; they count as one burn. A notarization may burn
  nothing (value 0): the fee pays for the record.
- `GET /burns/payload?purpose=&subject=&label=` returns the bytes, for
  clients that build their own transactions.

## Weight

A burn of `v` units at height `h` weighs `v / block_reward(h)` **blocks**: what it
cost in issuance. Its weight halves every year after (`HALF_LIFE` = 525,600
blocks, one halving of the reward). A subject's **backing** at height `t` is the
sum of its burns' weights then:

```text
backing(t) = Σ  v_i / block_reward(h_i) · 2^(−(t − h_i) / 525,600)
```

Decay is the same for every burn, so subjects keep their order between burns:
with `anchor(v, h) = v / block_reward(h) · 2^(h / 525,600)`, a subject's backing
at any `t` is `Σ anchor · 2^(−t / 525,600)`, and the index ranks subjects by
their anchor sums, which change only when burns do.

Wallets name three tiers: an hour of issuance (60 blocks) is **backed**, a day
(1,440) **well backed**, a week (10,080) **strongly backed**. They are counted in
blocks, so they mean the same in every year.

### Why blocks

Holding `B` blocks of backing costs `B·ln 2 ≈ 0.69·B` blocks a year to keep up,
whatever the year. So all the backing everyone holds, `ΣB`, burns
`ΣB · ln 2 / 525,600` of each year's issuance, in any year:

| backing held, all subjects | share of each year's issuance kept up |
|---|---|
| 76,000 blocks (~7.5 weeks) | 10% |
| 380,000 blocks | 50% |
| 682,000 blocks | 90% |

A fixed price behaves differently. Names burn the same gMDS each year, so their
share doubles at each halving. For a steady 106,350 gMDS a year (the strong
scenario reviewed for MSN-1):

| year | issued (gMDS) | names' share |
|---|---|---|
| 1 | 525,600 | 20% |
| 2 | 262,800 | 40% |
| 3 | 131,400 | 81% |
| 4 | 65,700 | 162% |

MSN-1's prices are consensus and stay as they are. MSB-1 adds burning that holds
its share across halvings. The explorer's Burns page shows the chain's actual
burn against its issuance, by kind, so the aim can be checked rather than
assumed.

## What backing buys

Backing is a cost that a throwaway identity would not carry. Wallets use it
where a cheap identity would otherwise cost the reader:

- **Paying a name.** The Send page shows what backs the name being paid. A
  name with none is flagged: *make sure it is the name you mean.* A lookalike
  would have to burn as much to look the same.
- **Sealed messages.** The inbox has a bar: verified names unless its user
  picks another (everyone, names backed an hour or more, or a day or more).
  Messages below it wait under Requests, never dropped. Conversations show
  their sender's tier.
- **The ETH and MDW swap markets.** A seller's verified name shows its backing
  as a tier; on the ETH market, with its Base account's added.
- **Channel hubs.** Hubs are listed most-backed first: a hub's adverts are its
  own claims, but its backing cost it something.
- **Mining pools.** A pool's dashboard shows its backing (purpose `pool`, its
  fee address), asked of its node every five minutes. Miners' own dashboards
  show it too.
- **The name market.** Backing goes with a name's registration, so a sale
  passes it to the buyer. Listings show their tier.
- **Leaderboards.** The explorer ranks names, pools, hubs, accounts and keys by
  backing.

Anyone can back anything. Backing is what someone burned for a subject, not the
subject's own statement. A pool that wants its backing to mean "we stand behind
this" burns it itself, and says so in the label.

## When a name changes hands

A name's backing counts the burns made for it since its **current
registration** began. A registration begins when someone claims a name no one
held (a new name, or a premium name's first claim), or when a key other than
its last holder's reclaims it after it lapsed. Nothing else ends one:

| what happens to the name | its backing |
|---|---|
| sold, transferred, its key rotated | kept: it goes with the name |
| rented out, renewed, renewed late by its holder | kept |
| lapsed, and claimed by another key | starts again from nothing |
| never registered, or a premium name never claimed | none counts |

So a lapsed name cannot be picked up, backing and all, for the price of its
registration: whoever takes it over starts from nothing. Burns for a name
before it is registered never count; back a name once it is held.

Why a sale keeps it: a sale off the market is a transfer its holder signs,
which the chain cannot tell from a key rotation (wallets rotate as their
one-time signatures run low), and resetting on market sales alone would only
push sales off the market. A lapse is the one change a holder did not agree
to.

The rule is the readers', not consensus. The index keeps every burn; it counts
a name's from the height its registration began (`registered` in
`GET /name/:name`), and `GET /burns/name/:name` says where backing counts from
(`counts_from`) and, for each burn, whether it `counts`.

## Notarizing

A `notary` burn puts a document's hash on chain: anyone can later show the
document existed by that block and has not changed. The document never leaves
the wallet; only its hash does. Two hashes are offered: BLAKE3, midstate's own,
and SHA-256, which every system has. A lookup checks both. The explorer hashes a
picked file in the browser and looks both up, and a search for a 64-hex hash
finds its notarizations. Notarizing proves existence by a time, not authorship.

## Node

`storage::burn_index` keeps every burn on the chain, always on, as the name
index does: its hooks run inside the write that stores or deletes a block, a
reorg rewinds before it applies, and rewinding reads only the index's own
per-block rows, so it unwinds exactly on a pruned node too. A worker backfills
from the lowest block the node holds; a node that pruned its early blocks
reports from where its totals run (`from`).

- `GET /burns`: totals; each year's issuance and burn by kind (names, backing
  by purpose, dex, other); the last 365 days; all the backing held, and the
  share of issuance keeping it up burns.
- `GET /burns/days?days=`: burned and issued by day.
- `GET /burns/top?purpose=&limit=`: the most-backed subjects, by the backing
  that counts.
- `GET /burns/recent?limit=`: the newest MSB-1 burns.
- `GET /burns/:purpose/:subject?limit=`: a subject's backing and its burns,
  newest first (a name, a `0x` account, a checksummed address or 64 hex):
  `counts_from`, the height its burns count from (`null`: none do), and
  each burn's `counts`.
- `GET /burns/notary/:hash`: a document hash under both notary purposes.
- `GET /burns/payload?purpose=&subject=&label=`: the bytes of an MSB-1 burn.
- `GET /name/:name` now includes the name's `backing`, and `registered`,
  the height its current registration began.

The index also counts every other burn: MSN-1 (`MSN1…` payloads), the DEX's
order books (`MDX…`, usually value 0) and anything else, including outputs paid
to the all-zero burn address. Burns of value 0 are not stored, except MSB-1's.

## Wallets

- **Desktop, the Burns page:** back a name, a Base account, a pool, a hub or
  any key, by an hour, a day or a week of issuance or an amount, with a label;
  a confirm says what it burns. The page shows this wallet's own names,
  account and hub with their backing, the chain's burn against its issuance,
  the most-backed names, pools and hubs, and recent burns. Drop a file on the
  window to notarize it or check it.
- **CLI:** `midstate wallet burn --purpose name --subject nick --blocks 1440
  --label "for the shop"` (or `--units`; `--file` for a notarization), and
  `midstate wallet backing --purpose name --subject nick`.

## Compatibility

No consensus rule changes. Nodes without the burn index accept and relay MSB-1
burns like any burn. Wallets without MSB-1 treat them as burns with a payload.
New wallet API fields are optional (`#[serde(default)]`).

## Frozen

Version 1 is final. These are fixed, and burns already on chain keep the
meaning they were made with:

- the tag `MSB1` and the layout: purpose at byte 4, subject at bytes 5–36,
  the label after, 80 bytes at most (a label of up to 43);
- purposes 1–7 and what each one's subject is, including the canonical-name
  and zero-padding rules;
- the weight: `v / block_reward(h)` blocks, halving every 525,600 blocks;
- the tiers: 60, 1,440 and 10,080 blocks, named backed, well backed and
  strongly backed;
- the name rule above.

What may change, and how:

- **New purposes** take the next number, 8 and up, written here first. A number
  is never reused or redefined, even for a use given up. Readers already keep
  and index purposes they do not know (shown as `purpose-N`), so burns for a
  new purpose count from the first one; readers need an upgrade only to name
  it.
- **Anything else**, a layout, a weight or a tier changed, is a new tag
  (`MSB2`), read beside MSB-1, never instead of it.
- Wallets may set bars of their own, as the inbox's are; the three tier names
  keep their values.

### Test vectors

`core::burns::tests::the_format_is_frozen` checks these.

```text
name "nick", label "for the shop":
  4d534231016e69636b00000000000000000000000000000000000000000000000000000000666f72207468652073686f70
evm 0x00112233445566778899aabbccddeeff00112233, no label:
  4d5342310300112233445566778899aabbccddeeff00112233000000000000000000000000
```

| | value |
|---|---|
| `block_reward(0)` | 1,073,741,824 units (2^30) |
| weight of `block_reward(0)` units at height 0 | 1 block |
| the same units at height 525,600 | 2 blocks (the reward halved) |
| a burn of 1 block, 525,600 blocks later | 0.5 blocks |
| a day of backing at height 0 (`cost(1440, 0)`) | 1,546,188,226,560 units |
| an hour at height 525,600 (`cost(60, 525600)`) | 32,212,254,720 units |
| keeping up `B` blocks a year | `B · ln 2` blocks |
