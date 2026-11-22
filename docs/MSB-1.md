# MSB-1: burns that back things

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
- **Sealed messages.** The inbox has a bar: everyone, verified names, names
  backed an hour or more, or a day or more. Messages below it wait under
  Requests, never dropped. Conversations show their sender's tier.
- **The ETH and MDW swap markets.** A seller's verified name shows its backing
  as a tier; on the ETH market, with its Base account's added.
- **Channel hubs.** Hubs are listed most-backed first: a hub's adverts are its
  own claims, but its backing cost it something.
- **Mining pools.** A pool's dashboard shows its backing (purpose `pool`, its
  fee address), asked of its node every five minutes. Miners' own dashboards
  show it too.
- **The name market.** Backing belongs to the name, so it goes to the name's
  next holder. Listings show their tier.
- **Leaderboards.** The explorer ranks names, pools, hubs, accounts and keys by
  backing.

Anyone can back anything. Backing is what someone burned for a subject, not the
subject's own statement. A pool that wants its backing to mean "we stand behind
this" burns it itself, and says so in the label.

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
- `GET /burns/top?purpose=&limit=`: the most-backed subjects.
- `GET /burns/recent?limit=`: the newest MSB-1 burns.
- `GET /burns/:purpose/:subject?limit=`: a subject's backing and its burns,
  newest first (a name, a `0x` account, a checksummed address or 64 hex).
- `GET /burns/notary/:hash`: a document hash under both notary purposes.
- `GET /burns/payload?purpose=&subject=&label=`: the bytes of an MSB-1 burn.
- `GET /name/:name` now includes the name's `backing`.

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
