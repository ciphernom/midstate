# Inheritance plans and payment streams

Two covenants the desktop wallet offers on its **Plans** tab. Both run on
opcodes midstate already has (`CHECKSIG`, `CHECKTIMEVERIFY`, `SUM_TO_ADDR`,
`SUM_INPUT_VALUE`, `DIV`, `MUL`, `IF`/`ELSE`): no consensus rule changes, on
either chain. `SUM_INPUT_VALUE` makes them available from midstate block
371,500 (`core::plans::ACTIVATION_HEIGHT`).

Everything runs between wallets. There is no server, no operator and no web
page: people swap codes, and the chain's scripts do the rest. Nobody holds
anyone's coins. A code names coins and lets a wallet find them; it never lets
anyone spend them.

| Where | What |
|---|---|
| `src/core/plans.rs` | The scripts, the payout addresses, salts, encodings |
| `src/core/plans/tests.rs` | Every path through consensus, and the spends each refuses |
| `crates/walletd/src/plans.rs` | Codes, finding coins on chain, building spends, records |
| `crates/walletd/src/service_plans.rs` | The actor: sends, reveals, watching the chain |
| `crates/walletd/tests/plans.rs` | Plans and streams end to end, signed by real wallets |
| `apps/desktop/src/views/plans.rs` | The Plans tab |

## Inheritance plans

**What it does.** Your coins sit at a plan address. You spend them with your
own key whenever you like. If you leave them untouched past the plan's
deadline, anyone may move them, but only by paying each heir its share.
Checking in is one transaction that moves the coins to the next deadline.

**The script** (deadline `k`):

```text
IF    PUSH owner_pk CHECKSIG                  witness [sig, 1]
ELSE  PUSH deadline_k CHECKTIMEVERIFY         witness [0]
      SUM_INPUT_VALUE PUSH 65536 DIV          unit = what this claim moves / 65,536
      per heir i:
        DUP PUSH share_i MUL                  owed_i
        PUSH payout_i SUM_TO_ADDR             paid_i
        SWAP GREATER_OR_EQUAL VERIFY          paid_i ≥ owed_i
      DROP PUSH 1
ENDIF
```

**Deadlines follow a fixed schedule:** `start + k·period`. A check-in at height
`h` moves the coins to the first deadline at least one period away, so a
check-in always buys at least a full period (and less than two). Because the
schedule is fixed, an heir's wallet can find the coins at whichever deadline
they sit, from the plan code alone.

**Shares** are counted in parts of 65,536 and add up to exactly 65,535. The
last part is the fee: a claim needs no signature, so the script cannot be told
what fee the transaction needs. It keeps back one part in 65,536, plus the
rounding. If the shares added up to less, the rest would belong to whoever
built the claim, so the wallet refuses such plans.

**Who can do what:**

| | Before the deadline | From the deadline |
|---|---|---|
| Owner (signature) | spend, check in, close | the same, until a claim confirms |
| Anyone (no key) | nothing | pay every heir its share |
| An heir (its key) | nothing | sweep its own payout |

## Payment streams

**What it does.** A payer locks a run of payments at once, each at its own
address. Payment `j` becomes payable from `first + j·period`. From then anyone
may move it, but only by paying all of it to the recipient, bar the 1/65,536
fee. A *cancellable* stream also lets the payer take back any payment that has
not been collected.

**The script** (payment `j`):

```text
IF    PUSH payer_pk CHECKSIG                       witness [sig, 1]    (cancellable only)
ELSE  PUSH due_j CHECKTIMEVERIFY                   witness [0], or [] when not cancellable
      SUM_INPUT_VALUE PUSH 65536 DIV PUSH 65535 MUL
      PUSH payout_j SUM_TO_ADDR SWAP GREATER_OR_EQUAL
ENDIF
```

**What the payer can still do.** Script can say "at or after height N", never
"before". So the payer's path stays open after a payment falls due: until it
is collected, a cancellable stream's payer *can* take back a payment that is
already due. The recipient's wallet therefore collects each payment as soon as
it falls due, and anyone else may collect it for them. The wallet never takes
back a due payment: its cancel only touches payments still in the future, with
ten blocks to spare. A stream made without cancellation has no payer path at
all: once funded, nothing takes it back.

## Why every payout goes to an address of its own

`SUM_INPUT_VALUE` totals only the inputs that share the running script.
`SUM_TO_ADDR` totals every output to an address. Suppose two covenants paid the
same address: two deadlines of one plan, two plans with the same heirs, or two
payments of one stream. Then one transaction could spend both and satisfy each
check with the same outputs, and whoever built it would keep the smaller one.

So heirs and recipients are never paid at a plain address. They are paid at a
**payout address**:

```text
PUSH tag DROP PUSH pk CHECKSIG
```

The tag is unique to the plan, the deadline and the heir, or to the stream and
the payment. No two covenants share one, and the tests try the skim both ways.
The heir or recipient then **sweeps** its payouts into ordinary coins with its
key. The wallet does that in batches: one signature covers every coin a sweep
takes. The coins are safe where they are, since only that key can spend them.

## Finding coins without scanning

Coin salts are derived from the plan's or stream's secret:
`hash(domain ‖ secret ‖ k ‖ n)`. With the code, a wallet computes every coin id
the plan could have and checks the UTXO set directly. That is up to eight
coins per deadline, at most a thousand deadlines, one of 39 values each. It
works years later, after a restore, with no scan.

A stream's coin values are fixed by its amount, so its ids are exact.

Blocks are read only to find payouts made by **someone else's** claim or
collection. Their salts are the builder's choice and appear in the block that
spends the covenant coins. The wallet remembers what it last saw on chain.
When coins go without its doing, it searches forward from that height, a
budget of blocks at a time.

## Codes

| Code | Who makes it | Holds |
|---|---|---|
| `mdpayee1…` | an heir or a recipient | a new MSS key of its wallet, and a name |
| `mdplan1…` | the owner | the plan: owner key, heirs and shares, schedule, secret, heir names |
| `mdstream1…` | the payer | the stream: keys, schedule, amount, secret, a label |

Each code ends in a 4-byte checksum. Every payee code is a **new** key, so each
plan or stream paying a wallet has a key of its own, with 1,024 signatures for
its sweeps. Owner and payer keys have 256 (one per check-in, close or cancel).
All keys are MSS trees derived from the recovery phrase.

**Recovery.** The parameters are not derivable from the phrase. A wallet
restored from its phrase gets its keys back, and pasting the plan or stream
code on the Codes tab brings the plan or stream back with them. Owners should
keep their plan code, and give one to every heir: the heirs hold copies too.

## The wallet's part

- **Funding** a plan or a stream is an ordinary send of the wallet's coins.
  A plan holds at most eight coins of at least 2²⁵ units. The rest of the
  amount stays in the balance, which also bounds the fee a third party could
  take by claiming coin by coin.
- **Every other move** spends covenant coins only, with the fee taken from
  them. The moves are check-in, close, claim, cancel, collection and sweep.
  Each is recorded in full before its commit leaves: the inputs, their
  witnesses, and the signature once made. So it reveals the same way after a
  restart. A signature is kept, so a retried reveal sends the same one rather
  than spending another leaf.
- **Confirmation** is the send's own output appearing on chain. If another
  party spends the same coins first (a claim racing a late check-in, a
  collection racing a cancel), the send is dropped with nothing signed when
  it can be. If it was already signed, it is dropped once its coins are gone.
- **Collecting** happens automatically, once a block, for streams paying the
  wallet (it can be turned off per stream). **Claiming** a plan is a button,
  with a confirmation. **Sweeping** waits until eight payouts are waiting or
  the oldest is six hours old, and can be done by hand.
- **Reminders.** A plan within a tenth of its period of the deadline shows
  "check in soon"; one past it shows "past its deadline". The wallet can only
  remind while it is open.

## Trust and limits

- **The owner must check in.** A missed deadline lets anyone pay the heirs,
  and that cannot be undone once it confirms. Between the deadline and a
  claim, the owner can still check in: the first to confirm wins.
- **Heirs see the whole plan.** A plan code reveals every heir's key and
  share, and the plan's addresses, to whoever holds it. It reveals nothing
  that spends.
- **Shares and heirs are fixed** once a plan is made. Changing them means
  closing the plan and making a new one.
- **A cancellable stream is not a promise:** the payer can stop it, and can
  take back due payments the recipient's wallet has not collected. A
  non-cancellable stream is a promise, for exactly what was locked.
- **Fees.** Keyless moves pay 1/65,536 of what they move, plus rounding.
  Signed moves pay the relay rate, with room to spare.
- **Pruned nodes.** A payout made by someone else's claim is found by reading
  the block that made it. A node that pruned that block cannot find it. The
  coins stay safe at the payout address, but the wallet needs the block, or
  another node, to learn their salts.

Not legal or tax advice. An inheritance plan moves coins on a schedule; it is
not a will, and it does not settle who may own them.
