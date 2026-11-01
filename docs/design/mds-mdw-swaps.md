# Design: MDS ↔ MDW atomic swaps

**Status**: Implemented in walletd and the desktop Trade view. Not audited.
**Date**: 2026-11-02
**Related**: midwimble (`github.com/ciphernom/midwimble`), whose coin is MDW.

---

## 1. Summary

Two wallets trade MDS for MDW with no trusted party and **no change to either
chain's consensus**. Either both are paid or both get their coins back.

The obstacle is that the chains can check different things. Midstate's script
VM checks hash preimages; midwimble has no scripts, only kernels whose
signatures can carry an *adaptor*, which releases a scalar `t` to whoever holds
the pre-signature. A secret released on one chain must open a lock on the
other, so the secret takes both forms: a scalar `t` with point `T = t·G`, and
BLAKE3 preimages derived from it. A cut-and-choose proof (§3) shows,
off chain, that the two forms hide the same `t`.

| What the maker sells | Maker locks (first) | Taker locks (second) | Secret revealed on |
|---|---|---|---|
| MDS (`SellMds`) | MDS behind an **OR lock** | MDW in a 2-of-2 output | midwimble |
| MDW (`SellMdw`) | MDW in a 2-of-2 output | MDS behind an **AND lock** | midstate |

In both directions the maker holds `t`, locks first on the *follow leg*, and
reveals by claiming the taker's lock on the *reveal leg*. The taker then uses
what the claim published to claim the maker's lock.

## 2. Code

| Where | What |
|---|---|
| `src/core/script.rs` | `compile_mwb_or_lock`, `compile_mwb_and_lock` |
| `crates/walletd/src/mwb_link.rs` | Cut-and-choose prover and verifier; lock witnesses |
| `crates/walletd/src/mwb.rs` | The MDW account's derivation; the 2-of-2 lock and its signatures |
| `crates/walletd/src/mwb_wire.rs` | Chat-bus framing, session keys, messages |
| `crates/walletd/src/mwb_negotiate.rs` | The negotiation, both roles and directions |
| `crates/walletd/src/mwb_trade.rs` | Orders, the journal, `observe` and `decide` |
| `crates/walletd/src/mwb_account.rs` | Scanning midwimble for the account's coins |
| `crates/walletd/src/mwb_node.rs` | The embedded midwimble node |
| `crates/walletd/src/service_mwb.rs` | The actor's glue: bus, timers, broadcasts |
| `crates/walletd/tests/mwb_trade.rs` | Both directions and every refund path, end to end |

## 3. The link

The maker splits `t = a_j + b_j` at random for `j < k` and commits to each half
twice: `A_j = a_j·G` (so `B_j = T − A_j`) and the hashes `BLAKE3(a_j)`,
`BLAKE3(b_j)`. The taker challenges one half per instance; the maker opens it;
the taker checks the opened half against its point and its hash. The hashes of
the *unopened* halves build the midstate lock.

Whoever learns an unopened half `u_j` learns `t = o_j + u_j` from its opened
partner, and whoever learns `t` learns every `u_j = t − o_j`.

A cheating maker must leave every *bad* half unopened. Each lock is paired with
the direction in which a cheat needs *every* instance bad, so a cheat survives
with probability `2^-k`:

* **OR lock** (secret out on midwimble): the taker needs *one* good unopened
  hash to claim with. `k = 128`, so the challenge can be Fiat–Shamir.
* **AND lock** (secret out on midstate): the maker's claim publishes *every*
  unopened half, and the taker needs one consistent with its partner. The lock
  carries every hash on chain, so `k = 48` (midstate's stack allows 62), and a
  48-bit challenge could be ground: it is drawn by the taker after the
  commitment arrives.

On the wire the Fiat–Shamir proof is compressed (`LinkProof`): per instance the
opened half and the unopened hash; the verifier rebuilds `A_j` and the opened
hash from the half, recomputes the challenge from the rebuilt commitment, and
requires it to equal the one sent. 8,264 bytes at `k = 128`, not 16,416.

## 4. The locks

**Midstate.** Pay-to-script-hash, one coin per denomination of the amount.

```text
OR lock:   IF  SIZE 32 EQUALVERIFY HASH SWAP MERKLE_ROOT <root> EQUALVERIFY
               <claim_pk> CHECKSIGVERIFY
           ELSE <timeout> CHECKTIMEVERIFY <refund_pk> CHECKSIGVERIFY ENDIF 1
AND lock:  IF  HASH (SWAP HASH CONCAT HASH)×(k−1) <acc> EQUALVERIFY
               <claim_pk> CHECKSIGVERIFY
           ELSE <timeout> CHECKTIMEVERIFY <refund_pk> CHECKSIGVERIFY ENDIF 1
```

The OR lock's `SIZE 32` check stops an inner Merkle node (the hash of two
leaves the claimer knows) passing as a preimage. Claims check the claimer's
signature, so a public preimage lets nobody else take the coins. Each side's
key in the lock is a fresh WOTS key that signs exactly one message, the swap's
deterministic claim or refund (its salt and outputs derive from the swap id).

**Midwimble.** Spending is authorised by the output's owner key, so a 2-of-2
lock is an output whose owner key is `Ko_F + Ko_R`, each share with a proof of
knowledge (no rogue keys). The funder picks the blinding and proves the range
alone. One input signature (it signs only the commitment and owner key) serves
both spends. The funder's refund kernel has `min_height` = its refund height;
the receiver's claim kernel carries the adaptor in its owner nonce
(`R2 = R2_b + R2_o + T`), so publishing it publishes `t = s2 − s2_pre` to the
funder's pre-signature.

## 5. Deadlines

`reveal_height` is the reveal leg's refund height, on its chain;
`follow_height` the follow leg's. Planned by the maker from both tips
(`swap::plan_mwb_timings`, reveal leg 120 blocks) so that a reveal at the last
moment of a slow reveal leg still leaves an hour on a fast follow leg
(280 blocks). The taker re-checks them (`check_mwb_timings`) and refuses reveal
legs over 240 blocks.

* The maker reveals only while `reveal_height − tip ≥ 30` (`may_reveal`).
* The taker starts its lock only while `tip ≤ reveal_height − 50`; the maker
  starts its own only while the taker would still have 15 blocks.
* Each side builds on the other's lock once it is 2 blocks deep.
* A maker whose claim landed never refunds: the taker is owed its lock.

## 6. Messages

### Transport

Midstate's chat bus. A message's `words` are `[33, 255, kind]` ("swap ⚡ …");
its payload is the concatenation of up to four `Signature` attachments
(1,536 bytes each, the only attachment kind that takes arbitrary bytes and is
not filtered as graffiti). Longer payloads span up to eight chat messages,
each carrying its index and the count.

```text
kind 34  advert   VERSION | order_id[16] | side | mds u64 | mdw u64 | maker_key[32] | expires u64
kind 43  take     VERSION | order_id[16] | taker_key[32] | index | count | chunk
kind 32  session  VERSION | tag[16] | to (0 maker, 1 taker) | seq | index | count | chunk
```

All integers little-endian. `side` is 0 for `SellMds`, 1 for `SellMdw`.
Adverts repeat every 5 midstate blocks while the order stands, and a book
forgets an order not heard for 20.

### Sessions

`maker_key` is a Ristretto public key per order, `taker_key` one per take.
From `shared = DH(maker_key, taker_key)`:

```text
m       = shared ‖ order_id ‖ maker_key ‖ taker_key
key     = BLAKE3-derive("midstate.mwb-swap.session-key.v1", m)
tag     = BLAKE3-derive("midstate.mwb-swap.session-tag.v1", m)[..16]
swap_id = BLAKE3-derive("midstate.mwb-swap.swap-id.v1", m[32..])
sealed  = nonce[24] | XChaCha20-Poly1305(key, nonce, bincode(Msg), aad)
aad     = VERSION | kind | route | to | seq,   route = tag, or order_id ‖ taker_key
```

`swap_id` binds every proof of knowledge and the link's Fiat–Shamir challenge.
Messages are bincode (fixed-width integers, 40 KiB limit) of `mwb_wire::Msg`.

### Sequence

```text
 seq   selling MDS (taker funds MDW)          selling MDW (maker funds MDW)
  0    T→M  Take                              T→M  Take
  1    M→T  Offer: link proof, salts          M→T  Offer: link commitment, lock,
                                                          nonce commitments
  2    T→M  lock, nonce commitments           T→M  challenge, nonces, salts
  3    M→T  nonces                            M→T  opening, nonces, input partial,
                                                          refund plan
  4    T→M  nonces, input partial,            T→M  input and refund partials,
            refund plan                                claim plan
  5    M→T  input and refund partials,        M→T  claim partial      (done)
            claim plan, own claim partial
  6    T→M  claim partial            (done)
```

Each side's four nonces: the funder's are input (G), refund Pedersen (H),
refund owner as builder (G), claim owner as co-signer (G); the receiver's are
input (G), refund owner as co-signer (G), claim Pedersen (H), claim owner as
builder (G). The funder commits to its nonces before seeing the receiver's.

What each side checks before it signs:

| Check | Stops |
|---|---|
| Share proofs of knowledge | A rogue key making the 2-of-2 one party's |
| The lock's opening and owner key | A lock that is not the agreed 2-of-2, or holds less |
| Link proof or opening (`k` exact) | A midstate lock whose hashes have nothing to do with `T` |
| Nonces against their commitments | Choosing a nonce after seeing the other's |
| Refund plan's `min_height` | A refund the funder could take at once |
| Claim plan's adaptor point is `T` | A claim completed without revealing `t` |
| Every partial, then the whole | Holding a refund or pre-signature that does not verify |

### Reliability

Each side resends its last message every 30 s (up to 8 times) until it hears
the next; a repeat of the other side's previous message is answered by
resending ours. The side that sends the last message keeps the session for
10 minutes to answer a repeat. A negotiation that does not move for 5 minutes
is abandoned with an `Abort`; nothing is locked before both sides are done,
so abandoning costs nothing.

## 7. After the negotiation

The journal entry (`MwbSwap`, in `wallet.dat` with the rest of what spends)
holds everything either side needs; no more messages are exchanged. Every tick
the wallet reads both chains (`observe`) and takes one step (`decide`): lock,
claim, refund, or wait. Midstate spends go commit-then-reveal; the reveal is
rebuilt byte for byte from the journal.

## 8. Limits and open points

* **Orders are unauthenticated**, like hub adverts. A forged advert reaches
  only a taker whose take the real maker refuses, or a "maker" that must lock
  first. Free options remain: a taker can walk away after the maker locks,
  costing the maker its lock for ~5 hours.
* **The bus loses messages.** Each node keeps its last 100 chats and dedups
  by proof-of-work nonce alone, so a different chat with a colliding nonce
  (about 1 in 400 at 5,000 recent chats) is dropped. Resends cover it.
* **Nonces** are fresh OS randomness, committed before reveal. Deterministic
  or hedged nonces, or MuSig2, would remove the dependence on the RNG.
* **Reorgs** deeper than 2 blocks after a side has built on the other's lock
  are not handled beyond rebroadcasting.
* **Review.** The link, the adaptor construction and the deadlines need an
  independent cryptographic review before real value relies on them.
