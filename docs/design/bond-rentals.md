# Bond rentals

A midwimble mining bond is a midstate coin locked by the bond script. The
script names the mining key that signs blocks with it and the owner key that
spends it back after its lock height. Bond rentals let someone with coins
rent a bond to someone with hashrate, from the desktop wallet's Bonds tab
("Bonds for rent"). No consensus rule changed on either chain for this: the
offer is an ordinary MidstateScript covenant, and midwimble sees an ordinary
bond.

## Why a bond can be rented safely

The bond script already splits the two rights a rental needs to separate:

- **The coin stays the owner's.** Only the owner key can spend it, and only
  after the lock height. Midwimble has no slashing, so nothing the renter
  does with the bond costs the owner anything.
- **The mining right stays the renter's.** The mining key is part of the
  bond's address, so the owner cannot change it or take the bond back before
  the lock ends. Midwimble's retargets apply only to new registrations.

The only thing left to trust is the rent. The offer covenant removes that.

## The offer

An owner locks one coin at the covenant
`core::script::compile_bond_rental_offer` (source in the IDE language:
`docs/contracts/bond-rental-offer.mds`). It has two branches:

- **Take: `[mining_key, 01]`, no signature.** The script checks that the key
  is exactly 32 bytes and builds the bond script around it, byte for byte as
  midwimble does. It hashes that into the bond's address and then requires
  that the transaction:
  - locks every coin of this offer it spends at that address
    (`OP_SUM_TO_ADDR ≥ OP_SUM_INPUT_VALUE`), and
  - pays the rent for each bond's worth to the owner.

  The bond's owner key is the offer's, so the coin goes back to the owner at
  the lock height whoever takes it.
- **Cancel: `[owner_sig, 00]`.** The owner key alone.

The offer is announced as a zero-value `DataBurn` in the transaction that
funds it, as DEX asks are. The announcement holds the tag `MDBR`, the
version, log2 of the bond, the lock height, the rent and the owner key, 54
bytes in all. Any wallet reading the chain:

1. finds the announcement,
2. rebuilds the covenant from it,
3. finds the coin beside it in the same transaction, and
4. checks that the coin is unspent.

An announcement that doesn't sit beside its own coin is ignored.

### What the covenant has to get right

Tests in `core::rental` and `core::script` cover each of these:

- **The key must be exactly 32 bytes.** Any longer and the extra bytes run
  as opcodes inside the bond script. A taker could then make the coin
  unspendable for the price of the rent, and against some owner keys take it
  outright.
- **The rules use `OP_SUM_INPUT_VALUE`,** so two coins of one offer cannot
  be swept into one bond.
- **Every offer has its own owner key, and the rent goes to that key's own
  address.** If two offers shared a key, one bond and one payment would
  satisfy both, and the taker would keep a coin.

## Who signs what

Only one side signs any transaction:

- the owner signs the funding send and the cancel;
- the renter signs the take, with its own coins paying the rent and the fee.

Nobody signs first and waits on the other. So a counterparty can never
leave one-time (WOTS) keys spent on a transaction that never happens, and
the owner need not be online when the bond is rented.

Two renters can go for one offer at once, and only one of them gets it.
Before it signs, the losing wallet checks that the offer is still there, so
it usually signs nothing. The take prefers MSS coins, whose keys can sign
again, so a loss after signing costs one MSS leaf rather than one-time keys.

## In the desktop wallet

### Renting a bond out

1. Choose the days of mining and the rent.
2. The wallet makes a fresh MSS tree (height 5) for the offer, then sends
   one coin of midwimble's current registration minimum to the covenant,
   with the announcement beside it.

When someone rents it:

- the wallet finds the take on chain;
- it records the bond (this wallet's coin, the renter's mining key), which
  unbonds like any other once the lock ends;
- the rent arrives as ordinary coins at the tree's address.

An offer can be cancelled until someone rents it.

### Renting a bond

The wallet reads offers announced over the last ten weeks and lists those
whose coin is unspent, cheapest per day of mining first. Renting one is a
single send. The new bond names the wallet's own mining key, and the wallet
registers it with midwimble once it confirms. After that the miner signs
with it like any other bond.

### Lock length

A bond counts for mining while more than 30 days of its lock remain, by a
clock that runs 7 days ahead. So an offer of `d` days of mining locks the
coin for about `d + 38` days: a day to register, the `d` days, then about
five weeks.

### Records

The records live beside the wallet in `<wallet>.rentals`. After a restore,
the wallet adopts its own offers it finds announced in the window it reads.
