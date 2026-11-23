# MDS ↔ USDC on Base

Status: deployed on Base at `0x34214d2F2aE29Bd73a735D63ad5D857cE200B567`, verified on Basescan.

Stablecoins are what most people trading on Base expect to pay with. MDS
trades for USDC the way it trades for ETH: wallet to wallet, atomically,
with no one in the middle. Nothing about the protocol changes; the Base side
is paid in a token instead of ether.

## The contract

`MidstateTokenSwap.sol` is `MidstateAtomicSwap.sol` for one ERC-20, fixed
when it is deployed (USDC). The same two order shapes:

* **lock / claim / refund**: a buyer takes a sell order; the seller locks MDS
  on midstate, the buyer locks USDC here against the same hash, the seller
  claims it (publishing the secret), the buyer claims the MDS with it.
* **createBid / reserveBid / claimBid / cancelBid**: a resting buy order,
  USDC escrowed up front; a seller reserves it, locks MDS to the bidder,
  the bidder claims (publishing the secret on midstate), the seller collects.

The events, and the `swaps` and `bids` getters, are `MidstateAtomicSwap`'s
exactly, so one reader serves both contracts; a wallet tells them apart by
the address that emitted them. The BLAKE3 code is the same.

What tokens force:

* **Paying in.** Tokens are pulled with `transferFrom`. Each paying call has
  a `...WithPermit` form that takes an EIP-2612 permit for exactly the
  amount, so a payment is one transaction, and no standing approval is left
  behind. A permit copied out of the mempool and used first has already set
  the allowance, so a failed permit is passed over and the transfer decides.
* **The fill bond is in USDC**, so it is weighed against the escrow like for
  like (the wallet's one-tenth rule). Bids that ask no bond can be filled by
  a seller holding no USDC.
* **A token that delivers less than asked is refused** (a fee on transfer),
  and one that answers `false` is a failed transfer, not a payment.

It has no owner and nothing to configure. A blacklisted beneficiary cannot
be paid, so its claim fails; the locker takes the USDC back after the
timeout, as with any lock never claimed.

## On the bus

A USDC advert or take is the ETH message with a fourth word, `fiat`:

```text
[dex, ⚡, sell, fiat]   [dex, ⚡, take, fiat]
```

Wallets that predate USDC read three-word messages only, so they never see
an order they cannot settle, and a take cannot be read as one for the other
currency's order: a seller declines a take whose currency is not its
order's. Amounts in a USDC message are millionths of a dollar.

## In the wallet

Every trade, order, bid, lock and book entry carries its `Asset`, and every
amount in it is that asset's smallest unit. The two contracts are read into
one index, each lock tagged with the contract it came from, and a lock on
the other contract is never the lock a trade waits for, whatever its
numbers say: five million wei is not five dollars. Each act goes to its
trade's contract.

A permit takes the token's next nonce, and the token checks nonces in
order, so the wallet signs one USDC payment at a time: a second waits until
the first is mined (a tick or two), rather than sign a permit that could
only fail.

The Trade page has a market for each currency; the USDC one opens once the
contract's address is set (Connection), and shows the account's USDC beside
its ETH. USDC trades still need a little ETH for gas.

## Tests

* `tools/token-swap`: the contract in an EVM against a stand-in for USDC
  (126 checks).
* `crates/walletd/tests/eth_trade.rs`: both sides of USDC trades through the
  trade machine against simulated chains, and the cross-currency attacks
  (an ETH lock with a USDC trade's numbers, and the reverse).
* `crates/walletd/tests/usdc_devnet.rs`: the wallet's own Base code against
  a deployed contract on a real chain (skipped unless pointed at one).
* Unit tests: permits match ethers' `signTypedData`, and every call's
  encoding matches Solidity's.
