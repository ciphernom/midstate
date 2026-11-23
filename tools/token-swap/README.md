# MidstateTokenSwap: test and deploy

`MidstateTokenSwap.sol` (at the repository's root) is the USDC side of
MDS ↔ USDC trades: `MidstateAtomicSwap`'s protocol, rule for rule, for one
ERC-20 fixed at deployment. The design is in `docs/design/usdc-swaps.md`.

## Test

```sh
npm install
npm test
```

This compiles both swap contracts and a stand-in for USDC (`MockUSDC.sol`:
six decimals, EIP-2612 permits under USDC's domain, a blacklist, and two
misbehaviours to switch on), then runs the contract in an EVM (ethereumjs,
the Cancun rules Base runs, chain id 8453): every order shape and every
refusal, the permit forms (including a permit someone else used first),
tokens that charge a fee, return nothing or answer `false`, a beneficiary
the token will not pay, BLAKE3 against the reference implementation and
against `MidstateAtomicSwap`'s, and that both contracts' events and getters
read the same. It ends with the gas each call used.

## Deploy

The contract has no owner, no admin and nothing to set after deployment:
whoever deploys it holds no power over it. Its one constructor argument is
the token: USDC on Base, `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913`
(Base Sepolia: `0x036CbD53842c5426634e7929541eC2318f3dCF7e`).

```sh
# What it would do: the token checked, the gas estimated. Nothing is sent.
BASE_RPC=https://mainnet.base.org DEPLOYER_KEY=0x… npm run deploy

# Deploy, then read back token() and BLAKE3 from the deployed contract.
BASE_RPC=https://mainnet.base.org DEPLOYER_KEY=0x… npm run deploy -- --broadcast
```

About 2.5 million gas. Rehearse on Base Sepolia first
(`BASE_RPC=https://sepolia.base.org`), with test USDC from Circle's faucet.

To verify on Basescan: Solidity 0.8.26, optimizer on with 200 runs, EVM
version `cancun`, one file, constructor argument the token's address. Remix
or `forge create` work as well, with the same settings.

## Switch it on in the wallet

USDC trading stays off until the wallet knows the contract. Either:

* in the app, Trade → Connection → **USDC contract**, then *Save and
  rescan*; or
* for a release, set `BASE_MAINNET_TOKEN_CONTRACT` in
  `crates/walletd/src/evm.rs`, so every wallet has it.

Then the same contract, from the wallet's own code, on a real chain: a
permit checked by the token, a lock and its claim, a bid and its cancel
(two millionths of a dollar, all of it back):

```sh
DEVNET_RPC=https://sepolia.base.org DEVNET_KEY=0x… \
DEVNET_SWAP=0x… DEVNET_TOKEN=0x036CbD53842c5426634e7929541eC2318f3dCF7e \
cargo test -p midstate-walletd --test usdc_devnet -- --nocapture
```
