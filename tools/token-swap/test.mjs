// MidstateTokenSwap, run in an EVM (ethereumjs, the Cancun rules Base runs)
// against a stand-in for USDC: every order shape, every refusal, the permit
// forms, misbehaving tokens, and BLAKE3 against the reference.
//
//   npm install && npm test

import fs from "node:fs";
import assert from "node:assert/strict";
import { createEVM } from "@ethereumjs/evm";
import { createBlock } from "@ethereumjs/block";
import { createCustomCommon, Mainnet, Hardfork } from "@ethereumjs/common";
import { createAddressFromString } from "@ethereumjs/util";
import { blake3 } from "@noble/hashes/blake3.js";
import { ethers } from "ethers";

const CHAIN_ID = 8453n; // Base
const common = createCustomCommon({ chainId: Number(CHAIN_ID), name: "base" }, Mainnet, { hardfork: Hardfork.Cancun });
const evm = await createEVM({ common });

const load = (n) => JSON.parse(fs.readFileSync(new URL(`build/${n}.json`, import.meta.url), "utf8"));
const SWAP = load("MidstateTokenSwap");
const V1 = load("MidstateAtomicSwap");
const USDC = load("MockUSDC");
const FALSE = load("FalseToken");
const iSwap = new ethers.Interface(SWAP.abi);
const iV1 = new ethers.Interface(V1.abi);
const iUsdc = new ethers.Interface(USDC.abi);

let now = 1_750_000_000;
let height = 30_000_000;
const block = () => createBlock({ header: { timestamp: BigInt(now), number: BigInt(height), gasLimit: 60_000_000n } }, { common });
const addr = (a) => createAddressFromString(a);
const hex = (b) => ethers.hexlify(b);

const gas = {};
let passed = 0;

async function deploy(art, args = [], iface = null, from = DEPLOYER) {
  const data = art.bytecode + (iface ? iface.encodeDeploy(args).slice(2) : "");
  const r = await evm.runCall({ caller: addr(from), data: ethers.getBytes("0x" + data.replace(/^0x/, "")), block: block(), gasLimit: 30_000_000n });
  if (r.execResult.exceptionError) {
    const why = revertReason(r.execResult.returnValue);
    throw new Error(`deploy failed: ${why ?? r.execResult.exceptionError.error}`);
  }
  return r.createdAddress.toString();
}

function revertReason(ret) {
  if (!ret || ret.length < 4) return null;
  try {
    return ethers.AbiCoder.defaultAbiCoder().decode(["string"], ethers.dataSlice(ret, 4))[0];
  } catch {
    return null;
  }
}

/// A call from `from`; returns { ok, ret, logs, reason, gas }.
async function call(to, iface, fn, args, from, value = 0n) {
  const data = iface.encodeFunctionData(fn, args);
  const r = await evm.runCall({ caller: addr(from), origin: addr(from), to: addr(to), data: ethers.getBytes(data), value, block: block(), gasLimit: 10_000_000n });
  const e = r.execResult;
  const ok = !e.exceptionError;
  const out = { ok, ret: hex(e.returnValue), logs: e.logs ?? [], reason: ok ? null : revertReason(e.returnValue) ?? e.exceptionError.error, gas: Number(e.executionGasUsed) + 21_000 };
  if (ok && iface === iSwap) gas[fn] = Math.max(gas[fn] ?? 0, out.gas);
  return out;
}

async function view(to, iface, fn, args = []) {
  const r = await call(to, iface, fn, args, DEPLOYER);
  assert.ok(r.ok, `${fn} reverted: ${r.reason}`);
  return iface.decodeFunctionResult(fn, r.ret);
}

async function must(r, what) {
  assert.ok(r.ok, `${what} reverted: ${r.reason}`);
  passed++;
  return r;
}

async function refuses(r, reason, what) {
  assert.ok(!r.ok, `${what} should have been refused`);
  if (reason) assert.equal(r.reason, reason, `${what}: refused for "${r.reason}", not "${reason}"`);
  passed++;
}

function events(r, iface) {
  return r.logs.map(([address, topics, data]) => ({ address: hex(address), ...iface.parseLog({ topics: topics.map(hex), data: hex(data) }) }));
}

const wallet = (n) => new ethers.Wallet(ethers.zeroPadValue(ethers.toBeHex(n), 32));
const DEPLOYER = wallet(1).address;
const maker = wallet(2); // sells MDS, receives tokens; or makes a bid
const taker = wallet(3); // locks tokens; or fills a bid
const stranger = wallet(4);

const secret = () => ethers.randomBytes(32);
const H = (s) => hex(blake3(s));
const usd = (n) => BigInt(Math.round(n * 1e6));

async function balance(token, who) {
  return (await view(token, iUsdc, "balanceOf", [who]))[0];
}

async function permit(token, owner, spender, value, deadline = BigInt(now + 3600)) {
  const nonce = (await view(token, iUsdc, "nonces", [owner.address]))[0];
  const domain = { name: "USD Coin", version: "2", chainId: CHAIN_ID, verifyingContract: token };
  const types = { Permit: [
    { name: "owner", type: "address" },
    { name: "spender", type: "address" },
    { name: "value", type: "uint256" },
    { name: "nonce", type: "uint256" },
    { name: "deadline", type: "uint256" },
  ] };
  const sig = ethers.Signature.from(await owner.signTypedData(domain, types, { owner: owner.address, spender, value, nonce, deadline }));
  // The domain this test signs under is the one the token checks.
  const sep = (await view(token, iUsdc, "DOMAIN_SEPARATOR"))[0];
  assert.equal(sep, ethers.TypedDataEncoder.hashDomain(domain));
  return { deadline, v: sig.v, r: sig.r, s: sig.s };
}

// ── Deploy ──────────────────────────────────────────────────────────────────

const usdc = await deploy(USDC);
const swap = await deploy(SWAP, [usdc], iSwap);
assert.equal((await view(swap, iSwap, "token"))[0], ethers.getAddress(usdc));
await assert.rejects(deploy(SWAP, [stranger.address], iSwap), /not a token/);
passed++;
for (const w of [maker, taker, stranger]) await must(await call(usdc, iUsdc, "mint", [w.address, usd(1_000_000)], DEPLOYER), "mint");

// ── One reader for both contracts ───────────────────────────────────────────

for (const ev of ["Locked", "Claimed", "Refunded", "BidCreated", "BidReserved", "BidClaimed", "BidCancelled"]) {
  assert.equal(iSwap.getEvent(ev).topicHash, iV1.getEvent(ev).topicHash, `${ev} is not MidstateAtomicSwap's`);
  passed++;
}
for (const fn of ["swaps", "bids", "claim", "refund", "claimBid", "cancelBid", "blake3_256"]) {
  assert.equal(iSwap.getFunction(fn).selector, iV1.getFunction(fn).selector, `${fn} differs`);
  assert.deepEqual(iSwap.getFunction(fn).outputs.map((o) => o.type), iV1.getFunction(fn).outputs.map((o) => o.type), `${fn}'s outputs differ`);
  passed++;
}

// ── BLAKE3, as midstate hashes ──────────────────────────────────────────────

const v1 = await deploy(V1);
for (let i = 0; i < 24; i++) {
  const s = i === 0 ? new Uint8Array(32) : i === 1 ? new Uint8Array(32).fill(0xff) : secret();
  const want = H(s);
  assert.equal((await view(swap, iSwap, "blake3_256", [s]))[0], want, "BLAKE3 differs from the reference");
  assert.equal((await view(v1, iV1, "blake3_256", [s]))[0], want, "V1's BLAKE3 differs");
  passed++;
}

// ── Sell orders: the taker locks tokens ─────────────────────────────────────

{
  const s = secret();
  const amount = usd(1234.56);
  // No allowance, no lock.
  await refuses(await call(swap, iSwap, "lock", [H(s), maker.address, amount, 3600], taker.address), "token transfer failed", "lock without approval");
  await must(await call(usdc, iUsdc, "approve", [swap, amount], taker.address), "approve");
  for (const [args, why] of [
    [[H(s), maker.address, 0, 3600], "no value"],
    [[H(s), ethers.ZeroAddress, amount, 3600], "bad beneficiary"],
    [[ethers.ZeroHash, maker.address, amount, 3600], "bad hashlock"],
    [[H(s), maker.address, amount, 599], "bad delay"],
    [[H(s), maker.address, amount, 7 * 86400 + 1], "bad delay"],
  ]) await refuses(await call(swap, iSwap, "lock", args, taker.address), why, `lock ${why}`);

  const before = await balance(usdc, taker.address);
  const r = await must(await call(swap, iSwap, "lock", [H(s), maker.address, amount, 3600], taker.address), "lock");
  const ev = events(r, iSwap).filter((e) => e.address.toLowerCase() === swap.toLowerCase());
  assert.equal(ev.length, 1);
  assert.equal(ev[0].name, "Locked");
  const id = ev[0].args.swapId;
  assert.equal(ev[0].args.beneficiary, maker.address);
  assert.equal(ev[0].args.refundTo, taker.address);
  assert.equal(ev[0].args.amount, amount);
  assert.equal(ev[0].args.timeout, BigInt(now + 3600));
  assert.equal(ev[0].args.hashlock, H(s));
  assert.equal(before - (await balance(usdc, taker.address)), amount);
  assert.equal(await balance(usdc, swap), amount);
  // The id is MidstateAtomicSwap's formula, with the amount for msg.value.
  const want = ethers.keccak256(ethers.AbiCoder.defaultAbiCoder().encode(
    ["address", "address", "bytes32", "uint256", "uint64", "address", "uint256"],
    [taker.address, maker.address, H(s), amount, now + 3600, swap, CHAIN_ID]));
  assert.equal(id, want);
  const st = await view(swap, iSwap, "swaps", [id]);
  assert.deepEqual([st.beneficiary, st.refundTo, st.amount, st.timeout, st.hashlock, st.settled], [maker.address, taker.address, amount, BigInt(now + 3600), H(s), false]);

  // The same lock again in the same second is the same id.
  await must(await call(usdc, iUsdc, "approve", [swap, amount], taker.address), "approve");
  await refuses(await call(swap, iSwap, "lock", [H(s), maker.address, amount, 3600], taker.address), "swap exists", "a lock twice");

  await refuses(await call(swap, iSwap, "claim", [id, secret()], maker.address), "bad preimage", "claim with a wrong secret");
  await refuses(await call(swap, iSwap, "refund", [id], taker.address), "too early", "refund before the timeout");
  // Anyone holding the secret may claim; the tokens go to the beneficiary.
  const m0 = await balance(usdc, maker.address);
  const c = await must(await call(swap, iSwap, "claim", [id, s], stranger.address), "claim by a stranger");
  const ce = events(c, iSwap).find((e) => e.name === "Claimed");
  assert.equal(ce.args.preimage, hex(s), "the claim publishes the secret");
  assert.equal((await balance(usdc, maker.address)) - m0, amount);
  assert.equal(await balance(usdc, swap), 0n);
  await refuses(await call(swap, iSwap, "claim", [id, s], maker.address), "settled", "a second claim");
  now += 3600;
  await refuses(await call(swap, iSwap, "refund", [id], taker.address), "settled", "a refund after the claim");
  await refuses(await call(swap, iSwap, "claim", [ethers.ZeroHash, s], maker.address), "not found", "claim of nothing");
}

{
  // A lock nobody claims comes back after its timeout, to the locker only.
  const s = secret();
  const amount = usd(50);
  await must(await call(usdc, iUsdc, "approve", [swap, amount], taker.address), "approve");
  const r = await must(await call(swap, iSwap, "lock", [H(s), maker.address, amount, 600], taker.address), "lock");
  const id = events(r, iSwap).find((e) => e.name === "Locked").args.swapId;
  now += 599;
  await refuses(await call(swap, iSwap, "refund", [id], taker.address), "too early", "refund a second early");
  now += 1;
  const t0 = await balance(usdc, taker.address);
  const rf = await must(await call(swap, iSwap, "refund", [id], stranger.address), "refund (sent by anyone, paid to the locker)");
  assert.ok(events(rf, iSwap).some((e) => e.name === "Refunded"));
  assert.equal((await balance(usdc, taker.address)) - t0, amount);
  await refuses(await call(swap, iSwap, "claim", [id, s], maker.address), "settled", "claim after the refund");
}

{
  // One transaction with a permit, and a permit someone else used first.
  const s = secret();
  const amount = usd(77.7);
  const p = await permit(usdc, taker, swap, amount);
  const r = await must(await call(swap, iSwap, "lockWithPermit", [H(s), maker.address, amount, 3600, p.deadline, p.v, p.r, p.s], taker.address), "lockWithPermit");
  assert.ok(events(r, iSwap).some((e) => e.name === "Locked"));
  assert.equal((await view(usdc, iUsdc, "allowance", [taker.address, swap]))[0], 0n, "the permit covers exactly the lock");

  const s2 = secret();
  const p2 = await permit(usdc, taker, swap, amount);
  await must(await call(usdc, iUsdc, "permit", [taker.address, swap, amount, p2.deadline, p2.v, p2.r, p2.s], stranger.address), "a griefer uses the permit first");
  await must(await call(swap, iSwap, "lockWithPermit", [H(s2), maker.address, amount, 3600, p2.deadline, p2.v, p2.r, p2.s], taker.address), "the lock still goes");

  // A permit signed by someone else moves nothing of theirs, or the caller's.
  const p3 = await permit(usdc, stranger, swap, amount);
  await refuses(await call(swap, iSwap, "lockWithPermit", [H(secret()), maker.address, amount, 3600, p3.deadline, p3.v, p3.r, p3.s], taker.address), "token transfer failed", "lock on someone else's permit");
  // An expired permit is no allowance.
  const p4 = await permit(usdc, taker, swap, amount, BigInt(now - 1));
  await refuses(await call(swap, iSwap, "lockWithPermit", [H(secret()), maker.address, amount, 3600, p4.deadline, p4.v, p4.r, p4.s], taker.address), "token transfer failed", "lock on an expired permit");
}

// ── Resting bids ────────────────────────────────────────────────────────────

const mdsAddr = hex(ethers.randomBytes(32));

async function newBid(s, amount, bond, ttl = 86400, mds = 5_000_000) {
  await must(await call(usdc, iUsdc, "approve", [swap, amount], maker.address), "approve the escrow");
  const r = await must(await call(swap, iSwap, "createBid", [H(s), amount, mds, mdsAddr, ttl, bond], maker.address), "createBid");
  const ev = events(r, iSwap).find((e) => e.name === "BidCreated");
  assert.equal(ev.args.maker, maker.address);
  assert.equal(ev.args.amount, amount);
  assert.equal(ev.args.fillBond, bond);
  assert.equal(ev.args.mdsAmount, BigInt(mds));
  assert.equal(ev.args.makerMdsAddr, mdsAddr);
  assert.equal(ev.args.expiry, BigInt(now + ttl));
  return ev.args.bidId;
}

{
  // A bid filled: reserved with a bond, claimed with the secret the maker
  // published on midstate; escrow and bond to the filler.
  const s = secret();
  const amount = usd(2500);
  const bond = usd(25);
  for (const [args, why] of [
    [[H(s), 0, 1, mdsAddr, 3600, bond], "no value"],
    [[H(s), amount, 0, mdsAddr, 3600, bond], "no mds amount"],
    [[ethers.ZeroHash, amount, 1, mdsAddr, 3600, bond], "bad hashlock"],
    [[H(s), amount, 1, ethers.ZeroHash, 3600, bond], "bad mds addr"],
    [[H(s), amount, 1, mdsAddr, 3599, bond], "bad ttl"],
    [[H(s), amount, 1, mdsAddr, 90 * 86400 + 1, bond], "bad ttl"],
  ]) await refuses(await call(swap, iSwap, "createBid", args, maker.address), why, `createBid ${why}`);
  const m0 = await balance(usdc, maker.address);
  const id = await newBid(s, amount, bond);
  assert.equal(m0 - (await balance(usdc, maker.address)), amount);
  await must(await call(usdc, iUsdc, "approve", [swap, amount], maker.address), "approve");
  await refuses(await call(swap, iSwap, "createBid", [H(s), amount, 1, mdsAddr, 3600, 0], maker.address), "hashlock reused", "a bid on a used secret");

  await refuses(await call(swap, iSwap, "claimBid", [id, s], taker.address), "not reserved", "claim before a reservation");
  await refuses(await call(swap, iSwap, "reserveBid", [id, 4 * 3600], taker.address), "token transfer failed", "a reservation without its bond");
  await must(await call(usdc, iUsdc, "approve", [swap, bond], taker.address), "approve the bond");
  await refuses(await call(swap, iSwap, "reserveBid", [id, 2 * 3600 - 1], taker.address), "bad window", "a short window");
  await refuses(await call(swap, iSwap, "reserveBid", [id, 3 * 86400 + 1], taker.address), "bad window", "a long window");
  const t0 = await balance(usdc, taker.address);
  const rr = await must(await call(swap, iSwap, "reserveBid", [id, 4 * 3600], taker.address), "reserveBid");
  const rev = events(rr, iSwap).find((e) => e.name === "BidReserved");
  assert.equal(rev.args.filler, taker.address);
  assert.equal(rev.args.fillDeadline, BigInt(now + 4 * 3600));
  assert.equal(t0 - (await balance(usdc, taker.address)), bond, "the bond is staked");
  await refuses(await call(swap, iSwap, "reserveBid", [id, 4 * 3600], stranger.address), "already reserved", "a second reservation");
  await refuses(await call(swap, iSwap, "cancelBid", [id], maker.address), "reserved", "cancel during the reservation");
  await refuses(await call(swap, iSwap, "claimBid", [id, secret()], taker.address), "bad preimage", "claim with a wrong secret");
  const b = await view(swap, iSwap, "bids", [id]);
  assert.deepEqual([b.maker, b.filler, b.amount, b.fillBond, b.hashlock, b.settled], [maker.address, taker.address, amount, bond, H(s), false]);

  now += 3600;
  // Anyone holding the secret may claim; the filler is paid.
  const t1 = await balance(usdc, taker.address);
  const c = await must(await call(swap, iSwap, "claimBid", [id, s], stranger.address), "claimBid");
  assert.equal(events(c, iSwap).find((e) => e.name === "BidClaimed").args.preimage, hex(s));
  assert.equal((await balance(usdc, taker.address)) - t1, amount + bond, "escrow and bond to the filler");
  await refuses(await call(swap, iSwap, "cancelBid", [id], maker.address), "settled", "cancel after the claim");
}

{
  // A reservation that lapses: the maker takes the escrow and the bond.
  const s = secret();
  const amount = usd(300);
  const bond = usd(10);
  const id = await newBid(s, amount, bond);
  await must(await call(usdc, iUsdc, "approve", [swap, bond], taker.address), "approve the bond");
  await must(await call(swap, iSwap, "reserveBid", [id, 2 * 3600], taker.address), "reserveBid");
  now += 2 * 3600 + 1;
  await refuses(await call(swap, iSwap, "claimBid", [id, s], taker.address), "fill lapsed", "a claim past the deadline");
  await refuses(await call(swap, iSwap, "cancelBid", [id], stranger.address), "not maker", "cancel by a stranger");
  const m0 = await balance(usdc, maker.address);
  const r = await must(await call(swap, iSwap, "cancelBid", [id], maker.address), "cancelBid after the lapse");
  assert.ok(events(r, iSwap).some((e) => e.name === "BidCancelled"));
  assert.equal((await balance(usdc, maker.address)) - m0, amount + bond, "the bond is forfeit to the maker");
  await refuses(await call(swap, iSwap, "reserveBid", [id, 2 * 3600], stranger.address), "settled", "a reservation after the cancel");
}

{
  // Unreserved: cancelled at once. Expired: never reserved.
  const id = await newBid(secret(), usd(40), 0n, 3600);
  const m0 = await balance(usdc, maker.address);
  await must(await call(swap, iSwap, "cancelBid", [id], maker.address), "cancel an open bid");
  assert.equal((await balance(usdc, maker.address)) - m0, usd(40));
  const old = await newBid(secret(), usd(40), 0n, 3600);
  now += 3600;
  await refuses(await call(swap, iSwap, "reserveBid", [old, 2 * 3600], taker.address), "expired", "a reservation of an expired bid");
  // No bond: nothing to approve.
  const free = await newBid(secret(), usd(40), 0n, 3600);
  await must(await call(swap, iSwap, "reserveBid", [free, 2 * 3600], stranger.address), "reserve a bid that asks no bond");
}

{
  // The permit forms of a bid and of its reservation.
  const s = secret();
  const amount = usd(99);
  const bond = usd(3);
  const p = await permit(usdc, maker, swap, amount);
  const r = await must(await call(swap, iSwap, "createBidWithPermit", [H(s), amount, 4096, mdsAddr, 3600, bond, p.deadline, p.v, p.r, p.s], maker.address), "createBidWithPermit");
  const id = events(r, iSwap).find((e) => e.name === "BidCreated").args.bidId;
  const q = await permit(usdc, taker, swap, bond);
  await must(await call(swap, iSwap, "reserveBidWithPermit", [id, 2 * 3600, q.deadline, q.v, q.r, q.s], taker.address), "reserveBidWithPermit");
  await must(await call(swap, iSwap, "claimBid", [id, s], taker.address), "claimBid");
}

// ── Tokens that misbehave ───────────────────────────────────────────────────

{
  // A fee on transfer: what would be locked is less than asked. Refused.
  await must(await call(usdc, iUsdc, "setFee", [100], DEPLOYER), "fee on");
  await must(await call(usdc, iUsdc, "approve", [swap, usd(10)], taker.address), "approve");
  await refuses(await call(swap, iSwap, "lock", [H(secret()), maker.address, usd(10), 3600], taker.address), "token delivered less", "a lock in a fee-taking token");
  await must(await call(usdc, iUsdc, "setFee", [0], DEPLOYER), "fee off");

  // A token that returns nothing: taken.
  await must(await call(usdc, iUsdc, "setSilent", [true], DEPLOYER), "silent on");
  const s = secret();
  await must(await call(usdc, iUsdc, "approve", [swap, usd(10)], taker.address), "approve");
  const r = await must(await call(swap, iSwap, "lock", [H(s), maker.address, usd(10), 3600], taker.address), "a lock in a silent token");
  const id = events(r, iSwap).find((e) => e.name === "Locked").args.swapId;
  await must(await call(swap, iSwap, "claim", [id, s], maker.address), "a claim in a silent token");
  await must(await call(usdc, iUsdc, "setSilent", [false], DEPLOYER), "silent off");

  // A token that answers false: refused, not taken for a payment.
  const f = await deploy(FALSE);
  const fswap = await deploy(SWAP, [f], iSwap);
  await refuses(await call(fswap, iSwap, "lock", [H(secret()), maker.address, 5, 3600], taker.address), "token transfer failed", "a lock in a token that answers false");

  // A beneficiary the token will not pay: the claim fails, and the locker
  // takes the tokens back after the timeout.
  const s2 = secret();
  await must(await call(usdc, iUsdc, "approve", [swap, usd(20)], taker.address), "approve");
  const r2 = await must(await call(swap, iSwap, "lock", [H(s2), maker.address, usd(20), 600], taker.address), "lock");
  const id2 = events(r2, iSwap).find((e) => e.name === "Locked").args.swapId;
  await must(await call(usdc, iUsdc, "setBlacklisted", [maker.address, true], DEPLOYER), "blacklist the beneficiary");
  await refuses(await call(swap, iSwap, "claim", [id2, s2], maker.address), "token transfer failed", "a claim the token will not pay");
  now += 600;
  await must(await call(swap, iSwap, "refund", [id2], taker.address), "the refund still comes");
  await must(await call(usdc, iUsdc, "setBlacklisted", [maker.address, false], DEPLOYER), "unblacklist");
}

// The contract holds exactly what is still locked or escrowed.
const left = await balance(usdc, swap);
// The two permit locks, unclaimed; the expired bid, not cancelled; the free
// bid, reserved and not claimed.
const expect = usd(77.7) * 2n + usd(40) * 2n;
assert.equal(left, expect, `the contract holds ${left}, not ${expect}`);
passed++;

console.log(`${passed} checks passed`);
console.log("gas (with the 21,000 base):", Object.entries(gas).map(([k, v]) => `${k} ${v}`).join(", "));
