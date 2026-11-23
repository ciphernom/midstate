// Deploys MidstateTokenSwap on Base for USDC, then reads it back.
//
//   BASE_RPC=https://mainnet.base.org DEPLOYER_KEY=0x… npm run deploy               (what it would do)
//   BASE_RPC=https://mainnet.base.org DEPLOYER_KEY=0x… npm run deploy -- --broadcast (deploys)
//
// The key only pays for the deployment: the contract has no owner, no admin
// and nothing to configure, so the account that deploys it holds no power
// over it. On Base Sepolia (84532) pass --token with that network's USDC.

import fs from "node:fs";
import { ethers } from "ethers";
import { blake3 } from "@noble/hashes/blake3.js";

const USDC = { 8453: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913", 84532: "0x036CbD53842c5426634e7929541eC2318f3dCF7e" };
const arg = (name) => {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : undefined;
};
const broadcast = process.argv.includes("--broadcast");

const rpc = process.env.BASE_RPC;
const key = process.env.DEPLOYER_KEY;
if (!rpc || !key) {
  console.error("Set BASE_RPC and DEPLOYER_KEY (the key pays the gas; it gets no power over the contract).");
  process.exit(1);
}
const provider = new ethers.JsonRpcProvider(rpc);
const wallet = new ethers.Wallet(key, provider);
const { chainId } = await provider.getNetwork();
const token = arg("--token") ?? USDC[Number(chainId)];
if (!token) {
  console.error(`chain ${chainId} is not Base or Base Sepolia: pass --token <address> to deploy anyway`);
  process.exit(1);
}

// The token must be a token, and take permits (the wallet pays in with them).
const erc20 = new ethers.Contract(token, ["function decimals() view returns (uint8)", "function symbol() view returns (string)", "function DOMAIN_SEPARATOR() view returns (bytes32)"], provider);
const [decimals, symbol] = [await erc20.decimals(), await erc20.symbol()];
await erc20.DOMAIN_SEPARATOR();
console.log(`chain ${chainId}; token ${token}: ${symbol}, ${decimals} decimals, takes permits`);
if (Number(decimals) !== 6) console.warn("warning: the wallet counts this market in millionths, as USDC's six decimals");

const art = JSON.parse(fs.readFileSync(new URL("build/MidstateTokenSwap.json", import.meta.url), "utf8"));
const factory = new ethers.ContractFactory(art.abi, art.bytecode, wallet);
const tx = await factory.getDeployTransaction(token);
const gas = await provider.estimateGas({ ...tx, from: wallet.address });
const fee = await provider.getFeeData();
console.log(`from ${wallet.address} (${ethers.formatEther(await provider.getBalance(wallet.address))} ETH): about ${gas} gas, ${ethers.formatEther(gas * (fee.maxFeePerGas ?? 0n))} ETH at most`);
if (!broadcast) {
  console.log("Not sent. Add --broadcast to deploy.");
  process.exit(0);
}

const c = await factory.deploy(token);
console.log(`sent ${c.deploymentTransaction().hash}`);
await c.waitForDeployment();
const address = await c.getAddress();
// Read it back: the token it was made for, and BLAKE3 as midstate hashes.
const zero = new Uint8Array(32);
if ((await c.token()).toLowerCase() !== token.toLowerCase()) throw new Error("token() is not the token deployed for");
if ((await c.blake3_256(zero)) !== ethers.hexlify(blake3(zero))) throw new Error("blake3_256 disagrees with BLAKE3");
console.log(`\nMidstateTokenSwap: ${address}`);
console.log("Set it in the wallet under Trade → Connection → USDC contract (or as BASE_MAINNET_TOKEN_CONTRACT in crates/walletd/src/evm.rs for a release).");
