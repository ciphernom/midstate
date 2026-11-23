// Compiles the swap contracts (from the repository's root) and the test
// token with solc-js, writing build/<Contract>.json: ABI, creation code and
// runtime code. The settings are the ones to deploy and verify with.
import solc from "solc";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");
const files = [path.join(root, "MidstateTokenSwap.sol"), path.join(root, "MidstateAtomicSwap.sol"), path.join(here, "MockUSDC.sol")];

export const SETTINGS = { optimizer: { enabled: true, runs: 200 }, evmVersion: "cancun" };

const sources = {};
for (const f of files) sources[path.basename(f)] = { content: fs.readFileSync(f, "utf8") };
const input = {
  language: "Solidity",
  sources,
  settings: { ...SETTINGS, outputSelection: { "*": { "*": ["abi", "evm.bytecode.object", "evm.deployedBytecode.object"] } } },
};
const out = JSON.parse(solc.compile(JSON.stringify(input)));
let failed = false;
for (const e of out.errors ?? []) {
  console.error(e.formattedMessage);
  if (e.severity === "error") failed = true;
}
if (failed) process.exit(1);
fs.mkdirSync(path.join(here, "build"), { recursive: true });
for (const [file, contracts] of Object.entries(out.contracts)) {
  for (const [name, c] of Object.entries(contracts)) {
    if (!c.evm.bytecode.object) continue; // interfaces
    fs.writeFileSync(path.join(here, "build", `${name}.json`), JSON.stringify({ abi: c.abi, bytecode: c.evm.bytecode.object, deployed: c.evm.deployedBytecode.object }, null, 1));
    console.log(`${file}:${name}: ${c.evm.deployedBytecode.object.length / 2} bytes deployed`);
  }
}
console.log(`solc ${solc.version()}, optimizer ${SETTINGS.optimizer.runs} runs, evm ${SETTINGS.evmVersion}`);
