# verify_early_blocks.py

Check Midstate's early blocks yourself: when they were mined, how hard they were to mine, what they paid, how much of today's supply they make up, and how many of the reward coins have never moved. The script needs Python 3 and nothing else (standard library only), and it reads only public node endpoints.

## Usage

```
python3 verify_early_blocks.py                          # blocks 1-50,000, every 50th block (~25 min at 2 req/s)
python3 verify_early_blocks.py --step 500               # quick pass, every 500th block (~3 min)
python3 verify_early_blocks.py --step 1                 # every block (slow, ~150k requests)
python3 verify_early_blocks.py --catchup-search         # also find where minimum difficulty ended
python3 verify_early_blocks.py --rpc http://127.0.0.1:8545   # use your own node (recommended)
```

Options:

- `--rpc` sets the node. The default is https://rpc.cypherpunk.gold, and the mirror is https://rpc.usa.cypherpunk.gold. You can and should point it at your own node.
- `--start` / `--end` set the block range (default 1 / 50000).
- `--step N` samples block heights that are multiples of N. `--start` and `--end` are always fetched for the time span. `--step 1` scans every block.
- `--rate` sets the maximum requests per second (default 2). Please keep it polite on the public nodes.
- `--catchup-search` binary-searches for the first block whose target is above the minimum.
- `--out` sets where the full JSON results go (default `results.json`).
- `--cache` sets the resume file (default `verify_early_blocks.cache.json`). If a run is interrupted, run the same command again and it picks up where it stopped. `--recheck-unspent` asks the node again about spent/unspent status instead of using the cached answer.
- `--range-size` sets the width of the per-range table (default 10000). `--cross-check N` sets how many coins are checked a second way, with `POST /check_output` (default 6).

## What it checks

- **Timestamps:** the timestamps of the first and last block, the time span in UTC, and the seconds per block.
- **Difficulty:** the target of every fetched block, recomputed locally from the consensus rule (ASERT, `src/core/state.rs:19-66`), plus where the minimum-difficulty (all-ff target) period ends.
- **Reward:** the reward is checked against `block_reward()`: 2^30 base units, halving every 525,600 blocks (`src/core/types.rs:992`, `:1009-1012`).
- **Share of supply:** what the range issued as a share of current circulating supply from `/supply`. The script also checks that the node's `issued` figure equals the schedule.
- **Addresses:** how many distinct addresses the sampled rewards went to.
- **Unspent coins:** whether each sampled reward coin is still unspent. The coin id is computed locally as BLAKE3(address‖value‖salt) and looked up with `GET /coin/:id`, and a few coins are cross-checked with `POST /check_output`.

Units: amounts are in **MDS**, the base unit, and **1 gMDS = 2^30 MDS**. The block reward is 1 gMDS. The `/supply` endpoint reports base MDS.

## What it does not show

- **Timestamps are set by miners.** Before block 250,000 a block could be up to 10 hours ahead of the node's clock, so real elapsed time may be shorter than the timestamp span.
- **Addresses don't count miners.** Solo-mining rewards go to a fresh one-time address every block, so one miner and a hundred miners look the same.
- **Spent coins are not traced.** "Spent" only means the coin is no longer in the UTXO set.
- **A sample is a sample.** Unless you use `--step 1`, the per-block figures describe the sampled blocks only.

Code references are to the midstate repository at commit cf8990f.

sha256 (verify_early_blocks.py): `a50b8f2cc4b272685850c5a70f3c4e7f33b47a66dfd822c54bf4bbc1ca0fdf46`
