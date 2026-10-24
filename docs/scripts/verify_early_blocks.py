#!/usr/bin/env python3
"""
verify_early_blocks.py - independently check what happened in Midstate's early blocks.

Uses only public node endpoints (GET /state, /block/:h, /headers/:start/:count,
/coin/:id, /supply and POST /check_output) and Python 3's standard library.

What it checks, for blocks --start..--end (default 1..50,000):
  * timestamps and the time span (UTC)
  * each sampled block's difficulty target, recomputed from the consensus rule
    (ASERT, src/core/state.rs:19-66) and compared with what the block carries,
    and where the minimum-difficulty (all-ff target) period ends
  * the block reward against the code's schedule (2^30 base units, halving every
    525,600 blocks)
  * how much the range issued, as a share of current circulating supply (/supply)
  * how many distinct addresses the sampled rewards went to
  * whether each sampled reward coin is still unspent

Point it at your own node if you can:  --rpc http://127.0.0.1:8545
Public nodes: https://rpc.cypherpunk.gold (default) and https://rpc.usa.cypherpunk.gold

Code references are to the midstate repository at commit cf8990f.
"""

import argparse
import datetime
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

VERSION = "1.0"

# ── Consensus constants (midstate @ cf8990f) ────────────────────────────────
# src/core/types.rs:918   TARGET_BLOCK_TIME = 60
# src/core/types.rs:989   BLOCKS_PER_YEAR = 365*24*3600 / TARGET_BLOCK_TIME = 525,600
# src/core/types.rs:992   INITIAL_REWARD = 1_073_741_824 (2^30 base units)
# src/core/types.rs:1009-1012  block_reward(h) = INITIAL_REWARD >> min(h / BLOCKS_PER_YEAR, 30)
TARGET_BLOCK_TIME = 60
BLOCKS_PER_YEAR = 365 * 24 * 3600 // TARGET_BLOCK_TIME
INITIAL_REWARD = 1 << 30

# Units: the base unit is called "MDS"; 2^30 MDS is "1 gMDS".
#   src/rpc/types.rs:610      "Every amount is in base MDS units (1 gMDS = 2^30 MDS)"
#   apps/desktop/src/theme.rs:471-472  UNIT_NAMES MDS/kMDS/mMDS/gMDS = 1, 2^10, 2^20, 2^30
#   src/rpc/explorer.html:1745-1747    MDS_PER_GMDS = 1073741824
GMDS = 1 << 30

# Genesis (src/core/types.rs:265 and :332-337) and ASERT half-life (types.rs:921)
GENESIS_TIMESTAMP = 1772274770                       # 2026-02-28 10:32:50 UTC
GENESIS_TARGET = int("0011" + "ff" * 30, 16)
ASERT_HALF_LIFE = 4 * 60 * 60
CEILING = (1 << 256) - 1                             # all-ff target = minimum difficulty

# src/core/types.rs:948 TIMEWARP_FIX_ACTIVATION_HEIGHT; src/core/state.rs:100-104
TIMEWARP_FIX_HEIGHT = 250_000


def block_reward(height):
    return INITIAL_REWARD >> min(height // BLOCKS_PER_YEAR, 30)


def cumulative_reward(blocks):
    """Sum of block_reward(h) for h in 0..blocks-1 (types.rs:1040-1054)."""
    total, era = 0, 0
    while True:
        start = era * BLOCKS_PER_YEAR
        if start >= blocks:
            return total
        if era >= 30:
            return total + (blocks - start)
        end = min(blocks, start + BLOCKS_PER_YEAR)
        total += (end - start) * (INITIAL_REWARD >> era)
        era += 1


def calculate_target(height, prev_timestamp):
    """Port of calculate_target (src/core/state.rs:19-66). The target a block at
    `height` must carry is calculate_target(height, timestamp of block height-1)
    (src/sync.rs:120)."""
    if height == 0:
        return GENESIS_TARGET
    drift = (prev_timestamp - GENESIS_TIMESTAMP) - height * TARGET_BLOCK_TIME
    e = drift * 65536
    exponent = -((-e) // ASERT_HALF_LIFE) if e < 0 else e // ASERT_HALF_LIFE  # Rust truncation
    shifts = exponent >> 16
    frac = exponent & 0xFFFF
    factor = 65536
    factor += (frac * 45426) >> 16
    factor += (frac * frac * 15746) >> 32
    factor += (frac * frac * frac * 3643) >> 48
    t = GENESIS_TARGET // 65536 * factor + (GENESIS_TARGET % 65536) * factor // 65536
    if shifts > 0:
        s = min(shifts, 255)
        t = CEILING if t > (CEILING >> s) else t << s
    elif shifts < 0:
        t >>= min(-shifts, 255)
    if t > CEILING:
        t = CEILING
    if t == 0:
        t = 1
    return t


# ── Minimal BLAKE3 (single chunk, inputs up to 1024 bytes) ──────────────────
# Coin IDs are BLAKE3(address || value_le_u64 || salt) (src/core/types.rs:126-135),
# a 72-byte input. hashlib has no BLAKE3, so this is a small pure-Python version
# so the script does not have to trust the node's own coin_id.
_IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
       0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
_PERM = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
_M32 = 0xFFFFFFFF


def _rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & _M32


def _g(s, a, b, c, d, mx, my):
    s[a] = (s[a] + s[b] + mx) & _M32; s[d] = _rotr(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & _M32;      s[b] = _rotr(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b] + my) & _M32; s[d] = _rotr(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & _M32;      s[b] = _rotr(s[b] ^ s[c], 7)


def _compress(cv, m, counter, block_len, flags):
    s = list(cv) + _IV[:4] + [counter & _M32, (counter >> 32) & _M32, block_len, flags]
    m = list(m)
    for r in range(7):
        _g(s, 0, 4, 8, 12, m[0], m[1]);   _g(s, 1, 5, 9, 13, m[2], m[3])
        _g(s, 2, 6, 10, 14, m[4], m[5]);  _g(s, 3, 7, 11, 15, m[6], m[7])
        _g(s, 0, 5, 10, 15, m[8], m[9]);  _g(s, 1, 6, 11, 12, m[10], m[11])
        _g(s, 2, 7, 8, 13, m[12], m[13]); _g(s, 3, 4, 9, 14, m[14], m[15])
        if r < 6:
            m = [m[i] for i in _PERM]
    return [s[i] ^ s[i + 8] for i in range(8)]


def blake3_small(data):
    if len(data) > 1024:
        raise ValueError("blake3_small handles a single chunk (<= 1024 bytes)")
    blocks = [data[i:i + 64] for i in range(0, len(data), 64)] or [b""]
    cv = list(_IV)
    for i, blk in enumerate(blocks):
        flags = (1 if i == 0 else 0) | ((2 | 8) if i == len(blocks) - 1 else 0)
        words = struct.unpack("<16I", blk.ljust(64, b"\0"))
        cv = _compress(cv, words, 0, len(blk), flags)
    return struct.pack("<8I", *cv)


def coin_id(address_hex, value, salt_hex):
    return blake3_small(bytes.fromhex(address_hex) + struct.pack("<Q", value)
                        + bytes.fromhex(salt_hex)).hex()


assert blake3_small(b"").hex() == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"


# ── HTTP with rate limit and retries ─────────────────────────────────────────
class Node:
    def __init__(self, base, rate, retries=6, timeout=60):
        self.base = base.rstrip("/")
        self.min_gap = 1.0 / rate if rate > 0 else 0
        self.retries = retries
        self.timeout = timeout
        self.last = 0.0
        self.requests = 0

    def _wait(self):
        gap = time.monotonic() - self.last
        if gap < self.min_gap:
            time.sleep(self.min_gap - gap)
        self.last = time.monotonic()

    def call(self, path, body=None):
        url = self.base + path
        data = json.dumps(body).encode() if body is not None else None
        headers = {"User-Agent": "midstate-verify-early-blocks/" + VERSION, "Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        delay = 2.0
        for attempt in range(1, self.retries + 1):
            self._wait()
            self.requests += 1
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                retryable = e.code in (429, 500, 502, 503, 504)
                msg = "HTTP %d" % e.code
                try:
                    msg += ": " + e.read().decode()[:200]
                except Exception:
                    pass
                if not retryable or attempt == self.retries:
                    raise RuntimeError("%s -> %s" % (path, msg))
                ra = e.headers.get("Retry-After") if e.headers else None
                wait = float(ra) if ra and ra.isdigit() else delay
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as e:
                if attempt == self.retries:
                    raise RuntimeError("%s -> %s" % (path, e))
                msg, wait = str(e), delay
            print("  retry %d/%d for %s in %.0fs (%s)" % (attempt, self.retries - 1, path, wait, msg),
                  file=sys.stderr, flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 60)


# ── Cache ────────────────────────────────────────────────────────────────────
def load_cache(path, rpc):
    cache = {"version": VERSION, "nodes": {}}
    if os.path.exists(path):
        try:
            with open(path) as f:
                cache = json.load(f)
        except (OSError, ValueError):
            print("warning: cache %s unreadable, starting fresh" % path, file=sys.stderr)
    node = cache.setdefault("nodes", {}).setdefault(rpc, {})
    node.setdefault("blocks", {})
    return cache, node["blocks"]


def save_cache(path, cache):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


# ── Helpers ──────────────────────────────────────────────────────────────────
def hx(b):
    return bytes(b).hex()


def utc(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def short_target(t):
    return ("%064x" % t)[:12] + "…"


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def fetch_block(node, h):
    """One sampled block: the header pair (h-1, h) and the full block."""
    hdrs = node.call("/headers/%d/2" % (h - 1))["headers"]
    if len(hdrs) != 2 or hdrs[1]["height"] != h:
        raise RuntimeError("unexpected /headers response for %d" % h)
    b = node.call("/block/%d" % h)
    prev_ts = hdrs[0]["timestamp"]
    target = int(hx(b["target"]), 16)
    cbs = []
    for c in b["coinbase"]:
        a, s = hx(c["address"]), hx(c["salt"])
        cbs.append({"address": a, "value": c["value"], "salt": s, "coin_id": coin_id(a, c["value"], s)})
    return {
        "ts": b["timestamp"],
        "prev_ts": prev_ts,
        "target": "%064x" % target,
        "header_matches_block": hdrs[1]["timestamp"] == b["timestamp"] and hx(hdrs[1]["target"]) == "%064x" % target,
        "expected_target": "%064x" % calculate_target(h, prev_ts),
        "ntx": len(b["transactions"]),
        "coinbase": cbs,
    }


def header_at(node, h):
    hd = node.call("/headers/%d/1" % h)["headers"]
    if not hd:
        raise RuntimeError("no header at %d" % h)
    return hd[0]


def catchup_search(node, lo, tip, out):
    """Binary search for the first block at or above `lo` whose target is not the
    all-ff minimum, assuming the minimum period is one contiguous run."""
    print("\n== Catch-up search (binary search on /headers) ==", flush=True)
    t_lo = int(hx(header_at(node, lo)["target"]), 16)
    if t_lo != CEILING:
        print("block %d is not at minimum difficulty; nothing to search from there" % lo)
        return None
    hi = lo
    step = 1024
    while True:
        hi = min(lo + step, tip)
        if int(hx(header_at(node, hi)["target"]), 16) != CEILING:
            break
        if hi == tip:
            print("every probed block up to the tip (%d) is at minimum difficulty" % tip)
            return None
        lo, step = hi, step * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if int(hx(header_at(node, mid)["target"]), 16) == CEILING:
            lo = mid
        else:
            hi = mid
    last_min, first_not = lo, hi
    hp = node.call("/headers/%d/2" % last_min)["headers"]
    prev = header_at(node, last_min - 1)
    rows = []
    for hdr, p in ((hp[0], prev), (hp[1], hp[0])):
        t = int(hx(hdr["target"]), 16)
        exp = calculate_target(hdr["height"], p["timestamp"])
        drift = (p["timestamp"] - GENESIS_TIMESTAMP) - hdr["height"] * TARGET_BLOCK_TIME
        rows.append((hdr["height"], hdr["timestamp"], t, exp, drift))
        print("  block %-7d %s  target %s  recomputed %s  (chain %.1f h behind the 60 s schedule)"
              % (hdr["height"], utc(hdr["timestamp"]), short_target(t),
                 "matches" if t == exp else "MISMATCH", drift / 3600))
    print("  last minimum-difficulty block: %d   first block above minimum: %d" % (last_min, first_not))
    print("  why: ASERT measures time since the genesis timestamp (%s). The chain restarted in"
          % utc(GENESIS_TIMESTAMP))
    print("       April with that February timestamp, so it was far 'behind schedule' and the")
    print("       target sat at the all-ff ceiling until the block count caught up with the clock.")
    print("       (Genesis target 0x0011ff… needs ~11.8 doublings, ~47 h of drift at a 4 h")
    print("       half-life, to reach the ceiling, which matches the drift shown above.)")
    out["catchup"] = {"last_min_block": last_min, "first_above_min": first_not,
                      "boundary_checks": [{"height": h, "ts": ts, "ts_utc": utc(ts), "target": "%064x" % t,
                                           "target_matches_rule": t == e, "drift_s": d}
                                          for h, ts, t, e, d in rows],
                      "method": "binary search, assumes one contiguous minimum-difficulty run"}
    return last_min


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Verify Midstate early-block facts from public node endpoints.",
        epilog="Public nodes: https://rpc.cypherpunk.gold, https://rpc.usa.cypherpunk.gold. "
               "Better: run your own node and use --rpc http://127.0.0.1:8545")
    ap.add_argument("--rpc", default="https://rpc.cypherpunk.gold",
                    help="node RPC base URL (default %(default)s; mirror https://rpc.usa.cypherpunk.gold; "
                         "your own node e.g. http://127.0.0.1:8545)")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=50000)
    ap.add_argument("--step", type=int, default=50,
                    help="sample every Nth block: heights that are multiples of N, plus --start and --end "
                         "for the time span (default %(default)s; 1 = full scan)")
    ap.add_argument("--rate", type=float, default=2.0, help="max requests per second (default %(default)s)")
    ap.add_argument("--range-size", type=int, default=10000, help="per-range breakdown width (default %(default)s)")
    ap.add_argument("--catchup-search", action="store_true",
                    help="binary-search for the first block whose target is above the minimum")
    ap.add_argument("--cross-check", type=int, default=6,
                    help="re-check this many coins with POST /check_output as well as GET /coin (default %(default)s)")
    ap.add_argument("--recheck-unspent", action="store_true",
                    help="ignore cached spent/unspent status and ask the node again")
    ap.add_argument("--cache", default="verify_early_blocks.cache.json", help="resume cache (default %(default)s)")
    ap.add_argument("--out", default="results.json", help="write full results here (default %(default)s)")
    args = ap.parse_args()

    if args.start < 1 or args.end < args.start or args.step < 1:
        ap.error("need 1 <= --start <= --end and --step >= 1")

    node = Node(args.rpc, args.rate)
    print("Midstate early-blocks verifier %s" % VERSION)
    print("node: %s" % node.base)
    state = node.call("/state")
    tip = state["height"] - 1 if state.get("height") else 0  # /state height counts blocks
    print("chain: %d blocks (tip height %d), syncing=%s" % (state["height"], tip, state.get("is_syncing")))
    if args.end > tip:
        print("note: --end %d is past the tip, clamping to %d" % (args.end, tip))
        args.end = tip

    grid = list(range(((args.start + args.step - 1) // args.step) * args.step, args.end + 1, args.step))
    extra = sorted({args.start, args.end} - set(grid))
    heights = sorted(set(grid) | set(extra))
    is_sample = args.step > 1
    print("range %d..%d, %d sampled blocks (step %d)%s, ~%d requests, rate %.1f/s"
          % (args.start, args.end, len(grid), args.step,
             (" + endpoints %s for the span" % extra) if extra else "",
             len(heights) * 3, args.rate))

    cache, blocks = load_cache(args.cache, node.base)
    out = {"tool": "verify_early_blocks.py", "version": VERSION, "rpc": node.base,
           "run_at_utc": utc(time.time()), "args": vars(args), "chain_blocks": state["height"]}

    # 1. fetch blocks + unspent status
    t0 = time.time()
    done = 0
    try:
        for i, h in enumerate(heights):
            key = str(h)
            rec = blocks.get(key)
            if rec is None:
                rec = fetch_block(node, h)
                blocks[key] = rec
            for c in rec["coinbase"]:
                if "unspent" not in c or args.recheck_unspent:
                    r = node.call("/coin/" + c["coin_id"])
                    if r.get("coin") != c["coin_id"]:
                        raise RuntimeError("/coin echoed a different id for block %d" % h)
                    c["unspent"] = bool(r["exists"])
                    c["checked_at"] = int(time.time())
            done += 1
            if done % 25 == 0 or i == len(heights) - 1:
                save_cache(args.cache, cache)
                el = time.time() - t0
                print("  %d/%d blocks (height %d), %.0fs" % (i + 1, len(heights), h, el), flush=True)
    except KeyboardInterrupt:
        save_cache(args.cache, cache)
        print("\ninterrupted; progress saved to %s, rerun the same command to resume" % args.cache)
        sys.exit(130)
    except RuntimeError as e:
        save_cache(args.cache, cache)
        print("\nerror: %s\nprogress saved to %s, rerun to resume" % (e, args.cache), file=sys.stderr)
        sys.exit(1)

    recs = {h: blocks[str(h)] for h in heights}
    problems = []

    # 2. cross-check a few coins with POST /check_output
    spent = [(h, c) for h in grid for c in recs[h]["coinbase"] if not c["unspent"]]
    unsp = [(h, c) for h in grid for c in recs[h]["coinbase"] if c["unspent"]]
    picks = unsp[:(args.cross_check + 1) // 2] + spent[:args.cross_check // 2]
    xc = []
    for h, c in picks:
        r = node.call("/check_output", {"address": c["address"], "value": c["value"], "salt": c["salt"]})
        ok = r.get("coin_id") == c["coin_id"] and bool(r.get("exists")) == c["unspent"]
        xc.append({"height": h, "coin_id": c["coin_id"], "agree": ok})
        if not ok:
            problems.append("cross-check disagreement at block %d" % h)
    save_cache(args.cache, cache)

    # 3. catch-up search
    last_min = None
    if args.catchup_search:
        start_from = max((h for h in heights if int(recs[h]["target"], 16) == CEILING), default=None)
        if start_from is None:
            print("\n(no sampled block is at minimum difficulty, skipping catch-up search)")
        else:
            last_min = catchup_search(node, start_from, tip, out)

    supply = node.call("/supply")

    # ── Report ──────────────────────────────────────────────────────────────
    W = 78
    print("\n" + "=" * W)
    print("RESULTS  blocks %d..%d  via %s" % (args.start, args.end, node.base))
    if is_sample:
        print("NOTE: this is a SAMPLE of %d blocks (every %dth). Per-block figures (targets, addresses,"
              % (len(grid), args.step))
        print("      unspent share) describe the sample, not every block. Use --step 1 for a full scan.")
    print("=" * W)

    # timestamps
    print("\n-- Timestamps --")
    a, b = recs[args.start], recs[args.end]
    span = b["ts"] - a["ts"]
    print("block %-7d %s" % (args.start, utc(a["ts"])))
    print("block %-7d %s" % (args.end, utc(b["ts"])))
    nblk = args.end - args.start
    print("span %d s = %.2f h for %d block intervals -> %.3f s per block (target is %d s)"
          % (span, span / 3600, nblk, span / nblk if nblk else 0, TARGET_BLOCK_TIME))
    rates = []
    for x, y in zip(heights, heights[1:]):
        rates.append((recs[y]["ts"] - recs[x]["ts"]) / (y - x))
    if rates:
        print("between consecutive samples: min %.3f, max %.3f s per block" % (min(rates), max(rates)))
    gaps1 = sum(1 for h in heights if recs[h]["ts"] - recs[h]["prev_ts"] == 1)
    print("blocks whose timestamp is exactly 1 s after the previous block: %d of %d checked"
          % (gaps1, len(heights)))
    if 1 in recs:
        print("(block 1 follows the February genesis block, so its gap is ~%.0f days)"
              % ((recs[1]["ts"] - recs[1]["prev_ts"]) / 86400))

    # targets
    print("\n-- Difficulty target --")
    n_min = [h for h in heights if int(recs[h]["target"], 16) == CEILING]
    not_min = [h for h in heights if int(recs[h]["target"], 16) != CEILING]
    bad_rule = [h for h in heights if recs[h]["target"] != recs[h]["expected_target"]]
    bad_hdr = [h for h in heights if not recs[h]["header_matches_block"]]
    print("all-ff (minimum difficulty) target: %d of %d checked blocks" % (len(n_min), len(heights)))
    if not_min:
        print("not at minimum: %s" % ", ".join("%d (%s)" % (h, short_target(int(recs[h]["target"], 16)))
                                               for h in not_min[:10]) + (" …" if len(not_min) > 10 else ""))
    if 1 in not_min:
        print("block 1 is expected to be below the ceiling: its target comes from the genesis block's")
        print("timestamp, which is on schedule by definition. From block 2 on the April timestamps")
        print("put the chain far behind schedule, so the rule gives the all-ff ceiling.")
    print("target equals the ASERT rule recomputed locally (state.rs:19-66): %d of %d%s"
          % (len(heights) - len(bad_rule), len(heights), ("  MISMATCH at %s" % bad_rule[:10]) if bad_rule else ""))
    if bad_hdr:
        problems.append("/headers and /block disagree at %s" % bad_hdr[:10])
    if bad_rule:
        problems.append("target does not match the consensus rule at %s" % bad_rule[:10])
    if last_min is not None:
        print("minimum-difficulty period ends at block %d (catch-up search)" % last_min)
    else:
        print("(run with --catchup-search to find where the minimum-difficulty period ends)")

    # reward
    print("\n-- Block reward (types.rs:992, :1009-1012: 2^30 >> (height / 525,600)) --")
    reward_bad, ncb, ntx = [], {}, 0
    for h in heights:
        r = recs[h]
        vals = [c["value"] for c in r["coinbase"]]
        ncb[len(vals)] = ncb.get(len(vals), 0) + 1
        ntx += r["ntx"]
        # coinbase = reward + fees; with no transactions there are no fees
        if (r["ntx"] == 0 and sum(vals) != block_reward(h)) or sum(vals) < block_reward(h):
            reward_bad.append(h)
    print("coinbase total equals block_reward(height) = %d MDS (1 gMDS): %d of %d checked blocks"
          % (INITIAL_REWARD, len(heights) - len(reward_bad), len(heights)))
    print("coinbase outputs per block: %s; transactions in checked blocks: %d"
          % (", ".join("%d output%s: %d blocks" % (k, "" if k == 1 else "s", v) for k, v in sorted(ncb.items())), ntx))
    if reward_bad:
        problems.append("reward differs from schedule at %s" % reward_bad[:10])

    # issuance vs supply
    print("\n-- Issuance vs current supply (/supply) --")
    issued_range = cumulative_reward(args.end + 1) - cumulative_reward(args.start)
    circ = supply.get("circulating_supply")
    print("/supply unit field: %r (base units; 1 gMDS = 2^30 = %d MDS)" % (supply.get("unit"), GMDS))
    print("/supply height %d, verified=%s, index=%s" % (supply.get("height", -1), supply.get("verified"),
                                                         (supply.get("index") or {}).get("status")))
    if supply.get("issued") is not None:
        exp_iss = cumulative_reward(supply["height"] + 1)
        print("node's 'issued' equals the schedule summed to its height: %s" % (supply["issued"] == exp_iss))
        if supply["issued"] != exp_iss:
            problems.append("/supply issued differs from schedule")
    if circ is None:
        circ = supply["issued"] - supply.get("burned_genesis", 0)
        print("(circulating_supply not served, index not ready; using issued - genesis burn)")
    print("current circulating: %d MDS = %.3f gMDS" % (circ, circ / GMDS))
    print("blocks %d..%d issued %d MDS = %.3f gMDS = %.2f%% of current circulating"
          % (args.start, args.end, issued_range, issued_range / GMDS, pct(issued_range, circ)))
    if last_min is not None:
        iss_cu = cumulative_reward(last_min + 1) - cumulative_reward(1)
        print("blocks 1..%d (whole minimum-difficulty period) issued %.3f gMDS = %.2f%% of current circulating"
              % (last_min, iss_cu / GMDS, pct(iss_cu, circ)))

    # addresses
    print("\n-- Reward addresses (sampled blocks) --")
    addrs = {}
    for h in grid:
        for c in recs[h]["coinbase"]:
            addrs[c["address"]] = addrs.get(c["address"], 0) + 1
    outs = sum(addrs.values())
    reused = {a: n for a, n in addrs.items() if n > 1}
    print("%d reward outputs, %d distinct addresses, %d addresses used more than once"
          % (outs, len(addrs), len(reused)))

    # unspent
    print("\n-- Are the sampled reward coins still unspent? (GET /coin/:id) --")
    tot_n = sum(len(recs[h]["coinbase"]) for h in grid)
    tot_u = sum(1 for h in grid for c in recs[h]["coinbase"] if c["unspent"])
    print("unspent: %d of %d = %.1f%%" % (tot_u, tot_n, pct(tot_u, tot_n)))
    print("coin ids computed locally with BLAKE3(address||value||salt) (types.rs:126-135)")
    print("cross-check with POST /check_output: %d of %d agree" % (sum(x["agree"] for x in xc), len(xc)))

    # per-range
    print("\n-- Per range (%d blocks each; sampled blocks only) --" % args.range_size)
    print("%-15s %7s %9s %9s %12s  %-23s %-23s" % ("blocks", "sampled", "min-diff", "unspent", "unspent %",
                                                   "first sampled ts", "last sampled ts"))
    ranges = []
    r0 = (args.start - 1) // args.range_size
    r1 = (args.end - 1) // args.range_size
    for r in range(r0, r1 + 1):
        lo_h, hi_h = r * args.range_size + 1, (r + 1) * args.range_size
        hs = [h for h in grid if lo_h <= h <= hi_h]
        if not hs:
            continue
        n = sum(len(recs[h]["coinbase"]) for h in hs)
        u = sum(1 for h in hs for c in recs[h]["coinbase"] if c["unspent"])
        m = sum(1 for h in hs if int(recs[h]["target"], 16) == CEILING)
        label = "%d-%d" % (max(lo_h, args.start), min(hi_h, args.end))
        print("%-15s %7d %9d %9d %11.1f%%  %-23s %-23s" % (label, len(hs), m, u, pct(u, n),
                                                           utc(recs[hs[0]]["ts"]), utc(recs[hs[-1]]["ts"])))
        ranges.append({"blocks": label, "sampled": len(hs), "min_difficulty": m, "unspent": u,
                       "outputs": n, "unspent_pct": round(pct(u, n), 2)})

    # limits
    print("\n-- What this does NOT show --")
    print("* Timestamps are miner-supplied. Before block %d a block could be up to 10 hours"
          % TIMEWARP_FIX_HEIGHT)
    print("  ahead of the validating node's clock (state.rs:100-104), so real elapsed time can be")
    print("  shorter than the timestamp span. After %d the limit is 15 minutes." % TIMEWARP_FIX_HEIGHT)
    print("* Each reward going to a different address says nothing about how many miners there were.")
    print("  Solo miners get a fresh one-time address per block, so one miner and a hundred look the same.")
    print("* Spent coins are not traced. 'Spent' only means the coin left the UTXO set; where the value")
    print("  went (fresh change addresses, power-of-two splits, optional coinjoin) is not followed.")
    print("* Unspent status is as of this run (%s) and as seen by %s." % (out["run_at_utc"], node.base))
    if is_sample:
        print("* Sample only (every %dth block). Use --step 1 for every block." % args.step)

    print("\n" + ("PROBLEMS: " + "; ".join(problems) if problems else "All consistency checks passed."))
    print("requests made this run: %d" % node.requests)

    out.update({
        "sample": is_sample, "sampled_heights": len(grid),
        "span": {"start_ts_utc": utc(a["ts"]), "end_ts_utc": utc(b["ts"]), "seconds": span,
                 "sec_per_block": span / nblk if nblk else None},
        "targets": {"checked": len(heights), "minimum": len(n_min), "not_minimum": not_min,
                    "rule_mismatches": bad_rule},
        "reward": {"checked": len(heights), "mismatches": reward_bad, "outputs_per_block": ncb},
        "supply": {"unit": supply.get("unit"), "height": supply.get("height"), "circulating_mds": circ,
                   "issued_in_range_mds": issued_range, "share_pct": round(pct(issued_range, circ), 3)},
        "addresses": {"outputs": outs, "distinct": len(addrs), "reused": len(reused)},
        "unspent": {"unspent": tot_u, "outputs": tot_n, "pct": round(pct(tot_u, tot_n), 2),
                    "cross_check": xc},
        "ranges": ranges, "problems": problems,
        "blocks": {str(h): recs[h] for h in heights},
    })
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("full results written to %s" % args.out)


if __name__ == "__main__":
    main()
