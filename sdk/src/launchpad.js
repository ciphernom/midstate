// launchpad.js — pump.fun-style token launches on Midstate's v6 opcodes.
//
// ═══════════════════════════════════════════════════════════════════════════
//  What a launch is
// ═══════════════════════════════════════════════════════════════════════════
//
// One script, two phases, one holder ledger:
//
//   curve  the k-th token sold costs (k-1)·P MDS, until T tokens are sold;
//   pool   then the MDS raised (x) and T/2 reserved tokens (y) trade as a
//          constant-product pool with a 0.3% fee that stays in the pool.
//
// The pool opens at x/y = P(T-1), the price of the curve's last token, so the
// price does not jump at graduation. Holders are rows in a Merkle ledger of
// 2^depth slots, each row `le8(balance) ‖ owner24`, proven with OP_MERKLE_ROOT.
//
// ═══════════════════════════════════════════════════════════════════════════
//  Why the state lives in the address
// ═══════════════════════════════════════════════════════════════════════════
//
// Consensus runs a script once per input, and an input only sees its own
// commitment. A state coin carries no value and a value coin carries no state,
// so a contract built as "state thread + reserve coin" can never spend its
// reserve: the reserve coin's run has no state to read. (That is why the
// older `pump.js` and `launcher.js` curves cannot work on chain.)
//
// Here the bytecode is `PUSH_DATA(state) DROP code`. Every coin at an address
// knows the whole state, because the state is part of what the address hashes.
// The witness carries a copy of `code`, and the script checks
//
//     hash(PUSH ‖ state ‖ DROP ‖ code) == this_address()
//
// so it can compute the address of the NEXT state and require the entire
// reserve to move there. `sum_input_value() == x + seed` makes every trade
// spend every reserve coin, so a state cannot be forked unless someone first
// pays a full second copy of the reserve to it — which only funds the fork.
//
// ═══════════════════════════════════════════════════════════════════════════
//  Why every trade leaves a receipt
// ═══════════════════════════════════════════════════════════════════════════
//
// Nodes strip witnesses from the blocks they serve (HTTP and light protocol
// alike), so the ledger cannot be rebuilt from witnesses. Output 0 of every
// trade is therefore a zero-value state coin AT THE NEXT ADDRESS whose 32 bytes
// are `owner24 ‖ le8(new balance)`, and the script requires it. From a stripped
// block a reader sees the state being spent (it is in the input's bytecode), the
// receipt, and the reserve outputs, which is enough to work out the next state.
// Without the requirement one trader could leave a row nobody else can
// reconstruct, and with it every other proof: the token would freeze.

import { blake3_hash_hex, compute_coin_id_hex } from '../pkg/wasm_wallet.js';

/** Launches use OP_MERKLE_ROOT, so they exist from the v6 activation height. */
export const LAUNCHPAD_HEIGHT = 371_500n;
export const STATE_BYTES = 56;
/** PUSH_DATA with a little-endian u16 length of 56. */
export const PREFIX = '013800';
export const DROP = '10';
/** Witness positions the wallet fills: the trader's public key and signature. */
export const PK_ITEM = 6;
export const SIG_ITEM = 12;
export const WITNESS_ITEMS = 13;
/**
 * No witness item but the signature may be this long. Nodes record the "leaf"
 * of anything in a witness that parses as an MSS signature (620+ bytes, and a
 * zero auth length parses whatever follows), and treat a 576-byte first item as
 * a WOTS signature. A code copy read that way would be marked spent by the
 * first trade and make every later trade look like key reuse. So the code
 * travels in two halves, and the ledger is at most 16 deep (a 528-byte proof).
 */
export const MAX_PLAIN_ITEM = 575;
export const POOL_FEE_NUM = 3n;
export const POOL_FEE_DEN = 1000n;
export const MAX_DEPTH = 16;
/** x·y at graduation stays below this, leaving room for the pool's checks and fee growth in u64. */
export const MAX_POOL_PRODUCT = 1n << 60n;

const U64 = 1n << 64n;
const clean = (h) => String(h ?? '').replace(/^0x/, '').toLowerCase();
const isHex = (h, bytes) => /^[0-9a-f]*$/.test(h) && h.length % 2 === 0 && (bytes == null || h.length === bytes * 2);

// ── Numbers ─────────────────────────────────────────────────────────────────

/** Little-endian u64, 8 bytes: the form every number takes in state and witness. */
export function le8(v) {
    let x = BigInt(v);
    if (x < 0n || x >= U64) throw new Error(`launchpad: ${v} does not fit in a u64`);
    let out = '';
    for (let i = 0; i < 8; i++) { out += Number(x & 0xffn).toString(16).padStart(2, '0'); x >>= 8n; }
    return out;
}

/** Read a little-endian u64 at a byte offset. */
export function readLe8(hex, offset = 0) {
    const h = clean(hex).slice(offset * 2, offset * 2 + 16);
    if (h.length !== 16) throw new Error('launchpad: short u64');
    return BigInt('0x' + h.match(/.{2}/g).reverse().join(''));
}

const beHex = (v, bytes) => BigInt(v).toString(16).padStart(bytes * 2, '0');
const utf8Hex = (s) => Array.from(new TextEncoder().encode(s), (b) => b.toString(16).padStart(2, '0')).join('');
const hexUtf8 = (h) => new TextDecoder('utf-8', { fatal: true }).decode(Uint8Array.from((h.match(/.{2}/g) || []).map((b) => parseInt(b, 16))));

// ── Parameters ──────────────────────────────────────────────────────────────

/**
 * Check and normalise launch parameters.
 *
 * @param {Object} p
 * @param {number} p.depth      ledger depth: 2^depth holder rows (1..16)
 * @param {number|bigint} p.threshold  T, tokens the curve sells before it graduates (even)
 * @param {number|bigint} p.price      P, MDS per step: token k costs (k-1)·P
 * @param {number|bigint} p.seed       MDS left at the address in every state (a power of two)
 * @returns {{depth:number, threshold:bigint, price:bigint, seed:bigint, pool:bigint, raise:bigint}}
 */
export function validateParams(p) {
    const depth = Number(p.depth);
    const threshold = BigInt(p.threshold), price = BigInt(p.price), seed = BigInt(p.seed ?? 1);
    if (!Number.isInteger(depth) || depth < 1 || depth > MAX_DEPTH) throw new Error(`launchpad: depth must be 1..${MAX_DEPTH}`);
    if (threshold < 2n || threshold % 2n || threshold >= 1n << 32n) throw new Error('launchpad: threshold must be even, from 2 to 2^32');
    if (price < 1n || price >= 1n << 52n) throw new Error('launchpad: price must be from 1 to 2^52');
    if (seed < 1n || seed >= 1n << 52n || (seed & (seed - 1n))) throw new Error('launchpad: seed must be a power of two below 2^52');
    const pool = threshold / 2n;
    const raise = curveReserve(price, threshold);
    if (raise * pool >= MAX_POOL_PRODUCT) {
        throw new Error('launchpad: the pool would be too large for the contract\'s u64 arithmetic; lower the price or the threshold');
    }
    return { depth, threshold, price, seed, pool, raise };
}

/** MDS held by the curve at supply s. */
export function curveReserve(price, s) {
    s = BigInt(s);
    return (s * s - s) / 2n * BigInt(price);
}

// ── The contract ────────────────────────────────────────────────────────────

/**
 * Contract source for a launch. `stateHex` only fills the leading push: the
 * code after `PUSH(state) DROP` is the same for every state of one launch.
 */
export function contractSource(params, stateHex = '00'.repeat(STATE_BYTES)) {
    const { depth, threshold, price, seed } = validateParams(params);
    const PB = depth * 33;
    return `
// LAUNCHPAD — a bonding curve that becomes a constant-product pool.
// State (in the address): supply 8 | x (MDS) 8 | y (pool tokens) 8 | root 32.
"${clean(stateHex)}"; drop();

witness T {
    code_a,
    code_b,
    st: ${STATE_BYTES},
    proof: ${PB},
    owner_old: 24,
    owner_new: 24,
    pk: 32,
    old_bal: 8,
    new_bal: 8,
    s_new: 8,
    x_new: 8,
    y_new: 8,
    sig
}

macro st_s() { T.st; 0; 8; slice(); 0; add(); }
macro st_x() { T.st; 8; 8; slice(); 0; add(); }
macro st_y() { T.st; 16; 8; slice(); 0; add(); }

{
    // Who am I: the hash of this state and this code is my address. The size
    // check matters: a longer state copy would let the code be re-split, and
    // the "next address" become any tail of this script.
    assert(size(T.st) == ${STATE_BYTES});
    assert(hash(concat(concat(concat(concat("013800", T.st), "10"), T.code_a), T.code_b)) == this_address());

    // Fixed widths, so every preimage has exactly one reading. A short field in
    // the next state would hash to a script nothing can ever spend.
    assert(size(T.proof) == ${PB});
    assert(size(T.old_bal) == 8);
    assert(size(T.new_bal) == 8);
    assert(size(T.s_new) == 8);
    assert(size(T.x_new) == 8);
    assert(size(T.y_new) == 8);

    // The trader signs this transaction. A row holding tokens answers only to
    // its owner, who may also hand it to a new owner.
    T.sig; require_sig(T.pk);
    if (T.old_bal >= 1) { assert(slice(hash(T.pk), 0, 24) == T.owner_old); }

    // The trader's row is in the ledger.
    hash(concat(T.old_bal, T.owner_old)); T.proof; MERKLE_ROOT;
    assert(slice(T.st, 24, 32) == pop_hex());

    var s = st_s();
    var x = st_x();
    var y = st_y();

    // Every coin of this state is spent: none is left behind to fork from.
    assert(sum_input_value() == x + ${seed});

    // Tokens are conserved: ledger + pool + unsold curve tokens is constant.
    assert(T.new_bal + T.y_new + s == T.old_bal + y + T.s_new);
    assert(${threshold} >= T.s_new);

    // EQUAL compares bytes, so a raw 8-byte witness number gets '+ 0' before
    // it is compared with a computed (minimal) one.
    if (s == ${threshold}) {
        // Pool: supply is frozen, and x*y may not fall after the fee.
        assert(T.s_new + 0 == s);
        if (T.y_new >= y) {
            assert(T.x_new * (T.y_new - (T.y_new - y) * 3 / 1000) >= x * y);
        } else {
            assert((T.x_new - (T.x_new - x) * 3 / 1000) * T.y_new >= x * y);
        }
    } else {
        // Curve: the pool side waits, and the reserve is exactly the curve's.
        assert(T.y_new + 0 == y);
        assert(T.x_new + 0 == (T.s_new * T.s_new - T.s_new) / 2 * ${price});
    }

    // The receipt: output 0 carries the new row in the clear.
    assert(read_output_state(0) == concat(T.owner_new, T.new_bal));

    // The whole reserve moves to the next state's address, which the receipt names.
    hash(concat(T.new_bal, T.owner_new)); T.proof; MERKLE_ROOT;
    concat(concat(concat("013800", T.s_new), T.x_new), T.y_new);
    swap(); concat(); "10"; concat(); T.code_a; concat(); T.code_b; concat(); hash();
    dup(); assert(output_address(0) == pop_hex());
    sum_to_addr(pop_hex());
    assert(T.x_new + ${seed} == pop_int());
}
drop(); drop(); drop(); drop(); drop(); drop(); drop(); drop(); drop(); drop(); drop(); drop(); drop();
true;
`;
}

export const bytecodeFor = (code, stateHex) => PREFIX + clean(stateHex) + DROP + clean(code);

/** The code copy's two witness halves (see MAX_PLAIN_ITEM). */
export function splitCode(code) {
    const c = clean(code), k = Math.ceil(c.length / 4) * 2;
    return [c.slice(0, k), c.slice(k)];
}
export const addressFor = (code, stateHex) => blake3_hash_hex(bytecodeFor(code, stateHex));

/** Split a launch script into its state and its state-independent code. */
export function splitBytecode(bytecode) {
    const b = clean(bytecode);
    if (b.slice(0, 6) !== PREFIX || b.slice(118, 120) !== DROP) throw new Error('launchpad: not a launch script');
    return { state: b.slice(6, 118), code: b.slice(120) };
}

/**
 * Compile a launch's code.
 * @param {Object} params
 * @param {Function} compileFn `compile` from compiler.js
 */
export function compileLaunch(params, compileFn) {
    if (typeof compileFn !== 'function') throw new Error('launchpad: compileLaunch needs compile() from compiler.js');
    const c = compileFn(contractSource(params), { height: LAUNCHPAD_HEIGHT });
    const { code } = splitBytecode(c.bytecode);
    return { code, sizeBytes: c.sizeBytes, sigops: c.sigops };
}

// ── State ───────────────────────────────────────────────────────────────────

export function encodeState({ supply, x, y, root }) {
    const r = clean(root);
    if (!isHex(r, 32)) throw new Error('launchpad: root must be 32 bytes');
    return le8(supply) + le8(x) + le8(y) + r;
}

export function decodeState(hex) {
    const h = clean(hex);
    if (!isHex(h, STATE_BYTES)) throw new Error('launchpad: state must be 56 bytes');
    return { supply: readLe8(h, 0), x: readLe8(h, 8), y: readLe8(h, 16), root: h.slice(48) };
}

// ── Ledger ──────────────────────────────────────────────────────────────────

/** Owner id for a public key: the first 24 bytes of its hash. */
export const ownerOf = (pubkeyHex) => blake3_hash_hex(clean(pubkeyHex)).slice(0, 48);
export const leafHash = (balance, owner24) => blake3_hash_hex(le8(balance) + clean(owner24));
const H = (a, b) => blake3_hash_hex(a + b);

/**
 * Holder ledger: 2^depth rows. Untouched rows carry the launch tag as their
 * owner, so every launch starts from its own root and its own address.
 */
export class Ledger {
    constructor(depth, tag24) {
        this.depth = depth;
        this.size = 2 ** depth;
        this.tag = clean(tag24);
        if (!isHex(this.tag, 24)) throw new Error('launchpad: tag must be 24 bytes');
        this.owners = Array(this.size).fill(this.tag);
        this.balances = Array(this.size).fill(0n);
        const empty = leafHash(0n, this.tag);
        this.layers = [Array(this.size).fill(empty)];
        while (this.layers.at(-1).length > 1) {
            const cur = this.layers.at(-1), next = [];
            for (let i = 0; i < cur.length; i += 2) next.push(i && cur[i] === cur[0] && cur[i + 1] === cur[1] ? next[0] : H(cur[i], cur[i + 1]));
            this.layers.push(next);
        }
    }

    get root() { return this.layers.at(-1)[0]; }

    /** OP_MERKLE_ROOT proof: per level a 32-byte sibling, then 00 (we are left) or 01 (we are right). */
    proof(slot) {
        let out = '', i = slot;
        for (const layer of this.layers.slice(0, -1)) {
            const right = i % 2 === 1;
            out += layer[right ? i - 1 : i + 1] + (right ? '01' : '00');
            i >>= 1;
        }
        return out;
    }

    set(slot, balance, owner) {
        this.balances[slot] = BigInt(balance);
        this.owners[slot] = clean(owner);
        let i = slot;
        this.layers[0][i] = leafHash(this.balances[slot], this.owners[slot]);
        for (let l = 1; l < this.layers.length; l++) {
            i >>= 1;
            this.layers[l][i] = H(this.layers[l - 1][2 * i], this.layers[l - 1][2 * i + 1]);
        }
        return this;
    }

    /** Rows holding tokens for an owner. */
    rowsOf(owner) {
        const o = clean(owner), out = [];
        for (let i = 0; i < this.size; i++) if (this.owners[i] === o && this.balances[i] > 0n) out.push(i);
        return out;
    }
    slotOf(owner) { const r = this.rowsOf(owner); return r.length ? r[0] : -1; }
    /** The lowest empty row: where a new holder goes. */
    freeSlot() { return this.balances.findIndex((b) => b === 0n); }
    holders() {
        const out = [];
        for (let i = 0; i < this.size; i++) if (this.balances[i] > 0n) out.push({ slot: i, owner: this.owners[i], balance: this.balances[i] });
        return out.sort((a, b) => (b.balance > a.balance ? 1 : b.balance < a.balance ? -1 : a.slot - b.slot));
    }
    total() { return this.balances.reduce((a, b) => a + b, 0n); }
}

export function rootFromProof(leaf, proofHex) {
    const p = clean(proofHex);
    let acc = leaf;
    for (let l = 0; l * 66 < p.length; l++) {
        const sib = p.slice(l * 66, l * 66 + 64);
        acc = p.slice(l * 66 + 64, l * 66 + 66) === '01' ? H(sib, acc) : H(acc, sib);
    }
    return acc;
}

// ── Prices ──────────────────────────────────────────────────────────────────

/** Smallest pool x the contract accepts when n tokens leave a pool (x, y). */
export function poolBuyX(x, y, n) {
    x = BigInt(x); y = BigInt(y); n = BigInt(n);
    if (n <= 0n || n >= y) throw new Error('launchpad: the pool cannot sell that many');
    const y2 = y - n, k = x * y;
    const ok = (x2) => x2 >= x && (x2 - (x2 - x) * POOL_FEE_NUM / POOL_FEE_DEN) * y2 >= k;
    let x2 = x + ((k + y2 - 1n) / y2 - x) * POOL_FEE_DEN / (POOL_FEE_DEN - POOL_FEE_NUM);
    while (!ok(x2)) x2 += 1n;
    while (x2 > x && ok(x2 - 1n)) x2 -= 1n;
    return x2;
}

/** Smallest pool x the contract accepts when n tokens enter a pool (x, y). */
export function poolSellX(x, y, n) {
    x = BigInt(x); y = BigInt(y); n = BigInt(n);
    const y2 = y + n, eff = y2 - n * POOL_FEE_NUM / POOL_FEE_DEN, k = x * y;
    return (k + eff - 1n) / eff;
}

/**
 * The next state for trading n tokens, and what it costs.
 * @returns {{supply, x, y, mds}} mds > 0: paid in; mds < 0: paid out.
 */
export function quote(params, st, side, n) {
    const p = validateParams(params);
    n = BigInt(n);
    if (n < 0n) throw new Error('launchpad: negative amount');
    const { supply: s, x, y } = st;
    let s2 = s, x2 = x, y2 = y;
    if (s === p.threshold) {
        if (n === 0n) { /* hand-over */ }
        else if (side === 'buy') { y2 = y - n; x2 = poolBuyX(x, y, n); }
        else { y2 = y + n; x2 = poolSellX(x, y, n); }
    } else {
        s2 = side === 'buy' ? s + n : s - n;
        if (s2 < 0n) throw new Error('launchpad: the curve has not sold that many');
        if (s2 > p.threshold) throw new Error(`launchpad: only ${p.threshold - s} tokens are left on the curve`);
        x2 = curveReserve(p.price, s2);
    }
    return { supply: s2, x: x2, y: y2, mds: x2 - x };
}

/** Most tokens `mds` buys from the current state (0 if none). */
export function tokensFor(params, st, mds) {
    const p = validateParams(params);
    mds = BigInt(mds);
    let hi = st.supply === p.threshold ? st.y - 1n : p.threshold - st.supply, lo = 0n;
    while (lo < hi) {
        const mid = (lo + hi + 1n) / 2n;
        if (quote(p, st, 'buy', mid).mds <= mds) lo = mid; else hi = mid - 1n;
    }
    return lo;
}

/** Price of the next token, in MDS (curve: (s)·P; pool: x/y, as a fraction). */
export function spotPrice(params, st) {
    const p = validateParams(params);
    if (st.supply === p.threshold) return { num: st.x, den: st.y };
    return { num: st.supply * p.price, den: 1n };
}

/** Powers of two summing to v, largest first: a reserve is held in these. */
export function pow2Split(v) {
    const out = [];
    let x = BigInt(v);
    for (let b = 63n; b >= 0n; b--) if (x & (1n << b)) out.push(1n << b);
    return out;
}

// ── Announcement and registry ───────────────────────────────────────────────
//
// A launch transaction pays the seed to the genesis address, one unit to the
// registry, and carries one DataBurn:
//
//   "MLP1" | ver 1 | depth 1 | T u32 | P u64 | log2(seed) 1 | nonce 8 | ticker ‖ 00 ‖ name
//
// (big-endian, like the other announcement wires). The launch's tag is the
// first 24 bytes of the payload's hash, so the name, ticker and parameters
// are all bound into the genesis root, and so into every address the launch
// ever has. A copy announced under another name is a different launch.

export const ANN_MAGIC = '4d4c5031';  // "MLP1"
export const ANN_VER = 1;
export const ANN_HEAD_BYTES = 27;
export const MAX_BURN_BYTES = 80;

/** Bytecode nobody can spend (`PUSH tag DROP PUSH 0 VERIFY`). Its address marks launch blocks. */
export const REGISTRY_BYTECODE = '01' + '1500' + utf8Hex('midstate/launchpad/v1') + DROP + '01010000' + '21';
export const registryAddress = () => blake3_hash_hex(REGISTRY_BYTECODE);

export function encodeAnnouncement({ params, ticker, name, nonce }) {
    const p = validateParams(params);
    const t = String(ticker || '').trim(), nm = String(name || '').trim();
    if (!t || /\0/.test(t + nm)) throw new Error('launchpad: a ticker is required');
    const n = clean(nonce);
    if (!isHex(n, 8)) throw new Error('launchpad: nonce must be 8 bytes');
    const exp = p.seed.toString(2).length - 1;
    const hex = ANN_MAGIC + beHex(ANN_VER, 1) + beHex(p.depth, 1) + beHex(p.threshold, 4) + beHex(p.price, 8)
        + beHex(exp, 1) + n + utf8Hex(t) + '00' + utf8Hex(nm);
    if (hex.length / 2 > MAX_BURN_BYTES) throw new Error(`launchpad: ticker and name are ${hex.length / 2 - ANN_HEAD_BYTES - 1} bytes; at most ${MAX_BURN_BYTES - ANN_HEAD_BYTES - 1} fit`);
    return hex;
}

/** @returns {null|{params, ticker, name, nonce, tag, payload}} */
export function decodeAnnouncement(payloadHex) {
    const h = clean(payloadHex);
    if (h.slice(0, 8) !== ANN_MAGIC || h.length < (ANN_HEAD_BYTES + 2) * 2 || h.length > MAX_BURN_BYTES * 2) return null;
    if (parseInt(h.slice(8, 10), 16) !== ANN_VER) return null;
    try {
        const depth = parseInt(h.slice(10, 12), 16);
        const threshold = BigInt('0x' + h.slice(12, 20));
        const price = BigInt('0x' + h.slice(20, 36));
        const seed = 1n << BigInt(parseInt(h.slice(36, 38), 16));
        const nonce = h.slice(38, 54);
        const text = h.slice(54), z = text.match(/^((?:[0-9a-f]{2})*?)00/);
        if (!z) return null;
        const ticker = hexUtf8(z[1]), name = hexUtf8(text.slice(z[0].length));
        const params = validateParams({ depth, threshold, price, seed });
        return { params, ticker, name, nonce, tag: blake3_hash_hex(h).slice(0, 48), payload: h };
    } catch { return null; }
}

// ── A launch, and following it on chain ─────────────────────────────────────

const randomHex = (bytes) => {
    const a = new Uint8Array(bytes);
    globalThis.crypto.getRandomValues(a);
    return Array.from(a, (b) => b.toString(16).padStart(2, '0')).join('');
};

/**
 * One launch: its parameters, code, ledger, state and the coins holding its
 * reserve. Built from an announcement, then advanced trade by trade.
 */
export class Launch {
    /**
     * @param {Object} a decoded announcement
     * @param {string} code from {@link compileLaunch}
     */
    constructor(a, code) {
        this.params = validateParams(a.params);
        this.ticker = a.ticker; this.name = a.name; this.nonce = a.nonce; this.tag = a.tag;
        this.payload = a.payload;
        this.code = clean(code);
        this.ledger = new Ledger(this.params.depth, this.tag);
        this.state = encodeState({ supply: 0n, x: 0n, y: this.params.pool, root: this.ledger.root });
        this.genesisAddress = this.address;
        /** Coins at the current address that the next trade spends: [{coinId, value, salt}] */
        this.coins = [];
        this.trades = [];
        this.created = null;
    }

    get address() { return addressFor(this.code, this.state); }
    get bytecode() { return bytecodeFor(this.code, this.state); }
    get st() { return decodeState(this.state); }
    get graduated() { return this.st.supply === this.params.threshold; }
    get reserve() { return this.coins.reduce((a, c) => a + BigInt(c.value), 0n); }

    /**
     * Outputs and burn for the launch transaction (wallet `fund` shape).
     * @returns {{fundings, burns, genesisAddress}}
     */
    launchOutputs({ salt = randomHex(32) } = {}) {
        return {
            fundings: [
                { address: this.genesisAddress, amount: Number(this.params.seed), salt },
                { address: registryAddress(), amount: 1 },
            ],
            burns: [this.payload],
            genesisAddress: this.genesisAddress,
        };
    }

    /**
     * Build a trade for the wallet to sign and send.
     *
     * The witness is complete except for the signature: item SIG_ITEM is a
     * zero placeholder of `sigBytes`, which the wallet replaces with its
     * site key's signature over the transaction commitment (the commitment
     * does not cover witnesses, so it can be known before they are final).
     *
     * @param {Object} o
     * @param {'buy'|'sell'} o.side
     * @param {bigint} o.n          tokens (0 with `to` hands the row over)
     * @param {string} o.pubkey     the trader's site-key public key (32 bytes)
     * @param {number} o.sigBytes   length of that key's signatures
     * @param {string} [o.to]       new owner id for the row (defaults to the trader)
     * @param {number} [o.slot]     row to use (defaults: the trader's row, else the first empty one)
     */
    prepareTrade({ side, n, pubkey, sigBytes, to = null, slot = null }) {
        if (!this.coins.length) throw new Error('launchpad: this launch has no reserve coin to spend yet');
        if (this.reserve !== this.st.x + this.params.seed) throw new Error('launchpad: the coins being tracked do not add up to the reserve');
        n = BigInt(n);
        const me = ownerOf(pubkey);
        if (slot === null) {
            slot = this.ledger.slotOf(me);
            if (slot < 0) {
                if (side === 'sell') throw new Error('launchpad: you hold none of this token');
                slot = this.ledger.freeSlot();
                if (slot < 0) throw new Error('launchpad: every row of this ledger is taken');
            }
        }
        const oldBal = this.ledger.balances[slot], oldOwner = this.ledger.owners[slot];
        if (oldBal > 0n && oldOwner !== me) throw new Error('launchpad: that row belongs to someone else');
        const st = this.st;
        const q = quote(this.params, st, side, n);
        const newBal = side === 'sell' ? oldBal - n : oldBal + n;
        if (newBal < 0n) throw new Error(`launchpad: you hold ${oldBal}`);
        const owner = to ? clean(to) : me;
        if (!isHex(owner, 24)) throw new Error('launchpad: an owner id is 24 bytes');
        const proof = this.ledger.proof(slot);
        const root = rootFromProof(leafHash(newBal, owner), proof);
        const next = encodeState({ supply: q.supply, x: q.x, y: q.y, root });
        const nextAddress = addressFor(this.code, next);
        const receipt = owner + le8(newBal);
        const [codeA, codeB] = splitCode(this.code);
        const witness = [codeA, codeB, this.state, proof, oldOwner, owner, clean(pubkey), le8(oldBal), le8(newBal),
                         le8(q.supply), le8(q.x), le8(q.y), '00'.repeat(sigBytes)];
        witness.forEach((w, i) => {
            if (i !== SIG_ITEM && w.length / 2 > MAX_PLAIN_ITEM) throw new Error(`launchpad: witness item ${i} is ${w.length / 2} bytes`);
        });
        const w = witness.join(',');
        const reserve = pow2Split(q.x + this.params.seed);
        if (reserve.some((v) => v > BigInt(Number.MAX_SAFE_INTEGER))) throw new Error('launchpad: a reserve coin is too large for the wallet');
        const outputs = [
            { out_type: 'confidential', address: nextAddress, value: 0, state: receipt, salt: randomHex(32) },
            ...reserve.map((v) => ({ out_type: 'standard', address: nextAddress, value: Number(v), salt: randomHex(32) })),
        ];
        return {
            spend: {
                bytecode: this.bytecode,
                contractAddress: this.address,
                inputs: this.coins.map((c) => ({ coinId: c.coinId, value: Number(c.value), salt: c.salt, witness: w })),
                outputs,
                siteSig: { item: SIG_ITEM, pk: clean(pubkey) },
            },
            slot, owner, oldBal, newBal, next, nextAddress, quote: q, witness,
        };
    }

    /**
     * Advance over one block transaction, if it spends this launch's current
     * state. Works on blocks with their witnesses stripped.
     *
     * @param {Object} tx a block transaction (`{Reveal: …}` / `{Consolidate: …}` or already normalised)
     * @param {number} [height]
     * @returns {null|Object} the trade, or null if the tx does not touch this launch
     */
    applyTx(tx, height = null) {
        const t = normalizeTx(tx);
        if (!t) return null;
        const here = this.address;
        const spent = t.inputs.filter((i) => i.address === here);
        if (!spent.length) return null;
        if (splitBytecode(spent[0].bytecode).state !== this.state) throw new Error('launchpad: replay is out of step with the chain');
        const r = t.outputs[0];
        if (!r || r.type !== 'confidential') throw new Error('launchpad: a trade without a receipt was accepted — this is not a launch script');
        const nextAddress = r.address, owner = r.commitment.slice(0, 48), newBal = readLe8(r.commitment, 24);
        const x2 = t.outputs.filter((o) => o.type === 'standard' && o.address === nextAddress)
            .reduce((a, o) => a + BigInt(o.value), 0n) - this.params.seed;
        const st = this.st, T = this.params.threshold;

        // The receipt names the owner and balance, not the row. Try the rows the
        // page itself would have used first, then every row.
        // A hand-over keeps the balance, so rows already holding it come next.
        const tried = new Set(), order = [...this.ledger.rowsOf(owner), this.ledger.freeSlot()];
        if (newBal > 0n) this.ledger.balances.forEach((b, i) => { if (b === newBal) order.push(i); });
        const attempt = (slot) => {
            if (slot < 0 || tried.has(slot)) return null;
            tried.add(slot);
            const oldBal = this.ledger.balances[slot];
            let s2 = st.supply, y2 = st.y;
            if (st.supply === T) {
                y2 = st.y + oldBal - newBal;
                if (y2 < 1n) return null;
            } else {
                s2 = st.supply + newBal - oldBal;
                if (s2 < 0n || s2 > T || curveReserve(this.params.price, s2) !== x2) return null;
            }
            const root = rootFromProof(leafHash(newBal, owner), this.ledger.proof(slot));
            const next = encodeState({ supply: s2, x: x2, y: y2, root });
            return addressFor(this.code, next) === nextAddress ? { slot, oldBal, next } : null;
        };
        let hit = null;
        for (const s of order) if ((hit = attempt(s))) break;
        for (let s = 0; !hit && s < this.ledger.size; s++) hit = attempt(s);
        if (!hit) throw new Error('launchpad: could not work out the next state from this trade');

        const before = st, oldOwner = this.ledger.owners[hit.slot];
        this.ledger.set(hit.slot, newBal, owner);
        this.state = hit.next;
        this.coins = t.outputs.filter((o) => o.type === 'standard' && o.address === nextAddress)
            .map((o) => ({ coinId: compute_coin_id_hex(nextAddress, BigInt(o.value), o.salt), value: BigInt(o.value), salt: o.salt }));
        const after = this.st;
        const trade = {
            height, slot: hit.slot, owner, oldOwner, oldBal: hit.oldBal, newBal,
            tokens: newBal - hit.oldBal,                 // + bought, - sold, 0 handed over
            mds: after.x - before.x,                     // + paid in, - paid out
            phase: before.supply === T ? 'pool' : 'curve',
            graduated: before.supply !== T && after.supply === T,
            supply: after.supply, x: after.x, y: after.y,
        };
        this.trades.push(trade);
        return trade;
    }
}

/**
 * Find launches in a block transaction: a DataBurn announcement plus the seed
 * at the address it implies.
 *
 * @param {Object} tx block transaction
 * @param {Function} codeFor (params) => code, usually memoised compileLaunch
 * @returns {Launch[]}
 */
export function launchesInTx(tx, codeFor, height = null) {
    const t = normalizeTx(tx);
    if (!t) return [];
    const out = [];
    for (const o of t.outputs) {
        if (o.type !== 'burn') continue;
        const a = decodeAnnouncement(o.payload);
        if (!a) continue;
        const L = new Launch(a, codeFor(a.params));
        const seed = t.outputs.filter((s) => s.type === 'standard' && s.address === L.genesisAddress);
        if (seed.reduce((v, s) => v + BigInt(s.value), 0n) !== L.params.seed) continue;
        L.coins = seed.map((s) => ({ coinId: compute_coin_id_hex(s.address, BigInt(s.value), s.salt), value: BigInt(s.value), salt: s.salt }));
        L.created = height;
        out.push(L);
    }
    return out;
}

// ── Block transactions ──────────────────────────────────────────────────────

/** Hex from a hex string or a serde byte array. */
export function toHex(v) {
    if (v == null) return null;
    if (typeof v === 'string') return clean(v);
    if (Array.isArray(v) || v instanceof Uint8Array) return Array.from(v, (b) => Number(b).toString(16).padStart(2, '0')).join('');
    throw new Error('launchpad: expected bytes');
}

/**
 * A block transaction in one shape:
 * `{inputs: [{address, bytecode, value, salt, commitment}], outputs: [{type, address, value, salt, commitment, payload}]}`.
 * Commits and coinbase return null.
 */
export function normalizeTx(tx) {
    if (!tx) return null;
    if (Array.isArray(tx.inputs) && Array.isArray(tx.outputs) && tx._normalised) return tx;
    const body = tx.Reveal || tx.reveal || tx.Consolidate || tx.consolidate || (tx.inputs && tx.outputs ? tx : null);
    if (!body) return null;
    const inputs = (body.inputs || []).map((i) => {
        const bytecode = toHex(i.predicate?.Script?.bytecode ?? i.predicate?.script?.bytecode ?? i.bytecode);
        return { address: blake3_hash_hex(bytecode), bytecode, value: BigInt(i.value ?? 0), salt: toHex(i.salt), commitment: toHex(i.commitment) };
    });
    const outputs = (body.outputs || []).map((o) => {
        const s = o.Standard || o.standard, c = o.Confidential || o.confidential, d = o.DataBurn || o.data_burn;
        if (s) return { type: 'standard', address: toHex(s.address), value: BigInt(s.value), salt: toHex(s.salt) };
        if (c) return { type: 'confidential', address: toHex(c.address), value: 0n, salt: toHex(c.salt), commitment: toHex(c.commitment) };
        if (d) return { type: 'burn', payload: toHex(d.payload), value: BigInt(d.value_burned ?? 0) };
        if (o.type === 'standard') return { type: 'standard', address: clean(o.address), value: BigInt(o.value), salt: clean(o.salt) };
        if (o.type === 'confidential') return { type: 'confidential', address: clean(o.address), value: 0n, salt: clean(o.salt), commitment: clean(o.commitment ?? o.state) };
        if (o.type === 'data_burn') return { type: 'burn', payload: clean(o.payload), value: BigInt(o.value_burned ?? 0) };
        return { type: 'unknown' };
    });
    return { inputs, outputs, _normalised: true };
}

/** Every address a launch has watched so far is enough for compact filters: current address only. */
export function watchAddresses(launches) {
    return [registryAddress(), ...launches.map((l) => l.address)];
}
