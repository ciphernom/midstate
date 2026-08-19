// dex.js — On-chain DEX order announcements and the limit-order covenant flow.
//
// ═══════════════════════════════════════════════════════════════════════════
//  Reasoning (module level)
// ═══════════════════════════════════════════════════════════════════════════
//
// Ported from `core/dex.rs`, which is itself the canonical port of the browser
// wallet, so all three read and write the same order book. Every integer here is
// **big-endian**, unlike the Q-Bolt channel wire and unlike the bincode codec in
// `bincode.js`, both of which are little-endian. Mixing them produces
// announcements that decode to plausible garbage rather than failing, which is
// why this lives in its own module and why the tests below assert byte layouts
// against hand-built vectors rather than round-tripping only against themselves.
//
// ── Why announcements exist ──
//
// A swap's coin salt is the one piece of state a wallet cannot re-derive from
// its seed. Lose it and the funds are stranded even though the key is intact.
// Publishing the salt — never the preimage — in a zero-value `DataBurn` makes
// every order recoverable from seed alone. That is the entire purpose: this is a
// recovery channel first and an order book second.
//
// ── Why fragmentation exists ──
//
// Consensus caps a burn payload at `MAX_BURN_DATA_SIZE` = 80 bytes. A
// self-contained MDXA is a 72-byte header plus 81 bytes per unit, so it can
// never fit in one burn. An announcement is therefore split into MDXF fragments
// and all of them ride as separate burns inside the SAME funding transaction, so
// they land in one block and reassemble trivially. The pool is nonetheless
// tolerant of fragments arriving apart, because a reorg or a partial scan can
// split them.
//
// ── One order, one secret, one coin ──
//
// A single hashlock backs exactly one trustless fill: the maker reveals the
// preimage to claim the counter-leg, and from that moment H is public. So a
// "large order" is N separate units, each with its own fresh secret and its own
// power-of-two covenant coin. Unit values are powers of two because a coin is,
// and because `max_claim` equal to the full unit value means the whole coin is
// claimed atomically with no remainder to drain.

import { blake3_hash_hex, build_limit_order_covenant_bytecode_hex } from '../pkg/wasm_wallet.js';

export const ANN_MAGIC = '4d445841';   // "MDXA" — maker order
export const FRAG_MAGIC = '4d445846';  // "MDXF" — fragment
export const TAKER_MAGIC = '4d445854'; // "MDXT" — taker lock
export const ANN_VER = 1;
export const TAKER_VER = 1;

/** Consensus cap on a burn payload (`MAX_BURN_DATA_SIZE`). Do not raise. */
export const MAX_BURN_DATA_SIZE = 80;
/** magic 4 + groupId 6 + idx 1 + total 1 */
export const FRAG_HEADER_BYTES = 12;
export const FRAG_PAYLOAD_BYTES = MAX_BURN_DATA_SIZE - FRAG_HEADER_BYTES; // 68

/** Chat opcodes the wallets use for fast (non-durable) order gossip. */
export const CHAT_OPCODES = {
    OFFER:     [255, 200],
    ACCEPT:    [255, 201],
    LOCKING:   [255, 202],
    LOCKED:    [255, 203],
    SUBMARINE: [255, 204],
    BIDFILL:   [255, 205],
    BIDSECRET: [255, 206],
};

// ── byte helpers (big-endian) ───────────────────────────────────────────────

const hexToBytes = (h) => {
    const s = String(h || '').replace(/^0x/, '').toLowerCase();
    if (s.length % 2) throw new Error('dex: odd-length hex');
    const a = new Uint8Array(s.length / 2);
    for (let i = 0; i < a.length; i++) a[i] = parseInt(s.substr(i * 2, 2), 16);
    return a;
};
const bytesToHex = (b) => Array.from(b).map((x) => x.toString(16).padStart(2, '0')).join('');
const beBytes = (v, n) => {
    const a = new Uint8Array(n);
    let x = BigInt(v);
    for (let i = n - 1; i >= 0; i--) { a[i] = Number(x & 0xffn); x >>= 8n; }
    return a;
};
const beRead = (a, o, n) => {
    let v = 0n;
    for (let i = 0; i < n; i++) v = (v << 8n) | BigInt(a[o + i]);
    return v;
};
const concat = (...parts) => {
    const len = parts.reduce((s, p) => s + p.length, 0);
    const out = new Uint8Array(len);
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
};

/**
 * Exact base-2 logarithm.
 *
 * Unit values are stored on the wire as a single exponent byte, so a value that
 * is not a power of two cannot be represented at all. Failing here rather than
 * rounding is deliberate: a silently rounded value produces an announcement
 * whose coin cannot be found, which looks like a lost order.
 */
export function log2Exact(v) {
    let n = BigInt(v), e = 0;
    if (n <= 0n) throw new Error(`dex: unit value ${v} must be positive`);
    while (n > 1n) {
        if (n & 1n) throw new Error(`dex: unit value ${v} is not a power of two`);
        n >>= 1n; e++;
    }
    return e;
}

// ── MDXA: maker orders ──────────────────────────────────────────────────────

/**
 * Encode a maker announcement: 72-byte header + 81 bytes per unit.
 *
 * Layout (all big-endian):
 *   MAGIC(4) VER(1) makerEvmAddr(20) makerMdsPk(32) timeoutHeight(8)
 *   groupId(6) unitCount(1), then per unit:
 *   secretHash(32) salt(32) valueExponent(1) weiAmount(16)
 *
 * The covenant address and coin id are NOT stored: a reader recomputes them
 * from (secretHash, value, timeoutHeight, makerMdsPk), so putting them on the
 * wire would only create a second, disagreeable source of truth.
 */
export function encodeMakerAnnouncement({ makerEvmAddr, makerMdsPk, timeoutHeight, groupId, units }) {
    if (!Array.isArray(units) || units.length === 0 || units.length > 255) {
        throw new Error('dex: unit count out of range (1..=255)');
    }
    const parts = [
        hexToBytes(ANN_MAGIC),
        Uint8Array.of(ANN_VER),
        hexToBytes(makerEvmAddr),
        hexToBytes(makerMdsPk),
        beBytes(timeoutHeight, 8),
        hexToBytes(String(groupId || '').padStart(12, '0')).slice(0, 6),
        Uint8Array.of(units.length),
    ];
    for (const u of units) {
        parts.push(hexToBytes(u.secretHash), hexToBytes(u.salt),
            Uint8Array.of(log2Exact(u.value)), beBytes(u.weiAmount, 16));
    }
    return bytesToHex(concat(...parts));
}

/** @returns {Object|null} Decoded maker announcement, or null if this isn't one. */
export function decodeMakerAnnouncement(hexOrBytes) {
    const b = typeof hexOrBytes === 'string' ? hexToBytes(hexOrBytes) : hexOrBytes;
    let o = 0;
    const need = (n) => { if (o + n > b.length) throw new RangeError('short'); };
    try {
        need(4); if (bytesToHex(b.subarray(0, 4)) !== ANN_MAGIC) return null; o += 4;
        need(1); if (b[o++] !== ANN_VER) return null;
        need(20); const makerEvmAddr = bytesToHex(b.subarray(o, o + 20)); o += 20;
        need(32); const makerMdsPk = bytesToHex(b.subarray(o, o + 32)); o += 32;
        need(8); const timeoutHeight = beRead(b, o, 8); o += 8;
        need(6); const groupId = bytesToHex(b.subarray(o, o + 6)); o += 6;
        need(1); const n = b[o++];
        if (n === 0) return null;

        const units = [];
        for (let i = 0; i < n; i++) {
            need(32); const secretHash = bytesToHex(b.subarray(o, o + 32)); o += 32;
            need(32); const salt = bytesToHex(b.subarray(o, o + 32)); o += 32;
            need(1); const exp = b[o++];
            if (exp >= 64) return null;      // would overflow a u64 coin value
            need(16); const weiAmount = beRead(b, o, 16); o += 16;
            units.push({ secretHash, salt, value: 1n << BigInt(exp), weiAmount });
        }
        return { makerEvmAddr, makerMdsPk, timeoutHeight, groupId, units };
    } catch {
        return null;
    }
}

// ── MDXT: taker locks ───────────────────────────────────────────────────────

export function encodeTakerAnnouncement({ takerMdsPk, secretHash, salt, receiverAddr, timeoutHeight, value, weiAmount }) {
    return bytesToHex(concat(
        hexToBytes(TAKER_MAGIC), Uint8Array.of(TAKER_VER),
        hexToBytes(takerMdsPk), hexToBytes(secretHash), hexToBytes(salt), hexToBytes(receiverAddr),
        beBytes(timeoutHeight, 8), Uint8Array.of(log2Exact(value)), beBytes(weiAmount, 16),
    ));
}

/** @returns {Object|null} */
export function decodeTakerAnnouncement(hexOrBytes) {
    const b = typeof hexOrBytes === 'string' ? hexToBytes(hexOrBytes) : hexOrBytes;
    let o = 0;
    const need = (n) => { if (o + n > b.length) throw new RangeError('short'); };
    try {
        need(4); if (bytesToHex(b.subarray(0, 4)) !== TAKER_MAGIC) return null; o += 4;
        need(1); if (b[o++] !== TAKER_VER) return null;
        need(32); const takerMdsPk = bytesToHex(b.subarray(o, o + 32)); o += 32;
        need(32); const secretHash = bytesToHex(b.subarray(o, o + 32)); o += 32;
        need(32); const salt = bytesToHex(b.subarray(o, o + 32)); o += 32;
        need(32); const receiverAddr = bytesToHex(b.subarray(o, o + 32)); o += 32;
        need(8); const timeoutHeight = beRead(b, o, 8); o += 8;
        need(1); const exp = b[o++];
        if (exp >= 64) return null;
        need(16); const weiAmount = beRead(b, o, 16); o += 16;
        return { takerMdsPk, secretHash, salt, receiverAddr, timeoutHeight, value: 1n << BigInt(exp), weiAmount };
    } catch {
        return null;
    }
}

// ── MDXF: fragmentation ─────────────────────────────────────────────────────

/**
 * Split an encoded announcement into burn-sized fragments.
 *
 * All fragments MUST be published in the same transaction. They are separate
 * zero-value burns, which consensus already permits, so this needs no protocol
 * change — but split across transactions they can land in different blocks, or
 * one can be dropped, leaving the group unreassemblable.
 */
export function fragment(bodyHex, groupId) {
    const g6 = bytesToHex(hexToBytes(String(groupId || '').padStart(12, '0')).slice(0, 6));
    const body = String(bodyHex).toLowerCase();
    const step = FRAG_PAYLOAD_BYTES * 2;
    const total = Math.max(1, Math.ceil(body.length / step));
    if (total > 255) throw new Error(`dex: announcement needs ${total} fragments (max 255)`);
    const out = [];
    for (let i = 0; i < total; i++) {
        out.push(FRAG_MAGIC + g6
            + i.toString(16).padStart(2, '0')
            + total.toString(16).padStart(2, '0')
            + body.slice(i * step, (i + 1) * step));
    }
    return out;
}

/** @returns {{groupId:string, idx:number, total:number, chunk:string}|null} */
export function parseFragment(hexOrBytes) {
    const hex = (typeof hexOrBytes === 'string' ? hexOrBytes : bytesToHex(hexOrBytes))
        .replace(/^0x/, '').toLowerCase();
    if (hex.length <= FRAG_HEADER_BYTES * 2 || hex.slice(0, 8) !== FRAG_MAGIC) return null;
    const groupId = hex.slice(8, 20);
    const idx = parseInt(hex.slice(20, 22), 16);
    const total = parseInt(hex.slice(22, 24), 16);
    if (!total || idx >= total) return null;
    return { groupId, idx, total, chunk: hex.slice(24) };
}

/**
 * Accumulates fragments until a group is complete.
 *
 * Fragments from one announcement normally arrive together, but a reorg or a
 * partial scan can split them, so this is deliberately tolerant. It is also
 * bounded: a long-running scan would otherwise grow without limit on fragments
 * whose siblings never arrive.
 */
export class FragmentPool {
    constructor(maxGroups = 512) {
        this.groups = new Map();
        this.maxGroups = maxGroups;
    }

    /** @returns {string|null} The reassembled body hex once every piece is in. */
    add(frag) {
        const key = `${frag.groupId}:${frag.total}`;
        let slots = this.groups.get(key);
        if (!slots) {
            if (this.groups.size >= this.maxGroups) {
                // Drop the oldest incomplete group rather than grow unbounded.
                this.groups.delete(this.groups.keys().next().value);
            }
            slots = new Array(frag.total).fill(null);
            this.groups.set(key, slots);
        }
        slots[frag.idx] = frag.chunk;
        if (slots.some((s) => s === null)) return null;
        this.groups.delete(key);
        return slots.join('');
    }

    get pending() { return this.groups.size; }
    clear() { this.groups.clear(); }
}

/**
 * Classify one burn payload.
 *
 * Order matters: a taker announcement is tried before a maker one because both
 * are fixed-prefix formats and only the magic distinguishes them, and fragments
 * are tried last because a fragment's chunk can begin with anything at all.
 *
 * @returns {{kind:'maker'|'taker', value:Object}|null}
 */
export function ingest(payloadHex, pool) {
    const t = decodeTakerAnnouncement(payloadHex);
    if (t) return { kind: 'taker', value: t };
    const m = decodeMakerAnnouncement(payloadHex);
    if (m) return { kind: 'maker', value: m };

    const f = parseFragment(payloadHex);
    if (f && pool) {
        const body = pool.add(f);
        if (body) {
            const tm = decodeTakerAnnouncement(body);
            if (tm) return { kind: 'taker', value: tm };
            const mm = decodeMakerAnnouncement(body);
            if (mm) return { kind: 'maker', value: mm };
        }
    }
    return null;
}

// ── Covenant derivation ─────────────────────────────────────────────────────

/**
 * Rebuild a limit-order covenant address from its announced parameters.
 *
 * # Reasoning
 *
 * This is what makes an announcement self-sufficient. A reader has
 * (secretHash, value, timeoutHeight, makerMdsPk) from the wire and derives the
 * address itself, so a maker cannot announce one set of terms and fund a
 * different covenant — the address either matches a real funded coin or it does
 * not.
 *
 * `maxClaim` is the full unit value so the whole coin is claimed atomically
 * with no remainder, which is also why unit values must be powers of two.
 */
export function deriveCovenantAddress({ secretHash, value, timeoutHeight, makerMdsPk }) {
    const script = build_limit_order_covenant_bytecode_hex(
        String(secretHash).replace(/^0x/, ''),
        BigInt(value),
        BigInt(timeoutHeight),
        String(makerMdsPk).replace(/^0x/, ''),
    );
    return { script, address: blake3_hash_hex(script) };
}

/**
 * Expand an announcement into resolvable orders.
 *
 * Each unit becomes one order with its covenant address and coin id derived
 * locally, so a caller can look the coin up on-chain and confirm the order is
 * actually funded before acting on it.
 *
 * @param {Object} ann Decoded maker announcement.
 * @param {Function} computeCoinId `(addressHex, value, saltHex) => coinIdHex`.
 */
export function expandOrders(ann, computeCoinId) {
    return ann.units.map((u, i) => {
        const { script, address } = deriveCovenantAddress({
            secretHash: u.secretHash,
            value: u.value,
            timeoutHeight: ann.timeoutHeight,
            makerMdsPk: ann.makerMdsPk,
        });
        return {
            groupId: ann.groupId,
            unitIndex: i,
            makerEvmAddr: ann.makerEvmAddr,
            makerMdsPk: ann.makerMdsPk,
            timeoutHeight: ann.timeoutHeight,
            secretHash: u.secretHash,
            salt: u.salt,
            value: u.value,
            weiAmount: u.weiAmount,
            covenantScript: script,
            covenantAddress: address,
            coinId: computeCoinId ? computeCoinId(address, u.value, u.salt) : null,
            // Price in wei per base unit, for sorting an order book.
            priceWeiPerUnit: u.value > 0n ? Number(u.weiAmount) / Number(u.value) : Infinity,
        };
    });
}

/**
 * Pull every DataBurn payload out of a block, whatever shape serde produced.
 *
 * `LightRequest::GetBlock` serializes core types with `derive(Serialize)`:
 * externally-tagged enums with `Vec<u8>` as JSON **number arrays** —
 * `{"DataBurn":{"payload":[77,68,...]}}` — which no hex-run regex will ever
 * match. So this walks the object tree and accepts either shape.
 */
export function extractBurnPayloads(node, out = []) {
    if (!node || typeof node !== 'object') return out;
    const push = (p) => {
        if (Array.isArray(p)) out.push(bytesToHex(Uint8Array.from(p)));
        else if (typeof p === 'string') out.push(p.replace(/^0x/, '').toLowerCase());
    };
    if (node.DataBurn && node.DataBurn.payload !== undefined) push(node.DataBurn.payload);
    if (node.data_burn && node.data_burn.payload !== undefined) push(node.data_burn.payload);
    if (node.type === 'data_burn' && node.payload !== undefined) push(node.payload);
    for (const k of Object.keys(node)) {
        const v = node[k];
        if (v && typeof v === 'object') extractBurnPayloads(v, out);
    }
    return out;
}

/**
 * Generate a fresh secret and its hash.
 *
 * The MAKER generates the secret, matching the counter-chain contract's
 * protocol. One secret backs exactly one unit: reusing it across units would
 * let the first fill unlock all of them.
 */
export function newSecret() {
    const bytes = new Uint8Array(32);
    (globalThis.crypto || require('node:crypto').webcrypto).getRandomValues(bytes);
    const secret = bytesToHex(bytes);
    return { secret, secretHash: blake3_hash_hex(secret) };
}

/** A random 6-byte group id, linking the units of one announcement. */
export function newGroupId() {
    const b = new Uint8Array(6);
    (globalThis.crypto || require('node:crypto').webcrypto).getRandomValues(b);
    return bytesToHex(b);
}
