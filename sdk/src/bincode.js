// bincode.js — Wire codec for the Midstate native P2P protocol (/midstate/2.0.0).
//
// The node serializes `Message` with `bincode::DefaultOptions::new()`, which is
// bincode's *varint* configuration: every integer, every collection length and
// every enum discriminant is varint-encoded. This module implements exactly
// that encoding and nothing else.
//
// It is deliberately free of transport dependencies. The wire format is the
// part most likely to drift against the node and the part most worth testing,
// and it should not require a libp2p install to exercise. Everything here is a
// pure function over bytes.
//
// Reference: node src/network/protocol.rs (`Message`) and src/chat.rs
// (`ChatAttachment`, hand-written Serialize/Deserialize impls).

// ── Varint ───────────────────────────────────────────────────────────────────
//
// bincode varint layout:
//   0..=250  → that single byte
//   251      → u16 little-endian follows
//   252      → u32 little-endian follows
//   253      → u64 little-endian follows
//   254      → u128 (not produced by this protocol)

const SINGLE_BYTE_MAX = 250;

export function encodeVarint(n) {
    const v = typeof n === 'bigint' ? n : BigInt(n);
    if (v < 0n) throw new Error('varint must be non-negative');
    if (v <= 250n) return Uint8Array.of(Number(v));
    if (v < (1n << 16n)) { const b = new Uint8Array(3); b[0] = 251; new DataView(b.buffer).setUint16(1, Number(v), true); return b; }
    if (v < (1n << 32n)) { const b = new Uint8Array(5); b[0] = 252; new DataView(b.buffer).setUint32(1, Number(v), true); return b; }
    if (v < (1n << 64n)) { const b = new Uint8Array(9); b[0] = 253; new DataView(b.buffer).setBigUint64(1, v, true); return b; }
    throw new Error('varint too large');
}

export function decodeVarint(buf, off = 0) {
    const first = buf[off];
    if (first === undefined) throw new RangeError('varint: out of bytes');
    if (first <= SINGLE_BYTE_MAX) return { value: BigInt(first), size: 1 };
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    // Each multi-byte form must fit entirely inside the buffer; a truncated
    // frame would otherwise read adjacent heap bytes as protocol data.
    if (first === 251) { if (off + 3 > buf.length) throw new RangeError('varint: truncated u16'); return { value: BigInt(dv.getUint16(off + 1, true)), size: 3 }; }
    if (first === 252) { if (off + 5 > buf.length) throw new RangeError('varint: truncated u32'); return { value: BigInt(dv.getUint32(off + 1, true)), size: 5 }; }
    if (first === 253) { if (off + 9 > buf.length) throw new RangeError('varint: truncated u64'); return { value: dv.getBigUint64(off + 1, true), size: 9 }; }
    throw new Error('varint invalid');
}

// ── Framing ──────────────────────────────────────────────────────────────────
//
// Both the native and light protocols frame payloads as a 4-byte little-endian
// length followed by the payload itself.

export function encodeBinaryFrame(payload) {
    const msg = new Uint8Array(4 + payload.length);
    new DataView(msg.buffer).setUint32(0, payload.length, true);
    msg.set(payload, 4);
    return msg;
}

export function encodeFrame(obj) {
    return encodeBinaryFrame(new TextEncoder().encode(JSON.stringify(obj)));
}

// ── Message discriminants ────────────────────────────────────────────────────
//
// Index into `enum Message` in src/network/protocol.rs. The node's own comment
// warns that reordering shifts every discriminant and breaks the whole
// protocol, so these are stable by contract.

export const MSG_GET_ADDR = 5;
export const MSG_ADDR = 6;
export const MSG_PONG = 8;
export const MSG_CHAT_V2 = 19;

export function encodeGetAddr() { return encodeVarint(MSG_GET_ADDR); }

/** `Message::Pong { nonce }` — the ACK for an inbound native-protocol message. */
export function encodePong(nonce = 0) {
    const disc = encodeVarint(MSG_PONG);
    const n = encodeVarint(nonce);
    const out = new Uint8Array(disc.length + n.length);
    out.set(disc, 0); out.set(n, disc.length);
    return out;
}

export function decodeAddr(payload) {
    let off = 0;
    const disc = decodeVarint(payload, off); off += disc.size;
    if (disc.value !== BigInt(MSG_ADDR)) throw new Error('expected Addr discriminant');
    const count = decodeVarint(payload, off); off += count.size;
    // The node caps Addr at 1000 entries in deserialize_bin; refuse anything
    // claiming more rather than trying to allocate for it.
    if (count.value > 1000n) throw new Error(`Addr count ${count.value} exceeds max 1000`);
    const out = [];
    const dec = new TextDecoder();
    for (let i = 0; i < Number(count.value); i++) {
        const len = decodeVarint(payload, off); off += len.size;
        const n = Number(len.value);
        // A truncated payload must be rejected, not silently decoded into a
        // short string. subarray() clamps rather than throwing, so without this
        // check a peer could send a length that overruns the buffer and get a
        // quietly wrong multiaddr back — which then gets dialed.
        if (off + n > payload.length) {
            throw new RangeError(`Addr entry ${i} claims ${n} bytes, only ${payload.length - off} remain`);
        }
        out.push(dec.decode(payload.subarray(off, off + n))); off += n;
    }
    return out;
}

// ── ChatAttachment ───────────────────────────────────────────────────────────
//
// Mirrors the hand-written `impl Serialize for ChatAttachment` in src/chat.rs —
// specifically the `BincodeHelper` branch, taken whenever the serializer is not
// human-readable. Variant order there IS the wire tag, and chat.rs documents
// the tags as immutable, so this table is the contract:
//
//   tag 0 address            [u8; 32]
//   tag 1 coin_id            [u8; 32]
//   tag 2 mix_id             [u8; 32]
//   tag 3 commitment         [u8; 32]
//   tag 4 block_hash         [u8; 32]
//   tag 5 midstate           [u8; 32]
//   tag 6 data_hash          [u8; 32]
//   tag 7 license_challenge  [u8; 32] ⌢ varint(u64 height) ⌢ [u8; 32]
//   tag 8 signature          varint(len) ⌢ len bytes
//
// Three things were wrong in the previous decoder, and because attachments are
// parsed in sequence each one corrupted every attachment that followed it
// rather than only the one it mis-read:
//
//  1. Signature was read at tag 9. There is no tag 9 — Signature is the ninth
//     variant but the eighth index, so a real signature fell through to the
//     catch-all and was dropped.
//  2. Its length was read as a fixed 4-byte LE u32. bincode varint-encodes
//     collection lengths, so a 576-byte WOTS signature is `251 40 02`.
//  3. The catch-all advanced a flat 32 bytes, but license_challenge is
//     32 + varint + 32.
//
// Unknown future tags remain genuinely unskippable — their width is unknowable
// — so decoding stops cleanly and says so rather than inventing a width.

const ATTACHMENT_KINDS_32 = [
    'address', 'coin_id', 'mix_id', 'commitment',
    'block_hash', 'midstate', 'data_hash',
];
const ATT_TAG_LICENSE_CHALLENGE = 7n;
const ATT_TAG_SIGNATURE = 8n;

/** Max attachments per message, from MAX_CHAT_ATTACHMENTS in the node. */
export const MAX_CHAT_ATTACHMENTS = 4;

const toHex = (bytes) => Array.from(bytes).map((b) => b.toString(16).padStart(2, '0')).join('');

/**
 * Decode a bincode `Vec<ChatAttachment>` beginning at `offset`.
 *
 * @param {Uint8Array} payload
 * @param {number} offset - Index of the attachment-count varint.
 * @returns {{attachments: Object[], offset: number, truncated: boolean}}
 *   `offset` is the first byte past the attachments. `truncated` is true when
 *   decoding stopped early (unknown tag, or the buffer ran short).
 */
export function decodeChatAttachments(payload, offset) {
    let off = offset;
    const need = (n) => {
        if (n < 0 || off + n > payload.length) throw new RangeError('attachment truncated');
    };

    const countVi = decodeVarint(payload, off); off += countVi.size;
    const count = Number(countVi.value);

    const attachments = [];
    let truncated = false;

    try {
        for (let i = 0; i < count; i++) {
            const tagVi = decodeVarint(payload, off); off += tagVi.size;
            const tag = tagVi.value;

            if (tag < BigInt(ATTACHMENT_KINDS_32.length)) {
                need(32);
                attachments.push({ kind: ATTACHMENT_KINDS_32[Number(tag)], value: toHex(payload.subarray(off, off + 32)) });
                off += 32;
            } else if (tag === ATT_TAG_LICENSE_CHALLENGE) {
                need(32);
                const commitment = toHex(payload.subarray(off, off + 32)); off += 32;
                const heightVi = decodeVarint(payload, off); off += heightVi.size;
                need(32);
                const salt = toHex(payload.subarray(off, off + 32)); off += 32;
                attachments.push({ kind: 'license_challenge', value: { commitment, height: heightVi.value, salt } });
            } else if (tag === ATT_TAG_SIGNATURE) {
                const lenVi = decodeVarint(payload, off); off += lenVi.size;
                const len = Number(lenVi.value);
                need(len);
                attachments.push({ kind: 'signature', value: toHex(payload.subarray(off, off + len)) });
                off += len;
            } else {
                truncated = true;
                break;
            }
        }
    } catch (e) {
        if (!(e instanceof RangeError)) throw e;
        truncated = true;
    }

    return { attachments, offset: off, truncated };
}

/**
 * Decode a full `Message::ChatV2` payload (discriminant byte included).
 *
 * Field order, from `enum Message` in protocol.rs:
 *   sender: String, timestamp: u64, nonce: u64,
 *   reply_to: Option<u64>, words: Vec<u8>, attachments: Vec<ChatAttachment>
 *
 * @param {Uint8Array} payload
 * @returns {Object|null} The decoded message, or null if this isn't a ChatV2.
 */
export function decodeChatV2(payload) {
    if (!payload || payload.length === 0 || payload[0] !== MSG_CHAT_V2) return null;

    let off = 1;
    const senderLen = decodeVarint(payload, off); off += senderLen.size;
    const senderBytes = Number(senderLen.value);
    if (off + senderBytes > payload.length) throw new RangeError('chat: sender truncated');
    const sender = new TextDecoder().decode(payload.subarray(off, off + senderBytes));
    off += senderBytes;

    const tsVi = decodeVarint(payload, off); off += tsVi.size;
    const nonceVi = decodeVarint(payload, off); off += nonceVi.size;

    // Option<u64>: one tag byte, then the value only when the tag is 1.
    let replyTo = null;
    if (payload[off] === 1) {
        const repVi = decodeVarint(payload, off + 1);
        replyTo = Number(repVi.value);
        off += 1 + repVi.size;
    } else {
        off += 1;
    }

    const wordsLen = decodeVarint(payload, off); off += wordsLen.size;
    const wordCount = Number(wordsLen.value);
    if (off + wordCount > payload.length) throw new RangeError('chat: words truncated');
    const words = Array.from(payload.subarray(off, off + wordCount));
    off += wordCount;

    const { attachments, truncated } = decodeChatAttachments(payload, off);

    return {
        sender,
        timestamp: Number(tsVi.value),
        nonce: Number(nonceVi.value),
        reply_to: replyTo,
        words,
        attachments,
        truncated,
    };
}
