// test-bincode.mjs — Offline conformance tests for the native P2P wire codec.
//
// Vectors are constructed here byte-by-byte from the Rust definitions
// (src/network/protocol.rs `Message`, src/chat.rs `ChatAttachment`) rather than
// captured from a live node, so a drift in the node's enum ordering shows up as
// a failure here instead of as silent corruption at runtime.
//
// Run: node test-bincode.mjs

import {
    encodeVarint, decodeVarint, encodeGetAddr, decodeAddr, encodePong,
    decodeChatV2, decodeChatAttachments,
    MSG_CHAT_V2, MSG_GET_ADDR, MSG_ADDR, MSG_PONG,
} from './src/bincode.js';

let pass = 0, fail = 0;
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
function t(name, fn) {
    try { fn(); console.log(`  ✅ ${name}`); pass++; }
    catch (e) { console.log(`  ❌ ${name} — ${e.message}`); fail++; }
}
function assert(c, m) { if (!c) throw new Error(m || 'assertion failed'); }

const enc = new TextEncoder();
const cat = (...a) => { const t = a.reduce((s, x) => s + x.length, 0); const o = new Uint8Array(t); let f = 0; for (const x of a) { o.set(x, f); f += x.length; } return o; };
const hex = (b) => Array.from(b).map((x) => x.toString(16).padStart(2, '0')).join('');

console.log('\n━━━ varint ━━━');

t('single byte boundary (0, 250)', () => {
    assert(eq([...encodeVarint(0)], [0]));
    assert(eq([...encodeVarint(250)], [250]));
});

t('u16 / u32 / u64 forms use the right marker', () => {
    assert(encodeVarint(251)[0] === 251, '251 should take the u16 form');
    assert(encodeVarint(65536)[0] === 252, '2^16 should take the u32 form');
    assert(encodeVarint(1n << 32n)[0] === 253, '2^32 should take the u64 form');
});

t('encode/decode round-trip across magnitudes', () => {
    for (const v of [0n, 1n, 250n, 251n, 65535n, 65536n, (1n << 32n) - 1n, 1n << 32n, (1n << 63n)]) {
        const d = decodeVarint(encodeVarint(v), 0);
        assert(d.value === v, `round-trip failed for ${v}: got ${d.value}`);
        assert(d.size === encodeVarint(v).length, `size mismatch for ${v}`);
    }
});

t('576 (WOTS signature length) encodes as the u16 form 251 40 02', () => {
    // This is the exact byte sequence the old decoder mis-read as a 4-byte u32.
    assert(eq([...encodeVarint(576)], [251, 0x40, 0x02]), hex(encodeVarint(576)));
});

t('truncated multi-byte varint throws rather than reading past the buffer', () => {
    let threw = false;
    try { decodeVarint(Uint8Array.of(253, 1, 2), 0); } catch (e) { threw = e instanceof RangeError; }
    assert(threw, 'expected RangeError on truncated u64 varint');
});

console.log('\n━━━ message discriminants ━━━');

t('discriminants match protocol.rs enum order', () => {
    assert(MSG_GET_ADDR === 5 && MSG_ADDR === 6 && MSG_PONG === 8 && MSG_CHAT_V2 === 19);
});

t('GetAddr encodes to a single byte 5', () => {
    assert(eq([...encodeGetAddr()], [5]));
});

t('Pong { nonce: 0 } encodes to [8, 0]', () => {
    assert(eq([...encodePong(0)], [8, 0]));
});

t('Addr round-trips a peer list', () => {
    const addrs = ['/ip4/1.2.3.4/tcp/9333/p2p/12D3KooWabc', '/ip4/5.6.7.8/tcp/9333/p2p/12D3KooWdef'];
    const body = cat(encodeVarint(MSG_ADDR), encodeVarint(addrs.length),
        ...addrs.flatMap((a) => [encodeVarint(enc.encode(a).length), enc.encode(a)]));
    assert(eq(decodeAddr(body), addrs));
});

t('truncated Addr payload is rejected, not silently short-decoded', () => {
    let threw = false;
    // disc 6, count 1, length 250, then nothing.
    try { decodeAddr(Uint8Array.of(6, 1, 250)); } catch { threw = true; }
    assert(threw, 'truncated Addr must throw');
});

t('Addr rejects an implausible entry count', () => {
    let threw = false;
    try { decodeAddr(cat(encodeVarint(6), encodeVarint(5000))); } catch { threw = true; }
    assert(threw, 'over-long Addr count must throw');
});

t('Addr round-trips an entry needing the u16 length marker', () => {
    const long = '/ip4/198.51.100.2/tcp/9333/p2p/' + 'Q'.repeat(260);
    const sb = enc.encode(long);
    const body = cat(encodeVarint(6), encodeVarint(1), encodeVarint(sb.length), sb);
    assert(eq(decodeAddr(body), [long]));
});

console.log('\n━━━ chat attachments ━━━');

const A32 = (fill) => new Uint8Array(32).fill(fill);

t('all seven 32-byte kinds decode at their correct tags', () => {
    const kinds = ['address', 'coin_id', 'mix_id', 'commitment', 'block_hash', 'midstate', 'data_hash'];
    for (let tag = 0; tag < kinds.length; tag++) {
        const buf = cat(encodeVarint(1), encodeVarint(tag), A32(tag));
        const r = decodeChatAttachments(buf, 0);
        assert(r.attachments.length === 1, `tag ${tag}: expected 1 attachment`);
        assert(r.attachments[0].kind === kinds[tag], `tag ${tag}: got kind ${r.attachments[0].kind}`);
        assert(r.offset === buf.length, `tag ${tag}: desync`);
        assert(!r.truncated, `tag ${tag}: unexpectedly truncated`);
    }
});

t('signature is tag 8 with a varint length (regression: was tag 9 + u32)', () => {
    const sig = new Uint8Array(576).fill(0xab);
    const buf = cat(encodeVarint(1), encodeVarint(8), encodeVarint(sig.length), sig);
    const r = decodeChatAttachments(buf, 0);
    assert(r.attachments.length === 1, 'signature not decoded');
    assert(r.attachments[0].kind === 'signature');
    assert(r.attachments[0].value === 'ab'.repeat(576), 'signature bytes wrong');
    assert(r.offset === buf.length, 'desync after signature');
});

t('license_challenge (tag 7) consumes 32 + varint + 32', () => {
    const buf = cat(encodeVarint(1), encodeVarint(7), A32(0x11), encodeVarint(123456), A32(0x22));
    const r = decodeChatAttachments(buf, 0);
    assert(r.attachments.length === 1);
    assert(r.attachments[0].kind === 'license_challenge');
    assert(r.attachments[0].value.height === 123456n, `height ${r.attachments[0].value.height}`);
    assert(r.attachments[0].value.commitment === '11'.repeat(32));
    assert(r.attachments[0].value.salt === '22'.repeat(32));
    assert(r.offset === buf.length, 'desync after license_challenge');
});

t('a variable-width attachment does not corrupt the ones after it', () => {
    // The core regression: sequence decoding means one bad width poisons the
    // rest. license_challenge and signature are both variable-width.
    const sig = new Uint8Array(1280).fill(0x7f);   // MSS(height=20) signature
    const buf = cat(
        encodeVarint(4),
        encodeVarint(7), A32(0x11), encodeVarint(999), A32(0x22),   // license_challenge
        encodeVarint(8), encodeVarint(sig.length), sig,             // signature
        encodeVarint(0), A32(0x33),                                 // address
        encodeVarint(3), A32(0x44),                                 // commitment
    );
    const r = decodeChatAttachments(buf, 0);
    assert(r.attachments.length === 4, `expected 4 attachments, got ${r.attachments.length}`);
    assert(eq(r.attachments.map((a) => a.kind), ['license_challenge', 'signature', 'address', 'commitment']));
    assert(r.attachments[2].value === '33'.repeat(32), 'address after variable-width fields corrupted');
    assert(r.attachments[3].value === '44'.repeat(32), 'commitment after variable-width fields corrupted');
    assert(r.offset === buf.length, 'desync');
});

t('unknown tag stops cleanly and reports truncation', () => {
    const buf = cat(encodeVarint(2), encodeVarint(0), A32(1), encodeVarint(99), A32(2));
    const r = decodeChatAttachments(buf, 0);
    assert(r.attachments.length === 1, 'should keep what was decodable');
    assert(r.truncated === true, 'should flag truncation on unknown tag');
});

t('short buffer is reported, not read past', () => {
    const buf = cat(encodeVarint(1), encodeVarint(0), new Uint8Array(10)); // 32 bytes promised, 10 given
    const r = decodeChatAttachments(buf, 0);
    assert(r.truncated === true, 'should flag truncation');
    assert(r.attachments.length === 0);
});

console.log('\n━━━ full ChatV2 message ━━━');

function buildChatV2({ sender, timestamp, nonce, replyTo, words, attachments }) {
    const parts = [encodeVarint(MSG_CHAT_V2), encodeVarint(enc.encode(sender).length), enc.encode(sender),
        encodeVarint(timestamp), encodeVarint(nonce)];
    parts.push(replyTo === null ? Uint8Array.of(0) : cat(Uint8Array.of(1), encodeVarint(replyTo)));
    parts.push(encodeVarint(words.length), Uint8Array.from(words));
    parts.push(encodeVarint(attachments.length), ...attachments);
    return cat(...parts);
}

const SENDER = '12D3KooWPbR63SQg1UBLpAMiNngqrRHGM4LaMP8ieAJUxhfw7dxv';

t('decodes a message carrying address + signature', () => {
    const sig = new Uint8Array(576).fill(0xab);
    const frame = buildChatV2({
        sender: SENDER, timestamp: 1700000000, nonce: 42, replyTo: 7, words: [160, 168],
        attachments: [cat(encodeVarint(0), A32(0x11)), cat(encodeVarint(8), encodeVarint(576), sig)],
    });
    const m = decodeChatV2(frame);
    assert(m !== null, 'returned null');
    assert(m.sender === SENDER, 'sender');
    assert(m.timestamp === 1700000000, 'timestamp');
    assert(m.nonce === 42, 'nonce');
    assert(m.reply_to === 7, 'reply_to');
    assert(eq(m.words, [160, 168]), 'words');
    assert(m.attachments.length === 2, `attachments: got ${m.attachments.length}`);
    assert(m.attachments[1].kind === 'signature' && m.attachments[1].value.length === 1152, 'signature');
    assert(m.truncated === false, 'should not be truncated');
});

t('reply_to: None decodes as null', () => {
    const frame = buildChatV2({ sender: SENDER, timestamp: 1, nonce: 2, replyTo: null, words: [], attachments: [] });
    assert(decodeChatV2(frame).reply_to === null);
});

t('non-ChatV2 discriminant returns null instead of throwing', () => {
    assert(decodeChatV2(Uint8Array.of(MSG_PONG, 0)) === null);
    assert(decodeChatV2(new Uint8Array(0)) === null);
});

t('max attachments (4) all survive', () => {
    const frame = buildChatV2({
        sender: SENDER, timestamp: 1, nonce: 2, replyTo: null, words: [1],
        attachments: [0, 1, 2, 3].map((tag) => cat(encodeVarint(tag), A32(tag))),
    });
    const m = decodeChatV2(frame);
    assert(m.attachments.length === 4, `got ${m.attachments.length}`);
    assert(eq(m.attachments.map((a) => a.kind), ['address', 'coin_id', 'mix_id', 'commitment']));
});

console.log(`\n━━━ ${pass} passed, ${fail} failed ━━━\n`);
process.exit(fail ? 1 : 0);
