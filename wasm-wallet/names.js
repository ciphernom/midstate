// names.js: paying `nick.mds` (MSN-1) from the browser, checked rather than
// taken on a node's word. No DOM: the wallet worker runs it, and
// tests/names.test.mjs runs it against a real node's answer.
//
// A node answers GET /name/:name/proof with the name's leaf in the registry,
// the records and terms the leaf commits to, its Merkle proof to the
// registry's root, and the registry coin that root belongs to. Here:
//
// * the leaf's key is the name's (H("MSN1" ‖ name)[..8]), so the leaf is
//   this name's and not another's;
// * the records and terms hash to the leaf's `data` and `deal`;
// * the proof folds the leaf up to the root;
// * the root is the commitment of a coin at the registry covenant's address,
//   so it is a registry thread's root and not a number made up for the
//   occasion;
// * the name is held, and not lapsing within a day.
//
// The wallet then asks its node that the coin is unspent (as it asks about
// its own coins), and, for a name that pays an MSS key, how many signatures
// the key has left. What no single answer shows is that the coin is the
// registry's own thread rather than a look-alike at the same address: that
// is a lineage only a node that followed the chain knows. A lying node would
// have to publish and keep unspent a look-alike thread on the real chain.
//
// The same checks as midwimble's src/names.rs (`check`) and midstate's
// core::names.

export const SUFFIX = '.mds';
export const MAX_NAME_LEN = 32;
/** The registry covenant's address (midstate's core::names::address()). */
export const REGISTRY_ADDRESS = 'c71ab5f9589e9ef822e950cce14fa0e40952fc7103d6358b13427ed96b0b742e';
export const DEPTH = 24;
/** Wallets refuse a name that lapses within a day of blocks. */
export const LAPSING = 1440;
/** Signatures an MSS key must keep: a name that pays one with fewer left
 *  pays nothing (core::names::SIGNATURES_KEPT). */
export const SIGNATURES_KEPT = 16;

export const TAG = { MDS_KEY: 1, MDS_ADDRESS: 2, MDW: 3, EVM: 4, IDENTITY: 5, TEXT: 6, KEM: 8, SCRIPT: 9, POSTAGE: 10 };

/** Whether something typed is a name rather than an address. */
export const isName = s => String(s || '').trim().toLowerCase().endsWith(SUFFIX);

/** A name's canonical form, without `.mds`: a-z, 0-9 and -, 1 to 32
 *  characters, no - at either end. Throws a reason otherwise. */
export function normalize(input) {
    const lower = String(input || '').trim().toLowerCase();
    const s = lower.endsWith(SUFFIX) ? lower.slice(0, -SUFFIX.length) : lower;
    if (!s || s.length > MAX_NAME_LEN) throw new Error(`A name has 1 to ${MAX_NAME_LEN} characters`);
    if (!/^[a-z0-9-]+$/.test(s)) throw new Error('A name uses only a-z, 0-9 and -');
    if (s.startsWith('-') || s.endsWith('-')) throw new Error('A name cannot start or end with -');
    return s;
}

const enc = new TextEncoder();
const toHex = bytes => Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
function fromHex(h, what, len) {
    const s = String(h ?? '');
    if (!/^([0-9a-f]{2})*$/i.test(s) || (len != null && s.length !== 2 * len)) {
        throw new Error(`The answer’s ${what} is not ${len ?? 'whole'} bytes of hex`);
    }
    const out = new Uint8Array(s.length / 2);
    for (let i = 0; i < out.length; i++) out[i] = parseInt(s.slice(2 * i, 2 * i + 2), 16);
    return out;
}
const cat = (...parts) => {
    const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let i = 0;
    for (const p of parts) { out.set(p, i); i += p.length; }
    return out;
};
const u64le = n => {
    const out = new Uint8Array(8);
    let v = BigInt(n);
    for (let i = 0; i < 8; i++) { out[i] = Number(v & 0xffn); v >>= 8n; }
    return out;
};
const readU64le = b => b.reduce((acc, byte, i) => acc | (BigInt(byte) << BigInt(8 * i)), 0n);
const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

/** A name's key in the registry: H("MSN1" ‖ name)[..8]. `blake3` hashes
 *  hex to hex (the wallet's WebAssembly `blake3_hash_hex`). */
export function nameKey(name, blake3) {
    return fromHex(blake3(toHex(cat(enc.encode('MSN1'), enc.encode(name))))).slice(0, 8);
}

/** The records, tag by tag: `[{ tag, value }]`. */
export function parseRecords(bytes) {
    const out = [];
    let i = 0;
    while (i < bytes.length) {
        if (i + 2 > bytes.length) throw new Error('The records are cut short');
        const tag = bytes[i];
        const len = bytes[i + 1];
        if (i + 2 + len > bytes.length) throw new Error('The records are cut short');
        out.push({ tag, value: bytes.slice(i + 2, i + 2 + len) });
        i += 2 + len;
    }
    return out;
}

/**
 * Checks a node's `/name/:name/proof` answer for `name` (canonical, no
 * `.mds`), whatever it says: the leaf against the registry root, the root
 * against the registry coin, and the records and terms against the leaf. For
 * a name never registered the leaf is its neighbour's, and its key must sit
 * between the neighbour's and the next. Returns `{ name, registered, slot,
 * leaf, records, terms, expires, height, root, coin, salt, proof }`
 * (records parsed; leaf fields as the node gives them) or throws why the
 * answer does not hold.
 */
export function checkAnswer(name, v, blake3) {
    if (v && typeof v.error === 'string') throw new Error(`The node says: ${v.error}`);
    const H = bytes => fromHex(blake3(toHex(bytes)));
    const leaf = (v && v.leaf) || {};
    const lkey = fromHex(leaf.key, 'leaf key', 8);
    const next = fromHex(leaf.next, 'leaf next', 8);
    const want = nameKey(name, blake3);
    const registered = v.registered === true;
    if (registered && !same(lkey, want)) throw new Error('The answer’s leaf is another name’s');
    if (!registered) {
        // The neighbour's link skips over the key: nothing is registered there.
        const [k, lo, hi] = [readU64le(want), readU64le(lkey), readU64le(next)];
        if (!(lo < k && k < hi)) throw new Error(`The answer says ${name}.mds is not registered, but its proof does not show that`);
    }
    const records = fromHex(v.records, 'records');
    const terms = fromHex(v.terms, 'terms');
    const data = fromHex(leaf.data, 'leaf data', 32);
    const deal = fromHex(leaf.deal, 'leaf deal', 32);
    if (!same(H(records), data)) throw new Error('The records are not the ones the leaf commits to');
    if (!same(H(terms), deal)) throw new Error('The terms are not the ones the leaf commits to');
    const expires = leaf.expires;
    if (!Number.isSafeInteger(expires) || expires < 0) throw new Error('The answer has no expiry');
    const tier = leaf.tier;
    if (!Number.isInteger(tier) || tier < 0 || tier > 3) throw new Error('The answer has no tier');
    const owner = fromHex(leaf.owner, 'owner', 32);
    const leafBytes = cat(lkey, next, u64le(expires), Uint8Array.of(tier), owner, data, deal);
    const proof = fromHex(v.proof, 'proof');
    if (proof.length !== DEPTH * 33) throw new Error(`The proof is not ${DEPTH} levels`);
    let acc = H(leafBytes);
    let slot = 0;
    for (let l = 0; l < DEPTH; l++) {
        const sib = proof.slice(l * 33, l * 33 + 32);
        const right = proof[l * 33 + 32] & 1;
        if (right) slot += 2 ** l;
        acc = right ? H(cat(sib, acc)) : H(cat(acc, sib));
    }
    if (v.slot != null && v.slot !== slot) throw new Error('The proof is for another slot than the answer says');
    const root = fromHex(v.root, 'root', 32);
    if (!same(acc, root)) throw new Error('The proof does not reach the registry root the node gives');
    const coin = fromHex(v.thread && v.thread.coin, 'thread coin', 32);
    const salt = fromHex(v.thread && v.thread.salt, 'thread salt', 32);
    const committed = H(cat(enc.encode('CONFIDENTIAL'), fromHex(REGISTRY_ADDRESS), root, salt));
    if (!same(committed, coin)) throw new Error('The root is not the commitment of a registry coin');
    const height = v.height;
    if (!Number.isSafeInteger(height)) throw new Error('The answer has no height');
    return {
        name, registered, slot, records: parseRecords(records), expires, height, root: toHex(root), coin: toHex(coin), salt: toHex(salt),
        proof: toHex(proof), recordsHex: toHex(records), termsHex: toHex(terms),
        leaf: { key: toHex(lkey), next: toHex(next), expires, tier, owner: toHex(owner), data: toHex(data), deal: toHex(deal) },
    };
}

/**
 * Checks a node's `/name/:name/proof` answer for `name` (canonical, no
 * `.mds`), for paying it: it must be held. Returns `{ name, records,
 * expires, height, root, coin }`, the records parsed, or throws why not. A
 * name that lapses within a day is refused, as a payment must be;
 * `{ lapsingOk: true }` takes it while it is still held (sealed messages,
 * which a lapsing name still takes).
 */
export function checkProof(name, v, blake3, { lapsingOk = false } = {}) {
    const a = checkAnswer(name, v, blake3);
    if (!a.registered) throw new Error(`No one has the name ${name}.mds`);
    if (a.expires < a.height) throw new Error(`${name}.mds has lapsed: its holder did not renew it`);
    if (!lapsingOk && a.expires < a.height + LAPSING) throw new Error(`${name}.mds lapses within a day, so it can’t be paid now`);
    return { name, records: a.records, expires: a.expires, height: a.height, root: a.root, coin: a.coin };
}

/**
 * Where an MDS payment to a name goes, from its records: the first record
 * that pays, as midstate's core::names::pays has it. An MDS address pays as
 * it is; an MSS key pays its address while it keeps SIGNATURES_KEPT
 * signatures. `used(keyHex)` gives the key's next signature index (the
 * node's mss_state); `p2pk(keyHex)` its address (the WebAssembly's
 * compute_p2pk_address_hex). Resolves to `{ address, kind }` or null.
 */
export async function payAddress(records, used, p2pk) {
    for (const r of records) {
        if (r.tag === TAG.MDS_ADDRESS && r.value.length === 32) return { address: toHex(r.value), kind: 'address' };
        if (r.tag === TAG.MDS_KEY && r.value.length === 33) {
            const key = toHex(r.value.slice(0, 32));
            const height = r.value[32];
            if (height > 30) continue;
            const left = 2 ** height - Number(await used(key));
            if (left >= SIGNATURES_KEPT) return { address: p2pk(key), kind: 'key', left };
        }
    }
    return null;
}

/** A record's value as text, for the ones that are text (MDW, a line). */
export const recordText = r => new TextDecoder().decode(r.value);

/** Records as the registry keeps them: tag, length, value, each value at
 *  most 255 bytes and all of them at most MAX_RECORDS. `value` is bytes or
 *  hex. Returns hex. */
export const MAX_RECORDS = 1536;
export function encodeRecords(list) {
    const parts = [];
    for (const r of list) {
        const value = typeof r.value === 'string' ? fromHex(r.value, 'record') : Uint8Array.from(r.value);
        if (!Number.isInteger(r.tag) || r.tag < 0 || r.tag > 255) throw new Error('A record’s tag is a byte');
        if (value.length > 255) throw new Error('A record holds at most 255 bytes');
        parts.push(Uint8Array.of(r.tag, value.length), value);
    }
    const out = cat(...parts);
    if (out.length > MAX_RECORDS) throw new Error(`A name’s records hold at most ${MAX_RECORDS} bytes`);
    return toHex(out);
}

/** What a name costs: its tier by length (premium names are listed in the
 *  registry from genesis with theirs), a year of each tier burning its rate. */
export const TIERS = ['platinum', 'gold', 'silver', 'bronze'];
export const RATES = [2 ** 36, 2 ** 33, 2 ** 30, 2 ** 27];
export const YEAR = 525_600;
export const MAX_YEARS = 10;
export const tierForLen = len => Math.min(Math.max(len, 2) - 2, 3);
export const costOf = (tier, years) => RATES[tier] * years;

/** What an answer says of a name, for someone who might want it: `available`
 *  (never registered), `unclaimed` (a premium name no one has claimed yet),
 *  `lapsed` (its holder did not renew), or `held`; with its tier, holder and
 *  expiry. `owner` is the control key that holds it. */
export function standing(a) {
    if (!a.registered) return { state: 'available', tier: tierForLen(a.name.length), height: a.height };
    const zero = /^0+$/.test(a.leaf.owner);
    const state = zero && a.leaf.expires === 0 ? 'unclaimed' : a.expires < a.height ? 'lapsed' : 'held';
    const t = fromHex(a.termsHex || '00', 'terms');
    const deal = t[0] === 1 ? 'sale' : t[0] === 2 ? 'rent' : t[0] === 3 ? 'rented' : null;
    return { state, tier: a.leaf.tier, owner: a.leaf.owner, expires: a.expires, height: a.height, deal };
}

/** The hash of the name's sealed-message key, if it takes sealed messages. */
export function kemHash(records) {
    const r = records.find(x => x.tag === TAG.KEM && x.value.length === 32);
    return r ? toHex(r.value) : null;
}

/** The name's identity key (hex), if it gives one: what is asked for its
 *  wallet's KEM key, and what signs as the name. */
export function identityOf(records) {
    const r = records.find(x => x.tag === TAG.IDENTITY && x.value.length === 32);
    return r ? toHex(r.value) : null;
}

/** The stamps a `postage` record may ask (names::POSTAGE_BITS), and what a
 *  name without one asks (sealed::STAMP_BITS). */
export const POSTAGE_BITS = [16, 40];
export const POSTAGE_STANDARD = 24;

/** The stamp bits a stranger's sealed message to the name must show: its
 *  first good postage record (one byte, 16 to 40), or the standard. */
export function postageOf(records) {
    const r = records.find(x => x.tag === TAG.POSTAGE && x.value.length === 1 && x.value[0] >= POSTAGE_BITS[0] && x.value[0] <= POSTAGE_BITS[1]);
    return r ? r.value[0] : POSTAGE_STANDARD;
}
