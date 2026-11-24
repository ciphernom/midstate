// room.js: the public room, the chat bus as everyone on it sees it.
//
// A public message is up to ten of the bus's 256 fixed words, sent as their
// indices (chat::CHAT_DICTIONARY), with up to four typed attachments. Two
// other things ride the same bus and are not public messages: wire frames
// (words that start with 255, as walletd's channels::parse_frame and the
// classic page read them) and sealed messages (encrypted_message
// attachments, which only their recipient's wallet opens). The room leaves
// both out.
//
// Pure functions only: the worker keeps what the bus says, the page shows it.

/** chat::CHAT_DICTIONARY. Fixed by the chat protocol: index 42 means the same
 *  word on every peer, so this list never changes. */
export const DICTIONARY = Object.freeze([
    'midstate', 'network', 'node', 'peer', 'block', 'blocks', 'tx', 'transaction', 'mempool', 'hash',
    'pow', 'mine', 'mining', 'miner', 'sync', 'wallet', 'address', 'key', 'seed', 'utxo',
    'airdrop', 'incoming', 'post', 'claim', 'free', 'giveaway', 'reward', 'bounty', 'pool', 'liquidity',
    'buy', 'sell', 'trade', 'swap', 'market', 'price', 'fiat', 'dex', 'cex', 'value',
    'send', 'receive', 'give', 'take', 'make', 'do', 'get', 'need', 'want', 'have',
    'check', 'verify', 'update', 'upgrade', 'restart', 'connect', 'drop', 'build', 'fix', 'run',
    'is', 'are', 'was', 'were', 'be', 'been', 'has', 'had', 'will', 'can',
    'could', 'should', 'would', 'might', 'must', 'stop', 'wait', 'see', 'look', 'know',
    'I', 'you', 'we', 'they', 'he', 'she', 'it', 'this', 'that', 'these',
    'those', 'who', 'what', 'where', 'when', 'why', 'how', 'which', 'my', 'your',
    'at', 'to', 'from', 'in', 'out', 'on', 'off', 'for', 'by', 'about',
    'as', 'but', 'if', 'then', 'else', 'and', 'or', 'not', 'with', 'without',
    'good', 'bad', 'fast', 'slow', 'full', 'empty', 'high', 'low', 'urgent', 'ready',
    'online', 'offline', 'hot', 'cold', 'big', 'small', 'hard', 'easy', 'safe', 'new',
    'all', 'none', 'some', 'any', 'many', 'much', 'more', 'less', 'every', 'only',
    'now', 'later', 'soon', 'early', 'today', 'tomorrow', 'yesterday', 'time', 'always', 'never',
    'gm', 'gn', 'lol', 'lfg', 'wagmi', 'ngmi', 'ser', 'anon', 'mate', 'based',
    'wtf', 'omg', 'moon', 'pump', 'dump', 'bull', 'bear', 'scam', 'rug', 'fren',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '10', '20', '50', '100', '200', '500', '1k', '10k', '100k', '1m',
    'wots', 'mss', 'smt', 'sig', 'data', 'disk', 'linux', 'pi', 'hardware', 'software',
    'code', 'rust', 'server', 'client', 'ip', 'webrtc', 'error', 'bug', 'issue', 'help',
    'please', 'thanks', 'ok', 'yes', 'no', 'maybe', 'here', 'there', 'again', 'done',
    'first', 'last', 'old', 'true', 'false', 'up', 'down', 'left', 'right', 'back',
    '?', '!', '.', ',', '...', ':)', ':(', '🔥', '🚀', '💀',
    '💎', '👀', '🤝', '📈', '📉', '⚡',
]);

export const MAX_WORDS = 10;        // the node's submit_chat and Message::deserialize_bin
export const MAX_ATTACHMENTS = 4;   // chat::MAX_CHAT_ATTACHMENTS
export const FRAME = 255;           // a first word of 255 marks a wire frame
export const KEEP = 200;            // public messages the worker holds

/** Punctuation that sits against the word before it. */
const TIGHT = new Set(['?', '!', '.', ',', '...']);
/** Endings peeled off a typed word: "ok?" is "ok" then "?". */
const ENDINGS = ['...', '?', '!', '.', ','];

const lower = s => String(s).toLowerCase();
const indexOf = (dict) => {
    const m = new Map();
    dict.forEach((w, i) => { if (!m.has(lower(w))) m.set(lower(w), i); });
    return m;
};
const INDEX = indexOf(DICTIONARY);

/** The bus's key for a message: what the node deduplicates on, and the
 *  sender, so that two senders' nonces never collide here. */
export const keyOf = m => `${m.nonce}:${m.sender}`;

/** Whether a chat is a public message: neither a wire frame nor a sealed one,
 *  and something in it to show. */
export function isPublic(m) {
    if (!m || typeof m !== 'object' || !Array.isArray(m.words)) return false;
    if (typeof m.sender !== 'string' || !m.sender) return false;
    if (m.words.length && m.words[0] === FRAME) return false;
    const atts = Array.isArray(m.attachments) ? m.attachments : [];
    if (atts.some(a => a && a.kind === 'encrypted_message')) return false;
    return m.words.length > 0 || atts.length > 0;
}

/** Words as the room reads them: "gm fren!" for [160, 179, 241]. */
export function textOf(words, dict = DICTIONARY) {
    let out = '';
    for (const w of words || []) {
        const t = dict[w] ?? '?';
        out += (out && !TIGHT.has(t) ? ' ' : '') + t;
    }
    return out;
}

/**
 * What a typed line says in the bus's words. Each space-separated piece is a
 * word of the dictionary, matched without regard to case, or a word with
 * punctuation from the dictionary after it ("ok?"). Returns the indices, the
 * pieces that are not words, and why the line can't go as it is (or null).
 */
export function parseDraft(text, dict = DICTIONARY) {
    const index = dict === DICTIONARY ? INDEX : indexOf(dict);
    const words = [];
    const unknown = [];
    for (const piece of String(text || '').split(/\s+/).filter(Boolean)) {
        let rest = piece;
        const tail = [];
        while (!index.has(lower(rest))) {
            const end = ENDINGS.find(e => rest.length > e.length && rest.endsWith(e));
            if (!end) break;
            tail.unshift(index.get(end));
            rest = rest.slice(0, -end.length);
        }
        if (!index.has(lower(rest))) { unknown.push(piece); continue; }
        words.push(index.get(lower(rest)), ...tail);
    }
    let problem = null;
    if (unknown.length) problem = unknown.length === 1 ? `“${unknown[0]}” isn’t one of the room’s words.` : `${unknown.map(u => `“${u}”`).join(', ')} aren’t among the room’s words.`;
    else if (words.length > MAX_WORDS) problem = `That’s ${words.length} words: a message has ${MAX_WORDS} at most.`;
    else if (words.length && words[0] === FRAME) problem = `A message can’t start with ${dict[FRAME]}: wallets read that as a payment-channel frame.`;
    return { words, unknown, problem };
}

/** Words that start with what is being typed, for the picker under the box:
 *  the word itself first, if it is one, then the rest in the bus's order. */
export function suggest(prefix, n = 8, dict = DICTIONARY) {
    const p = lower(String(prefix || '').trim());
    if (!p) return [];
    const exact = dict.filter(w => lower(w) === p);
    const starts = dict.filter(w => lower(w) !== p && lower(w).startsWith(p));
    return [...exact, ...starts].slice(0, n);
}

/** A short name for a sender: what a 64-hex key or a libp2p peer id is told
 *  apart by at a glance. */
export function senderLabel(sender) {
    const s = String(sender || '');
    if (/^[0-9a-f]{64}$/i.test(s)) return s.slice(0, 8).toLowerCase();
    return s.length > 8 ? `…${s.slice(-6)}` : s || 'unknown';
}

/** Messages in, without repeats, the newest `keep` in time order. */
export function merge(held, incoming, keep = KEEP) {
    const map = new Map((held || []).map(m => [keyOf(m), m]));
    for (const m of incoming || []) if (isPublic(m)) map.set(keyOf(m), m);
    return [...map.values()]
        .sort((a, b) => (a.timestamp - b.timestamp) || (a.nonce - b.nonce))
        .slice(-keep);
}

/** The room as the page draws it: each message's text, what it answers, and
 *  whether this wallet sent it. */
export function view(messages, mine, dict = DICTIONARY) {
    const byNonce = new Map();
    for (const m of messages || []) if (!byNonce.has(m.nonce)) byNonce.set(m.nonce, m);
    return (messages || []).map((m) => {
        const parent = m.reply_to != null ? byNonce.get(m.reply_to) : null;
        return {
            key: keyOf(m),
            nonce: m.nonce,
            sender: m.sender,
            at: m.timestamp,
            text: textOf(m.words, dict),
            attachments: (m.attachments || []).map(a => ({ kind: a.kind, value: typeof a.value === 'string' ? a.value : JSON.stringify(a.value) })),
            replyTo: m.reply_to != null ? (parent ? { sender: parent.sender, text: textOf(parent.words, dict), mine: parent.sender === mine } : { sender: null, text: null, mine: false }) : null,
            mine: !!mine && m.sender === mine,
        };
    });
}

/** What a send would carry, checked as the node checks it. */
export function checkOut(words, attachments) {
    if (!Array.isArray(words) || words.some(w => !Number.isInteger(w) || w < 0 || w >= DICTIONARY.length)) return 'Those aren’t the room’s words.';
    if (!Array.isArray(attachments)) return 'Those attachments aren’t right.';
    if (!words.length && !attachments.length) return 'Say something first.';
    if (words.length > MAX_WORDS) return `A message has ${MAX_WORDS} words at most.`;
    if (words.length && words[0] === FRAME) return `A message can’t start with ${DICTIONARY[FRAME]}.`;
    if (attachments.length > MAX_ATTACHMENTS) return `A message carries ${MAX_ATTACHMENTS} attachments at most.`;
    if (attachments.some(a => !a || a.kind !== 'address' || !/^[0-9a-f]{72}$/.test(String(a.value)))) return 'Only addresses can be attached here.';
    return null;
}
