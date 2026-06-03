import { compute_coin_id_hex, blake3_hash_hex } from '../pkg/wasm_wallet.js';

const MDS_KILO = 1024;
const MDS_MEGA = 1048576;
const MDS_GIGA = 1073741824;

export const CHAT_DICTIONARY = [
    "midstate", "network", "node", "peer", "block", "blocks", "tx", "transaction", "mempool", "hash",
    "pow", "mine", "mining", "miner", "sync", "wallet", "address", "key", "seed", "utxo",
    "airdrop", "incoming", "post", "claim", "free", "giveaway", "reward", "bounty", "pool", "liquidity",
    "buy", "sell", "trade", "swap", "market", "price", "fiat", "dex", "cex", "value",
    "send", "receive", "give", "take", "make", "do", "get", "need", "want", "have",
    "check", "verify", "update", "upgrade", "restart", "connect", "drop", "build", "fix", "run",
    "is", "are", "was", "were", "be", "been", "has", "had", "will", "can",
    "could", "should", "would", "might", "must", "stop", "wait", "see", "look", "know",
    "I", "you", "we", "they", "he", "she", "it", "this", "that", "these",
    "those", "who", "what", "where", "when", "why", "how", "which", "my", "your",
    "at", "to", "from", "in", "out", "on", "off", "for", "by", "about",
    "as", "but", "if", "then", "else", "and", "or", "not", "with", "without",
    "good", "bad", "fast", "slow", "full", "empty", "high", "low", "urgent", "ready",
    "online", "offline", "hot", "cold", "big", "small", "hard", "easy", "safe", "new",
    "all", "none", "some", "any", "many", "much", "more", "less", "every", "only",
    "now", "later", "soon", "early", "today", "tomorrow", "yesterday", "time", "always", "never",
    "gm", "gn", "lol", "lfg", "wagmi", "ngmi", "ser", "anon", "mate", "based",
    "wtf", "omg", "moon", "pump", "dump", "bull", "bear", "scam", "rug", "fren",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "20", "50", "100", "200", "500", "1k", "10k", "100k", "1m",
    "wots", "mss", "smt", "sig", "data", "disk", "linux", "pi", "hardware", "software",
    "code", "rust", "server", "client", "ip", "webrtc", "error", "bug", "issue", "help",
    "please", "thanks", "ok", "yes", "no", "maybe", "here", "there", "again", "done",
    "first", "last", "old", "true", "false", "up", "down", "left", "right", "back",
    "?", "!", ".", ",", "...", ":)", ":(", "🔥", "🚀", "💀",
    "💎", "👀", "🤝", "📈", "📉", "⚡"
];

export const MidstateUtils = {
    formatMDS(n) {
        n = Number(n) || 0;
        if (n === 0) return { value: '0', prefix: 'MDS' };
        if (n >= MDS_GIGA) return { value: parseFloat((n / MDS_GIGA).toFixed(4)).toString(), prefix: 'gMDS' };
        if (n >= MDS_MEGA) return { value: parseFloat((n / MDS_MEGA).toFixed(4)).toString(), prefix: 'mMDS' };
        if (n >= MDS_KILO) return { value: parseFloat((n / MDS_KILO).toFixed(4)).toString(), prefix: 'kMDS' };
        return { value: n.toLocaleString('en'), prefix: 'MDS' };
    },
    computeCoinId(addressHex, value, saltHex) {
        return compute_coin_id_hex(addressHex, BigInt(value), saltHex);
    },
    hash(hexData) {
        return blake3_hash_hex(hexData);
    },
    // Convert array of indices [160, 168] -> "gm mate"
    indicesToWords(indices) {
        return indices.map(i => CHAT_DICTIONARY[i] || "???").join(" ");
    },
    // Convert string "gm mate" -> [160, 168]
    textToIndices(text) {
        return text.split(/\s+/).map(w => CHAT_DICTIONARY.indexOf(w)).filter(i => i !== -1);
    }
};
