// format.js: amounts, addresses and times as the wallet shows them. No DOM,
// so it is tested on its own (tests/format.test.mjs).
//
// Every MDS amount is shown and typed in gMDS: 2^30 base units, the block
// reward the chain started with. The desktop app's simple view does the
// same, so a figure reads the same in both.

export const UNIT = 'gMDS';
export const ONE = 2 ** 30;
const ONE_N = 1n << 30n;

function decimals(units, places, grouped) {
    const v = BigInt(Math.max(0, Math.floor(Number(units) || 0)));
    const scale = 10n ** BigInt(places);
    let whole = v / ONE_N;
    let frac = ((v % ONE_N) * scale + ONE_N / 2n) / ONE_N;
    if (frac >= scale) { whole += 1n; frac -= scale; }
    let out = grouped ? whole.toLocaleString('en-US') : whole.toString();
    if (frac > 0n) out += '.' + frac.toString().padStart(places, '0').replace(/0+$/, '');
    return out;
}

/** An amount in gMDS for people: up to four decimals, more only when a
 *  small amount would otherwise read as nothing. */
export function gmds(units) {
    const s = decimals(units, 4, true);
    return s === '0' && Number(units) > 0 ? decimals(units, 9, true) : s;
}

/** `gmds` with the unit after it. */
export const withUnit = units => `${gmds(units)} ${UNIT}`;

/** Every decimal an amount has (up to nine), grouped: for a payment being
 *  reviewed, where 4.499999 must not read as 4.5. */
export const gmdsExact = units => decimals(units, 9, true);
export const exactUnit = units => `${gmdsExact(units)} ${UNIT}`;

/** An amount as it goes back into a field: every decimal it has (up to
 *  nine), no thousands separators, so it reads back as itself. */
export const plainAmount = units => decimals(units, 9, false);

/** gMDS typed by a person, in base units: `{ units }` or `{ error }` (an
 *  empty `error` for an empty field). Decimals round to the nearest base unit,
 *  about a billionth of a gMDS, so 0.1 means 0.1. Spaces are ignored. Commas:
 *  beside a dot, or more than one, they group thousands (1,000.5 and
 *  1,000,000); a single one is a decimal point (2,5), unless three digits
 *  follow it, when 1,234 could be either and is refused rather than guessed. */
export function parseAmount(text) {
    let t = String(text ?? '').replace(/\s+/g, '');
    if (!t) return { error: '' };
    const commas = (t.match(/,/g) || []).length;
    if (commas) {
        const dot = t.indexOf('.');
        if (dot >= 0 && t.lastIndexOf(',') > dot) return { error: 'That isn’t a number' };
        if (dot >= 0 || commas > 1) t = t.replace(/,/g, '');
        else if (t.split(',')[1].length === 3) return { error: 'Use a dot for decimals' };
        else t = t.replace(',', '.');
    }
    const m = /^(\d*)(?:\.(\d*))?$/.exec(t);
    if (!m || (!m[1] && !m[2])) return { error: 'That isn’t a number' };
    const [, int = '', frac = ''] = m;
    if (frac.length > 9) return { error: 'Use at most 9 decimals' };
    const scale = 10n ** BigInt(frac.length);
    const units = BigInt(int || '0') * ONE_N + ((BigInt(frac || '0') * ONE_N + scale / 2n) / scale);
    if (units === 0n) return { error: 'Enter an amount' };
    if (units > BigInt(Number.MAX_SAFE_INTEGER)) return { error: 'That is too much' };
    return { units: Number(units) };
}

/** An address by its ends. */
export function shortAddr(a, n = 6) {
    const s = String(a || '');
    return s.length > 2 * n + 1 ? `${s.slice(0, n)}…${s.slice(-4)}` : s;
}

/** An address in groups of `n`, so it can be read out and compared. */
export const groupHex = (a, n = 4) => String(a || '').replace(new RegExp(`(.{${n}})`, 'g'), '$1 ').trim();

/** Seconds ago, in words. */
export function ago(secs) {
    const s = Math.max(0, Math.round(secs));
    if (s < 60) return 'just now';
    if (s < 3600) return `${Math.round(s / 60)} min ago`;
    if (s < 86_400) { const n = Math.round(s / 3600); return `${n} hour${n === 1 ? '' : 's'} ago`; }
    const n = Math.round(s / 86_400);
    return `${n} day${n === 1 ? '' : 's'} ago`;
}

/** The day a timestamp falls on, for headings: Today, Yesterday, a weekday
 *  within the week, then the date. `now` is for tests. */
export function dayLabel(ts, now = Date.now()) {
    const d = new Date(ts * 1000);
    const n = new Date(now);
    const start = new Date(n.getFullYear(), n.getMonth(), n.getDate()).getTime();
    const day = 86_400_000;
    if (d.getTime() >= start) return 'Today';
    if (d.getTime() >= start - day) return 'Yesterday';
    if (d.getTime() >= start - 6 * day) return d.toLocaleDateString(undefined, { weekday: 'long' });
    return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: d.getFullYear() === n.getFullYear() ? undefined : 'numeric' });
}

/** A clock time. */
export const clock = ts => new Date(ts * 1000).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });

/** A checksummed midstate address as typed or pasted: 72 hex characters
 *  (64 of address, 8 of checksum), spaces and a 0x ignored. */
export function cleanAddress(s) {
    const t = String(s || '').replace(/\s+/g, '').replace(/^0x/i, '').toLowerCase();
    return /^[0-9a-f]{72}$/.test(t) ? t : null;
}

/** A send's steps, as the worker names them, in plain words with how far
 *  along it is. */
export function sendStep(step) {
    switch (step) {
        case 'built': return ['Preparing', 0.1];
        case 'pow': return ['Preparing', 0.2];
        case 'pow_done': return ['Preparing', 0.3];
        case 'commit_sent': return ['Waiting for the network', 0.45];
        case 'commit_mined': return ['Waiting for the network', 0.65];
        case 'reveal_sent': return ['Finishing', 0.85];
        case 'reveal_mined': return ['Sent', 1];
        default: return ['Preparing', 0.05];
    }
}

/** A long-running wallet message (a tidy-up reports only these) in plain
 *  words, with how far along it is. */
export function progressWords(msg) {
    const m = String(msg || '').toLowerCase();
    if (/\(2\/2\)|phase 2|waiting for (the )?(reveal|defrag)/.test(m)) return ['Finishing', 0.85];
    if (/reveal/.test(m)) return ['Finishing', 0.7];
    if (/\(1\/2\)|phase 1|waiting for (the )?commit/.test(m)) return ['Waiting for the network', 0.5];
    if (/commit/.test(m)) return ['Waiting for the network', 0.4];
    if (/pow|proof/.test(m)) return ['Preparing', 0.25];
    return ['Preparing', 0.1];
}

/** Attempts a second, for the miner. */
export function speed(nps) {
    const n = Number(nps) || 0;
    if (n < 1000) return `${Math.round(n)} attempts/s`;
    if (n < 1e6) return `${(n / 1e3).toFixed(n < 1e4 ? 2 : 1)} k attempts/s`;
    return `${(n / 1e6).toFixed(2)} M attempts/s`;
}

/** A date and time in full, for a payment's details. */
export const dateTime = ts => new Date(ts * 1000).toLocaleString(undefined, {
    weekday: 'short', day: 'numeric', month: 'short', year: 'numeric', hour: 'numeric', minute: '2-digit',
});

/** A recovery phrase as typed or pasted: lower case, single spaces, and
 *  any numbering ("1." or "1)") from a copied list taken out. */
export function cleanPhrase(text) {
    return String(text || '').toLowerCase().split(/\s+/)
        .filter(w => w && !/^\d+[.)]?$/.test(w))
        .map(w => w.replace(/^\d+[.)]/, ''))
        .filter(Boolean)
        .join(' ');
}

/** Whether a 72-character address carries its own checksum: its last 8 hex
 *  characters are BLAKE3(address)[..4], as the wallet's parse_address_wasm
 *  checks. `blake3` hashes hex to hex (the WebAssembly's blake3_hash_hex). */
export function addressChecksumOk(address, blake3) {
    const a = String(address || '').toLowerCase();
    return /^[0-9a-f]{72}$/.test(a) && blake3(a.slice(0, 64)).slice(0, 8) === a.slice(64);
}

// ── sealed messages ─────────────────────────────────────────────────────────

/** What a name asks of a stranger's message, by the name the desktop app's
 *  picker gives the price. */
export const PRICES = [[24, 'Standard'], [28, 'Raised'], [32, 'High'], [36, 'Very high'], [40, 'Maximum']];
export const priceLabel = bits => (PRICES.find(([b]) => b === bits) || [bits, `${bits} bits`])[1];

/** A stretch of time, roughly, as people say it: "about 5 minutes". Each unit
 *  takes over at one and a half of it, so there is never "1 minute". */
export function about(secs) {
    const say = (n, what) => `about ${Math.max(2, Math.round(n))} ${what}s`;
    if (!(secs >= 1.5)) return 'about a second';
    if (secs < 90) return say(secs, 'second');
    if (secs < 90 * 60) return say(secs / 60, 'minute');
    if (secs < 36 * 3600) return say(secs / 3600, 'hour');
    return say(secs / 86_400, 'day');
}

/** A conversation's timer, for people: "1 hour", "4 weeks", "off". */
export function timerText(ttl) {
    const unit = (n, what) => `${n} ${what}${n === 1 ? '' : 's'}`;
    if (!ttl) return 'off';
    if (ttl % (7 * 86_400) === 0) return unit(ttl / (7 * 86_400), 'week');
    if (ttl % 86_400 === 0) return unit(ttl / 86_400, 'day');
    if (ttl % 3_600 === 0) return unit(ttl / 3_600, 'hour');
    return unit(Math.ceil(ttl / 60), 'minute');
}

/** When a conversation last moved, for its row: a time today, a day before. */
export const lately = (ts, now = Date.now()) => (dayLabel(ts, now) === 'Today' ? clock(ts) : dayLabel(ts, now));

/** The letter on a conversation's tile. */
export function initial(title) {
    const c = String(title || '').match(/[\p{L}\p{N}]/u);
    return c && title !== 'Anonymous' ? c[0].toUpperCase() : '?';
}
