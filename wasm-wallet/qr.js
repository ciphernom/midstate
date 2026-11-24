// qr.js: QR Code Model 2 encoder, byte and alphanumeric modes, all 40
// versions. A compact rendition of Project Nayuki's QR Code generator (MIT
// License, https://www.nayuki.io/page/qr-code-generator-library), so that
// the wallet loads no third-party script.
//
//   qrEncode(text, "M") → { size, dark(x, y) }
//
// Addresses go in upper case (`text.toUpperCase()`), which bech32 allows
// and which fits QR's alphanumeric mode: a smaller, easier code.

const ECL = { L: [0, 1], M: [1, 0], Q: [2, 3], H: [3, 2] }; // [table row, format bits]

const ECC_PER_BLOCK = [
    [-1, 7, 10, 15, 20, 26, 18, 20, 24, 30, 18, 20, 24, 26, 30, 22, 24, 28, 30, 28, 28, 28, 28, 30, 30, 26, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
    [-1, 10, 16, 26, 18, 24, 16, 18, 22, 22, 26, 30, 22, 22, 24, 24, 28, 28, 26, 26, 26, 26, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28],
    [-1, 13, 22, 18, 26, 18, 24, 18, 22, 20, 24, 28, 26, 24, 20, 30, 24, 28, 28, 26, 30, 28, 30, 30, 30, 30, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
    [-1, 17, 28, 22, 16, 22, 28, 26, 26, 24, 28, 24, 28, 22, 24, 24, 30, 28, 28, 26, 28, 30, 24, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
];
const ECC_BLOCKS = [
    [-1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 7, 8, 8, 9, 9, 10, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 24, 25],
    [-1, 1, 1, 1, 2, 2, 4, 4, 4, 5, 5, 5, 8, 9, 9, 10, 10, 11, 13, 14, 16, 17, 17, 18, 20, 21, 23, 25, 26, 28, 29, 31, 33, 35, 37, 38, 40, 43, 45, 47, 49],
    [-1, 1, 1, 2, 2, 4, 4, 6, 6, 8, 8, 8, 10, 12, 16, 12, 17, 16, 18, 21, 20, 23, 23, 25, 27, 29, 34, 34, 35, 38, 40, 43, 45, 48, 51, 53, 56, 59, 62, 65, 68],
    [-1, 1, 1, 2, 4, 4, 4, 5, 6, 8, 8, 11, 11, 16, 16, 18, 16, 19, 21, 25, 25, 25, 34, 30, 32, 35, 37, 40, 42, 45, 48, 51, 54, 57, 60, 63, 66, 70, 74, 77, 81],
];
const ALNUM = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:';

function rawModules(ver) {
    let r = (16 * ver + 128) * ver + 64;
    if (ver >= 2) {
        const n = Math.floor(ver / 7) + 2;
        r -= (25 * n - 10) * n - 55;
        if (ver >= 7) r -= 36;
    }
    return r;
}

const dataCodewords = (ver, e) => Math.floor(rawModules(ver) / 8) - ECC_PER_BLOCK[e][ver] * ECC_BLOCKS[e][ver];

function gfMul(x, y) {
    let z = 0;
    for (let i = 7; i >= 0; i--) {
        z = (z << 1) ^ ((z >>> 7) * 0x11d);
        z ^= ((y >>> i) & 1) * x;
    }
    return z;
}

function rsDivisor(degree) {
    const r = new Array(degree).fill(0);
    r[degree - 1] = 1;
    let root = 1;
    for (let i = 0; i < degree; i++) {
        for (let j = 0; j < r.length; j++) {
            r[j] = gfMul(r[j], root);
            if (j + 1 < r.length) r[j] ^= r[j + 1];
        }
        root = gfMul(root, 0x02);
    }
    return r;
}

function rsRemainder(data, divisor) {
    const r = divisor.map(() => 0);
    for (const b of data) {
        const factor = b ^ r.shift();
        r.push(0);
        divisor.forEach((c, i) => { r[i] ^= gfMul(c, factor); });
    }
    return r;
}

/** Mode, character count bits per version band, and the segment's bits. */
function segment(text) {
    if ([...text].every(c => ALNUM.includes(c))) {
        const bits = [];
        for (let i = 0; i + 1 < text.length; i += 2) {
            push(bits, ALNUM.indexOf(text[i]) * 45 + ALNUM.indexOf(text[i + 1]), 11);
        }
        if (text.length % 2) push(bits, ALNUM.indexOf(text[text.length - 1]), 6);
        return { mode: 0x2, counts: [9, 11, 13], chars: text.length, bits };
    }
    const bytes = new TextEncoder().encode(text);
    const bits = [];
    for (const b of bytes) push(bits, b, 8);
    return { mode: 0x4, counts: [8, 16, 16], chars: bytes.length, bits };
}

function push(bits, value, len) {
    for (let i = len - 1; i >= 0; i--) bits.push((value >>> i) & 1);
}

export function qrEncode(text, level = 'M') {
    const [e, formatBits] = ECL[level];
    const seg = segment(text);
    let ver = 1;
    let used;
    for (; ; ver++) {
        if (ver > 40) throw new Error('too much data for a QR code');
        const cc = seg.counts[ver < 10 ? 0 : ver < 27 ? 1 : 2];
        if (seg.chars >= 1 << cc) continue;
        used = 4 + cc + seg.bits.length;
        if (used <= dataCodewords(ver, e) * 8) break;
    }
    const cap = dataCodewords(ver, e) * 8;
    const bits = [];
    push(bits, seg.mode, 4);
    push(bits, seg.chars, seg.counts[ver < 10 ? 0 : ver < 27 ? 1 : 2]);
    bits.push(...seg.bits);
    push(bits, 0, Math.min(4, cap - bits.length));
    push(bits, 0, (8 - (bits.length % 8)) % 8);
    for (let pad = 0xec; bits.length < cap; pad ^= 0xec ^ 0x11) push(bits, pad, 8);
    const data = [];
    for (let i = 0; i < bits.length; i += 8) {
        let b = 0;
        for (let j = 0; j < 8; j++) b = (b << 1) | bits[i + j];
        data.push(b);
    }

    // Error correction, split into blocks and interleaved.
    const numBlocks = ECC_BLOCKS[e][ver];
    const eccLen = ECC_PER_BLOCK[e][ver];
    const raw = Math.floor(rawModules(ver) / 8);
    const numShort = numBlocks - (raw % numBlocks);
    const shortLen = Math.floor(raw / numBlocks);
    const div = rsDivisor(eccLen);
    const blocks = [];
    for (let i = 0, k = 0; i < numBlocks; i++) {
        const dat = data.slice(k, k + shortLen - eccLen + (i < numShort ? 0 : 1));
        k += dat.length;
        const ecc = rsRemainder(dat, div);
        if (i < numShort) dat.push(0);
        blocks.push(dat.concat(ecc));
    }
    const codewords = [];
    for (let i = 0; i < blocks[0].length; i++) {
        blocks.forEach((b, j) => {
            if (i !== shortLen - eccLen || j >= numShort) codewords.push(b[i]);
        });
    }

    // The symbol.
    const size = ver * 4 + 17;
    const mod = Array.from({ length: size }, () => new Array(size).fill(false));
    const fn = Array.from({ length: size }, () => new Array(size).fill(false));
    const set = (x, y, d) => { mod[y][x] = d; fn[y][x] = true; };
    for (let i = 0; i < size; i++) { set(6, i, i % 2 === 0); set(i, 6, i % 2 === 0); }
    const finder = (cx, cy) => {
        for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) {
            const x = cx + dx, y = cy + dy, d = Math.max(Math.abs(dx), Math.abs(dy));
            if (x >= 0 && x < size && y >= 0 && y < size) set(x, y, d !== 2 && d !== 4);
        }
    };
    finder(3, 3); finder(size - 4, 3); finder(3, size - 4);
    const align = [];
    if (ver > 1) {
        const n = Math.floor(ver / 7) + 2;
        const step = Math.floor((ver * 8 + n * 3 + 5) / (n * 4 - 4)) * 2;
        for (let pos = size - 7; align.length < n - 1; pos -= step) align.unshift(pos);
        align.unshift(6);
    }
    align.forEach((ax, i) => align.forEach((ay, j) => {
        if ((i === 0 && j === 0) || (i === 0 && j === align.length - 1) || (i === align.length - 1 && j === 0)) return;
        for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) set(ax + dx, ay + dy, Math.max(Math.abs(dx), Math.abs(dy)) !== 1);
    }));
    const format = (mask) => {
        const d = (formatBits << 3) | mask;
        let rem = d;
        for (let i = 0; i < 10; i++) rem = (rem << 1) ^ ((rem >>> 9) * 0x537);
        const b = ((d << 10) | rem) ^ 0x5412;
        const bit = i => ((b >>> i) & 1) !== 0;
        for (let i = 0; i <= 5; i++) set(8, i, bit(i));
        set(8, 7, bit(6)); set(8, 8, bit(7)); set(7, 8, bit(8));
        for (let i = 9; i < 15; i++) set(14 - i, 8, bit(i));
        for (let i = 0; i < 8; i++) set(size - 1 - i, 8, bit(i));
        for (let i = 8; i < 15; i++) set(8, size - 15 + i, bit(i));
        set(8, size - 8, true);
    };
    format(0);
    if (ver >= 7) {
        let rem = ver;
        for (let i = 0; i < 12; i++) rem = (rem << 1) ^ ((rem >>> 11) * 0x1f25);
        const b = (ver << 12) | rem;
        for (let i = 0; i < 18; i++) {
            const d = ((b >>> i) & 1) !== 0, a = size - 11 + (i % 3), c = Math.floor(i / 3);
            set(a, c, d); set(c, a, d);
        }
    }
    let i = 0;
    for (let right = size - 1; right >= 1; right -= 2) {
        if (right === 6) right = 5;
        for (let v = 0; v < size; v++) for (let j = 0; j < 2; j++) {
            const x = right - j, up = ((right + 1) & 2) === 0, y = up ? size - 1 - v : v;
            if (!fn[y][x] && i < codewords.length * 8) {
                mod[y][x] = ((codewords[i >>> 3] >>> (7 - (i & 7))) & 1) !== 0;
                i++;
            }
        }
    }

    const masked = (m) => {
        const out = mod.map(r => r.slice());
        for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
            let inv;
            switch (m) {
                case 0: inv = (x + y) % 2 === 0; break;
                case 1: inv = y % 2 === 0; break;
                case 2: inv = x % 3 === 0; break;
                case 3: inv = (x + y) % 3 === 0; break;
                case 4: inv = (Math.floor(x / 3) + Math.floor(y / 2)) % 2 === 0; break;
                case 5: inv = ((x * y) % 2) + ((x * y) % 3) === 0; break;
                case 6: inv = (((x * y) % 2) + ((x * y) % 3)) % 2 === 0; break;
                default: inv = (((x + y) % 2) + ((x * y) % 3)) % 2 === 0;
            }
            if (!fn[y][x] && inv) out[y][x] = !out[y][x];
        }
        return out;
    };
    let best = null, bestScore = Infinity;
    for (let m = 0; m < 8; m++) {
        format(m);
        const grid = masked(m);
        const score = penalty(grid, size);
        if (score < bestScore) { bestScore = score; best = { grid, m }; }
    }
    format(best.m);
    // format() wrote the chosen mask's format bits into `mod`; copy them.
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) if (fn[y][x]) best.grid[y][x] = mod[y][x];
    return { size, version: ver, dark: (x, y) => best.grid[y][x] };
}

function penalty(g, size) {
    let score = 0;
    const lines = (get) => {
        for (let a = 0; a < size; a++) {
            let color = false, run = 0;
            const hist = [0, 0, 0, 0, 0, 0, 0];
            for (let b = 0; b < size; b++) {
                if (get(a, b) === color) {
                    run++;
                    if (run === 5) score += 3;
                    else if (run > 5) score++;
                } else {
                    addHistory(run, hist, size);
                    if (!color) score += finderLike(hist) * 40;
                    color = get(a, b);
                    run = 1;
                }
            }
            if (color) { addHistory(run, hist, size); run = 0; }
            run += size;
            addHistory(run, hist, size);
            score += finderLike(hist) * 40;
        }
    };
    lines((a, b) => g[a][b]);
    lines((a, b) => g[b][a]);
    for (let y = 0; y < size - 1; y++) for (let x = 0; x < size - 1; x++) {
        const c = g[y][x];
        if (c === g[y][x + 1] && c === g[y + 1][x] && c === g[y + 1][x + 1]) score += 3;
    }
    let dark = 0;
    for (const row of g) for (const c of row) if (c) dark++;
    const total = size * size;
    score += (Math.ceil(Math.abs(dark * 20 - total * 10) / total) - 1) * 10;
    return score;
}

function addHistory(run, hist, size) {
    if (hist[0] === 0) run += size;
    hist.pop();
    hist.unshift(run);
}

function finderLike(h) {
    const n = h[1];
    const core = n > 0 && h[2] === n && h[3] === n * 3 && h[4] === n && h[5] === n;
    return (core && h[0] >= n * 4 && h[6] >= n ? 1 : 0) + (core && h[6] >= n * 4 && h[0] >= n ? 1 : 0);
}

/** An <svg> element (built with DOM calls, so CSP needs no exceptions). */
export function qrSvg(text, level = 'M', border = 3) {
    const q = qrEncode(text, level);
    const NS = 'http://www.w3.org/2000/svg';
    const n = q.size + border * 2;
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${n} ${n}`);
    svg.setAttribute('shape-rendering', 'crispEdges');
    svg.setAttribute('role', 'img');
    const bg = document.createElementNS(NS, 'rect');
    bg.setAttribute('width', n); bg.setAttribute('height', n); bg.setAttribute('fill', '#fff');
    svg.appendChild(bg);
    let d = '';
    for (let y = 0; y < q.size; y++) for (let x = 0; x < q.size; x++) {
        if (q.dark(x, y)) d += `M${x + border},${y + border}h1v1h-1z`;
    }
    const path = document.createElementNS(NS, 'path');
    path.setAttribute('d', d);
    path.setAttribute('fill', '#000');
    svg.appendChild(path);
    return svg;
}
