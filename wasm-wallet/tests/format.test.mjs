// node --test wasm-wallet/tests/
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
    gmds, withUnit, plainAmount, parseAmount, ONE, shortAddr, groupHex, cleanAddress, sendStep, dayLabel,
    progressWords, speed, cleanPhrase, addressChecksumOk, gmdsExact, exactUnit, priceLabel, about, timerText, lately, initial, clock,
} from '../format.js';
import { initSync, blake3_hash_hex, address_to_checksummed_hex } from '../pkg/wasm_wallet.js';

initSync({ module: fs.readFileSync(new URL('../pkg/wasm_wallet_bg.wasm', import.meta.url)) });

test('amounts read in gMDS, as the desktop shows them', () => {
    assert.equal(gmds(0), '0');
    assert.equal(gmds(ONE), '1');
    assert.equal(gmds(37 * ONE + ONE / 2), '37.5');
    assert.equal(gmds(ONE / 4), '0.25');
    assert.equal(gmds(1234 * ONE), '1,234');
    assert.equal(gmds(Math.floor(ONE / 3)), '0.3333');
    assert.equal(gmds(2 * ONE - 1), '2', 'rounds up into the next whole');
    assert.equal(gmds(61), '0.000000057', 'a fee is not nothing');
    assert.equal(withUnit(ONE / 2), '0.5 gMDS');
});

test('typed amounts become base units', () => {
    assert.deepEqual(parseAmount('1'), { units: ONE });
    assert.deepEqual(parseAmount(' 2.5 '), { units: 2 * ONE + ONE / 2 });
    assert.deepEqual(parseAmount('.5'), { units: ONE / 2 });
    assert.deepEqual(parseAmount('2,5'), { units: 2 * ONE + ONE / 2 }, 'a decimal comma');
    assert.deepEqual(parseAmount('1,000.5'), { units: 1000 * ONE + ONE / 2 });
    assert.deepEqual(parseAmount('1,000,000'), { units: 1_000_000 * ONE });
    assert.deepEqual(parseAmount('1,234'), { error: 'Use a dot for decimals' });
    assert.deepEqual(parseAmount('0.1'), { units: 107_374_182 }, 'rounds to the nearest unit');
    assert.deepEqual(parseAmount('0.000000001'), { units: 1 });
    assert.deepEqual(parseAmount(''), { error: '' });
    assert.deepEqual(parseAmount('0'), { error: 'Enter an amount' });
    assert.deepEqual(parseAmount('0.0000000001'), { error: 'Use at most 9 decimals' });
    assert.deepEqual(parseAmount('1.2.3'), { error: 'That isn’t a number' });
    assert.deepEqual(parseAmount('-1'), { error: 'That isn’t a number' });
    assert.deepEqual(parseAmount('1e3'), { error: 'That isn’t a number' });
    assert.deepEqual(parseAmount('99999999999'), { error: 'That is too much' });
    // What is typed reads back as itself, and Max's figure reads back too.
    for (const t of ['1', '2.5', '0.25', '37.5', '1,234.5']) assert.equal(gmds(parseAmount(t).units), t);
    for (const v of [1, 61, Math.floor(ONE / 3), 37 * ONE + ONE / 2, 40_265_318_339]) {
        assert.ok(Math.abs(parseAmount(plainAmount(v)).units - v) <= 1, `${v}`);
    }
});

test('addresses', () => {
    const a = 'ab'.repeat(36);
    assert.equal(cleanAddress(' 0x' + a.toUpperCase() + ' '), a);
    assert.equal(cleanAddress(a.slice(2)), null, 'too short');
    assert.equal(cleanAddress('zz' + a.slice(2)), null);
    assert.equal(shortAddr(a), 'ababab…abab');
    assert.equal(groupHex('abcdefgh'), 'abcd efgh');
});

test('send steps only move forward', () => {
    const order = ['built', 'pow', 'pow_done', 'commit_sent', 'commit_mined', 'reveal_sent', 'reveal_mined'];
    const f = order.map(s => sendStep(s)[1]);
    assert.ok(f.every((x, i) => i === 0 || x > f[i - 1]), f.join());
    assert.equal(sendStep('reveal_mined')[0], 'Sent');
});

test('days', () => {
    const now = new Date(2026, 10, 23, 20, 0).getTime();
    const ts = d => Math.floor(d.getTime() / 1000);
    assert.equal(dayLabel(ts(new Date(2026, 10, 23, 9, 0)), now), 'Today');
    assert.equal(dayLabel(ts(new Date(2026, 10, 22, 9, 0)), now), 'Yesterday');
    const thu = new Date(2026, 10, 19, 9, 0);
    assert.equal(dayLabel(ts(thu), now), thu.toLocaleDateString(undefined, { weekday: 'long' }));
    // Older days are dates, in this computer's own order; a past year shows.
    assert.match(dayLabel(ts(new Date(2026, 9, 1, 9, 0)), now), /\b1\b/);
    assert.doesNotMatch(dayLabel(ts(new Date(2026, 9, 1, 9, 0)), now), /2026/);
    assert.match(dayLabel(ts(new Date(2025, 9, 1, 9, 0)), now), /2025/);
});

test('a tidy-up’s messages only move it forward', () => {
    const said = [
        'Preparing defragmentation...', 'Mining PoW...', 'Broadcasting commit...',
        'Waiting for commit to be mined (1/2)...', 'Broadcasting defrag reveal...', 'Waiting for defrag to be mined (2/2)...',
    ].map(progressWords);
    for (let i = 1; i < said.length; i++) assert.ok(said[i][1] >= said[i - 1][1], `${i}: ${said[i]}`);
    assert.equal(said[0][0], 'Preparing');
    assert.equal(said[3][0], 'Waiting for the network');
    assert.equal(said[5][0], 'Finishing');
});

test('mining speed', () => {
    assert.equal(speed(0), '0 attempts/s');
    assert.equal(speed(950.4), '950 attempts/s');
    assert.equal(speed(1234), '1.23 k attempts/s');
    assert.equal(speed(56789), '56.8 k attempts/s');
    assert.equal(speed(2.5e6), '2.50 M attempts/s');
});

test('a pasted recovery phrase loses its numbering and spacing', () => {
    assert.equal(cleanPhrase('  Abandon   ability\nable '), 'abandon ability able');
    assert.equal(cleanPhrase('1. abandon 2. ability 3) able'), 'abandon ability able');
    assert.equal(cleanPhrase('1.abandon 2)ability'), 'abandon ability');
    assert.equal(cleanPhrase(''), '');
});

test('an address carries its own checksum', () => {
    const plain = 'ab'.repeat(32);
    const full = address_to_checksummed_hex(plain);
    assert.equal(full.length, 72);
    assert.ok(addressChecksumOk(full, blake3_hash_hex));
    assert.ok(addressChecksumOk(full.toUpperCase(), blake3_hash_hex));
    // One character off anywhere: the checksum no longer matches.
    const typo = full.slice(0, 10) + (full[10] === '0' ? '1' : '0') + full.slice(11);
    assert.ok(!addressChecksumOk(typo, blake3_hash_hex));
    const badSum = full.slice(0, 71) + (full[71] === '0' ? '1' : '0');
    assert.ok(!addressChecksumOk(badSum, blake3_hash_hex));
    assert.ok(!addressChecksumOk(plain, blake3_hash_hex));
    assert.ok(!addressChecksumOk('', blake3_hash_hex));
});

test('a reviewed amount shows every decimal it has', () => {
    assert.equal(gmdsExact(4.5 * ONE), '4.5');
    assert.equal(gmdsExact(4.5 * ONE - 905), '4.499999157');
    assert.equal(gmdsExact(1234 * ONE), '1,234');
    assert.equal(exactUnit(483), '0.00000045 gMDS');
    assert.equal(gmdsExact(0), '0');
});

test('the words around sealed messages', () => {
    assert.equal(priceLabel(24), 'Standard');
    assert.equal(priceLabel(36), 'Very high');
    assert.equal(priceLabel(30), '30 bits');
    assert.equal(about(0.2), 'about a second');
    assert.equal(about(1.6), 'about 2 seconds');
    assert.equal(about(89), 'about 89 seconds');
    assert.equal(about(90), 'about 2 minutes');
    assert.equal(about(4 * 3600), 'about 4 hours');
    assert.equal(about(40 * 3600), 'about 2 days');
    assert.equal(about(NaN), 'about a second');
    assert.equal(timerText(0), 'off');
    assert.equal(timerText(3_600), '1 hour');
    assert.equal(timerText(86_400), '1 day');
    assert.equal(timerText(7 * 86_400), '1 week');
    assert.equal(timerText(28 * 86_400), '4 weeks');
    assert.equal(timerText(90), '2 minutes');
    assert.equal(initial('nick.mds'), 'N');
    assert.equal(initial('Anonymous'), '?');
    assert.equal(initial('9lives.mds'), '9');
    const now = new Date(2026, 10, 24, 15, 0).getTime();
    assert.equal(lately(now / 1000 - 60, now), clock(now / 1000 - 60));
    assert.equal(lately(now / 1000 - 86_400, now), 'Yesterday');
});
