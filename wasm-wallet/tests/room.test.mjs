// node --test wasm-wallet/tests/
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
    DICTIONARY, MAX_WORDS, isPublic, textOf, parseDraft, suggest, senderLabel, merge, view, checkOut, keyOf,
} from '../room.js';

const idx = w => DICTIONARY.indexOf(w);
const chat = (nonce, sender, words, extra = {}) => ({ sender, timestamp: 1_700_000_000 + nonce, nonce, reply_to: null, words, attachments: [], ...extra });
const ADDR = 'ab'.repeat(32) + '01020304';

test('the dictionary is the node’s, word for word', () => {
    const rust = fs.readFileSync(new URL('../../src/chat.rs', import.meta.url), 'utf8');
    const body = rust.slice(rust.indexOf('pub const CHAT_DICTIONARY'), rust.indexOf('];', rust.indexOf('pub const CHAT_DICTIONARY')));
    const words = [...body.slice(body.indexOf('[')).matchAll(/"([^"]*)"/g)].map(m => m[1]);
    assert.equal(words.length, 256);
    assert.deepEqual([...DICTIONARY], words);
});

test('frames and sealed messages are not public messages', () => {
    assert.equal(isPublic(chat(1, 'p', [idx('gm')])), true);
    assert.equal(isPublic(chat(2, 'p', [255, 69])), false, 'a wire frame');
    assert.equal(isPublic(chat(3, 'p', [], { attachments: [{ kind: 'encrypted_message', value: '00' }] })), false, 'a sealed message');
    assert.equal(isPublic(chat(4, 'p', [], { attachments: [{ kind: 'address', value: ADDR }] })), true, 'an address alone');
    assert.equal(isPublic(chat(5, 'p', [])), false, 'nothing to show');
    assert.equal(isPublic({ words: [1] }), false, 'no sender');
    assert.equal(isPublic(null), false);
});

test('words read as the room reads them, punctuation against the word before', () => {
    assert.equal(textOf([idx('gm'), idx('fren'), idx('!')]), 'gm fren!');
    assert.equal(textOf([idx('?')]), '?');
    assert.equal(textOf([idx('wait'), idx('...'), idx('ok')]), 'wait... ok');
    assert.equal(textOf([300]), '?', 'a word this page does not know');
});

test('a typed line becomes the bus’s words', () => {
    let r = parseDraft('GM fren!');
    assert.deepEqual(r.words, [idx('gm'), idx('fren'), idx('!')]);
    assert.equal(r.problem, null);
    r = parseDraft('i need help...');
    assert.deepEqual(r.words, [idx('I'), idx('need'), idx('help'), idx('...')]);
    r = parseDraft('  :)  🚀 ');
    assert.deepEqual(r.words, [idx(':)'), idx('🚀')]);
    r = parseDraft('hello world');
    assert.deepEqual(r.unknown, ['hello', 'world']);
    assert.match(r.problem, /aren’t among the room’s words/);
    r = parseDraft('gm hello');
    assert.match(r.problem, /“hello” isn’t one of the room’s words/);
    r = parseDraft('1 2 3 4 5 6 7 8 9 10 100');
    assert.equal(r.words.length, 11);
    assert.match(r.problem, new RegExp(`${MAX_WORDS} at most`));
    r = parseDraft('⚡ gm');
    assert.match(r.problem, /can’t start with ⚡/, 'it would be read as a frame');
    assert.equal(parseDraft('gm ⚡').problem, null);
    assert.deepEqual(parseDraft('').words, []);
});

test('the picker offers words that start with what is typed', () => {
    assert.deepEqual(suggest('wa'), ['wallet', 'want', 'was', 'wait', 'wagmi']);
    assert.deepEqual(suggest('i'), ['I', 'incoming', 'is', 'it', 'in', 'if', 'ip', 'issue'], 'the word itself first');
    assert.equal(suggest('x').length, 0);
    assert.equal(suggest('').length, 0);
    assert.equal(suggest('s', 3).length, 3);
});

test('senders are told apart at a glance', () => {
    assert.equal(senderLabel('AB'.repeat(32)), 'abababab');
    assert.equal(senderLabel('12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN'), '…x6nXTN');
    assert.equal(senderLabel(''), 'unknown');
});

test('the room keeps public messages once each, in time order, the newest kept', () => {
    const a = chat(1, 'p', [idx('gm')]);
    const b = chat(2, 'q', [idx('gn')]);
    const frame = chat(3, 'p', [255, 70]);
    let held = merge([], [b, a, frame]);
    assert.deepEqual(held.map(keyOf), [keyOf(a), keyOf(b)]);
    held = merge(held, [a, chat(1, 'q', [idx('ok')])]);
    assert.equal(held.length, 3, 'the same nonce from another sender is another message');
    const many = Array.from({ length: 30 }, (_, i) => chat(100 + i, 'p', [idx('ok')]));
    assert.equal(merge(held, many, 10).length, 10);
    assert.equal(merge(held, many, 10)[9].nonce, 129);
});

test('the view says what each message answers and which are this wallet’s', () => {
    const me = 'cd'.repeat(32);
    const a = chat(1, 'p', [idx('gm')]);
    const b = chat(2, me, [idx('gm'), idx('fren')], { reply_to: 1 });
    const c = chat(3, 'q', [idx('ok')], { reply_to: 99 });
    const d = chat(4, 'q', [], { attachments: [{ kind: 'address', value: ADDR }] });
    const v = view([a, b, c, d], me);
    assert.equal(v[0].mine, false);
    assert.equal(v[1].mine, true);
    assert.deepEqual(v[1].replyTo, { sender: 'p', text: 'gm', mine: false });
    assert.deepEqual(v[2].replyTo, { sender: null, text: null, mine: false }, 'answers something older than the room holds');
    assert.equal(v[3].text, '');
    assert.deepEqual(v[3].attachments, [{ kind: 'address', value: ADDR }]);
    assert.equal(view([a], null)[0].mine, false);
});

test('a send is checked as the node checks it', () => {
    assert.equal(checkOut([idx('gm')], []), null);
    assert.equal(checkOut([], [{ kind: 'address', value: ADDR }]), null);
    assert.match(checkOut([], []), /Say something/);
    assert.match(checkOut(new Array(11).fill(1), []), /10 words/);
    assert.match(checkOut([255, 1], []), /can’t start/);
    assert.match(checkOut([256], []), /room’s words/);
    assert.match(checkOut([1], new Array(5).fill({ kind: 'address', value: ADDR })), /4 attachments/);
    assert.match(checkOut([1], [{ kind: 'signature', value: '00' }]), /Only addresses/);
    assert.match(checkOut([1], [{ kind: 'address', value: 'ab'.repeat(32) }]), /Only addresses/, 'checksummed, as the node gives them');
});
