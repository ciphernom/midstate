// bridge.js: MidstateConnect, the wallet's side. A site (the IDE, a dApp)
// opens this page as a popup with ?dapp=1 and asks, over postMessage, to see
// the address (connect), fund a contract (fund) or run one (spend). Each
// request is shown to the person, who approves or rejects it; nothing
// happens without a click. Moved here from the old page (classic.html) with
// the same protocol, so midstate-provider.js is unchanged.
//
// Security, as before: a message is accepted only from an allowed origin AND
// from window.opener; every reply goes to that exact origin. Requests are
// drawn as text, never as HTML.

import { h, dialog } from './ui.js';
import { withUnit } from './format.js';

const PROTOCOL = 'midstate-bridge';
const VERSION = 1;

// The wallet and the IDE are served from one origin. Add other dApp origins
// here only for dApps hosted elsewhere. Never '*'.
const ALLOWED = [location.origin, 'http://localhost:8080'];

/** Starts the bridge if this page is a dApp's popup. `hooks`: post (to the
 *  wallet worker), isUnlocked(), primaryAddress(), onWorker(fn). */
export function startBridge(hooks) {
    if (new URLSearchParams(location.search).get('dapp') !== '1') return;
    if (!window.opener) return;

    let active = null; // { id, origin, source }
    let watching = false;

    const reply = (source, origin, msg) => {
        try { source.postMessage({ __mid: PROTOCOL, v: VERSION, ...msg }, origin); } catch (_) {}
    };
    const done = () => {
        active = null;
        setTimeout(() => { try { window.close(); } catch (_) {} }, 400);
    };

    // Until the wallet is unlocked, tell the site now and then that the person
    // is still here, so its idle timer doesn't give up.
    function whenUnlocked(id, then) {
        if (hooks.isUnlocked()) { then(); return; }
        let last = 0;
        const t = setInterval(() => {
            if (hooks.isUnlocked()) { clearInterval(t); then(); return; }
            if (active && Date.now() - last > 8000) {
                last = Date.now();
                reply(active.source, active.origin, { type: 'progress', id, message: 'Waiting for wallet unlock...' });
            }
        }, 300);
    }

    // The worker's progress on the request being run, out to the site.
    function relayProgress() {
        if (watching) return;
        watching = true;
        hooks.onWorker((data) => {
            if (!active) return;
            const p = data.payload;
            if (data.type === 'CONTRACT_TX_PROGRESS' && p && p.reqId === active.id) {
                reply(active.source, active.origin, { type: 'progress', id: active.id, message: p.msg });
            } else if (data.type === 'CONTRACT_TX_COMPLETE' && p && p.reqId === active.id) {
                reply(active.source, active.origin, { type: 'result', id: active.id, data: { txid: p.txid } });
                done();
            } else if (data.type === 'ERROR') {
                reply(active.source, active.origin, { type: 'error', id: active.id, message: String(p) });
                done();
            }
        });
    }

    async function approve(origin, title, lines) {
        const r = await dialog({
            title,
            body: h('div', { class: 'stack' },
                h('p', { class: 'muted small break' }, `Asked by ${origin}`),
                lines),
            buttons: [{ label: 'Reject', value: false }, { label: 'Approve', value: true }],
        });
        return r.value === true;
    }

    const printable = (hex) => {
        const s = String(hex || '').replace(/^0x/, '');
        let out = '';
        for (let i = 0; i + 1 < s.length; i += 2) {
            const c = parseInt(s.substr(i, 2), 16);
            out += c >= 32 && c < 127 ? String.fromCharCode(c) : '·';
        }
        return out;
    };
    const amount = v => withUnit(Number(v) || 0);
    const short = a => `${String(a || '').slice(0, 16)}…`;

    function handle(req, origin, source) {
        active = { id: req.id, origin, source };
        whenUnlocked(req.id, async () => {
            const p = req.params || {};
            const reject = (why) => { reply(source, origin, { type: 'error', id: req.id, message: why }); done(); };

            if (req.action === 'connect') {
                const addr = hooks.primaryAddress();
                const ok = await approve(origin, 'Connect to this site?',
                    [h('p', null, 'It will see your address:'), h('p', { class: 'mono small break' }, addr || '')]);
                if (!ok) return reject('User rejected the connection.');
                reply(source, origin, { type: 'result', id: req.id, data: { address: addr } });
                done();
                return;
            }

            if (req.action === 'fund' && Array.isArray(p.fundings)) {
                const ok = await approve(origin, 'Pay into a contract?', [
                    p.summary ? h('p', null, h('b', null, String(p.summary))) : null,
                    h('p', { class: 'muted small' }, 'Payments'),
                    p.fundings.map(x => h('p', { class: 'small' }, `${amount(x.amount)} to `, h('span', { class: 'mono' }, short(x.address)))),
                    (p.burns || []).length ? [h('p', { class: 'muted small' }, 'Public data in the payment'),
                        p.burns.map(b => h('p', { class: 'mono small break' }, printable(b)))] : null,
                    h('p', { class: 'muted small' }, 'Plus the network fee.'),
                ]);
                if (!ok) return reject('User rejected the funding request.');
                relayProgress();
                hooks.post({ type: 'FUND_CONTRACT', payload: { reqId: req.id, fundings: p.fundings, burns: p.burns || [] } });
                return;
            }

            if (req.action === 'fund') {
                const ok = await approve(origin, 'Pay into a contract?', [
                    h('p', { class: 'mono small break' }, String(p.contractAddress || '')),
                    h('p', null, 'Amount: ', h('b', null, amount(p.amount))),
                    p.state ? h('p', { class: 'muted small break' }, `Starting state: ${p.state}`) : null,
                    h('p', { class: 'muted small' }, 'Plus the network fee.'),
                ]);
                if (!ok) return reject('User rejected the funding request.');
                relayProgress();
                hooks.post({ type: 'FUND_CONTRACT', payload: { reqId: req.id, contractAddress: p.contractAddress, amount: p.amount, state: p.state || null } });
                return;
            }

            if (req.action === 'spend') {
                // One line per destination: a reserve split into power-of-two
                // coins would otherwise be a screenful of lines to one address.
                const byAddr = [];
                const seen = {};
                for (const o of p.outputs || []) {
                    const a = String(o.address || '');
                    if (!seen[a]) { seen[a] = { a, value: 0, coins: 0, states: 0 }; byAddr.push(seen[a]); }
                    seen[a].value += Number(o.value || 0);
                    if (o.state) seen[a].states++; else seen[a].coins++;
                }
                // With explicit inputs the net is known: what the outputs ask for
                // beyond the contract's own coins is paid from this wallet, and
                // anything left over comes back to it.
                const explicit = Array.isArray(p.inputs) && p.inputs.length;
                let net = null;
                if (explicit) {
                    let tin = 0;
                    let tout = 0;
                    for (const i of p.inputs) tin += Number(i.value || 0);
                    for (const o of p.outputs || []) tout += Number(o.value || 0);
                    net = tout > tin
                        ? h('p', null, 'You pay ', h('b', null, amount(tout - tin)), ', plus the network fee.')
                        : h('p', null, 'You receive ', h('b', null, amount(tin - tout)), ', less the network fee.');
                }
                const ok = await approve(origin, 'Run a contract?', [
                    p.summary ? h('p', null, h('b', null, String(p.summary))) : null,
                    h('p', { class: 'mono small break' }, String(p.contractAddress || '(from its code)')),
                    byAddr.length
                        ? [h('p', { class: 'muted small' }, 'Outputs'), byAddr.map(g => h('p', { class: 'small' },
                            `${amount(g.value)} to `, h('span', { class: 'mono' }, short(g.a)),
                            g.coins > 1 ? ` (${g.coins} coins)` : '', g.states ? ' (+ state)' : ''))]
                        : h('p', { class: 'muted small' }, 'No payments: it only changes the contract’s state.'),
                    net,
                ]);
                if (!ok) return reject('User rejected the execution request.');
                relayProgress();
                // Explicit inputs need no discovery; watching would rescan per trade.
                if (!explicit) hooks.post({ type: 'WATCH_CONTRACT', payload: { address: p.contractAddress } });
                // The origin comes from the browser, after the site's own fields,
                // so a site cannot name another.
                hooks.post({ type: 'SPEND_CONTRACT', payload: { reqId: req.id, ...p, origin } });
                return;
            }

            reject(`Unknown action: ${req.action}`);
        });
    }

    window.addEventListener('message', (event) => {
        if (!ALLOWED.includes(event.origin)) return;
        if (event.source !== window.opener) return;
        const d = event.data;
        if (!d || d.__mid !== PROTOCOL || d.v !== VERSION || d.type !== 'request') return;
        if (active) {
            reply(event.source, event.origin, { type: 'error', id: d.id, message: 'Wallet is busy with another request.' });
            return;
        }
        handle(d, event.origin, event.source);
    });

    // Ready: the site then sends its request. The ping carries nothing
    // private, so '*' is safe here; the site checks the origin and source of
    // what comes back, and every real reply goes to the exact origin.
    const announce = () => { try { window.opener.postMessage({ __mid: PROTOCOL, v: VERSION, type: 'ready', id: null }, '*'); } catch (_) {} };
    announce();
    setTimeout(announce, 150);
    setTimeout(announce, 500);
}
