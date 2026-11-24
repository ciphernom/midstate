// ui.js: the page's building blocks: elements, icons, rows, dialogs and
// notices. No wallet state lives here; app.js puts the pieces together.
// Everything from the chain or the user goes in as text, never as HTML.

/** An element: h('a', { href, class, onclick }, ...children). Children may be
 *  nodes, text, numbers, arrays, or null/false (left out). */
export function h(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
        if (v == null || v === false) continue;
        if (k === 'class') n.className = v;
        else if (k === 'value') n.value = v;
        else if (k === 'hidden' || k === 'checked' || k === 'disabled' || k === 'readOnly') n[k] = !!v;
        else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
        else n.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat(Infinity)) {
        if (kid == null || kid === false) continue;
        n.appendChild(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return n;
}

/** Replaces an element's children, taken as h() takes them. */
export function fill(el, ...kids) {
    el.replaceChildren(...kids.flat(Infinity).filter(k => k != null && k !== false)
        .map(k => (k instanceof Node ? k : document.createTextNode(String(k)))));
    return el;
}

export const errOf = e => (e && e.message) || String(e);
export const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── icons: drawn here, in the text's colour ─────────────────────────────────

const NS = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs) {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
}

// Each part: c(x,y,r), e(x,y,rx,ry), r(x,y,w,h,radius), or path data.
const ICONS = {
    home: ['M4 11l8-6.5 8 6.5', 'M6 9.8V20h12V9.8', 'M10 20v-5h4v5'],
    clock: ['c12,12,8.5', 'M12 7.5V12l3 2'],
    up: ['M12 19V5.5', 'M6.5 11L12 5.5l5.5 5.5'],
    down: ['M12 5v13.5', 'M6.5 13l5.5 5.5 5.5-5.5'],
    pick: ['M10.5 3.5Q19 5 20.5 13.5', 'M4 20L16.5 7.5'],
    lock: ['r5,11,14,10,2', 'M8 11V8a4 4 0 0 1 8 0v3'],
    sun: ['c12,12,4', 'M12 2.5v2', 'M12 19.5v2', 'M2.5 12h2', 'M19.5 12h2', 'M5.3 5.3l1.4 1.4', 'M17.3 17.3l1.4 1.4', 'M5.3 18.7l1.4-1.4', 'M17.3 6.7l1.4-1.4'],
    moon: ['M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z'],
    copy: ['r8,8,12,12,2', 'M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3'],
    paste: ['r6,4,12,17,2', 'M9 4V3h6v1', 'M9 10h6', 'M9 14h6'],
    scan: ['M4 9V4h5', 'M15 4h5v5', 'M20 15v5h-5', 'M9 20H4v-5', 'M8 12h8'],
    share: ['M12 15V3', 'M7 8l5-5 5 5', 'M5 12v8h14v-8'],
    check: ['M5 13l4 4L19 7'],
    left: ['M15 5l-7 7 7 7'],
    right: ['M9 5l7 7-7 7'],
    download: ['M12 4v11', 'M7 10l5 5 5-5', 'M5 20h14'],
    file: ['M6 3h8l4 4v14H6z', 'M14 3v4h4'],
    key: ['c8,15,4', 'M11 12l9-9', 'M17 6l3 3', 'M15 8l2 2'],
    globe: ['c12,12,9', 'e12,12,4,9', 'M3 12h18'],
    rescan: ['M20 12a8 8 0 1 1-2.3-5.7', 'M20 4v4h-4'],
    coins: ['e12,7,7,3', 'M5 7v5c0 1.7 3.1 3 7 3s7-1.3 7-3V7', 'M5 12v5c0 1.7 3.1 3 7 3s7-1.3 7-3v-5'],
    info: ['c12,12,9', 'M12 11v6', 'M12 7.6v.4'],
    warn: ['M12 3l10 18H2z', 'M12 10v5', 'M12 18v.4'],
    shield: ['M12 3l8 3v6c0 4.5-3.4 8.3-8 9-4.6-.7-8-4.5-8-9V6z'],
    list: ['M4 6h16', 'M4 12h16', 'M4 18h10'],
    external: ['M14 4h6v6', 'M20 4l-9 9', 'M18 14v6H4V6h6'],
    box: ['r3,4,18,5,1', 'M5 9v11h14V9', 'M10 13h4'],
    plus: ['M12 5v14', 'M5 12h14'],
    chat: ['M4 5h16v11H11l-5 4v-4H4z'],
    tag: ['M3.5 4h8l9 9-7.5 7.5-9-9z', 'c8,8.5,1.3'],
    hash: ['M10 4L8 20', 'M16 4l-2 16', 'M4.5 9h15', 'M4 15h15'],
    contrast: ['c12,12,8.5', 'M12 3.5a8.5 8.5 0 0 1 0 17z'],
};

/** A small line icon. */
export function icon(name) {
    if (name === 'gear') return gearIcon();
    const s = svgEl('svg', { viewBox: '0 0 24 24', 'aria-hidden': 'true', class: 'icon' });
    for (const part of ICONS[name] || []) {
        const n = part.slice(1).split(',').map(Number);
        if (part[0] === 'c') s.appendChild(svgEl('circle', { cx: n[0], cy: n[1], r: n[2] }));
        else if (part[0] === 'e') s.appendChild(svgEl('ellipse', { cx: n[0], cy: n[1], rx: n[2], ry: n[3] }));
        else if (part[0] === 'r') s.appendChild(svgEl('rect', { x: n[0], y: n[1], width: n[2], height: n[3], rx: n[4] }));
        else s.appendChild(svgEl('path', { d: part }));
    }
    return s;
}

function gearIcon() {
    const [cx, cy, outer, root, teeth, hole] = [12, 12, 9.5, 7.3, 8, 3];
    const step = Math.PI * 2 / teeth;
    const pts = [];
    for (let i = 0; i < teeth; i++) {
        const a = i * step;
        pts.push([a - step * 0.3, root], [a - step * 0.2, outer], [a + step * 0.2, outer], [a + step * 0.3, root]);
    }
    const p = ([a, r]) => `${(cx + r * Math.cos(a)).toFixed(2)} ${(cy + r * Math.sin(a)).toFixed(2)}`;
    const d = 'M' + pts.map(p).join('L') + 'Z'
        + `M${cx + hole} ${cy}a${hole} ${hole} 0 1 0 ${-2 * hole} 0a${hole} ${hole} 0 1 0 ${2 * hole} 0Z`;
    const s = svgEl('svg', { viewBox: '0 0 24 24', 'aria-hidden': 'true', class: 'icon' });
    s.appendChild(svgEl('path', { d }));
    return s;
}

export const spinner = () => h('span', { class: 'spinner', 'aria-hidden': 'true' });

// ── rows and groups ─────────────────────────────────────────────────────────

const labelOf = (label, sub) => h('span', { class: 'label' }, label, sub ? h('small', null, sub) : null);
const valueOf = value => (value != null && value !== '' ? h('span', { class: 'value' }, value) : null);

export const group = (...rows) => h('div', { class: 'group' }, ...rows);

/** A row that goes somewhere. */
export function linkRow(label, href, { value, sub, iconName, external } = {}) {
    return h('a', { class: 'row link', href, target: external ? '_blank' : null, rel: external ? 'noopener' : null },
        iconName ? icon(iconName) : null, labelOf(label, sub), valueOf(value), h('span', { class: 'chev' }, icon(external ? 'external' : 'right')));
}

/** A row that does something. */
export function actionRow(label, onclick, { sub, value, iconName, danger } = {}) {
    return h('button', { class: 'row action' + (danger ? ' danger' : ''), type: 'button', onclick },
        iconName ? icon(iconName) : null, labelOf(label, sub), valueOf(value));
}

/** A row that says something. */
export function infoRow(label, value, { sub, mono, wrap } = {}) {
    return h('div', { class: 'row' + (wrap ? ' stacked' : '') }, labelOf(label, sub),
        h('span', { class: 'value' + (mono ? ' mono' : '') + (wrap ? ' wrap' : '') }, value));
}

export const textRow = (...kids) => h('div', { class: 'row text' }, ...kids);

/** A row with a control on the right. */
export function controlRow(label, control, sub) {
    return h('div', { class: 'row' }, labelOf(label, sub), control);
}

/** A row of buttons, one pressed: options are [value, label]. */
export function seg(options, current, onpick, label) {
    const el = h('div', { class: 'seg', role: 'group', 'aria-label': label || null });
    const draw = cur => fill(el, options.map(([v, text]) =>
        h('button', { type: 'button', class: v === cur ? 'on' : null, 'aria-pressed': String(v === cur), onclick: () => { draw(v); onpick(v); } }, text)));
    draw(current);
    return el;
}

/** A checkbox drawn as a switch. */
export function toggle(on, onchange, label) {
    const input = h('input', { type: 'checkbox', role: 'switch', 'aria-label': label });
    input.checked = !!on;
    input.addEventListener('change', () => onchange(input.checked));
    return h('label', { class: 'switch' }, input, h('span', { class: 'track' }));
}

/** A progress bar; `fraction` null for one that only shows something is
 *  happening. */
export function bar(fraction) {
    const fillEl = h('i');
    const el = h('div', { class: 'progress' + (fraction == null ? ' busy' : '') }, fillEl);
    if (fraction != null) fillEl.style.width = Math.min(100, Math.max(0, fraction * 100)) + '%';
    return el;
}

/** A labelled field. */
export function field(label, input, hint) {
    return h('label', { class: 'field' }, h('span', null, label), input, hint || null);
}

// ── dialogs ─────────────────────────────────────────────────────────────────

let openDialogs = 0;
export const dialogOpen = () => openDialogs > 0;

/**
 * A dialog. `buttons` are [{ label, value, cls }], the last the default (Enter),
 * the first what Escape does. `input` adds a field ({ type, placeholder,
 * label }); `body` adds nodes. `check(text)` may refuse the default button:
 * it returns an error to show, or nothing. Resolves { value, text }.
 */
export function dialog({ title, message, body, input, buttons, check }) {
    return new Promise((resolve) => {
        const fieldEl = input ? h('input', {
            type: input.type || 'text', placeholder: input.placeholder || '', autocomplete: input.autocomplete || 'off',
            'aria-label': input.label || title, spellcheck: 'false', value: input.value || '',
            maxlength: input.maxlength || null,
        }) : null;
        const err = h('div', { class: 'err', role: 'alert' });
        const back = h('div', { class: 'modal-back' });
        const prevFocus = document.activeElement;
        openDialogs++;
        const close = (b) => {
            back.remove();
            openDialogs--;
            document.removeEventListener('keydown', onKey, true);
            if (prevFocus && prevFocus.focus) try { prevFocus.focus(); } catch (_) {}
            resolve({ value: b.value, text: fieldEl ? fieldEl.value : null });
        };
        const choose = async (b, isDefault) => {
            if (isDefault && check) {
                const problem = await check(fieldEl ? fieldEl.value : null);
                if (problem) { err.textContent = problem; if (fieldEl) fieldEl.focus(); return; }
            }
            close(b);
        };
        const onKey = (e) => {
            if (e.key === 'Escape') { e.preventDefault(); close(buttons[0]); }
            else if (e.key === 'Enter' && !(e.target && e.target.tagName === 'TEXTAREA')) { e.preventDefault(); choose(buttons[buttons.length - 1], true); }
        };
        const box = h('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true', 'aria-label': title },
            h('h2', null, title),
            message ? h('p', { class: 'muted' }, message) : null,
            body || null, fieldEl, err,
            h('div', { class: 'btns' }, buttons.map((b, i) => h('button', {
                type: 'button', class: 'btn ' + (b.cls || (i === buttons.length - 1 ? 'primary' : '')),
                onclick: () => choose(b, i === buttons.length - 1),
            }, b.label))));
        back.appendChild(box);
        back.addEventListener('click', (e) => { if (e.target === back) close(buttons[0]); });
        document.body.appendChild(back);
        document.addEventListener('keydown', onKey, true);
        setTimeout(() => (fieldEl || box.querySelector('.btn.primary') || box.querySelector('button')).focus(), 0);
    });
}

/** A modal whose content `build(close)` makes; resolves with what `close`
 *  is given (null for Escape or a click outside). */
export function sheet(title, build) {
    return new Promise((resolve) => {
        const back = h('div', { class: 'modal-back' });
        let closed = false;
        const close = (value) => {
            if (closed) return;
            closed = true;
            back.remove();
            openDialogs--;
            document.removeEventListener('keydown', onKey, true);
            resolve(value);
        };
        const onKey = (e) => { if (e.key === 'Escape') { e.preventDefault(); close(null); } };
        openDialogs++;
        back.appendChild(h('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true', 'aria-label': title },
            h('h2', null, title), build(close)));
        back.addEventListener('click', (e) => { if (e.target === back) close(null); });
        document.body.appendChild(back);
        document.addEventListener('keydown', onKey, true);
    });
}

/** Yes or no. */
export async function confirmBox(title, message, yes = 'Continue', no = 'Cancel') {
    const r = await dialog({ title, message, buttons: [{ label: no, value: false }, { label: yes, value: true }] });
    return r.value === true;
}

/** Something to read, with one button. */
export async function okBox(title, message, body) {
    await dialog({ title, message, body, buttons: [{ label: 'OK', value: true }] });
}

// ── notices ─────────────────────────────────────────────────────────────────

let $toasts = null;
/** A short notice in the corner that goes by itself. */
export function toast(text, ms = 4000) {
    if (!$toasts) { $toasts = h('div', { class: 'toasts', role: 'status', 'aria-live': 'polite' }); document.body.appendChild($toasts); }
    const t = h('div', { class: 'toast' }, text);
    $toasts.appendChild(t);
    setTimeout(() => t.remove(), ms);
}

/** Copies text, and says so. */
export async function copyText(text, what = 'Copied') {
    try {
        await navigator.clipboard.writeText(text);
        toast(what);
    } catch (_) {
        // Older browsers, or no permission: select it in a field instead.
        const t = h('textarea', { class: 'offscreen', readOnly: true }, text);
        document.body.appendChild(t);
        t.select();
        let ok = false;
        try { ok = document.execCommand('copy'); } catch (_) {}
        t.remove();
        toast(ok ? what : 'Couldn’t copy. Select it and copy it by hand.');
    }
}

/** Saves text as a file the browser downloads. */
export function download(name, text, type = 'application/json') {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = h('a', { href: url, download: name, hidden: true });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
