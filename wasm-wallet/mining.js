// mining.js: mining in the browser, for the page. Moved here from the old
// single-file page (classic.html) with the same protocol and the same
// safeguards; only the drawing is gone. The page reads `mining.state` and
// listens with `mining.on`.
//
// Mining threads are miner.js workers. They search nonces in the wallet's
// WebAssembly; worker 0 tries WebGPU first. The wallet worker (worker.js)
// builds the work: GET_TEMPLATE → TEMPLATE_READY, from a pool (the default:
// small, frequent payouts) or, solo, from a node reached directly over
// WebRTC. A hit is SUBMIT_MINED_BLOCK → BLOCK_SUBMITTED. A pool that fails
// its fairness audit (POOL_AUDIT_FAILED) stops mining outright.

export const DEFAULT_POOL_URL = 'https://rpc.cypherpunk.gold/pool';
/** Midstate Walk replaces the BLAKE3 chain at this height; only for the
 *  countdown before it. */
export const WALK_ACTIVATION_HEIGHT = 371500;
const TEMPLATE_POLL_INTERVAL = 10_000;

export const MAX_THREADS = Math.max(1, (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4);
export const IS_MOBILE = typeof navigator !== 'undefined' && (
    (navigator.userAgentData && navigator.userAgentData.mobile)
    || /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || ''));
/** Phones heat up and throttle under all-core load, so they start on two
 *  threads; everything else leaves two cores for the page and the OS. */
export const DEFAULT_THREADS = IS_MOBILE ? Math.max(1, Math.min(2, MAX_THREADS - 1)) : Math.max(1, MAX_THREADS - 2);

/** How hard each thread works: the pause between searches. */
export const EFFORTS = [
    { label: 'Light', ms: 4000 },
    { label: 'Medium', ms: 1500 },
    { label: 'Balanced', ms: 400 },
    { label: 'High', ms: 50 },
    { label: 'Full', ms: 0 },
];
export const DEFAULT_EFFORT = IS_MOBILE ? 0 : 2;

/** Expected attempts to beat a 64-hex big-endian 256-bit target. */
export function expectedAttempts(targetHex) {
    const hex = String(targetHex || '').toLowerCase();
    if (!/^[0-9a-f]{64}$/.test(hex)) return 0;
    const lead = hex.replace(/^0+/, '');
    if (!lead) return 0;
    const zeros = hex.length - lead.length;
    const top = parseInt(lead.slice(0, 13).padEnd(13, '0'), 16) / Math.pow(16, 13);
    const fraction = top * Math.pow(16, -zeros);
    return fraction > 0 ? 1 / fraction : 0;
}

/** Expected base units a day at `nps` attempts a second, or null. In a
 *  pool, its share of the network's blocks less the pool's fee. */
export function earningsPerDay(nps, tmpl, pooled) {
    if (!tmpl || !(nps > 0)) return null;
    const attempts = expectedAttempts(pooled ? tmpl.network_target : tmpl.target);
    const reward = Number(tmpl.blockReward) || 0;
    if (!(attempts > 0) || !(reward > 0)) return null;
    const fee = pooled ? Math.min(100, Math.max(0, Number(tmpl.poolFeePercent) || 0)) : 0;
    return (nps * 86400 / attempts) * reward * (1 - fee / 100);
}

const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch (_) { return d; } },
    set(k, v) { try { v === null || v === '' ? localStorage.removeItem(k) : localStorage.setItem(k, String(v)); } catch (_) {} },
};

const MIDWIMBLE_ADDRESS = /^mw1[qpzry9x8gf2tvdw0s3jn54khce6mua7l]{100,}$/;

export class Mining {
    /** `wallet`: the wallet worker. `isWebRTC()`: a node reached directly.
     *  `log(text)`. */
    constructor({ wallet, isWebRTC, log }) {
        this.wallet = wallet;
        this.isWebRTC = isWebRTC;
        this.log = log || (() => {});
        this.listeners = new Set();
        this.miners = [];
        this.template = null;
        this.gpuWaiters = new Map();
        this.pollTimer = null;
        this.tickTimer = null;
        this.wakeLock = null;
        this.battery = null;
        this.batteryPaused = false;
        this.blocks = [];          // this session's mined blocks, newest first
        this.state = {
            active: false,
            nps: 0,
            since: 0,
            nonces: 0,
            blocksFound: 0,
            poolScore: -1,
            mwScore: null,
            merge: undefined,      // the pool's word on merged mining
            backend: 'cpu',        // or the GPU adapter's name
            pow: null,
            tip: 0,
            note: '',              // why it is not mining, when it should be
            ...this.loadSettings(),
        };
        this.watchBattery();
        if (typeof document !== 'undefined') {
            document.addEventListener('visibilitychange', () => {
                if (document.visibilityState === 'visible' && this.state.active) this.holdScreen();
            });
        }
    }

    loadSettings() {
        // Earlier builds saved "pool" without acting on it; the old page reset
        // such a saved choice to solo once and said so. Keep that, so a saved
        // choice means the same thing it did.
        let mode = store.get('mineMode', null);
        if (mode === 'pool' && !store.get('poolModeMigrated', null)) {
            mode = 'solo';
            store.set('mineMode', 'solo');
            store.set('poolModeMigrated', '1');
        }
        return {
            mode: mode === 'solo' ? 'solo' : 'pool',
            poolUrl: store.get('poolUrl', DEFAULT_POOL_URL),
            midwimble: store.get('poolMidwimbleAddress', ''),
            threads: Math.min(MAX_THREADS, Math.max(1, Number(store.get('mineThreads', 0)) || DEFAULT_THREADS)),
            effort: Math.min(EFFORTS.length - 1, Math.max(0, Number(store.get('mineEffort', DEFAULT_EFFORT)))),
            pauseOnBattery: store.get('minePauseOnBattery', '1') === '1',
        };
    }

    on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
    emit(kind, data) { for (const fn of this.listeners) { try { fn(kind, data); } catch (_) {} } }
    changed() { this.emit('change'); }

    get pooled() { return this.state.mode === 'pool'; }
    get throttleMs() { return EFFORTS[this.state.effort].ms; }
    modePayload() { return { mode: this.state.mode, url: this.state.poolUrl.trim(), midwimble: this.state.midwimble }; }

    // ── settings ───────────────────────────────────────────────────────────

    setMode(mode) {
        this.state.mode = mode === 'solo' ? 'solo' : 'pool';
        store.set('mineMode', this.state.mode);
        store.set('poolModeMigrated', '1');
        this.wallet.postMessage({ type: 'SET_MINING_MODE', payload: this.modePayload() });
        this.changed();
    }
    setPoolUrl(url) {
        this.state.poolUrl = String(url || '').trim() || DEFAULT_POOL_URL;
        store.set('poolUrl', this.state.poolUrl === DEFAULT_POOL_URL ? null : this.state.poolUrl);
        this.wallet.postMessage({ type: 'SET_MINING_MODE', payload: this.modePayload() });
        this.changed();
    }
    /** A midwimble address paid the midwimble part of each share, or ''. */
    setMidwimble(addr) {
        const a = String(addr || '').trim().toLowerCase();
        if (a && !MIDWIMBLE_ADDRESS.test(a)) throw new Error('That isn’t a midwimble address (mw1…)');
        this.state.midwimble = a;
        this.state.mwScore = null;
        store.set('poolMidwimbleAddress', a);
        this.wallet.postMessage({ type: 'SET_MINING_MODE', payload: this.modePayload() });
        this.changed();
    }
    setThreads(n) {
        const count = Math.min(MAX_THREADS, Math.max(1, Math.round(n)));
        this.state.threads = count;
        store.set('mineThreads', count === DEFAULT_THREADS ? null : count);
        if (this.state.active) this.resize(count);
        this.changed();
    }
    setEffort(i) {
        this.state.effort = Math.min(EFFORTS.length - 1, Math.max(0, Math.round(i)));
        store.set('mineEffort', this.state.effort);
        for (const m of this.miners) m.worker.postMessage({ type: 'THROTTLE', payload: { throttle_ms: this.throttleMs } });
        this.changed();
    }
    setPauseOnBattery(on) {
        this.state.pauseOnBattery = !!on;
        store.set('minePauseOnBattery', on ? '1' : '0');
        this.checkBattery();
        this.changed();
    }

    // ── starting and stopping ────────────────────────────────────────────

    /** Start; throws why it cannot. */
    start() {
        if (this.state.active) return;
        if (!this.pooled && !this.isWebRTC()) throw new Error('Solo mining needs a direct connection to a node (Settings → Network). Pool mining works without one.');
        if (this.pooled && !this.state.poolUrl.trim()) throw new Error('Choose a pool first.');
        if (this.batteryTooLow()) throw new Error('The battery is below 20%. Plug in to mine, or turn off the battery pause.');
        this.wallet.postMessage({ type: 'SET_MINING_MODE', payload: this.modePayload() });
        // The wallet worker gates its fresh-template-on-new-block on this.
        this.wallet.postMessage({ type: 'SET_MINING_ACTIVE', payload: { active: true } });
        Object.assign(this.state, { active: true, nps: 0, since: Date.now(), nonces: 0, blocksFound: 0, poolScore: -1, note: '' });
        this.batteryPaused = false;
        this.log(`Starting ${this.state.threads} mining thread(s), ${this.state.mode} mining`);
        this.spawn(this.state.threads);
        this.tickTimer = setInterval(() => this.tick(), 1500);
        this.holdScreen();
        this.changed();
    }

    stop(byHand = true) {
        if (byHand) this.batteryPaused = false;
        this.state.active = false;
        this.state.note = '';
        this.wallet.postMessage({ type: 'SET_MINING_ACTIVE', payload: { active: false } });
        for (const m of this.miners) { m.worker.postMessage({ type: 'STOP' }); m.worker.terminate(); }
        this.miners = [];
        if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
        if (this.tickTimer) { clearInterval(this.tickTimer); this.tickTimer = null; }
        this.template = null;
        this.state.nps = 0;
        if (this.wakeLock) { this.wakeLock.release().catch(() => {}); this.wakeLock = null; }
        this.changed();
    }

    newMiner(idx, useGpu, onReady) {
        const w = new Worker('./miner.js', { type: 'module' });
        const m = { worker: w, nps: 0, nonces: 0 };
        w.onerror = (err) => this.log(`Miner ${idx}: ${err.message || err}`);
        w.onmessage = (e) => {
            const { type, payload } = e.data;
            if (type === 'READY') onReady(m);
            else if (type === 'HASHRATE') {
                m.nps = payload.nps;
                m.nonces = payload.nonces;
                this.state.nonces += payload.nonces;
            } else if (type === 'FOUND') this.found(payload.nonce);
            else if (type === 'LOG') this.log(`[miner ${idx}] ${payload}`);
            else if (type === 'GPU_VERIFY_REQUEST') {
                // The wallet worker owns the WebWallet that checks GPU work;
                // remember which miner asked, for the answer.
                this.gpuWaiters.set(payload.id, w);
                this.wallet.postMessage({ type: 'GPU_VERIFY', payload });
            } else if (type === 'GPU_READY') {
                this.state.backend = payload.adapter || 'GPU';
                this.changed();
            } else if (type === 'ERROR') this.log(`[miner ${idx}] ${payload}`);
        };
        // Only thread 0 tries WebGPU: a GPU device cannot be shared between
        // workers, so it takes the GPU and the rest stay on the CPU.
        w.postMessage({ type: 'INIT', payload: { use_gpu: useGpu } });
        return m;
    }

    spawn(count) {
        this.miners = [];
        let ready = 0;
        for (let i = 0; i < count; i++) {
            this.miners.push(this.newMiner(i, i === 0, () => {
                if (++ready === count) {
                    this.requestTemplate();
                    this.pollTimer = setInterval(() => this.requestTemplate(), TEMPLATE_POLL_INTERVAL);
                }
            }));
        }
    }

    resize(count) {
        const current = this.miners.length;
        if (count < current) {
            for (const m of this.miners.splice(count)) { m.worker.postMessage({ type: 'STOP' }); m.worker.terminate(); }
        } else {
            for (let i = current; i < count; i++) {
                this.miners.push(this.newMiner(i, false, (m) => {
                    if (this.template) m.worker.postMessage({ type: 'START', payload: this.startPayload() });
                }));
            }
        }
    }

    requestTemplate() {
        if (!this.state.active) return;
        // Solo needs a node reached directly; a pool serves its own template.
        if (!this.pooled && !this.isWebRTC()) {
            this.state.note = 'Waiting for a direct connection to a node…';
            this.changed();
            return;
        }
        this.wallet.postMessage({ type: 'GET_TEMPLATE' });
    }

    startPayload() {
        const t = this.template;
        // In a pool a hit is a share and the work goes on after it.
        return { midstate: t.mining_midstate, target: t.target, pow: t.pow, throttle_ms: this.throttleMs, continue_on_found: this.pooled };
    }

    found(nonce) {
        if (!this.template) return;
        if (!this.pooled) {
            // A solo hit is a block: hold the threads until the next template.
            for (const m of this.miners) { m.worker.postMessage({ type: 'STOP' }); m.nps = 0; }
            this.state.blocksFound++;
        }
        this.wallet.postMessage({ type: 'SUBMIT_MINED_BLOCK', payload: { template: this.template, nonce } });
    }

    tick() {
        if (!this.state.active) return;
        this.state.nps = this.miners.reduce((s, m) => s + m.nps, 0);
        this.changed();
    }

    /** The wallet worker's messages about mining. Returns true if it was one. */
    handle(type, payload) {
        switch (type) {
            case 'TEMPLATE_READY': {
                if (!this.state.active) return true;
                if (!payload) { this.state.note = 'No work yet: the wallet may still be catching up.'; this.changed(); return true; }
                if (typeof payload.poolShares === 'number') this.state.poolScore = payload.poolShares;
                if (this.pooled) this.state.merge = payload.midwimble;
                if (payload.midwimble && this.state.midwimble && payload.midwimble.score != null) this.state.mwScore = payload.midwimble.score;
                this.state.tip = Number(payload.chainHeight) || this.state.tip;
                this.state.note = '';
                // A new midstate (a new block or new transactions), or in a pool
                // a new job: shares against an old job are refused as stale.
                const jobChanged = payload.job_id !== undefined && payload.job_id !== (this.template && this.template.job_id);
                if (!this.template || payload.mining_midstate !== this.template.mining_midstate || jobChanged) {
                    this.template = payload;
                    this.state.pow = payload.pow || null;
                    const start = this.startPayload();
                    for (const m of this.miners) m.worker.postMessage({ type: 'START', payload: start });
                }
                this.changed();
                return true;
            }
            case 'POOL_AUDIT_FAILED':
                this.stop();
                this.emit('error', `The pool failed its fairness check, so mining stopped: ${payload && payload.reason}`);
                return true;
            case 'TEMPLATE_ERROR':
                this.log(`Template error: ${payload}`);
                this.state.note = 'The pool or node did not hand out work. Trying again…';
                this.changed();
                return true;
            case 'BLOCK_SUBMITTED': {
                if (!payload || payload.rejectReason === 'duplicate') return true;
                if (payload.isShare && !payload.isBlock) {
                    if (typeof payload.poolShares === 'number') this.state.poolScore = payload.poolShares;
                    if (typeof payload.midwimbleScore === 'number') this.state.mwScore = payload.midwimbleScore;
                    this.changed();
                    return true;
                }
                if (payload.isBlock) this.state.blocksFound++;
                this.blocks.unshift(payload);
                if (payload.accepted) this.emit('block', payload);
                if (this.state.active && payload.accepted) {
                    this.template = null;
                    setTimeout(() => this.requestTemplate(), 2000);
                }
                this.changed();
                return true;
            }
            case 'GPU_VERIFY_RESULT': {
                const w = this.gpuWaiters.get(payload.id);
                this.gpuWaiters.delete(payload.id);
                if (w) w.postMessage({ type: 'GPU_VERIFY_RESULT', payload });
                return true;
            }
        }
        return false;
    }

    // ── the numbers ──────────────────────────────────────────────────────

    /** Expected base units a day at this speed, or null. */
    perDay() { return earningsPerDay(this.state.nps, this.template, this.pooled); }

    /** Blocks until the Walk starts, or 0 once it has. */
    walkIn(tip = this.state.tip) {
        return tip && tip < WALK_ACTIVATION_HEIGHT ? WALK_ACTIVATION_HEIGHT - tip : 0;
    }

    shareUrl() { return `${location.origin}${location.pathname}#/mine`; }

    // ── battery and screen ───────────────────────────────────────────────

    batteryTooLow() {
        return !!(this.state.pauseOnBattery && this.battery && !this.battery.charging && this.battery.level < 0.20);
    }

    checkBattery() {
        if (!this.battery) return;
        if (this.state.active && this.batteryTooLow()) {
            this.stop(false);
            this.batteryPaused = true;
            this.state.note = 'Paused: the battery is low. It starts again when you plug in.';
            this.changed();
        } else if (this.batteryPaused && !this.state.active
            && (!this.state.pauseOnBattery || this.battery.charging || this.battery.level >= 0.30)) {
            this.batteryPaused = false;
            this.state.note = '';
            try { this.start(); } catch (_) {}
        }
    }

    watchBattery() {
        if (typeof navigator === 'undefined' || !navigator.getBattery) return;
        navigator.getBattery().then((b) => {
            this.battery = b;
            b.addEventListener('levelchange', () => this.checkBattery());
            b.addEventListener('chargingchange', () => this.checkBattery());
            this.changed();
        }).catch(() => {});
    }

    get hasBattery() { return !!this.battery; }

    holdScreen() {
        if (typeof navigator === 'undefined' || !('wakeLock' in navigator) || this.wakeLock) return;
        navigator.wakeLock.request('screen').then((lock) => {
            this.wakeLock = lock;
            lock.addEventListener('release', () => { this.wakeLock = null; });
        }).catch(() => {});
    }
}
