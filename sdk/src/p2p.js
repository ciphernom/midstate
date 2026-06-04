import { createLibp2p } from 'libp2p';
import { multiaddr } from '@multiformats/multiaddr';

const LIGHT_PROTOCOL = '/midstate/light/2.0.0';
const LIGHT_PUSH_PROTOCOL = '/midstate/light-push/2.0.0';
const FULL_NODE_PROTOCOL = '/midstate/2.0.0';
const REQUEST_TIMEOUT_MS = 15_000;
const RECONNECT_DELAY_MS = 3_000;
const MAX_RECONNECT_ATTEMPTS = 5;

const isBrowser = typeof window !== 'undefined' && typeof window.document !== 'undefined';

// --- Binary Parsing Helpers ---
const SINGLE_BYTE_MAX = 250;
export function encodeVarint(n) {
    const v = typeof n === 'bigint' ? n : BigInt(n);
    if (v < 0n) throw new Error('varint must be non-negative');
    if (v <= 250n) return Uint8Array.of(Number(v));
    if (v < (1n << 16n)) { const b = new Uint8Array(3); b[0] = 251; new DataView(b.buffer).setUint16(1, Number(v), true); return b; }
    if (v < (1n << 32n)) { const b = new Uint8Array(5); b[0] = 252; new DataView(b.buffer).setUint32(1, Number(v), true); return b; }
    if (v < (1n << 64n)) { const b = new Uint8Array(9); b[0] = 253; new DataView(b.buffer).setBigUint64(1, v, true); return b; }
    throw new Error('varint too large');
}

function decodeVarint(buf, off = 0) {
    const first = buf[off];
    if (first === undefined) throw new Error('varint: out of bytes');
    if (first <= SINGLE_BYTE_MAX) return { value: BigInt(first), size: 1 };
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    if (first === 251) return { value: BigInt(dv.getUint16(off + 1, true)), size: 3 };
    if (first === 252) return { value: BigInt(dv.getUint32(off + 1, true)), size: 5 };
    if (first === 253) return { value: dv.getBigUint64(off + 1, true), size: 9 };
    throw new Error('varint invalid');
}

export function encodeGetAddr() { return encodeVarint(5); }

export function decodeAddr(payload) {
    let off = 0;
    const disc = decodeVarint(payload, off); off += disc.size;
    if (disc.value !== 6n) throw new Error(`expected Addr discriminant`);
    const count = decodeVarint(payload, off); off += count.size;
    const out = [];
    const dec = new TextDecoder();
    for (let i = 0; i < Number(count.value); i++) {
        const len = decodeVarint(payload, off); off += len.size;
        const n = Number(len.value);
        out.push(dec.decode(payload.subarray(off, off + n))); off += n;
    }
    return out;
}

function encodeBinaryFrame(payload) {
    const msg = new Uint8Array(4 + payload.length);
    new DataView(msg.buffer).setUint32(0, payload.length, true);
    msg.set(payload, 4);
    return msg;
}

function isDialableTcpAddr(addr) {
    if (typeof addr !== 'string') return false;
    if (addr.includes('webrtc')) return false;
    if (!addr.includes('/tcp/') || !addr.includes('/p2p/')) return false;
    if (addr.includes('/127.0.0.1/') || addr.includes('/::1/') || addr.includes('/0.0.0.0/')) return false;
    return true;
}

function encodeFrame(obj) {
    const jsonBytes = new TextEncoder().encode(JSON.stringify(obj));
    const msg = new Uint8Array(4 + jsonBytes.length);
    new DataView(msg.buffer).setUint32(0, jsonBytes.length, true);
    msg.set(jsonBytes, 4);
    return msg;
}

function waitForDrain(stream) {
    return new Promise((resolve, reject) => {
        const cleanup = () => { stream.removeEventListener('drain', onDrain); stream.removeEventListener('close', onClose); stream.removeEventListener('error', onClose); };
        const onDrain = () => { cleanup(); resolve(); };
        const onClose = (evt) => { cleanup(); reject(evt?.reason ?? new Error('stream closed while draining')); };
        stream.addEventListener('drain', onDrain); stream.addEventListener('close', onClose); stream.addEventListener('error', onClose);
    });
}

async function sendAll(stream, bytes) {
    let last = true;
    for (let i = 0; i < bytes.length; i += 16384) {
        const slice = bytes.subarray(i, i + 16384);
        const res = stream.send(slice);
        if (res === false) { last = false; await waitForDrain(stream); }
        else if (res && typeof res.then === 'function') { await res; }
        else { last = res; }
    }
    return last;
}

function readFrame(stream, timeoutMs) {
    return new Promise((resolve, reject) => {
        const chunks = [];
        let total = 0;
        const assemble = () => { const raw = new Uint8Array(total); let off = 0; for (const c of chunks) { raw.set(c, off); off += c.length; } return raw; };
        const tryComplete = () => {
            if (total < 4) return false;
            const raw = assemble();
            const len = new DataView(raw.buffer, raw.byteOffset).getUint32(0, true);
            if (raw.length < 4 + len) return false;
            const json = new TextDecoder().decode(raw.subarray(4, 4 + len));
            cleanup();
            try { resolve(JSON.parse(json)); } catch (e) { reject(e); }
            return true;
        };
        const onMessage = (evt) => { chunks.push(evt.data.subarray ? evt.data.subarray() : new Uint8Array(evt.data)); total += chunks[chunks.length-1].length; tryComplete(); };
        const onClose = (evt) => { if (!tryComplete()) { cleanup(); reject(evt?.reason ?? new Error('stream closed')); } };
        const onError = (evt) => { cleanup(); reject(evt?.reason ?? new Error('stream error')); };
        const onRemoteCloseWrite = () => {};
        const cleanup = () => { clearTimeout(timer); stream.removeEventListener('message', onMessage); stream.removeEventListener('close', onClose); stream.removeEventListener('error', onError); stream.removeEventListener('remoteCloseWrite', onRemoteCloseWrite); };
        const timer = setTimeout(() => { cleanup(); reject(new Error('timeout')); }, timeoutMs);
        stream.addEventListener('message', onMessage); stream.addEventListener('close', onClose); stream.addEventListener('error', onError); stream.addEventListener('remoteCloseWrite', onRemoteCloseWrite);
    });
}

function readBinaryFrame(stream, timeoutMs) {
    return new Promise((resolve, reject) => {
        const chunks = [];
        let total = 0;
        const assemble = () => { const raw = new Uint8Array(total); let off = 0; for (const c of chunks) { raw.set(c, off); off += c.length; } return raw; };
        const tryComplete = () => {
            if (total < 4) return false;
            const raw = assemble();
            const len = new DataView(raw.buffer, raw.byteOffset).getUint32(0, true);
            if (raw.length < 4 + len) return false;
            cleanup();
            resolve(raw.subarray(4, 4 + len));
            return true;
        };
        const onMessage = (evt) => { chunks.push(evt.data.subarray ? evt.data.subarray() : new Uint8Array(evt.data)); total += chunks[chunks.length-1].length; tryComplete(); };
        const onClose = (evt) => { if (!tryComplete()) { cleanup(); reject(evt?.reason ?? new Error('stream closed')); } };
        const onError = (evt) => { cleanup(); reject(evt?.reason ?? new Error('stream error')); };
        const cleanup = () => { clearTimeout(timer); stream.removeEventListener('message', onMessage); stream.removeEventListener('close', onClose); stream.removeEventListener('error', onError); };
        const timer = setTimeout(() => { cleanup(); reject(new Error('timeout')); }, timeoutMs);
        stream.addEventListener('message', onMessage); stream.addEventListener('close', onClose); stream.addEventListener('error', onError);
    });
}

export class P2PClient {
    constructor() {
        this.node = null;
        this.connectedPeer = null;
        this.connectedAddr = null;
        this.knownMultiaddrs = new Set();
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this._onStatusChange = null;
        this._onPushEvent = null;
        this._stopping = false;
    }

    onStatusChange(cb) { this._onStatusChange = cb; }
    onPushEvent(cb) { this._onPushEvent = cb; }
    _emitStatus(status) { if (this._onStatusChange) this._onStatusChange(status); }

    async start(addrs) {
        const transports = [];
        const connectionEncrypters = [];
        const streamMuxers = [];
        const services = {};

        if (isBrowser) {
            const { webRTCDirect } = await import('@libp2p/webrtc');
            transports.push(webRTCDirect());
        } else {
            const { tcp } = await import('@libp2p/tcp');
            const { noise } = await import('@chainsafe/libp2p-noise');
            const { yamux } = await import('@chainsafe/libp2p-yamux');
            const { identify } = await import('@libp2p/identify');
            const { ping } = await import('@libp2p/ping');

            transports.push(tcp());
            connectionEncrypters.push(noise());
            streamMuxers.push(yamux());
            services.identify = identify();
            services.ping = ping();
        }

        this.node = await createLibp2p({
            transports,
            connectionEncrypters: connectionEncrypters.length > 0 ? connectionEncrypters : undefined,
            streamMuxers: streamMuxers.length > 0 ? streamMuxers : undefined,
            services: Object.keys(services).length > 0 ? services : undefined
        });

        // ── NATIVE SDK BINARY HANDLER ──
        this.node.handle(FULL_NODE_PROTOCOL, async (data) => {
            const stream = data.stream || data;
            try {
                const payload = await readBinaryFrame(stream, 5000);
                if (payload.length > 0 && payload[0] === 19) {
                    let off = 1;
                    const senderLen = decodeVarint(payload, off); off += senderLen.size;
                    const sender = new TextDecoder().decode(payload.subarray(off, off + Number(senderLen.value))); off += Number(senderLen.value);
                    const timestampVi = decodeVarint(payload, off); off += timestampVi.size;
                    const timestamp = Number(timestampVi.value);

                    const nonceVi = decodeVarint(payload, off); off += nonceVi.size;
                    const nonce = Number(nonceVi.value);
                                    
                    let replyTo = null;
                    if (payload[off] === 1) { 
                        const repVi = decodeVarint(payload, off + 1); 
                        replyTo = Number(repVi.value); 
                        off += 1 + repVi.size; 
                    } else { off += 1; }

                    const wordsLen = decodeVarint(payload, off); off += wordsLen.size;
                    const words = Array.from(payload.subarray(off, off + Number(wordsLen.value))); off += Number(wordsLen.value);
                    
                    const attLen = decodeVarint(payload, off); off += attLen.size;
                    const attachments = [];
                    for (let i = 0; i < Number(attLen.value); i++) {
                        const attTag = decodeVarint(payload, off); off += attTag.size;
                        if (attTag.value === 0n) {
                            const addrBytes = payload.subarray(off, off + 32); off += 32;
                            attachments.push({ kind: "address", value: Array.from(addrBytes).map(b => b.toString(16).padStart(2,'0')).join('') });
                        } else {
                            off += 32; // Skip other 32-byte types to prevent crash
                        }
                    }

                    if (this._onPushEvent) {
                        this._onPushEvent({ ChatMessage: { sender, timestamp, nonce, reply_to: replyTo, words, attachments } });
                    }
                }
                
                // ACK
                const ackPayload = new Uint8Array([8, 0]);
                await sendAll(stream, encodeBinaryFrame(ackPayload));
                try { await stream.close(); } catch(e){}
            } catch (e) {
                try { stream.abort(e); } catch(err){}
            }
        });

        // Incoming Light Push Notifications (WebRTC)
        this.node.handle(LIGHT_PUSH_PROTOCOL, (stream) => {
            const chunks = [];
            let total = 0;
            const drainFrames = () => {
                let raw = new Uint8Array(total);
                let off = 0;
                for (const c of chunks) { raw.set(c, off); off += c.length; }

                let consumed = 0;
                while (raw.length - consumed >= 4) {
                    const len = new DataView(raw.buffer, consumed).getUint32(0, true);
                    if (raw.length - consumed < 4 + len) break;
                    const json = new TextDecoder().decode(raw.subarray(consumed + 4, consumed + 4 + len));
                    consumed += 4 + len;
                    try { if (this._onPushEvent) this._onPushEvent(JSON.parse(json)); }
                    catch (e) {}
                }
                if (consumed > 0) {
                    const remainder = raw.subarray(consumed);
                    chunks.length = 0;
                    total = remainder.length;
                    if (remainder.length) chunks.push(new Uint8Array(remainder));
                }
            };
            stream.addEventListener('message', (evt) => {
                chunks.push(evt.data.subarray ? evt.data.subarray() : new Uint8Array(evt.data));
                total += chunks[chunks.length-1].length;
                drainFrames();
            });
            stream.addEventListener('close', () => { try { stream.close(); } catch (_) {} });
        });

        await this.node.start();

        this.node.addEventListener('peer:connect', (evt) => {
            if (!this.isConnected || !this.connectedPeer) {
                this.connectedPeer = evt.detail;
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this._emitStatus('connected');
            }
        });

        this.node.addEventListener('peer:disconnect', (evt) => {
            if (this._stopping) return;
            if (this.connectedPeer?.toString() === evt.detail.toString()) {
                const spare = this._anyConnectedPeer(evt.detail);
                if (spare) {
                    this.connectedPeer = spare;
                    this.isConnected = true;
                    this._emitStatus('connected');
                } else {
                    this.isConnected = false;
                    this.connectedPeer = null;
                    this._emitStatus('disconnected');
                    this._scheduleReconnect();
                }
            }
        });

        addrs.forEach(a => this.knownMultiaddrs.add(a));
        for (const addr of this.knownMultiaddrs) {
            try {
                const ma = multiaddr(addr);
                const connection = await this.node.dial(ma, { signal: AbortSignal.timeout(5000) });
                this.connectedAddr = addr;
                this.connectedPeer = connection.remotePeer;
                this.isConnected = true;
                this._emitStatus('connected');
                this._discoverPeers().catch(()=>{});
                return;
            } catch (e) {}
        }
        throw new Error("Could not connect to any P2P peers");
    }

    _scheduleReconnect() {
        if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
            this._emitStatus('failed');
            return;
        }
        this.reconnectAttempts++;
        setTimeout(async () => {
            for (const addr of this.knownMultiaddrs) {
                try {
                    const connection = await this.node.dial(multiaddr(addr), { signal: AbortSignal.timeout(5000) });
                    this.connectedAddr = addr;
                    this.connectedPeer = connection.remotePeer;
                    this.isConnected = true;
                    return;
                } catch (_) {}
            }
            this._scheduleReconnect();
        }, RECONNECT_DELAY_MS * this.reconnectAttempts);
    }

    _anyConnectedPeer(exceptPeer) {
        const except = exceptPeer?.toString();
        try {
            for (const conn of this.node.getConnections()) {
                const pid = conn.remotePeer;
                if (pid && pid.toString() !== except) return pid;
            }
        } catch (_) {}
        return null;
    }

    async _rawBinaryRequest(peer, payload, timeoutMs = REQUEST_TIMEOUT_MS) {
        const conns = this.node.getConnections(peer);
        if (!conns || conns.length === 0) throw new Error('no connection for binary request');
        const stream = await conns[0].newStream(FULL_NODE_PROTOCOL, { signal: AbortSignal.timeout(timeoutMs) });
        const responsePromise = readBinaryFrame(stream, timeoutMs);
        responsePromise.catch(() => {});
        try {
            await sendAll(stream, encodeBinaryFrame(payload));
            try { await stream.close({ signal: AbortSignal.timeout(timeoutMs) }); } catch (_) {}
            return await responsePromise;
        } catch (e) {
            try { stream.abort(e instanceof Error ? e : new Error(String(e))); } catch (_) {}
            throw e;
        }
    }

    async _discoverPeers({ maxDial = 3 } = {}) {
        if (isBrowser || !this.connectedPeer) return;
        let addrs;
        try {
            const resp = await this._rawBinaryRequest(this.connectedPeer, encodeGetAddr());
            addrs = decodeAddr(resp);
        } catch (e) { return; }

        const myId = this.node.peerId.toString();
        const fresh = [];
        for (const addr of addrs) {
            if (!isDialableTcpAddr(addr) || addr.includes(myId) || this.knownMultiaddrs.has(addr)) continue;
            this.knownMultiaddrs.add(addr);
            fresh.push(addr);
        }

        let dialed = 0;
        for (const addr of fresh) {
            if (dialed >= maxDial) break;
            try {
                await this.node.dial(multiaddr(addr), { signal: AbortSignal.timeout(5000) });
                dialed++;
            } catch (e) {}
        }
    }

    async request(req, _retries = 2) {
        if (!this.isConnected || !this.connectedPeer) throw new Error('Not connected to any peer');
        let conns = this.node.getConnections(this.connectedPeer);
        if (!conns || conns.length === 0) {
            const spare = this._anyConnectedPeer(this.connectedPeer);
            if (spare) {
                this.connectedPeer = spare;
                conns = this.node.getConnections(spare);
            }
            if (!conns || conns.length === 0) throw new Error('No active connection to peer');
        }

        const stream = await conns[0].newStream(LIGHT_PROTOCOL, { signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
        const responsePromise = readFrame(stream, REQUEST_TIMEOUT_MS);
        responsePromise.catch(() => {});

        try {
            const frame = encodeFrame(req);
            await sendAll(stream, frame);
            try { await stream.close({ signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) }); } catch (e) {}
            return await responsePromise;
        } catch (e) {
            try { stream.abort(e instanceof Error ? e : new Error(String(e))); } catch (_) {}
            if (_retries > 0) return this.request(req, _retries - 1);
            throw e;
        }
    }

    async stop() {
        this._stopping = true;
        if (this.node) {
            await this.node.stop();
            this.isConnected = false;
            this.connectedPeer = null;
        }
    }
}
