// pool_client.js — Browser-side libp2p client for the Midstate mining pool.
//
// Dials the pool directly over WebRTC-direct, the same transport light_client.js
// uses to reach full nodes: no HTTPS, no domain, no certificate authority. The
// pool is authenticated by the /certhash/ component of its multiaddr.
//
// Why this exists alongside the HTTP path:
//
//   • No mixed content. A wallet served over HTTPS cannot fetch an http:// pool;
//     the browser blocks it before the request is sent. WebRTC-direct is not
//     subject to that rule, so the pool needs no CA-signed certificate.
//   • No CORS. There is no preflight and no origin check to misconfigure.
//   • Push, not poll. The HTTP client asks for a template once a second and gets
//     the whole batch back even when nothing changed. Here the pool pushes
//     `notify` the moment a job rotates, over /midstate/pool-push/1.0.0.
//
// Wire format matches light_protocol.rs exactly: 4-byte little-endian length
// prefix followed by JSON, one request/response per stream.
//
//   Request:  { method: "get_template", params: { address } }
//   Response: { ok: true, data: {...} } | { ok: false, error: "..." }

import { createLibp2p } from 'libp2p';
import { webRTCDirect } from '@libp2p/webrtc';
import { multiaddr } from '@multiformats/multiaddr';

const POOL_PROTOCOL = '/midstate/pool/1.0.0';
const POOL_PUSH_PROTOCOL = '/midstate/pool-push/1.0.0';
const REQUEST_TIMEOUT_MS = 15_000;

/** Frames a JSON value as 4-byte LE length + UTF-8 body. */
function frame(value) {
    const jsonBytes = new TextEncoder().encode(JSON.stringify(value));
    const msg = new Uint8Array(4 + jsonBytes.length);
    new DataView(msg.buffer).setUint32(0, jsonBytes.length, true);
    msg.set(jsonBytes, 4);
    return msg;
}

export class PoolClient {
    constructor() {
        this.node = null;
        this.peer = null;
        this.isConnected = false;
        this._onNotify = null;
        this._onStatus = null;
    }

    /** Registers a callback fired when the pool pushes a new job. */
    onNotify(cb) { this._onNotify = cb; }
    onStatus(cb) { this._onStatus = cb; }

    _emit(status, detail) {
        if (this._onStatus) { try { this._onStatus(status, detail); } catch (_) {} }
    }

    /**
     * Starts libp2p and dials the pool.
     *
     * @param {string} addr - Pool multiaddr, including /certhash/ and /p2p/.
     */
    async start(addr) {
        this.node = await createLibp2p({
            transports: [webRTCDirect()],
            // Same relaxation as the light client: without this, dialing a pool
            // on a LAN address or 127.0.0.1 is refused, which makes local
            // testing impossible.
            connectionGater: { denyDialMultiaddr: async () => false },
        });

        // Server-initiated job notifications. One framed JSON per stream, then
        // the pool closes it — the same one-shot shape as /midstate/light-push.
        this.node.handle(POOL_PUSH_PROTOCOL, async (stream) => {
            if (!stream) return;
            try {
                const chunks = [];
                let total = 0;
                const source = stream.source || stream;
                for await (const chunk of source) {
                    const bytes = chunk.subarray ? chunk.subarray() : chunk;
                    chunks.push(bytes);
                    total += bytes.length;
                    if (total < 4) continue;

                    const buf = new Uint8Array(total);
                    let off = 0;
                    for (const c of chunks) { buf.set(c, off); off += c.length; }

                    const expected = new DataView(buf.buffer, buf.byteOffset).getUint32(0, true);
                    // Bound the read against the declared length before slicing,
                    // so a truncated or hostile stream can't drive allocation.
                    if (total >= 4 + expected) {
                        const notif = JSON.parse(new TextDecoder().decode(buf.slice(4, 4 + expected)));
                        if (this._onNotify) this._onNotify(notif);
                        break;
                    }
                }
            } catch (e) {
                console.warn('[pool] push stream error', e);
            } finally {
                try { if (stream.close) stream.close(); } catch (_) {}
            }
        });

        await this.node.start();

        this.node.addEventListener('peer:disconnect', (evt) => {
            if (this.peer?.toString() === evt.detail.toString()) {
                this.isConnected = false;
                this._emit('disconnected');
            }
        });

        const ma = multiaddr(addr);
        const conn = await this.node.dial(ma);
        this.peer = conn.remotePeer;
        this.isConnected = true;
        this._emit('connected', this.peer.toString());
        return this.peer.toString();
    }

    async stop() {
        this.isConnected = false;
        this.peer = null;
        try { await this.node?.stop(); } catch (_) {}
        this.node = null;
    }

    /**
     * Sends one request and awaits its response.
     *
     * Opens a fresh stream per request — matching the pool's server side, which
     * reads exactly one framed request per inbound stream and then closes.
     */
    async requestRaw(method, params = {}, _retry = 1) {
        if (!this.isConnected || !this.peer) throw new Error('Not connected to pool');
        const conns = this.node.getConnections(this.peer);
        if (!conns || conns.length === 0) throw new Error('No active connection to pool');

        const stream = await conns[0].newStream([POOL_PROTOCOL]);

        try {
            const response = new Promise((resolve, reject) => {
                let buf = new Uint8Array(0);
                let expected = null;

                const onMessage = (evt) => {
                    const d = evt.data;
                    const bytes = d instanceof Uint8Array ? d : d.subarray();
                    const next = new Uint8Array(buf.length + bytes.length);
                    next.set(buf, 0); next.set(bytes, buf.length);
                    buf = next;

                    if (expected === null && buf.length >= 4) {
                        expected = new DataView(buf.buffer, buf.byteOffset).getUint32(0, true);
                    }
                    if (expected !== null && buf.length >= 4 + expected) {
                        cleanup();
                        resolve(buf.slice(4, 4 + expected));
                    }
                };
                const onClose = (evt) => {
                    cleanup();
                    reject(evt?.error ?? new Error('Stream closed before completing response'));
                };
                const timer = setTimeout(() => {
                    cleanup();
                    reject(new Error('Pool request timeout'));
                }, REQUEST_TIMEOUT_MS);
                const cleanup = () => {
                    clearTimeout(timer);
                    stream.removeEventListener('message', onMessage);
                    stream.removeEventListener('close', onClose);
                };

                stream.addEventListener('message', onMessage);
                stream.addEventListener('close', onClose);
            });
            // Suppress unhandled-rejection noise if we bail during send(); the
            // promise still rejects normally where it is awaited.
            response.catch(() => {});

            const msg = frame({ method, params });
            // send() returns false when the transport buffer is full; wait for
            // 'drain' rather than dropping the write on the floor.
            if (!stream.send(msg)) {
                await new Promise((resolve, reject) => {
                    const onDrain = () => { off(); resolve(); };
                    const onClose = (evt) => { off(); reject(evt?.error ?? new Error('Stream closed during send')); };
                    const off = () => {
                        stream.removeEventListener('drain', onDrain);
                        stream.removeEventListener('close', onClose);
                    };
                    stream.addEventListener('drain', onDrain);
                    stream.addEventListener('close', onClose);
                });
            }

            const bytes = await response;
            // Returns the RAW envelope: { ok, data } | { ok:false, error }.
            // Callers must be able to tell "the pool said no" from "the request
            // never completed" — collapsing both into a thrown Error is what let
            // a dropped stream masquerade as the pool omitting a miner from its
            // precommitment, which halts mining.
            return JSON.parse(new TextDecoder().decode(bytes));
        } catch (e) {
            // libp2p streams get reset under normal operation (peer churn, the
            // remote closing early). One retry on a fresh stream turns a routine
            // blip into a non-event rather than a visible failure.
            if (_retry > 0 && /reset|closed|abort/i.test(e.message || '')) {
                return this.requestRaw(method, params, _retry - 1);
            }
            throw e;
        } finally {
            try { if (stream.close) await stream.close(); } catch (_) {}
        }
    }

    /** Throws on a pool-reported error. Use for calls where absence is fatal. */
    async request(method, params = {}) {
        const parsed = await this.requestRaw(method, params);
        if (!parsed.ok) throw new Error(parsed.error || 'Pool returned an error');
        return parsed.data;
    }

    getTemplate(address) { return this.request('get_template', { address }); }
    /** Raw envelope: a miner absent from the tree is an ANSWER, not an exception. */
    getProof(address)    { return this.requestRaw('get_proof', { address }); }
    submitShare(address, jobId, nonce, worker) {
        // Nonce as a decimal string: it is a u64 and JSON numbers lose precision
        // above 2^53.
        return this.request('submit_share', {
            address, job_id: jobId, nonce: String(nonce), worker: worker || 'browser',
        });
    }
}
