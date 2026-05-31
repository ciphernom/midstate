/**
 * @fileoverview Universal P2P Transport Layer
 * Dynamically uses TCP (Node.js) or WebRTC (Browser) to connect to the Midstate network.
 *
 * js-libp2p v3 MessageStream API (shipped): streams are EventTargets.
 *   - write:     stream.send(bytes)  (returns false => wait for 'drain')
 *   - read:      'message' events; evt.data is a Uint8ArrayList
 *   - teardown:  await stream.close()  (graceful, flushes) / stream.abort(err) (reset)
 * NOTE: there is no closeWrite()/closeRead() — that was a v3 design draft that did
 * not ship. The Midstate light protocol reads a fixed-length frame server-side, so
 * no write-side half-close is needed; we send the request and read the response.
 * The transport handles WebRTC datachannel framing below the Stream interface,
 * so no manual protobuf decoding is needed — evt.data is application bytes.
 */

import { createLibp2p } from 'libp2p';
import { multiaddr } from '@multiformats/multiaddr';

const LIGHT_PROTOCOL = '/midstate/light/2.0.0';
const LIGHT_PUSH_PROTOCOL = '/midstate/light-push/2.0.0';
const FULL_NODE_PROTOCOL = '/midstate/2.0.0';
const REQUEST_TIMEOUT_MS = 15_000;
const RECONNECT_DELAY_MS = 3_000;
const MAX_RECONNECT_ATTEMPTS = 5;

// Set MIDSTATE_P2P_DEBUG=1 to trace the request/stream lifecycle. Helps diagnose
// "stream closed before full response": shows bytes sent, bytes received, and the
// close/error reason so we can tell "server sent no response" from a transport reset.
const DEBUG = (typeof process !== 'undefined' && process.env && process.env.MIDSTATE_P2P_DEBUG === '1');
const dlog = (...a) => { if (DEBUG) console.log('[P2P:debug]', ...a); };
const WRITE_CHUNK_SIZE = 16_384; // keep individual sends under the WebRTC SCTP limit

const isBrowser = typeof window !== 'undefined' && typeof window.document !== 'undefined';

// ── PEX peer discovery over the binary /midstate/2.0.0 protocol ──────────────
//
// The full node gossips peer addresses via PEX: a `GetAddr` request (Message
// enum discriminant 5) is answered on the same request_response stream with
// `Addr(Vec<String>)` (discriminant 6) — see node.rs `Message::GetAddr =>
// send_response(Addr(pex_addrs()))`. We use this so the SDK isn't pinned to a
// single hardcoded bootstrap node.
//
// The binary protocol is bincode with `DefaultOptions` = little-endian VARINT
// encoding (NOT fixed width). Per bincode 1.3 config/int.rs: a value <= 250 is
// a single byte; otherwise a marker byte (251=u16, 252=u32, 253=u64) precedes
// the LE integer. Enum discriminants are varint-encoded too. So `GetAddr` is
// the single byte 0x05; an `Addr` reply is [disc 6][vec-len][len+utf8 per str].
// Framing is the same [4-byte LE length][payload] used by the light protocol.
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

// Decode a varint at byte offset; returns { value: BigInt, size }.
function decodeVarint(buf, off = 0) {
    const first = buf[off];
    if (first === undefined) throw new Error('varint: out of bytes');
    if (first <= SINGLE_BYTE_MAX) return { value: BigInt(first), size: 1 };
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    if (first === 251) return { value: BigInt(dv.getUint16(off + 1, true)), size: 3 };
    if (first === 252) return { value: BigInt(dv.getUint32(off + 1, true)), size: 5 };
    if (first === 253) return { value: dv.getBigUint64(off + 1, true), size: 9 };
    if (first === 254) throw new Error('varint: u128 not supported');
    throw new Error('varint: byte 255 is an extension point (invalid)');
}

// Message::GetAddr → discriminant 5 (single varint byte).
export function encodeGetAddr() { return encodeVarint(5); }

// Decode Message::Addr(Vec<String>) from a binary payload (frame already stripped).
export function decodeAddr(payload) {
    let off = 0;
    const disc = decodeVarint(payload, off); off += disc.size;
    if (disc.value !== 6n) throw new Error(`expected Addr discriminant 6, got ${disc.value}`);
    const count = decodeVarint(payload, off); off += count.size;
    if (count.value > 1000n) throw new Error(`Addr count ${count.value} exceeds sane max`);
    const out = [];
    const dec = new TextDecoder();
    for (let i = 0; i < Number(count.value); i++) {
        const len = decodeVarint(payload, off); off += len.size;
        const n = Number(len.value);
        if (off + n > payload.length) throw new Error('Addr payload truncated: string length exceeds remaining bytes');
        out.push(dec.decode(payload.subarray(off, off + n))); off += n;
    }
    return out;
}

// Frame a raw binary payload as [4-byte LE length][payload].
function encodeBinaryFrame(payload) {
    const msg = new Uint8Array(4 + payload.length);
    new DataView(msg.buffer).setUint32(0, payload.length, true);
    msg.set(payload, 4);
    return msg;
}

// A TCP multiaddr we can actually dial: has /tcp/ and /p2p/, and is neither
// webrtc-direct nor localhost. The node's own dial_addr applies the same
// filter — its PEX list legitimately contains webrtc-direct addrs (served for
// the browser UI) that a TCP-only client must skip.
function isDialableTcpAddr(addr) {
    if (typeof addr !== 'string') return false;
    if (addr.includes('webrtc')) return false;
    if (!addr.includes('/tcp/') || !addr.includes('/p2p/')) return false;
    if (addr.includes('/127.0.0.1/') || addr.includes('/::1/') || addr.includes('/0.0.0.0/')) return false;
    return true;
}

// Frame a JSON value as [4-byte little-endian length][utf8 JSON].
function encodeFrame(obj) {
    const jsonBytes = new TextEncoder().encode(JSON.stringify(obj));
    const msg = new Uint8Array(4 + jsonBytes.length);
    new DataView(msg.buffer).setUint32(0, jsonBytes.length, true);
    msg.set(jsonBytes, 4);
    return msg;
}

// Wait for a 'drain' event (back pressure relief), rejecting if the stream dies first.
function waitForDrain(stream) {
    return new Promise((resolve, reject) => {
        const cleanup = () => {
            stream.removeEventListener('drain', onDrain);
            stream.removeEventListener('close', onClose);
            stream.removeEventListener('error', onClose);
        };
        const onDrain = () => { cleanup(); resolve(); };
        const onClose = (evt) => { cleanup(); reject(evt?.reason ?? new Error('stream closed while draining')); };
        stream.addEventListener('drain', onDrain);
        stream.addEventListener('close', onClose);
        stream.addEventListener('error', onClose);
    });
}

// Send all bytes, honoring back pressure. Tolerates send() returning either a
// boolean (v3 release API) or a promise (defensive against minor version drift).
async function sendAll(stream, bytes) {
    let last = true;
    for (let i = 0; i < bytes.length; i += WRITE_CHUNK_SIZE) {
        const slice = bytes.subarray(i, i + WRITE_CHUNK_SIZE);
        const res = stream.send(slice);
        if (res === false) {
            last = false;
            await waitForDrain(stream);
        } else if (res && typeof res.then === 'function') {
            await res;
        } else {
            last = res;
        }
    }
    return last;
}

// Read incoming bytes until one complete [len][JSON] frame is assembled, then resolve.
function readFrame(stream, timeoutMs) {
    return new Promise((resolve, reject) => {
        const chunks = [];
        let total = 0;

        const assemble = () => {
            const raw = new Uint8Array(total);
            let off = 0;
            for (const c of chunks) { raw.set(c, off); off += c.length; }
            return raw;
        };

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

        const onMessage = (evt) => {
            const part = evt.data.subarray ? evt.data.subarray() : new Uint8Array(evt.data);
            chunks.push(part);
            total += part.length;
            dlog(`recv message: +${part.length}B (total ${total}B)`);
            tryComplete();
        };
        const onClose = (evt) => {
            dlog(`recv 'close' event; reason=${evt?.reason?.message ?? 'none'}; bytes so far=${total}`);
            if (!tryComplete()) { cleanup(); reject(evt?.reason ?? new Error('stream closed before full response')); }
        };
        const onError = (evt) => {
            dlog(`recv 'error' event; reason=${evt?.reason?.message ?? 'unknown'}; bytes so far=${total}`);
            cleanup(); reject(evt?.reason ?? new Error('stream error'));
        };
        const onRemoteCloseWrite = () => dlog(`recv 'remoteCloseWrite' (server finished sending); bytes so far=${total}`);
        const cleanup = () => {
            clearTimeout(timer);
            stream.removeEventListener('message', onMessage);
            stream.removeEventListener('close', onClose);
            stream.removeEventListener('error', onError);
            stream.removeEventListener('remoteCloseWrite', onRemoteCloseWrite);
        };

        const timer = setTimeout(() => { dlog(`read TIMEOUT after ${timeoutMs}ms; bytes=${total}`); cleanup(); reject(new Error('Stream read timeout')); }, timeoutMs);
        stream.addEventListener('message', onMessage);
        stream.addEventListener('close', onClose);
        stream.addEventListener('error', onError);
        stream.addEventListener('remoteCloseWrite', onRemoteCloseWrite);
    });
}

// Like readFrame, but resolves with the raw payload bytes (no JSON parse) — used
// for the bincode binary protocol (PEX). Same [4-byte LE length][payload] framing.
function readBinaryFrame(stream, timeoutMs) {
    return new Promise((resolve, reject) => {
        const chunks = [];
        let total = 0;
        const assemble = () => {
            const raw = new Uint8Array(total);
            let off = 0;
            for (const c of chunks) { raw.set(c, off); off += c.length; }
            return raw;
        };
        const tryComplete = () => {
            if (total < 4) return false;
            const raw = assemble();
            const len = new DataView(raw.buffer, raw.byteOffset).getUint32(0, true);
            if (raw.length < 4 + len) return false;
            cleanup();
            resolve(raw.subarray(4, 4 + len));
            return true;
        };
        const onMessage = (evt) => {
            const part = evt.data.subarray ? evt.data.subarray() : new Uint8Array(evt.data);
            chunks.push(part); total += part.length;
            tryComplete();
        };
        const onClose = (evt) => { if (!tryComplete()) { cleanup(); reject(evt?.reason ?? new Error('stream closed before full response')); } };
        const onError = (evt) => { cleanup(); reject(evt?.reason ?? new Error('stream error')); };
        const cleanup = () => {
            clearTimeout(timer);
            stream.removeEventListener('message', onMessage);
            stream.removeEventListener('close', onClose);
            stream.removeEventListener('error', onError);
        };
        const timer = setTimeout(() => { cleanup(); reject(new Error('Stream read timeout')); }, timeoutMs);
        stream.addEventListener('message', onMessage);
        stream.addEventListener('close', onClose);
        stream.addEventListener('error', onError);
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

        // v3 handler signature: (stream, connection) — NOT ({ stream }).
        this.node.handle(FULL_NODE_PROTOCOL, (stream) => {
            stream.close();
        });

        // Incoming Light Push Notifications. Parse complete [len][JSON] frames out
        // of the message stream as they arrive (handles one-shot or streamed pushes).
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
                    catch (e) { console.warn('[P2P] Bad push frame', e); }
                }
                if (consumed > 0) {
                    const remainder = raw.subarray(consumed);
                    chunks.length = 0;
                    total = remainder.length;
                    if (remainder.length) chunks.push(new Uint8Array(remainder));
                }
            };

            stream.addEventListener('message', (evt) => {
                const part = evt.data.subarray ? evt.data.subarray() : new Uint8Array(evt.data);
                chunks.push(part);
                total += part.length;
                drainFrames();
            });
            stream.addEventListener('close', () => { try { stream.close(); } catch (_) {} });
            stream.addEventListener('error', () => {});
        });

        await this.node.start();

        this.node.addEventListener('peer:connect', (evt) => {
            // Only adopt a new primary on a real disconnected→connected transition.
            // PEX dials extra peers; without this guard every extra (and every
            // inbound) connection would reassign connectedPeer and thrash the
            // primary that request() depends on.
            if (!this.isConnected || !this.connectedPeer) {
                this.connectedPeer = evt.detail;
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this._emitStatus('connected');
            }
        });

        this.node.addEventListener('peer:disconnect', (evt) => {
            // During an intentional stop(), don't promote spares or schedule
            // reconnects — that would flap the status mid-teardown. Let stop()
            // drive the final state.
            if (this._stopping) return;
            if (this.connectedPeer?.toString() === evt.detail.toString()) {
                // Primary dropped. Promote any other live connection as the new
                // primary before falling back to a timed reconnect — PEX may have
                // left us several warm peers to choose from.
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

                console.log(`[P2P] libp2p started. Engine: ${isBrowser ? 'WebRTC' : 'TCP'}. Local PeerId:`, this.node.peerId.toString());
                console.log('[P2P] Successfully secured connection to Node:', this.connectedPeer.toString());

                this._emitStatus('connected');

                // Best-effort: ask this peer for more peers and warm a few spare
                // connections so we're not pinned to one bootstrap node. Never let
                // a discovery failure affect the primary connection we just made.
                this._discoverPeers().catch((e) => dlog(`peer discovery failed (non-fatal): ${e?.message}`));

                return;
            } catch (e) {
                console.warn(`[P2P] Failed to connect to ${addr}:`, e.message);
            }
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

    // Return a connected PeerId other than `exceptPeer` (used to promote a spare
    // when the primary drops). Returns null if no other peer is connected.
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

    // Send one binary /midstate/2.0.0 request to `peer` and read the framed
    // bincode response payload. Mirrors the proven light-protocol request flow:
    // open stream → attach reader → send framed payload → half-close (FIN) → read.
    // TCP/Node only — the browser build doesn't speak this protocol to nodes.
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

    // PEX: ask the current peer for more peers, then warm a few spare TCP
    // connections so the SDK isn't pinned to one bootstrap node. Entirely
    // best-effort — every failure is swallowed and never disturbs the primary.
    async _discoverPeers({ maxDial = 3 } = {}) {
        if (isBrowser) return;                 // TCP-only: node won't gossip dialable WebRTC addrs
        if (!this.connectedPeer) return;

        let addrs;
        try {
            const resp = await this._rawBinaryRequest(this.connectedPeer, encodeGetAddr());
            addrs = decodeAddr(resp);
        } catch (e) {
            dlog(`GetAddr failed (non-fatal): ${e?.message}`);
            return;
        }
        dlog(`PEX: peer returned ${addrs.length} addr(s)`);

        const myId = this.node.peerId.toString();
        const fresh = [];
        for (const addr of addrs) {
            if (!isDialableTcpAddr(addr)) continue;
            if (addr.includes(myId)) continue;             // don't dial ourselves
            if (this.knownMultiaddrs.has(addr)) continue;  // already known/seeded
            this.knownMultiaddrs.add(addr);
            fresh.push(addr);
        }
        dlog(`PEX: ${fresh.length} new dialable TCP addr(s) added to pool (pool size ${this.knownMultiaddrs.size})`);

        // Warm a handful of spares. These extras must NOT become the primary
        // (the peer:connect guard handles that); we only want them connected so
        // a primary drop can promote one instantly.
        let dialed = 0;
        for (const addr of fresh) {
            if (dialed >= maxDial) break;
            try {
                await this.node.dial(multiaddr(addr), { signal: AbortSignal.timeout(5000) });
                dialed++;
                dlog(`PEX: warmed spare connection to ${addr}`);
            } catch (e) {
                dlog(`PEX: dial ${addr} failed (non-fatal): ${e?.message}`);
            }
        }
    }

    async request(req, _retries = 2) {
        if (!this.isConnected || !this.connectedPeer) throw new Error('Not connected to any peer');
        let conns = this.node.getConnections(this.connectedPeer);
        if (!conns || conns.length === 0) {
            // Primary has no live connection right now — promote any other
            // connected peer (PEX may have warmed spares) instead of failing.
            const spare = this._anyConnectedPeer(this.connectedPeer);
            if (spare) {
                this.connectedPeer = spare;
                conns = this.node.getConnections(spare);
            }
            if (!conns || conns.length === 0) throw new Error('No active connection to peer');
        }

        const stream = await conns[0].newStream(LIGHT_PROTOCOL, {
            signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)
        });
        dlog(`stream opened for ${req.method}; protocol=${stream.protocol}; typeof close=${typeof stream.close}, typeof abort=${typeof stream.abort}`);

        // Attach the reader BEFORE writing so we never miss an early response chunk.
        // (In v3, inbound bytes are buffered until a 'message' listener exists, but
        // attaching first avoids any microtask-ordering surprises.)
        const responsePromise = readFrame(stream, REQUEST_TIMEOUT_MS);
        // Mark the promise as handled so an early bail-out (e.g. a write error before
        // we await) doesn't surface as an unhandledRejection.
        responsePromise.catch(() => {});

        try {
            const frame = encodeFrame(req);
            const sent = await sendAll(stream, frame);
            dlog(`sent request ${req.method}: ${frame.length}B (send() => ${sent})`);

            // Half-close our write side: flush the request and send a FIN. In v3,
            // close() is a *graceful write-close* — it resolves once our unsent data
            // is written and signals end-of-write to the peer, while our read side
            // stays open so we can still read the response. This mirrors the working
            // browser client's sendCloseWrite(), and some server stream readers only
            // complete (or flush) once they observe this FIN. Bounded by a signal so a
            // dead connection can't hang us here.
            try {
                await stream.close({ signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
                dlog('write side half-closed (FIN sent); awaiting response');
            } catch (e) {
                dlog(`close() (half-close) threw: ${e?.message}; continuing to read anyway`);
            }

            const result = await responsePromise;
            dlog(`response received for ${req.method}`);
            return result;
        } catch (e) {
            dlog(`request ${req.method} failed: ${e?.message}; retries left=${_retries}`);
            // Discard the stream on any failure; abort() resets it and frees the
            // server-side per-peer stream slot immediately.
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
