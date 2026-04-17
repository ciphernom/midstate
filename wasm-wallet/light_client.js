// light_client.js — Browser-side libp2p light client for Midstate wallet
//
// Connects to full nodes over WebRTC direct (no HTTPS, no domain, no cert authority).
// Speaks the /midstate/light/1.0.0 JSON protocol over libp2p streams.
//
// Usage in worker.js:
//   import { LightClient } from './light_client.js';
//   const client = new LightClient();
//   await client.start(bootstrapMultiaddr);
//   const state = await client.request({ method: 'get_state' });
//
// Dependencies (install via npm):
//   npm install libp2p @libp2p/webrtc @chainsafe/libp2p-noise @chainsafe/libp2p-yamux
//   npm install it-length-prefixed uint8arrays it-pipe

import { createLibp2p } from 'libp2p';
import { webRTCDirect } from '@libp2p/webrtc';
import { pipe } from 'it-pipe';
import { multiaddr } from '@multiformats/multiaddr';

const LIGHT_PROTOCOL = '/midstate/light/1.0.0';
const REQUEST_TIMEOUT_MS = 60_000;
const RECONNECT_DELAY_MS = 3_000;
const MAX_RECONNECT_ATTEMPTS = 5;

// ── WebRTC stream protobuf decoder ──────────────────────────────────────────
// The js-libp2p WebRTC transport wraps application data in protobuf frames:
//   [varint msg_len][protobuf Message]...
//   Message { Flag flag = 1; bytes message = 2; }
//   Flag: FIN=0, STOP_SENDING=1, RESET=2
// incomingData yields the raw datachannel bytes with this framing intact.
// This function strips the framing and returns only the application data.

function readVarint(buf, offset) {
    let val = 0, shift = 0;
    while (offset < buf.length) {
        const b = buf[offset++];
        val |= (b & 0x7f) << shift;
        if ((b & 0x80) === 0) return [val, offset];
        shift += 7;
        if (shift > 35) throw new Error('varint too long');
    }
    throw new Error('truncated varint');
}

function decodeWebRTCStreamData(raw) {
    let offset = 0;
    const chunks = [];
    let totalLen = 0;

    while (offset < raw.length) {
        // Read message length (varint)
        let msgLen;
        [msgLen, offset] = readVarint(raw, offset);
        const msgEnd = offset + msgLen;
        if (msgEnd > raw.length) break; // truncated

        // Parse protobuf fields within this message
        let pos = offset;
        while (pos < msgEnd) {
            const tag = raw[pos++];
            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (wireType === 0) { // varint (flag field)
                let _val;
                [_val, pos] = readVarint(raw, pos);
                // flag=0 (FIN), flag=2 (RESET) — we just keep extracting data
            } else if (wireType === 2) { // length-delimited (message/data field)
                let dLen;
                [dLen, pos] = readVarint(raw, pos);
                if (fieldNum === 2 && pos + dLen <= msgEnd) {
                    chunks.push(raw.slice(pos, pos + dLen));
                    totalLen += dLen;
                }
                pos += dLen;
            } else {
                break; // unknown wire type, skip rest
            }
        }

        offset = msgEnd;
    }

    // Concatenate all data chunks
    const result = new Uint8Array(totalLen);
    let off = 0;
    for (const chunk of chunks) {
        result.set(chunk, off);
        off += chunk.length;
    }
    return result;
}

// Check if a raw protobuf-framed chunk contains a specific flag value.
// Flag is: field 1 (tag byte 0x08), varint value.
// FIN=0, STOP_SENDING=1, RESET=2
// The flag message can appear standalone (3 bytes: 02 08 XX) or
// appended after data messages in the same chunk.
function chunkContainsFlag(raw, flagValue) {
    let offset = 0;
    try {
        while (offset < raw.length) {
            let msgLen;
            [msgLen, offset] = readVarint(raw, offset);
            const msgEnd = offset + msgLen;
            if (msgEnd > raw.length) return false;
            let pos = offset;
            while (pos < msgEnd) {
                const tag = raw[pos++];
                const fieldNum = tag >> 3;
                const wireType = tag & 0x7;
                if (wireType === 0) {
                    let val;
                    [val, pos] = readVarint(raw, pos);
                    if (fieldNum === 1 && val === flagValue) return true;
                } else if (wireType === 2) {
                    let dLen;
                    [dLen, pos] = readVarint(raw, pos);
                    pos += dLen;
                } else {
                    break;
                }
            }
            offset = msgEnd;
        }
    } catch (_) {}
    return false;
}

function chunkContainsFin(raw) { return chunkContainsFlag(raw, 0); }
function chunkContainsReset(raw) { return chunkContainsFlag(raw, 2); }

export class LightClient {
    constructor() {
        this.node = null;
        this.connectedPeer = null;        // PeerId of currently connected full node
        this.knownMultiaddrs = new Set();  // Set of multiaddr strings we can try
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this._onStatusChange = null;       // callback: (status) => void
    }

onPushEvent(cb) {
        this._onPushEvent = cb;
    }

    /// Start the libp2p node and connect to a bootstrap peer list.
    ///
    /// addrs: Array of multiaddr strings
    async start(addrs) {
        this.node = await createLibp2p({
            transports: [webRTCDirect()],
            // WebRTC handles DTLS/SCTP automatically.
        });

                // --- LISTEN FOR SERVER PUSHES ---
        this.node.handle('/midstate/light-push/1.0.0', async ({ stream }) => {
            try {
                const chunks = [];
                let total = 0;
                for await (const chunk of stream.source) {
                    const bytes = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk.subarray());
                    chunks.push(bytes);
                    total += bytes.length;
                }
                const rawBuf = new Uint8Array(total);
                let offset = 0;
                for (const c of chunks) { rawBuf.set(c, offset); offset += c.length; }
                
                // Strip the WebRTC Protobuf framing before parsing
                const appData = decodeWebRTCStreamData(rawBuf);
                
                if (appData.length >= 4) {
                    const len = new DataView(appData.buffer, appData.byteOffset).getUint32(0, true);
                    const jsonStr = new TextDecoder().decode(appData.slice(4, 4 + len));
                    const notif = JSON.parse(jsonStr);
                    if (this._onPushEvent) this._onPushEvent(notif);
                }
            } catch (e) { console.warn("[light] Push handle error", e); }
        });

        await this.node.start();
        console.log('[light] libp2p started. PeerId:', this.node.peerId.toString());

        // Track connection lifecycle
        this.node.addEventListener('peer:connect', (evt) => {
            console.log('[light] Peer connected:', evt.detail.toString());
            this.connectedPeer = evt.detail;
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this._emitStatus('connected');
        });

        this.node.addEventListener('peer:disconnect', (evt) => {
            console.log('[light] Peer disconnected:', evt.detail.toString());
            if (this.connectedPeer?.toString() === evt.detail.toString()) {
                this.isConnected = false;
                this.connectedPeer = null;
                this._emitStatus('disconnected');
                this._scheduleReconnect();
            }
        });

        // Try connecting to the provided addresses in order
        if (addrs && addrs.length > 0) {
            for (const addr of addrs) {
                this.knownMultiaddrs.add(addr);
            }
            
            let connected = false;
            for (const addr of addrs) {
                try {
                    await this.connectTo(addr);
                    connected = true;
                    break; // Stop at the first successful connection
                } catch (e) {
                    console.warn('[light] Skipping unreachable peer:', addr);
                }
            }
            
            if (!connected) {
                throw new Error("Could not connect to any WebRTC peers");
            }
        }
    }

    /// Connect to a specific multiaddr with a 5-second timeout.
    async connectTo(addrStr) {
        const ma = multiaddr(addrStr);
        // AbortSignal.timeout ensures we don't hang forever on firewalled peers
        const conn = await this.node.dial(ma, { signal: AbortSignal.timeout(5000) });
        console.log('[light] Dialed:', addrStr);
        return conn;
    }

    /// Send a JSON request over the light protocol and return the parsed response.
    ///
    /// req: { method: 'get_state' } or { method: 'get_block', params: { height: 42 } }
    /// Returns: parsed LightResponse { ok, data?, error? }
async request(req, _retries = 2) {
        if (!this.isConnected || !this.connectedPeer) {
            throw new Error('Not connected to any peer');
        }

        const dialResult = await this.node.dialProtocol(this.connectedPeer, LIGHT_PROTOCOL);
        const stream = dialResult.stream || dialResult;

        try {
            const jsonBytes = new TextEncoder().encode(JSON.stringify(req));
            const lenBuf = new Uint8Array(4);
            new DataView(lenBuf.buffer).setUint32(0, jsonBytes.length, true);
            const msg = new Uint8Array(4 + jsonBytes.length);
            msg.set(lenBuf, 0);
            msg.set(jsonBytes, 4);

            const chunks = [];
            let totalLen = 0;
            let gotReset = false;

            // 1. WRITE and READ in one clean pipeline.
            // pipe(source, duplex_stream, sink_function)
            const streamPipeline = pipe(
                [msg],
                stream,
                async (source) => {
                    for await (const chunk of source) {
                        const bytes = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk.subarray ? chunk.subarray() : chunk);
                        chunks.push(bytes);
                        totalLen += bytes.length;

                        if (chunkContainsReset(bytes)) {
                            gotReset = true;
                            break;
                        }
                        if (chunkContainsFin(bytes)) {
                            break;
                        }
                    }
                }
            );

            const timeoutPromise = new Promise((_, reject) =>
                setTimeout(() => reject(new Error('Stream read timeout')), REQUEST_TIMEOUT_MS)
            );

            // 2. Await the pipeline with a timeout safety net
            await Promise.race([streamPipeline, timeoutPromise]);

            const rawBuf = new Uint8Array(totalLen);
            let offset = 0;
            for (const c of chunks) {
                rawBuf.set(c, offset);
                offset += c.length;
            };       

            // If the node reset the stream (happens occasionally on first connect), retry
            if (gotReset && _retries > 0) {
                try { stream.abort(new Error('reset')); } catch (_) {}
                return this.request(req, _retries - 1);
            }

            const appData = decodeWebRTCStreamData(rawBuf);

            if (appData.length < 4) throw new Error('Response too short');
            const respLen = new DataView(appData.buffer, appData.byteOffset).getUint32(0, true);
            const respJson = new TextDecoder().decode(appData.slice(4, 4 + respLen));
            return JSON.parse(respJson);
            
        } finally {
            try { stream.abort(new Error('done')); } catch (_) {}
            try { stream.close(); } catch (_) {}
        }
    }

    // ── Convenience Methods (match the RPC endpoints the wallet uses) ────────

    async getState() {
        const resp = await this.request({ method: 'get_state' });
        if (!resp.ok) throw new Error(resp.error);
        return resp.data;
    }

    async getBlock(height) {
        const resp = await this.request({ method: 'get_block', params: { height } });
        if (!resp.ok) throw new Error(resp.error);
        return resp.data;
    }

    async getFilters(startHeight, endHeight) {
        const resp = await this.request({ method: 'get_filters', params: { start_height: startHeight, end_height: endHeight } });
        if (!resp.ok) throw new Error(resp.error);
        return resp.data;
    }

    async getMempool() {
        const resp = await this.request({ method: 'get_mempool' });
        if (!resp.ok) throw new Error(resp.error);
        return resp.data;
    }

    async submitBatch(batch) {
        const resp = await this.request({ method: 'submit_batch', params: { batch } });
        return resp; // Caller checks resp.ok
    }
    async getBlockTemplate(coinbase) {
        const resp = await this.request({ method: 'block_template', params: { coinbase } });
        return {
            ok: resp.ok,
            // If the node provided expected_total during an error (409 Conflict), map it correctly
            status: resp.ok ? 200 : (resp.data && resp.data.expected_total ? 409 : 500),
            json: () => Promise.resolve(resp.data),
            // Always stringify the data payload if it exists, even on failure
            text: () => Promise.resolve(
                resp.data ? JSON.stringify(resp.data) : (resp.error || 'error')
            ),
        };
    }
    async commit(commitmentHex, spamNonce) {
        const resp = await this.request({ method: 'commit', params: { commitment: commitmentHex, spam_nonce: spamNonce } });
        return resp;
    }

    async send(revealPayload) {
        const resp = await this.request({ method: 'send', params: { reveal: revealPayload } });
        return resp;
    }

    async checkCoin(coinHex) {
        const resp = await this.request({ method: 'check', params: { coin: coinHex } });
        if (!resp.ok) throw new Error(resp.error);
        return resp.data;
    }

    async checkCommitment(commitmentHex) {
        const resp = await this.request({ method: 'check_commitment', params: { commitment: commitmentHex } });
        if (!resp.ok) throw new Error(resp.error);
        return resp.data;
    }

    async mssState(masterPkHex) {
        const resp = await this.request({ method: 'mss_state', params: { master_pk: masterPkHex } });
        if (!resp.ok) throw new Error(resp.error);
        return resp.data;
    }

    // ── Peer Discovery ──────────────────────────────────────────────────────

    /// Ask the connected node for its known peers' WebRTC multiaddrs.
    /// These can be used for failover if the current connection drops.
    async discoverPeers() {
        try {
            const state = await this.getState();
            // If the node exposes webrtc_addrs in state response:
            if (state.webrtc_addrs) {
                for (const addr of state.webrtc_addrs) {
                    this.knownMultiaddrs.add(addr);
                }
            }
        } catch (e) {
            // Silent — we'll use whatever addrs we already know
        }
    }

    // ── Connection Management ───────────────────────────────────────────────

    _scheduleReconnect() {
        if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
            console.error('[light] Max reconnect attempts reached');
            this._emitStatus('failed');
            return;
        }

        this.reconnectAttempts++;
        const delay = RECONNECT_DELAY_MS * this.reconnectAttempts;
        console.log(`[light] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

        setTimeout(async () => {
            // Try known multiaddrs in order
            for (const addr of this.knownMultiaddrs) {
                try {
                    await this.connectTo(addr);
                    return; // Success — peer:connect event handles the rest
                } catch (_) {
                    continue;
                }
            }
            // All failed — try again
            this._scheduleReconnect();
        }, delay);
    }

    _emitStatus(status) {
        if (this._onStatusChange) {
            this._onStatusChange(status);
        }
    }

    /// Register a status change callback.
    /// status: 'connected' | 'disconnected' | 'failed'
    onStatusChange(cb) {
        this._onStatusChange = cb;
    }

    /// Graceful shutdown.
    async stop() {
        if (this.node) {
            await this.node.stop();
            this.node = null;
            this.isConnected = false;
            this.connectedPeer = null;
        }
    }
}
