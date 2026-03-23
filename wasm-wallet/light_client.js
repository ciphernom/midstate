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
import { noise } from '@chainsafe/libp2p-noise';
import { yamux } from '@chainsafe/libp2p-yamux';
import { multiaddr } from '@multiformats/multiaddr';

const LIGHT_PROTOCOL = '/midstate/light/1.0.0';
const REQUEST_TIMEOUT_MS = 30_000;
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

    /// Start the libp2p node and connect to a bootstrap peer.
    ///
    /// bootstrapAddr: string like "/ip4/203.0.113.10/udp/9335/webrtc-direct/certhash/uEi..."
    async start(bootstrapAddr) {
        this.node = await createLibp2p({
            transports: [webRTCDirect()],
            connectionEncrypters: [noise()],
            streamMuxers: [yamux()],
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

        // Connect to bootstrap
        if (bootstrapAddr) {
            this.knownMultiaddrs.add(bootstrapAddr);
            await this.connectTo(bootstrapAddr);
        }
    }

    /// Connect to a specific multiaddr.
    async connectTo(addrStr) {
        try {
            const ma = multiaddr(addrStr);
            const conn = await this.node.dial(ma);
            console.log('[light] Dialed:', addrStr);
            return conn;
        } catch (e) {
            console.warn('[light] Failed to dial', addrStr, e.message);
            throw e;
        }
    }

    /// Send a JSON request over the light protocol and return the parsed response.
    ///
    /// req: { method: 'get_state' } or { method: 'get_block', params: { height: 42 } }
    /// Returns: parsed LightResponse { ok, data?, error? }
async request(req, _retries = 2) {
    if (!this.isConnected || !this.connectedPeer) {
        throw new Error('Not connected to any peer');
    }

    const stream = await this.node.dialProtocol(this.connectedPeer, LIGHT_PROTOCOL);

    try {
        const jsonBytes = new TextEncoder().encode(JSON.stringify(req));
        const lenBuf = new Uint8Array(4);
        new DataView(lenBuf.buffer).setUint32(0, jsonBytes.length, true);
        const msg = new Uint8Array(4 + jsonBytes.length);
        msg.set(lenBuf, 0);
        msg.set(jsonBytes, 4);

        await stream.sendData(msg);
        stream.sendCloseWrite();

        // Collect raw protobuf-framed bytes from incomingData.
        // The iterator never terminates on stream close — we must detect
        // FIN/RESET flags in the protobuf framing and break manually.
        let rawBuf = new Uint8Array(0);
        let gotReset = false;
        const readAll = async () => {
            try {
                for await (const chunk of stream.incomingData) {
                    const bytes = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
                    const merged = new Uint8Array(rawBuf.length + bytes.length);
                    merged.set(rawBuf);
                    merged.set(bytes, rawBuf.length);
                    rawBuf = merged;
                    // Check for RESET — server killed the stream (e.g. first
                    // request after connect gets clobbered by binary protocol probe).
                    // Flag can be standalone or appended to a data chunk.
                    if (chunkContainsReset(bytes)) {
                        gotReset = true;
                        break;
                    }
                    // Check for FIN — server is done writing.
                    // For large responses FIN arrives in the same chunk as
                    // the last data, not as a separate 3-byte chunk.
                    if (chunkContainsFin(bytes)) {
                        break;
                    }
                }
            } catch (_) {
                // Stream close may throw AggregateError as background rejection
            }
        };

        const timeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Stream read timeout')), REQUEST_TIMEOUT_MS)
        );

        await Promise.race([readAll(), timeoutPromise]);

        // If we got RESET, retry (server's initial binary protocol probe
        // can kill the first light stream after a new connection)
        if (gotReset && _retries > 0) {
            try { stream.close(); } catch (_) {}
            return this.request(req, _retries - 1);
        }

        // Decode protobuf framing to extract application data
        const appData = decodeWebRTCStreamData(rawBuf);

        if (appData.length < 4) throw new Error('Response too short');
        const respLen = new DataView(appData.buffer, appData.byteOffset).getUint32(0, true);
        const respJson = new TextDecoder().decode(appData.slice(4, 4 + respLen));
        return JSON.parse(respJson);
    } finally {
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
            status: resp.ok ? 200 : 500,
            json: () => Promise.resolve(resp.data),
            text: () => Promise.resolve(resp.ok ? JSON.stringify(resp.data) : (resp.error || 'error')),
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
