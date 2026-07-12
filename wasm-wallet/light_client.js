// light_client.js — Browser-side libp2p light client for Midstate wallet
//
// Connects to full nodes over WebRTC direct (no HTTPS, no domain, no cert authority).
// Speaks the /midstate/light/2.0.0 JSON protocol over libp2p streams.
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
import { multiaddr } from '@multiformats/multiaddr';

const LIGHT_PROTOCOL = '/midstate/light/2.0.0';
const REQUEST_TIMEOUT_MS = 15_000;
const RECONNECT_DELAY_MS = 3_000;
const MAX_RECONNECT_ATTEMPTS = 5;

// ── Load-balancing placement ────────────────────────────────────────────────
// Spread light clients across full nodes instead of all dialing the first
// bootstrap address. On startup we shuffle the candidate list (random
// placement) and, if a node reports it's near its light-peer capacity, hop to
// another — bounded by MAX_PLACEMENT_HOPS so we always connect quickly even
// when every node is busy. Load is read from get_state (`light_load`, or
// `light_connections`/`max_light_connections`); nodes that omit it are treated
// as available, so this is a no-op against older nodes.
const LIGHT_LOAD_SOFT_THRESHOLD = 0.85; // hop off nodes above 85% light-peer capacity
const MAX_PLACEMENT_HOPS = 3;           // cap on nodes to probe before settling

// Fisher–Yates shuffle (returns a new array). Used to randomize the order in
// which candidate nodes are tried so clients don't all pile onto the first one.
function shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

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
            // --- FIX: Disable the local IP filter so we can dial 127.0.0.1 / 192.168.x.x ---
            connectionGater: {
                denyDialMultiaddr: async () => false,
            },
        });

                // --- LISTEN FOR SERVER PUSHES ---
        /**
        * 
         * ## 1. Reasoning
         * Listens for asynchronous Push streams initiated by the Midstate full node 
         * (e.g., new block announcements, incoming P2P chat messages).
         * 
         * By using standard libp2p `stream.source` abstraction, this cleanly reads the 
         * length-prefixed application JSON without manual WebRTC-Protobuf un-framing, 
         * preventing payload corruption.
         * 
         * ## 2. Formal Specification
         * 
         * ```text
         * Pre:
         *   - Peer opens a new stream with protocol '/midstate/light-push/2.0.0'
         *   - Data follows: `u32LE(len) || JSON_Bytes`
         * 
         * Post:
         *   - Reassembles chunks until `total_bytes >= 4 + expected_len`
         *   - Emits parsed JSON to `this._onPushEvent`
         *   - Ignores malformed streams securely without throwing unhandled exceptions.
         * ```
         * 
         * ## 3. Safety / Invariants
         * - Protects against memory exhaustion by strictly adhering to the `expectedLen` 
         *   boundary before slicing the buffer and parsing.
         */
        this.node.handle('/midstate/light-push/2.0.0', async (stream, connection) => {
            console.log(`[light] Push stream incoming...`);
            if (!stream) return;
            
            try {
                const chunks = [];
                let totalLen = 0;
                
                // Ensure we read from the async iterable source
                const source = stream.source || stream;
                
                for await (const chunk of source) {
                    const bytes = chunk.subarray ? chunk.subarray() : chunk;
                    chunks.push(bytes);
                    totalLen += bytes.length;
                    
                    if (totalLen >= 4) {
                        const rawBuf = new Uint8Array(totalLen);
                        let offset = 0;
                        for (const c of chunks) { rawBuf.set(c, offset); offset += c.length; }
                        
                        const expectedLen = new DataView(rawBuf.buffer, rawBuf.byteOffset).getUint32(0, true);
                        console.log(`[light] Push stream read ${totalLen} bytes. Expected payload length: ${expectedLen}.`);

                        if (totalLen >= 4 + expectedLen) {
                            console.log(`[light] Push stream payload fully received. Parsing...`);
                            const jsonStr = new TextDecoder().decode(rawBuf.slice(4, 4 + expectedLen));
                            const notif = JSON.parse(jsonStr);
                            if (this._onPushEvent) this._onPushEvent(notif);
                            break;
                        }
                    }
                }
            } catch (e) { 
                console.warn("[light] Push handle error", e); 
            } finally {
                try { if (stream.close) stream.close(); } catch (_) {}
            }
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

        // Seed the candidate set, then place our connection load-aware across a
        // SHUFFLED list rather than always dialing the first bootstrap address.
        // See _placeConnection.
        if (addrs && addrs.length > 0) {
            for (const addr of addrs) {
                this.knownMultiaddrs.add(addr);
            }
            await this._placeConnection([...this.knownMultiaddrs]);
        }
    }

    // ── Load-aware placement ─────────────────────────────────────────────────
    //
    // Connect across a shuffled candidate set. If a node reports it's near its
    // light-peer capacity (via get_state), drop it and try another — bounded by
    // MAX_PLACEMENT_HOPS so startup stays fast even when every node is busy.
    // The first reachable node and the least-loaded node seen are remembered, so
    // heavy load can never leave us unconnected: if every sampled node is hot we
    // reconnect to the lightest one. Nodes that don't advertise load are treated
    // as available (no-op against older nodes).
    async _placeConnection(candidatesIn) {
        const candidates = shuffle(candidatesIn);
        let firstReachable = null;
        let lightestAddr = null;
        let lightestLoad = Infinity;
        let hops = 0;

        for (const addr of candidates) {
            if (hops >= MAX_PLACEMENT_HOPS) break;

            try {
                await this.connectTo(addr);
            } catch (e) {
                console.warn('[light] Skipping unreachable peer:', addr);
                continue;
            }

            // peer:connect sets isConnected asynchronously — wait briefly for it.
            await this._waitConnected(2500);
            if (!this.isConnected) continue;

            hops++;
            if (!firstReachable) firstReachable = addr;

            // Best-effort: read load + learn about more peers for failover.
            let load = 0; // unknown / older node → treat as available
            try {
                const state = await this.getState();
                if (state && Array.isArray(state.webrtc_addrs)) {
                    for (const a of state.webrtc_addrs) this.knownMultiaddrs.add(a);
                }
                const r = this._loadRatio(state);
                if (r !== null) load = r;
            } catch (_) { /* couldn't read state; treat as available */ }

            if (load < lightestLoad) { lightestLoad = load; lightestAddr = addr; }

            // Comfortably loaded (or no load info) → settle here.
            if (load < LIGHT_LOAD_SOFT_THRESHOLD) return true;

            // Node is hot: drop it and try another. firstReachable / lightestAddr
            // are remembered so we can always fall back.
            console.log(`[light] ${addr} busy (${Math.round(load * 100)}% of light capacity); trying another node`);
            await this._disconnectCurrent();
        }

        // Everything we sampled was busy (or we just dropped the last one).
        // Reconnect to the least-loaded node we saw — never fail purely due to load.
        if (this.isConnected) return true;
        const target = lightestAddr || firstReachable;
        if (target) {
            try {
                await this.connectTo(target);
                await this._waitConnected(2500);
            } catch (_) {}
            if (this.isConnected) return true;
        }
        throw new Error('Could not connect to any WebRTC peers');
    }

    /// Poll until connected (peer:connect sets the flag) or timeout. Returns isConnected.
    async _waitConnected(ms) {
        const deadline = Date.now() + ms;
        while (!this.isConnected && Date.now() < deadline) {
            await new Promise(r => setTimeout(r, 50));
        }
        return this.isConnected;
    }

    /// Hang up the current peer WITHOUT triggering the reconnect scheduler.
    /// We null connectedPeer first so the peer:disconnect handler's guard sees
    /// no match and skips _scheduleReconnect — this is a deliberate hop during
    /// placement, not a dropped connection.
    async _disconnectCurrent() {
        const peer = this.connectedPeer;
        this.isConnected = false;
        this.connectedPeer = null;
        if (peer && this.node) {
            try { await this.node.hangUp(peer); } catch (_) {}
        }
    }

    /// Extract a light-peer load ratio in [0,1] from a get_state response,
    /// or null if the node doesn't advertise load.
    _loadRatio(state) {
        if (!state) return null;
        if (typeof state.light_load === 'number') {
            return Math.max(0, Math.min(1, state.light_load));
        }
        if (typeof state.light_connections === 'number'
            && typeof state.max_light_connections === 'number'
            && state.max_light_connections > 0) {
            return Math.max(0, Math.min(1, state.light_connections / state.max_light_connections));
        }
        return null;
    }

    /// Connect to a specific multiaddr with a 5-second timeout.
    async connectTo(addrStr) {
        const ma = multiaddr(addrStr);
        // AbortSignal.timeout ensures we don't hang forever on firewalled peers
        const conn = await this.node.dial(ma, { signal: AbortSignal.timeout(5000) });
        console.log('[light] Dialed:', addrStr);
        return conn;
    }

async submitChat(sender, timestamp, nonce, replyTo, words, attachments = []) {
        const resp = await this.request({
            method: 'submit_chat',
            params: { sender, timestamp, nonce, reply_to: replyTo, words, attachments },
        });
        return resp;
    }

/**
     * 
     * ## 1. Reasoning
     * The `request` method handles the lifecycle of an outbound RPC call over the 
     * Midstate Light Protocol. Previous iterations attempted to manually parse WebRTC 
     * Protobuf framing, which corrupted already-unwrapped application data yielded 
     * by modern `js-libp2p` multiplexed streams. 
     * 
     * This implementation uses standard `stream.source` and `stream.sink` abstractions, 
     * ensuring protocol compatibility regardless of the underlying transport (WebRTC, 
     * TCP, etc.). It maintains strict timeout and retry safety.
     * 
     * ## 2. Formal Specification
     * 
     * ```text
     * Pre:
     *   - this.isConnected == true
     *   - req is a valid JSON-serializable object
     * 
     * Post:
     *   result = Ok(response) =>
     *     - A stream was opened to `connectedPeer`
     *     - `len(req_bytes)_LE || req_bytes` was successfully sent
     *     - A well-formed JSON response was received and parsed before REQUEST_TIMEOUT_MS
     * 
     *   result = Err(e) =>
     *     - Stream was closed/aborted to prevent resource leaks
     * ```
     * 
     * ## 3. Safety / Invariants
     * - **Resource Exhaustion Guard:** The stream must be explicitly aborted/closed in `finally` 
     *   to prevent leaking multiplexer channels on timeouts or malformed peer data.
     * - **Data Integrity:** Reads strictly adhere to the 4-byte Little Endian length prefix 
     *   enforced by the Midstate rust node (`LightRequest` / `LightResponse` serialization).
     */
    async request(req, _retries = 2) {
        console.log(`[light] ---> Requesting ${req.method} (retries left: ${_retries})`);
        
        if (!this.isConnected || !this.connectedPeer) {
            throw new Error('Not connected to any peer');
        }

        const conns = this.node.getConnections(this.connectedPeer);
        if (!conns || conns.length === 0) throw new Error('No active connection to peer');
        
        let stream;
        try {
            stream = await conns[0].newStream([LIGHT_PROTOCOL]);
        } catch (e) {
            console.error(`[light] newStream failed for ${req.method}:`, e);
            throw e;
        }

        try {
            const jsonBytes = new TextEncoder().encode(JSON.stringify(req));
            const lenBuf = new Uint8Array(4);
            new DataView(lenBuf.buffer).setUint32(0, jsonBytes.length, true);
            const msg = new Uint8Array(4 + jsonBytes.length);
            msg.set(lenBuf, 0);
            msg.set(jsonBytes, 4);

            console.log(`[light] ${req.method}: Writing ${msg.length} bytes...`);

            // Use standard js-libp2p sink if available
            if (typeof stream.sink === 'function') {
                await stream.sink((async function*() { yield msg; })());
            } else {
                console.warn(`[light] stream.sink is missing, trying fallback stream.send...`);
                if (typeof stream.sendData === 'function') stream.sendData(msg);
                else if (typeof stream.send === 'function') stream.send(msg);
                if (typeof stream.sendCloseWrite === 'function') stream.sendCloseWrite().catch(()=>{});
                else if (typeof stream.closeWrite === 'function') stream.closeWrite().catch(()=>{});
            }

            console.log(`[light] ${req.method}: Awaiting response...`);

            const readWithTimeout = async () => {
                const chunks = [];
                let totalLen = 0;
                
                // Fallback to iterable stream if stream.source is missing
                const source = stream.source || stream; 
                
                for await (const chunk of source) {
                    const bytes = chunk.subarray ? chunk.subarray() : chunk;
                    chunks.push(bytes);
                    totalLen += bytes.length;
                    
                    if (totalLen >= 4) {
                        const rawBuf = new Uint8Array(totalLen);
                        let offset = 0;
                        for (const c of chunks) { rawBuf.set(c, offset); offset += c.length; }
                        
                        const expectedLen = new DataView(rawBuf.buffer, rawBuf.byteOffset).getUint32(0, true);
                        console.log(`[light] ${req.method}: Read ${totalLen} bytes. Expected payload length: ${expectedLen}.`);
                        
                        if (totalLen >= 4 + expectedLen) {
                            console.log(`[light] ${req.method}: Payload fully received.`);
                            return rawBuf.slice(0, 4 + expectedLen); 
                        }
                    }
                }
                throw new Error("Stream closed before completing response");
            };

            const appData = await Promise.race([
                readWithTimeout(),
                new Promise((_, reject) =>
                    setTimeout(() => reject(new Error('Stream read timeout')), REQUEST_TIMEOUT_MS)
                )
            ]);

            const respLen = new DataView(appData.buffer, appData.byteOffset).getUint32(0, true);
            const respJson = new TextDecoder().decode(appData.slice(4, 4 + respLen));
            return JSON.parse(respJson);

        } catch (err) {
            console.error(`[light] Request ${req.method} failed:`, err);
            if (_retries > 0) {
                console.log(`[light] Retrying ${req.method}...`);
                try { if (stream.abort) stream.abort(new Error('retry')); } catch (_) {}
                return this.request(req, _retries - 1);
            }
            throw err;
        } finally {
            try { if (stream.close) stream.close(); } catch (_) {}
        }
    }

    /// Send a JSON request over the light protocol and return the parsed response.
    ///
    /// req: { method: 'get_state' } or { method: 'get_block', params: { height: 42 } }
    /// Returns: parsed LightResponse { ok, data?, error? }
async OLDrequest(req, _retries = 2) {
    if (!this.isConnected || !this.connectedPeer) {
        throw new Error('Not connected to any peer');
    }

    const conns = this.node.getConnections(this.connectedPeer);
    if (!conns || conns.length === 0) throw new Error('No active connection to peer');
    const stream = await conns[0].newStream([LIGHT_PROTOCOL]);

   try {
            // Build the length-prefixed message
            const jsonBytes = new TextEncoder().encode(JSON.stringify(req));
            const lenBuf = new Uint8Array(4);
            new DataView(lenBuf.buffer).setUint32(0, jsonBytes.length, true);
            const msg = new Uint8Array(4 + jsonBytes.length);
            msg.set(lenBuf, 0);
            msg.set(jsonBytes, 4);

            // Write in chunks to respect WebRTC SCTP message size limits (16 KB)
            const CHUNK_SIZE = 16384; 
            for (let i = 0; i < msg.length; i += CHUNK_SIZE) {
                stream.sendData(msg.slice(i, i + CHUNK_SIZE));
                
                // THE FIX: Yield to the JS event loop for 10ms so the WebRTC buffer can drain!
                await new Promise(r => setTimeout(r, 10)); 
            }
            stream.sendCloseWrite();

            // Read: incomingData is an async iterator
            const chunks = [];
            let totalLen = 0;
            let gotReset = false;

            const readWithTimeout = async () => {
                for await (const chunk of stream.incomingData) {
                    const bytes = chunk instanceof Uint8Array
                        ? chunk
                        : new Uint8Array(chunk.buffer ?? chunk);
                    chunks.push(bytes);
                    totalLen += bytes.length;
                    if (chunkContainsReset(bytes)) { gotReset = true; break; }
                    if (chunkContainsFin(bytes)) { break; }
                }
            };

            await Promise.race([
                readWithTimeout(),
                new Promise((_, reject) =>
                    // Uses the new shorter REQUEST_TIMEOUT_MS
                    setTimeout(() => reject(new Error('Stream read timeout')), REQUEST_TIMEOUT_MS)
                )
            ]);

            if (gotReset && _retries > 0) {
                try { stream.abort(new Error('reset')); } catch (_) {}
                return this.request(req, _retries - 1);
            }

            const rawBuf = new Uint8Array(totalLen);
            let offset = 0;
            for (const c of chunks) { rawBuf.set(c, offset); offset += c.length; }

            const appData = decodeWebRTCStreamData(rawBuf);
            if (appData.length < 4) throw new Error('Response too short');
            const respLen = new DataView(appData.buffer, appData.byteOffset).getUint32(0, true);
            const respJson = new TextDecoder().decode(appData.slice(4, 4 + respLen));
            return JSON.parse(respJson);

        } finally {
            try { stream.abort(new Error('done')); } catch (_) {}
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

    /**
     * Originate a chat over the WebRTC light protocol.
     *
     * Server-side this hits `LightRequest::SendChat` in `src/node.rs`,
     * which enqueues a `NodeCommand::SendChat` with `sender_override =
     * Some(<our light-peer-id>)`. The node mines v2 PoW (~10 ms) and
     * broadcasts as `Message::ChatV2`.
     *
     * @param {number[]}                                words      0..=10 indices into CHAT_DICTIONARY
     * @param {number|null}                             replyTo    Parent message nonce, or null
     * @param {{kind:"address",value:string}[]} [attachments=[]]   0..=4 typed attachments
     *                                                             (value: 64-char lowercase hex for address)
     */
    async sendChat(words, replyTo, attachments = []) {
        const resp = await this.request({
            method: 'send_chat',
            params: { words, reply_to: replyTo, attachments },
        });
        return resp;
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
            // Try known multiaddrs in RANDOM order so reconnecting clients spread
            // across nodes instead of all reconnecting to the same one.
            for (const addr of shuffle([...this.knownMultiaddrs])) {
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
