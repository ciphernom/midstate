import { P2PClient } from './p2p.js';

/**
 * Midstate RPC & P2P Client
 * 
 * Provides a unified interface to interact with the Midstate network.
 * Automatically routes requests over direct P2P connections (WebRTC in Browser, 
 * TCP in Node.js) or falls back to traditional HTTPS REST endpoints.
 */
export class MidstateClient {
    /**
     * Initialize the client.
     * 
     * # Formal Specification
     * ```text
     * pre  target is a valid HTTP URL string OR an Array of Multiaddrs
     * post isP2P = true ⇔ target contains "/p2p/"
     * ```
     * 
     * @param {string|string[]} target - A single HTTP URL, or an Array of P2P Multiaddrs.
     */
    constructor(target = "https://rpc.cypherpunk.gold") {
        this.isP2P = Array.isArray(target) || target.includes("/p2p/");
        
        if (this.isP2P) {
            this.p2pAddrs = Array.isArray(target) ? target : [target];
            this.p2pClient = new P2PClient();
        } else {
            this.rpcUrl = target.replace(/\/+$/, '');
        }
    }

    /**
     * The underlying P2P transport, or null when this client is HTTP-backed.
     * Exposes peer-discovery internals (knownMultiaddrs, connectedPeer) for
     * diagnostics and tests without making callers reach into a private field.
     */
    getP2P() {
        return this.isP2P ? this.p2pClient : null;
    }

    /**
     * Establish the P2P connection to the network overlay.
     * Required if initialized with P2P Multiaddrs. No-op for HTTP.
     * 
     * # Formal Specification
     * ```text
     * pre  isP2P = true
     * post p2pClient.isConnected = true
     * ```
     */
    async connect() {
        if (this.isP2P) {
            await this.p2pClient.start(this.p2pAddrs);
        }
    }

    /**
     * Gracefully close the P2P connection.
     */
    async disconnect() {
        if (this.isP2P) {
            await this.p2pClient.stop();
        }
    }

    /**
     * Subscribe to real-time P2P Push events (e.g., new blocks, chat messages).
     * 
     * # Formal Specification
     * ```text
     * pre  isP2P = true
     * post callback is registered for LightNotification events
     * ```
     * 
     * @param {Function} callback - Function to execute on incoming events.
     */
    onPushEvent(callback) {
        if (this.isP2P) {
            this.p2pClient.onPushEvent(callback);
        } else {
            console.warn("Push events are only supported over direct P2P connections.");
        }
    }

    /**
     * Subscribe to connection status changes (connected, disconnected, failed).
     */
    onStatusChange(callback) {
        if (this.isP2P) {
            this.p2pClient.onStatusChange(callback);
        }
    }

    /**
     * Internal router: Dispatches calls to P2P Protocol or HTTP Fetch.
     * @private
     */
    async _route(method, endpoint, httpOptions = {}, rpcParams = {}) {
        if (this.isP2P) {
            // The server's LightRequest enum is adjacently tagged
            // (#[serde(tag="method", content="params")]). Unit variants like
            // `get_state` and `get_mempool` take NO content, so sending
            // `params: {}` for them fails to deserialize server-side — the node
            // then closes the stream without a response ("stream closed before
            // full response"). Only include `params` when there's actually data,
            // matching the reference web client.
            const hasParams = rpcParams && Object.keys(rpcParams).length > 0;
            const reqObj = hasParams ? { method, params: rpcParams } : { method };
            const resp = await this.p2pClient.request(reqObj);
            if (!resp.ok) throw new Error(`P2P RPC Error: ${resp.error}`);
            return resp.data !== undefined ? resp.data : resp.body;
        } else {
            const res = await fetch(`${this.rpcUrl}${endpoint}`, {
                ...httpOptions,
                headers: { 'Content-Type': 'application/json', ...httpOptions?.headers }
            });
            if (!res.ok) {
                let errorMsg = await res.text();
                try { errorMsg = JSON.parse(errorMsg).error || errorMsg; } catch (e) {}
                throw new Error(`HTTP Error (${res.status}): ${errorMsg}`);
            }
            return res.json();
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Public Network API
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Get the current network consensus state.
     * 
     * # Formal Specification
     * ```text
     * post result! = { height, depth, safe_depth, midstate, required_pow, ... }
     * ```
     */
    async getState() { 
        return this._route('get_state', '/state'); 
    }
    
    /**
     * Get raw block data by height.
     * 
     * @param {number} height - Block height
     */
    async getBlock(height) { 
        return this._route('get_block', `/block/${height}`, {}, { height }); 
    }
    
    /**
     * Get the current mempool transactions and size.
     */
    async getMempool() { 
        return this._route('get_mempool', '/mempool'); 
    }

    /**
     * Fetch compact block filters for light-client syncing.
     * 
     * @param {number} startHeight 
     * @param {number} endHeight 
     */
    async getFilters(startHeight, endHeight) {
        return this._route('get_filters', '/filters', 
            { method: 'POST', body: JSON.stringify({ start_height: startHeight, end_height: endHeight }) },
            { start_height: startHeight, end_height: endHeight }
        );
    }
    
    /**
     * Check if a specific coin_id exists in the current UTXO set.
     * 
     * @param {string} coinIdHex - 32-byte hex string
     */
    async checkCoin(coinIdHex) {
        return this._route('check', '/check', 
            { method: 'POST', body: JSON.stringify({ coin: coinIdHex }) }, 
            { coin: coinIdHex }
        );
    }

    /**
     * Check if a specific Phase 1 commitment exists in the chain state.
     * 
     * @param {string} commitmentHex - 32-byte hex string
     */
    async checkCommitment(commitmentHex) {
        return this._route('check_commitment', '/check_commitment', 
            { method: 'POST', body: JSON.stringify({ commitment: commitmentHex }) },
            { commitment: commitmentHex }
        );
    }

    /**
     * Retrieve the highest used index for an MSS Master Public Key.
     * 
     * @param {string} masterPkHex - 32-byte hex string
     */
    async getMssState(masterPkHex) {
        return this._route('mss_state', '/mss_state', 
            { method: 'POST', body: JSON.stringify({ master_pk: masterPkHex }) },
            { master_pk: masterPkHex }
        );
    }

    /**
     * Ask the node to construct a mining template.
     * 
     * @param {Object[]} coinbase - Array of {address, value, salt} objects
     */
    async getBlockTemplate(coinbase) {
        if (this.isP2P) {
            const resp = await this.p2pClient.request({ method: 'block_template', params: { coinbase } });
            if (!resp.ok) {
                // To maintain HTTP parity, we return the expected_total if it was a fee mismatch
                if (resp.data && resp.data.expected_total) {
                    return { ok: false, status: 409, json: () => Promise.resolve(resp.data) };
                }
                throw new Error(resp.error || "Failed to get block template");
            }
            return { ok: true, status: 200, json: () => Promise.resolve(resp.data) };
        }
        
        const res = await fetch(`${this.rpcUrl}/block_template`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ coinbase })
        });
        return { ok: res.ok, status: res.status, json: () => res.json() };
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Transaction Submission API
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Submit a completed Block to the network.
     * 
     * @param {Object} batch - The finalized batch object
     */
    async submitBatch(batch) {
        if (this.isP2P) {
            const resp = await this.p2pClient.request({ method: 'submit_batch', params: { batch } });
            return { ok: resp.ok, body: resp.ok ? null : (resp.error || 'rejected') };
        }
        const r = await fetch(`${this.rpcUrl}/api/internal/submit_batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(batch)
        });
        return { ok: r.ok, body: r.ok ? null : await r.text() };
    }

    /**
     * Phase 1: Submit a transaction commitment with Anti-Spam PoW.
     * 
     * # Formal Specification
     * ```text
     * pre  commitmentHex is 64 hex chars
     * pre  spamNonce is a valid integer satisfying network difficulty
     * post result.ok = true ⇔ Mempool accepts the commitment
     * ```
     * 
     * @param {string} commitmentHex 
     * @param {number} spamNonce 
     */
    async commit(commitmentHex, spamNonce) {
        if (this.isP2P) {
            const resp = await this.p2pClient.request({ 
                method: 'commit', params: { commitment: commitmentHex, spam_nonce: spamNonce } 
            });
            return { ok: resp.ok, body: resp.ok ? null : (resp.error || "Commit rejected") };
        }
        const r = await fetch(`${this.rpcUrl}/commit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ commitment: commitmentHex, spam_nonce: spamNonce })
        });
        return { ok: r.ok, body: r.ok ? null : await r.text() };
    }

    /**
     * Phase 2: Submit a Reveal transaction containing inputs and signatures.
     * 
     * # Formal Specification
     * ```text
     * pre  Reveal transaction payload is well-formed
     * pre  Corresponding commitment exists on-chain
     * post result.ok = true ⇔ Node validates signatures and broadcasts tx
     * ```
     * 
     * @param {Object|string} revealPayloadJson 
     */
    async send(revealPayloadJson) {
        const parsedPayload = typeof revealPayloadJson === 'string' ? JSON.parse(revealPayloadJson) : revealPayloadJson;
        if (this.isP2P) {
            const resp = await this.p2pClient.request({ method: 'send', params: { reveal: parsedPayload } });
            return { ok: resp.ok, body: resp.ok ? null : (resp.error || "Reveal rejected") };
        }
        const r = await fetch(`${this.rpcUrl}/send`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: typeof revealPayloadJson === 'string' ? revealPayloadJson : JSON.stringify(revealPayloadJson)
        });
        return { ok: r.ok, body: r.ok ? null : await r.text() };
    }

    /**
     * Originate a chat message onto the P2P overlay.
     * 
     * # Formal Specification
     * ```text
     * pre  words.length ≤ 10
     * pre  attachments.length ≤ 4
     * post Node mines ChatV2 PoW locally and broadcasts Message::ChatV2
     * ```
     * 
     * @param {number[]} words - Array of dictionary indices.
     * @param {number|null} replyTo - Nonce of parent message, or null.
     * @param {Object[]} attachments - Array of typed attachments.
     */
    async sendChat(words, replyTo = null, attachments = []) {
        if (this.isP2P) {
            const resp = await this.p2pClient.request({ 
                method: 'send_chat', params: { words, reply_to: replyTo, attachments } 
            });
            return { ok: resp.ok, body: resp.ok ? null : (resp.error || "Chat rejected") };
        }
        const r = await fetch(`${this.rpcUrl}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ words, reply_to: replyTo, attachments })
        });
        return { ok: r.ok, body: r.ok ? null : await r.text() };
    }
}
