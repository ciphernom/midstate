// pool.js — Client for the Midstate mining pool.
//
// ═══════════════════════════════════════════════════════════════════════════
//  Reasoning (module level)
// ═══════════════════════════════════════════════════════════════════════════
//
// Solo mining on this chain is a lottery a light client cannot realistically
// win. `create_extension` is a 1,000,000-iteration sequential BLAKE3 chain, so a
// browser or laptop lands 15–30 nonces/second against a network target set for
// the whole hashrate. Pool mining is how such a client actually earns: the pool
// hands out an easier `share_target`, credits work against it, and rebuilds and
// submits the block itself when a share also clears `network_target`.
//
// Three consequences shape this module.
//
// ── A share is not a block, and the job survives it ──
//
// Finding a share does NOT invalidate the template. The `mining_midstate` is
// unchanged, so a miner that halts on a hit — correct solo behaviour, since
// there the tip has moved — will sit idle until the next network block, because
// the "restart on new midstate" path never fires. Pool mining must continue from
// the next nonce. This is the `continueOnFound` distinction, and getting it
// wrong looks like a miner that works for one share and then stops.
//
// ── The pool rebuilds the block, so submission is tiny ──
//
// A share is `(address, job_id, nonce, worker)`. `job_id` is what lets the pool
// reject work mined against a superseded template; it must be echoed back
// exactly. The `batch_template` in a job is informational — for showing a
// transaction count — and is not what gets submitted.
//
// ── "The pool said no" is not "the request failed" ──
//
// `requestRaw` returns the envelope rather than throwing on `ok: false`. A miner
// absent from the precommitment tree is an *answer* to `get_proof`, not an
// exception; collapsing both into a thrown Error is what lets a dropped stream
// masquerade as the pool omitting a miner, which halts mining for the wrong
// reason. `request` throws, and is for calls where absence really is fatal.
//
// Wire format is the same as the light protocol: a 4-byte little-endian length
// prefix followed by JSON, one request/response per stream.

import { encodeFrame } from './bincode.js';
import { blake3_hash_hex } from '../pkg/wasm_wallet.js';

export const POOL_PROTOCOL = '/midstate/pool/1.0.0';
export const POOL_PUSH_PROTOCOL = '/midstate/pool-push/1.0.0';
const REQUEST_TIMEOUT_MS = 15_000;

/**
 * Pool client.
 *
 * The transport is injected rather than built here: the SDK runs under Node
 * (TCP) and in browsers (WebRTC-direct), and the pool protocol itself is
 * identical over both. `MidstateP2P` supplies a compatible dialer; tests supply
 * a stub. This also keeps the framing independently testable without libp2p.
 */
export class PoolClient {
    /**
     * @param {Object} transport Must provide
     *   `openStream(protocol) => { send(bytes), onMessage(cb), onClose(cb), close() }`.
     */
    constructor(transport) {
        this.transport = transport;
        this.isConnected = false;
        this._onNotify = null;
        this._onStatus = null;
    }

    onNotify(cb) { this._onNotify = cb; return this; }
    onStatus(cb) { this._onStatus = cb; return this; }

    _emit(status, detail) {
        if (this._onStatus) { try { this._onStatus(status, detail); } catch { /* never fatal */ } }
    }

    async connect(addr) {
        const id = await this.transport.dial(addr, {
            // The pool pushes `notify` the instant a job rotates, so the client
            // does not have to poll for a template it already has.
            [POOL_PUSH_PROTOCOL]: (json) => {
                if (this._onNotify) { try { this._onNotify(json); } catch { /* never fatal */ } }
            },
        });
        this.isConnected = true;
        this._emit('connected', id);
        return id;
    }

    async disconnect() {
        this.isConnected = false;
        try { await this.transport.close(); } catch { /* already gone */ }
        this._emit('disconnected');
    }

    /**
     * One request, one stream, returning the raw envelope.
     *
     * Retries once on a stream reset. libp2p streams are reset under entirely
     * normal conditions (peer churn, the remote closing early), and a single
     * retry on a fresh stream turns a routine blip into a non-event.
     *
     * @returns {Promise<{ok:boolean, data?:Object, error?:string}>}
     */
    async requestRaw(method, params = {}, retry = 1) {
        if (!this.isConnected) throw new Error('Not connected to pool');
        try {
            const bytes = await this.transport.request(
                POOL_PROTOCOL, encodeFrame({ method, params }), REQUEST_TIMEOUT_MS
            );
            return JSON.parse(new TextDecoder().decode(bytes));
        } catch (e) {
            if (retry > 0 && /reset|closed|abort/i.test(e.message || '')) {
                return this.requestRaw(method, params, retry - 1);
            }
            throw e;
        }
    }

    /** Throws on a pool-reported error. For calls where absence is fatal. */
    async request(method, params = {}) {
        const parsed = await this.requestRaw(method, params);
        if (!parsed.ok) throw new Error(parsed.error || 'Pool returned an error');
        return parsed.data;
    }

    /**
     * Current job.
     *
     * @returns {Promise<Object>} `{ job_id, mining_midstate, target,
     *   network_target, batch_template, height, merkle_root, ... }`. Note
     *   `target` is the SHARE target; `network_target` is informational, for
     *   showing the real block odds behind a share.
     */
    getTemplate(address) { return this.request('get_template', { address }); }

    /**
     * Precommitment proof for an address.
     *
     * Raw envelope on purpose: a miner absent from the tree is an answer.
     */
    getProof(address) { return this.requestRaw('get_proof', { address }); }

    /**
     * Submit a share.
     *
     * The nonce goes as a decimal STRING. It is a u64, and a JSON number loses
     * precision above 2^53 — a silently corrupted nonce is a share the pool
     * cannot reproduce and therefore rejects.
     */
    /**
     * Fetch and verify this miner's inclusion in the pool's precommitment.
     *
     * Returns a structured verdict rather than throwing, because "the pool has
     * not recorded you yet" is a normal state for a miner that has just
     * connected — distinguishing it from "the pool is lying" is the whole point.
     *
     * @param {string} address Payout address.
     * @param {string} rootHex `merkle_root` from the current job.
     * @returns {Promise<{included:boolean, verified:boolean, score:number|null, reason:string|null}>}
     */
    async auditInclusion(address, rootHex) {
        const env = await this.getProof(address);
        if (!env.ok) {
            return { included: false, verified: false, score: null, reason: env.error || 'not in tree' };
        }
        const d = env.data || {};
        const proof = d.proof || [];
        const index = d.index ?? d.leaf_index;
        const score = d.score ?? 0;
        if (index === undefined || index === null) {
            return { included: true, verified: false, score, reason: 'proof carried no leaf index' };
        }
        const verified = verifyPoolProof(address, score, index, proof, rootHex);
        return {
            included: true,
            verified,
            score: Number(score),
            reason: verified ? null : 'proof does not reconstruct the job merkle_root',
        };
    }

    submitShare(address, jobId, nonce, worker = 'sdk') {
        return this.request('submit_share', {
            address, job_id: jobId, nonce: String(nonce), worker,
        });
    }
}

/**
 * Verify a pool inclusion proof against a job's precommitment root.
 *
 * # Reasoning
 *
 * `get_proof` is only useful if someone checks it. Without verification a pool
 * can hand a miner a plausible-looking proof, or omit it from the tree entirely,
 * and the miner keeps grinding shares it will never be paid for. The proof is
 * cheap to check — O(log N) hashes — and the root it must reach is already in
 * every job, so there is no reason not to.
 *
 * The reconstruction mirrors `ShareMerkleTree::build` in the node exactly:
 *
 *   leaf      = BLAKE3(address ⌢ le64(score))
 *   parent    = BLAKE3(left ⌢ right)
 *   odd node  = BLAKE3(node ⌢ node)      (a lone node is paired with itself)
 *
 * The odd case matters: `generate_proof` clamps the sibling index to the last
 * element, so a lone right-most node lists *itself* as its sibling. Verifying
 * with a conventional "promote the odd node unchanged" rule would fail on
 * exactly the miners at the end of the tree.
 *
 * Leaves are sorted by address before hashing, so `index` is a position in that
 * sorted order and the caller cannot choose it.
 *
 * # Formal Specification
 *
 * ```text
 * Pre:
 *   - #proof? = ceil(log2(#leaves))
 *
 * Post:
 *   ok! = (fold_up(leaf(address?, score?), index?, proof?) = root?)
 * ```
 *
 * @param {string} addressHex  64-hex miner address.
 * @param {number|bigint} score  Share score claimed in the leaf.
 * @param {number} index       Leaf index in the sorted tree.
 * @param {string[]} proof     Sibling hashes, leaf level upward.
 * @param {string} rootHex     Expected root (from the job's `merkle_root`).
 * @returns {boolean}
 */
export function verifyPoolProof(addressHex, score, index, proof, rootHex) {
    const addr = String(addressHex).replace(/^0x/, '').toLowerCase();
    if (addr.length !== 64) return false;
    if (!Array.isArray(proof)) return false;

    // leaf = BLAKE3(address ⌢ score as little-endian u64)
    const scoreLe = new Uint8Array(8);
    new DataView(scoreLe.buffer).setBigUint64(0, BigInt(score), true);
    const scoreHex = Array.from(scoreLe).map((b) => b.toString(16).padStart(2, '0')).join('');

    let node = blake3_hash_hex(addr + scoreHex);
    let idx = Number(index);

    for (const sibRaw of proof) {
        const sib = String(sibRaw).replace(/^0x/, '').toLowerCase();
        if (sib.length !== 64) return false;
        // A lone right-most node is its own sibling, so sib === node there and
        // the order below still produces BLAKE3(node ⌢ node).
        node = (idx % 2 === 1) ? blake3_hash_hex(sib + node) : blake3_hash_hex(node + sib);
        idx = Math.floor(idx / 2);
    }
    return node === String(rootHex).replace(/^0x/, '').toLowerCase();
}

/**
 * Adapts a `MidstateP2P` instance to the transport shape `PoolClient` expects.
 *
 * Kept separate so the pool protocol does not depend on the SDK's libp2p setup,
 * and so tests can drive `PoolClient` without a network.
 */
export function libp2pPoolTransport(p2p) {
    return {
        async dial(addr, pushHandlers) {
            return p2p.dialPeer(addr, pushHandlers);
        },
        async request(protocol, frameBytes, timeoutMs) {
            return p2p.requestRawFramed(protocol, frameBytes, timeoutMs);
        },
        async close() { return p2p.disconnect(); },
    };
}
