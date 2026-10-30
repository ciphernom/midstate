// pow.js — Which proof of work a template asks for.
//
// # Reasoning
//
// Midstate changes proof of work at a height. Blocks below the node's
// `WALK_ACTIVATION_HEIGHT` (420,000) carry the BLAKE3 chain: `create_extension`,
// 1,000,000 sequential BLAKE3 hashes per nonce. Blocks from that height carry
// Midstate Walk: a 256 KiB scratchpad walked 2^18 steps by a chain of
// multiply, divide and AES-round operations (`docs/MIDSTATE_WALK.md` in the
// node repository).
//
// The node and the pool name the algorithm in every template as `pow`, so a
// miner never needs to know the activation height: it reads the field. A
// template without one comes from a node or pool that predates the switch, and
// such a server only builds BLAKE3-chain blocks, so that is the default.
//
// An unknown name is refused rather than mined. `search_nonces_pow` returns
// "nothing found" for a name it does not know, so passing one through would
// burn CPU on a search that can never succeed.

/** Below `WALK_ACTIVATION_HEIGHT`. */
export const BLAKE3_CHAIN = 'blake3-chain';
/** From `WALK_ACTIVATION_HEIGHT`. */
export const WALK = 'walk';
/** Every proof of work this SDK can mine. */
export const POW_NAMES = Object.freeze([BLAKE3_CHAIN, WALK]);

/**
 * The proof of work a template or pool job asks for.
 *
 * @param {Object} template A node template or pool job.
 * @returns {string} One of {@link POW_NAMES}.
 * @throws {Error} If the template names a proof of work this SDK cannot mine.
 */
export function templatePow(template) {
    const pow = template?.pow ?? BLAKE3_CHAIN;
    if (!POW_NAMES.includes(pow)) {
        throw new Error(`Unknown proof of work "${pow}". This SDK mines ${POW_NAMES.join(' and ')}; update it.`);
    }
    return pow;
}

/**
 * A `(midstate, target, nonce, iterations) => nonce | undefined` search for one
 * proof of work, over `iterations × 4` nonces from `nonce`, as `search_nonces`.
 *
 * @param {Object} wasm The `pkg/wasm_wallet.js` module namespace.
 * @param {string} pow  One of {@link POW_NAMES}.
 * @throws {Error} If `pkg/` predates `search_nonces_pow` and `pow` is not the BLAKE3 chain.
 */
export function searchFor(wasm, pow) {
    if (!POW_NAMES.includes(pow)) throw new Error(`Unknown proof of work "${pow}".`);
    if (typeof wasm.search_nonces_pow === 'function') {
        return (midstate, target, nonce, iterations) =>
            wasm.search_nonces_pow(pow, midstate, target, nonce, iterations);
    }
    if (pow === BLAKE3_CHAIN) return wasm.search_nonces;
    throw new Error(`pkg/ predates the ${pow} proof of work; rebuild it from the node's wasm-wallet.`);
}

/**
 * The extension (`{ nonce, final_hash }` as JSON) a block carries for `nonce`,
 * recomputed with the template's proof of work.
 *
 * @param {Object} inner A `WebWallet` (`Wallet.inner`).
 * @returns {string|null} `null` if `pkg/` cannot compute this proof of work.
 */
export function buildExtension(inner, pow, midstate, nonce) {
    if (typeof inner.build_solo_extension_pow === 'function') {
        return inner.build_solo_extension_pow(pow, midstate, nonce) ?? null;
    }
    return pow === BLAKE3_CHAIN ? inner.build_solo_extension(midstate, nonce) ?? null : null;
}
