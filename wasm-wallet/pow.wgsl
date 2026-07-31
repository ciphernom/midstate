// BLAKE3 mining kernel for the PoW extension chain.
//
// COPY OF `src/core/pow.wgsl`'s equivalent — the canonical version is the inline
// `SHADER` literal in the node's `src/core/gpu_mining.rs`. A fix there should be
// ported here (and vice versa).
//
// Divergence is caught, not silent: `gpu_miner.js` refuses to mine until this
// kernel reproduces the WASM `build_solo_extension` bit-for-bit over a full
// 1,000,000-iteration chain, and re-checks every candidate it surfaces. So a
// stale copy fails loudly at startup rather than producing rejected shares.
//
// CONSENSUS-CRITICAL: every value below mirrors `create_extension` /
// `simd_mining::compress_4way` exactly. The per-nonce result MUST be
// bit-identical to the scalar reference or the miner produces rejected
// blocks/shares. The compression body is machine-generated from the same
// MSG_SCHEDULE the CPU path uses.
//
//   per nonce:  h = blake3_40(midstate || nonce_le8)        // block_len = 40
//               repeat EXTENSION_ITERATIONS:  h = blake3_32(h)  // block_len = 32
//   the chaining value fed into every compression is the IV (each step is a
//   fresh BLAKE3 of a 32-byte input, NOT a running hash).

const IV0: u32 = 0x6A09E667u;
const IV1: u32 = 0xBB67AE85u;
const IV2: u32 = 0x3C6EF372u;
const IV3: u32 = 0xA54FF53Au;
const IV4: u32 = 0x510E527Fu;
const IV5: u32 = 0x9B05688Cu;
const IV6: u32 = 0x1F83D9ABu;
const IV7: u32 = 0x5BE0CD19u;
const FLAGS: u32 = 11u; // CHUNK_START | CHUNK_END | ROOT  (1 | 2 | 8)

struct Params {
    midstate: array<u32, 8>,  // u32::from_le_bytes of each 4-byte group of the 32-byte midstate
    tgt:      array<u32, 8>,  // from_be_bytes of each 4-byte group of the 32-byte target ('target' is a WGSL reserved word)
    pool:     array<u32, 8>,  // same encoding as target; used only when has_pool != 0
    base_lo:  u32,
    base_hi:  u32,
    n_nonces: u32,
    iters:    u32,            // k_step: how many 32-byte iterations to apply this dispatch
    has_pool: u32,
    pad0: u32, pad1: u32, pad2: u32,
};

struct Winners {
    count: atomic<u32>,
    cap: u32,
    pad0: u32, pad1: u32,
    nonce_lo: array<u32, 256>,
    nonce_hi: array<u32, 256>,
    kind:     array<u32, 256>,   // 0 = block, 1 = share
};

@group(0) @binding(0) var<storage, read>       P:     Params;
@group(0) @binding(1) var<storage, read_write> state: array<u32>;   // n_nonces * 8 chaining words
@group(0) @binding(2) var<storage, read_write> out:   Winners;

fn rotr(x: u32, n: u32) -> u32 {
    return (x >> n) | (x << (32u - n));
}

// Reverse the 4 bytes of a word: turns a little-endian hash word into the
// big-endian key whose numeric order matches the [u8;32] lexicographic order.
fn bswap(x: u32) -> u32 {
    return ((x & 0xFFu) << 24u) | ((x & 0xFF00u) << 8u) | ((x >> 8u) & 0xFF00u) | ((x >> 24u) & 0xFFu);
}

fn compress(m: array<u32,16>, block_len: u32) -> array<u32,8> {
    var v0 = IV0; var v1 = IV1; var v2 = IV2; var v3 = IV3;
    var v4 = IV4; var v5 = IV5; var v6 = IV6; var v7 = IV7;
    var v8 = IV0; var v9 = IV1; var v10 = IV2; var v11 = IV3;
    var v12 = 0u; var v13 = 0u; var v14 = block_len; var v15 = FLAGS;
  // round 0
  v0 = v0 + v4 + m[0]; v12 = rotr(v12 ^ v0, 16u); v8 = v8 + v12; v4 = rotr(v4 ^ v8, 12u);
  v0 = v0 + v4 + m[1]; v12 = rotr(v12 ^ v0, 8u);  v8 = v8 + v12; v4 = rotr(v4 ^ v8, 7u);
  v1 = v1 + v5 + m[2]; v13 = rotr(v13 ^ v1, 16u); v9 = v9 + v13; v5 = rotr(v5 ^ v9, 12u);
  v1 = v1 + v5 + m[3]; v13 = rotr(v13 ^ v1, 8u);  v9 = v9 + v13; v5 = rotr(v5 ^ v9, 7u);
  v2 = v2 + v6 + m[4]; v14 = rotr(v14 ^ v2, 16u); v10 = v10 + v14; v6 = rotr(v6 ^ v10, 12u);
  v2 = v2 + v6 + m[5]; v14 = rotr(v14 ^ v2, 8u);  v10 = v10 + v14; v6 = rotr(v6 ^ v10, 7u);
  v3 = v3 + v7 + m[6]; v15 = rotr(v15 ^ v3, 16u); v11 = v11 + v15; v7 = rotr(v7 ^ v11, 12u);
  v3 = v3 + v7 + m[7]; v15 = rotr(v15 ^ v3, 8u);  v11 = v11 + v15; v7 = rotr(v7 ^ v11, 7u);
  v0 = v0 + v5 + m[8]; v15 = rotr(v15 ^ v0, 16u); v10 = v10 + v15; v5 = rotr(v5 ^ v10, 12u);
  v0 = v0 + v5 + m[9]; v15 = rotr(v15 ^ v0, 8u);  v10 = v10 + v15; v5 = rotr(v5 ^ v10, 7u);
  v1 = v1 + v6 + m[10]; v12 = rotr(v12 ^ v1, 16u); v11 = v11 + v12; v6 = rotr(v6 ^ v11, 12u);
  v1 = v1 + v6 + m[11]; v12 = rotr(v12 ^ v1, 8u);  v11 = v11 + v12; v6 = rotr(v6 ^ v11, 7u);
  v2 = v2 + v7 + m[12]; v13 = rotr(v13 ^ v2, 16u); v8 = v8 + v13; v7 = rotr(v7 ^ v8, 12u);
  v2 = v2 + v7 + m[13]; v13 = rotr(v13 ^ v2, 8u);  v8 = v8 + v13; v7 = rotr(v7 ^ v8, 7u);
  v3 = v3 + v4 + m[14]; v14 = rotr(v14 ^ v3, 16u); v9 = v9 + v14; v4 = rotr(v4 ^ v9, 12u);
  v3 = v3 + v4 + m[15]; v14 = rotr(v14 ^ v3, 8u);  v9 = v9 + v14; v4 = rotr(v4 ^ v9, 7u);
  // round 1
  v0 = v0 + v4 + m[2]; v12 = rotr(v12 ^ v0, 16u); v8 = v8 + v12; v4 = rotr(v4 ^ v8, 12u);
  v0 = v0 + v4 + m[6]; v12 = rotr(v12 ^ v0, 8u);  v8 = v8 + v12; v4 = rotr(v4 ^ v8, 7u);
  v1 = v1 + v5 + m[3]; v13 = rotr(v13 ^ v1, 16u); v9 = v9 + v13; v5 = rotr(v5 ^ v9, 12u);
  v1 = v1 + v5 + m[10]; v13 = rotr(v13 ^ v1, 8u);  v9 = v9 + v13; v5 = rotr(v5 ^ v9, 7u);
  v2 = v2 + v6 + m[7]; v14 = rotr(v14 ^ v2, 16u); v10 = v10 + v14; v6 = rotr(v6 ^ v10, 12u);
  v2 = v2 + v6 + m[0]; v14 = rotr(v14 ^ v2, 8u);  v10 = v10 + v14; v6 = rotr(v6 ^ v10, 7u);
  v3 = v3 + v7 + m[4]; v15 = rotr(v15 ^ v3, 16u); v11 = v11 + v15; v7 = rotr(v7 ^ v11, 12u);
  v3 = v3 + v7 + m[13]; v15 = rotr(v15 ^ v3, 8u);  v11 = v11 + v15; v7 = rotr(v7 ^ v11, 7u);
  v0 = v0 + v5 + m[1]; v15 = rotr(v15 ^ v0, 16u); v10 = v10 + v15; v5 = rotr(v5 ^ v10, 12u);
  v0 = v0 + v5 + m[11]; v15 = rotr(v15 ^ v0, 8u);  v10 = v10 + v15; v5 = rotr(v5 ^ v10, 7u);
  v1 = v1 + v6 + m[12]; v12 = rotr(v12 ^ v1, 16u); v11 = v11 + v12; v6 = rotr(v6 ^ v11, 12u);
  v1 = v1 + v6 + m[5]; v12 = rotr(v12 ^ v1, 8u);  v11 = v11 + v12; v6 = rotr(v6 ^ v11, 7u);
  v2 = v2 + v7 + m[9]; v13 = rotr(v13 ^ v2, 16u); v8 = v8 + v13; v7 = rotr(v7 ^ v8, 12u);
  v2 = v2 + v7 + m[14]; v13 = rotr(v13 ^ v2, 8u);  v8 = v8 + v13; v7 = rotr(v7 ^ v8, 7u);
  v3 = v3 + v4 + m[15]; v14 = rotr(v14 ^ v3, 16u); v9 = v9 + v14; v4 = rotr(v4 ^ v9, 12u);
  v3 = v3 + v4 + m[8]; v14 = rotr(v14 ^ v3, 8u);  v9 = v9 + v14; v4 = rotr(v4 ^ v9, 7u);
  // round 2
  v0 = v0 + v4 + m[3]; v12 = rotr(v12 ^ v0, 16u); v8 = v8 + v12; v4 = rotr(v4 ^ v8, 12u);
  v0 = v0 + v4 + m[4]; v12 = rotr(v12 ^ v0, 8u);  v8 = v8 + v12; v4 = rotr(v4 ^ v8, 7u);
  v1 = v1 + v5 + m[10]; v13 = rotr(v13 ^ v1, 16u); v9 = v9 + v13; v5 = rotr(v5 ^ v9, 12u);
  v1 = v1 + v5 + m[12]; v13 = rotr(v13 ^ v1, 8u);  v9 = v9 + v13; v5 = rotr(v5 ^ v9, 7u);
  v2 = v2 + v6 + m[13]; v14 = rotr(v14 ^ v2, 16u); v10 = v10 + v14; v6 = rotr(v6 ^ v10, 12u);
  v2 = v2 + v6 + m[2]; v14 = rotr(v14 ^ v2, 8u);  v10 = v10 + v14; v6 = rotr(v6 ^ v10, 7u);
  v3 = v3 + v7 + m[7]; v15 = rotr(v15 ^ v3, 16u); v11 = v11 + v15; v7 = rotr(v7 ^ v11, 12u);
  v3 = v3 + v7 + m[14]; v15 = rotr(v15 ^ v3, 8u);  v11 = v11 + v15; v7 = rotr(v7 ^ v11, 7u);
  v0 = v0 + v5 + m[6]; v15 = rotr(v15 ^ v0, 16u); v10 = v10 + v15; v5 = rotr(v5 ^ v10, 12u);
  v0 = v0 + v5 + m[5]; v15 = rotr(v15 ^ v0, 8u);  v10 = v10 + v15; v5 = rotr(v5 ^ v10, 7u);
  v1 = v1 + v6 + m[9]; v12 = rotr(v12 ^ v1, 16u); v11 = v11 + v12; v6 = rotr(v6 ^ v11, 12u);
  v1 = v1 + v6 + m[0]; v12 = rotr(v12 ^ v1, 8u);  v11 = v11 + v12; v6 = rotr(v6 ^ v11, 7u);
  v2 = v2 + v7 + m[11]; v13 = rotr(v13 ^ v2, 16u); v8 = v8 + v13; v7 = rotr(v7 ^ v8, 12u);
  v2 = v2 + v7 + m[15]; v13 = rotr(v13 ^ v2, 8u);  v8 = v8 + v13; v7 = rotr(v7 ^ v8, 7u);
  v3 = v3 + v4 + m[8]; v14 = rotr(v14 ^ v3, 16u); v9 = v9 + v14; v4 = rotr(v4 ^ v9, 12u);
  v3 = v3 + v4 + m[1]; v14 = rotr(v14 ^ v3, 8u);  v9 = v9 + v14; v4 = rotr(v4 ^ v9, 7u);
  // round 3
  v0 = v0 + v4 + m[10]; v12 = rotr(v12 ^ v0, 16u); v8 = v8 + v12; v4 = rotr(v4 ^ v8, 12u);
  v0 = v0 + v4 + m[7]; v12 = rotr(v12 ^ v0, 8u);  v8 = v8 + v12; v4 = rotr(v4 ^ v8, 7u);
  v1 = v1 + v5 + m[12]; v13 = rotr(v13 ^ v1, 16u); v9 = v9 + v13; v5 = rotr(v5 ^ v9, 12u);
  v1 = v1 + v5 + m[9]; v13 = rotr(v13 ^ v1, 8u);  v9 = v9 + v13; v5 = rotr(v5 ^ v9, 7u);
  v2 = v2 + v6 + m[14]; v14 = rotr(v14 ^ v2, 16u); v10 = v10 + v14; v6 = rotr(v6 ^ v10, 12u);
  v2 = v2 + v6 + m[3]; v14 = rotr(v14 ^ v2, 8u);  v10 = v10 + v14; v6 = rotr(v6 ^ v10, 7u);
  v3 = v3 + v7 + m[13]; v15 = rotr(v15 ^ v3, 16u); v11 = v11 + v15; v7 = rotr(v7 ^ v11, 12u);
  v3 = v3 + v7 + m[15]; v15 = rotr(v15 ^ v3, 8u);  v11 = v11 + v15; v7 = rotr(v7 ^ v11, 7u);
  v0 = v0 + v5 + m[4]; v15 = rotr(v15 ^ v0, 16u); v10 = v10 + v15; v5 = rotr(v5 ^ v10, 12u);
  v0 = v0 + v5 + m[0]; v15 = rotr(v15 ^ v0, 8u);  v10 = v10 + v15; v5 = rotr(v5 ^ v10, 7u);
  v1 = v1 + v6 + m[11]; v12 = rotr(v12 ^ v1, 16u); v11 = v11 + v12; v6 = rotr(v6 ^ v11, 12u);
  v1 = v1 + v6 + m[2]; v12 = rotr(v12 ^ v1, 8u);  v11 = v11 + v12; v6 = rotr(v6 ^ v11, 7u);
  v2 = v2 + v7 + m[5]; v13 = rotr(v13 ^ v2, 16u); v8 = v8 + v13; v7 = rotr(v7 ^ v8, 12u);
  v2 = v2 + v7 + m[8]; v13 = rotr(v13 ^ v2, 8u);  v8 = v8 + v13; v7 = rotr(v7 ^ v8, 7u);
  v3 = v3 + v4 + m[1]; v14 = rotr(v14 ^ v3, 16u); v9 = v9 + v14; v4 = rotr(v4 ^ v9, 12u);
  v3 = v3 + v4 + m[6]; v14 = rotr(v14 ^ v3, 8u);  v9 = v9 + v14; v4 = rotr(v4 ^ v9, 7u);
  // round 4
  v0 = v0 + v4 + m[12]; v12 = rotr(v12 ^ v0, 16u); v8 = v8 + v12; v4 = rotr(v4 ^ v8, 12u);
  v0 = v0 + v4 + m[13]; v12 = rotr(v12 ^ v0, 8u);  v8 = v8 + v12; v4 = rotr(v4 ^ v8, 7u);
  v1 = v1 + v5 + m[9]; v13 = rotr(v13 ^ v1, 16u); v9 = v9 + v13; v5 = rotr(v5 ^ v9, 12u);
  v1 = v1 + v5 + m[11]; v13 = rotr(v13 ^ v1, 8u);  v9 = v9 + v13; v5 = rotr(v5 ^ v9, 7u);
  v2 = v2 + v6 + m[15]; v14 = rotr(v14 ^ v2, 16u); v10 = v10 + v14; v6 = rotr(v6 ^ v10, 12u);
  v2 = v2 + v6 + m[10]; v14 = rotr(v14 ^ v2, 8u);  v10 = v10 + v14; v6 = rotr(v6 ^ v10, 7u);
  v3 = v3 + v7 + m[14]; v15 = rotr(v15 ^ v3, 16u); v11 = v11 + v15; v7 = rotr(v7 ^ v11, 12u);
  v3 = v3 + v7 + m[8]; v15 = rotr(v15 ^ v3, 8u);  v11 = v11 + v15; v7 = rotr(v7 ^ v11, 7u);
  v0 = v0 + v5 + m[7]; v15 = rotr(v15 ^ v0, 16u); v10 = v10 + v15; v5 = rotr(v5 ^ v10, 12u);
  v0 = v0 + v5 + m[2]; v15 = rotr(v15 ^ v0, 8u);  v10 = v10 + v15; v5 = rotr(v5 ^ v10, 7u);
  v1 = v1 + v6 + m[5]; v12 = rotr(v12 ^ v1, 16u); v11 = v11 + v12; v6 = rotr(v6 ^ v11, 12u);
  v1 = v1 + v6 + m[3]; v12 = rotr(v12 ^ v1, 8u);  v11 = v11 + v12; v6 = rotr(v6 ^ v11, 7u);
  v2 = v2 + v7 + m[0]; v13 = rotr(v13 ^ v2, 16u); v8 = v8 + v13; v7 = rotr(v7 ^ v8, 12u);
  v2 = v2 + v7 + m[1]; v13 = rotr(v13 ^ v2, 8u);  v8 = v8 + v13; v7 = rotr(v7 ^ v8, 7u);
  v3 = v3 + v4 + m[6]; v14 = rotr(v14 ^ v3, 16u); v9 = v9 + v14; v4 = rotr(v4 ^ v9, 12u);
  v3 = v3 + v4 + m[4]; v14 = rotr(v14 ^ v3, 8u);  v9 = v9 + v14; v4 = rotr(v4 ^ v9, 7u);
  // round 5
  v0 = v0 + v4 + m[9]; v12 = rotr(v12 ^ v0, 16u); v8 = v8 + v12; v4 = rotr(v4 ^ v8, 12u);
  v0 = v0 + v4 + m[14]; v12 = rotr(v12 ^ v0, 8u);  v8 = v8 + v12; v4 = rotr(v4 ^ v8, 7u);
  v1 = v1 + v5 + m[11]; v13 = rotr(v13 ^ v1, 16u); v9 = v9 + v13; v5 = rotr(v5 ^ v9, 12u);
  v1 = v1 + v5 + m[5]; v13 = rotr(v13 ^ v1, 8u);  v9 = v9 + v13; v5 = rotr(v5 ^ v9, 7u);
  v2 = v2 + v6 + m[8]; v14 = rotr(v14 ^ v2, 16u); v10 = v10 + v14; v6 = rotr(v6 ^ v10, 12u);
  v2 = v2 + v6 + m[12]; v14 = rotr(v14 ^ v2, 8u);  v10 = v10 + v14; v6 = rotr(v6 ^ v10, 7u);
  v3 = v3 + v7 + m[15]; v15 = rotr(v15 ^ v3, 16u); v11 = v11 + v15; v7 = rotr(v7 ^ v11, 12u);
  v3 = v3 + v7 + m[1]; v15 = rotr(v15 ^ v3, 8u);  v11 = v11 + v15; v7 = rotr(v7 ^ v11, 7u);
  v0 = v0 + v5 + m[13]; v15 = rotr(v15 ^ v0, 16u); v10 = v10 + v15; v5 = rotr(v5 ^ v10, 12u);
  v0 = v0 + v5 + m[3]; v15 = rotr(v15 ^ v0, 8u);  v10 = v10 + v15; v5 = rotr(v5 ^ v10, 7u);
  v1 = v1 + v6 + m[0]; v12 = rotr(v12 ^ v1, 16u); v11 = v11 + v12; v6 = rotr(v6 ^ v11, 12u);
  v1 = v1 + v6 + m[10]; v12 = rotr(v12 ^ v1, 8u);  v11 = v11 + v12; v6 = rotr(v6 ^ v11, 7u);
  v2 = v2 + v7 + m[2]; v13 = rotr(v13 ^ v2, 16u); v8 = v8 + v13; v7 = rotr(v7 ^ v8, 12u);
  v2 = v2 + v7 + m[6]; v13 = rotr(v13 ^ v2, 8u);  v8 = v8 + v13; v7 = rotr(v7 ^ v8, 7u);
  v3 = v3 + v4 + m[4]; v14 = rotr(v14 ^ v3, 16u); v9 = v9 + v14; v4 = rotr(v4 ^ v9, 12u);
  v3 = v3 + v4 + m[7]; v14 = rotr(v14 ^ v3, 8u);  v9 = v9 + v14; v4 = rotr(v4 ^ v9, 7u);
  // round 6
  v0 = v0 + v4 + m[11]; v12 = rotr(v12 ^ v0, 16u); v8 = v8 + v12; v4 = rotr(v4 ^ v8, 12u);
  v0 = v0 + v4 + m[15]; v12 = rotr(v12 ^ v0, 8u);  v8 = v8 + v12; v4 = rotr(v4 ^ v8, 7u);
  v1 = v1 + v5 + m[5]; v13 = rotr(v13 ^ v1, 16u); v9 = v9 + v13; v5 = rotr(v5 ^ v9, 12u);
  v1 = v1 + v5 + m[0]; v13 = rotr(v13 ^ v1, 8u);  v9 = v9 + v13; v5 = rotr(v5 ^ v9, 7u);
  v2 = v2 + v6 + m[1]; v14 = rotr(v14 ^ v2, 16u); v10 = v10 + v14; v6 = rotr(v6 ^ v10, 12u);
  v2 = v2 + v6 + m[9]; v14 = rotr(v14 ^ v2, 8u);  v10 = v10 + v14; v6 = rotr(v6 ^ v10, 7u);
  v3 = v3 + v7 + m[8]; v15 = rotr(v15 ^ v3, 16u); v11 = v11 + v15; v7 = rotr(v7 ^ v11, 12u);
  v3 = v3 + v7 + m[6]; v15 = rotr(v15 ^ v3, 8u);  v11 = v11 + v15; v7 = rotr(v7 ^ v11, 7u);
  v0 = v0 + v5 + m[14]; v15 = rotr(v15 ^ v0, 16u); v10 = v10 + v15; v5 = rotr(v5 ^ v10, 12u);
  v0 = v0 + v5 + m[10]; v15 = rotr(v15 ^ v0, 8u);  v10 = v10 + v15; v5 = rotr(v5 ^ v10, 7u);
  v1 = v1 + v6 + m[2]; v12 = rotr(v12 ^ v1, 16u); v11 = v11 + v12; v6 = rotr(v6 ^ v11, 12u);
  v1 = v1 + v6 + m[12]; v12 = rotr(v12 ^ v1, 8u);  v11 = v11 + v12; v6 = rotr(v6 ^ v11, 7u);
  v2 = v2 + v7 + m[3]; v13 = rotr(v13 ^ v2, 16u); v8 = v8 + v13; v7 = rotr(v7 ^ v8, 12u);
  v2 = v2 + v7 + m[4]; v13 = rotr(v13 ^ v2, 8u);  v8 = v8 + v13; v7 = rotr(v7 ^ v8, 7u);
  v3 = v3 + v4 + m[7]; v14 = rotr(v14 ^ v3, 16u); v9 = v9 + v14; v4 = rotr(v4 ^ v9, 12u);
  v3 = v3 + v4 + m[13]; v14 = rotr(v14 ^ v3, 8u);  v9 = v9 + v14; v4 = rotr(v4 ^ v9, 7u);
    return array<u32,8>(v0 ^ v8, v1 ^ v9, v2 ^ v10, v3 ^ v11, v4 ^ v12, v5 ^ v13, v6 ^ v14, v7 ^ v15);
}

fn nonce_for(gid: u32) -> vec2<u32> {
    let lo = P.base_lo + gid;          // gid < n_nonces <= 2^18, so at most one carry
    var carry = 0u;
    if (lo < P.base_lo) { carry = 1u; }
    let hi = P.base_hi + carry;
    return vec2<u32>(lo, hi);
}

fn first_compress(gid: u32) -> array<u32,8> {
    var m: array<u32,16>;
    m[0] = P.midstate[0]; m[1] = P.midstate[1]; m[2] = P.midstate[2]; m[3] = P.midstate[3];
    m[4] = P.midstate[4]; m[5] = P.midstate[5]; m[6] = P.midstate[6]; m[7] = P.midstate[7];
    let n = nonce_for(gid);
    m[8] = n.x; m[9] = n.y;
    m[10] = 0u; m[11] = 0u; m[12] = 0u; m[13] = 0u; m[14] = 0u; m[15] = 0u;
    return compress(m, 40u);
}

fn iterate(h: array<u32,8>) -> array<u32,8> {
    var m: array<u32,16>;
    m[0] = h[0]; m[1] = h[1]; m[2] = h[2]; m[3] = h[3];
    m[4] = h[4]; m[5] = h[5]; m[6] = h[6]; m[7] = h[7];
    m[8] = 0u; m[9] = 0u; m[10] = 0u; m[11] = 0u; m[12] = 0u; m[13] = 0u; m[14] = 0u; m[15] = 0u;
    return compress(m, 32u);
}

// final_hash[u8;32] < ref ?  (byte 0 most significant), unrolled to avoid
// dynamic indexing into value arrays.
fn lt8(h: array<u32,8>, r: array<u32,8>) -> bool {
    var k: u32;
    k = bswap(h[0]); if (k < r[0]) { return true; } if (k > r[0]) { return false; }
    k = bswap(h[1]); if (k < r[1]) { return true; } if (k > r[1]) { return false; }
    k = bswap(h[2]); if (k < r[2]) { return true; } if (k > r[2]) { return false; }
    k = bswap(h[3]); if (k < r[3]) { return true; } if (k > r[3]) { return false; }
    k = bswap(h[4]); if (k < r[4]) { return true; } if (k > r[4]) { return false; }
    k = bswap(h[5]); if (k < r[5]) { return true; } if (k > r[5]) { return false; }
    k = bswap(h[6]); if (k < r[6]) { return true; } if (k > r[6]) { return false; }
    k = bswap(h[7]); if (k < r[7]) { return true; } if (k > r[7]) { return false; }
    return false;
}

fn load_state(gid: u32) -> array<u32,8> {
    let b = gid * 8u;
    var h: array<u32,8>;
    h[0] = state[b + 0u]; h[1] = state[b + 1u]; h[2] = state[b + 2u]; h[3] = state[b + 3u];
    h[4] = state[b + 4u]; h[5] = state[b + 5u]; h[6] = state[b + 6u]; h[7] = state[b + 7u];
    return h;
}

fn store_state(gid: u32, h: array<u32,8>) {
    let b = gid * 8u;
    state[b + 0u] = h[0]; state[b + 1u] = h[1]; state[b + 2u] = h[2]; state[b + 3u] = h[3];
    state[b + 4u] = h[4]; state[b + 5u] = h[5]; state[b + 6u] = h[6]; state[b + 7u] = h[7];
}

@compute @workgroup_size(64)
fn k_init(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= P.n_nonces) { return; }
    store_state(gid, first_compress(gid));
}

@compute @workgroup_size(64)
fn k_step(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= P.n_nonces) { return; }
    var h = load_state(gid);
    for (var i = 0u; i < P.iters; i = i + 1u) {
        h = iterate(h);
    }
    store_state(gid, h);
}

@compute @workgroup_size(64)
fn k_test(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= P.n_nonces) { return; }
    let h = load_state(gid);
    var kind = 0xFFFFFFFFu;
    if (lt8(h, P.tgt)) {
        kind = 0u;
    } else if (P.has_pool != 0u && lt8(h, P.pool)) {
        kind = 1u;
    }
    if (kind != 0xFFFFFFFFu) {
        let idx = atomicAdd(&out.count, 1u);
        if (idx < out.cap) {
            let n = nonce_for(gid);
            out.nonce_lo[idx] = n.x;
            out.nonce_hi[idx] = n.y;
            out.kind[idx] = kind;
        }
    }
}
