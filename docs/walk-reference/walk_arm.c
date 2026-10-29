/* Midstate Walk v1 in C, for checking the ARMv8 AES path under emulation.
 *
 * Mirrors walk.rs's aarch64 code: one AES round is AESMC(AESE(s, 0)) ^ k,
 * and the (v, k) pair is packed with vcombine_u64 / unpacked with
 * vgetq_lane_u64 exactly as the Rust intrinsics do. Built with
 * -march=armv8.2-a+crypto and run under qemu-aarch64.
 *
 *   walk_arm [--fast] <mining_hash_hex> <nonce>
 */
#include <arm_neon.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "blake3.h"

static size_t LINES = 4096, STEPS = 1u << 18, REMIX = 1u << 12;

static uint64_t le64(const uint8_t *b) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | b[i];
    return v;
}

static void put64(uint8_t *b, uint64_t v) {
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (8 * i));
}

static void aes_round(uint8_t out[16], const uint8_t s[16], const uint8_t k[16]) {
    uint8x16_t o = veorq_u8(vaesmcq_u8(vaeseq_u8(vld1q_u8(s), vdupq_n_u8(0))), vld1q_u8(k));
    vst1q_u8(out, o);
}

static void aes_pair(uint64_t lo, uint64_t hi, uint64_t *olo, uint64_t *ohi) {
    uint8x16_t s = vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(lo), vcreate_u64(hi)));
    uint64x2_t o = vreinterpretq_u64_u8(vaesmcq_u8(vaeseq_u8(s, vdupq_n_u8(0))));
    *olo = vgetq_lane_u64(o, 0);
    *ohi = vgetq_lane_u64(o, 1);
}

static void program(uint64_t z, uint8_t p[8]) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)i;
    for (int i = 7; i >= 1; i--) {
        uint64_t n = (uint64_t)i + 1;
        uint8_t j = (uint8_t)(z % n), tmp = p[i];
        p[i] = p[j];
        p[j] = tmp;
        z /= n;
    }
}

static void regs_bytes(const uint64_t r[8], uint8_t out[64]) {
    for (int k = 0; k < 8; k++) put64(out + 8 * k, r[k]);
}

static void walk(const uint8_t m[32], uint64_t nonce, uint8_t out[32]) {
    uint8_t nb[8], seed[200], rb[64], o[72];
    put64(nb, nonce);
    blake3_hasher h;
    blake3_hasher_init_derive_key(&h, "midstate walk v1 init");
    blake3_hasher_update(&h, m, 32);
    blake3_hasher_update(&h, nb, 8);
    blake3_hasher_finalize(&h, seed, sizeof seed);

    uint64_t r[8];
    for (int k = 0; k < 8; k++) r[k] = le64(seed + 8 * k);
    uint8_t prog[8];
    program(le64(seed + 192), prog);

    uint64_t (*pad)[8] = malloc(LINES * sizeof *pad);
    uint8_t lanes[4][16], keys[4][16];
    for (int q = 0; q < 4; q++) {
        memcpy(lanes[q], seed + 64 + 16 * q, 16);
        memcpy(keys[q], seed + 128 + 16 * q, 16);
    }
    for (size_t n = 0; n < LINES; n++)
        for (int q = 0; q < 4; q++) {
            aes_round(lanes[q], lanes[q], keys[q]);
            pad[n][2 * q] = le64(lanes[q]);
            pad[n][2 * q + 1] = le64(lanes[q] + 8);
        }

    uint64_t c = 0;
    for (size_t i = 0; i < STEPS; i++) {
        size_t s = i & 7;
        uint64_t x = r[s] ^ c;
        size_t a = x & (LINES - 1);
        size_t t = (s + 1 + ((x >> 12) & 3)) & 7;
        size_t j = (x >> 14) & 7;
        size_t e = (x >> 17) & (LINES - 1);
        size_t b = (x >> 32) & (LINES - 1);
        size_t f = (x >> 44) & (LINES - 1);
        uint64_t m0 = pad[a][j], m1 = pad[b][j ^ 4], m2 = pad[e][j ^ 2], m3 = pad[f][j ^ 6];
        uint64_t k = r[t], v = x ^ m0 ^ m2, y, hi = 0;
        switch (prog[s]) {
        case 0: y = v * (k | 1); break;
        case 1: y = v ^ (uint64_t)(((unsigned __int128)v * k) >> 64); break;
        case 2: y = v + k / (((uint64_t)(uint32_t)v) | 0x80000001u); break;
        case 3: aes_pair(v, k, &y, &hi); break;
        case 4: { unsigned n = k & 63; y = n ? (v >> n) | (v << (64 - n)) : v; } break;
        case 5: y = v + k; break;
        case 6: y = v - k; break;
        default: y = v ^ k; break;
        }
        uint64_t out = y ^ m1 ^ m3;
        r[s] = out;
        r[t] ^= hi;
        c = out;
        pad[a][j] = m0 ^ y;
        pad[b][j ^ 4] = m1 + x;
        pad[e][j ^ 2] = m2 + y;
        pad[f][j ^ 6] = m3 ^ x;
        if ((i + 1) % REMIX == 0) {
            regs_bytes(r, rb);
            blake3_hasher_init_derive_key(&h, "midstate walk v1 remix");
            blake3_hasher_update(&h, rb, 64);
            blake3_hasher_finalize(&h, o, sizeof o);
            for (int q = 0; q < 8; q++) r[q] = le64(o + 8 * q);
            program(le64(o + 64), prog);
        }
    }
    free(pad);
    regs_bytes(r, rb);
    blake3_hasher_init_derive_key(&h, "midstate walk v1 final");
    blake3_hasher_update(&h, m, 32);
    blake3_hasher_update(&h, nb, 8);
    blake3_hasher_update(&h, rb, 64);
    blake3_hasher_finalize(&h, out, 32);
}

int main(int argc, char **argv) {
    int ai = 1;
    if (argc > 1 && strcmp(argv[1], "--fast") == 0) {
        LINES = 256; STEPS = 1u << 10; REMIX = 1u << 8; ai = 2;
    }
    if (argc - ai != 2) { fprintf(stderr, "usage: walk_arm [--fast] <hash hex> <nonce>\n"); return 2; }
    uint8_t m[32], out[32];
    for (int i = 0; i < 32; i++) sscanf(argv[ai] + 2 * i, "%2hhx", &m[i]);
    uint64_t nonce = strtoull(argv[ai + 1], NULL, 10);

    /* FIPS-197 Appendix B round 1 through the ARM path. */
    static const uint8_t fin[16] = {0x19,0x3d,0xe3,0xbe,0xa0,0xf4,0xe2,0x2b,0x9a,0xc6,0x8d,0x2a,0xe9,0xf8,0x48,0x08};
    static const uint8_t fk[16]  = {0xa0,0xfa,0xfe,0x17,0x88,0x54,0x2c,0xb1,0x23,0xa3,0x39,0x39,0x2a,0x6c,0x76,0x05};
    static const uint8_t fout[16]= {0xa4,0x9c,0x7f,0xf2,0x68,0x9f,0x35,0x2b,0x6b,0x5b,0xea,0x43,0x02,0x6a,0x50,0x49};
    uint8_t got[16];
    aes_round(got, fin, fk);
    if (memcmp(got, fout, 16) != 0) { fprintf(stderr, "FIPS-197 round check FAILED\n"); return 1; }

    walk(m, nonce, out);
    for (int i = 0; i < 32; i++) printf("%02x", out[i]);
    printf("\n");
    return 0;
}
