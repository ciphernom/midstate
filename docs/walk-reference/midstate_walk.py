"""Independent reference implementation of Midstate Walk v1, written from the
specification in walk.rs's module docs (not from the Rust code).

Slow (pure Python), but small enough to audit line by line. Used to cross-check
the Rust implementation's golden vectors.

    python3 midstate_walk.py [--fast] <mining_hash_hex> <nonce>
"""
import sys
from blake3 import blake3

M64 = (1 << 64) - 1

# AES S-box generated from the definition (multiplicative inverse in GF(2^8)
# followed by the affine map), so it is independent of any table in walk.rs.
def _gmul(a, b):
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= 0x1B
        b >>= 1
    return p

def _sbox():
    inv = [0] * 256
    for x in range(1, 256):
        for y in range(1, 256):
            if _gmul(x, y) == 1:
                inv[x] = y
                break
    box = []
    for x in range(256):
        b = inv[x]
        s = b
        for i in range(1, 5):
            s ^= ((b << i) | (b >> (8 - i))) & 0xFF
        box.append(s ^ 0x63)
    return box

SBOX = _sbox()

def aesenc(state, key):
    """One AES round as x86 AESENC: MixColumns(ShiftRows(SubBytes(s))) ^ key.
    Byte i of the 16-byte state is row i % 4, column i // 4."""
    s = [SBOX[b] for b in state]
    # ShiftRows: row r rotates left by r: new[r][c] = old[r][(c + r) % 4]
    sr = [0] * 16
    for c in range(4):
        for r in range(4):
            sr[4 * c + r] = s[4 * ((c + r) % 4) + r]
    out = [0] * 16
    for c in range(4):
        a0, a1, a2, a3 = sr[4 * c:4 * c + 4]
        out[4 * c + 0] = _gmul(a0, 2) ^ _gmul(a1, 3) ^ a2 ^ a3
        out[4 * c + 1] = a0 ^ _gmul(a1, 2) ^ _gmul(a2, 3) ^ a3
        out[4 * c + 2] = a0 ^ a1 ^ _gmul(a2, 2) ^ _gmul(a3, 3)
        out[4 * c + 3] = _gmul(a0, 3) ^ a1 ^ a2 ^ _gmul(a3, 2)
    return bytes(o ^ k for o, k in zip(out, key))

def le64(b):
    return int.from_bytes(b[:8], "little")

def program(z):
    p = list(range(8))
    for i in range(7, 0, -1):
        n = i + 1
        j = z % n
        p[i], p[j] = p[j], p[i]
        z //= n
    return p

def regs_bytes(r):
    return b"".join(w.to_bytes(8, "little") for w in r)

def walk(mining_hash, nonce, fast=False):
    LINES, STEPS, REMIX = (256, 1 << 10, 1 << 8) if fast else (4096, 1 << 18, 1 << 12)
    nb = nonce.to_bytes(8, "little")
    seed = blake3(mining_hash + nb, derive_key_context="midstate walk v1 init").digest(length=200)
    r = [le64(seed[8 * k:]) for k in range(8)]
    lanes = [seed[64 + 16 * q:80 + 16 * q] for q in range(4)]
    keys = [seed[128 + 16 * q:144 + 16 * q] for q in range(4)]
    prog = program(le64(seed[192:200]))
    pad = []
    for n in range(LINES):
        line = []
        for q in range(4):
            lanes[q] = aesenc(lanes[q], keys[q])
            line += [le64(lanes[q][:8]), le64(lanes[q][8:])]
        pad.append(line)

    # AES of a (v, k) pair under a zero key, memoised: it is the slow part.
    zero = bytes(16)
    cache = {}
    def aes_pair(v, k):
        key = (v, k)
        if key not in cache:
            o = aesenc(v.to_bytes(8, "little") + k.to_bytes(8, "little"), zero)
            cache[key] = (le64(o[:8]), le64(o[8:]))
        return cache[key]

    c = 0
    for i in range(STEPS):
        s = i % 8
        x = r[s] ^ c
        a = x % LINES
        t = (s + 1 + (x >> 12) % 4) % 8
        j = (x >> 14) % 8
        e = (x >> 17) % LINES
        b = (x >> 32) % LINES
        f = (x >> 44) % LINES
        m0 = pad[a][j]
        m1 = pad[b][j ^ 4]
        m2 = pad[e][j ^ 2]
        m3 = pad[f][j ^ 6]
        k = r[t]
        v = x ^ m0 ^ m2
        h = 0
        op = prog[s]
        if op == 0:
            y = (v * (k | 1)) & M64
        elif op == 1:
            y = v ^ ((v * k) >> 64)
        elif op == 2:
            y = (v + k // ((v & 0xFFFFFFFF) | 0x80000001)) & M64
        elif op == 3:
            y, h = aes_pair(v, k)
        elif op == 4:
            n = k % 64
            y = ((v >> n) | (v << (64 - n))) & M64 if n else v
        elif op == 5:
            y = (v + k) & M64
        elif op == 6:
            y = (v - k) & M64
        else:
            y = v ^ k
        out = y ^ m1 ^ m3
        r[s] = out
        r[t] ^= h
        c = out
        pad[a][j] = m0 ^ y
        pad[b][j ^ 4] = (m1 + x) & M64
        pad[e][j ^ 2] = (m2 + y) & M64
        pad[f][j ^ 6] = m3 ^ x
        if (i + 1) % REMIX == 0:
            o = blake3(regs_bytes(r), derive_key_context="midstate walk v1 remix").digest(length=72)
            r = [le64(o[8 * q:]) for q in range(8)]
            prog = program(le64(o[64:72]))

    return blake3(mining_hash + nb + regs_bytes(r), derive_key_context="midstate walk v1 final").digest()

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--fast"]
    fast = "--fast" in sys.argv
    # FIPS-197 Appendix B round 1 check of the AES round.
    assert aesenc(bytes.fromhex("193de3bea0f4e22b9ac68d2ae9f84808"),
                  bytes.fromhex("a0fafe1788542cb123a339392a6c7605")).hex() == "a49c7ff2689f352b6b5bea43026a5049"
    print(walk(bytes.fromhex(args[0]), int(args[1]), fast).hex())
