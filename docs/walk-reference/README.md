# Midstate Walk reference implementations

Two implementations of Midstate Walk v1. Both were written from the
definition in `src/core/walk.rs`'s module docs, not from its code, and both
reproduce the golden vectors pinned in that file's tests.

| File | What it checks |
|---|---|
| `midstate_walk.py` | The whole definition, readably. The AES S-box is derived from GF(2^8) inversion and the affine map, independent of any table. |
| `walk_arm.c` | The ARMv8 path: one AES round as `AESMC(AESE(s, 0)) ^ k`, with the 64-bit lanes packed as the Rust intrinsics pack them. |

```sh
pip install blake3
python3 midstate_walk.py [--fast] <mining_hash_hex> <nonce>

# Cross-compiled for a Cortex-A76 (Raspberry Pi 5), run natively there or
# under qemu-user. Needs the BLAKE3 C sources (github.com/BLAKE3-team/BLAKE3, c/).
B3=path/to/BLAKE3/c
aarch64-linux-gnu-gcc -O2 -mcpu=cortex-a76+crypto -static -I"$B3" -DBLAKE3_USE_NEON=1 \
    walk_arm.c "$B3/blake3.c" "$B3/blake3_dispatch.c" "$B3/blake3_portable.c" "$B3/blake3_neon.c" \
    -o walk_arm
qemu-aarch64 -cpu cortex-a76 ./walk_arm [--fast] <mining_hash_hex> <nonce>
```

`--fast` selects the test-build parameters (`fast-mining`). The golden
vectors:

| mining hash | nonce | final hash (mainnet) | final hash (`--fast`) |
|---|---|---|---|
| `00…00` | 0 | `a934fced2d9a53fb09d6abe089f3d2f5bba9a421b41ee5726233a525cc7438e7` | `592cd5f394f7dc705343585a7cb48c58ce8da5f646dadb81af440e0d7a470e40` |
| `ff…ff` | 18446744073709551615 | `65d237f595f07bee156c635ad45f721987ce38d79ea14c89e40c5555613d0768` | `f1488d17741b4eaaa909374c90ea70adfe89ceaa16d7ca8605c0da25f031e9f3` |
| `374bf98b3b8f7f41a9e291154fc582a65bb3e14180cb9a07fef08e55f3006c50` | 42 | `5f2da9d320de391aef28bcaab63e3d1a25402b4541cdff8680059a87c7d515b7` | `8f9d7f1d42826a15f81dbee9fd7e25ca3edee11776dfb48e078feebfeab2ab13` |
