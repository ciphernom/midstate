//! Midstate Walk on any GPU wgpu reaches (Vulkan, Metal, DX12): the Raspberry
//! Pi 5's VideoCore VII through Mesa's v3dv first of all, and AMD, Intel and
//! Apple GPUs alike. For now it is measured, not mined with:
//! `midstate bench-pow --gpu` (`main.rs`) runs it.
//!
//! # The split
//!
//! - The CPU seeds each walk (BLAKE3) and grows its 256 KiB pad (AES), with
//!   [`walk::walk_start`]: microseconds, with the CPU's AES instructions.
//! - The GPU runs the 2^18 steps, BLAKE3 re-mixes included, in launches of a
//!   few thousand steps (a long launch trips a GPU's hang timer: the Pi's
//!   kernel driver resets the GPU after half a second).
//! - The CPU hashes the registers the steps leave, with
//!   [`walk::walk_finish`].
//!
//! # The kernel
//!
//! One invocation runs one walk. The kernel is written out by [`shader`] for
//! each variant, so that every step of a round of eight names its register
//! directly (no indexed register arrays, which a GPU keeps in slow memory):
//!
//! - Words are `vec2<u32>` (low, high). WGSL has no 64-bit integers, and the
//!   Pi's GPU none either.
//! - Multiplies go through 16-bit limbs, whose products fit the 24-bit
//!   multiplier the Pi's GPU has; 64×64→128 high is sixteen of them.
//! - The division's divisor has its top bit set (`lo32(v) | 0x8000_0001`), so
//!   64÷32 is two base-2^16 digits of Knuth's algorithm D, exact after at
//!   most two corrections each, rather than 32 shift-subtracts.
//! - The AES round is four 1 KiB tables (`TE` rotated): sixteen lookups.
//! - The program's draw needs only `z mod 8!`, since Fisher–Yates reads its
//!   digits in radices 8, 7, …, 2: 32-bit arithmetic throughout.
//! - Pads are `array<vec2<u32>>`: one 8-byte load per word.
//!
//! # Grouping walks by their programs
//!
//! A GPU runs its lanes in lockstep, 16 on the Pi's, 32 on NVIDIA's. Each
//! walk's program puts its eight operations in its own order, so at almost
//! every step some lane wants the AES round, the division or the high
//! multiply, and every lane waits through it. But which lane runs which walk
//! is free: the kernel reads its walk's number from a table. At every re-mix,
//! when programs change, [`group_order`] sorts the walks by where their AES
//! round falls, then the division, then the high multiply, then the multiply,
//! so neighbouring lanes take the costly paths at the same steps. With a few
//! hundred walks in flight, a group of 16 has its AES at one step of the eight
//! instead of at nearly all of them. The hashes do not change; the benchmark
//! measures whether the time does.
//!
//! # Probes
//!
//! Two variants change the walk on purpose, to find what limits a GPU:
//! every operation an XOR (only the memory traffic is left), and a pad of
//! eight lines (it stays in the GPU's cache, leaving the arithmetic). Their
//! hashes are not the walk's.

#![cfg(not(target_arch = "wasm32"))]

use super::walk::{self, WalkState, WALK_LINES, WALK_REMIX, WALK_STEPS};
use std::time::{Duration, Instant};

/// `vec2<u32>` words per pad.
const PAD_WORDS: u64 = (WALK_LINES * 8) as u64;
/// `vec2<u32>` words per walk's state: r0..r7, c, (program, 0).
const STATE_WORDS: u64 = 10;
/// The longest a launch should take: well inside the Pi's half-second
/// GPU hang timer, short enough to stop promptly.
pub const LAUNCH_TARGET: Duration = Duration::from_millis(60);

/// What the kernel computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Variant {
    /// Invocations per workgroup.
    pub workgroup: u32,
    /// The walk's own operations, or (a probe) an XOR for each.
    pub real_ops: bool,
    /// Lines of pad used: the walk's own, or (a probe) eight, which all walks
    /// share so that they stay in the GPU's cache.
    pub lines: u32,
    /// Walks regrouped at every re-mix so that lanes running together have
    /// their costly operations at the same steps ([`group_order`]).
    pub grouped: bool,
    /// Steps from the launch's `limit` on are skipped: for running a walk
    /// to any step, not only to the end of a round of eight, when looking
    /// for where a GPU goes wrong.
    pub limit: bool,
}

impl Variant {
    /// The walk itself.
    pub fn walk(workgroup: u32) -> Self {
        Variant { workgroup, real_ops: true, lines: WALK_LINES as u32, grouped: false, limit: false }
    }

    /// Whether this computes the walk (not a probe).
    pub fn is_walk(&self) -> bool {
        self.real_ops && self.lines == WALK_LINES as u32
    }

    /// Its pipeline constants, as [`kernel_source`] declares them.
    pub fn constants(&self) -> [(&'static str, f64); 5] {
        let b = |x: bool| if x { 1.0 } else { 0.0 };
        [
            ("WG", self.workgroup as f64),
            ("LINES", self.lines as f64),
            ("REAL_OPS", b(self.real_ops)),
            ("GROUPED", b(self.grouped)),
            ("LIMITED", b(self.limit)),
        ]
    }
}

// ── The kernel ──────────────────────────────────────────────────────────────

const COMMON: &str = r#"
struct Params {
    walks: u32,
    steps: u32,   // steps in this launch, a multiple of 8
    start: u32,   // steps every walk has run before it
    limit: u32,   // with the limit variant: no step from this one on
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> st: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> pads: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> consts: array<u32>;   // TE0..TE3 (1024), then the remix key (8)
@group(0) @binding(4) var<storage, read> order: array<u32>;    // the walk each invocation runs, when grouped
@group(0) @binding(5) var<storage, read> upload: array<vec2<u32>>;  // what the host sends, for copy_in

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

fn sub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}

fn lt64(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

// a * b as 64 bits, from 16-bit limbs: each partial product fits 32 bits
// (and a 24-bit multiplier).
fn wide(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu; let a1 = a >> 16u;
    let b0 = b & 0xffffu; let b1 = b >> 16u;
    let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    return vec2<u32>((p00 & 0xffffu) | (mid << 16u), p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u));
}

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    var p = wide(a.x, b.x);
    p.y = p.y + a.x * b.y + a.y * b.x;
    return p;
}

// The high 64 bits of the 128-bit product.
fn mulhi64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let ll = wide(a.x, b.x);
    let lh = wide(a.x, b.y);
    let hl = wide(a.y, b.x);
    let hh = wide(a.y, b.y);
    var t = ll.y + lh.x;
    var carry = select(0u, 1u, t < lh.x);
    let t2 = t + hl.x;
    carry = carry + select(0u, 1u, t2 < hl.x);
    var r = add64(hh, vec2<u32>(lh.y, 0u));
    r = add64(r, vec2<u32>(hl.y, 0u));
    return add64(r, vec2<u32>(carry, 0u));
}

// One base-2^16 digit of (top * 2^16 + next) / d, for top < d and d >= 2^31
// (Knuth's algorithm D, step D3: exact after at most two corrections, since
// the test uses the divisor's both digits).
fn div_digit(top: u32, next: u32, d1: u32, d0: u32) -> u32 {
    var q = top / d1;
    var r = top - q * d1;
    for (var i = 0u; i < 2u; i = i + 1u) {
        if (r >= 65536u) { break; }
        if (q >= 65536u || q * d0 > ((r << 16u) | next)) {
            q = q - 1u;
            r = r + d1;
        } else {
            break;
        }
    }
    return q;
}

// k / d as 64 bits, for d >= 2^31: the quotient is below 2^33.
fn div64n(k: vec2<u32>, d: u32) -> vec2<u32> {
    let qh = select(0u, 1u, k.y >= d);
    let r1 = k.y - qh * d;
    let d1 = d >> 16u;
    let d0 = d & 0xffffu;
    let u1 = k.x >> 16u;
    let q1 = div_digit(r1, u1, d1, d0);
    let rem = sub64(vec2<u32>((r1 << 16u) | u1, r1 >> 16u), wide(q1, d)).x;
    let q0 = div_digit(rem, k.x & 0xffffu, d1, d0);
    return vec2<u32>((q1 << 16u) | q0, qh);
}

fn rotr64(v: vec2<u32>, n: u32) -> vec2<u32> {
    let a = select(v, v.yx, n >= 32u);
    let m = n & 31u;
    if (m == 0u) { return a; }
    return vec2<u32>((a.x >> m) | (a.y << (32u - m)), (a.y >> m) | (a.x << (32u - m)));
}

// AESENC(v ‖ k, 0): (low 64, high 64) as (x, y) and (z, w).
fn aes(v: vec2<u32>, k: vec2<u32>) -> vec4<u32> {
    let w0 = v.x; let w1 = v.y; let w2 = k.x; let w3 = k.y;
    let o0 = consts[w0 & 255u] ^ consts[256u + ((w1 >> 8u) & 255u)] ^ consts[512u + ((w2 >> 16u) & 255u)] ^ consts[768u + (w3 >> 24u)];
    let o1 = consts[w1 & 255u] ^ consts[256u + ((w2 >> 8u) & 255u)] ^ consts[512u + ((w3 >> 16u) & 255u)] ^ consts[768u + (w0 >> 24u)];
    let o2 = consts[w2 & 255u] ^ consts[256u + ((w3 >> 8u) & 255u)] ^ consts[512u + ((w0 >> 16u) & 255u)] ^ consts[768u + (w1 >> 24u)];
    let o3 = consts[w3 & 255u] ^ consts[256u + ((w0 >> 8u) & 255u)] ^ consts[512u + ((w1 >> 16u) & 255u)] ^ consts[768u + (w2 >> 24u)];
    return vec4<u32>(o0, o1, o2, o3);
}

// The program a draw z decodes to, as eight 3-bit fields (register s's
// operation at bits 3s). Fisher–Yates reads z's digits in radices 8..2, so
// only z mod 8! = 40320 matters; 2^32 mod 40320 = 256.
fn program(z: vec2<u32>) -> u32 {
    var zz = ((z.y % 40320u) * 256u + z.x % 40320u) % 40320u;
    var p = 0xfac688u;   // 0, 1, .., 7
    var i = 7u;
    loop {
        if (i == 0u) { break; }
        let n = i + 1u;
        let j = zz % n;
        zz = zz / n;
        let fi = (p >> (3u * i)) & 7u;
        let fj = (p >> (3u * j)) & 7u;
        p = p & ~((7u << (3u * i)) | (7u << (3u * j)));
        p = p | (fj << (3u * i)) | (fi << (3u * j));
        i = i - 1u;
    }
    return p;
}

// ── BLAKE3, for the re-mix: one block, derive_key mode, two output blocks ──

fn rotr32(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }

fn g(v: ptr<function, array<u32, 16>>, a: u32, b: u32, c: u32, d: u32, x: u32, y: u32) {
    (*v)[a] = (*v)[a] + (*v)[b] + x;
    (*v)[d] = rotr32((*v)[d] ^ (*v)[a], 16u);
    (*v)[c] = (*v)[c] + (*v)[d];
    (*v)[b] = rotr32((*v)[b] ^ (*v)[c], 12u);
    (*v)[a] = (*v)[a] + (*v)[b] + y;
    (*v)[d] = rotr32((*v)[d] ^ (*v)[a], 8u);
    (*v)[c] = (*v)[c] + (*v)[d];
    (*v)[b] = rotr32((*v)[b] ^ (*v)[c], 7u);
}

// The 64-byte output block `counter` of BLAKE3 in derive_key mode, keyed by
// the context key, over the one 64-byte block m.
fn blake3_out(m_in: array<u32, 16>, counter: u32) -> array<u32, 16> {
    var cv: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) { cv[i] = consts[1024u + i]; }
    var m = m_in;
    // CHUNK_START | CHUNK_END | ROOT | DERIVE_KEY_MATERIAL
    var v = array<u32, 16>(cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, counter, 0u, 64u, 75u);
    for (var round = 0u; round < 7u; round = round + 1u) {
        g(&v, 0u, 4u, 8u, 12u, m[0], m[1]);
        g(&v, 1u, 5u, 9u, 13u, m[2], m[3]);
        g(&v, 2u, 6u, 10u, 14u, m[4], m[5]);
        g(&v, 3u, 7u, 11u, 15u, m[6], m[7]);
        g(&v, 0u, 5u, 10u, 15u, m[8], m[9]);
        g(&v, 1u, 6u, 11u, 12u, m[10], m[11]);
        g(&v, 2u, 7u, 8u, 13u, m[12], m[13]);
        g(&v, 3u, 4u, 9u, 14u, m[14], m[15]);
        m = array<u32, 16>(m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13],
                           m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]);
    }
    var out: array<u32, 16>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        out[i] = v[i] ^ v[i + 8u];
        out[i + 8u] = v[i + 8u] ^ cv[i];
    }
    return out;
}
"#;

/// One step on register `s`, written out for the kernel. The pipeline
/// constants pick what it does: the walk's operation or (`REAL_OPS` false)
/// an XOR, and with `LIMITED`, nothing from `P.limit` on.
fn step_source(s: usize) -> String {
    let t = |d: usize| (s + 1 + d) % 8;
    format!(
        r#"
    if (!LIMITED || P.start + n * 8u + {s}u < P.limit) {{
        let x = r{s} ^ c;
        let a = x.x & (LINES - 1u);
        let toff = (x.x >> 12u) & 3u;
        let j = (x.x >> 14u) & 7u;
        let e = (x.x >> 17u) & (LINES - 1u);
        let b = x.y & (LINES - 1u);
        let f = (x.y >> 12u) & (LINES - 1u);
        let ia = base + a * 8u + j;
        let ib = base + b * 8u + (j ^ 4u);
        let ie = base + e * 8u + (j ^ 2u);
        let if_ = base + f * 8u + (j ^ 6u);
        let m0 = pads[ia];
        let m1 = pads[ib];
        let m2 = pads[ie];
        let m3 = pads[if_];
        let k = select(select(r{t0}, r{t1}, toff == 1u), select(r{t2}, r{t3}, toff == 3u), toff >= 2u);
        let v = x ^ m0 ^ m2;
        var y = v ^ k;
        var h = vec2<u32>(0u, 0u);
        if (REAL_OPS) {{
            switch ((prog >> {shift}u) & 7u) {{
                case 0u: {{ y = mul64(v, k | vec2<u32>(1u, 0u)); }}
                case 1u: {{ y = v ^ mulhi64(v, k); }}
                case 2u: {{ y = add64(v, div64n(k, v.x | 0x80000001u)); }}
                case 3u: {{ let o = aes(v, k); y = o.xy; h = o.zw; }}
                case 4u: {{ y = rotr64(v, k.x & 63u); }}
                case 5u: {{ y = add64(v, k); }}
                case 6u: {{ y = sub64(v, k); }}
                default: {{}}
            }}
        }}
        let out = y ^ m1 ^ m3;
        r{s} = out;
        r{t0} = r{t0} ^ select(vec2<u32>(0u, 0u), h, toff == 0u);
        r{t1} = r{t1} ^ select(vec2<u32>(0u, 0u), h, toff == 1u);
        r{t2} = r{t2} ^ select(vec2<u32>(0u, 0u), h, toff == 2u);
        r{t3} = r{t3} ^ select(vec2<u32>(0u, 0u), h, toff == 3u);
        c = out;
        pads[ia] = m0 ^ y;
        pads[ib] = add64(m1, x);
        pads[ie] = add64(m2, y);
        pads[if_] = m3 ^ x;
    }}"#,
        shift = 3 * s,
        t0 = t(0),
        t1 = t(1),
        t2 = t(2),
        t3 = t(3),
    )
}

/// The kernel's WGSL: the common functions, the walk's steps kernel
/// (`walk_steps`), and two test kernels (`test_ops`, `test_remix`). One
/// source for every [`Variant`]: its fields are pipeline constants
/// ([`Variant::constants`]). `bench --dump-wgsl` writes it out and `--wgsl`
/// runs an edited one, with nothing compiled.
pub fn kernel_source() -> String {
    let steps: String = (0..8).map(step_source).collect();
    let regs_in: String = (0..8).map(|i| format!("    var r{i} = st[sb + {i}u];\n")).collect();
    let regs_out: String = (0..8).map(|i| format!("    st[sb + {i}u] = r{i};\n")).collect();
    let regs_m = (0..8).map(|i| format!("r{i}.x, r{i}.y")).collect::<Vec<_>>().join(", ");
    let regs_set: String = (0..8).map(|i| format!("            r{i} = vec2<u32>(o[{}], o[{}]);\n", 2 * i, 2 * i + 1)).collect();
    format!(
        r#"// Midstate Walk for GPUs, written out by core::wgpu_walk (midstate {version}).
// The pipeline constants below are set for each variant the bench runs.
override WG: u32 = 64u;           // invocations per workgroup
override LINES: u32 = {lines}u;      // pad lines used: the walk's own, or 8 (a probe)
override REAL_OPS: bool = true;   // the walk's operations, or an XOR for each (a probe)
override GROUPED: bool = false;   // lanes run walks in the order the host sorts them
override LIMITED: bool = false;   // no step from P.limit on
const PAD_WORDS: u32 = {pad_words}u;
const REMIX: u32 = {remix}u;
{common}

@compute @workgroup_size(WG)
fn walk_steps(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= P.walks) {{ return; }}
    let w = select(gid.x, order[gid.x], GROUPED);
    let sb = w * {state_words}u;
    // A probe's few lines are shared by every walk, so they stay cached.
    let base = select(w * PAD_WORDS, 0u, LINES < {lines}u);
{regs_in}    var c = st[sb + 8u];
    var prog = st[sb + 9u].x;
    for (var n = 0u; n < P.steps / 8u; n = n + 1u) {{
{steps}
        if (((P.start + (n + 1u) * 8u) & (REMIX - 1u)) == 0u && (!LIMITED || P.start + (n + 1u) * 8u <= P.limit)) {{
            let m = array<u32, 16>({regs_m});
            let o = blake3_out(m, 0u);
{regs_set}            let o1 = blake3_out(m, 1u);
            prog = program(vec2<u32>(o1[0], o1[1]));
        }}
    }}
{regs_out}    st[sb + 8u] = c;
    st[sb + 9u] = vec2<u32>(prog, 0u);
}}

// Copies P.walks words of upload to st (P.limit == 0) or pads (otherwise),
// from word P.start on. The host never writes st or pads itself: on some
// drivers (the Pi 5's v3dv) a kernel then reads what it wrote there before.
@compute @workgroup_size(64)
fn copy_in(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= P.walks) {{ return; }}
    if (P.limit == 0u) {{ st[P.start + i] = upload[i]; }} else {{ pads[P.start + i] = upload[i]; }}
}}

// Each operation on input pairs st[2i], st[2i+1], written from st[2 * walks + 12i].
@compute @workgroup_size(64)
fn test_ops(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= P.walks) {{ return; }}
    let a = st[2u * i];
    let b = st[2u * i + 1u];
    let o = 2u * P.walks + 12u * i;
    st[o] = mul64(a, b | vec2<u32>(1u, 0u));
    st[o + 1u] = a ^ mulhi64(a, b);
    st[o + 2u] = add64(a, div64n(b, a.x | 0x80000001u));
    let x = aes(a, b);
    st[o + 3u] = x.xy;
    st[o + 4u] = x.zw;
    st[o + 5u] = rotr64(a, b.x & 63u);
    st[o + 6u] = add64(a, b);
    st[o + 7u] = sub64(a, b);
    st[o + 8u] = a ^ b;
    st[o + 9u] = vec2<u32>(program(a), 0u);
    st[o + 10u] = wide(a.x, b.x);
    st[o + 11u] = div64n(b, a.x | 0x80000001u);
}}

// The re-mix of each walk state's registers, in place.
@compute @workgroup_size(64)
fn test_remix(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= P.walks) {{ return; }}
    let sb = i * {state_words}u;
    var m: array<u32, 16>;
    for (var k = 0u; k < 8u; k = k + 1u) {{
        m[2u * k] = st[sb + k].x;
        m[2u * k + 1u] = st[sb + k].y;
    }}
    let o = blake3_out(m, 0u);
    for (var k = 0u; k < 8u; k = k + 1u) {{
        st[sb + k] = vec2<u32>(o[2u * k], o[2u * k + 1u]);
    }}
    let o1 = blake3_out(m, 1u);
    st[sb + 9u] = vec2<u32>(program(vec2<u32>(o1[0], o1[1])), 0u);
}}
"#,
        version = env!("CARGO_PKG_VERSION"),
        lines = WALK_LINES,
        pad_words = PAD_WORDS,
        remix = WALK_REMIX,
        common = COMMON,
        state_words = STATE_WORDS,
    )
}

/// Which walk each lane runs, so that lanes side by side have their costly
/// operations at the same steps: walks sorted by the step of their AES round,
/// then of the division, the high multiply and the multiply.
pub fn group_order(progs: &[[u8; 8]]) -> Vec<u32> {
    let key = |p: &[u8; 8]| {
        let at = |op: u8| p.iter().position(|o| *o == op).unwrap_or(0) as u32;
        (at(3) << 9) | (at(2) << 6) | (at(1) << 3) | at(0)
    };
    let mut order: Vec<u32> = (0..progs.len() as u32).collect();
    order.sort_by_key(|&w| key(&progs[w as usize]));
    order
}

/// The constants the kernel reads: the AES table four ways rotated, then the
/// re-mix's context key.
pub fn consts() -> Vec<u32> {
    let mut out = Vec::with_capacity(1032);
    let te: Vec<u32> = (0..256)
        .map(|x| {
            let s = walk::SBOX[x] as u32;
            let s2 = ((s << 1) ^ if s & 0x80 != 0 { 0x11b } else { 0 }) & 0xff;
            s2 | (s << 8) | (s << 16) | ((s2 ^ s) << 24)
        })
        .collect();
    for rot in [0, 8, 16, 24] {
        out.extend(te.iter().map(|t| t.rotate_left(rot)));
    }
    let key = blake3::hazmat::hash_derive_key_context("midstate walk v1 remix");
    out.extend(key.chunks(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])));
    out
}

/// A program as the kernel holds it: register s's operation at bits 3s.
pub fn pack_program(p: &[u8; 8]) -> u32 {
    p.iter().enumerate().map(|(s, op)| (*op as u32) << (3 * s)).sum()
}

fn unpack_program(p: u32) -> [u8; 8] {
    std::array::from_fn(|s| ((p >> (3 * s)) & 7) as u8)
}

/// A walk's state as the kernel holds it: r0..r7, c, (program, 0).
fn state_words(st: &WalkState) -> Vec<u32> {
    let mut words: Vec<u32> = st.r.iter().flat_map(|r| word(*r)).collect();
    words.extend(word(st.c));
    words.extend([pack_program(&st.prog), 0]);
    words
}

fn word(v: u64) -> [u32; 2] {
    [v as u32, (v >> 32) as u32]
}

fn words_bytes(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ── The host ────────────────────────────────────────────────────────────────

/// A GPU wgpu found.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub name: String,
    pub backend: String,
    pub kind: String,
    pub driver: String,
    /// Whether it is a software renderer (llvmpipe, SwiftShader).
    pub software: bool,
}

fn info_of(a: &wgpu::AdapterInfo) -> GpuInfo {
    GpuInfo {
        name: a.name.clone(),
        backend: format!("{:?}", a.backend),
        kind: format!("{:?}", a.device_type),
        driver: [a.driver.as_str(), a.driver_info.as_str()].iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join(" "),
        software: a.device_type == wgpu::DeviceType::Cpu,
    }
}

fn instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env())
}

/// Every GPU wgpu finds, software ones included.
pub fn gpus() -> Vec<GpuInfo> {
    pollster::block_on(instance().enumerate_adapters(wgpu::Backends::all()))
        .iter()
        .map(|a| info_of(&a.get_info()))
        .collect()
}

/// A GPU opened for walks: its buffers for up to `capacity` of them.
pub struct GpuWalker {
    pub info: GpuInfo,
    /// The most walks at once.
    pub capacity: u32,
    /// The largest workgroup it takes.
    pub max_workgroup: u32,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: std::collections::HashMap<(Variant, &'static str), wgpu::ComputePipeline>,
    /// The kernel run instead of [`kernel_source`], if any.
    source: Option<String>,
    module: Option<wgpu::ShaderModule>,
    params: wgpu::Buffer,
    state: wgpu::Buffer,
    pads: wgpu::Buffer,
    readback: wgpu::Buffer,
    order: wgpu::Buffer,
    upload: wgpu::Buffer,
    /// Uploads written straight into the state and pads, as before the copy
    /// kernel: only to show a driver's fault in [`check_uploads`].
    pub direct: bool,
    bind: wgpu::BindGroup,
}

/// Where an upload goes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Target {
    State,
    Pads,
}

/// 8-byte words an upload sends at a time (8 MiB): a whole load of 1,024
/// pads (256 MiB) at once was more than the Pi's driver could stage.
const UPLOAD_WORDS: u64 = 1 << 20;

fn entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

impl GpuWalker {
    /// Opens the first GPU whose name contains `name` (any case), or the first
    /// hardware one, for up to `walks` walks (fewer if its memory limits say
    /// so). Software renderers only when named.
    pub fn open(name: Option<&str>, walks: u32) -> Result<Self, String> {
        pollster::block_on(Self::open_async(name, walks))
    }

    async fn open_async(name: Option<&str>, walks: u32) -> Result<Self, String> {
        let instance = instance();
        let mut adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
        // A GPU can be listed once per API (the Pi's through Vulkan and
        // OpenGL ES): the native compute APIs first.
        adapters.sort_by_key(|a| match a.get_info().backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Dx12 => 0,
            _ => 1,
        });
        let want = name.map(str::to_lowercase);
        let adapter = adapters
            .into_iter()
            .find(|a| {
                let i = a.get_info();
                match &want {
                    Some(w) => i.name.to_lowercase().contains(w.as_str()),
                    None => i.device_type != wgpu::DeviceType::Cpu,
                }
            })
            .ok_or_else(|| match name {
                Some(n) => format!("no GPU named like \"{n}\""),
                None => "no GPU found".to_string(),
            })?;
        let info = info_of(&adapter.get_info());
        let limits = adapter.limits();
        let per_walk = PAD_WORDS * 8;
        let most = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size) / per_walk;
        let capacity = (walks as u64).min(most).max(1) as u32;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("midstate-walk"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("could not open {}: {e}", info.name))?;

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("walk"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(5, wgpu::BufferBindingType::Storage { read_only: true }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("walk"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let buffer = |label, size: u64, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let params = buffer("params", 16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        // Room for the test kernels too: 2 inputs and 12 outputs a pair.
        let state_words = (capacity as u64 * STATE_WORDS).max(14 * 4096);
        let state = buffer("state", state_words * 8, usage);
        let pads = buffer("pads", capacity as u64 * per_walk, usage);
        let consts_buf = buffer("consts", 1032 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        queue.write_buffer(&consts_buf, 0, &words_bytes(&consts()));
        let readback = buffer("readback", state_words * 8, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let order = buffer("order", capacity as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        queue.write_buffer(&order, 0, &words_bytes(&(0..capacity).collect::<Vec<u32>>()));
        let upload = buffer("upload", UPLOAD_WORDS * 8, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("walk"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pads.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: consts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: order.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: upload.as_entire_binding() },
            ],
        });
        Ok(GpuWalker {
            info,
            capacity,
            max_workgroup: limits.max_compute_workgroup_size_x.min(limits.max_compute_invocations_per_workgroup),
            device,
            queue,
            pipeline_layout,
            pipelines: Default::default(),
            source: None,
            module: None,
            params,
            state,
            pads,
            readback,
            order,
            upload,
            direct: false,
            bind,
        })
    }

    /// The pipeline for `entry` of variant `v`, compiled once.
    /// Runs the kernel from `source` instead of [`kernel_source`]: an edited
    /// copy, to try a change on a GPU with nothing compiled. Before any run.
    pub fn use_source(&mut self, source: String) {
        self.source = Some(source);
        self.module = None;
        self.pipelines.clear();
    }

    fn pipeline(&mut self, v: Variant, entry: &'static str) -> Result<&wgpu::ComputePipeline, String> {
        if !self.pipelines.contains_key(&(v, entry)) {
            let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let source = self.source.clone().unwrap_or_else(kernel_source);
            let module = self
                .module
                .get_or_insert_with(|| {
                    self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("walk"),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    })
                })
                .clone();
            let constants = v.constants();
            let pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&self.pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    // The test kernels are the same whatever the variant.
                    constants: if entry == "walk_steps" { &constants } else { &[] },
                    ..Default::default()
                },
                cache: None,
            });
            if let Some(e) = pollster::block_on(scope.pop()) {
                return Err(format!("the kernel did not compile for {}: {e}", self.info.name));
            }
            self.pipelines.insert((v, entry), pipe);
        }
        Ok(&self.pipelines[&(v, entry)])
    }

    fn wait(&self) -> Result<(), String> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|e| format!("the GPU stopped answering: {e}"))
    }

    /// Runs `entry` over `n` invocations with `steps` from `start` (and, for
    /// the limit variant, none from `limit` on), and waits.
    fn launch(&mut self, v: Variant, entry: &'static str, n: u32, steps: u32, start: u32, limit: u32) -> Result<Duration, String> {
        let groups = n.div_ceil(if entry == "walk_steps" { v.workgroup } else { 64 });
        self.queue.write_buffer(&self.params, 0, &words_bytes(&[n, steps, start, limit]));
        let pipe = self.pipeline(v, entry)?.clone();
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(entry) });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(entry), timestamp_writes: None });
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        let begun = Instant::now();
        self.queue.submit([enc.finish()]);
        self.wait()?;
        Ok(begun.elapsed())
    }

    fn read_words(&self, words: u64) -> Result<Vec<u32>, String> {
        self.read_from(&self.state, 0, words)
    }

    /// `words` 8-byte words of `buf` from word `at`, as 32-bit halves.
    fn read_from(&self, buf: &wgpu::Buffer, at: u64, words: u64) -> Result<Vec<u32>, String> {
        let bytes = words * 8;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("read") });
        enc.copy_buffer_to_buffer(buf, at * 8, &self.readback, 0, bytes);
        self.queue.submit([enc.finish()]);
        let slice = self.readback.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.wait()?;
        rx.recv().map_err(|e| e.to_string())?.map_err(|e| format!("reading the GPU back: {e}"))?;
        let out = slice.get_mapped_range().chunks(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        self.readback.unmap();
        Ok(out)
    }

    /// Puts `words` (32-bit halves, two to a word) into the state or the
    /// pads from 8-byte word `at`: through the upload buffer and the copy
    /// kernel, a batch at a time, so the host never writes memory a kernel
    /// writes. Waits until it is there.
    fn put(&mut self, into: Target, at: u64, words: &[u32]) -> Result<(), String> {
        if self.direct {
            let buf = if into == Target::State { &self.state } else { &self.pads };
            self.queue.write_buffer(buf, at * 8, &words_bytes(words));
            return self.wait();
        }
        for (k, batch) in words.chunks(2 * UPLOAD_WORDS as usize).enumerate() {
            self.queue.write_buffer(&self.upload, 0, &words_bytes(batch));
            let start = at + k as u64 * UPLOAD_WORDS;
            let n = (batch.len() / 2) as u32;
            self.launch(Variant::walk(64), "copy_in", n, 0, start as u32, (into == Target::Pads) as u32)?;
        }
        Ok(())
    }

    /// Seeds the walks of `nonces` on the CPU and loads them: their pads and
    /// states.
    pub fn load(&mut self, mining_hash: &[u8; 32], nonces: &[u64]) -> Result<Vec<WalkState>, String> {
        if nonces.len() > self.capacity as usize {
            return Err(format!("{} walks, room for {}", nonces.len(), self.capacity));
        }
        let per_batch = (UPLOAD_WORDS / PAD_WORDS).max(1) as usize;
        let mut pad = walk::new_pad();
        let mut states = Vec::with_capacity(nonces.len());
        let mut words = Vec::with_capacity(nonces.len() * STATE_WORDS as usize * 2);
        for (b, batch) in nonces.chunks(per_batch).enumerate() {
            let mut flat: Vec<u32> = Vec::with_capacity(batch.len() * PAD_WORDS as usize * 2);
            for nonce in batch {
                let st = walk::walk_start(mining_hash, *nonce, &mut pad);
                flat.extend(pad.iter().flat_map(|line| line.iter().flat_map(|w| word(*w))));
                for r in st.r {
                    words.extend(word(r));
                }
                words.extend(word(st.c));
                words.extend([pack_program(&st.prog), 0]);
                states.push(st);
            }
            self.put(Target::Pads, (b * per_batch) as u64 * PAD_WORDS, &flat)?;
        }
        self.put(Target::State, 0, &words)?;
        Ok(states)
    }

    /// Loads one walk part-way: `st` and its `pad`, as walk 0.
    pub fn load_state(&mut self, st: &WalkState, pad: &walk::Pad) -> Result<(), String> {
        let flat: Vec<u32> = pad.iter().flat_map(|line| line.iter().flat_map(|w| word(*w))).collect();
        self.put(Target::Pads, 0, &flat)?;
        self.put(Target::State, 0, &state_words(st))
    }

    /// Walk 0's pad, as it is now.
    pub fn pad0(&self) -> Result<Vec<u64>, String> {
        let w = self.read_from(&self.pads, 0, PAD_WORDS)?;
        Ok(w.chunks(2).map(|p| p[0] as u64 | (p[1] as u64) << 32).collect())
    }

    /// Runs `steps` more steps (a multiple of 8) of the `n` loaded walks,
    /// which have run `start` so far. How long the GPU took.
    pub fn run(&mut self, v: Variant, n: u32, start: u32, steps: u32) -> Result<Duration, String> {
        self.run_to(v, n, start, steps, u32::MAX)
    }

    /// As [`GpuWalker::run`], skipping every step from `limit` on (the
    /// limit variant).
    fn run_to(&mut self, v: Variant, n: u32, start: u32, steps: u32, limit: u32) -> Result<Duration, String> {
        assert!(steps % 8 == 0 && start % 8 == 0, "whole rounds of eight");
        self.launch(v, "walk_steps", n, steps, start, limit)
    }

    /// Regroups the `n` loaded walks by their programs now ([`group_order`]).
    pub fn regroup(&mut self, n: u32) -> Result<(), String> {
        let progs: Vec<[u8; 8]> = self.states(n)?.iter().map(|s| s.prog).collect();
        self.queue.write_buffer(&self.order, 0, &words_bytes(&group_order(&progs)));
        Ok(())
    }

    /// Runs `steps` steps from `start` in launches sized to
    /// [`LAUNCH_TARGET`], regrouping at every re-mix for a grouped variant,
    /// beginning with launches of `chunk` steps. The chunk to go on with.
    fn run_span(&mut self, v: Variant, n: u32, start: u32, steps: u32, chunk: u32) -> Result<u32, String> {
        self.run_span_to(v, n, start, steps, chunk, u32::MAX)
    }

    /// As `run_span`, skipping every step from `limit` on (the limit variant).
    fn run_span_to(&mut self, v: Variant, n: u32, start: u32, steps: u32, mut chunk: u32, limit: u32) -> Result<u32, String> {
        let mut done = start;
        while done < start + steps {
            if v.grouped && done % WALK_REMIX as u32 == 0 {
                self.regroup(n)?;
            }
            let c = chunk.min(start + steps - done);
            let took = self.run_to(v, n, done, c, limit)?;
            done += c;
            chunk = next_chunk(chunk, took);
        }
        Ok(chunk)
    }

    /// Runs walk 0 from `st` and `pad` to step `to` (any step), skipping
    /// what lies past it: for the search for where a GPU goes wrong.
    fn run_one_to(&mut self, v: Variant, st: &WalkState, pad: &walk::Pad, to: usize) -> Result<(), String> {
        let lim = Variant { limit: true, grouped: false, ..v };
        self.load_state(st, pad)?;
        let rounds = (to - st.steps).div_ceil(8) as u32 * 8;
        self.run_span_to(lim, 1, st.steps as u32, rounds, 64, to as u32).map(|_| ())
    }

    /// The `n` loaded walks' states, as they are now.
    pub fn states(&self, n: u32) -> Result<Vec<WalkState>, String> {
        let w = self.read_words(n as u64 * STATE_WORDS)?;
        Ok((0..n as usize)
            .map(|i| {
                let s = &w[i * 20..i * 20 + 20];
                let u = |k: usize| s[2 * k] as u64 | (s[2 * k + 1] as u64) << 32;
                WalkState { r: std::array::from_fn(u), prog: unpack_program(s[18]), c: u(8), steps: 0 }
            })
            .collect())
    }

    /// The proof-of-work hashes of `nonces`, the steps run on the GPU in
    /// launches sized to [`LAUNCH_TARGET`].
    pub fn hash(&mut self, v: Variant, mining_hash: &[u8; 32], nonces: &[u64]) -> Result<Vec<[u8; 32]>, String> {
        let n = nonces.len() as u32;
        self.load(mining_hash, nonces)?;
        self.run_span(v, n, 0, WALK_STEPS as u32, 64)?;
        Ok(self.states(n)?.iter().zip(nonces).map(|(s, nonce)| walk::walk_finish(mining_hash, *nonce, &s.r)).collect())
    }

    /// Runs the test kernel over operand pairs: for each, the twelve results
    /// `test_ops` writes.
    pub fn test_ops(&mut self, pairs: &[(u64, u64)]) -> Result<Vec<[u64; 12]>, String> {
        let n = pairs.len() as u32;
        let words: Vec<u32> = pairs.iter().flat_map(|(a, b)| word(*a).into_iter().chain(word(*b))).collect();
        self.put(Target::State, 0, &words)?;
        self.launch(Variant::walk(64), "test_ops", n, 0, 0, 0)?;
        let w = self.read_words(14 * n as u64)?;
        let base = 4 * n as usize;
        Ok((0..n as usize)
            .map(|i| std::array::from_fn(|k| {
                let o = base + 24 * i + 2 * k;
                w[o] as u64 | (w[o + 1] as u64) << 32
            }))
            .collect())
    }

    /// Runs the GPU's re-mix on each register set: the new registers and
    /// program.
    pub fn test_remix(&mut self, regs: &[[u64; 8]]) -> Result<Vec<([u64; 8], [u8; 8])>, String> {
        let n = regs.len() as u32;
        let words: Vec<u32> = regs.iter().flat_map(|r| r.iter().flat_map(|x| word(*x)).chain([0, 0, 0, 0])).collect();
        self.put(Target::State, 0, &words)?;
        self.launch(Variant::walk(64), "test_remix", n, 0, 0, 0)?;
        Ok(self.states(n)?.into_iter().map(|s| (s.r, s.prog)).collect())
    }
}

/// The next launch's steps: aimed at [`LAUNCH_TARGET`], a power of two from
/// 8 to one re-mix's worth, at most doubling each time.
pub fn next_chunk(chunk: u32, took: Duration) -> u32 {
    let target = LAUNCH_TARGET.as_secs_f64();
    let t = took.as_secs_f64().max(1e-6);
    let want = (chunk as f64 * target / t).clamp(8.0, (chunk * 2) as f64);
    let mut c = 8u32;
    while c * 2 <= want as u32 && c * 2 <= WALK_REMIX as u32 {
        c *= 2;
    }
    c
}

// ── Checking a GPU ──────────────────────────────────────────────────────────

/// What `test_ops` computes for each pair, by name.
pub const TEST_OPS: [&str; 12] = ["mul", "mulhi", "div", "aes (low half)", "aes (high half)", "rotr", "add", "sub", "xor", "program", "32x32 multiply", "64/32 division"];

/// What `test_ops` should give for (a, b), from the CPU.
pub fn ops_reference(a: u64, b: u64) -> [u64; 12] {
    let d = (a as u32 | 0x8000_0001) as u64;
    let (aes_lo, aes_hi) = walk::aes_round_pair(a, b);
    [
        a.wrapping_mul(b | 1),
        a ^ ((a as u128 * b as u128) >> 64) as u64,
        a.wrapping_add(b / d),
        aes_lo,
        aes_hi,
        a.rotate_right((b & 63) as u32),
        a.wrapping_add(b),
        a.wrapping_sub(b),
        a ^ b,
        pack_program(&walk::program_of(a)) as u64,
        (a as u32 as u64) * (b as u32 as u64),
        b / d,
    ]
}

/// Operand pairs for checking: every pair of edge values, then `total`
/// in all from a fixed generator, every seventh with its division's
/// quotient near a digit boundary.
pub fn op_pairs(total: usize) -> Vec<(u64, u64)> {
    let edges = [0u64, 1, 2, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, 0xffff_ffff, 1 << 32, (1 << 32) + 1, u64::MAX, u64::MAX - 1, 1 << 63, 0xffff_0000_ffff_0000, 0x0000_ffff_0000_ffff];
    let mut pairs: Vec<(u64, u64)> = edges.iter().flat_map(|a| edges.iter().map(move |b| (*a, *b))).collect();
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    while pairs.len() < total {
        let a = next();
        let r = next();
        let b = if pairs.len() % 7 == 0 { ((a as u32 | 0x8000_0001) as u64).wrapping_mul(r >> 33) } else { r };
        pairs.push((a, b));
    }
    pairs
}

/// Runs every operation on `total` operand pairs on the GPU against the
/// CPU: a line for each that came out wrong, with its first case.
pub fn check_ops(g: &mut GpuWalker, total: usize) -> Result<Vec<String>, String> {
    let mut wrong = [0usize; 12];
    let mut first: [Option<String>; 12] = Default::default();
    for batch in op_pairs(total).chunks(4096) {
        for (&(a, b), got) in batch.iter().zip(g.test_ops(batch)?) {
            let want = ops_reference(a, b);
            for k in 0..12 {
                if got[k] != want[k] {
                    wrong[k] += 1;
                    first[k].get_or_insert_with(|| format!("a={a:#018x} b={b:#018x}: the GPU gave {:#018x}, the CPU {:#018x}", got[k], want[k]));
                }
            }
        }
    }
    Ok((0..12)
        .filter(|&k| wrong[k] > 0)
        .map(|k| format!("{} wrong {} times, first {}", TEST_OPS[k], wrong[k], first[k].as_deref().unwrap_or("")))
        .collect())
}

/// Runs the re-mix on `total` register sets on the GPU against the CPU: how
/// many came out wrong, and the first.
pub fn check_remix(g: &mut GpuWalker, total: usize) -> Result<Option<String>, String> {
    let mut wrong = 0;
    let mut first = None;
    let sets: Vec<[u64; 8]> = (0..total as u64).map(|i| std::array::from_fn(|k| (i ^ (k as u64) << 56).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(k as u32 * 7))).collect();
    for batch in sets.chunks(4096) {
        for (r, got) in batch.iter().zip(g.test_remix(batch)?) {
            if got != walk::remix_of(r) {
                wrong += 1;
                first.get_or_insert_with(|| format!("{r:x?}"));
            }
        }
    }
    Ok(first.map(|f| format!("{wrong} of {total} wrong, first of registers {f}")))
}

/// Whether a kernel sees what was just uploaded into memory it has written:
/// four times, a walk's state is put in place, run through zero steps (the
/// kernel reads it and writes it back) and read back. With uploads written
/// straight into the state (`direct`) and through the copy kernel: how many
/// of the four each got right.
pub fn check_uploads(g: &mut GpuWalker) -> Result<(usize, usize), String> {
    let v = Variant::walk(64.min(g.max_workgroup));
    let mut right = [0usize; 2];
    for (m, direct) in [true, false].into_iter().enumerate() {
        g.direct = direct;
        // The kernel writes the state once first, as any earlier run has.
        g.run(v, 1, 0, 0)?;
        for k in 0..4u64 {
            let st = WalkState {
                r: std::array::from_fn(|i| (k * 8 + i as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
                prog: walk::program_of(k * 77 + 3),
                c: k.wrapping_mul(0xdead_beef),
                steps: 0,
            };
            g.put(Target::State, 0, &state_words(&st))?;
            g.run(v, 1, 0, 0)?;
            let back = g.states(1)?[0];
            if (back.r, back.prog, back.c) == (st.r, st.prog, st.c) {
                right[m] += 1;
            }
        }
    }
    g.direct = false;
    Ok((right[0], right[1]))
}

/// Whether walk 0 on the GPU is `st` with `pad`; if not, what differs.
fn differences(g: &GpuWalker, st: &WalkState, pad: &walk::Pad) -> Result<Vec<String>, String> {
    let gpu = g.states(1)?[0];
    let mut out = Vec::new();
    for k in 0..8 {
        if gpu.r[k] != st.r[k] {
            out.push(format!("r{k}: GPU {:#018x}, CPU {:#018x}", gpu.r[k], st.r[k]));
        }
    }
    if gpu.c != st.c {
        out.push(format!("c: GPU {:#018x}, CPU {:#018x}", gpu.c, st.c));
    }
    if gpu.prog != st.prog {
        out.push(format!("program: GPU {:?}, CPU {:?}", gpu.prog, st.prog));
    }
    let gp = g.pad0()?;
    let flat = pad.iter().flat_map(|l| l.iter().copied());
    let bad: Vec<String> = gp
        .iter()
        .zip(flat)
        .enumerate()
        .filter(|(_, (a, b))| *a != b)
        .map(|(i, (a, b))| format!("pad word {i} (line {}, word {}): GPU {a:#018x}, CPU {b:#018x}", i / 8, i % 8))
        .collect();
    if !bad.is_empty() {
        out.push(format!("{} pad word{} differ", bad.len(), if bad.len() == 1 { "" } else { "s" }));
        out.extend(bad.into_iter().take(6));
    }
    Ok(out)
}

/// Finds the first step at which the GPU's walk of `nonce` leaves the CPU's,
/// and what is wrong about it: runs the walk in stretches of one re-mix,
/// checking each, runs a wrong stretch again to see whether the GPU repeats
/// itself, then narrows it to one step. A report to print.
pub fn find_divergence(g: &mut GpuWalker, v: Variant, mining_hash: &[u8; 32], nonce: u64) -> Result<String, String> {
    let mut pad = walk::new_pad();
    let mut cpu = walk::walk_start(mining_hash, nonce, &mut pad);
    g.load(mining_hash, &[nonce])?;
    let mut chunk = 64;
    let mut report = Vec::new();
    while cpu.steps < WALK_STEPS {
        let (a, start, start_pad) = (cpu.steps, cpu, pad.clone());
        chunk = g.run_span(v, 1, a as u32, WALK_REMIX as u32, chunk)?;
        walk::walk_steps(&mut cpu, &mut pad, WALK_REMIX);
        if differences(g, &cpu, &pad)?.is_empty() {
            continue;
        }
        report.push(format!("The walk is right up to step {a} and wrong by step {}.", a + WALK_REMIX));

        // The same stretch again, from the CPU's state at its start.
        g.load_state(&start, &start_pad)?;
        g.run_span(v, 1, a as u32, WALK_REMIX as u32, 64)?;
        if differences(g, &cpu, &pad)?.is_empty() {
            report.push("Run again from the same state, the GPU got that stretch right: it does not repeat itself.".into());
            report.push("So the fault comes and goes: a GPU reset (look for v3d in: sudo dmesg | tail -50), or a race in reading back what a step wrote.".into());
            return Ok(report.join("\n"));
        }
        report.push("Run again from the same state, the GPU got it wrong again.".into());

        // The first step after which the GPU differs: (lo] right, [hi wrong.
        let (mut lo, mut hi) = (0usize, WALK_REMIX);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            g.run_one_to(v, &start, &start_pad, a + mid)?;
            let (mut st, mut p) = (start, start_pad.clone());
            walk::walk_steps(&mut st, &mut p, mid);
            if differences(g, &st, &p)?.is_empty() {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // The step itself, from the CPU's state just before it, and what
        // the stretch's steps wrote before it.
        let (mut before, mut before_pad) = (start, start_pad.clone());
        let mut written: std::collections::HashMap<usize, (usize, u64)> = Default::default();
        for _ in 0..hi - 1 {
            let i = walk::step_inputs(&before, &before_pad);
            for (w, m) in i.words.iter().zip(i.m) {
                written.insert(*w, (before.steps, m));
            }
            walk::walk_steps(&mut before, &mut before_pad, 1);
        }
        let i = walk::step_inputs(&before, &before_pad);
        let (mut after, mut after_pad) = (before, before_pad.clone());
        walk::walk_steps(&mut after, &mut after_pad, 1);
        g.run_one_to(v, &start, &start_pad, a + hi)?;
        let gpu = g.states(1)?[0];
        report.push(format!(
            "The first wrong step is step {} of the walk (register r{}, operation {}).",
            before.steps,
            i.s,
            walk::OP_NAMES[i.op as usize]
        ));
        if hi == WALK_REMIX {
            report.push("It is the stretch's last step, so the re-mix follows it.".into());
            let (gr, gp) = g.test_remix(&[after_regs_before_remix(&before, &before_pad)])?[0];
            if (gr, gp) != walk::remix_of(&after_regs_before_remix(&before, &before_pad)) {
                report.push("The GPU's re-mix of these registers alone is wrong.".into());
            }
        }
        report.push(format!("  x={:#018x} k=r{}={:#018x} v={:#018x}", i.x, i.t, i.k, i.v));
        for (n, (w, m)) in i.words.iter().zip(i.m).enumerate() {
            let when = match written.get(w) {
                Some((at, _)) => format!("written by step {at}, {} step(s) before", before.steps - at),
                None => "not written since step ".to_string() + &a.to_string(),
            };
            report.push(format!("  m{n} = pad word {w} = {m:#018x}, {when}"));
        }
        for d in differences(g, &after, &after_pad)? {
            report.push(format!("  after it: {d}"));
        }

        // The operation alone, on the same inputs.
        let alone = g.test_ops(&[(i.v, i.k)])?[0];
        let want = ops_reference(i.v, i.k);
        let slot = [0usize, 1, 2, 3, 5, 6, 7, 8][i.op as usize];
        report.push(if alone[slot] == want[slot] && (i.op != 3 || alone[4] == want[4]) {
            format!("Alone, on these inputs, the GPU's {} is right: the fault is in the step around it.", walk::OP_NAMES[i.op as usize])
        } else {
            format!("Alone, on these inputs, the GPU's {} is wrong: {:#018x}, not {:#018x}.", walk::OP_NAMES[i.op as usize], alone[slot], want[slot])
        });

        // Did it read a word as it was before an earlier step wrote it?
        for n in 0..4 {
            if let Some((at, old)) = written.get(&i.words[n]) {
                let mut m = i.m;
                m[n] = *old;
                let (y, _) = walk::apply_op(i.op, i.x ^ m[0] ^ m[2], i.k);
                if y ^ m[1] ^ m[3] == gpu.c {
                    report.push(format!(
                        "The GPU's result is what the step gives with m{n} as it was before step {at} wrote it, {} step(s) earlier: the GPU read a stale value.",
                        before.steps - at
                    ));
                }
            }
        }
        return Ok(report.join("\n"));
    }
    Ok("Run again, the whole walk came out right: the fault comes and goes (a GPU reset, or a race). Look for v3d in: sudo dmesg | tail -50".into())
}

/// The registers after `st`'s next step, before the re-mix that follows it.
fn after_regs_before_remix(st: &WalkState, pad: &walk::Pad) -> [u64; 8] {
    let i = walk::step_inputs(st, pad);
    let (y, h) = walk::apply_op(i.op, i.v, i.k);
    let mut r = st.r;
    r[i.s] = y ^ i.m[1] ^ i.m[3];
    r[i.t] ^= h;
    r
}

/// Cores' worth of CPU time something else is using, sampled over a second.
fn busy_cores() -> Option<f64> {
    let read = || -> Option<(u64, u64)> {
        let stat = std::fs::read_to_string("/proc/stat").ok()?;
        let v: Vec<u64> = stat.lines().next()?.split_whitespace().skip(1).filter_map(|x| x.parse().ok()).collect();
        let idle = v.get(3)? + v.get(4).unwrap_or(&0);
        Some((v.iter().sum(), idle))
    };
    let (t0, i0) = read()?;
    std::thread::sleep(Duration::from_secs(1));
    let (t1, i1) = read()?;
    let total = (t1 - t0).max(1) as f64;
    Some((1.0 - (i1 - i0) as f64 / total) * cores() as f64)
}

// ── The benchmark ───────────────────────────────────────────────────────────

/// What `midstate bench-pow --gpu` was asked for.
#[derive(Clone, Debug)]
pub struct BenchOptions {
    /// Part of the GPU's name; the first hardware GPU otherwise.
    pub adapter: Option<String>,
    /// The most walks at once to try.
    pub max_walks: u32,
    /// Seconds for each CPU measurement; each GPU point takes a fifth.
    pub seconds: u64,
    /// CPU threads (0: every core).
    pub threads: usize,
    /// Also run the CPU and the GPU together.
    pub with_cpu: bool,
    /// The board's temperature and power, in words, for before and after.
    pub board: fn() -> String,
    /// A kernel to run instead of [`kernel_source`] (an edited copy).
    pub wgsl: Option<String>,
}

/// One measurement: attempts a second, and the steps a second behind it.
#[derive(Clone, Copy, Debug)]
struct Point {
    variant: Variant,
    walks: u32,
    attempts: f64,
    steps: f64,
}

/// Runs `walks` walks of `v` for about `secs` from fresh seeds: attempts a
/// second, from the steps done (seeding is the CPU's and not counted).
fn measure(g: &mut GpuWalker, v: Variant, walks: u32, secs: f64, salt: u64) -> Result<Point, String> {
    let mining_hash = *blake3::hash(b"midstate bench-pow --gpu").as_bytes();
    let (mut timed, mut chunk, mut busy, mut round) = (0f64, 8u32, 0f64, 0u64);
    while busy < secs {
        // Fresh walks whenever the last ran out of steps; seeding them is
        // the CPU's work and not counted.
        let nonces: Vec<u64> = (0..walks as u64).map(|i| (salt << 40) | (round << 24) | i).collect();
        round += 1;
        g.load(&mining_hash, &nonces)?;
        // The first launch compiles the kernel and warms the caches: not timed.
        g.run(v, walks, 0, 8)?;
        let mut done = 8u32;
        while busy < secs && done + chunk <= WALK_STEPS as u32 {
            // Regrouping is part of the work, so it is timed with it.
            let begun = Instant::now();
            let c = chunk;
            chunk = g.run_span(v, walks, done, c, c)?;
            busy += begun.elapsed().as_secs_f64();
            timed += c as f64;
            done += c;
        }
    }
    let steps = timed * walks as f64 / busy.max(1e-9);
    Ok(Point { variant: v, walks, attempts: steps / WALK_STEPS as f64, steps })
}

/// This machine's cores.
fn cores() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// `midstate bench-pow --gpu`: checks the kernel against the known answers
/// on this GPU, finds its best workgroup and number of walks, probes what
/// limits it, and sets it against the CPU. Prints as it goes.
pub fn bench(o: &BenchOptions) {
    let found = gpus();
    println!("GPUs wgpu finds:");
    if found.is_empty() {
        println!("  none");
    }
    for i in &found {
        println!("  {} [{} via {}{}]{}", i.name, i.kind, i.backend, if i.driver.is_empty() { String::new() } else { format!(", {}", i.driver) }, if i.software { " (software)" } else { "" });
    }
    let mut g = match GpuWalker::open(o.adapter.as_deref(), o.max_walks) {
        Ok(g) => g,
        Err(e) => {
            println!("\nNo GPU to measure: {e}.");
            println!("On a Raspberry Pi 5 the GPU needs Mesa's Vulkan driver: sudo apt install mesa-vulkan-drivers");
            println!("and a user in the render group (groups; sudo usermod -aG render $USER, then log in again).");
            return;
        }
    };
    println!(
        "\nMeasuring {} ({}, {}). Room for {} walks of 256 KiB; workgroups up to {}.",
        g.info.name, g.info.backend, g.info.kind, g.capacity, g.max_workgroup
    );
    if let Some(src) = &o.wgsl {
        println!("Running the kernel given (--wgsl), not the built-in one.");
        g.use_source(src.clone());
    }
    if let Some(busy) = busy_cores().filter(|b| *b > 0.5) {
        println!("Something else is using {busy:.1} cores' worth of CPU now. Stop it for clean numbers (the miner: sudo systemctl stop midstate-hasher).");
    }
    let wg0 = 64.min(g.max_workgroup);

    // 1. Uploads, the kernel's parts, then the known answers as whole walks.
    println!("\nChecking the kernel on this GPU, part by part:");
    let mut right = true;
    match check_uploads(&mut g) {
        Ok((direct, copied)) => {
            println!(
                "  Uploads into memory the kernel has written: seen {direct} times of 4 when written straight in, {copied} of 4 through the copy kernel{}.",
                if direct < 4 { " (this driver's fault, worked around)" } else { "" }
            );
            if copied < 4 {
                right = false;
                println!("  WRONG: even through the copy kernel, the kernel did not see what was uploaded.");
            }
        }
        Err(e) => {
            println!("  The upload check failed to run: {e}");
            return;
        }
    }
    match check_ops(&mut g, 1 << 20) {
        Ok(p) if p.is_empty() => println!("  Operations: right on {} operand pairs.", 1 << 20),
        Ok(p) => {
            right = false;
            for line in p {
                println!("  WRONG {line}");
            }
        }
        Err(e) => {
            println!("  The operations test failed to run: {e}");
            return;
        }
    }
    match check_remix(&mut g, 1 << 16) {
        Ok(None) => println!("  Re-mix: right on {} register sets.", 1 << 16),
        Ok(Some(p)) => {
            right = false;
            println!("  WRONG re-mix: {p}");
        }
        Err(e) => println!("  The re-mix test failed to run: {e}"),
    }
    println!("Known answers, as whole walks, one at a time:");
    for (k, (m, nonce, want)) in walk::GOLDEN.iter().enumerate() {
        let mut mh = [0u8; 32];
        if hex::decode_to_slice(m, &mut mh).is_err() {
            continue;
        }
        let started = Instant::now();
        match g.hash(Variant::walk(wg0), &mh, &[*nonce]) {
            Ok(h) if hex::encode(h[0]) == *want => println!("  answer {k}: right ({:.1}s)", started.elapsed().as_secs_f64()),
            Ok(h) => {
                right = false;
                println!("  answer {k}: WRONG, {} not {want} ({:.1}s)", hex::encode(h[0]), started.elapsed().as_secs_f64());
                println!("Finding the first step where the GPU leaves the CPU (a walk and a few reruns of part of it)...");
                match find_divergence(&mut g, Variant::walk(wg0), &mh, *nonce) {
                    Ok(r) => {
                        for line in r.lines() {
                            println!("  {line}");
                        }
                    }
                    Err(e) => println!("  The search failed: {e}"),
                }
            }
            Err(e) => {
                println!("  answer {k}: the GPU failed: {e}");
                return;
            }
        }
    }
    if !right {
        println!("\nTHE KERNEL IS WRONG ON THIS GPU. Please send this whole output.");
        println!("The rates below are of a kernel that does the work but gets some of it wrong: what it costs, not yet what it earns.");
    }

    // 2. Workgroup size and walks at once.
    let per = (o.seconds as f64 / 5.0).max(1.0);
    println!("\nWalk rate by workgroup and walks at once ({per:.0}s each):");
    println!("  {:>9} {:>6} {:>14} {:>16}", "workgroup", "walks", "attempts/s", "steps/s");
    let mut best: Option<Point> = None;
    let mut salt = 1u64;
    for wg in [8u32, 16, 32, 64, 128, 256] {
        if wg > g.max_workgroup {
            continue;
        }
        let mut walks = wg;
        let mut flat = 0;
        let mut last = 0.0;
        while walks <= g.capacity {
            salt += 1;
            match measure(&mut g, Variant::walk(wg), walks, per, salt) {
                Ok(p) => {
                    println!("  {:>9} {:>6} {:>14.2} {:>16.0}", wg, walks, p.attempts, p.steps);
                    if best.is_none_or(|b| p.attempts > b.attempts) {
                        best = Some(p);
                    }
                    flat = if p.attempts < last * 1.05 { flat + 1 } else { 0 };
                    last = last.max(p.attempts);
                    if flat >= 2 {
                        break;
                    }
                }
                Err(e) => {
                    println!("  {:>9} {:>6}  failed: {e}", wg, walks);
                    break;
                }
            }
            walks *= 2;
        }
    }
    let Some(mut best) = best else {
        println!("Nothing could be measured.");
        return;
    };
    // The best workgroup again, with the walks regrouped by their programs at
    // every re-mix. Grouping needs many walks to each group of lanes, so it
    // starts at eight groups' worth.
    println!("Grouping the walks by where their costly operations fall (workgroup {}):", best.variant.workgroup);
    let grouped = Variant { grouped: true, ..best.variant };
    let mut walks = (best.walks).max(8 * 16).min(g.capacity);
    let (mut last, mut flat) = (0.0, 0);
    loop {
        match measure(&mut g, grouped, walks, per, 500 + walks as u64) {
            Ok(p) => {
                println!("  {:>9} {:>6} {:>14.2} {:>16.0}  grouped", grouped.workgroup, walks, p.attempts, p.steps);
                if p.attempts > best.attempts {
                    best = p;
                }
                flat = if p.attempts < last * 1.05 { flat + 1 } else { 0 };
                last = f64::max(last, p.attempts);
            }
            Err(e) => {
                println!("  grouping failed: {e}");
                break;
            }
        }
        if flat >= 2 || walks * 2 > g.capacity {
            break;
        }
        walks *= 2;
    }

    println!(
        "Best: workgroup {}, {} walks at once{}: {:.2} attempts/s.",
        best.variant.workgroup,
        best.walks,
        if best.variant.grouped { ", grouped" } else { "" },
        best.attempts
    );

    // 3. What limits it.
    let probe = |g: &mut GpuWalker, real_ops: bool, lines: u32, salt: u64| {
        measure(g, Variant { real_ops, lines, ..best.variant }, best.walks, per, salt).map(|p| p.attempts)
    };
    let mem = probe(&mut g, false, WALK_LINES as u32, 900);
    let alu = probe(&mut g, true, 8, 901);
    let floor = probe(&mut g, false, 8, 902);
    println!("\nWhat limits it (the same walks, changed on purpose):");
    let show = |r: &Result<f64, String>| match r {
        Ok(a) => format!("{a:>10.2} attempts/s ({:.1}x the walk)", a / best.attempts),
        Err(e) => format!("failed: {e}"),
    };
    println!("  memory only (every operation an XOR):   {}", show(&mem));
    println!("  arithmetic only (pad of 8 lines, cached): {}", show(&alu));
    println!("  neither (XOR, 8 lines):                 {}", show(&floor));
    if let (Ok(m), Ok(a)) = (&mem, &alu) {
        let verdict = if *m < best.attempts * 1.25 && *a > best.attempts * 1.5 {
            "memory: taking the arithmetic away barely helps, so a better kernel cannot do much"
        } else if *a < best.attempts * 1.25 && *m > best.attempts * 1.5 {
            "arithmetic: the 64-bit emulation, the division, AES and diverging programs"
        } else {
            "both memory and arithmetic"
        };
        println!("  Bound by {verdict}.");
    }
    // Each step reads four words and writes four, each in its own line.
    let accesses = best.steps * 8.0;
    println!(
        "  At the best point: {:.1} M steps/s, {:.0} M random 8-byte accesses/s ({:.2} GB/s if each moves a 64-byte line).",
        best.steps / 1e6,
        accesses / 1e6,
        accesses * 64.0 / 1e9
    );

    // 4. The CPU, for scale.
    let threads = if o.threads == 0 { cores() } else { o.threads };
    let (cpu, good) = walk::bench(threads, Duration::from_secs(o.seconds.max(1)));
    println!(
        "\nThis CPU, {threads} threads: {cpu:.1} attempts/s{}. The GPU adds {:.1}% to it.",
        if good { "" } else { " (and a core got a known answer wrong)" },
        100.0 * best.attempts / cpu
    );

    // 5. Both at once: what the board does when both work, and how hot.
    if o.with_cpu {
        println!("\nThe CPU and the GPU together, {}s (before: {}):", o.seconds, (o.board)());
        let cpu_side = std::thread::spawn({
            let secs = o.seconds;
            move || walk::bench(threads, Duration::from_secs(secs.max(1))).0
        });
        let gpu_side = measure(&mut g, best.variant, best.walks, o.seconds as f64, 777);
        let cpu_with = cpu_side.join().unwrap_or(0.0);
        match gpu_side {
            Ok(p) => {
                println!("  CPU {cpu_with:.1} attempts/s ({:+.1}% against alone)", 100.0 * (cpu_with / cpu - 1.0));
                println!("  GPU {:.2} attempts/s ({:+.1}% against alone)", p.attempts, 100.0 * (p.attempts / best.attempts - 1.0));
                println!(
                    "  Together {:.1} attempts/s: {:+.2}% against the CPU alone. After: {}.",
                    cpu_with + p.attempts,
                    100.0 * ((cpu_with + p.attempts) / cpu - 1.0),
                    (o.board)()
                );
            }
            Err(e) => println!("  the GPU failed: {e}"),
        }
    }
    println!(
        "\nSummary{}: {} | best wg {} × {} walks{} = {:.2} att/s | mem-only {} | alu-only {} | CPU {:.1} att/s",
        if right { "" } else { " (KERNEL WRONG HERE)" },
        g.info.name,
        best.variant.workgroup,
        best.walks,
        if best.variant.grouped { " grouped" } else { "" },
        best.attempts,
        mem.as_ref().map_or("failed".into(), |a| format!("{a:.2}")),
        alu.as_ref().map_or("failed".into(), |a| format!("{a:.2}")),
        cpu
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn programs_pack_and_launches_aim_at_their_target() {
        for z in [0u64, 1, 40_319, 40_320, u64::MAX, 0x1234_5678_9abc_def0] {
            let p = walk::program_of(z);
            assert_eq!(unpack_program(pack_program(&p)), p);
        }
        assert_eq!(pack_program(&[0, 1, 2, 3, 4, 5, 6, 7]), 0xfac688, "the kernel's identity");
        assert_eq!((1u64 << 32) % 40_320, 256, "the kernel's 2^32 mod 8!");
        // The draw's digits read only z mod 8!.
        for z in [5u64, 99_999, u64::MAX - 7] {
            assert_eq!(walk::program_of(z), walk::program_of(z % 40_320));
        }
        assert_eq!(next_chunk(64, Duration::from_millis(1)), 128, "at most doubling");
        let remix = WALK_REMIX as u32;
        assert_eq!(next_chunk(remix, Duration::from_millis(1)), remix, "at most a re-mix's worth");
        assert_eq!(next_chunk(1024, Duration::from_millis(240)), 256);
        assert_eq!(next_chunk(64, Duration::from_secs(5)), 8);
        // One source, every variant's differences in its pipeline constants.
        let src = kernel_source();
        assert!(src.contains("fn walk_steps") && src.contains("r7 = vec2<u32>(o[14], o[15]);"));
        assert!(src.contains("if (!LIMITED || P.start + n * 8u + 7u < P.limit) {"));
        for (name, _) in Variant::walk(64).constants() {
            assert!(src.contains(&format!("override {name}:")), "{name} is declared");
        }
        let probe = Variant { workgroup: 16, real_ops: false, lines: 8, grouped: true, limit: true };
        assert_eq!(probe.constants(), [("WG", 16.0), ("LINES", 8.0), ("REAL_OPS", 0.0), ("GROUPED", 1.0), ("LIMITED", 1.0)]);

        // Grouping sorts by the AES round's step first, and is a permutation.
        let progs: Vec<[u8; 8]> = (0..500u64).map(|i| walk::program_of(i.wrapping_mul(0x9e37_79b9_7f4a_7c15))).collect();
        let order = group_order(&progs);
        let mut seen = order.clone();
        seen.sort();
        assert_eq!(seen, (0..500).collect::<Vec<u32>>());
        let aes_at = |w: u32| progs[w as usize].iter().position(|o| *o == 3).unwrap();
        assert!(order.windows(2).all(|p| aes_at(p[0]) <= aes_at(p[1])));
        // Groups of 16 lanes: how many distinct steps hold an AES round.
        let spread = |o: &[u32]| o.chunks(16).map(|g| g.iter().map(|&w| aes_at(w)).collect::<std::collections::HashSet<_>>().len()).sum::<usize>() as f64 / o.chunks(16).count() as f64;
        let plain: Vec<u32> = (0..500).collect();
        assert!(spread(&order) < 1.5 && spread(&plain) > 6.0, "grouped {:.2}, as they come {:.2}", spread(&order), spread(&plain));
    }

    /// The kernel against the CPU on a GPU, when there is one (a software
    /// one will do: `VK_ICD_FILENAMES` pointing at SwiftShader or lavapipe,
    /// and `MIDSTATE_TEST_GPU` naming it). Skipped without.
    #[test]
    fn the_gpu_walk_is_the_walk() {
        let name = std::env::var("MIDSTATE_TEST_GPU").ok();
        let mut g = match GpuWalker::open(name.as_deref(), 8) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no GPU to test the kernel on ({e}): skipped");
                return;
            }
        };
        eprintln!("testing the kernel on {} ({}, {})", g.info.name, g.info.backend, g.info.kind);

        // Uploads reach the kernel (straight in, too, on a driver without
        // the Pi's fault).
        assert_eq!(check_uploads(&mut g).unwrap(), (4, 4));

        // Every operation, and the re-mix, against the CPU. The re-mix is
        // the CPU's own (walk::remix_of), a second BLAKE3 not written here.
        assert_eq!(check_ops(&mut g, 8192).unwrap(), Vec::<String>::new());
        assert_eq!(check_remix(&mut g, 256).unwrap(), None);

        // Steps against the CPU, across the first re-mix, then whole walks.
        let m = walk::GOLDEN.map(|(m, _, _)| {
            let mut b = [0u8; 32];
            hex::decode_to_slice(m, &mut b).unwrap();
            b
        });
        let nonces = [7u64, 8, 9];
        let mut states = g.load(&m[2], &nonces).unwrap();
        let v = Variant::walk(64.min(g.max_workgroup));
        g.run(v, 3, 0, WALK_REMIX as u32).unwrap();
        g.run(v, 3, WALK_REMIX as u32, 16).unwrap();
        let gpu = g.states(3).unwrap();
        for (i, nonce) in nonces.iter().enumerate() {
            let mut pad = walk::new_pad();
            states[i] = walk::walk_start(&m[2], *nonce, &mut pad);
            walk::walk_steps(&mut states[i], &mut pad, WALK_REMIX + 16);
            assert_eq!((gpu[i].r, gpu[i].prog, gpu[i].c), (states[i].r, states[i].prog, states[i].c), "nonce {nonce} after {} steps", WALK_REMIX + 16);
        }
        for (k, (_, nonce, want)) in walk::GOLDEN.iter().enumerate() {
            let got = g.hash(v, &m[k], &[*nonce]).unwrap();
            assert_eq!(hex::encode(got[0]), *want, "known answer {k} (nonce {nonce})");
        }

        // Grouped: which lane runs which walk changes at every re-mix, and
        // the hashes do not.
        let nonces: Vec<u64> = (100..108).collect();
        let grouped = g.hash(Variant { grouped: true, ..v }, &m[1], &nonces).unwrap();
        for (nonce, h) in nonces.iter().zip(&grouped) {
            assert_eq!(*h, walk::walk_hash(&m[1], *nonce), "grouped, nonce {nonce}");
        }

        // A load in two upload batches: walks past the first batch are right.
        let per_batch = (UPLOAD_WORDS / PAD_WORDS) as u64;
        let mut big = GpuWalker::open(name.as_deref(), per_batch as u32 + 2).unwrap();
        let nonces: Vec<u64> = (0..per_batch + 2).collect();
        let hashes = big.hash(v, &m[2], &nonces).unwrap();
        for k in [per_batch - 1, per_batch, per_batch + 1] {
            assert_eq!(hashes[k as usize], walk::walk_hash(&m[2], k), "walk {k}, across upload batches");
        }
        drop(big);

        // From any state to any step, as the search for a fault runs it.
        let mut pad = walk::new_pad();
        let mut start = walk::walk_start(&m[0], 5, &mut pad);
        walk::walk_steps(&mut start, &mut pad, WALK_REMIX);
        for n in [1usize, 5, 8, 13, WALK_REMIX - 1, WALK_REMIX] {
            g.run_one_to(v, &start, &pad, start.steps + n).unwrap();
            let (mut st, mut p) = (start, pad.clone());
            walk::walk_steps(&mut st, &mut p, n);
            assert_eq!(differences(&g, &st, &p).unwrap(), Vec::<String>::new(), "{n} steps on");
        }

        // The search: the right kernel is right throughout, and a wrong one
        // (every operation an XOR) is caught at its first step whose
        // operation is not one.
        let right = find_divergence(&mut g, v, &m[0], 5).unwrap();
        assert!(right.contains("came out right"), "{right}");
        let xor_only = Variant { real_ops: false, ..v };
        let mut probe = walk::new_pad();
        let mut st = walk::walk_start(&m[0], 5, &mut probe);
        let first = loop {
            let i = walk::step_inputs(&st, &probe);
            let (y, h) = walk::apply_op(i.op, i.v, i.k);
            if (y, h) != (i.v ^ i.k, 0) {
                break st.steps;
            }
            walk::walk_steps(&mut st, &mut probe, 1);
        };
        let report = find_divergence(&mut g, xor_only, &m[0], 5).unwrap();
        assert!(report.contains(&format!("The first wrong step is step {first} of the walk")), "{report}");
        assert!(report.contains("got it wrong again") && report.contains("the fault is in the step around it"), "{report}");
    }
}
