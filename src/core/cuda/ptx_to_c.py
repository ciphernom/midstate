#!/usr/bin/env python3
"""Translates walk_miner.ptx into C that does what the PTX says, so CI can run
the miner's kernels as compiled, not only their source (ptx_check.cpp).

    python3 ptx_to_c.py walk_miner.ptx > walk_miner_ptx.c

host_check.cpp runs walk_miner.cu compiled by g++, which tests the source. It
cannot catch clang compiling that source to wrong PTX, and that happened: clang
18 lowered the walk's 64-bit rotate to two 64-bit shifts without reducing the
amount mod 64, PTX clamps such shifts to zero, and every walk came out wrong on
the GPU. Translated here instruction by instruction, with PTX's rules (shifts
of the register width or more give zero, funnel shifts take their amount mod
32, bfe), that PTX gives the same wrong hash a Quadro M6000 gave.

Each kernel becomes `void name(const uint64_t *param)`, its parameters in
order. Every state space is ordinary host memory: a kernel's pointers are host
pointers, `ms_te` and `ms_keys` are arrays the caller fills, and a kernel's
shared table is a static array. One thread runs at a time, `tid_x`, `ntid_x`
and `ctaid_x` saying which, and a barrier does nothing, so a block's shared
table must be filled by one thread before another reads it (ptx_check.cpp
runs blocks of one thread first, as host_check.cpp does).

Only the instructions walk_miner.ptx uses are translated; anything else stops
the translation, naming the instruction.
"""
import re
import sys

PRELUDE = r"""
#include <stdbool.h>
#include <stdint.h>
extern uint32_t tid_x, ntid_x, ctaid_x;
uint8_t ms_te[1024] __attribute__((aligned(16)));
uint8_t ms_keys[96] __attribute__((aligned(16)));
static inline uint32_t shl32(uint32_t a, uint32_t n) { return n >= 32 ? 0 : a << n; }
static inline uint64_t shl64(uint64_t a, uint32_t n) { return n >= 64 ? 0 : a << n; }
static inline uint32_t shr32(uint32_t a, uint32_t n) { return n >= 32 ? 0 : a >> n; }
static inline uint64_t shr64(uint64_t a, uint32_t n) { return n >= 64 ? 0 : a >> n; }
static inline uint32_t sar32(uint32_t a, uint32_t n) { return (uint32_t)((int32_t)a >> (n >= 32 ? 31 : n)); }
static inline uint64_t sar64(uint64_t a, uint32_t n) { return (uint64_t)((int64_t)a >> (n >= 64 ? 63 : n)); }
/* shf.l.wrap: the top half of {b,a} << (c mod 32); shf.r.wrap: the bottom half of {b,a} >> (c mod 32). */
static inline uint32_t shfl32(uint32_t a, uint32_t b, uint32_t c) { uint32_t n = c & 31; return n ? (b << n) | (a >> (32 - n)) : b; }
static inline uint32_t shfr32(uint32_t a, uint32_t b, uint32_t c) { uint32_t n = c & 31; return n ? (a >> n) | (b << (32 - n)) : a; }
static inline uint32_t bfe32(uint32_t a, uint32_t pos, uint32_t len) {
  pos &= 255; len &= 255;
  if (len == 0 || pos >= 32) return 0;
  uint32_t v = a >> pos;
  return len >= 32 ? v : v & ((1u << len) - 1);
}
static inline uint64_t mulhi64(uint64_t a, uint64_t b) { return (uint64_t)(((unsigned __int128)a * b) >> 64); }
"""

CTYPE = {'32': 'uint32_t', '64': 'uint64_t'}
SIGNED = {'s32': 'int32_t', 'u32': 'uint32_t', 'b32': 'uint32_t', 's64': 'int64_t', 'u64': 'uint64_t', 'b64': 'uint64_t'}
BINARY = {'add': '+', 'sub': '-', 'and': '&', 'or': '|', 'xor': '^'}
COMPARE = {'eq': '==', 'ne': '!=', 'gt': '>', 'lt': '<', 'ge': '>=', 'le': '<='}


def width(ptx_type):
    return '64' if ptx_type.endswith('64') else '32'


def fail(msg, line):
    sys.exit(f'ptx_to_c: {msg}: {line.strip()}')


def operand(a, w, line):
    """A source operand as a C expression of width `w`."""
    m = re.fullmatch(r'%(r|rd|p)(\d+)', a)
    if m:
        return f'{m.group(1)}[{m.group(2)}]'
    if a in ('%tid.x', '%ntid.x', '%ctaid.x'):
        return a[1:].replace('.', '_')
    m = re.fullmatch(r'%(\w+)', a)
    if m:
        return f'L_{m.group(1)}'  # a register declared in a { } block
    if re.fullmatch(r'-?\d+|0x[0-9a-fA-F]+', a):
        return f'(({CTYPE[w]}){int(a, 0) & ((1 << int(w)) - 1)}ULL)'
    if re.fullmatch(r'\w+', a):
        return f'((uint64_t)(uintptr_t){a})'  # a symbol: its address
    fail(f'operand {a!r}', line)


def address(a, line):
    """An address operand: ('param', index) or ('mem', C expression)."""
    inner = a.strip()[1:-1]
    m = re.fullmatch(r'\w+_param_(\d+)', inner)
    if m:
        return 'param', int(m.group(1))
    m = re.fullmatch(r'(%\w+|\w+)(?:\+(-?\d+))?', inner)
    if not m:
        fail(f'address {a!r}', line)
    return 'mem', f'({operand(m.group(1), "64", line)} + (uint64_t)({int(m.group(2) or 0)}LL))'


def instruction(op, args, line):
    if op in ('bra', 'bra.uni'):
        return f'goto {args[0].lstrip("$")};'
    if op == 'bar.sync':
        return ';'
    if op == 'ret':
        return 'return;'
    ty = op.split('.')[-1]
    w = width(ty)
    T = CTYPE[w]
    base = op.split('.')[0]
    d = operand(args[0], w, line) if args and not args[0].startswith('[') else None  # st's is an address
    src = lambda i, ww=w: operand(args[i], ww, line)
    if base == 'ld':
        kind, where = address(args[1], line)
        return f'{d} = ({T})param[{where}];' if kind == 'param' else f'{d} = *({T} *)(uintptr_t){where};'
    if base == 'st':
        _, where = address(args[0], line)
        return f'*({T} *)(uintptr_t){where} = ({T}){src(1)};'
    if op in ('mov.u32', 'mov.u64', 'mov.b32', 'mov.b64'):
        return f'{d} = ({T}){src(1)};'
    if op == 'cvta.to.global.u64':
        return f'{d} = {src(1)};'
    if op == 'cvt.u64.u32':
        return f'{d} = (uint64_t)(uint32_t){src(1, "32")};'
    if op == 'cvt.u32.u64':
        return f'{d} = (uint32_t){src(1, "64")};'
    if base in BINARY and len(op.split('.')) == 2:
        return f'{d} = ({T})(({T}){src(1)} {BINARY[base]} ({T}){src(2)});'
    if op in ('mul.lo.s32', 'mul.lo.u32', 'mul.lo.s64', 'mul.lo.u64'):
        return f'{d} = ({T})(({T}){src(1)} * ({T}){src(2)});'
    if op == 'mul.wide.u32':
        return f'{d} = (uint64_t)(uint32_t){src(1, "32")} * (uint64_t)(uint32_t){src(2, "32")};'
    if op == 'mul.hi.u64':
        return f'{d} = mulhi64({src(1)}, {src(2)});'
    if op in ('mad.lo.s32', 'mad.lo.u32'):
        return f'{d} = (uint32_t)((uint32_t){src(1)} * (uint32_t){src(2)} + (uint32_t){src(3)});'
    if op in ('div.u32', 'div.u64', 'rem.u32', 'rem.u64'):
        sym = '/' if base == 'div' else '%'
        return f'{d} = ({T})(({T}){src(1)} {sym} ({T}){src(2)});'
    if op in ('shl.b32', 'shl.b64'):
        return f'{d} = shl{w}({src(1)}, (uint32_t){src(2, "32")});'
    if op in ('shr.u32', 'shr.b32', 'shr.u64', 'shr.b64'):
        return f'{d} = shr{w}({src(1)}, (uint32_t){src(2, "32")});'
    if op in ('shr.s32', 'shr.s64'):
        return f'{d} = sar{w}({src(1)}, (uint32_t){src(2, "32")});'
    if op == 'shf.l.wrap.b32':
        return f'{d} = shfl32({src(1)}, {src(2)}, {src(3)});'
    if op == 'shf.r.wrap.b32':
        return f'{d} = shfr32({src(1)}, {src(2)}, {src(3)});'
    if op == 'bfe.u32':
        return f'{d} = bfe32({src(1)}, {src(2)}, {src(3)});'
    if base == 'setp':
        _, cmp, t = op.split('.')
        ww = width(t)
        ct = SIGNED[t]
        return (f'{operand(args[0], "32", line)} = ({ct})({CTYPE[ww]}){src(1, ww)} '
                f'{COMPARE[cmp]} ({ct})({CTYPE[ww]}){src(2, ww)};')
    if op in ('selp.b32', 'selp.b64'):
        return f'{d} = {operand(args[3], "32", line)} ? ({T}){src(1)} : ({T}){src(2)};'
    fail(f'instruction {op!r}', line)


def kernel(chunk):
    name = re.match(r'\.visible \.entry (\w+)\(', chunk).group(1)
    body = chunk[chunk.index('{') + 1:]
    c = [f'void {name}(const uint64_t *param) {{']
    for raw in body.split('\n'):
        line = raw.split('//')[0].strip()
        if not line:
            continue
        if raw.startswith('}'):  # the kernel's own closing brace
            break
        if line in ('{', '}'):
            c.append(line)
            continue
        m = re.fullmatch(r'\{ \.reg \.b32 tmp; mov\.b64 \{tmp, %r(\d+)\}, %rd(\d+); \}', line)
        if m:  # the high half of a 64-bit register
            c.append(f'r[{m.group(1)}] = (uint32_t)(rd[{m.group(2)}] >> 32);')
            continue
        m = re.fullmatch(r'\{ \.reg \.b32 tmp; mov\.b64 \{%r(\d+), tmp\}, %rd(\d+); \}', line)
        if m:  # the low half
            c.append(f'r[{m.group(1)}] = (uint32_t)rd[{m.group(2)}];')
            continue
        m = re.fullmatch(r'\.reg \.(\w+)\s+%(\w+)<(\d+)>;', line)
        if m:
            t, reg, n = m.groups()
            ct = 'bool' if t == 'pred' else CTYPE[width(t)]
            c.append(f'{ct} {reg}[{n}];')
            continue
        m = re.fullmatch(r'\.reg \.(\w+)\s+%(\w+);', line)
        if m:
            c.append(f'{CTYPE[width(m.group(1))]} L_{m.group(2)};')
            continue
        m = re.fullmatch(r'\.shared \.align \d+ \.b8 (\w+)\[(\d+)\];', line)
        if m:
            c.append(f'static uint8_t {m.group(1)}[{m.group(2)}] __attribute__((aligned(16)));')
            continue
        m = re.fullmatch(r'\$(\w+):', line)
        if m:
            c.append(f'{m.group(1)}:;')
            continue
        if line.startswith('.pragma'):
            continue
        if line.startswith('.'):
            fail('directive', raw)
        guard = ''
        m = re.fullmatch(r'@(!?)%p(\d+)\s+(.*)', line)
        if m:
            guard = f'if ({m.group(1)}p[{m.group(2)}]) '
            line = m.group(3)
        parts = line.rstrip(';').split(None, 1)
        args = [a.strip() for a in re.split(r',(?![^\[]*\])', parts[1])] if len(parts) > 1 else []
        c.append(guard + instruction(parts[0], args, raw))
    c.append('}')
    return c


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    text = open(sys.argv[1], encoding='utf-8').read()
    chunks = re.split(r'\n(?=\.visible \.entry )', text)
    out = [PRELUDE]
    for chunk in chunks[1:]:
        out.extend(kernel(chunk))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
