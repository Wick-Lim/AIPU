#!/usr/bin/env python3
"""fp8_huff.py -- OFFLINE canonical-Huffman encoder + golden generator for the
on-chip streaming weight decompressor src/weight_decomp.v (IMPROVEMENT_PLAN P2.1).

WHY THIS FILE EXISTS
  weight_decomp.v is a streaming lossless CANONICAL (static) Huffman decoder over
  a 257-symbol alphabet (0..255 FP8 E4M3 bytes + one EOB=256 symbol).  Its header
  names THIS module -- "the matching OFFLINE ENCODER (length-limited package-merge
  Huffman + canonical-code assignment) lives in tools/fp8_huff.py; the TB compresses
  with it and checks bit-exact round-trip" -- but the file was never written, so the
  decoder shipped FUNCTIONALLY UNVERIFIED.  This is the matching encoder + the golden
  emitter that test/weight_decomp_tb.v consumes.

WHAT IT PRODUCES  (mirrors tools/q4k_matmul_gen.py's whitespace-hex vector idiom)
  build/weight_decomp_vec.txt -- one canonical-Huffman table + compressed byte
  stream + expected decoded byte stream per test, read by the Verilog TB via $fscanf.

THE CONTRACT (must be bit-exact to src/weight_decomp.v's decode recurrence)
  Tables:
    count_table[len]  (len=1..MAXLEN): #codewords of each length.
    symbol_table[i]   (i=0..ncodes-1): symbols in CANONICAL order = sorted by
                       (length, then symbol value).
  Canonical code assignment (the exact inverse of the decoder's per-bit loop):
    first_code[1]=0 ;  first_code[l+1] = (first_code[l] + count[l]) << 1
    the k-th (0-based) length-l symbol (symbols of a length taken in increasing
    symbol order) gets code first_code[l]+k, and sits at symbol_table[cindex[l]+k]
    where cindex[l] = sum_{m<l} count[m].
  Bit packing: MSB-first -- the first code bit is bit 7 of the first compressed
    byte (weight_decomp reads bitbuf[BUFW-1] first).  EOB is appended once; the
    final byte is zero-padded (pad bits are never decoded -- `done` gates them).

SELF-TEST DISCIPLINE (like tools/q4k_ref.py)
  Every emitted block is round-tripped through an INDEPENDENT reference decoder
  (decode_ref, a line-for-line model of the RTL recurrence) and asserted to
  reconstruct the original bytes exactly, and the code is asserted COMPLETE
  (Kraft equality sum 2^-len == 1) with max length <= MAXLEN.  A generator that
  cannot reproduce its own stream aborts NONZERO -- it never emits a fake golden.

Run:  python3 tools/fp8_huff.py            # self-test + emit build/weight_decomp_vec.txt
      python3 tools/fp8_huff.py <outdir>   # emit into <outdir> instead of build
No deps beyond the stdlib.  Deterministic (fixed seeds).
"""
import sys, os, random

MAXLEN  = 15    # weight_decomp default WD_MAXLEN
EOB_SYM = 256   # weight_decomp default WD_EOB_SYM


# ------------------------------------------------------------------ code lengths
def package_merge(weights, L):
    """Length-limited (<=L) optimal prefix-code lengths via the package-merge
    algorithm (Larmore-Hirschberg).  weights: list of positive ints, one per
    symbol.  Returns a list of code lengths (each 1..L).  n>=2 required; n==1
    returns [1] (a single symbol still needs a 1-bit code in this scheme)."""
    n = len(weights)
    assert n >= 1
    if n == 1:
        return [1]
    # a "coin" is (weight, tuple-of-symbol-indices-it-covers)
    coins = sorted((weights[i], (i,)) for i in range(n))
    nodes = list(coins)                     # deepest denomination (2^-L) level
    for _ in range(L - 1):
        packaged = []
        j = 0
        while j + 1 < len(nodes):           # merge adjacent pairs; drop odd tail
            packaged.append((nodes[j][0] + nodes[j + 1][0],
                             nodes[j][1] + nodes[j + 1][1]))
            j += 2
        nodes = sorted(coins + packaged)    # merge with this level's originals
    lengths = [0] * n
    for _w, members in nodes[:2 * n - 2]:   # the 2n-2 lowest-weight selections
        for idx in members:
            lengths[idx] += 1
    assert all(1 <= l <= L for l in lengths), f"length out of range: {lengths}"
    return lengths


# --------------------------------------------------------- canonical assignment
def canonical(sym_len):
    """sym_len: dict {symbol: code_length}.  Returns (count, sym_order, codes):
      count[l]        for l in 0..MAXLEN  (count[0]=0, decoder never reads it)
      sym_order       symbols sorted by (length, symbol) -- the symbol_table
      codes[sym]      = (code_value, length)"""
    maxl = max(sym_len.values())
    count = [0] * (MAXLEN + 1)
    for l in sym_len.values():
        count[l] += 1
    sym_order = sorted(sym_len, key=lambda s: (sym_len[s], s))
    first = [0] * (MAXLEN + 2)
    code = 0
    for l in range(1, maxl + 1):
        first[l] = code
        code = (code + count[l]) << 1
    nxt = list(first)
    codes = {}
    for s in sym_order:                     # sym_order groups by length then value
        l = sym_len[s]
        codes[s] = (nxt[l], l)
        nxt[l] += 1
    return count, sym_order, codes


def build_code(data):
    """data: iterable of byte values (0..255).  Frequencies from data plus one
    EOB.  Returns (count, sym_order, codes, sym_len)."""
    freq = {}
    for b in data:
        assert 0 <= b <= 255
        freq[b] = freq.get(b, 0) + 1
    freq[EOB_SYM] = freq.get(EOB_SYM, 0) + 1   # EOB appears once per block
    syms = sorted(freq)
    lengths = package_merge([freq[s] for s in syms], MAXLEN)
    sym_len = {s: lengths[i] for i, s in enumerate(syms)}
    # Kraft completeness: a valid, COMPLETE prefix code satisfies sum 2^-len == 1.
    kraft = sum(1 << (MAXLEN - sym_len[s]) for s in syms)
    assert kraft == (1 << MAXLEN), f"code not complete (Kraft={kraft} != {1<<MAXLEN})"
    count, sym_order, codes = canonical(sym_len)
    return count, sym_order, codes, sym_len


# ----------------------------------------------------------------------- encode
def encode(data, codes):
    """MSB-first pack of each symbol's code, EOB appended, final byte zero-padded.
    Returns a list of compressed byte values."""
    bits = []
    for b in list(data) + [EOB_SYM]:
        val, ln = codes[b]
        for i in range(ln - 1, -1, -1):
            bits.append((val >> i) & 1)
    out = []
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        byte = 0
        for j, bit in enumerate(chunk):      # chunk[0] -> bit 7 (MSB) of the byte
            byte |= bit << (7 - j)
        out.append(byte)
    return out


# --------------------------------------------------- reference decoder (RTL model)
def decode_ref(comp, count, sym_order):
    """Independent, line-for-line model of weight_decomp.v's per-bit canonical
    recurrence.  Returns (decoded_bytes, saw_eob).  Used to self-verify the
    encoder before any RTL runs -- two independent implementations must agree."""
    out = []
    ccode = cfirst = cindex = 0
    clen = 1
    for byte in comp:
        for i in range(7, -1, -1):
            b = (byte >> i) & 1
            code_L = (ccode << 1) | b
            cnt = count[clen] if clen <= MAXLEN else 0
            diff = code_L - cfirst
            if diff < cnt:
                sym = sym_order[cindex + diff]
                if sym == EOB_SYM:
                    return out, True
                out.append(sym & 0xFF)
                ccode = cfirst = cindex = 0
                clen = 1
            else:
                ccode = code_L
                cfirst = (cfirst + cnt) << 1
                cindex += cnt
                clen += 1
    return out, False


# ------------------------------------------------------------------- test corpus
def _corpus():
    """(name, data, out_ready_mode) tuples.  mode 1 makes the TB toggle out_ready
    (exercises the decoder's output back-pressure / can_proc stall path)."""
    rnd = random.Random(0xF8E4A3)
    tests = []

    # 1) hand-crafted small mix (repeats -> short codes for 0/3).
    tests.append(("mix", [0, 1, 2, 3, 3, 3, 2, 1, 0, 0, 0, 0, 255, 255, 128, 7, 7], 0))

    # 2) degenerate single symbol repeated (both data sym and EOB are length 1).
    tests.append(("single", [0xA5] * 24, 0))

    # 3) two-symbol alternation, WITH output back-pressure (length-1 codes).
    tests.append(("binbp", [0x00, 0xFF] * 20 + [0x00], 1))

    # 4) Laplacian-ish skew over the full byte range (many code lengths), back-pressure.
    lap = []
    for _ in range(400):
        v = int(round(rnd.gauss(0.0, 22.0)))
        lap.append(max(0, min(255, 128 + v)))
    tests.append(("laplace", lap, 1))

    # 5) near-uniform over all 256 byte codes (worst case: ratio ~1, deep codes).
    uni = [rnd.randrange(256) for _ in range(512)]
    tests.append(("uniform", uni, 0))

    # 6) heavy skew that pushes some code lengths toward (but within) MAXLEN.
    #    A geometric-ish tail exercises the longest legal codewords.
    skew = []
    for _ in range(600):
        # p(k) ~ 2^-k gives a long thin tail -> long codewords for rare symbols.
        k = min(200, int(rnd.expovariate(1.0 / 6.0)))
        skew.append(k & 0xFF)
    tests.append(("skew", skew, 0))

    # 7) Power-of-two "caterpillar" frequencies -> a maximally DEEP code.  When
    #    each weight strictly exceeds the sum of all smaller ones (freqs 1,2,4,..,
    #    2^14), the OPTIMAL code is a unique caterpillar tree, forcing the rarest
    #    symbols to the MAXLEN=15 codeword length.  This is the ONLY block that
    #    drives clen -> 15, exercising the per-bit recurrence AND the full-width
    #    (CW=MAXLEN+1=16b) ccode/cfirst accumulators at their maximum.  With
    #    back-pressure.  (freqs sum to 2^15-1 = 32767 data bytes; ~ms in iverilog.)
    deep = []
    for i in range(15):                       # symbol i appears 2^i times
        deep += [(i * 16 + 5) & 0xFF] * (1 << i)
    rnd.shuffle(deep)
    tests.append(("deep", deep, 1))

    return tests


# ------------------------------------------------------------------------- emit
def emit(outdir):
    tests = _corpus()
    lines = []
    lines.append(f"{len(tests)} {MAXLEN} {EOB_SYM}")
    total_in = total_out = 0
    print(f"fp8_huff: {len(tests)} blocks, MAXLEN={MAXLEN}, EOB={EOB_SYM}")
    for name, data, mode in tests:
        count, sym_order, codes, sym_len = build_code(data)
        comp = encode(data, codes)

        # SELF-CHECK: independent reference decoder must reproduce the input exactly.
        dec, saw_eob = decode_ref(comp, count, sym_order)
        assert saw_eob, f"[{name}] reference decode never hit EOB"
        assert dec == list(data), f"[{name}] round-trip mismatch: {dec} != {list(data)}"

        ncodes = len(sym_order)
        maxl = max(sym_len.values())
        ratio = (8.0 * len(data)) / (8.0 * len(comp)) if comp else 0.0
        total_in += len(data)
        total_out += len(comp)
        print(f"  [{name:8s}] N={len(data):4d} ncodes={ncodes:3d} "
              f"comp={len(comp):4d}B maxlen={maxl:2d} ratio={ratio:4.2f}x mode={mode}")

        lines.append(f"{ncodes} {len(comp)} {len(data)} {mode}")
        lines.append(" ".join(str(count[l]) for l in range(1, MAXLEN + 1)))
        lines.append(" ".join(str(s) for s in sym_order))
        lines.append(" ".join(f"{b:02x}" for b in comp))
        lines.append(" ".join(f"{b:02x}" for b in data))

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "weight_decomp_vec.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    agg = (8.0 * total_in) / (8.0 * total_out) if total_out else 0.0
    print(f"fp8_huff: self-test OK (encode==decode_ref, Kraft-complete, len<=MAXLEN "
          f"on all {len(tests)} blocks)")
    print(f"wrote {path}: {total_in} raw bytes -> {total_out} compressed "
          f"(aggregate {agg:.2f}x)")
    return path


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "build"
    emit(outdir)
