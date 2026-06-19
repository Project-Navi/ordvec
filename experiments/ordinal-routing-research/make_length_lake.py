#!/usr/bin/env python3
"""Path B — build a CHUNK-LENGTH-MIXTURE lake (one domain, many length geometries).

Unlike make_lake.py (multi-DOMAIN union -> separated cones), this unions the SAME
documents embedded at several chunk lengths {128,256,512,1100}. Domain is held
constant; only the length-geometry varies. Models the real-lake pathology where one
S3 bucket holds the same content chunked every which way -> a *mixture of cones of
different tightness* quantized under one global set of rank-bin edges.

Stdlib only (struct/array), matches make_lake.py's npy IO. Deterministic.
"""
import struct, array, sys, argparse

def read_npy(path):
    b = open(path, "rb").read()
    assert b[:6] == b"\x93NUMPY"
    hl = struct.unpack("<H", b[8:10])[0]
    h = b[10:10+hl].decode()
    s = h.split("'shape':")[1]
    n, d = [int(x) for x in s[s.find("(")+1:s.find(")")].split(",") if x.strip()]
    a = array.array("f"); a.frombytes(b[10+hl:10+hl+n*d*4])
    return a, n, d

def write_npy(path, a, n, d):
    hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (n, d)
    pad = (64 - (10 + len(hdr) + 1) % 64) % 64
    hb = (hdr + " "*pad + "\n").encode()
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY" + bytes([1,0])); f.write(struct.pack("<H", len(hb))); f.write(hb)
        f.write(a.tobytes())
    print(f"# wrote {path}: {n} x {d}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True, help="per-length corpus .npy files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-nan", action="store_true", help="skip rows with NaN/zero norm")
    args = ap.parse_args()

    dim = None; rows = array.array("f"); total = 0; dropped = 0
    for p in args.parts:
        a, n, d = read_npy(p)
        dim = d if dim is None else dim
        assert d == dim, f"dim mismatch {p}"
        kept = 0
        for i in range(n):
            seg = a[i*d:(i+1)*d]
            if args.drop_nan:
                ss = 0.0; bad = False
                for x in seg:
                    if x != x:  # NaN
                        bad = True; break
                    ss += x*x
                if bad or ss == 0.0:
                    dropped += 1; continue
            rows.extend(seg); kept += 1
        total += kept
        print(f"# {p}: +{kept} docs", file=sys.stderr)
    if dropped:
        print(f"# dropped {dropped} NaN/zero rows", file=sys.stderr)
    write_npy(args.out, rows, total, dim)

if __name__ == "__main__":
    main()
