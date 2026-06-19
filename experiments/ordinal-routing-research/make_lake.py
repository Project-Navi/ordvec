#!/usr/bin/env python3
"""Build a synthetic multi-cone 'lake' from existing per-domain .npy embeddings.
Stdlib only (struct/array). Models two real enterprise-lake pathologies:
  1. MULTI-CONE: concatenate N domain corpora (each its own cone) into one corpus.
  2. TEMPLATED HUBS: optionally inject near-duplicate clusters (one base vector +
     tiny gaussian noise) at a target prevalence -> artificial hubs.
Queries: concatenate the per-domain held-out query sets (ground truth spans cones).
Deterministic (fixed LCG); no numpy.
"""
import struct, array, sys, math, argparse

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

# simple deterministic LCG gaussian (Box-Muller), no numpy
_state = [88172645463325252]
def _u():
    x = _state[0]; x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF; x ^= x >> 7; x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    _state[0] = x; return (x & 0xFFFFFFFF) / 0x100000000
def _g():
    u1 = max(_u(), 1e-12); u2 = _u()
    return math.sqrt(-2*math.log(u1)) * math.cos(2*math.pi*u2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True, help="domain corpus .npy files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hub-frac", type=float, default=0.0, help="fraction of corpus that is templated near-dupes")
    ap.add_argument("--hub-clusters", type=int, default=50, help="number of distinct templated clusters")
    ap.add_argument("--hub-noise", type=float, default=0.02, help="gaussian noise added to base vec (pre-normalize)")
    ap.add_argument("--cap", type=int, default=0, help="cap docs per part (0=all)")
    args = ap.parse_args()

    dim = None; rows = array.array("f"); total = 0
    for p in args.parts:
        a, n, d = read_npy(p)
        dim = d if dim is None else dim
        assert d == dim, f"dim mismatch {p}"
        take = n if args.cap == 0 else min(n, args.cap)
        rows.extend(a[:take*d]); total += take
        print(f"# {p}: +{take} docs", file=sys.stderr)

    # inject templated hubs: replace a fraction of rows with near-dupes of K bases
    if args.hub_frac > 0:
        n_hub = int(total * args.hub_frac)
        # pick K base vectors from existing rows (deterministic indices)
        bases = [ [rows[(i*9973 % total)*dim + j] for j in range(dim)] for i in range(args.hub_clusters) ]
        for h in range(n_hub):
            base = bases[h % args.hub_clusters]
            tgt = (h * 7919) % total  # deterministic overwrite positions
            row = [base[j] + args.hub_noise * _g() for j in range(dim)]
            nrm = math.sqrt(sum(x*x for x in row)) or 1.0
            for j in range(dim):
                rows[tgt*dim + j] = row[j] / nrm
        print(f"# injected {n_hub} templated-hub docs ({args.hub_frac:.0%}) across {args.hub_clusters} clusters", file=sys.stderr)

    write_npy(args.out, rows, total, dim)

if __name__ == "__main__":
    main()
