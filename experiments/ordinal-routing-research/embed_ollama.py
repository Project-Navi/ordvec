#!/usr/bin/env python3
"""Generate real sentence embeddings via ollama (GPU) -> .npy for the probes.

ollama runs nomic-embed-text (768-d) resident on the GPU. We embed a corpus of
short texts and write a 2-D little-endian float32 C-order .npy matching the
format bench_rank / the conjecture probes read (--corpus-npy).

Corpus source (in priority order):
  1. --texts FILE   : one text per line
  2. built-in       : a few thousand templated sentences across many topics,
                      enough varied geometry for a first real-encoder pass
                      (no external download, no datasets lib).

Usage:
  python examples/embed_ollama.py --out corpus_real.npy --n 5000
  python examples/embed_ollama.py --texts mylines.txt --out corpus_real.npy
"""
import argparse, json, struct, sys, urllib.request

OLLAMA = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"


def embed_batch(texts):
    req = urllib.request.Request(
        OLLAMA,
        data=json.dumps({"model": MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        embs = json.load(r)["embeddings"]
    # E2 fix: ollama must return exactly one vector per input, IN ORDER.
    # A mismatch would silently misalign rows against the source texts.
    if len(embs) != len(texts):
        raise RuntimeError(
            f"embedding count {len(embs)} != input count {len(texts)} "
            "(row misalignment) — aborting rather than writing corrupt .npy"
        )
    return embs


def build_builtin_corpus(n):
    # Templated sentences spanning many semantic clusters -> real anisotropy,
    # real cluster structure, without an external dataset. Deterministic.
    subjects = ["the engineer", "a chef", "the astronomer", "my neighbor",
                "the orchestra", "a startup", "the river", "an old library",
                "the algorithm", "a mountain village", "the immune system",
                "a jazz quartet", "the supply chain", "an ancient manuscript",
                "the coral reef", "a chess grandmaster", "the power grid",
                "a desert caravan", "the neural network", "a vineyard"]
    verbs = ["studied", "rebuilt", "abandoned", "celebrated", "measured",
             "optimized", "flooded", "catalogued", "trained", "harvested",
             "defended", "improvised", "rerouted", "deciphered", "mapped"]
    objects = ["under heavy rain", "for three decades", "with great precision",
               "against all odds", "in the summer of 1998", "without any funding",
               "across twelve countries", "before the deadline", "at dawn",
               "using only open data", "despite the noise", "on a tight budget",
               "in complete silence", "after the merger", "beyond the horizon"]
    out = []
    i = 0
    while len(out) < n:
        s = subjects[i % len(subjects)]
        v = verbs[(i // len(subjects)) % len(verbs)]
        o = objects[(i // (len(subjects) * len(verbs))) % len(objects)]
        out.append(f"{s} {v} {o}.")
        i += 1
    return out[:n]


def write_npy(path, vecs):
    if not vecs:
        raise ValueError("no vectors to write (empty corpus?)")
    n = len(vecs)
    dim = len(vecs[0])
    header = ("{'descr': '<f4', 'fortran_order': False, "
              f"'shape': ({n}, {dim}), }}")
    hb = header.encode()
    pad = (64 - (10 + len(hb) + 1) % 64) % 64
    hb = hb + b" " * pad + b"\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY" + bytes([1, 0]))
        f.write(struct.pack("<H", len(hb)))
        f.write(hb)
        for v in vecs:
            f.write(struct.pack(f"<{dim}f", *v))
    print(f"# wrote {path}: {n} x {dim} f32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--texts")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    if args.texts:
        with open(args.texts, encoding="utf-8") as fh:
            texts = [ln.strip() for ln in fh if ln.strip()][: args.n]
    else:
        texts = build_builtin_corpus(args.n)
    print(f"# embedding {len(texts)} texts via ollama/{MODEL} (GPU)")

    vecs = []
    for i in range(0, len(texts), args.batch):
        batch = texts[i : i + args.batch]
        vecs.extend(embed_batch(batch))
        if (i // args.batch) % 10 == 0:
            print(f"  {len(vecs)}/{len(texts)}", file=sys.stderr)
    write_npy(args.out, vecs)


if __name__ == "__main__":
    main()
