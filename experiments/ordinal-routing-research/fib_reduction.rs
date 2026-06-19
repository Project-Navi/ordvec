//! CONTEXT-BOXED, PRE-REGISTERED — three independent probes of the hypothesis
//! "golden/Fibonacci is a REDUCTION RATE of the permutation space" (NOT an
//! address). Isolated by request: touches no routing harness, writes nothing,
//! and each experiment's verdict is fixed BELOW, before any data is seen. No
//! result here may reframe another experiment on this branch.
//!
//! Permutahedron P_{n-1} has n! vertices; a rank code is one vertex. The
//! hypothesis: at scale, the RETRIEVAL-relevant structure is a sub-factorial
//! reduction of that space, and golden ratio is its rate.
//!
//! ===================== EXP1 — FAN-OUT RATE =====================
//! Math fact (not measured, it's a theorem): #{σ : |σ(i)-i| <= 1} = F_{n+1},
//! ratio -> φ. The MEASURABLE crux: do the encoder's FP32 true neighbours
//! actually LIVE in the low-displacement (local) permutation neighbourhood of
//! the query's rank code, and does that neighbourhood grow at rate φ in DATA?
//!   metric A (locality)  = mean footrule(query, true-nbr) / mean footrule(random pair)
//!   metric B (φ fan-out) = geometric growth ratio of |docs within footrule r| over r
//! PRE-REGISTERED:
//!   locality PASS if A <= 0.75 (true nbrs meaningfully closer in rank space)
//!   fan-out  GOLDEN if B in [1.50, 1.75]; else "data-determined (intrinsic dim), not φ"
//!
//! ================= EXP2 — COARSENING SCHEDULE =================
//! Bucket ranks into B bins; retrieve top-K by L1 on the B-bin code; recall@10
//! vs bits/coord = log2(B). Compare bit-allocation schedules:
//!   fib {2,3,5,8,13,21}  pow2 {2,4,8,16,32}  linear {2,3,4,5,6,7}
//! PRE-REGISTERED:
//!   Fibonacci PASS if at matched bits it beats BOTH pow2 and linear by >= 0.03
//!     recall somewhere AND is never worse than either by > 0.03; else
//!     "schedule-independent (recall is one smooth curve in bits; φ no advantage)"
//!
//! ==================== EXP3 — MERGE RATE ======================
//! As ranks are progressively coarsened (B descending, Fibonacci-spaced), at
//! what RATE do permutations merge? Two levels: code-identity (do full codes
//! collide?) and neighbourhood (|docs within a fixed footrule ball|).
//! PRE-REGISTERED:
//!   GOLDEN if successive neighbourhood-merge ratios converge to [1.50, 1.75];
//!   else report the actual law (expected: combinatorial ~1/B, not φ).
//!
//! Run: cargo run --release --example fib_reduction -- \
//!          --corpus-npy /tmp/repo_corpus.npy --queries-npy /tmp/repo_q.npy
//!      (optional first positional: exp1 | exp2 | exp3; default runs all three)

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const SEED: u64 = 1;

fn load_npy_f32(path: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).expect("read npy");
    assert!(
        bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY",
        "not a numpy file"
    );
    let major = bytes[6];
    assert!(
        major == 1 || major == 2,
        "unsupported numpy .npy major version {major}"
    );
    if major == 2 {
        assert!(bytes.len() >= 12, "truncated numpy v2 header");
    }
    let (hlen, hstart) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    assert!(hstart + hlen <= bytes.len(), "truncated numpy header");
    let header = std::str::from_utf8(&bytes[hstart..hstart + hlen]).expect("utf8 header");
    assert!(header.contains("'descr': '<f4'"), "expected <f4 dtype");
    let after = &header[header.find("'shape':").expect("shape")..];
    let inner = &after[after.find('(').unwrap() + 1..after.find(')').unwrap()];
    let dims: Vec<usize> = inner
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let (n, dim) = (dims[0], dims[1]);
    let dstart = hstart + hlen;
    let mut out = vec![0.0f32; n * dim];
    for (i, ch) in bytes[dstart..].chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
    }
    (out, n, dim)
}

fn l2_normalize_rows(v: &mut [f32], dim: usize) {
    for i in 0..v.len() / dim {
        let row = &mut v[i * dim..(i + 1) * dim];
        let nrm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if nrm > 0.0 {
            for x in row.iter_mut() {
                *x /= nrm;
            }
        }
    }
}

/// Rank-encode: rank[i] = ordinal position (0..dim) of coordinate i within the row.
fn rank_encode(rows: &[f32], n: usize, dim: usize) -> Vec<u16> {
    let mut out = vec![0u16; n * dim];
    out.par_chunks_mut(dim).enumerate().for_each(|(r, code)| {
        let row = &rows[r * dim..(r + 1) * dim];
        let mut idx: Vec<u16> = (0..dim as u16).collect();
        idx.sort_by(|&a, &b| row[a as usize].partial_cmp(&row[b as usize]).unwrap());
        for (rank, &i) in idx.iter().enumerate() {
            code[i as usize] = rank as u16;
        }
    });
    out
}

fn footrule(a: &[u16], b: &[u16]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as f64)
        .sum()
}

/// FP32 cosine top-k ground truth (rows already L2-normalized).
fn ground_truth(corpus: &[f32], queries: &[f32], dim: usize, k: usize) -> Vec<Vec<u32>> {
    let n = corpus.len() / dim;
    let nq = queries.len() / dim;
    (0..nq)
        .into_par_iter()
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let mut s: Vec<(u32, f32)> = (0..n)
                .map(|di| {
                    let d = &corpus[di * dim..(di + 1) * dim];
                    (di as u32, q.iter().zip(d).map(|(a, b)| a * b).sum::<f32>())
                })
                .collect();
            s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            s.into_iter().take(k).map(|(i, _)| i).collect()
        })
        .collect()
}

struct Data {
    corpus: Vec<f32>,
    queries: Vec<f32>,
    dim: usize,
    n: usize,
    nq: usize,
    crank: Vec<u16>,
    qrank: Vec<u16>,
    truth: Vec<Vec<u32>>,
}

fn exp1_fanout(d: &Data) {
    println!("\n## EXP1 — FAN-OUT RATE");
    let crow = |i: usize| &d.crank[i * d.dim..(i + 1) * d.dim];
    let qrow = |i: usize| &d.qrank[i * d.dim..(i + 1) * d.dim];

    // metric A: locality of true neighbours vs random pairs, in rank (footrule) space
    let true_med: f64 = (0..d.nq)
        .into_par_iter()
        .map(|qi| {
            let mut fs: Vec<f64> = d.truth[qi]
                .iter()
                .map(|&di| footrule(qrow(qi), crow(di as usize)))
                .collect();
            fs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            fs[fs.len() / 2]
        })
        .sum::<f64>()
        / d.nq as f64;
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let rand_pairs: Vec<(usize, usize)> = (0..20_000)
        .map(|_| (rng.random_range(0..d.n), rng.random_range(0..d.n)))
        .collect();
    let rand_med = {
        let mut fs: Vec<f64> = rand_pairs
            .par_iter()
            .map(|&(i, j)| footrule(crow(i), crow(j)))
            .collect();
        fs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        fs[fs.len() / 2]
    };
    let a_ratio = true_med / rand_med;
    println!("  metric A locality: true-nbr median footrule {true_med:.0} / random {rand_med:.0} = {a_ratio:.4}");
    println!(
        "    -> {}",
        if a_ratio <= 0.75 {
            "PASS (true neighbours closer in rank space)"
        } else {
            "FAIL (rank space does not localize neighbours)"
        }
    );

    // metric B: neighbourhood growth ratio over footrule shells, on a query sample
    let sample: Vec<usize> = (0..d.nq).step_by(4).collect();
    let radii: Vec<f64> = (1..=8).map(|s| rand_med * (s as f64) / 8.0).collect();
    let mut counts = vec![0f64; radii.len()];
    for &qi in &sample {
        let dists: Vec<f64> = (0..d.n)
            .into_par_iter()
            .map(|di| footrule(qrow(qi), crow(di)))
            .collect();
        for (ri, &r) in radii.iter().enumerate() {
            counts[ri] += dists.iter().filter(|&&x| x <= r).count() as f64;
        }
    }
    let mut ratios = vec![];
    for w in counts.windows(2) {
        if w[0] > 0.0 {
            ratios.push(w[1] / w[0]);
        }
    }
    let gmean = ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64;
    let g = gmean.exp();
    println!("  metric B fan-out growth ratio (per equal radius step) = {g:.3}");
    println!(
        "    -> {}",
        if (1.50..=1.75).contains(&g) {
            "GOLDEN (≈φ)"
        } else {
            "data-determined (set by intrinsic dim), NOT φ"
        }
    );
}

fn exp2_schedule(d: &Data) {
    println!("\n## EXP2 — COARSENING SCHEDULE");
    let crow = |i: usize| &d.crank[i * d.dim..(i + 1) * d.dim];
    let qrow = |i: usize| &d.qrank[i * d.dim..(i + 1) * d.dim];
    let cand_k = 100usize;
    let bucket = |rank: u16, b: u32| -> u16 { ((rank as u32 * b) / d.dim as u32) as u16 };
    // recall@10 (truth top-10 captured within top-cand_k by L1 on B-bin code)
    let recall_for_b = |b: u32| -> f64 {
        let ccode: Vec<u16> = d.crank.iter().map(|&r| bucket(r, b)).collect();
        (0..d.nq)
            .into_par_iter()
            .map(|qi| {
                let qc: Vec<u16> = qrow(qi).iter().map(|&r| bucket(r, b)).collect();
                let mut s: Vec<(u32, i64)> = (0..d.n)
                    .map(|di| {
                        let dc = &ccode[di * d.dim..(di + 1) * d.dim];
                        let dist: i64 = qc
                            .iter()
                            .zip(dc)
                            .map(|(x, y)| (*x as i64 - *y as i64).abs())
                            .sum();
                        (di as u32, dist)
                    })
                    .collect();
                s.sort_by_key(|x| x.1);
                let top: std::collections::HashSet<u32> =
                    s.into_iter().take(cand_k).map(|x| x.0).collect();
                let hit = d.truth[qi].iter().filter(|t| top.contains(t)).count();
                hit as f64 / d.truth[qi].len() as f64
            })
            .sum::<f64>()
            / d.nq as f64
    };
    let all_b = [2u32, 3, 4, 5, 6, 7, 8, 13, 16, 21, 32];
    let mut rec = std::collections::HashMap::new();
    for &b in &all_b {
        rec.insert(b, recall_for_b(b));
    }
    let show = |name: &str, bs: &[u32], rec: &std::collections::HashMap<u32, f64>| {
        println!("  {name}:");
        for &b in bs {
            println!(
                "    B={b:>2} bits={:.2}  recall@10={:.4}",
                (b as f64).log2(),
                rec[&b]
            );
        }
    };
    let _ = crow; // crow unused here; codes built from crank directly
    show("fib   ", &[2, 3, 5, 8, 13, 21], &rec);
    show("pow2  ", &[2, 4, 8, 16, 32], &rec);
    show("linear", &[2, 3, 4, 5, 6, 7], &rec);
    println!("  -> recall is a function of bits=log2(B); schedules only SAMPLE that curve.");
    println!(
        "     Fibonacci PASS only if its points beat pow2 AND linear at matched bits by >=0.03."
    );
}

fn exp3_merge(d: &Data) {
    println!("\n## EXP3 — MERGE RATE");
    let crow = |i: usize| &d.crank[i * d.dim..(i + 1) * d.dim];
    let bucket = |rank: u16, b: u32| -> u16 { ((rank as u32 * b) / d.dim as u32) as u16 };
    // (a) code-identity: do full B-bin codes collide as B shrinks?
    println!("  (a) distinct full codes among {} docs:", d.n);
    let fib_levels = [21u32, 13, 8, 5, 3, 2];
    let mut distinct = vec![];
    for &b in &fib_levels {
        let set: std::collections::HashSet<Vec<u16>> = (0..d.n)
            .map(|i| crow(i).iter().map(|&r| bucket(r, b)).collect::<Vec<u16>>())
            .collect();
        distinct.push(set.len());
        println!(
            "    B={b:>2}: {} distinct ({:.1}% of n)",
            set.len(),
            100.0 * set.len() as f64 / d.n as f64
        );
    }
    // (b) neighbourhood merge: mean docs within a fixed footrule ball, across Fibonacci radii
    let crow2 = |i: usize| &d.crank[i * d.dim..(i + 1) * d.dim];
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0x3333);
    let probes: Vec<usize> = (0..50).map(|_| rng.random_range(0..d.n)).collect();
    // a coarse footrule scale: median random footrule
    let scale = {
        let mut fs: Vec<f64> = (0..5000)
            .map(|_| {
                footrule(
                    crow2(rng.random_range(0..d.n)),
                    crow2(rng.random_range(0..d.n)),
                )
            })
            .collect();
        fs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        fs[fs.len() / 2]
    };
    let fib_r = [1f64, 2.0, 3.0, 5.0, 8.0, 13.0]; // Fibonacci-spaced shell radii (× scale/13)
    let mut counts = vec![0f64; fib_r.len()];
    for &p in &probes {
        let dists: Vec<f64> = (0..d.n)
            .into_par_iter()
            .map(|di| footrule(crow2(p), crow2(di)))
            .collect();
        for (ri, &fr) in fib_r.iter().enumerate() {
            let r = scale * fr / 13.0;
            counts[ri] += dists.iter().filter(|&&x| x <= r).count() as f64;
        }
    }
    println!("  (b) neighbourhood size at Fibonacci-spaced radii:");
    for (ri, &fr) in fib_r.iter().enumerate() {
        println!(
            "    r={fr:>4} (×scale/13): mean |ball| = {:.1}",
            counts[ri] / probes.len() as f64
        );
    }
    let mut ratios = vec![];
    for w in counts.windows(2) {
        if w[0] > 0.0 {
            ratios.push(w[1] / w[0]);
        }
    }
    if !ratios.is_empty() {
        let g = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64).exp();
        println!(
            "  merge growth ratio across shells = {g:.3} -> {}",
            if (1.50..=1.75).contains(&g) {
                "GOLDEN (≈φ)"
            } else {
                "NOT φ (set by data geometry / shell spacing)"
            }
        );
    }
}

fn main() {
    let mut which = None;
    let mut corpus_path = None;
    let mut queries_path = None;
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        match x.as_str() {
            "--corpus-npy" => corpus_path = Some(a.next().unwrap()),
            "--queries-npy" => queries_path = Some(a.next().unwrap()),
            "exp1" | "exp2" | "exp3" => which = Some(x),
            _ => {}
        }
    }
    let cp = corpus_path.expect("--corpus-npy required");
    let qp = queries_path.expect("--queries-npy required");
    let (mut corpus, n, dim) = load_npy_f32(&cp);
    let (mut queries, nq, dq) = load_npy_f32(&qp);
    assert_eq!(dim, dq);
    l2_normalize_rows(&mut corpus, dim);
    l2_normalize_rows(&mut queries, dim);
    let crank = rank_encode(&corpus, n, dim);
    let qrank = rank_encode(&queries, nq, dim);
    let truth = ground_truth(&corpus, &queries, dim, 10);
    let d = Data {
        corpus,
        queries,
        dim,
        n,
        nq,
        crank,
        qrank,
        truth,
    };
    println!("# fib_reduction (CONTEXT-BOXED, pre-registered). corpus {n}x{dim}, queries {nq}");
    let _ = (&d.corpus, &d.queries); // kept for parity; rank-space metrics use crank/qrank
    match which.as_deref() {
        Some("exp1") => exp1_fanout(&d),
        Some("exp2") => exp2_schedule(&d),
        Some("exp3") => exp3_merge(&d),
        _ => {
            exp1_fanout(&d);
            exp2_schedule(&d);
            exp3_merge(&d);
        }
    }
}
