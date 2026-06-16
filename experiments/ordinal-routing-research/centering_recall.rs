//! CONTEXT-BOXED, PRE-REGISTERED — the DECISIVE gate for the centering finding.
//! overlap_decomp.rs showed per-coord mean-centering removes the cone and
//! amplifies true-neighbour overlap 2-5x. Necessary but NOT sufficient: the only
//! metric that promotes this to an ordvec change is RECALL at matched bytes,
//! and it must survive at b=4 (the incumbent that beat every other idea on this
//! branch). Isolated by request: touches no harness, writes nothing.
//!
//! TEST: full-scan RankQuant-style retrieval, R@10 vs FP32 cosine ground truth.
//!   Encode = dimension-wise rank -> bucket into 2^B equal-width bins (B bits/coord).
//!   Score  = -L1 distance between B-bin codes (symmetric rank metric; identical
//!            scoring for both arms, so only the ENCODING differs).
//!   Ground truth = FP32 cosine top-10 on L2-normalized RAW vectors (the real task;
//!                  centering changes the CODE, never the target).
//! Two arms:
//!   raw      : rank the L2-normalized vector directly.
//!   centered : subtract the per-coord CORPUS mean (a stored, data-oblivious
//!              statistic — no codebook, no per-query training) before ranking.
//!              The SAME mean is applied to queries (deployable asymmetric setup).
//! Also reports the two-stage candidate-recall (CR@M): fraction of FP32 top-10
//! captured in the top-M by code distance — the bitmap-prefilter-relevant number.
//!
//! PRE-REGISTERED VERDICT (fixed before running):
//!   FEATURE-WORTHY if centered R@10 - raw R@10 >= 0.02 at b=2 AND >= 0.02 at b=4
//!     (must survive at the incumbent b=4), AND centered never worse than raw by
//!     > 0.005 at any B.
//!   PARTIAL if it helps at low bits (b<=2) but the b=4 gain is < 0.02 (a
//!     low-bit-only tweak, not a headline change).
//!   FAIL if centered <= raw at b=4, or net-negative anywhere material.
//!   Report actual R@10 and CR@M for both arms regardless.
//!
//! Run: cargo run --release --example centering_recall -- \
//!          --corpus-npy /tmp/repo_corpus.npy --queries-npy /tmp/repo_q.npy

use rayon::prelude::*;

fn load_npy_f32(path: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).expect("read npy");
    assert!(
        bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY",
        "not a numpy file"
    );
    let major = bytes[6];
    let (hlen, hstart) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
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

fn coord_mean(corpus: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut m = vec![0f32; dim];
    for i in 0..n {
        for c in 0..dim {
            m[c] += corpus[i * dim + c];
        }
    }
    for m_c in m.iter_mut() {
        *m_c /= n as f32;
    }
    m
}

/// Bucketed-rank code: rank each (optionally mean-subtracted) coord, then bucket
/// the rank into 2^bits equal-width bins. One u8 bucket id per coord.
fn encode(rows: &[f32], n: usize, dim: usize, bits: u32, sub: Option<&[f32]>) -> Vec<u8> {
    let nb = 1u32 << bits;
    let mut out = vec![0u8; n * dim];
    out.par_chunks_mut(dim).enumerate().for_each(|(r, code)| {
        let mut idx: Vec<u16> = (0..dim as u16).collect();
        let val = |c: u16| rows[r * dim + c as usize] - sub.map_or(0.0, |s| s[c as usize]);
        idx.sort_by(|&a, &b| val(a).partial_cmp(&val(b)).unwrap());
        for (rank, &c) in idx.iter().enumerate() {
            let bucket = (rank as u32 * nb) / dim as u32;
            code[c as usize] = bucket as u8;
        }
    });
    out
}

fn l1(a: &[u8], b: &[u8]) -> i64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as i64 - *y as i64).abs())
        .sum()
}

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

/// Full-scan R@10 and candidate-recall@M for a given arm's codes.
fn evaluate(
    ccode: &[u8],
    qcode: &[u8],
    n: usize,
    nq: usize,
    dim: usize,
    truth: &[Vec<u32>],
    m: usize,
) -> (f64, f64) {
    let res: Vec<(f64, f64)> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let qc = &qcode[qi * dim..(qi + 1) * dim];
            let mut s: Vec<(u32, i64)> = (0..n)
                .map(|di| (di as u32, l1(qc, &ccode[di * dim..(di + 1) * dim])))
                .collect();
            s.sort_by_key(|x| x.1);
            let truth_set: std::collections::HashSet<u32> = truth[qi].iter().copied().collect();
            // R@10: top-10 by code distance that are in FP32 top-10
            let top10: std::collections::HashSet<u32> = s.iter().take(10).map(|x| x.0).collect();
            let r10 = top10.intersection(&truth_set).count() as f64 / truth[qi].len() as f64;
            // CR@M: FP32 top-10 captured within top-M by code distance
            let topm: std::collections::HashSet<u32> = s.iter().take(m).map(|x| x.0).collect();
            let cr = truth[qi].iter().filter(|t| topm.contains(t)).count() as f64
                / truth[qi].len() as f64;
            (r10, cr)
        })
        .collect();
    let r10 = res.iter().map(|x| x.0).sum::<f64>() / nq as f64;
    let cr = res.iter().map(|x| x.1).sum::<f64>() / nq as f64;
    (r10, cr)
}

fn main() {
    let mut cp = None;
    let mut qp = None;
    let mut m_arg = 100usize;
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        match x.as_str() {
            "--corpus-npy" => cp = Some(a.next().unwrap()),
            "--queries-npy" => qp = Some(a.next().unwrap()),
            "--m" => m_arg = a.next().unwrap().parse().unwrap(),
            _ => {}
        }
    }
    let (mut corpus, n, dim) = load_npy_f32(&cp.expect("--corpus-npy"));
    let (mut queries, nq, dq) = load_npy_f32(&qp.expect("--queries-npy"));
    assert_eq!(dim, dq);
    l2_normalize_rows(&mut corpus, dim);
    l2_normalize_rows(&mut queries, dim);
    let truth = ground_truth(&corpus, &queries, dim, 10);
    let mean = coord_mean(&corpus, n, dim);
    let m = m_arg; // CR@M operating point (default 100, --m to override)

    println!("# centering_recall (CONTEXT-BOXED, pre-registered). corpus {n}x{dim}, queries {nq}, CR@M M={m}");
    println!("# R@10 vs FP32 cosine top-10; symmetric -L1 on B-bin rank codes; only ENCODING differs between arms.");
    println!("B\tbytes/vec\traw_R@10\tcen_R@10\tΔR@10\traw_CR\tcen_CR\tΔCR");
    let mut deltas = vec![];
    for &bits in &[1u32, 2, 4] {
        let bytes = dim * bits as usize / 8;
        let craw = encode(&corpus, n, dim, bits, None);
        let qraw = encode(&queries, nq, dim, bits, None);
        let ccen = encode(&corpus, n, dim, bits, Some(&mean));
        let qcen = encode(&queries, nq, dim, bits, Some(&mean));
        let (r_raw, cr_raw) = evaluate(&craw, &qraw, n, nq, dim, &truth, m);
        let (r_cen, cr_cen) = evaluate(&ccen, &qcen, n, nq, dim, &truth, m);
        let d_r = r_cen - r_raw;
        let d_cr = cr_cen - cr_raw;
        deltas.push((bits, d_r));
        println!("{bits}\t{bytes}\t{r_raw:.4}\t{r_cen:.4}\t{d_r:+.4}\t{cr_raw:.4}\t{cr_cen:.4}\t{d_cr:+.4}");
    }
    println!("\n# PRE-REGISTERED: FEATURE-WORTHY if ΔR@10 >= +0.02 at b=2 AND b=4, and never worse than -0.005.");
    let b2 = deltas.iter().find(|x| x.0 == 2).unwrap().1;
    let b4 = deltas.iter().find(|x| x.0 == 4).unwrap().1;
    let worst = deltas.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
    let verdict = if b2 >= 0.02 && b4 >= 0.02 && worst >= -0.005 {
        "FEATURE-WORTHY"
    } else if b4 < 0.02 && deltas.iter().any(|x| x.1 >= 0.02) {
        "PARTIAL (low-bit only; does not survive at incumbent b=4)"
    } else {
        "FAIL"
    };
    println!("# VERDICT: {verdict}  (Δb2={b2:+.4}, Δb4={b4:+.4}, worst={worst:+.4})");
}
