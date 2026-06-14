//! Profiling harness for the contingency / projection surface (issue #219),
//! on a SYNTHETIC seeded corpus — NOT real data, fully reproducible.
//!
//! Two APIs, three regimes:
//!
//! * API 1 — stateless dense [`ordvec::Contingency`]: the `nb × nb`
//!   bucket-overlap table for two `&[u8]` code slices, plus its named
//!   projections.
//! * API 2 — the indexed [`ordvec::MultiBucketBitmap`] twin:
//!   `contingency_row` / `diagonal_overlap_row` / `project_all_batched`
//!   accumulate the same table straight from the stored bucket bitmaps.
//!
//! Regimes (issue #219 acceptance):
//!
//!   (a) ONE PAIR / MANY PROJECTIONS — build the dense `Contingency` table
//!       once, then apply all ~7 projections off the cached integer table,
//!       vs naively recomputing each projection from the raw `&[u8]` codes
//!       every time. Shows the build-once-project-many win.
//!
//!   (b) ONE QUERY / MANY DOCS / ONE PROJECTION — the indexed diagonal fast
//!       path over the whole corpus (dispatched = SIMD-when-available, and a
//!       forced-scalar twin), vs the dense pairwise loop that rebuilds the
//!       full `nb × nb` table per doc just to read its diagonal.
//!
//!   (c) ONE QUERY / MANY DOCS / MANY PROJECTIONS — the indexed batched
//!       `project_all_batched` (one table per doc, K projections off it), vs
//!       calling the single-projection indexed path K times (one corpus
//!       rescan per projection). Shows the no-rescan win, dispatched vs
//!       forced-scalar.
//!
//! Timing is median-of-N wall clock (same discipline as
//! `examples/bench_retrieval.rs`). Every number is SYNTHETIC.
//!
//! NOTE on SIMD: the indexed kernels dispatch to an AVX-512 VPOPCNTDQ
//! popcount-AND path only when the host advertises `avx512f` AND
//! `avx512vpopcntdq` (the dense API-1 kernel additionally needs `avx512bw`).
//! On a host without VPOPCNTDQ the "dispatched" column runs the SAME scalar
//! kernel as the forced-scalar column, so the SIMD/scalar ratio is ~1.0 —
//! that is the honest result, not a bug. The header prints the host features
//! so the ratio can be read in context.
//!
//! Run: cargo run --release --features experimental --example bench_contingency

#[cfg(not(feature = "experimental"))]
fn main() {
    eprintln!(
        "bench_contingency requires --features experimental (Contingency / \
         MultiBucketBitmap). Re-run: cargo run --release --features experimental \
         --example bench_contingency"
    );
}

#[cfg(feature = "experimental")]
fn main() {
    bench::run();
}

#[cfg(feature = "experimental")]
mod bench {
    use std::time::Instant;

    use ordvec::{Contingency, MultiBucketBitmap, Projection};
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    // ---- timing -----------------------------------------------------------

    /// Median wall-clock seconds over `reps` timed runs (one warmup first),
    /// each run doing `iters` repetitions of `f`. Returns seconds PER `f`
    /// call (total / iters), so callers report cost of a single operation.
    fn median_secs(reps: usize, iters: usize, mut f: impl FnMut()) -> f64 {
        // warmup
        for _ in 0..iters {
            f();
        }
        let mut lats = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            for _ in 0..iters {
                f();
            }
            lats.push(t.elapsed().as_secs_f64() / iters as f64);
        }
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        lats[lats.len() / 2]
    }

    // ---- corpus -----------------------------------------------------------

    fn gauss(rng: &mut ChaCha8Rng) -> f32 {
        let u1: f32 = rng.random_range(1e-9..1.0);
        let u2: f32 = rng.random_range(0.0..1.0);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Clustered gaussian corpus + a single query, seeded for reproducibility.
    /// Each doc/query draws a cluster prototype plus noise (queries quieter),
    /// so bucket assignments correlate the way real ordinal codes would —
    /// pure iid noise would make every contingency table look the same.
    fn make_data(n: usize, dim: usize, n_clusters: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut protos = vec![0.0f32; n_clusters * dim];
        for x in protos.iter_mut() {
            *x = gauss(&mut rng);
        }
        let emit = |count: usize, noise: f32, rng: &mut ChaCha8Rng| -> Vec<f32> {
            let mut out = vec![0.0f32; count * dim];
            for i in 0..count {
                let c = rng.random_range(0..n_clusters as u32) as usize;
                for j in 0..dim {
                    out[i * dim + j] = protos[c * dim + j] + noise * gauss(rng);
                }
            }
            out
        };
        let corpus = emit(n, 1.0, &mut rng);
        let query = emit(1, 0.5, &mut rng);
        (corpus, query)
    }

    /// Reconstruct the per-coordinate bucket codes (`0..nb`) a doc was encoded
    /// with by reading back the index's stored bitmaps via the indexed
    /// contingency: feeding a one-hot "query" for bucket `b` recovers, per
    /// coordinate, whether the doc sits in bucket `b`. Simpler here: rebuild
    /// the codes from `query_bitmaps_from_ranks` of the raw float vector, which
    /// is exactly the encoding `add` applies.
    fn codes_from_bitmaps(bitmaps: &[u64], nb: usize, qpb: usize, dim: usize) -> Vec<u8> {
        let mut codes = vec![0u8; dim];
        for b in 0..nb {
            let off = b * qpb;
            for (j, code) in codes.iter_mut().enumerate() {
                if (bitmaps[off + j / 64] >> (j % 64)) & 1 == 1 {
                    *code = b as u8;
                }
            }
        }
        codes
    }

    // ---- the ~7 projections, as both named ops (API 1) and weight matrices --

    /// The named-projection battery used in regime (a): the full surface a
    /// caller might evaluate off one table (top-overlap, top-group, diagonal,
    /// two bands, L1 transport, centred outer product).
    fn named_projections(nb: usize) -> Vec<Projection> {
        let mut p = vec![
            Projection::TopOverlap,
            Projection::TopGroupOverlap { width: 2 },
            Projection::DiagonalAgreement,
            Projection::BandAgreement { radius: 1 },
            Projection::BucketL1Distance,
            Projection::RankQuantSymmetric,
        ];
        if nb > 4 {
            p.push(Projection::BandAgreement { radius: 2 });
        }
        p
    }

    /// The same battery expressed as raw `nb × nb` weight matrices, for the
    /// indexed `project_all_batched` path in regime (c). (TopOverlap,
    /// TopGroup, diagonal, band1, band2, outer product — six dense matrices;
    /// L1 distance is also expressible as a weight matrix `|a-b|`.)
    fn weight_matrices(nb: usize) -> Vec<Vec<f32>> {
        let c = (nb as f32 - 1.0) / 2.0;
        let mut out: Vec<Vec<f32>> = Vec::new();

        // top-overlap: single cell (nb-1, nb-1)
        let mut w = vec![0.0f32; nb * nb];
        w[(nb - 1) * nb + (nb - 1)] = 1.0;
        out.push(w);

        // top-group width 2
        let mut w = vec![0.0f32; nb * nb];
        let start = nb - 2;
        for a in start..nb {
            for b in start..nb {
                w[a * nb + b] = 1.0;
            }
        }
        out.push(w);

        // diagonal
        let mut w = vec![0.0f32; nb * nb];
        for a in 0..nb {
            w[a * nb + a] = 1.0;
        }
        out.push(w);

        // band radius 1 (unit weights)
        let mut w = vec![0.0f32; nb * nb];
        for a in 0..nb {
            for b in 0..nb {
                if a.abs_diff(b) <= 1 {
                    w[a * nb + b] = 1.0;
                }
            }
        }
        out.push(w);

        // bucket-L1 transport cost |a-b|
        let mut w = vec![0.0f32; nb * nb];
        for a in 0..nb {
            for b in 0..nb {
                w[a * nb + b] = a.abs_diff(b) as f32;
            }
        }
        out.push(w);

        // centred outer product (a-c)(b-c)
        let mut w = vec![0.0f32; nb * nb];
        for a in 0..nb {
            for b in 0..nb {
                w[a * nb + b] = (a as f32 - c) * (b as f32 - c);
            }
        }
        out.push(w);

        if nb > 4 {
            // band radius 2
            let mut w = vec![0.0f32; nb * nb];
            for a in 0..nb {
                for b in 0..nb {
                    if a.abs_diff(b) <= 2 {
                        w[a * nb + b] = 1.0;
                    }
                }
            }
            out.push(w);
        }
        out
    }

    pub fn run() {
        #[cfg(target_arch = "x86_64")]
        {
            println!(
                "# host SIMD: avx2={} avx512f={} avx512bw={} avx512vpopcntdq={}",
                is_x86_feature_detected!("avx2"),
                is_x86_feature_detected!("avx512f"),
                is_x86_feature_detected!("avx512bw"),
                is_x86_feature_detected!("avx512vpopcntdq"),
            );
            if !is_x86_feature_detected!("avx512vpopcntdq") {
                println!(
                    "# NOTE: no avx512vpopcntdq on this host — the 'dispatched' \
                     and 'scalar' columns run the SAME kernel; ratio ~1.0 is honest."
                );
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        println!("# host SIMD: non-x86_64 target — scalar kernels only");

        println!("# bench_contingency — SYNTHETIC clustered gaussian corpus (not real data)");

        for &bits in &[2u8, 4u8] {
            run_bits(bits);
            println!();
        }
    }

    fn run_bits(bits: u8) {
        let dim = 1024usize;
        let n = 20_000usize;
        let n_clusters = 64usize;
        let seed = 219u64;
        let nb = 1usize << bits;
        let qpb = dim / 64;

        println!("# ===== bits={bits} (nb={nb})  dim={dim}  n={n}  clusters={n_clusters}  seed={seed} =====");

        let (corpus, query) = make_data(n, dim, n_clusters, seed);

        // Build the indexed surface (API 2).
        let mut idx = MultiBucketBitmap::new(dim, bits);
        idx.add(&corpus);
        let q_bitmaps = idx.query_bitmaps_from_ranks(&query);

        // Per-coord codes for API 1 (dense). `query_bitmaps_from_ranks` applies
        // the exact same rank→bucket encoding `add` uses, so re-encoding doc 0's
        // float vector and the query recovers the real codes the index holds.
        let q_codes = codes_from_bitmaps(&q_bitmaps, nb, qpb, dim);
        let doc0 = &corpus[0..dim];
        let d0_bitmaps = idx.query_bitmaps_from_ranks(doc0);
        let d_codes = codes_from_bitmaps(&d0_bitmaps, nb, qpb, dim);

        // ---- regime (a): ONE PAIR / MANY PROJECTIONS (dense API 1) --------
        regime_a(&q_codes, &d_codes, nb);

        // ---- regime (b): ONE QUERY / MANY DOCS / ONE PROJECTION ----------
        regime_b(&idx, &q_bitmaps, n, nb);

        // ---- regime (c): ONE QUERY / MANY DOCS / MANY PROJECTIONS --------
        regime_c(&idx, &q_bitmaps, nb);
    }

    /// (a) Build the dense `Contingency` once, apply all projections off the
    /// cached table — vs recomputing each projection from the raw codes (build
    /// a fresh table per projection). Single pair, repeated for timing.
    fn regime_a(q_codes: &[u8], d_codes: &[u8], nb: usize) {
        let projs = named_projections(nb);
        let n_proj = projs.len();

        // Build-once-project-many: one table, then all projections.
        let build_once = median_secs(25, 200, || {
            let c = Contingency::new(q_codes, d_codes, nb).unwrap();
            let mut acc = 0.0f32;
            for p in &projs {
                acc += p.score(&c);
            }
            std::hint::black_box(acc);
        });

        // Naive: rebuild the table from raw codes for EVERY projection.
        let rebuild_each = median_secs(25, 200, || {
            let mut acc = 0.0f32;
            for p in &projs {
                let c = Contingency::new(q_codes, d_codes, nb).unwrap();
                acc += p.score(&c);
            }
            std::hint::black_box(acc);
        });

        let speedup = rebuild_each / build_once;
        println!("# (a) ONE PAIR / {n_proj} PROJECTIONS — dense Contingency (API 1)");
        println!("{:<34} {:>14} {:>14}", "approach", "us/pair", "speedup");
        println!(
            "{:<34} {:>14.3} {:>14}",
            "build-once, project-many",
            build_once * 1e6,
            "1.00x (ref)"
        );
        println!(
            "{:<34} {:>14.3} {:>13.2}x",
            "rebuild-per-projection",
            rebuild_each * 1e6,
            speedup
        );
        println!(
            "DATA\ta\tbits={}\tbuild_once_us={:.3}\trebuild_each_us={:.3}\tn_proj={}\tspeedup={:.2}",
            (nb as f32).log2() as u32,
            build_once * 1e6,
            rebuild_each * 1e6,
            n_proj,
            speedup
        );
    }

    /// (b) ONE QUERY / MANY DOCS / ONE PROJECTION — indexed diagonal fast path
    /// (dispatched + forced-scalar) over the corpus, vs the dense pairwise
    /// loop (full table per doc, read its diagonal).
    fn regime_b(idx: &MultiBucketBitmap, q_bitmaps: &[u64], n: usize, nb: usize) {
        // dispatched diagonal fast path over every doc
        let diag_dispatched = median_secs(7, 3, || {
            let mut acc = 0u64;
            for di in 0..n {
                let d = idx.diagonal_overlap_row(q_bitmaps, di);
                acc += d.iter().map(|&x| x as u64).sum::<u64>();
            }
            std::hint::black_box(acc);
        });

        // forced-scalar diagonal fast path
        let diag_scalar = median_secs(7, 3, || {
            let mut acc = 0u64;
            for di in 0..n {
                let d = idx.diagonal_overlap_row_scalar(q_bitmaps, di);
                acc += d.iter().map(|&x| x as u64).sum::<u64>();
            }
            std::hint::black_box(acc);
        });

        // dense pairwise loop: build the FULL nb*nb table per doc (dispatched
        // contingency_row), then take its diagonal trace. This is the work the
        // diagonal fast path avoids (nb passes vs nb*nb).
        let dense_full = median_secs(7, 3, || {
            let mut acc = 0u64;
            for di in 0..n {
                let table = idx.contingency_row(q_bitmaps, di);
                for a in 0..nb {
                    acc += table[a * nb + a] as u64;
                }
            }
            std::hint::black_box(acc);
        });

        let simd_speedup = diag_scalar / diag_dispatched;
        let fastpath_speedup = dense_full / diag_dispatched;

        let per = |s: f64| s / n as f64 * 1e9; // ns per doc
        println!("# (b) ONE QUERY / {n} DOCS / ONE PROJECTION (diagonal) — indexed (API 2)");
        println!(
            "{:<34} {:>14} {:>16}",
            "approach", "ns/doc", "vs dispatched"
        );
        println!(
            "{:<34} {:>14.2} {:>16}",
            "diagonal fast path (dispatched)",
            per(diag_dispatched),
            "1.00x (ref)"
        );
        println!(
            "{:<34} {:>14.2} {:>15.2}x",
            "diagonal fast path (scalar)",
            per(diag_scalar),
            simd_speedup
        );
        println!(
            "{:<34} {:>14.2} {:>15.2}x",
            "dense full-table diagonal",
            per(dense_full),
            fastpath_speedup
        );
        println!(
            "DATA\tb\tbits={}\tdiag_disp_ns={:.2}\tdiag_scalar_ns={:.2}\tdense_full_ns={:.2}\tsimd_speedup={:.2}\tfastpath_speedup={:.2}",
            (nb as f32).log2() as u32,
            per(diag_dispatched),
            per(diag_scalar),
            per(dense_full),
            simd_speedup,
            fastpath_speedup
        );
    }

    /// (c) ONE QUERY / MANY DOCS / MANY PROJECTIONS — batched
    /// `project_all_batched` (table once, K projections off it) vs calling the
    /// single-projection path K times (one corpus rescan per projection).
    /// Both dispatched and forced-scalar.
    fn regime_c(idx: &MultiBucketBitmap, q_bitmaps: &[u64], nb: usize) {
        let mats = weight_matrices(nb);
        let k = mats.len();
        let weights: Vec<&[f32]> = mats.iter().map(|w| w.as_slice()).collect();
        let n = idx.len();

        // batched dispatched: one table per doc, all K projections off it.
        let batched_disp = median_secs(7, 3, || {
            let out = idx.project_all_batched(q_bitmaps, &weights);
            std::hint::black_box(out.len());
        });

        // batched forced-scalar.
        let batched_scalar = median_secs(7, 3, || {
            let out = idx.project_all_batched_scalar(q_bitmaps, &weights);
            std::hint::black_box(out.len());
        });

        // rescan: call the single-projection batched path K times (each call
        // rescans the whole corpus building a fresh table per doc). Dispatched.
        let rescan_disp = median_secs(7, 3, || {
            let mut acc = 0usize;
            for w in &weights {
                let single: [&[f32]; 1] = [w];
                let out = idx.project_all_batched(q_bitmaps, &single);
                acc += out.len();
            }
            std::hint::black_box(acc);
        });

        let norescan_speedup = rescan_disp / batched_disp;
        let simd_speedup = batched_scalar / batched_disp;

        let per = |s: f64| s / n as f64 * 1e9; // ns per doc (over all K)
        println!("# (c) ONE QUERY / {n} DOCS / {k} PROJECTIONS — indexed batched (API 2)");
        println!(
            "{:<38} {:>14} {:>16}",
            "approach", "ns/doc(allK)", "vs batched"
        );
        println!(
            "{:<38} {:>14.2} {:>16}",
            "project_all_batched (dispatched)",
            per(batched_disp),
            "1.00x (ref)"
        );
        println!(
            "{:<38} {:>14.2} {:>15.2}x",
            "project_all_batched (scalar)",
            per(batched_scalar),
            simd_speedup
        );
        println!(
            "{:<38} {:>14.2} {:>15.2}x",
            "rescan: single-proj x K (dispatched)",
            per(rescan_disp),
            norescan_speedup
        );
        println!(
            "DATA\tc\tbits={}\tbatched_disp_ns={:.2}\tbatched_scalar_ns={:.2}\trescan_disp_ns={:.2}\tk={}\tnorescan_speedup={:.2}\tsimd_speedup={:.2}",
            (nb as f32).log2() as u32,
            per(batched_disp),
            per(batched_scalar),
            per(rescan_disp),
            k,
            norescan_speedup,
            simd_speedup
        );
    }
}
