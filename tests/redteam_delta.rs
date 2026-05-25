//! Red-team hardening suite, fourth pass (`delta`).
//!
//! Adversarial pre-publication fuzzing of the `ordvec` public API,
//! authored by an offensive-security review. The earlier `alpha` /
//! `beta` / `gamma` suites pin specific historical fixes (SIMD-dispatch
//! lane invariants, the `k = usize::MAX` capacity-overflow clamp, the
//! subset out-of-range guard, the `rank_to_bucket` / `bucket_centre`
//! domain asserts). This suite covers the *remaining* boundary surface a
//! motivated attacker would poke at, and pins it as a regression so the
//! guarantees survive future refactors:
//!
//! - **DELTA-A (loaders, highest value).** Adversarial header geometry
//!   the structural-fuzz in `tests/index/main.rs` does not cover: a
//!   declared `n_vectors == MAX_VECTORS` paired with an empty file (the
//!   "tiny header claims gigabytes" DoS — must reject in microseconds
//!   without allocating), `n_vectors == MAX_VECTORS + 1`, `dim == 1`
//!   (just under the `[2, MAX_DIM]` floor), per-format `dim` ceilings
//!   (`MAX_DIM` for TVBM, `MAX_SIGN_BITMAP_DIM` for TVSB), bad version
//!   bytes, and all-`0xFF` files at every header length. Every loader
//!   must return `Err`, never panic / hang / OOM.
//! - **DELTA-B (integer overflow on 64-bit).** The symmetric `Rank`
//!   search accumulates `Σ_d (2·q − (D−1))·(2·doc − (D−1))` into an
//!   `i64`. At the true `dim = u16::MAX` ceiling with the rank extremes
//!   this is the worst case; pin that it stays finite (no `i32`/`i64`
//!   wrap, no `rank_norm` overflow) and the asymmetric path likewise.
//! - **DELTA-C (`search_asymmetric_subset` candidate list).** Empty list,
//!   `k == 0`, `k > m`, duplicate ids (the same doc scored more than
//!   once), and a duplicate-with-out-of-range id. The duplicate case is
//!   the interesting one: it is *accepted* and the doc is returned once
//!   per occurrence — documented here as the contract (the API does not
//!   deduplicate), with the out-of-range guard still firing when a bad id
//!   is mixed in.
//! - **DELTA-D (empty-index / empty-input search).** Search before any
//!   `add`, `add(&[])`, `swap_remove` down to empty then search,
//!   `swap_remove` + re-`add` (buffer integrity), `body_overlap_*` with
//!   an empty `doc_ids` (and with an id against an empty index), and a
//!   large `nq` × small `k` (the `result_buffer_len(nq, k)` axis).
//! - **DELTA-E (documented fail-loud contracts).** The bench-only
//!   `search_asymmetric_byte_lut` panics on a `b = 1` index by design;
//!   pin it so the documented restriction cannot silently regress into a
//!   wrong-result path.
//!
//! Verdict of the pass: **no genuine bug found** — every probe confirmed
//! a guard holding (clean `Err`, intentional fail-loud, or correct
//! result). All tests below are passing assertions of correct behaviour;
//! none are `#[ignore]`d.

use std::io::Write;

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use ordvec::rank::rank_norm;
use ordvec::{search_asymmetric_byte_lut, Bitmap, Rank, RankQuant, SignBitmap};

/// `MAX_VECTORS` from `rank_io` — the on-disk document-count ceiling.
/// Re-declared here (not imported) to keep the test independent of
/// whether the constant is re-exported; if the crate value ever changes,
/// the loader-rejection tests below still exercise the *boundary*
/// behaviour around whatever the loaders enforce.
const MAX_VECTORS_U32: u32 = 64 * 1024 * 1024;
/// `MAX_DIM` (= `u16::MAX`) and `MAX_SIGN_BITMAP_DIM` (= `1 << 24`).
const MAX_DIM_U32: u32 = u16::MAX as u32;
const MAX_SIGN_BITMAP_DIM_U32: u32 = 1 << 24;

fn make_corpus(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.random_range(-1.0..1.0)).collect()
}

/// Write `bytes` to a uniquely-named temp file and return its path.
/// Mirrors the `forge` helper in `src/rank_io.rs`'s test module
/// (pid + nanosecond nonce + suffix) so concurrent test binaries never
/// collide, and uses only `std::fs` / `std::env::temp_dir` (no
/// `tempfile` dev-dependency).
fn forge(suffix: &str, bytes: &[u8]) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!(
        "ordvec_redteam_delta_{}_{}_{}",
        std::process::id(),
        nonce,
        suffix
    ));
    std::fs::File::create(&p).unwrap().write_all(bytes).unwrap();
    p
}

/// Run all four `T::load` entry points against one forged file and assert
/// each returns `Err` without panicking. `catch_unwind` enforces the
/// no-panic half of the contract (a malformed file must never abort the
/// process); `is_err` enforces the rejection half.
fn assert_all_loaders_reject(path: &std::path::Path, label: &str) {
    let p = path.to_path_buf();
    let r1 = std::panic::catch_unwind(|| Rank::load(&p));
    assert!(r1.is_ok(), "Rank::load panicked on {label}");
    assert!(r1.unwrap().is_err(), "Rank::load accepted {label}");

    let r2 = std::panic::catch_unwind(|| RankQuant::load(&p));
    assert!(r2.is_ok(), "RankQuant::load panicked on {label}");
    assert!(r2.unwrap().is_err(), "RankQuant::load accepted {label}");

    let r3 = std::panic::catch_unwind(|| Bitmap::load(&p));
    assert!(r3.is_ok(), "Bitmap::load panicked on {label}");
    assert!(r3.unwrap().is_err(), "Bitmap::load accepted {label}");

    let r4 = std::panic::catch_unwind(|| SignBitmap::load(&p));
    assert!(r4.is_ok(), "SignBitmap::load panicked on {label}");
    assert!(r4.unwrap().is_err(), "SignBitmap::load accepted {label}");
}

// =====================================================================
// DELTA-A — loaders: adversarial header geometry.
// =====================================================================

/// DELTA-A1 (DoS): a forged TVR1 header declaring `n_vectors ==
/// MAX_VECTORS` with a valid `dim` but **no payload bytes**. The implied
/// payload is `1024 * 64Mi * 2 ≈ 137 GiB`; a naive loader that sizes a
/// buffer from the declared length before checking it against the file
/// would attempt a 137 GiB allocation. `check_payload_matches_file` runs
/// *before* any allocation, so the loader must reject this in negligible
/// time. We bound the wall-clock to catch a regression that re-orders the
/// allocation ahead of the size check.
#[test]
fn delta_a1_loader_rejects_huge_declared_nvectors_with_empty_payload() {
    let mut v = Vec::new();
    v.extend_from_slice(b"TVR1");
    v.push(1); // version
    v.extend_from_slice(&1024u32.to_le_bytes()); // dim (valid)
    v.extend_from_slice(&MAX_VECTORS_U32.to_le_bytes()); // n_vectors at the cap
                                                         // No payload bytes — declared payload ~137 GiB, file is header-only.
    let p = forge("dos_huge_nvectors.tvr", &v);

    let start = std::time::Instant::now();
    let r = std::panic::catch_unwind(|| Rank::load(&p));
    let elapsed = start.elapsed();
    std::fs::remove_file(&p).ok();

    assert!(r.is_ok(), "Rank::load panicked on the DoS header");
    assert!(
        r.unwrap().is_err(),
        "Rank::load must reject a header declaring a gigabyte payload over an empty file"
    );
    // The size check is O(1) (a `stream_position` + integer compare); a
    // generous ceiling still catches a regression that tries to allocate
    // ~137 GiB first.
    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "loader took {elapsed:?} to reject a tiny-file/huge-payload header — \
         a size guard must precede allocation",
    );
}

/// DELTA-A2: `n_vectors == MAX_VECTORS + 1` (one past the cap) must be
/// rejected by `check_n_vectors` for every format. The structural fuzz in
/// `main.rs` only exercises `u32::MAX`; this pins the exact boundary.
#[test]
fn delta_a2_loader_rejects_nvectors_one_past_max() {
    let over = MAX_VECTORS_U32 + 1;
    let mut v = Vec::new();
    v.extend_from_slice(b"TVR1");
    v.push(1);
    v.extend_from_slice(&1024u32.to_le_bytes()); // dim valid
    v.extend_from_slice(&over.to_le_bytes());
    let p = forge("nvectors_over_max.tvr", &v);
    let r = std::panic::catch_unwind(|| Rank::load(&p));
    std::fs::remove_file(&p).ok();
    assert!(r.is_ok(), "Rank::load panicked on n_vectors = MAX+1");
    assert!(
        r.unwrap().is_err(),
        "Rank::load must reject n_vectors = MAX_VECTORS + 1"
    );
}

/// DELTA-A3: `dim == 1` is one below the `[2, MAX_DIM]` floor enforced by
/// `check_dim`. A 1-dimensional rank vector is degenerate (the rank
/// transform / analytical norm assume `dim >= 2`), so the loader must
/// reject it — even though `dim == 1` would not, on its own, overflow any
/// size arithmetic.
#[test]
fn delta_a3_loader_rejects_dim_one() {
    let mut v = Vec::new();
    v.extend_from_slice(b"TVR1");
    v.push(1);
    v.extend_from_slice(&1u32.to_le_bytes()); // dim = 1 (< 2)
    v.extend_from_slice(&0u32.to_le_bytes()); // n_vectors = 0
    let p = forge("dim_one.tvr", &v);
    let r = std::panic::catch_unwind(|| Rank::load(&p));
    std::fs::remove_file(&p).ok();
    assert!(r.is_ok(), "Rank::load panicked on dim = 1");
    assert!(r.unwrap().is_err(), "Rank::load must reject dim = 1");
}

/// DELTA-A4: a `dim == 0` header must be rejected by *all four* loaders.
/// `dim = 0` satisfies `dim % 64 == 0` (a trap for the bitmap formats)
/// and would yield `qwords_per_vec == 0` / a div-by-zero downstream, so
/// the loaders must catch it at the header. Payload is empty to isolate
/// the dim gate.
#[test]
fn delta_a4_all_loaders_reject_dim_zero() {
    // TVR1
    {
        let mut v = Vec::new();
        v.extend_from_slice(b"TVR1");
        v.push(1);
        v.extend_from_slice(&0u32.to_le_bytes()); // dim
        v.extend_from_slice(&0u32.to_le_bytes()); // n_vectors
        let p = forge("dim0.tvr", &v);
        let r = std::panic::catch_unwind(|| Rank::load(&p));
        std::fs::remove_file(&p).ok();
        assert!(r.is_ok() && r.unwrap().is_err(), "TVR1 dim=0 must Err");
    }
    // TVRQ (extra `bits` byte)
    {
        let mut v = Vec::new();
        v.extend_from_slice(b"TVRQ");
        v.push(1);
        v.push(2); // bits = 2
        v.extend_from_slice(&0u32.to_le_bytes()); // dim
        v.extend_from_slice(&0u32.to_le_bytes()); // n_vectors
        let p = forge("dim0.tvrq", &v);
        let r = std::panic::catch_unwind(|| RankQuant::load(&p));
        std::fs::remove_file(&p).ok();
        assert!(r.is_ok() && r.unwrap().is_err(), "TVRQ dim=0 must Err");
    }
    // TVBM (extra `n_top` field)
    {
        let mut v = Vec::new();
        v.extend_from_slice(b"TVBM");
        v.push(1);
        v.extend_from_slice(&0u32.to_le_bytes()); // dim
        v.extend_from_slice(&0u32.to_le_bytes()); // n_top
        v.extend_from_slice(&0u32.to_le_bytes()); // n_vectors
        let p = forge("dim0.tvbm", &v);
        let r = std::panic::catch_unwind(|| Bitmap::load(&p));
        std::fs::remove_file(&p).ok();
        assert!(r.is_ok() && r.unwrap().is_err(), "TVBM dim=0 must Err");
    }
    // TVSB
    {
        let mut v = Vec::new();
        v.extend_from_slice(b"TVSB");
        v.push(1);
        v.extend_from_slice(&0u32.to_le_bytes()); // dim
        v.extend_from_slice(&0u32.to_le_bytes()); // n_vectors
        let p = forge("dim0.tvsb", &v);
        let r = std::panic::catch_unwind(|| SignBitmap::load(&p));
        std::fs::remove_file(&p).ok();
        assert!(r.is_ok() && r.unwrap().is_err(), "TVSB dim=0 must Err");
    }
}

/// DELTA-A5: TVBM `dim` that is a multiple of 64 but exceeds `MAX_DIM`
/// (`u16::MAX`). `Bitmap` carries the same `u16` rank-storage invariant as
/// `Rank`, so `check_dim` caps it at `MAX_DIM`; `65536` is the smallest
/// multiple of 64 above the cap and must be rejected (otherwise a loaded
/// index would panic on the first query's `dim as u16` truncation).
#[test]
fn delta_a5_tvbm_rejects_dim_over_max_dim() {
    let dim = MAX_DIM_U32 + 1; // 65536, a multiple of 64
    assert_eq!(dim % 64, 0, "test fixture: 65536 must be a multiple of 64");
    let mut v = Vec::new();
    v.extend_from_slice(b"TVBM");
    v.push(1);
    v.extend_from_slice(&dim.to_le_bytes());
    v.extend_from_slice(&100u32.to_le_bytes()); // n_top
    v.extend_from_slice(&0u32.to_le_bytes()); // n_vectors
    let p = forge("bm_dim_over_max.tvbm", &v);
    let r = std::panic::catch_unwind(|| Bitmap::load(&p));
    std::fs::remove_file(&p).ok();
    assert!(r.is_ok(), "Bitmap::load panicked on dim > MAX_DIM");
    assert!(
        r.unwrap().is_err(),
        "Bitmap::load must reject dim > MAX_DIM even when dim % 64 == 0"
    );
}

/// DELTA-A6: TVSB `dim` above `MAX_SIGN_BITMAP_DIM` (`1 << 24`). Sign
/// bitmaps do *not* share the `u16` rank cap (they round-trip dims well
/// above `u16::MAX`, pinned by `sign_bitmap`'s own `large_dim` test), so
/// their ceiling is the higher `MAX_SIGN_BITMAP_DIM`. Just over it, with a
/// 64-multiple dim, must still be rejected.
#[test]
fn delta_a6_tvsb_rejects_dim_over_max_sign_bitmap_dim() {
    let dim = MAX_SIGN_BITMAP_DIM_U32 + 64; // 16_777_280, a multiple of 64
    assert_eq!(dim % 64, 0, "test fixture: dim must be a multiple of 64");
    let mut v = Vec::new();
    v.extend_from_slice(b"TVSB");
    v.push(1);
    v.extend_from_slice(&dim.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes()); // n_vectors
    let p = forge("sb_dim_over_max.tvsb", &v);
    let r = std::panic::catch_unwind(|| SignBitmap::load(&p));
    std::fs::remove_file(&p).ok();
    assert!(
        r.is_ok(),
        "SignBitmap::load panicked on dim > MAX_SIGN_BITMAP_DIM"
    );
    assert!(
        r.unwrap().is_err(),
        "SignBitmap::load must reject dim > MAX_SIGN_BITMAP_DIM"
    );
}

/// DELTA-A7: version bytes other than `1`. `0`, `2`, and `255` must each
/// be rejected by every loader's `if ver[0] != VERSION` check (a forged or
/// future-format file must not be silently parsed under the v1 layout).
#[test]
fn delta_a7_all_loaders_reject_bad_version_bytes() {
    for ver in [0u8, 2u8, 255u8] {
        let mut v = Vec::new();
        v.extend_from_slice(b"TVR1");
        v.push(ver);
        v.extend_from_slice(&1024u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        let p = forge(&format!("ver_{ver}.tvr"), &v);
        let r = std::panic::catch_unwind(|| Rank::load(&p));
        std::fs::remove_file(&p).ok();
        assert!(r.is_ok(), "Rank::load panicked on version {ver}");
        assert!(
            r.unwrap().is_err(),
            "Rank::load must reject version byte {ver}"
        );
    }
}

/// DELTA-A8: all-`0xFF` files at each format's header length and a few
/// larger sizes. `0xFF` magic fails the magic check; `0xFF` version fails
/// the version check; a larger all-`0xFF` file decodes to an absurd
/// `dim`/`n_vectors` that trips the size guards. Every loader must `Err`
/// without panicking on every length.
#[test]
fn delta_a8_all_loaders_reject_all_ff_files() {
    for len in [13usize, 14, 17, 32, 64, 256, 4096] {
        let bytes = vec![0xFFu8; len];
        let p = forge(&format!("all_ff_{len}.bin"), &bytes);
        assert_all_loaders_reject(&p, &format!("all-0xFF len={len}"));
        std::fs::remove_file(&p).ok();
    }
}

// =====================================================================
// DELTA-B — integer overflow at the dimension ceiling (64-bit).
// =====================================================================

/// DELTA-B1: symmetric `Rank::search` at the true `dim = u16::MAX`
/// ceiling. The inner loop accumulates
/// `acc: i64 += (2·q − (D−1)) · (2·doc − (D−1))` over `dim` coordinates.
/// At `dim = 65535` the per-term magnitude peaks near `65535² ≈ 4.3e9`
/// (already past `i32::MAX`, hence the deliberate `as i64` widening) and
/// the sum over `dim` terms peaks near `2.8e14` — comfortably inside
/// `i64` but worth pinning at the exact boundary so a future refactor
/// that narrows the accumulator (or `mean_2x`'s `i32`) is caught. Also
/// exercises `rank_norm(65535)`, which forms its product in `f64` to
/// avoid `f32` overflow. Both score paths must stay finite.
#[test]
fn delta_b1_rank_symmetric_no_overflow_at_max_dim() {
    let dim = u16::MAX as usize; // 65535 — the largest constructible dim
    let n = 3;
    let corpus = make_corpus(8001, n, dim);
    let mut idx = Rank::new(dim);
    idx.add(&corpus);
    let query = make_corpus(8002, 1, dim);

    // rank_norm must not overflow to a non-finite value at the ceiling.
    let norm = rank_norm(dim);
    assert!(
        norm.is_finite() && norm > 0.0,
        "rank_norm({dim}) = {norm} must be finite and positive"
    );

    let res = idx.search(&query, n);
    assert_eq!(res.k, n);
    for &s in res.scores_for_query(0) {
        assert!(
            s.is_finite(),
            "symmetric score at dim={dim} must be finite (no i64/i32 wrap, no norm overflow); got {s}"
        );
    }

    // Asymmetric path shares the same dim ceiling and the f32 dot/accumulate.
    let resa = idx.search_asymmetric(&query, n);
    for &s in resa.scores_for_query(0) {
        assert!(
            s.is_finite(),
            "asymmetric score at dim={dim} must be finite; got {s}"
        );
    }
}

// =====================================================================
// DELTA-C — search_asymmetric_subset: candidate-list edge cases.
// =====================================================================

/// DELTA-C1: an empty candidate list returns empty `(scores, indices)`
/// (the `m == 0 → k_eff == 0` early-out), and `k == 0` with a non-empty
/// list does the same. Neither path may panic on the zero-length scratch
/// buffer or the `vec![0u8; m * bpv]` gather.
#[test]
fn delta_c1_subset_empty_list_and_zero_k() {
    let dim = 64;
    let n = 32;
    let corpus = make_corpus(8101, n, dim);
    let mut idx = RankQuant::new(dim, 2);
    idx.add(&corpus);
    let query = make_corpus(8102, 1, dim);

    let (s0, g0) = idx.search_asymmetric_subset(&query, &[], 5);
    assert!(
        s0.is_empty() && g0.is_empty(),
        "empty candidate list must return empty"
    );

    let (s1, g1) = idx.search_asymmetric_subset(&query, &[0, 1, 2], 0);
    assert!(s1.is_empty() && g1.is_empty(), "k == 0 must return empty");
}

/// DELTA-C2 (contract pin): duplicate candidate ids are **accepted**, and
/// the same global doc is returned once per occurrence. This is the
/// documented gather behaviour — the subset API scores each candidate
/// position independently (`sub_packed[i*bpv..]` is a copy per slot) and
/// `TopK` keeps distinct *local* positions that map back to the same
/// global id. The API deliberately does not deduplicate; a caller that
/// passes a deduped list gets deduped results. Pinned so the behaviour is
/// a conscious contract, not an accident: feeding `[7, 7, 7]` with `k = 3`
/// returns three identical scores all mapping to global id 7.
#[test]
fn delta_c2_subset_duplicate_ids_returned_per_occurrence() {
    let dim = 64;
    let n = 32;
    let corpus = make_corpus(8201, n, dim);
    let mut idx = RankQuant::new(dim, 2);
    idx.add(&corpus);
    let query = make_corpus(8202, 1, dim);

    let cands: Vec<u32> = vec![7, 7, 7];
    let (scores, global) = idx.search_asymmetric_subset(&query, &cands, 3);
    assert_eq!(scores.len(), 3);
    assert_eq!(global.len(), 3);
    // Every returned id is the duplicated candidate.
    assert!(
        global.iter().all(|&g| g == 7),
        "duplicate candidate list [7,7,7] must map every result to global id 7; got {global:?}",
    );
    // The repeated doc yields the identical score each time (same bytes
    // gathered, same kernel) and is finite.
    let s0 = scores[0];
    assert!(s0.is_finite(), "duplicate-candidate score must be finite");
    for &s in &scores {
        assert_eq!(
            s, s0,
            "all occurrences of one duplicated doc must score identically"
        );
    }

    // Cross-check against the single-occurrence score: the per-doc value
    // is independent of how many times the id appears.
    let (single, _) = idx.search_asymmetric_subset(&query, &[7], 1);
    assert!(
        (single[0] - s0).abs() < 1e-6,
        "duplicate-occurrence score {s0} must equal the single-occurrence score {}",
        single[0],
    );
}

/// DELTA-C3: `k` larger than the candidate count clamps to `m`
/// (`k_eff = k.min(m)`) — even with a duplicate in the list. `k = 10`
/// over a 3-element list `[3, 3, 9]` must return exactly 3 results, not
/// pad to 10 or over-read.
#[test]
fn delta_c3_subset_k_greater_than_m_clamps() {
    let dim = 64;
    let n = 32;
    let corpus = make_corpus(8301, n, dim);
    let mut idx = RankQuant::new(dim, 2);
    idx.add(&corpus);
    let query = make_corpus(8302, 1, dim);

    let cands: Vec<u32> = vec![3, 3, 9];
    let (scores, global) = idx.search_asymmetric_subset(&query, &cands, 10);
    assert_eq!(scores.len(), 3, "k must clamp to candidate count m");
    assert_eq!(global.len(), 3);
    // Returned globals are drawn only from the candidate set.
    for &g in &global {
        assert!(
            g < 0 || cands.contains(&(g as u32)),
            "result id {g} not in candidate set {cands:?}"
        );
    }
}

/// DELTA-C4: a duplicated id mixed with an out-of-range id
/// (`[5, 999, 5]`, `n_vectors == 32`) must still trip the bounds assert —
/// the out-of-range guard scans the *whole* list, so a dup neither masks
/// nor bypasses it. Pins the `alpha` "subset rejects out-of-range" guard
/// against a duplicate-laden adversarial list.
#[test]
#[should_panic(expected = "candidate id out of range")]
fn delta_c4_subset_dup_plus_oob_still_rejected() {
    let dim = 64;
    let n = 32;
    let corpus = make_corpus(8401, n, dim);
    let mut idx = RankQuant::new(dim, 2);
    idx.add(&corpus);
    let query = make_corpus(8402, 1, dim);
    // 999 >= n_vectors (32): must panic before the gather, dup or not.
    let _ = idx.search_asymmetric_subset(&query, &[5, 999, 5], 3);
}

// =====================================================================
// DELTA-D — empty-index / empty-input search paths.
// =====================================================================

/// DELTA-D1: searching an index that has had no `add` (and a degenerate
/// `add(&[])`) must clamp `k` to `0` and return a correctly-shaped empty
/// result across all four types, never panic on `par_chunks_mut(0)` or an
/// empty scan.
#[test]
fn delta_d1_search_before_any_add() {
    let dim = 128;
    let query = make_corpus(8501, 1, dim);

    // Rank
    let idx = Rank::new(dim);
    let r = idx.search(&query, 5);
    assert_eq!(r.k, 0);
    assert!(r.scores.is_empty() && r.indices.is_empty());

    // RankQuant (both symmetric and asymmetric)
    let idx = RankQuant::new(dim, 2);
    assert_eq!(idx.search(&query, 5).k, 0);
    assert_eq!(idx.search_asymmetric(&query, 5).k, 0);

    // Bitmap
    let idx = Bitmap::new(dim, 32);
    assert_eq!(idx.search(&query, 5).k, 0);
    assert!(idx.top_m_candidates(&query, 5).is_empty());

    // SignBitmap — single and batched candidate paths.
    let idx = SignBitmap::new(dim);
    assert!(idx.top_m_candidates(&query, 5).is_empty());
    let batched = idx.top_m_candidates_batched(&make_corpus(8502, 3, dim), 5);
    assert_eq!(batched.len(), 3);
    assert!(batched.iter().all(|v| v.is_empty()));

    // Degenerate add(&[]) is a no-op (n == 0) and leaves the index empty.
    let mut idx = Rank::new(dim);
    idx.add(&[]);
    assert_eq!(idx.len(), 0);
}

/// DELTA-D2: `body_overlap_scores_subset` with an empty `doc_ids` slice
/// is a no-op (empty `out`), even though the AVX-512 dispatch is reached
/// (the `n == 0` loop body never runs). And an in-range-looking id `0`
/// against an *empty* index must fail the bounds assert (`0 < 0` is
/// false) — the guard does not special-case the empty corpus.
#[test]
fn delta_d2_body_overlap_empty_doc_ids_and_empty_index() {
    let dim = 128;
    let n_top = 32;

    // Empty doc_ids on a populated index: no-op, no panic.
    let mut idx = Bitmap::new(dim, n_top);
    idx.add(&make_corpus(8601, 8, dim));
    let q = make_corpus(8602, 1, dim);
    let qb = idx.build_query_bitmap_fp32(&q);
    let mut out: Vec<u32> = Vec::new();
    idx.body_overlap_scores_subset(&qb, &[], &mut out);
    assert!(out.is_empty(), "empty doc_ids must leave out empty");

    // Id 0 against an empty index must be rejected by the bounds assert.
    let empty = Bitmap::new(dim, n_top);
    let qb_empty = empty.build_query_bitmap_fp32(&q);
    let mut out1 = vec![0u32; 1];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        empty.body_overlap_scores_subset(&qb_empty, &[0], &mut out1)
    }));
    assert!(
        r.is_err(),
        "body_overlap_scores_subset must reject id 0 against an empty index"
    );
}

/// DELTA-D3: `swap_remove` down to empty, then search; and `swap_remove`
/// of a middle element followed by re-`add` (buffer-integrity check). The
/// remove must keep the packed buffer consistent so a later search over
/// the surviving + newly-added docs returns exactly the live count.
#[test]
fn delta_d3_swap_remove_to_empty_then_readd() {
    let dim = 64;

    // Remove the only element → empty, then search returns nothing.
    let mut idx = Rank::new(dim);
    idx.add(&make_corpus(8701, 1, dim));
    let moved = idx.swap_remove(0);
    assert_eq!(moved, 0, "removing the sole element reports last index 0");
    assert!(idx.is_empty());
    let q = make_corpus(8702, 1, dim);
    assert_eq!(idx.search(&q, 5).k, 0, "search over emptied index is empty");

    // Remove a middle element, re-add, and verify the live count searches
    // cleanly (no stale bytes, no over/under count).
    let mut idx = Rank::new(dim);
    idx.add(&make_corpus(8703, 4, dim)); // ids 0..4
    let last_moved = idx.swap_remove(1); // pulls id 3 into slot 1
    assert_eq!(last_moved, 3);
    assert_eq!(idx.len(), 3);
    idx.add(&make_corpus(8704, 2, dim)); // back to 5
    assert_eq!(idx.len(), 5);
    let res = idx.search(&q, 100); // k clamps to 5
    let valid = res.indices_for_query(0).iter().filter(|&&i| i >= 0).count();
    assert_eq!(
        valid, 5,
        "all 5 live docs must be returned after remove+readd"
    );
}

/// DELTA-D4: a large query count `nq` with a small `k` exercises the
/// `result_buffer_len(nq, k) = nq * k` axis (the half of the
/// capacity-overflow guard the `beta` suite covers from the `k` side).
/// `5000 * 2 = 10000` slots must allocate and fill cleanly, with every
/// per-query block correctly sliceable.
#[test]
fn delta_d4_large_nq_small_k() {
    let dim = 64;
    let n = 8;
    let mut idx = Rank::new(dim);
    idx.add(&make_corpus(8801, n, dim));
    let nq = 5000;
    let queries = make_corpus(8802, nq, dim);

    let res = idx.search(&queries, 2);
    assert_eq!(res.nq, nq);
    assert_eq!(res.k, 2);
    assert_eq!(res.scores.len(), nq * 2);
    assert_eq!(res.indices.len(), nq * 2);
    // Spot-check a couple of per-query blocks: each returns min(k, n) live ids.
    for qi in [0usize, nq / 2, nq - 1] {
        let valid = res
            .indices_for_query(qi)
            .iter()
            .filter(|&&i| i >= 0)
            .count();
        assert_eq!(
            valid, 2,
            "query {qi} must return k=2 live results (n=8 >= 2)"
        );
    }
}

// =====================================================================
// DELTA-E — documented fail-loud contracts.
// =====================================================================

/// DELTA-E1: the bench-only `search_asymmetric_byte_lut` is documented to
/// support only `bits ∈ {2, 4}` and to panic on a `b = 1` index. Pin the
/// panic so the documented restriction cannot silently regress into a
/// wrong-result path (the production `RankQuant::search_asymmetric` routes
/// `b = 1` to the scalar LUT and is unaffected — covered by the `beta`
/// suite). This is an intentional, documented contract, not a bug.
#[test]
#[should_panic(expected = "byte-LUT path only supports bits")]
fn delta_e1_byte_lut_panics_on_b1_index() {
    let dim = 64;
    let mut idx = RankQuant::new(dim, 1);
    idx.add(&make_corpus(8901, 8, dim));
    let query = make_corpus(8902, 1, dim);
    let _ = search_asymmetric_byte_lut(&idx, &query, 3);
}
