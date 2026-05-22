//! Red-team hardening tests for the rank-mode loaders (ported from
//! turbovec).
//!
//! These exercise two deserialization gaps at the `rank_io::*` layer.
//! Ported verbatim from turbovec's `redteam_delta.rs`; that suite was
//! already rank-only (no FFI/pyo3 cases), so the full set carries over.
//! The `extern crate blas_src;` line is dropped because `ordvec` has no
//! BLAS dependency.
//!
//! * **TV-DESER-004** — [`rank_io::load_rankquant`] validated `bits`
//!   but not the `dim % (1 << bits) == 0` / `dim % (8 / bits) == 0`
//!   constant-composition invariant, even though the module doc
//!   (rank_io.rs ~26-27) claims it is enforced on load. The invariant
//!   was only re-checked one layer up in `RankQuantIndex::load`, so a
//!   direct caller of `load_rankquant` (or any future consumer) could
//!   silently accept a malformed packed buffer.
//! * **TV-DESER-005** — [`rank_io::load_rank`] accepted rank values
//!   `>= dim`. Out-of-range ranks are not an OOB read (they index a
//!   per-query LUT sized to `dim` downstream), but they silently
//!   corrupt Spearman scores. A loader is the right boundary to reject
//!   them.
//!
//! Both loaders must return `Err(InvalidData)`, never panic.

use std::io::Write;
use std::path::PathBuf;

use ordvec::rank_io::{load_rank, load_rankquant};

/// Write `bytes` to a uniquely-named temp file and return its path.
fn forge(suffix: &str, bytes: &[u8]) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!(
        "redteam_delta_{}_{}_{}",
        std::process::id(),
        nonce,
        suffix
    ));
    std::fs::File::create(&p).unwrap().write_all(bytes).unwrap();
    p
}

// -------------------------------------------------------------------
// TV-DESER-004: load_rankquant must enforce dim % (1<<bits) == 0.
// -------------------------------------------------------------------

/// A `.tvrq` header with `bits=2, dim=6` violates the
/// constant-composition invariant (`6 % 4 != 0`). The packed payload
/// is supplied at the *header-consistent* size (`n_vectors * dim *
/// bits / 8`) so the file passes every length/overflow check and the
/// loader is forced to either reject on the divisibility rule or
/// (pre-fix) wrongly return `Ok`.
#[test]
fn tvdeser004_load_rankquant_rejects_dim_not_multiple_of_2pow_bits() {
    // n_vectors=0 keeps the payload empty (0 * 6 * 2 / 8 = 0), so the
    // only thing under test is the divisibility gate, not payload
    // length. dim=6, bits=2 → 6 % 4 = 2 ≠ 0.
    let mut v = Vec::new();
    v.extend_from_slice(b"TVRQ");
    v.push(1); // version
    v.push(2); // bits = 2  → n_buckets = 4
    v.extend_from_slice(&6u32.to_le_bytes()); // dim = 6 (not a multiple of 4)
    v.extend_from_slice(&0u32.to_le_bytes()); // n_vectors = 0
    let path = forge("tvrq_dim6_bits2.tvrq", &v);

    let result = std::panic::catch_unwind(|| load_rankquant(&path));
    std::fs::remove_file(&path).ok();

    let result = result.expect("load_rankquant panicked on bits=2 dim=6");
    assert!(
        result.is_err(),
        "load_rankquant accepted bits=2 dim=6 (dim % 4 != 0); expected Err"
    );
}

/// A `.tvrq` with `bits=4, dim=4` is divisible by `1<<bits = 16`?  No:
/// `4 % 16 != 0`. This guards the same gate from the other side
/// (large bucket count, small dim) and also trips `dim % (8/bits)` —
/// `8/4 = 2`, `4 % 2 == 0` is fine, so the rejection is owed purely to
/// the 2^bits rule, mirroring the documented invariant ordering.
#[test]
fn tvdeser004_load_rankquant_rejects_dim_smaller_than_buckets() {
    let mut v = Vec::new();
    v.extend_from_slice(b"TVRQ");
    v.push(1); // version
    v.push(4); // bits = 4 → n_buckets = 16
    v.extend_from_slice(&4u32.to_le_bytes()); // dim = 4 (not a multiple of 16)
    v.extend_from_slice(&0u32.to_le_bytes()); // n_vectors = 0
    let path = forge("tvrq_dim4_bits4.tvrq", &v);

    let result = std::panic::catch_unwind(|| load_rankquant(&path));
    std::fs::remove_file(&path).ok();

    let result = result.expect("load_rankquant panicked on bits=4 dim=4");
    assert!(
        result.is_err(),
        "load_rankquant accepted bits=4 dim=4 (dim % 16 != 0); expected Err"
    );
}

/// Sanity guard: a *valid* `.tvrq` (bits=2, dim=8, both invariants
/// satisfied, empty corpus) must still load. Ensures the new
/// divisibility gate does not over-reject the happy path.
#[test]
fn tvdeser004_load_rankquant_accepts_valid_dim() {
    let mut v = Vec::new();
    v.extend_from_slice(b"TVRQ");
    v.push(1); // version
    v.push(2); // bits = 2
    v.extend_from_slice(&8u32.to_le_bytes()); // dim = 8 (8 % 4 == 0, 8 % 4 == 0)
    v.extend_from_slice(&0u32.to_le_bytes()); // n_vectors = 0
    let path = forge("tvrq_dim8_bits2.tvrq", &v);

    let result = load_rankquant(&path);
    std::fs::remove_file(&path).ok();

    let (bits, dim, n, packed) = result.expect("valid TVRQ should load");
    assert_eq!(bits, 2);
    assert_eq!(dim, 8);
    assert_eq!(n, 0);
    assert!(packed.is_empty());
}

// -------------------------------------------------------------------
// TV-DESER-005: load_rank must reject any rank value >= dim.
// -------------------------------------------------------------------

/// A `.tvr` with `dim=4, n_vectors=1, ranks=[60000, 1, 2, 3]`. The
/// payload length (4 ranks * 2 bytes = 8) matches the header, so all
/// length/overflow checks pass and the loader reaches the value scan.
/// `60000 >= 4`, so the loader must reject it.
#[test]
fn tvdeser005_load_rank_rejects_rank_value_ge_dim() {
    let dim: u32 = 4;
    let n_vectors: u32 = 1;
    let ranks: [u16; 4] = [60000, 1, 2, 3];

    let mut v = Vec::new();
    v.extend_from_slice(b"TVR1");
    v.push(1); // version
    v.extend_from_slice(&dim.to_le_bytes());
    v.extend_from_slice(&n_vectors.to_le_bytes());
    for &r in &ranks {
        v.extend_from_slice(&r.to_le_bytes());
    }
    let path = forge("tvr_rank_ge_dim.tvr", &v);

    let result = std::panic::catch_unwind(|| load_rank(&path));
    std::fs::remove_file(&path).ok();

    let result = result.expect("load_rank panicked on rank >= dim");
    assert!(
        result.is_err(),
        "load_rank accepted ranks=[60000,1,2,3] with dim=4 (60000 >= dim); expected Err"
    );
}

/// Sanity guard: a *valid* `.tvr` (a true permutation of 0..dim) must
/// still load. Ensures the value scan does not over-reject.
#[test]
fn tvdeser005_load_rank_accepts_valid_permutation() {
    let dim: u32 = 4;
    let n_vectors: u32 = 2;
    // Two valid rows, each a permutation of 0..4.
    let ranks: [u16; 8] = [0, 1, 2, 3, 3, 2, 1, 0];

    let mut v = Vec::new();
    v.extend_from_slice(b"TVR1");
    v.push(1); // version
    v.extend_from_slice(&dim.to_le_bytes());
    v.extend_from_slice(&n_vectors.to_le_bytes());
    for &r in &ranks {
        v.extend_from_slice(&r.to_le_bytes());
    }
    let path = forge("tvr_valid_perm.tvr", &v);

    let result = load_rank(&path);
    std::fs::remove_file(&path).ok();

    let (d, n, loaded) = result.expect("valid TVR1 should load");
    assert_eq!(d, 4);
    assert_eq!(n, 2);
    assert_eq!(loaded, ranks.to_vec());
}
