//! Loader *semantic*-validation tests.
//!
//! The structural loader-fuzz test in `main.rs` proves malformed *shapes*
//! return `Err` without panicking. These tests cover the complementary
//! contract: a file with valid structure (correct magic / version / dim /
//! n_vectors / payload length) but a payload that violates the type's
//! *semantic* invariant must also be rejected — the scoring math depends on
//! those invariants, and a silently-wrong score is worse than a clean error.
//! Each case pairs a positive control (a freshly-written valid index still
//! round-trips) with a corrupted-but-well-shaped negative case.

use std::io::Write;

use ordvec::{Bitmap, Rank, RankQuant, SignBitmap};

use crate::{make_corpus, D, N};

fn read_bytes(p: &std::path::Path) -> Vec<u8> {
    std::fs::read(p).unwrap()
}
fn write_bytes(p: &std::path::Path, b: &[u8]) {
    std::fs::File::create(p).unwrap().write_all(b).unwrap();
}
fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "ordvec_loadval_{}_{}.bin",
        name,
        std::process::id()
    ))
}

#[test]
fn load_rank_rejects_non_permutation_row() {
    let corpus = make_corpus(1);
    let mut idx = Rank::new(D);
    idx.add(&corpus);
    let p = tmp("rank_perm");
    idx.write(&p).unwrap();
    // Positive control: the valid file round-trips.
    assert!(Rank::load(&p).is_ok(), "valid Rank file must load");

    // TVR1 header is 13 bytes; payload is u16 LE ranks. Force ranks[1] ==
    // ranks[0] in row 0, turning the row into a non-permutation (a repeat).
    let mut bytes = read_bytes(&p);
    let (a, b) = (13usize, 15usize); // byte offsets of the first two u16 ranks
    bytes[b] = bytes[a];
    bytes[b + 1] = bytes[a + 1];
    write_bytes(&p, &bytes);

    let r = Rank::load(&p);
    std::fs::remove_file(&p).ok();
    assert!(
        r.is_err(),
        "Rank::load must reject a row that is not a permutation of [0, dim)"
    );
}

#[test]
fn load_rankquant_rejects_skewed_composition() {
    let corpus = make_corpus(2);
    let mut idx = RankQuant::new(D, 2);
    idx.add(&corpus);
    let p = tmp("rq_comp");
    idx.write(&p).unwrap();
    assert!(
        RankQuant::load(&p).is_ok(),
        "valid RankQuant file must load"
    );

    // TVRQ header is 14 bytes. Zero the entire packed payload so every
    // coordinate decodes to bucket 0 — a maximally skewed composition that
    // violates the dim/2^bits-per-bucket invariant on the very first row.
    let mut bytes = read_bytes(&p);
    for byte in bytes.iter_mut().skip(14) {
        *byte = 0;
    }
    write_bytes(&p, &bytes);

    let r = RankQuant::load(&p);
    std::fs::remove_file(&p).ok();
    assert!(
        r.is_err(),
        "RankQuant::load must reject a document whose bucket composition is not uniform"
    );
}

#[test]
fn load_bitmap_rejects_wrong_popcount_row() {
    let corpus = make_corpus(3);
    let n_top = D / 4;
    let mut idx = Bitmap::new(D, n_top);
    idx.add(&corpus);
    let p = tmp("bm_pop");
    idx.write(&p).unwrap();
    assert!(Bitmap::load(&p).is_ok(), "valid Bitmap file must load");

    // TVBM header is 17 bytes; payload is u64 LE words, qpv = dim/64 per doc.
    // Zero the first document's whole row so its popcount becomes 0 != n_top.
    let qpv = D / 64;
    let mut bytes = read_bytes(&p);
    for byte in bytes.iter_mut().skip(17).take(qpv * 8) {
        *byte = 0;
    }
    write_bytes(&p, &bytes);

    let r = Bitmap::load(&p);
    std::fs::remove_file(&p).ok();
    assert!(
        r.is_err(),
        "Bitmap::load must reject a document whose popcount != n_top"
    );
}

#[test]
fn load_sign_bitmap_accepts_any_bit_pattern() {
    // SignBitmap has no composition invariant: any bit pattern is a valid
    // document. A corrupted-but-well-shaped payload must therefore still load
    // (no false rejection) — the deliberate complement of the three checks
    // above, pinning that the sign-bitmap loader is intentionally
    // structural-only.
    let corpus = make_corpus(4);
    let mut idx = SignBitmap::new(D);
    idx.add(&corpus);
    let p = tmp("sb_any");
    idx.write(&p).unwrap();

    // TVSB header is 13 bytes. Flip bits across the payload; the result is
    // still a structurally valid sign bitmap of the same shape.
    let mut bytes = read_bytes(&p);
    for byte in bytes.iter_mut().skip(13) {
        *byte ^= 0xAA;
    }
    write_bytes(&p, &bytes);

    let loaded = SignBitmap::load(&p);
    std::fs::remove_file(&p).ok();
    let loaded = loaded.expect("SignBitmap::load must accept any bit pattern");
    assert_eq!(
        loaded.len(),
        N,
        "sign bitmap doc count preserved after edit"
    );
}

#[test]
fn loaders_reject_trailing_bytes() {
    // Every v1 format's payload is the file's final section, so the loader
    // requires the declared payload to consume the rest of the file exactly
    // (`check_payload_matches_file`). A structurally-valid file with even one
    // extra trailing byte must be rejected on all four formats — otherwise a
    // record could smuggle data past a smaller declared payload, or silent
    // corruption would pass unnoticed. One byte is appended past the payload
    // and each loader must now error.
    let corpus = make_corpus(5);
    let n_top = D / 4;

    // Rank (.tvr)
    {
        let mut idx = Rank::new(D);
        idx.add(&corpus);
        let p = tmp("rank_trail");
        idx.write(&p).unwrap();
        assert!(Rank::load(&p).is_ok(), "valid Rank file must load");
        let mut bytes = read_bytes(&p);
        bytes.push(0x00);
        write_bytes(&p, &bytes);
        let r = Rank::load(&p);
        std::fs::remove_file(&p).ok();
        assert!(r.is_err(), "Rank::load must reject trailing bytes");
    }

    // RankQuant (.tvrq)
    {
        let mut idx = RankQuant::new(D, 2);
        idx.add(&corpus);
        let p = tmp("rq_trail");
        idx.write(&p).unwrap();
        assert!(
            RankQuant::load(&p).is_ok(),
            "valid RankQuant file must load"
        );
        let mut bytes = read_bytes(&p);
        bytes.push(0x00);
        write_bytes(&p, &bytes);
        let r = RankQuant::load(&p);
        std::fs::remove_file(&p).ok();
        assert!(r.is_err(), "RankQuant::load must reject trailing bytes");
    }

    // Bitmap (.tvbm)
    {
        let mut idx = Bitmap::new(D, n_top);
        idx.add(&corpus);
        let p = tmp("bm_trail");
        idx.write(&p).unwrap();
        assert!(Bitmap::load(&p).is_ok(), "valid Bitmap file must load");
        let mut bytes = read_bytes(&p);
        bytes.push(0x00);
        write_bytes(&p, &bytes);
        let r = Bitmap::load(&p);
        std::fs::remove_file(&p).ok();
        assert!(r.is_err(), "Bitmap::load must reject trailing bytes");
    }

    // SignBitmap (.tvsb) — no composition invariant, but the exact-EOF
    // length guard still applies.
    {
        let mut idx = SignBitmap::new(D);
        idx.add(&corpus);
        let p = tmp("sb_trail");
        idx.write(&p).unwrap();
        assert!(
            SignBitmap::load(&p).is_ok(),
            "valid SignBitmap file must load"
        );
        let mut bytes = read_bytes(&p);
        bytes.push(0x00);
        write_bytes(&p, &bytes);
        let r = SignBitmap::load(&p);
        std::fs::remove_file(&p).ok();
        assert!(r.is_err(), "SignBitmap::load must reject trailing bytes");
    }
}
