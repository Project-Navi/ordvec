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

use std::io::{Cursor, Write};

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

fn assert_load_err_contains<T>(result: std::io::Result<T>, expected: &str) {
    let Err(err) = result else {
        panic!("expected error containing {expected:?}, got Ok(_)");
    };
    let text = err.to_string();
    assert!(
        text.contains(expected),
        "expected error containing {expected:?}, got {text:?}"
    );
}

fn set_u32_field(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn prefixed_cursor(bytes: &[u8]) -> Cursor<Vec<u8>> {
    const PREFIX: &[u8] = b"container-prefix";
    let mut prefixed = PREFIX.to_vec();
    prefixed.extend_from_slice(bytes);
    let mut cursor = Cursor::new(prefixed);
    cursor.set_position(PREFIX.len() as u64);
    cursor
}

fn append_trailer(mut bytes: Vec<u8>) -> Cursor<Vec<u8>> {
    bytes.extend_from_slice(b"next-record");
    Cursor::new(bytes)
}

#[test]
fn public_stream_persistence_roundtrips_core_formats() {
    let corpus = make_corpus(90_001);
    let query = &make_corpus(90_002)[..D];

    {
        let mut idx = Rank::new(D);
        idx.add(&corpus);
        let mut bytes = Vec::new();
        idx.write_to(&mut bytes).unwrap();
        assert_eq!(&bytes[..4], b"OVR1");

        let p = tmp("rank_stream_bytes");
        idx.write(&p).unwrap();
        assert_eq!(read_bytes(&p), bytes);
        std::fs::remove_file(&p).ok();

        let from_bytes = Rank::load_from_bytes(&bytes).unwrap();
        let from_reader = Rank::read_from(prefixed_cursor(&bytes)).unwrap();
        assert_eq!(from_bytes.len(), idx.len());
        assert_eq!(from_reader.dim(), idx.dim());
        assert_eq!(
            from_bytes.search(query, 10).indices_for_query(0),
            idx.search(query, 10).indices_for_query(0)
        );
        assert_eq!(
            from_reader.search(query, 10).indices_for_query(0),
            idx.search(query, 10).indices_for_query(0)
        );
    }

    {
        let mut idx = RankQuant::new(D, 2);
        idx.add(&corpus);
        let mut bytes = Vec::new();
        idx.write_to(&mut bytes).unwrap();
        assert_eq!(&bytes[..4], b"OVRQ");

        let p = tmp("rankquant_stream_bytes");
        idx.write(&p).unwrap();
        assert_eq!(read_bytes(&p), bytes);
        std::fs::remove_file(&p).ok();

        let from_bytes = RankQuant::load_from_bytes(&bytes).unwrap();
        let from_reader = RankQuant::read_from(prefixed_cursor(&bytes)).unwrap();
        assert_eq!(from_bytes.len(), idx.len());
        assert_eq!(from_reader.bits(), idx.bits());
        assert_eq!(
            from_bytes.search_asymmetric(query, 10).indices_for_query(0),
            idx.search_asymmetric(query, 10).indices_for_query(0)
        );
        assert_eq!(
            from_reader
                .search_asymmetric(query, 10)
                .indices_for_query(0),
            idx.search_asymmetric(query, 10).indices_for_query(0)
        );
    }

    {
        let mut idx = Bitmap::new(D, D / 4);
        idx.add(&corpus);
        let mut bytes = Vec::new();
        idx.write_to(&mut bytes).unwrap();
        assert_eq!(&bytes[..4], b"OVBM");

        let p = tmp("bitmap_stream_bytes");
        idx.write(&p).unwrap();
        assert_eq!(read_bytes(&p), bytes);
        std::fs::remove_file(&p).ok();

        let from_bytes = Bitmap::load_from_bytes(&bytes).unwrap();
        let from_reader = Bitmap::read_from(prefixed_cursor(&bytes)).unwrap();
        assert_eq!(from_bytes.len(), idx.len());
        assert_eq!(from_reader.n_top(), idx.n_top());
        assert_eq!(
            from_bytes.search(query, 10).indices_for_query(0),
            idx.search(query, 10).indices_for_query(0)
        );
        assert_eq!(
            from_reader.top_m_candidates(query, 32),
            idx.top_m_candidates(query, 32)
        );
    }

    {
        let mut idx = SignBitmap::new(D);
        idx.add(&corpus);
        let mut bytes = Vec::new();
        idx.write_to(&mut bytes).unwrap();
        assert_eq!(&bytes[..4], b"OVSB");

        let p = tmp("sign_bitmap_stream_bytes");
        idx.write(&p).unwrap();
        assert_eq!(read_bytes(&p), bytes);
        std::fs::remove_file(&p).ok();

        let from_bytes = SignBitmap::load_from_bytes(&bytes).unwrap();
        let from_reader = SignBitmap::read_from(prefixed_cursor(&bytes)).unwrap();
        assert_eq!(from_bytes.len(), idx.len());
        assert_eq!(from_reader.dim(), idx.dim());
        assert_eq!(from_bytes.score_all(query), idx.score_all(query));
        assert_eq!(
            from_reader.top_m_candidates(query, 32),
            idx.top_m_candidates(query, 32)
        );
    }
}

#[test]
fn public_readers_do_not_buffer_past_reported_trailing_bytes() {
    let corpus = make_corpus(90_101);

    {
        let mut idx = Rank::new(D);
        idx.add(&corpus);
        let mut bytes = Vec::new();
        idx.write_to(&mut bytes).unwrap();
        let mut cursor = append_trailer(bytes);
        assert_load_err_contains(
            Rank::read_from(&mut cursor),
            "OVR1 payload has trailing bytes",
        );
        assert_eq!(
            cursor.position(),
            13,
            "Rank reader should stop after header"
        );
    }

    {
        let mut idx = RankQuant::new(D, 2);
        idx.add(&corpus);
        let mut bytes = Vec::new();
        idx.write_to(&mut bytes).unwrap();
        let mut cursor = append_trailer(bytes);
        assert_load_err_contains(
            RankQuant::read_from(&mut cursor),
            "OVRQ payload has trailing bytes",
        );
        assert_eq!(
            cursor.position(),
            14,
            "RankQuant reader should stop after header"
        );
    }

    {
        let mut idx = Bitmap::new(D, D / 4);
        idx.add(&corpus);
        let mut bytes = Vec::new();
        idx.write_to(&mut bytes).unwrap();
        let mut cursor = append_trailer(bytes);
        assert_load_err_contains(
            Bitmap::read_from(&mut cursor),
            "OVBM payload has trailing bytes",
        );
        assert_eq!(
            cursor.position(),
            17,
            "Bitmap reader should stop after header"
        );
    }

    {
        let mut idx = SignBitmap::new(D);
        idx.add(&corpus);
        let mut bytes = Vec::new();
        idx.write_to(&mut bytes).unwrap();
        let mut cursor = append_trailer(bytes);
        assert_load_err_contains(
            SignBitmap::read_from(&mut cursor),
            "OVSB payload has trailing bytes",
        );
        assert_eq!(
            cursor.position(),
            13,
            "SignBitmap reader should stop after header"
        );
    }
}

fn rank_payload_cases(dim: usize) -> (Vec<u8>, Vec<u8>) {
    let p = tmp("rank_empty_payload_case");
    Rank::new(dim).write(&p).unwrap();
    let trailing = read_bytes(&p);
    std::fs::remove_file(&p).ok();
    assert_eq!(trailing.len(), 13, "empty Rank file is header-only");
    let mut truncated = trailing.clone();
    set_u32_field(&mut truncated, 9, 1);
    (truncated, trailing)
}

fn rankquant_payload_cases(bits: u8, dim: usize) -> (Vec<u8>, Vec<u8>) {
    let p = tmp("rankquant_empty_payload_case");
    RankQuant::new(dim, bits).write(&p).unwrap();
    let trailing = read_bytes(&p);
    std::fs::remove_file(&p).ok();
    assert_eq!(trailing.len(), 14, "empty RankQuant file is header-only");
    let mut truncated = trailing.clone();
    set_u32_field(&mut truncated, 10, 1);
    (truncated, trailing)
}

fn bitmap_payload_cases(dim: usize, n_top: usize) -> (Vec<u8>, Vec<u8>) {
    let p = tmp("bitmap_empty_payload_case");
    Bitmap::new(dim, n_top).write(&p).unwrap();
    let trailing = read_bytes(&p);
    std::fs::remove_file(&p).ok();
    assert_eq!(trailing.len(), 17, "empty Bitmap file is header-only");
    let mut truncated = trailing.clone();
    set_u32_field(&mut truncated, 13, 1);
    (truncated, trailing)
}

fn sign_bitmap_payload_cases(dim: usize) -> (Vec<u8>, Vec<u8>) {
    let p = tmp("sign_bitmap_empty_payload_case");
    SignBitmap::new(dim).write(&p).unwrap();
    let trailing = read_bytes(&p);
    std::fs::remove_file(&p).ok();
    assert_eq!(trailing.len(), 13, "empty SignBitmap file is header-only");
    let mut truncated = trailing.clone();
    set_u32_field(&mut truncated, 9, 1);
    (truncated, trailing)
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

    // OVR1 header is 13 bytes; payload is u16 LE ranks. Force ranks[1] ==
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

    // OVRQ header is 14 bytes. Zero the entire packed payload so every
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

    // OVBM header is 17 bytes; payload is u64 LE words, qpv = dim/64 per doc.
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

    // OVSB header is 13 bytes. Flip bits across the payload; the result is
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
fn public_loaders_report_stable_malformed_payload_context() {
    let rank = rank_payload_cases(4);
    let rankquant = rankquant_payload_cases(2, 8);
    let bitmap = bitmap_payload_cases(64, 16);
    let sign_bitmap = sign_bitmap_payload_cases(64);
    let cases: [(&str, Vec<u8>, Vec<u8>, &str); 4] = [
        ("rank", rank.0, rank.1, "OVR1"),
        ("rankquant", rankquant.0, rankquant.1, "OVRQ"),
        ("bitmap", bitmap.0, bitmap.1, "OVBM"),
        ("sign_bitmap", sign_bitmap.0, sign_bitmap.1, "OVSB"),
    ];

    for (suffix, truncated_header, mut trailing_bytes, label) in cases {
        let truncated = tmp(&format!("{suffix}_truncated_context"));
        write_bytes(&truncated, &truncated_header);
        match label {
            "OVR1" => assert_load_err_contains(
                Rank::load(&truncated),
                &format!("{label} payload truncated"),
            ),
            "OVRQ" => assert_load_err_contains(
                RankQuant::load(&truncated),
                &format!("{label} payload truncated"),
            ),
            "OVBM" => assert_load_err_contains(
                Bitmap::load(&truncated),
                &format!("{label} payload truncated"),
            ),
            "OVSB" => assert_load_err_contains(
                SignBitmap::load(&truncated),
                &format!("{label} payload truncated"),
            ),
            _ => unreachable!(),
        }
        match label {
            "OVR1" => assert_load_err_contains(
                Rank::load_from_bytes(&truncated_header),
                &format!("{label} payload truncated"),
            ),
            "OVRQ" => assert_load_err_contains(
                RankQuant::load_from_bytes(&truncated_header),
                &format!("{label} payload truncated"),
            ),
            "OVBM" => assert_load_err_contains(
                Bitmap::load_from_bytes(&truncated_header),
                &format!("{label} payload truncated"),
            ),
            "OVSB" => assert_load_err_contains(
                SignBitmap::load_from_bytes(&truncated_header),
                &format!("{label} payload truncated"),
            ),
            _ => unreachable!(),
        }
        std::fs::remove_file(&truncated).ok();

        trailing_bytes.push(0);
        let trailing = tmp(&format!("{suffix}_trailing_context"));
        write_bytes(&trailing, &trailing_bytes);
        match label {
            "OVR1" => assert_load_err_contains(
                Rank::load(&trailing),
                &format!("{label} payload has trailing bytes"),
            ),
            "OVRQ" => assert_load_err_contains(
                RankQuant::load(&trailing),
                &format!("{label} payload has trailing bytes"),
            ),
            "OVBM" => assert_load_err_contains(
                Bitmap::load(&trailing),
                &format!("{label} payload has trailing bytes"),
            ),
            "OVSB" => assert_load_err_contains(
                SignBitmap::load(&trailing),
                &format!("{label} payload has trailing bytes"),
            ),
            _ => unreachable!(),
        }
        match label {
            "OVR1" => assert_load_err_contains(
                Rank::load_from_bytes(&trailing_bytes),
                &format!("{label} payload has trailing bytes"),
            ),
            "OVRQ" => assert_load_err_contains(
                RankQuant::load_from_bytes(&trailing_bytes),
                &format!("{label} payload has trailing bytes"),
            ),
            "OVBM" => assert_load_err_contains(
                Bitmap::load_from_bytes(&trailing_bytes),
                &format!("{label} payload has trailing bytes"),
            ),
            "OVSB" => assert_load_err_contains(
                SignBitmap::load_from_bytes(&trailing_bytes),
                &format!("{label} payload has trailing bytes"),
            ),
            _ => unreachable!(),
        }
        std::fs::remove_file(&trailing).ok();
    }
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
