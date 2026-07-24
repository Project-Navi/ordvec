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

use std::cell::Cell;
use std::io::{self, Cursor, Read, Write};
use std::rc::Rc;

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

struct FragmentedInterruptingSpy {
    bytes: Vec<u8>,
    record_len: usize,
    position: usize,
    interrupt_next: bool,
    overread_attempts: Rc<Cell<usize>>,
}

impl FragmentedInterruptingSpy {
    fn new(bytes: Vec<u8>, record_len: usize, overread_attempts: Rc<Cell<usize>>) -> Self {
        Self {
            bytes,
            record_len,
            position: 0,
            interrupt_next: true,
            overread_attempts,
        }
    }
}

impl Read for FragmentedInterruptingSpy {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.interrupt_next {
            self.interrupt_next = false;
            return Err(io::Error::new(io::ErrorKind::Interrupted, "spy interrupt"));
        }
        self.interrupt_next = true;
        if self.position == self.bytes.len() || buf.is_empty() {
            return Ok(0);
        }
        let read = buf.len().min(3).min(self.bytes.len() - self.position);
        if self.position.saturating_add(read) > self.record_len {
            self.overread_attempts
                .set(self.overread_attempts.get().saturating_add(1));
        }
        buf[..read].copy_from_slice(&self.bytes[self.position..self.position + read]);
        self.position += read;
        Ok(read)
    }
}

struct CountingReader {
    inner: Cursor<Vec<u8>>,
    calls: Rc<Cell<usize>>,
}

impl Read for CountingReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if !buf.is_empty() {
            self.calls.set(self.calls.get().saturating_add(1));
        }
        self.inner.read(buf)
    }
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
fn sized_forward_only_readers_handle_fragmentation_interrupts_and_never_overread() {
    let corpus = make_corpus(90_051);

    let mut rank_quant = RankQuant::new(D, 2);
    rank_quant.add(&corpus);
    let mut rank_quant_bytes = Vec::new();
    rank_quant.write_to(&mut rank_quant_bytes).unwrap();
    let rank_quant_len = rank_quant_bytes.len();
    rank_quant_bytes.extend_from_slice(b"unrelated-next-record");
    let rank_quant_overreads = Rc::new(Cell::new(0));
    let decoded = RankQuant::read_from_sized(
        FragmentedInterruptingSpy::new(
            rank_quant_bytes,
            rank_quant_len,
            Rc::clone(&rank_quant_overreads),
        ),
        rank_quant_len as u64,
    )
    .unwrap();
    assert_eq!(decoded.len(), rank_quant.len());
    assert_eq!(rank_quant_overreads.get(), 0);

    let mut sign = SignBitmap::new(D);
    sign.add(&corpus);
    let mut sign_bytes = Vec::new();
    sign.write_to(&mut sign_bytes).unwrap();
    let sign_len = sign_bytes.len();
    sign_bytes.extend_from_slice(b"unrelated-next-record");
    let sign_overreads = Rc::new(Cell::new(0));
    let decoded = SignBitmap::read_from_sized(
        FragmentedInterruptingSpy::new(sign_bytes, sign_len, Rc::clone(&sign_overreads)),
        sign_len as u64,
    )
    .unwrap();
    assert_eq!(decoded.len(), sign.len());
    assert_eq!(sign_overreads.get(), 0);
}

#[test]
fn sized_forward_only_readers_reject_inexact_lengths_before_payload_allocation() {
    let corpus = make_corpus(90_052);

    let mut rank_quant = RankQuant::new(D, 2);
    rank_quant.add(&corpus);
    let mut bytes = Vec::new();
    rank_quant.write_to(&mut bytes).unwrap();
    assert_load_err_contains(
        RankQuant::read_from_sized(Cursor::new(&bytes), bytes.len() as u64 - 1),
        "payload truncated",
    );
    assert_load_err_contains(
        RankQuant::read_from_sized(Cursor::new(&bytes), u64::MAX),
        "payload has trailing bytes",
    );

    let mut sign = SignBitmap::new(D);
    sign.add(&corpus);
    let mut bytes = Vec::new();
    sign.write_to(&mut bytes).unwrap();
    assert_load_err_contains(
        SignBitmap::read_from_sized(Cursor::new(&bytes), bytes.len() as u64 - 1),
        "payload truncated",
    );
    assert_load_err_contains(
        SignBitmap::read_from_sized(Cursor::new(&bytes), u64::MAX),
        "payload has trailing bytes",
    );
}

#[test]
fn sized_forward_only_readers_never_cross_an_undersized_header_boundary() {
    let corpus = make_corpus(90_053);

    let mut rank_quant = RankQuant::new(D, 2);
    rank_quant.add(&corpus);
    let mut rank_quant_bytes = Vec::new();
    rank_quant.write_to(&mut rank_quant_bytes).unwrap();
    let rank_quant_declared = 13;
    let rank_quant_overreads = Rc::new(Cell::new(0));
    assert_load_err_contains(
        RankQuant::read_from_sized(
            FragmentedInterruptingSpy::new(
                rank_quant_bytes,
                rank_quant_declared,
                Rc::clone(&rank_quant_overreads),
            ),
            rank_quant_declared as u64,
        ),
        "header truncated",
    );
    assert_eq!(rank_quant_overreads.get(), 0);

    let mut sign = SignBitmap::new(D);
    sign.add(&corpus);
    let mut sign_bytes = Vec::new();
    sign.write_to(&mut sign_bytes).unwrap();
    let sign_declared = 12;
    let sign_overreads = Rc::new(Cell::new(0));
    assert_load_err_contains(
        SignBitmap::read_from_sized(
            FragmentedInterruptingSpy::new(sign_bytes, sign_declared, Rc::clone(&sign_overreads)),
            sign_declared as u64,
        ),
        "header truncated",
    );
    assert_eq!(sign_overreads.get(), 0);
}

#[test]
fn sized_sign_reader_batches_typed_payload_reads() {
    let corpus = make_corpus(90_054);
    let mut sign = SignBitmap::new(D);
    for _ in 0..32 {
        sign.add(&corpus);
    }
    let mut bytes = Vec::new();
    sign.write_to(&mut bytes).unwrap();
    assert!(bytes.len() > 64 * 1024);

    let calls = Rc::new(Cell::new(0));
    let reader = CountingReader {
        inner: Cursor::new(bytes.clone()),
        calls: Rc::clone(&calls),
    };
    let decoded = SignBitmap::read_from_sized(reader, bytes.len() as u64).unwrap();
    assert_eq!(decoded.len(), sign.len());
    assert!(
        calls.get() < 32,
        "fixed-chunk decoding regressed to {} underlying reads",
        calls.get()
    );
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
