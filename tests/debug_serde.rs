use ordvec::{Bitmap, Rank, RankQuant, RankQuantFastscan, SearchResults, SignBitmap};

fn assert_param_debug(debug: &str, type_name: &str) {
    assert!(debug.contains(type_name), "{type_name} missing: {debug}");
    assert!(debug.contains("dim"), "{type_name} missing dim: {debug}");
    assert!(
        !debug.contains("packed")
            && !debug.contains("ranks")
            && !debug.contains("bitmaps")
            && !debug.contains("scores:")
            && !debug.contains("indices:"),
        "{type_name} debug output leaks storage fields: {debug}"
    );
}

#[test]
fn public_types_debug_prints_shape_not_storage() {
    let rank = Rank::new(8);
    assert_param_debug(&format!("{rank:?}"), "Rank");

    let rq = RankQuant::new(64, 2);
    let rq_dbg = format!("{rq:?}");
    assert_param_debug(&rq_dbg, "RankQuant");
    assert!(rq_dbg.contains("bits"));

    let bitmap = Bitmap::new(64, 16);
    let bitmap_dbg = format!("{bitmap:?}");
    assert_param_debug(&bitmap_dbg, "Bitmap");
    assert!(bitmap_dbg.contains("n_top"));

    let sign = SignBitmap::new(64);
    assert_param_debug(&format!("{sign:?}"), "SignBitmap");

    let fastscan = RankQuantFastscan::new(64);
    assert_param_debug(&format!("{fastscan:?}"), "RankQuantFastscan");

    let results = SearchResults {
        scores: vec![0.5, 0.25],
        indices: vec![7, 3],
        nq: 1,
        k: 2,
    };
    let results_dbg = format!("{results:?}");
    assert!(results_dbg.contains("SearchResults"));
    assert!(results_dbg.contains("scores_len"));
    assert!(results_dbg.contains("indices_len"));
    assert!(!results_dbg.contains("scores:") && !results_dbg.contains("indices:"));
}

#[cfg(feature = "serde")]
#[test]
fn search_results_implements_serde_traits_with_serde_feature() {
    fn assert_serde<T: serde::Serialize + for<'de> serde::Deserialize<'de>>() {}

    assert_serde::<SearchResults>();
}
