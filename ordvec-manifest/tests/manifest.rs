use ordvec::{Bitmap, Rank, RankQuant, SignBitmap};
use ordvec_manifest::{
    create_manifest_for_index, create_manifest_for_index_with_options, load_manifest_file,
    load_manifest_file_with_options, sha256_file, verify_index_manifest, verify_manifest_with_base,
    CalibrationOrdinalization, CalibrationProfileRef, CreateManifestOptions, CreateRowIdentity,
    EncoderSpec, ManifestIndexParams, NullModelSpec, ProfileArtifactRef, ProfileParameterization,
    ResourceLimits, RowIdentity, VerifyOptions, CALIBRATION_SCHEMA_VERSION,
};
use serde_json::json;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn write_index(dir: &Path) -> PathBuf {
    let path = dir.join("index.tvrq");
    let mut index = RankQuant::new(16, 2);
    let docs: Vec<f32> = (0..32).map(|i| i as f32 - 12.0).collect();
    index.add(&docs);
    index.write(&path).unwrap();
    path
}

fn write_rankquant_index(dir: &Path, rows: usize) -> PathBuf {
    let path = dir.join("index.tvrq");
    let mut index = RankQuant::new(16, 2);
    let docs: Vec<f32> = (0..16 * rows).map(|i| i as f32 - 12.0).collect();
    index.add(&docs);
    index.write(&path).unwrap();
    path
}

#[derive(Clone, Copy)]
enum FixtureKind {
    Rank,
    RankQuant,
    Bitmap,
    SignBitmap,
}

fn write_index_kind(dir: &Path, kind: FixtureKind) -> PathBuf {
    match kind {
        FixtureKind::Rank => {
            let path = dir.join("index.tvr");
            let mut index = Rank::new(8);
            index.add(&[
                1.0, 3.0, 2.0, 4.0, 8.0, 7.0, 6.0, 5.0, 8.0, 6.0, 7.0, 5.0, 1.0, 2.0, 3.0, 4.0,
            ]);
            index.write(&path).unwrap();
            path
        }
        FixtureKind::RankQuant => write_index(dir),
        FixtureKind::Bitmap => {
            let path = dir.join("index.tvbm");
            let mut index = Bitmap::new(64, 16);
            let docs: Vec<f32> = (0..128).map(|i| ((i * 17) % 31) as f32).collect();
            index.add(&docs);
            index.write(&path).unwrap();
            path
        }
        FixtureKind::SignBitmap => {
            let path = dir.join("index.tvsb");
            let mut index = SignBitmap::new(64);
            let docs: Vec<f32> = (0usize..128)
                .map(|i| if i.is_multiple_of(3) { 1.0 } else { -1.0 })
                .collect();
            index.add(&docs);
            index.write(&path).unwrap();
            path
        }
    }
}

fn write_row_map(path: &Path, rows: &[(&str, Option<&str>)]) {
    let mut file = fs::File::create(path).unwrap();
    for (row_id, (db_id, parent_id)) in rows.iter().enumerate() {
        let value = if let Some(parent_id) = parent_id {
            json!({"row_id": row_id, "db_id": db_id, "parent_id": parent_id})
        } else {
            json!({"row_id": row_id, "db_id": db_id})
        };
        writeln!(file, "{value}").unwrap();
    }
}

fn identity_manifest(dir: &Path) -> (tempfile::TempDir, ordvec_manifest::IndexManifest, PathBuf) {
    let temp = tempfile::tempdir_in(dir).unwrap();
    let index = write_index(temp.path());
    let manifest_path = temp.path().join("manifest.json");
    let manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    (temp, manifest, manifest_path)
}

fn write_profile(path: &Path, size_bytes: usize) -> ordvec_manifest::FileHash {
    fs::write(path, vec![0u8; size_bytes]).unwrap();
    sha256_file(path).unwrap()
}

fn uniform_calibration(
    manifest: &ordvec_manifest::IndexManifest,
    ordinalization: CalibrationOrdinalization,
) -> CalibrationProfileRef {
    CalibrationProfileRef {
        schema_version: CALIBRATION_SCHEMA_VERSION.to_string(),
        profile_id: "urn:uuid:7c66ad6e-bdde-49a8-b420-f1136d04f5bd".to_string(),
        created_at: Some("2026-05-29T06:00:00Z".to_string()),
        calibrated_for: EncoderSpec {
            model: manifest.embedding.model.clone(),
            dim: manifest.embedding.dim,
            model_revision: manifest.embedding.model_revision.clone(),
            normalization: manifest.embedding.normalization.clone(),
        },
        ordinalization,
        profile: None,
        null_model: NullModelSpec::UniformHypergeometric,
    }
}

fn weighted_calibration(
    manifest: &ordvec_manifest::IndexManifest,
    path: impl Into<String>,
    hash: ordvec_manifest::FileHash,
    ordinalization: CalibrationOrdinalization,
    parameterization: ProfileParameterization,
    shape: Vec<usize>,
) -> CalibrationProfileRef {
    CalibrationProfileRef {
        schema_version: CALIBRATION_SCHEMA_VERSION.to_string(),
        profile_id: "urn:uuid:7c66ad6e-bdde-49a8-b420-f1136d04f5bd".to_string(),
        created_at: Some("2026-05-29T06:00:00Z".to_string()),
        calibrated_for: EncoderSpec {
            model: manifest.embedding.model.clone(),
            dim: manifest.embedding.dim,
            model_revision: manifest.embedding.model_revision.clone(),
            normalization: manifest.embedding.normalization.clone(),
        },
        ordinalization,
        profile: Some(ProfileArtifactRef {
            path: path.into(),
            sha256: hash.sha256,
            file_size_bytes: hash.size_bytes,
            dim: manifest.artifact.dim,
            sample_count: 100,
            parameterization,
            format: "raw_f64_le".to_string(),
            shape,
            source_digest: None,
        }),
        null_model: NullModelSpec::WeightedMarginalProfile { parameterization },
    }
}

fn error_codes(report: &ordvec_manifest::VerificationReport) -> Vec<&str> {
    report
        .errors
        .iter()
        .map(|issue| issue.code.as_str())
        .collect()
}

#[test]
fn create_then_verify_identity_manifest_for_all_persisted_formats() {
    let temp = tempfile::tempdir().unwrap();
    for (kind, expected) in [
        (FixtureKind::Rank, ordvec_manifest::ManifestIndexKind::Rank),
        (
            FixtureKind::RankQuant,
            ordvec_manifest::ManifestIndexKind::RankQuant,
        ),
        (
            FixtureKind::Bitmap,
            ordvec_manifest::ManifestIndexKind::Bitmap,
        ),
        (
            FixtureKind::SignBitmap,
            ordvec_manifest::ManifestIndexKind::SignBitmap,
        ),
    ] {
        let case = tempfile::tempdir_in(temp.path()).unwrap();
        let index = write_index_kind(case.path(), kind);
        let manifest_path = case.path().join("manifest.json");
        let manifest = create_manifest_for_index(
            &index,
            CreateRowIdentity::RowIdIdentity,
            "test-embedding",
            &manifest_path,
        )
        .unwrap();

        let report = verify_manifest_with_base(manifest, case.path(), VerifyOptions::default());
        assert!(report.ok, "{:?}", report.errors);
        assert_eq!(report.skipped_checks, ["attestations_absent"]);
        assert_eq!(report.artifact.metadata.unwrap().kind, expected);
    }
}

#[test]
fn create_manifest_creates_output_parent_for_programmatic_callers() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let manifest_path = temp.path().join("nested").join("manifest.json");

    let manifest = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            allow_path_escape: true,
            ..CreateManifestOptions::default()
        },
    )
    .unwrap();

    assert!(manifest_path.parent().unwrap().is_dir());
    assert_eq!(manifest.row_identity.row_count(), 2);
}

#[test]
fn schema_rejects_unknown_fields_and_bad_extension_keys() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());

    let mut value = serde_json::to_value(&manifest).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("unknown".to_string(), json!(true));
    let parsed = serde_json::from_value::<ordvec_manifest::IndexManifest>(value);
    assert!(
        parsed.is_err(),
        "schema-owned structs must reject unknown fields"
    );

    manifest
        .extensions
        .insert("policy".to_string(), json!({"decision": "deny"}));
    let report = verify_manifest_with_base(manifest.clone(), temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "extension_key_not_namespaced"));

    manifest.extensions.clear();
    manifest.extensions.insert(
        "com.example.policy".to_string(),
        json!({"decision": "allow"}),
    );
    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(report.ok, "{:?}", report.errors);
}

#[test]
fn manifest_loader_enforces_size_limit_with_exact_boundary_success() {
    let root = tempfile::tempdir().unwrap();
    let (_temp, manifest, manifest_path) = identity_manifest(root.path());
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let manifest_bytes = fs::metadata(&manifest_path).unwrap().len();

    let err = load_manifest_file_with_options(
        &manifest_path,
        &VerifyOptions {
            limits: ResourceLimits {
                max_manifest_bytes: manifest_bytes - 1,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
    )
    .unwrap_err();
    assert_eq!(err.code(), Some("manifest_file_too_large"));

    let loaded = load_manifest_file_with_options(
        &manifest_path,
        &VerifyOptions {
            limits: ResourceLimits {
                max_manifest_bytes: manifest_bytes,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
    )
    .unwrap();
    assert_eq!(loaded.manifest.manifest_id, manifest.manifest_id);
}

#[test]
fn row_identity_jsonl_line_limit_is_overridable() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_rankquant_index(temp.path(), 1);
    let rows = temp.path().join("rows.jsonl");
    write_row_map(&rows, &[("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", None)]);
    let line_len = fs::read(&rows).unwrap().len();
    let manifest_path = temp.path().join("manifest.json");
    let manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::Jsonl(rows.clone()),
        "test-embedding",
        &manifest_path,
    )
    .unwrap();

    let report = verify_manifest_with_base(
        manifest.clone(),
        temp.path(),
        VerifyOptions {
            limits: ResourceLimits {
                max_row_identity_jsonl_line_bytes: line_len - 1,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
    );
    assert!(error_codes(&report).contains(&"row_identity_line_too_large"));

    let report = verify_manifest_with_base(
        manifest,
        temp.path(),
        VerifyOptions {
            limits: ResourceLimits {
                max_row_identity_jsonl_line_bytes: line_len,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
    );
    assert!(report.ok, "{:?}", report.errors);
}

#[test]
fn row_identity_row_limit_rejects_declared_overrun() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_rankquant_index(temp.path(), 2);
    let rows = temp.path().join("rows.jsonl");
    write_row_map(
        &rows,
        &[
            ("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", None),
            ("ffffffff-1111-4222-8333-444444444444", None),
        ],
    );
    let manifest_path = temp.path().join("manifest.json");
    let manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::Jsonl(rows),
        "test-embedding",
        &manifest_path,
    )
    .unwrap();

    let report = verify_manifest_with_base(
        manifest.clone(),
        temp.path(),
        VerifyOptions {
            limits: ResourceLimits {
                max_row_identity_rows: 1,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
    );
    assert!(error_codes(&report).contains(&"row_identity_row_count_limit_exceeded"));

    let report = verify_manifest_with_base(
        manifest,
        temp.path(),
        VerifyOptions {
            limits: ResourceLimits {
                max_row_identity_rows: 2,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
    );
    assert!(report.ok, "{:?}", report.errors);
}

#[test]
fn row_identity_extra_rows_reports_mismatch_not_limit() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_rankquant_index(temp.path(), 2);
    let rows = temp.path().join("rows.jsonl");
    write_row_map(
        &rows,
        &[
            ("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", None),
            ("ffffffff-1111-4222-8333-444444444444", None),
            ("55555555-6666-4777-8888-999999999999", None),
        ],
    );
    let row_hash = sha256_file(&rows).unwrap();
    let manifest_path = temp.path().join("manifest.json");
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    manifest.row_identity = RowIdentity::Jsonl {
        path: "rows.jsonl".to_string(),
        sha256: row_hash.sha256,
        row_count: 2,
        id_kind: "uuid".to_string(),
        db: None,
    };

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    let codes = error_codes(&report);
    assert!(codes.contains(&"row_identity_row_count_mismatch"));
    assert!(!codes.contains(&"row_identity_row_count_limit_exceeded"));
}

#[test]
fn row_identity_duplicate_tracking_limit_is_bounded() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_rankquant_index(temp.path(), 2);
    let rows = temp.path().join("rows.jsonl");
    let first_id = "a".repeat(24);
    let second_id = "b".repeat(24);
    let mut file = fs::File::create(&rows).unwrap();
    writeln!(file, "{}", json!({"row_id": 0, "db_id": first_id})).unwrap();
    writeln!(file, "{}", json!({"row_id": 1, "db_id": second_id})).unwrap();
    let row_hash = sha256_file(&rows).unwrap();
    let manifest_path = temp.path().join("manifest.json");
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    manifest.row_identity = RowIdentity::Jsonl {
        path: "rows.jsonl".to_string(),
        sha256: row_hash.sha256,
        row_count: 2,
        id_kind: "uuid".to_string(),
        db: None,
    };

    let report = verify_manifest_with_base(
        manifest,
        temp.path(),
        VerifyOptions {
            limits: ResourceLimits {
                max_row_identity_tracked_db_id_bytes: 32,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
    );
    assert!(error_codes(&report).contains(&"row_identity_duplicate_tracking_limit_exceeded"));
}

#[test]
fn create_row_identity_limit_is_overridable() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_rankquant_index(temp.path(), 2);
    let rows = temp.path().join("rows.jsonl");
    write_row_map(
        &rows,
        &[
            ("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", None),
            ("ffffffff-1111-4222-8333-444444444444", None),
        ],
    );
    let manifest_path = temp.path().join("manifest.json");

    let err = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::Jsonl(rows.clone()),
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            limits: ResourceLimits {
                max_row_identity_rows: 1,
                ..ResourceLimits::default()
            },
            ..CreateManifestOptions::default()
        },
    )
    .unwrap_err();
    assert_eq!(err.code(), Some("row_identity_row_count_limit_exceeded"));

    let manifest = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::Jsonl(rows),
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            limits: ResourceLimits {
                max_row_identity_rows: 2,
                ..ResourceLimits::default()
            },
            ..CreateManifestOptions::default()
        },
    )
    .unwrap();
    assert_eq!(manifest.row_identity.row_count(), 2);
}

#[test]
fn create_row_identity_line_limit_preserves_code() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_rankquant_index(temp.path(), 1);
    let rows = temp.path().join("rows.jsonl");
    write_row_map(&rows, &[("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", None)]);
    let line_len = fs::read(&rows).unwrap().len();
    let manifest_path = temp.path().join("manifest.json");

    let err = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::Jsonl(rows),
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            limits: ResourceLimits {
                max_row_identity_jsonl_line_bytes: line_len - 1,
                ..ResourceLimits::default()
            },
            ..CreateManifestOptions::default()
        },
    )
    .unwrap_err();
    assert_eq!(err.code(), Some("row_identity_line_too_large"));
}

#[test]
fn create_row_identity_duplicate_tracking_limit_preserves_code() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_rankquant_index(temp.path(), 2);
    let rows = temp.path().join("rows.jsonl");
    let first_id = "a".repeat(24);
    let second_id = "b".repeat(24);
    let mut file = fs::File::create(&rows).unwrap();
    writeln!(file, "{}", json!({"row_id": 0, "db_id": first_id})).unwrap();
    writeln!(file, "{}", json!({"row_id": 1, "db_id": second_id})).unwrap();
    let manifest_path = temp.path().join("manifest.json");

    let err = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::Jsonl(rows),
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            limits: ResourceLimits {
                max_row_identity_tracked_db_id_bytes: 32,
                ..ResourceLimits::default()
            },
            ..CreateManifestOptions::default()
        },
    )
    .unwrap_err();
    assert_eq!(
        err.code(),
        Some("row_identity_duplicate_tracking_limit_exceeded")
    );
}

#[test]
fn row_identity_report_issue_limit_truncates_per_row_errors() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    let rows = temp.path().join("rows.jsonl");
    fs::write(&rows, b"{}\n{}\n{}\n{}\n").unwrap();
    let hash = sha256_file(&rows).unwrap();
    manifest.row_identity = RowIdentity::Jsonl {
        path: "rows.jsonl".to_string(),
        sha256: hash.sha256,
        row_count: 4,
        id_kind: "uuid".to_string(),
        db: None,
    };

    let report = verify_manifest_with_base(
        manifest,
        temp.path(),
        VerifyOptions {
            limits: ResourceLimits {
                max_report_issues: 2,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
    );
    assert_eq!(report.errors.len(), 2);
    assert!(error_codes(&report).contains(&"verification_report_issue_limit_exceeded"));
}

#[test]
fn schema_enforces_lowercase_sha256_and_optional_field_shapes() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    manifest.artifact.sha256 = manifest.artifact.sha256.to_ascii_uppercase();
    manifest.row_identity = RowIdentity::Jsonl {
        path: "rows.jsonl".to_string(),
        sha256: "A".repeat(64),
        row_count: 2,
        id_kind: "uuid".to_string(),
        db: None,
    };
    manifest.embedding.model_revision = Some("".to_string());
    manifest.embedding.corpus_digest = Some("A".repeat(64));
    manifest.embedding.embedding_matrix_digest = Some("not-a-digest".to_string());
    manifest.embedding.normalization = Some("".to_string());
    manifest.build.as_mut().unwrap().source_repo = Some("".to_string());

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    for code in [
        "artifact_sha256_invalid",
        "row_identity_sha256_invalid",
        "embedding_model_revision_empty",
        "embedding_corpus_digest_invalid",
        "embedding_matrix_digest_invalid",
        "embedding_normalization_empty",
        "build_source_repo_empty",
    ] {
        assert!(
            report.errors.iter().any(|issue| issue.code == code),
            "missing {code}: {:?}",
            report.errors
        );
    }
}

#[test]
fn calibration_schema_shape_is_strict_and_optional() {
    let root = tempfile::tempdir().unwrap();
    let (temp, manifest, _manifest_path) = identity_manifest(root.path());
    let report = verify_manifest_with_base(manifest.clone(), temp.path(), VerifyOptions::default());
    assert!(report.ok, "{:?}", report.errors);
    assert!(!report.calibration.present);

    let mut with_unknown = manifest.clone();
    with_unknown.calibration = Some(uniform_calibration(
        &with_unknown,
        CalibrationOrdinalization::Bucket {
            dim: with_unknown.artifact.dim,
            bits: 2,
        },
    ));
    let mut value = serde_json::to_value(&with_unknown).unwrap();
    value["calibration"]
        .as_object_mut()
        .unwrap()
        .insert("unknown".to_string(), json!(true));
    let parsed = serde_json::from_value::<ordvec_manifest::IndexManifest>(value);
    assert!(parsed.is_err(), "calibration must reject unknown fields");

    let mut bad = manifest;
    let mut calibration = uniform_calibration(
        &bad,
        CalibrationOrdinalization::Bucket {
            dim: bad.artifact.dim,
            bits: 2,
        },
    );
    calibration.schema_version = "ordvec.calibration.v2".to_string();
    calibration.created_at = Some("not-rfc3339".to_string());
    bad.calibration = Some(calibration);
    let report = verify_manifest_with_base(bad, temp.path(), VerifyOptions::default());
    for code in [
        "calibration_schema_version_unsupported",
        "calibration_created_at_invalid",
    ] {
        assert!(
            error_codes(&report).contains(&code),
            "missing {code}: {:?}",
            report.errors
        );
    }
}

#[test]
fn calibration_encoder_identity_must_match_embedding() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    manifest.embedding.model_revision = Some("rev-a".to_string());
    manifest.embedding.normalization = Some("l2".to_string());
    let mut calibration = uniform_calibration(
        &manifest,
        CalibrationOrdinalization::Bucket {
            dim: manifest.artifact.dim,
            bits: 2,
        },
    );
    calibration.calibrated_for.model = "other-model".to_string();
    calibration.calibrated_for.dim += 1;
    calibration.calibrated_for.model_revision = Some("rev-b".to_string());
    calibration.calibrated_for.normalization = Some("as_provided".to_string());
    manifest.calibration = Some(calibration);

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    for code in [
        "calibration_encoder_model_mismatch",
        "calibration_encoder_dim_mismatch",
        "calibration_encoder_model_revision_mismatch",
        "calibration_encoder_normalization_mismatch",
    ] {
        assert!(
            error_codes(&report).contains(&code),
            "missing {code}: {:?}",
            report.errors
        );
    }
}

#[test]
fn calibration_ordinalization_matches_artifact_formats() {
    let temp = tempfile::tempdir().unwrap();

    let bitmap_case = tempfile::tempdir_in(temp.path()).unwrap();
    let bitmap = write_index_kind(bitmap_case.path(), FixtureKind::Bitmap);
    let bitmap_manifest_path = bitmap_case.path().join("manifest.json");
    let mut bitmap_manifest = create_manifest_for_index(
        &bitmap,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &bitmap_manifest_path,
    )
    .unwrap();
    bitmap_manifest.calibration = Some(uniform_calibration(
        &bitmap_manifest,
        CalibrationOrdinalization::TopK {
            dim: bitmap_manifest.artifact.dim,
            k: 16,
        },
    ));
    let report = verify_manifest_with_base(
        bitmap_manifest.clone(),
        bitmap_case.path(),
        VerifyOptions::default(),
    );
    assert!(report.ok, "{:?}", report.errors);
    bitmap_manifest.calibration = Some(uniform_calibration(
        &bitmap_manifest,
        CalibrationOrdinalization::TopK {
            dim: bitmap_manifest.artifact.dim,
            k: 8,
        },
    ));
    let report = verify_manifest_with_base(
        bitmap_manifest,
        bitmap_case.path(),
        VerifyOptions::default(),
    );
    assert!(error_codes(&report).contains(&"calibration_ordinalization_artifact_mismatch"));

    let rq_case = tempfile::tempdir_in(temp.path()).unwrap();
    let rank_quant = write_index_kind(rq_case.path(), FixtureKind::RankQuant);
    let rq_manifest_path = rq_case.path().join("manifest.json");
    let rq_profile_hash = write_profile(&rq_case.path().join("bucket.f64"), 16 * 4 * 8);
    let mut rq_manifest = create_manifest_for_index(
        &rank_quant,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &rq_manifest_path,
    )
    .unwrap();
    rq_manifest.calibration = Some(weighted_calibration(
        &rq_manifest,
        "bucket.f64",
        rq_profile_hash.clone(),
        CalibrationOrdinalization::Bucket {
            dim: rq_manifest.artifact.dim,
            bits: 2,
        },
        ProfileParameterization::BucketFrequency,
        vec![rq_manifest.artifact.dim, 4],
    ));
    let report = verify_manifest_with_base(
        rq_manifest.clone(),
        rq_case.path(),
        VerifyOptions::default(),
    );
    assert!(report.ok, "{:?}", report.errors);
    rq_manifest.calibration = Some(weighted_calibration(
        &rq_manifest,
        "bucket.f64",
        rq_profile_hash,
        CalibrationOrdinalization::Bucket {
            dim: rq_manifest.artifact.dim,
            bits: 4,
        },
        ProfileParameterization::BucketFrequency,
        vec![rq_manifest.artifact.dim, 4],
    ));
    let report = verify_manifest_with_base(rq_manifest, rq_case.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"calibration_ordinalization_artifact_mismatch"));

    let sign_case = tempfile::tempdir_in(temp.path()).unwrap();
    let sign = write_index_kind(sign_case.path(), FixtureKind::SignBitmap);
    let sign_manifest_path = sign_case.path().join("manifest.json");
    let sign_profile_hash = write_profile(&sign_case.path().join("sign.f64"), 64 * 8);
    let mut sign_manifest = create_manifest_for_index(
        &sign,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &sign_manifest_path,
    )
    .unwrap();
    sign_manifest.calibration = Some(weighted_calibration(
        &sign_manifest,
        "sign.f64",
        sign_profile_hash,
        CalibrationOrdinalization::Sign {
            dim: sign_manifest.artifact.dim,
        },
        ProfileParameterization::SignFrequency,
        vec![sign_manifest.artifact.dim],
    ));
    let report =
        verify_manifest_with_base(sign_manifest, sign_case.path(), VerifyOptions::default());
    assert!(report.ok, "{:?}", report.errors);

    let rank_case = tempfile::tempdir_in(temp.path()).unwrap();
    let rank = write_index_kind(rank_case.path(), FixtureKind::Rank);
    let rank_manifest_path = rank_case.path().join("manifest.json");
    let rank_profile_hash = write_profile(&rank_case.path().join("rank-position.f64"), 8 * 8 * 8);
    let mut rank_manifest = create_manifest_for_index(
        &rank,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &rank_manifest_path,
    )
    .unwrap();
    rank_manifest.calibration = Some(weighted_calibration(
        &rank_manifest,
        "rank-position.f64",
        rank_profile_hash,
        CalibrationOrdinalization::RankPosition {
            dim: rank_manifest.artifact.dim,
        },
        ProfileParameterization::RankPositionFrequency,
        vec![rank_manifest.artifact.dim, rank_manifest.artifact.dim],
    ));
    let report =
        verify_manifest_with_base(rank_manifest, rank_case.path(), VerifyOptions::default());
    assert!(report.ok, "{:?}", report.errors);
}

#[test]
fn uniform_hypergeometric_requires_top_k_ordinalization() {
    let temp = tempfile::tempdir().unwrap();

    let bitmap_case = tempfile::tempdir_in(temp.path()).unwrap();
    let bitmap = write_index_kind(bitmap_case.path(), FixtureKind::Bitmap);
    let bitmap_manifest_path = bitmap_case.path().join("manifest.json");
    let mut bitmap_manifest = create_manifest_for_index(
        &bitmap,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &bitmap_manifest_path,
    )
    .unwrap();
    bitmap_manifest.calibration = Some(uniform_calibration(
        &bitmap_manifest,
        CalibrationOrdinalization::TopK {
            dim: bitmap_manifest.artifact.dim,
            k: 16,
        },
    ));
    let report = verify_manifest_with_base(
        bitmap_manifest,
        bitmap_case.path(),
        VerifyOptions::default(),
    );
    assert!(report.ok, "{:?}", report.errors);

    for (kind, ordinalization) in [
        (
            FixtureKind::RankQuant,
            CalibrationOrdinalization::Bucket { dim: 16, bits: 2 },
        ),
        (
            FixtureKind::SignBitmap,
            CalibrationOrdinalization::Sign { dim: 64 },
        ),
        (
            FixtureKind::Rank,
            CalibrationOrdinalization::RankPosition { dim: 8 },
        ),
    ] {
        let case = tempfile::tempdir_in(temp.path()).unwrap();
        let index = write_index_kind(case.path(), kind);
        let manifest_path = case.path().join("manifest.json");
        let mut manifest = create_manifest_for_index(
            &index,
            CreateRowIdentity::RowIdIdentity,
            "test-embedding",
            &manifest_path,
        )
        .unwrap();
        manifest.calibration = Some(uniform_calibration(&manifest, ordinalization));
        let report = verify_manifest_with_base(manifest, case.path(), VerifyOptions::default());
        assert!(
            error_codes(&report).contains(&"calibration_null_model_ordinalization_mismatch"),
            "expected uniform_hypergeometric rejection: {:?}",
            report.errors
        );
    }
}

#[test]
fn calibration_profile_artifact_checks_are_enforced() {
    let temp = tempfile::tempdir().unwrap();
    let case = tempfile::tempdir_in(temp.path()).unwrap();
    let profile_dir = case.path().join("profiles");
    fs::create_dir(&profile_dir).unwrap();
    let index = write_index_kind(case.path(), FixtureKind::Bitmap);
    let manifest_path = case.path().join("manifest.json");
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    let profile_hash = write_profile(
        &profile_dir.join("profile.f64"),
        manifest.artifact.dim * std::mem::size_of::<f64>(),
    );
    manifest.calibration = Some(weighted_calibration(
        &manifest,
        "profiles/profile.f64",
        profile_hash.clone(),
        CalibrationOrdinalization::TopK {
            dim: manifest.artifact.dim,
            k: 16,
        },
        ProfileParameterization::MarginalTopKFrequency,
        vec![manifest.artifact.dim],
    ));
    let report = verify_manifest_with_base(manifest.clone(), case.path(), VerifyOptions::default());
    assert!(report.ok, "{:?}", report.errors);
    assert!(report.calibration.present);
    assert_eq!(
        report.calibration.profile_sha256.as_deref(),
        Some(profile_hash.sha256.as_str())
    );

    let mut missing_profile = manifest.clone();
    missing_profile.calibration.as_mut().unwrap().profile = None;
    let report = verify_manifest_with_base(missing_profile, case.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"calibration_profile_required"));

    let mut unexpected_profile = manifest.clone();
    unexpected_profile.calibration.as_mut().unwrap().null_model =
        NullModelSpec::UniformHypergeometric;
    let report =
        verify_manifest_with_base(unexpected_profile, case.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"calibration_profile_unexpected"));

    let mut hash_mismatch = manifest.clone();
    hash_mismatch
        .calibration
        .as_mut()
        .unwrap()
        .profile
        .as_mut()
        .unwrap()
        .sha256 = "b".repeat(64);
    let report = verify_manifest_with_base(hash_mismatch, case.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"calibration_profile_sha256_mismatch"));

    let mut size_mismatch = manifest.clone();
    size_mismatch
        .calibration
        .as_mut()
        .unwrap()
        .profile
        .as_mut()
        .unwrap()
        .file_size_bytes += 8;
    let report = verify_manifest_with_base(size_mismatch, case.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"calibration_profile_file_size_mismatch"));

    let mut zero_sample = manifest.clone();
    zero_sample
        .calibration
        .as_mut()
        .unwrap()
        .profile
        .as_mut()
        .unwrap()
        .sample_count = 0;
    let report = verify_manifest_with_base(zero_sample, case.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"calibration_profile_sample_count_zero"));

    let mut wrong_parameterization = manifest.clone();
    let wrong_calibration = wrong_parameterization.calibration.as_mut().unwrap();
    let wrong_profile = wrong_calibration.profile.as_mut().unwrap();
    wrong_profile.parameterization = ProfileParameterization::BucketFrequency;
    wrong_profile.shape.clear();
    wrong_calibration.null_model = NullModelSpec::WeightedMarginalProfile {
        parameterization: ProfileParameterization::BucketFrequency,
    };
    let report = verify_manifest_with_base(
        wrong_parameterization,
        case.path(),
        VerifyOptions::default(),
    );
    assert!(error_codes(&report)
        .contains(&"calibration_profile_parameterization_ordinalization_mismatch"));

    let outside = temp.path().join("outside-profile.f64");
    let outside_hash = write_profile(&outside, manifest.artifact.dim * std::mem::size_of::<f64>());
    let mut escaped = manifest.clone();
    let escaped_profile = escaped
        .calibration
        .as_mut()
        .unwrap()
        .profile
        .as_mut()
        .unwrap();
    escaped_profile.path = "../outside-profile.f64".to_string();
    escaped_profile.sha256 = outside_hash.sha256.clone();
    escaped_profile.file_size_bytes = outside_hash.size_bytes;
    let report = verify_manifest_with_base(escaped, case.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"calibration_profile_path_escape_rejected"));

    let mut absolute = manifest;
    let absolute_profile = absolute
        .calibration
        .as_mut()
        .unwrap()
        .profile
        .as_mut()
        .unwrap();
    absolute_profile.path = outside.display().to_string();
    absolute_profile.sha256 = outside_hash.sha256;
    absolute_profile.file_size_bytes = outside_hash.size_bytes;
    let report = verify_manifest_with_base(absolute, case.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"calibration_profile_absolute_path_rejected"));
}

#[test]
fn artifact_metadata_mismatches_are_reported_with_stable_codes() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    manifest.artifact.dim += 1;
    manifest.embedding.dim += 1;

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(!report.ok);
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_dim_mismatch"));

    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    manifest.artifact.params = ManifestIndexParams::RankQuant { bits: 4 };
    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_params_mismatch"));

    let case = tempfile::tempdir_in(root.path()).unwrap();
    let bitmap = write_index_kind(case.path(), FixtureKind::Bitmap);
    let manifest_path = case.path().join("bitmap.manifest.json");
    let mut manifest = create_manifest_for_index(
        &bitmap,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    manifest.artifact.params = ManifestIndexParams::Bitmap { n_top: 8 };
    let report = verify_manifest_with_base(manifest, case.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_params_mismatch"));
}

#[test]
fn missing_artifact_and_row_count_mismatch_are_reported() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    manifest.row_identity = RowIdentity::RowIdIdentity { row_count: 1 };
    let report = verify_manifest_with_base(manifest.clone(), temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_row_count_mismatch"));

    manifest.row_identity = RowIdentity::RowIdIdentity { row_count: 2 };
    fs::remove_file(temp.path().join(&manifest.artifact.path)).unwrap();
    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_path_unavailable"));
}

#[test]
fn path_policy_rejects_escapes_and_absolute_paths_by_default() {
    let root = tempfile::tempdir().unwrap();
    let base = root.path().join("manifests");
    fs::create_dir(&base).unwrap();
    let index = write_index(root.path());
    let manifest_path = base.join("manifest.json");
    let mut manifest = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            allow_path_escape: true,
            ..CreateManifestOptions::default()
        },
    )
    .unwrap();

    manifest.artifact.path = "../index.tvrq".to_string();
    let report = verify_manifest_with_base(manifest.clone(), &base, VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_path_escape_rejected"));

    let report = verify_manifest_with_base(
        manifest.clone(),
        &base,
        VerifyOptions {
            allow_path_escape: true,
            ..VerifyOptions::default()
        },
    );
    assert!(report.ok, "{:?}", report.errors);

    manifest.artifact.path = index.display().to_string();
    let report = verify_manifest_with_base(manifest.clone(), &base, VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_absolute_path_rejected"));

    let report = verify_manifest_with_base(
        manifest,
        &base,
        VerifyOptions {
            allow_absolute_paths: true,
            allow_path_escape: true,
            ..VerifyOptions::default()
        },
    );
    assert!(report.ok, "{:?}", report.errors);
}

#[cfg(unix)]
#[test]
fn symlink_escape_reports_observed_canonical_path() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().unwrap();
    let base = root.path().join("base");
    let outside = root.path().join("outside");
    fs::create_dir(&base).unwrap();
    fs::create_dir(&outside).unwrap();
    let index = write_index(&outside);
    symlink(&index, base.join("link.tvrq")).unwrap();
    let manifest_path = base.join("manifest.json");
    let mut manifest = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            allow_path_escape: true,
            ..CreateManifestOptions::default()
        },
    )
    .unwrap();
    manifest.artifact.path = "link.tvrq".to_string();

    let report = verify_manifest_with_base(manifest.clone(), &base, VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_path_escape_rejected"));

    let report = verify_manifest_with_base(
        manifest,
        &base,
        VerifyOptions {
            allow_path_escape: true,
            ..VerifyOptions::default()
        },
    );
    assert!(report.ok, "{:?}", report.errors);
    assert_eq!(
        PathBuf::from(report.artifact.canonical_path.unwrap()),
        fs::canonicalize(index).unwrap()
    );
}

#[test]
fn jsonl_row_identity_is_strict_and_duplicate_ids_need_opt_in() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let rows = temp.path().join("rows.jsonl");
    write_row_map(
        &rows,
        &[
            ("00000000-0000-0000-0000-000000000001", None),
            ("00000000-0000-0000-0000-000000000001", None),
        ],
    );
    let row_hash = sha256_file(&rows).unwrap();
    let manifest_path = temp.path().join("manifest.json");
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    manifest.row_identity = RowIdentity::Jsonl {
        path: "rows.jsonl".to_string(),
        sha256: row_hash.sha256,
        row_count: 2,
        id_kind: "uuid".to_string(),
        db: None,
    };

    let report = verify_manifest_with_base(manifest.clone(), temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "row_identity_duplicate_db_id"));

    let report = verify_manifest_with_base(
        manifest,
        temp.path(),
        VerifyOptions {
            allow_duplicate_db_ids: true,
            ..VerifyOptions::default()
        },
    );
    assert!(report.ok, "{:?}", report.errors);

    fs::write(
        &rows,
        "{\"row_id\":1,\"db_id\":\"\"}\n{\"row_id\":1,\"db_id\":\"ok\",\"extra\":true}\n",
    )
    .unwrap();
    let row_hash = sha256_file(&rows).unwrap();
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    manifest.row_identity = RowIdentity::Jsonl {
        path: "rows.jsonl".to_string(),
        sha256: row_hash.sha256,
        row_count: 2,
        id_kind: "uuid".to_string(),
        db: None,
    };
    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "row_identity_jsonl_invalid_json"));
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "row_identity_row_id_mismatch"));
}

#[test]
fn attestation_shape_requires_matching_subject_sha256() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    manifest.attestations.push(json!({
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {"builder": {"id": "builder"}},
        "subject": [{"name": "index.tvrq", "digest": {"sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}]
    }));

    let report = verify_manifest_with_base(manifest.clone(), temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "attestation_subject_sha256_mismatch"));

    let sha = manifest.artifact.sha256.clone();
    manifest.attestations[0]["subject"][0]["digest"]["sha256"] = json!(sha);
    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(report.ok, "{:?}", report.errors);
    assert_eq!(
        report.attestation_shape_checks[0].predicate_type.as_deref(),
        Some("https://slsa.dev/provenance/v1")
    );
}

#[test]
fn cli_create_verify_and_exit_codes() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let manifest = temp.path().join("manifest.json");
    let bin = env!("CARGO_BIN_EXE_ordvec-manifest");

    let output = Command::new(bin)
        .args([
            "create",
            "--index",
            index.to_str().unwrap(),
            "--row-id-is-identity",
            "--embedding-model",
            "test-embedding",
            "--out",
            manifest.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(bin)
        .args(["verify", "--manifest", manifest.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mut document = load_manifest_file(&manifest).unwrap();
    document.manifest.artifact.sha256 =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string();
    fs::write(
        &manifest,
        serde_json::to_string_pretty(&document.manifest).unwrap(),
    )
    .unwrap();
    let output = Command::new(bin)
        .args(["verify", "--manifest", manifest.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));

    let output = Command::new(bin)
        .args([
            "create",
            "--index",
            index.to_str().unwrap(),
            "--embedding-model",
            "test-embedding",
            "--out",
            manifest.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));

    let rows = temp.path().join("rows.jsonl");
    write_row_map(
        &rows,
        &[
            ("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", None),
            ("ffffffff-1111-4222-8333-444444444444", None),
        ],
    );
    let output = Command::new(bin)
        .args([
            "create",
            "--index",
            index.to_str().unwrap(),
            "--row-map",
            rows.to_str().unwrap(),
            "--embedding-model",
            "test-embedding",
            "--out",
            manifest.to_str().unwrap(),
            "--max-row-map-rows",
            "1",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));

    let output = Command::new(bin)
        .args([
            "create",
            "--index",
            index.to_str().unwrap(),
            "--row-map",
            rows.to_str().unwrap(),
            "--embedding-model",
            "test-embedding",
            "--out",
            manifest.to_str().unwrap(),
            "--max-row-map-tracked-id-bytes",
            "10",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));

    let output = Command::new(bin)
        .args([
            "create",
            "--index",
            index.to_str().unwrap(),
            "--row-map",
            rows.to_str().unwrap(),
            "--embedding-model",
            "test-embedding",
            "--out",
            manifest.to_str().unwrap(),
            "--max-row-map-rows",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn create_outside_manifest_dir_requires_explicit_path_policy() {
    let temp = tempfile::tempdir().unwrap();
    let outside = temp.path().join("outside");
    let manifests = temp.path().join("manifests");
    fs::create_dir(&outside).unwrap();
    let index = write_index(&outside);
    let manifest_path = manifests.join("manifest.json");

    let err = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap_err();
    assert!(err.to_string().contains("outside manifest directory"));

    let bin = env!("CARGO_BIN_EXE_ordvec-manifest");
    let output = Command::new(bin)
        .args([
            "create",
            "--index",
            index.to_str().unwrap(),
            "--row-id-is-identity",
            "--embedding-model",
            "test-embedding",
            "--out",
            manifest_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));

    let output = Command::new(bin)
        .args([
            "create",
            "--index",
            index.to_str().unwrap(),
            "--row-id-is-identity",
            "--embedding-model",
            "test-embedding",
            "--out",
            manifest_path.to_str().unwrap(),
            "--allow-path-escape",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(bin)
        .args(["verify", "--manifest", manifest_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));

    let output = Command::new(bin)
        .args([
            "verify",
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--allow-path-escape",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn verify_index_manifest_uses_explicit_index_override() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let manifest_path = temp.path().join("manifest.json");
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    manifest.artifact.path = "missing.tvrq".to_string();
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let report = verify_index_manifest(
        PathBuf::from("index.tvrq"),
        &manifest_path,
        VerifyOptions::default(),
    )
    .unwrap();
    assert!(report.ok, "{:?}", report.errors);
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_cache_is_explicit_and_activation_reverifies_by_default() {
    use rusqlite::Connection;
    use std::fs::OpenOptions;

    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let manifest_path = temp.path().join("manifest.json");
    let manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let document = load_manifest_file(&manifest_path).unwrap();
    let db = temp.path().join("registry.sqlite");

    let report = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        true,
    )
    .unwrap();
    assert!(report.ok, "{:?}", report.errors);

    let second_fresh = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        false,
    )
    .unwrap();
    assert!(second_fresh.ok, "{:?}", second_fresh.errors);

    let conn = Connection::open(&db).unwrap();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM verification_reports", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert!(
        count >= 2,
        "rapid verifications must preserve report history"
    );

    OpenOptions::new()
        .append(true)
        .open(&index)
        .unwrap()
        .write_all(b"\0")
        .unwrap();

    let cached = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        true,
    )
    .unwrap();
    assert!(
        !cached.ok,
        "cache key mismatch must force fresh verification"
    );

    let fresh = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        false,
    )
    .unwrap();
    assert!(!fresh.ok);

    let activation = ordvec_manifest::sqlite::activate(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        false,
    )
    .unwrap();
    assert!(!activation.ok);

    let forced = ordvec_manifest::sqlite::activate(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        true,
    )
    .unwrap();
    assert!(!forced.ok);
    assert!(forced
        .warnings
        .iter()
        .any(|issue| issue.code == "sqlite_activation_forced"));

    let bin = env!("CARGO_BIN_EXE_ordvec-manifest");
    let output = Command::new(bin)
        .args([
            "sqlite",
            "activate",
            "--db",
            db.to_str().unwrap(),
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--force",
            "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    let forced_report: ordvec_manifest::VerificationReport =
        serde_json::from_slice(&output.stdout).unwrap();
    assert!(!forced_report.ok);
    assert!(forced_report
        .warnings
        .iter()
        .any(|issue| issue.code == "sqlite_activation_forced"));
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_cache_key_includes_calibration_profile_bytes() {
    let temp = tempfile::tempdir().unwrap();
    let profile_dir = temp.path().join("profiles");
    fs::create_dir(&profile_dir).unwrap();
    let index = write_index_kind(temp.path(), FixtureKind::Bitmap);
    let manifest_path = temp.path().join("manifest.json");
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    let profile_path = profile_dir.join("profile.f64");
    let profile_hash = write_profile(
        &profile_path,
        manifest.artifact.dim * std::mem::size_of::<f64>(),
    );
    manifest.calibration = Some(weighted_calibration(
        &manifest,
        "profiles/profile.f64",
        profile_hash,
        CalibrationOrdinalization::TopK {
            dim: manifest.artifact.dim,
            k: 16,
        },
        ProfileParameterization::MarginalTopKFrequency,
        vec![manifest.artifact.dim],
    ));
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let document = load_manifest_file(&manifest_path).unwrap();
    let db = temp.path().join("registry.sqlite");

    let report = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        true,
    )
    .unwrap();
    assert!(report.ok, "{:?}", report.errors);

    fs::write(
        &profile_path,
        vec![1u8; manifest.artifact.dim * std::mem::size_of::<f64>()],
    )
    .unwrap();
    let cached = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        true,
    )
    .unwrap();
    assert!(!cached.ok, "profile drift must force fresh verification");
    assert!(error_codes(&cached).contains(&"calibration_profile_sha256_mismatch"));
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_cache_key_includes_limits_and_bounds_cached_report_size() {
    use rusqlite::Connection;

    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let manifest_path = temp.path().join("manifest.json");
    let manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let document = load_manifest_file(&manifest_path).unwrap();
    let db = temp.path().join("registry.sqlite");

    let options_a = VerifyOptions {
        limits: ResourceLimits {
            max_report_issues: 17,
            ..ResourceLimits::default()
        },
        ..VerifyOptions::default()
    };
    let report = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        options_a.clone(),
        true,
    )
    .unwrap();
    assert!(report.ok, "{:?}", report.errors);
    let cached = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        options_a,
        true,
    )
    .unwrap();
    assert!(cached.ok, "{:?}", cached.errors);

    let conn = Connection::open(&db).unwrap();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM verification_reports", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 1, "same limits should reuse the cached report");

    let options_b = VerifyOptions {
        limits: ResourceLimits {
            max_report_issues: 18,
            ..ResourceLimits::default()
        },
        ..VerifyOptions::default()
    };
    let report = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        options_b,
        true,
    )
    .unwrap();
    assert!(report.ok, "{:?}", report.errors);
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM verification_reports", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 2, "limit changes must produce a distinct cache key");

    let tiny_cache_limit = VerifyOptions {
        limits: ResourceLimits {
            max_cached_report_bytes: 1,
            ..ResourceLimits::default()
        },
        ..VerifyOptions::default()
    };
    let err = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        tiny_cache_limit,
        true,
    )
    .unwrap_err();
    assert_eq!(err.code(), Some("sqlite_cached_report_too_large"));
}
