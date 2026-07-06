use ordvec::{Bitmap, Rank, RankQuant, SignBitmap};
use ordvec_manifest::{
    create_manifest_for_index, create_manifest_for_index_with_options, load_manifest_file,
    load_manifest_file_with_options, sha256_file, verify_document_for_load, verify_for_load,
    verify_index_manifest, verify_manifest_with_base, AuxiliaryArtifact, AuxiliaryArtifactState,
    CalibrationOrdinalization, CalibrationProfileRef, CreateAuxiliaryArtifact,
    CreateManifestOptions, CreateRowIdentity, DistortionBounds, DistortionEvidence,
    DistortionEvidenceKind, DistortionProfileArtifactRef, DistortionScope,
    EncoderDistortionProfileRef, EncoderSpec, ManifestIndexKind, ManifestIndexParams, MetricSpec,
    NullModelSpec, ProfileArtifactRef, ProfileParameterization, RequireAuxiliaryError,
    ResourceLimits, RowIdentity, VerifiedLoadPlanError, VerifyOptions, CALIBRATION_SCHEMA_VERSION,
    ENCODER_DISTORTION_SCHEMA_VERSION,
};
use serde_json::json;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
#[cfg(feature = "cli")]
use std::process::Command;

fn write_index(dir: &Path) -> PathBuf {
    let path = dir.join("index.ovrq");
    let mut index = RankQuant::new(16, 2);
    let docs: Vec<f32> = (0..32).map(|i| i as f32 - 12.0).collect();
    index.add(&docs);
    index.write(&path).unwrap();
    path
}

fn write_rankquant_index(dir: &Path, rows: usize) -> PathBuf {
    let path = dir.join("index.ovrq");
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
            let path = dir.join("index.ovr");
            let mut index = Rank::new(8);
            index.add(&[
                1.0, 3.0, 2.0, 4.0, 8.0, 7.0, 6.0, 5.0, 8.0, 6.0, 7.0, 5.0, 1.0, 2.0, 3.0, 4.0,
            ]);
            index.write(&path).unwrap();
            path
        }
        FixtureKind::RankQuant => write_index(dir),
        FixtureKind::Bitmap => {
            let path = dir.join("index.ovbm");
            let mut index = Bitmap::new(64, 16);
            let docs: Vec<f32> = (0..128).map(|i| ((i * 17) % 31) as f32).collect();
            index.add(&docs);
            index.write(&path).unwrap();
            path
        }
        FixtureKind::SignBitmap => {
            let path = dir.join("index.ovsb");
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

fn auxiliary_artifact(
    name: &str,
    path: &str,
    hash: ordvec_manifest::FileHash,
    required: bool,
) -> AuxiliaryArtifact {
    AuxiliaryArtifact {
        name: name.to_string(),
        path: path.to_string(),
        sha256: hash.sha256,
        file_size_bytes: hash.size_bytes,
        required,
    }
}

fn sha256_uri(byte: char) -> String {
    format!("sha256:{}", byte.to_string().repeat(64))
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

fn distortion_profile(
    manifest: &ordvec_manifest::IndexManifest,
    path: Option<String>,
    hash: Option<ordvec_manifest::FileHash>,
    evidence_kind: DistortionEvidenceKind,
) -> EncoderDistortionProfileRef {
    EncoderDistortionProfileRef {
        schema_version: ENCODER_DISTORTION_SCHEMA_VERSION.to_string(),
        profile_id: "urn:uuid:a8c39375-ae65-4924-92f5-8088adfab9b5".to_string(),
        created_at: Some("2026-06-01T08:00:00Z".to_string()),
        encoder: EncoderSpec {
            model: manifest.embedding.model.clone(),
            dim: manifest.embedding.dim,
            model_revision: manifest.embedding.model_revision.clone(),
            normalization: manifest.embedding.normalization.clone(),
        },
        tokenizer_revision: manifest.embedding.tokenizer_revision.clone(),
        pooling: manifest.embedding.pooling.clone(),
        source_metric: MetricSpec {
            name: "qrel_distance".to_string(),
            version: Some("v1".to_string()),
            digest: Some(sha256_uri('a')),
        },
        embedding_metric: MetricSpec {
            name: "cosine".to_string(),
            version: None,
            digest: None,
        },
        bounds: DistortionBounds {
            declared_lower_bound: Some(0.5),
            declared_upper_bound: Some(2.0),
            estimated_distortion: Some(4.0),
            violation_rate: Some(0.01),
            max_observed_violation: Some(0.05),
            quantile_observed_violation: Some(0.02),
        },
        scope: DistortionScope {
            corpus_digest: manifest
                .embedding
                .corpus_digest
                .as_ref()
                .map(|digest| format!("sha256:{digest}")),
            query_set_digest: Some(sha256_uri('b')),
            pair_sample_digest: Some(sha256_uri('c')),
            domain: Some("arxiv-abstracts".to_string()),
            sample_size: Some(10_000),
            confidence: Some(0.997),
            coverage: Some(0.99),
            estimator_version: Some("encoder-distortion-estimator/0.1.0".to_string()),
        },
        evidence: DistortionEvidence {
            kind: evidence_kind,
            estimator_id: Some("ordvec-harness/distortion-profile".to_string()),
            estimator_hash: Some(sha256_uri('d')),
        },
        profile: path.map(|path| {
            let hash = hash.expect("profile hash required when path is present");
            DistortionProfileArtifactRef {
                path,
                sha256: hash.sha256,
                file_size_bytes: hash.size_bytes,
                format: "json".to_string(),
                source_digest: Some(sha256_uri('e')),
            }
        }),
        calibration_profile_id: None,
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
fn create_manifest_declares_auxiliary_artifacts_for_load_plan_lookup() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let ids = temp.path().join("ids.bin");
    let optional = temp.path().join("optional.json");
    fs::write(&ids, 7u64.to_le_bytes()).unwrap();
    fs::write(&optional, br#"{"optional":true}"#).unwrap();
    let ids_hash = sha256_file(&ids).unwrap();
    let optional_hash = sha256_file(&optional).unwrap();
    let manifest_path = temp.path().join("manifest.json");

    let manifest = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            auxiliary_artifacts: vec![
                CreateAuxiliaryArtifact {
                    name: " app.ids ".to_string(),
                    path: ids.clone(),
                    required: true,
                },
                CreateAuxiliaryArtifact {
                    name: "optional.stats".to_string(),
                    path: optional.clone(),
                    required: false,
                },
            ],
            ..CreateManifestOptions::default()
        },
    )
    .unwrap();

    assert_eq!(manifest.auxiliary_artifacts.len(), 2);
    assert_eq!(manifest.auxiliary_artifacts[0].name, "app.ids");
    assert_eq!(manifest.auxiliary_artifacts[0].path, "ids.bin");
    assert_eq!(manifest.auxiliary_artifacts[0].sha256, ids_hash.sha256);
    assert_eq!(
        manifest.auxiliary_artifacts[0].file_size_bytes,
        ids_hash.size_bytes
    );
    assert!(manifest.auxiliary_artifacts[0].required);
    assert_eq!(manifest.auxiliary_artifacts[1].name, "optional.stats");
    assert_eq!(manifest.auxiliary_artifacts[1].path, "optional.json");
    assert_eq!(manifest.auxiliary_artifacts[1].sha256, optional_hash.sha256);
    assert!(!manifest.auxiliary_artifacts[1].required);

    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    fs::remove_file(&optional).unwrap();

    let plan = verify_for_load(&manifest_path, VerifyOptions::default()).unwrap();
    assert_eq!(
        plan.require_auxiliary("app.ids").unwrap(),
        fs::canonicalize(&ids).unwrap().as_path()
    );
    assert_eq!(
        plan.require_auxiliary(" app.ids ").unwrap(),
        fs::canonicalize(&ids).unwrap().as_path()
    );
    assert_eq!(
        plan.auxiliary_by_name("optional.stats").unwrap().state(),
        AuxiliaryArtifactState::OptionalAbsent
    );
    assert!(matches!(
        plan.require_auxiliary("missing"),
        Err(RequireAuxiliaryError::MissingDeclaration { .. })
    ));
}

#[test]
fn create_manifest_rejects_invalid_auxiliary_artifact_declarations() {
    let root = tempfile::tempdir().unwrap();
    let case = tempfile::tempdir_in(root.path()).unwrap();
    let index = write_index(case.path());
    let sidecar = case.path().join("ids.bin");
    fs::write(&sidecar, b"sidecar").unwrap();
    let manifest_path = case.path().join("manifest.json");

    let err = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            auxiliary_artifacts: vec![CreateAuxiliaryArtifact {
                name: " ".to_string(),
                path: sidecar.clone(),
                required: true,
            }],
            ..CreateManifestOptions::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("name must be non-empty"));

    let err = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            auxiliary_artifacts: vec![
                CreateAuxiliaryArtifact {
                    name: "dup".to_string(),
                    path: sidecar.clone(),
                    required: true,
                },
                CreateAuxiliaryArtifact {
                    name: "dup".to_string(),
                    path: sidecar.clone(),
                    required: false,
                },
            ],
            ..CreateManifestOptions::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("duplicated"));

    let err = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            limits: ResourceLimits {
                max_auxiliary_artifacts: 0,
                ..ResourceLimits::default()
            },
            auxiliary_artifacts: vec![CreateAuxiliaryArtifact {
                name: "ids".to_string(),
                path: sidecar.clone(),
                required: true,
            }],
            ..CreateManifestOptions::default()
        },
    )
    .unwrap_err();
    assert_eq!(err.code(), Some("auxiliary_artifact_count_limit_exceeded"));

    let err = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            limits: ResourceLimits {
                max_auxiliary_artifact_bytes: 1,
                ..ResourceLimits::default()
            },
            auxiliary_artifacts: vec![CreateAuxiliaryArtifact {
                name: "ids".to_string(),
                path: sidecar.clone(),
                required: true,
            }],
            ..CreateManifestOptions::default()
        },
    )
    .unwrap_err();
    assert_eq!(err.code(), Some("auxiliary_artifact_file_too_large"));

    let missing = case.path().join("missing.bin");
    let err = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            auxiliary_artifacts: vec![CreateAuxiliaryArtifact {
                name: "missing".to_string(),
                path: missing,
                required: true,
            }],
            ..CreateManifestOptions::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("No such file") || err.to_string().contains("not found"));

    let outside = root.path().join("outside.bin");
    fs::write(&outside, b"outside").unwrap();
    let err = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            auxiliary_artifacts: vec![CreateAuxiliaryArtifact {
                name: "outside".to_string(),
                path: outside,
                required: true,
            }],
            ..CreateManifestOptions::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("outside manifest directory"));
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
    assert_eq!(loaded.manifest.artifact.sha256, manifest.artifact.sha256);
}

#[test]
fn manifest_loader_rejects_unenforceable_size_limit() {
    let root = tempfile::tempdir().unwrap();
    let (_temp, manifest, manifest_path) = identity_manifest(root.path());
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = load_manifest_file_with_options(
        &manifest_path,
        &VerifyOptions {
            limits: ResourceLimits {
                max_manifest_bytes: u64::MAX,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
    )
    .unwrap_err();
    assert_eq!(err.code(), Some("manifest_file_too_large"));
    assert!(err.to_string().contains("too large to enforce"));
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
fn row_identity_validated_rows_excludes_unparsed_overrun_line() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_rankquant_index(temp.path(), 1);
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
        row_count: 1,
        id_kind: "uuid".to_string(),
        db: None,
    };

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert_eq!(report.row_identity.validated_rows, Some(1));
    assert!(report.row_identity.sha256.is_none());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "row_identity_row_count_mismatch"
            && issue.message.contains("more than declared row_count=1")));
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
fn row_identity_zero_report_issue_limit_reports_configured_limit() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    let rows = temp.path().join("rows.jsonl");
    fs::write(&rows, b"{}\n{}\n").unwrap();
    let hash = sha256_file(&rows).unwrap();
    manifest.row_identity = RowIdentity::Jsonl {
        path: "rows.jsonl".to_string(),
        sha256: hash.sha256,
        row_count: 2,
        id_kind: "uuid".to_string(),
        db: None,
    };

    let report = verify_manifest_with_base(
        manifest,
        temp.path(),
        VerifyOptions {
            limits: ResourceLimits {
                max_report_issues: 0,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
    );
    assert_eq!(report.errors.len(), 1);
    assert_eq!(
        report.errors[0].code,
        "verification_report_issue_limit_exceeded"
    );
    assert!(report.errors[0].message.contains("max_report_issues=0"));
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
    manifest.embedding.tokenizer_revision = Some("".to_string());
    manifest.embedding.pooling = Some("".to_string());
    manifest.embedding.corpus_digest = Some("A".repeat(64));
    manifest.embedding.embedding_matrix_digest = Some("not-a-digest".to_string());
    manifest.embedding.normalization = Some("".to_string());
    manifest.build = Some(ordvec_manifest::BuildInfo {
        invocation_id: "urn:uuid:7c66ad6e-bdde-49a8-b420-f1136d04f5bd".to_string(),
        builder_id: None,
        source_repo: Some("".to_string()),
        source_commit: None,
        ci_provider: None,
        ci_run_id: None,
    });

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    for code in [
        "artifact_sha256_invalid",
        "row_identity_sha256_invalid",
        "embedding_model_revision_empty",
        "embedding_tokenizer_revision_empty",
        "embedding_pooling_empty",
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
fn encoder_distortion_schema_shape_is_strict_and_optional() {
    let root = tempfile::tempdir().unwrap();
    let (temp, manifest, _manifest_path) = identity_manifest(root.path());
    let report = verify_manifest_with_base(manifest.clone(), temp.path(), VerifyOptions::default());
    assert!(report.ok, "{:?}", report.errors);
    assert!(!report.encoder_distortion.present);

    let mut with_unknown = manifest.clone();
    with_unknown.encoder_distortion = Some(distortion_profile(
        &with_unknown,
        None,
        None,
        DistortionEvidenceKind::CallerAsserted,
    ));
    let mut value = serde_json::to_value(&with_unknown).unwrap();
    value["encoder_distortion"]
        .as_object_mut()
        .unwrap()
        .insert("unknown".to_string(), json!(true));
    let parsed = serde_json::from_value::<ordvec_manifest::IndexManifest>(value);
    assert!(
        parsed.is_err(),
        "encoder_distortion must reject unknown fields"
    );

    let mut valid = manifest.clone();
    valid.embedding.model_revision = Some("rev-a".to_string());
    valid.embedding.tokenizer_revision = Some("tok-a".to_string());
    valid.embedding.pooling = Some("mean".to_string());
    valid.embedding.normalization = Some("l2".to_string());
    valid.encoder_distortion = Some(distortion_profile(
        &valid,
        None,
        None,
        DistortionEvidenceKind::CallerAsserted,
    ));
    let report = verify_manifest_with_base(valid, temp.path(), VerifyOptions::default());
    assert!(report.ok, "{:?}", report.errors);
    assert!(report.encoder_distortion.present);
    assert_eq!(
        report.encoder_distortion.evidence_kind.as_deref(),
        Some("caller_asserted")
    );

    let mut bad = manifest;
    let mut profile = distortion_profile(&bad, None, None, DistortionEvidenceKind::CallerAsserted);
    profile.schema_version = "ordvec.encoder_distortion.v2".to_string();
    profile.profile_id.clear();
    profile.created_at = Some("not-rfc3339".to_string());
    profile.source_metric.name.clear();
    profile.source_metric.version = Some(String::new());
    profile.source_metric.digest = Some("not-a-sha-uri".to_string());
    profile.embedding_metric.name.clear();
    profile.evidence.estimator_id = Some(String::new());
    profile.evidence.estimator_hash = Some(sha256_uri('A'));
    profile.bounds = DistortionBounds {
        declared_lower_bound: None,
        declared_upper_bound: None,
        estimated_distortion: None,
        violation_rate: None,
        max_observed_violation: None,
        quantile_observed_violation: None,
    };
    bad.encoder_distortion = Some(profile);
    let report = verify_manifest_with_base(bad, temp.path(), VerifyOptions::default());
    for code in [
        "encoder_distortion_schema_version_unsupported",
        "encoder_distortion_profile_id_empty",
        "encoder_distortion_created_at_invalid",
        "encoder_distortion_source_metric_name_empty",
        "encoder_distortion_source_metric_version_empty",
        "encoder_distortion_source_metric_digest_invalid",
        "encoder_distortion_embedding_metric_name_empty",
        "encoder_distortion_evidence_estimator_id_empty",
        "encoder_distortion_evidence_estimator_hash_invalid",
        "encoder_distortion_bounds_empty",
    ] {
        assert!(
            error_codes(&report).contains(&code),
            "missing {code}: {:?}",
            report.errors
        );
    }
}

#[test]
fn encoder_distortion_identity_bounds_and_scope_are_checked() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    manifest.embedding.model_revision = Some("rev-a".to_string());
    manifest.embedding.tokenizer_revision = Some("tok-a".to_string());
    manifest.embedding.pooling = Some("mean".to_string());
    manifest.embedding.normalization = Some("l2".to_string());
    manifest.embedding.corpus_digest = Some("b".repeat(64));

    let mut profile = distortion_profile(
        &manifest,
        None,
        None,
        DistortionEvidenceKind::CallerAsserted,
    );
    profile.encoder.model = "other-model".to_string();
    profile.encoder.dim += 1;
    profile.encoder.model_revision = Some("rev-b".to_string());
    profile.encoder.normalization = Some("as_provided".to_string());
    profile.tokenizer_revision = Some("tok-b".to_string());
    profile.pooling = Some("cls".to_string());
    profile.bounds.declared_lower_bound = Some(3.0);
    profile.bounds.declared_upper_bound = Some(2.0);
    profile.bounds.estimated_distortion = Some(99.0);
    profile.bounds.violation_rate = Some(2.0);
    profile.bounds.max_observed_violation = Some(-1.0);
    profile.bounds.quantile_observed_violation = Some(-0.1);
    profile.scope.corpus_digest = Some("not-a-sha-uri".to_string());
    profile.scope.query_set_digest = Some(sha256_uri('Q'));
    profile.scope.domain = Some(String::new());
    profile.scope.sample_size = Some(0);
    profile.scope.confidence = Some(-0.1);
    profile.scope.coverage = Some(1.1);
    profile.scope.estimator_version = Some(String::new());
    manifest.encoder_distortion = Some(profile);

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    for code in [
        "encoder_distortion_encoder_model_mismatch",
        "encoder_distortion_encoder_dim_mismatch",
        "encoder_distortion_encoder_model_revision_mismatch",
        "encoder_distortion_encoder_normalization_mismatch",
        "encoder_distortion_tokenizer_revision_mismatch",
        "encoder_distortion_pooling_mismatch",
        "encoder_distortion_bounds_order_invalid",
        "encoder_distortion_distortion_mismatch",
        "encoder_distortion_violation_rate_invalid",
        "encoder_distortion_max_observed_violation_invalid",
        "encoder_distortion_quantile_observed_violation_invalid",
        "encoder_distortion_scope_corpus_digest_invalid",
        "encoder_distortion_scope_query_set_digest_invalid",
        "encoder_distortion_scope_domain_empty",
        "encoder_distortion_scope_sample_size_zero",
        "encoder_distortion_scope_confidence_invalid",
        "encoder_distortion_scope_coverage_invalid",
        "encoder_distortion_scope_estimator_version_empty",
    ] {
        assert!(
            error_codes(&report).contains(&code),
            "missing {code}: {:?}",
            report.errors
        );
    }
}

#[test]
fn encoder_distortion_bounds_ratio_overflow_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    let mut profile = distortion_profile(
        &manifest,
        None,
        None,
        DistortionEvidenceKind::CallerAsserted,
    );
    profile.bounds.declared_lower_bound = Some(f64::MIN_POSITIVE);
    profile.bounds.declared_upper_bound = Some(f64::MAX);
    profile.bounds.estimated_distortion = Some(1.0);
    manifest.encoder_distortion = Some(profile);

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(
        error_codes(&report).contains(&"encoder_distortion_distortion_mismatch"),
        "{:?}",
        report.errors
    );
}

#[test]
fn encoder_distortion_profile_artifact_checks_are_enforced() {
    let temp = tempfile::tempdir().unwrap();
    let case = tempfile::tempdir_in(temp.path()).unwrap();
    let profile_dir = case.path().join("profiles");
    fs::create_dir(&profile_dir).unwrap();
    let index = write_index(case.path());
    let manifest_path = case.path().join("manifest.json");
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    let profile_path = profile_dir.join("distortion.json");
    let profile_hash = write_profile(&profile_path, 128);
    manifest.encoder_distortion = Some(distortion_profile(
        &manifest,
        Some("profiles/distortion.json".to_string()),
        Some(profile_hash.clone()),
        DistortionEvidenceKind::EmpiricalSample,
    ));
    let report = verify_manifest_with_base(manifest.clone(), case.path(), VerifyOptions::default());
    assert!(report.ok, "{:?}", report.errors);
    assert_eq!(
        report.encoder_distortion.profile_sha256.as_deref(),
        Some(profile_hash.sha256.as_str())
    );

    let report = verify_manifest_with_base(
        manifest.clone(),
        case.path(),
        VerifyOptions {
            limits: ResourceLimits {
                max_encoder_distortion_profile_bytes: 16,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
    );
    assert!(error_codes(&report).contains(&"encoder_distortion_profile_too_large"));

    let mut missing_profile = manifest.clone();
    let missing = missing_profile.encoder_distortion.as_mut().unwrap();
    missing.profile = None;
    missing.evidence.kind = DistortionEvidenceKind::EmpiricalSample;
    let report = verify_manifest_with_base(missing_profile, case.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"encoder_distortion_profile_required"));

    let mut hash_mismatch = manifest.clone();
    hash_mismatch
        .encoder_distortion
        .as_mut()
        .unwrap()
        .profile
        .as_mut()
        .unwrap()
        .sha256 = "b".repeat(64);
    let report = verify_manifest_with_base(hash_mismatch, case.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"encoder_distortion_profile_sha256_mismatch"));

    let mut size_mismatch = manifest.clone();
    size_mismatch
        .encoder_distortion
        .as_mut()
        .unwrap()
        .profile
        .as_mut()
        .unwrap()
        .file_size_bytes += 1;
    let report = verify_manifest_with_base(size_mismatch, case.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"encoder_distortion_profile_file_size_mismatch"));

    let mut bad_shape = manifest.clone();
    let bad_profile = bad_shape
        .encoder_distortion
        .as_mut()
        .unwrap()
        .profile
        .as_mut()
        .unwrap();
    bad_profile.path.clear();
    bad_profile.sha256 = sha256_uri('b');
    bad_profile.file_size_bytes = 0;
    bad_profile.format.clear();
    bad_profile.source_digest = Some("not-a-sha-uri".to_string());
    let report = verify_manifest_with_base(bad_shape, case.path(), VerifyOptions::default());
    for code in [
        "encoder_distortion_profile_path_empty",
        "encoder_distortion_profile_sha256_invalid",
        "encoder_distortion_profile_file_size_zero",
        "encoder_distortion_profile_format_empty",
        "encoder_distortion_profile_source_digest_invalid",
    ] {
        assert!(
            error_codes(&report).contains(&code),
            "missing {code}: {:?}",
            report.errors
        );
    }

    let outside = temp.path().join("outside-distortion.json");
    let outside_hash = write_profile(&outside, 128);
    let mut escaped = manifest.clone();
    let escaped_profile = escaped
        .encoder_distortion
        .as_mut()
        .unwrap()
        .profile
        .as_mut()
        .unwrap();
    escaped_profile.path = "../outside-distortion.json".to_string();
    escaped_profile.sha256 = outside_hash.sha256.clone();
    escaped_profile.file_size_bytes = outside_hash.size_bytes;
    let report = verify_manifest_with_base(escaped, case.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"encoder_distortion_profile_path_escape_rejected"));

    let mut absolute = manifest;
    let absolute_profile = absolute
        .encoder_distortion
        .as_mut()
        .unwrap()
        .profile
        .as_mut()
        .unwrap();
    absolute_profile.path = outside.display().to_string();
    absolute_profile.sha256 = outside_hash.sha256;
    absolute_profile.file_size_bytes = outside_hash.size_bytes;
    let report = verify_manifest_with_base(absolute, case.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"encoder_distortion_profile_absolute_path_rejected"));
}

#[test]
fn profile_ref_paths_must_be_canonical() {
    let temp = tempfile::tempdir().unwrap();
    let profile_dir = temp.path().join("profiles");
    fs::create_dir(&profile_dir).unwrap();
    let index = write_index_kind(temp.path(), FixtureKind::RankQuant);
    let manifest_path = temp.path().join("manifest.json");
    let distortion_hash = write_profile(&profile_dir.join("distortion.json"), 128);
    let bucket_hash = write_profile(&temp.path().join("bucket.f64"), 16 * 4 * 8);
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    manifest.encoder_distortion = Some(distortion_profile(
        &manifest,
        Some("profiles/./distortion.json".to_string()),
        Some(distortion_hash),
        DistortionEvidenceKind::EmpiricalSample,
    ));
    manifest.calibration = Some(weighted_calibration(
        &manifest,
        "a/../bucket.f64",
        bucket_hash,
        CalibrationOrdinalization::Bucket {
            dim: manifest.artifact.dim,
            bits: 2,
        },
        ProfileParameterization::BucketFrequency,
        vec![manifest.artifact.dim, 4],
    ));
    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    for code in [
        "encoder_distortion_profile_path_not_canonical",
        "calibration_profile_path_not_canonical",
    ] {
        assert!(
            error_codes(&report).contains(&code),
            "missing {code}: {:?}",
            report.errors
        );
    }
}

#[test]
fn encoder_distortion_can_bind_to_calibration_profile_id() {
    let temp = tempfile::tempdir().unwrap();
    let case = tempfile::tempdir_in(temp.path()).unwrap();
    let index = write_index_kind(case.path(), FixtureKind::RankQuant);
    let manifest_path = case.path().join("manifest.json");
    let profile_hash = write_profile(&case.path().join("bucket.f64"), 16 * 4 * 8);
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    manifest.calibration = Some(weighted_calibration(
        &manifest,
        "bucket.f64",
        profile_hash,
        CalibrationOrdinalization::Bucket {
            dim: manifest.artifact.dim,
            bits: 2,
        },
        ProfileParameterization::BucketFrequency,
        vec![manifest.artifact.dim, 4],
    ));
    let mut distortion = distortion_profile(
        &manifest,
        None,
        None,
        DistortionEvidenceKind::CallerAsserted,
    );
    distortion.calibration_profile_id =
        Some(manifest.calibration.as_ref().unwrap().profile_id.clone());
    manifest.encoder_distortion = Some(distortion);
    let report = verify_manifest_with_base(manifest.clone(), case.path(), VerifyOptions::default());
    assert!(report.ok, "{:?}", report.errors);

    let mut mismatch = manifest.clone();
    mismatch
        .encoder_distortion
        .as_mut()
        .unwrap()
        .calibration_profile_id = Some("urn:uuid:00000000-0000-0000-0000-000000000000".to_string());
    let report = verify_manifest_with_base(mismatch, case.path(), VerifyOptions::default());
    assert!(
        error_codes(&report).contains(&"encoder_distortion_calibration_profile_mismatch"),
        "{:?}",
        report.errors
    );

    let mut padded = manifest.clone();
    let calibration_profile_id = padded.calibration.as_ref().unwrap().profile_id.clone();
    padded
        .encoder_distortion
        .as_mut()
        .unwrap()
        .calibration_profile_id = Some(format!(" {calibration_profile_id} "));
    let report = verify_manifest_with_base(padded, case.path(), VerifyOptions::default());
    assert!(
        error_codes(&report).contains(&"encoder_distortion_calibration_profile_id_whitespace"),
        "{:?}",
        report.errors
    );

    let mut missing = manifest;
    missing.calibration = None;
    let report = verify_manifest_with_base(missing, case.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"encoder_distortion_calibration_missing"));
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
fn calibration_invalid_bucket_bits_reports_without_panic() {
    let temp = tempfile::tempdir().unwrap();
    let case = tempfile::tempdir_in(temp.path()).unwrap();
    let index = write_index_kind(case.path(), FixtureKind::RankQuant);
    let manifest_path = case.path().join("manifest.json");
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    let profile_hash = write_profile(
        &case.path().join("bucket.f64"),
        manifest.artifact.dim * std::mem::size_of::<f64>(),
    );
    manifest.calibration = Some(weighted_calibration(
        &manifest,
        "bucket.f64",
        profile_hash,
        CalibrationOrdinalization::Bucket {
            dim: manifest.artifact.dim,
            bits: 255,
        },
        ProfileParameterization::BucketFrequency,
        vec![manifest.artifact.dim, 1],
    ));

    let report = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        verify_manifest_with_base(manifest, case.path(), VerifyOptions::default())
    }))
    .expect("invalid bucket bits must report errors instead of panicking");
    assert!(error_codes(&report).contains(&"calibration_ordinalization_artifact_mismatch"));
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

    let report = verify_manifest_with_base(
        manifest.clone(),
        case.path(),
        VerifyOptions {
            limits: ResourceLimits {
                max_calibration_profile_bytes: 16,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
    );
    assert!(error_codes(&report).contains(&"calibration_profile_too_large"));

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

    manifest.artifact.path = "../index.ovrq".to_string();
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
fn single_backslash_artifact_path_fails_canonical_check_under_all_policies() {
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

    // On Unix a backslash is an ordinary file-name byte, so a crafted bundle
    // can carry a matching artifact literally named `\evil`. Classifying a
    // single leading backslash as absolute skipped the canonical-form check
    // while resolution still treated the path as relative, so this manifest
    // verified successfully before the fix. The lint misreads the backslash
    // as a path separator; on Unix this join produces a child file name.
    #[allow(clippy::join_absolute_paths)]
    fs::copy(&index, temp.path().join("\\evil")).unwrap();
    manifest.artifact.path = "\\evil".to_string();

    let report = verify_manifest_with_base(manifest.clone(), temp.path(), VerifyOptions::default());
    assert!(!report.ok);
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_path_not_canonical"));

    let report = verify_manifest_with_base(
        manifest,
        temp.path(),
        VerifyOptions {
            allow_absolute_paths: true,
            allow_path_escape: true,
            ..VerifyOptions::default()
        },
    );
    assert!(!report.ok);
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_path_not_canonical"));
}

#[cfg(unix)]
#[test]
fn unc_artifact_path_stays_policy_gated_and_never_resolves_relative() {
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
    manifest.artifact.path = "\\\\server\\share\\index.ovrq".to_string();

    // UNC paths remain classified absolute, so the default policy rejects
    // them outright.
    let report = verify_manifest_with_base(manifest.clone(), temp.path(), VerifyOptions::default());
    assert!(!report.ok);
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_absolute_path_rejected"));

    // Even with absolute paths allowed, a path that is absolute for policy
    // purposes must never silently resolve relative to the manifest base on
    // a platform (Unix) that cannot resolve it as absolute.
    let report = verify_manifest_with_base(
        manifest,
        temp.path(),
        VerifyOptions {
            allow_absolute_paths: true,
            allow_path_escape: true,
            ..VerifyOptions::default()
        },
    );
    assert!(!report.ok);
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_absolute_path_unresolvable"));
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
    symlink(&index, base.join("link.ovrq")).unwrap();
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
    manifest.artifact.path = "link.ovrq".to_string();

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
fn verify_for_load_returns_resolved_plan_and_report() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let sidecar = temp.path().join("neighbors.json");
    fs::write(&sidecar, br#"{"kind":"neighbors"}"#).unwrap();
    let sidecar_hash = sha256_file(&sidecar).unwrap();
    let manifest_path = temp.path().join("manifest.json");
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    manifest.auxiliary_artifacts.push(AuxiliaryArtifact {
        name: "neighbors".to_string(),
        path: "neighbors.json".to_string(),
        sha256: sidecar_hash.sha256.clone(),
        file_size_bytes: sidecar_hash.size_bytes,
        required: true,
    });
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let plan = verify_for_load(&manifest_path, VerifyOptions::default()).unwrap();

    assert_eq!(plan.manifest_path(), Some(manifest_path.as_path()));
    assert_eq!(
        plan.artifact_path(),
        fs::canonicalize(&index).unwrap().as_path()
    );
    assert_eq!(plan.metadata().kind, ManifestIndexKind::RankQuant);
    assert_eq!(plan.row_identity().kind(), "row_id_identity");
    assert_eq!(plan.row_identity().row_count(), 2);
    assert!(plan.report().ok, "{:?}", plan.report().errors);

    let sidecar_plan = &plan.auxiliary_artifacts()[0];
    let canonical_sidecar = fs::canonicalize(&sidecar).unwrap();
    assert_eq!(sidecar_plan.name(), "neighbors");
    assert_eq!(sidecar_plan.state(), AuxiliaryArtifactState::Verified);
    assert_eq!(sidecar_plan.path(), Some(canonical_sidecar.as_path()));
    assert_eq!(sidecar_plan.sha256(), Some(sidecar_hash.sha256.as_str()));

    let document = load_manifest_file(&manifest_path).unwrap();
    let document_plan = verify_document_for_load(&document, VerifyOptions::default()).unwrap();
    assert_eq!(document_plan.artifact_path(), plan.artifact_path());
}

#[test]
fn verify_for_load_uses_explicit_index_override() {
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
    manifest.artifact.path = "missing.ovrq".to_string();
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let plan = verify_for_load(
        &manifest_path,
        VerifyOptions {
            index_override: Some(PathBuf::from("index.ovrq")),
            ..VerifyOptions::default()
        },
    )
    .unwrap();

    assert_eq!(
        plan.artifact_path(),
        fs::canonicalize(&index).unwrap().as_path()
    );
    assert_eq!(
        plan.report().artifact.observed_path.as_deref(),
        Some("index.ovrq")
    );
}

#[test]
fn verify_for_load_returns_row_map_path_and_optional_absent_auxiliary() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let rows = temp.path().join("rows.jsonl");
    write_row_map(
        &rows,
        &[
            ("00000000-0000-0000-0000-000000000001", None),
            ("00000000-0000-0000-0000-000000000002", None),
        ],
    );
    let manifest_path = temp.path().join("manifest.json");
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::Jsonl(rows.clone()),
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    manifest.auxiliary_artifacts.push(AuxiliaryArtifact {
        name: "optional-neighbors".to_string(),
        path: "missing-neighbors.json".to_string(),
        sha256: "0".repeat(64),
        file_size_bytes: 0,
        required: false,
    });
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let plan = verify_for_load(&manifest_path, VerifyOptions::default()).unwrap();

    assert_eq!(
        plan.row_identity().path(),
        Some(fs::canonicalize(&rows).unwrap().as_path())
    );
    assert_eq!(plan.row_identity().validated_rows(), Some(2));

    let sidecar_plan = &plan.auxiliary_artifacts()[0];
    assert_eq!(sidecar_plan.name(), "optional-neighbors");
    assert_eq!(sidecar_plan.state(), AuxiliaryArtifactState::OptionalAbsent);
    assert_eq!(
        sidecar_plan.reason_code(),
        Some("auxiliary_artifact_optional_absent")
    );
    assert_eq!(sidecar_plan.path(), None);
}

#[cfg(unix)]
#[test]
fn verify_for_load_preserves_non_utf8_base_paths() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    let root = tempfile::tempdir().unwrap();
    let case = root
        .path()
        .join(OsString::from_vec(b"manifest-\xff".to_vec()));
    fs::create_dir(&case).unwrap();
    let index = write_index(&case);
    let manifest_path = case.join("manifest.json");
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

    let plan = verify_for_load(&manifest_path, VerifyOptions::default()).unwrap();

    assert_eq!(
        plan.artifact_path(),
        fs::canonicalize(index).unwrap().as_path()
    );
}

#[test]
fn verify_for_load_fails_closed_with_report_for_default_path_policy() {
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
    manifest.artifact.path = "../index.ovrq".to_string();
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = verify_for_load(&manifest_path, VerifyOptions::default()).unwrap_err();
    let VerifiedLoadPlanError::VerificationFailed(report) = err else {
        panic!("expected verification failure");
    };
    assert!(error_codes(&report).contains(&"artifact_path_escape_rejected"));

    let plan = verify_for_load(
        &manifest_path,
        VerifyOptions {
            allow_path_escape: true,
            ..VerifyOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        plan.artifact_path(),
        fs::canonicalize(index).unwrap().as_path()
    );
}

#[test]
fn verify_for_load_fails_closed_with_report_for_corrupted_artifact() {
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
    // Corrupt in place (same size): the declared-size read bound is
    // satisfied, so verification proceeds to the digest and fails there.
    let mut bytes = fs::read(&index).unwrap();
    bytes[0] ^= 0xFF;
    fs::write(&index, &bytes).unwrap();

    let err = verify_for_load(&manifest_path, VerifyOptions::default()).unwrap_err();
    let VerifiedLoadPlanError::VerificationFailed(report) = err else {
        panic!("expected verification failure");
    };
    assert!(error_codes(&report).contains(&"artifact_sha256_mismatch"));
}

#[test]
fn verify_for_load_plan_is_not_a_byte_pin() {
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

    let plan = verify_for_load(&manifest_path, VerifyOptions::default()).unwrap();
    assert_eq!(
        plan.artifact_path(),
        fs::canonicalize(&index).unwrap().as_path()
    );

    fs::OpenOptions::new()
        .append(true)
        .open(&index)
        .unwrap()
        .write_all(b"\0")
        .unwrap();

    assert!(plan.report().ok);
    assert!(
        RankQuant::load(plan.artifact_path()).is_err(),
        "a previously returned plan still resolves to the current mutable path"
    );

    let err = verify_for_load(&manifest_path, VerifyOptions::default()).unwrap_err();
    let VerifiedLoadPlanError::VerificationFailed(report) = err else {
        panic!("expected verification failure");
    };
    // The artifact grew past its declared size, so re-verification fails
    // fast at the declared-size read bound.
    assert!(error_codes(&report).contains(&"artifact_file_too_large"));
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
fn jsonl_row_identity_rejects_non_uuid_ids() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_rankquant_index(temp.path(), 2);
    let rows = temp.path().join("rows.jsonl");
    write_row_map(&rows, &[("doc-a", None), ("doc-b", Some("doc-a"))]);
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
    assert!(codes.contains(&"row_identity_db_id_invalid_uuid"));
    assert!(codes.contains(&"row_identity_parent_id_invalid_uuid"));
}

#[test]
fn jsonl_row_identity_uuid_error_message_is_v1_scoped() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_rankquant_index(temp.path(), 1);
    let rows = temp.path().join("rows.jsonl");
    write_row_map(&rows, &[("doc-a", None)]);
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
        row_count: 1,
        id_kind: "u64".to_string(),
        db: None,
    };

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    let codes = error_codes(&report);
    assert!(codes.contains(&"row_identity_id_kind_unsupported"));
    let issue = report
        .errors
        .iter()
        .find(|issue| issue.code == "row_identity_db_id_invalid_uuid")
        .expect("non-UUID db_id should still report v1 UUID validation");
    assert!(issue.message.contains("must be a UUID in v1"));
    assert!(!issue
        .message
        .contains("because row_identity.id_kind is uuid"));
}

#[test]
fn jsonl_row_identity_rejects_reserved_db_metadata() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_rankquant_index(temp.path(), 1);
    let rows = temp.path().join("rows.jsonl");
    write_row_map(&rows, &[("00000000-0000-0000-0000-000000000001", None)]);
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
        row_count: 1,
        id_kind: "uuid".to_string(),
        db: Some(ordvec_manifest::RowIdentityDb {
            path: Some("/etc/passwd".to_string()),
            table: Some("documents".to_string()),
            id_column: Some("id".to_string()),
        }),
    };

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"row_identity_db_unsupported"));
}

#[test]
fn auxiliary_artifacts_verify_and_report_deterministically() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    fs::write(temp.path().join("zeta.bin"), b"zeta").unwrap();
    fs::write(temp.path().join("alpha.bin"), b"alpha").unwrap();
    let zeta_hash = sha256_file(temp.path().join("zeta.bin")).unwrap();
    let alpha_hash = sha256_file(temp.path().join("alpha.bin")).unwrap();

    manifest.auxiliary_artifacts = vec![
        auxiliary_artifact("zeta", "zeta.bin", zeta_hash, true),
        AuxiliaryArtifact {
            name: "optional-model".to_string(),
            path: "missing-model.json".to_string(),
            sha256: "0".repeat(64),
            file_size_bytes: 0,
            required: false,
        },
        auxiliary_artifact("alpha", "alpha.bin", alpha_hash.clone(), true),
    ];

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(report.ok, "{:?}", report.errors);
    assert_eq!(
        report
            .auxiliary_artifacts
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>(),
        ["alpha", "optional-model", "zeta"]
    );
    assert_eq!(
        report.auxiliary_artifacts[0].state,
        AuxiliaryArtifactState::Verified
    );
    assert_eq!(report.auxiliary_artifacts[0].manifest_path, "alpha.bin");
    assert!(report.auxiliary_artifacts[0]
        .resolved_path
        .as_deref()
        .unwrap()
        .ends_with("alpha.bin"));
    assert_eq!(
        report.auxiliary_artifacts[0].expected_sha256.as_deref(),
        Some(alpha_hash.sha256.as_str())
    );
    assert_eq!(
        report.auxiliary_artifacts[0].expected_size_bytes,
        Some(alpha_hash.size_bytes)
    );
    assert_eq!(
        report.auxiliary_artifacts[1].state,
        AuxiliaryArtifactState::OptionalAbsent
    );
    assert_eq!(
        report.auxiliary_artifacts[1].reason_code.as_deref(),
        Some("auxiliary_artifact_optional_absent")
    );
    assert_eq!(
        report.auxiliary_artifacts[1].expected_sha256.as_deref(),
        Some("0000000000000000000000000000000000000000000000000000000000000000")
    );
    assert_eq!(report.auxiliary_artifacts[1].expected_size_bytes, Some(0));
    assert!(report.auxiliary_artifacts[1]
        .resolved_path
        .as_deref()
        .unwrap()
        .ends_with("missing-model.json"));
    assert_eq!(
        report.auxiliary_artifacts[2].state,
        AuxiliaryArtifactState::Verified
    );
}

#[test]
fn auxiliary_artifacts_fail_closed_on_tamper_missing_and_path_escape() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    let outside = root.path().join("outside.bin");
    fs::write(&outside, b"outside").unwrap();
    fs::write(temp.path().join("tampered.bin"), b"original").unwrap();
    fs::write(temp.path().join("wrong-size.bin"), b"size").unwrap();
    let tampered_hash = sha256_file(temp.path().join("tampered.bin")).unwrap();
    let wrong_size_hash = sha256_file(temp.path().join("wrong-size.bin")).unwrap();
    fs::write(temp.path().join("tampered.bin"), b"changed").unwrap();

    manifest.auxiliary_artifacts = vec![
        AuxiliaryArtifact {
            name: "missing".to_string(),
            path: "missing.bin".to_string(),
            sha256: "0".repeat(64),
            file_size_bytes: 0,
            required: true,
        },
        auxiliary_artifact("tampered", "tampered.bin", tampered_hash, true),
        AuxiliaryArtifact {
            name: "wrong-size".to_string(),
            path: "wrong-size.bin".to_string(),
            sha256: wrong_size_hash.sha256,
            file_size_bytes: wrong_size_hash.size_bytes + 1,
            required: true,
        },
        AuxiliaryArtifact {
            name: "escape".to_string(),
            path: "../outside.bin".to_string(),
            sha256: sha256_file(outside).unwrap().sha256,
            file_size_bytes: 7,
            required: true,
        },
    ];

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(!report.ok);
    let codes = error_codes(&report);
    assert!(codes.contains(&"auxiliary_artifact_missing_required"));
    assert!(codes.contains(&"auxiliary_artifact_sha256_mismatch"));
    assert!(codes.contains(&"auxiliary_artifact_file_size_mismatch"));
    assert!(codes.contains(&"auxiliary_artifact_path_escape_rejected"));
    let missing = report
        .auxiliary_artifacts
        .iter()
        .find(|entry| entry.name == "missing")
        .unwrap();
    assert_eq!(missing.state, AuxiliaryArtifactState::MissingRequired);
    assert_eq!(
        missing.expected_sha256.as_deref(),
        Some("0000000000000000000000000000000000000000000000000000000000000000")
    );
    assert_eq!(missing.expected_size_bytes, Some(0));
    assert!(missing
        .resolved_path
        .as_deref()
        .unwrap()
        .ends_with("missing.bin"));
}

#[test]
fn manifest_shape_rejects_zero_declared_file_sizes_for_required_artifacts() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    fs::write(temp.path().join("extra.bin"), b"extra").unwrap();
    let extra_hash = sha256_file(temp.path().join("extra.bin")).unwrap();

    manifest.artifact.file_size_bytes = 0;
    manifest.auxiliary_artifacts = vec![AuxiliaryArtifact {
        name: "extra".to_string(),
        path: "extra.bin".to_string(),
        sha256: extra_hash.sha256,
        file_size_bytes: 0,
        required: true,
    }];

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(!report.ok);
    let codes = error_codes(&report);
    assert!(codes.contains(&"artifact_file_size_zero"), "{codes:?}");
    assert!(
        codes.contains(&"auxiliary_artifact_file_size_zero"),
        "{codes:?}"
    );
}

#[test]
fn optional_absent_zero_size_placeholder_is_not_flagged_zero_size() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    manifest.auxiliary_artifacts = vec![AuxiliaryArtifact {
        name: "optional-model".to_string(),
        path: "missing-model.json".to_string(),
        sha256: "0".repeat(64),
        file_size_bytes: 0,
        required: false,
    }];

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(report.ok, "{:?}", report.errors);
    assert!(!error_codes(&report).contains(&"auxiliary_artifact_file_size_zero"));
}

#[test]
fn auxiliary_artifact_schema_rejects_unknown_fields_and_duplicate_names() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    fs::write(temp.path().join("sidecar.bin"), b"sidecar").unwrap();
    let sidecar_hash = sha256_file(temp.path().join("sidecar.bin")).unwrap();

    manifest.auxiliary_artifacts = vec![
        auxiliary_artifact("duplicate", "sidecar.bin", sidecar_hash.clone(), true),
        auxiliary_artifact("duplicate", "sidecar.bin", sidecar_hash.clone(), false),
    ];
    let report = verify_manifest_with_base(manifest.clone(), temp.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"auxiliary_artifact_name_duplicate"));

    let mut padded = manifest.clone();
    padded.auxiliary_artifacts = vec![auxiliary_artifact(
        " duplicate ",
        "sidecar.bin",
        sidecar_hash,
        true,
    )];
    let report = verify_manifest_with_base(padded, temp.path(), VerifyOptions::default());
    assert!(error_codes(&report).contains(&"auxiliary_artifact_name_not_trimmed"));

    let mut value = serde_json::to_value(&manifest).unwrap();
    value["auxiliary_artifacts"][0]["unexpected"] = json!(true);
    let parsed = serde_json::from_value::<ordvec_manifest::IndexManifest>(value);
    assert!(parsed.is_err());
}

#[test]
fn auxiliary_artifact_count_limit_is_enforced_before_verification() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    fs::write(temp.path().join("a.bin"), b"a").unwrap();
    fs::write(temp.path().join("b.bin"), b"b").unwrap();
    let a_hash = sha256_file(temp.path().join("a.bin")).unwrap();
    let b_hash = sha256_file(temp.path().join("b.bin")).unwrap();
    manifest.auxiliary_artifacts = vec![
        auxiliary_artifact("a", "a.bin", a_hash, true),
        auxiliary_artifact("b", "b.bin", b_hash, true),
    ];

    let report = verify_manifest_with_base(
        manifest,
        temp.path(),
        VerifyOptions {
            limits: ResourceLimits {
                max_auxiliary_artifacts: 1,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
    );
    assert!(error_codes(&report).contains(&"auxiliary_artifact_count_limit_exceeded"));
    assert!(report.auxiliary_artifacts.is_empty());
}

#[test]
fn auxiliary_artifact_byte_limit_is_enforced_before_hashing() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    let sidecar = temp.path().join("sidecar.bin");
    fs::write(&sidecar, b"sidecar").unwrap();
    let sidecar_hash = sha256_file(&sidecar).unwrap();
    manifest.auxiliary_artifacts = vec![auxiliary_artifact(
        "sidecar",
        "sidecar.bin",
        sidecar_hash.clone(),
        true,
    )];

    let report = verify_manifest_with_base(
        manifest.clone(),
        temp.path(),
        VerifyOptions {
            limits: ResourceLimits {
                max_auxiliary_artifact_bytes: sidecar_hash.size_bytes - 1,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
    );
    assert!(error_codes(&report).contains(&"auxiliary_artifact_file_too_large"));
    assert_eq!(report.auxiliary_artifacts[0].sha256, None);
    assert_eq!(
        report.auxiliary_artifacts[0].reason_code.as_deref(),
        Some("auxiliary_artifact_file_too_large")
    );

    let report = verify_manifest_with_base(
        manifest,
        temp.path(),
        VerifyOptions {
            limits: ResourceLimits {
                max_auxiliary_artifact_bytes: sidecar_hash.size_bytes,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
    );
    assert!(report.ok, "{:?}", report.errors);
    assert_eq!(
        report.auxiliary_artifacts[0].sha256.as_deref(),
        Some(sidecar_hash.sha256.as_str())
    );
}

#[test]
fn verification_report_deserializes_missing_auxiliary_artifacts_field() {
    let root = tempfile::tempdir().unwrap();
    let (temp, manifest, _manifest_path) = identity_manifest(root.path());
    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    let mut value = serde_json::to_value(&report).unwrap();
    value.as_object_mut().unwrap().remove("auxiliary_artifacts");

    let parsed: ordvec_manifest::VerificationReport = serde_json::from_value(value).unwrap();
    assert!(parsed.auxiliary_artifacts.is_empty());
}

#[test]
fn verification_report_deserializes_missing_encoder_distortion_field() {
    let root = tempfile::tempdir().unwrap();
    let (temp, manifest, _manifest_path) = identity_manifest(root.path());
    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    let mut value = serde_json::to_value(&report).unwrap();
    value.as_object_mut().unwrap().remove("encoder_distortion");

    let parsed: ordvec_manifest::VerificationReport = serde_json::from_value(value).unwrap();
    assert!(!parsed.encoder_distortion.present);
}

#[test]
fn attestation_shape_requires_matching_subject_sha256() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    manifest.attestations.push(json!({
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {"builder": {"id": "builder"}},
        "subject": [{"name": "index.ovrq", "digest": {"sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}]
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

#[cfg(feature = "cli")]
#[test]
fn cli_create_verify_and_exit_codes() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let ids = temp.path().join("ids.bin");
    let optional = temp.path().join("optional.json");
    fs::write(&ids, 7u64.to_le_bytes()).unwrap();
    fs::write(&optional, br#"{"optional":true}"#).unwrap();
    let aux_arg = format!("app.ids={}", ids.display());
    let optional_aux_arg = format!("optional.stats={}", optional.display());
    let manifest = temp.path().join("manifest.json");
    let bin = env!("CARGO_BIN_EXE_ordvec-manifest");

    let output = Command::new(bin)
        .arg("create")
        .arg("--index")
        .arg(index.to_str().unwrap())
        .arg("--row-id-is-identity")
        .arg("--aux")
        .arg(&aux_arg)
        .arg("--optional-aux")
        .arg(&optional_aux_arg)
        .arg("--embedding-model")
        .arg("test-embedding")
        .arg("--out")
        .arg(manifest.to_str().unwrap())
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
    let profile_path = temp.path().join("distortion.json");
    let profile_hash = write_profile(&profile_path, 128);
    document.manifest.encoder_distortion = Some(distortion_profile(
        &document.manifest,
        Some("distortion.json".to_string()),
        Some(profile_hash),
        DistortionEvidenceKind::EmpiricalSample,
    ));
    fs::write(
        &manifest,
        serde_json::to_string_pretty(&document.manifest).unwrap(),
    )
    .unwrap();
    let output = Command::new(bin)
        .args([
            "verify",
            "--manifest",
            manifest.to_str().unwrap(),
            "--max-encoder-distortion-profile-bytes",
            "16",
            "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("encoder_distortion_profile_too_large")
    );

    document.manifest.encoder_distortion = None;
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

#[cfg(feature = "cli")]
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
    manifest.artifact.path = "missing.ovrq".to_string();
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let report = verify_index_manifest(
        PathBuf::from("index.ovrq"),
        &manifest_path,
        VerifyOptions::default(),
    )
    .unwrap();
    assert!(report.ok, "{:?}", report.errors);
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_refuses_to_migrate_unknown_verification_reports_table() {
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
    let db = temp.path().join("foreign.sqlite");
    let conn = Connection::open(&db).unwrap();
    conn.execute("CREATE TABLE verification_reports(id INTEGER)", [])
        .unwrap();
    drop(conn);

    let err = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        true,
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("unsupported verification_reports schema"));

    let conn = Connection::open(&db).unwrap();
    let columns = conn
        .prepare("PRAGMA table_info(verification_reports)")
        .unwrap()
        .query_map([], |row| row.get::<_, String>(1))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(columns, vec!["id"]);
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_migrates_legacy_verification_reports_by_required_column_names() {
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
    let db = temp.path().join("legacy.sqlite");
    let conn = Connection::open(&db).unwrap();
    conn.execute(
        "CREATE TABLE verification_reports(
            report_json TEXT,
            checked_at TEXT,
            extra TEXT,
            ok INTEGER,
            manifest_path TEXT,
            manifest_id TEXT
        )",
        [],
    )
    .unwrap();
    drop(conn);

    let report = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        true,
    )
    .unwrap();
    assert!(report.ok, "{:?}", report.errors);

    let conn = Connection::open(&db).unwrap();
    let columns = conn
        .prepare("PRAGMA table_info(verification_reports)")
        .unwrap()
        .query_map([], |row| row.get::<_, String>(1))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert!(columns.contains(&"report_id".to_string()));
    assert!(columns.contains(&"manifest_sha256".to_string()));
    assert!(!columns.contains(&"extra".to_string()));
    assert!(!columns.contains(&"manifest_id".to_string()));
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

    #[cfg(feature = "cli")]
    {
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
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_activate_recreates_legacy_active_manifest_schema() {
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
    {
        let conn = Connection::open(&db).unwrap();
        // Legacy pre-schema-v2 registry: activate() no longer supplies
        // manifest_id, so this NOT NULL column must trigger the
        // drop-and-recreate on init instead of failing the INSERT.
        conn.execute_batch(
            "CREATE TABLE active_manifest(
                id INTEGER PRIMARY KEY CHECK(id = 1),
                manifest_id TEXT NOT NULL,
                manifest_path TEXT NOT NULL,
                activated_at TEXT NOT NULL,
                forced INTEGER NOT NULL
            );
            INSERT INTO active_manifest(id, manifest_id, manifest_path, activated_at, forced)
            VALUES(1, 'urn:uuid:legacy', 'legacy-manifest.json', '2026-01-01T00:00:00Z', 0);",
        )
        .unwrap();
    }

    let report = ordvec_manifest::sqlite::activate(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        false,
    )
    .unwrap();
    assert!(report.ok, "{:?}", report.errors);

    // Re-activating against the recreated table must stay idempotent.
    let second = ordvec_manifest::sqlite::activate(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        false,
    )
    .unwrap();
    assert!(second.ok, "{:?}", second.errors);

    let conn = Connection::open(&db).unwrap();
    let columns = {
        let mut stmt = conn.prepare("PRAGMA table_info(active_manifest)").unwrap();
        let columns = stmt
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        columns
    };
    assert_eq!(
        columns,
        vec!["id", "manifest_path", "activated_at", "forced"]
    );
    let (active_path, forced): (String, i64) = conn
        .query_row(
            "SELECT manifest_path, forced FROM active_manifest WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(active_path, manifest_path.display().to_string());
    assert_eq!(forced, 0);
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_combined_verified_load_matrix_respects_limits_paths_and_cache() {
    use rusqlite::Connection;

    let temp = tempfile::tempdir().unwrap();
    let assets = temp.path().join("assets");
    let manifests = temp.path().join("manifests");
    fs::create_dir_all(&assets).unwrap();
    fs::create_dir_all(&manifests).unwrap();

    let index = write_index_kind(&assets, FixtureKind::RankQuant);
    let row_map_path = assets.join("rows.jsonl");
    write_row_map(
        &row_map_path,
        &[
            ("00000000-0000-0000-0000-000000000001", None),
            (
                "00000000-0000-0000-0000-000000000002",
                Some("00000000-0000-0000-0000-000000000001"),
            ),
        ],
    );
    let required_path = assets.join("required-sidecar.json");
    fs::write(&required_path, b"{\"required\":true}\n").unwrap();
    let required_hash = sha256_file(&required_path).unwrap();
    let optional_path = assets.join("optional-sidecar.json");
    fs::write(&optional_path, b"{\"optional\":true}\n").unwrap();
    let optional_hash = sha256_file(&optional_path).unwrap();
    fs::remove_file(&optional_path).unwrap();

    let manifest_path = manifests.join("manifest.json");
    let mut manifest = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::Jsonl(row_map_path.clone()),
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            allow_path_escape: true,
            ..CreateManifestOptions::default()
        },
    )
    .unwrap();
    let profile_path = assets.join("calibration.f64");
    let profile_hash = write_profile(
        &profile_path,
        manifest.artifact.dim * 4 * std::mem::size_of::<f64>(),
    );
    manifest.calibration = Some(weighted_calibration(
        &manifest,
        "../assets/calibration.f64",
        profile_hash,
        CalibrationOrdinalization::Bucket {
            dim: manifest.artifact.dim,
            bits: 2,
        },
        ProfileParameterization::BucketFrequency,
        vec![manifest.artifact.dim, 4],
    ));
    let distortion_path = assets.join("distortion.json");
    let distortion_hash = write_profile(&distortion_path, 128);
    let mut distortion = distortion_profile(
        &manifest,
        Some("../assets/distortion.json".to_string()),
        Some(distortion_hash.clone()),
        DistortionEvidenceKind::EmpiricalSample,
    );
    distortion.calibration_profile_id = manifest
        .calibration
        .as_ref()
        .map(|calibration| calibration.profile_id.clone());
    manifest.encoder_distortion = Some(distortion);
    manifest.auxiliary_artifacts = vec![
        auxiliary_artifact(
            "optional",
            "../assets/optional-sidecar.json",
            optional_hash.clone(),
            false,
        ),
        auxiliary_artifact(
            "required",
            "../assets/required-sidecar.json",
            required_hash.clone(),
            true,
        ),
    ];
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = verify_for_load(&manifest_path, VerifyOptions::default()).unwrap_err();
    let VerifiedLoadPlanError::VerificationFailed(report) = err else {
        panic!("expected path-policy verification failure");
    };
    let codes = error_codes(&report);
    for code in [
        "artifact_path_escape_rejected",
        "row_identity_path_escape_rejected",
        "calibration_profile_path_escape_rejected",
        "encoder_distortion_profile_path_escape_rejected",
        "auxiliary_artifact_path_escape_rejected",
    ] {
        assert!(codes.contains(&code), "missing {code}: {:?}", report.errors);
    }

    let options = VerifyOptions {
        allow_path_escape: true,
        limits: ResourceLimits {
            max_row_identity_rows: 2,
            max_row_identity_jsonl_line_bytes: 512,
            max_auxiliary_artifacts: 2,
            max_auxiliary_artifact_bytes: required_hash.size_bytes.max(optional_hash.size_bytes),
            max_encoder_distortion_profile_bytes: distortion_hash.size_bytes,
            max_report_issues: 64,
            ..ResourceLimits::default()
        },
        ..VerifyOptions::default()
    };
    let plan = verify_for_load(&manifest_path, options.clone()).unwrap();
    assert_eq!(
        plan.artifact_path(),
        fs::canonicalize(&index).unwrap().as_path()
    );
    assert_eq!(
        plan.row_identity().path(),
        Some(fs::canonicalize(&row_map_path).unwrap().as_path())
    );
    assert_eq!(plan.row_identity().validated_rows(), Some(2));
    assert_eq!(plan.auxiliary_artifacts().len(), 2);
    assert_eq!(plan.auxiliary_artifacts()[0].name(), "optional");
    assert_eq!(
        plan.auxiliary_artifacts()[0].state(),
        AuxiliaryArtifactState::OptionalAbsent
    );
    assert_eq!(plan.auxiliary_artifacts()[1].name(), "required");
    assert_eq!(
        plan.auxiliary_artifacts()[1].state(),
        AuxiliaryArtifactState::Verified
    );
    let required_canonical = fs::canonicalize(&required_path).unwrap();
    let required_canonical_display = required_canonical.display().to_string();
    assert_eq!(
        plan.auxiliary_artifacts()[1].path(),
        Some(required_canonical.as_path())
    );
    assert!(!plan.auxiliary_artifacts()[0].required());
    assert_eq!(
        plan.auxiliary_artifacts()[0].reason_code(),
        Some("auxiliary_artifact_optional_absent")
    );
    assert_eq!(plan.auxiliary_artifacts()[0].sha256(), None);
    assert_eq!(plan.auxiliary_artifacts()[0].size_bytes(), None);
    assert!(plan.auxiliary_artifacts()[1].required());
    assert_eq!(plan.auxiliary_artifacts()[1].reason_code(), None);
    assert_eq!(
        plan.auxiliary_artifacts()[1].sha256(),
        Some(required_hash.sha256.as_str())
    );
    assert_eq!(
        plan.auxiliary_artifacts()[1].size_bytes(),
        Some(required_hash.size_bytes)
    );
    let auxiliary_report = &plan.report().auxiliary_artifacts;
    assert_eq!(auxiliary_report.len(), 2);
    assert_eq!(
        auxiliary_report[0].manifest_path,
        "../assets/optional-sidecar.json"
    );
    assert_eq!(
        auxiliary_report[0].expected_sha256.as_deref(),
        Some(optional_hash.sha256.as_str())
    );
    assert_eq!(
        auxiliary_report[0].expected_size_bytes,
        Some(optional_hash.size_bytes)
    );
    assert_eq!(auxiliary_report[0].canonical_path, None);
    assert_eq!(auxiliary_report[0].sha256, None);
    assert_eq!(auxiliary_report[0].size_bytes, None);
    assert_eq!(
        auxiliary_report[0].reason_code.as_deref(),
        Some("auxiliary_artifact_optional_absent")
    );
    assert_eq!(
        auxiliary_report[1].manifest_path,
        "../assets/required-sidecar.json"
    );
    assert_eq!(
        auxiliary_report[1].expected_sha256.as_deref(),
        Some(required_hash.sha256.as_str())
    );
    assert_eq!(
        auxiliary_report[1].expected_size_bytes,
        Some(required_hash.size_bytes)
    );
    assert_eq!(
        auxiliary_report[1].canonical_path.as_deref(),
        Some(required_canonical_display.as_str())
    );
    assert_eq!(
        auxiliary_report[1].sha256.as_deref(),
        Some(required_hash.sha256.as_str())
    );
    assert_eq!(
        auxiliary_report[1].size_bytes,
        Some(required_hash.size_bytes)
    );
    assert_eq!(auxiliary_report[1].reason_code, None);
    assert!(plan.report().calibration.present);
    assert!(plan.report().calibration.profile_sha256.is_some());
    assert!(plan.report().calibration.profile_canonical_path.is_some());
    assert!(plan.report().encoder_distortion.present);
    assert!(plan.report().encoder_distortion.profile_sha256.is_some());
    assert!(plan
        .report()
        .encoder_distortion
        .profile_canonical_path
        .is_some());

    let document = load_manifest_file(&manifest_path).unwrap();
    let document_plan = verify_document_for_load(&document, options.clone()).unwrap();
    assert_eq!(document_plan.artifact_path(), plan.artifact_path());

    let db = temp.path().join("registry.sqlite");
    let first = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        options.clone(),
        true,
    )
    .unwrap();
    assert!(first.ok, "{:?}", first.errors);

    let conn = Connection::open(&db).unwrap();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM verification_reports", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 1);
    let (
        row_identity_sha256,
        calibration_profile_sha256,
        auxiliary_artifacts_sha256,
        encoder_distortion_profile_sha256,
    ): (
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
    ) = conn
        .query_row(
            "SELECT row_identity_sha256,
                    calibration_profile_sha256,
                    auxiliary_artifacts_sha256,
                    encoder_distortion_profile_sha256
             FROM verification_reports
             ORDER BY report_id
             LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert!(row_identity_sha256.is_some());
    assert!(calibration_profile_sha256.is_some());
    assert!(auxiliary_artifacts_sha256.is_some());
    assert!(encoder_distortion_profile_sha256.is_some());

    let second = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        options.clone(),
        true,
    )
    .unwrap();
    assert!(second.ok, "{:?}", second.errors);
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM verification_reports", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 1, "unchanged verification should reuse the cache");

    fs::write(&distortion_path, vec![1u8; 128]).unwrap();
    let drift = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        options,
        true,
    )
    .unwrap();
    assert!(!drift.ok, "distortion drift must force fresh verification");
    assert!(error_codes(&drift).contains(&"encoder_distortion_profile_sha256_mismatch"));
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM verification_reports", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 2);
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

    let limited = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions {
            limits: ResourceLimits {
                max_calibration_profile_bytes: 16,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
        true,
    )
    .unwrap();
    assert!(
        error_codes(&limited).contains(&"calibration_profile_too_large"),
        "{:?}",
        limited.errors
    );

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
fn sqlite_cache_key_is_scoped_to_manifest_location() {
    use rusqlite::Connection;

    let root = tempfile::tempdir().unwrap();
    let case_a = root.path().join("case-a");
    let case_b = root.path().join("case-b");
    fs::create_dir(&case_a).unwrap();
    fs::create_dir(&case_b).unwrap();

    let index_a = write_index(&case_a);
    let manifest_a = case_a.join("manifest.json");
    let manifest = create_manifest_for_index(
        &index_a,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_a,
    )
    .unwrap();
    fs::write(
        &manifest_a,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let index_b = case_b.join("index.ovrq");
    let manifest_b = case_b.join("manifest.json");
    fs::copy(&index_a, &index_b).unwrap();
    fs::copy(&manifest_a, &manifest_b).unwrap();

    let document_a = load_manifest_file(&manifest_a).unwrap();
    let document_b = load_manifest_file(&manifest_b).unwrap();
    let db = root.path().join("registry.sqlite");

    let report_a = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document_a,
        &manifest_a,
        VerifyOptions::default(),
        true,
    )
    .unwrap();
    assert!(report_a.ok, "{:?}", report_a.errors);
    assert_eq!(
        report_a.artifact.canonical_path.as_deref(),
        Some(fs::canonicalize(&index_a).unwrap().to_str().unwrap())
    );

    let report_b = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document_b,
        &manifest_b,
        VerifyOptions::default(),
        true,
    )
    .unwrap();
    assert!(report_b.ok, "{:?}", report_b.errors);
    assert_eq!(
        report_b.artifact.canonical_path.as_deref(),
        Some(fs::canonicalize(&index_b).unwrap().to_str().unwrap())
    );

    let conn = Connection::open(&db).unwrap();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM verification_reports", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(
        count, 2,
        "copied manifests at distinct locations must not reuse canonical-path reports"
    );
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_cache_key_includes_jsonl_row_identity_bytes() {
    use rusqlite::Connection;

    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let rows = temp.path().join("rows.jsonl");
    write_row_map(
        &rows,
        &[
            ("00000000-0000-0000-0000-000000000001", None),
            ("00000000-0000-0000-0000-000000000002", None),
        ],
    );
    let manifest_path = temp.path().join("manifest.json");
    let manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::Jsonl(rows.clone()),
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

    write_row_map(
        &rows,
        &[
            ("00000000-0000-0000-0000-000000000011", None),
            ("00000000-0000-0000-0000-000000000012", None),
        ],
    );
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
        "row identity drift must force fresh verification"
    );
    assert!(error_codes(&cached).contains(&"row_identity_sha256_mismatch"));

    let conn = Connection::open(&db).unwrap();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM verification_reports", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 2, "row-map drift must store a fresh report");
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_cache_key_includes_auxiliary_artifact_bytes() {
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
    let sidecar_path = temp.path().join("sidecar.json");
    fs::write(&sidecar_path, b"{\"version\":1}\n").unwrap();
    let sidecar_hash = sha256_file(&sidecar_path).unwrap();
    manifest.auxiliary_artifacts = vec![auxiliary_artifact(
        "sidecar",
        "sidecar.json",
        sidecar_hash,
        true,
    )];
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

    fs::write(&sidecar_path, b"{\"version\":2}\n").unwrap();
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
        "auxiliary artifact drift must force fresh verification"
    );
    assert!(error_codes(&cached).contains(&"auxiliary_artifact_sha256_mismatch"));
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_cache_key_includes_failed_auxiliary_artifact_observed_bytes() {
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
    let sidecar_path = temp.path().join("sidecar.json");
    fs::write(&sidecar_path, b"{\"version\":1}\n").unwrap();
    let expected_hash = sha256_file(&sidecar_path).unwrap();
    manifest.auxiliary_artifacts = vec![auxiliary_artifact(
        "sidecar",
        "sidecar.json",
        expected_hash,
        true,
    )];
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let document = load_manifest_file(&manifest_path).unwrap();
    let db = temp.path().join("registry.sqlite");

    fs::write(&sidecar_path, b"{\"version\":2}\n").unwrap();
    let first_observed = sha256_file(&sidecar_path).unwrap();
    let report = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        true,
    )
    .unwrap();
    assert!(!report.ok);
    assert_eq!(
        report.auxiliary_artifacts[0].sha256.as_deref(),
        Some(first_observed.sha256.as_str())
    );

    fs::write(&sidecar_path, b"{\"version\":3}\n").unwrap();
    let second_observed = sha256_file(&sidecar_path).unwrap();
    let cached = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        true,
    )
    .unwrap();
    assert!(!cached.ok);
    assert_eq!(
        cached.auxiliary_artifacts[0].sha256.as_deref(),
        Some(second_observed.sha256.as_str())
    );
    assert_ne!(first_observed.sha256, second_observed.sha256);
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_cache_key_distinguishes_optional_auxiliary_absent_and_present() {
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
    let optional_path = temp.path().join("optional.json");
    fs::write(&optional_path, b"{\"enabled\":true}\n").unwrap();
    let optional_hash = sha256_file(&optional_path).unwrap();
    fs::remove_file(&optional_path).unwrap();
    manifest.auxiliary_artifacts = vec![auxiliary_artifact(
        "optional",
        "optional.json",
        optional_hash,
        false,
    )];
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

    assert_eq!(
        report.auxiliary_artifacts[0].state,
        AuxiliaryArtifactState::OptionalAbsent
    );

    fs::write(&optional_path, b"{\"enabled\":true}\n").unwrap();
    let present = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        true,
    )
    .unwrap();
    assert!(present.ok, "{:?}", present.errors);
    assert_eq!(
        present.auxiliary_artifacts[0].state,
        AuxiliaryArtifactState::Verified
    );
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_cache_key_includes_encoder_distortion_profile_bytes() {
    let temp = tempfile::tempdir().unwrap();
    let profile_dir = temp.path().join("profiles");
    fs::create_dir(&profile_dir).unwrap();
    let index = write_index(temp.path());
    let manifest_path = temp.path().join("manifest.json");
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    let profile_path = profile_dir.join("distortion.json");
    let profile_hash = write_profile(&profile_path, 128);
    manifest.encoder_distortion = Some(distortion_profile(
        &manifest,
        Some("profiles/distortion.json".to_string()),
        Some(profile_hash),
        DistortionEvidenceKind::EmpiricalSample,
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

    fs::write(&profile_path, vec![1u8; 128]).unwrap();
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
        "distortion profile drift must force fresh verification"
    );
    assert!(error_codes(&cached).contains(&"encoder_distortion_profile_sha256_mismatch"));
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_cache_key_does_not_hash_oversized_encoder_distortion_profile() {
    let temp = tempfile::tempdir().unwrap();
    let profile_dir = temp.path().join("profiles");
    fs::create_dir(&profile_dir).unwrap();
    let index = write_index(temp.path());
    let manifest_path = temp.path().join("manifest.json");
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    let profile_path = profile_dir.join("distortion.json");
    let profile_hash = write_profile(&profile_path, 128);
    manifest.encoder_distortion = Some(distortion_profile(
        &manifest,
        Some("profiles/distortion.json".to_string()),
        Some(profile_hash),
        DistortionEvidenceKind::EmpiricalSample,
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

    let limited = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions {
            limits: ResourceLimits {
                max_encoder_distortion_profile_bytes: 16,
                ..ResourceLimits::default()
            },
            ..VerifyOptions::default()
        },
        true,
    )
    .unwrap();
    assert!(
        error_codes(&limited).contains(&"encoder_distortion_profile_too_large"),
        "{:?}",
        limited.errors
    );
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

#[test]
fn grown_profiles_fail_fast_at_declared_size_under_default_limits() {
    // Derived-limits regression coverage for the two profile call sites:
    // a profile grown past its manifest-declared size must fail fast with
    // the *_too_large code at DEFAULT options (bound = declared size), not
    // be hashed in full and reported as a digest mismatch.
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

    let calibration_path = profile_dir.join("profile.f64");
    let calibration_hash = write_profile(
        &calibration_path,
        manifest.artifact.dim * std::mem::size_of::<f64>(),
    );
    manifest.calibration = Some(weighted_calibration(
        &manifest,
        "profiles/profile.f64",
        calibration_hash,
        CalibrationOrdinalization::TopK {
            dim: manifest.artifact.dim,
            k: 16,
        },
        ProfileParameterization::MarginalTopKFrequency,
        vec![manifest.artifact.dim],
    ));

    let distortion_path = profile_dir.join("distortion.json");
    let distortion_hash = write_profile(&distortion_path, 128);
    manifest.encoder_distortion = Some(distortion_profile(
        &manifest,
        Some("profiles/distortion.json".to_string()),
        Some(distortion_hash),
        DistortionEvidenceKind::EmpiricalSample,
    ));

    let report = verify_manifest_with_base(manifest.clone(), case.path(), VerifyOptions::default());
    assert!(report.ok, "{:?}", report.errors);

    // Grow both profile files past their declarations.
    for path in [&calibration_path, &distortion_path] {
        let mut file = fs::OpenOptions::new().append(true).open(path).unwrap();
        file.write_all(&[0u8; 64]).unwrap();
    }

    let report = verify_manifest_with_base(manifest, case.path(), VerifyOptions::default());
    let codes = error_codes(&report);
    assert!(
        codes.contains(&"calibration_profile_too_large"),
        "grown calibration profile must fail the declared-size bound, got {codes:?}",
    );
    assert!(
        codes.contains(&"encoder_distortion_profile_too_large"),
        "grown encoder distortion profile must fail the declared-size bound, got {codes:?}",
    );
}
