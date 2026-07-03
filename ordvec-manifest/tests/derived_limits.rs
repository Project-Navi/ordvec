//! Derived artifact size bounds: create bounds reads by the artifact's own
//! observed size, verify bounds reads by the manifest-declared size. The flat
//! `ResourceLimits` byte caps remain enforceable as explicit opt-in ceilings
//! but no longer reject large legitimate artifacts by default.

use ordvec::RankQuant;
use ordvec_manifest::{
    create_manifest_for_index, create_manifest_for_index_with_options, verify_manifest_with_base,
    CreateAuxiliaryArtifact, CreateManifestOptions, CreateRowIdentity, VerificationReport,
    VerifyOptions,
};
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

const LEGACY_AUX_CAP: u64 = 64 * 1024 * 1024;

fn write_index(dir: &Path) -> PathBuf {
    let path = dir.join("index.ovrq");
    let mut index = RankQuant::new(16, 2);
    let docs: Vec<f32> = (0..32).map(|i| i as f32 - 12.0).collect();
    index.add(&docs);
    index.write(&path).unwrap();
    path
}

fn error_codes(report: &VerificationReport) -> Vec<&str> {
    report
        .errors
        .iter()
        .map(|issue| issue.code.as_str())
        .collect()
}

fn create_with_aux(dir: &Path, aux_path: &Path) -> (ordvec_manifest::IndexManifest, PathBuf) {
    let index = write_index(dir);
    let manifest_path = dir.join("manifest.json");
    let manifest = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            auxiliary_artifacts: vec![CreateAuxiliaryArtifact {
                name: "sidecar".to_string(),
                path: aux_path.to_path_buf(),
                required: true,
            }],
            ..CreateManifestOptions::default()
        },
    )
    .unwrap();
    (manifest, manifest_path)
}

/// Default options must accept auxiliary artifacts larger than the legacy
/// 64 MiB flat cap, end to end: create records the artifact, verify passes.
/// (A 1.26M-row dim=1024 sign sidecar is ~161 MB; the default cap made such
/// bundles impossible to write.)
#[test]
fn default_limits_accept_aux_artifact_above_legacy_cap() {
    let temp = tempfile::tempdir().unwrap();
    let aux_path = temp.path().join("sidecar.bin");
    let aux_len = LEGACY_AUX_CAP + 4096;
    let file = fs::File::create(&aux_path).unwrap();
    file.set_len(aux_len).unwrap();
    drop(file);

    let (manifest, _) = create_with_aux(temp.path(), &aux_path);
    assert_eq!(manifest.auxiliary_artifacts.len(), 1);
    assert_eq!(manifest.auxiliary_artifacts[0].file_size_bytes, aux_len);

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert_eq!(
        error_codes(&report),
        Vec::<&str>::new(),
        "expected clean verification for a {aux_len}-byte auxiliary artifact under defaults",
    );
}

/// An auxiliary artifact that grew after manifest creation must be rejected
/// by the declared-size read bound (fail-fast, without hashing the excess),
/// keeping the established `auxiliary_artifact_file_too_large` reason code.
#[test]
fn verify_bounds_aux_read_by_declared_size_when_grown() {
    let temp = tempfile::tempdir().unwrap();
    let aux_path = temp.path().join("sidecar.bin");
    fs::write(&aux_path, vec![7u8; 8192]).unwrap();

    let (manifest, _) = create_with_aux(temp.path(), &aux_path);

    let mut file = OpenOptions::new().append(true).open(&aux_path).unwrap();
    file.write_all(&[7u8; 4096]).unwrap();
    drop(file);

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(
        error_codes(&report).contains(&"auxiliary_artifact_file_too_large"),
        "grown artifact must fail the declared-size bound, got {:?}",
        error_codes(&report),
    );
    assert_eq!(
        report.auxiliary_artifacts[0].reason_code.as_deref(),
        Some("auxiliary_artifact_file_too_large"),
    );
}

/// Regression guard: a truncated auxiliary artifact still fails verification
/// (size mismatch below the declared bound; the bound itself must not
/// misclassify a smaller-than-declared file).
#[test]
fn verify_rejects_truncated_aux_artifact() {
    let temp = tempfile::tempdir().unwrap();
    let aux_path = temp.path().join("sidecar.bin");
    fs::write(&aux_path, vec![7u8; 8192]).unwrap();

    let (manifest, _) = create_with_aux(temp.path(), &aux_path);
    let file = OpenOptions::new().write(true).open(&aux_path).unwrap();
    file.set_len(4096).unwrap();
    drop(file);

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(
        error_codes(&report).contains(&"auxiliary_artifact_file_size_mismatch"),
        "truncated artifact must fail size equality, got {:?}",
        error_codes(&report),
    );
}

/// Regression guard: a manifest whose declared auxiliary size was inflated
/// (bytes on disk unchanged) still fails the size-equality check even though
/// the SHA-256 matches.
#[test]
fn verify_rejects_inflated_declared_aux_size() {
    let temp = tempfile::tempdir().unwrap();
    let aux_path = temp.path().join("sidecar.bin");
    fs::write(&aux_path, vec![7u8; 8192]).unwrap();

    let (mut manifest, _) = create_with_aux(temp.path(), &aux_path);
    manifest.auxiliary_artifacts[0].file_size_bytes = 1 << 30;

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(
        error_codes(&report).contains(&"auxiliary_artifact_file_size_mismatch"),
        "inflated declaration must fail size equality, got {:?}",
        error_codes(&report),
    );
}

/// An explicitly configured flat cap remains an enforceable ceiling on
/// verify even when the declared size is within bounds.
#[test]
fn explicit_flat_cap_still_enforced_on_verify() {
    let temp = tempfile::tempdir().unwrap();
    let aux_path = temp.path().join("sidecar.bin");
    fs::write(&aux_path, vec![7u8; 8192]).unwrap();

    let (manifest, _) = create_with_aux(temp.path(), &aux_path);
    let mut options = VerifyOptions::default();
    options.limits.max_auxiliary_artifact_bytes = 4096;

    let report = verify_manifest_with_base(manifest, temp.path(), options);
    assert!(
        error_codes(&report).contains(&"auxiliary_artifact_file_too_large"),
        "explicit tight cap must still reject, got {:?}",
        error_codes(&report),
    );
}

/// The primary index artifact gains a declared-size read bound: a primary
/// artifact that grew after manifest creation fails fast with a dedicated
/// reason code instead of being hashed in full.
#[test]
fn verify_bounds_primary_read_by_declared_size_when_grown() {
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

    let mut file = OpenOptions::new().append(true).open(&index).unwrap();
    file.write_all(&[0u8; 4096]).unwrap();
    drop(file);

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(
        error_codes(&report).contains(&"artifact_file_too_large"),
        "grown primary artifact must fail the declared-size bound, got {:?}",
        error_codes(&report),
    );
}

/// The primary index artifact honors an explicitly configured opt-in
/// ceiling, mirroring the auxiliary/profile artifact classes (CIPHER-02).
#[test]
fn explicit_index_ceiling_enforced_on_primary() {
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

    let mut options = VerifyOptions::default();
    options.limits.max_index_artifact_bytes = 8;

    let report = verify_manifest_with_base(manifest, temp.path(), options);
    assert!(
        error_codes(&report).contains(&"artifact_file_too_large"),
        "explicit index ceiling must reject, got {:?}",
        error_codes(&report),
    );
}
