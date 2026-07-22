use ordvec::RankQuant;
use ordvec_manifest::{
    create_manifest_for_index, create_manifest_for_index_with_options, load_manifest_file,
    sha256_file, verify_manifest_with_base, write_manifest_file, CreateAuxiliaryArtifact,
    CreateManifestOptions, CreateRowIdentity, ManifestError, VerifyOptions, SCHEMA_VERSION,
};
use serde_json::{json, Map, Value};
use std::fs;
use std::path::{Path, PathBuf};

fn write_index(dir: &Path) -> PathBuf {
    let path = dir.join("index.ovrq");
    let mut index = RankQuant::new(16, 2);
    let docs: Vec<f32> = (0..32).map(|i| i as f32 - 12.0).collect();
    index.add(&docs);
    index.write(&path).unwrap();
    path
}

fn aux_input(dir: &Path, name: &str, contents: &[u8]) -> CreateAuxiliaryArtifact {
    let path = dir.join(name);
    fs::write(&path, contents).unwrap();
    CreateAuxiliaryArtifact {
        name: name.to_string(),
        path,
        required: true,
    }
}

/// Builds the fixed synthetic bundle used by the determinism tests and
/// returns the serialized manifest bytes.
fn build_manifest_bytes(dir: &Path, aux_names: &[&str]) -> Vec<u8> {
    let index = write_index(dir);
    let manifest_path = dir.join("manifest.json");
    let auxiliary_artifacts = aux_names
        .iter()
        .map(|name| aux_input(dir, name, name.as_bytes()))
        .collect();
    let manifest = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            auxiliary_artifacts,
            ..CreateManifestOptions::default()
        },
    )
    .unwrap();
    write_manifest_file(&manifest, &manifest_path).unwrap();
    fs::read(&manifest_path).unwrap()
}

#[test]
fn identical_inputs_produce_byte_identical_manifests() {
    let temp_a = tempfile::tempdir().unwrap();
    let temp_b = tempfile::tempdir().unwrap();
    let bytes_a = build_manifest_bytes(temp_a.path(), &["aux-a.bin", "aux-b.bin"]);
    let bytes_b = build_manifest_bytes(temp_b.path(), &["aux-a.bin", "aux-b.bin"]);
    assert_eq!(bytes_a, bytes_b);
    assert_eq!(
        sha256_file(temp_a.path().join("manifest.json"))
            .unwrap()
            .sha256,
        sha256_file(temp_b.path().join("manifest.json"))
            .unwrap()
            .sha256,
    );
}

#[test]
fn manifest_bytes_match_checked_in_golden() {
    let temp = tempfile::tempdir().unwrap();
    let bytes = build_manifest_bytes(temp.path(), &["aux-a.bin", "aux-b.bin"]);
    let golden = include_bytes!("golden/manifest.v2.json");
    assert_eq!(
        bytes, golden,
        "checked-in golden manifest bytes changed. The canonical byte form is \
         the bundle's content address, so this is deliberate only if you changed \
         the manifest serializer (a schema-version event) or the ordvec index \
         encoding the fixture embeds (an .ovrq format_version event). If instead \
         an editor reflowed or newline-normalized golden/manifest.v2.json, revert \
         that — the fixture is intentionally byte-exact with no trailing newline."
    );
}

#[test]
fn manifest_bytes_change_when_artifact_content_changes() {
    let temp_a = tempfile::tempdir().unwrap();
    let temp_b = tempfile::tempdir().unwrap();
    let bytes_a = build_manifest_bytes(temp_a.path(), &[]);

    let index = temp_b.path().join("index.ovrq");
    let mut altered = RankQuant::new(16, 2);
    // Different rank order per vector, so the encoded index bytes (and the
    // manifest-embedded sha256) actually change.
    let docs: Vec<f32> = (0..32).map(|i| ((i * 17) % 31) as f32).collect();
    altered.add(&docs);
    altered.write(&index).unwrap();
    let manifest_path = temp_b.path().join("manifest.json");
    let manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    write_manifest_file(&manifest, &manifest_path).unwrap();
    let bytes_b = fs::read(&manifest_path).unwrap();

    assert_ne!(bytes_a, bytes_b);
}

#[test]
fn manifest_bytes_change_when_auxiliary_entry_added_or_removed() {
    let temp_a = tempfile::tempdir().unwrap();
    let temp_b = tempfile::tempdir().unwrap();
    let temp_c = tempfile::tempdir().unwrap();
    let without_aux = build_manifest_bytes(temp_a.path(), &[]);
    let with_one = build_manifest_bytes(temp_b.path(), &["aux-a.bin"]);
    let with_two = build_manifest_bytes(temp_c.path(), &["aux-a.bin", "aux-b.bin"]);
    assert_ne!(without_aux, with_one);
    assert_ne!(with_one, with_two);
}

#[test]
fn auxiliary_declaration_order_does_not_change_manifest_bytes() {
    let temp_a = tempfile::tempdir().unwrap();
    let temp_b = tempfile::tempdir().unwrap();
    let bytes_a = build_manifest_bytes(temp_a.path(), &["aux-a.bin", "aux-b.bin"]);
    let bytes_b = build_manifest_bytes(temp_b.path(), &["aux-b.bin", "aux-a.bin"]);
    assert_eq!(bytes_a, bytes_b);
}

#[test]
fn nested_json_order_and_number_spelling_do_not_change_manifest_bytes() {
    let temp_a = tempfile::tempdir().unwrap();
    let temp_b = tempfile::tempdir().unwrap();
    let index_a = write_index(temp_a.path());
    let index_b = write_index(temp_b.path());
    let path_a = temp_a.path().join("manifest.json");
    let path_b = temp_b.path().join("manifest.json");
    let mut manifest_a = create_manifest_for_index(
        &index_a,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &path_a,
    )
    .unwrap();
    let mut manifest_b = create_manifest_for_index(
        &index_b,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &path_b,
    )
    .unwrap();

    let mut descending = Map::new();
    descending.insert("z".to_string(), serde_json::from_str("1e0").unwrap());
    descending.insert("a".to_string(), json!({"y": true, "b": false}));
    let mut ascending = Map::new();
    ascending.insert("a".to_string(), json!({"b": false, "y": true}));
    ascending.insert("z".to_string(), serde_json::from_str("1.0").unwrap());
    manifest_a
        .extensions
        .insert("nested".to_string(), Value::Object(descending));
    manifest_b
        .extensions
        .insert("nested".to_string(), Value::Object(ascending));

    write_manifest_file(&manifest_a, &path_a).unwrap();
    write_manifest_file(&manifest_b, &path_b).unwrap();
    assert_eq!(fs::read(path_a).unwrap(), fs::read(path_b).unwrap());
}

#[test]
fn arbitrary_precision_numbers_that_would_alias_are_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    for (case, number) in [
        ("first", "18446744073709551616"),
        ("second", "18446744073709551617"),
        ("decimal", "0.100000000000000005"),
    ] {
        let manifest_path = temp.path().join(format!("{case}.json"));
        let mut manifest = create_manifest_for_index(
            &index,
            CreateRowIdentity::RowIdIdentity,
            "test-embedding",
            &manifest_path,
        )
        .unwrap();
        manifest
            .extensions
            .insert("precise".to_string(), serde_json::from_str(number).unwrap());
        let err = write_manifest_file(&manifest, &manifest_path).unwrap_err();
        assert!(
            err.to_string().contains("outside the exact canonical"),
            "{number}: {err}"
        );
        assert!(!manifest_path.exists());
    }
}

#[test]
fn manifest_write_atomically_replaces_the_destination_inode() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let manifest_path = temp.path().join("manifest.json");
    let old_inode_link = temp.path().join("old-manifest-inode");
    let manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();

    fs::write(&manifest_path, b"old manifest bytes").unwrap();
    fs::hard_link(&manifest_path, &old_inode_link).unwrap();
    write_manifest_file(&manifest, &manifest_path).unwrap();

    assert_eq!(fs::read(old_inode_link).unwrap(), b"old manifest bytes");
    assert_eq!(
        load_manifest_file(manifest_path)
            .unwrap()
            .manifest
            .schema_version,
        SCHEMA_VERSION
    );
}

#[cfg(unix)]
#[test]
fn manifest_write_matches_create_mode_and_preserves_existing_mode() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let manifest_path = temp.path().join("manifest.json");
    let control_path = temp.path().join("create-mode-control");
    fs::File::create(&control_path).unwrap();
    let create_mode = fs::metadata(&control_path).unwrap().permissions().mode() & 0o777;
    let manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();

    write_manifest_file(&manifest, &manifest_path).unwrap();
    assert_eq!(
        fs::metadata(&manifest_path).unwrap().permissions().mode() & 0o777,
        create_mode
    );

    fs::set_permissions(&manifest_path, fs::Permissions::from_mode(0o640)).unwrap();
    write_manifest_file(&manifest, &manifest_path).unwrap();
    assert_eq!(
        fs::metadata(&manifest_path).unwrap().permissions().mode() & 0o777,
        0o640
    );
}

#[cfg(unix)]
#[test]
fn manifest_write_opens_parent_before_replacing_destination() {
    use std::os::unix::fs::PermissionsExt;

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
    fs::write(&manifest_path, b"original manifest").unwrap();

    let original_permissions = fs::metadata(temp.path()).unwrap().permissions();
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o300)).unwrap();
    let result = write_manifest_file(&manifest, &manifest_path);
    fs::set_permissions(temp.path(), original_permissions).unwrap();

    match result.unwrap_err() {
        ManifestError::Io(error) => assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied),
        error => panic!("expected parent-directory permission error, got {error}"),
    }
    assert_eq!(fs::read(manifest_path).unwrap(), b"original manifest");
}

#[cfg(unix)]
#[test]
fn manifest_write_does_not_replace_a_symlink_destination() {
    use std::os::unix::fs::symlink;

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
    symlink("missing-manifest-target", &manifest_path).unwrap();

    assert!(write_manifest_file(&manifest, &manifest_path).is_err());
    assert!(fs::symlink_metadata(manifest_path)
        .unwrap()
        .file_type()
        .is_symlink());
}

#[test]
fn old_schema_manifest_fails_with_clear_schema_version_error() {
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
    let mut value = serde_json::to_value(&manifest).unwrap();
    let object = value.as_object_mut().unwrap();
    object.insert(
        "schema_version".to_string(),
        json!("ordvec.index_manifest.v1"),
    );
    object.insert(
        "manifest_id".to_string(),
        json!("urn:uuid:11111111-1111-4111-8111-111111111111"),
    );
    object.insert("created_at".to_string(), json!("2026-06-09T00:00:00Z"));
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&value).unwrap(),
    )
    .unwrap();

    let message = load_manifest_file(&manifest_path).unwrap_err().to_string();
    assert!(message.contains("ordvec.index_manifest.v1"), "{message}");
    assert!(message.contains(SCHEMA_VERSION), "{message}");
    assert!(message.contains("older or newer"), "{message}");
}

#[test]
fn current_shape_with_wrong_schema_version_fails_at_load() {
    // A document that is otherwise valid v2 but claims an unsupported
    // schema_version has no unknown fields, so `deny_unknown_fields` accepts
    // it — the loader must reject it on the version alone, not defer to verify.
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
    let mut value = serde_json::to_value(&manifest).unwrap();
    value.as_object_mut().unwrap().insert(
        "schema_version".to_string(),
        json!("ordvec.index_manifest.v1"),
    );
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&value).unwrap(),
    )
    .unwrap();

    let message = load_manifest_file(&manifest_path).unwrap_err().to_string();
    assert!(message.contains("ordvec.index_manifest.v1"), "{message}");
    assert!(message.contains(SCHEMA_VERSION), "{message}");
    assert!(message.contains("older or newer"), "{message}");
}

#[test]
fn unknown_fields_on_current_schema_keep_the_parse_error() {
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
    let mut value = serde_json::to_value(&manifest).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("unknown".to_string(), json!(true));
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&value).unwrap(),
    )
    .unwrap();

    let message = load_manifest_file(&manifest_path).unwrap_err().to_string();
    assert!(message.contains("unknown"), "{message}");
    assert!(!message.contains("older or newer"), "{message}");
}

#[test]
fn non_canonical_manifest_paths_are_rejected_at_validation() {
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

    let mut dotted = manifest.clone();
    dotted.artifact.path = "./index.ovrq".to_string();
    let report = verify_manifest_with_base(dotted, temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_path_not_canonical"));

    let mut backslashed = manifest;
    backslashed.artifact.path = "sub\\index.ovrq".to_string();
    let report = verify_manifest_with_base(backslashed, temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_path_not_canonical"));
}

#[test]
fn contained_parent_dir_segments_are_not_canonical_by_default() {
    let temp = tempfile::tempdir().unwrap();
    write_index(temp.path());
    fs::create_dir(temp.path().join("a")).unwrap();
    let manifest_path = temp.path().join("manifest.json");
    let manifest = create_manifest_for_index(
        temp.path().join("index.ovrq"),
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();

    // `a/../index.ovrq` resolves to the same file as `index.ovrq` without
    // ever escaping the bundle, so it slips past escape/containment checks;
    // canonicality must reject it or one bundle has many verified identities.
    let mut aliased = manifest;
    aliased.artifact.path = "a/../index.ovrq".to_string();
    let report = verify_manifest_with_base(aliased.clone(), temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_path_not_canonical"));

    // `..` segments remain available under the explicit escape policy.
    let report = verify_manifest_with_base(
        aliased,
        temp.path(),
        VerifyOptions {
            allow_path_escape: true,
            ..VerifyOptions::default()
        },
    );
    assert!(report.ok, "{:?}", report.errors);
}

#[test]
fn absolute_path_strings_are_policy_governed_not_canonicality_errors() {
    let temp = tempfile::tempdir().unwrap();
    write_index(temp.path());
    let manifest_path = temp.path().join("manifest.json");
    let manifest = create_manifest_for_index(
        temp.path().join("index.ovrq"),
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();

    // Windows-style absolute strings must fall to the allow_absolute_paths
    // policy at resolution, not to the canonical-form check, so the retained
    // absolute-path opt-in keeps working across platforms.
    for absolute in ["C:/bundles/index.ovrq", "//?/C:/bundles/index.ovrq"] {
        let mut manifest = manifest.clone();
        manifest.artifact.path = absolute.to_string();
        let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
        assert!(!report.ok);
        assert!(
            report
                .errors
                .iter()
                .all(|issue| issue.code != "artifact_path_not_canonical"),
            "{absolute} must be governed by path policy, got {:?}",
            report.errors
        );
    }
}

#[cfg(unix)]
#[test]
fn create_rejects_paths_it_cannot_embed_canonically() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let manifest_path = temp.path().join("manifest.json");
    // A legal Unix filename containing a backslash cannot be embedded without
    // aliasing the manifest path separator; creation must fail instead of
    // minting a manifest that fails its own default verification.
    let aux = aux_input(temp.path(), "back\\slash.bin", b"aux");
    let err = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            auxiliary_artifacts: vec![aux],
            ..CreateManifestOptions::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("cannot be embedded"), "{err}");
}

#[cfg(target_os = "linux")]
#[test]
fn create_rejects_non_utf8_paths_instead_of_embedding_them_lossily() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let manifest_path = temp.path().join("manifest.json");
    let non_utf8 = temp
        .path()
        .join(OsString::from_vec(b"aux-\xff.bin".to_vec()));
    fs::write(&non_utf8, b"aux").unwrap();
    let err = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            auxiliary_artifacts: vec![CreateAuxiliaryArtifact {
                name: "aux".to_string(),
                path: non_utf8,
                required: true,
            }],
            ..CreateManifestOptions::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("not valid UTF-8"), "{err}");
}
