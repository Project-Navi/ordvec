use ordvec::RankQuant;
#[cfg(any(unix, windows))]
use ordvec_manifest::VerifiedArtifactTypeRejection;
use ordvec_manifest::{
    create_manifest_for_index_with_options, parse_current_manifest_bytes,
    read_manifest_bytes_bounded, verify_for_load, write_manifest_file, AuxiliaryArtifactState,
    CreateAuxiliaryArtifact, CreateManifestOptions, CreateRowIdentity, ManifestIndexKind,
    VerifiedArtifactChange, VerifiedArtifactUseError, VerifiedLoadPlan, VerifyOptions,
};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use tempfile::TempDir;

struct Fixture {
    _dir: TempDir,
    manifest_path: PathBuf,
    index_path: PathBuf,
    auxiliary_path: PathBuf,
    plan: VerifiedLoadPlan,
}

fn fixture(optional_auxiliary_absent: bool) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let index_path = dir.path().join("index.ovrq");
    let auxiliary_path = dir.path().join("ids.bin");
    let manifest_path = dir.path().join("index.manifest.json");

    let mut index = RankQuant::new(8, 2);
    index.add(&[8.0, 3.0, 6.0, 1.0, 7.0, 2.0, 5.0, 4.0]);
    index.write(&index_path).unwrap();
    fs::write(&auxiliary_path, b"final auxiliary representation").unwrap();

    let manifest = create_manifest_for_index_with_options(
        &index_path,
        CreateRowIdentity::RowIdIdentity,
        "fixture-model",
        &manifest_path,
        CreateManifestOptions {
            auxiliary_artifacts: vec![CreateAuxiliaryArtifact {
                name: "ids".to_string(),
                path: auxiliary_path.clone(),
                required: false,
            }],
            ..CreateManifestOptions::default()
        },
    )
    .unwrap();
    write_manifest_file(&manifest, &manifest_path).unwrap();
    if optional_auxiliary_absent {
        fs::remove_file(&auxiliary_path).unwrap();
    }
    let plan = verify_for_load(&manifest_path, VerifyOptions::default()).unwrap();

    Fixture {
        _dir: dir,
        manifest_path,
        index_path,
        auxiliary_path,
        plan,
    }
}

#[test]
fn public_manifest_byte_reader_and_current_parser_share_the_strict_loader_path() {
    let fixture = fixture(false);
    let bytes = read_manifest_bytes_bounded(&fixture.manifest_path, 1024 * 1024).unwrap();
    let parsed = parse_current_manifest_bytes(&bytes).unwrap();
    assert_eq!(parsed.schema_version, ordvec_manifest::SCHEMA_VERSION);

    let err =
        read_manifest_bytes_bounded(&fixture.manifest_path, bytes.len() as u64 - 1).unwrap_err();
    assert_eq!(err.code(), Some("manifest_file_too_large"));
}

#[test]
fn primary_and_auxiliary_decode_from_plan_verified_forward_only_readers() {
    let fixture = fixture(false);
    let decoded = fixture
        .plan
        .decode_primary_with(ManifestIndexKind::RankQuant, |reader, encoded_len| {
            RankQuant::read_from_sized(reader, encoded_len)
        })
        .unwrap();
    assert_eq!(decoded.len(), 1);
    assert_eq!(
        fixture.plan.primary_identity().expected_size_bytes,
        fs::metadata(&fixture.index_path).unwrap().len()
    );

    let auxiliary = fixture.plan.require_auxiliary("ids").unwrap();
    assert_eq!(
        auxiliary,
        fs::canonicalize(&fixture.auxiliary_path).unwrap()
    );
    let decoded = fixture
        .plan
        .auxiliary_by_name("ids")
        .unwrap()
        .decode_verified_with(|reader, _| {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes)?;
            Ok::<_, io::Error>(bytes)
        })
        .unwrap();
    assert_eq!(decoded, b"final auxiliary representation");
}

#[test]
fn primary_kind_mismatch_is_typed_and_does_not_open_the_file() {
    let fixture = fixture(false);
    fs::remove_file(&fixture.index_path).unwrap();
    let err = fixture
        .plan
        .decode_primary_with(ManifestIndexKind::SignBitmap, |_, _| Ok::<_, io::Error>(()))
        .unwrap_err();
    assert!(matches!(
        err,
        VerifiedArtifactUseError::KindMismatch {
            expected: ManifestIndexKind::SignBitmap,
            observed: ManifestIndexKind::RankQuant,
            ..
        }
    ));
}

#[test]
fn intact_early_parser_failure_wins_after_the_remainder_is_drained() {
    let fixture = fixture(false);
    let err = fixture
        .plan
        .decode_primary_with(ManifestIndexKind::RankQuant, |reader, _| {
            let mut first = [0u8; 1];
            reader.read_exact(&mut first)?;
            Err::<(), _>(io::Error::new(
                io::ErrorKind::InvalidData,
                "deliberate early parser failure",
            ))
        })
        .unwrap_err();
    match err {
        VerifiedArtifactUseError::Decoder { source, .. } => {
            assert_eq!(source.kind(), io::ErrorKind::InvalidData);
            assert!(source
                .to_string()
                .contains("deliberate early parser failure"));
        }
        other => panic!("expected decoder error, got {other:?}"),
    }
}

#[test]
fn stale_digest_wins_over_an_early_parser_failure() {
    let fixture = fixture(false);
    let mut bytes = fs::read(&fixture.index_path).unwrap();
    *bytes.last_mut().unwrap() ^= 0x01;
    fs::write(&fixture.index_path, bytes).unwrap();

    let err = fixture
        .plan
        .decode_primary_with(ManifestIndexKind::RankQuant, |reader, _| {
            let mut first = [0u8; 1];
            reader.read_exact(&mut first)?;
            Err::<(), _>(io::Error::new(
                io::ErrorKind::InvalidData,
                "deliberate early parser failure",
            ))
        })
        .unwrap_err();
    assert!(matches!(
        err,
        VerifiedArtifactUseError::Stale { ref changes, .. }
            if changes.iter().any(|change| matches!(change, VerifiedArtifactChange::Digest { .. }))
    ));
}

#[test]
fn initial_size_change_is_stale_before_decoder_invocation() {
    let fixture = fixture(false);
    OpenOptions::new()
        .append(true)
        .open(&fixture.index_path)
        .unwrap()
        .write_all(b"growth")
        .unwrap();
    let decoder_called = std::cell::Cell::new(false);
    let err = fixture
        .plan
        .decode_primary_with(ManifestIndexKind::RankQuant, |_, _| {
            decoder_called.set(true);
            Ok::<_, io::Error>(())
        })
        .unwrap_err();
    assert!(!decoder_called.get());
    assert!(matches!(
        err,
        VerifiedArtifactUseError::Stale { ref changes, .. }
            if matches!(changes.as_slice(), [VerifiedArtifactChange::InitialSize { .. }])
    ));
}

#[test]
fn initial_truncation_is_stale_before_decoder_invocation() {
    let fixture = fixture(false);
    let mut bytes = fs::read(&fixture.index_path).unwrap();
    bytes.pop().unwrap();
    fs::write(&fixture.index_path, bytes).unwrap();
    let decoder_called = std::cell::Cell::new(false);
    let err = fixture
        .plan
        .decode_primary_with(ManifestIndexKind::RankQuant, |_, _| {
            decoder_called.set(true);
            Ok::<_, io::Error>(())
        })
        .unwrap_err();
    assert!(!decoder_called.get());
    assert!(matches!(
        err,
        VerifiedArtifactUseError::Stale { ref changes, .. }
            if matches!(changes.as_slice(), [VerifiedArtifactChange::InitialSize { expected, observed }] if observed + 1 == *expected)
    ));
}

#[test]
fn missing_artifact_is_a_typed_open_access_failure() {
    let fixture = fixture(false);
    fs::remove_file(&fixture.index_path).unwrap();
    let err = fixture
        .plan
        .decode_primary_with(ManifestIndexKind::RankQuant, |_, _| Ok::<_, io::Error>(()))
        .unwrap_err();
    assert!(matches!(
        err,
        VerifiedArtifactUseError::Access {
            stage: ordvec_manifest::VerifiedArtifactAccessStage::Open,
            ..
        }
    ));
}

#[test]
fn final_descriptor_growth_is_stale_without_reading_past_the_plan_boundary() {
    let fixture = fixture(false);
    let path = fixture.index_path.clone();
    let err = fixture
        .plan
        .decode_primary_with(ManifestIndexKind::RankQuant, move |reader, _| {
            let mut first = [0u8; 1];
            reader.read_exact(&mut first)?;
            OpenOptions::new()
                .append(true)
                .open(path)?
                .write_all(b"growth")?;
            Err::<(), _>(io::Error::new(
                io::ErrorKind::InvalidData,
                "deliberate early parser failure",
            ))
        })
        .unwrap_err();
    assert!(matches!(
        err,
        VerifiedArtifactUseError::Stale { ref changes, .. }
            if changes.iter().any(|change| matches!(change, VerifiedArtifactChange::FinalSize { .. }))
    ));
}

#[test]
fn successful_decoder_must_consume_the_entire_declared_artifact() {
    let fixture = fixture(false);
    let err = fixture
        .plan
        .decode_primary_with(ManifestIndexKind::RankQuant, |reader, _| {
            let mut first = [0u8; 1];
            reader.read_exact(&mut first)?;
            Ok::<_, io::Error>(first[0])
        })
        .unwrap_err();
    assert!(matches!(
        err,
        VerifiedArtifactUseError::IncompleteConsumption {
            consumed: 1,
            expected,
            ..
        } if expected > 1
    ));
}

#[test]
fn optional_absent_auxiliary_is_a_typed_state() {
    let fixture = fixture(true);
    let auxiliary = fixture.plan.auxiliary_by_name("ids").unwrap();
    assert_eq!(auxiliary.state(), AuxiliaryArtifactState::OptionalAbsent);
    let err = auxiliary
        .decode_verified_with(|_, _| Ok::<_, io::Error>(()))
        .unwrap_err();
    assert!(matches!(
        err,
        VerifiedArtifactUseError::OptionalAbsent { ref name } if name == "ids"
    ));
}

#[cfg(unix)]
#[test]
fn final_symlink_is_rejected_by_manifest_and_artifact_openers() {
    use std::os::unix::fs::symlink;

    let fixture = fixture(false);
    let saved_index = fixture.index_path.with_extension("saved");
    fs::rename(&fixture.index_path, &saved_index).unwrap();
    symlink(&saved_index, &fixture.index_path).unwrap();
    let err = fixture
        .plan
        .decode_primary_with(ManifestIndexKind::RankQuant, |_, _| Ok::<_, io::Error>(()))
        .unwrap_err();
    assert!(matches!(
        err,
        VerifiedArtifactUseError::TypeRejected {
            rejection: VerifiedArtifactTypeRejection::FinalSymlinkOrReparsePoint,
            ..
        }
    ));

    let saved_manifest = fixture.manifest_path.with_extension("saved");
    fs::rename(&fixture.manifest_path, &saved_manifest).unwrap();
    symlink(&saved_manifest, &fixture.manifest_path).unwrap();
    let err = read_manifest_bytes_bounded(&fixture.manifest_path, 1024 * 1024).unwrap_err();
    assert!(err.to_string().contains("symlink or reparse point"));
}

#[cfg(windows)]
#[test]
fn final_junction_reparse_point_is_rejected() {
    let fixture = fixture(false);
    fs::remove_file(&fixture.index_path).unwrap();
    let junction_target = fixture.index_path.with_extension("junction-target");
    fs::create_dir(&junction_target).unwrap();
    let status = std::process::Command::new("cmd.exe")
        .args(["/D", "/C", "mklink", "/J"])
        .arg(&fixture.index_path)
        .arg(&junction_target)
        .status()
        .expect("cmd.exe must be available on supported Windows test hosts");
    assert!(status.success(), "failed to create test junction: {status}");

    let err = fixture
        .plan
        .decode_primary_with(ManifestIndexKind::RankQuant, |_, _| Ok::<_, io::Error>(()))
        .unwrap_err();
    assert!(matches!(
        err,
        VerifiedArtifactUseError::TypeRejected {
            rejection: VerifiedArtifactTypeRejection::FinalSymlinkOrReparsePoint,
            ..
        }
    ));
}

#[cfg(unix)]
#[test]
fn fifo_decode_child() {
    if std::env::var_os("ORDVEC_MANIFEST_FIFO_CHILD").is_none() {
        return;
    }
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let fixture = fixture(false);
    fs::remove_file(&fixture.index_path).unwrap();
    let path = CString::new(fixture.index_path.as_os_str().as_bytes()).unwrap();
    // SAFETY: `path` is a NUL-terminated pathname and the mode is valid.
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    let err = fixture
        .plan
        .decode_primary_with(ManifestIndexKind::RankQuant, |_, _| Ok::<_, io::Error>(()))
        .unwrap_err();
    assert!(matches!(
        err,
        VerifiedArtifactUseError::TypeRejected {
            rejection: VerifiedArtifactTypeRejection::NonRegularFile,
            ..
        }
    ));
}

#[cfg(unix)]
#[test]
fn fifo_rejection_subprocess_cannot_hang() {
    use std::process::{Command, Stdio};
    use std::thread;
    use std::time::{Duration, Instant};

    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "fifo_decode_child", "--nocapture"])
        .env("ORDVEC_MANIFEST_FIFO_CHILD", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "FIFO child failed with {status}");
            break;
        }
        if Instant::now() >= deadline {
            child.kill().ok();
            let output = child.wait_with_output().unwrap();
            panic!(
                "FIFO child hung; stdout={} stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        thread::sleep(Duration::from_millis(20));
    }
}
