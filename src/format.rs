//! Persisted-format registry shared by probes, bindings, and manifest tooling.
//!
//! This module is the single source of truth for on-disk ordvec magics and the
//! support stance of each persisted format. Loader and probe implementations
//! still live with their owning index code, but dispatchers should identify
//! formats through this registry so new magics cannot silently drift across
//! trust-boundary surfaces.

use crate::rank_io::IndexKind;

// Current ordvec magics — written by this crate going forward.
pub(crate) const OVR_MAGIC: &[u8; 4] = b"OVR1";
pub(crate) const OVRQ_MAGIC: &[u8; 4] = b"OVRQ";
pub(crate) const OVBM_MAGIC: &[u8; 4] = b"OVBM";
pub(crate) const OVSB_MAGIC: &[u8; 4] = b"OVSB";
// FastScan b=2 block-32 layout (`RankQuantFastscan`). New in the ordvec format:
// there is no turbovec-era counterpart, so it has no legacy magic.
pub(crate) const OVFS_MAGIC: &[u8; 4] = b"OVFS";

// Legacy turbovec-era magics — still accepted on load for backward
// compatibility, never written.
pub(crate) const TVR_MAGIC: &[u8; 4] = b"TVR1";
pub(crate) const TVRQ_MAGIC: &[u8; 4] = b"TVRQ";
pub(crate) const TVBM_MAGIC: &[u8; 4] = b"TVBM";
pub(crate) const TVSB_MAGIC: &[u8; 4] = b"TVSB";

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PersistedFormat {
    Rank,
    RankQuant,
    Bitmap,
    SignBitmap,
    RankQuantFastscan,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProbeCoverage {
    Covered,
    NotCovered {
        tracking_issue: u32,
        reason: &'static str,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ManifestCoverage {
    Covered,
    NotCovered {
        tracking_issue: u32,
        reason: &'static str,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FfiLoadSupport {
    Supported,
    Unsupported { reason: &'static str },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FormatSpec {
    pub format: PersistedFormat,
    pub extension: &'static str,
    pub magic: &'static [u8; 4],
    pub legacy_magic: Option<&'static [u8; 4]>,
    pub kind: IndexKind,
    pub probe: ProbeCoverage,
    pub manifest: ManifestCoverage,
    pub ffi_load: FfiLoadSupport,
}

const FASTSCAN_PROBE_UNSUPPORTED: &str = "OVFS (RankQuantFastscan) metadata probing is not \
supported in this version; load the index with RankQuantFastscan::load (tracked in #232)";
const FASTSCAN_MANIFEST_UNSUPPORTED: &str =
    "RankQuantFastscan (.ovfs) is not covered by ordvec-manifest v1 (tracked in #232)";
const FFI_CORE_ONLY: &str = "ABI v1 supports only RankQuant and Bitmap indexes";
const FFI_FASTSCAN_UNSUPPORTED: &str =
    "ABI v1 does not support RankQuantFastscan indexes (tracked in #232)";

pub const FORMATS: &[FormatSpec] = &[
    FormatSpec {
        format: PersistedFormat::Rank,
        extension: "ovr",
        magic: OVR_MAGIC,
        legacy_magic: Some(TVR_MAGIC),
        kind: IndexKind::Rank,
        probe: ProbeCoverage::Covered,
        manifest: ManifestCoverage::Covered,
        ffi_load: FfiLoadSupport::Unsupported {
            reason: FFI_CORE_ONLY,
        },
    },
    FormatSpec {
        format: PersistedFormat::RankQuant,
        extension: "ovrq",
        magic: OVRQ_MAGIC,
        legacy_magic: Some(TVRQ_MAGIC),
        kind: IndexKind::RankQuant,
        probe: ProbeCoverage::Covered,
        manifest: ManifestCoverage::Covered,
        ffi_load: FfiLoadSupport::Supported,
    },
    FormatSpec {
        format: PersistedFormat::Bitmap,
        extension: "ovbm",
        magic: OVBM_MAGIC,
        legacy_magic: Some(TVBM_MAGIC),
        kind: IndexKind::Bitmap,
        probe: ProbeCoverage::Covered,
        manifest: ManifestCoverage::Covered,
        ffi_load: FfiLoadSupport::Supported,
    },
    FormatSpec {
        format: PersistedFormat::SignBitmap,
        extension: "ovsb",
        magic: OVSB_MAGIC,
        legacy_magic: Some(TVSB_MAGIC),
        kind: IndexKind::SignBitmap,
        probe: ProbeCoverage::Covered,
        manifest: ManifestCoverage::Covered,
        ffi_load: FfiLoadSupport::Unsupported {
            reason: FFI_CORE_ONLY,
        },
    },
    FormatSpec {
        format: PersistedFormat::RankQuantFastscan,
        extension: "ovfs",
        magic: OVFS_MAGIC,
        legacy_magic: None,
        kind: IndexKind::RankQuantFastscan,
        probe: ProbeCoverage::NotCovered {
            tracking_issue: 232,
            reason: FASTSCAN_PROBE_UNSUPPORTED,
        },
        manifest: ManifestCoverage::NotCovered {
            tracking_issue: 232,
            reason: FASTSCAN_MANIFEST_UNSUPPORTED,
        },
        ffi_load: FfiLoadSupport::Unsupported {
            reason: FFI_FASTSCAN_UNSUPPORTED,
        },
    },
];

pub fn lookup_magic(magic: &[u8; 4]) -> Option<&'static FormatSpec> {
    FORMATS
        .iter()
        .find(|spec| spec.magic == magic || spec.legacy_magic == Some(magic))
}

pub fn spec(format: PersistedFormat) -> &'static FormatSpec {
    FORMATS
        .iter()
        .find(|spec| spec.format == format)
        .expect("every PersistedFormat must have a FormatSpec")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_one_row_per_current_format() {
        assert_eq!(
            FORMATS.len(),
            5,
            "new formats must update this registry test"
        );
        assert_eq!(spec(PersistedFormat::Rank).kind, IndexKind::Rank);
        assert_eq!(spec(PersistedFormat::RankQuant).kind, IndexKind::RankQuant);
        assert_eq!(spec(PersistedFormat::Bitmap).kind, IndexKind::Bitmap);
        assert_eq!(
            spec(PersistedFormat::SignBitmap).kind,
            IndexKind::SignBitmap
        );
        assert_eq!(
            spec(PersistedFormat::RankQuantFastscan).kind,
            IndexKind::RankQuantFastscan
        );
    }

    #[test]
    fn every_magic_and_legacy_magic_resolves_to_its_row() {
        for spec in FORMATS {
            assert_eq!(
                lookup_magic(spec.magic).map(|found| found.format),
                Some(spec.format)
            );
            if let Some(legacy_magic) = spec.legacy_magic {
                assert_eq!(
                    lookup_magic(legacy_magic).map(|found| found.format),
                    Some(spec.format)
                );
            }
        }
        assert!(lookup_magic(b"NOPE").is_none());
    }

    #[test]
    fn manifest_coverage_is_explicit_for_every_format() {
        for spec in FORMATS {
            match spec.manifest {
                ManifestCoverage::Covered => {}
                ManifestCoverage::NotCovered {
                    tracking_issue,
                    reason,
                } => {
                    assert!(tracking_issue > 0);
                    assert!(!reason.trim().is_empty());
                }
            }
        }
        assert!(matches!(
            spec(PersistedFormat::RankQuantFastscan).manifest,
            ManifestCoverage::NotCovered {
                tracking_issue: 232,
                ..
            }
        ));
    }

    #[test]
    fn ffi_load_support_is_limited_to_abi_v1_formats() {
        for spec in FORMATS {
            match (spec.format, spec.ffi_load) {
                (
                    PersistedFormat::RankQuant | PersistedFormat::Bitmap,
                    FfiLoadSupport::Supported,
                ) => {}
                (
                    PersistedFormat::Rank
                    | PersistedFormat::SignBitmap
                    | PersistedFormat::RankQuantFastscan,
                    FfiLoadSupport::Unsupported { reason },
                ) => assert!(!reason.trim().is_empty()),
                other => panic!("unexpected FFI load support matrix entry: {other:?}"),
            }
        }
    }
}
