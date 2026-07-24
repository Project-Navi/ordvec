"""Python bindings for the ordvec manifest verifier.

The package wraps the Rust ``ordvec-manifest`` crate and returns plain Python
``dict`` objects for JSON-shaped verifier outputs. It is intentionally a
verification API, not a policy engine: callers still decide where to store
artifacts, how to trust keys, and when to load verified bytes.
"""

from ._ordvec_manifest import (
    CALIBRATION_SCHEMA_VERSION,
    DEFAULT_MAX_AUXILIARY_ARTIFACT_BYTES,
    DEFAULT_MAX_AUXILIARY_ARTIFACTS,
    DEFAULT_MAX_CACHED_REPORT_BYTES,
    DEFAULT_MAX_CALIBRATION_PROFILE_BYTES,
    DEFAULT_MAX_ENCODER_DISTORTION_PROFILE_BYTES,
    DEFAULT_MAX_MANIFEST_BYTES,
    DEFAULT_MAX_REPORT_ISSUES,
    DEFAULT_MAX_ROW_IDENTITY_JSONL_LINE_BYTES,
    DEFAULT_MAX_ROW_IDENTITY_ROWS,
    DEFAULT_MAX_ROW_IDENTITY_TRACKED_DB_ID_BYTES,
    ENCODER_DISTORTION_SCHEMA_VERSION,
    SCHEMA_VERSION,
    create_manifest,
    default_resource_limits,
    inspect_manifest,
    sha256_file,
    verify_for_load,
    verify_manifest,
)

__all__ = [
    "SCHEMA_VERSION",
    "CALIBRATION_SCHEMA_VERSION",
    "ENCODER_DISTORTION_SCHEMA_VERSION",
    "DEFAULT_MAX_MANIFEST_BYTES",
    "DEFAULT_MAX_ROW_IDENTITY_JSONL_LINE_BYTES",
    "DEFAULT_MAX_ROW_IDENTITY_ROWS",
    "DEFAULT_MAX_ROW_IDENTITY_TRACKED_DB_ID_BYTES",
    "DEFAULT_MAX_AUXILIARY_ARTIFACTS",
    "DEFAULT_MAX_AUXILIARY_ARTIFACT_BYTES",
    "DEFAULT_MAX_CALIBRATION_PROFILE_BYTES",
    "DEFAULT_MAX_ENCODER_DISTORTION_PROFILE_BYTES",
    "DEFAULT_MAX_REPORT_ISSUES",
    "DEFAULT_MAX_CACHED_REPORT_BYTES",
    "default_resource_limits",
    "sha256_file",
    "inspect_manifest",
    "verify_manifest",
    "verify_for_load",
    "create_manifest",
]

__version__ = "0.6.0"
