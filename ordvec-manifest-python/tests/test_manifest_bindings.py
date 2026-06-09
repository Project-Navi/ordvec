from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import ordvec_manifest


def write_rankquant_index(path: Path, *, dim: int = 16, rows: int = 2, bits: int = 2):
    bytes_per_vec = dim * bits // 8
    path.write_bytes(
        b"TVRQ"
        + bytes([1, bits])
        + dim.to_bytes(4, "little")
        + rows.to_bytes(4, "little")
        + (b"\x00" * (rows * bytes_per_vec))
    )


def write_unloadable_manifest(tmp_path):
    artifact = tmp_path / "index.tvrq"
    artifact.write_bytes(b"not an ordvec index")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "schema_version": ordvec_manifest.SCHEMA_VERSION,
        "manifest_id": "urn:uuid:7c66ad6e-bdde-49a8-b420-f1136d04f5bd",
        "created_at": "2026-06-09T00:00:00Z",
        "artifact": {
            "path": artifact.name,
            "sha256": digest,
            "kind": "rank_quant",
            "format_version": 1,
            "dim": 16,
            "vector_count": 1,
            "bytes_per_vec": 4,
            "params": {"kind": "rank_quant", "bits": 2},
            "file_size_bytes": artifact.stat().st_size,
        },
        "embedding": {"model": "test-embedding", "dim": 16},
        "row_identity": {"kind": "row_id_identity", "row_count": 1},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return artifact, manifest_path


def test_hash_and_limits(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"manifest bindings")

    result = ordvec_manifest.sha256_file(path)

    assert result == {
        "sha256": hashlib.sha256(b"manifest bindings").hexdigest(),
        "size_bytes": len(b"manifest bindings"),
    }
    limits = ordvec_manifest.default_resource_limits()
    assert limits["max_manifest_bytes"] == ordvec_manifest.DEFAULT_MAX_MANIFEST_BYTES
    assert "max_row_map_line_bytes" in limits
    assert "max_row_identity_jsonl_line_bytes" not in limits

    _, manifest_path = write_unloadable_manifest(tmp_path)
    report = ordvec_manifest.verify_manifest(manifest_path, **limits)
    assert report["ok"] is False


def test_inspect_and_verify_return_dicts(tmp_path):
    _, manifest_path = write_unloadable_manifest(tmp_path)

    manifest = ordvec_manifest.inspect_manifest(manifest_path)
    report = ordvec_manifest.verify_manifest(manifest_path)

    assert manifest["schema_version"] == ordvec_manifest.SCHEMA_VERSION
    assert report["ok"] is False
    assert report["artifact"]["sha256"] == manifest["artifact"]["sha256"]
    assert any(error["code"] == "artifact_probe_failed" for error in report["errors"])


def test_verify_for_load_raises_when_report_is_not_loadable(tmp_path):
    _, manifest_path = write_unloadable_manifest(tmp_path)

    with pytest.raises(ValueError, match="manifest verification failed"):
        ordvec_manifest.verify_for_load(manifest_path)


def test_verify_for_load_preserves_manifest_io_errors(tmp_path):
    with pytest.raises(OSError):
        ordvec_manifest.verify_for_load(tmp_path / "missing.json")


def test_create_manifest_requires_explicit_row_identity(tmp_path):
    index = tmp_path / "index.tvrq"
    index.write_bytes(b"not an ordvec index")

    with pytest.raises(ValueError, match="row_map or row_id_is_identity"):
        ordvec_manifest.create_manifest(index, tmp_path / "manifest.json", "model")


def test_create_manifest_accepts_auxiliary_artifacts(tmp_path):
    index = tmp_path / "index.tvrq"
    ids = tmp_path / "ids.bin"
    optional = tmp_path / "optional.json"
    manifest_path = tmp_path / "manifest.json"
    write_rankquant_index(index)
    ids.write_bytes((7).to_bytes(8, "little") + (9).to_bytes(8, "little"))
    optional.write_text('{"optional": true}', encoding="utf-8")

    manifest = ordvec_manifest.create_manifest(
        index,
        manifest_path,
        "model",
        row_id_is_identity=True,
        auxiliary_artifacts=[
            {"name": "ordinaldb.ids", "path": ids},
            {"name": "optional.stats", "path": optional, "required": False},
        ],
    )

    assert manifest["row_identity"] == {"kind": "row_id_identity", "row_count": 2}
    assert manifest["auxiliary_artifacts"][0]["name"] == "ordinaldb.ids"
    assert manifest["auxiliary_artifacts"][0]["path"] == "ids.bin"
    assert manifest["auxiliary_artifacts"][0].get("required", True) is True
    assert manifest["auxiliary_artifacts"][1]["name"] == "optional.stats"
    assert manifest["auxiliary_artifacts"][1]["required"] is False

    optional.unlink()
    plan = ordvec_manifest.verify_for_load(manifest_path)
    auxiliary = {artifact["name"]: artifact for artifact in plan["auxiliary_artifacts"]}
    assert auxiliary["ordinaldb.ids"]["state"] == "verified"
    assert Path(auxiliary["ordinaldb.ids"]["path"]) == ids.resolve()
    assert auxiliary["optional.stats"]["state"] == "optional_absent"
