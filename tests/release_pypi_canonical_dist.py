#!/usr/bin/env python3
"""Canonical PyPI dist handling for the release workflow.

The normal release path publishes the wheels/sdist built by the current run.
The recovery path for an immutable PyPI version downloads the already-published
files from PyPI, verifies their published SHA-256 digests, and makes those bytes
the canonical Python dist for the GitHub Release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_PROJECT = "ordvec"
DIST_SUFFIXES = (".whl", ".tar.gz")

# A wheel carries its license text under `<dist>.dist-info/licenses/<name>`
# (the PEP 639 location maturin writes to); an sdist carries it at the archive
# root `<root>/<name>`. We deliberately ignore deeper copies (e.g. a vendored
# workspace member's own LICENSE inside the sdist) — the regression this guards
# is the license missing from the canonical location the packaging metadata
# points at, not the absence of every copy.
_WHEEL_LICENSE_MEMBER = re.compile(r"[^/]+\.dist-info/licenses/([^/]+)")
_SDIST_ROOT_MEMBER = re.compile(r"[^/]+/([^/]+)")


class PyPIReadError(RuntimeError):
    """PyPI returned an unusable response for a retryable read."""


def fail(message: str) -> None:
    print(f"::error::{message}", file=sys.stderr)
    raise SystemExit(1)


def notice(message: str) -> None:
    print(f"::notice::{message}")


def set_output(name: str, value: str) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(f"{name}={value}\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dist_files(directory: Path) -> dict[str, Path]:
    files = {
        path.name: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name.endswith(DIST_SUFFIXES)
    }
    if not files:
        fail(f"no wheel/sdist files found in {directory}")
    return files


def canonical_license_basenames(path: Path) -> set[str]:
    """Basenames present at the canonical license location of a built dist
    archive — where a license file MUST appear to be detected by tooling:
    `*.dist-info/licenses/` for a wheel (license-only by construction), and the
    archive root `<root>/` for an sdist (which also holds PKG-INFO, pyproject,
    etc.). Callers intersect this with the required license names, so the extra
    sdist-root entries are harmless. Deeper copies (a vendored member's own
    LICENSE) are intentionally excluded. Unknown suffixes return an empty set."""
    name = path.name
    if name.endswith(".whl"):
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive.namelist()
        except (zipfile.BadZipFile, OSError) as exc:
            fail(f"could not read wheel {name}: {exc!r}")
        pattern = _WHEEL_LICENSE_MEMBER
    elif name.endswith(".tar.gz"):
        try:
            with tarfile.open(path, "r:gz") as archive:
                members = archive.getnames()
        except (tarfile.TarError, OSError) as exc:
            fail(f"could not read sdist {name}: {exc!r}")
        pattern = _SDIST_ROOT_MEMBER
    else:
        return set()
    found: set[str] = set()
    for member in members:
        match = pattern.fullmatch(member.replace("\\", "/"))
        if match:
            found.add(match.group(1))
    return found


def check_license_members(files: dict[str, Path], required: tuple[str, ...]) -> None:
    """Fail unless every required license file is present in the canonical
    license location of every wheel and sdist. Closes the regression class where
    a crate declares `license = "MIT OR Apache-2.0"` but ships no license text."""
    if not required:
        return
    required_set = set(required)
    for filename, path in sorted(files.items()):
        present = canonical_license_basenames(path) & required_set
        missing = sorted(required_set - present)
        if missing:
            fail(
                f"{filename} is missing required license file(s) in its canonical "
                f"license location: {', '.join(missing)}"
            )


def validate_expected_dist(
    files: dict[str, Any],
    *,
    expected_wheels: int | None = None,
    expected_sdists: int | None = None,
    required_wheel_tags: tuple[str, ...] = (),
) -> None:
    wheels = sorted(name for name in files if name.endswith(".whl"))
    sdists = sorted(name for name in files if name.endswith(".tar.gz"))
    if expected_wheels is not None and len(wheels) != expected_wheels:
        fail(f"expected {expected_wheels} wheel files, found {len(wheels)}: {wheels!r}")
    if expected_sdists is not None and len(sdists) != expected_sdists:
        fail(f"expected {expected_sdists} sdist files, found {len(sdists)}: {sdists!r}")
    missing_tags = [
        tag for tag in required_wheel_tags if not any(tag in wheel for wheel in wheels)
    ]
    if missing_tags:
        fail(
            "wheel dist is missing required platform tag substrings: "
            f"missing={missing_tags!r} wheels={wheels!r}"
        )


def fetch_pypi_payload(project: str, version: str) -> dict[str, Any] | None:
    url = f"https://pypi.org/pypi/{project}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise PyPIReadError(f"could not read {url}: HTTP {exc.code}") from exc
    except Exception as exc:  # noqa: BLE001 - release diagnostics should be direct.
        raise PyPIReadError(f"could not read {url}: {exc!r}") from exc
    raise AssertionError("unreachable")


def pypi_dist_map(payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    dist: dict[str, dict[str, str]] = {}
    for item in payload.get("urls", []):
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        url = item.get("url")
        sha256 = item.get("digests", {}).get("sha256")
        if not (
            isinstance(filename, str)
            and filename.endswith(DIST_SUFFIXES)
            and isinstance(url, str)
            and isinstance(sha256, str)
        ):
            continue
        dist[filename] = {"url": url, "sha256": sha256}
    if not dist:
        raise PyPIReadError("PyPI JSON did not contain any wheel/sdist files")
    return dist


def prepare_empty_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        fail(f"{path} must be empty before canonical dist is written")


def download_verified(url: str, expected_sha256: str, target: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except Exception as exc:  # noqa: BLE001 - release diagnostics should be direct.
        fail(f"could not download {url}: {exc!r}")
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != expected_sha256:
        fail(f"downloaded {target.name} hash mismatch: {actual_sha256} != {expected_sha256}")
    target.write_bytes(data)


def ensure_same_filenames(local: dict[str, Path], remote: dict[str, dict[str, str]]) -> None:
    local_names = set(local)
    remote_names = set(remote)
    if local_names != remote_names:
        only_local = sorted(local_names - remote_names)
        only_remote = sorted(remote_names - local_names)
        fail(
            "current build and PyPI have different dist filename sets: "
            f"only_local={only_local!r} only_pypi={only_remote!r}"
        )


def canonicalize(
    project: str,
    version: str,
    built_dir: Path,
    out_dir: Path,
    *,
    expected_wheels: int | None = None,
    expected_sdists: int | None = None,
    required_wheel_tags: tuple[str, ...] = (),
    required_license_files: tuple[str, ...] = (),
) -> None:
    built = dist_files(built_dir)
    validate_expected_dist(
        built,
        expected_wheels=expected_wheels,
        expected_sdists=expected_sdists,
        required_wheel_tags=required_wheel_tags,
    )
    check_license_members(built, required_license_files)
    prepare_empty_dir(out_dir)
    try:
        payload = fetch_pypi_payload(project, version)
    except PyPIReadError as exc:
        fail(str(exc))

    if payload is None:
        for filename, path in built.items():
            shutil.copy2(path, out_dir / filename)
        set_output("source", "build")
        set_output("pypi_exists", "false")
        print(f"OK: PyPI has no {project} {version}; canonical dist uses current build")
        return

    try:
        remote = pypi_dist_map(payload)
    except PyPIReadError as exc:
        fail(str(exc))
    ensure_same_filenames(built, remote)
    validate_expected_dist(
        remote,
        expected_wheels=expected_wheels,
        expected_sdists=expected_sdists,
        required_wheel_tags=required_wheel_tags,
    )

    mismatched: list[str] = []
    for filename, path in built.items():
        built_sha256 = sha256_file(path)
        remote_sha256 = remote[filename]["sha256"]
        if built_sha256 != remote_sha256:
            mismatched.append(filename)

    if mismatched:
        notice(
            "PyPI already has immutable files whose bytes differ from this rebuild; "
            f"using PyPI-canonical bytes for {', '.join(mismatched)}"
        )

    for filename, item in remote.items():
        download_verified(item["url"], item["sha256"], out_dir / filename)

    set_output("source", "pypi")
    set_output("pypi_exists", "true")
    print(f"OK: PyPI already has {project} {version}; canonical dist uses verified PyPI files")


def remote_hashes(project: str, version: str) -> dict[str, str] | None:
    payload = fetch_pypi_payload(project, version)
    if payload is None:
        return None
    return {name: item["sha256"] for name, item in pypi_dist_map(payload).items()}


def local_hashes(
    dist_dir: Path,
    *,
    expected_wheels: int | None = None,
    expected_sdists: int | None = None,
    required_wheel_tags: tuple[str, ...] = (),
) -> dict[str, str]:
    files = dist_files(dist_dir)
    validate_expected_dist(
        files,
        expected_wheels=expected_wheels,
        expected_sdists=expected_sdists,
        required_wheel_tags=required_wheel_tags,
    )
    return {name: sha256_file(path) for name, path in files.items()}


def verify(
    project: str,
    version: str,
    dist_dir: Path,
    attempts: int,
    sleep_seconds: float,
    *,
    expected_wheels: int | None = None,
    expected_sdists: int | None = None,
    required_wheel_tags: tuple[str, ...] = (),
    required_license_files: tuple[str, ...] = (),
) -> None:
    local = local_hashes(
        dist_dir,
        expected_wheels=expected_wheels,
        expected_sdists=expected_sdists,
        required_wheel_tags=required_wheel_tags,
    )
    check_license_members(dist_files(dist_dir), required_license_files)
    url = f"https://pypi.org/pypi/{project}/{version}/json"
    last_error = "not checked"
    for attempt in range(1, attempts + 1):
        try:
            remote = remote_hashes(project, version)
            if remote == local:
                print(f"OK: PyPI-served hashes match canonical dist for {project} {version}")
                return
            last_error = f"local={local!r} remote={remote!r}"
        except PyPIReadError as exc:
            last_error = str(exc)
        print(f"waiting for PyPI JSON/hash propagation ({attempt}/{attempts}): {last_error}", file=sys.stderr)
        if attempt != attempts:
            time.sleep(sleep_seconds)
    fail(f"PyPI hash verification failed for {url}: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    canonical = subparsers.add_parser("canonicalize")
    canonical.add_argument("--project", default=DEFAULT_PROJECT)
    canonical.add_argument("--version", required=True)
    canonical.add_argument("--built-dir", required=True, type=Path)
    canonical.add_argument("--out-dir", required=True, type=Path)
    canonical.add_argument("--expected-wheels", type=int)
    canonical.add_argument("--expected-sdists", type=int)
    canonical.add_argument(
        "--required-wheel-tag",
        action="append",
        default=[],
        help="Require at least one wheel filename containing this substring; may be repeated.",
    )
    canonical.add_argument(
        "--require-license-file",
        action="append",
        default=[],
        help="Require this license basename in every wheel/sdist's canonical "
        "license location; may be repeated.",
    )

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--project", default=DEFAULT_PROJECT)
    verify_parser.add_argument("--version", required=True)
    verify_parser.add_argument("--dist-dir", required=True, type=Path)
    verify_parser.add_argument("--attempts", default=24, type=int)
    verify_parser.add_argument("--sleep-seconds", default=5.0, type=float)
    verify_parser.add_argument("--expected-wheels", type=int)
    verify_parser.add_argument("--expected-sdists", type=int)
    verify_parser.add_argument(
        "--required-wheel-tag",
        action="append",
        default=[],
        help="Require at least one wheel filename containing this substring; may be repeated.",
    )
    verify_parser.add_argument(
        "--require-license-file",
        action="append",
        default=[],
        help="Require this license basename in every wheel/sdist's canonical "
        "license location; may be repeated.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "canonicalize":
        canonicalize(
            args.project,
            args.version,
            args.built_dir,
            args.out_dir,
            expected_wheels=args.expected_wheels,
            expected_sdists=args.expected_sdists,
            required_wheel_tags=tuple(args.required_wheel_tag),
            required_license_files=tuple(args.require_license_file),
        )
        return
    if args.command == "verify":
        verify(
            args.project,
            args.version,
            args.dist_dir,
            args.attempts,
            args.sleep_seconds,
            expected_wheels=args.expected_wheels,
            expected_sdists=args.expected_sdists,
            required_wheel_tags=tuple(args.required_wheel_tag),
            required_license_files=tuple(args.require_license_file),
        )
        return
    raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
