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
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT = "ordvec"
DIST_SUFFIXES = (".whl", ".tar.gz")


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


def fetch_pypi_payload(version: str) -> dict[str, Any] | None:
    url = f"https://pypi.org/pypi/{PROJECT}/{version}/json"
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


def canonicalize(version: str, built_dir: Path, out_dir: Path) -> None:
    built = dist_files(built_dir)
    prepare_empty_dir(out_dir)
    try:
        payload = fetch_pypi_payload(version)
    except PyPIReadError as exc:
        fail(str(exc))

    if payload is None:
        for filename, path in built.items():
            shutil.copy2(path, out_dir / filename)
        set_output("source", "build")
        set_output("pypi_exists", "false")
        print(f"OK: PyPI has no {PROJECT} {version}; canonical dist uses current build")
        return

    try:
        remote = pypi_dist_map(payload)
    except PyPIReadError as exc:
        fail(str(exc))
    ensure_same_filenames(built, remote)

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
    print(f"OK: PyPI already has {PROJECT} {version}; canonical dist uses verified PyPI files")


def remote_hashes(version: str) -> dict[str, str] | None:
    payload = fetch_pypi_payload(version)
    if payload is None:
        return None
    return {name: item["sha256"] for name, item in pypi_dist_map(payload).items()}


def local_hashes(dist_dir: Path) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in dist_files(dist_dir).items()}


def verify(version: str, dist_dir: Path, attempts: int, sleep_seconds: float) -> None:
    local = local_hashes(dist_dir)
    url = f"https://pypi.org/pypi/{PROJECT}/{version}/json"
    last_error = "not checked"
    for attempt in range(1, attempts + 1):
        try:
            remote = remote_hashes(version)
            if remote == local:
                print(f"OK: PyPI-served hashes match canonical dist for {PROJECT} {version}")
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
    canonical.add_argument("--version", required=True)
    canonical.add_argument("--built-dir", required=True, type=Path)
    canonical.add_argument("--out-dir", required=True, type=Path)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--version", required=True)
    verify_parser.add_argument("--dist-dir", required=True, type=Path)
    verify_parser.add_argument("--attempts", default=24, type=int)
    verify_parser.add_argument("--sleep-seconds", default=5.0, type=float)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "canonicalize":
        canonicalize(args.version, args.built_dir, args.out_dir)
        return
    if args.command == "verify":
        verify(args.version, args.dist_dir, args.attempts, args.sleep_seconds)
        return
    raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
