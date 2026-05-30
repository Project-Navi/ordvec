#!/usr/bin/env python3
"""Unit tests for release_pypi_canonical_dist.py."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


SCRIPT = Path(__file__).with_name("release_pypi_canonical_dist.py")
SPEC = importlib.util.spec_from_file_location("release_pypi_canonical_dist", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
canonical = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(canonical)


def write(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class CanonicalPyPIDistTests(unittest.TestCase):
    def test_missing_pypi_release_uses_current_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = root / "built"
            out = root / "out"
            built.mkdir()
            write(built / "ordvec-0.3.0.tar.gz", b"fresh sdist")
            write(built / "ordvec-0.3.0-cp310-abi3-win_amd64.whl", b"fresh wheel")

            old_fetch = canonical.fetch_pypi_payload
            canonical.fetch_pypi_payload = lambda version: None
            try:
                with redirect_stdout(io.StringIO()):
                    canonical.canonicalize("0.3.0", built, out)
            finally:
                canonical.fetch_pypi_payload = old_fetch

            self.assertEqual((out / "ordvec-0.3.0.tar.gz").read_bytes(), b"fresh sdist")
            self.assertEqual((out / "ordvec-0.3.0-cp310-abi3-win_amd64.whl").read_bytes(), b"fresh wheel")

    def test_existing_pypi_release_uses_verified_remote_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = root / "built"
            remote = root / "remote"
            out = root / "out"
            built.mkdir()
            remote.mkdir()

            write(built / "ordvec-0.3.0.tar.gz", b"rebuilt sdist")
            write(built / "ordvec-0.3.0-cp310-abi3-win_amd64.whl", b"rebuilt wheel")
            sdist_sha = write(remote / "ordvec-0.3.0.tar.gz", b"pypi sdist")
            wheel_sha = write(remote / "ordvec-0.3.0-cp310-abi3-win_amd64.whl", b"pypi wheel")

            payload = {
                "urls": [
                    {
                        "filename": "ordvec-0.3.0.tar.gz",
                        "url": (remote / "ordvec-0.3.0.tar.gz").as_uri(),
                        "digests": {"sha256": sdist_sha},
                    },
                    {
                        "filename": "ordvec-0.3.0-cp310-abi3-win_amd64.whl",
                        "url": (remote / "ordvec-0.3.0-cp310-abi3-win_amd64.whl").as_uri(),
                        "digests": {"sha256": wheel_sha},
                    },
                ]
            }

            old_fetch = canonical.fetch_pypi_payload
            canonical.fetch_pypi_payload = lambda version: payload
            try:
                with redirect_stdout(io.StringIO()):
                    canonical.canonicalize("0.3.0", built, out)
            finally:
                canonical.fetch_pypi_payload = old_fetch

            self.assertEqual((out / "ordvec-0.3.0.tar.gz").read_bytes(), b"pypi sdist")
            self.assertEqual((out / "ordvec-0.3.0-cp310-abi3-win_amd64.whl").read_bytes(), b"pypi wheel")

    def test_existing_pypi_release_rejects_filename_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = root / "built"
            out = root / "out"
            built.mkdir()
            write(built / "ordvec-0.3.0.tar.gz", b"fresh sdist")

            payload = {
                "urls": [
                    {
                        "filename": "ordvec-0.3.0-cp310-abi3-win_amd64.whl",
                        "url": "file:///unused",
                        "digests": {"sha256": "0" * 64},
                    }
                ]
            }

            old_fetch = canonical.fetch_pypi_payload
            canonical.fetch_pypi_payload = lambda version: payload
            try:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    canonical.canonicalize("0.3.0", built, out)
            finally:
                canonical.fetch_pypi_payload = old_fetch

    def test_verify_retries_after_transient_pypi_fetch_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            wheel_sha = write(dist / "ordvec-0.3.0-cp310-abi3-win_amd64.whl", b"canonical wheel")
            payload = {
                "urls": [
                    {
                        "filename": "ordvec-0.3.0-cp310-abi3-win_amd64.whl",
                        "url": "file:///unused",
                        "digests": {"sha256": wheel_sha},
                    }
                ]
            }
            responses = [canonical.PyPIReadError("temporary PyPI 503"), payload]
            sleeps: list[float] = []

            old_fetch = canonical.fetch_pypi_payload
            old_sleep = canonical.time.sleep
            def fetch(version: str) -> dict[str, object] | None:
                response = responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response

            canonical.fetch_pypi_payload = fetch
            canonical.time.sleep = sleeps.append
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    canonical.verify("0.3.0", dist, attempts=2, sleep_seconds=0.25)
            finally:
                canonical.fetch_pypi_payload = old_fetch
                canonical.time.sleep = old_sleep

            self.assertEqual(sleeps, [0.25])

    def test_verify_retries_after_empty_pypi_dist_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            sdist_sha = write(dist / "ordvec-0.3.0.tar.gz", b"canonical sdist")
            payload = {
                "urls": [
                    {
                        "filename": "ordvec-0.3.0.tar.gz",
                        "url": "file:///unused",
                        "digests": {"sha256": sdist_sha},
                    }
                ]
            }
            responses = [{"urls": []}, payload]
            sleeps: list[float] = []

            old_fetch = canonical.fetch_pypi_payload
            old_sleep = canonical.time.sleep
            canonical.fetch_pypi_payload = lambda version: responses.pop(0)
            canonical.time.sleep = sleeps.append
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    canonical.verify("0.3.0", dist, attempts=2, sleep_seconds=0.5)
            finally:
                canonical.fetch_pypi_payload = old_fetch
                canonical.time.sleep = old_sleep

            self.assertEqual(sleeps, [0.5])

    def test_canonicalize_reports_pypi_read_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = root / "built"
            out = root / "out"
            built.mkdir()
            write(built / "ordvec-0.3.0.tar.gz", b"fresh sdist")

            old_fetch = canonical.fetch_pypi_payload
            canonical.fetch_pypi_payload = lambda version: (_ for _ in ()).throw(
                canonical.PyPIReadError("temporary PyPI 503")
            )
            try:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    canonical.canonicalize("0.3.0", built, out)
            finally:
                canonical.fetch_pypi_payload = old_fetch


if __name__ == "__main__":
    unittest.main()
