#!/usr/bin/env python3
"""Unit tests for release_pypi_canonical_dist.py."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import tarfile
import tempfile
import unittest
import zipfile
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


def write_complete_release_dist(directory: Path, project: str = "ordvec") -> dict[str, str]:
    files = {
        f"{project}-0.3.0.tar.gz": b"sdist",
        f"{project}-0.3.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": b"linux x86_64",
        f"{project}-0.3.0-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl": b"linux aarch64",
        f"{project}-0.3.0-cp310-abi3-macosx_11_0_arm64.whl": b"macos arm64",
        f"{project}-0.3.0-cp310-abi3-win_amd64.whl": b"windows amd64",
    }
    return {name: write(directory / name, data) for name, data in files.items()}


def make_wheel(
    path: Path,
    project: str = "ordvec_manifest",
    version: str = "0.5.0",
    license_members: tuple[str, ...] = ("LICENSE-MIT", "LICENSE-APACHE-2.0"),
    *,
    nested_only: bool = False,
) -> None:
    """Write a minimal wheel (zip) whose license text lives under
    `<dist>.dist-info/licenses/`, matching the layout maturin produces. With
    `nested_only`, the license is written somewhere other than that canonical
    location so the matcher must treat it as absent."""
    dist_info = f"{project}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{project}/__init__.py", b"")
        archive.writestr(f"{dist_info}/METADATA", b"Metadata-Version: 2.4\n")
        archive.writestr(f"{dist_info}/RECORD", b"")
        for member in license_members:
            location = (
                f"{project}/vendor/{member}"
                if nested_only
                else f"{dist_info}/licenses/{member}"
            )
            archive.writestr(location, b"license text\n")


def make_sdist(
    path: Path,
    project: str = "ordvec_manifest",
    version: str = "0.5.0",
    license_members: tuple[str, ...] = ("LICENSE-MIT", "LICENSE-APACHE-2.0"),
    *,
    nested_only: bool = False,
) -> None:
    """Write a minimal sdist (tar.gz) whose license text lives at the archive
    root `<root>/<name>`. With `nested_only`, the license is only placed inside a
    vendored subdirectory so the matcher must treat the root as missing it."""
    root = f"{project}-{version}"

    def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    with tarfile.open(path, "w:gz") as archive:
        add_bytes(archive, f"{root}/PKG-INFO", b"Metadata-Version: 2.4\n")
        add_bytes(archive, f"{root}/pyproject.toml", b"[project]\n")
        for member in license_members:
            location = (
                f"{root}/vendored-crate/{member}"
                if nested_only
                else f"{root}/{member}"
            )
            add_bytes(archive, location, b"license text\n")


class LicenseMemberTests(unittest.TestCase):
    REQUIRED = ("LICENSE-MIT", "LICENSE-APACHE-2.0")

    def test_wheel_license_basenames_read_from_dist_info_licenses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "ordvec_manifest-0.5.0-cp310-abi3-win_amd64.whl"
            make_wheel(wheel)
            # The `.dist-info/licenses/` dir holds only license files.
            self.assertEqual(
                canonical.canonical_license_basenames(wheel),
                {"LICENSE-MIT", "LICENSE-APACHE-2.0"},
            )

    def test_sdist_license_basenames_read_from_archive_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "ordvec_manifest-0.5.0.tar.gz"
            make_sdist(sdist)
            # The sdist root holds the licenses plus PKG-INFO/pyproject; the
            # licenses must be present among the root members.
            found = canonical.canonical_license_basenames(sdist)
            self.assertTrue({"LICENSE-MIT", "LICENSE-APACHE-2.0"} <= found)

    def test_check_license_members_passes_for_compliant_dist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel = root / "ordvec_manifest-0.5.0-cp310-abi3-win_amd64.whl"
            sdist = root / "ordvec_manifest-0.5.0.tar.gz"
            make_wheel(wheel)
            make_sdist(sdist)
            # Must not raise.
            canonical.check_license_members(
                {wheel.name: wheel, sdist.name: sdist}, self.REQUIRED
            )

    def test_check_license_members_rejects_wheel_missing_license(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "ordvec_manifest-0.5.0-cp310-abi3-win_amd64.whl"
            make_wheel(wheel, license_members=("LICENSE-MIT",))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                canonical.check_license_members({wheel.name: wheel}, self.REQUIRED)

    def test_check_license_members_rejects_sdist_missing_license(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "ordvec_manifest-0.5.0.tar.gz"
            make_sdist(sdist, license_members=("LICENSE-APACHE-2.0",))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                canonical.check_license_members({sdist.name: sdist}, self.REQUIRED)

    def test_check_license_members_rejects_nested_only_copies(self) -> None:
        # A license buried in a vendored subdirectory does NOT satisfy the
        # canonical-location requirement — this is the exact regression class.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel = root / "ordvec_manifest-0.5.0-cp310-abi3-win_amd64.whl"
            sdist = root / "ordvec_manifest-0.5.0.tar.gz"
            make_wheel(wheel, nested_only=True)
            make_sdist(sdist, nested_only=True)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                canonical.check_license_members({wheel.name: wheel}, self.REQUIRED)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                canonical.check_license_members({sdist.name: sdist}, self.REQUIRED)

    def test_check_license_members_noop_when_no_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "ordvec_manifest-0.5.0-cp310-abi3-win_amd64.whl"
            make_wheel(wheel, license_members=())
            # No required license files → no inspection, no failure.
            canonical.check_license_members({wheel.name: wheel}, ())

    def test_canonicalize_rejects_built_dist_without_license(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = root / "built"
            out = root / "out"
            built.mkdir()
            make_sdist(built / "ordvec_manifest-0.5.0.tar.gz")
            make_wheel(
                built / "ordvec_manifest-0.5.0-cp310-abi3-win_amd64.whl",
                license_members=("LICENSE-MIT",),
            )

            old_fetch = canonical.fetch_pypi_payload
            canonical.fetch_pypi_payload = lambda project, version: self.fail(
                "license check must fail before any PyPI fetch"
            )
            try:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    canonical.canonicalize(
                        "ordvec-manifest",
                        "0.5.0",
                        built,
                        out,
                        required_license_files=("LICENSE-MIT", "LICENSE-APACHE-2.0"),
                    )
            finally:
                canonical.fetch_pypi_payload = old_fetch


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
            canonical.fetch_pypi_payload = lambda project, version: None
            try:
                with redirect_stdout(io.StringIO()):
                    canonical.canonicalize("ordvec", "0.3.0", built, out)
            finally:
                canonical.fetch_pypi_payload = old_fetch

            self.assertEqual((out / "ordvec-0.3.0.tar.gz").read_bytes(), b"fresh sdist")
            self.assertEqual((out / "ordvec-0.3.0-cp310-abi3-win_amd64.whl").read_bytes(), b"fresh wheel")

    def test_missing_pypi_release_accepts_complete_expected_release_dist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = root / "built"
            out = root / "out"
            built.mkdir()
            write_complete_release_dist(built)

            old_fetch = canonical.fetch_pypi_payload
            canonical.fetch_pypi_payload = lambda project, version: None
            try:
                with redirect_stdout(io.StringIO()):
                    canonical.canonicalize(
                        "ordvec",
                        "0.3.0",
                        built,
                        out,
                        expected_wheels=4,
                        expected_sdists=1,
                        required_wheel_tags=("x86_64", "aarch64", "macosx", "win_amd64"),
                    )
            finally:
                canonical.fetch_pypi_payload = old_fetch

            self.assertEqual(len(list(out.glob("*.whl"))), 4)
            self.assertEqual(len(list(out.glob("*.tar.gz"))), 1)

    def test_canonicalize_rejects_incomplete_expected_wheel_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = root / "built"
            out = root / "out"
            built.mkdir()
            write(built / "ordvec-0.3.0.tar.gz", b"fresh sdist")
            write(built / "ordvec-0.3.0-cp310-abi3-win_amd64.whl", b"fresh wheel")

            old_fetch = canonical.fetch_pypi_payload
            canonical.fetch_pypi_payload = lambda project, version: self.fail("unexpected PyPI fetch")
            try:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    canonical.canonicalize(
                        "ordvec",
                        "0.3.0",
                        built,
                        out,
                        expected_wheels=4,
                        expected_sdists=1,
                        required_wheel_tags=("x86_64", "aarch64", "macosx", "win_amd64"),
                    )
            finally:
                canonical.fetch_pypi_payload = old_fetch

    def test_canonicalize_rejects_missing_required_platform_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = root / "built"
            out = root / "out"
            built.mkdir()
            write(built / "ordvec-0.3.0.tar.gz", b"fresh sdist")
            write(built / "ordvec-0.3.0-cp310-abi3-manylinux_2_17_x86_64.whl", b"linux x86_64")
            write(built / "ordvec-0.3.0-cp310-abi3-manylinux_2_17_aarch64.whl", b"linux aarch64")
            write(built / "ordvec-0.3.0-cp310-abi3-macosx_11_0_arm64.whl", b"macos arm64")
            write(built / "ordvec-0.3.0-cp310-abi3-macosx_12_0_universal2.whl", b"extra macos")

            old_fetch = canonical.fetch_pypi_payload
            canonical.fetch_pypi_payload = lambda project, version: self.fail("unexpected PyPI fetch")
            try:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    canonical.canonicalize(
                        "ordvec",
                        "0.3.0",
                        built,
                        out,
                        expected_wheels=4,
                        expected_sdists=1,
                        required_wheel_tags=("x86_64", "aarch64", "macosx", "win_amd64"),
                    )
            finally:
                canonical.fetch_pypi_payload = old_fetch

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
            canonical.fetch_pypi_payload = lambda project, version: payload
            try:
                with redirect_stdout(io.StringIO()):
                    canonical.canonicalize("ordvec", "0.3.0", built, out)
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
            canonical.fetch_pypi_payload = lambda project, version: payload
            try:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    canonical.canonicalize("ordvec", "0.3.0", built, out)
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
            def fetch(project: str, version: str) -> dict[str, object] | None:
                response = responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response

            canonical.fetch_pypi_payload = fetch
            canonical.time.sleep = sleeps.append
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    canonical.verify("ordvec", "0.3.0", dist, attempts=2, sleep_seconds=0.25)
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
            canonical.fetch_pypi_payload = lambda project, version: responses.pop(0)
            canonical.time.sleep = sleeps.append
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    canonical.verify("ordvec", "0.3.0", dist, attempts=2, sleep_seconds=0.5)
            finally:
                canonical.fetch_pypi_payload = old_fetch
                canonical.time.sleep = old_sleep

            self.assertEqual(sleeps, [0.5])

    def test_verify_rejects_incomplete_local_dist_before_remote_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            write(dist / "ordvec-0.3.0.tar.gz", b"canonical sdist")
            write(dist / "ordvec-0.3.0-cp310-abi3-win_amd64.whl", b"canonical wheel")

            old_fetch = canonical.fetch_pypi_payload
            canonical.fetch_pypi_payload = lambda project, version: self.fail("unexpected PyPI fetch")
            try:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    canonical.verify(
                        "ordvec",
                        "0.3.0",
                        dist,
                        attempts=1,
                        sleep_seconds=0.0,
                        expected_wheels=4,
                        expected_sdists=1,
                        required_wheel_tags=("x86_64", "aarch64", "macosx", "win_amd64"),
                    )
            finally:
                canonical.fetch_pypi_payload = old_fetch

    def test_canonicalize_reports_pypi_read_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = root / "built"
            out = root / "out"
            built.mkdir()
            write(built / "ordvec-0.3.0.tar.gz", b"fresh sdist")

            old_fetch = canonical.fetch_pypi_payload
            canonical.fetch_pypi_payload = lambda project, version: (_ for _ in ()).throw(
                canonical.PyPIReadError("temporary PyPI 503")
            )
            try:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    canonical.canonicalize("ordvec", "0.3.0", built, out)
            finally:
                canonical.fetch_pypi_payload = old_fetch


if __name__ == "__main__":
    unittest.main()
