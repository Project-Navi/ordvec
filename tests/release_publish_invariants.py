#!/usr/bin/env python3
"""Structural release publish invariants for registry upload jobs."""

from __future__ import annotations

import json
import os
import posixpath
import shutil
import subprocess
import sys
from typing import Any


WORKFLOW_PATH = os.environ.get("RELEASE_WORKFLOW_PATH", ".github/workflows/release.yml")


def fail(message: str) -> None:
    print(f"::error::release-publish invariant violated: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_workflow(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        yq = shutil.which("yq")
        if yq is None:
            fail("PyYAML or yq is required to parse .github/workflows/release.yml")
        try:
            raw = subprocess.check_output([yq, "-o=json", ".", path], text=True)
        except subprocess.CalledProcessError as exc:
            fail(f"{path}: yq could not parse workflow YAML: {exc}")
        try:
            workflow = json.loads(raw)
        except json.JSONDecodeError as exc:
            fail(f"{path}: yq emitted invalid JSON: {exc}")
    else:
        try:
            with open(path, encoding="utf-8") as fh:
                workflow = yaml.safe_load(fh)
        except OSError as exc:
            fail(f"{path}: could not read workflow: {exc}")
        except Exception as exc:  # PyYAML exposes several parser exception types.
            fail(f"{path}: could not parse workflow YAML: {exc}")

    if not isinstance(workflow, dict):
        fail(f"{path}: workflow root must be a mapping")
    return workflow


def mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{context} must be a mapping")
    return value


def sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{context} must be a sequence")
    return value


def action_name(step: dict[str, Any]) -> str | None:
    uses = step.get("uses")
    if not isinstance(uses, str):
        return None
    return uses.split("@", 1)[0].lower()


def norm_path(value: Any) -> str:
    if value is None:
        return ""
    path = str(value).strip().replace("\\", "/")
    if not path:
        return ""
    normalized = posixpath.normpath(path)
    return "" if normalized == "." else normalized.rstrip("/")


def boolish_true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


def step_label(index: int, step: dict[str, Any]) -> str:
    name = step.get("name")
    if isinstance(name, str) and name:
        return f"step {index + 1} ({name!r})"
    return f"step {index + 1}"


def empty(value: Any) -> bool:
    return value is None or value == ""


def check_publish_pypi(workflow: dict[str, Any], path: str) -> None:
    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    job = mapping(jobs.get("publish-pypi"), f"{path}: jobs.publish-pypi")
    steps = sequence(job.get("steps"), f"{path}: jobs.publish-pypi.steps")

    publish_steps: list[tuple[int, dict[str, Any]]] = []
    artifact_downloads: list[tuple[int, dict[str, Any], dict[str, Any]]] = []

    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.publish-pypi.steps[{index}]")
        action = action_name(step)
        if action == "pypa/gh-action-pypi-publish":
            publish_steps.append((index, step))
        if action != "actions/download-artifact":
            continue

        with_block = step.get("with", {})
        with_map = mapping(with_block, f"{path}: {step_label(index, step)} with")
        artifact_downloads.append((index, step, with_map))

    if len(publish_steps) != 1:
        fail(f"{path}: publish-pypi must have exactly one pypa/gh-action-pypi-publish step")

    publish_index, publish_step = publish_steps[0]
    publish_with = mapping(
        publish_step.get("with", {}), f"{path}: {step_label(publish_index, publish_step)} with"
    )
    if norm_path(publish_with.get("packages-dir")) != "dist":
        fail(f"{path}: PyPI publish step must upload packages-dir: dist")
    if not boolish_true(publish_with.get("skip-existing")):
        fail(
            f"{path}: PyPI publish step must set skip-existing: true so a recovery "
            "rerun is idempotent after PyPI has already accepted the version"
        )

    wheels: list[int] = []
    sdists: list[int] = []
    for index, step, with_map in artifact_downloads:
        label = step_label(index, step)
        artifact_path = norm_path(with_map.get("path"))
        if artifact_path != "dist":
            fail(
                f"{path}: {label} downloads artifacts to {artifact_path or 'the default path'!r}; "
                "publish-pypi may only download wheels-* and sdist into dist"
            )
        if index > publish_index:
            fail(f"{path}: {label} downloads into dist after the PyPI publish step")

        name = with_map.get("name")
        pattern = with_map.get("pattern")
        is_wheels = (
            pattern == "wheels-*"
            and empty(name)
            and boolish_true(with_map.get("merge-multiple"))
        )
        is_sdist = name == "sdist" and empty(pattern)

        if is_wheels:
            wheels.append(index)
            continue
        if is_sdist:
            sdists.append(index)
            continue

        fail(
            f"{path}: {label} downloads into dist but is not the allowed "
            "'pattern: wheels-*' or 'name: sdist' artifact"
        )

    if len(wheels) != 1:
        fail(f"{path}: publish-pypi must download exactly one wheels-* artifact set into dist")
    if len(sdists) != 1:
        fail(f"{path}: publish-pypi must download exactly one sdist artifact into dist")


def check_publish_crate(workflow: dict[str, Any], path: str) -> None:
    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    job = mapping(jobs.get("publish-crate"), f"{path}: jobs.publish-crate")
    steps = sequence(job.get("steps"), f"{path}: jobs.publish-crate.steps")

    crate_downloads: list[tuple[int, dict[str, Any], dict[str, Any]]] = []

    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.publish-crate.steps[{index}]")
        if action_name(step) != "actions/download-artifact":
            continue
        with_block = step.get("with", {})
        with_map = mapping(with_block, f"{path}: {step_label(index, step)} with")
        if with_map.get("name") == "dist-crate":
            crate_downloads.append((index, step, with_map))

    if len(crate_downloads) != 1:
        fail(f"{path}: publish-crate must download exactly one dist-crate artifact")

    index, step, with_map = crate_downloads[0]
    label = step_label(index, step)
    artifact_path = norm_path(with_map.get("path"))
    if artifact_path != "${{ runner.temp }}/attested":
        fail(
            f"{path}: {label} downloads dist-crate to {artifact_path or 'the default path'!r}; "
            "it must use ${{ runner.temp }}/attested so cargo package sees a clean checkout"
        )

    verify_step_names = {
        "Verify byte-identity vs the attested .crate",
        "Post-publish byte-identity (download from crates.io == attested)",
    }
    verify_steps: list[dict[str, Any]] = []
    found_names: set[str] = set()
    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.publish-crate.steps[{index}]")
        name = step.get("name")
        if name in verify_step_names:
            verify_steps.append(step)
            found_names.add(name)
    if found_names != verify_step_names:
        fail(f"{path}: publish-crate must have both attested .crate verification steps")

    for step in verify_steps:
        name = step.get("name")
        run = step.get("run")
        if not isinstance(run, str):
            fail(f"{path}: publish-crate step {name!r} must be a run step")
        if "${RUNNER_TEMP}/attested/ordvec-${VERSION}.crate" not in run:
            fail(
                f"{path}: publish-crate step {name!r} must read the attested .crate "
                "from ${RUNNER_TEMP}/attested"
            )


def main() -> None:
    workflow = load_workflow(WORKFLOW_PATH)
    check_publish_crate(workflow, WORKFLOW_PATH)
    check_publish_pypi(workflow, WORKFLOW_PATH)


if __name__ == "__main__":
    main()
