#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

RECEIPT_SCHEMA = "ordvec.index_authority.v0.1"
POLICY_SCHEMA = "ordvec.index_authority.verifier_policy.v0.1"

VALID_DECISIONS = {
    "ALLOW_INDEX_FIRST",
    "REQUIRE_DENSE_FALLBACK",
    "REQUIRE_HNSW_COMPARISON",
    "DENY_UNSCOPED_CLAIM",
}

REQUIRED_TOP_LEVEL = [
    "schema",
    "subject",
    "baseline",
    "ifc",
    "evidence",
    "economics",
    "decision",
    "scope",
    "limitations",
]


def die(msg, code=2):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def reject_json_constant(value):
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def load_json(path: Path, label: str):
    try:
        return json.loads(path.read_text(), parse_constant=reject_json_constant)
    except Exception as e:
        die(f"cannot read {label}: {e}")


def sha(obj):
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(b).hexdigest()


def require_keys(obj, keys, label):
    if not isinstance(obj, dict):
        die(f"{label} must be an object")

    missing = [k for k in keys if k not in obj]
    if missing:
        die(f"{label} missing required field(s): {', '.join(missing)}")


def require_number(obj, key, label):
    value = obj.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        die(f"{label}.{key} must be a number")
    value = float(value)
    if not math.isfinite(value):
        die(f"{label}.{key} must be finite")
    return value


def require_list(obj, key, label):
    value = obj.get(key)
    if not isinstance(value, list):
        die(f"{label}.{key} must be a list")
    return value


def require_nonempty_string_list(obj, key, label):
    value = require_list(obj, key, label)
    cleaned = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            die(f"{label}.{key}[{i}] must be a non-empty string")
        cleaned.append(item.strip())
    if not cleaned:
        die(f"{label}.{key} must contain at least one non-empty string")
    return cleaned


def require_ifc_enabled(ifc):
    if not isinstance(ifc, dict):
        die("ifc must be an object")
    if ifc.get("enabled") is not True:
        die("ifc.enabled must be true for an index authority receipt")

    compute_path = ifc.get("compute_path")
    if isinstance(compute_path, str):
        if not compute_path.strip():
            die("ifc.compute_path must be non-empty")
    elif isinstance(compute_path, list):
        if not compute_path or any(not isinstance(x, str) or not x.strip() for x in compute_path):
            die("ifc.compute_path must contain non-empty string entries")
    else:
        die("ifc.compute_path must be a non-empty string or list of strings")


def has_concrete_hnsw_comparison(evidence):
    h = evidence.get("hnsw_comparison")
    if not isinstance(h, dict) or not h:
        return False

    artifact = h.get("artifact") or h.get("artifact_ref") or h.get("evidence_ref") or h.get("receipt_ref")
    has_artifact = isinstance(artifact, str) and bool(artifact.strip())

    metric_pairs = [
        ("baseline_latency_ms", "candidate_latency_ms"),
        ("baseline_qps", "candidate_qps"),
        ("baseline_recall", "candidate_recall"),
        ("baseline_score", "candidate_score"),
    ]

    has_metric_pair = any(
        isinstance(h.get(a), (int, float))
        and isinstance(h.get(b), (int, float))
        and math.isfinite(float(h.get(a)))
        and math.isfinite(float(h.get(b)))
        for a, b in metric_pairs
    )

    nested_latency = h.get("single_query_latency_ms")
    has_nested_latency = (
        isinstance(nested_latency, dict)
        and isinstance(nested_latency.get("baseline"), (int, float))
        and isinstance(nested_latency.get("candidate"), (int, float))
        and math.isfinite(float(nested_latency.get("baseline")))
        and math.isfinite(float(nested_latency.get("candidate")))
    )

    return has_artifact and (has_metric_pair or has_nested_latency)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt", type=Path)
    ap.add_argument(
        "--policy",
        type=Path,
        default=Path("policies/index-authority.default-policy.json"),
        help="Verifier-owned acceptance policy. Receipt policy fields are ignored.",
    )
    args = ap.parse_args()

    r = load_json(args.receipt, "receipt")
    policy = load_json(args.policy, "policy")

    require_keys(r, REQUIRED_TOP_LEVEL, "receipt")

    if r["schema"] != RECEIPT_SCHEMA:
        die(f"bad receipt schema: {r['schema']}")

    if policy.get("schema") != POLICY_SCHEMA:
        die(f"bad policy schema: {policy.get('schema')}")

    require_keys(
        policy,
        [
            "min_storage_reduction_x",
            "min_single_query_speedup_x",
            "max_quality_delta_loss",
            "require_scope",
            "require_limitations",
            "require_hnsw_comparison_for_parallel_claims",
        ],
        "policy",
    )

    e = r["evidence"]
    econ = r["economics"]
    base = r["baseline"]
    ifc = r["ifc"]
    decision_obj = r["decision"]
    scope = r["scope"]
    limitations = r["limitations"]

    require_ifc_enabled(ifc)

    require_keys(e, ["candidate_score", "baseline_score", "delta_vs_baseline", "within_bootstrap_noise"], "evidence")
    require_keys(base, ["mode", "bytes_per_vector"], "baseline")
    require_keys(
        econ,
        ["candidate_bytes_per_vector", "storage_reduction_x", "single_query_latency_ms", "single_query_speedup_x"],
        "economics",
    )
    require_keys(econ["single_query_latency_ms"], ["baseline", "candidate"], "economics.single_query_latency_ms")
    require_keys(decision_obj, ["recommended"], "decision")
    require_keys(scope, ["applies_to", "does_not_claim"], "scope")

    recommended = decision_obj["recommended"]
    if recommended not in VALID_DECISIONS:
        die(f"invalid recommended decision: {recommended}")

    candidate_score = require_number(e, "candidate_score", "evidence")
    baseline_score = require_number(e, "baseline_score", "evidence")
    declared_delta = require_number(e, "delta_vs_baseline", "evidence")

    baseline_bytes = require_number(base, "bytes_per_vector", "baseline")
    candidate_bytes = require_number(econ, "candidate_bytes_per_vector", "economics")
    declared_storage = require_number(econ, "storage_reduction_x", "economics")

    latency = econ["single_query_latency_ms"]
    baseline_latency = require_number(latency, "baseline", "economics.single_query_latency_ms")
    candidate_latency = require_number(latency, "candidate", "economics.single_query_latency_ms")
    declared_speedup = require_number(econ, "single_query_speedup_x", "economics")

    if baseline_bytes <= 0 or candidate_bytes <= 0:
        die("bytes_per_vector values must be positive")
    if baseline_latency <= 0 or candidate_latency <= 0:
        die("latency values must be positive")

    expected_delta = candidate_score - baseline_score
    if abs(declared_delta - expected_delta) > 0.0001:
        die("delta_vs_baseline mismatch")

    expected_storage = baseline_bytes / candidate_bytes
    if abs(declared_storage - expected_storage) > 0.02:
        die("storage_reduction_x mismatch")

    expected_speedup = baseline_latency / candidate_latency
    if abs(declared_speedup - expected_speedup) > 0.02:
        die("single_query_speedup_x mismatch")

    applies_to = require_nonempty_string_list(scope, "applies_to", "scope")
    does_not_claim = require_nonempty_string_list(scope, "does_not_claim", "scope")

    if not isinstance(limitations, list):
        die("limitations must be a list")
    for i, item in enumerate(limitations):
        if not isinstance(item, str) or not item.strip():
            die(f"limitations[{i}] must be a non-empty string")

    decision = "ALLOW_INDEX_FIRST"

    scope_missing = not applies_to or not does_not_claim
    limitations_missing = not limitations

    quality_loss = max(0.0, baseline_score - candidate_score)
    outside_bootstrap_noise = e["within_bootstrap_noise"] is not True
    quality_too_low = quality_loss > float(policy["max_quality_delta_loss"]) and outside_bootstrap_noise

    economics_too_weak = (
        declared_storage < float(policy["min_storage_reduction_x"])
        or declared_speedup < float(policy["min_single_query_speedup_x"])
    )

    claims_text = " ".join(str(x).lower() for x in applies_to)
    claims_parallel_or_production = any(
        marker in claims_text
        for marker in [
            "parallel",
            "threaded",
            "multi-thread",
            "multithread",
            "concurrent",
            "throughput",
            "high-qps",
            "high qps"
        ]
    )

    has_hnsw_comparison = has_concrete_hnsw_comparison(e)

    if policy["require_scope"] and scope_missing:
        decision = "DENY_UNSCOPED_CLAIM"
    elif policy["require_limitations"] and limitations_missing:
        decision = "DENY_UNSCOPED_CLAIM"
    elif quality_too_low or economics_too_weak:
        decision = "REQUIRE_DENSE_FALLBACK"
    elif (
        policy["require_hnsw_comparison_for_parallel_claims"]
        and claims_parallel_or_production
        and not has_hnsw_comparison
    ):
        decision = "REQUIRE_HNSW_COMPARISON"

    print(f"decision: {decision}")
    print(f"mode: {r['subject'].get('mode')}")
    print(f"baseline: {base.get('mode')}")
    print(f"quality_within_bootstrap_noise: {str(e['within_bootstrap_noise']).lower()}")
    print(f"storage_reduction: {declared_storage}x")
    print(f"single_query_speedup: {declared_speedup}x")
    print(f"receipt_hash: {sha(r)}")
    print(f"policy_hash: {sha(policy)}")

    if decision != recommended:
        die(f"decision mismatch: receipt recommends {recommended}, verifier computed {decision}", code=3)

    print("verified: true")


if __name__ == "__main__":
    main()
