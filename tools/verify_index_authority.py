#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

def die(msg, code=2):
    print("ERROR:", msg, file=sys.stderr)
    raise SystemExit(code)

def sha(obj):
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(b).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt", type=Path)
    args = ap.parse_args()

    try:
        r = json.loads(args.receipt.read_text())
    except Exception as e:
        die(f"cannot read receipt: {e}")

    for k in ["schema","subject","baseline","ifc","evidence","economics","decision","scope","limitations"]:
        if k not in r:
            die(f"missing field {k}")

    if r["schema"] != "ordvec.index_authority.v0.1":
        die("bad schema")

    e = r["evidence"]
    econ = r["economics"]
    base = r["baseline"]
    policy = r["decision"]["policy"]

    expected_delta = e["candidate_score"] - e["baseline_score"]
    if abs(e["delta_vs_baseline"] - expected_delta) > 0.0001:
        die("delta_vs_baseline mismatch")

    expected_storage = base["bytes_per_vector"] / econ["candidate_bytes_per_vector"]
    if abs(econ["storage_reduction_x"] - expected_storage) > 0.02:
        die("storage_reduction_x mismatch")

    expected_speedup = econ["single_query_latency_ms"]["baseline"] / econ["single_query_latency_ms"]["candidate"]
    if abs(econ["single_query_speedup_x"] - expected_speedup) > 0.02:
        die("single_query_speedup_x mismatch")

    decision = "ALLOW_INDEX_FIRST"
    if policy["require_quality_within_bootstrap_noise"] and not e["within_bootstrap_noise"]:
        decision = "REQUIRE_DENSE_FALLBACK"
    if econ["storage_reduction_x"] < policy["min_storage_reduction_x"]:
        decision = "REQUIRE_DENSE_FALLBACK"
    if econ["single_query_speedup_x"] < policy["min_single_query_speedup_x"]:
        decision = "REQUIRE_DENSE_FALLBACK"
    if policy["require_scope"] and (not r["scope"]["applies_to"] or not r["scope"]["does_not_claim"]):
        decision = "DENY_UNSCOPED_CLAIM"
    if policy["require_limitations"] and not r["limitations"]:
        decision = "DENY_UNSCOPED_CLAIM"

    print(f"decision: {decision}")
    print(f"mode: {r['subject']['mode']}")
    print(f"baseline: {base['mode']}")
    print(f"quality_within_bootstrap_noise: {str(e['within_bootstrap_noise']).lower()}")
    print(f"storage_reduction: {econ['storage_reduction_x']}x")
    print(f"single_query_speedup: {econ['single_query_speedup_x']}x")
    print(f"receipt_hash: {sha(r)}")

    if decision != r["decision"]["recommended"]:
        die(f"declared decision {r['decision']['recommended']} does not match computed decision {decision}", 3)

if __name__ == "__main__":
    main()
