import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "tools" / "verify_index_authority.py"
RECEIPT = ROOT / "examples" / "caif" / "trec-covid-sign-rq2.index-authority.json"
POLICY = ROOT / "policies" / "index-authority.default-policy.json"

def run_verify(path):
    return subprocess.run(
        [sys.executable, str(VERIFY), str(path), "--policy", str(POLICY)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

def test_valid_receipt_passes():
    result = run_verify(RECEIPT)
    assert result.returncode == 0, result.stderr + result.stdout

def test_missing_required_field_rejected(tmp_path):
    data = json.loads(RECEIPT.read_text())
    data.pop("evidence")
    bad = tmp_path / "missing-evidence.json"
    bad.write_text(json.dumps(data))
    result = run_verify(bad)
    assert result.returncode != 0

def test_metric_tampering_rejected(tmp_path):
    data = json.loads(RECEIPT.read_text())
    data["economics"]["storage_reduction_x"] = 999
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(data))
    result = run_verify(bad)
    assert result.returncode != 0

def test_decision_mismatch_exit_code_3(tmp_path):
    data = json.loads(RECEIPT.read_text())
    data["decision"]["recommended"] = "DENY_UNSCOPED_CLAIM"
    bad = tmp_path / "decision-mismatch.json"
    bad.write_text(json.dumps(data))
    result = run_verify(bad)
    assert result.returncode == 3

def test_ifc_disabled_rejected(tmp_path):
    data = json.loads(RECEIPT.read_text())
    data["ifc"]["enabled"] = False
    bad = tmp_path / "ifc-disabled.json"
    bad.write_text(json.dumps(data))
    result = run_verify(bad)
    assert result.returncode != 0
    assert "ifc.enabled must be true" in result.stderr

def test_ifc_empty_compute_path_rejected(tmp_path):
    data = json.loads(RECEIPT.read_text())
    data["ifc"]["compute_path"] = ""
    bad = tmp_path / "ifc-empty-path.json"
    bad.write_text(json.dumps(data))
    result = run_verify(bad)
    assert result.returncode != 0
    assert "ifc.compute_path" in result.stderr

def test_nan_metrics_rejected(tmp_path):
    bad = tmp_path / "nan.json"
    text = RECEIPT.read_text().replace('"storage_reduction_x":', '"storage_reduction_x": NaN, "old_storage_reduction_x":', 1)
    bad.write_text(text)
    result = run_verify(bad)
    assert result.returncode != 0
    assert "non-finite" in result.stderr

def test_blank_scope_entries_rejected(tmp_path):
    data = json.loads(RECEIPT.read_text())
    data["scope"]["applies_to"] = [""]
    data["scope"]["does_not_claim"] = ["  "]
    data["limitations"] = [""]
    bad = tmp_path / "blank-scope.json"
    bad.write_text(json.dumps(data))
    result = run_verify(bad)
    assert result.returncode != 0

def test_significant_quality_improvement_allowed(tmp_path):
    data = json.loads(RECEIPT.read_text())
    data["evidence"]["candidate_score"] = data["evidence"]["baseline_score"] + 0.05
    data["evidence"]["delta_vs_baseline"] = 0.05
    data["evidence"]["within_bootstrap_noise"] = False
    data["decision"]["recommended"] = "ALLOW_INDEX_FIRST"
    bad = tmp_path / "quality-improvement.json"
    bad.write_text(json.dumps(data))
    result = run_verify(bad)
    assert result.returncode == 0, result.stderr + result.stdout

def test_parallel_claim_requires_concrete_hnsw_evidence(tmp_path):
    data = json.loads(RECEIPT.read_text())
    data["scope"]["applies_to"] = ["highly parallel threaded serving"]
    data["evidence"]["compared_against_hnsw"] = True
    data["evidence"]["hnsw_comparison"] = {}
    data["decision"]["recommended"] = "ALLOW_INDEX_FIRST"
    bad = tmp_path / "empty-hnsw.json"
    bad.write_text(json.dumps(data))
    result = run_verify(bad)
    assert result.returncode == 3
    assert "REQUIRE_HNSW_COMPARISON" in result.stderr + result.stdout

def test_single_query_production_does_not_require_hnsw(tmp_path):
    data = json.loads(RECEIPT.read_text())
    data["scope"]["applies_to"] = ["single-query production serving"]
    data["evidence"].pop("hnsw_comparison", None)
    data["evidence"].pop("compared_against_hnsw", None)
    data["decision"]["recommended"] = "ALLOW_INDEX_FIRST"
    bad = tmp_path / "single-query-prod.json"
    bad.write_text(json.dumps(data))
    result = run_verify(bad)
    assert result.returncode == 0, result.stderr + result.stdout
