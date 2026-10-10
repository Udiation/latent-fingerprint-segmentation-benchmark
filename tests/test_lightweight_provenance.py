from __future__ import annotations

from segbench.provenance import run_provenance


def test_lightweight_provenance_has_stable_schema(tmp_path):
    provenance = run_provenance(
        ["predict_classical.py", "--help"],
        capture_timing="process_start",
        repository=tmp_path,
    )
    assert provenance["command_argv"] == ["predict_classical.py", "--help"]
    assert provenance["capture_timing"] == "process_start"
    assert provenance["git_commit"] == "unavailable"
    assert provenance["git_dirty"] is None
    assert set(provenance["versions"]) == {
        "python",
        "torch",
        "numpy",
        "opencv",
        "pandas",
        "pyyaml",
    }
