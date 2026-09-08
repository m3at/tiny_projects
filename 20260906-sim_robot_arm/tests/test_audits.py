"""Audit entry points must not report success without input data."""

import importlib.util
from pathlib import Path

import pytest


def test_dataset_audit_rejects_empty_cache(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / "scripts/dataset_audit.py"
    spec = importlib.util.spec_from_file_location("dataset_audit", source)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    monkeypatch.setattr(audit, "DATA", tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="No KanjiVG SVGs found; run make data"):
        audit.main()
    assert not (tmp_path / "runs").exists()
