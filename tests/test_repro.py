"""Tests for hpccoworklab.repro (HPC-25): module-env parsing + snapshot assembly.
job_resources is live-only (verified on Libra, HPC-10)."""

from hpccoworklab.repro import loaded_modules, build_snapshot


def test_loaded_modules_colon_separated():
    env = {"LOADEDMODULES": "gcc/9.2.0:cuda/11.4:openmpi/4.1.0"}
    assert loaded_modules(env) == ["gcc/9.2.0", "cuda/11.4", "openmpi/4.1.0"]


def test_loaded_modules_empty_absent():
    assert loaded_modules({}) == []
    assert loaded_modules({"LOADEDMODULES": ""}) == []


def test_build_snapshot_includes_facts():
    txt = build_snapshot(jobid="123", modules=["gcc/9", "cuda/11"],
                         resources="123|gpu|COMPLETED|cpu=4,mem=8G|8G|...",
                         cluster="SLU Libra", python="3.12.0", host="comp01")
    assert "123" in txt
    assert "SLU Libra" in txt
    assert "python: 3.12.0" in txt
    assert "host: comp01" in txt
    assert "cuda/11" in txt
    assert "cpu=4,mem=8G" in txt


def test_build_snapshot_no_modules_notes_none():
    txt = build_snapshot(jobid=None, modules=[])
    assert "none detected" in txt.lower()
    assert "current session" in txt
