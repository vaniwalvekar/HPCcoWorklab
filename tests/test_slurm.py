"""
Tests for hpccoworklab.slurm and the Slurm-command wrappers' error handling (HPC-5).

These exercise the failure paths we cannot hit without a real cluster by
patching subprocess.run: missing binary, timeout, and non-zero exit must all
become a clean SlurmCommandError, while "empty output, exit 0" stays a normal
"not found" result (None), never an error.
"""

import subprocess

import pytest

import hpccoworklab.slurm as slurm
from hpccoworklab.slurm import run_slurm, SlurmCommandError


class _FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_slurm_returns_stdout_on_success(monkeypatch):
    monkeypatch.setattr(slurm.subprocess, "run",
                        lambda *a, **k: _FakeResult(0, "12345|RUNNING\n"))
    assert run_slurm(["squeue", "-j", "1"]) == "12345|RUNNING\n"


def test_run_slurm_empty_stdout_zero_exit_is_not_an_error(monkeypatch):
    monkeypatch.setattr(slurm.subprocess, "run", lambda *a, **k: _FakeResult(0, ""))
    assert run_slurm(["squeue", "-j", "1"]) == ""


def test_run_slurm_nonzero_exit_raises_with_stderr(monkeypatch):
    monkeypatch.setattr(slurm.subprocess, "run",
                        lambda *a, **k: _FakeResult(1, "", "slurm_load_jobs error: Invalid job id"))
    with pytest.raises(SlurmCommandError) as exc:
        run_slurm(["squeue", "-j", "abc"])
    assert "Invalid job id" in str(exc.value)


def test_run_slurm_missing_binary_raises(monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError()
    monkeypatch.setattr(slurm.subprocess, "run", _boom)
    with pytest.raises(SlurmCommandError) as exc:
        run_slurm(["squeue", "-j", "1"])
    assert "not found" in str(exc.value).lower()


def test_run_slurm_timeout_raises(monkeypatch):
    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(["sacct"], timeout=1)
    monkeypatch.setattr(slurm.subprocess, "run", _boom)
    with pytest.raises(SlurmCommandError) as exc:
        run_slurm(["sacct", "-j", "1"])
    assert "timed out" in str(exc.value).lower()


def test_run_squeue_propagates_command_error(monkeypatch):
    import hpccoworklab.monitor as monitor
    monkeypatch.setattr(slurm.subprocess, "run",
                        lambda *a, **k: _FakeResult(2, "", "controller down"))
    with pytest.raises(SlurmCommandError):
        monitor.run_squeue(12345)


def test_run_squeue_not_found_returns_none(monkeypatch):
    import hpccoworklab.monitor as monitor
    monkeypatch.setattr(slurm.subprocess, "run", lambda *a, **k: _FakeResult(0, ""))
    assert monitor.run_squeue(999999) is None
