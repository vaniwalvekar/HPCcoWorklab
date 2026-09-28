"""Tests for HPC-31: log-content progress detection.

Covers read_log_progress (pure, real files in tmp_path), the verdict it drives
in assess_job_health, and health_exit_code. No live Slurm needed - squeue/sstat
inputs are dicts.
"""

import os
import time

from hpclint.monitor import (
    read_log_progress,
    assess_job_health,
    health_exit_code,
    _format_eta,
    _last_progress,
)

PATTERNS = [{"regex": r"step\s+(\d+)", "total_regex": r"of\s+(\d+)"}]
COMPLETION = [r"\b(done|finished|completed)\b"]


def _write(tmp_path, text, age_minutes=0):
    p = tmp_path / "job.out"
    p.write_text(text)
    if age_minutes:
        old = time.time() - age_minutes * 60
        os.utime(p, (old, old))
    return str(p)


def test_missing_log_not_available(tmp_path):
    info = read_log_progress(str(tmp_path / "nope.out"), patterns=PATTERNS)
    assert info["available"] is False
    assert info["has_markers"] is False


def test_markers_present_with_total_and_percent(tmp_path):
    path = _write(tmp_path, "step 10 of 100\nstep 40 of 100\n")
    info = read_log_progress(path, patterns=PATTERNS)
    assert info["has_markers"] is True
    assert info["latest"] == 40          # LAST occurrence wins
    assert info["total"] == 100
    assert info["percent"] == 40.0
    assert info["stalled"] is False      # just written


def test_no_markers_when_nothing_matches(tmp_path):
    path = _write(tmp_path, "loading model...\nwarmup\n")
    info = read_log_progress(path, patterns=PATTERNS)
    assert info["has_markers"] is False


def test_completion_marker_sets_finished(tmp_path):
    path = _write(tmp_path, "step 100 of 100\nJob completed\n")
    info = read_log_progress(path, patterns=PATTERNS, completion_markers=COMPLETION)
    assert info["finished"] is True


def test_old_log_is_stalled(tmp_path):
    path = _write(tmp_path, "step 5 of 100\n", age_minutes=90)
    info = read_log_progress(path, patterns=PATTERNS, stale_after_minutes=30)
    assert info["stalled"] is True


def test_last_progress_picks_latest():
    text = "step 1\nstep 7 of 50\nstep 9\n"
    value, total = _last_progress(text, PATTERNS)
    assert value == 9 and total == 50


def test_format_eta():
    assert _format_eta(None) == "unknown"
    assert _format_eta(90) == "1m30s"
    assert _format_eta(3725) == "1h02m"


# --- verdicts driven by log content ----------------------------------------

def _running(qtime="02:00:00"):
    return {"state": "RUNNING", "time_used": qtime, "time_limit": "1-00:00:00",
            "nodes": 1, "cpus": 8}


def test_verdict_progressing_log_is_healthy(tmp_path):
    info = read_log_progress(_write(tmp_path, "step 40 of 100\n"), patterns=PATTERNS)
    verdict = assess_job_health(_running(), {"exists": True}, {"ave_cpu": "01:50:00"}, info)
    assert "advancing" in verdict.lower()


def test_verdict_stalled_log_with_cpu_is_loop(tmp_path):
    info = read_log_progress(_write(tmp_path, "step 40 of 100\n", age_minutes=90),
                             patterns=PATTERNS, stale_after_minutes=30)
    verdict = assess_job_health(_running(), {"exists": True}, {"ave_cpu": "01:50:00"}, info)
    assert "loop" in verdict.lower() or "spinning" in verdict.lower()


def test_verdict_stalled_log_idle_is_blocked(tmp_path):
    info = read_log_progress(_write(tmp_path, "step 40 of 100\n", age_minutes=90),
                             patterns=PATTERNS, stale_after_minutes=30)
    verdict = assess_job_health(_running(), {"exists": True}, {"ave_cpu": "00:00:02"}, info)
    assert "blocked" in verdict.lower()


def test_verdict_finished_log(tmp_path):
    info = read_log_progress(_write(tmp_path, "step 100 of 100\ndone\n"),
                             patterns=PATTERNS, completion_markers=COMPLETION)
    verdict = assess_job_health(_running(), {"exists": True}, {"ave_cpu": "01:50:00"}, info)
    assert "finished" in verdict.lower()


def test_no_markers_falls_back_to_directory_check(tmp_path):
    info = read_log_progress(_write(tmp_path, "no markers here\n"), patterns=PATTERNS)
    verdict = assess_job_health(_running(), {"exists": True, "file_count": 0, "stale": False},
                                {"ave_cpu": "01:50:00"}, info)
    assert "output directory" in verdict.lower()


# --- exit codes from the log dimension -------------------------------------

def test_exit_code_progressing_is_0(tmp_path):
    info = read_log_progress(_write(tmp_path, "step 40 of 100\n"), patterns=PATTERNS)
    assert health_exit_code(_running(), {"exists": True}, {"ave_cpu": "01:50:00"}, info) == 0


def test_exit_code_stalled_is_1(tmp_path):
    info = read_log_progress(_write(tmp_path, "step 40 of 100\n", age_minutes=90),
                             patterns=PATTERNS, stale_after_minutes=30)
    assert health_exit_code(_running(), {"exists": True}, {"ave_cpu": "01:50:00"}, info) == 1


def test_exit_code_finished_is_0(tmp_path):
    info = read_log_progress(_write(tmp_path, "done\n"),
                             patterns=PATTERNS, completion_markers=COMPLETION)
    assert health_exit_code(_running(), {"exists": True}, {"ave_cpu": "01:50:00"}, info) == 0
