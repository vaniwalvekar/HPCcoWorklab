"""
Tests for hpccoworklab.monitor — parsing and health-assessment logic only.

These test the pure functions against realistic fixture text, not the
actual subprocess calls (run_squeue/run_sstat), since that needs a real
Slurm cluster. Those two functions get exercised for real once this is
tested on Libra, per the project's plan to verify everything end-to-end
at that point rather than piecemeal.
"""

import os
import time

from hpccoworklab.monitor import (
    parse_squeue_line,
    parse_sstat_line,
    check_output_activity,
    assess_job_health,
    _parse_cpu_time_to_seconds,
    select_squeue_line,
    health_exit_code,
)

# --- squeue multi-line selection (HPC-2: array/step jobs) ------------------

def test_select_squeue_single_line():
    out = "12345|RUNNING|01:23:45|1-00:00:00|2|16\n"
    assert select_squeue_line(out, 12345) == "12345|RUNNING|01:23:45|1-00:00:00|2|16"

def test_select_squeue_picks_main_among_steps():
    out = (
        "12345|RUNNING|01:23:45|1-00:00:00|2|16\n"
        "12345.batch|RUNNING|01:23:44|1-00:00:00|2|16\n"
        "12345.extern|RUNNING|01:23:44|1-00:00:00|2|16\n"
    )
    assert select_squeue_line(out, 12345).startswith("12345|")

def test_select_squeue_array_falls_back_to_first_task():
    out = (
        "12345_0|RUNNING|00:10:00|01:00:00|1|8\n"
        "12345_1|PENDING|00:00:00|01:00:00|1|8\n"
    )
    assert select_squeue_line(out, 12345).startswith("12345_0|")

def test_select_squeue_empty_returns_none():
    assert select_squeue_line("", 12345) is None
    assert select_squeue_line("   \n  \n", 12345) is None

# --- CPU/elapsed time parsing (HPC-1: D-HH:MM:SS crash) --------------------

def test_parse_time_hhmmss():
    assert _parse_cpu_time_to_seconds("01:23:45") == 5025

def test_parse_time_mmss():
    assert _parse_cpu_time_to_seconds("23:45") == 1425

def test_parse_time_seconds_only():
    assert _parse_cpu_time_to_seconds("45") == 45

def test_parse_time_with_days_no_crash():
    # '2-03:04:05' == 2 days, 3h, 4m, 5s  -> the crash case from HPC-1
    assert _parse_cpu_time_to_seconds("2-03:04:05") == 2 * 86400 + 3 * 3600 + 4 * 60 + 5

def test_parse_time_days_only():
    assert _parse_cpu_time_to_seconds("5-00:00:00") == 5 * 86400

def test_parse_time_garbage_returns_none():
    assert _parse_cpu_time_to_seconds("N/A") is None
    assert _parse_cpu_time_to_seconds("bogus") is None

def test_assess_health_long_job_does_not_crash():
    info = {"jobid": "9", "state": "RUNNING", "time_used": "2-03:04:05",
            "nodes": "1", "cpus": "8"}
    activity = {"exists": True, "file_count": 0, "stale": False, "minutes_since_change": None}
    verdict = assess_job_health(info, activity)
    assert isinstance(verdict, str) and "RUNNING" in verdict


# --- squeue parsing ---------------------------------------------------------

def test_parse_squeue_line_running_job():
    line = "12345|RUNNING|01:23:45|1-00:00:00|2|16"
    info = parse_squeue_line(line)
    assert info == {
        "jobid": "12345",
        "state": "RUNNING",
        "time_used": "01:23:45",
        "time_limit": "1-00:00:00",
        "nodes": 2,
        "cpus": 16,
    }


# --- HPC-7: padded/odd field robustness ------------------------------------

def test_parse_squeue_line_strips_padded_fields():
    line = " 12345 | RUNNING | 01:23:45 | 1-00:00:00 | 2 | 16 "
    info = parse_squeue_line(line)
    assert info["state"] == "RUNNING"
    assert info["nodes"] == 2
    assert info["cpus"] == 16

def test_parse_squeue_line_nonnumeric_counts_do_not_crash():
    # Some builds render %C as 'allocated/idle/other/available'.
    line = "12345|RUNNING|01:23:45|1-00:00:00|1|16/0/0/0"
    info = parse_squeue_line(line)
    assert info["cpus"] == 0
    assert info["state"] == "RUNNING"


def test_parse_squeue_line_pending_job():
    line = "99999|PENDING|0:00|02:00:00|1|4"
    info = parse_squeue_line(line)
    assert info["state"] == "PENDING"


def test_parse_squeue_line_empty_returns_none():
    assert parse_squeue_line("") is None
    assert parse_squeue_line(None) is None


def test_parse_squeue_line_malformed_returns_none():
    assert parse_squeue_line("not|enough|fields") is None


# --- sstat parsing -----------------------------------------------------------

def test_parse_sstat_line():
    line = "12345.batch|00:45:12|2048576K|1024288K"
    info = parse_sstat_line(line)
    assert info == {
        "jobid": "12345.batch",
        "ave_cpu": "00:45:12",
        "max_rss": "2048576K",
        "ave_rss": "1024288K",
    }


def test_parse_sstat_line_empty_returns_none():
    assert parse_sstat_line("") is None
    assert parse_sstat_line(None) is None


# --- output directory activity ----------------------------------------------

def test_output_activity_missing_dir(tmp_path):
    result = check_output_activity(str(tmp_path / "does_not_exist"))
    assert result["exists"] is False


def test_output_activity_fresh_file_not_stale(tmp_path):
    f = tmp_path / "output.log"
    f.write_text("progress...")
    result = check_output_activity(str(tmp_path), stale_after_minutes=30)
    assert result["exists"] is True
    assert result["stale"] is False


def test_output_activity_old_file_is_stale(tmp_path):
    f = tmp_path / "output.log"
    f.write_text("progress...")
    old_time = time.time() - (60 * 60)  # 1 hour ago
    os.utime(f, (old_time, old_time))
    result = check_output_activity(str(tmp_path), stale_after_minutes=30)
    assert result["exists"] is True
    assert result["stale"] is True


def test_output_activity_empty_dir_is_stale(tmp_path):
    result = check_output_activity(str(tmp_path), stale_after_minutes=30)
    assert result["exists"] is True
    assert result["stale"] is True
    assert result["most_recent_file"] is None
    assert result["file_count"] == 0


def test_output_activity_reports_file_count(tmp_path):
    (tmp_path / "a.log").write_text("x")
    (tmp_path / "b.log").write_text("y")
    result = check_output_activity(str(tmp_path))
    assert result["file_count"] == 2


# --- combined health verdict --------------------------------------------------

def test_assess_health_job_not_found():
    verdict = assess_job_health(None, {"exists": False})
    assert "not found" in verdict.lower()


def test_assess_health_not_yet_running():
    squeue_info = {"jobid": "1", "state": "PENDING", "time_used": "0:00", "time_limit": "1:00:00", "nodes": 1, "cpus": 1}
    verdict = assess_job_health(squeue_info, {"exists": False})
    assert "PENDING" in verdict


def test_assess_health_busy_but_silent_flagged():
    # Regression target: this is the exact pattern we set out to catch —
    # real accrued CPU time, but the output directory has gone stale.
    squeue_info = {"jobid": "1", "state": "RUNNING", "time_used": "02:15:00", "time_limit": "1-00:00:00", "nodes": 1, "cpus": 8}
    activity_info = {"exists": True, "stale": True, "minutes_since_change": 90, "most_recent_file": "/data/out.log"}
    verdict = assess_job_health(squeue_info, activity_info)
    assert "busy-but-silent" in verdict or "stuck" in verdict.lower()


def test_assess_health_healthy_job_not_flagged():
    squeue_info = {"jobid": "1", "state": "RUNNING", "time_used": "02:15:00", "time_limit": "1-00:00:00", "nodes": 1, "cpus": 8}
    activity_info = {"exists": True, "stale": False, "minutes_since_change": 2, "most_recent_file": "/data/out.log"}
    verdict = assess_job_health(squeue_info, activity_info)
    assert "healthy" in verdict.lower() or "actively updated" in verdict.lower()


def test_assess_health_early_startup_not_flagged_as_stuck():
    # Low CPU time + stale output is treated differently from high CPU +
    # stale output — this is likely just startup, not the busy-but-silent case.
    squeue_info = {"jobid": "1", "state": "RUNNING", "time_used": "0:05", "time_limit": "1-00:00:00", "nodes": 1, "cpus": 8}
    activity_info = {"exists": True, "stale": True, "minutes_since_change": 40, "most_recent_file": None}
    verdict = assess_job_health(squeue_info, activity_info)
    assert "busy-but-silent" not in verdict


def test_assess_health_long_running_with_zero_files_ever_flagged():
    # The specific gap being fixed: a job that has run a long time and
    # never written a single file, not just one that's gone stale partway.
    squeue_info = {"jobid": "1", "state": "RUNNING", "time_used": "05:00:00", "time_limit": "1-00:00:00", "nodes": 1, "cpus": 8}
    activity_info = {"exists": True, "stale": True, "minutes_since_change": None, "most_recent_file": None, "file_count": 0}
    verdict = assess_job_health(squeue_info, activity_info)
    assert "never written" in verdict.lower()
    assert "05:00:00" in verdict


def test_assess_health_just_started_with_zero_files_not_alarming():
    # Same zero-files situation, but the job just started - this should
    # read as "too early to tell," not "stuck."
    squeue_info = {"jobid": "1", "state": "RUNNING", "time_used": "0:10", "time_limit": "1-00:00:00", "nodes": 1, "cpus": 8}
    activity_info = {"exists": True, "stale": True, "minutes_since_change": None, "most_recent_file": None, "file_count": 0}
    verdict = assess_job_health(squeue_info, activity_info)
    assert "hasn't written anything" in verdict.lower()
    assert "not concerning yet" in verdict.lower()


# --- HPC-3: busy determination must use sstat AveCPU, not wall time --------

def test_assess_health_busy_driven_by_avecpu_not_walltime():
    # Short wall time (30s, below threshold) but high real CPU -> AveCPU says
    # busy, so a stale output dir is flagged as busy-but-silent. Under the old
    # wall-time-only logic this would NOT have been flagged.
    squeue_info = {"jobid": "1", "state": "RUNNING", "time_used": "00:30", "time_limit": "1-00:00:00", "nodes": 1, "cpus": 8}
    sstat_info = {"jobid": "1", "ave_cpu": "05:00:00", "max_rss": "1G", "ave_rss": "800M"}
    activity_info = {"exists": True, "stale": True, "minutes_since_change": 120, "most_recent_file": "/data/out.log"}
    verdict = assess_job_health(squeue_info, activity_info, sstat_info)
    assert "busy-but-silent" in verdict or "stuck" in verdict.lower()


def test_assess_health_low_avecpu_long_wall_flagged_blocked():
    # Long wall time but ~zero real CPU + no output -> blocked/deadlocked, NOT
    # "busy". This is the distinction wall time alone could not make.
    squeue_info = {"jobid": "1", "state": "RUNNING", "time_used": "02:00:00", "time_limit": "1-00:00:00", "nodes": 1, "cpus": 8}
    sstat_info = {"jobid": "1", "ave_cpu": "00:00:03", "max_rss": "10M", "ave_rss": "8M"}
    activity_info = {"exists": True, "file_count": 0, "stale": True, "minutes_since_change": None, "most_recent_file": None}
    verdict = assess_job_health(squeue_info, activity_info, sstat_info)
    assert "blocked" in verdict.lower() or "almost no cpu" in verdict.lower()
    assert "busy-but-silent" not in verdict


def test_assess_health_early_startup_with_low_avecpu_not_blocked():
    # Just started (short wall) and little CPU -> too early to call blocked.
    squeue_info = {"jobid": "1", "state": "RUNNING", "time_used": "00:20", "time_limit": "1-00:00:00", "nodes": 1, "cpus": 8}
    sstat_info = {"jobid": "1", "ave_cpu": "00:00:02", "max_rss": "10M", "ave_rss": "8M"}
    activity_info = {"exists": True, "file_count": 0, "stale": True, "minutes_since_change": None, "most_recent_file": None}
    verdict = assess_job_health(squeue_info, activity_info, sstat_info)
    assert "not concerning yet" in verdict.lower()
    assert "blocked" not in verdict.lower()


# --- HPC-6: watch exit codes ------------------------------------------------

def test_exit_code_not_found_is_2():
    assert health_exit_code(None, {"exists": False}) == 2

def test_exit_code_pending_is_0():
    assert health_exit_code({"state": "PENDING", "time_used": "0:00"}, {"exists": False}) == 0

def test_exit_code_healthy_is_0():
    q = {"state": "RUNNING", "time_used": "02:15:00", "cpus": 8}
    act = {"exists": True, "stale": False, "file_count": 5, "minutes_since_change": 2}
    assert health_exit_code(q, act, {"ave_cpu": "02:10:00"}) == 0

def test_exit_code_busy_but_silent_is_1():
    q = {"state": "RUNNING", "time_used": "00:30", "cpus": 8}
    act = {"exists": True, "stale": True, "file_count": 1, "minutes_since_change": 120}
    assert health_exit_code(q, act, {"ave_cpu": "05:00:00"}) == 1

def test_exit_code_blocked_zero_files_is_1():
    q = {"state": "RUNNING", "time_used": "02:00:00", "cpus": 8}
    act = {"exists": True, "stale": True, "file_count": 0, "minutes_since_change": None}
    assert health_exit_code(q, act, {"ave_cpu": "00:00:03"}) == 1

def test_exit_code_just_started_is_0():
    q = {"state": "RUNNING", "time_used": "00:20", "cpus": 8}
    act = {"exists": True, "stale": True, "file_count": 0, "minutes_since_change": None}
    assert health_exit_code(q, act, {"ave_cpu": "00:00:02"}) == 0

def test_exit_code_matches_verdict_for_busy_but_silent():
    # Guard: the message and the code must agree on the same scenario.
    q = {"state": "RUNNING", "time_used": "00:30", "cpus": 8}
    act = {"exists": True, "stale": True, "file_count": 1, "minutes_since_change": 120}
    verdict = assess_job_health(q, act, {"ave_cpu": "05:00:00"})
    assert "busy-but-silent" in verdict and health_exit_code(q, act, {"ave_cpu": "05:00:00"}) == 1


# --- HPC-14: queue position -------------------------------------------------

def test_parse_pending_ids():
    from hpccoworklab.monitor import parse_pending_ids
    assert parse_pending_ids("10\n11\n\n12\n") == ["10", "11", "12"]
    assert parse_pending_ids("") == []


def test_compute_queue_position_exact():
    from hpccoworklab.monitor import compute_queue_position
    assert compute_queue_position(["10", "11", "12"], "11") == 2


def test_compute_queue_position_array_parent():
    from hpccoworklab.monitor import compute_queue_position
    assert compute_queue_position(["5_0", "5_1", "9"], "5") == 1


def test_compute_queue_position_missing():
    from hpccoworklab.monitor import compute_queue_position
    assert compute_queue_position(["1", "2"], "99") is None
