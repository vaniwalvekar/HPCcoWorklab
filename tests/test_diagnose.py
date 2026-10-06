"""
Tests for hpccoworklab.diagnose — parsing and translation logic only, tested
against realistic fixture sacct output text (the real subprocess call
gets verified on Libra later, same as monitor.py).
"""

from hpccoworklab.diagnose import parse_sacct_line, diagnose, diagnose_exit_code


# --- sacct parsing -----------------------------------------------------------

def test_parse_sacct_line_completed():
    line = "12345|COMPLETED|0:0"
    info = parse_sacct_line(line)
    assert info == {"jobid": "12345", "state": "COMPLETED", "exit_code": "0"}


def test_parse_sacct_line_oom():
    line = "12345|OUT_OF_MEMORY|0:125"
    info = parse_sacct_line(line)
    assert info == {"jobid": "12345", "state": "OUT_OF_MEMORY", "exit_code": "0"}


def test_parse_sacct_line_failed_with_signal():
    line = "12345|FAILED|137:9"
    info = parse_sacct_line(line)
    assert info["exit_code"] == "137"


def test_parse_sacct_line_empty_returns_none():
    assert parse_sacct_line("") is None
    assert parse_sacct_line(None) is None


def test_parse_sacct_line_malformed_returns_none():
    assert parse_sacct_line("not|enough") is None


# --- diagnosis / translation --------------------------------------------------

def test_diagnose_no_record_found():
    result = diagnose(None)
    assert "no accounting record" in result.lower()


def test_diagnose_completed_job():
    info = {"jobid": "1", "state": "COMPLETED", "exit_code": "0"}
    result = diagnose(info)
    assert "COMPLETED" in result
    assert "successful" in result.lower()


def test_diagnose_timeout_mentions_time_flag():
    info = {"jobid": "1", "state": "TIMEOUT", "exit_code": "0"}
    result = diagnose(info)
    assert "--time" in result


def test_diagnose_oom_mentions_mem_flag():
    info = {"jobid": "1", "state": "OUT_OF_MEMORY", "exit_code": "0"}
    result = diagnose(info)
    assert "--mem" in result


def test_diagnose_exit_137_flags_likely_oom_even_if_state_differs():
    # Regression target: exit code 137 (SIGKILL) is a strong OOM signal
    # even when Slurm's own State field says something more generic.
    info = {"jobid": "1", "state": "FAILED", "exit_code": "137"}
    result = diagnose(info)
    assert "out-of-memory" in result.lower() or "oom" in result.lower()


def test_diagnose_exit_139_flags_segfault():
    info = {"jobid": "1", "state": "FAILED", "exit_code": "139"}
    result = diagnose(info)
    assert "segmentation fault" in result.lower()


def test_diagnose_unknown_state_still_returns_something_useful():
    info = {"jobid": "1", "state": "SOME_NEW_STATE", "exit_code": "0"}
    result = diagnose(info)
    assert "SOME_NEW_STATE" in result


# --- HPC-4: cancelled-by-uid normalisation + transient/ghost states ---------

def test_diagnose_parse_handles_cancelled_with_uid():
    # sacct -P keeps 'CANCELLED by 1042' in the State field.
    info = parse_sacct_line("12345|CANCELLED by 1042|0:0")
    assert info["state"] == "CANCELLED by 1042"

def test_diagnose_cancelled_by_uid_explained():
    info = {"jobid": "12345", "state": "CANCELLED by 1042", "exit_code": "0"}
    result = diagnose(info)
    assert "cancel" in result.lower()
    assert "no specific explanation" not in result

def test_diagnose_pending_is_transient_not_failure():
    info = {"jobid": "12345", "state": "PENDING", "exit_code": "0"}
    result = diagnose(info)
    assert "hasn't failed" in result
    assert "ghost" in result.lower()
    assert "no specific explanation" not in result

def test_diagnose_running_is_transient():
    info = {"jobid": "12345", "state": "RUNNING", "exit_code": "0:0"}
    result = diagnose(info)
    assert "running" in result.lower()
    assert "no specific explanation" not in result


# --- HPC-6: diagnose exit codes --------------------------------------------

def test_diag_exit_no_record_is_2():
    assert diagnose_exit_code(None) == 2

def test_diag_exit_completed_is_0():
    assert diagnose_exit_code({"state": "COMPLETED", "exit_code": "0"}) == 0

def test_diag_exit_failed_is_1():
    assert diagnose_exit_code({"state": "FAILED", "exit_code": "1"}) == 1

def test_diag_exit_oom_is_1():
    assert diagnose_exit_code({"state": "OUT_OF_MEMORY", "exit_code": "0:125"}) == 1

def test_diag_exit_cancelled_is_0():
    assert diagnose_exit_code({"state": "CANCELLED by 1042", "exit_code": "0"}) == 0

def test_diag_exit_pending_is_0():
    assert diagnose_exit_code({"state": "PENDING", "exit_code": "0"}) == 0

def test_diag_exit_unknown_state_is_1():
    assert diagnose_exit_code({"state": "WEIRD_NEW_STATE", "exit_code": "0"}) == 1

def test_diag_exit_completed_but_nonzero_exit_bumped_to_1():
    assert diagnose_exit_code({"state": "COMPLETED", "exit_code": "2"}) == 1
