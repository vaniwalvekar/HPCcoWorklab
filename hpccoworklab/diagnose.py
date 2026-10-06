"""
hpccoworklab.diagnose — post-run analysis for a finished Slurm job.

Wraps `sacct` to pull a job's final state and exit code, and translates
common failure patterns (OOM kill, timeout, segfault, node failure) into
plain language with a likely cause and next step.

Same split as monitor.py: parsing/translation logic is pure and tested
against realistic fixture text; the subprocess wrapper needs a real
Slurm cluster and gets verified there per the project's end-to-end
verification plan.
"""

from .slurm import run_slurm

# Common Slurm job end states and what they mean in plain language.
_STATE_EXPLANATIONS = {
    "COMPLETED": "Finished normally, with a successful exit code.",
    "FAILED": "The job's command exited with a non-zero (error) status.",
    "TIMEOUT": "The job hit its --time limit and was killed before finishing. "
               "If it was making real progress, resubmit with a longer --time.",
    "OUT_OF_MEMORY": "The job used more memory than it requested and was killed. "
                      "Increase --mem, or check for a memory leak / unexpectedly large input.",
    "CANCELLED": "The job was cancelled — either by you, an admin, or a dependency failure.",
    "NODE_FAIL": "The compute node itself failed during the job (hardware/system issue, "
                 "not your code). Usually safe to resubmit as-is.",
    "PREEMPTED": "The job was preempted by a higher-priority job (common on shared/priority "
                 "partitions). Usually safe to resubmit.",
}

# States meaning the job is queued or still in flight, not a finished result.
# A sacct record stuck in one of these while squeue no longer lists the job is
# the classic "ghost record" case (e.g. a job killed before it really started).
_TRANSIENT_STATES = {
    "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUE_HOLD",
}

# Common exit codes and their typical meaning. Exit code alone is ambiguous
# (128+signal is a common convention but not guaranteed), so this is treated
# as a hint, not a certainty.
_EXIT_CODE_HINTS = {
    "0": "Success.",
    "1": "Generic error — check the job's stderr/output log for details.",
    "137": "Killed with SIGKILL (128+9) — very often an out-of-memory kill, "
           "even if the job's State doesn't say OUT_OF_MEMORY.",
    "139": "Segmentation fault (128+11) — the program crashed, often from a bug or "
           "incompatible library/module version.",
    "124": "Command timed out (common convention from the `timeout` utility, if used "
           "inside the script).",
}


def run_sacct(jobid):
    """Run sacct for one job and return its raw pipe-delimited output line
    for the main job step, or None if not found."""
    cmd = ["sacct", "-j", str(jobid), "--format=JobID,State,ExitCode", "-P", "-n"]
    lines = [l for l in run_slurm(cmd).strip().split("\n") if l]
    if not lines:
        return None
    # The first line is normally the main job (jobid with no suffix like .batch/.extern)
    for line in lines:
        parts = line.split("|")
        if len(parts) == 3 and "." not in parts[0]:
            return line
    return lines[0]


def parse_sacct_line(line):
    """Parse one line of `sacct --format=JobID,State,ExitCode -P` output.
    e.g. '12345|OUT_OF_MEMORY|0:125' or '12345|COMPLETED|0:0'
    """
    if not line:
        return None
    parts = line.strip().split("|")
    if len(parts) != 3:
        return None
    jobid, state, exit_code = parts
    # exit_code is "code:signal" — split out the code portion
    code_part = exit_code.split(":")[0] if exit_code else None
    return {"jobid": jobid, "state": state, "exit_code": code_part}


def diagnose(sacct_info):
    """Turn a parsed sacct result into a plain-language explanation and,
    where possible, a suggested next step."""
    if sacct_info is None:
        return "No accounting record found for this job ID — check the ID, or it may not have run yet."

    raw_state = sacct_info.get("state", "")
    # sacct reports cancellations as 'CANCELLED by 1042' - normalise to the
    # bare state for lookup, but keep the raw string for display.
    state = raw_state.split()[0] if raw_state else ""
    exit_code = sacct_info.get("exit_code")

    lines = []

    state_explanation = _STATE_EXPLANATIONS.get(state)
    if state_explanation:
        lines.append(f"State: {raw_state} — {state_explanation}")
    elif state in _TRANSIENT_STATES:
        lines.append(
            f"State: {raw_state} — the job hasn't failed; it is still {state.lower()} in the "
            f"accounting log. If `hpccoworklab watch` cannot find this job in squeue, that is usually "
            f"a stale 'ghost' record - trust squeue for the live state."
        )
    else:
        lines.append(f"State: {raw_state} (no specific explanation on file for this state)")

    if exit_code and exit_code != "0":
        exit_hint = _EXIT_CODE_HINTS.get(exit_code)
        if exit_hint:
            lines.append(f"Exit code {exit_code}: {exit_hint}")
        else:
            lines.append(f"Exit code {exit_code}: no specific hint on file for this code — "
                         f"check the job's stderr/output log.")

    return "\n".join(lines)


# States that are not a fault of the job itself.
_BENIGN_STATES = {"COMPLETED", "CANCELLED", "PREEMPTED"}


def diagnose_exit_code(sacct_info):
    """Process exit code for a diagnosis, mirroring diagnose():

        0 = benign outcome (completed, cancelled, preempted, still queued/running)
        1 = a failure/problem (FAILED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, ...) or unknown
        2 = cannot determine (no accounting record found)

    A non-zero exit code bumps an otherwise-benign state up to 1.
    """
    if sacct_info is None:
        return 2
    raw_state = sacct_info.get("state", "") or ""
    state = raw_state.split()[0] if raw_state else ""
    exit_code = sacct_info.get("exit_code")

    if state in _BENIGN_STATES or state in _TRANSIENT_STATES:
        return 1 if (exit_code and exit_code != "0") else 0
    return 1
