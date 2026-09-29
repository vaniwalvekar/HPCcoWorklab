"""
hpclint.report - post-run utilization analysis (HPC-21).

Answers the question researchers hate: "am I wasting my allocation?" It pulls a
user's recent jobs from `sacct` and compares what each job *asked for* (CPUs x
walltime) against what it actually *used* (TotalCPU), surfacing chronic
over-requesting so future scripts can ask for less and queue faster.

Parsing/summary logic is pure and fixture-tested; only run_sacct_usage needs a
live cluster (and its exact column availability is confirmed on Libra).
"""

from .slurm import run_slurm
from .monitor import _parse_cpu_time_to_seconds
from .checker import parse_mem_to_gb

_FIELDS = ["JobID", "JobName", "Partition", "AllocCPUS", "TotalCPU",
           "Elapsed", "MaxRSS", "ReqMem"]


def run_sacct_usage(user=None, since="7 days"):
    """Run sacct for a user's recent jobs. Needs a real cluster to exercise."""
    cmd = ["sacct", "-X", "-P", "-n", "--noheader", "--starttime", since,
           "--format=" + ",".join(_FIELDS), "--units=MB"]
    if user:
        cmd += ["-u", user]
    return run_slurm(cmd)


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _req_mem_gb(v):
    if not v or v.strip().upper() in ("N/A", ""):
        return None
    try:
        return parse_mem_to_gb(v)
    except Exception:
        return None


def parse_usage_line(line):
    """Parse one sacct --format row into a usage record; None if malformed."""
    if not line or not line.strip():
        return None
    parts = line.strip().split("|")
    if len(parts) != len(_FIELDS):
        return None
    d = dict(zip(_FIELDS, parts))
    return {
        "jobid": d["JobID"],
        "name": d["JobName"],
        "partition": d["Partition"],
        "alloc_cpus": _to_int(d["AllocCPUS"]),
        "totalcpu_seconds": _parse_cpu_time_to_seconds(d["TotalCPU"]),
        "elapsed_seconds": _parse_cpu_time_to_seconds(d["Elapsed"]),
        "maxrss_mb": _to_float(d["MaxRSS"]),
        "req_mem": d["ReqMem"] or None,
        "req_mem_gb": _req_mem_gb(d["ReqMem"]),
    }


def summarize(records):
    """Aggregate core-hour utilization and list the worst offenders.

    utilization = TotalCPU / (AllocCPUS x Elapsed). Low = you asked for CPUs
    you didn't use (wasted allocation + longer queues for everyone)."""
    used = 0.0
    alloc = 0.0
    count = 0
    worst = []
    for r in records:
        if not r:
            continue
        count += 1
        a = r["alloc_cpus"]
        e = r["elapsed_seconds"]
        if not a or not e:
            continue
        alloc_secs = a * e
        used_secs = r["totalcpu_seconds"] or 0
        alloc += alloc_secs
        used += used_secs
        worst.append((used_secs / alloc_secs, r["jobid"], r["name"], a,
                      alloc_secs / 3600.0, used_secs / 3600.0))
    worst.sort(key=lambda x: x[0])
    return {
        "jobs": count,
        "cpu_alloc_core_hours": alloc / 3600.0,
        "cpu_used_core_hours": used / 3600.0,
        "cpu_utilization": (used / alloc) if alloc else None,
        "worst": worst[:5],
    }
