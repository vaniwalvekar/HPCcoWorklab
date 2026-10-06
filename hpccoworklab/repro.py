"""
hpccoworklab.repro - reproducibility snapshot (HPC-25).

Capture the environment a job ran (or will run) in: loaded modules, the job's
requested resources from `sacct`, Python/cluster/host, so you can attach an
exact "how this was run" record to a paper or re-run it faithfully.

Module capture uses the LOADEDMODULES environment variable (set by both Lmod and
Environment-Modules), which is reliable and needs no fragile `module` shell call.
Pure parts are tested here; job_resources needs a live cluster (HPC-10).
"""

import os
import re
import sys
import socket

from .slurm import run_slurm


def loaded_modules(environ=None):
    """Module names currently loaded, parsed from $LOADEDMODULES (':' separated,
    sometimes space). Returns [] when the variable is absent."""
    e = environ if environ is not None else os.environ
    raw = (e.get("LOADEDMODULES") or "").strip()
    if not raw:
        return []
    return [m for m in re.split(r"[:\s]+", raw) if m]


def job_resources(jobid):
    """One-line requested-resource summary for a job from sacct (AllocTRES, etc.).
    Needs a real cluster to exercise; returns None if nothing found."""
    cmd = ["sacct", "-j", str(jobid), "-X", "-P", "-n", "--noheader",
           "--format=JobID,Partition,State,AllocTRES,ReqMem,Submit,Start,End"]
    out = run_slurm(cmd)
    line = next((l for l in out.splitlines() if l.strip()), "")
    return line.strip() or None


def build_snapshot(jobid=None, modules=None, resources=None, cluster=None,
                   python=None, host=None):
    """Assemble a plain-text reproducibility manifest from already-gathered facts
    (pure + testable)."""
    modules = modules or []
    lines = ["# hpccoworklab reproducibility snapshot"]
    lines.append(f"job: {jobid or '(current session)'}")
    lines.append(f"cluster: {cluster or 'unknown'}")
    lines.append(f"host: {host or socket.gethostname()}")
    lines.append(f"python: {python or sys.version.split()[0]}")
    if resources:
        lines.append(f"sacct(JobID|Partition|State|AllocTRES|ReqMem|Submit|Start|End): {resources}")
    lines.append("modules:")
    if modules:
        lines.extend(f"  - {m}" for m in modules)
    else:
        lines.append("  (none detected via $LOADEDMODULES)")
    return "\n".join(lines)
