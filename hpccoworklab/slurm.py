"""
hpccoworklab.slurm — one safe way to run Slurm CLI commands (squeue/sstat/sacct).

Centralises the things every wrapper must get right so a problem on a real
cluster produces a clear message instead of a stack trace:
  - never shells out (args are always a list)
  - a hard timeout, so an unreachable slurmctld/SlurmDBD can't hang forever
  - a clean error type for "command missing", "timed out", or "exited non-zero"

A non-zero exit is treated as a *failure*, distinct from an empty-but-successful
result (which means "job not found"), so callers never confuse the two.
"""

import subprocess

DEFAULT_TIMEOUT_SECONDS = 20


class SlurmCommandError(Exception):
    """A Slurm command could not be run, timed out, or reported a failure."""


def run_slurm(args, timeout=DEFAULT_TIMEOUT_SECONDS):
    """Run a Slurm command and return stdout as text.

    Raises SlurmCommandError if the binary is missing, the command times out,
    or it exits non-zero. An empty stdout with exit code 0 is returned as-is
    (callers interpret that as "nothing found").
    """
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise SlurmCommandError(
            f"Slurm command '{args[0]}' was not found. Run hpccoworklab on a login/compute "
            f"node where Slurm is available (after any required 'module load')."
        )
    except subprocess.TimeoutExpired:
        raise SlurmCommandError(
            f"Slurm command '{' '.join(args)}' timed out after {timeout}s — the "
            f"scheduler controller (slurmctld) or accounting DB may be unreachable."
        )

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise SlurmCommandError(
            f"'{args[0]}' failed (exit {result.returncode})" + (f": {detail}" if detail else "")
        )

    return result.stdout
