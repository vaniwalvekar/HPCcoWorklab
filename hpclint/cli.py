"""
hpclint.cli — command-line entry point for hpclint.

Subcommands:
    hpclint check <script> --config <config.yaml>   Pre-submission script check
    hpclint watch <jobid> [--output-dir DIR]         Live job health check
    hpclint diagnose <jobid>                         Post-run failure explanation
"""

import sys
import os
import argparse

from .checker import check_script, load_config
from .monitor import (
    run_squeue,
    run_sstat,
    parse_squeue_line,
    parse_sstat_line,
    check_output_activity,
    assess_job_health,
    health_exit_code,
    read_log_progress,
)
from .diagnose import run_sacct, parse_sacct_line, diagnose, diagnose_exit_code
from .slurm import SlurmCommandError


def _run_check(args):
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: could not find config file '{args.config}'")
        sys.exit(2)

    try:
        params_checked, issues = check_script(args.script, config)
    except FileNotFoundError:
        print(f"Error: could not find script file '{args.script}'")
        sys.exit(2)

    print("Checks completed. Here's the result:\n")
    print(f"Cluster:       {config.get('cluster_name', 'unknown')}")
    print(f"File checked:  {args.script}\n")

    print("Parameters checked:")
    for key, value in params_checked.items():
        display_value = value if value is not None else "(not set)"
        print(f"  --{key} = {display_value}")
    print()

    if issues:
        print(f"Found {len(issues)} issue(s):\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        sys.exit(1)
    else:
        print("No issues found.")
        sys.exit(0)


def _maybe_read_log(args):
    log_path = getattr(args, "log", None)
    if not log_path:
        return None
    patterns = completion = None
    cfg_path = getattr(args, "config", None)
    if cfg_path:
        try:
            cfg = load_config(cfg_path) or {}
            patterns = cfg.get("progress_patterns")
            completion = cfg.get("completion_markers")
        except FileNotFoundError:
            pass
    info = read_log_progress(log_path, patterns=patterns, completion_markers=completion,
                             stale_after_minutes=args.stale_minutes)
    if info.get("available") and not info.get("has_markers"):
        suffix = "" if patterns else " (no progress_patterns in config)"
        print(f"note: no progress markers found in the log{suffix} - using the directory check\n")
    return info


def _run_watch(args):
    squeue_raw = run_squeue(args.jobid)
    squeue_info = parse_squeue_line(squeue_raw)

    sstat_raw = run_sstat(args.jobid)
    sstat_info = parse_sstat_line(sstat_raw)

    activity_info = check_output_activity(args.output_dir, stale_after_minutes=args.stale_minutes)

    print(f"Job {args.jobid} — live status\n")

    if squeue_info:
        print(f"State:         {squeue_info['state']}")
        print(f"Time used:     {squeue_info['time_used']} / {squeue_info['time_limit']}")
        print(f"Nodes/CPUs:    {squeue_info['nodes']} / {squeue_info['cpus']}")
    else:
        print("State:         not found in queue (may have finished)")

    if sstat_info:
        print(f"Avg CPU:       {sstat_info['ave_cpu']}")
        print(f"Max RSS:       {sstat_info['max_rss']}")

    print(f"Output dir:    {args.output_dir}")
    if activity_info.get("exists") and activity_info.get("minutes_since_change") is not None:
        print(f"Last change:   {activity_info['minutes_since_change']:.1f} minutes ago "
              f"({activity_info['most_recent_file']})")

    print()
    log_info = _maybe_read_log(args)
    print(assess_job_health(squeue_info, activity_info, sstat_info, log_info))
    return health_exit_code(squeue_info, activity_info, sstat_info, log_info)


def _run_diagnose(args):
    sacct_raw = run_sacct(args.jobid)
    sacct_info = parse_sacct_line(sacct_raw)

    print(f"Job {args.jobid} — post-run diagnosis\n")
    print(diagnose(sacct_info))
    return diagnose_exit_code(sacct_info)


def main():
    parser = argparse.ArgumentParser(description="hpclint — a cluster-agnostic Slurm job assistant.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Check a job script before submission")
    check_parser.add_argument("script", help="Path to the Slurm job script (.sh/.slurm) to check")
    check_parser.add_argument(
        "--config",
        default=os.environ.get("HPCLINT_DEFAULT_CONFIG"),
        required=os.environ.get("HPCLINT_DEFAULT_CONFIG") is None,
        help="Path to your cluster's YAML config file. Falls back to $HPCLINT_DEFAULT_CONFIG if set "
             "(e.g. via `module load hpclint`), so it's optional in that case.",
    )

    watch_parser = subparsers.add_parser("watch", help="Check a running job's live status")
    watch_parser.add_argument("jobid", help="Slurm job ID to check")
    watch_parser.add_argument("--output-dir", required=True, help="Directory the job writes output to")
    watch_parser.add_argument("--stale-minutes", type=int, default=30,
                               help="Minutes of no file activity before flagging as stale (default: 30)")
    watch_parser.add_argument("--log", help="Path to the job's stdout/stderr log, to detect real progress (HPC-31)")
    watch_parser.add_argument("--config", default=os.environ.get("HPCLINT_DEFAULT_CONFIG"),
                              help="Cluster/app config providing progress_patterns/completion_markers (used with --log)")

    diagnose_parser = subparsers.add_parser("diagnose", help="Explain why a finished job failed (or didn't)")
    diagnose_parser.add_argument("jobid", help="Slurm job ID to diagnose")

    args = parser.parse_args()

    if args.command == "check":
        _run_check(args)
        return

    try:
        if args.command == "watch":
            sys.exit(_run_watch(args))
        elif args.command == "diagnose":
            sys.exit(_run_diagnose(args))
    except SlurmCommandError as exc:
        print(f"Error: could not query Slurm — {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
