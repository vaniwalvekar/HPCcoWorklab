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

from .checker import check_script, load_config, read_script
from .monitor import (
    run_squeue,
    run_sstat,
    parse_squeue_line,
    parse_sstat_line,
    check_output_activity,
    assess_job_health,
    health_exit_code,
    read_log_progress,
    run_squeue_pending_ids,
    compute_queue_position,
)
from .diagnose import run_sacct, parse_sacct_line, diagnose, diagnose_exit_code
from .advisor import suggest_submission
from .report import run_sacct_usage, parse_usage_line, summarize
from .repro import loaded_modules, job_resources, build_snapshot
from .ai import review_script, AiNotConfigured, AiError
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

    code = 1 if issues else 0
    if issues:
        print(f"Found {len(issues)} issue(s):\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("No issues found.")

    if getattr(args, "ai", False):
        _print_ai_notes(args.script, config)

    sys.exit(code)


def _print_ai_notes(script_path, config):
    """Optional, advisory AI review. Never raises and never changes exit codes."""
    try:
        content = read_script(script_path)
    except FileNotFoundError:
        content = ""
    try:
        findings = review_script(content, config)
    except AiNotConfigured as exc:
        print(f"\nAI: skipped - {exc}")
        return
    except Exception as exc:  # fail-safe: AI must never break `check`
        print(f"\nAI: skipped due to error: {exc}")
        return
    print("\nAI-assisted observations (advisory, not authoritative):")
    if findings:
        for f in findings:
            print(f"  - {f}")
    else:
        print("  (none beyond what the deterministic checks found)")


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
                             stale_after_minutes=args.stale_minutes,
                             sample_seconds=getattr(args, "sample_seconds", 0))
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
        if squeue_info["state"] == "PENDING":
            try:
                pos = compute_queue_position(run_squeue_pending_ids(args.jobid), args.jobid)
            except SlurmCommandError:
                pos = None
            if pos:
                print(f"Queue position: {pos}")
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


def _run_ask(args):
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: could not find config file '{args.config}'")
        sys.exit(2)
    partition, modules, script = suggest_submission(
        config, gpus=args.gpus, cpus=args.cpus, mem_gb=args.mem,
        time_limit=args.time, app=args.app)
    print("Suggested submission\n")
    print(f"Partition: {partition or '(no partition fits your request)'}")
    if modules:
        print("Modules:   " + ", ".join(modules))
    print("\n" + script)
    return 0


def _run_report(args):
    try:
        raw = run_sacct_usage(user=args.user, since=args.since)
    except SlurmCommandError as exc:
        print(f"Error: could not read accounting data - {exc}")
        return 2
    records = []
    for line in raw.splitlines():
        rec = parse_usage_line(line)
        if rec:
            records.append(rec)
    if not records:
        print(f"No jobs found in accounting for the last {args.since}.")
        return 0
    s = summarize(records)
    print("Job utilization report\n")
    print(f"Jobs analyzed:   {s['jobs']}")
    print(f"Requested:       {s['cpu_alloc_core_hours']:.1f} core-hours")
    print(f"Actually used:   {s['cpu_used_core_hours']:.1f} core-hours")
    if s["cpu_utilization"] is not None:
        print(f"CPU utilization: {s['cpu_utilization'] * 100:.0f}%")
    if s["worst"]:
        print("\nLowest-utilization jobs (asked vs used core-hours):")
        for util, jid, name, alloc, ah, uh in s["worst"]:
            print(f"  {jid:>10} {name[:22]:22} {util * 100:3.0f}%  ({uh:.1f} used / {ah:.1f} asked)")
    return 0


def _run_repro(args):
    cluster = None
    cfg_path = getattr(args, "config", None)
    if cfg_path:
        try:
            cluster = (load_config(cfg_path) or {}).get("cluster_name")
        except FileNotFoundError:
            pass
    resources = None
    if args.jobid:
        try:
            resources = job_resources(args.jobid)
        except SlurmCommandError as exc:
            resources = f"(sacct unavailable: {exc})"
    snap = build_snapshot(args.jobid, loaded_modules(), resources, cluster)
    out_path = args.output or f"hpclint-repro-{args.jobid or 'env'}.txt"
    print(snap)
    try:
        with open(out_path, "w") as f:
            f.write(snap + "\n")
        print(f"\nSaved -> {out_path}")
    except OSError as exc:
        print(f"\n(would save to {out_path}, but: {exc})")
    return 0


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
    check_parser.add_argument("--ai", action="store_true",
                              help="Optional advisory LLM review (needs HPCLINT_AI_BASE_URL + HPCLINT_AI_API_KEY); never changes exit codes")

    watch_parser = subparsers.add_parser("watch", help="Check a running job's live status")
    watch_parser.add_argument("jobid", help="Slurm job ID to check")
    watch_parser.add_argument("--output-dir", required=True, help="Directory the job writes output to")
    watch_parser.add_argument("--stale-minutes", type=int, default=30,
                               help="Minutes of no file activity before flagging as stale (default: 30)")
    watch_parser.add_argument("--log", help="Path to the job's stdout/stderr log, to detect real progress (HPC-31)")
    watch_parser.add_argument("--config", default=os.environ.get("HPCLINT_DEFAULT_CONFIG"),
                              help="Cluster/app config providing progress_patterns/completion_markers (used with --log)")
    watch_parser.add_argument("--sample-seconds", type=int, default=0,
                              help="Re-sample the log N seconds later to compute progress rate + ETA (HPC-41)")

    ask_parser = subparsers.add_parser("ask", help="Suggest partition, modules, and an sbatch skeleton from a short spec")
    ask_parser.add_argument("--gpus", type=int, default=0)
    ask_parser.add_argument("--cpus", type=int, default=1, help="cpus-per-task")
    ask_parser.add_argument("--mem", type=int, default=None, help="Requested memory in GB")
    ask_parser.add_argument("--time", dest="time", default=None, help="Walltime, e.g. 04:00:00")
    ask_parser.add_argument("--app", default=None, help="Application name from config software_map")
    ask_parser.add_argument("--config", default=os.environ.get("HPCLINT_DEFAULT_CONFIG"),
                            required=os.environ.get("HPCLINT_DEFAULT_CONFIG") is None,
                            help="Cluster YAML providing partitions + software_map")

    report_parser = subparsers.add_parser("report", help="Summarize CPU over-requesting across your recent jobs (HPC-21)")
    report_parser.add_argument("--user", default=os.environ.get("USER"), help="Account to report on (default: $USER)")
    report_parser.add_argument("--since", default="7 days", help="Time window for sacct (default: '7 days')")

    repro_parser = subparsers.add_parser("repro", help="Write a reproducibility snapshot (modules + resources) (HPC-25)")
    repro_parser.add_argument("jobid", nargs="?", help="Optional job id to include its sacct resources")
    repro_parser.add_argument("--config", default=os.environ.get("HPCLINT_DEFAULT_CONFIG"),
                              help="Cluster YAML for the cluster_name field (optional)")
    repro_parser.add_argument("--output", default=None, help="Path to write the snapshot file")

    diagnose_parser = subparsers.add_parser("diagnose", help="Explain why a finished job failed (or didn't)")
    diagnose_parser.add_argument("jobid", help="Slurm job ID to diagnose")

    args = parser.parse_args()

    if args.command == "check":
        _run_check(args)
        return

    if args.command == "ask":
        sys.exit(_run_ask(args))
        return

    if args.command == "report":
        sys.exit(_run_report(args))
        return

    if args.command == "repro":
        sys.exit(_run_repro(args))
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
