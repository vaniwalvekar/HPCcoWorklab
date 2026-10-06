# Changelog

All notable changes to HPCcoWorklab are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`hpccoworklab check`** — pre-submission validation: partition validity, required/
  recommended fields, GPU sanity, CPU ceilings, core-overflow math, unit-safe
  memory parsing (`M/G/T`, bare number = MB), slow-storage (`$HOME`) warnings,
  missing referenced files, MPI-vs-threaded awareness.
  - Batch 1: environment→`module load` suggestions (`software_map`), partition
    recommender, Apptainer/Singularity `.sif` + `--bind` existence check, long-
    walltime checkpointing nudge.
  - Batch 2: `--array` validator (malformed spec + unused task-id), config-driven
    conflicting-module detection.
- **`hpccoworklab watch`** — live status via `squeue` + `sstat`; busy detection uses
  real `AveCPU` (not wall time); distinguishes *busy-but-silent* vs *blocked*;
  output-directory activity (stale vs never-written); `--log` progress detection
  (rate/ETA) via config `progress_patterns`/`completion_markers`; queue position
  for pending jobs; `--sample-seconds` to compute throughput.
- **`hpccoworklab diagnose`** — translates `sacct` state + exit code to plain English;
  handles `CANCELLED by <uid>`, transient/ghost `PENDING`/`RUNNING`.
- **`hpccoworklab ask`** — config-driven submission advisor (partition + modules +
  sbatch skeleton) from a short spec.
- **`hpccoworklab report`** — `sacct`-based CPU over-request analysis (core-hours asked
  vs used, utilization, worst offenders).
- **`hpccoworklab repro`** — reproducibility snapshot (loaded modules via
  `$LOADEDMODULES`, `sacct` resources, cluster/python/host).
- **`hpccoworklab check --ai`** — optional, advisory, fail-safe LLM review (off by
  default; never changes exit codes; user-configured non-NRP endpoint).
- Cluster-agnostic YAML configs; bundled into the wheel (`--config libra` resolves
  to a packaged config). Meaningful exit codes for every command
  (`0` ok · `1` problem · `2` couldn't determine). GitHub Actions CI across
  Python 3.8–3.12 plus a build + `twine check` job.

### Fixed
- Memory-unit parsing (bare `--mem` treated as MB), MPI false-positive on
  thread-launch scripts, core-overflow math, `>=24h` elapsed-time crash, array/
  step jobs misread as missing, sstat AveCPU ignored for busy detection, ghost
  `PENDING` records, unhandled `CANCELLED by <uid>`, subprocess tracebacks when
  Slurm is absent, and `squeue` field padding/non-numeric counts.

## [0.1.0] - 2026
- Initial packaged release: `hpccoworklab` console entry point, editable install,
  `check` command, YAML cluster configs, TestPyPI publication.
