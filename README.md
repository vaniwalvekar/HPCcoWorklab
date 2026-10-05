# hpclint

**The missing pre-flight check for HPC jobs — and the first piece of a
complete Slurm job assistant.**

Every HPC user has felt this: you write a job script, submit it, wait in
queue for twenty minutes, and watch it die in the first second because you
asked for more memory than any node has. Or worse — it runs for six hours,
using 100% CPU the whole time, and produces nothing, because it was quietly
stuck the entire way through.

hpclint exists so that stops happening.

## The idea

A Slurm job goes through a lifecycle — written, submitted, queued, run,
finished (or not) — and at almost every stage, something can quietly go
wrong that nobody notices until it's too late. hpclint is being built to
watch the whole lifecycle, not just one moment of it.

### 🧭 Before you submit
*Catch the mistake while it's still free to fix.*
- Check `#SBATCH` resource requests against your cluster's real hardware
- Understand how a job actually parallelizes (MPI vs. threaded vs. hybrid)
  so it's never flagged for doing the right thing the "wrong" way
- Warn about slow storage paths (`$HOME`) before a data-heavy job crawls
- Catch core-overflow mistakes and missing referenced files

### 📡 While it runs
*A job that's "running" isn't the same as a job that's working.*
- A unified live view combining `squeue` state and `sstat` resource usage
- **Detect a job that's stuck, not working** — running, maybe even pegged
  at 100% CPU, but producing nothing: no new output files, no progress.
  That combination — busy but silent — is one of the hardest failure
  modes to notice manually, and one of the most expensive, since it can
  burn an entire walltime allocation before anyone checks in

### 🔍 After it finishes
*Turn "it failed" into "here's why, and here's the fix."*
- `hpclint diagnose` translates cryptic exit codes, OOM kills, timeouts,
  segfaults, cancellations, and node failures into plain language with a
  likely cause and a next step
- (planned) Track your own jobs over time and spot recurring over-request patterns

### 🗺️ Anywhere on the cluster
*The stuff that has nothing to do with any one job, but eats time anyway. (planned)*
- "I need GROMACS" → the right `module load` command, instantly
- Storage quota checks before `$HOME` quietly fills up
- "Which partition should I use?" — recommend resources from a plain
  description of the workload

## Where things stand today

All three job-lifecycle commands are functional:

- `hpclint check` — validates `#SBATCH` directives against a cluster's real
  hardware and rules (YAML-configured, works on any Slurm cluster):
  MPI-vs-threaded aware, core-overflow math, missing referenced files,
  slow-storage warnings, unit-safe memory parsing.
- `hpclint watch` — combines live `squeue`/`sstat` status with output-directory
  activity, and decides "busy vs stuck" from real **CPU time (`AveCPU`)**, not
  wall-clock — so a deadlocked low-CPU job reads as *blocked*, distinct from a
  CPU-bound-but-silent one.
- `hpclint diagnose` — turns a finished job's `sacct` state + exit code into a
  plain-language cause and fix (OOM, timeout, segfault, cancel, node failure).
- CI runs a **160+ test** suite (pure-logic plus realistic Slurm fixture text)
  across Python 3.8–3.12 on every push; end-to-end verification against live
  Slurm on Libra is the current milestone.
- `check`, `watch`, and `diagnose` each return meaningful exit codes
  (`0` ok · `1` problem · `2` couldn't determine), so they're scriptable.

```
$ hpclint check scripts/train.sh --config libra

Checks completed. Here's the result:

Cluster:       SLU Libra
File checked:  scripts/train.sh

Parameters checked:
  --partition = compute
  --nodes = 1
  --gpus = (not set)
  --account = (not set)
  --cpus-per-task = 4
  --ntasks = (not set)
  --ntasks-per-node = (not set)
  --mem = 16G
  --time = 01:00:00

Found 2 issue(s):

1. No --gpus set. This cluster requires a GPU count on every job script
   (use --gpus=0 on 'compute', which has no GPUs).
2. No --account set. This cluster recommends always setting --account.
```

All three core commands — **before** (`check`), **while** (`watch`), and
**after** (`diagnose`) — are built and unit-tested; end-to-end verification on
live Slurm (Libra) is the current milestone. It's published on TestPyPI for now,
with a real PyPI release planned once that verification is done. **Anywhere on
the cluster** is still on the roadmap, not built yet. It's an early,
actively-developed project, built in the open on purpose: if you run Slurm and
any of this resonates, your cluster's quirks and your ideas are exactly what
would make this better.

## Why this doesn't already exist

Plenty of great tools cover pieces of this — `jobstats`, `seff`, and
`reportseff` are all excellent at telling you how a *completed* job
performed. Nothing widely adopted watches the whole journey: before,
during, and after, tuned to your specific cluster's rules. That's the gap
hpclint is aiming to fill.

## Installation

**From TestPyPI** (current release, while the project is still early):

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hpclint
```

**From source** (for development, or to get the latest unreleased changes):

```bash
git clone https://github.com/vaniwalvekar/hpclint.git
cd hpclint
pip install -e .
```

## Usage

**Check a script before submitting:**
```bash
hpclint check <path_to_script> --config <path_to_cluster_config.yaml>
```

**Watch a running job's live status:**
```bash
hpclint watch <jobid> --output-dir <path_the_job_writes_to> [--stale-minutes 30]
```
Combines `squeue` + `sstat` with output-directory activity. It decides whether a
job is actually *working* from real CPU time (`AveCPU`), so it distinguishes the
**busy-but-silent** pattern (high CPU, no new output — likely stuck in a loop)
from a **blocked** one (long-running but using almost no CPU — likely deadlocked
or waiting on a resource). Both can silently burn an entire allocation.

**Diagnose a finished job:**
```bash
hpclint diagnose <jobid>
```
Reads the job's `sacct` state and exit code and explains it in plain language —
OOM, timeout, segfault, cancellation, node failure, or a normal finish — with a
likely next step.

**Get a submission skeleton (no Slurm needed):**
```bash
hpclint ask --gpus 1 --cpus 8 --mem 32 --app gromacs --config libra
```
Suggests a fitting partition + `module load`s (from the config `software_map`)
and prints a ready-to-edit sbatch skeleton.

**Audit CPU over-requesting:**
```bash
hpclint report [--user NAME] [--since "7 days"]
```
Compares core-hours *requested* vs *used* (from `sacct`) and lists the most
wasteful recent jobs.

**Reproducibility snapshot:**
```bash
hpclint repro [jobid] [--config libra] [--output file.txt]
```
Writes loaded modules (`$LOADEDMODULES`), the job's `sacct` resources, and
cluster/python/host — something to attach to a paper.

**Optional AI review (advisory, off by default):**
```bash
hpclint check script.sh --config libra --ai
```
With `HPCLINT_AI_BASE_URL`/`HPCLINT_AI_API_KEY` set (a non-NRP, OpenAI-compatible
endpoint), adds non-authoritative observations. The deterministic checks and exit
codes stay the source of truth; AI failures degrade to a note.

### Exit codes

| Code | `check` | `watch` | `diagnose` |
|------|---------|---------|------------|
| `0` | no issues | nothing concerning | benign outcome (ok/cancelled/still queued) |
| `1` | issues found | problem (stuck/blocked/busy-silent) | failure / unrecognized state |
| `2` | usage or file error | couldn't determine (job not in queue / Slurm error) | no record / Slurm command error |

## Writing a config for your cluster

See [`hpclint/configs/libra.yaml`](hpclint/configs/libra.yaml) for a full real-world
example. The shape is:

```yaml
cluster_name: "SLU Libra"

partitions:
  compute:
    is_default: true
    has_gpu: false
    cpus_per_task_max: 64
    mem_gb_max: 1024

  gpu:
    is_default: false
    has_gpu: true
    gpu_max: 2
    cpus_per_task_max: 64
    mem_gb_max: 512

required_fields: ["nodes", "gpus"]
recommended_fields: ["account"]
default_time_days: 7

slow_io_paths:
  - path_markers: ["/home/", "$HOME"]
    recommend: "$SCRATCH"
```

Omit any section your cluster doesn't need — missing sections are treated
as "not applicable," not errors.

## Running the tests

```bash
pip install -e ".[dev]"
python3 -m pytest tests/ -v
```

The suite locks in real bugs found during development (an MPI false
positive, a memory-unit parsing bug, core-overflow math) and validates
the monitoring logic against realistic sample Slurm output, so these
can't silently break later.

## Contributing

Contributions are welcome — cluster configs for other institutions, new
checks for parallelism patterns, or a hand with any item on the roadmap
above. Open an issue before a large PR so we can talk it through first.

## License

MIT — see [LICENSE](LICENSE).
