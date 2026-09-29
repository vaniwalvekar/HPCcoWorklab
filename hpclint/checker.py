"""
hpclint.checker — core logic for checking a Slurm job script against a
cluster's real hardware limits and submission rules.
"""

import re
import os

try:
    import yaml
except ImportError as e:
    raise ImportError("Missing dependency: PyYAML. Install with: pip install pyyaml") from e


# Patterns for common ways a script references another file it depends on.
# This is deliberately heuristic, not a full shell parser — it catches the
# common cases (interpreter + script, common input flags, ./relative exec)
# and skips anything with a shell variable ($HOME, $SCRATCH, $SLURM_*, etc.)
# since we can't know its value without actually running the job.
_FILE_REFERENCE_PATTERNS = [
    r"\b(?:python3?|Rscript|perl|bash|sh)\s+([^\s$][^\s]*\.(?:py|R|pl|sh|jl|m))",
    r"--(?:input|in|infile|script)[= ]([^\s$][^\s]*)",
    r"(?<!\S)\.\/([^\s$][^\s]*)",
]


def find_referenced_files(content, script_dir):
    """Return a list of (reference_text, resolved_path) for files the script
    references that don't actually exist on disk. Best-effort only."""
    missing = []
    seen = set()
    for pattern in _FILE_REFERENCE_PATTERNS:
        for match in re.finditer(pattern, content):
            ref = match.group(1)
            if ref in seen:
                continue
            seen.add(ref)
            resolved = ref if os.path.isabs(ref) else os.path.join(script_dir, ref)
            if not os.path.exists(resolved):
                missing.append((ref, resolved))
    return missing


def read_script(path):
    with open(path, "r") as f:
        return f.read()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Long --key <-> short -x equivalence for the options hpclint inspects.
_SHORT_ALIASES = {
    "partition": "p",
    "nodes": "N",
    "ntasks": "n",
    "cpus-per-task": "c",
    "account": "A",
}


def find_sbatch_value(content, key):
    """Find a #SBATCH option value, accepting --key=val, --key val,
    -x val and -xval (short aliases). Each #SBATCH line is read as argv-like
    tokens, so a short '-p' can never match inside '--partition'."""
    short = _SHORT_ALIASES.get(key)
    for line_match in re.finditer(r"#SBATCH(.*)", content):
        tokens = line_match.group(1).split()
        for i, tok in enumerate(tokens):
            if tok == f"--{key}" and i + 1 < len(tokens):
                return tokens[i + 1]
            if tok.startswith(f"--{key}="):
                return tok.split("=", 1)[1]
            if short:
                if tok == f"-{short}" and i + 1 < len(tokens):
                    return tokens[i + 1]
                if (tok.startswith(f"-{short}") and not tok.startswith("--")
                        and len(tok) > len(short) + 1):
                    return tok[len(short) + 1:]
    return None


def parse_mem_to_gb(mem_str):
    if mem_str is None:
        return None
    num = float(re.sub(r"[A-Za-z]", "", mem_str))
    unit = re.sub(r"[0-9.]", "", mem_str).upper().rstrip("B")
    if unit == "T":
        return num * 1024
    elif unit == "M":
        return num / 1024
    elif unit == "K":
        return num / (1024 * 1024)
    if unit == "":        # bare number: Slurm's default --mem unit is MB
        return num / 1024
    return num            # 'G'/'GB' -> gigabytes


_DEFAULT_INTERACTIVE_MARKERS = [
    r"\bjupyter\b", r"\bipython\b", r"\bcode-server\b", r"\bvscode\b",
    r"\bx11\b", r"--pty",
]


def detect_interactive(content, markers=None):
    """Return the first interactive marker found in the script (HPC-13),
    or None. Markers are regexes; defaults cover Jupyter/IPython/code-server/
    VS Code/X11/--pty. A cluster can override via config `interactive_markers`."""
    for m in (markers or _DEFAULT_INTERACTIVE_MARKERS):
        mo = re.search(m, content, re.IGNORECASE)
        if mo:
            return mo.group(0)
    return None


def parse_gpu_count(gpus_str):
    """Number of GPUs requested, or None if unparseable.
    Accepts '2', '0', 'a100:2' (type:count), '2:1' (count:per-node), or a
    bare type name ('a100' -> 1)."""
    if gpus_str is None:
        return None
    s = str(gpus_str).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if ":" in s:
        for part in s.split(":"):
            if part.isdigit():
                return int(part)
        return None
    return 1


_DEFAULT_ENV_PATH_PATTERNS = [
    r"[^\s'\"]+bin[/\\]activate",          # .../bin/activate
    r"[^\s'\"]+\.conda[/\\]envs[/\\][^\s'\"]+",
    r"[^\s'\"]+miniconda3[/\\][^\s'\"]+",
    r"[^\s'\"]+anaconda3[/\\][^\s'\"]+",
    r"[^\s'\"]+[/\\]\.venv\b",
    r"[^\s'\"]+[/\\]venvs?[/\\][^\s'\"]+",
]


def find_envs_on_slow_paths(content, slow_markers, patterns=None):
    """Return env activation paths in the script that live under a slow-storage
    marker (HPC-12). Only explicit paths count, so a bare `conda activate name`
    (whose location is unknown) is never a false positive."""
    hits = []
    seen = set()
    for pat in (patterns or _DEFAULT_ENV_PATH_PATTERNS):
        for m in re.finditer(pat, content):
            p = m.group(0)
            if p in seen:
                continue
            seen.add(p)
            if any(sm and sm in p for sm in slow_markers):
                hits.append(p)
    # Drop a hit that is nested inside another (one venv matched by several
    # patterns, e.g. '$HOME/.venv' inside '$HOME/.venv/bin/activate') so we
    # report the environment once.
    hits = [h for h in hits if not any(h != o and h in o for o in hits)]
    return hits


def _time_str_to_seconds(time_str):
    """Parse Slurm MM:SS / HH:MM:SS / D-HH:MM:SS to seconds; None if invalid."""
    if not time_str:
        return None
    s = str(time_str).strip()
    days = 0
    if "-" in s:
        dpart, _, s = s.partition("-")
        try:
            days = int(dpart)
        except ValueError:
            return None
    parts = s.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    while len(nums) < 3:
        nums.insert(0, 0)
    h, m, sec = nums[-3:]
    return days * 86400 + h * 3600 + m * 60 + sec


def _fmt_duration(seconds):
    if seconds >= 86400:
        return f"{seconds // 86400}d {(seconds % 86400) // 3600}h"
    if seconds >= 3600:
        return f"{seconds // 3600}h"
    return f"{seconds // 60}m"


_CONTAINER_CMD_RE = re.compile(r"\b(?:apptainer|singularity)\b")
_CHECKPOINT_RE = re.compile(
    r"\b(checkpoint|checkpointing|c3radical|crac|restart|resume|snapshot|save[-_ ]?interval)\b",
    re.IGNORECASE,
)


def check_containers(content, script_dir):
    """HPC-27: verify apptainer/singularity image files and --bind host paths
    exist. Only literal paths are checked - anything with a shell variable is
    skipped because its value isn't known until the job actually runs."""
    issues = []
    if not _CONTAINER_CMD_RE.search(content):
        return issues
    seen = set()
    for m in re.finditer(r"([^\s'\"]+\.(?:sif|squashfs|img))\b", content):
        img = m.group(1)
        if "$" in img or img in seen:
            continue
        seen.add(img)
        resolved = img if os.path.isabs(img) else os.path.join(script_dir, img)
        if not os.path.exists(resolved):
            issues.append(
                f"Container image '{img}' (used with apptainer/singularity) was not found at "
                f"'{resolved}' - the job will fail when it tries to launch it."
            )
    for m in re.finditer(r"(?:--bind|-B)[= ]([^\s'\"]+)", content):
        for pair in m.group(1).split(","):
            src = pair.split(":")[0]
            if not src or "$" in src:
                continue
            resolved = src if os.path.isabs(src) else os.path.join(script_dir, src)
            if not os.path.exists(resolved):
                issues.append(
                    f"Container bind path '{src}' does not exist; --bind will fail or mount nothing."
                )
    return issues


def recommend_checkpointing(time_limit, content, threshold_hours=24):
    """HPC-28: a long walltime with no checkpoint/restart logic is a real risk."""
    secs = _time_str_to_seconds(time_limit)
    if secs is None or secs < threshold_hours * 3600:
        return None
    if _CHECKPOINT_RE.search(content):
        return None
    return (
        f"--time requests {_fmt_duration(secs)} with no checkpoint/restart logic detected. For a job "
        f"this long, a preemption or node failure loses all progress - consider periodic checkpointing."
    )


def recommend_modules(content, software_map):
    """HPC-22: map detected software usage to suggested module loads. Fully
    config-driven (software_map), so no library names are hardcoded in the tool."""
    recs = []
    for name, spec in (software_map or {}).items():
        if isinstance(spec, dict):
            pats = spec.get("match") or [name]
            mods = spec.get("modules") or []
        else:
            pats, mods = [name], [spec]
        if isinstance(pats, str):
            pats = [pats]
        if isinstance(mods, str):
            mods = [mods]
        try:
            if re.search("|".join(pats), content, re.IGNORECASE):
                recs.append((name, mods))
        except re.error:
            continue
    return recs


def recommend_partitions(config, gpu_count, mem_gb):
    """HPC-23: partition names that can actually satisfy the request."""
    out = []
    for name, spec in (config.get("partitions") or {}).items():
        if gpu_count and gpu_count > 0:
            if not spec.get("has_gpu"):
                continue
            gmax = spec.get("gpu_max")
            if gmax is not None and gpu_count > gmax:
                continue
        mem_max = spec.get("mem_gb_max")
        if mem_gb is not None and mem_max is not None and mem_gb > mem_max:
            continue
        out.append(name)
    return out


def check_script(script_path, config):
    content = read_script(script_path)
    issues = []
    script_dir = os.path.dirname(os.path.abspath(script_path))

    partitions = config.get("partitions", {})
    valid_partitions = set(partitions.keys())
    default_partition = next(
        (name for name, spec in partitions.items() if spec.get("is_default")),
        next(iter(valid_partitions), None),
    )
    required_fields = config.get("required_fields", [])
    recommended_fields = config.get("recommended_fields", [])
    default_time_days = config.get("default_time_days")
    slow_io_paths = config.get("slow_io_paths", [])

    partition = find_sbatch_value(content, "partition")
    cpus = find_sbatch_value(content, "cpus-per-task")
    mem = find_sbatch_value(content, "mem")
    time_limit = find_sbatch_value(content, "time")
    nodes = find_sbatch_value(content, "nodes")
    gpus = find_sbatch_value(content, "gpus")
    account = find_sbatch_value(content, "account")
    ntasks = find_sbatch_value(content, "ntasks")
    ntasks_per_node = find_sbatch_value(content, "ntasks-per-node")

    is_mpi_job = bool(
        (ntasks or ntasks_per_node)
        and re.search(r"\b(mpirun|mpiexec|srun)\b", content)
    )

    params_checked = {
        "partition": partition,
        "nodes": nodes,
        "gpus": gpus,
        "account": account,
        "cpus-per-task": cpus,
        "ntasks": ntasks,
        "ntasks-per-node": ntasks_per_node,
        "mem": mem,
        "time": time_limit,
    }

    # --- Partition validity ---
    if partition and partition not in valid_partitions:
        issues.append(
            f"Partition '{partition}' is not valid for {config.get('cluster_name', 'this cluster')}. "
            f"Valid options are: {', '.join(sorted(valid_partitions))}."
        )
    effective_partition = partition if partition in valid_partitions else default_partition
    partition_spec = partitions.get(effective_partition, {})

    # --- Required / recommended fields (generic to any Slurm cluster) ---
    if "nodes" in required_fields and not nodes:
        issues.append("No --nodes set. This cluster requires --nodes on every job script.")

    has_gpu = partition_spec.get("has_gpu", False)
    gpu_max = partition_spec.get("gpu_max")
    if "gpus" in required_fields and gpus is None:
        note = f" (use --gpus=0 on '{effective_partition}', which has no GPUs)" if not has_gpu else ""
        issues.append(f"No --gpus set. This cluster requires a GPU count on every job script{note}.")
    elif gpus is not None:
        gpu_count = parse_gpu_count(gpus)
        if gpu_count is None:
            issues.append(
                f"--gpus='{gpus}' isn't in a form hpclint can count; double-check the syntax."
            )
        elif not has_gpu and gpu_count != 0:
            issues.append(
                f"Requested --gpus={gpus}, but '{effective_partition}' has no GPUs. Set --gpus=0 or use a GPU partition."
            )
        elif has_gpu and gpu_max is not None and gpu_count > gpu_max:
            issues.append(
                f"Requested --gpus={gpus}, but '{effective_partition}' nodes only have {gpu_max} GPUs. Lower --gpus."
            )

    for field in recommended_fields:
        if not find_sbatch_value(content, field):
            issues.append(f"No --{field} set. This cluster recommends always setting --{field}.")

    # --- Resource ceilings for the resolved partition ---
    cpus_max = partition_spec.get("cpus_per_task_max")
    if cpus:
        if cpus_max is not None and int(cpus) > cpus_max:
            issues.append(
                f"Requested {cpus} CPUs, but '{effective_partition}' nodes only have {cpus_max} threads. "
                f"Lower --cpus-per-task."
            )
    elif not is_mpi_job:
        issues.append("No --cpus-per-task set (the scheduler will default to 1, which may waste your time budget).")
    # If it's an MPI job, absence of --cpus-per-task is expected (parallelism comes from --ntasks instead).

    # --- Total cores per node overflow check (ntasks-per-node x cpus-per-task) ---
    # Each piece can look fine alone; only the combination reveals it won't fit on one node.
    tasks_per_node = None
    if ntasks_per_node:
        tasks_per_node = int(ntasks_per_node)
    elif ntasks and nodes:
        # ntasks is a total across all nodes; estimate the per-node share.
        tasks_per_node = -(-int(ntasks) // int(nodes))  # ceiling division

    if tasks_per_node and cpus_max is not None:
        cpus_per_task_int = int(cpus) if cpus else 1  # Slurm defaults --cpus-per-task to 1 if unset
        total_cores_per_node = tasks_per_node * cpus_per_task_int
        if total_cores_per_node > cpus_max:
            issues.append(
                f"{tasks_per_node} tasks/node x {cpus_per_task_int} cpus-per-task = {total_cores_per_node} "
                f"cores per node, but '{effective_partition}' nodes only have {cpus_max} threads. "
                f"Lower --ntasks-per-node or --cpus-per-task."
            )

    mem_max = partition_spec.get("mem_gb_max")
    if mem:
        mem_gb = parse_mem_to_gb(mem)
        if mem_max is not None and mem_gb > mem_max:
            issues.append(
                f"Requested {mem} (~{mem_gb:.0f}G), but '{effective_partition}' nodes only have {mem_max}G RAM. "
                f"Lower --mem."
            )
    else:
        issues.append("No --mem set (job may get a low default allocation).")

    if not time_limit and default_time_days is not None:
        issues.append(
            f"No --time set. This cluster defaults to a {default_time_days}-day walltime if omitted — "
            f"fine for short jobs, but set it explicitly for anything you want the scheduler to plan around."
        )

    interactive_marker = detect_interactive(content, config.get("interactive_markers"))
    if interactive_marker:
        interactive_partitions = config.get("interactive_partitions", [])
        if interactive_partitions and effective_partition not in interactive_partitions:
            issues.append(
                f"This looks like an interactive session (matched '{interactive_marker}'), but it's "
                f"targeting partition '{effective_partition}'. Interactive work usually needs a "
                f"login/interactive queue ({', '.join(interactive_partitions)}) or `srun --pty`, "
                f"not a batch partition."
            )
        else:
            issues.append(
                f"This looks like an interactive session (matched '{interactive_marker}'). Interactive "
                f"work typically runs via `srun --pty` or an interactive queue; as a batch job it may "
                f"idle and waste your walltime allocation."
            )

    for rule in slow_io_paths:
        if any(marker in content for marker in rule.get("path_markers", [])):
            issues.append(
                f"Script references a slow storage path. Consider using {rule.get('recommend', 'a faster filesystem')} "
                f"for data-intensive read/write instead."
            )

    # --- Python environments living on slow storage (HPC-12) ---
    slow_markers = [m for rule in slow_io_paths for m in rule.get("path_markers", [])]
    if slow_markers:
        recommend = next((r.get("recommend", "a faster filesystem") for r in slow_io_paths),
                         "a faster filesystem")
        for env_path in find_envs_on_slow_paths(content, slow_markers, config.get("env_path_patterns")):
            issues.append(
                f"Python environment '{env_path}' lives on slow storage. Recreating it under "
                f"{recommend} speeds up job startup (env resolution off $HOME) and eases home-quota pressure."
            )

    # --- Containers: apptainer/singularity image + bind paths (HPC-27) ---
    issues.extend(check_containers(content, script_dir))

    # --- Checkpointing nudge for long walltimes (HPC-28) ---
    _ck = recommend_checkpointing(time_limit, content,
                                  config.get("checkpoint_time_threshold_hours", 24))
    if _ck:
        issues.append(_ck)

    # --- Software -> module-load recommendations (HPC-22) ---
    for _name, _mods in recommend_modules(content, config.get("software_map")):
        if _mods:
            issues.append(
                f"Detected '{_name}' usage - consider `module load {' '.join(_mods)}` "
                f"(verify with `module avail {_name}`)."
            )

    # --- Partition recommendation when none/invalid given (HPC-23) ---
    if partition is None or partition not in valid_partitions:
        _cands = recommend_partitions(
            config, parse_gpu_count(gpus), parse_mem_to_gb(mem) if mem else None)
        if _cands:
            issues.append(
                "No usable --partition set; based on your resource request, suitable "
                f"partition(s): {', '.join(sorted(_cands))}."
            )

    # --- Referenced file existence (best-effort; skips anything using a shell variable) ---
    for ref, resolved in find_referenced_files(content, script_dir):
        issues.append(
            f"Script references '{ref}', but no file was found at '{resolved}'. "
            f"Double-check the path before submitting — this job will fail immediately if it's missing."
        )

    return params_checked, issues
