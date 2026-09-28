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


def check_script(script_path, config):
    content = read_script(script_path)
    issues = []

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

    for rule in slow_io_paths:
        if any(marker in content for marker in rule.get("path_markers", [])):
            issues.append(
                f"Script references a slow storage path. Consider using {rule.get('recommend', 'a faster filesystem')} "
                f"for data-intensive read/write instead."
            )

    # --- Referenced file existence (best-effort; skips anything using a shell variable) ---
    script_dir = os.path.dirname(os.path.abspath(script_path))
    for ref, resolved in find_referenced_files(content, script_dir):
        issues.append(
            f"Script references '{ref}', but no file was found at '{resolved}'. "
            f"Double-check the path before submitting — this job will fail immediately if it's missing."
        )

    return params_checked, issues
