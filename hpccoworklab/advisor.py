"""
hpccoworklab.advisor — turn a short description of a workload into a suggested
partition, module loads, and a ready-to-edit sbatch skeleton (HPC-24, `ask`).

Deterministic and config-driven: it reads only the cluster YAML
(partitions + software_map). No LLM, no live Slurm required.
"""


def _partition_fits(spec, gpus, cpus, mem_gb):
    if gpus and gpus > 0:
        if not spec.get("has_gpu"):
            return False
        gmax = spec.get("gpu_max")
        if gmax is not None and gpus > gmax:
            return False
    cmax = spec.get("cpus_per_task_max")
    if cmax is not None and cpus is not None and cpus > cmax:
        return False
    mmax = spec.get("mem_gb_max")
    if mem_gb is not None and mmax is not None and mem_gb > mmax:
        return False
    return True


def suggest_partition(config, gpus=0, cpus=None, mem_gb=None):
    """First partition that can satisfy the request; prefer the default when
    it fits. Returns the partition name or None."""
    parts = config.get("partitions") or {}
    fits = [name for name, spec in parts.items()
            if _partition_fits(spec, gpus, cpus, mem_gb)]
    for name in fits:
        if parts[name].get("is_default") and not gpus:
            return name
    return fits[0] if fits else None


def suggest_modules(config, app):
    """Module load commands for a named application from software_map."""
    if not app:
        return []
    spec = (config.get("software_map") or {}).get(app)
    if isinstance(spec, dict):
        mods = spec.get("modules") or []
    elif isinstance(spec, str):
        mods = [spec]
    else:
        return []
    return list(mods)


def build_sbatch(partition, cpus=1, mem_gb=None, gpus=0, time_limit=None,
                 nodes=1, modules=None):
    lines = ["#!/bin/bash", f"#SBATCH --nodes={nodes}"]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if gpus:
        lines.append(f"#SBATCH --gpus={gpus}")
    lines.append(f"#SBATCH --cpus-per-task={cpus}")
    if mem_gb is not None:
        lines.append(f"#SBATCH --mem={mem_gb}G")
    lines.append(f"#SBATCH --time={time_limit or '01:00:00'}")
    for mod in (modules or []):
        lines.append(f"module load {mod}")
    lines.append("")
    lines.append("# TODO: replace with your run command, e.g.:")
    lines.append("# srun python train.py")
    lines.append("")
    return "\n".join(lines)


def suggest_submission(config, gpus=0, cpus=1, mem_gb=None, time_limit=None, app=None):
    """Return (partition, modules, script_text) recommendation. Pure + testable."""
    partition = suggest_partition(config, gpus=gpus, cpus=cpus, mem_gb=mem_gb)
    modules = suggest_modules(config, app)
    script = build_sbatch(partition, cpus=cpus, mem_gb=mem_gb, gpus=gpus,
                          time_limit=time_limit, modules=modules)
    return partition, modules, script
