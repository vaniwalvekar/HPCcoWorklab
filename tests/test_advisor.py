"""Tests for hpclint.advisor (HPC-24 `ask`): pure, config-driven, no Slurm."""

from hpclint.advisor import (
    suggest_partition, suggest_modules, build_sbatch, suggest_submission,
)

CFG = {
    "partitions": {
        "compute": {"is_default": True, "has_gpu": False, "cpus_per_task_max": 64, "mem_gb_max": 1024},
        "gpu": {"has_gpu": True, "gpu_max": 2, "cpus_per_task_max": 64, "mem_gb_max": 512},
    },
    "software_map": {"gromacs": {"match": ["gmx"], "modules": ["GROMACS"]}},
}


def test_default_partition_when_no_gpu():
    assert suggest_partition(CFG, gpus=0, cpus=8, mem_gb=16) == "compute"


def test_gpu_partition_when_gpus_requested():
    assert suggest_partition(CFG, gpus=1, cpus=8) == "gpu"


def test_no_partition_when_gpu_count_too_high():
    assert suggest_partition(CFG, gpus=8, cpus=8) is None


def test_no_partition_when_mem_too_high():
    assert suggest_partition(CFG, gpus=0, cpus=8, mem_gb=2048) is None


def test_suggest_modules_known_unknown_none():
    assert suggest_modules(CFG, "gromacs") == ["GROMACS"]
    assert suggest_modules(CFG, "nope") == []
    assert suggest_modules(CFG, None) == []


def test_build_sbatch_reflects_choices():
    txt = build_sbatch("gpu", cpus=4, mem_gb=32, gpus=1, modules=["GROMACS"])
    assert "--partition=gpu" in txt
    assert "--gpus=1" in txt
    assert "--mem=32G" in txt
    assert "module load GROMACS" in txt
    assert "#SBATCH --cpus-per-task=4" in txt


def test_suggest_submission_tuple():
    part, mods, txt = suggest_submission(CFG, gpus=2, cpus=4, mem_gb=16, app="gromacs")
    assert part == "gpu"
    assert mods == ["GROMACS"]
    assert "--partition=gpu" in txt
