"""
Regression suite for hpccoworklab.

Each test here corresponds to something we found and fixed by hand during
development (the MPI false positive, the memory-unit bug, the overflow
check, etc.). The point of this file is that none of those can silently
come back when the code changes later.
"""

import os

import pytest

from hpccoworklab import check_script, parse_mem_to_gb, load_config

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRA_CONFIG = os.path.join(REPO_ROOT, "hpccoworklab", "configs", "libra.yaml")


def fixture_path(name):
    return os.path.join(FIXTURES, name)


@pytest.fixture(scope="module")
def libra_config():
    return load_config(LIBRA_CONFIG)


def issue_text(issues):
    """Flatten issues into one string for simple substring assertions."""
    return " | ".join(issues)


# --- Memory unit parsing -----------------------------------------------

def test_mem_parses_mb_correctly():
    # Regression: 8000MB was once misread as 8000GB because the parser only
    # recognized single-letter units.
    assert parse_mem_to_gb("8000MB") == pytest.approx(7.8125)


def test_mem_parses_gb_and_g_the_same():
    assert parse_mem_to_gb("500GB") == parse_mem_to_gb("500G") == 500


def test_mem_parses_tb_and_t_the_same():
    assert parse_mem_to_gb("2TB") == parse_mem_to_gb("2T") == 2048


def test_mem_no_unit_is_mb():
    # Slurm treats a bare --mem value as MEGABYTES, not GB.
    assert parse_mem_to_gb("1200") == pytest.approx(1200 / 1024)
    assert parse_mem_to_gb("8000") == pytest.approx(7.8125)


# --- HPC-34: directive + GPU-spec parsing robustness ----------------------

from hpccoworklab.checker import (
    find_sbatch_value, parse_gpu_count, get_gpu_request,
    recommend_checkpointing, recommend_modules, recommend_partitions,
    check_containers, _time_str_to_seconds,
    check_array_usage, check_conflicting_modules,
)


def test_find_sbatch_value_space_form():
    assert find_sbatch_value("#SBATCH --nodes 2", "nodes") == "2"


def test_find_sbatch_value_equals_form_still_works():
    assert find_sbatch_value("#SBATCH --nodes=3", "nodes") == "3"


def test_find_sbatch_value_short_option():
    assert find_sbatch_value("#SBATCH -N 2", "nodes") == "2"
    assert find_sbatch_value("#SBATCH -p gpu", "partition") == "gpu"


def test_find_sbatch_value_bundled_short_option():
    assert find_sbatch_value("#SBATCH -N2", "nodes") == "2"


def test_find_sbatch_value_short_not_confused_with_long():
    # -p must not accidentally match the 'p' inside '--partition'
    assert find_sbatch_value("#SBATCH --partition=compute", "partition") == "compute"


def test_parse_gpu_count_plain():
    assert parse_gpu_count("2") == 2
    assert parse_gpu_count("0") == 0


def test_parse_gpu_count_name_colon():
    assert parse_gpu_count("a100:2") == 2
    assert parse_gpu_count("gpu:4") == 4


def test_parse_gpu_count_bare_name_is_one():
    assert parse_gpu_count("a100") == 1


def test_parse_gpu_count_unparseable_is_none():
    assert parse_gpu_count("garbage:xyz") is None


# --- HPC-13: interactive-vs-batch detection --------------------------------

_CFG = {
    "cluster_name": "X",
    "partitions": {"compute": {"is_default": True, "has_gpu": False,
                               "cpus_per_task_max": 8, "mem_gb_max": 64}},
    "required_fields": [], "recommended_fields": [],
}


def _job(tmp_path, text):
    p = tmp_path / "job.sh"
    p.write_text(text)
    return str(p)


def test_interactive_detected(tmp_path):
    s = _job(tmp_path, "#!/bin/bash\n#SBATCH --partition compute\njupyter notebook\n")
    _, issues = check_script(s, _CFG)
    assert any("interactive" in i.lower() for i in issues)


def test_interactive_wrong_partition_names_alternatives(tmp_path):
    cfg = dict(_CFG)
    cfg["interactive_partitions"] = ["interactive", "login"]
    s = _job(tmp_path, "#!/bin/bash\n#SBATCH --partition compute\njupyter lab\n")
    _, issues = check_script(s, cfg)
    assert any("interactive" in i.lower() and "login" in i for i in issues)


def test_normal_batch_job_not_flagged_interactive(tmp_path):
    s = _job(tmp_path, "#!/bin/bash\n#SBATCH --partition compute\n#SBATCH --mem 4G\npython train.py\n")
    _, issues = check_script(s, _CFG)
    assert not any("interactive" in i.lower() for i in issues)


# --- MPI vs threaded detection -------------------------------------------

def test_mpi_job_not_flagged_for_missing_cpus_per_task(libra_config):
    # Regression: a real production GROMACS script (8-rank MPI, no
    # --cpus-per-task) was wrongly flagged as "may waste your time budget."
    _, issues = check_script(fixture_path("mpi_job.sh"), libra_config)
    assert not any("cpus-per-task set" in i for i in issues)


def test_non_mpi_job_still_flagged_for_missing_cpus_per_task(libra_config):
    # Same rule, opposite direction: no --ntasks and no mpirun/srun means
    # Slurm really would default to 1 core, so this should still be flagged.
    _, issues = check_script(fixture_path("no_cpus_specified_job.sh"), libra_config)
    assert any("cpus-per-task set" in i for i in issues)


# --- Core overflow check --------------------------------------------------

def test_overflow_detected(libra_config):
    _, issues = check_script(fixture_path("overflow_job.sh"), libra_config)
    assert any("cores per node" in i for i in issues)


def test_no_overflow_for_reasonable_mpi_job(libra_config):
    _, issues = check_script(fixture_path("mpi_job.sh"), libra_config)
    assert not any("cores per node" in i for i in issues)


# --- File existence check -------------------------------------------------

def test_missing_referenced_files_flagged(libra_config):
    _, issues = check_script(fixture_path("missing_file_job.sh"), libra_config)
    text = issue_text(issues)
    assert "nonexistent_script.py" in text
    assert "missing_data.csv" in text


def test_existing_referenced_file_not_flagged(libra_config):
    _, issues = check_script(fixture_path("existing_file_job.sh"), libra_config)
    assert not any("no file was found" in i for i in issues)


def test_env_var_paths_are_skipped_not_false_flagged(libra_config):
    # We can't resolve $SCRATCH/$HOME without running the job, so these
    # should be silently skipped rather than reported as missing.
    _, issues = check_script(fixture_path("env_var_job.sh"), libra_config)
    assert not any("no file was found" in i for i in issues)


# --- Partition handling ----------------------------------------------------

def test_typo_partition_flagged(libra_config):
    _, issues = check_script(fixture_path("typo_partition_job.sh"), libra_config)
    assert any("not valid" in i for i in issues)


def test_valid_gpu_partition_uses_its_own_limits(libra_config):
    _, issues = check_script(fixture_path("gpu_job.sh"), libra_config)
    # 2 GPUs is within the gpu partition's limit of 2 (per configs/libra.yaml)
    assert not any("GPUs" in i for i in issues)


# --- Storage guidance -------------------------------------------------------

def test_home_dir_usage_flagged(libra_config):
    _, issues = check_script(fixture_path("home_dir_job.sh"), libra_config)
    assert any("slow storage path" in i for i in issues)


# --- HPC-12: python environments on slow storage ---------------------------

def _env_cfg():
    return {
        "cluster_name": "X",
        "partitions": {"compute": {"is_default": True, "has_gpu": False,
                                   "cpus_per_task_max": 8, "mem_gb_max": 64}},
        "required_fields": [], "recommended_fields": [],
        "slow_io_paths": [{"path_markers": ["/home/", "$HOME"], "recommend": "$SCRATCH"}],
    }


def _env_job(tmp_path, text):
    p = tmp_path / "job.sh"
    p.write_text(text)
    return str(p)


def _env_warnings(issues):
    return [i for i in issues if "environment" in i.lower() and "slow" in i.lower()]


def test_env_on_home_warned(tmp_path):
    s = _env_job(tmp_path, "#!/bin/bash\n#SBATCH --partition compute\nsource $HOME/.venv/bin/activate\n")
    _, issues = check_script(s, _env_cfg())
    assert _env_warnings(issues)


def test_env_on_scratch_not_warned(tmp_path):
    s = _env_job(tmp_path, "#!/bin/bash\n#SBATCH --partition compute\nsource /scratch/u/.venv/bin/activate\n")
    _, issues = check_script(s, _env_cfg())
    assert not _env_warnings(issues)


def test_bare_conda_activate_no_path_no_warning(tmp_path):
    s = _env_job(tmp_path, "#!/bin/bash\n#SBATCH --partition compute\nconda activate myenv\n")
    _, issues = check_script(s, _env_cfg())
    assert not _env_warnings(issues)


def test_miniconda_under_home_warned(tmp_path):
    s = _env_job(tmp_path, "#!/bin/bash\n#SBATCH --partition compute\nconda activate $HOME/miniconda3/envs/ml\n")
    _, issues = check_script(s, _env_cfg())
    assert _env_warnings(issues)


# --- HPC-51: bundled config resolution -------------------------------------

def test_load_config_bundled_by_name():
    from hpccoworklab.checker import load_config
    cfg = load_config("libra")          # no path -> resolves to packaged hpccoworklab/configs/libra.yaml
    assert cfg.get("cluster_name") == "SLU Libra"


# --- HPC-35: --gres=gpu:N recognized as a GPU request ----------------------

def test_gres_gpu_counted():
    r = get_gpu_request("#SBATCH --gres=gpu:1\n")
    assert r["present"] and r["per_node"] == 1

def test_gpus_still_counted():
    r = get_gpu_request("#SBATCH --gpus=2\n")
    assert r["present"] and r["total"] == 2

def test_gres_type_colon_count():
    r = get_gpu_request("#SBATCH --gres=gpu:a100:2\n")
    assert r["per_node"] == 2

def test_gpus_per_node_counted():
    r = get_gpu_request("#SBATCH --gpus-per-node=1\n")
    assert r["present"] and r["per_node"] == 1

def test_no_gpu_request():
    assert not get_gpu_request("#SBATCH --partition compute\n")["present"]

def test_gres_non_gpu_ignored():
    assert not get_gpu_request("#SBATCH --gres=mpol:2\n")["present"]

def test_gres_job_not_falsely_missing_gpu(tmp_path):
    cfg = {"cluster_name": "X",
           "partitions": {"gpu": {"is_default": True, "has_gpu": True, "gpu_max": 2,
                                  "cpus_per_task_max": 64, "mem_gb_max": 512}},
           "required_fields": ["gpus"], "recommended_fields": ["account"]}
    s = _job(tmp_path, "#!/bin/bash\n#SBATCH --partition gpu\n#SBATCH --gres=gpu:1\n"
                       "#SBATCH --account a1\n#SBATCH --mem=8G\n#SBATCH --time 01:00:00\n"
                       "#SBATCH --cpus-per-task 2\n")
    _, issues = check_script(s, cfg)
    assert not any("No GPU request" in i for i in issues)
    assert not any("has no GPUs" in i for i in issues)


# --- Batch 1 (HPC-22/23/27/28) ---------------------------------------------

def test_time_str_to_seconds_forms():
    assert _time_str_to_seconds("01:00:00") == 3600
    assert _time_str_to_seconds("1-00:00:00") == 86400
    assert _time_str_to_seconds("30:00") == 1800
    assert _time_str_to_seconds("bogus") is None


def test_checkpoint_nudge_long_no_marker():
    assert recommend_checkpointing("48:00:00", "python train.py")


def test_checkpoint_none_short():
    assert recommend_checkpointing("01:00:00", "python train.py") is None


def test_checkpoint_none_when_present():
    assert recommend_checkpointing("48:00:00", "python train.py --checkpoint ckpt.pt") is None


def test_recommend_modules_config_driven():
    sm = {"gromacs": {"match": [r"\bgmx(_mpi)?\b"], "modules": ["GROMACS"]}}
    assert recommend_modules("gmx_mpi mdrun -s top.tpr", sm) == [("gromacs", ["GROMACS"])]
    assert recommend_modules("python x.py", sm) == []
    assert recommend_modules("anything", {}) == []


def test_recommend_partitions():
    cfg = {"partitions": {"compute": {"has_gpu": False, "mem_gb_max": 1024},
                          "gpu": {"has_gpu": True, "gpu_max": 2, "mem_gb_max": 512}}}
    assert recommend_partitions(cfg, 2, 100) == ["gpu"]
    assert sorted(recommend_partitions(cfg, 0, 100)) == ["compute", "gpu"]
    assert recommend_partitions(cfg, 8, 100) == []      # no partition has 8 GPUs
    assert recommend_partitions(cfg, 0, 4096) == []     # mem exceeds all nodes


def test_check_containers_missing_image(tmp_path):
    issues = check_containers("apptainer run /no/such/img.sif\n", str(tmp_path))
    assert any("Container image" in i for i in issues)


def test_check_containers_existing_image(tmp_path):
    img = tmp_path / "app.sif"; img.write_text("x")
    issues = check_containers(f"apptainer run {img} data\n", str(tmp_path))
    assert not any("Container image" in i for i in issues)


def test_check_containers_bind_missing(tmp_path):
    issues = check_containers("singularity exec --bind /definitely/missing:/data img.sif\n", str(tmp_path))
    assert any("bind path" in i for i in issues)


def test_check_containers_skips_variable_paths():
    # $SCRATCH etc. are unknown until run; must NOT be flagged as missing.
    issues = check_containers("apptainer run --bind $SCRATCH/data:/data $IMAGES/app.sif\n", "/x")
    assert issues == []


def test_check_partition_recommendation_when_missing(tmp_path):
    cfg = {"cluster_name": "X",
           "partitions": {"compute": {"is_default": True, "has_gpu": False, "cpus_per_task_max": 8, "mem_gb_max": 64},
                          "gpu": {"is_default": False, "has_gpu": True, "gpu_max": 2, "cpus_per_task_max": 8, "mem_gb_max": 64}},
           "required_fields": [], "recommended_fields": []}
    s = _job(tmp_path, "#!/bin/bash\n#SBATCH --gpus=2\n#SBATCH --mem=16G\n#SBATCH --time 01:00:00\npython t.py\n")
    _, issues = check_script(s, cfg)
    assert any("partition" in i.lower() and "gpu" in i for i in issues)


def test_check_checkpoint_nudge_in_output(tmp_path):
    s = _job(tmp_path, "#!/bin/bash\n#SBATCH --partition compute\n#SBATCH --time 48:00:00\npython train.py\n")
    _, issues = check_script(s, _CFG)
    assert any("checkpoint" in i.lower() for i in issues)


# --- Batch 2 (HPC-30 array / HPC-42 conflicting modules) -------------------

def test_array_without_task_id_flagged():
    issues = check_array_usage("#SBATCH --array=1-100\npython run.py")
    assert any("SLURM_ARRAY_TASK_ID" in i for i in issues)


def test_array_with_task_id_ok():
    issues = check_array_usage("#SBATCH --array=1-100\npython run.py --id $SLURM_ARRAY_TASK_ID")
    assert not any("task id" in i.lower() or "SLURM_ARRAY_TASK_ID" in i for i in issues)


def test_array_malformed_flagged():
    issues = check_array_usage("#SBATCH --array=abc\npython run.py $SLURM_ARRAY_TASK_ID")
    assert any("valid Slurm array" in i for i in issues)


def test_array_no_array_no_output():
    assert check_array_usage("#SBATCH --partition compute\npython run.py") == []


def test_array_with_max_percent_valid():
    issues = check_array_usage("#SBATCH --array=1-1000%8\nfor f in data/$SLURM_ARRAY_TASK_ID/*; do run $f; done")
    assert issues == []


def test_conflicting_modules_detected():
    cfg_groups = [["cuda/11", "cuda/12"]]
    issues = check_conflicting_modules("module load cuda/11 hdf5\nmodule load cuda/12", cfg_groups)
    assert issues and "cuda/11" in issues[0] and "cuda/12" in issues[0]


def test_conflicting_modules_single_load_ok():
    assert check_conflicting_modules("module load cuda/12", [["cuda/11", "cuda/12"]]) == []


def test_conflicting_modules_empty_config_noop():
    assert check_conflicting_modules("module load cuda/11 cuda/12", []) == []


def test_check_conflicting_integration(tmp_path):
    cfg = dict(_CFG); cfg["conflicting_modules"] = [["gcc/9", "gcc/12"]]
    s = _job(tmp_path, "#!/bin/bash\n#SBATCH --partition compute\nmodule load gcc/9\ngcc --version\nmodule load gcc/12\n")
    _, issues = check_script(s, cfg)
    assert any("conflict" in i.lower() for i in issues)
