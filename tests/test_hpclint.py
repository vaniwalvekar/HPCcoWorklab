"""
Regression suite for hpclint.

Each test here corresponds to something we found and fixed by hand during
development (the MPI false positive, the memory-unit bug, the overflow
check, etc.). The point of this file is that none of those can silently
come back when the code changes later.
"""

import os

import pytest

from hpclint import check_script, parse_mem_to_gb, load_config

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRA_CONFIG = os.path.join(REPO_ROOT, "configs", "libra.yaml")


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

from hpclint.checker import find_sbatch_value, parse_gpu_count


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
