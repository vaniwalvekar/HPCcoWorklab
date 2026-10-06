"""hpccoworklab — a cluster-agnostic linter for Slurm job scripts."""

from .checker import check_script, parse_mem_to_gb, load_config, find_referenced_files

__version__ = "0.1.0"

__all__ = ["check_script", "parse_mem_to_gb", "load_config", "find_referenced_files"]
