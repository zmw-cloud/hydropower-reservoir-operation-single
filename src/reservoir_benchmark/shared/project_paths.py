"""
Shared project path utility module for the unified reservoir operation framework.

This module provides a centralized set of helper functions for locating and
managing the core directory structure of the project, including:
1) the repository root directory,
2) the shared data directory,
3) the outputs directory,
4) policy-generation output folders for individual algorithms, and
5) file-name formatting utilities for parameterized experiment outputs.

The purpose of this module is to avoid repeated path definitions across
different algorithms and scripts, and to ensure that all modules follow the
same repository-level directory convention for reading inputs and saving results.
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    Return the repository root directory.

    File location:
        src/reservoir_benchmark/shared/project_paths.py

    Therefore:
        parents[0] -> shared
        parents[1] -> reservoir_benchmark
        parents[2] -> src
        parents[3] -> repository root
    """
    return Path(__file__).resolve().parents[3]


def get_data_dir() -> Path:
    """
    Return the shared data directory under the repository root.
    """
    return get_project_root() / "data"


def get_outputs_dir() -> Path:
    """
    Return the outputs directory under the repository root.
    """
    return get_project_root() / "outputs"

def get_formal_outputs_dir() -> Path:
    """Physically isolated root for the formal violation-count objective."""
    return get_outputs_dir() / "violation_count_penalty" / "xiaowan_single_violation_count_v1"

def get_formal_algorithm_dir(algorithm: str) -> Path:
    return get_formal_outputs_dir() / algorithm


def get_policy_generation_dir(algorithm: str) -> Path:
    """
    Return the policy-generation output directory for a given algorithm.

    Example:
        outputs/policy_generation/dp
    """
    return get_outputs_dir() / "policy_generation" / algorithm

def get_preprocessed_dir(algorithm: str) -> Path:
    return get_policy_generation_dir(algorithm) / "preprocessed"

def format_lambda_for_filename(lam: float) -> str:
    """
    Format lambda values for file names.

    Examples:
        0.0 -> "0"
        0.4 -> "0.4"
        1.0 -> "1"
    """
    if abs(lam - int(lam)) < 1e-9:
        return str(int(lam))
    return str(lam)
