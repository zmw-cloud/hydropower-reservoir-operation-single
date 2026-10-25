"""
Shared input/output utility module for the unified reservoir operation framework.

This module provides common file-reading utilities used across different
algorithms and workflow stages, including:
1) loading and validating monthly inflow data from Excel files, and
2) loading algorithm-specific penalty coefficient candidates from YAML settings.

The purpose of this module is to centralize repeated data-loading logic,
reduce duplicated code across algorithm modules, and ensure that all algorithms
use a consistent and validated interface for reading shared input files and
experiment settings.
"""

from pathlib import Path
import calendar

import numpy as np
import pandas as pd
import yaml


def load_monthly_inflows_with_calendar(
    file_path: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load and validate monthly inflow data from Excel.

    The current logic follows the original DP implementation:
    - read the Excel file
    - skip the first column
    - keep the remaining monthly inflow columns

    Returns:
        A pandas DataFrame of shape [n_years, 12]
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Inflow file not found: {file_path}")

    raw_data = pd.read_excel(file_path)
    if raw_data.shape[1] != 13:
        raise ValueError(
            f"Expected Year plus 12 monthly inflow columns in {file_path}, "
            f"got {raw_data.shape[1]} columns."
        )

    years_numeric = pd.to_numeric(raw_data.iloc[:, 0], errors="coerce")
    if years_numeric.isnull().any():
        raise ValueError("Inflow data contains invalid calendar years.")
    years = years_numeric.to_numpy(dtype=float)
    if np.any(years != np.floor(years)):
        raise ValueError("Inflow calendar years must be whole numbers.")
    years = years.astype(np.int64)

    inflows = raw_data.iloc[:, 1:].copy()

    if len(inflows.shape) != 2:
        raise ValueError("Inflow data must be a 2D table.")
    if inflows.isnull().any().any():
        raise ValueError("Inflow data contains missing values.")
    if (inflows < 0).any().any():
        raise ValueError("Inflow data contains negative values.")

    if inflows.shape[1] != 12:
        raise ValueError("Inflow data must contain exactly 12 monthly columns.")

    return inflows, years


def load_monthly_inflows_from_excel(file_path: Path) -> pd.DataFrame:
    """Backward-compatible inflow-only loader."""
    inflows, _ = load_monthly_inflows_with_calendar(file_path)
    return inflows


def calendar_days_matrix(years: np.ndarray) -> np.ndarray:
    """Return Gregorian calendar days with shape ``[years, 12]``."""
    values = np.asarray(years)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Calendar years must be a nonempty one-dimensional array.")
    if np.any(values != np.floor(values)):
        raise ValueError("Calendar years must be whole numbers.")
    result = np.empty((len(values), 12), dtype=np.float64)
    for row, year in enumerate(values.astype(np.int64)):
        result[row] = [calendar.monthrange(int(year), month)[1] for month in range(1, 13)]
    return result


def mean_calendar_days_by_month(years: np.ndarray) -> np.ndarray:
    """Return the training-calendar mean duration for each calendar month."""
    return calendar_days_matrix(years).mean(axis=0)


def load_penalty_candidates(file_path: Path, algorithm: str) -> list[float]:
    """
    Load penalty candidates from penalty_settings.yaml.

    Expected structure:
        algorithm_name:
          candidates: [0.0, 0.1, ...]

    Example:
        dp:
          candidates: [0.0]
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Penalty settings file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    if algorithm not in settings or "candidates" not in settings[algorithm]:
        raise KeyError(f"Missing '{algorithm}.candidates' in penalty_settings.yaml.")

    candidates = [float(x) for x in settings[algorithm]["candidates"]]

    if len(candidates) == 0:
        raise ValueError(f"{algorithm}.candidates in penalty_settings.yaml is empty.")

    return candidates

def load_formal_penalty_candidates(data_dir: Path, algorithm: str) -> list[float]:
    """Load only the new-objective list; never fall back to legacy lambdas."""
    return load_penalty_candidates(data_dir / "violation_count_penalty_settings.yaml", algorithm)

def load_sdp_inflow_intervals(file_path: Path, n_inflow_discrete: int):
    """
    Load discretized inflow intervals for SDP training.

    Expected CSV columns:
        Month
        Inflow_Class
        Lower_Bound_m3s
        Upper_Bound_m3s
        Probability

    Returns:
        inflows: representative inflow DataFrame [12 x n]
        flow_probs: probability array [n]
        flow_intervals: interval bounds DataFrame [12 x n]
    """
    if not file_path.exists():
        raise FileNotFoundError(f"SDP inflow interval file not found: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    required_columns = {
        "Month",
        "Inflow_Class",
        "Lower_Bound_m3s",
        "Upper_Bound_m3s",
        "Probability",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in {file_path}: {sorted(missing)}")

    n = n_inflow_discrete
    inflows = pd.DataFrame(index=range(1, 13), columns=range(n))
    flow_intervals = pd.DataFrame(index=range(1, 13), columns=range(n))
    flow_probs = np.zeros(n)

    for month in range(1, 13):
        month_data = df[df["Month"] == month].sort_values("Inflow_Class")

        for i in range(n):
            if i < len(month_data):
                row = month_data.iloc[i]
                inflows.loc[month, i] = (
                    row["Lower_Bound_m3s"] + row["Upper_Bound_m3s"]
                ) / 2.0
                flow_intervals.loc[month, i] = (
                    row["Lower_Bound_m3s"],
                    row["Upper_Bound_m3s"],
                )
                if month == 1:
                    flow_probs[i] = row["Probability"]

    inflows = inflows.astype(float)
    return inflows, flow_probs, flow_intervals

def load_experiment_settings(file_path: Path) -> dict:
    """
    Load experiment_settings.yaml.

    Returns:
        Entire yaml dictionary.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Experiment settings file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    if settings is None:
        raise ValueError(
            "experiment_settings.yaml is empty."
        )

    return settings

def load_algorithm_settings(
    file_path: Path,
    algorithm: str,
) -> dict:
    """
    Load configuration section for one algorithm.
    """
    settings = load_experiment_settings(file_path)

    if algorithm not in settings:
        raise KeyError(
            f"Missing '{algorithm}' section "
            f"in experiment_settings.yaml."
        )

    return settings[algorithm]

def require_keys(settings: dict, keys: list[str], prefix: str) -> None:
    missing = [key for key in keys if key not in settings]
    if missing:
        raise KeyError(f"Missing required config keys in {prefix}: {missing}")
