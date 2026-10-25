"""Two-input, one-output reduction of V11 MonthlyCascadeISOPolicy.

Month is handled by twelve separate forests. Each regressor observes month-start
water level and the inflow of that same month, and predicts that month's end level.
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


FEATURE_NAMES = ("Start_Water_Level_m", "Inflow_m3s")
TARGET_NAME = "Target_Water_Level_m"
POLICY_VERSION = "single_monthly_iso_rf_v1"


def validate_hyperparameters(parameters: dict) -> dict:
    required = {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
    if set(parameters) != required:
        raise ValueError(f"RF parameters must be exactly {sorted(required)}")
    params = dict(parameters)
    for key in ("n_estimators", "min_samples_leaf"):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        params[key] = int(value)
    depth = params["max_depth"]
    if depth is not None:
        if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)) or depth <= 0:
            raise ValueError("max_depth must be null or a positive integer")
        params["max_depth"] = int(depth)
    features = params["max_features"]
    if features not in ("sqrt", "log2"):
        if isinstance(features, bool) or not isinstance(features, (float, int)) or not 0 < features <= 1:
            raise ValueError("max_features must be sqrt, log2 or a fraction in (0,1]")
        params["max_features"] = float(features)
    return params


@dataclass
class MonthlySingleRFPolicy:
    hyperparameters: dict
    random_seed: int
    month_models: dict
    month_domains: dict
    metadata: dict

    @classmethod
    def fit(cls, teacher: pd.DataFrame, *, hyperparameters: dict, random_seed: int,
            fit_n_jobs: int = 4, predict_n_jobs: int = 1, metadata: dict | None = None):
        params = validate_hyperparameters(hyperparameters)
        required = {"Calendar_Month", *FEATURE_NAMES, TARGET_NAME}
        if not required.issubset(teacher.columns):
            raise ValueError(f"Teacher is missing {sorted(required - set(teacher.columns))}")
        months = teacher["Calendar_Month"].to_numpy(dtype=float)
        if not np.all(np.isin(months, np.arange(1, 13))):
            raise ValueError("Teacher calendar months must be integers 1..12")
        if not np.isfinite(teacher[list(FEATURE_NAMES) + [TARGET_NAME]].to_numpy(float)).all():
            raise ValueError("Teacher features/targets must be finite")
        models, domains = {}, {}
        for month in range(1, 13):
            frame = teacher.loc[teacher["Calendar_Month"] == month]
            if frame.empty:
                raise ValueError(f"Teacher has no samples for month {month}")
            x = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
            y = frame[TARGET_NAME].to_numpy(dtype=float)
            model = RandomForestRegressor(
                random_state=int(random_seed), n_jobs=fit_n_jobs, **params
            )
            model.fit(x, y)
            model.n_jobs = predict_n_jobs
            models[month] = model
            domains[month] = [(float(x[:, i].min()), float(x[:, i].max())) for i in range(2)]
        record = {**(metadata or {}), "policy_version": POLICY_VERSION,
                  "feature_order": list(FEATURE_NAMES), "target": TARGET_NAME,
                  "month_handling": "twelve_separate_calendar_month_forests",
                  "hyperparameters": params, "random_seed": int(random_seed),
                  "fit_n_jobs": fit_n_jobs, "predict_n_jobs": predict_n_jobs}
        return cls(params, int(random_seed), models, domains, record)

    def predict(self, month: int, current_level: float, inflow: float):
        if month not in self.month_models:
            raise ValueError(f"Invalid prediction month {month}")
        row = np.asarray([current_level, inflow], dtype=float)
        if not np.isfinite(row).all():
            raise ValueError("Prediction features must be finite")
        raw = float(self.month_models[month].predict(row.reshape(1, 2))[0])
        if not np.isfinite(raw):
            raise FloatingPointError("RF produced a non-finite target")
        outside = np.asarray([
            row[i] < low or row[i] > high
            for i, (low, high) in enumerate(self.month_domains[month])
        ], dtype=bool)
        return raw, outside

    def save(self, path: Path):
        if self.metadata.get("policy_version") != POLICY_VERSION:
            raise ValueError("Policy metadata is incompatible")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path, *, required_role: str | None = None):
        # Only load trusted, locally produced model artifacts.
        with path.open("rb") as handle:
            policy = pickle.load(handle)
        if not isinstance(policy, cls) or policy.metadata.get("policy_version") != POLICY_VERSION:
            raise ValueError("Artifact is not a compatible ISO-RF policy")
        if required_role is not None and policy.metadata.get("artifact_role") != required_role:
            raise ValueError(f"Policy must have role={required_role}")
        if set(policy.month_models) != set(range(1, 13)):
            raise ValueError("Policy must contain twelve monthly forests")
        return policy
