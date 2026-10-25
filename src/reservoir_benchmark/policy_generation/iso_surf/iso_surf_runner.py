"""
ISO-SURF policy fitting and persistence module for the unified reservoir
operation framework.

This module fits monthly ISO-SURF rule surfaces from upstream DP training
trajectories and persists the fitted policy as a PKL artifact, including:
1) preparing month-wise training pairs (S_t, I_t, S_{t+1}),
2) fitting monthly spline surfaces with health checks,
3) falling back to linear planes when spline fitting is unstable,
4) predicting next-period water levels from the fitted rule surface, and
5) saving and loading the fitted ISO-SURF policy artifact.

The purpose of this module is to keep ISO-SURF policy generation independent
from evaluation and to standardize policy persistence under the unified
repository structure.
"""

from __future__ import annotations

import pickle
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline

from reservoir_benchmark.policy_generation.iso_surf.iso_preprocessor import ISOPreprocessor
from reservoir_benchmark.shared.io_utils import (
    load_algorithm_settings,
    load_formal_penalty_candidates,
    require_keys,
)
from reservoir_benchmark.shared.project_paths import (
    format_lambda_for_filename,
    get_data_dir,
    get_formal_algorithm_dir,
)
from reservoir_benchmark.shared.experiment_contract import (
    DDP_SOLVER_VARIANT, require_contract, sha256_file,
)


class ISOSurfPolicyFitter:
    """
    Fit and persist monthly ISO-SURF policy surfaces.
    """

    def __init__(
        self,
        smoothing_factor: float = 1.0,
        spline_order: tuple[int, int] = (3, 3),
        grid_size: int = 25,
        max_factor: float = 10000.0,
        min_samples_spline: int = 20,
    ):
        self.smoothing_factor = float(smoothing_factor)
        self.spline_order = tuple(int(x) for x in spline_order)
        self.grid_size = int(grid_size)
        self.max_factor = float(max_factor)
        self.min_samples_spline = int(min_samples_spline)

        if len(self.spline_order) != 2:
            raise ValueError("iso_surf.spline_order must contain exactly two integers.")

        self.month_models: dict[int, dict] = {}
        self.output_dir = get_formal_algorithm_dir("iso_surf") / "policy"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_dataframe(self, dp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DP trajectory data for month-wise ISO-SURF fitting.
        """
        # ISO-SURF fitting requires the current month index, current water level,
        # and current inflow as inputs. The target is the next-period water level.
        required_cols = [
            "Month",
            "Start_Water_Level_m",
            "Target_Water_Level_m",
            "Inflow_m3s",
        ]
        for col in required_cols:
            if col not in dp_df.columns:
                raise ValueError(f"DP training DataFrame is missing required column: {col}")

        df = dp_df.copy()

        # DDP exports explicit within-month timing.  The supervised pair is:
        # (month-start level, current-month inflow) -> month-end target level.
        # No cross-row shift is allowed because that would train an end_t -> end_t+1
        # mapping while deployment requests a start_t -> end_t action.
        df["Water_Level_m"] = df["Start_Water_Level_m"].astype(float)
        df["Next_Level_m"] = df["Target_Water_Level_m"].astype(float)

        # Convert the continuous month index in the DP trajectory into a month-of-year
        # label in the range 1-12, so each month can be fitted independently.
        if "Calendar_Month" in df.columns:
            df["MonthOfYear"] = df["Calendar_Month"].astype(int)
        else:
            df["MonthOfYear"] = ((df["Month"] - 1) % 12) + 1

        # Remove any incomplete rows before fitting.
        df = df.dropna(subset=["Water_Level_m", "Inflow_m3s", "Next_Level_m"])
        return df

    def fit_from_dp_dataframe(self, dp_df: pd.DataFrame) -> None:
        """
        Fit month-wise ISO-SURF rule surfaces from a DP monthly trajectory DataFrame.
        """
        df = self._prepare_dataframe(dp_df)

        # Use a positive smoothing factor. If the user provides a non-positive
        # value, fall back to a default value of 1.0.
        base_sf = self.smoothing_factor if self.smoothing_factor > 0 else 1.0

        def _check_spline_health(S_norm, I_norm, Z_data, spline, grid_size=25):
            """
            Check whether a fitted spline behaves reasonably on a regular grid.

            The spline is rejected if:
            1) it produces non-finite values,
            2) its predicted range is far outside the observed training range, or
            3) its overall variation is excessively large compared with the data.
            """
            Sg = np.linspace(S_norm.min(), S_norm.max(), grid_size)
            Ig = np.linspace(I_norm.min(), I_norm.max(), grid_size)
            Zg = spline(Sg, Ig)

            if not np.all(np.isfinite(Zg)):
                return False, np.nan, np.nan, np.nan, np.nan

            z_min = float(Zg.min())
            z_max = float(Zg.max())

            data_min = float(Z_data.min())
            data_max = float(Z_data.max())
            data_range = max(data_max - data_min, 1e-6)
            grid_range = z_max - z_min

            # Allow a moderate extrapolation margin, but reject strongly abnormal
            # predictions that go far beyond the observed target range.
            margin = max(5.0 * data_range, 50.0)
            if z_min < data_min - margin or z_max > data_max + margin:
                return False, z_min, z_max, data_min, data_max

            # Reject splines that oscillate too strongly over the fitted domain.
            max_allowed_range = max(10.0 * data_range, 200.0)
            if grid_range > max_allowed_range:
                return False, z_min, z_max, data_min, data_max

            return True, z_min, z_max, data_min, data_max

        # Clear any previously fitted month models before fitting a new policy.
        self.month_models = {}

        # Fit one rule surface per calendar month.
        for month in range(1, 13):
            df_m = df[df["MonthOfYear"] == month].copy()
            n_m = len(df_m)

            # If the available monthly samples are too few, skip spline fitting
            # and directly use the linear fallback model.
            if n_m < self.min_samples_spline:
                self._fit_linear_only(month, df_m)
                continue

            S_t = df_m["Water_Level_m"].values
            I_t = df_m["Inflow_m3s"].values
            Z = df_m["Next_Level_m"].values

            # Normalize the two input dimensions before spline fitting so that
            # water level and inflow have comparable numerical scales.
            S_mean, S_std = S_t.mean(), S_t.std()
            I_mean, I_std = I_t.mean(), I_t.std()
            S_std = S_std if S_std > 0 else 1.0
            I_std = I_std if I_std > 0 else 1.0

            S_norm = (S_t - S_mean) / S_std
            I_norm = (I_t - I_mean) / I_std

            # Start from the base smoothing level and gradually increase it if
            # the spline is unstable or Fitpack reports warnings.
            factor = base_sf
            max_factor = base_sf * self.max_factor
            best_spline = None

            while factor <= max_factor:
                s_val = factor * n_m

                # Capture warnings raised during spline fitting. Some warnings
                # indicate that the smoothing is too small or the fit is
                # numerically unstable.
                with warnings.catch_warnings(record=True) as wlist:
                    warnings.simplefilter("always")
                    try:
                        spline = SmoothBivariateSpline(
                            S_norm,
                            I_norm,
                            Z,
                            kx=self.spline_order[0],
                            ky=self.spline_order[1],
                            s=s_val,
                        )
                    except Exception:
                        # If the fit fails completely, increase smoothing and retry.
                        factor *= 5.0
                        continue

                has_fitpack_warning = False
                for w in wlist:
                    msg = str(w.message)
                    if (
                        "storage space" in msg
                        or "s too small" in msg
                        or "The required storage space" in msg
                    ):
                        has_fitpack_warning = True
                        break

                # If Fitpack indicates a problematic fit, use a larger smoothing value.
                if has_fitpack_warning:
                    factor *= 5.0
                    continue

                # Even if fitting formally succeeds, reject the spline if it is
                # numerically unhealthy on a regular evaluation grid.
                ok, _, _, _, _ = _check_spline_health(
                    S_norm,
                    I_norm,
                    Z,
                    spline,
                    grid_size=self.grid_size,
                )
                if not ok:
                    factor *= 5.0
                    continue

                # Accept the first spline that passes all health checks.
                best_spline = spline
                break

            # If no acceptable spline can be obtained, fall back to a linear surface.
            if best_spline is None:
                self._fit_linear_only(month, df_m)
                continue

            # Store the accepted month model together with the normalization
            # statistics required later for prediction.
            self.month_models[month] = {
                "mode": "spline",
                "spline": best_spline,
                "S_mean": S_mean,
                "S_std": S_std,
                "I_mean": I_mean,
                "I_std": I_std,
                "S_range": (float(S_t.min()), float(S_t.max())),
                "I_range": (float(I_t.min()), float(I_t.max())),
                "n_samples": n_m,
            }

        print("ISO-SURF month models fitted for months:", sorted(self.month_models.keys()))

    def _fit_linear_only(self, month: int, df_m: pd.DataFrame) -> None:
        """
        Fit a linear fallback plane:
            S_{t+1} = a + b S_t + c I_t
        """
        if df_m.empty:
            self.month_models[month] = {
                "mode": "identity",
                "n_samples": 0,
            }
            return

        S_t = df_m["Water_Level_m"].values
        I_t = df_m["Inflow_m3s"].values
        Z = df_m["Next_Level_m"].values

        # Fit a simple least-squares linear plane when spline fitting is not
        # possible or not reliable.
        X = np.column_stack([np.ones_like(S_t), S_t, I_t])
        coef, *_ = np.linalg.lstsq(X, Z, rcond=None)

        self.month_models[month] = {
            "mode": "linear",
            "coef": coef,
            "S_range": (float(S_t.min()), float(S_t.max())),
            "I_range": (float(I_t.min()), float(I_t.max())),
            "n_samples": len(df_m),
        }

    def predict_next_level(self, month: int, S_t: float, I_t: float) -> float:
        """
        Predict next-period water level S_{t+1} for a given month and state.
        """
        info = self.month_models.get(month, None)

        # If the requested month has no fitted model, return the current level
        # as a safe fallback.
        if info is None:
            return float(S_t)

        mode = info.get("mode", "spline")

        if mode == "identity":
            return float(S_t)

        if mode == "spline":
            spline = info["spline"]
            S_mean, S_std = info["S_mean"], info["S_std"]
            I_mean, I_std = info["I_mean"], info["I_std"]
            S_std = S_std if S_std > 0 else 1.0
            I_std = I_std if I_std > 0 else 1.0

            # Apply the same normalization used during fitting before calling
            # the spline predictor.
            S_norm = (S_t - S_mean) / S_std
            I_norm = (I_t - I_mean) / I_std
            z_pred = spline(S_norm, I_norm)[0][0]
            return float(z_pred)

        if mode == "linear":
            a, b, c = info["coef"]
            z_pred = a + b * S_t + c * I_t
            return float(z_pred)

        return float(S_t)

    def save_policy(self, lam: float, *, teacher_metadata: dict, teacher_path: Path) -> Path:
        """
        Save the fitted ISO-SURF policy as a PKL file.
        """
        lam_str = format_lambda_for_filename(lam)
        policy_path = self.output_dir / f"iso_surf_train_lambda_{lam_str}.pkl"

        # Save only the information needed later for evaluation and prediction.
        payload = {
            "smoothing_factor": self.smoothing_factor,
            "spline_order": self.spline_order,
            "grid_size": self.grid_size,
            "max_factor": self.max_factor,
            "min_samples_spline": self.min_samples_spline,
            "month_models": self.month_models,
            "artifact_metadata": {
                **teacher_metadata,
                "method": "ISO-SURF",
                "policy_model": "monthly_smooth_bivariate_spline_with_linear_fallback",
                "policy_source": str(teacher_path),
                "teacher_hash": sha256_file(teacher_path),
            },
        }

        with policy_path.open("wb") as f:
            pickle.dump(payload, f)

        print(f"ISO-SURF policy saved to: {policy_path}")
        return policy_path

    @classmethod
    def load_policy(cls, policy_path: str | Path, *, penalty_lambda: float) -> "ISOSurfPolicyFitter":
        """
        Load a fitted ISO-SURF policy from a PKL file.
        """
        policy_path = Path(policy_path)
        with policy_path.open("rb") as f:
            payload = pickle.load(f)
        require_contract(
            payload.get("artifact_metadata"), penalty_lambda=penalty_lambda,
            solver_variant=DDP_SOLVER_VARIANT)

        obj = cls(
            smoothing_factor=payload.get("smoothing_factor", 1.0),
            spline_order=tuple(payload.get("spline_order", (3, 3))),
            grid_size=payload.get("grid_size", 25),
            max_factor=payload.get("max_factor", 10000.0),
            min_samples_spline=payload.get("min_samples_spline", 20),
        )
        obj.month_models = payload["month_models"]
        obj.artifact_metadata = payload["artifact_metadata"]
        return obj


def main_iso_surf() -> None:
    """
    Generate upstream DP training data first, then fit and save ISO-SURF policies.
    """

    data_dir = get_data_dir()
    experiment_settings_file = data_dir / "experiment_settings.yaml"
    iso_settings = load_algorithm_settings(experiment_settings_file, "iso_surf")
    require_keys(
        iso_settings,
        [
            "smoothing_factor",
            "spline_order",
            "grid_size",
            "max_factor",
            "min_samples_spline",
        ],
        "iso_surf",
    )

    spline_order = tuple(int(x) for x in iso_settings["spline_order"])
    if len(spline_order) != 2:
        raise ValueError("iso_surf.spline_order must contain exactly two integers.")

    iso_root_dir = get_formal_algorithm_dir("iso_surf")
    preprocessed_dir = iso_root_dir / "teacher"
    policy_dir = iso_root_dir / "policy"
    penalty_settings_file = data_dir / "violation_count_penalty_settings.yaml"
    penalty_lambdas = load_formal_penalty_candidates(data_dir, "iso_surf")

    # Step 1: explicitly run ISO preprocessing to generate the upstream DP
    # training trajectories required for ISO-SURF fitting.
    preprocessor = ISOPreprocessor()
    preprocessor.run_all()

    print("\n============================================================")
    print("ISO-SURF policy fitting started")
    print("============================================================")
    print(f"Penalty settings file: {penalty_settings_file}")
    print(f"Preprocessed directory: {preprocessed_dir}")
    print(f"Policy directory: {policy_dir}")
    print(f"Penalty lambdas: {penalty_lambdas}")
    print("============================================================")

    for lam in penalty_lambdas:
        lam_str = format_lambda_for_filename(lam)
        dp_csv_path = preprocessed_dir / f"iso_upstream_dp_train_lambda_{lam_str}.csv"

        if not dp_csv_path.exists():
            raise FileNotFoundError(
                f"Required upstream DP training file not found: {dp_csv_path}"
            )

        print("\n------------------------------------------------------------")
        print(f"Fitting ISO-SURF policy for lambda = {lam}")
        print("------------------------------------------------------------")

        # Each lambda corresponds to one upstream DP trajectory file. That file
        # is loaded and used to fit a separate ISO-SURF monthly rule set.
        dp_df = pd.read_csv(dp_csv_path)
        teacher_metadata_path = dp_csv_path.with_suffix(".json")
        if not teacher_metadata_path.is_file():
            raise FileNotFoundError("Formal ISO teacher metadata is missing")
        teacher_metadata = json.loads(teacher_metadata_path.read_text(encoding="utf-8"))
        require_contract(teacher_metadata, penalty_lambda=lam,
                         solver_variant=DDP_SOLVER_VARIANT)
        if teacher_metadata.get("teacher_hash") != sha256_file(dp_csv_path):
            raise ValueError("ISO teacher hash does not match its metadata")

        fitter = ISOSurfPolicyFitter(
            smoothing_factor=float(iso_settings["smoothing_factor"]),
            spline_order=spline_order,
            grid_size=int(iso_settings["grid_size"]),
            max_factor=float(iso_settings["max_factor"]),
            min_samples_spline=int(iso_settings["min_samples_spline"]),
        )
        fitter.fit_from_dp_dataframe(dp_df)
        fitter.save_policy(float(lam), teacher_metadata=teacher_metadata,
                           teacher_path=dp_csv_path)


if __name__ == "__main__":
    main_iso_surf()
