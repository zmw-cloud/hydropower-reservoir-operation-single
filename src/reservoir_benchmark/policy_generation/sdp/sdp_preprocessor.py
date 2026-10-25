from __future__ import annotations

"""
SDP inflow discretization preprocessor for the unified reservoir operation
framework.

This module prepares the discretized inflow interval file required by SDP
training, including:
1) loading the monthly synthetic training inflow series,
2) extracting monthly inflow samples,
3) discretizing each month's inflow into equal-probability classes, and
4) exporting the standardized interval table for subsequent SDP optimization.

The purpose of this module is to integrate inflow discretization into the SDP
workflow so that SDP training can be launched directly from the runner without
requiring a separate preprocessing script.
"""

import numpy as np
import pandas as pd

from reservoir_benchmark.shared.io_utils import load_monthly_inflows_from_excel
from reservoir_benchmark.shared.project_paths import (
    get_data_dir,
    get_preprocessed_dir,
)


class SDPInflowDiscretizer:
    """
    Prepare the discretized inflow interval file required by SDP training.
    """

    def __init__(self, n_inflow_classes: int = 20):
        self.data_dir = get_data_dir()
        self.output_dir = get_preprocessed_dir("sdp")
        self.inflow_file = self.data_dir / "SynMonthlyInflow_train.xlsx"

        self.n_inflow_classes = int(n_inflow_classes)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cached interval table
        self._intervals_df: pd.DataFrame | None = None

    def _extract_monthly_samples(self) -> dict[int, np.ndarray]:
        """
        Load the synthetic training inflow series and extract monthly samples.

        Returns:
            A dictionary:
                {1: January samples, 2: February samples, ..., 12: December samples}
        """
        inflow_df = load_monthly_inflows_from_excel(self.inflow_file)

        if inflow_df.shape[1] != 12:
            raise ValueError(
                f"Expected 12 monthly columns in {self.inflow_file}, got {inflow_df.shape[1]}."
            )

        monthly_data: dict[int, np.ndarray] = {}
        for month in range(1, 13):
            monthly_data[month] = inflow_df.iloc[:, month - 1].dropna().to_numpy(dtype=float)

        return monthly_data

    def _get_intervals_df(self) -> pd.DataFrame:
        """
        Return the cached SDP inflow interval table.
        Load it from disk only once if it is not already cached.
        """
        if self._intervals_df is None:
            intervals_file = self.output_dir / "sdp_inflow_intervals_detailed.csv"
            if not intervals_file.exists():
                raise FileNotFoundError(
                    f"SDP inflow interval file not found: {intervals_file}"
                )
            self._intervals_df = pd.read_csv(intervals_file)

        return self._intervals_df

    def generate(self) -> Path:
        """
        Generate the standardized SDP inflow interval file required by SDP training.

        Output:
            sdp_inflow_intervals_detailed.csv

        Returns:
            intervals_file
        """
        monthly_data = self._extract_monthly_samples()

        intervals_detailed = []

        for month in range(1, 13):
            data = np.sort(monthly_data[month])

            for inflow_class in range(1, self.n_inflow_classes + 1):
                lower_quantile = (inflow_class - 1) / self.n_inflow_classes
                upper_quantile = inflow_class / self.n_inflow_classes

                lower_bound = np.percentile(data, lower_quantile * 100)
                upper_bound = np.percentile(data, upper_quantile * 100)

                intervals_detailed.append(
                    {
                        "Month": month,
                        "Inflow_Class": inflow_class,
                        "Lower_Bound_m3s": round(lower_bound, 2),
                        "Upper_Bound_m3s": round(upper_bound, 2),
                        "Probability": round(1 / self.n_inflow_classes, 4),
                    }
                )

        df_intervals = pd.DataFrame(intervals_detailed)
        intervals_file = self.output_dir / "sdp_inflow_intervals_detailed.csv"
        df_intervals.to_csv(intervals_file, index=False, encoding="utf-8-sig")

        # Update cache after regeneration
        self._intervals_df = df_intervals.copy()

        print("SDP inflow discretization completed.")
        print(f"Input inflow file: {self.inflow_file}")
        print(f"Output directory: {self.output_dir}")
        print(f"Generated: {intervals_file}")

        return intervals_file

    def classify_inflow(self, month: int, inflow_value: float) -> int:
        """
        Classify a given inflow value into one of the discretized inflow classes.

        Args:
            month: Month index (1-12)
            inflow_value: Inflow value to classify

        Returns:
            Inflow class index (1-20 by default)
        """
        df = self._get_intervals_df()
        month_data = df[df["Month"] == month]

        if month_data.empty:
            raise ValueError(f"No inflow interval data found for month {month}.")

        for _, row in month_data.iterrows():
            if row["Lower_Bound_m3s"] <= inflow_value <= row["Upper_Bound_m3s"]:
                return int(row["Inflow_Class"])

        if inflow_value < month_data["Lower_Bound_m3s"].min():
            return 1

        return self.n_inflow_classes