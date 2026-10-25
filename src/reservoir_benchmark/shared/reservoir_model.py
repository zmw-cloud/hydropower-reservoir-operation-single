"""
Shared reservoir characteristics module for the unified reservoir operation framework.

This module provides the common physical relationships and shared operating
parameters used by all four algorithms (DP, SDP, ISO-SURF, and DQN), including:
1) water level-to-storage conversion,
2) storage-to-water level conversion,
3) tailwater level estimation from discharge,
4) hydropower output calculation, and
5) unified access to reservoir constraints and basic operating parameters.

The purpose of this module is to ensure that all algorithms are evaluated under the
same reservoir environment, physical assumptions, constraints, and parameter
settings, so that their differences arise from the control strategy rather than
from inconsistencies in the underlying model.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd


class SharedReservoirModel:
    """Shared reservoir parameters, constraints, and physical relationships for all algorithms."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = (
            Path(data_dir).resolve()
            if data_dir is not None
            else Path(__file__).resolve().parents[3] / "data"
        )

        # Load reservoir parameters
        self.params = self._load_reservoir_parameters()

        # Expose shared reservoir parameters and operating constraints as attributes
        self.dead_level = float(self._get_param("dead_level"))
        self.normal_level = float(self._get_param("normal_level"))
        self.initial_level = float(self._get_param("initial_level"))
        self.terminal_level = float(
            self._get_param("terminal_level", default=self.initial_level)
        )
        self.flood_limit_level = float(
            self._get_param("flood_limit_level", default=self.normal_level)
        )
        self.flood_limit_action_start_month = int(
            self._get_param("flood_limit_action_start_month", default=5)
        )
        self.flood_limit_action_end_month = int(
            self._get_param("flood_limit_action_end_month", default=10)
        )
        self.max_turbine_discharge = float(self._get_param("max_turbine_discharge"))
        self.max_total_release = float(
            self._get_param("max_total_release", default=float("inf"))
        )
        self.max_power = float(self._get_param("max_power"))
        self.firm_power = float(self._get_param("firm_power"))
        self.power_coefficient = float(self._get_param("power_coefficient"))
        self.inflow_upper_bound = float(
            self._get_param("inflow_upper_bound", default=5000.0)
        )

        # Optional shared temporal parameter
        self.delta_t_days = float(self._get_param("delta_t_days", default=30.4))
        self.delta_t_hours = self.delta_t_days * 24.0
        self.delta_t_seconds = self.delta_t_hours * 3600.0

        # Load the original water level-storage table
        self._levels, self._storages = self._load_level_storage_data()

        # Prepare for interpolation: sort by water level
        level_sort_idx = np.argsort(self._levels)
        self._levels_h = self._levels[level_sort_idx].astype(float)
        self._storages_h = self._storages[level_sort_idx].astype(float)

        # Prepare for interpolation: sort by storage
        storage_sort_idx = np.argsort(self._storages)
        self._storages_s = self._storages[storage_sort_idx].astype(float)
        self._levels_s = self._levels[storage_sort_idx].astype(float)

        # Boundary limits
        self._curve_level_min = float(self._levels_h[0])
        self._curve_level_max = float(self._levels_h[-1])
        self._curve_storage_min = float(self._storages_s[0])
        self._curve_storage_max = float(self._storages_s[-1])

        # Fit the tailwater rating curve
        self.tailwater_curve_coefficients = self._fit_discharge_tailwater_curve()

    def _get_param(self, *keys, default=None):
        for key in keys:
            if key in self.params:
                return self.params[key]
        if default is not None:
            return default
        raise KeyError(f"Missing required parameter. Expected one of: {keys}")

    def _load_reservoir_parameters(self):
        file_path = self.data_dir / "reservoir_settings.txt"

        if not file_path.exists():
            raise FileNotFoundError(f"Cannot find file: {file_path}")

        text = file_path.read_text(encoding="utf-8")
        pattern = r'"([^"]+)"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        matches = re.findall(pattern, text)
        return {key: float(value) for key, value in matches}

    def _load_level_storage_data(self):
        """Load the original water level-storage curve data from the CSV file."""
        file_path = self.data_dir / "level_storage_curve.csv"
        df = pd.read_csv(file_path)
        water_levels = df["water_level_m"].values.astype(float)
        storage = df["storage_1e9_m3"].values.astype(float)
        return water_levels, storage

    def _load_tailwater_discharge_data(self):
        """Load the tailwater level-discharge curve data from the CSV file."""
        file_path = self.data_dir / "tailwater_discharge_curve.csv"
        df = pd.read_csv(file_path)
        tailwater_column = (
            "tailwater_level_m"
            if "tailwater_level_m" in df.columns
            else "tail_water_level_m"
        )
        tailwater_levels = df[tailwater_column].values.astype(float)
        discharge = df["discharge_m3s"].values.astype(float)
        anchors = []
        for index in (1, 2):
            discharge_key = f"high_flow_anchor_{index}_discharge"
            tailwater_key = f"high_flow_anchor_{index}_tailwater"
            if discharge_key in self.params and tailwater_key in self.params:
                anchors.append(
                    (float(self.params[discharge_key]), float(self.params[tailwater_key]))
                )
        if anchors:
            anchors_array = np.asarray(anchors, dtype=float)
            discharge = np.concatenate((discharge, anchors_array[:, 0]))
            tailwater_levels = np.concatenate((tailwater_levels, anchors_array[:, 1]))
        return tailwater_levels, discharge

    def _fit_discharge_tailwater_curve(self):
        """Fit a fourth-order polynomial tailwater rating curve as a function of discharge."""
        tailwater_levels, discharge = self._load_tailwater_discharge_data()
        return np.polyfit(discharge, tailwater_levels, 4)

    def calculate_storage(self, level: float) -> float:
        """Convert water level to storage (1e9 m³) using linear interpolation."""
        level = float(np.clip(float(level), self._curve_level_min, self._curve_level_max))
        return float(np.interp(level, self._levels_h, self._storages_h))

    def calculate_level(self, storage: float) -> float:
        """Convert storage (1e9 m³) to water level using linear interpolation."""
        storage = float(np.clip(float(storage), self._curve_storage_min, self._curve_storage_max))
        return float(np.interp(storage, self._storages_s, self._levels_s))

    def calculate_tailwater(self, discharge: float) -> float:
        """Calculate the tailwater level from the total discharge."""
        poly = np.poly1d(self.tailwater_curve_coefficients)
        return float(poly(float(discharge)))

    def calculate_power(
        self,
        turbine_discharge: float,
        total_discharge: float,
        current_level: float,
        next_level: float,
    ) -> float:
        """Calculate hydropower output from turbine discharge and average hydraulic head."""
        tailwater_level = self.calculate_tailwater(total_discharge)
        average_head = (current_level + next_level) / 2 - tailwater_level
        return self.power_coefficient * turbine_discharge * average_head

    def monthly_max_target_level(self, month: int) -> float:
        """Return the end-of-month target ceiling used by the cascade benchmark."""
        month = int(month)
        if not 1 <= month <= 12:
            raise ValueError("month must be in 1..12")
        if self.flood_limit_action_start_month <= month <= self.flood_limit_action_end_month:
            return self.flood_limit_level
        return self.normal_level

    def calculate_transition(
        self,
        current_level: float,
        requested_target_level: float,
        inflow: float,
        duration_days: float,
        *,
        month: int,
    ) -> dict:
        """Evaluate one physically corrected XiaoWan monthly transition."""
        duration_days = float(duration_days)
        if duration_days <= 0.0:
            raise ValueError("duration_days must be positive")
        inflow = float(inflow)
        if inflow < 0.0:
            raise ValueError("inflow must be nonnegative")

        monthly_max = self.monthly_max_target_level(month)
        requested = float(requested_target_level)
        bounded_target = float(np.clip(requested, self.dead_level, monthly_max))
        current_storage = self.calculate_storage(current_level)
        requested_storage = self.calculate_storage(bounded_target)
        seconds = duration_days * 86400.0
        available_storage = current_storage + inflow * seconds / 1e9
        end_storage = min(requested_storage, available_storage, self.calculate_storage(monthly_max))
        end_storage = max(end_storage, self.calculate_storage(self.dead_level))
        total_discharge = (
            current_storage + inflow * seconds / 1e9 - end_storage
        ) * 1e9 / seconds
        total_discharge = max(0.0, float(total_discharge))

        max_total_release_triggered = total_discharge > self.max_total_release
        unresolved_upper_storage_excess_m3 = 0.0
        if max_total_release_triggered:
            total_discharge = self.max_total_release
            uncropped_end_storage = current_storage + (inflow - total_discharge) * seconds / 1e9
            unresolved_upper_storage_excess_m3 = max(
                0.0,
                uncropped_end_storage - self.calculate_storage(monthly_max),
            ) * 1e9
            end_storage = uncropped_end_storage
            end_storage = float(
                np.clip(
                    end_storage,
                    self.calculate_storage(self.dead_level),
                    self.calculate_storage(monthly_max),
                )
            )

        end_level = self.calculate_level(end_storage)
        tailwater_level = self.calculate_tailwater(total_discharge)
        head = max(0.0, (float(current_level) + end_level) / 2.0 - tailwater_level)
        turbine_discharge = min(total_discharge, self.max_turbine_discharge)
        if head <= 0.0:
            turbine_discharge = 0.0
        else:
            turbine_discharge = min(
                turbine_discharge,
                self.max_power / (self.power_coefficient * head),
            )
        power_kw = min(
            self.max_power,
            self.power_coefficient * turbine_discharge * head,
        )
        spill_discharge = max(0.0, total_discharge - turbine_discharge)
        energy_gwh = power_kw * duration_days * 24.0 / 1e6
        deficit_kw = max(0.0, self.firm_power - power_kw)
        deficit_energy_gwh = deficit_kw * duration_days * 24.0 / 1e6
        target_reached = bool(
            np.isclose(end_level, requested, rtol=0.0, atol=1e-8)
        )
        water_balance_error_m3 = (
            (end_storage - current_storage) * 1e9
            - (inflow - total_discharge) * seconds
        )
        return {
            "requested_target_level": requested,
            "bounded_target_level": bounded_target,
            "monthly_max_target_level": monthly_max,
            "current_storage": current_storage,
            "end_storage": end_storage,
            "end_level": end_level,
            "total_discharge": total_discharge,
            "turbine_discharge": turbine_discharge,
            "spill_discharge": spill_discharge,
            "tailwater_level": tailwater_level,
            "head": head,
            "power_kw": power_kw,
            "energy_gwh": energy_gwh,
            "deficit_energy_gwh": deficit_energy_gwh,
            "target_reached": target_reached,
            "was_corrected": not target_reached,
            "correction_amount_m": end_level - requested,
            "water_balance_error_m3": water_balance_error_m3,
            "max_total_release_triggered": bool(max_total_release_triggered),
            "unresolved_upper_storage_excess_m3": float(unresolved_upper_storage_excess_m3),
        }

    def feasible_target_level_mask(
        self,
        current_level: float,
        inflow: float,
        duration_days: float,
        *,
        month: int,
        target_levels: np.ndarray,
        level_atol: float = 1e-8,
    ) -> np.ndarray:
        """Vectorized reachability mask with the same water balance as transitions."""
        levels = np.asarray(target_levels, dtype=float)
        if levels.ndim != 1 or len(levels) == 0 or np.any(~np.isfinite(levels)):
            raise ValueError("target_levels must be a finite nonempty vector")
        duration_days = float(duration_days)
        if duration_days <= 0.0 or inflow < 0.0:
            raise ValueError("duration_days must be positive and inflow nonnegative")
        monthly_max = self.monthly_max_target_level(month)
        inside = (
            (levels >= self.dead_level - level_atol)
            & (levels <= monthly_max + level_atol)
        )
        bounded = np.clip(levels, self.dead_level, monthly_max)
        target_storage = np.interp(bounded, self._levels_h, self._storages_h)
        current_storage = self.calculate_storage(current_level)
        seconds = duration_days * 86400.0
        available = current_storage + float(inflow) * seconds / 1e9
        end_storage = np.minimum(
            np.minimum(target_storage, available),
            self.calculate_storage(monthly_max),
        )
        end_storage = np.maximum(end_storage, self.calculate_storage(self.dead_level))
        release = (
            current_storage + float(inflow) * seconds / 1e9 - end_storage
        ) * 1e9 / seconds
        capped = release > self.max_total_release
        capped_storage = current_storage + (
            float(inflow) - self.max_total_release
        ) * seconds / 1e9
        capped_storage = np.clip(
            capped_storage,
            self.calculate_storage(self.dead_level),
            self.calculate_storage(monthly_max),
        )
        end_storage = np.where(capped, capped_storage, end_storage)
        end_level = np.interp(end_storage, self._storages_s, self._levels_s)
        return inside & np.isclose(
            end_level,
            levels,
            rtol=0.0,
            atol=float(level_atol),
        )
