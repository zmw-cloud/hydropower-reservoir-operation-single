"""One-source reduction of the V11 monthly equal-width inflow model."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from reservoir_benchmark.shared.io_utils import mean_calendar_days_by_month


N_INFLOW_CLASSES = 10


@dataclass
class MonthlyInflowModel:
    edges: np.ndarray          # [12, 11]
    probabilities: np.ndarray  # [12, 10]
    representatives: np.ndarray
    counts: np.ndarray
    mean_days: np.ndarray
    metadata: dict

    @classmethod
    def fit(cls, inflows, years):
        values = np.asarray(inflows, dtype=float)
        years = np.asarray(years, dtype=int)
        if values.shape != (len(years), 12) or len(years) == 0:
            raise ValueError("SDP inflows must have [years, 12] shape")
        edges = np.empty((12, 11), dtype=float)
        reps = np.empty((12, N_INFLOW_CLASSES), dtype=float)
        counts = np.zeros((12, N_INFLOW_CLASSES), dtype=np.int64)
        for month in range(12):
            sample = values[:, month]
            bounds = np.linspace(sample.min(), sample.max(), N_INFLOW_CLASSES + 1)
            edges[month] = bounds
            reps[month] = bounds[:-1] + np.diff(bounds) / 2.0
            indices = (np.zeros(len(sample), dtype=int) if bounds[0] == bounds[-1]
                       else np.clip(np.searchsorted(bounds, sample, side="right") - 1, 0, N_INFLOW_CLASSES - 1))
            counts[month] = np.bincount(indices, minlength=N_INFLOW_CLASSES)
        digest = hashlib.sha256()
        digest.update(years.astype(np.int64).tobytes())
        digest.update(values.astype(np.float64).tobytes())
        model = cls(edges, counts / float(len(years)), reps, counts,
                    mean_calendar_days_by_month(years), {
                        "schema_version": 1,
                        "class_numbering": "zero_based_0_to_9",
                        "classification": "10_equal_width_intervals_from_training_month_min_max",
                        "representative_inflow": "interval_midpoint",
                        "probability_definition": "class_count_divided_by_training_years",
                        "training_years": len(years),
                        "training_content_sha256": digest.hexdigest(),
                    })
        model.validate()
        return model

    def validate(self):
        for array, shape in ((self.edges, (12, 11)), (self.probabilities, (12, 10)),
                             (self.representatives, (12, 10)), (self.counts, (12, 10)),
                             (self.mean_days, (12,))):
            if np.asarray(array).shape != shape or np.any(~np.isfinite(array)):
                raise ValueError(f"Invalid SDP probability array {shape}")
        if not np.allclose(self.probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-14):
            raise ValueError("Monthly SDP probabilities must sum to one")

    def classify(self, month: int, inflow: float) -> int:
        index = int(month) - 1
        if not 0 <= index < 12 or not np.isfinite(inflow) or inflow < 0:
            raise ValueError("Invalid month/inflow for SDP classification")
        bounds = self.edges[index]
        return (0 if bounds[0] == bounds[-1] else
                int(np.clip(np.searchsorted(bounds, inflow, side="right") - 1, 0, 9)))

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, edges=self.edges, probabilities=self.probabilities,
                            representatives=self.representatives, counts=self.counts,
                            mean_days=self.mean_days,
                            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)))
        return target

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as saved:
            model = cls(saved["edges"], saved["probabilities"], saved["representatives"],
                        saved["counts"], saved["mean_days"],
                        json.loads(str(saved["metadata_json"].item())))
        model.validate()
        return model
