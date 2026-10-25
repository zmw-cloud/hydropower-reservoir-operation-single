"""DDP-only batch physics and bounded threaded transition caching.

No interpolation of continuation values is introduced. Invalid actions are
represented by NaN in a float64 power cache; all remaining values are physical
monthly power, independent of the penalty coefficient.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def worker_count() -> int:
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    workers = int(os.environ.get("DDP_NUM_WORKERS", min(4, available)))
    if workers < 1:
        raise ValueError("DDP_NUM_WORKERS must be positive")
    return workers


def cache_signature(reservoir, inflows, days, levels) -> str:
    digest = hashlib.sha256()
    digest.update(repr(sorted(reservoir.params.items())).encode())
    for array in (inflows, days, levels, reservoir._levels_h,
                  reservoir._storages_h, reservoir._storages_s,
                  reservoir._levels_s, reservoir.tailwater_curve_coefficients):
        digest.update(np.asarray(array, dtype=np.float64).tobytes())
    for month in range(1, 13):
        digest.update(repr(reservoir.monthly_max_target_level(month)).encode())
    return digest.hexdigest()


def month_power_matrix(reservoir, levels, storages, inflow, days, month):
    """Broadcast the scalar calculate_transition equations without float32."""
    ceiling = reservoir.monthly_max_target_level(month)
    start = storages[:, None]
    target = np.clip(levels, reservoir.dead_level, ceiling)
    target_storage = np.interp(target, reservoir._levels_h, reservoir._storages_h)
    seconds = float(days) * 86400.0
    available = start + float(inflow) * seconds / 1e9
    end_storage = np.minimum(np.minimum(target_storage[None, :], available),
                             reservoir.calculate_storage(ceiling))
    end_storage = np.maximum(end_storage, reservoir.calculate_storage(reservoir.dead_level))
    release = np.maximum(0.0, (start + float(inflow) * seconds / 1e9 - end_storage) * 1e9 / seconds)
    capped = release > reservoir.max_total_release
    release = np.minimum(release, reservoir.max_total_release)
    capped_storage = np.clip(start + (float(inflow) - release) * seconds / 1e9,
                             reservoir.calculate_storage(reservoir.dead_level),
                             reservoir.calculate_storage(ceiling))
    end_storage = np.where(capped, capped_storage, end_storage)
    end_level = np.interp(np.clip(end_storage, reservoir._curve_storage_min,
                                 reservoir._curve_storage_max),
                          reservoir._storages_s, reservoir._levels_s)
    feasible = (levels[None, :] <= ceiling + 1e-8) & np.isclose(
        end_level, levels[None, :], rtol=0.0, atol=1e-8)
    tailwater = np.polyval(reservoir.tailwater_curve_coefficients, release)
    head = np.maximum(0.0, (levels[:, None] + end_level) / 2.0 - tailwater)
    turbine = np.minimum(release, reservoir.max_turbine_discharge)
    capacity_flow = np.divide(reservoir.max_power, reservoir.power_coefficient * head,
                              out=np.full_like(head, np.inf), where=head > 0.0)
    turbine = np.where(head <= 0.0, 0.0, np.minimum(turbine, capacity_flow))
    power = np.minimum(reservoir.max_power, reservoir.power_coefficient * turbine * head)
    if not np.all(np.isfinite(power)):
        raise ValueError("DDP batch physics produced non-finite power")
    return np.where(feasible, power, np.nan)


class TransitionCache:
    """Temporary, disk-backed power table reusable across lambdas on one solver.

    Workers write disjoint monthly slices. The directory is removed on close;
    it is a computation cache, not a checkpoint or a reusable experiment result.
    """
    def __init__(self, reservoir, inflows, days, levels):
        self.signature = cache_signature(reservoir, inflows, days, levels)
        self._directory = TemporaryDirectory(prefix="single_ddp_", dir=os.environ.get("DDP_CACHE_DIR"))
        self.power = None
        self.workers = worker_count()
        try:
            self.power = np.lib.format.open_memmap(
                Path(self._directory.name) / "power.npy", mode="w+", dtype=np.float64,
                shape=(len(inflows), len(levels), len(levels)))
            storages = np.asarray([reservoir.calculate_storage(z) for z in levels])
            def fill_block(start):
                for t in range(start, min(start + 32, len(inflows))):
                    self.power[t] = month_power_matrix(
                        reservoir, levels, storages, inflows[t], days[t], t % 12 + 1)

            starts = range(0, len(inflows), 32)
            print(f"DDP transition cache: {self.workers} workers, "
                  f"{self.power.nbytes / 1024**2:.1f} MiB disk-backed float64", flush=True)
            if self.workers == 1:
                for start in starts:
                    fill_block(start)
            else:
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    for _ in pool.map(fill_block, starts):
                        pass
            self.power.flush()
            print("DDP transition cache ready", flush=True)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.power is not None:
            self.power._mmap.close()
            self.power = None
        self._directory.cleanup()

    def __del__(self):
        if hasattr(self, "_directory"):
            self.close()
