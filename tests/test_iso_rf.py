"""Small synthetic regression tests; no formal hydrological experiment is run."""
from __future__ import annotations

import contextlib
import copy
from io import StringIO
from pathlib import Path
import sys
import tempfile
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml

from reservoir_benchmark.policy_generation.iso_rf.policy import MonthlySingleRFPolicy
from reservoir_benchmark.policy_generation.iso_rf.workflow import (
    Context, TeacherBuilder, split_training, simulate_policy, rank_hyperparameters, rank_lambdas,
)
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel


def small_teacher():
    rows = []
    for year_index in range(6):
        for month in range(1, 13):
            rows.append({"Calendar_Month": month,
                         "Start_Water_Level_m": 1200 + year_index,
                         "Inflow_m3s": 500 + year_index * 30,
                         "Target_Water_Level_m": 1190 + month + year_index})
    return pd.DataFrame(rows)


class IsoRFTests(unittest.TestCase):
    def test_split_does_not_expose_validation_to_teacher_input(self):
        years = np.arange(2009, 2409)
        raw = pd.DataFrame(np.ones((400, 12)))
        fit, fit_years, validation, validation_years = split_training(raw, years)
        raw.iloc[320:] = 999999
        changed_fit, changed_years, _, _ = split_training(raw, years)
        np.testing.assert_array_equal(fit.to_numpy(), changed_fit.to_numpy())
        np.testing.assert_array_equal(fit_years, changed_years)
        self.assertEqual((len(fit), len(validation)), (320, 80))
        self.assertEqual((fit_years[-1], validation_years[0]), (2328, 2329))
        with self.assertRaises(ValueError):
            split_training(raw.iloc[:399], years[:399])

    def test_monthly_features_and_targets_are_same_row(self):
        seen = []
        class FakeForest:
            def __init__(self, **kwargs):
                pass
            def fit(self, x, y):
                seen.append((x.copy(), y.copy()))
        teacher = small_teacher()
        with patch("reservoir_benchmark.policy_generation.iso_rf.policy.RandomForestRegressor", FakeForest):
            MonthlySingleRFPolicy.fit(
                teacher, hyperparameters={"n_estimators": 2, "max_depth": 2,
                                          "min_samples_leaf": 1, "max_features": 1.0},
                random_seed=7, fit_n_jobs=1,
            )
        for month, (x, y) in enumerate(seen, 1):
            subset = teacher.loc[teacher["Calendar_Month"] == month]
            np.testing.assert_array_equal(x[:, 0], subset["Start_Water_Level_m"])
            np.testing.assert_array_equal(x[:, 1], subset["Inflow_m3s"])
            np.testing.assert_array_equal(y, subset["Target_Water_Level_m"])

    def test_forest_reproducibility_and_final_role_guard(self):
        kwargs = dict(hyperparameters={"n_estimators": 3, "max_depth": 2,
                                      "min_samples_leaf": 1, "max_features": 0.5},
                      random_seed=7, fit_n_jobs=1, metadata={"artifact_role": "development_policy"})
        first = MonthlySingleRFPolicy.fit(small_teacher(), **kwargs)
        second = MonthlySingleRFPolicy.fit(small_teacher(), **kwargs)
        self.assertEqual(first.predict(3, 1202, 560)[0], second.predict(3, 1202, 560)[0])
        _, outside = first.predict(3, 1240, 10000)
        self.assertTrue(outside.all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pkl"
            first.save(path)
            with self.assertRaises(ValueError):
                MonthlySingleRFPolicy.load(path, required_role="final_policy")

    def test_ddp_builder_receives_only_320_years_and_reuses_verified_teacher(self):
        years = np.arange(2009, 2409)
        raw = pd.DataFrame(np.ones((400, 12)))
        raw.iloc[320:] = 99999
        fit, fit_years, validation, validation_years = split_training(raw, years)
        reservoir = SimpleNamespace(initial_level=1240, terminal_level=1240)
        calls = []
        class FakeDDP:
            def __init__(self, reservoir, inflows, month_days):
                calls.append(inflows.copy())
                self.inflows, self.days = inflows, month_days
            def optimize(self, firm_power_penalty_lambda):
                n = self.inflows.size
                result = [None] * 13
                result[2] = np.full(n, 1240.0)
                result[4] = np.full(n, 1000.0)
                result[5] = np.ones(n)
                result[11] = self.days.ravel()
                return tuple(result)
            def clear_transition_cache(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            ctx = SimpleNamespace(output=Path(directory), contract={"test": "synthetic"},
                                  reservoir=reservoir, train=raw, train_years=years,
                                  fit=fit, fit_years=fit_years)
            with patch("reservoir_benchmark.policy_generation.iso_rf.workflow.ReservoirOperation", FakeDDP), contextlib.redirect_stdout(StringIO()):
                builder = TeacherBuilder(ctx)
                teacher, meta = builder.get(1200)
                repeated, _ = builder.get(1200)
                builder.close()
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].shape, (320, 12))
            self.assertTrue((calls[0] == 1).all())
            self.assertEqual(len(teacher), 3840)
            self.assertEqual(meta["teacher_last_year"], 2328)
            self.assertFalse(meta["validation_future_used_by_ddp"])
            self.assertEqual(len(repeated), 3840)

    def test_continuous_physical_feedback_and_energy_identity(self):
        reservoir = SharedReservoirModel(data_dir=ROOT / "data")
        class Policy:
            def __init__(self):
                self.inputs = []
            def predict(self, month, level, inflow):
                self.inputs.append(level)
                return (1300.0 if 5 <= month <= 10 else 1220.0), np.array([False, False])
        policy = Policy()
        frame = pd.DataFrame(np.full((2, 12), 1000.0))
        schedule, summary = simulate_policy(policy, frame, np.array([2009, 2010]), reservoir, 1200)
        np.testing.assert_allclose(policy.inputs[1:], schedule["Water_Level_m"].iloc[:-1])
        self.assertEqual(policy.inputs[12], schedule["Water_Level_m"].iloc[11])
        self.assertNotEqual(policy.inputs[12], 1240.0)
        # Level/storage inverse interpolation may differ at float64 round-off.
        self.assertLess(summary["max_abs_storage_jump_m3"], 1e-4)
        self.assertEqual(summary["flood_limit_exceedance_count"], 0)
        np.testing.assert_allclose(schedule["Electricity_GWh"], schedule["Power_kW"] * schedule["Duration_Days"] * 24 / 1e6)
        expected = schedule["Electricity_GWh"].sum() - 1200 * schedule["Firm_Power_Violation"].sum()
        self.assertAlmostEqual(expected, summary["total_penalized_objective_gwh"])
        self.assertEqual(summary["months"], 24)

    def test_hyperparameter_and_lambda_selection_use_different_metrics(self):
        frame = pd.DataFrame([
            {"candidate_id": "a", "lambda": 500, "mean_annual_penalized_objective_gwh": 19000,
             "mean_annual_generation_gwh": 21000, "guarantee_rate": .60},
            {"candidate_id": "b", "lambda": 650, "mean_annual_penalized_objective_gwh": 17000,
             "mean_annual_generation_gwh": 20000, "guarantee_rate": .80},
        ])
        self.assertEqual(rank_hyperparameters(frame).iloc[0]["candidate_id"], "a")
        self.assertEqual(rank_lambdas(frame, .80).iloc[0]["candidate_id"], "b")

    def test_preflight_does_not_open_independent_evaluation(self):
        from reservoir_benchmark.policy_generation.iso_rf import workflow
        original = workflow.load_monthly_inflows_with_calendar
        opened = []
        def guard(path):
            opened.append(Path(path).name)
            if "evaluation" in Path(path).name:
                raise AssertionError("Independent evaluation opened during development preflight")
            return original(path)
        with patch.object(workflow, "load_monthly_inflows_with_calendar", guard):
            ctx = Context.build(ROOT / "data/iso_rf_settings.yaml")
        self.assertEqual(opened, ["SynMonthlyInflow_train.xlsx"])
        self.assertEqual(len(ctx.fit), 320)

    def test_staged_workflow_with_synthetic_inputs_and_mock_ddp(self):
        """Exercise artifacts end to end; mock DDP and use two-tree toy forests."""
        from reservoir_benchmark.policy_generation.iso_rf import workflow

        class FakeDDP:
            def __init__(self, reservoir, inflows, month_days):
                self.reservoir, self.inflows, self.days = reservoir, inflows, month_days
            def optimize(self, firm_power_penalty_lambda):
                n = self.inflows.size
                levels = np.full(n, 1220.0)
                levels[-1] = self.reservoir.terminal_level
                result = [None] * 13
                result[2] = levels
                result[4] = np.full(n, 2000000.0)
                result[5] = result[4] * self.days.ravel() * 24 / 1e6
                result[11] = self.days.ravel()
                return tuple(result)
            def clear_transition_cache(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for name in ("reservoir_settings.txt", "level_storage_curve.csv", "tailwater_discharge_curve.csv"):
                shutil.copyfile(ROOT / "data" / name, data / name)
            years = np.arange(2009, 2409)
            columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            train = pd.DataFrame(np.full((400, 12), 1000.0), columns=columns)
            train.insert(0, "Year", years)
            evaluation = train.copy()
            evaluation.loc[:, columns] = 1100.0
            train.to_excel(data / "SynMonthlyInflow_train.xlsx", index=False)
            evaluation.to_excel(data / "SynMonthlyInflow_evaluation.xlsx", index=False)
            config = yaml.safe_load((ROOT / "data/iso_rf_settings.yaml").read_text(encoding="utf-8"))
            config["protocol"]["lambda_candidates"] = [600, 1200]
            config["random_forest"]["candidates"] = copy.deepcopy(config["random_forest"]["candidates"][:2])
            for candidate in config["random_forest"]["candidates"]:
                candidate["n_estimators"] = 2
            config["random_forest"]["fit_n_jobs"] = 1
            config["random_forest"]["predict_n_jobs"] = 1
            config_path = root / "settings.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            original_loader = workflow.load_monthly_inflows_with_calendar
            opened = []
            def training_only(path):
                opened.append(Path(path).name)
                if "evaluation" in Path(path).name:
                    raise AssertionError("Independent data opened before evaluation stage")
                return original_loader(path)

            with patch.object(workflow, "get_data_dir", return_value=data), \
                 patch.object(workflow, "ReservoirOperation", FakeDDP), \
                 patch.object(workflow, "load_monthly_inflows_with_calendar", training_only), \
                 contextlib.redirect_stdout(StringIO()):
                ctx = Context.build(config_path, root / "results")
                workflow.run_development(ctx)
                workflow.run_development(ctx)  # complete candidates are reused
                workflow.run_lambda_scan(ctx)
                workflow.select_lambda(ctx)
                workflow.run_final_training(ctx)
            self.assertEqual(opened, ["SynMonthlyInflow_train.xlsx"])
            selected = workflow.read_json(ctx.output / "selection/selected_lambda.json")
            tag = str(int(selected["lambda"]))
            meta = workflow.read_json(ctx.output / "final" / f"lambda_{tag}" / "iso_rf_final_policy.json")
            self.assertEqual(meta["fit_years"], 400)
            self.assertEqual(meta["artifact_role"], "final_policy")
            with contextlib.redirect_stdout(StringIO()):
                workflow.run_evaluation(ctx)
            result = workflow.read_json(ctx.output / "evaluation" / f"iso_rf_eval_lambda_{tag}.json")
            self.assertEqual(result["months"], 4800)
            self.assertFalse(result["evaluation_used_for_selection"])
            self.assertEqual(result["data_split"], "evaluation")
            final_path = ctx.output / "final" / f"lambda_{tag}" / "iso_rf_final_policy.pkl"
            policy = MonthlySingleRFPolicy.load(final_path)
            policy.metadata["artifact_role"] = "development_policy"
            policy.save(final_path)
            with self.assertRaises(ValueError):
                workflow.run_evaluation(ctx)


if __name__ == "__main__":
    unittest.main()
