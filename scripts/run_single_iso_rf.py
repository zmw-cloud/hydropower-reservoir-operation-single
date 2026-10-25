"""Run one explicit ISO-RF stage; development never opens independent evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reservoir_benchmark.policy_generation.iso_rf.workflow import (
    Context, preflight, run_development, run_lambda_scan, select_lambda,
    run_final_training, run_evaluation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "development", "lambda-scan",
                                          "select-lambda", "final-train", "evaluation"))
    parser.add_argument("--config", type=Path, default=ROOT / "data" / "iso_rf_settings.yaml")
    parser.add_argument("--output-root", type=Path, help="Isolated output root for this whole workflow")
    parser.add_argument("--target", type=float, help="Guarantee target in percent, only for select-lambda")
    args = parser.parse_args()
    if args.target is not None and args.stage != "select-lambda":
        parser.error("--target is only used by select-lambda")
    ctx = Context.build(args.config, args.output_root)
    if args.stage == "select-lambda":
        select_lambda(ctx, target=None if args.target is None else args.target / 100)
    else:
        {"preflight": preflight, "development": run_development,
         "lambda-scan": run_lambda_scan, "final-train": run_final_training,
         "evaluation": run_evaluation}[args.stage](ctx)


if __name__ == "__main__":
    main()
