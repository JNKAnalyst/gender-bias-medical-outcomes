"""End-to-end driver for the Silent Disparities pipeline.

Loads MIMIC-IV tables, builds the modeling cohort, fits LR/RF/GBT, evaluates
AUC and AUPR, and reports per-group fairness metrics. Designed to run on
Databricks; locally requires PySpark installed and the MIMIC-IV files at the
path given in `config.yaml`.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from pyspark.ml.evaluation import BinaryClassificationEvaluator

from . import data_preprocessing as dp
from . import feature_engineering as fe
from .fairness_metrics import fairness_gaps, per_group_metrics
from .models import build_model
from .utils import get_spark, load_config


def evaluate_binary(predictions, metric: str) -> float:
    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName=metric,
    )
    return float(evaluator.evaluate(predictions))


def run(config_path: str) -> Dict[str, Any]:
    cfg = load_config(config_path)
    spark = get_spark()

    base = cfg["data"]["mimic_path"]
    patients = dp.load_table(spark, base, "patients.csv")
    admissions = dp.load_table(spark, base, "admissions.csv")
    cohort = dp.build_cohort(patients, admissions)

    pipeline = fe.build_feature_pipeline()
    featurized = fe.transform(cohort, pipeline)

    train, test = dp.split_train_test(
        featurized,
        test_fraction=cfg["model"]["test_size"],
        seed=cfg["model"]["random_state"],
    )

    results: Dict[str, Any] = {}
    for name in ("logistic_regression", "random_forest", "gbt"):
        estimator = build_model(name, cfg.get("hyperparameters", {}).get(name))
        model = estimator.fit(train)
        preds = model.transform(test)

        auc = evaluate_binary(preds, "areaUnderROC")
        aupr = evaluate_binary(preds, "areaUnderPR")
        per_group = per_group_metrics(preds, cfg["fairness"]["protected_attribute"])
        gaps = fairness_gaps(per_group)

        results[name] = {
            "auc": auc,
            "aupr": aupr,
            "per_group": {k: v.__dict__ for k, v in per_group.items()},
            "fairness_gaps": gaps,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Silent Disparities pipeline.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--out", default=None, help="Optional path to write JSON results")
    args = parser.parse_args()

    results = run(args.config)
    rendered = json.dumps(results, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(rendered)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
