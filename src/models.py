"""Classifiers used for the gender-bias study: LR, RF, GBT.

Each builder returns a Spark ML estimator configured with sensible defaults;
hyperparameters are exposed via the config file rather than hard-coded.
"""

from __future__ import annotations

from typing import Any, Dict

from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)


def logistic_regression(params: Dict[str, Any] | None = None) -> LogisticRegression:
    p = params or {}
    return LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=p.get("max_iter", 100),
        regParam=p.get("reg_param", 0.0),
        elasticNetParam=p.get("elastic_net_param", 0.0),
    )


def random_forest(params: Dict[str, Any] | None = None) -> RandomForestClassifier:
    p = params or {}
    return RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=p.get("num_trees", 100),
        maxDepth=p.get("max_depth", 8),
        seed=p.get("seed", 42),
    )


def gbt(params: Dict[str, Any] | None = None) -> GBTClassifier:
    p = params or {}
    return GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=p.get("max_iter", 50),
        maxDepth=p.get("max_depth", 5),
        seed=p.get("seed", 42),
    )


MODEL_REGISTRY = {
    "logistic_regression": logistic_regression,
    "random_forest": random_forest,
    "gbt": gbt,
}


def build_model(name: str, params: Dict[str, Any] | None = None):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](params)
