"""Group-wise fairness metrics for binary classifiers.

Computes per-group recall, precision, FPR/FNR, and demographic parity / equal
opportunity gaps between male and female cohorts. Inputs are Spark DataFrames
with `prediction`, `label`, and the protected-attribute column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pyspark.sql import DataFrame, functions as F


@dataclass
class GroupMetrics:
    group: str
    n: int
    positive_rate: float
    tpr: float
    fpr: float
    precision: float


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def per_group_metrics(df: DataFrame, protected_col: str) -> Dict[str, GroupMetrics]:
    rows = (
        df.groupBy(protected_col)
        .agg(
            F.count("*").alias("n"),
            F.sum(F.col("label")).alias("pos"),
            F.sum(((F.col("prediction") == 1) & (F.col("label") == 1)).cast("int")).alias("tp"),
            F.sum(((F.col("prediction") == 1) & (F.col("label") == 0)).cast("int")).alias("fp"),
            F.sum(((F.col("prediction") == 0) & (F.col("label") == 1)).cast("int")).alias("fn"),
            F.sum(((F.col("prediction") == 0) & (F.col("label") == 0)).cast("int")).alias("tn"),
            F.avg(F.col("prediction").cast("double")).alias("pred_pos_rate"),
        )
        .collect()
    )

    out: Dict[str, GroupMetrics] = {}
    for r in rows:
        group = r[protected_col]
        out[group] = GroupMetrics(
            group=group,
            n=int(r["n"]),
            positive_rate=float(r["pred_pos_rate"]),
            tpr=_safe_div(r["tp"], r["tp"] + r["fn"]),
            fpr=_safe_div(r["fp"], r["fp"] + r["tn"]),
            precision=_safe_div(r["tp"], r["tp"] + r["fp"]),
        )
    return out


def fairness_gaps(metrics: Dict[str, GroupMetrics]) -> Dict[str, float]:
    """Compute |male - female| gaps for the headline fairness metrics."""
    if "M" not in metrics or "F" not in metrics:
        raise ValueError("Expected groups 'M' and 'F' to be present.")
    m, f = metrics["M"], metrics["F"]
    return {
        "demographic_parity_gap": abs(m.positive_rate - f.positive_rate),
        "equal_opportunity_gap": abs(m.tpr - f.tpr),
        "fpr_gap": abs(m.fpr - f.fpr),
        "precision_gap": abs(m.precision - f.precision),
    }
