"""Loaders and cleaning steps for the MIMIC-IV cohort used by the pipeline.

The functions here intentionally take Spark DataFrames and Spark paths so they
can run unchanged on a Databricks cluster against the credentialed MIMIC-IV
release. Schema names follow the public MIMIC-IV documentation.
"""

from __future__ import annotations

from typing import Iterable

from pyspark.sql import DataFrame, SparkSession, functions as F


REQUIRED_PATIENT_COLS: tuple[str, ...] = ("subject_id", "gender", "anchor_age")
REQUIRED_ADMISSION_COLS: tuple[str, ...] = (
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "hospital_expire_flag",
)


def load_table(spark: SparkSession, base_path: str, name: str) -> DataFrame:
    """Read a MIMIC-IV CSV/Parquet table from the configured base path."""
    path = f"{base_path.rstrip('/')}/{name}"
    if path.endswith(".parquet") or "parquet" in name:
        return spark.read.parquet(path)
    return spark.read.option("header", True).csv(path)


def assert_columns(df: DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_cohort(patients: DataFrame, admissions: DataFrame) -> DataFrame:
    """Join patients to admissions and produce the modeling cohort."""
    assert_columns(patients, REQUIRED_PATIENT_COLS)
    assert_columns(admissions, REQUIRED_ADMISSION_COLS)

    joined = (
        admissions.alias("a")
        .join(patients.alias("p"), on="subject_id", how="inner")
        .select(
            F.col("subject_id"),
            F.col("hadm_id"),
            F.col("p.gender").alias("gender"),
            F.col("p.anchor_age").cast("int").alias("age"),
            F.col("a.admittime").cast("timestamp").alias("admittime"),
            F.col("a.dischtime").cast("timestamp").alias("dischtime"),
            F.col("a.hospital_expire_flag").cast("int").alias("label"),
        )
        .where(F.col("gender").isin("M", "F"))
        .where(F.col("age").between(18, 95))
    )

    return joined.withColumn(
        "los_hours",
        (F.col("dischtime").cast("long") - F.col("admittime").cast("long")) / 3600.0,
    )


def split_train_test(df: DataFrame, test_fraction: float, seed: int) -> tuple[DataFrame, DataFrame]:
    train, test = df.randomSplit([1.0 - test_fraction, test_fraction], seed=seed)
    return train, test
