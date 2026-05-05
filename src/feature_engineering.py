"""Feature engineering for the MIMIC-IV mortality cohort.

Produces a Spark ML pipeline that one-hot encodes categorical variables,
buckets continuous ones, and assembles the final feature vector. Gender is
deliberately kept out of the model features so it can be used as the
protected attribute during fairness evaluation.
"""

from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Bucketizer,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import DataFrame


CATEGORICAL_COLS: list[str] = []
CONTINUOUS_COLS: list[str] = ["age", "los_hours"]
AGE_SPLITS = [18.0, 30.0, 45.0, 60.0, 75.0, 95.0]


def build_feature_pipeline(categorical: list[str] | None = None) -> Pipeline:
    categorical = categorical or CATEGORICAL_COLS
    stages: list = []

    indexed_cols: list[str] = []
    encoded_cols: list[str] = []
    for col in categorical:
        idx = f"{col}_idx"
        enc = f"{col}_enc"
        stages.append(StringIndexer(inputCol=col, outputCol=idx, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCol=idx, outputCol=enc))
        indexed_cols.append(idx)
        encoded_cols.append(enc)

    stages.append(
        Bucketizer(
            splits=AGE_SPLITS,
            inputCol="age",
            outputCol="age_bucket",
            handleInvalid="keep",
        )
    )

    assembled_inputs = encoded_cols + ["age_bucket", "los_hours"]
    stages.append(
        VectorAssembler(
            inputCols=assembled_inputs,
            outputCol="features",
            handleInvalid="skip",
        )
    )
    return Pipeline(stages=stages)


def transform(df: DataFrame, pipeline: Pipeline) -> DataFrame:
    model = pipeline.fit(df)
    return model.transform(df)
