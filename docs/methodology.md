# Methodology

This document explains how the Silent Disparities pipeline measures gender
bias in mortality-risk prediction on MIMIC-IV.

## Cohort definition

- Source tables: `patients`, `admissions`.
- Inclusion: adults aged 18-95 with a recorded gender of `M` or `F`.
- Outcome: `hospital_expire_flag` (in-hospital mortality).
- Features used at training time: bucketed age, length of stay (hours).
- The protected attribute (`gender`) is **excluded** from model features and
  used only for fairness evaluation.

## Models

Three Spark ML classifiers are trained on the same train/test split:

1. Logistic Regression (baseline, interpretable coefficients).
2. Random Forest (non-linear, captures interactions).
3. Gradient-Boosted Trees (typically the strongest single model on tabular
   clinical data).

## Evaluation

Discriminative performance:

- AUC (area under ROC).
- AUPR (area under precision-recall) — preferred under class imbalance.

Fairness:

- Per-group TPR / FPR / precision (M vs F).
- Demographic-parity gap: |P(Ŷ=1|M) − P(Ŷ=1|F)|.
- Equal-opportunity gap: |TPR_M − TPR_F|.
- FPR gap and precision gap.

## Bias mitigation

When a meaningful equal-opportunity gap appears, the project explores three
post-hoc / in-processing strategies and re-evaluates against the same
dashboard:

1. **Threshold tuning per group** — different decision thresholds for M vs F
   to equalize TPR.
2. **Class-weighted training** — penalize errors on the under-served group
   more heavily.
3. **Reweighing** — adjust the joint distribution of (label, group) before
   training.

## Reproducibility

- Random seeds are pinned in `config/config.yaml`.
- Splits are produced with `randomSplit` using the configured seed.
- Cluster runtime: Databricks Runtime 13.x ML (PySpark 3.5).
