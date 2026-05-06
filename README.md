# Gender Bias in Medical Outcomes (Silent Disparities)

> End-to-end PySpark ML pipeline identifying gender bias in medical outcomes using the MIMIC-IV clinical dataset

## Overview

This project builds scalable PySpark ML pipelines in Databricks to process large clinical datasets, evaluate classification model performance (Logistic Regression, Random Forest, GBT), and surface statistically significant gender-based bias patterns with corrective recommendations — evaluated with AUC, AUPR, and fairness metrics.

**Key Result:** Bias evaluation identified a **9-point recall gap** between male and female cohorts; post-correction accuracy improved to **91%**.

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| PySpark | Distributed data processing and ML pipelines |
| Databricks | Cloud ML platform and notebook environment |
| MIMIC-IV | Clinical dataset (requires credentialed access) |
| Logistic Regression / Random Forest / GBT | Classification models |
| AUC / AUPR | Model evaluation metrics |

## Repository Structure

```
gender-bias-medical-outcomes/
├── README.md
├── docs/
│   ├── methodology.md
│   └── data_access.md
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_fairness_analysis.ipynb
├── src/
│   ├── pipeline.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── fairness_metrics.py
│   └── utils.py
├── config/
│   └── config.yaml
├── requirements.txt
└── .env.example
```

## Environment Setup

### Prerequisites
- Python 3.9+
- Databricks account (Community Edition works)
- MIMIC-IV access ([PhysioNet credentials required](https://physionet.org/content/mimiciv/))

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/JNKAnalyst/gender-bias-medical-outcomes.git
   cd gender-bias-medical-outcomes
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Mac/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Databricks token and data paths
   ```

### Databricks Setup

1. Create a new Databricks cluster (Runtime 13.x ML recommended)
2. Import notebooks from the `notebooks/` folder into your Databricks workspace
3. Attach the cluster to each notebook
4. Upload the MIMIC-IV data to DBFS or mount an external storage path
5. Update `config/config.yaml` with your data path

## Configuration

### `config/config.yaml`
```yaml
data:
  mimic_path: "/dbfs/mnt/mimic-iv/"
  output_path: "/dbfs/mnt/results/"

model:
  test_size: 0.2
  random_state: 42
  cv_folds: 5

fairness:
  protected_attribute: "gender"
  positive_label: 1
  threshold: 0.5
```

### `.env.example`
```
DATABRICKS_HOST=https://<your-workspace>.azuredatabricks.net
DATABRICKS_TOKEN=<your-token>
DATA_PATH=/dbfs/mnt/mimic-iv/
```

## Running the Pipeline

```bash
# Run full pipeline locally (requires PySpark)
python -m src.pipeline --config config/config.yaml

# Or run notebooks sequentially in Databricks:
# 01 → 02 → 03 → 04
```

## Requirements

```
pyspark==3.5.0
pandas==2.1.0
numpy==1.26.0
scikit-learn==1.3.0
matplotlib==3.8.0
seaborn==0.13.0
pyyaml==6.0
python-dotenv==1.0.0
```

## Data Access

This project uses the **MIMIC-IV** clinical dataset. Access requires:
1. Complete CITI training
2. Sign the data use agreement on [PhysioNet](https://physionet.org/content/mimiciv/)
3. Download and place data in the path specified in `config.yaml`

See [`docs/data_access.md`](docs/data_access.md) for a step-by-step guide and
[`docs/methodology.md`](docs/methodology.md) for cohort definition, evaluation
metrics, and bias-mitigation strategies.

## Author

**Joash**  
[GitHub](https://github.com/JNKAnalyst) | [Portfolio](https://jnkanalyst.github.io/portfolio/)
