# Obtaining MIMIC-IV

This project uses the **MIMIC-IV** clinical dataset, which is freely
available for research but requires credentialed access. The repository does
**not** ship any patient data.

## Steps

1. Create an account on [PhysioNet](https://physionet.org/).
2. Complete the required CITI "Data or Specimens Only Research" training and
   upload your completion report to your PhysioNet profile.
3. Sign the data use agreement on the MIMIC-IV project page:
   <https://physionet.org/content/mimiciv/>.
4. Once approval is granted (typically within a few days), download the
   release as either CSV or Parquet.

## Layout expected by the pipeline

By default the pipeline reads the following files from `data.mimic_path` in
`config/config.yaml`:

```
<mimic_path>/
├── patients.csv     # subject_id, gender, anchor_age, ...
└── admissions.csv   # subject_id, hadm_id, admittime, dischtime, hospital_expire_flag, ...
```

Parquet files work too — `data_preprocessing.load_table` switches on the
extension automatically.

## Databricks

For Databricks, upload the files to DBFS (or mount cloud storage) and set
`mimic_path` to the corresponding `/dbfs/...` path. Avoid using personal
Workspace storage for protected health data; use a properly-governed mount
point with access controls.
