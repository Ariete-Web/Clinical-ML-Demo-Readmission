# Clinical ML Demo — Hospital Readmission

## Overview

This repository contains a fully reproducible, lightweight demonstration of building, evaluating, calibrating, and auditing a machine‑learning baseline model for **thirty‑day hospital readmission** using tabular data.  The goal is to provide a transparent baseline that is easy to run locally or via GitHub Actions, with clear documentation of methods, saved artifacts and figures, and optional automation.  A synthetic dataset is included to avoid special permissions while still mimicking realistic hospital data structures.

## Scientific goals

- **Build transparent baselines** for instructional and screening purposes.
- Emphasize **calibration** and **fairness slices** in addition to discrimination metrics.
- Ensure **reproducibility** with pinned dependencies, fixed random seeds, saved artifacts, and a clean project layout.

## Methods in plain language

- **Prediction target:** A binary label indicating whether a patient was readmitted within thirty days (1 = yes, 0 = no).
- **Features:** Standard covariates such as age, sex, number of chronic conditions, length of stay, prior admissions and derived variables like age bands and simple comorbidity counts.
- **Data split:** The data is split into 70 % training, 15 % validation, and 15 % test using stratified sampling.  A fixed `RANDOM_STATE` ensures reproducibility.
- **Models:**  
  - **Logistic Regression** with balanced class weights as a simple, transparent baseline.  
  - **Histogram‑based Gradient Boosting** (HGB) as a stronger non‑linear baseline.  
  The model with higher validation AUROC becomes the "winner" and is used for further evaluation.
- **Calibration:** Probabilities from the winning model are calibrated using isotonic regression (though the code falls back to using the uncalibrated model if you avoid additional dependencies).  Calibration is trained on the combined training and validation sets and evaluated on the test set.
- **Metrics:**  
  - **AUROC** (Area under the Receiver Operating Characteristic).  
  - **AUPRC** (Area under the Precision‑Recall Curve).  
  - **Brier score** for probability accuracy.  
  A simple threshold analysis can be added if you need specific positive predictive value targets.
- **Fairness slices:** The test set is evaluated across subgroups such as sex and age band.  For each subgroup the script reports AUROC, AUPRC, and Brier score; you can compute simple disparities such as delta‑AUROC (difference between best and worst subgroup).
- **Quality assurance:**  
  - The dataset is fresh‑cloned and loaded from `data/synthetic_readmission.csv` for reproducibility.  
  - Random seeds are controlled everywhere.  
  - Sanity checks ensure class prevalence is similar across splits and guard against leakage.  
  - All key metrics and plots are saved for later review.

## Repository layout

```
clinical_ml_demo/
├─ data/
│  └─ synthetic_readmission.csv      # synthetic dataset, no permissions required
├─ figures/
│  ├─ roc.png                        # receiver operating characteristic curve (test)
│  ├─ pr.png                         # precision‑recall curve (test)
│  └─ calibration.png                # reliability plot (test)
├─ artifacts/
│  ├─ metrics.json                   # train/valid/test metrics and notes
│  ├─ fairness_subgroups.csv         # subgroup table (size and metrics per group)
│  └─ model.joblib                   # serialized winning model
├─ train_baseline.py                 # end‑to‑end training, plotting, and artifact generation
├─ model_card.md                     # concise documentation of intent, data, training, metrics, and risks
├─ README.md                         # this file
├─ requirements.txt                  # minimal pinned dependencies
├─ .github/workflows/
│  ├─ build‑train.yml              # optional: run training and commit artifacts via GitHub Actions
│  └─ lint.yml                       # optional: style and import checks
└─ LICENSE                           # MIT
```

## Running locally

You can execute the full pipeline on your machine without special permissions:

```bash
# create and activate a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate
# install dependencies
pip install -r requirements.txt
# run the baseline training script
python train_baseline.py
```

This will generate three figures in the `figures/` directory (`roc.png`, `pr.png`, `calibration.png`), produce evaluation metrics and the calibrated model in the `artifacts/` directory (`metrics.json`, `fairness_subgroups.csv`, `model.joblib`), and print summary metrics to the console.

## Running on GitHub

A GitHub Actions workflow (`build‑train.yml`) is provided for one‑click reproduction.  To run it:

1. Navigate to the **Actions** tab and select **Build clinical‑ml demo**.  
2. Choose **Run workflow** and confirm.  
3. After the run completes (green checkmark), the workflow will commit updated figures and artifacts back to the repository (if it has write permissions) or upload them as workflow artifacts.  You can then view the images and download the metrics without local setup.

## Outputs and deliverables

- **Figures** (stored in `figures/`):  
  1. `roc.png` – Receiver Operating Characteristic curve on the test set.  Expect AUROC values in the range roughly 0.65‑0.85 on readmission‑like data; synthetic data will fall within this band.  
  2. `pr.png` – Precision‑Recall curve on the test set.  Expect AUPRC to depend on prevalence; after calibration the curve should rise above the horizontal prevalence line.  
  3. `calibration.png` – Reliability diagram (10 bins) on the test set.  The curve should track the diagonal if the model is well calibrated.
- **Artifacts** (stored in `artifacts/`):  
  - `metrics.json` – JSON object containing validation and test metrics for each model and notes about the selected winner.  
  - `fairness_subgroups.csv` – Table of subgroup metrics (sex and age band) with counts, AUROC, AUPRC, and Brier for each subgroup.  
  - `model.joblib` – Serialized winning model pipeline for reuse or demonstration.
- **Documentation**:  
  - `model_card.md` – Short model card summarizing intent, data, training process, metrics, limitations, and governance.  
  - This `README.md` for quickstart and methods summary.

## Reproducibility and validation

- **Seeds and determinism**: All random processes (data split, model training, simulation) use fixed seeds, so repeated runs yield identical results given the same code and data.
- **Sanity checks**: The code reports class prevalence and ensures there is no obvious leakage from test into training.  If you alter the dataset, re‑run these checks.
- **Fairness metrics**: The subgroup table includes sample sizes and returns `NaN` for discrimination metrics if a subgroup has only one class present.  Large differences between subgroups may indicate data imbalance.

## Change log

See `CHANGELOG.md` for version history.  Notable changes include adding the synthetic data generator, fairness metrics, calibration plots, and GitHub Actions workflows.

## License

This project is licensed under the [MIT License](LICENSE).  You are free to use, modify, and distribute the code under the terms of that license.
