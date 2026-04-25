# Predict Calorie Expenditure — Stacked Ridge Ensemble

**CS6140 Machine Learning · Kaggle Playground Series S5E5**

**Authors:** Keith Fernandes · Shreyans Jain (Northeastern University)

---

## Overview

This project builds a regression pipeline to predict calorie expenditure from physiological and exercise features. It trains ten diverse base learners under identical 5-fold cross-validation and combines their out-of-fold predictions with a Ridge-regression meta-learner (stacking).

**Primary metric:** RMSLE (equivalent to RMSE on `log1p(y)`).  
**Secondary metric:** Calorie-scale RMSE, reported for interpretability.

**Best CV score:** Ridge ensemble RMSLE = **0.059161**

---

## Repository Structure

```
.
├── Datasets/
│   ├── train.csv                  # Kaggle competition training set
│   ├── test.csv                   # Kaggle competition test set
│   ├── calories.csv               # External dataset (augments training folds)
│   └── dataset-KF-SJ-CS6140.zip  # Source zip (extract to populate CSVs)
├── Results/
│   ├── figures/                   # Generated plots (PNG)
│   ├── models/                    # Cached model objects (joblib .pkl)
│   ├── oof_preds/                 # OOF and test predictions (.parquet)
│   ├── submissions/               # Kaggle submission CSVs
│   └── tables/                    # CSV summary tables
├── s05e05_calorie_expenditure_prediction_ridge_v2.ipynb  # Main notebook
├── requirements.txt
└── Keith Fernandes & Shreyans Jain - CS6140 - Final Report.pdf
```

---

## Dataset

| File | Description |
|------|-------------|
| `train.csv` | Kaggle S5E5 training split |
| `test.csv`  | Kaggle S5E5 test split (no labels) |
| `calories.csv` | External calorie dataset used to augment each training fold |

**Features:** `Sex`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`

**Engineered features:**
- **BMI** = `Weight / (Height / 100)²`
- **Duration × Heart_Rate** — exercise-intensity proxy

**Preprocessing:**
- `Sex` binary-encoded: `male → 0`, `female → 1`
- Target log-transformed with `log1p`; predictions back-transformed with `expm1`

---

## Pipeline

```
Raw features
     │
     ├── Feature Engineering (BMI, Duration × Heart_Rate)
     │
     ├── 5-Fold Cross-Validation (KFold, shuffle, seed=42)
     │       │
     │       ├── + External calories.csv appended to each training fold
     │       │
     │       └── Train 10 base learners independently per fold
     │
     ├── Collect OOF predictions → meta-feature matrix
     │
     └── Ridge meta-learner (alpha tuned by Optuna, 250 TPE trials)
              │
              └── Final stacked prediction
```

---

## Models

### Base Learners

| # | Model | Library |
|---|-------|---------|
| 1 | HistGradientBoosting | scikit-learn |
| 2 | LightGBM (gbdt) | LightGBM |
| 3 | LightGBM (goss) | LightGBM |
| 4 | XGBoost | XGBoost |
| 5 | CatBoost | CatBoost |
| 6 | Random Forest | scikit-learn |
| 7 | Extra Trees | scikit-learn |
| 8 | Yggdrasil GBT (YDF) | ydf |
| 9 | AutoGluon (medium quality, 600 s/fold) | AutoGluon |
| 10 | Keras ANN (Dense SELU 256→128→64→1, 3-seed avg) | Keras 3 / TensorFlow |

### Meta-Learner

Ridge regression with `alpha` and `tol` tuned by Optuna (TPE sampler, 250 trials, 5-fold CV on OOF meta-features).

---

## Results

### Model Benchmark (5-fold CV, sorted by mean RMSLE)

| Model | RMSLE (mean) | RMSLE (std) | RMSLE (OOF) | RMSE (cal) |
|-------|-------------|------------|------------|-----------|
| **Ridge (ensemble)** | **0.059159** | 0.000384 | **0.059161** | **3.5631** |
| AutoGluon | 0.059403 | 0.000376 | 0.059404 | 3.5846 |
| LightGBM (gbdt) | 0.059754 | 0.000311 | 0.059755 | 3.6061 |
| LightGBM (goss) | 0.059778 | 0.000361 | 0.059779 | 3.5959 |
| CatBoost | 0.059793 | 0.000334 | 0.059794 | 3.6423 |
| HistGB | 0.059890 | 0.000434 | 0.059892 | 3.6293 |
| Yggdrasil | 0.060398 | 0.000336 | 0.060399 | 3.6796 |
| KerasANN | 0.060609 | 0.000624 | 0.060612 | 3.6659 |
| XGBoost | 0.060708 | 0.000480 | 0.060710 | 3.7202 |
| ExtraTrees | 0.061240 | 0.000556 | 0.061243 | 3.7060 |
| RandomForest | 0.061803 | 0.000643 | 0.061806 | 3.7383 |

### Feature Engineering Ablation (LightGBM gbdt)

| Variant | Features | RMSLE (OOF) |
|---------|----------|------------|
| With engineered features (BMI, Duration × HR) | 9 | 0.059755 |
| Without engineered features | 7 | 0.059790 |

Engineered features provide a small but consistent improvement.

---

## Generated Artifacts

All outputs are written to `Results/` and are ready to reference from the final report.

| Path | Contents |
|------|----------|
| `Results/figures/distributions.png` | Feature distribution histograms |
| `Results/figures/corr_heatmaps.png` | Correlation heatmaps |
| `Results/figures/mutual_info_train.png` | Mutual information (train features) |
| `Results/figures/mutual_info_original.png` | Mutual information (original dataset) |
| `Results/figures/model_benchmark.png` | Model comparison bar & box plots |
| `Results/figures/ridge_coeffs.png` | Ridge meta-learner coefficients |
| `Results/figures/ablation_features.png` | Feature engineering ablation bar chart |
| `Results/tables/data_quality.csv` | Missing values and IQR outlier counts |
| `Results/tables/model_benchmark.csv` | Full model benchmark table |
| `Results/tables/fold_scores.csv` | Raw per-fold RMSLE for every model |
| `Results/tables/ablation_features.csv` | Ablation study results |
| `Results/tables/mutual_info_train.csv` | Mutual information scores (train) |
| `Results/tables/mutual_info_original.csv` | Mutual information scores (original) |
| `Results/models/*.pkl` | Cached trained model objects |
| `Results/oof_preds/*.parquet` | OOF and test predictions per model |
| `Results/submissions/sub_ridge_0.059161.csv` | Kaggle submission file |

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
# Heavy optional dependency (AutoGluon, ~500 MB):
pip install autogluon.tabular
# Required base packages (not in requirements.txt):
pip install xgboost lightgbm catboost ydf tensorflow optuna seaborn
```

### 2. Prepare the dataset

Place `dataset-KF-SJ-CS6140.zip` inside `Datasets/`. The notebook will extract `train.csv`, `test.csv`, and `calories.csv` automatically on first run.

Alternatively, download the zip from [Google Drive](https://drive.google.com/file/d/1pfQ8wWWYp1Z75_ZZw7qJA93ZaZXjzS4p/view?usp=sharing).

### 3. Run the notebook

```bash
jupyter notebook s05e05_calorie_expenditure_prediction_ridge_v2.ipynb
```

The notebook runs locally by default and can also be uploaded to Google Colab without modification. Google Drive persistence is supported on Colab via `CFG.use_drive = True`.

### Configuration (`CFG` class)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_folds` | 5 | Number of CV folds |
| `run_optuna` | `True` | Tune Ridge meta-learner with Optuna |
| `n_optuna_trials` | 250 | Optuna trials for Ridge tuning |
| `run_autogluon` | `True` | Train AutoGluon base learner |
| `autogluon_time_limit` | 600 | Seconds per fold for AutoGluon |
| `nn_seeds` | 3 | Seeds averaged per fold for Keras ANN |
| `run_ablation` | `True` | Run feature engineering ablation |
| `force_retrain` | `False` | Bust model caches and retrain all models |
| `use_gpu` | `"auto"` | GPU mode: `"auto"`, `"yes"`, or `"no"` |

### GPU Support

- **CUDA**: accelerates XGBoost and CatBoost (auto-detected via `nvidia-smi`).
- **Apple Silicon MPS**: accelerates Keras/TensorFlow (requires `tensorflow-metal`).
- LightGBM uses CPU by default (GPU wheel must be built separately).

---

## Report

See `Keith Fernandes & Shreyans Jain - CS6140 - Final Report.pdf` for full methodology, analysis, and discussion.
