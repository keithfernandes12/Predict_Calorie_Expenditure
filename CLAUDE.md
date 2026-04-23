# Predict Calorie Expenditure — Project Context

## Overview
Kaggle **Playground Series S5E5 — Predict Calorie Expenditure** (regression). Doubles as the course project for **CS6140 Machine Learning** (Northeastern, Spring 2026). Teammates: Shreyans Jain + Keith Fernandes.

> Note: the parent folder is named `CS6400` but the course is **CS6140**. Use CS6140 in any course-facing writing.

## Competition
- **Goal:** Predict calories burned during a workout.
- **Metric:** RMSLE (Root Mean Squared Logarithmic Error).
- **Strategy:** Notebook optimizes RMSE on `log1p(target)`, which is equivalent to RMSLE.

## Repo Layout

| Path | Purpose |
|---|---|
| `Datasets/train.csv` | Kaggle training data (~750k rows). Target: `Calories`. |
| `Datasets/test.csv` | Kaggle test set. |
| `Datasets/calories.csv` | Original "Calories Burnt Prediction" dataset, used to augment training per Kaggle rules. |
| `Datasets/sample_submission.csv` | Submission format reference. |
| `Documents/` | CS6140 project proposal (docx / pptx / pdf). |
| `Test_Notebooks/s05e05-calorie-expenditure-prediction-ridge.ipynb` | Main modeling notebook. |
| `Test_Notebooks/s05e05-calorie-expenditure-prediction-overview.md` | Written overview of the notebook's approach. |
| `Kaggle S5 E5 - Predict Calorie Expenditure.md` | Competition brief. |

## Features
`Sex, Age, Height, Weight, Duration, Heart_Rate, Body_Temp` → target `Calories`.

- `Sex` is binary-encoded (`male → 0`, `female → 1`).
- Target is log-transformed via `np.log1p(y)` for training; predictions are back-transformed with `np.expm1()` at submission.

## Modeling Approach
Multi-model **stacking ensemble** with **5-fold CV**, external dataset appended in each fold.

**Base models:**
- HistGradientBoosting (sklearn)
- LightGBM (gbdt)
- LightGBM (goss)
- XGBoost
- CatBoost
- AutoGluon (OOF/test preds loaded from external source)
- Yggdrasil / YDF (Google), sklearn-wrapped
- Keras NN: `Dense(256) → Dense(128) → Dense(64) → Dense(1)`, SELU activations, AdamW, bagged 3×

**Meta-learner:** Ridge regression on OOF predictions, hyperparameters tuned with Optuna (TPESampler, multivariate).

**Why Ridge:** base predictions are highly collinear (all target the same variable); L2 regularization stabilizes the weights.

**Why OOF:** avoids leakage — meta-learner sees only predictions made on unseen folds.

## Environments
- **Shreyans:** macOS.
- **Keith:** Windows (`C:/Users/keith/...`). The Windows paths in `.claude/settings.local.json` are intentional — leave them alone.
