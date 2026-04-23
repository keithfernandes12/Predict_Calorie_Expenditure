# Notebook Analysis: `s05e05_calorie_expenditure_prediction_ridge_v2`

## What the notebook does

End-to-end Kaggle S5E5 pipeline for predicting calorie expenditure. Writes all artifacts (figures, tables, OOF preds, submissions, model caches) to `Results/`. Flow:

1. **Data**: 750 000 train rows, 250 000 test rows, plus 15 000 rows from the external `calories.csv` appended to each training fold.
2. **Preprocessing**: `Sex → {male:0, female:1}`; target log-transformed (`log1p`) so RMSE on logs = RMSLE.
3. **Feature engineering** (2 added → 9 total): `BMI = Weight / (Height/100)²`, `Duration_x_HR = Duration × Heart_Rate`.
4. **Base models** (all run under identical 5-fold `KFold(shuffle=True, seed=42)`): HistGB, LightGBM gbdt, LightGBM goss, XGBoost, CatBoost, RandomForest, ExtraTrees, Yggdrasil (YDF), AutoGluon (`medium_quality`, 600 s/fold), Keras SELU stack 256→128→64→1 bagged ×3 seeds.
5. **Meta-learner**: Ridge on OOF predictions, tuned with Optuna (250 trials, TPE multivariate) → `alpha ≈ 9.999`, `tol ≈ 8.6e-3`.
6. **Submission** written to `sub_ridge_0.059161.csv`.

## Data quality signal

From `Results/tables/data_quality.csv` (cell 12): no missing values in any column. The only notable outlier mass is `Body_Temp` with **14 919 IQR outliers** (~2 %) — interesting but benign for tree models. `Height` and `Weight` have a handful each.

## Model benchmark (5-fold CV)

From the benchmark table output of the notebook:

| Model | RMSLE_mean | RMSLE_std | RMSE (cal) |
|---|---|---|---|
| **Ridge (ensemble)** | **0.059159** | 0.000384 | **3.5631** |
| AutoGluon | 0.059403 | 0.000376 | 3.5846 |
| LightGBM (gbdt) | 0.059754 | 0.000311 | 3.6061 |
| LightGBM (goss) | 0.059778 | 0.000361 | 3.5959 |
| CatBoost | 0.059793 | 0.000334 | 3.6423 |
| HistGB | 0.059890 | 0.000434 | 3.6293 |
| Yggdrasil | 0.060398 | 0.000336 | 3.6796 |
| KerasANN | 0.060609 | 0.000624 | 3.6659 |
| XGBoost | 0.060708 | 0.000480 | 3.7202 |
| ExtraTrees | 0.061240 | 0.000556 | 3.7060 |
| RandomForest | 0.061803 | 0.000643 | 3.7383 |

### Observations

- **Ensemble wins**, but narrowly. Ridge lifts ~0.00024 RMSLE over the best single model (AutoGluon) — small, but stacking behaves as advertised: the meta-learner beats every base learner, not just the average.
- **Tier 1 (boosting ensembles)**: AutoGluon + the four GBMs + HistGB all cluster within a ~0.0005 band (0.0594–0.0599). They dominate the leaderboard.
- **Tier 2 (trees/NN)**: Yggdrasil, KerasANN, XGBoost at ~0.0604–0.0607. XGBoost underperforming LightGBM/CatBoost is the one slightly surprising result — likely because its `n_estimators=50 000` with `min_child_weight=96` and `max_depth=11` is tuned differently than the LightGBM pair and early-stops sooner.
- **Tier 3 (bagged trees)**: RandomForest and ExtraTrees trail by ~0.002 RMSLE — consistent with bagging being weaker than boosting on a smooth regression target.
- **Spread is tiny**: 0.0021 RMSLE separates best from worst (~0.18 calories RMSE). The problem is well-modeled by anything reasonable.
- **Std across folds is ~0.0004** for nearly every model → scores are stable; the ordering isn't noise.

## Ridge meta-learner

- Optuna converged to `alpha ≈ 9.999` (saturating the upper bound of `[0, 10]`). That's a signal the search space should probably be widened — a strong ceiling bias suggests Ridge wants even more regularization, consistent with the heavy collinearity of OOF predictions (every base model is targeting the same variable).
- **Action item for the report**: rerun Optuna with `alpha ∈ [0, 100]` or similar to confirm the optimum isn't above 10.
- Ridge coefficients were saved to `Results/figures/ridge_coeffs.png` but text weights weren't printed in the notebook — worth surfacing in the report.

## Ablation: engineered features

From the ablation cell (LightGBM gbdt, identical CV):

| Variant | Features | RMSLE_oof | RMSE (cal) |
|---|---|---|---|
| With BMI + Duration_x_HR | 9 | 0.059755 | 3.6061 |
| Without | 7 | 0.059790 | 3.6126 |
| **Δ** | | **−0.000035** | **−0.0065** |

The engineered features help, but **barely** — ~0.06 % relative RMSLE improvement, well within fold-std (0.0003). Tree ensembles can reconstruct BMI and Duration×HR from the raw columns on their own; the features mostly help convergence speed, not final accuracy. Honest framing for the report: "engineered features give a marginal, not-statistically-significant lift under 5-fold CV — their main value is interpretability, not performance."

## Submission

Final: `sub_ridge_0.059161.csv` (CV RMSLE embedded in filename). Predictions are `expm1`-back-transformed before submission.

## Suggested follow-ups

1. **Widen Ridge `alpha` search** — the optimum is pinned at the upper bound.
2. **Report Ridge coefficients as a table** in the write-up — the bar plot exists but per-model weights aren't in any CSV.
3. **Consider dropping the bottom tier** (RandomForest/ExtraTrees) from the stack and re-tuning Ridge — high-bias bases can add noise once Ridge has enough diverse signal.
4. **Ablation caveat**: the current ablation only runs on LightGBM. A more defensible claim would re-run it on a second model (Keras, where manual features matter more to NNs).
