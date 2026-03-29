# Calorie Expenditure Prediction: Notebook Overview
### Kaggle Playground Series S5E5 | Ridge Ensemble

---

## Problem Statement

Predict the number of **calories burned** during exercise based on physiological and activity features. The evaluation metric is **RMSE (Root Mean Squared Error)**, and the approach is a multi-model stacking ensemble with Ridge Regression as the meta-learner.

---

## Dataset

| Source | Description |
|---|---|
| `train.csv` | Kaggle Playground S5E5 training data |
| `test.csv` | Kaggle test set for submission |
| `calories.csv` | External original dataset used to augment training |

**Target variable:** `Calories`

**Key preprocessing:**
- `Sex` column is binary-encoded: `male -> 0`, `female -> 1`
- Target is **log-transformed** using `np.log1p(y)` before training to reduce skew
- Predictions are back-transformed using `np.expm1()` at submission time

---

## Exploratory Data Analysis

Two quick analyses are run on both the Kaggle dataset and the original dataset:

**1. Mutual Information Scores**
Features are ranked by their mutual information with the target. This gives a non-linear measure of how predictive each feature is, shown as a color-coded bar chart.

**2. Correlation Heatmaps**
Lower-triangle correlation heatmaps are plotted side-by-side for both datasets to visualize feature relationships and identify multicollinearity.

---

## Configuration (CFG)

```
n_folds    = 5          # 5-fold cross-validation
seed       = 42         # Random seed for reproducibility
metric     = RMSE       # Evaluation metric
cv         = KFold      # Shuffled K-Fold split
run_optuna = True       # Tune Ridge hyperparameters with Optuna
```

---

## Training: Base Models

All base models are trained using **5-fold cross-validation** via a shared `Trainer` utility class (from `koolbox`). The original external dataset is appended as extra training data in each fold.

For each model, the trainer outputs:
- **Fold scores** (per-fold RMSE)
- **OOF predictions** (out-of-fold predictions on the training set)
- **Test predictions** (averaged across folds)

### Models Trained

| Model | Library | Notes |
|---|---|---|
| **HistGradientBoosting** | scikit-learn | Tuned hyperparameters (learning rate, depth, regularization) |
| **LightGBM (gbdt)** | lightgbm | Standard gradient boosting, early stopping on RMSE |
| **LightGBM (goss)** | lightgbm | Gradient-one-side-sampling variant, early stopping |
| **XGBoost** | xgboost | Early stopping enabled |
| **CatBoost** | catboost | Early stopping, `use_best_model=True` |
| **AutoGluon** | automl | Pre-trained OOF/test preds loaded from a separate Kaggle dataset input |
| **Yggdrasil (YDF)** | ydf (Google) | Gradient Boosted Trees wrapped in a custom sklearn-compatible class |
| **Keras Neural Network** | keras + scikeras | 3-layer dense network (256 to 128 to 64 to 1), SELU activations, AdamW optimizer, bagged 3x via `BaggingRegressor` |

### Neural Network Architecture

```
Input -> Dense(256, SELU) -> Dense(128, SELU) -> Dense(64, SELU) -> Dense(1)
```

- Optimizer: AdamW (lr=0.001)
- Loss: Mean Squared Error
- Callbacks: EarlyStopping (patience=7), ReduceLROnPlateau
- Bagged 3 times to reduce variance

---

## Ensembling with Ridge Regression

### Why Stacking?

Instead of a simple average of all base models, a **Ridge regression meta-learner** is trained on top of the base model predictions. This learns optimal, data-driven weights for each model.

### How It Works

1. **Build the meta-feature matrix**: Stack all OOF predictions into a DataFrame where each column is one base model's predictions.
2. **Train Ridge on OOF predictions**: Ridge learns how to best combine the models to minimize RMSE on the training set.
3. **Predict on test set**: Use each base model's test predictions as input to the trained Ridge model.
4. **Back-transform**: Apply `np.expm1()` to convert log-scale predictions back to actual calorie values.

### Why Ridge (not plain linear regression)?

Base model predictions are **highly correlated** with each other since they all predict the same target. Ridge's L2 regularization stabilizes the coefficients by penalizing large weights, preventing any single model from dominating due to multicollinearity.

### Ridge Hyperparameter Tuning (Optuna)

```
alpha   -> sampled from [0, 10]       (regularization strength)
tol     -> sampled from [1e-6, 1e-2]  (convergence tolerance)
sampler = TPESampler (multivariate)
```

### Why OOF Predictions (Not Full Training Predictions)?

Using full training predictions would cause **data leakage**. Ridge would learn to trust whichever base model memorized the training data best, not the one that generalizes best. OOF predictions ensure each prediction was made on data the base model never saw during training.

---

## Ridge Coefficient Visualization

After training, the average Ridge coefficients across all 5 folds are computed and plotted as a horizontal bar chart. This shows the **relative contribution** of each base model to the final ensemble prediction.

---

## Submission

```python
sub["Calories"] = np.expm1(ridge_trainer.predict(X_test))
sub.to_csv(f"sub_ridge_{mean_cv_score:.6f}.csv", index=False)
```

The submission filename includes the ensemble's mean CV RMSE for easy tracking.

---

## Results

A combined results view is generated showing:
- **Boxplot** of per-fold RMSE scores for each model (ordered by mean score)
- **Comparison** of individual model performance vs. the Ridge ensemble

Models are sorted ascending by mean RMSE (lower is better).

---

## Key Design Choices

| Choice | Reason |
|---|---|
| Log-transform target | Reduces skew in calorie values, improves model fit |
| Augment with original dataset | More training data, reduces overfitting |
| OOF-based stacking | Prevents data leakage in the meta-learner |
| Ridge as meta-learner | Handles multicollinearity between base model predictions |
| Optuna for Ridge tuning | Even simple meta-learners benefit from proper hyperparameter search |
| Bagging the neural net | Reduces variance from random weight initialization |
