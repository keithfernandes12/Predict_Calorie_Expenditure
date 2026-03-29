# Predict Calorie Expenditure
**Playground Series - Season 5, Episode 5**  
**Competition URL:** https://www.kaggle.com/competitions/playground-series-s5e5

---

## Overview

Welcome to the 2025 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Goal:** Predict how many calories were burned during a workout.

---

## Evaluation

The evaluation metric for this competition is **Root Mean Squared Logarithmic Error (RMSLE)**.

$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(\ln(\hat{y}_i + 1) - \ln(y_i + 1)\right)^2}$$

Where:
- $n$ = total number of observations in the test set
- $\hat{y}_i$ = predicted value for instance $i$
- $y_i$ = actual value for instance $i$
- $\ln$ = natural logarithm

---

## About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science.

- Competition duration generally only lasts a few weeks (may vary)
- Uses fairly light-weight datasets that are **synthetically generated from real-world data**
- Provides an opportunity to quickly iterate through model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows Kaggle to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows hosting competitions with more interesting datasets. The goal is to produce datasets that have far fewer artifacts than in earlier iterations of the series.

---

## Dataset Description

The dataset (both train and test) was generated from a deep learning model trained on the **Calories Burnt Prediction** dataset. Feature distributions are close to, but not exactly the same as, the original. The original dataset may be freely used as part of this competition — both to explore differences and to see whether incorporating it in training improves model performance.

### Files

| File | Description |
|---|---|
| `train.csv` | Training dataset; `Calories` is the continuous target |
| `test.csv` | Test dataset; predict `Calories` for each row |
| `sample_submission.csv` | Sample submission file in the correct format |

---

## Citation

Walter Reade and Elizabeth Park. *Predict Calorie Expenditure*. https://kaggle.com/competitions/playground-series-s5e5, 2025. Kaggle.
