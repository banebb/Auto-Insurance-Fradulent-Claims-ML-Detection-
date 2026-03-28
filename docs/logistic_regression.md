# Logistic Regression - Documentation

## Overview

This document describes the logistic regression classifier built for fraud detection on the
auto insurance claims dataset. It covers the full pipeline: data preparation, model architecture,
experiment runs with progressive improvements, and results analysis.

All experiments are logged in `notebooks/logistic_regression_experiments.ipynb`, where each run
occupies its own cells for reproducibility and comparison.

---

## Project Structure & What Was Implemented

### Source Modules (`src/`)

| Module | Purpose | Why it exists |
|--------|---------|---------------|
| `src/data/load_data.py` | Loads raw and processed CSV data | Separates I/O from logic; both the notebook and scripts can reuse the same loading functions |
| `src/data/preprocess.py` | Label encoding, standard scaling, train/val/test splitting | Encapsulates all preprocessing so the same transformations are applied consistently everywhere |
| `src/models/model.py` | Creates LogisticRegression with configurable parameters | Single place to change model configuration; avoids duplicating constructor params across files |
| `src/models/train.py` | `train_model()` and `cross_validate()` functions | Separates training logic from model definition; CV is a standalone utility |
| `src/models/predict.py` | `predict()` and `predict_proba()` wrappers | Thin wrappers that provide a consistent interface; `predict_proba` extracts only the positive class probability |
| `src/evaluation/metrics.py` | `evaluate_model()` returns dict of all metrics; `print_metrics()` for display | Centralizes metric computation so every evaluation uses the same set of metrics |
| `src/utils/helpers.py` | Config loading (YAML) and directory creation | Shared utilities used by both scripts and notebooks |
| `src/config/config.yaml` | All hyperparameters, paths, split ratios | Single source of truth for configuration; change params without editing code |

### Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | End-to-end training pipeline: load data, preprocess, train, cross-validate, evaluate, save model |
| `scripts/evaluate.py` | Load a saved model and evaluate on the test set |

### Notebook (`notebooks/`)

| Notebook | Purpose |
|----------|---------|
| `notebooks/logistic_regression_experiments.ipynb` | Experiment log — each run is a separate section with its own cells, markdown explaining the reasoning, and results |

---

## Data Preparation Pipeline

### Input
`data/processed/selected_features.csv` — 1000 samples, 20 features selected by MRMR + 1 target column (`fraud_reported`).

### Step 1: Target Encoding
`fraud_reported` is converted from `Y/N` strings to `1/0` integers. Binary encoding is required by sklearn's logistic regression.

### Step 2: Categorical Feature Encoding
12 categorical features (e.g., `incident_severity`, `collision_type`, `insured_hobbies`) are label-encoded — each unique category gets an integer ID.

**Why label encoding and not one-hot?** With 12 categorical features, one-hot encoding would expand the feature space significantly (some features like `incident_city` have many categories). For a dataset of only 1000 samples, this creates a sparse high-dimensional space that logistic regression struggles with. Label encoding keeps dimensionality low. The trade-off is that it introduces an artificial ordering, but logistic regression can still learn useful coefficients.

### Step 3: Train / Validation / Test Split
- **70% training** (700 samples) — model learns from this
- **15% validation** (150 samples) — used for hyperparameter tuning (C, threshold)
- **15% test** (150 samples) — held out completely until final evaluation
- All splits are **stratified** to maintain the ~75/25 non-fraud/fraud ratio

**Why three sets instead of two?** If we tune hyperparameters on the test set, we're indirectly fitting to it and the reported metrics become optimistically biased. The validation set absorbs this bias, keeping the test set as a truly unseen evaluation.

### Step 4: Feature Scaling
StandardScaler normalizes each feature to zero mean and unit variance. The scaler is **fit on training data only**, then applied to validation and test sets.

**Why fit on train only?** If we fit the scaler on the full dataset, information about the validation/test distribution leaks into the training pipeline. This is "data leakage" — it makes metrics look better during development but the model performs worse on truly new data.

---

## Why Logistic Regression?

Logistic regression was chosen as the first classifier for several reasons:

1. **Interpretability** — coefficients directly show which features push toward fraud vs non-fraud, and by how much. This is valuable for a fraud detection system where decisions need to be explainable.

2. **Baseline** — it's a simple, well-understood model. If it performs well, complex models may not be needed. If it doesn't, we know the performance floor.

3. **Probabilistic output** — it naturally outputs calibrated probabilities, which allows threshold tuning (important for fraud detection where we can trade precision for recall).

4. **Fast training** — on 700 samples with 20 features, it trains in milliseconds, enabling rapid experimentation.

### How Logistic Regression Works

Logistic regression models the probability of the positive class (fraud) as:

```
P(fraud | X) = sigmoid(w0 + w1*x1 + w2*x2 + ... + w20*x20)
```

Where:
- `sigmoid(z) = 1 / (1 + e^(-z))` maps any real number to [0, 1]
- `w0` is the intercept (bias)
- `w1...w20` are coefficients learned during training
- `x1...x20` are the feature values

The model learns coefficients by maximizing the log-likelihood (or equivalently, minimizing the log-loss / cross-entropy) using the L-BFGS optimizer.

**Regularization:** An L2 penalty is added to prevent overfitting:
```
Loss = cross_entropy + (1/C) * sum(w_i^2)
```
Smaller C = stronger regularization = smaller coefficients = simpler model.

---

## Experiment Runs

### Run 1: Baseline (C=1.0, balanced weights)

**Configuration:** Default regularization (C=1.0), balanced class weights, L-BFGS solver.

**Why balanced weights?** The dataset is ~75% non-fraud. Without balanced weights, the model can achieve 75% accuracy by predicting everything as non-fraud — useless for catching fraud. `class_weight='balanced'` automatically sets class weights inversely proportional to frequency:
- Non-fraud weight: 1000 / (2 * 753) = 0.66
- Fraud weight: 1000 / (2 * 247) = 2.02

This makes a fraud misclassification ~3x more costly, forcing the model to pay attention to fraud patterns.

**Results:**

| Set | Accuracy | Precision | Recall | F1 | ROC AUC |
|-----|----------|-----------|--------|----|---------|
| Train | 0.7200 | 0.4591 | 0.7457 | 0.5683 | 0.7830 |
| Validation | 0.7133 | 0.4500 | 0.7297 | 0.5567 | 0.7828 |
| Test | 0.7133 | 0.4583 | 0.8919 | 0.6055 | 0.8496 |

**Analysis:** Train and validation metrics are close, so the model is not overfitting. Recall is high (73-89%) meaning we catch most fraud cases, but precision is low (45%) meaning many flagged claims are actually legitimate. The F1 of ~0.56-0.61 is our baseline to beat.

### Run 2: Cross-Validation Stability Check

**Goal:** Verify the baseline metrics are stable, not a fluke of the particular train/val split.

**Method:** 5-fold stratified CV on the training set.

**Results:**
| Metric | Mean | Std |
|--------|------|-----|
| F1 | 0.528 | 0.051 |
| Precision | 0.428 | 0.058 |
| Recall | 0.699 | 0.063 |
| ROC AUC | 0.760 | 0.038 |

**Analysis:** Standard deviations are moderate (0.04-0.06), indicating the model is reasonably stable but there's some variance from the small dataset size. The CV F1 (0.528) is slightly below the single-split validation F1 (0.557), suggesting the validation split was slightly optimistic.

### Run 3: Regularization Tuning (C parameter)

**Goal:** Find the optimal C value.

**Method:** Tested C values across 4 orders of magnitude: [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0].

**Key finding:** Best validation F1 at **C=0.01** (F1=0.571) — stronger regularization than the default (C=1.0) helps slightly. This makes sense: with only 700 training samples and 20 features, there's limited data to learn from, so constraining the model more (lower C) prevents fitting to noise.

For C >= 1.0, performance plateaus — the model has enough data to fit the patterns and additional flexibility doesn't help.

### Run 4: Best C with CV Confirmation

**Goal:** Confirm C=0.01 is genuinely better via cross-validation.

**Results:** CV F1 with C=0.01: 0.536 (+/- 0.021) — improved slightly over C=1.0 baseline CV, and the lower std (0.021 vs 0.051) indicates more stable performance.

Validation set metrics with C=0.01:
| Set | Accuracy | Precision | Recall | F1 | ROC AUC |
|-----|----------|-----------|--------|----|---------|
| Validation | 0.7200 | 0.4590 | 0.7568 | 0.5714 | 0.7797 |
| Test | 0.7133 | 0.4583 | 0.8919 | 0.6055 | 0.8398 |

### Run 5: Threshold Tuning

**Goal:** Optimize the classification threshold (default 0.5).

**Result:** The F1-optimal threshold turned out to be 0.50 — the default was already optimal for this model. At lower thresholds recall increases but precision drops too fast; at higher thresholds, precision improves but we miss too many fraud cases.

The threshold at 0.60 provides an interesting alternative: Precision=0.55, Recall=0.57, F1=0.56 — a more balanced trade-off if false positives are costly.

### Run 6: No Class Weighting (Comparison)

**Goal:** Quantify the impact of balanced class weights.

**Results WITHOUT balanced weights:**

| Set | Accuracy | Precision | Recall | F1 | ROC AUC |
|-----|----------|-----------|--------|----|---------|
| Validation | 0.7667 | 1.0000 | 0.0541 | 0.1026 | 0.7778 |
| Test | 0.7600 | 1.0000 | 0.0270 | 0.0526 | 0.8414 |

**Analysis:** Without balanced weights, the model achieves "higher accuracy" (76.7%) but catches only 2.7-5.4% of fraud cases — it essentially predicts non-fraud for almost everything. When it does predict fraud, it's always right (100% precision), but it's useless as a detector. This confirms that `class_weight='balanced'` is essential for this imbalanced dataset.

---

## Final Results Summary

| Run | Val F1 | Val Precision | Val Recall | Val ROC AUC | Test F1 | Test ROC AUC |
|-----|--------|---------------|------------|-------------|---------|--------------|
| V1: Baseline (C=1.0, balanced) | 0.5567 | 0.4500 | 0.7297 | 0.7828 | 0.6055 | 0.8496 |
| V2: Tuned C=0.01 | 0.5714 | 0.4590 | 0.7568 | 0.7797 | 0.6055 | 0.8398 |
| V3: Tuned C=0.01, threshold=0.50 | 0.5714 | 0.4590 | 0.7568 | 0.7797 | 0.6055 | 0.8398 |
| V4: No class weighting | 0.1026 | 1.0000 | 0.0541 | 0.7778 | 0.0526 | 0.8414 |

**Best configuration:** C=0.01, class_weight='balanced', threshold=0.50

**Test set performance:**
- F1: 0.6055 — reasonable for a linear model on a challenging fraud dataset
- ROC AUC: 0.8398 — the model has good discriminative ability
- Recall: 89.2% — catches most fraud cases
- Precision: 45.8% — about half of fraud predictions are correct

### Feature Importance (Top Coefficients)

The logistic regression coefficients reveal which features drive fraud predictions. Positive coefficients increase fraud probability, negative coefficients decrease it. The most influential features are `incident_severity` and `insured_hobbies`, which aligns with the MRMR feature selection results.

---

## Limitations & Next Steps

1. **Precision is low (45.8%)** — more than half of flagged cases are false alarms. Non-linear models (Random Forest, XGBoost) may capture complex feature interactions that improve precision without sacrificing recall.

2. **Label encoding assumes ordinal relationships** — features like `incident_city` have no natural ordering. Tree-based models handle categorical features more naturally.

3. **Small dataset (1000 samples)** — limits model capacity and makes evaluation noisy. More data would improve both model quality and metric reliability.

4. **Feature engineering opportunities** — interactions between features (e.g., claim amount relative to policy premium), temporal features from dates, or binning continuous variables could help.

---

## Files

| File | Description |
|------|-------------|
| `src/data/load_data.py` | Data loading functions |
| `src/data/preprocess.py` | Encoding, scaling, splitting |
| `src/models/model.py` | Logistic regression model factory |
| `src/models/train.py` | Training and cross-validation |
| `src/models/predict.py` | Prediction utilities |
| `src/evaluation/metrics.py` | Evaluation metrics |
| `src/config/config.yaml` | Hyperparameters and paths |
| `scripts/train.py` | End-to-end training script |
| `scripts/evaluate.py` | Evaluation script |
| `notebooks/logistic_regression_experiments.ipynb` | Experiment notebook with all runs |
| `models/logistic_regression.joblib` | Saved trained model |
| `models/scaler.joblib` | Saved feature scaler |
