# Random Forest - Documentation

## Overview

This document describes the Random Forest classifier built for fraud detection on the
auto insurance claims dataset. It covers the full pipeline: data preparation, model architecture,
experiment runs with progressive improvements, and results analysis.

Unlike the logistic regression experiments which used only the MRMR-selected features, this notebook
runs every experiment on **two feature sets** to directly compare the effect of feature selection:

| Label | Source | Features |
|-------|--------|----------|
| **Partially Selected (PS)** | `data/processed/partially_selected_features.csv` | 33 features (all after column cleanup) |
| **Selected (S)** | `data/processed/selected_features.csv` | 20 features (MRMR-selected subset) |

All experiments are logged in `notebooks/random_forest_experiments.ipynb`, where each run
occupies its own cells for reproducibility and comparison.

---

## Data Preparation Pipeline

### Input
Two datasets are used simultaneously:
- `data/processed/selected_features.csv` — 1000 samples, 20 MRMR-selected features + target
- `data/processed/partially_selected_features.csv` — 1000 samples, 33 features + target

### Step 1: Target Encoding
`fraud_reported` is converted from `Y/N` strings to `1/0` integers.

### Step 2: Categorical Feature Encoding
Categorical features are label-encoded — each unique category gets an integer ID.

### Step 3: Train / Validation / Test Split
- **70% training** (700 samples) — model learns from this
- **15% validation** (150 samples) — used for hyperparameter tuning
- **15% test** (150 samples) — held out completely until final evaluation
- All splits are **stratified** to maintain the ~75/25 non-fraud/fraud ratio
- Both feature sets use `random_state=42`, so the same rows land in the same split

### No Feature Scaling
Unlike logistic regression, Random Forest does **not** require feature scaling. Trees split on
thresholds (`feature_i <= value`), and this decision is invariant to monotonic transformations.
Scaling would not change any split point or model behavior, so it is skipped entirely.

---

## Why Random Forest?

Random Forest was chosen as the second classifier for several reasons:

1. **Non-linearity** — logistic regression can only learn linear decision boundaries. Random Forest
   captures non-linear relationships and feature interactions automatically through recursive
   splitting.

2. **Handles categorical features better** — label encoding introduces artificial ordering that
   logistic regression treats as meaningful. Trees can split on any value, effectively undoing
   the ordering assumption.

3. **Built-in feature importance** — Gini importance ranks features by how much they contribute
   to splitting decisions across all trees, providing a different perspective from mutual information.

4. **Robustness** — less sensitive to outliers and feature scaling than logistic regression.
   Individual trees overfit, but the ensemble (bagging) averages out the noise.

### How Random Forest Works

Random Forest builds an ensemble of decision trees, each trained on a different bootstrap sample
(random subset with replacement) of the training data. At each node split, only a random subset
of features is considered. This double randomness (data + features) ensures the trees are diverse.

For classification, each tree "votes" for a class, and the final prediction is the majority vote
(or the average probability across trees).

**Key hyperparameters:**
- `n_estimators` — number of trees. More trees = lower variance, diminishing returns
- `max_depth` — maximum tree depth. Controls overfitting of individual trees
- `min_samples_split` / `min_samples_leaf` — minimum samples to split a node / form a leaf

---

## Evaluation Methodology

**Important:** Throughout all experiment runs, the test set is **never** used for model selection or
hyperparameter tuning. All intermediate evaluations use only the training and validation sets.
The test set is evaluated exactly **once** at the very end. Both feature sets follow the same
protocol.

---

## Experiment Runs

### Run 1: Baseline (100 trees, no depth limit, balanced weights)

**Configuration:** 100 trees, unlimited depth, balanced class weights, all other defaults.

**Results — Selected (20 features):**

| Set | Accuracy | Precision | Recall | F1 | ROC AUC |
|-----|----------|-----------|--------|----|---------|
| Train | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Validation | 0.8000 | 0.6400 | 0.4324 | 0.5161 | 0.8282 |

**Results — Partially Selected (33 features):**

| Set | Accuracy | Precision | Recall | F1 | ROC AUC |
|-----|----------|-----------|--------|----|---------|
| Train | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Validation | 0.7800 | 0.6667 | 0.2162 | 0.3265 | 0.8370 |

**Analysis:** Perfect training scores with unlimited depth are expected — each tree memorizes its
bootstrap sample. The large train-val gap signals heavy overfitting. Precision is reasonable but
recall is low, especially for PS (21.6%) — the model misses most fraud cases. Interestingly, PS
has higher ROC AUC (0.837 vs 0.828) but much lower F1 (0.327 vs 0.516), suggesting the extra
features add noise that hurts the default threshold decision.

### Run 2: Cross-Validation Stability Check

**Goal:** Verify baseline metrics are stable across folds.

**Results — Selected (20 features):**

| Metric | Mean | Std |
|--------|------|-----|
| F1 | 0.3922 | 0.0998 |
| Precision | 0.5610 | 0.1152 |
| Recall | 0.3072 | 0.0981 |
| ROC AUC | 0.8444 | 0.0127 |

**Results — Partially Selected (33 features):**

| Metric | Mean | Std |
|--------|------|-----|
| F1 | 0.2648 | 0.0826 |
| Precision | 0.5403 | 0.0915 |
| Recall | 0.1787 | 0.0608 |
| ROC AUC | 0.8360 | 0.0235 |

**Analysis:** High variance in F1 and recall (std ~0.08-0.10) indicates the baseline model is
unstable — performance varies significantly across folds. ROC AUC is more stable, suggesting the
model's ranking of samples is decent but the decision boundary at threshold=0.5 is poor. The
unlimited-depth trees are overfitting, which depth tuning should address.

### Run 3: Number of Trees (n_estimators)

**Goal:** Find where performance plateaus.

**Key finding:** Both feature sets peaked at **n_estimators=100**. More trees did not improve
validation F1 — the variance reduction from bagging saturated early, likely because the dataset
is small (700 training samples) and the trees are highly correlated.

### Run 4: Max Depth Tuning

**Goal:** Control individual tree overfitting.

**Results — Selected (20 features):**

| max_depth | Train F1 | Val F1 | Val ROC AUC |
|-----------|----------|--------|-------------|
| 3 | 0.6947 | 0.6667 | 0.8464 |
| **5** | **0.7778** | **0.6829** | **0.8244** |
| 7 | 0.8804 | 0.6410 | 0.8299 |
| 10 | 1.0000 | 0.5588 | 0.8366 |
| None | 1.0000 | 0.5161 | 0.8282 |

**Results — Partially Selected (33 features):**

| max_depth | Train F1 | Val F1 | Val ROC AUC |
|-----------|----------|--------|-------------|
| 3 | 0.7028 | 0.6098 | 0.8338 |
| **5** | **0.7887** | **0.6988** | **0.8400** |
| 7 | 0.9136 | 0.6316 | 0.8436 |
| 10 | 1.0000 | 0.5246 | 0.8273 |
| None | 1.0000 | 0.3265 | 0.8370 |

**Analysis:** `max_depth=5` is optimal for both feature sets. This is the most impactful
hyperparameter — it improved validation F1 from 0.516 to 0.683 (Selected) and from 0.327 to
0.699 (PS). Shallower trees (depth=5) generalize much better because they can't memorize noise.
The train-val gap shrinks dramatically (from perfect train to ~0.78 train F1).

### Run 5: Min Samples Split & Min Samples Leaf

**Goal:** Further regularization beyond depth limiting.

**Best configuration — Selected:** `min_samples_split=10, min_samples_leaf=4` (Val F1=0.6988)

**Best configuration — Partially Selected:** `min_samples_split=2, min_samples_leaf=1` — defaults were already optimal (Val F1=0.6988)

**Analysis:** With depth already limited to 5, additional min_samples constraints had minimal effect.
For the Selected features, slightly higher min_samples improved F1 marginally. Both feature sets
converged to the same validation F1 (0.6988).

### Run 6: Best Configuration with CV Confirmation

**Goal:** Confirm the tuned hyperparameters are genuinely better.

**Selected (20 features):** CV F1 = 0.6697 (+/- 0.0550) — up from baseline 0.3922

| Set | Accuracy | Precision | Recall | F1 | ROC AUC |
|-----|----------|-----------|--------|----|---------|
| Train | 0.8614 | 0.6652 | 0.8844 | 0.7593 | 0.9355 |
| Validation | 0.8333 | 0.6304 | 0.7838 | 0.6988 | 0.8419 |

**Partially Selected (33 features):** CV F1 = 0.6032 (+/- 0.0584) — up from baseline 0.2648

| Set | Accuracy | Precision | Recall | F1 | ROC AUC |
|-----|----------|-----------|--------|----|---------|
| Train | 0.8829 | 0.7116 | 0.8844 | 0.7887 | 0.9600 |
| Validation | 0.8333 | 0.6304 | 0.7838 | 0.6988 | 0.8400 |

**Analysis:** Depth tuning was transformative. The tuned model now has a reasonable train-val gap
and catches 78% of fraud cases (vs 43% and 22% in the baseline). Both feature sets achieve
identical validation F1 (0.6988) and recall (0.7838), but PS has slightly higher CV F1 variance.

### Run 7: Threshold Tuning

**Goal:** Optimize the classification threshold for fraud detection.

**Selected (20 features):** Best threshold = **0.45** (F1=0.7143)
- At t=0.45: Precision=0.6383, Recall=0.8108 — catches more fraud than the default

**Partially Selected (33 features):** Best threshold = **0.50** — default was already optimal (F1=0.6988)

**Analysis:** For the Selected features, lowering the threshold to 0.45 improved F1 from 0.699 to
0.714 by increasing recall from 78% to 81% with only a small precision drop. The PS model's
probability distribution is more concentrated, so the default 0.50 was already optimal.

### Run 8: No Class Weighting (Comparison)

**Goal:** Quantify the impact of balanced class weights.

**Selected without balanced weights:**

| Set | Accuracy | Precision | Recall | F1 | ROC AUC |
|-----|----------|-----------|--------|----|---------|
| Validation | 0.7867 | 0.6471 | 0.2973 | 0.4074 | 0.8369 |

**Partially Selected without balanced weights:**

| Set | Accuracy | Precision | Recall | F1 | ROC AUC |
|-----|----------|-----------|--------|----|---------|
| Validation | 0.7533 | 0.5000 | 0.0541 | 0.0976 | 0.8249 |

**Analysis:** Without balanced weights, recall collapses: 29.7% for S and 5.4% for PS. The model
defaults to predicting non-fraud for nearly everything. PS is especially affected — with 33
features and no class weighting, the trees find many ways to classify the majority class correctly
while ignoring the minority class. This confirms `class_weight='balanced'` is essential.

---

## Model Selection (Validation Metrics)

### Selected (20 features)

| Run | Val F1 | Val Precision | Val Recall | Val ROC AUC |
|-----|--------|---------------|------------|-------------|
| V1: Baseline (100 trees, no depth limit) | 0.5161 | 0.6400 | 0.4324 | 0.8282 |
| V2: Tuned (n=100, depth=5, split=10, leaf=4) | 0.6988 | 0.6304 | 0.7838 | 0.8419 |
| V3: Tuned + threshold=0.45 | 0.7143 | 0.6383 | 0.8108 | 0.8419 |
| V4: No class weighting | 0.4074 | 0.6471 | 0.2973 | 0.8369 |

### Partially Selected (33 features)

| Run | Val F1 | Val Precision | Val Recall | Val ROC AUC |
|-----|--------|---------------|------------|-------------|
| V1: Baseline (100 trees, no depth limit) | 0.3265 | 0.6667 | 0.2162 | 0.8370 |
| V2: Tuned (n=100, depth=5, split=2, leaf=1) | 0.6988 | 0.6304 | 0.7838 | 0.8400 |
| V3: Tuned + threshold=0.50 | 0.6988 | 0.6304 | 0.7838 | 0.8400 |
| V4: No class weighting | 0.0976 | 0.5000 | 0.0541 | 0.8249 |

**Best configuration — Selected:** n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=4, class_weight='balanced', threshold=0.45

**Best configuration — Partially Selected:** n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, class_weight='balanced', threshold=0.50

---

## Cross-Dataset Comparison

Both feature sets achieve very similar tuned performance (Val F1 = 0.699-0.714), but the path
there is different:

- **PS (33 features)** had a much worse baseline (F1=0.327 vs 0.516) because the extra features
  caused more overfitting with unlimited depth
- After depth tuning, the extra features in PS neither helped nor hurt — the model converged to
  similar performance
- The Selected set benefited from threshold tuning (0.45 improved F1 to 0.714) while PS did not

**Conclusion:** MRMR feature selection slightly benefits Random Forest, primarily by reducing
overfitting risk and enabling threshold tuning gains. The 13 additional features in the PS set
do not provide meaningful extra signal for this model.

---

## Final Test Evaluation (ONE-TIME)

The best model from each feature set is evaluated on the held-out test set exactly once.

**Selected (20 features) — threshold=0.45:**

| Accuracy | Precision | Recall | F1 | ROC AUC |
|----------|-----------|--------|----|---------|
| 0.8400 | 0.6226 | 0.8919 | 0.7333 | 0.8675 |

**Partially Selected (33 features) — threshold=0.50:**

| Accuracy | Precision | Recall | F1 | ROC AUC |
|----------|-----------|--------|----|---------|
| 0.8533 | 0.6596 | 0.8378 | 0.7381 | 0.8469 |

**Analysis:** Both feature sets perform similarly on the test set. PS has slightly higher F1 (0.738 vs
0.733) and precision (0.660 vs 0.623), while S has higher recall (0.892 vs 0.838) and ROC AUC
(0.868 vs 0.847). The difference is marginal — within the noise expected from 150 test samples.

### Feature Importance (Top 10 — Gini Importance)

**Selected (20 features):**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | incident_severity | 0.3904 |
| 2 | insured_hobbies | 0.1535 |
| 3 | total_claim_amount | 0.0707 |
| 4 | property_claim | 0.0659 |
| 5 | vehicle_claim | 0.0633 |
| 6 | injury_claim | 0.0496 |
| 7 | auto_make | 0.0267 |
| 8 | incident_state | 0.0263 |
| 9 | insured_relationship | 0.0248 |
| 10 | authorities_contacted | 0.0236 |

**Partially Selected (33 features):**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | incident_severity | 0.2623 |
| 2 | insured_hobbies | 0.1125 |
| 3 | total_claim_amount | 0.0596 |
| 4 | vehicle_claim | 0.0512 |
| 5 | property_claim | 0.0455 |
| 6 | insured_occupation | 0.0394 |
| 7 | policy_annual_premium | 0.0389 |
| 8 | injury_claim | 0.0370 |
| 9 | months_as_customer | 0.0330 |
| 10 | incident_state | 0.0302 |

The top 2 features (`incident_severity` and `insured_hobbies`) dominate in both feature sets,
consistent with the MRMR mutual information scores from feature selection. In the PS set, the
importance is spread more thinly across 33 features, diluting the top features' share.

---

## Comparison with Logistic Regression

| Metric | LR (Selected, Val) | RF (Selected, Val) |
|--------|--------------------|--------------------|
| F1 | 0.5714 | 0.7143 |
| Precision | 0.4590 | 0.6383 |
| Recall | 0.7568 | 0.8108 |
| ROC AUC | 0.7797 | 0.8419 |

Random Forest outperforms logistic regression on all validation metrics. The non-linear model
captures feature interactions that the linear model cannot, resulting in higher precision (+0.18)
without sacrificing recall (+0.05). The ROC AUC improvement (+0.06) confirms better overall
discrimination ability.

---

## Limitations & Next Steps

1. **Train-test gap still present** — even with depth limiting, Train F1 (~0.76) exceeds Val F1
   (~0.70), indicating some residual overfitting. More data or stronger regularization could help.

2. **Gradient boosting** — XGBoost or LightGBM may improve further by sequentially correcting
   errors rather than averaging independent trees.

3. **Feature engineering** — claim amount ratios, temporal features, or feature interactions
   could provide additional signal.

4. **Hyperparameter search** — a more exhaustive grid or randomized search over all parameters
   simultaneously (rather than sequential tuning) may find better configurations.

---

## Files

| File | Description |
|------|-------------|
| `src/data/load_data.py` | Data loading functions |
| `src/data/preprocess.py` | Encoding, scaling, splitting |
| `src/models/predict.py` | Prediction utilities |
| `src/evaluation/metrics.py` | Evaluation metrics |
| `src/config/config.yaml` | Hyperparameters and paths |
| `notebooks/random_forest_experiments.ipynb` | Experiment notebook with all runs |
| `data/processed/selected_features.csv` | MRMR-selected features (20 + target) |
| `data/processed/partially_selected_features.csv` | All features after cleanup (33 + target) |
