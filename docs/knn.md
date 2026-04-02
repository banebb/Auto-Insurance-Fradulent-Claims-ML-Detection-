# K-Nearest Neighbors - Documentation

## Overview

This document describes the K-Nearest Neighbors (KNN) classifier built for fraud detection on the
auto insurance claims dataset. It covers the full pipeline: data preparation, model architecture,
experiment runs with progressive improvements, and results analysis.

Each experiment is run on **two feature sets** to compare the effect of feature selection:

| Label | Source | Features |
|-------|--------|----------|
| **Partially Selected (PS)** | `data/processed/partially_selected_features.csv` | 33 features (all after column cleanup) |
| **Selected (S)** | `data/processed/selected_features.csv` | 20 features (MRMR-selected subset) |

All experiments are logged in `notebooks/knn_experiments.ipynb`, where each run
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

### Step 4: Feature Scaling
StandardScaler normalizes each feature to zero mean and unit variance. The scaler is **fit on
training data only**, then applied to validation and test sets.

**Why scaling is essential for KNN:** KNN uses distance (e.g. Euclidean) to find nearest neighbors.
Without scaling, features with large ranges (e.g. `total_claim_amount` in thousands) dominate
the distance calculation, making features with small ranges (e.g. `witnesses` in 0-5) irrelevant.
This is different from Random Forest (scale-invariant) but similar to logistic regression.

---

## Why KNN?

KNN was chosen as the third classifier to provide a fundamentally different approach:

1. **Non-parametric** — makes no assumptions about the data distribution (unlike logistic regression
   which assumes a linear boundary, or Random Forest which builds explicit decision rules).

2. **Instance-based** — no training phase; at prediction time it searches for the K most similar
   training samples. This means it can capture arbitrarily complex decision boundaries.

3. **Local patterns** — classifies based on nearby examples, which can detect localized fraud
   patterns that global models might miss.

4. **Diagnostic value** — if KNN performs poorly, it suggests that fraud and non-fraud cases are
   not well-separated in the feature space, which informs feature engineering.

### How KNN Works

For each new sample to classify:
1. Compute the distance to all training samples
2. Select the K nearest (most similar) training samples
3. Take a majority vote: if more than half the neighbors are fraud, predict fraud

The probability output is the fraction of K neighbors that are fraud: `P(fraud) = count(fraud neighbors) / K`.

**Key hyperparameters:**
- `n_neighbors (K)` — number of neighbors to consider. Small K = sensitive to noise, large K = smoother boundaries
- `weights` — `uniform` (all K neighbors vote equally) or `distance` (closer neighbors count more)
- `metric` — distance function (Euclidean, Manhattan, etc.)

### The Class Imbalance Challenge

**KNN has no `class_weight` parameter.** This is a critical difference from logistic regression and
Random Forest. With ~75% non-fraud in the training set, most neighborhoods are majority non-fraud.
At the default threshold of 0.5, a sample needs more than K/2 fraud neighbors to be classified
as fraud — which is unlikely when only 25% of training data is fraud.

**Solution:** Aggressive threshold tuning. By lowering the threshold (e.g., to 0.25-0.30), we
accept samples as fraud even when a minority of their neighbors are fraud. This is equivalent
to implicitly adjusting for class imbalance through the decision threshold rather than through
sample weights.

---

## Evaluation Methodology

**Important:** Throughout all experiment runs, the test set is **never** used for model selection or
hyperparameter tuning. All intermediate evaluations use only the training and validation sets.
The test set is evaluated exactly **once** at the very end. Both feature sets follow the same
protocol.

---

## Experiment Runs

### Run 1: Baseline (K=5, uniform, Euclidean)

**Configuration:** K=5, uniform weights, Euclidean distance — the most common defaults.

**Results — Selected (20 features):**

| Set | Accuracy | Precision | Recall | F1 | ROC AUC |
|-----|----------|-----------|--------|----|---------|
| Train | 0.8200 | 0.7527 | 0.4046 | 0.5263 | 0.8404 |
| Validation | 0.7200 | 0.3333 | 0.1351 | 0.1923 | 0.6260 |

**Results — Partially Selected (33 features):**

| Set | Accuracy | Precision | Recall | F1 | ROC AUC |
|-----|----------|-----------|--------|----|---------|
| Train | 0.7886 | 0.6667 | 0.2890 | 0.4032 | 0.8024 |
| Validation | 0.6933 | 0.2353 | 0.1081 | 0.1481 | 0.5996 |

**Analysis:** The baseline is very poor — recall is 13.5% (S) and 10.8% (PS), meaning the model
catches almost no fraud. This is the class imbalance problem: with K=5, a sample needs 3+ fraud
neighbors to exceed the 0.5 threshold. PS performs worse due to the curse of dimensionality — in
33 dimensions, distances become less meaningful and neighborhoods are less informative.

### Run 2: Cross-Validation Stability Check

**Goal:** Verify baseline metrics and check ROC AUC (threshold-independent).

**Results — Selected (20 features):**

| Metric | Mean | Std |
|--------|------|-----|
| F1 | 0.2852 | 0.0740 |
| Precision | 0.3821 | 0.0455 |
| Recall | 0.2319 | 0.0777 |
| ROC AUC | 0.6014 | 0.0393 |

**Results — Partially Selected (33 features):**

| Metric | Mean | Std |
|--------|------|-----|
| F1 | 0.2287 | 0.0340 |
| Precision | 0.4030 | 0.0689 |
| Recall | 0.1618 | 0.0291 |
| ROC AUC | 0.5724 | 0.0408 |

**Analysis:** Low F1/recall at default threshold is expected (no class weighting). ROC AUC is
only 0.60-0.57, suggesting KNN's ranking quality is mediocre with K=5. The curse of dimensionality
hurts PS more (AUC 0.57 vs 0.60).

### Run 3: K Tuning with Threshold Optimization

**Goal:** Find the optimal K, evaluated at each K's best threshold (not just 0.5).

**Key insight:** At default threshold (t=0.5), higher K values appear useless (F1 drops to 0).
But with threshold tuning, higher K values are actually much better — they produce smoother
probability estimates that enable better precision-recall trade-offs.

**Results — Selected (20 features):**

| K | Val F1 (t=0.5) | Best Threshold | Val F1 (tuned) | Val ROC AUC |
|---|----------------|----------------|----------------|-------------|
| 1 | 0.2857 | 0.05 | 0.2857 | 0.5203 |
| 5 | 0.1923 | 0.05 | 0.4414 | 0.6260 |
| 11 | 0.2128 | 0.10 | 0.4595 | 0.6577 |
| 21 | 0.1026 | 0.20 | 0.4833 | 0.6631 |
| **51** | **0.0000** | **0.30** | **0.5063** | **0.7315** |

**Results — Partially Selected (33 features):**

| K | Val F1 (t=0.5) | Best Threshold | Val F1 (tuned) | Val ROC AUC |
|---|----------------|----------------|----------------|-------------|
| 1 | 0.3030 | 0.05 | 0.3030 | 0.5511 |
| 5 | 0.1481 | 0.05 | 0.4324 | 0.5996 |
| 11 | 0.1333 | 0.20 | 0.4954 | 0.7239 |
| **15** | **0.0952** | **0.25** | **0.5138** | **0.7010** |
| 51 | 0.0000 | 0.20 | 0.4571 | 0.6854 |

**Analysis:** Best K is 51 for Selected and 15 for Partially Selected. The pattern is clear:
higher K produces better AUC and threshold-tuned F1, despite appearing worse at t=0.5. The
optimal threshold decreases as K increases — with more neighbors, a lower fraction of fraud
neighbors becomes statistically meaningful.

### Run 4: Weighting Scheme

**Goal:** Compare uniform vs distance-weighted voting.

**Result:** Uniform weighting won for both feature sets. With large K values (15-51), many
neighbors are at similar distances, so distance weighting provides minimal benefit.

### Run 5: Distance Metric

**Goal:** Compare Euclidean vs Manhattan distance.

**Results — Selected (20 features):** Manhattan (F1=0.5570 at t=0.30) beat Euclidean (F1=0.5063) — Manhattan's robustness to outliers and better behavior in high dimensions helped.

**Results — Partially Selected (33 features):** Euclidean (F1=0.5138) beat Manhattan (F1=0.4828) — with 33 features, Euclidean's sensitivity to magnitude differences (amplified by scaling) provided slightly better discrimination.

### Run 6: Best Configuration with CV Confirmation

**Selected (20 features):** K=51, uniform, Manhattan

| Set | Accuracy | Precision | Recall | F1 | ROC AUC |
|-----|----------|-----------|--------|----|---------|
| Validation (t=0.5) | 0.7600 | 1.0000 | 0.0270 | 0.0526 | 0.7914 |

CV ROC AUC: 0.7331 (+/- 0.0391) — decent ranking ability despite poor default-threshold F1.

**Partially Selected (33 features):** K=15, uniform, Euclidean

| Set | Accuracy | Precision | Recall | F1 | ROC AUC |
|-----|----------|-----------|--------|----|---------|
| Validation (t=0.5) | 0.7467 | 0.4000 | 0.0541 | 0.0952 | 0.7010 |

CV ROC AUC: 0.6230 (+/- 0.0426)

### Run 7: Threshold Tuning

**Selected (20 features):** Best threshold = **0.30** (F1=0.5570)
- At t=0.30: Precision=0.5238, Recall=0.5946

**Partially Selected (33 features):** Best threshold = **0.25** (F1=0.5138)
- At t=0.25: Precision=0.3889, Recall=0.7568

**Analysis:** Threshold tuning is transformative for KNN — it converts a near-useless classifier
(F1=0.05) into a functional one (F1=0.51-0.56). The optimal thresholds (0.25-0.30) are well below
the default 0.50, reflecting the class imbalance compensation.

---

## Model Selection (Validation Metrics)

### Selected (20 features)

| Run | Val F1 | Val Precision | Val Recall | Val ROC AUC |
|-----|--------|---------------|------------|-------------|
| V1: Baseline (K=5, uniform, euclidean) | 0.1923 | 0.3333 | 0.1351 | 0.6260 |
| V2: Tuned (K=51, uniform, manhattan) | 0.0526 | 1.0000 | 0.0270 | 0.7914 |
| V3: Tuned + threshold=0.30 | 0.5570 | 0.5238 | 0.5946 | 0.7914 |

### Partially Selected (33 features)

| Run | Val F1 | Val Precision | Val Recall | Val ROC AUC |
|-----|--------|---------------|------------|-------------|
| V1: Baseline (K=5, uniform, euclidean) | 0.1481 | 0.2353 | 0.1081 | 0.5996 |
| V2: Tuned (K=15, uniform, euclidean) | 0.0952 | 0.4000 | 0.0541 | 0.7010 |
| V3: Tuned + threshold=0.25 | 0.5138 | 0.3889 | 0.7568 | 0.7010 |

**Best configuration — Selected:** K=51, uniform, Manhattan, threshold=0.30

**Best configuration — Partially Selected:** K=15, uniform, Euclidean, threshold=0.25

---

## Cross-Dataset Comparison

Selected (20 features) outperforms PS (33 features):
- **Val F1:** S=0.5570 vs PS=0.5138
- **Val ROC AUC:** S=0.7914 vs PS=0.7010

The curse of dimensionality hits KNN harder than tree-based models. In 33 dimensions, the concept
of "nearest neighbor" becomes less meaningful because distances between all points converge.
MRMR feature selection removing 13 irrelevant features makes a meaningful difference for KNN.

---

## Final Test Evaluation (ONE-TIME)

**Selected (20 features) — K=51, Manhattan, threshold=0.30:**

| Accuracy | Precision | Recall | F1 | ROC AUC |
|----------|-----------|--------|----|---------|
| 0.7600 | 0.5116 | 0.5946 | 0.5500 | 0.7791 |

**Partially Selected (33 features) — K=15, Euclidean, threshold=0.25:**

| Accuracy | Precision | Recall | F1 | ROC AUC |
|----------|-----------|--------|----|---------|
| 0.6533 | 0.3810 | 0.6486 | 0.4800 | 0.7181 |

---

## Comparison with Other Models (Validation — Selected Features)

| Model | Val F1 | Val Precision | Val Recall | Val ROC AUC |
|-------|--------|---------------|------------|-------------|
| Logistic Regression | 0.5714 | 0.4590 | 0.7568 | 0.7797 |
| Random Forest | 0.7143 | 0.6383 | 0.8108 | 0.8419 |
| KNN | 0.5570 | 0.5238 | 0.5946 | 0.7914 |

KNN performs comparably to logistic regression but significantly worse than Random Forest.
KNN achieves better precision than LR (0.52 vs 0.46) but lower recall (0.59 vs 0.76),
meaning it flags fewer cases but is slightly more accurate when it does. Random Forest
dominates on all metrics.

KNN's weakness on this dataset is likely due to:
1. **Label-encoded categoricals** — Euclidean/Manhattan distance between integer-encoded
   categories is not meaningful (e.g., distance between city=3 and city=7 has no real meaning)
2. **Mixed feature types** — the dataset has both numerical and categorical features, which
   is challenging for distance-based methods
3. **Small dataset** — 700 training samples in 20 dimensions means the feature space is sparsely
   populated, making nearest-neighbor lookup less reliable

---

## Limitations & Next Steps

1. **Categorical feature handling** — label encoding creates meaningless distances for categorical
   features. Techniques like target encoding, or using a mixed-distance metric (Gower distance)
   that handles categorical features natively, could help.

2. **Curse of dimensionality** — KNN degrades in high dimensions. Dimensionality reduction
   (PCA, UMAP) before KNN could improve performance.

3. **Class imbalance** — relying solely on threshold tuning is a blunt instrument. Oversampling
   (SMOTE) or undersampling the training set could provide KNN with more balanced neighborhoods.

4. **Computational cost** — KNN stores the entire training set and searches it at prediction time.
   On larger datasets, this becomes a bottleneck (though not an issue with 700 samples).

---

## Files

| File | Description |
|------|-------------|
| `src/data/load_data.py` | Data loading functions |
| `src/data/preprocess.py` | Encoding, scaling, splitting |
| `src/models/predict.py` | Prediction utilities |
| `src/evaluation/metrics.py` | Evaluation metrics |
| `src/config/config.yaml` | Hyperparameters and paths |
| `notebooks/knn_experiments.ipynb` | Experiment notebook with all runs |
| `data/processed/selected_features.csv` | MRMR-selected features (20 + target) |
| `data/processed/partially_selected_features.csv` | All features after cleanup (33 + target) |
