# Feature Selection Documentation

## Overview

Feature selection was performed on the auto insurance claims dataset (1000 samples, 40 features) to identify the most relevant features for fraud detection. Two methods were used: **Mutual Information (MI)** and **MRMR (Minimum Redundancy Maximum Relevance)**.

---

## How Mutual Information Works

### Theory

Mutual Information (MI) is rooted in information theory (Shannon, 1948). It quantifies how much knowing the value of one variable reduces uncertainty about another. For two random variables X (a feature) and Y (the target):

```
MI(X; Y) = H(Y) - H(Y|X)
```

Where:
- **H(Y)** is the entropy (uncertainty) of Y alone
- **H(Y|X)** is the conditional entropy of Y given X (remaining uncertainty after observing X)
- The difference is how much uncertainty about Y is *removed* by knowing X

Equivalently, MI can be expressed using probability distributions:

```
MI(X; Y) = sum over all x,y of p(x,y) * log( p(x,y) / (p(x) * p(y)) )
```

This compares the joint distribution p(x,y) against what it would be if X and Y were independent (p(x)*p(y)). If knowing X tells us nothing about Y, these are equal and MI = 0.

**Key properties:**
- MI >= 0 always (non-negative)
- MI = 0 only when X and Y are statistically independent
- Unlike correlation, MI captures *any* kind of dependency (linear, nonlinear, categorical patterns)
- MI is symmetric: MI(X;Y) = MI(Y;X)

### How sklearn Estimates MI

For continuous features, the true probability distributions are unknown, so sklearn uses the **k-nearest neighbors (KNN) estimator** by Kraskov, Stogbauer, and Grassberger (2004). The idea:
1. For each data point, find its k nearest neighbors in the joint (X, Y) space
2. Count how many points fall within the same distance in the marginal X and Y spaces separately
3. Use these counts to estimate the log-ratio of joint vs marginal densities

For discrete (categorical) features, MI is computed directly by counting co-occurrences to estimate the joint and marginal probabilities, then plugging into the MI formula.

In our notebook, we pass `discrete_features=discrete_mask` so sklearn knows which columns are categorical (direct counting) and which are continuous (KNN estimation). We use `n_neighbors=5` (the k in KNN estimation).

### Limitation

MI scores each feature *independently* against the target. It does not account for relationships between features. So if `injury_claim`, `property_claim`, and `vehicle_claim` all correlate with fraud, MI will rank all three highly even though they carry largely the same information (they sum to `total_claim_amount`). This is why MI alone can produce a redundant feature set.

---

## How MRMR Works

### Theory

MRMR (Minimum Redundancy Maximum Relevance) was proposed by Peng, Long, and Ding (2005). It addresses MI's blind spot by selecting features that are:
- **Maximally relevant** to the target variable
- **Minimally redundant** with features already selected

At each step, MRMR selects the next feature by solving:

```
max_f [ relevance(f, target) - (1/|S|) * sum over s in S of redundancy(f, s) ]
```

Where:
- **f** is a candidate feature
- **S** is the set of already-selected features
- **relevance(f, target)** = MI(f; target) or F-statistic between f and the target
- **redundancy(f, s)** = MI(f; s) or correlation between f and each already-selected feature s
- The factor `1/|S|` averages the redundancy over all previously selected features

### Step-by-step Algorithm

1. **First feature:** Select the feature with the highest relevance to the target (same as pure MI ranking). In our case: `incident_severity` (MI = 0.1237).

2. **Second feature:** For each remaining candidate, compute:
   - Its relevance to the target
   - Its redundancy with `incident_severity`
   - Score = relevance - redundancy
   - Pick the candidate with the highest score. Result: `umbrella_limit` — it has moderate relevance but is *very different* from `incident_severity`.

3. **Third feature and beyond:** Same process, but redundancy is averaged across all features already in S. This is why `witnesses` (MI = 0.0000 individually!) gets selected at rank 5 — it carries non-redundant information that complements what's already been chosen.

### Why MRMR Differs from MI Ranking

In our results, the key differences come from redundancy penalization:
- **MI ranks `injury_claim` (6th), `property_claim` (10th), and `vehicle_claim` (19th) all in top 20** — but they're sub-components of `total_claim_amount`. MRMR selects `total_claim_amount` (rank 6) and pushes the sub-claims lower because they're redundant with it.
- **MRMR promotes `witnesses` to rank 5** despite near-zero MI, because it provides unique information not captured by any previously selected feature.
- **`auto_model` (MI rank 4) is dropped by MRMR** — its information overlaps with `auto_make` and other features.

### Implementation Details

The `mrmr-selection` library uses an efficient implementation that computes F-statistics for relevance (between each feature and the target class) and Pearson correlation for redundancy (between feature pairs). This is a practical approximation that's faster than computing MI between all feature pairs while still capturing the core MRMR principle.

In our notebook:
```python
mrmr.mrmr_classif(X=features, y=target, K=20)
```
- `K=20`: Select the top 20 features
- The function returns features in selection order (most important first)
- Classification variant is used since our target is binary (fraud: Y/N)

---

## Step 1: Column Removal

The following columns were dropped before feature selection as they carry no predictive value:

| Column | Reason |
|--------|--------|
| `_c39` | 100% missing values (empty column) |
| `policy_number` | Unique identifier |
| `policy_bind_date` | Raw date string |
| `insured_zip` | High cardinality identifier |
| `incident_location` | Too specific, high cardinality |
| `incident_date` | Raw date string |

**Remaining features after removal: 34 (33 predictors + 1 target)**

## Step 2: Handling Missing/Invalid Values

Several columns contained `?` as a placeholder for missing values, and `authorities_contacted` had actual NaN values:

| Column | Missing Count | Missing % | Imputation |
|--------|--------------|-----------|------------|
| `property_damage` | 360 | 36.00% | Mode (`NO`) |
| `police_report_available` | 343 | 34.30% | Mode (`NO`) |
| `collision_type` | 178 | 17.80% | Mode (`Rear Collision`) |
| `authorities_contacted` | 91 | 9.10% | Mode (`Police`) |

## Step 3: Feature Encoding

All 17 categorical features were label-encoded to enable numerical MI computation.

## Step 4: Mutual Information Scores

MI scores measure statistical dependency between each feature and the target (`fraud_reported`).

**Top 12 features by MI score:**

| Rank | Feature | MI Score |
|------|---------|----------|
| 1 | `incident_severity` | 0.1237 |
| 2 | `insured_hobbies` | 0.0724 |
| 3 | `total_claim_amount` | 0.0296 |
| 4 | `auto_model` | 0.0245 |
| 5 | `collision_type` | 0.0182 |
| 6 | `injury_claim` | 0.0179 |
| 7 | `incident_type` | 0.0172 |
| 8 | `auto_year` | 0.0165 |
| 9 | `authorities_contacted` | 0.0154 |
| 10 | `property_claim` | 0.0133 |
| 11 | `umbrella_limit` | 0.0116 |
| 12 | `policy_annual_premium` | 0.0114 |

**Features with near-zero MI (< 0.01):** `capital-loss`, `capital-gains`, `insured_occupation`, `incident_state`, `auto_make`, `bodily_injuries`, `vehicle_claim`, `property_damage`, `number_of_vehicles_involved`, `insured_relationship`, `incident_hour_of_the_day`, `incident_city`, `policy_csl`, `insured_education_level`, `policy_state`, `insured_sex`, `police_report_available`, `months_as_customer`, `age`, `policy_deductable`, `witnesses`

## Step 5: MRMR Feature Selection

MRMR selects features that are maximally relevant to the target while minimally redundant with each other. Top 20 features were selected.

**MRMR selected features (in order of selection):**

1. `incident_severity`
2. `umbrella_limit`
3. `property_damage`
4. `vehicle_claim`
5. `witnesses`
6. `total_claim_amount`
7. `insured_hobbies`
8. `collision_type`
9. `incident_city`
10. `authorities_contacted`
11. `property_claim`
12. `incident_state`
13. `policy_state`
14. `bodily_injuries`
15. `insured_sex`
16. `policy_csl`
17. `injury_claim`
18. `auto_make`
19. `number_of_vehicles_involved`
20. `insured_relationship`

## MI vs MRMR Comparison

- **Features in both top-20 lists:** 13
- **Only in MI top-20:** `auto_year`, `incident_type`, `policy_annual_premium`, `capital-loss`, `capital-gains`, `insured_occupation`, `auto_model`
- **Only in MRMR top-20:** `number_of_vehicles_involved`, `incident_city`, `policy_csl`, `witnesses`, `insured_relationship`, `policy_state`, `insured_sex`

Key difference: MRMR promotes features like `witnesses` and `property_damage` that have low MI individually but add non-redundant information. Meanwhile, MI ranks correlated features like `injury_claim`, `property_claim`, and `vehicle_claim` highly despite their redundancy with `total_claim_amount`.

## Step 6: Final Selection

The MRMR-selected 20 features were used as the final feature set (MRMR accounts for both relevance and redundancy). The 13 dropped features are:

`months_as_customer`, `age`, `policy_deductable`, `policy_annual_premium`, `capital-gains`, `capital-loss`, `incident_hour_of_the_day`, `insured_education_level`, `insured_occupation`, `incident_type`, `auto_model`, `auto_year`, `police_report_available`

## Output

The selected features dataset was saved to: `data/processed/selected_features.csv` (1000 rows, 21 columns: 20 features + target)

## Notebooks

- `notebooks/data_overview.ipynb` - Initial data exploration and quality analysis
- `notebooks/feature_selection.ipynb` - Feature selection pipeline (MI + MRMR)
