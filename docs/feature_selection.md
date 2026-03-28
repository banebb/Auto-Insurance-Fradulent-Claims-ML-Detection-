# Feature Selection Documentation

## Overview

Feature selection was performed on the auto insurance claims dataset (1000 samples, 40 features) to identify the most relevant features for fraud detection. Two methods were used: **Mutual Information (MI)** and **MRMR (Minimum Redundancy Maximum Relevance)**.

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
