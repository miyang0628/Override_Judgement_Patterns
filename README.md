# Detecting Hidden Evaluator Judgment Patterns in Credit Rating Override Decisions: A Two-Layer Early-Warning Factor Library with Nested Cross-Fold Validation

> **Blind review submission** — Author information withheld.
> Submitted to [Anonymous Journal], [Anonymous Date].

---

## Overview

This repository contains the full replication code for a two-paper series on evaluator upgrade overrides in corporate credit rating.

**Study 1** (companion paper) established that upgrade overrides are systematically associated with elevated default rates using the Polish Companies Bankruptcy dataset.

**This study** asks the deeper question: *which specific financial patterns do evaluators overlook when they upgrade a firm's grade, and can these be formalised into a pre-emptive warning system that generalises beyond the sample used to discover it?*

The answer is a **two-layer warning factor library** — single-variable factors (Layer 1) and composite-pattern rules (Layer 2) — that flags high-risk upgrade decisions in real time, before the outcome is known. Unlike a single-sample retrospective analysis, the library and its validation are built on a **nested 5-fold cross-validation design**: within each fold, the system credit-grading model, grade thresholds, override simulation, and the entire warning factor library are re-estimated using *only* that fold's training partition (80% of the sample), and validated on a held-out partition (20%) never used in any stage of factor discovery. This prevents information from held-out firms leaking into the factors being validated against them, and directly tests whether the identified patterns generalise to firms not used to discover them.

---

## Key Findings

| Finding | Value |
|---------|-------|
| Upgrade cohort size (full sample) | 15,684 firm-years |
| Per-fold outer-training / outer-holdout split | ~80% / ~20% (stratified 5-fold) |
| Layer 1 factors (single-variable), per fold | 40–44 (mean 43.0, SD 1.7) |
| Layer 2 factors (composite-pattern), per fold | 16–49 (mean 25.8, SD 14.0) |
| **Total warning factors, per fold** | **58–93 (mean 68.8, SD 14.8)** |
| Factor stability: variables selected as Layer 1 in all 5 folds | 39 / 49 (80%) |
| Layer 1 vs Layer 2 stability (coefficient of variation) | 4.0% vs 54.2% |
| Mean train–holdout hit-rate gap, per factor (5-fold average) | +0.0061 (SD 0.0060), range −0.0017 to +0.0136 |
| Best-performing grade transition (all folds) | CCC→B — train F1 = 0.2893 ± 0.0059; holdout F1 = 0.2787 ± 0.0130 |
| Default rate ratio, 30+ warnings vs no warnings (holdout, 5-fold average) | 5.05× ± 2.24× |
| Robust optimal warning-count threshold (5-fold search, range 1–50) | 27 (mean holdout F1 = 0.2062, SD 0.0149) |
| Defaulting override cases already showing 20+ warnings at override time (holdout, 5-fold average) | 59.8% (range 56.9%–66.0%) |
| K-Means cluster count (K), across folds | Unstable: K = 3–5, Silhouette 0.137–0.371 |

> All figures above are cross-fold aggregates. Fold 0's individual-fold results (used for initial pipeline validation) and the full per-fold breakdown are provided in `results/tables/NB08_all_folds_raw.csv` and `NB08_5fold_aggregate_summary.csv`.

---

## Repository Structure

```
├── notebooks/
│   ├── NB01_data_preparation.ipynb       # Dataset loading, outer 5-fold split, per-fold system model/cohort construction
│   ├── NB02_eda.ipynb                    # Mann-Whitney U tests (per-fold training partition)
│   ├── NB03_univariate_logistic.ipynb    # Firth logistic regression, FDR correction, exact-duplicate removal, VIF screening
│   ├── NB03b_data_dictionary_check.ipynb # Variable definition mapping, union-find duplicate detection
│   ├── NB04_association_rules.ipynb      # Apriori ARM, binned-label agreement grouping, group-redundancy filtering
│   ├── NB05_clustering.ipynb             # K-Means clustering, K selection (Silhouette + DBI)
│   ├── NB06_judgment_factors.ipynb       # Two-layer warning factor library construction (per-fold, with explicit thresholds)
│   ├── NB07_simulation.ipynb             # Out-of-fold retrospective simulation (Q1–Q5), train vs holdout
│   └── NB08_five_fold_loop.ipynb         # Cross-fold orchestration and aggregation (folds 1–4), stability scoring,
│                                          # robust threshold search, evaluator-gap analysis
│
├── src/
│   └── firth_utils.py                    # Self-contained Firth (1993) bias-reduced logistic regression implementation
│
├── data/
│   ├── raw/                              # Place UCI .arff files here (not included — see below)
│   └── processed/
│       ├── fold_0/ … fold_4/             # Per-fold master/cohort parquet files (outer_split: train / holdout)
│       └── fold_assignment_summary.csv   # Cross-fold balance and AUC stability check
│
├── results/
│   ├── figures/                          # All output figures, per fold where applicable (auto-created)
│   └── tables/                           # All output CSV tables, per fold where applicable (auto-created)
│
└── README.md
```

---

## Data

This study uses the **Polish Companies Bankruptcy dataset** (Zięba et al., 2016), publicly available from the UCI Machine Learning Repository.

**Download link:** https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data

After downloading, place the five `.arff` files in `data/raw/`:

```
data/raw/1year.arff
data/raw/2year.arff
data/raw/3year.arff
data/raw/4year.arff
data/raw/5year.arff
```

> The raw data files are **not included** in this repository in compliance with UCI's terms of use. All processed outputs are reproducible by running the notebooks in order.

---

## Environment Setup

### Requirements

- Python 3.9+
- Conda (recommended) or pip

### Installation

```bash
# Create environment
conda create -n credit_override_env python=3.9
conda activate credit_override_env

# Install dependencies
pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn
pip install mlxtend
pip install pyarrow fastparquet   # for parquet I/O
pip install jupyterlab             # to run notebooks
```

### Key package versions used in this study

| Package | Version |
|---------|---------|
| numpy | 1.26.4 |
| pandas | 2.x |
| scikit-learn | 1.7.2 |
| statsmodels | 0.14+ |
| mlxtend | 0.23+ |
| matplotlib | 3.8+ |
| seaborn | 0.13+ |

> **Note on Firth regression:** The third-party `firthlogist` package (v0.5.0) has a compatibility issue with scikit-learn ≥ 1.6 (`_validate_data` API change) and fails on every call in this environment. This study uses a self-contained Firth (1993) bias-reduced logistic regression implementation (`src/firth_utils.py`) with no dependency on `firthlogist`, validated against standard MLE odds ratios (maximum discrepancy < 0.021 across all folds, consistent with the original paper's own validation check).

---

## Execution Order

### Stage 1 — Fold construction (run once)

```
NB01
```

NB01 constructs the outer stratified 5-fold split and, **for each of the 5 folds independently**, re-fits the system credit-grading model, re-assigns grades, re-simulates overrides, and extracts the upgrade cohort using only that fold's training firms. This single run produces all 5 folds' `fold_{f}/upgrade_cohort_fold{f}.parquet` files.

### Stage 2 — Factor discovery and validation (per fold)

```
NB02 → NB03 → NB03b → NB04 → NB05 → NB06 → NB07
```

Each notebook in this stage operates on ONE fold's outer-training partition (factor discovery) and, in NB07, validates the resulting factor library against that SAME fold's outer-holdout partition. This sequence was developed and validated in full on fold 0.

### Stage 3 — Cross-fold orchestration (folds 1–4, then aggregation)

```
NB08
```

NB08 wraps the validated NB02–NB07 logic into reusable functions (with all discovery-stage parameters fixed at the values established during fold 0 development — see Key Parameters below), executes them for folds 1–4, and aggregates all 5 folds' results into cross-fold stability and performance summaries, including the factor stability score, the 5-fold robust threshold search, and the evaluator quantitative-gap analysis.

Expected total runtime: approximately 1.5–3 hours for the full 5-fold pipeline, depending on hardware (NB03's Firth regression and NB04's Apriori mining are the most time-intensive steps, repeated 5×).

---

## Key Parameters (fixed across all folds)

Several parameters were established during fold 0 development, based on diagnostics not anticipated in the original single-sample analysis, and then held fixed across folds 1–4 for methodological consistency:

| Parameter | Value | Rationale |
|---|---|---|
| VIF threshold (NB03) | **5** (revised from 10) | Threshold 10 retained 22–27 variables per fold and produced Apriori rule-count inflation (thousands of near-duplicate rules); threshold 5 reduces this while remaining a standard multicollinearity cutoff |
| Exact-duplicate correlation threshold (NB03) | 0.999 | Removes variables that are algebraically identical under different names (e.g., `ebit_to_assets` / `Attr14` / `Attr18`) before VIF screening |
| Near-duplicate correlation threshold (NB03b) | **0.90** (revised from the original paper's 0.98) | A stricter cutoff than the single-sample analysis, to compensate for a larger starting Layer 1 pool per fold |
| Binned-label agreement threshold for group formation (NB04) | **0.75** (new) | Groups variables whose *discretised* (tertile) distress labels agree on ≥75% of risky-cohort firms — a quantity not captured by raw-value correlation, which was found (via diagnostic testing) to under-detect the redundancy actually driving Apriori's rule-count inflation |
| Group-redundancy rule filter (NB04) | New filter | Excludes association rules where 2+ items across the *combined* antecedent and consequent belong to the same binned-label-agreement group, preventing near-tautological rules (e.g., two proxies for the same underlying financial factor) from inflating the Layer 2 library |
| Layer 1 FDR threshold (NB06) | 0.05 (unchanged from original) | |
| Layer 2 confidence / lift thresholds (NB06) | 0.75 / 2.0× (unchanged from original) | |

---

## Notebook Descriptions

### NB01 — Data Preparation and Outer 5-Fold Construction
Loads the five-year `.arff` files and applies standard cleaning (1st/99th percentile clipping, median imputation, computed on the full sample). Constructs a stratified outer 5-fold split (stratified on default × year_horizon). For each fold, independently: fits the system logistic regression model (C=0.1) using only that fold's training firms, assigns quantile-based grades (AAA–CCC) using training-derived thresholds, simulates the Base scenario override distribution, and extracts the `upgrade_cohort` with `risky_upgrade` / `safe_upgrade` group labels and an `outer_split` (train/holdout) flag.

**Cross-fold system model stability:** outer-training AUC = 0.7904 ± 0.0035; outer-holdout AUC = 0.7818 ± 0.0125 (true out-of-sample estimate).

**Outputs:** `data/processed/fold_{0..4}/polish_master_fold{f}.parquet`, `upgrade_cohort_fold{f}.parquet`, `fold_assignment_summary.csv`

---

### NB02 — Exploratory Data Analysis (per fold, training partition only)
Applies Mann-Whitney U tests (two-sided) to all 64 financial variables comparing `risky_upgrade` vs `safe_upgrade`, using only the fold's outer-training firms.

**Outputs:** `results/tables/NB02_02_mannwhitney_results_fold{f}.csv`

---

### NB03 — Univariate Firth Logistic Regression (per fold)
Applies a self-contained Firth (1993) bias-reduced logistic regression (see `src/firth_utils.py`) to all Mann-Whitney-significant variables. FDR correction (Benjamini-Hochberg) applied across all tests. Firth odds ratios are validated against standard MLE odds ratios (maximum discrepancy < 0.021 across all folds). Exact near-duplicate variables (|r| > 0.999) are removed prior to VIF screening (threshold = 5, revised from the original paper's 10 — see Key Parameters above).

**Outputs:** `results/tables/NB03_01_firth_results_fold{f}.csv`, `NB03_02_top_candidates_fold{f}.csv`

---

### NB03b — Data Dictionary Check (per fold)
Maps Attr-coded variable names to their UCI definitions. Applies a union-find near-duplicate check (|r| > 0.90, revised from the original paper's 0.98 — see Key Parameters above), retaining the variable with the strongest univariate signal within each duplicate group.

**Outputs:** `results/tables/NB03b_variable_map_fold{f}.csv`

---

### NB04 — Association Rule Mining (per fold)
Discretises candidate variables into tertile bins (LOW/MID/HIGH), computed on the fold's training partition. Applies Apriori to `risky_upgrade` and `safe_upgrade` cohorts separately. Groups variables by binned-label agreement (threshold = 0.75) to identify latent financial-factor families, and filters out association rules where 2+ items across the antecedent and consequent belong to the same group (see Key Parameters above). Filters remaining risky-specific rules: support ≥ 0.20, confidence ≥ 0.60, lift ≥ 2.0×, conf_diff ≥ 0.10, antecedent max length = 3.

**Outputs:** `results/tables/NB04_01_rules_risky_fold{f}.csv`, `NB04_02_rules_summary_fold{f}.csv`, `NB04_04_variable_groups_fold{f}.csv`, `NB04_05_signature_diversity_fold{f}.csv`

---

### NB05 — K-Means Clustering (per fold)
Clusters each fold's `risky_upgrade` firms using K-Means. Optimal K selected via combined Silhouette + Davies-Bouldin score with minimum cluster size constraint (≥ 5% of sample), falling back to Elbow K when the constraint cannot be satisfied.

**Key result:** cluster structure was the least stable component of the pipeline across folds — optimal K ranged from 3 to 5, with Silhouette coefficients ranging from 0.137 to 0.371. Two of five folds required the Elbow-K fallback. This instability is consistent with the original paper's own characterisation of its clustering result as reflecting overlapping rather than sharply bounded financial distress profiles.

**Outputs:** `results/tables/NB05_01_cluster_profiles_fold{f}.csv`, `NB05_02_cluster_assignments_fold{f}.csv`, `NB05_03_k_selection_fold{f}.csv`

---

### NB06 — Warning Factor Library (per fold)
Constructs the two-layer factor library for the fold:
- **Layer 1**: all FDR-significant, exact-duplicate-free variables, trigger = distress tertile (threshold stored explicitly for out-of-fold reuse)
- **Layer 2**: group-filtered risky-specific rules with confidence ≥ 0.75 and lift ≥ 2.0×, deduplicated by unique antecedent, with **per-condition tertile thresholds saved as explicit columns** (`cond1_p33`, `cond1_p67`, …) so NB07 can apply the exact training-derived thresholds to held-out firms without recomputing them.

**Outputs:** `NB06_L1_single_factors_fold{f}.csv`, `NB06_L2_composite_factors_fold{f}.csv`, `NB06_full_factor_library_fold{f}.csv`

---

### NB07 — Out-of-Fold Retrospective Simulation (per fold)
Applies the fold's frozen, training-derived factor library to **both** the training and holdout partitions (never recomputing thresholds from holdout data), and answers five operational questions on each:

| Q | Question | Train vs holdout consistency |
|---|----------|-------------------------------|
| Q1 | Warning count vs default rate | Both partitions show a consistent monotonic relationship in every fold |
| Q2 | Which factors were most accurate, and do they generalise? | Mean train-holdout hit-rate gap per factor: +0.0061 ± 0.0060 (5-fold average) |
| Q3 | Coverage-precision tradeoff | Coverage and precision at threshold ≥ 1 closely track between train and holdout in every fold |
| Q4 | Where does the system work best? | CCC→B remains the strongest-performing transition in every fold (holdout F1 = 0.2787 ± 0.0130) |
| Q5 | Cost of ignoring warnings | 30+ vs no-warning default-rate ratio: 5.05× ± 2.24× (holdout, 5-fold average) |

**Outputs:** `NB07_01_firm_level_warnings_fold{f}_{TRAIN,HOLDOUT}.csv`, `NB07_Q1` through `NB07_Q5` tables (per fold, train and holdout columns side by side)

---

### NB08 — Cross-Fold Orchestration and Aggregation
Wraps the NB02–NB07 logic (validated on fold 0) into reusable functions using fixed parameters (see Key Parameters), executes them for folds 1–4, and produces:

- **Cross-fold raw and aggregate summaries** (`NB08_all_folds_raw.csv`, `NB08_5fold_aggregate_summary.csv`) — all 5 folds' key metrics, plus mean/SD/min/max across folds
- **Factor stability scores** (`NB08_factor_stability_scores.csv`) — how many of the 5 folds selected each financial variable as a Layer 1 factor and/or a Layer 2 rule component; 39 of 49 distinct variables (80%) were selected as Layer 1 factors in all 5 folds
- **5-fold robust threshold search** (`NB08_robust_threshold_search.csv`) — scans warning-count thresholds 1–50 across all folds' holdout partitions to identify the value maximising mean holdout F1 (optimal threshold = 27, mean F1 = 0.2062 ± 0.0149), rather than reporting performance only at a small number of pre-selected thresholds
- **Evaluator quantitative-gap analysis** — quantifies, across all 5 folds' holdout partitions, what share of firms that received an upgrade override and subsequently defaulted had already triggered 20+ (later-identified) warning factors at the time of override (59.8% on average, range 56.9%–66.0%) — evidence that the identified factors fill an area where no comparable quantitative check previously existed, rather than merely refining existing rules

---

## Reproducibility

All random states are fixed (`RANDOM_SEED = 42`). The outer 5-fold split uses `StratifiedKFold(shuffle=True, random_state=42)`. Each fold's override simulation uses a fold-specific but fixed random stream (`np.random.default_rng(42 + fold_id)`), so results are reproducible per fold without reusing identical draws across folds. Results are fully reproducible given the same input data and package versions.

Minor numerical differences may occur across operating systems due to floating-point handling in `scipy` and `sklearn`.

---

## Citation

> [Author information withheld for blind review]. (Anonymous Year). Detecting hidden evaluator judgment patterns in credit rating override decisions: A two-layer early-warning factor library with nested cross-fold validation. *[Anonymous Journal]*. [Anonymous DOI].

Upon acceptance, this placeholder will be replaced with full citation details.

---

## Related Work

This repository is the companion codebase to Study 1:

> [Author information withheld for blind review]. (Anonymous Year). Detecting default risk in evaluator upgrade overrides: A machine learning-based early warning framework for corporate credit rating. *Expert Systems with Applications*. [Anonymous DOI].

Study 1 replication code is available at: [link withheld for blind review]

---

## License

Code released under the MIT License. Data (Polish Companies Bankruptcy dataset) is subject to UCI Machine Learning Repository terms of use.

---

## References

Zięba, M., Tomczak, S. K., & Tomczak, J. M. (2016). Ensemble boosted trees with synthetic features generation in application to bankruptcy prediction. *Expert Systems with Applications*, 58, 93–101.

Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27–38.

Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.

Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 1(2), 224–227.

Basel Committee on Banking Supervision (2006). *International convergence of capital measurement and capital standards: A revised framework*. Bank for International Settlements.
