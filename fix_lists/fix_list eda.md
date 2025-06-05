Below is a **plain-markdown** “red-flag” review of the **EDA notebook (`eda_full.txt`)**—no HTML tables or special extensions—so you can paste it straight into Claude Sonnet 4 (or any LLM) for further processing.

---

# Red-Flag Review of the EDA Notebook

## 1 Data Loading & Initial Assumptions

### 1.1 “No Missing Values” Stated Without Proof

* **Issue** – The markdown summary says *“Data Quality: Complete dataset (no missing values)”* but the notebook never calls `isnull().sum()` or similar.&#x20;
* **Impact** – Undetected NaNs would break later preprocessing or bias statistics.
* **Fix** – Run an explicit missing-value audit and display counts for every column.

### 1.2 Hard-Fail on Config Keys

* **Issue** – If `continuous_features` or `categorical_features` is absent in `config.toml`, the code raises `ValueError` and halts the notebook.&#x20;
* **Impact** – Fragile to minor config changes; hampers reproducibility.
* **Fix** – Provide sensible fall-backs (e.g., infer types from `df.dtypes`) and warn instead of crash.

### 1.3 Silent Path Dependence

* **Issue** – `tools.load_dataset('dataset_raw.csv')` relies on the current working directory but never sets it.&#x20;
* **Impact** – Running the notebook from a different folder fails silently or loads the wrong file.
* **Fix** – Resolve paths relative to the notebook location (`Path(__file__).parent`) or the project root.

---

## 2 Feature Lists & Type Handling

### 2.1 Column Normalisation Duplicates Work

* **Issue** – Columns are lower-cased and snake-cased *after* earlier steps already referenced the original names.&#x20;
* **Impact** – Later cells that forget the rename will break; easy source of bugs.
* **Fix** – Standardise column names immediately after reading the CSV and use that canonical form everywhere.

### 2.2 Manual Spell-Fix for “nacionality”

* **Issue** – The rename logic only fires if the *misspelling* is present; downstream code assumes the corrected name.&#x20;
* **Impact** – If a future data pull already contains `nationality`, the list `categorical_features` still holds the wrong token, raising a `ValueError`.
* **Fix** – Clean the config once; don’t patch at runtime. Better: call `.str.normalize()` on all headers.

---

## 3 Target Transformation

### 3.1 Outcome Leakage in “Continuation vs Withdrawal”

* **Issue** – Students labelled *“Continuation”* include both *Enrolled* **and** *Graduate*. Graduation status is only known **years** after the prediction point.
* **Impact** – Any model trained later on this binary target will have future information, inflating scores.
* **Fix** – Exclude *Graduate* from the positive class when evaluating early-warning systems, or censor features that occur after the first year.

### 3.2 Class-Ratio Claim Mis-calculated

* **Issue** – Markdown table says binary split is **3,003 : 1,421 (67.9 : 32.1)** but `value_counts()` shows 3,003/4,424 = 67.9 % indeed—good—but the code never verifies after renaming.
* **Impact** – If upstream counts change, the text becomes stale.
* **Fix** – Generate those numbers dynamically inside the markdown (e.g., with `jupyterlab-execute-time` or literate programming).

---

## 4 Continuous-Feature Analysis

### 4.1 Heat-Map Includes Highly Redundant Pairs

* **Issue** – All 18 continuous features enter the correlation heat-map even though four pairs exceed 0.9 correlation.&#x20;
* **Impact** – Visual clutter hides actionable insights; risks normalising the redundancy.
* **Fix** – Filter the matrix to show only |ρ| > 0.3 or drop one feature from each near-duplicate pair.

### 4.2 VIF Mis-interpretation

* **Issue** – Flags any VIF > 5 as “problematic”, but then labels a VIF of 4.98 as *not* problematic because it simply falls below the threshold.&#x20;
* **Impact** – Arbitrary cliff effects; ignores domain context.
* **Fix** – Plot VIF distribution and discuss diminishing returns instead of binary good/bad.

---

## 5 Categorical-Feature Analysis

### 5.1 Multiple-Testing Not Controlled

* **Issue** – Performs 18 chi-square tests but does not adjust p-values (Bonferroni, Benjamini-Hochberg, etc.).&#x20;
* **Impact** – High likelihood of false positives, especially with large N.
* **Fix** – Report adjusted p-values or flag features that survive FDR ≤ 0.05.

### 5.2 LabelEncoder for Mutual Information

* **Issue** – Encodes *nominal* categories as arbitrary integers before `mutual_info_classif`.&#x20;
* **Impact** – Imposes fake ordinality; MI estimate becomes metric-dependent.
* **Fix** – Use `OrdinalEncoder(handle_unknown='use_encoded_value', …)` with random shuffling or switch to a hashing trick; better, compute target encoding on the fly.

### 5.3 Severe Cardinality Ignored in One-Hot Plan

* **Issue** – Parents’ occupations (46 categories) will explode to 46 dummies; notebook merely *notes* the curse of dimensionality.&#x20;
* **Impact** – k-NN distance dominated by sparsely populated axes; logistic regression over-parametrised.
* **Fix** – Group rare labels, try target encoding, or drop after feature-importance screening.

---

## 6 Outlier Analysis

### 6.1 IQR Treats Legitimate Zeros as “Outliers”

* **Issue** – Grades of 0 (non-attendance) are flagged as outliers though they are business-critical signals.&#x20;
* **Impact** – Down-stream removal or winsorisation would destroy signal.
* **Fix** – Tag zeros with a boolean **indicator variable** and keep raw values.

### 6.2 No Robust Scaling Demonstrated

* **Issue** – Recommendations mention “robust scaling”, but the notebook never computes or stores a scaler.
* **Impact** – Analysts may forget by the time modelling starts.
* **Fix** – Create a `RobustScaler` object now and pickle it with the dataset version.

---

## 7 Narrative vs Code Drift

* Markdown conclusions (e.g., “economic indicators show **very clean data** with <1 % outliers”) are not backed by numeric checks—`outlier_summary` lists GDP outlier % but the text ignores it.&#x20;
* Some recommendations (PCA, regularisation) appear before any model is trained; risk of confirmation bias.

---

## 8 Code Quality & Reproducibility

| Concern                                                                                                         | Impact                                       | Suggested Remedy                                |
| --------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ----------------------------------------------- |
| **Multiple global variables** (`continuous_features`, `categorical_features`, `df_dataset`) mutate across cells | Hidden side-effects in later notebooks       | Encapsulate in functions or a lightweight class |
| Reliance on `rich.print` for status messages                                                                    | Logs vanish in plain consoles / CI pipelines | Use Python logging with configurable handlers   |
| Heavy use of `plt.show()` inside function loops                                                                 | Jupyter-only; scripts will block             | Return the `Figure` and let caller decide       |

---

## Highest-Priority Fixes

1. **Verify data integrity** – run a full missing-value and duplicate-row audit.
2. **Resolve target leakage** – decide whether “Graduate” belongs in the positive class given the intended prediction horizon.
3. **Control for multiple tests** in chi-square / MI statistics to avoid false discoveries.
4. **Pre-empt dimensionality blow-ups** by grouping or encoding high-cardinality categories now rather than later.

Addressing these points will make the exploratory findings trustworthy and prevent accidental leakage or over-fitting in the subsequent modelling notebooks.

---

*(End of markdown)*
