Below is a **plain-markdown** “red-flag” review of the *k-NN notebook*.
I kept the structure identical to the earlier logistic-regression critique so you can copy-paste it straight into Claude Sonnet 4 (or any LLM) with no rendering surprises.

---

# Red-Flag Review of the k-Nearest-Neighbours Notebook

## 1 Data Preparation & Feature Engineering

### 1.1 Target-Aware Encoding *before* the Train/Test Split

* **Issue** – `encode_categorical_withdrawal_rate()` replaces `course` and `application_mode` with their overall withdrawal rates *before* the split.
* **Impact** – Classic data leakage: the model “learns” statistics from the test fold.
* **Fix** – Fit the withdrawal-rate map on the **training** fold inside a `ColumnTransformer`, then apply `.transform()` to validation/test.

### 1.2 Scaler Fit on the *Entire* Dataset

* **Issue** – `scaler.fit_transform()` is run on **all rows** (cell #28), then the data are split.
* **Impact** – Test-set means and variances leak into training; neighbours in production will be measured against different scales.
* **Fix** – Fit the scaler *after* the split (or inside a `Pipeline`) and call only `.transform()` on hold-out data.

### 1.3 Unregularised Withdrawal-Rate Encoding

* **Issue** – Rare categories receive extreme 0/1 values.
* **Impact** – High-variance distance spikes; risk of over-weighting small cohorts.
* **Fix** – Apply additive smoothing (e.g. m-estimate) or a James-Stein shrinkage toward the global rate.

### 1.4 Binary Features Left Unscaled but Continuous Ones Min-Max’d

* **Issue** – Binary 0/1 dimensions now span the *same length* (1 unit) as min-max’d “continuous” features (also 0–1).
* **Impact** – Each binary switch counts as much as moving from the 5th to the 95th percentile of `admission_grade`.
* **Fix** – Consider **distance-weighted k-NN** or a metric-learning step, or scale binaries by √p (1 − p).

---

## 2 Model Definition & Training

### 2.1 Cross-Validation Finds *k = 9* but Model Trains with *k = 5*

* **Issue** – `find_optimal_k()` returns 9, but the subsequent cell overwrites it with the value from `config.toml` (5).
* **Impact** – Reported “optimal” hyper-parameter is silently ignored; results are irreproducible.
* **Fix** – Either honour the CV result or lock the config; never both.

### 2.2 Search Range Too Narrow

* **Issue** – `n_neighbors_range = [3, 10]` with step 1.
* **Impact** – Misses potentially better odd numbers above 10 (large-k often smooths minority-class noise).
* **Fix** – Extend to, say, 1–31 and include *distance weighting* and *Minkowski p* in the grid.

### 2.3 No `Pipeline` ⇒ Train/Test Mismatch Risk

* **Issue** – Encoding, scaling and model are separate steps.
* **Impact** – Easy to forget a transform when deploying; CV scores optimistic because the scaler is cross-contaminated.
* **Fix** – Wrap **all** preprocessing + `KNeighborsClassifier` in a single `Pipeline`.

### 2.4 Randomness Not Controlled

* **Issue** – `cross_val_score` uses K-Fold with shuffling = False by default; `train_test_split` has fixed `random_state=42`.
* **Impact** – CV scores depend on row order in the file.
* **Fix** – Use `StratifiedKFold(shuffle=True, random_state=42)`.

---

## 3 Evaluation & Reporting

### 3.1 “Specificity” Is Actually *Withdrawal Recall*

* **Issue** – With label 0 = withdrawn, `specificity = TN / (TN + FP)` measures recall of class 0, *not* specificity (which would refer to class 1).
* **Impact** – Metric names mislead stakeholders and confuse dashboarding.
* **Fix** – Either swap label coding (make withdrawals the *positive* class) or rename metric to “withdrawal recall”.

### 3.2 Confusion-Matrix Labels Swapped

* **Issue** – Heat-map annotates row 0/col 1 as “False Positive”; it is actually a **False Negative** when class 1 is positive.
* **Impact** – Wrong quadrant explanations; incorrect error counts in the printed interpretation.
* **Fix** – Re-map labels or call `ConfusionMatrixDisplay.from_predictions()` and let scikit-learn handle the axes.

### 3.3 AUC Uses Only Class-1 Probability but Business Focus Is Class 0

* **Impact** – AUC may look fine while withdrawal-recall is poor.
* **Fix** – Report both conventional AUC and the *inverted* AUC (1 − proba) or use *PR-AUC* for the withdrawal class.

### 3.4 “Prediction Confidence” Formula Is Arbitrary

* **Issue** – `abs(p − 0.5) + 0.5` maps 0.5→0.5 and 0/1→1, but k-NN probabilities are coarse (steps of 1/k).
* **Impact** – All misclassifications end up with ≥0.6 “confidence”, so the low-confidence analysis finds *zero* cases.
* **Fix** – Use raw distance ratio or the *margin* (difference between class-vote fractions).

---

## 4 Code Quality & Reproducibility

* Notebook again mixes utilities, exploratory plots and production code.
* Absolute paths depend on `os.getcwd()`; running from a different working directory breaks `load_dataset`.
* Multiple sources of truth (notebook cells vs `config.toml`) cause silent overrides.
* No saving of the fitted scaler/encoder, so live inference would measure distances in a different space.

---

## Highest-Priority Fixes

1. **Bundle encoding + scaling + k-NN into a scikit-learn `Pipeline`, and fit it *after* splitting.**
2. **Regularise target encodings and remove *all* test-set peeking (withdrawal-rate maps, scaler fit).**
3. **Let CV pick *k* and stick to it; expand the search grid to include distance weighting.**
4. **Correct the metric names / confusion-matrix labels so dashboards reflect the real error types.**

Addressing these four items will eliminate the hidden optimism currently inflating the 0.57 “specificity” figure and make the model’s withdrawal-detection claims defensible.

---

*(End of markdown)*
