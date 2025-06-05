Below is a **plain-markdown** version of the review—no HTML, no exotic extensions—so you can copy-paste it directly.
I replaced most tables with bullet lists (many LLMs, including Claude Sonnet 4, find nested lists easier to parse than pipe-tables).

---

# Red-Flag Review of the Notebook

## 1  Data Preparation & Feature Engineering

### 1.1 Target-Aware Encoding Before Train/Test Split

* **Issue** – Encodes each category with its *overall* withdrawal rate *prior* to splitting.
* **Impact** – Data leakage: the model “sees” test-fold statistics.
* **Fix** – Split first; compute encodings only on the training fold, then *transform* the test fold.

### 1.2 Scaling One-Hot Dummies

* **Issue** – `StandardScaler` applied to 0/1 dummy variables.
* **Impact** – Uninterpretable coefficients and unnecessary variance.
* **Fix** – Leave dummies unscaled (or set `with_mean=False` for sparse matrices).

### 1.3 Leaky Second-Semester Features

* **Issue** – Column `curricular_units_2nd_sem_without_evaluations` kept, although its value is known only *after* the prediction point.
* **Impact** – Perfect signal for early drop-outs → leakage.
* **Fix** – Drop any feature written after the prediction moment.

### 1.4 Unregularised Frequency Encoding

* **Issue** – Rare categories get extreme 0/1 codes.
* **Impact** – High variance, over-fitting.
* **Fix** – Use additive smoothing / leave-one-out or clip to \[ε, 1 − ε].

---

## 2  Model Definition & Training

### 2.1 Wrong Class Weight

* **Issue** – Loss weights the *continuation* class, not the minority withdrawal class.
* **Fix** – Swap label meanings in the loss **or** output the withdrawal logit and weight it.

### 2.2 “Specificity” Mis-labelled

* **Issue** – The code calls minority-class *recall* “specificity”.
* **Fix** – Rename or recompute.

  * Specificity = TN / (TN + FP)
  * Withdrawal recall = TP / (TP + FN)

### 2.3 Casting Labels to Float

* **Issue** – Labels changed to `float32` before `GridSearchCV`.
* **Impact** – Undefined behaviour for some scorers.
* **Fix** – Keep labels as integer.

### 2.4 Manual `[1 − p, p]` Stacking

* **Issue** – Creates a 2-column tensor each pass.
* **Fix** – Return a single logit; use `BCEWithLogitsLoss`.

### 2.5 Unfixed Random Seeds Across Folds

* **Impact** – Scores vary run-to-run.
* **Fix** – Set a single `random_state` per grid-search run.

### 2.6 Threshold Tuning on Test Set

* **Impact** – Peeking; optimistic metrics.
* **Fix** – Tune threshold inside CV or on a separate validation set.

### 2.7 Mixed Metrics Without Cost Model

* **Issue** – Business and statistical metrics mixed with equal weight.
* **Fix** – Pick one primary metric or define an explicit cost matrix.

---

## 3  Evaluation & Reporting

* **3.1 Confusion-Matrix Labels** – TN mislabeled as “correctly identified withdrawals”.
* **3.2 Coefficient Plot** – Uses scaled continuous vars vs unscaled dummies ⇒ magnitudes incomparable.
* **3.3 Model Persistence** – `pos_weight` not stored with the model; models may reload with default 1.0.

---

## 4  Code Quality & Reproducibility

* Analysis, production code, and long prints are interleaved—hard to unit-test.
* Multiple conflicting config sources (`config.toml`, hard-coded defaults, ad-hoc params).
* Paths assume `os.getcwd()`; running from another directory breaks them.

---

## Highest-Priority Fixes

1. **Move all target-dependent transforms into a `Pipeline` / `ColumnTransformer` *after* the train/test split.**
2. **Apply class weighting to the correct label and fix the specificity vs recall naming.**
3. **Remove every instance of test-set peeking (threshold tuning, encodings, etc.).**
4. **Re-run evaluation—expect lower scores—then iterate with regularised encoders and calibrated probabilities.**

---

*(End of markdown)*
