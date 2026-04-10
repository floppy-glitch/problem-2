# Problem Set 02 – Bank Marketing: Term Deposit Subscription Prediction

## Overview

This project applies **Logistic Regression** to predict whether a bank customer will subscribe to a **term deposit** (`y = yes / no`). The model is trained on the UCI Bank Marketing dataset, which contains 17 attributes covering customer demographics, account information, and details of previous marketing campaigns.

---

## Dataset

| Attribute | Description |
|-----------|-------------|
| `age` | Customer age |
| `job` | Type of employment |
| `marital` | Marital status |
| `education` | Education level |
| `default` | Has credit in default? |
| `balance` | Average yearly balance (€) |
| `housing` | Has housing loan? |
| `loan` | Has personal loan? |
| `contact` | Contact communication type |
| `day` / `month` | Last contact date |
| `duration` | Last contact duration (seconds) |
| `campaign` | Contacts performed in this campaign |
| `pdays` | Days since last contact from previous campaign |
| `previous` | Contacts before this campaign |
| `poutcome` | Outcome of previous campaign |
| **`y`** | **Target – subscribed to term deposit?** |

- **Class distribution**: Heavily imbalanced (~88% No, ~12% Yes)

---

## Approach & Methodology

### 1. Exploratory Data Analysis (EDA)

- Visualised target class distribution to confirm imbalance.
- Plotted histograms of all numerical features to understand spread and skewness.
- Generated a **correlation heatmap** to identify multicollinear features.
- Observed that `duration` (call duration) is the most correlated numerical feature with the target — longer calls correlate strongly with subscription.

### 2. Pre-processing

| Step | Action |
|------|--------|
| Missing values | None found in this dataset |
| Categorical encoding | **One-Hot Encoding** (`pd.get_dummies`, `drop_first=True` to avoid dummy trap) |
| Target encoding | `yes → 1`, `no → 0` |
| Feature scaling | `StandardScaler` inside the pipeline (applied only to training data; test data transformed using training statistics) |

Using a **scikit-learn Pipeline** ensures that the scaler is fit only on training data, completely preventing **data leakage**.

### 3. Handling Class Imbalance

Two strategies are applied together:

1. **`class_weight="balanced"`** in `LogisticRegression` — automatically adjusts the loss function to penalise misclassification of the minority class (subscribers) proportionally more.
2. **Stratified train/test split** — preserves the original class ratio in both sets.

### 4. Model: Logistic Regression

Logistic Regression models the log-odds of subscription:

```
log(p / (1 − p)) = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Key hyperparameters:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `C` | 1.0 | Moderate L2 regularisation (inverse strength) |
| `penalty` | `l2` | Ridge — shrinks coefficients, handles correlated features |
| `solver` | `lbfgs` | Efficient for small-to-medium datasets with L2 |
| `max_iter` | 1000 | Ensures convergence |
| `class_weight` | `balanced` | Addresses class imbalance |

### 5. Validation Strategy

**5-fold Stratified Cross-Validation** on the training set to get a robust performance estimate:

- Stratified folds preserve the class ratio in every fold.
- Metrics: Accuracy, F1, ROC-AUC.

### 6. Evaluation Metrics

| Metric | Why it matters here |
|--------|-------------------|
| Accuracy | General correctness |
| Precision | Of predicted subscribers, how many truly subscribed? |
| Recall | Of all actual subscribers, how many were caught? |
| F1 Score | Harmonic mean — useful under class imbalance |
| ROC-AUC | Measures separability across all thresholds |

---

## Key Findings

1. **Class imbalance** is the dominant challenge — the bank's no-subscription rate (~88%) means a naive model that always predicts "No" would appear 88% accurate. Balanced weighting corrects this.

2. **`duration`** (last call duration in seconds) is the strongest predictor of subscription, followed by `poutcome_success` (success of the previous campaign). These align with domain intuition: engaged customers stay on the call longer.

3. **Previous campaign outcome** features (`poutcome_success`, `pdays`) strongly influence predictions, suggesting that customers with positive prior interactions are far more likely to subscribe again.

4. **`contact_unknown`** (no known contact method) is negatively correlated with subscription — reaching customers through known channels increases conversion.

5. Logistic Regression is appropriate here because:
   - The relationship between features and log-odds is approximately linear for most banking predictors.
   - Coefficients are interpretable by business stakeholders.
   - L2 regularisation handles the mild multicollinearity among demographic features.

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas scikit-learn matplotlib seaborn numpy

# 2. Place the dataset file (bank.csv) in the same directory.
#    The file uses semicolons (;) as delimiters — do NOT convert to commas.

# 3. Run
python logistic_regression_bank.py
```

Outputs produced:
- `target_distribution.png` — bar chart of class balance
- `numerical_distributions.png` — histograms of numerical features
- `correlation_heatmap.png` — Pearson correlation matrix
- `confusion_matrix.png` — test-set confusion matrix
- `roc_curve.png` — ROC curve with AUC
- `feature_coefficients.png` — top 20 feature importances by coefficient magnitude

---

## File Structure

```
problem_set_02/
├── logistic_regression_bank.py   # Full pipeline: EDA → preprocessing → training → evaluation
└── README.md                     # This file
```

---

## Dependencies

```
pandas
scikit-learn >= 1.0
matplotlib
seaborn
numpy
```
