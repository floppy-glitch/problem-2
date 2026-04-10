"""
Problem Set 02: Bank Marketing – Term Deposit Subscription Prediction
Model: Logistic Regression
Dataset: Bank Marketing Data Set (17 attributes, binary target: y = yes/no)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
# The dataset uses semicolon (;) as delimiter — common for UCI Bank Marketing sets
df = pd.read_csv("bank.csv", sep=";")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn dtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# ─────────────────────────────────────────────
# 2. Exploratory Data Analysis
# ─────────────────────────────────────────────
print("\nTarget distribution:\n", df["y"].value_counts())
print(f"Subscription rate: {(df['y'] == 'yes').mean()*100:.2f}%")

# Separate feature types
categorical_cols = df.select_dtypes(include="object").columns.tolist()
categorical_cols.remove("y")           # target column
numerical_cols   = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nCategorical features:", categorical_cols)
print("Numerical features  :", numerical_cols)

# ── Plot: target balance ──
plt.figure(figsize=(5, 4))
df["y"].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"], edgecolor="black")
plt.title("Target Class Distribution")
plt.xlabel("Subscribed (y)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("target_distribution.png", dpi=150)
plt.show()

# ── Plot: numerical distributions ──
df[numerical_cols].hist(figsize=(14, 8), bins=25, edgecolor="black", color="#4C72B0")
plt.suptitle("Numerical Feature Distributions", fontsize=14)
plt.tight_layout()
plt.savefig("numerical_distributions.png", dpi=150)
plt.show()

# ── Plot: correlation heatmap ──
plt.figure(figsize=(10, 7))
corr = df[numerical_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap (Numerical Features)")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 3. Pre-processing
# ─────────────────────────────────────────────

# 3a. Encode target
df["y"] = (df["y"] == "yes").astype(int)   # yes → 1, no → 0

# 3b. One-Hot Encode categorical features
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"\nShape after encoding: {df_encoded.shape}")

# 3c. Split features / target
X = df_encoded.drop("y", axis=1)
y = df_encoded["y"]

print(f"\nFeatures: {X.shape[1]}  |  Samples: {X.shape[0]}")

# 3d. Train / Test split  (stratified to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}   Test: {X_test.shape}")

# ─────────────────────────────────────────────
# 4. Build Pipeline  (Scaler + Logistic Regression)
# ─────────────────────────────────────────────

# Compute class weights to handle imbalance
cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(cw))
print(f"\nClass weights: {cw_dict}")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        C=1.0,                    # inverse regularisation strength
        penalty="l2",             # Ridge regularisation
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",  # handles class imbalance automatically
        random_state=42,
    )),
])

# ─────────────────────────────────────────────
# 5. Cross-Validation
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_accuracy = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
cv_f1       = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
cv_auc      = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

print("\n── 5-Fold Cross-Validation (on Training Set) ──")
print(f"  Accuracy : {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
print(f"  F1 Score : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
print(f"  ROC-AUC  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# ─────────────────────────────────────────────
# 6. Train on Full Training Set & Evaluate
# ─────────────────────────────────────────────
pipeline.fit(X_train, y_train)

y_pred      = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)

print("\n── Test Set Performance ──")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

# ─────────────────────────────────────────────
# 7. Visualisations
# ─────────────────────────────────────────────

# ── Confusion Matrix ──
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix – Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ── ROC Curve ──
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.4f})", color="#4C72B0", lw=2)
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Term Deposit Subscription")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()

# ── Feature Importance (Coefficients) ──
lr_model   = pipeline.named_steps["lr"]
coef       = lr_model.coef_[0]
feat_names = X.columns.tolist()

coef_df = pd.DataFrame({"Feature": feat_names, "Coefficient": coef})
coef_df["Abs"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values("Abs", ascending=False).head(20)

plt.figure(figsize=(10, 7))
colors = ["#DD8452" if c > 0 else "#4C72B0" for c in coef_df["Coefficient"]]
plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="black")
plt.xlabel("Coefficient Value")
plt.title("Top 20 Feature Coefficients (Logistic Regression)")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("feature_coefficients.png", dpi=150)
plt.show()

print("\nAll plots saved.")
print("Done.")
