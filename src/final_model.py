# ================================================================
# ScamShield India
# FINAL DEPLOYABLE PHISHING URL MODEL
# ================================================================

import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.calibration import CalibratedClassifierCV

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ================================================================
# CONFIGURATION
# ================================================================

DATA_PATH = Path(
    "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
)

OUTPUT_DIR = Path(
    "outputs/final_model"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42


# ================================================================
# LOAD DATA
# ================================================================

print("=" * 75)
print("SCAMSHIELD FINAL MODEL")
print("=" * 75)

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH
)

# Remove duplicate URLs
df = df.drop_duplicates(
    subset=["URL"]
).copy()


print(
    f"Rows after duplicate removal: "
    f"{len(df):,}"
)


# ================================================================
# TARGET
# ================================================================

# Original PhiUSIIL labels:
#
# 0 = phishing
# 1 = legitimate
#
# Convert:
#
# 0 = legitimate
# 1 = phishing

df["phishing_label"] = (
    df["label"] == 0
).astype(int)


# ================================================================
# DOMAIN
# ================================================================

df["domain_group"] = (
    df["Domain"]
    .fillna("")
    .astype(str)
    .str.lower()
)


# ================================================================
# DOMAIN-DISJOINT SPLIT
# ================================================================

print(
    "\n" + "=" * 75
)

print(
    "CREATING DOMAIN-DISJOINT HOLDOUT"
)

print(
    "=" * 75
)


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)


train_idx, test_idx = next(
    splitter.split(
        df,
        df["phishing_label"],
        groups=df["domain_group"]
    )
)


train_df = df.iloc[
    train_idx
].copy()

test_df = df.iloc[
    test_idx
].copy()


print(
    f"Training rows : "
    f"{len(train_df):,}"
)

print(
    f"Testing rows  : "
    f"{len(test_df):,}"
)


# ================================================================
# VERIFY DOMAIN SEPARATION
# ================================================================

train_domains = set(
    train_df["domain_group"]
)

test_domains = set(
    test_df["domain_group"]
)


overlap = (
    train_domains &
    test_domains
)


print(
    f"Domain overlap: "
    f"{len(overlap)}"
)


if overlap:

    raise RuntimeError(
        "Domain leakage detected."
    )


print(
    "✓ Zero domain overlap confirmed."
)


# ================================================================
# DATA
# ================================================================

X_train = (
    train_df["URL"]
    .astype(str)
)

X_test = (
    test_df["URL"]
    .astype(str)
)

y_train = (
    train_df["phishing_label"]
)

y_test = (
    test_df["phishing_label"]
)


# ================================================================
# FINAL MODEL
# ================================================================

print(
    "\n" + "=" * 75
)

print(
    "TRAINING CALIBRATED CHARACTER TF-IDF + LINEAR SVM"
)

print(
    "=" * 75
)


base_svm = Pipeline([

    (
        "tfidf",

        TfidfVectorizer(

            analyzer="char",

            ngram_range=(3, 5),

            min_df=2,

            max_features=80000,

            sublinear_tf=True,

            lowercase=True
        )
    ),

    (
        "svm",

        LinearSVC(

            C=1.0,

            dual="auto",

            max_iter=5000,

            random_state=RANDOM_STATE
        )
    )
])


# Calibrate the SVM so we can obtain
# usable probability estimates.

model = CalibratedClassifierCV(
    estimator=base_svm,
    method="sigmoid",
    cv=3,
    n_jobs=-1
)


print(
    "\nTraining..."
)

model.fit(
    X_train,
    y_train
)

print(
    "✓ Training complete."
)


# ================================================================
# PREDICTIONS
# ================================================================

print(
    "\nGenerating predictions..."
)


predictions = model.predict(
    X_test
)


probabilities = (
    model.predict_proba(
        X_test
    )[:, 1]
)


# ================================================================
# METRICS
# ================================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)


print(
    "\n" + "=" * 75
)

print(
    "FINAL MODEL PERFORMANCE"
)

print(
    "=" * 75
)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print(
    f"PR-AUC    : {pr_auc:.4f}"
)


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print(
    "\n" + "=" * 75
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=" * 75
)

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Legitimate",
            "Phishing"
        ],
        digits=4
    )
)


# ================================================================
# CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_test,
    predictions
)


print(
    "\nConfusion Matrix:"
)

print(cm)


# ================================================================
# SAVE MODEL
# ================================================================

MODEL_PATH = (
    OUTPUT_DIR /
    "scamshield_phishing_model.joblib"
)


joblib.dump(
    model,
    MODEL_PATH
)


print(
    "\n" + "=" * 75
)

print(
    "MODEL SAVED"
)

print(
    "=" * 75
)

print(
    MODEL_PATH
)


# ================================================================
# SAVE MODEL METADATA
# ================================================================

metadata = pd.DataFrame({

    "metric": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "training_rows",
        "testing_rows"
    ],

    "value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc,
        len(X_train),
        len(X_test)
    ]
})


metadata.to_csv(
    OUTPUT_DIR /
    "model_metadata.csv",
    index=False
)


# ================================================================
# SAVE TEST PREDICTIONS
# ================================================================

prediction_df = pd.DataFrame({

    "URL": X_test.values,

    "Actual": y_test.values,

    "Predicted": predictions,

    "Phishing_Probability": probabilities

})


prediction_df.to_csv(
    OUTPUT_DIR /
    "test_predictions.csv",
    index=False
)


# ================================================================
# FINAL
# ================================================================

print(
    "\n" + "=" * 75
)

print(
    "FINAL SCAMSHIELD MODEL READY"
)

print(
    "=" * 75
)

print(
    "\nThe model now supports:"
)

print(
    "✓ Raw URL input"
)

print(
    "✓ Character-level feature learning"
)

print(
    "✓ Domain-disjoint evaluation"
)

print(
    "✓ Calibrated phishing probability"
)

print(
    "✓ Future risk-score generation"
)

print(
    "\nNext step:"
)

print(
    "Build predict_url.py"
)