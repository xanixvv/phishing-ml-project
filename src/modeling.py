# ================================================================
# ScamShield India
# Phishing URL ML Model Comparison
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


# ================================================================
# CONFIGURATION
# ================================================================

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/models")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# IMPORTANT LABEL DEFINITION
# ================================================================

# PhiUSIIL dataset:
#
# 0 = Phishing
# 1 = Legitimate
#
# For our project, PHISHING is the positive class.

PHISHING_LABEL = 0


# ================================================================
# LOAD DATA
# ================================================================

print("=" * 70)
print("LOADING PROCESSED DATA")
print("=" * 70)

X_train = pd.read_csv(
    DATA_DIR / "X_train.csv"
)

X_test = pd.read_csv(
    DATA_DIR / "X_test.csv"
)

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
).squeeze()

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).squeeze()


print(
    f"Training features : {X_train.shape}"
)

print(
    f"Testing features  : {X_test.shape}"
)

print(
    f"Training labels   : {y_train.shape}"
)

print(
    f"Testing labels    : {y_test.shape}"
)


# ================================================================
# CONVERT TARGET TO PHISHING-POSITIVE FORMAT
# ================================================================

y_train_phishing = (
    y_train == PHISHING_LABEL
).astype(int)

y_test_phishing = (
    y_test == PHISHING_LABEL
).astype(int)


print("\n" + "=" * 70)
print("TARGET ENCODING")
print("=" * 70)

print("Original labels:")
print(y_train.value_counts().sort_index())

print("\nOur ML evaluation labels:")
print("0 = Legitimate")
print("1 = Phishing")

print("\nTraining:")
print(
    y_train_phishing.value_counts()
    .sort_index()
)

print("\nTesting:")
print(
    y_test_phishing.value_counts()
    .sort_index()
)


# ================================================================
# NUMERICAL CLEANUP
# ================================================================

print("\n" + "=" * 70)
print("FEATURE CLEANUP")
print("=" * 70)

X_train = X_train.apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


# IMPORTANT:
# Use only training medians for both train and test.

train_medians = X_train.median()

X_train = X_train.fillna(
    train_medians
)

X_test = X_test.fillna(
    train_medians
)


print("Infinite values handled.")
print("Missing values handled.")
print("Training-only medians used for imputation.")


# ================================================================
# MODELS
# ================================================================

print("\n" + "=" * 70)
print("CREATING MODELS")
print("=" * 70)


models = {

    # ------------------------------------------------------------
    # 1. Logistic Regression
    # ------------------------------------------------------------

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]),


    # ------------------------------------------------------------
    # 2. Decision Tree
    # ------------------------------------------------------------

    "Decision Tree": DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=10,
        random_state=42
    ),


    # ------------------------------------------------------------
    # 3. Random Forest
    # ------------------------------------------------------------

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),


    # ------------------------------------------------------------
    # 4. Linear SVM
    # ------------------------------------------------------------

    "Linear SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LinearSVC(
                C=1.0,
                dual=False,
                max_iter=5000,
                random_state=42
            )
        )
    ])
}


print(
    f"Models to evaluate: {len(models)}"
)

for name in models:
    print(f"  ✓ {name}")


# ================================================================
# STORAGE
# ================================================================

results = []

trained_models = {}

roc_data = {}

pr_data = {}


# ================================================================
# TRAIN MODELS
# ================================================================

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)


for name, model in models.items():

    print("\n" + "-" * 70)
    print(f"Training: {name}")
    print("-" * 70)

    start_time = time()

    model.fit(
        X_train,
        y_train_phishing
    )

    elapsed = time() - start_time

    print(
        f"Training completed in {elapsed:.2f} seconds."
    )


    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # ------------------------------------------------------------
    # Prediction score
    # ------------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        # Probability of class 1,
        # which WE defined as phishing.
        phishing_probability = (
            model.predict_proba(X_test)[:, 1]
        )

    else:

        # LinearSVC returns higher scores for class 1.
        # Since our transformed target uses:
        #
        # 1 = phishing
        #
        # the decision function already points toward
        # phishing.
        phishing_probability = (
            model.decision_function(X_test)
        )


    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    accuracy = accuracy_score(
        y_test_phishing,
        y_pred
    )

    precision = precision_score(
        y_test_phishing,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test_phishing,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test_phishing,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test_phishing,
        phishing_probability
    )

    pr_auc = average_precision_score(
        y_test_phishing,
        phishing_probability
    )


    # ------------------------------------------------------------
    # Store results
    # ------------------------------------------------------------

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Training Time (s)": elapsed
    })

    trained_models[name] = model


    # ------------------------------------------------------------
    # Print metrics
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # ROC
    # ------------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test_phishing,
        phishing_probability
    )

    roc_data[name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc
    }


    # ------------------------------------------------------------
    # Precision-Recall
    # ------------------------------------------------------------

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_test_phishing,
            phishing_probability
        )
    )

    pr_data[name] = {
        "precision": precision_curve,
        "recall": recall_curve,
        "auc": pr_auc
    }


# ================================================================
# RESULTS TABLE
# ================================================================

results_df = pd.DataFrame(
    results
)

# F1 is our primary ranking metric.
results_df = results_df.sort_values(
    "F1 Score",
    ascending=False
).reset_index(
    drop=True
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ================================================================
# SAVE RESULTS
# ================================================================

results_path = (
    OUTPUT_DIR /
    "model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(
    f"\nSaved: {results_path}"
)


# ================================================================
# BEST MODEL
# ================================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)

best_model = (
    trained_models[best_model_name]
)


print("\n" + "=" * 70)
print("CURRENT BEST MODEL")
print("=" * 70)

print(
    f"Model: {best_model_name}"
)

for metric in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC",
    "PR-AUC"
]:

    print(
        f"{metric}: "
        f"{results_df.iloc[0][metric]:.4f}"
    )


# ================================================================
# MODEL COMPARISON PLOTS
# ================================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC",
    "PR-AUC"
]


for metric in metrics:

    sorted_df = results_df.sort_values(
        metric,
        ascending=True
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        sorted_df["Model"],
        sorted_df[metric]
    )

    plt.xlim(
        0,
        1
    )

    plt.xlabel(
        metric
    )

    plt.ylabel(
        "Model"
    )

    plt.title(
        f"Model Comparison - {metric}"
    )

    for i, value in enumerate(
        sorted_df[metric]
    ):

        plt.text(
            value + 0.01,
            i,
            f"{value:.3f}",
            va="center"
        )

    plt.tight_layout()

    filename = (
        "comparison_"
        + metric.lower().replace(
            " ",
            "_"
        )
        + ".png"
    )

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# ROC CURVES
# ================================================================

plt.figure(
    figsize=(10, 8)
)

for name, data in roc_data.items():

    plt.plot(
        data["fpr"],
        data["tpr"],
        label=(
            f"{name} "
            f"(AUC = {data['auc']:.3f})"
        )
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves for Phishing Detection"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# PRECISION-RECALL CURVES
# ================================================================

plt.figure(
    figsize=(10, 8)
)

for name, data in pr_data.items():

    plt.plot(
        data["recall"],
        data["precision"],
        label=(
            f"{name} "
            f"(AP = {data['auc']:.3f})"
        )
    )


plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curves for Phishing Detection"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "precision_recall_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# CONFUSION MATRICES
# ================================================================

print("\n" + "=" * 70)
print("CONFUSION MATRICES")
print("=" * 70)


for name, model in trained_models.items():

    y_pred = model.predict(
        X_test
    )

    cm = confusion_matrix(
        y_test_phishing,
        y_pred
    )

    print(f"\n{name}")
    print(cm)


    plt.figure(
        figsize=(7, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Legitimate",
            "Phishing"
        ],
        yticklabels=[
            "Legitimate",
            "Phishing"
        ]
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.tight_layout()

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    plt.savefig(
        OUTPUT_DIR /
        f"confusion_matrix_{safe_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# BEST MODEL CLASSIFICATION REPORT
# ================================================================

best_predictions = (
    best_model.predict(X_test)
)

print("\n" + "=" * 70)
print("BEST MODEL CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test_phishing,
        best_predictions,
        target_names=[
            "Legitimate",
            "Phishing"
        ],
        digits=4
    )
)


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)


if hasattr(
    best_model,
    "named_steps"
):

    final_estimator = (
        best_model.named_steps["model"]
    )

else:

    final_estimator = best_model


if hasattr(
    final_estimator,
    "feature_importances_"
):

    importances = (
        final_estimator.feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importances
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    importance_df.to_csv(
        OUTPUT_DIR /
        "feature_importance.csv",
        index=False
    )

    top_features = (
        importance_df
        .head(20)
        .sort_values(
            "Importance"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"Top Features - {best_model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "top_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


else:

    print(
        "Best model does not provide "
        "tree-based feature importance."
    )


# ================================================================
# LOGISTIC REGRESSION COEFFICIENTS
# ================================================================

logistic_model = (
    trained_models["Logistic Regression"]
)

logistic_estimator = (
    logistic_model.named_steps["model"]
)

coefficients = (
    logistic_estimator.coef_[0]
)

coefficient_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": coefficients,
    "AbsoluteCoefficient": np.abs(coefficients)
})

coefficient_df = (
    coefficient_df
    .sort_values(
        "AbsoluteCoefficient",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\nTop Logistic Regression features:")

print(
    coefficient_df.head(20).to_string(
        index=False
    )
)

coefficient_df.to_csv(
    OUTPUT_DIR /
    "logistic_coefficients.csv",
    index=False
)


# ================================================================
# FINAL MESSAGE
# ================================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nOutputs saved to:\n"
    f"{OUTPUT_DIR.resolve()}"
)

print("\nModels evaluated:")

for name in trained_models:
    print(f"  ✓ {name}")

print(
    "\nNext step: leakage and robustness analysis."
)