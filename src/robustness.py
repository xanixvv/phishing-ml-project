# ================================================================
# ScamShield India
# Robustness + Leakage Audit
# PhiUSIIL Phishing URL Dataset
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from time import time

from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

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

DATA_PATH = Path(
    "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
)

OUTPUT_DIR = Path(
    "outputs/robustness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42


# ================================================================
# LABEL DEFINITION
# ================================================================

# PhiUSIIL:
#
# 0 = Phishing
# 1 = Legitimate
#
# We convert this to:
#
# 0 = Legitimate
# 1 = Phishing

PHISHING_LABEL = 0


# ================================================================
# LOAD DATA
# ================================================================

print("=" * 75)
print("SCAMSHIELD ROBUSTNESS + LEAKAGE AUDIT")
print("=" * 75)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {df.shape[1]}"
)


# ================================================================
# BASIC DATASET AUDIT
# ================================================================

print("\n" + "=" * 75)
print("DATASET AUDIT")
print("=" * 75)

print(
    f"Unique URLs    : {df['URL'].nunique():,}"
)

print(
    f"Duplicate URLs : "
    f"{len(df) - df['URL'].nunique():,}"
)

print(
    f"Unique domains : "
    f"{df['Domain'].nunique():,}"
)

domain_label_counts = (
    df.groupby("Domain")["label"]
    .nunique()
)

mixed_domains = (
    domain_label_counts > 1
).sum()

print(
    f"Domains appearing in both classes : "
    f"{mixed_domains:,}"
)


# ================================================================
# URLSIMILARITY AUDIT
# ================================================================

print("\n" + "=" * 75)
print("URLSIMILARITYINDEX AUDIT")
print("=" * 75)

similarity_by_class = (
    df.groupby("label")["URLSimilarityIndex"]
    .agg([
        "count",
        "mean",
        "std",
        "min",
        "median",
        "max"
    ])
)

print(similarity_by_class)


print("\nPercentage of each class with URLSimilarityIndex = 100:")

similarity_100 = (
    df.groupby("label")["URLSimilarityIndex"]
    .apply(
        lambda x: (x == 100).mean() * 100
    )
)

print(
    similarity_100.round(2)
)


# ================================================================
# TARGET CONVERSION
# ================================================================

df["phishing_label"] = (
    df["label"] == PHISHING_LABEL
).astype(int)


# ================================================================
# FEATURE GROUPS
# ================================================================

# ------------------------------------------------
# URL-ONLY FEATURES
# ------------------------------------------------
#
# These can ultimately be calculated from the URL
# itself, making them suitable for our future
# web application.

URL_ONLY_FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLD",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS"
]


# ------------------------------------------------
# WEB-ENHANCED FEATURES
# ------------------------------------------------
#
# These include URL + webpage characteristics.
#
# We deliberately remove URLSimilarityIndex because
# its class separation is extremely strong and would
# make the benchmark less informative for our
# generalization study.
#
# TLDLegitimateProb is also excluded because it is
# explicitly a prior probability of TLD legitimacy.

DROP_FOR_WEB_ROBUSTNESS = [
    "label",
    "phishing_label",
    "FILENAME",
    "URL",
    "Domain",
    "Title",
    "URLSimilarityIndex",
    "TLDLegitimateProb"
]


WEB_FEATURES = [
    column
    for column in df.columns
    if column not in DROP_FOR_WEB_ROBUSTNESS
]


print("\n" + "=" * 75)
print("FEATURE GROUPS")
print("=" * 75)

print(
    f"URL-only features      : {len(URL_ONLY_FEATURES)}"
)

print(
    f"Web-enhanced features  : {len(WEB_FEATURES)}"
)


# ================================================================
# VERIFY FEATURE LISTS
# ================================================================

missing_url_features = [
    feature
    for feature in URL_ONLY_FEATURES
    if feature not in df.columns
]

if missing_url_features:

    raise ValueError(
        "Missing URL features:\n"
        + "\n".join(missing_url_features)
    )


# ================================================================
# MODEL BUILDING FUNCTION
# ================================================================

def build_models(
    numerical_features,
    categorical_features
):

    numeric_imputer = SimpleImputer(
        strategy="median"
    )

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ])


    # ------------------------------------------------------------
    # Model 1: Logistic Regression
    # ------------------------------------------------------------

    logistic_preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]),
            numerical_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])


    logistic = Pipeline([
        (
            "preprocessor",
            logistic_preprocessor
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ])


    # ------------------------------------------------------------
    # Model 2: Decision Tree
    # ------------------------------------------------------------

    tree_preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_imputer,
            numerical_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])


    decision_tree = Pipeline([
        (
            "preprocessor",
            tree_preprocessor
        ),

        (
            "model",
            DecisionTreeClassifier(
                max_depth=20,
                min_samples_split=10,
                random_state=RANDOM_STATE
            )
        )
    ])


    # ------------------------------------------------------------
    # Model 3: Random Forest
    # ------------------------------------------------------------

    forest_preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_imputer,
            numerical_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])


    random_forest = Pipeline([
        (
            "preprocessor",
            forest_preprocessor
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])


    # ------------------------------------------------------------
    # Model 4: Linear SVM
    # ------------------------------------------------------------

    svm_preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),

                (
                    "scaler",
                    StandardScaler()
                )
            ]),
            numerical_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])


    svm = Pipeline([
        (
            "preprocessor",
            svm_preprocessor
        ),

        (
            "model",
            LinearSVC(
                C=1.0,
                dual=False,
                max_iter=5000,
                random_state=RANDOM_STATE
            )
        )
    ])


    return {
        "Logistic Regression": logistic,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
        "Linear SVM": svm
    }


# ================================================================
# EVALUATION FUNCTION
# ================================================================

def evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_group,
    split_type
):

    print("\n" + "=" * 75)
    print(
        f"EVALUATING: {feature_group} | {split_type}"
    )
    print("=" * 75)


    # Identify feature types
    categorical_features = [
        column
        for column in X_train.columns
        if X_train[column].dtype == "object"
        or str(X_train[column].dtype) == "category"
    ]

    numerical_features = [
        column
        for column in X_train.columns
        if column not in categorical_features
    ]


    print(
        f"Numerical features  : "
        f"{len(numerical_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )


    models = build_models(
        numerical_features,
        categorical_features
    )


    results = []

    trained_models = {}


    for name, model in models.items():

        print("\n" + "-" * 75)
        print(f"Training {name}")
        print("-" * 75)

        start = time()

        model.fit(
            X_train,
            y_train
        )

        training_time = (
            time() - start
        )


        # --------------------------------------------------------
        # Predictions
        # --------------------------------------------------------

        y_pred = model.predict(
            X_test
        )


        # --------------------------------------------------------
        # Scores for ROC/PR
        # --------------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            score = (
                model
                .predict_proba(X_test)[:, 1]
            )

        else:

            score = (
                model
                .decision_function(X_test)
            )


        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            score
        )

        pr_auc = average_precision_score(
            y_test,
            score
        )


        results.append({
            "Feature Group": feature_group,
            "Split": split_type,
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "Training Time (s)": training_time
        })


        trained_models[name] = model


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

        print(
            f"Time      : {training_time:.2f}s"
        )


        # --------------------------------------------------------
        # Confusion matrix
        # --------------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        plt.figure(
            figsize=(6, 5)
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
            f"{feature_group} | "
            f"{split_type} | "
            f"{name}"
        )

        plt.tight_layout()

        safe_name = (
            f"{feature_group}_"
            f"{split_type}_"
            f"{name}"
            .replace(" ", "_")
            .replace("/", "_")
        )

        plt.savefig(
            OUTPUT_DIR /
            f"cm_{safe_name}.png",
            dpi=250,
            bbox_inches="tight"
        )

        plt.close()


    return (
        pd.DataFrame(results),
        trained_models
    )


# ================================================================
# CREATE RANDOM SPLIT
# ================================================================

print("\n" + "=" * 75)
print("EXPERIMENT 1 - RANDOM HOLDOUT")
print("=" * 75)


# Use web features
X_web = df[WEB_FEATURES].copy()

y = df["phishing_label"]


X_train_random, X_test_random, y_train_random, y_test_random = (
    train_test_split(
        X_web,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


random_results, random_models = evaluate_models(
    X_train_random,
    X_test_random,
    y_train_random,
    y_test_random,
    "WebEnhanced",
    "RandomSplit"
)


# ================================================================
# CREATE DOMAIN-DISJOINT SPLIT
# ================================================================

print("\n" + "=" * 75)
print("EXPERIMENT 2 - DOMAIN-DISJOINT HOLDOUT")
print("=" * 75)


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)


train_indices, test_indices = next(
    splitter.split(
        df,
        y,
        groups=df["Domain"]
    )
)


train_df = df.iloc[
    train_indices
].copy()

test_df = df.iloc[
    test_indices
].copy()


print(
    f"Training rows : {len(train_df):,}"
)

print(
    f"Testing rows  : {len(test_df):,}"
)

print(
    f"Training domains: "
    f"{train_df['Domain'].nunique():,}"
)

print(
    f"Testing domains : "
    f"{test_df['Domain'].nunique():,}"
)


train_domains = set(
    train_df["Domain"]
)

test_domains = set(
    test_df["Domain"]
)

overlap = train_domains.intersection(
    test_domains
)

print(
    f"Domain overlap: {len(overlap)}"
)


if len(overlap) != 0:

    raise RuntimeError(
        "Domain overlap detected. "
        "Group split failed."
    )

else:

    print(
        "✓ Zero domain overlap confirmed."
    )


# ------------------------------------------------
# Web-enhanced domain-disjoint model
# ------------------------------------------------

X_train_web_group = train_df[
    WEB_FEATURES
]

X_test_web_group = test_df[
    WEB_FEATURES
]

y_train_web_group = train_df[
    "phishing_label"
]

y_test_web_group = test_df[
    "phishing_label"
]


group_web_results, group_web_models = evaluate_models(
    X_train_web_group,
    X_test_web_group,
    y_train_web_group,
    y_test_web_group,
    "WebEnhanced",
    "DomainDisjoint"
)


# ================================================================
# EXPERIMENT 3 - URL-ONLY MODEL
# ================================================================

print("\n" + "=" * 75)
print("EXPERIMENT 3 - URL-ONLY DOMAIN-DISJOINT")
print("=" * 75)


X_train_url = train_df[
    URL_ONLY_FEATURES
].copy()

X_test_url = test_df[
    URL_ONLY_FEATURES
].copy()

y_train_url = train_df[
    "phishing_label"
]

y_test_url = test_df[
    "phishing_label"
]


url_results, url_models = evaluate_models(
    X_train_url,
    X_test_url,
    y_train_url,
    y_test_url,
    "URLOnly",
    "DomainDisjoint"
)


# ================================================================
# COMBINE RESULTS
# ================================================================

print("\n" + "=" * 75)
print("COMBINING RESULTS")
print("=" * 75)


all_results = pd.concat(
    [
        random_results,
        group_web_results,
        url_results
    ],
    ignore_index=True
)


all_results = all_results.sort_values(
    [
        "Feature Group",
        "Split",
        "F1 Score"
    ],
    ascending=[
        True,
        True,
        False
    ]
)


results_path = (
    OUTPUT_DIR /
    "robustness_comparison.csv"
)

all_results.to_csv(
    results_path,
    index=False
)


print(
    all_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print(
    f"\nSaved: {results_path}"
)


# ================================================================
# F1 COMPARISON
# ================================================================

plt.figure(
    figsize=(12, 7)
)

plot_df = all_results.copy()

plot_df["Configuration"] = (
    plot_df["Feature Group"]
    + " | "
    + plot_df["Split"]
    + " | "
    + plot_df["Model"]
)


plot_df = plot_df.sort_values(
    "F1 Score"
)


plt.barh(
    plot_df["Configuration"],
    plot_df["F1 Score"]
)

plt.xlim(
    0,
    1.05
)

plt.xlabel(
    "F1 Score"
)

plt.ylabel(
    "Configuration"
)

plt.title(
    "Robustness Comparison Across Feature Sets and Splits"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "robustness_f1_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# URL-ONLY MODEL PERFORMANCE
# ================================================================

url_sorted = url_results.sort_values(
    "F1 Score",
    ascending=False
)


plt.figure(
    figsize=(9, 6)
)

plt.bar(
    url_sorted["Model"],
    url_sorted["F1 Score"]
)

plt.ylim(
    0,
    1.05
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "F1 Score"
)

plt.title(
    "URL-Only Phishing Detection Performance"
)

for i, value in enumerate(
    url_sorted["F1 Score"]
):

    plt.text(
        i,
        value + 0.01,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "url_only_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# BEST URL-ONLY MODEL REPORT
# ================================================================

best_url_name = (
    url_results
    .sort_values(
        "F1 Score",
        ascending=False
    )
    .iloc[0]["Model"]
)

best_url_model = (
    url_models[best_url_name]
)

best_url_predictions = (
    best_url_model.predict(
        X_test_url
    )
)


print("\n" + "=" * 75)
print("BEST URL-ONLY MODEL")
print("=" * 75)

print(
    f"Model: {best_url_name}"
)

print(
    classification_report(
        y_test_url,
        best_url_predictions,
        target_names=[
            "Legitimate",
            "Phishing"
        ],
        digits=4
    )
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 75)
print("ROBUSTNESS AUDIT COMPLETE")
print("=" * 75)

print(
    "\nOutputs saved to:"
)

print(
    OUTPUT_DIR.resolve()
)

print(
    "\nKey experiments:"
)

print(
    "✓ Random split benchmark"
)

print(
    "✓ Domain-disjoint web-enhanced benchmark"
)

print(
    "✓ Domain-disjoint URL-only model"
)

print(
    "\nNext step:"
)

print(
    "Use the URL-only model to build our "
    "real-time feature extraction pipeline."
)