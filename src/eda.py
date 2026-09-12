# ================================================================
# ScamShield India - Exploratory Data Analysis
# PhiUSIIL Phishing URL Dataset
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ================================================================
# CONFIGURATION
# ================================================================

DATA_PATH = Path("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
OUTPUT_DIR = Path("outputs/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# HELPER FUNCTION
# ================================================================

def save_plot(filename):
    """
    Save the current matplotlib figure.
    """
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


# ================================================================
# LOAD DATA
# ================================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")


# ================================================================
# BASIC INFORMATION
# ================================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ================================================================
# TARGET DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

target_counts = df["label"].value_counts().sort_index()

print(target_counts)

print("\nPercentages:")
print((target_counts / len(df) * 100).round(2))


# ------------------------------------------------
# Plot 1 - Target Distribution
# ------------------------------------------------

plt.figure(figsize=(8, 6))

labels = ["Phishing", "Legitimate"]
values = [
    target_counts.get(0, 0),
    target_counts.get(1, 0)
]

plt.bar(labels, values)

plt.title("Distribution of Phishing and Legitimate URLs")
plt.xlabel("URL Class")
plt.ylabel("Number of URLs")

for i, value in enumerate(values):
    plt.text(
        i,
        value,
        f"{value:,}",
        ha="center",
        va="bottom"
    )

save_plot("01_target_distribution.png")


# ================================================================
# NUMERICAL FEATURES
# ================================================================

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

numeric_columns.remove("label")

print("\n" + "=" * 70)
print("NUMERICAL FEATURES")
print("=" * 70)

print(f"Number of numerical features: {len(numeric_columns)}")

print("\nDescriptive statistics:")
print(
    df[numeric_columns]
    .describe()
    .T
    .round(3)
)


# ================================================================
# PHISHING VS LEGITIMATE MEAN COMPARISON
# ================================================================

print("\n" + "=" * 70)
print("PHISHING VS LEGITIMATE FEATURE COMPARISON")
print("=" * 70)

feature_means = df.groupby("label")[numeric_columns].mean().T

feature_means.columns = [
    "Phishing" if col == 0 else "Legitimate"
    for col in feature_means.columns
]

feature_means["Absolute Difference"] = abs(
    feature_means["Phishing"] -
    feature_means["Legitimate"]
)

feature_means = feature_means.sort_values(
    "Absolute Difference",
    ascending=False
)

print(feature_means.head(20).round(3))


# ================================================================
# TOP FEATURES BY CORRELATION WITH LABEL
# ================================================================

print("\n" + "=" * 70)
print("FEATURE CORRELATION WITH TARGET")
print("=" * 70)

correlations = (
    df[numeric_columns + ["label"]]
    .corr()["label"]
    .drop("label")
    .sort_values(key=abs, ascending=False)
)

print(correlations.head(20).round(4))


# ------------------------------------------------
# Plot 2 - Top Correlated Features
# ------------------------------------------------

top_corr = correlations.head(15).sort_values()

plt.figure(figsize=(10, 7))

plt.barh(
    top_corr.index,
    top_corr.values
)

plt.title("Top 15 Numerical Features Correlated with URL Class")
plt.xlabel("Correlation with Label")
plt.ylabel("Feature")

save_plot("02_top_feature_correlations.png")


# ================================================================
# SELECTED FEATURE DISTRIBUTIONS
# ================================================================

selected_features = [
    "URLLength",
    "DomainLength",
    "NoOfSubDomain",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "SpacialCharRatioInURL",
    "NoOfURLRedirect",
    "NoOfPopup",
    "NoOfiFrame"
]

selected_features = [
    feature
    for feature in selected_features
    if feature in df.columns
]


# ================================================================
# DISTRIBUTION PLOTS
# ================================================================

print("\n" + "=" * 70)
print("GENERATING FEATURE DISTRIBUTIONS")
print("=" * 70)

for feature in selected_features:

    plt.figure(figsize=(9, 6))

    sns.histplot(
        data=df,
        x=feature,
        hue="label",
        bins=40,
        kde=True,
        stat="density",
        common_norm=False
    )

    plt.title(
        f"{feature}: Phishing vs Legitimate URLs"
    )

    plt.xlabel(feature)
    plt.ylabel("Density")

    save_plot(
        f"03_distribution_{feature}.png"
    )


# ================================================================
# BOXPLOTS
# ================================================================

print("\n" + "=" * 70)
print("GENERATING BOXPLOTS")
print("=" * 70)

boxplot_features = [
    "URLLength",
    "DomainLength",
    "NoOfSubDomain",
    "ObfuscationRatio",
    "SpacialCharRatioInURL",
    "NoOfURLRedirect"
]

boxplot_features = [
    feature
    for feature in boxplot_features
    if feature in df.columns
]


for feature in boxplot_features:

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="label",
        y=feature
    )

    plt.xticks(
        [0, 1],
        ["Phishing", "Legitimate"]
    )

    plt.title(
        f"{feature} by URL Class"
    )

    plt.xlabel("URL Class")
    plt.ylabel(feature)

    save_plot(
        f"04_boxplot_{feature}.png"
    )


# ================================================================
# HTTPS ANALYSIS
# ================================================================

if "IsHTTPS" in df.columns:

    print("\n" + "=" * 70)
    print("HTTPS ANALYSIS")
    print("=" * 70)

    https_table = pd.crosstab(
        df["IsHTTPS"],
        df["label"],
        normalize="index"
    ) * 100

    https_table.columns = [
        "Phishing (%)",
        "Legitimate (%)"
    ]

    print(https_table.round(2))

    plt.figure(figsize=(8, 6))

    sns.countplot(
        data=df,
        x="IsHTTPS",
        hue="label"
    )

    plt.title(
        "HTTPS Usage by URL Class"
    )

    plt.xlabel("Uses HTTPS")
    plt.ylabel("Number of URLs")

    plt.legend(
        title="Class",
        labels=["Phishing", "Legitimate"]
    )

    save_plot("05_https_analysis.png")


# ================================================================
# DOMAIN IP ANALYSIS
# ================================================================

if "IsDomainIP" in df.columns:

    print("\n" + "=" * 70)
    print("DOMAIN IP ANALYSIS")
    print("=" * 70)

    ip_table = pd.crosstab(
        df["IsDomainIP"],
        df["label"],
        normalize="index"
    ) * 100

    ip_table.columns = [
        "Phishing (%)",
        "Legitimate (%)"
    ]

    print(ip_table.round(2))

    plt.figure(figsize=(8, 6))

    sns.countplot(
        data=df,
        x="IsDomainIP",
        hue="label"
    )

    plt.title(
        "IP Address Usage in URLs"
    )

    plt.xlabel("Domain is an IP Address")
    plt.ylabel("Number of URLs")

    plt.legend(
        title="Class",
        labels=["Phishing", "Legitimate"]
    )

    save_plot("06_domain_ip_analysis.png")


# ================================================================
# OBFUSCATION ANALYSIS
# ================================================================

if "HasObfuscation" in df.columns:

    print("\n" + "=" * 70)
    print("URL OBFUSCATION ANALYSIS")
    print("=" * 70)

    obfuscation_table = pd.crosstab(
        df["HasObfuscation"],
        df["label"],
        normalize="index"
    ) * 100

    obfuscation_table.columns = [
        "Phishing (%)",
        "Legitimate (%)"
    ]

    print(obfuscation_table.round(2))

    plt.figure(figsize=(8, 6))

    sns.countplot(
        data=df,
        x="HasObfuscation",
        hue="label"
    )

    plt.title(
        "URL Obfuscation by Class"
    )

    plt.xlabel("Contains Obfuscation")
    plt.ylabel("Number of URLs")

    plt.legend(
        title="Class",
        labels=["Phishing", "Legitimate"]
    )

    save_plot("07_obfuscation_analysis.png")


# ================================================================
# SUBDOMAIN ANALYSIS
# ================================================================

if "NoOfSubDomain" in df.columns:

    print("\n" + "=" * 70)
    print("SUBDOMAIN ANALYSIS")
    print("=" * 70)

    subdomain_stats = (
        df.groupby("label")["NoOfSubDomain"]
        .agg(["mean", "median", "max"])
    )

    print(subdomain_stats)

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        data=df,
        x="label",
        y="NoOfSubDomain"
    )

    plt.xticks(
        [0, 1],
        ["Phishing", "Legitimate"]
    )

    plt.title(
        "Number of Subdomains by URL Class"
    )

    plt.xlabel("URL Class")
    plt.ylabel("Number of Subdomains")

    save_plot("08_subdomain_analysis.png")


# ================================================================
# SPECIAL CHARACTER ANALYSIS
# ================================================================

special_features = [
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL"
]

special_features = [
    feature
    for feature in special_features
    if feature in df.columns
]


print("\n" + "=" * 70)
print("SPECIAL CHARACTER ANALYSIS")
print("=" * 70)

special_means = (
    df.groupby("label")[special_features]
    .mean()
    .T
)

special_means.columns = [
    "Phishing",
    "Legitimate"
]

print(special_means.round(3))


# ================================================================
# CORRELATION HEATMAP
# ================================================================

print("\n" + "=" * 70)
print("GENERATING CORRELATION HEATMAP")
print("=" * 70)

# Use top correlated features instead of all features
# to keep the heatmap readable.

heatmap_features = list(
    correlations.head(20).index
)

heatmap_features.append("label")

corr_matrix = df[heatmap_features].corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap of Top Features"
)

save_plot("09_correlation_heatmap.png")


# ================================================================
# MULTIVARIATE ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("MULTIVARIATE ANALYSIS")
print("=" * 70)

# Select a few meaningful features
multivariate_features = [
    "URLLength",
    "NoOfSubDomain",
    "ObfuscationRatio",
    "SpacialCharRatioInURL",
    "NoOfURLRedirect",
    "label"
]

multivariate_features = [
    feature
    for feature in multivariate_features
    if feature in df.columns
]


# Use a sample so plotting stays fast
sample_size = min(10000, len(df))

plot_df = df[
    multivariate_features
].sample(
    n=sample_size,
    random_state=42
)


sns.pairplot(
    plot_df,
    hue="label",
    diag_kind="hist"
)

plt.suptitle(
    "Multivariate Relationships Among Selected Features",
    y=1.02
)

plt.savefig(
    OUTPUT_DIR / "10_multivariate_pairplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Multivariate pairplot generated using {sample_size:,} samples."
)


# ================================================================
# PCA VISUALIZATION
# ================================================================

print("\n" + "=" * 70)
print("PCA VISUALIZATION")
print("=" * 70)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Use numerical features only
pca_features = numeric_columns.copy()

# Remove extremely problematic columns if necessary
X_pca = df[pca_features].copy()

# Replace infinite values
X_pca = X_pca.replace(
    [np.inf, -np.inf],
    np.nan
)

# Median imputation
X_pca = X_pca.fillna(
    X_pca.median()
)

# Standardize
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_pca)


# PCA
pca = PCA(
    n_components=2,
    random_state=42
)

X_pca_2d = pca.fit_transform(X_scaled)


pca_df = pd.DataFrame(
    {
        "PC1": X_pca_2d[:, 0],
        "PC2": X_pca_2d[:, 1],
        "label": df["label"].values
    }
)


# Sample for visualization
pca_sample = pca_df.sample(
    n=min(15000, len(pca_df)),
    random_state=42
)


plt.figure(figsize=(10, 8))

sns.scatterplot(
    data=pca_sample,
    x="PC1",
    y="PC2",
    hue="label",
    alpha=0.5,
    s=30
)

explained_1 = pca.explained_variance_ratio_[0] * 100
explained_2 = pca.explained_variance_ratio_[1] * 100

plt.title(
    "PCA Projection of Phishing URL Dataset"
)

plt.xlabel(
    f"Principal Component 1 ({explained_1:.2f}% variance)"
)

plt.ylabel(
    f"Principal Component 2 ({explained_2:.2f}% variance)"
)

plt.legend(
    title="Class",
    labels=["Phishing", "Legitimate"]
)

save_plot("11_pca_visualization.png")

print(
    f"PC1 explained variance: {explained_1:.2f}%"
)

print(
    f"PC2 explained variance: {explained_2:.2f}%"
)


# ================================================================
# TLD ANALYSIS
# ================================================================

if "TLD" in df.columns:

    print("\n" + "=" * 70)
    print("TOP TLD ANALYSIS")
    print("=" * 70)

    # Top 15 TLDs overall
    top_tlds = (
        df["TLD"]
        .value_counts()
        .head(15)
    )

    print(top_tlds)

    plt.figure(figsize=(12, 7))

    top_tlds.sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Top 15 Most Common Top-Level Domains"
    )

    plt.xlabel("Number of URLs")
    plt.ylabel("TLD")

    save_plot("12_top_tlds.png")


# ================================================================
# SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(f"\nAll visualizations saved to:")
print(OUTPUT_DIR.resolve())

print("\nGenerated visualizations:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  ✓ {file.name}")

print("\nReady for the modeling stage.")