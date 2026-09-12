import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
PROCESSED_DIR = Path("data/processed")


# ============================================================
# LOAD DATASET
# ============================================================

def load_data():
    """Load the original PhiUSIIL phishing URL dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found at: {DATA_PATH}\n"
            "Make sure the CSV is inside data/raw/"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("DATASET LOADED")
    print("=" * 70)
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]:,}")

    return df


# ============================================================
# INITIAL CLEANING
# ============================================================

def clean_data(df):
    """Perform basic cleaning."""

    print("\n" + "=" * 70)
    print("INITIAL CLEANING")
    print("=" * 70)

    original_rows = len(df)

    # Remove exact duplicate rows
    df = df.drop_duplicates().copy()

    duplicates_removed = original_rows - len(df)

    print(f"Duplicate rows removed : {duplicates_removed:,}")

    # Remove rows with missing target
    before = len(df)

    df = df.dropna(subset=["label"]).copy()

    target_rows_removed = before - len(df)

    print(f"Rows with missing label : {target_rows_removed:,}")

    return df


# ============================================================
# TARGET ANALYSIS
# ============================================================

def analyze_target(df):
    """Display target distribution."""

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    counts = df["label"].value_counts().sort_index()

    print(counts)

    percentages = (
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nPercentage:")
    for label, percentage in percentages.items():
        print(f"Class {label}: {percentage}%")


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):
    """
    Separate target from input features and remove columns
    that should not be directly used by the initial models.
    """

    print("\n" + "=" * 70)
    print("FEATURE PREPARATION")
    print("=" * 70)

    # Target
    y = df["label"].copy()

    # Raw/high-cardinality/text columns
    columns_to_drop = [
        "label",
        "FILENAME",
        "URL",
        "Title"
    ]

    existing_columns = [
        column
        for column in columns_to_drop
        if column in df.columns
    ]

    X = df.drop(columns=existing_columns).copy()

    print("Dropped columns:")

    for column in existing_columns:
        print(f"  - {column}")

    print(f"\nRemaining features: {X.shape[1]}")

    return X, y


# ============================================================
# HANDLE CATEGORICAL FEATURES
# ============================================================

def encode_features(X):
    """
    Handle categorical features safely.

    Domain has extremely high cardinality, so it is removed
    instead of one-hot encoded.

    Low-cardinality categorical features such as TLD are
    one-hot encoded.
    """

    print("\n" + "=" * 70)
    print("CATEGORICAL FEATURE HANDLING")
    print("=" * 70)

    # --------------------------------------------------------
    # HIGH-CARDINALITY DOMAIN
    # --------------------------------------------------------

    if "Domain" in X.columns:

        unique_domains = X["Domain"].nunique()

        print("High-cardinality feature detected:")
        print(f"  Domain unique values: {unique_domains:,}")

        print("\nDropping Domain instead of one-hot encoding.")

        X = X.drop(columns=["Domain"])

    # --------------------------------------------------------
    # FIND REMAINING CATEGORICAL COLUMNS
    # --------------------------------------------------------

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if categorical_columns:

        print("\nCategorical columns to encode:")

        for column in categorical_columns:
            print(
                f"  - {column} "
                f"({X[column].nunique():,} unique values)"
            )

        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=True,
            dtype=np.int8
        )

        print(
            f"\nFeatures after one-hot encoding: "
            f"{X.shape[1]}"
        )

    else:

        print("\nNo remaining categorical columns.")

    return X


# ============================================================
# NUMERICAL CLEANUP
# ============================================================

def clean_numeric_values(X):
    """
    Convert features to numeric values and handle
    infinite/missing values.
    """

    print("\n" + "=" * 70)
    print("NUMERICAL CLEANUP")
    print("=" * 70)

    # Convert boolean columns to integers
    boolean_columns = X.select_dtypes(
        include=["bool"]
    ).columns

    if len(boolean_columns) > 0:
        X[boolean_columns] = X[boolean_columns].astype(np.int8)

    # Convert everything to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace infinity
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    missing_before = int(X.isnull().sum().sum())

    print(
        f"Missing values before filling: "
        f"{missing_before:,}"
    )

    # Median imputation
    if missing_before > 0:

        X = X.fillna(
            X.median(numeric_only=True)
        )

    missing_after = int(X.isnull().sum().sum())

    print(
        f"Missing values after filling : "
        f"{missing_after:,}"
    )

    return X


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):
    """
    Split the dataset into training and testing sets.

    Stratification preserves the phishing/legitimate
    class distribution.
    """

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training samples : {len(X_train):,}")
    print(f"Testing samples  : {len(X_test):,}")

    print("\nTraining class distribution:")

    print(
        y_train
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
    )

    print("\nTesting class distribution:")

    print(
        y_test
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test
):
    """Save processed train/test datasets."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X_train.to_csv(
        PROCESSED_DIR / "X_train.csv",
        index=False
    )

    X_test.to_csv(
        PROCESSED_DIR / "X_test.csv",
        index=False
    )

    y_train.to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False
    )

    y_test.to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"✓ {PROCESSED_DIR / 'X_train.csv'}"
    )

    print(
        f"✓ {PROCESSED_DIR / 'X_test.csv'}"
    )

    print(
        f"✓ {PROCESSED_DIR / 'y_train.csv'}"
    )

    print(
        f"✓ {PROCESSED_DIR / 'y_test.csv'}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    # 1. Load dataset
    df = load_data()

    # 2. Clean dataset
    df = clean_data(df)

    # 3. Analyze target
    analyze_target(df)

    # 4. Prepare X and y
    X, y = prepare_features(df)

    # 5. Handle categorical features
    X = encode_features(X)

    # 6. Clean numerical values
    X = clean_numeric_values(X)

    # 7. Train/test split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # 8. Save processed datasets
    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()