# ================================================================
# ScamShield India
# Advanced Deployable URL Model
#
# Goal:
#   Build a phishing detector that can operate using ONLY
#   information available from a raw URL.
#
# Experiments:
#   1. Structured URL features
#   2. Character-level TF-IDF features
#   3. Hybrid URL model = TF-IDF + structured features
#
# Evaluation:
#   Domain-disjoint train/test split
#
# ================================================================


import re
import math
import ipaddress
import joblib

import numpy as np
import pandas as pd

from pathlib import Path
from urllib.parse import urlparse

from sklearn.model_selection import GroupShuffleSplit

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

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
    "outputs/advanced_model"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42


# PhiUSIIL:
#
# 0 = Phishing
# 1 = Legitimate
#
# Convert to:
#
# 0 = Legitimate
# 1 = Phishing


# ================================================================
# SUSPICIOUS KEYWORDS
# ================================================================

SUSPICIOUS_KEYWORDS = [
    "login",
    "signin",
    "sign-in",
    "verify",
    "verification",
    "account",
    "secure",
    "security",
    "update",
    "confirm",
    "confirmation",
    "password",
    "passwd",
    "credential",
    "credentials",
    "bank",
    "banking",
    "payment",
    "billing",
    "invoice",
    "wallet",
    "support",
    "recover",
    "recovery",
    "unlock",
    "suspend",
    "suspended",
    "authenticate",
    "authentication",
    "webscr",
    "ebayisapi",
    "paypal"
]


# Common URL-shortening services
SHORTENER_PATTERNS = [
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "shorturl.at",
    "rebrand.ly"
]


# ================================================================
# ENTROPY
# ================================================================

def calculate_entropy(text):
    """
    Shannon entropy of a string.

    Higher entropy can indicate more random-looking
    or obfuscated URL components.
    """

    if not text:
        return 0.0

    counts = {}

    for char in text:
        counts[char] = counts.get(char, 0) + 1

    length = len(text)

    entropy = 0.0

    for count in counts.values():

        probability = count / length

        entropy -= (
            probability *
            math.log2(probability)
        )

    return entropy


# ================================================================
# IP ADDRESS DETECTION
# ================================================================

def is_ip_address(hostname):
    """
    Determine whether the hostname is an IPv4 or IPv6 address.
    """

    if not hostname:
        return 0

    try:

        ipaddress.ip_address(hostname)

        return 1

    except ValueError:

        return 0


# ================================================================
# URL FEATURE EXTRACTION
# ================================================================

def extract_url_features(url):
    """
    Extract features that can be calculated directly from
    a raw URL.

    No webpage request is performed.
    """

    url = str(url).strip()

    # ------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------

    parse_target = url

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        parse_target
    ):

        parse_target = (
            "http://"
            + parse_target
        )

    try:

        parsed = urlparse(
            parse_target
        )

    except Exception:

        parsed = urlparse(
            "http://" + url
        )


    scheme = (
        parsed.scheme.lower()
    )

    hostname = (
        parsed.hostname or ""
    ).lower()

    path = (
        parsed.path or ""
    )

    query = (
        parsed.query or ""
    )

    fragment = (
        parsed.fragment or ""
    )


    # ------------------------------------------------------------
    # Basic lengths
    # ------------------------------------------------------------

    url_length = len(url)

    hostname_length = len(
        hostname
    )

    path_length = len(
        path
    )

    query_length = len(
        query
    )

    fragment_length = len(
        fragment
    )


    # ------------------------------------------------------------
    # Character counts
    # ------------------------------------------------------------

    digit_count = sum(
        char.isdigit()
        for char in url
    )

    letter_count = sum(
        char.isalpha()
        for char in url
    )

    special_count = sum(
        not char.isalnum()
        for char in url
    )


    # ------------------------------------------------------------
    # Ratios
    # ------------------------------------------------------------

    digit_ratio = (
        digit_count / url_length
        if url_length > 0
        else 0.0
    )

    special_ratio = (
        special_count / url_length
        if url_length > 0
        else 0.0
    )

    hostname_digit_count = sum(
        char.isdigit()
        for char in hostname
    )

    hostname_digit_ratio = (
        hostname_digit_count / hostname_length
        if hostname_length > 0
        else 0.0
    )


    # ------------------------------------------------------------
    # Structural features
    # ------------------------------------------------------------

    dot_count = url.count(".")

    hyphen_count = url.count("-")

    underscore_count = url.count("_")

    at_count = url.count("@")

    question_count = url.count("?")

    equals_count = url.count("=")

    ampersand_count = url.count("&")

    percent_count = url.count("%")

    colon_count = url.count(":")

    semicolon_count = url.count(";")

    slash_count = url.count("/")


    # ------------------------------------------------------------
    # Subdomain estimation
    # ------------------------------------------------------------

    hostname_parts = [
        part
        for part in hostname.split(".")
        if part
    ]

    # Approximation:
    # hostname = subdomain + domain + TLD
    #
    # For example:
    # login.paypal.com
    #
    # parts = login / paypal / com
    #
    # subdomains = 1

    subdomain_count = max(
        len(hostname_parts) - 2,
        0
    )


    # ------------------------------------------------------------
    # Path depth
    # ------------------------------------------------------------

    path_depth = len([
        part
        for part in path.split("/")
        if part
    ])


    # ------------------------------------------------------------
    # IP address
    # ------------------------------------------------------------

    domain_is_ip = is_ip_address(
        hostname
    )


    # ------------------------------------------------------------
    # HTTPS
    # ------------------------------------------------------------

    is_https = int(
        scheme == "https"
    )


    # ------------------------------------------------------------
    # Port
    # ------------------------------------------------------------

    try:

        port = (
            parsed.port
            if parsed.port is not None
            else 0
        )

        has_port = int(
            port != 0
        )

    except ValueError:

        has_port = 1


    # ------------------------------------------------------------
    # Punycode
    # ------------------------------------------------------------

    has_punycode = int(
        "xn--" in hostname
    )


    # ------------------------------------------------------------
    # Obfuscation-like indicators
    # ------------------------------------------------------------

    encoded_character_count = len(
        re.findall(
            r"%[0-9a-fA-F]{2}",
            url
        )
    )

    double_slash_inside = int(
        "//" in path
    )


    # ------------------------------------------------------------
    # Suspicious keywords
    # ------------------------------------------------------------

    lower_url = url.lower()

    suspicious_keyword_count = 0

    for keyword in SUSPICIOUS_KEYWORDS:

        if keyword in lower_url:

            suspicious_keyword_count += 1


    # ------------------------------------------------------------
    # URL shortener
    # ------------------------------------------------------------

    has_shortener = 0

    for shortener in SHORTENER_PATTERNS:

        if shortener in hostname:

            has_shortener = 1

            break


    # ------------------------------------------------------------
    # Fragment
    # ------------------------------------------------------------

    has_fragment = int(
        bool(fragment)
    )


    # ------------------------------------------------------------
    # Query
    # ------------------------------------------------------------

    has_query = int(
        bool(query)
    )


    # ------------------------------------------------------------
    # Email-like @ pattern
    # ------------------------------------------------------------

    has_at_symbol = int(
        "@" in url
    )


    # ------------------------------------------------------------
    # Numeric IP + suspicious URL structure
    # ------------------------------------------------------------

    repeated_symbol_count = (
        url.count("--")
        + url.count("__")
        + url.count("..")
    )


    # ------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------

    url_entropy = calculate_entropy(
        url
    )

    hostname_entropy = calculate_entropy(
        hostname
    )


    # ------------------------------------------------------------
    # TLD
    # ------------------------------------------------------------

    if hostname_parts:

        tld = hostname_parts[-1].lower()

    else:

        tld = ""


    tld_length = len(
        tld
    )


    # ------------------------------------------------------------
    # Final feature dictionary
    # ------------------------------------------------------------

    return {

        "URLLength": url_length,

        "DomainLength": hostname_length,

        "PathLength": path_length,

        "QueryLength": query_length,

        "FragmentLength": fragment_length,

        "NoOfDigits": digit_count,

        "NoOfLetters": letter_count,

        "NoOfSpecialChars": special_count,

        "DigitRatio": digit_ratio,

        "SpecialCharRatio": special_ratio,

        "HostnameDigitRatio": hostname_digit_ratio,

        "NoOfDots": dot_count,

        "NoOfHyphens": hyphen_count,

        "NoOfUnderscores": underscore_count,

        "NoOfAtSymbols": at_count,

        "NoOfQuestionMarks": question_count,

        "NoOfEquals": equals_count,

        "NoOfAmpersands": ampersand_count,

        "NoOfPercentEncoded": encoded_character_count,

        "NoOfColons": colon_count,

        "NoOfSemicolons": semicolon_count,

        "NoOfSlashes": slash_count,

        "NoOfSubdomains": subdomain_count,

        "PathDepth": path_depth,

        "IsDomainIP": domain_is_ip,

        "IsHTTPS": is_https,

        "HasPort": has_port,

        "HasPunycode": has_punycode,

        "HasAtSymbol": has_at_symbol,

        "HasQuery": has_query,

        "HasFragment": has_fragment,

        "HasDoubleSlashInPath": double_slash_inside,

        "HasShortener": has_shortener,

        "SuspiciousKeywordCount": suspicious_keyword_count,

        "RepeatedSymbolCount": repeated_symbol_count,

        "URLEntropy": url_entropy,

        "HostnameEntropy": hostname_entropy,

        "TLDLength": tld_length,

        "TLD": tld,
    }


# ================================================================
# BUILD STRUCTURED FEATURE DATAFRAME
# ================================================================

def create_feature_dataframe(url_series):

    records = []

    for index, url in enumerate(
        url_series
    ):

        if index % 25000 == 0:

            print(
                f"Extracting URL features: "
                f"{index:,}/{len(url_series):,}"
            )

        features = extract_url_features(
            url
        )

        records.append(features)

    return pd.DataFrame(
        records,
        index=url_series.index
    )


# ================================================================
# LOAD DATA
# ================================================================

print(
    "=" * 75
)

print(
    "SCAMSHIELD - ADVANCED URL MODEL"
)

print(
    "=" * 75
)

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Rows: {len(df):,}"
)


# ================================================================
# REMOVE DUPLICATE URLS
# ================================================================

print(
    "\nRemoving duplicate URLs..."
)

before = len(df)

df = df.drop_duplicates(
    subset=["URL"]
).copy()

print(
    f"Removed: {before - len(df):,}"
)

print(
    f"Remaining: {len(df):,}"
)


# ================================================================
# TARGET
# ================================================================

# Original:
# 0 = phishing
# 1 = legitimate
#
# New:
# 0 = legitimate
# 1 = phishing

df["phishing_label"] = (
    df["label"] == 0
).astype(int)


# ================================================================
# DOMAIN GROUP
# ================================================================

# Domain-disjoint evaluation:
# URLs from the same domain cannot appear in
# both training and testing.

df["domain_group"] = (
    df["Domain"]
    .fillna("")
    .astype(str)
    .str.lower()
)


# ================================================================
# EXTRACT URL FEATURES
# ================================================================

print(
    "\n" + "=" * 75
)

print(
    "EXTRACTING DEPLOYABLE URL FEATURES"
)

print(
    "=" * 75
)


structured_df = create_feature_dataframe(
    df["URL"]
)


# Make sure the URL feature rows line up.
structured_df.index = df.index


# ================================================================
# ADD TARGET + RAW URL
# ================================================================

structured_df["URL"] = (
    df["URL"]
    .astype(str)
)

structured_df["phishing_label"] = (
    df["phishing_label"]
)

structured_df["domain_group"] = (
    df["domain_group"]
)


# ================================================================
# SAVE ENGINEERED DATA
# ================================================================

feature_path = (
    OUTPUT_DIR /
    "engineered_url_features.csv"
)

structured_df.to_csv(
    feature_path,
    index=False
)

print(
    f"\nSaved engineered features:"
)

print(
    feature_path
)


# ================================================================
# DOMAIN-DISJOINT SPLIT
# ================================================================

print(
    "\n" + "=" * 75
)

print(
    "DOMAIN-DISJOINT TRAIN/TEST SPLIT"
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
        structured_df,
        structured_df["phishing_label"],
        groups=structured_df["domain_group"]
    )
)


train_df = structured_df.iloc[
    train_idx
].copy()

test_df = structured_df.iloc[
    test_idx
].copy()


print(
    f"Training rows : {len(train_df):,}"
)

print(
    f"Testing rows  : {len(test_df):,}"
)

print(
    f"Training domains: "
    f"{train_df['domain_group'].nunique():,}"
)

print(
    f"Testing domains : "
    f"{test_df['domain_group'].nunique():,}"
)


overlap = (
    set(train_df["domain_group"])
    &
    set(test_df["domain_group"])
)

print(
    f"Domain overlap: {len(overlap)}"
)


if overlap:

    raise RuntimeError(
        "ERROR: domain overlap detected."
    )

print(
    "✓ Domain-disjoint split confirmed."
)


# ================================================================
# LABELS
# ================================================================

y_train = train_df[
    "phishing_label"
]

y_test = test_df[
    "phishing_label"
]


# ================================================================
# STRUCTURED FEATURE COLUMNS
# ================================================================

STRUCTURED_FEATURES = [
    column
    for column in structured_df.columns
    if column not in [
        "URL",
        "phishing_label",
        "domain_group"
    ]
]


NUMERIC_FEATURES = [
    column
    for column in STRUCTURED_FEATURES
    if column != "TLD"
]


CATEGORICAL_FEATURES = [
    "TLD"
]


print(
    "\n" + "=" * 75
)

print(
    "FEATURE SUMMARY"
)

print(
    "=" * 75
)

print(
    f"Structured numerical features: "
    f"{len(NUMERIC_FEATURES)}"
)

print(
    f"Structured categorical features: "
    f"{len(CATEGORICAL_FEATURES)}"
)


# ================================================================
# PREPROCESSING FOR STRUCTURED FEATURES
# ================================================================

structured_preprocessor = ColumnTransformer(
    transformers=[

        # ========================================================
        # NUMERICAL FEATURES
        # ========================================================

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

            NUMERIC_FEATURES
        ),

        # ========================================================
        # TLD CATEGORICAL FEATURE
        # ========================================================

        (
            "tld",

            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),

                (
                    "onehot",
                    __import__(
                        "sklearn.preprocessing",
                        fromlist=["OneHotEncoder"]
                    ).OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True
                    )
                )
            ]),

            CATEGORICAL_FEATURES
        )
    ],

    remainder="drop"
)

# ================================================================
# MODEL 1 - LOGISTIC REGRESSION
# ================================================================

print(
    "\n" + "=" * 75
)

print(
    "MODEL 1 - STRUCTURED LOGISTIC REGRESSION"
)

print(
    "=" * 75
)


logistic_model = Pipeline([
    (
        "preprocessor",
        structured_preprocessor
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    )
])


start = pd.Timestamp.now()

logistic_model.fit(
    train_df[
        STRUCTURED_FEATURES
    ],
    y_train
)

elapsed = (
    pd.Timestamp.now() - start
).total_seconds()


logistic_predictions = (
    logistic_model.predict(
        test_df[
            STRUCTURED_FEATURES
        ]
    )
)

logistic_scores = (
    logistic_model
    .predict_proba(
        test_df[
            STRUCTURED_FEATURES
        ]
    )[:, 1]
)


# ================================================================
# METRICS FUNCTION
# ================================================================

def calculate_metrics(
    name,
    y_true,
    predictions,
    scores,
    training_time
):

    return {

        "Model": name,

        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            scores
        ),

        "PR-AUC": average_precision_score(
            y_true,
            scores
        ),

        "Training Time (s)": training_time
    }


results = []


results.append(
    calculate_metrics(
        "Structured Logistic Regression",
        y_test,
        logistic_predictions,
        logistic_scores,
        elapsed
    )
)


print(
    results[-1]
)


# ================================================================
# MODEL 2 - RANDOM FOREST
# ================================================================

print(
    "\n" + "=" * 75
)

print(
    "MODEL 2 - STRUCTURED RANDOM FOREST"
)

print(
    "=" * 75
)


rf_preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",

            SimpleImputer(
                strategy="median"
            ),

            NUMERIC_FEATURES
        )
    ],

    remainder="drop"
)


rf_model = Pipeline([
    (
        "preprocessor",
        rf_preprocessor
    ),

    (
        "classifier",

        RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
])


start = pd.Timestamp.now()

rf_model.fit(
    train_df[
        NUMERIC_FEATURES
    ],
    y_train
)

elapsed = (
    pd.Timestamp.now() - start
).total_seconds()


rf_predictions = (
    rf_model.predict(
        test_df[
            NUMERIC_FEATURES
        ]
    )
)

rf_scores = (
    rf_model
    .predict_proba(
        test_df[
            NUMERIC_FEATURES
        ]
    )[:, 1]
)


results.append(
    calculate_metrics(
        "Structured Random Forest",
        y_test,
        rf_predictions,
        rf_scores,
        elapsed
    )
)


print(
    results[-1]
)


# ================================================================
# MODEL 3 - CHARACTER TF-IDF + LINEAR SVM
# ================================================================

print(
    "\n" + "=" * 75
)

print(
    "MODEL 3 - CHARACTER TF-IDF LINEAR SVM"
)

print(
    "=" * 75
)


# Character n-grams learn recurring URL patterns without
# requiring manually defined keywords.

tfidf_svm = Pipeline([

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
        "classifier",

        LinearSVC(
            C=1.0,
            dual="auto",
            max_iter=5000,
            random_state=RANDOM_STATE
        )
    )
])


start = pd.Timestamp.now()

tfidf_svm.fit(
    train_df["URL"],
    y_train
)

elapsed = (
    pd.Timestamp.now() - start
).total_seconds()


tfidf_predictions = (
    tfidf_svm.predict(
        test_df["URL"]
    )
)

tfidf_scores = (
    tfidf_svm.decision_function(
        test_df["URL"]
    )
)


results.append(
    calculate_metrics(
        "Character TF-IDF + Linear SVM",
        y_test,
        tfidf_predictions,
        tfidf_scores,
        elapsed
    )
)


print(
    results[-1]
)


# ================================================================
# MODEL 4 - CHARACTER TF-IDF + LOGISTIC REGRESSION
# ================================================================

print(
    "\n" + "=" * 75
)

print(
    "MODEL 4 - CHARACTER TF-IDF LOGISTIC REGRESSION"
)

print(
    "=" * 75
)


tfidf_logistic = Pipeline([

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
        "classifier",

        LogisticRegression(
            max_iter=1000,
            C=4.0,
            random_state=RANDOM_STATE
        )
    )
])


start = pd.Timestamp.now()

tfidf_logistic.fit(
    train_df["URL"],
    y_train
)

elapsed = (
    pd.Timestamp.now() - start
).total_seconds()


tfidf_logistic_predictions = (
    tfidf_logistic.predict(
        test_df["URL"]
    )
)

tfidf_logistic_scores = (
    tfidf_logistic
    .predict_proba(
        test_df["URL"]
    )[:, 1]
)


results.append(
    calculate_metrics(
        "Character TF-IDF + Logistic Regression",
        y_test,
        tfidf_logistic_predictions,
        tfidf_logistic_scores,
        elapsed
    )
)


print(
    results[-1]
)


# ================================================================
# RESULTS
# ================================================================

results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values(
        "F1 Score",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    "\n" + "=" * 75
)

print(
    "ADVANCED MODEL COMPARISON"
)

print(
    "=" * 75
)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


results_df.to_csv(
    OUTPUT_DIR /
    "advanced_model_comparison.csv",
    index=False
)


# ================================================================
# BEST MODEL
# ================================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)


print(
    "\n" + "=" * 75
)

print(
    "BEST ADVANCED MODEL"
)

print(
    "=" * 75
)

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
# CLASSIFICATION REPORTS
# ================================================================

print(
    "\n" + "=" * 75
)

print(
    "CHARACTER SVM CLASSIFICATION REPORT"
)

print(
    "=" * 75
)

print(
    classification_report(
        y_test,
        tfidf_predictions,
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
    tfidf_predictions
)


print(
    "\nCharacter SVM confusion matrix:"
)

print(cm)


# ================================================================
# SAVE DEPLOYABLE MODELS
# ================================================================

print(
    "\n" + "=" * 75
)

print(
    "SAVING MODELS"
)

print(
    "=" * 75
)


joblib.dump(
    logistic_model,
    OUTPUT_DIR /
    "structured_logistic_model.joblib"
)


joblib.dump(
    tfidf_svm,
    OUTPUT_DIR /
    "character_tfidf_svm.joblib"
)


joblib.dump(
    tfidf_logistic,
    OUTPUT_DIR /
    "character_tfidf_logistic.joblib"
)


print(
    "✓ Structured Logistic Regression saved"
)

print(
    "✓ Character TF-IDF SVM saved"
)

print(
    "✓ Character TF-IDF Logistic Regression saved"
)


# ================================================================
# SAVE SAMPLE FEATURE EXTRACTION
# ================================================================

sample_urls = [
    "https://www.google.com",
    "https://secure-login.example.com/verify-account",
    "http://192.168.0.1/login",
    "https://paypal-secure-login.example.com/account/verify"
]


sample_features = pd.DataFrame(
    [
        extract_url_features(url)
        for url in sample_urls
    ]
)


sample_features.insert(
    0,
    "URL",
    sample_urls
)


sample_features.to_csv(
    OUTPUT_DIR /
    "feature_extraction_examples.csv",
    index=False
)


print(
    "\nSaved feature extraction examples."
)


# ================================================================
# FINAL
# ================================================================

print(
    "\n" + "=" * 75
)

print(
    "ADVANCED URL MODELING COMPLETE"
)

print(
    "=" * 75
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR.resolve()
)

print(
    "\nThe saved models can be connected to the future"
)

print(
    "ScamShield website without requiring webpage scraping."
)