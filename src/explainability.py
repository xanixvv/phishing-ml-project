# ================================================================
# ScamShield AI
# ROBUST MODEL EXPLAINABILITY
#
# Finds the fitted TF-IDF vectorizer and Linear SVM inside
# arbitrarily nested sklearn objects.
#
# Supports structures such as:
#
#   Pipeline
#       ├── TfidfVectorizer
#       └── LinearSVC
#
#   CalibratedClassifierCV
#       └── Pipeline
#             ├── TfidfVectorizer
#             └── LinearSVC
#
#   Nested lists / dictionaries / sklearn calibration objects
#
# ================================================================

from __future__ import annotations

from typing import Any

import numpy as np


# ================================================================
# GENERIC OBJECT GRAPH WALKER
# ================================================================

def _walk_object(
    obj: Any,
    visited: set[int] | None = None,
    depth: int = 0,
    max_depth: int = 12,
):
    """
    Recursively walk through an arbitrary Python/sklearn object.

    This deliberately does not assume a specific sklearn version
    or pipeline structure.
    """

    if visited is None:
        visited = set()

    if obj is None:
        return

    if depth > max_depth:
        return

    object_id = id(obj)

    if object_id in visited:
        return

    visited.add(object_id)

    yield obj


    # ------------------------------------------------------------
    # Dictionaries
    # ------------------------------------------------------------

    if isinstance(obj, dict):

        for value in obj.values():

            yield from _walk_object(
                value,
                visited,
                depth + 1,
                max_depth,
            )

        return


    # ------------------------------------------------------------
    # Lists / tuples / sets
    # ------------------------------------------------------------

    if isinstance(
        obj,
        (
            list,
            tuple,
            set,
        ),
    ):

        for value in obj:

            yield from _walk_object(
                value,
                visited,
                depth + 1,
                max_depth,
            )

        return


    # ------------------------------------------------------------
    # NumPy containers
    # ------------------------------------------------------------

    if isinstance(
        obj,
        np.ndarray,
    ):

        return


    # ------------------------------------------------------------
    # sklearn's named_steps
    # ------------------------------------------------------------

    if hasattr(
        obj,
        "named_steps",
    ):

        try:

            for value in (
                obj.named_steps.values()
            ):

                yield from _walk_object(
                    value,
                    visited,
                    depth + 1,
                    max_depth,
                )

        except Exception:

            pass


    # ------------------------------------------------------------
    # sklearn's calibrated estimators
    # ------------------------------------------------------------

    for attribute in (
        "calibrated_classifiers_",
        "estimators_",
    ):

        try:

            value = getattr(
                obj,
                attribute,
                None,
            )

            if value is not None:

                yield from _walk_object(
                    value,
                    visited,
                    depth + 1,
                    max_depth,
                )

        except Exception:

            pass


    # ------------------------------------------------------------
    # Common nested estimator attributes
    # ------------------------------------------------------------

    for attribute in (
        "estimator",
        "base_estimator",
        "transformer",
        "transformer_",
        "classifier",
        "classifier_",
        "model",
        "model_",
        "steps",
        "transformers",
        "transformers_",
    ):

        try:

            value = getattr(
                obj,
                attribute,
                None,
            )

            if value is not None:

                yield from _walk_object(
                    value,
                    visited,
                    depth + 1,
                    max_depth,
                )

        except Exception:

            pass


    # ------------------------------------------------------------
    # Generic __dict__ fallback
    #
    # This is the important part.
    # It allows us to find fitted objects hidden inside sklearn
    # objects even if their attribute name changed between versions.
    # ------------------------------------------------------------

    try:

        attributes = vars(
            obj
        )

    except Exception:

        attributes = {}


    for name, value in attributes.items():

        # Ignore large learned arrays and sparse matrices.
        if name.endswith("_"):

            if isinstance(
                value,
                (
                    np.ndarray,
                    list,
                    tuple,
                ),
            ):

                # Lists may still contain sklearn estimators,
                # so only skip obvious numeric arrays.

                if isinstance(
                    value,
                    np.ndarray,
                ):

                    continue

        # Avoid primitive values.
        if isinstance(
            value,
            (
                str,
                bytes,
                int,
                float,
                bool,
            ),
        ):

            continue

        yield from _walk_object(
            value,
            visited,
            depth + 1,
            max_depth,
        )


# ================================================================
# FIND TF-IDF VECTORIZER
# ================================================================

def find_vectorizer(
    model: Any,
):
    """
    Find a fitted TF-IDF vectorizer anywhere inside the model.
    """

    for obj in _walk_object(model):

        class_name = (
            obj.__class__.__name__
            .lower()
        )


        # Strong identification
        if (
            "tfidf" in class_name
            and hasattr(
                obj,
                "get_feature_names_out",
            )
            and hasattr(
                obj,
                "transform",
            )
        ):

            return obj


        # Fallback for vectorizers whose class name may differ
        if (
            hasattr(
                obj,
                "vocabulary_",
            )
            and hasattr(
                obj,
                "idf_",
            )
            and hasattr(
                obj,
                "get_feature_names_out",
            )
            and hasattr(
                obj,
                "transform",
            )
        ):

            return obj


    return None


# ================================================================
# FIND LINEAR CLASSIFIER
# ================================================================

def find_linear_estimators(
    model: Any,
):
    """
    Find all fitted linear estimators exposing coef_.
    """

    estimators = []

    seen = set()


    for obj in _walk_object(model):

        if not hasattr(
            obj,
            "coef_",
        ):

            continue


        object_id = id(obj)

        if object_id in seen:

            continue

        seen.add(
            object_id
        )


        try:

            coefficients = np.asarray(
                obj.coef_,
                dtype=float,
            )

        except Exception:

            continue


        if coefficients.ndim != 1 and (
            coefficients.ndim != 2
        ):

            continue


        estimators.append(
            obj
        )


    return estimators


# ================================================================
# GET COEFFICIENTS
# ================================================================

def get_average_coefficients(
    estimators,
):
    """
    Convert one or more LinearSVC coefficient arrays into one
    representative coefficient vector.

    For calibrated binary classifiers, the average across the
    fitted estimators is used.
    """

    coefficient_arrays = []


    for estimator in estimators:

        try:

            coefficients = np.asarray(
                estimator.coef_,
                dtype=float,
            )

        except Exception:

            continue


        # Binary classifier:
        # (1, n_features)

        if coefficients.ndim == 2:

            if coefficients.shape[0] != 1:

                continue

            coefficients = (
                coefficients[0]
            )


        if coefficients.ndim != 1:

            continue


        coefficient_arrays.append(
            coefficients
        )


    if not coefficient_arrays:

        return None


    feature_count = len(
        coefficient_arrays[0]
    )


    compatible = [

        coefficients

        for coefficients
        in coefficient_arrays

        if len(coefficients)
        == feature_count
    ]


    if not compatible:

        return None


    if len(compatible) == 1:

        return compatible[0]


    return np.mean(
        np.vstack(
            compatible
        ),
        axis=0,
    )


# ================================================================
# FORMAT FEATURE
# ================================================================

def format_feature(
    feature: str,
) -> str:

    return (
        str(feature)
        .replace(
            "\n",
            "\\n",
        )
        .replace(
            "\r",
            "\\r",
        )
        .replace(
            " ",
            "␠",
        )
    )


# ================================================================
# EXPLAIN URL
# ================================================================

def explain_url(
    model: Any,
    url: str,
    top_n: int = 8,
):
    """
    Explain an individual URL using active TF-IDF features.

    contribution =
        TF-IDF value × Linear SVM coefficient

    Positive contribution:
        pushes the decision toward phishing.

    Negative contribution:
        pushes the decision toward legitimate.
    """

    # ------------------------------------------------------------
    # Find vectorizer
    # ------------------------------------------------------------

    vectorizer = find_vectorizer(
        model
    )


    if vectorizer is None:

        return {
            "available": False,

            "reason":
                "TF-IDF vectorizer could not be located "
                "inside the saved model.",

            "positive": [],

            "negative": [],
        }


    # ------------------------------------------------------------
    # Find classifier
    # ------------------------------------------------------------

    estimators = (
        find_linear_estimators(
            model
        )
    )


    if not estimators:

        return {
            "available": False,

            "reason":
                "Linear SVM estimator could not be located "
                "inside the saved model.",

            "positive": [],

            "negative": [],
        }


    # ------------------------------------------------------------
    # Average coefficients
    # ------------------------------------------------------------

    coefficients = (
        get_average_coefficients(
            estimators
        )
    )


    if coefficients is None:

        return {
            "available": False,

            "reason":
                "Linear SVM coefficients could not be extracted.",

            "positive": [],

            "negative": [],
        }


    # ------------------------------------------------------------
    # Transform URL
    # ------------------------------------------------------------

    try:

        transformed = (
            vectorizer.transform(
                [url]
            )
        )


        feature_names = np.asarray(
            vectorizer
            .get_feature_names_out()
        )

    except Exception as error:

        return {
            "available": False,

            "reason":
                f"Could not transform URL: {error}",

            "positive": [],

            "negative": [],
        }


    # ------------------------------------------------------------
    # Verify dimensions
    # ------------------------------------------------------------

    if (
        transformed.shape[1]
        != len(coefficients)
    ):

        return {
            "available": False,

            "reason":
                (
                    "TF-IDF feature count "
                    f"({transformed.shape[1]}) does not match "
                    "SVM coefficient count "
                    f"({len(coefficients)})."
                ),

            "positive": [],

            "negative": [],
        }


    # ------------------------------------------------------------
    # Get active features
    # ------------------------------------------------------------

    row = transformed[0]

    indices = row.indices

    values = row.data


    records = []


    for feature_index, tfidf_value in zip(
        indices,
        values,
    ):

        coefficient = float(
            coefficients[
                feature_index
            ]
        )


        contribution = (
            float(tfidf_value)
            * coefficient
        )


        if contribution == 0:

            continue


        records.append({

            "feature":
                str(
                    feature_names[
                        feature_index
                    ]
                ),

            "tfidf":
                float(tfidf_value),

            "coefficient":
                coefficient,

            "contribution":
                contribution,
        })


    # ------------------------------------------------------------
    # Split by direction
    # ------------------------------------------------------------

    positive = sorted(

        [
            record
            for record in records
            if record[
                "contribution"
            ] > 0
        ],

        key=lambda record:
            record[
                "contribution"
            ],

        reverse=True,
    )


    negative = sorted(

        [
            record
            for record in records
            if record[
                "contribution"
            ] < 0
        ],

        key=lambda record:
            record[
                "contribution"
            ],
    )


    return {

        "available":
            True,

        "reason":
            "",

        "positive":
            positive[:top_n],

        "negative":
            negative[:top_n],
    }


# ================================================================
# DEBUG MODEL STRUCTURE
# ================================================================

def inspect_model_structure(
    model: Any,
):
    """
    Print useful information about the actual saved model.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "SCAMSHIELD MODEL STRUCTURE"
    )

    print(
        "=" * 70
    )


    print(
        "\nRoot object:"
    )

    print(
        type(model).__name__
    )


    vectorizer = find_vectorizer(
        model
    )


    if vectorizer is not None:

        print(
            "\n✓ TF-IDF vectorizer found:"
        )

        print(
            type(vectorizer).__name__
        )

        try:

            print(
                "Vocabulary size:",
                len(
                    vectorizer
                    .get_feature_names_out()
                )
            )

        except Exception:

            pass

    else:

        print(
            "\n✗ TF-IDF vectorizer NOT found."
        )


    estimators = (
        find_linear_estimators(
            model
        )
    )


    print(
        "\nLinear estimators found:",
        len(estimators)
    )


    for index, estimator in enumerate(
        estimators,
        start=1,
    ):

        print(
            f"  {index}. "
            f"{type(estimator).__name__}"
        )


    print(
        "=" * 70
    )