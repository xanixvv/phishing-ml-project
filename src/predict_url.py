# ================================================================
# ScamShield India
# Real-Time URL Prediction
# ================================================================

import re
import math
import ipaddress
import joblib

from pathlib import Path
from urllib.parse import urlparse


# ================================================================
# CONFIGURATION
# ================================================================

MODEL_PATH = Path(
    "outputs/final_model/scamshield_phishing_model.joblib"
)


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
    "paypal",
    "ebayisapi"
]


# ================================================================
# URL SHORTENERS
# ================================================================

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
    Calculate Shannon entropy.
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
# IP ADDRESS CHECK
# ================================================================

def is_ip_address(hostname):
    """
    Check whether hostname is an IP address.
    """

    if not hostname:
        return False

    try:

        ipaddress.ip_address(
            hostname
        )

        return True

    except ValueError:

        return False


# ================================================================
# URL INDICATOR ANALYSIS
# ================================================================

def analyze_url(url):
    """
    Generate human-readable security indicators.

    IMPORTANT:
    These indicators are supporting explanations based on
    URL characteristics. They are NOT direct explanations
    of individual SVM coefficients.
    """

    indicators = []

    url = url.strip()

    # ------------------------------------------------------------
    # Parse URL
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

    hostname = (
        parsed.hostname or ""
    ).lower()

    # ------------------------------------------------------------
    # URL length
    # ------------------------------------------------------------

    if len(url) > 100:

        indicators.append(
            (
                "HIGH",
                f"Very long URL ({len(url)} characters)"
            )
        )

    elif len(url) > 60:

        indicators.append(
            (
                "MEDIUM",
                f"Long URL ({len(url)} characters)"
            )
        )

    else:

        indicators.append(
            (
                "LOW",
                f"URL length is {len(url)} characters"
            )
        )

    # ------------------------------------------------------------
    # HTTPS
    # ------------------------------------------------------------

    if parsed.scheme.lower() == "https":

        indicators.append(
            (
                "LOW",
                "HTTPS is enabled"
            )
        )

    else:

        indicators.append(
            (
                "HIGH",
                "URL does not use HTTPS"
            )
        )

    # ------------------------------------------------------------
    # IP address
    # ------------------------------------------------------------

    if is_ip_address(hostname):

        indicators.append(
            (
                "HIGH",
                "Domain is an IP address"
            )
        )

    # ------------------------------------------------------------
    # Subdomains
    # ------------------------------------------------------------

    hostname_parts = [
        part
        for part in hostname.split(".")
        if part
    ]

    subdomain_count = max(
        len(hostname_parts) - 2,
        0
    )

    if subdomain_count >= 3:

        indicators.append(
            (
                "HIGH",
                f"Multiple subdomains detected ({subdomain_count})"
            )
        )

    elif subdomain_count == 2:

        indicators.append(
            (
                "MEDIUM",
                "Multiple subdomains detected"
            )
        )

    # ------------------------------------------------------------
    # @ symbol
    # ------------------------------------------------------------

    if "@" in url:

        indicators.append(
            (
                "HIGH",
                "Contains '@' character"
            )
        )

    # ------------------------------------------------------------
    # Punycode
    # ------------------------------------------------------------

    if "xn--" in hostname:

        indicators.append(
            (
                "HIGH",
                "Punycode detected in domain"
            )
        )

    # ------------------------------------------------------------
    # URL shortener
    # ------------------------------------------------------------

    is_shortener = False

    for shortener in SHORTENER_PATTERNS:

        if shortener in hostname:

            is_shortener = True
            break

    if is_shortener:

        indicators.append(
            (
                "MEDIUM",
                "Known URL-shortening domain detected"
            )
        )

    # ------------------------------------------------------------
    # Suspicious keywords
    # ------------------------------------------------------------

    found_keywords = []

    lower_url = url.lower()

    for keyword in SUSPICIOUS_KEYWORDS:

        if keyword in lower_url:

            found_keywords.append(
                keyword
            )

    if found_keywords:

        unique_keywords = list(
            dict.fromkeys(
                found_keywords
            )
        )

        indicators.append(
            (
                "HIGH",
                "Suspicious keywords: "
                + ", ".join(
                    unique_keywords[:6]
                )
            )
        )

    # ------------------------------------------------------------
    # Digits
    # ------------------------------------------------------------

    digit_count = sum(
        char.isdigit()
        for char in url
    )

    if digit_count >= 10:

        indicators.append(
            (
                "HIGH",
                f"High number of digits ({digit_count})"
            )
        )

    elif digit_count >= 5:

        indicators.append(
            (
                "MEDIUM",
                f"Several digits present ({digit_count})"
            )
        )

    # ------------------------------------------------------------
    # Special characters
    # ------------------------------------------------------------

    special_count = sum(
        not char.isalnum()
        for char in url
    )

    special_ratio = (
        special_count / len(url)
        if len(url) > 0
        else 0
    )

    if special_ratio > 0.25:

        indicators.append(
            (
                "HIGH",
                "High proportion of special characters"
            )
        )

    elif special_ratio > 0.15:

        indicators.append(
            (
                "MEDIUM",
                "Elevated special-character usage"
            )
        )

    # ------------------------------------------------------------
    # Encoded characters
    # ------------------------------------------------------------

    encoded_count = len(
        re.findall(
            r"%[0-9a-fA-F]{2}",
            url
        )
    )

    if encoded_count >= 5:

        indicators.append(
            (
                "HIGH",
                f"Multiple encoded characters ({encoded_count})"
            )
        )

    elif encoded_count > 0:

        indicators.append(
            (
                "MEDIUM",
                f"Encoded characters detected ({encoded_count})"
            )
        )

    # ------------------------------------------------------------
    # Repeated symbols
    # ------------------------------------------------------------

    repeated_patterns = [
        "..",
        "--",
        "__"
    ]

    repeated_found = []

    for pattern in repeated_patterns:

        if pattern in url:

            repeated_found.append(
                pattern
            )

    if repeated_found:

        indicators.append(
            (
                "MEDIUM",
                "Repeated symbols detected: "
                + ", ".join(
                    repeated_found
                )
            )
        )

    # ------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------

    url_entropy = calculate_entropy(
        url
    )

    if url_entropy >= 5.0:

        indicators.append(
            (
                "MEDIUM",
                f"High URL character entropy ({url_entropy:.2f})"
            )
        )

    return indicators


# ================================================================
# RISK LEVEL
# ================================================================

def get_risk_level(probability):
    """
    Convert calibrated phishing probability into
    an application risk category.
    """

    if probability >= 0.90:

        return "CRITICAL", "🚨"

    if probability >= 0.70:

        return "HIGH", "⚠️"

    if probability >= 0.40:

        return "MEDIUM", "🟠"

    if probability >= 0.15:

        return "LOW", "🟡"

    return "VERY LOW", "🟢"


# ================================================================
# LOAD MODEL
# ================================================================

print(
    "=" * 70
)

print(
    "SCAMSHIELD INDIA"
)

print(
    "AI-Powered Phishing URL Detection"
)

print(
    "=" * 70
)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        "\nModel not found.\n"
        f"Expected location:\n{MODEL_PATH}\n"
        "\nRun final_model.py first."
    )


print(
    "\nLoading trained model..."
)

model = joblib.load(
    MODEL_PATH
)

print(
    "✓ Model loaded successfully."
)


# ================================================================
# PREDICTION FUNCTION
# ================================================================

def predict_url(url):
    """
    Predict phishing probability for a URL.
    """

    probability = (
        model
        .predict_proba(
            [url]
        )[0, 1]
    )

    prediction = int(
        probability >= 0.50
    )

    risk_level, icon = (
        get_risk_level(
            probability
        )
    )

    indicators = analyze_url(
        url
    )

    return {
        "probability": probability,
        "prediction": prediction,
        "risk_level": risk_level,
        "icon": icon,
        "indicators": indicators
    }


# ================================================================
# DISPLAY RESULT
# ================================================================

def display_result(
    url,
    result
):

    probability = (
        result["probability"]
    )

    prediction = (
        result["prediction"]
    )

    risk_level = (
        result["risk_level"]
    )

    icon = (
        result["icon"]
    )

    indicators = (
        result["indicators"]
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "ANALYSIS RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nURL:"
    )

    print(
        url
    )


    print(
        "\n" + "-" * 70
    )


    if prediction == 1:

        print(
            f"{icon} PHISHING DETECTED"
        )

    else:

        print(
            f"{icon} LIKELY LEGITIMATE"
        )


    print(
        f"\nPhishing Probability : "
        f"{probability * 100:.2f}%"
    )

    print(
        f"Risk Level           : "
        f"{risk_level}"
    )


    print(
        "\n" + "-" * 70
    )

    print(
        "SECURITY INDICATORS"
    )

    print(
        "-" * 70
    )


    if not indicators:

        print(
            "No notable URL indicators detected."
        )

    else:

        for severity, message in indicators:

            print(
                f"[{severity:6}] {message}"
            )


    print(
        "\n" + "=" * 70
    )


# ================================================================
# INTERACTIVE LOOP
# ================================================================

while True:

    print()

    url = input(
        "Enter a URL (or type 'exit'): "
    ).strip()


    if url.lower() in [
        "exit",
        "quit",
        "q"
    ]:

        print(
            "\nScamShield closed."
        )

        break


    if not url:

        print(
            "Please enter a URL."
        )

        continue


    try:

        result = predict_url(
            url
        )

        display_result(
            url,
            result
        )

    except Exception as error:

        print(
            "\nPrediction failed:"
        )

        print(
            error
        )   