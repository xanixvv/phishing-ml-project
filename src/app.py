from __future__ import annotations

import html
import ipaddress
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import cv2
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# Optional: your working explainability module.
try:
    if (Path(__file__).resolve().parent / "src").exists():
        from src.explainability import explain_url, format_feature
    else:
        from explainability import explain_url, format_feature
    EXPLAINABILITY_AVAILABLE = True
except Exception:
    EXPLAINABILITY_AVAILABLE = False


# ================================================================
# PATHS
# ================================================================

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent if APP_DIR.name.lower() == "src" else APP_DIR

MODEL_PATH = (
    ROOT
    / "outputs"
    / "final_model"
    / "scamshield_phishing_model.joblib"
)

MODEL_METADATA_PATH = (
    ROOT
    / "outputs"
    / "final_model"
    / "model_metadata.csv"
)

TEST_PREDICTIONS_PATH = (
    ROOT
    / "outputs"
    / "final_model"
    / "test_predictions.csv"
)

ADVANCED_RESULTS_PATH = (
    ROOT
    / "outputs"
    / "advanced_model"
    / "advanced_model_comparison.csv"
)

MODEL_RESULTS_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "model_comparison.csv"
)

ROBUSTNESS_RESULTS_PATH = (
    ROOT
    / "outputs"
    / "robustness"
    / "robustness_comparison.csv"
)

HISTORY_DB = ROOT / "phishing_ml_history.db"


# ================================================================
# PAGE
# ================================================================

st.set_page_config(
    page_title="Phishing ML Project",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# STYLING
# ================================================================

st.markdown(
    """
<style>
:root {
    --bg: #070a12;
    --panel: #0f1523;
    --panel-2: #131b2d;
    --border: rgba(148, 163, 184, 0.14);
    --muted: #8e9aad;
    --text: #eef2ff;
    --purple: #8b5cf6;
    --purple-2: #a78bfa;
    --green: #34d399;
    --red: #fb7185;
    --amber: #fbbf24;
}

.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(99,102,241,.11), transparent 27%),
        radial-gradient(circle at 100% 5%, rgba(168,85,247,.09), transparent 22%),
        var(--bg);
}

.block-container {
    max-width: 1480px;
    padding-top: 1.45rem;
    padding-bottom: 4rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f1a 0%, #070a12 100%);
    border-right: 1px solid rgba(148,163,184,.08);
}

.hero {
    padding: 30px 34px;
    border-radius: 24px;
    background:
        linear-gradient(135deg, rgba(17,24,39,.98), rgba(10,14,25,.98));
    border: 1px solid var(--border);
    box-shadow: 0 20px 70px rgba(0,0,0,.20);
    margin-bottom: 24px;
}

.hero-kicker {
    color: var(--purple-2);
    text-transform: uppercase;
    letter-spacing: 1.8px;
    font-size: .74rem;
    font-weight: 800;
    margin-bottom: 9px;
}

.hero-title {
    font-size: 3rem;
    line-height: 1;
    font-weight: 850;
    letter-spacing: -1.5px;
}

.hero-title span {
    color: var(--purple-2);
}

.hero-subtitle {
    color: #98a4b7;
    font-size: 1rem;
    margin-top: 12px;
    max-width: 900px;
}

.card {
    padding: 20px 21px;
    border-radius: 20px;
    background: rgba(15,21,35,.86);
    border: 1px solid var(--border);
    box-shadow: 0 12px 38px rgba(0,0,0,.12);
}

.card-title {
    font-size: 1.02rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.card-muted {
    color: var(--muted);
    font-size: .88rem;
    line-height: 1.55;
}

.metric-card {
    padding: 17px 18px;
    min-height: 108px;
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(17,24,39,.95), rgba(13,18,30,.95));
    border: 1px solid var(--border);
}

.metric-label {
    color: var(--muted);
    font-size: .71rem;
    text-transform: uppercase;
    letter-spacing: .85px;
    font-weight: 750;
}

.metric-value {
    font-size: 1.68rem;
    font-weight: 850;
    margin-top: 7px;
}

.result-panel {
    padding: 26px;
    border-radius: 22px;
    background: linear-gradient(145deg, rgba(17,24,39,.96), rgba(12,17,29,.96));
    border: 1px solid var(--border);
}

.small-pill {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 999px;
    background: rgba(139,92,246,.12);
    color: #c4b5fd;
    font-size: .72rem;
    font-weight: 750;
    margin-right: 5px;
    margin-bottom: 5px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: .76rem;
    margin-top: 42px;
}

hr {
    border-color: rgba(148,163,184,.10);
}

div[data-testid="stTextInput"] input {
    border-radius: 13px;
}

button[kind="primary"] {
    border-radius: 12px;
}

[data-testid="stMetricValue"] {
    font-weight: 800;
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
}



/* Polished product layer */
.app-status-row {
    display:flex;
    gap:8px;
    flex-wrap:wrap;
    margin-top:12px;
}

.mini-pill {
    display:inline-flex;
    align-items:center;
    gap:7px;
    padding:6px 10px;
    border-radius:999px;
    background:rgba(139,92,246,.10);
    border:1px solid rgba(139,92,246,.20);
    color:#c4b5fd;
    font-size:.71rem;
    font-weight:700;
}

.mini-pill.green {
    background:rgba(52,211,153,.08);
    border-color:rgba(52,211,153,.16);
    color:#86efac;
}

.mini-pill.gray {
    background:rgba(148,163,184,.07);
    border-color:rgba(148,163,184,.12);
    color:#a9b3c3;
}

.quick-card {
    padding:22px;
    min-height:148px;
    border-radius:20px;
    background:linear-gradient(145deg,rgba(17,24,39,.97),rgba(10,14,25,.97));
    border:1px solid rgba(148,163,184,.12);
    box-shadow:0 12px 34px rgba(0,0,0,.12);
}

.quick-card:hover {
    border-color:rgba(139,92,246,.32);
}

.verdict {
    padding:24px 26px;
    border-radius:20px;
    border:1px solid rgba(148,163,184,.12);
    margin-bottom:14px;
}

.verdict-danger {
    background:linear-gradient(135deg,rgba(127,29,29,.28),rgba(29,13,22,.34));
    border-color:rgba(248,113,113,.24);
}

.verdict-safe {
    background:linear-gradient(135deg,rgba(6,78,59,.25),rgba(9,29,27,.34));
    border-color:rgba(52,211,153,.22);
}

.verdict-warning {
    background:linear-gradient(135deg,rgba(120,53,15,.24),rgba(36,25,11,.34));
    border-color:rgba(251,191,36,.22);
}

.verdict-label {
    color:#8d99ab;
    font-size:.70rem;
    text-transform:uppercase;
    letter-spacing:1.2px;
    font-weight:800;
}

.verdict-title {
    font-size:1.62rem;
    font-weight:850;
    margin-top:5px;
}

.url-box {
    padding:14px 16px;
    border-radius:14px;
    background:rgba(6,9,16,.62);
    border:1px solid rgba(148,163,184,.10);
    word-break:break-word;
}

</style>
""",
    unsafe_allow_html=True,
)


# ================================================================
# DATA HELPERS
# ================================================================

@st.cache_data
def load_csv(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_column(df: pd.DataFrame, names: set[str]):
    for col in df.columns:
        if col.lower() in names:
            return col
    return None


model_metadata = load_csv(MODEL_METADATA_PATH)
test_predictions = load_csv(TEST_PREDICTIONS_PATH)
advanced_results = load_csv(ADVANCED_RESULTS_PATH)
model_results = load_csv(MODEL_RESULTS_PATH)
robustness_results = load_csv(ROBUSTNESS_RESULTS_PATH)

metric_map = {}
if (
    model_metadata is not None
    and {"metric", "value"}.issubset(model_metadata.columns)
):
    metric_map = dict(
        zip(
            model_metadata["metric"],
            pd.to_numeric(model_metadata["value"], errors="coerce"),
        )
    )

comparison_source = (
    advanced_results.copy()
    if advanced_results is not None
    else model_results.copy()
    if model_results is not None
    else None
)


# ================================================================
# MODEL
# ================================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found:\n{MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error("The production phishing model could not be loaded.")
    st.code(str(exc))
    st.stop()


# ================================================================
# URL ANALYSIS
# ================================================================

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "verify", "verification",
    "account", "secure", "security", "update", "confirm",
    "confirmation", "password", "passwd", "credential",
    "credentials", "bank", "banking", "payment", "billing",
    "invoice", "wallet", "support", "recover", "recovery",
    "unlock", "suspend", "suspended", "authenticate",
    "authentication", "paypal", "ebayisapi",
]

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "cutt.ly", "shorturl.at", "rebrand.ly",
}


def normalize_url(url: str) -> str:
    value = str(url).strip()
    if not value:
        return ""

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        value,
    ):
        value = "https://" + value

    return value


def parse_url(url: str):
    return urlparse(normalize_url(url))


def is_ip_host(hostname: str) -> bool:
    if not hostname:
        return False

    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0

    counts: dict[str, int] = {}

    for char in text:
        counts[char] = counts.get(char, 0) + 1

    n = len(text)
    entropy = 0.0

    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)

    return entropy


def analyze_url(url: str):
    parsed = parse_url(url)
    hostname = (parsed.hostname or "").lower()

    findings = []

    # HTTPS
    if parsed.scheme.lower() == "https":
        findings.append(
            ("LOW", "HTTPS is enabled")
        )
    else:
        findings.append(
            ("HIGH", "URL does not use HTTPS")
        )

    # IP host
    if is_ip_host(hostname):
        findings.append(
            ("HIGH", "The hostname is an IP address")
        )

    # URL length
    length = len(url)

    if length >= 100:
        findings.append(
            ("HIGH", f"Very long URL ({length} characters)")
        )
    elif length >= 60:
        findings.append(
            ("MEDIUM", f"Long URL ({length} characters)")
        )

    # Suspicious terms
    lower_url = url.lower()

    keywords = []
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in lower_url:
            keywords.append(keyword)

    if keywords:
        findings.append(
            (
                "HIGH",
                "Sensitive-action keywords detected: "
                + ", ".join(dict.fromkeys(keywords)[:8]),
            )
        )

    # @
    if "@" in url:
        findings.append(
            ("HIGH", "Contains an '@' symbol")
        )

    # Punycode
    if "xn--" in hostname:
        findings.append(
            ("HIGH", "Punycode detected in hostname")
        )

    # Shortener
    for shortener in SHORTENERS:
        if hostname == shortener or hostname.endswith("." + shortener):
            findings.append(
                (
                    "MEDIUM",
                    f"URL shortener detected ({shortener})",
                )
            )
            break

    # Encoded characters
    encoded = len(
        re.findall(
            r"%[0-9a-fA-F]{2}",
            url,
        )
    )

    if encoded >= 5:
        findings.append(
            ("HIGH", f"Multiple encoded characters ({encoded})")
        )
    elif encoded > 0:
        findings.append(
            ("MEDIUM", f"Encoded characters present ({encoded})")
        )

    # Digits
    digits = sum(char.isdigit() for char in url)
    digit_ratio = digits / len(url) if url else 0

    if digits >= 10:
        findings.append(
            ("HIGH", f"High digit count ({digits})")
        )
    elif digit_ratio > 0.12:
        findings.append(
            (
                "MEDIUM",
                f"Elevated digit density ({digit_ratio * 100:.1f}%)",
            )
        )

    # Unusual special characters only.
    normal_url_chars = set(
        ":/?#[]@!$&'()*+,;=-._~%"
    )

    unusual_special = sum(
        not char.isalnum() and char not in normal_url_chars
        for char in url
    )

    unusual_ratio = (
        unusual_special / len(url)
        if url
        else 0
    )

    if unusual_ratio > 0.12:
        findings.append(
            ("HIGH", "Unusual special-character usage detected")
        )
    elif unusual_ratio > 0.05:
        findings.append(
            ("MEDIUM", "Some unusual special characters detected")
        )

    # Deep path
    path_parts = [
        part
        for part in parsed.path.split("/")
        if part
    ]

    if len(path_parts) >= 5:
        findings.append(
            (
                "MEDIUM",
                f"Deep URL path ({len(path_parts)} levels)",
            )
        )

    # Entropy
    entropy = shannon_entropy(url)

    if entropy >= 5.0:
        findings.append(
            (
                "MEDIUM",
                f"High character entropy ({entropy:.2f})",
            )
        )

    return findings


def predict_url(url: str):
    normalized = normalize_url(url)

    if not normalized:
        raise ValueError("URL cannot be empty.")

    probability = float(
        model.predict_proba([normalized])[0, 1]
    )

    prediction = int(probability >= 0.50)

    risk_score = round(probability * 100, 2)

    if risk_score >= 90:
        risk = "CRITICAL"
    elif risk_score >= 70:
        risk = "HIGH"
    elif risk_score >= 40:
        risk = "MODERATE"
    elif risk_score >= 15:
        risk = "LOW"
    else:
        risk = "VERY LOW"

    return {
        "url": normalized,
        "probability": probability,
        "risk_score": risk_score,
        "prediction": prediction,
        "classification": (
            "Phishing" if prediction else "Legitimate"
        ),
        "risk": risk,
        "findings": analyze_url(normalized),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ================================================================
# HISTORY
# ================================================================

def db_connect():
    conn = sqlite3.connect(HISTORY_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            url TEXT NOT NULL,
            classification TEXT NOT NULL,
            probability REAL NOT NULL,
            risk_score REAL NOT NULL,
            risk_level TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def save_scan(result: dict):
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO scans
            (created_at, url, classification, probability, risk_score, risk_level)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                result["url"],
                result["classification"],
                result["probability"],
                result["risk_score"],
                result["risk"],
            ),
        )
        conn.commit()


def get_history(limit: int = 250):
    with db_connect() as conn:
        return pd.read_sql_query(
            """
            SELECT
                id,
                created_at,
                url,
                classification,
                probability,
                risk_score,
                risk_level
            FROM scans
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


def get_history_summary():
    history = get_history(10000)

    if history.empty:
        return 0, 0, 0.0

    total = len(history)

    phishing = int(
        (
            history["classification"]
            == "Phishing"
        ).sum()
    )

    average_risk = float(
        history["risk_score"].mean()
    )

    return total, phishing, average_risk


# ================================================================
# QR DECODER
# ================================================================

def decode_qr_codes(image):
    detector = cv2.QRCodeDetector()
    found = []

    try:
        success, decoded, _, _ = (
            detector.detectAndDecodeMulti(image)
        )

        if success and decoded:
            for value in decoded:
                value = str(value).strip()
                if value and value not in found:
                    found.append(value)
    except Exception:
        pass

    if not found:
        try:
            value, _, _ = detector.detectAndDecode(image)
            value = str(value).strip()
            if value:
                found.append(value)
        except Exception:
            pass

    return found


# ================================================================
# REPORT
# ================================================================

def build_html_report(result: dict) -> bytes:
    rows = "".join(
        f"<li><b>{html.escape(severity)}</b> "
        f"{html.escape(message)}</li>"
        for severity, message in result["findings"]
    )

    document = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Phishing ML Project - URL Security Report</title>
<style>
body {{
    font-family: Inter, Arial, sans-serif;
    background:#080b14;
    color:#eef2ff;
    margin:0;
    padding:40px;
}}
.card {{
    background:#121927;
    border:1px solid #293247;
    border-radius:18px;
    padding:24px;
    margin-bottom:18px;
}}
.score {{
    font-size:54px;
    font-weight:800;
}}
.muted {{
    color:#94a3b8;
}}
code {{
    word-break:break-word;
    white-space:pre-wrap;
}}
</style>
</head>
<body>
<div class="card">
    <h1>Phishing ML Project</h1>
    <div class="muted">URL security analysis report · generated locally</div>
</div>
<div class="card">
    <h2>Result</h2>
    <div class="score">{result["risk_score"]:.1f}/100</div>
    <p><b>Classification:</b> {html.escape(result["classification"])}</p>
    <p><b>Risk:</b> {html.escape(result["risk"])}</p>
    <p><b>Phishing probability:</b> {result["probability"] * 100:.2f}%</p>
</div>
<div class="card">
    <h2>URL</h2>
    <code>{html.escape(result["url"])}</code>
</div>
<div class="card">
    <h2>Security indicators</h2>
    <ul>{rows}</ul>
</div>
<div class="card">
    <div class="muted">
        Model: Character TF-IDF + Calibrated Linear SVM
    </div>
</div>
</body>
</html>
"""
    return document.encode("utf-8")


# ================================================================
# UI HELPERS
# ================================================================

def metric_card(label: str, value: str):
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">{html.escape(label)}</div>
    <div class="metric-value">{html.escape(value)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_hero(
    kicker: str,
    title: str,
    subtitle: str,
):
    st.markdown(
        f"""
<div class="hero">
    <div class="hero-kicker">{html.escape(kicker)}</div>
    <div class="hero-title">{html.escape(title.split(" ")[0])} <span>{html.escape(" ".join(title.split(" ")[1:]))}</span></div>
    <div class="hero-subtitle">{html.escape(subtitle)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_verdict(result: dict):
    phishing = result["prediction"] == 1
    risk = result["risk"]

    if phishing:
        css = "verdict verdict-danger"
        icon = "🚨"
        title = "PHISHING DETECTED"
        subtitle = "The trained classifier estimates a high likelihood of phishing."
    elif risk in {"MODERATE", "LOW"}:
        css = "verdict verdict-warning"
        icon = "⚠️"
        title = "REVIEW RECOMMENDED"
        subtitle = "The result is not confidently safe. Inspect the URL before opening it."
    else:
        css = "verdict verdict-safe"
        icon = "🟢"
        title = "LIKELY LEGITIMATE"
        subtitle = "The trained classifier estimates a low phishing likelihood."

    st.markdown(
        f"""
<div class="{css}">
    <div class="verdict-label">Automated assessment</div>
    <div class="verdict-title">{icon} {title}</div>
    <div class="card-muted">{html.escape(subtitle)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_result(result: dict, show_explanation: bool = True):
    phishing = result["prediction"] == 1

    render_verdict(result)

    cols = st.columns(4)

    with cols[0]:
        metric_card("Risk score", f'{result["risk_score"]:.1f}/100')

    with cols[1]:
        metric_card(
            "Phishing probability",
            f'{result["probability"] * 100:.2f}%',
        )

    with cols[2]:
        metric_card("Risk level", result["risk"])

    with cols[3]:
        metric_card("Classification", result["classification"])

    st.markdown("### Risk level")
    st.progress(
        min(
            max(
                float(result["probability"]),
                0.0,
            ),
            1.0,
        )
    )

    st.markdown("### Security indicators")

    findings = result["findings"]

    if not findings:
        st.success(
            "No notable URL indicators were detected."
        )
    else:
        high_count = sum(1 for severity, _ in findings if severity == "HIGH")
        medium_count = sum(1 for severity, _ in findings if severity == "MEDIUM")
        low_count = sum(1 for severity, _ in findings if severity == "LOW")

        signal_cols = st.columns(3)

        with signal_cols[0]:
            metric_card("High-risk signals", str(high_count))

        with signal_cols[1]:
            metric_card("Medium signals", str(medium_count))

        with signal_cols[2]:
            metric_card("Informational", str(low_count))

        with st.expander("View security-indicator details"):
            for severity, message in findings:
                if severity == "HIGH":
                    st.error(f"🔴 {message}")
                elif severity == "MEDIUM":
                    st.warning(f"🟠 {message}")
                else:
                    st.info(f"🟢 {message}")

    st.markdown("### Analysis actions")

    action_cols = st.columns(3)

    with action_cols[0]:
        if st.button(
            "💾 Save scan",
            use_container_width=True,
            key=f"save_scan_{hash(result['url'])}",
        ):
            save_scan(result)
            st.success("Saved to local history.")

    with action_cols[1]:
        st.download_button(
            "⬇️ Security report",
            data=build_html_report(result),
            file_name="phishing_security_report.html",
            mime="text/html",
            use_container_width=True,
        )

    with action_cols[2]:
        st.download_button(
            "⬇️ JSON",
            data=json.dumps(
                result,
                indent=2,
                default=str,
            ),
            file_name="phishing_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("### Analyzed URL")
    st.markdown(
        f'<div class="url-box"><code>{html.escape(result["url"])}</code></div>',
        unsafe_allow_html=True,
    )

    if not show_explanation:
        return

    st.markdown("### 🧠 Model explainability")

    if not EXPLAINABILITY_AVAILABLE:
        st.info(
            "Explainability module is not available in this installation."
        )
        return

    st.caption(
        "These are active character-level TF-IDF contributions "
        "to the Linear SVM decision function. They are model "
        "evidence, not direct components of the calibrated probability."
    )

    explanation = explain_url(
        model,
        result["url"],
        top_n=8,
    )

    if not explanation.get("available"):
        st.info(
            explanation.get(
                "reason",
                "Feature-level explanation is unavailable.",
            )
        )
        return

    positive = explanation["positive"]
    negative = explanation["negative"]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 🔴 Phishing-side evidence")

        if positive:
            df = pd.DataFrame(
                {
                    "Pattern": [
                        format_feature(item["feature"])
                        for item in positive
                    ],
                    "Contribution": [
                        item["contribution"]
                        for item in positive
                    ],
                }
            ).sort_values("Contribution")

            fig = px.bar(
                df,
                x="Contribution",
                y="Pattern",
                orientation="h",
                text="Contribution",
                title="Learned patterns pushing toward phishing",
            )

            fig.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside",
            )

            fig.update_layout(
                template="plotly_dark",
                height=420,
                yaxis_title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"phish_explain_{hash(result['url'])}",
            )
        else:
            st.info("No active phishing-side features.")

    with c2:
        st.markdown("#### 🟢 Legitimate-side evidence")

        if negative:
            df = pd.DataFrame(
                {
                    "Pattern": [
                        format_feature(item["feature"])
                        for item in negative
                    ],
                    "Contribution": [
                        abs(item["contribution"])
                        for item in negative
                    ],
                }
            ).sort_values("Contribution")

            fig = px.bar(
                df,
                x="Contribution",
                y="Pattern",
                orientation="h",
                text="Contribution",
                title="Learned patterns pushing toward legitimate",
            )

            fig.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside",
            )

            fig.update_layout(
                template="plotly_dark",
                height=420,
                yaxis_title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"legit_explain_{hash(result['url'])}",
            )
        else:
            st.info("No active legitimate-side features.")

    rows = []

    for item in positive:
        rows.append(
            {
                "Direction": "Phishing",
                "Pattern": format_feature(item["feature"]),
                "TF-IDF": round(item["tfidf"], 5),
                "Coefficient": round(item["coefficient"], 5),
                "Contribution": round(item["contribution"], 5),
            }
        )

    for item in negative:
        rows.append(
            {
                "Direction": "Legitimate",
                "Pattern": format_feature(item["feature"]),
                "TF-IDF": round(item["tfidf"], 5),
                "Coefficient": round(item["coefficient"], 5),
                "Contribution": round(item["contribution"], 5),
            }
        )

    if rows:
        with st.expander("View learned feature details"):
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.markdown(
        """
<div style="font-size:1.22rem;font-weight:850">
    🛡️ Phishing ML Project
</div>
<div style="color:#8e9aad;font-size:.84rem;margin-top:4px">
    URL classification & threat analysis
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🔎 URL Scanner",
            "📷 QR Scanner",
            "📂 Batch Scanner",
            "🕘 History",
            "📊 ML Dashboard",
            "🧪 Research",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        '<span class="small-pill">MODEL ONLINE</span>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Character TF-IDF + Calibrated Linear SVM"
    )

    st.caption(
        "No webpage scraping required"
    )

    st.markdown(
        """
<div class="app-status-row">
    <span class="mini-pill green">● MODEL ONLINE</span>
    <span class="mini-pill gray">OFFLINE-FIRST UI</span>
</div>
""",
        unsafe_allow_html=True,
    )


# ================================================================
# OVERVIEW
# ================================================================

if page == "🏠 Overview":
    render_hero(
        "MACHINE LEARNING PROJECT",
        "Phishing ML Project",
        "A deployable phishing-URL classifier combining character-level "
        "machine learning, interpretable evidence and practical security tooling.",
    )

    total_scans, phishing_scans, average_risk = (
        get_history_summary()
    )

    st.markdown("## System overview")

    cols = st.columns(4)

    with cols[0]:
        metric_card(
            "Saved investigations",
            f"{total_scans:,}",
        )

    with cols[1]:
        metric_card(
            "Phishing detections",
            f"{phishing_scans:,}",
        )

    with cols[2]:
        metric_card(
            "Average saved risk",
            f"{average_risk:.1f}/100",
        )

    with cols[3]:
        metric_card(
            "Model status",
            "ONLINE",
        )

    st.markdown("## Quick actions")

    a1, a2, a3 = st.columns(3)

    with a1:
        if st.button(
            "🔎 Analyze a URL",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["quick_url"] = True
            st.rerun()

    with a2:
        if st.button(
            "📷 Scan a QR code",
            use_container_width=True,
        ):
            st.session_state["quick_qr"] = True
            st.rerun()

    with a3:
        if st.button(
            "📊 Open ML Dashboard",
            use_container_width=True,
        ):
            st.session_state["quick_ml"] = True
            st.rerun()

    st.markdown("## What the application does")

    cards = [
        (
            "🔎",
            "URL classification",
            "Normalizes a submitted URL, evaluates it with the trained "
            "classifier and returns a phishing probability.",
        ),
        (
            "📷",
            "QR investigation",
            "Decodes URLs from QR-code images and sends them through "
            "the same phishing model.",
        ),
        (
            "📂",
            "Batch analysis",
            "Processes a CSV of URLs and generates a downloadable "
            "classification report.",
        ),
        (
            "🧠",
            "Model evidence",
            "Shows active character-level TF-IDF contributions for "
            "individual predictions.",
        ),
    ]

    card_cols = st.columns(4)

    for col, (icon, title, description) in zip(
        card_cols,
        cards,
    ):
        with col:
            st.markdown(
                f"""
<div class="card">
    <div style="font-size:1.8rem">{icon}</div>
    <div class="card-title">{html.escape(title)}</div>
    <div class="card-muted">{html.escape(description)}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("## ML snapshot")

    if metric_map:
        snapshot = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
            "Score": [
                metric_map.get("accuracy", np.nan),
                metric_map.get("precision", np.nan),
                metric_map.get("recall", np.nan),
                metric_map.get("f1", np.nan),
                metric_map.get("roc_auc", np.nan),
                metric_map.get("pr_auc", np.nan),
            ],
        }).dropna()

        if not snapshot.empty:
            fig_snapshot = go.Figure()
            fig_snapshot.add_trace(
                go.Bar(
                    x=snapshot["Metric"],
                    y=snapshot["Score"],
                    text=snapshot["Score"],
                    texttemplate="%{text:.2%}",
                    textposition="outside",
                    name="Score",
                )
            )
            fig_snapshot.update_layout(
                template="plotly_dark",
                title="Production model performance",
                height=390,
                yaxis_range=[0, 1.02],
                xaxis_title="",
                yaxis_title="Score",
                showlegend=False,
            )
            st.plotly_chart(
                fig_snapshot,
                use_container_width=True,
            )

    st.markdown("## Current model")

    model_cols = st.columns(3)

    with model_cols[0]:
        metric_card(
            "Representation",
            "Character TF-IDF",
        )

    with model_cols[1]:
        metric_card(
            "Classifier",
            "Calibrated Linear SVM",
        )

    with model_cols[2]:
        metric_card(
            "Evaluation",
            "Domain-disjoint",
        )

    st.info(
        "The URL classifier is the trained ML component. QR decoding, "
        "human-readable indicators and reporting are supporting application layers."
    )


# ================================================================
# URL SCANNER
# ================================================================

elif page == "🔎 URL Scanner":
    render_hero(
        "LIVE INFERENCE",
        "URL Scanner",
        "Analyze a URL without visiting the target website.",
    )

    if "quick_url" in st.session_state:
        st.session_state.pop("quick_url", None)

    sample_cols = st.columns(3)

    with sample_cols[0]:
        if st.button(
            "Use safe example",
            use_container_width=True,
        ):
            st.session_state["scan_url"] = (
                "https://www.google.com"
            )

    with sample_cols[1]:
        if st.button(
            "Use phishing demo",
            use_container_width=True,
        ):
            st.session_state["scan_url"] = (
                "http://paypal-login-verify.example.com/"
                "account/secure/verification"
            )

    with sample_cols[2]:
        if st.button(
            "Use IP-based demo",
            use_container_width=True,
        ):
            st.session_state["scan_url"] = (
                "http://192.168.0.45/"
                "secure-login/account/verify"
                "?password=confirm&update=1"
            )

    with st.form("url_scan_form", clear_on_submit=False):
        url_input = st.text_input(
            "URL",
            value=st.session_state.get("scan_url", ""),
            placeholder="https://example.com/login",
            label_visibility="visible",
        )

        submitted = st.form_submit_button(
            "🔍 Analyze URL",
            type="primary",
            use_container_width=True,
        )

    st.caption(
        "The application analyzes the URL string locally; it does not open the submitted website."
    )

    if submitted:
        try:
            result = predict_url(url_input)
            st.session_state["current_result"] = result
            st.session_state["scan_url"] = result["url"]
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if "current_result" in st.session_state:
        st.divider()
        render_result(
            st.session_state["current_result"]
        )


# ================================================================
# QR SCANNER
# ================================================================

elif page == "📷 QR Scanner":
    render_hero(
        "IMAGE INPUT",
        "QR Scanner",
        "Decode QR-embedded URLs and analyze them with the same phishing classifier.",
    )

    st.markdown(
        """
<div class="app-status-row">
    <span class="mini-pill gray">📷 DETERMINISTIC QR DECODING</span>
    <span class="mini-pill gray">🔗 EXTRACT URL</span>
    <span class="mini-pill green">🧠 SAME ML CLASSIFIER</span>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="card">
    <div class="card-title">How QR analysis works</div>
    <div class="card-muted">
        QR image → OpenCV decoder → URL → character-level ML classifier.
        No separate QR training model is required.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    qr_file = st.file_uploader(
        "Upload a QR-code image",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if qr_file:
        data = qr_file.getvalue()

        image = cv2.imdecode(
            np.frombuffer(
                data,
                dtype=np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            st.error("Could not read the uploaded image.")
        else:
            st.image(
                image,
                channels="BGR",
                caption="Uploaded QR image",
                use_container_width=True,
            )

            urls = decode_qr_codes(image)

            if not urls:
                st.warning(
                    "No readable QR code was detected."
                )
            else:
                st.success(
                    f"Detected {len(urls)} unique QR URL(s)."
                )

                for index, qr_url in enumerate(
                    urls,
                    start=1,
                ):
                    with st.expander(
                        f"QR {index} · {qr_url}",
                        expanded=index == 1,
                    ):
                        try:
                            qr_result = predict_url(qr_url)
                            render_result(
                                qr_result,
                                show_explanation=False,
                            )
                        except Exception as exc:
                            st.error(str(exc))


# ================================================================
# BATCH SCANNER
# ================================================================

elif page == "📂 Batch Scanner":
    render_hero(
        "BULK ANALYSIS",
        "Batch Scanner",
        "Upload a CSV of URLs, classify them in one pass and export the results.",
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()

        candidate_columns = [
            col
            for col in batch_df.columns
            if col.lower()
            in {
                "url",
                "urls",
                "link",
                "links",
                "website",
            }
        ]

        if candidate_columns:
            url_column = candidate_columns[0]
        else:
            text_columns = (
                batch_df
                .select_dtypes(include="object")
                .columns
                .tolist()
            )

            if not text_columns:
                st.error("No URL-like column was found.")
                st.stop()

            url_column = text_columns[0]

        st.info(
            f"Detected URL column: `{url_column}`"
        )

        st.dataframe(
            batch_df.head(8),
            use_container_width=True,
            hide_index=True,
        )

        if st.button(
            "🚨 Scan all URLs",
            type="primary",
            use_container_width=True,
        ):
            records = []

            progress = st.progress(0)

            total = max(
                len(batch_df),
                1,
            )

            for index, raw_url in enumerate(
                batch_df[url_column],
                start=1,
            ):
                try:
                    result = predict_url(
                        str(raw_url).strip()
                    )

                    records.append(
                        {
                            "URL": result["url"],
                            "Prediction": result["classification"],
                            "Phishing Probability (%)": round(
                                result["probability"] * 100,
                                4,
                            ),
                            "Risk Score": result["risk_score"],
                            "Risk Level": result["risk"],
                        }
                    )

                except Exception as exc:
                    records.append(
                        {
                            "URL": str(raw_url),
                            "Prediction": "Error",
                            "Phishing Probability (%)": np.nan,
                            "Risk Score": np.nan,
                            "Risk Level": str(exc),
                        }
                    )

                progress.progress(
                    index / total
                )

            st.session_state["batch_results"] = pd.DataFrame(
                records
            )

    if "batch_results" in st.session_state:
        result_df = st.session_state["batch_results"]

        phishing_count = int(
            (
                result_df["Prediction"]
                == "Phishing"
            ).sum()
        )

        legitimate_count = int(
            (
                result_df["Prediction"]
                == "Legitimate"
            ).sum()
        )

        cols = st.columns(3)

        with cols[0]:
            metric_card(
                "URLs scanned",
                f"{len(result_df):,}",
            )

        with cols[1]:
            metric_card(
                "Phishing",
                f"{phishing_count:,}",
            )

        with cols[2]:
            metric_card(
                "Legitimate",
                f"{legitimate_count:,}",
            )

        st.markdown("### Batch results")

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download CSV report",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="phishing_batch_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

        chart_data = pd.DataFrame(
            {
                "Classification": [
                    "Phishing",
                    "Legitimate",
                ],
                "Count": [
                    phishing_count,
                    legitimate_count,
                ],
            }
        )

        fig = px.bar(
            chart_data,
            x="Classification",
            y="Count",
            text="Count",
            title="Batch Classification Summary",
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ================================================================
# HISTORY
# ================================================================

elif page == "🕘 History":
    render_hero(
        "LOCAL STORAGE",
        "Scan History",
        "Review previous investigations stored locally by this application.",
    )

    history = get_history()

    if history.empty:
        st.info(
            "No saved scans yet. Save a result from the URL scanner."
        )
    else:
        total = len(history)
        phishing = int(
            (
                history["classification"]
                == "Phishing"
            ).sum()
        )

        avg_risk = float(
            history["risk_score"].mean()
        )

        cols = st.columns(3)

        with cols[0]:
            metric_card(
                "Investigations",
                f"{total:,}",
            )

        with cols[1]:
            metric_card(
                "Phishing",
                f"{phishing:,}",
            )

        with cols[2]:
            metric_card(
                "Average risk",
                f"{avg_risk:.1f}/100",
            )

        history["created_at"] = pd.to_datetime(
            history["created_at"],
            errors="coerce",
            utc=True,
        )

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            history_filter = st.selectbox(
                "Filter result",
                ["All", "Phishing", "Legitimate"],
            )

        with filter_col2:
            search_term = st.text_input(
                "Search URL",
                placeholder="example.com",
            ).strip().lower()

        filtered = history.copy()

        if history_filter != "All":
            filtered = filtered[
                filtered["classification"] == history_filter
            ]

        if search_term:
            filtered = filtered[
                filtered["url"].str.lower().str.contains(
                    search_term,
                    na=False,
                )
            ]

        filtered["created_at"] = filtered["created_at"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )

        chart = (
            history["classification"]
            .value_counts()
            .rename_axis("Classification")
            .reset_index(name="Count")
        )

        fig = px.bar(
            chart,
            x="Classification",
            y="Count",
            text="Count",
            title="Saved Investigation Outcomes",
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ================================================================
# ML DASHBOARD
# ================================================================

elif page == "📊 ML Dashboard":
    render_hero(
        "MODEL EVIDENCE",
        "ML Dashboard",
        "Performance, discrimination, errors, probability behaviour and robustness.",
    )

    st.markdown("## Production performance")

    if metric_map:
        metrics = [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1 Score", "f1"),
            ("ROC-AUC", "roc_auc"),
            ("PR-AUC", "pr_auc"),
        ]

        cols = st.columns(6)

        for col, (label, key) in zip(
            cols,
            metrics,
        ):
            value = metric_map.get(key)

            with col:
                metric_card(
                    label,
                    (
                        f"{float(value) * 100:.2f}%"
                        if pd.notna(value)
                        else "N/A"
                    ),
                )
    else:
        st.warning(
            "model_metadata.csv was not found, so headline production metrics cannot be loaded."
        )

    # ------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------

    if (
        comparison_source is not None
        and not comparison_source.empty
    ):
        st.markdown("## Model comparison")

        numeric_columns = [
            col
            for col in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
                "PR-AUC",
            ]
            if col in comparison_source.columns
        ]

        show_columns = (
            ["Model"]
            + numeric_columns
            + (
                ["Training Time (s)"]
                if "Training Time (s)"
                in comparison_source.columns
                else []
            )
        )

        st.dataframe(
            comparison_source[show_columns].style.format(
                {
                    col: "{:.2%}"
                    for col in numeric_columns
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        long_df = (
            comparison_source[
                ["Model"] + numeric_columns
            ]
            .melt(
                id_vars="Model",
                var_name="Metric",
                value_name="Score",
            )
        )

        fig = px.bar(
            long_df,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Model Performance Comparison",
        )

        fig.update_layout(
            template="plotly_dark",
            height=560,
            yaxis_range=[0, 1.02],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ------------------------------------------------------------
    # Test predictions
    # ------------------------------------------------------------

    if test_predictions is None:
        st.info(
            "test_predictions.csv was not found. "
            "Confusion-matrix and threshold plots require saved test predictions."
        )
    else:
        actual_col = find_column(
            test_predictions,
            {"actual", "true", "label", "y_true"},
        )

        pred_col = find_column(
            test_predictions,
            {"predicted", "prediction", "pred", "y_pred"},
        )

        probability_col = find_column(
            test_predictions,
            {
                "phishing_probability",
                "probability",
                "phishing_prob",
                "score",
            },
        )

        if actual_col and pred_col:
            actual = (
                pd.to_numeric(
                    test_predictions[actual_col],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
            )

            predicted = (
                pd.to_numeric(
                    test_predictions[pred_col],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
            )

            matrix = np.zeros(
                (2, 2),
                dtype=int,
            )

            for a, p in zip(
                actual,
                predicted,
            ):
                if a in (0, 1) and p in (0, 1):
                    matrix[a, p] += 1

            st.markdown("## Error analysis")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                metric_card(
                    "True negatives",
                    f"{matrix[0, 0]:,}",
                )

            with c2:
                metric_card(
                    "False positives",
                    f"{matrix[0, 1]:,}",
                )

            with c3:
                metric_card(
                    "False negatives",
                    f"{matrix[1, 0]:,}",
                )

            with c4:
                metric_card(
                    "True positives",
                    f"{matrix[1, 1]:,}",
                )

            cm1, cm2 = st.columns(2)

            with cm1:
                fig = px.imshow(
                    matrix,
                    text_auto=True,
                    x=["Legitimate", "Phishing"],
                    y=["Legitimate", "Phishing"],
                    labels={
                        "x": "Predicted",
                        "y": "Actual",
                        "color": "Count",
                    },
                    title="Confusion Matrix",
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=470,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            with cm2:
                error_df = pd.DataFrame(
                    {
                        "Type": [
                            "True Negative",
                            "False Positive",
                            "False Negative",
                            "True Positive",
                        ],
                        "Count": [
                            matrix[0, 0],
                            matrix[0, 1],
                            matrix[1, 0],
                            matrix[1, 1],
                        ],
                    }
                )

                fig = px.bar(
                    error_df,
                    x="Type",
                    y="Count",
                    text="Count",
                    title="Confusion Matrix Breakdown",
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=470,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        if actual_col and probability_col:
            scores = pd.to_numeric(
                test_predictions[probability_col],
                errors="coerce",
            )

            actual_scores = pd.to_numeric(
                test_predictions[actual_col],
                errors="coerce",
            )

            valid = scores.notna() & actual_scores.notna()

            scores = scores[valid]
            actual_scores = actual_scores[
                valid
            ].astype(int)

            if not scores.empty and scores.max() > 1:
                scores = scores / 100

            if (
                len(scores) > 1
                and scores.nunique() > 1
                and actual_scores.nunique() > 1
            ):
                st.markdown("## ROC and Precision-Recall")

                fpr, tpr, _ = roc_curve(
                    actual_scores,
                    scores,
                )

                roc_auc = auc(
                    fpr,
                    tpr,
                )

                precision, recall, _ = (
                    precision_recall_curve(
                        actual_scores,
                        scores,
                    )
                )

                pr_auc = auc(
                    recall,
                    precision,
                )

                roc_col, pr_col = st.columns(2)

                with roc_col:
                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=fpr,
                            y=tpr,
                            mode="lines",
                            name=f"AUC {roc_auc:.4f}",
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode="lines",
                            name="Random",
                            line={"dash": "dash"},
                        )
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        title="ROC Curve",
                        height=480,
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                with pr_col:
                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=recall,
                            y=precision,
                            mode="lines",
                            name=f"AUC {pr_auc:.4f}",
                        )
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        title="Precision-Recall Curve",
                        height=480,
                        xaxis_title="Recall",
                        yaxis_title="Precision",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                st.markdown("## Score separation")

                score_df = pd.DataFrame(
                    {
                        "Phishing Probability (%)": (
                            scores * 100
                        ),
                        "Actual Class": actual_scores.map(
                            {
                                0: "Legitimate",
                                1: "Phishing",
                            }
                        ).values,
                    }
                )

                fig = px.histogram(
                    score_df,
                    x="Phishing Probability (%)",
                    color="Actual Class",
                    nbins=50,
                    barmode="overlay",
                    opacity=.72,
                    marginal="box",
                    title="Estimated Phishing Probability by Actual Class",
                )

                fig.add_vline(
                    x=50,
                    line_dash="dash",
                    annotation_text="Current threshold",
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=520,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                st.markdown("## Threshold analysis")

                thresholds = np.arange(
                    0.10,
                    0.91,
                    0.02,
                )

                records = []

                for threshold in thresholds:
                    threshold_prediction = (
                        scores >= threshold
                    ).astype(int)

                    tp = int(
                        (
                            (actual_scores == 1)
                            & (threshold_prediction == 1)
                        ).sum()
                    )

                    fp = int(
                        (
                            (actual_scores == 0)
                            & (threshold_prediction == 1)
                        ).sum()
                    )

                    fn = int(
                        (
                            (actual_scores == 1)
                            & (threshold_prediction == 0)
                        ).sum()
                    )

                    p = (
                        tp / (tp + fp)
                        if tp + fp
                        else 0
                    )

                    r = (
                        tp / (tp + fn)
                        if tp + fn
                        else 0
                    )

                    f1 = (
                        2 * p * r / (p + r)
                        if p + r
                        else 0
                    )

                    records.append(
                        {
                            "Threshold": threshold,
                            "Precision": p,
                            "Recall": r,
                            "F1 Score": f1,
                        }
                    )

                threshold_df = pd.DataFrame(records)

                fig = go.Figure()

                for column in [
                    "Precision",
                    "Recall",
                    "F1 Score",
                ]:
                    fig.add_trace(
                        go.Scatter(
                            x=threshold_df["Threshold"],
                            y=threshold_df[column],
                            mode="lines",
                            name=column,
                        )
                    )

                fig.add_vline(
                    x=.50,
                    line_dash="dash",
                    annotation_text="Current threshold",
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=500,
                    title="Threshold vs Precision / Recall / F1",
                    yaxis_range=[0, 1.05],
                    xaxis_title="Phishing probability threshold",
                    yaxis_title="Score",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                best = threshold_df.loc[
                    threshold_df["F1 Score"].idxmax()
                ]

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    metric_card(
                        "Best F1 threshold",
                        f'{best["Threshold"]:.2f}',
                    )

                with c2:
                    metric_card(
                        "Best F1",
                        f'{best["F1 Score"] * 100:.2f}%',
                    )

                with c3:
                    metric_card(
                        "Precision",
                        f'{best["Precision"] * 100:.2f}%',
                    )

                with c4:
                    metric_card(
                        "Recall",
                        f'{best["Recall"] * 100:.2f}%',
                    )

    # ------------------------------------------------------------
    # Robustness
    # ------------------------------------------------------------

    if (
        robustness_results is not None
        and not robustness_results.empty
    ):
        required = {
            "Feature Group",
            "Split",
            "Model",
            "F1 Score",
        }

        if required.issubset(
            robustness_results.columns
        ):
            st.markdown("## Robustness")

            robust = robustness_results.copy()

            robust["Configuration"] = (
                robust["Feature Group"].astype(str)
                + " • "
                + robust["Split"].astype(str)
                + " • "
                + robust["Model"].astype(str)
            )

            fig = px.bar(
                robust.sort_values("F1 Score"),
                x="F1 Score",
                y="Configuration",
                orientation="h",
                text="F1 Score",
                title="F1 Across Robustness Experiments",
            )

            fig.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside",
            )

            fig.update_layout(
                template="plotly_dark",
                height=max(
                    440,
                    len(robust) * 42,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# ================================================================
# RESEARCH
# ================================================================

elif page == "🧪 Research":
    render_hero(
        "PROJECT REPORT",
        "Research Overview",
        "The experimental story behind the deployed phishing classifier.",
    )

    st.markdown("## Research question")

    st.info(
        "Can URL-only character-level machine learning reliably distinguish "
        "phishing from legitimate URLs without retrieving webpage content, "
        "while retaining performance on previously unseen domains?"
    )

    st.markdown("## Methodology")

    methodology = [
        "Dataset acquisition",
        "Data cleaning",
        "Baseline model comparison",
        "Leakage audit",
        "Domain-disjoint holdout",
        "Character-level TF-IDF representation",
        "Linear SVM classification",
        "Probability calibration",
        "Threshold analysis",
        "Model explainability",
        "Application deployment",
    ]

    for i, stage in enumerate(
        methodology,
        start=1,
    ):
        st.markdown(
            f"**{i}. {stage}**"
        )
        if i != len(methodology):
            st.caption("↓")

    st.markdown("## Why this model?")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
<div class="card">
    <div class="card-title">Character-level TF-IDF</div>
    <div class="card-muted">
        URLs contain meaningful local character patterns, substrings
        and structural fragments. Character-level TF-IDF allows the
        classifier to learn those patterns directly from URL text.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
<div class="card">
    <div class="card-title">Calibrated Linear SVM</div>
    <div class="card-muted">
        A linear decision function provides a lightweight deployable
        classifier, while calibration allows the application to expose
        an estimated phishing probability.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("## Evaluation philosophy")

    st.warning(
        "The final evaluation is based on a domain-disjoint holdout. "
        "This is stricter than a simple random URL split because test "
        "domains are excluded from training."
    )

    st.markdown("## What is actually trained?")

    training_df = pd.DataFrame(
        {
            "Component": [
                "Phishing URL classifier",
                "QR decoder",
                "URL security indicators",
                "Risk-score mapping",
                "Batch scanner",
                "Model explainability",
            ],
            "Trained?": [
                "Yes",
                "No",
                "No",
                "No",
                "No",
                "No",
            ],
            "Technology": [
                "Character TF-IDF + Calibrated Linear SVM",
                "OpenCV QRCodeDetector",
                "Python heuristics",
                "Deterministic mapping",
                "Same trained classifier",
                "TF-IDF × SVM coefficient contribution",
            ],
        }
    )

    st.dataframe(
        training_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("## Deployment pipeline")

    pipeline = [
        "URL / QR / CSV input",
        "URL normalization",
        "Character TF-IDF",
        "Calibrated Linear SVM",
        "Probability",
        "Risk score",
        "Human-readable evidence",
        "Report / dashboard",
    ]

    cols = st.columns(4)

    for index, stage in enumerate(
        pipeline
    ):
        with cols[index % 4]:
            st.markdown(
                f"""
<div class="card" style="margin-bottom:14px">
    <div style="color:#8b5cf6;font-size:.72rem;font-weight:800">
        STEP {index + 1}
    </div>
    <div class="card-title">{html.escape(stage)}</div>
</div>
""",
                unsafe_allow_html=True,
            )


# ================================================================
# FOOTER
# ================================================================

st.markdown(
    """
<div class="footer">
    Phishing ML Project · Character-level phishing URL classification · Research & Development
</div>
""",
    unsafe_allow_html=True,
)
