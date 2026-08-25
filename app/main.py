"""
URLGuard — FastAPI inference app.

Loads the staged XGBoost model from the MLflow/DagsHub registry and the
fitted preprocessor (ColumnTransformer) from disk, exposes a single
/predict endpoint, and serves a small HTML UI.

IMPORTANT — read before deploying:
------------------------------------------------------------------------
This app has to turn a raw URL string (from an end user) into the exact
same ~50 engineered columns the model was trained on. The training
dataset (PhiUSIIL) ships those columns pre-computed; nothing in the
URLGuard pipeline so far derives them from a URL at inference time.
This file does that derivation. Two things are NOT fully solved here:

1. `URLSimilarityIndex`, `TLDLegitimateProb`, `URLCharProb` are
   dataset-internal statistics computed against the *training corpus*
   (e.g. Levenshtein similarity to known-legitimate domains, a learned
   per-TLD legitimacy rate). They cannot be correctly recomputed from a
   single incoming URL without the original lookup tables/methodology.
   They are stubbed to 0.0 below with a logged warning. These are also
   the three features flagged in the project handoff as the likely
   source of the suspicious 1.0 accuracy — check
   reports/feature_importance.csv. If they dominate, this app's
   predictions are not trustworthy until the model is retrained without
   them, regardless of anything else here.
2. Required columns are read from `preprocessor.feature_names_in_` at
   startup rather than hardcoded, since the exact column list produced
   by `data_preprocessing.py`/`feature_engineering.py` wasn't available
   to write this. Any column name this file doesn't know how to compute
   defaults to 0 and is logged — check the startup log for
   "UNRECOGNIZED COLUMN" warnings and fill those in.
------------------------------------------------------------------------
"""

import os
import re
import logging
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import mlflow
import dagshub
import numpy as np
import pandas as pd
import pickle
import requests
import time
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("urlguard")

MODEL_NAME = "URLGuard_model"
MODEL_ALIAS = "staging"
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")

# -------------------------------------------------------------------------------------
# MLflow / DagsHub tracking setup.
#
# LOCAL / TESTING (current): dagshub.init() opens a browser auth flow —
# fine for running this on your own machine, not for a headless container.
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri("https://dagshub.com/pradeepbishnoi9601/URLGuard.mlflow")
# dagshub.init(repo_owner="pradeepbishnoi9601", repo_name="URLGuard", mlflow=True)
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# PRODUCTION (swap in for Docker/deployment): token-based auth, no browser flow.
# Requires DAGSHUB_TOKEN env var set to a DagsHub access token.
# -------------------------------------------------------------------------------------
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
mlflow.set_tracking_uri("https://dagshub.com/pradeepbishnoi9601/URLGuard.mlflow")
# -------------------------------------------------------------------------------------

state = {}  # holds model, preprocessor, required_columns — populated at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading preprocessor from %s", PREPROCESSOR_PATH)
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    if hasattr(preprocessor, "feature_names_in_"):
        required_columns = list(preprocessor.feature_names_in_)
    else:
        raise RuntimeError(
            "preprocessor.pkl has no feature_names_in_ — it wasn't fit on a "
            "DataFrame with column names. Required columns can't be determined "
            "automatically; you'll need to hardcode the list from "
            "feature_engineering.py's NUMERIC + BINARY_FLAGS constants instead."
        )
    logger.info("Preprocessor expects %d raw feature columns", len(required_columns))

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    logger.info("Loading model from %s", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)

    state["preprocessor"] = preprocessor
    state["model"] = model
    state["required_columns"] = required_columns

    unknown = [c for c in required_columns if c not in KNOWN_EXTRACTORS]
    if unknown:
        logger.warning(
            "UNRECOGNIZED COLUMN(S) — will default to 0, predictions will be "
            "unreliable until these are implemented: %s",
            unknown,
        )

    yield
    state.clear()


app = FastAPI(title="URLGuard", lifespan=lifespan)

# Resolve relative to this file's own directory, not the process's CWD —
# otherwise "python app/main.py" run from the project root looks for
# ./templates instead of ./app/templates and fails with TemplateNotFound.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(_THIS_DIR, "templates"))

# ---------------------------------------------------------------------------
# Prometheus metrics — same custom-registry pattern as the reference app,
# ported from Flask's before/after-request style to explicit timing calls
# around each route body (FastAPI/Starlette doesn't have the same global
# request hooks Flask does, so this is done per-endpoint instead).
# ---------------------------------------------------------------------------
metrics_registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app",
    ["method", "endpoint"], registry=metrics_registry,
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds",
    ["endpoint"], registry=metrics_registry,
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class",
    ["prediction"], registry=metrics_registry,
)


class PredictRequest(BaseModel):
    url: str


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

SUSPICIOUS_TLDS_STUB = {"zip", "xyz", "tk", "top", "gq", "cf", "ml"}


def _fetch_page(url: str, timeout: float = 5.0):
    """Best-effort page fetch. Returns (html_text, status_ok) — never raises."""
    try:
        resp = requests.get(
            url, timeout=timeout, headers={"User-Agent": "URLGuard/1.0"}, verify=True
        )
        return resp.text, resp.status_code < 400
    except requests.RequestException as e:
        logger.warning("Could not fetch %s for content features: %s", url, e)
        return "", False


def extract_lexical_features(url: str) -> dict:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    domain = parsed.netloc.split(":")[0]
    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)
    special_chars = sum(not c.isalnum() for c in url)
    tld = domain.split(".")[-1] if "." in domain else ""

    return {
        "URLLength": len(url),
        "DomainLength": len(domain),
        "IsDomainIP": int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))),
        "TLDLength": len(tld),
        "NoOfSubDomain": max(domain.count(".") - 1, 0),
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letters / len(url) if url else 0.0,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digits / len(url) if url else 0.0,
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "NoOfOtherSpecialCharsInURL": special_chars,
        "SpacialCharRatioInURL": special_chars / len(url) if url else 0.0,
        "IsHTTPS": int(parsed.scheme == "https"),
        "HasObfuscation": int("%" in url),
        "NoOfObfuscatedChar": url.count("%"),
        "ObfuscationRatio": url.count("%") / len(url) if url else 0.0,
    }


def extract_content_features(url: str) -> dict:
    html, ok = _fetch_page(url)
    if not ok or not html:
        # Page unreachable — return conservative defaults, don't fabricate signal
        return {
            "LineOfCode": 0, "LargestLineLength": 0, "HasTitle": 0,
            "DomainTitleMatchScore": 0.0, "URLTitleMatchScore": 0.0,
            "HasFavicon": 0, "IsResponsive": 0, "NoOfURLRedirect": 0,
            "NoOfSelfRedirect": 0, "HasDescription": 0, "NoOfPopup": 0,
            "NoOfiFrame": 0, "HasExternalFormSubmit": 0, "HasSocialNet": 0,
            "HasSubmitButton": 0, "HasHiddenFields": 0, "HasPasswordField": 0,
            "Bank": 0, "Pay": 0, "Crypto": 0, "HasCopyrightInfo": 0,
            "NoOfImage": 0, "NoOfCSS": 0, "NoOfJS": 0, "NoOfSelfRef": 0,
            "NoOfEmptyRef": 0, "NoOfExternalRef": 0,
        }

    soup = BeautifulSoup(html, "html.parser")
    lines = html.splitlines()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    domain = urlparse(url).netloc.split(":")[0]
    forms = soup.find_all("form")
    links = soup.find_all("a", href=True)
    self_ref = sum(1 for a in links if domain in a.get("href", ""))
    empty_ref = sum(1 for a in links if a.get("href", "").strip() in ("", "#"))

    return {
        "LineOfCode": len(lines),
        "LargestLineLength": max((len(l) for l in lines), default=0),
        "HasTitle": int(bool(title)),
        "DomainTitleMatchScore": float(domain.split(".")[0].lower() in title.lower()) if title else 0.0,
        "URLTitleMatchScore": 0.0,  # needs a real string-similarity metric — placeholder
        "HasFavicon": int(bool(soup.find("link", rel=lambda v: v and "icon" in v.lower()))),
        "IsResponsive": int(bool(soup.find("meta", attrs={"name": "viewport"}))),
        "NoOfURLRedirect": 0,  # would need to follow redirect chain explicitly
        "NoOfSelfRedirect": 0,
        "HasDescription": int(bool(soup.find("meta", attrs={"name": "description"}))),
        "NoOfPopup": len(re.findall(r"window\.open", html, re.I)),
        "NoOfiFrame": len(soup.find_all("iframe")),
        "HasExternalFormSubmit": int(any(
            f.get("action", "").startswith("http") and domain not in f.get("action", "")
            for f in forms
        )),
        "HasSocialNet": int(bool(re.search(r"facebook\.com|twitter\.com|instagram\.com|linkedin\.com", html, re.I))),
        "HasSubmitButton": int(bool(soup.find("button", type="submit") or soup.find("input", type="submit"))),
        "HasHiddenFields": int(bool(soup.find("input", type="hidden"))),
        "HasPasswordField": int(bool(soup.find("input", type="password"))),
        "Bank": int(bool(re.search(r"\bbank\b", html, re.I))),
        "Pay": int(bool(re.search(r"\bpay(pal|ment)?\b", html, re.I))),
        "Crypto": int(bool(re.search(r"crypto|bitcoin|wallet", html, re.I))),
        "HasCopyrightInfo": int("©" in html or "copyright" in html.lower()),
        "NoOfImage": len(soup.find_all("img")),
        "NoOfCSS": len(soup.find_all("link", rel="stylesheet")),
        "NoOfJS": len(soup.find_all("script")),
        "NoOfSelfRef": self_ref,
        "NoOfEmptyRef": empty_ref,
        "NoOfExternalRef": max(len(links) - self_ref, 0),
    }


def extract_stub_features(url: str) -> dict:
    """
    Dataset-internal statistical features — NOT correctly derivable from a
    single URL at inference time. See module docstring. Stubbed to 0.0 so
    they're visibly inert rather than silently wrong-but-plausible.
    """
    return {
        "URLSimilarityIndex": 0.0,
        "TLDLegitimateProb": 0.0,
        "URLCharProb": 0.0,
    }


# Explicit list of every column name the extractors above actually produce.
# Used only for the startup "unrecognized column" warning — the runtime
# fallback in build_feature_row (defaulting to 0) is what actually protects
# against a missing column, this is just so you see the gap immediately
# in the startup log instead of discovering it via bad predictions.
KNOWN_EXTRACTORS = {
    k: True
    for k in list(extract_lexical_features("http://example.com").keys())
    + [
        "LineOfCode", "LargestLineLength", "HasTitle", "DomainTitleMatchScore",
        "URLTitleMatchScore", "HasFavicon", "IsResponsive", "NoOfURLRedirect",
        "NoOfSelfRedirect", "HasDescription", "NoOfPopup", "NoOfiFrame",
        "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton",
        "HasHiddenFields", "HasPasswordField", "Bank", "Pay", "Crypto",
        "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef",
        "NoOfEmptyRef", "NoOfExternalRef",
    ]
    + list(extract_stub_features("http://example.com").keys())
}


def build_feature_row(url: str, required_columns: list) -> pd.DataFrame:
    features = {}
    features.update(extract_lexical_features(url))
    features.update(extract_content_features(url))
    features.update(extract_stub_features(url))

    row = {}
    for col in required_columns:
        if col in features:
            row[col] = features[col]
        else:
            row[col] = 0  # unrecognized column — logged at startup, not silently ignored
    return pd.DataFrame([row], columns=required_columns)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = templates.TemplateResponse(request, "index.html", {})
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response


@app.post("/predict")
async def predict(payload: PredictRequest):
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    url = payload.url.strip()
    if not url:
        return JSONResponse({"error": "URL is required"}, status_code=400)
    if "://" not in url:
        url = f"http://{url}"

    try:
        row = build_feature_row(url, state["required_columns"])
        transformed = state["preprocessor"].transform(row)
        pred = state["model"].predict(transformed)

        # label direction is an ASSUMPTION per the project handoff (1 = legitimate),
        # not yet verified against PhiUSIIL's official docs — confirm before trusting this.
        pred_value = pred[0] if hasattr(pred, "__len__") else pred
        try:
            proba = state["model"]._model_impl.predict_proba(transformed)[0]
            confidence = float(max(proba))
        except Exception:
            confidence = None

        if pred_value == 1 and (confidence is None or confidence >= 0.8):
            label = "Safe"
        elif pred_value == 1:
            label = "Suspicious"
        else:
            label = "Phishing"

        PREDICTION_COUNT.labels(prediction=label).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        return {"url": url, "prediction": label, "confidence": confidence}

    except Exception as e:
        logger.error("Prediction failed for %s: %s", url, e)
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics from the custom registry."""
    return Response(generate_latest(metrics_registry), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": "model" in state}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)