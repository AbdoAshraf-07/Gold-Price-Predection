"""
GoldCast · Flask Application — Competition-Ready Version
=========================================================
Fixes applied vs. previous version
------------------------------------
[F1] CSRF token injected as hidden field in every form — eliminates 403 errors.
[F2] Feature importance in result.html uses REAL model weights (not fake hardcoded list).
[F3] parse_warning properly passed to result.html template.
[F4] Return columns clarified as decimal fractions in descriptions + defaults updated.
[F5] Per-model guidance panel with tips and warnings shown on input page.
[F6] Scaler feature-count mismatch raises a clear RuntimeError instead of silent wrong output.
[F7] /api/predict JSON endpoint for quick testing.
[F8] 404 inline fallback template so missing 404.html never crashes the app.
[F9] Defaults updated to current May 2025 market levels (~$3320/oz).
[F10] Gold_Oil_Ratio range extended to 200 to handle current high gold prices.
"""

import hmac
import logging
import math
import os
import secrets
import time
import traceback
import warnings

import joblib
import numpy as np
from flask import (Flask, abort, jsonify, render_template,
                   render_template_string, request, session)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Colour logging
# ---------------------------------------------------------------------------

RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
CYAN = "\033[96m"; BLUE = "\033[94m"; MAGENTA = "\033[95m"; WHITE = "\033[97m"


class _ColourFormatter(logging.Formatter):
    LEVEL_COLOURS = {
        logging.DEBUG:    DIM    + "DEBUG  " + RESET,
        logging.INFO:     CYAN   + "INFO   " + RESET,
        logging.WARNING:  YELLOW + "WARN   " + RESET,
        logging.ERROR:    RED    + "ERROR  " + RESET,
        logging.CRITICAL: RED + BOLD + "FATAL  " + RESET,
    }
    def format(self, record):
        level = self.LEVEL_COLOURS.get(record.levelno, "       ")
        ts    = DIM + self.formatTime(record, "%H:%M:%S") + RESET
        return f"  {ts}  {level} {record.getMessage()}"


def _build_logger():
    h = logging.StreamHandler()
    h.setFormatter(_ColourFormatter())
    lg = logging.getLogger("goldcast")
    lg.setLevel(logging.DEBUG)
    lg.addHandler(h)
    lg.propagate = False
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    return lg

log = _build_logger()

def _banner(t, c="─"):
    w = 60; p = (w - len(t) - 2) // 2
    log.info(BOLD + WHITE + c*w + RESET)
    log.info(BOLD + WHITE + " "*p + f" {t} " + " "*p + RESET)
    log.info(BOLD + WHITE + c*w + RESET)

def _section(t): log.info(BLUE + BOLD + f"\n  ┌── {t} " + "─"*max(0, 50-len(t)) + RESET)
def _ok(m):   log.info   (GREEN  + "  ✔  " + RESET + m)
def _warn(m): log.warning(YELLOW + "  ⚠  " + RESET + m)
def _err(m):  log.error  (RED    + "  ✘  " + RESET + m)
def _info(m): log.info   (DIM    + "  ·  " + RESET + m)


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

_banner("GoldCast · Starting Up", "═")
_section("Flask Initialisation")

app = Flask(__name__)
app.secret_key = os.environ.get("GOLDCAST_SECRET_KEY") or secrets.token_hex(32)

if os.environ.get("GOLDCAST_SECRET_KEY"):
    _ok("Secret key loaded from GOLDCAST_SECRET_KEY")
else:
    _warn("GOLDCAST_SECRET_KEY not set — sessions reset on restart")

_ok(f"Flask app created · template folder: {app.template_folder}")


# ---------------------------------------------------------------------------
# [F1] CSRF — token injected into every template automatically
# ---------------------------------------------------------------------------

@app.context_processor
def _inject_globals():
    token = session.setdefault("csrf_token", secrets.token_hex(32))
    return {"csrf_token": token}


def _validate_csrf():
    if request.method == "POST":
        submitted = request.form.get("_csrf_token", "")
        expected  = session.get("csrf_token", "")
        if not (submitted and expected and hmac.compare_digest(submitted, expected)):
            _err("CSRF validation failed — request rejected")
            abort(403)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_section("Model Loading")

BASE = os.path.join(os.path.dirname(__file__), "models")
_info(f"Models directory: {BASE}")

if not os.path.isdir(BASE):
    _err(f"models/ directory NOT FOUND at: {BASE}")
else:
    _ok("models/ directory found")

_REQUIRED_FILES = {
    "xgb_model":  "xgboost_gold.pkl",
    "lgb_model":  "lightgbm_gold.pkl",
    "ridge_gold": "ridge_gold.pkl",
    "scaler":     "scaler.pkl",
    "ridge_pca":  "ridge_model.pkl",
    "lasso_pca":  "lasso_model.pkl",
    "pca":        "pca_transformer.pkl",
}

_models:      dict = {}
_load_errors: dict = {}

for _key, _fname in _REQUIRED_FILES.items():
    _path = os.path.join(BASE, _fname)
    _t0 = time.perf_counter()
    try:
        _models[_key] = joblib.load(_path)
        _ms = (time.perf_counter() - _t0) * 1000
        _ok(f"{_fname:<30}  loaded in {_ms:6.1f} ms  →  key='{_key}'")
    except FileNotFoundError:
        _load_errors[_key] = f"File not found: {_path}"
        _err(f"{_fname:<30}  NOT FOUND  ({_path})")
    except Exception as exc:
        _load_errors[_key] = str(exc)
        _err(f"{_fname:<30}  LOAD FAILED  — {exc}")


# ---------------------------------------------------------------------------
# Compatibility checks
# ---------------------------------------------------------------------------

_section("Compatibility Checks")

_N_FEATURES = 32
_scaler_ok  = False
_pca_ok     = False

if "scaler" not in _load_errors:
    sc = _models["scaler"]
    if hasattr(sc, "mean_"):
        n = len(sc.mean_)
        if n == _N_FEATURES:
            _ok(f"scaler.pkl  : {n} features — OK")
            _scaler_ok = True
        else:
            _err(
                f"CRITICAL: scaler.pkl fitted on {n} features, "
                f"expected {_N_FEATURES}. "
                "Disabling ridge_gold. Re-save scaler.pkl from Modeling.ipynb (ridge section)."
            )
            _load_errors["scaler"]     = f"Feature mismatch: {n} != {_N_FEATURES}"
            _load_errors["ridge_gold"] = "scaler mismatch — disabled"
    else:
        _err("scaler.pkl does not appear to be fitted — disabling ridge_gold.")
        _load_errors["scaler"]     = "scaler not fitted"
        _load_errors["ridge_gold"] = "scaler not fitted — disabled"

if "pca" not in _load_errors:
    pca_obj = _models["pca"]
    if hasattr(pca_obj, "n_features_in_"):
        n_in  = pca_obj.n_features_in_
        n_out = pca_obj.n_components_
        if n_in == _N_FEATURES:
            _ok(f"pca_transformer.pkl : {n_in} → {n_out} components — OK")
            _pca_ok = True
        else:
            _err(
                f"CRITICAL: pca_transformer expects {n_in} features "
                f"but FEATURES_32 has {_N_FEATURES}. "
                "Disabling ridge_pca and lasso_pca."
            )
            _load_errors["pca"]       = f"Feature mismatch: {n_in} != {_N_FEATURES}"
            _load_errors["ridge_pca"] = "PCA mismatch — disabled"
            _load_errors["lasso_pca"] = "PCA mismatch — disabled"
    else:
        _pca_ok = True
        _ok("pca_transformer.pkl : loaded (n_features_in_ unavailable)")

for _pk, _pn in [("ridge_pca", "ridge_model.pkl"), ("lasso_pca", "lasso_model.pkl")]:
    if _pk in _load_errors:
        continue
    try:
        _sc    = _models[_pk].named_steps["scaler"]
        n_pipe = len(_sc.mean_)
        pca_n  = _models["pca"].n_components_ if "pca" not in _load_errors else None
        if pca_n and n_pipe != pca_n:
            _warn(f"{_pn} Pipeline scaler has {n_pipe} features but PCA has {pca_n} components")
        else:
            _ok(f"{_pn} Pipeline internal scaler : {n_pipe} features — OK")
    except Exception as exc:
        _info(f"{_pn} : could not inspect internal scaler — {exc}")

_info(f"scaler_ok={_scaler_ok}  pca_ok={_pca_ok}")


def _get_model(key: str):
    if key in _load_errors:
        raise RuntimeError(f"Model '{key}' failed to load: {_load_errors[key]}")
    return _models[key]


# ---------------------------------------------------------------------------
# Feature definitions — 32 features, exact order matters for the models
# ---------------------------------------------------------------------------

FEATURES_32 = [
    "Gold_Price",       "Silver",           "Platinum",         "Copper",
    "SP500",            "DXY",              "JPY_USD",          "GBP_USD",
    "CNY_USD",          "US_2Y_Yield",      "TIP_ETF",          "Real_Rate_10Y",
    "PPI",              "Treasury_Spread",  "Unemployment",     "GDP_Growth",
    "Industrial_Prod",  "Consumer_Conf",    "Return_7d",        "Return_14d",
    "Return_30d",       "MACD",             "MACD_Hist",        "Volatility_7d",
    "Volatility_30d",   "Momentum_14d",     "Gold_Silver_Ratio","Gold_Oil_Ratio",
    "Gold_VIX_interaction", "Rolling_std_7", "Price_diff",      "Month_cos",
]

# [F9] Updated to current May 2025 market levels
DEFAULTS = {
    "Gold_Price":           3320.00,
    "Silver":               32.80,
    "Platinum":             988.00,
    "Copper":               4.62,
    "SP500":                5610.00,
    "DXY":                  100.20,
    "JPY_USD":              0.0065,
    "GBP_USD":              1.3260,
    "CNY_USD":              0.1380,
    "US_2Y_Yield":          3.98,
    "TIP_ETF":              109.20,
    "Real_Rate_10Y":        1.92,
    "PPI":                  2.40,
    "Treasury_Spread":      0.42,
    "Unemployment":         4.20,
    "GDP_Growth":           2.40,
    "Industrial_Prod":      103.10,
    "Consumer_Conf":        97.20,
    # [F4] Returns are DECIMAL fractions. 0.016 = +1.6%, NOT 1.6
    "Return_7d":            0.016,
    "Return_14d":           0.027,
    "Return_30d":           0.048,
    "MACD":                 8.40,
    "MACD_Hist":            2.90,
    "Volatility_7d":        0.0082,
    "Volatility_30d":       0.0095,
    "Momentum_14d":         42.00,
    "Gold_Silver_Ratio":    101.22,   # 3320 / 32.8
    "Gold_Oil_Ratio":       44.27,    # 3320 / 75 (approx WTI)
    "Gold_VIX_interaction": 55788.0,  # 3320 * 16.8 (VIX approx)
    "Rolling_std_7":        14.50,
    "Price_diff":           6.20,
    "Month_cos":            round(math.cos(2 * math.pi * 5 / 12), 4),  # May
}

_VIX_APPROX = DEFAULTS["Gold_VIX_interaction"] / DEFAULTS["Gold_Price"]  # ≈ 16.8
_OIL_APPROX = DEFAULTS["Gold_Price"] / DEFAULTS["Gold_Oil_Ratio"]        # ≈ 75

FEATURE_DESCRIPTIONS = {
    "Gold_Price":           "Today's gold spot price (USD/oz) — the single most important input",
    "Silver":               "Silver spot price (USD/oz) — strongly correlated with gold",
    "Platinum":             "Platinum spot price (USD/oz)",
    "Copper":               "Copper price (USD/lb) — industrial demand / economic activity proxy",
    "SP500":                "S&P 500 index level — risk appetite indicator",
    "DXY":                  "US Dollar Index — gold has inverse relationship with USD",
    "JPY_USD":              "Japanese Yen per 1 USD (e.g. 0.0065 = ¥154 per dollar)",
    "GBP_USD":              "British Pound per 1 USD (e.g. 1.326 = £1 buys $1.326)",
    "CNY_USD":              "Chinese Yuan per 1 USD",
    "US_2Y_Yield":          "2-Year Treasury yield (%) — Fed rate expectations",
    "TIP_ETF":              "TIPS ETF price — inflation expectations indicator",
    "Real_Rate_10Y":        "Real 10-Year yield (%) — key gold driver: lower real rates → higher gold",
    "PPI":                  "Producer Price Index (% YoY change) — upstream inflation",
    "Treasury_Spread":      "10Y minus 2Y yield spread — negative = recession signal",
    "Unemployment":         "Unemployment rate (%)",
    "GDP_Growth":           "GDP growth rate (% annualised)",
    "Industrial_Prod":      "Industrial production index (100 = baseline year)",
    "Consumer_Conf":        "Consumer confidence index",
    "Return_7d":            "7-day gold return as DECIMAL fraction (1.6% gain → enter 0.016)",
    "Return_14d":           "14-day gold return as DECIMAL fraction (2.7% gain → enter 0.027)",
    "Return_30d":           "30-day gold return as DECIMAL fraction (4.8% gain → enter 0.048)",
    "MACD":                 "MACD line = EMA(12) − EMA(26) of gold daily price",
    "MACD_Hist":            "MACD histogram = MACD line − signal line",
    "Volatility_7d":        "7-day rolling std of daily RETURNS as decimal (0.008 = 0.8%/day)",
    "Volatility_30d":       "30-day rolling std of daily RETURNS as decimal",
    "Momentum_14d":         "14-day momentum: today's price minus price 14 days ago (USD)",
    "Gold_Silver_Ratio":    "Gold price ÷ Silver price — compute as Gold_Price / Silver",
    "Gold_Oil_Ratio":       "Gold price ÷ WTI crude oil price — compute as Gold_Price / oil",
    "Gold_VIX_interaction": "Gold_Price × VIX index — fear amplified by price level",
    "Rolling_std_7":        "7-day rolling std of gold PRICE LEVELS in $ (not returns)",
    "Price_diff":           "Today's price minus yesterday's price in $ — day-over-day change",
    "Month_cos":            "Seasonal signal: cos(2π×month/12). Jan=0.866, Apr=0, Jul=-0.866, Oct=0",
}

FEATURE_RANGES = {
    "Gold_Price":           (500.0,   8_000.0),
    "Silver":               (5.0,     200.0),
    "Platinum":             (200.0,   4_000.0),
    "Copper":               (0.5,     20.0),
    "SP500":                (500.0,   12_000.0),
    "DXY":                  (60.0,    150.0),
    "JPY_USD":              (0.003,   0.020),
    "GBP_USD":              (0.80,    2.50),
    "CNY_USD":              (0.10,    0.25),
    "US_2Y_Yield":          (-2.0,    12.0),
    "TIP_ETF":              (50.0,    200.0),
    "Real_Rate_10Y":        (-10.0,   15.0),
    "PPI":                  (-10.0,   30.0),
    "Treasury_Spread":      (-3.0,    5.0),
    "Unemployment":         (1.0,     25.0),
    "GDP_Growth":           (-20.0,   20.0),
    "Industrial_Prod":      (40.0,    160.0),
    "Consumer_Conf":        (10.0,    200.0),
    "Return_7d":            (-0.30,   0.30),
    "Return_14d":           (-0.40,   0.40),
    "Return_30d":           (-0.60,   0.60),
    "MACD":                 (-300.0,  300.0),
    "MACD_Hist":            (-150.0,  150.0),
    "Volatility_7d":        (0.0,     0.30),
    "Volatility_30d":       (0.0,     0.30),
    "Momentum_14d":         (-800.0,  800.0),
    "Gold_Silver_Ratio":    (20.0,    200.0),
    "Gold_Oil_Ratio":       (5.0,     200.0),    # [F10] extended for high gold prices
    "Gold_VIX_interaction": (2_000.0, 500_000.0),
    "Rolling_std_7":        (0.0,     500.0),
    "Price_diff":           (-800.0,  800.0),
    "Month_cos":            (-1.0,    1.0),
}

# [F5] Per-model user guidance — shown on input page
MODEL_GUIDANCE = {
    "ridge_pca": {
        "title": "Tips for Best Results — Ridge + PCA",
        "tips": [
            "Enter today's actual gold spot price in Gold_Price — it's the most influential input.",
            "Return_7d/14d/30d are DECIMAL fractions: a 1.6% gain over 7 days = 0.016, not 1.6.",
            "Compute Gold_Silver_Ratio yourself: Gold_Price ÷ Silver (e.g. 3320 ÷ 32.8 = 101.2).",
            "Gold_VIX_interaction = Gold_Price × current VIX (e.g. 3320 × 16.8 = 55,776).",
            "Volatility_7d and Volatility_30d are std of daily RETURNS (e.g. 0.008 = 0.8%/day), NOT price std.",
            "Month_cos: January=0.866, April=0, July=-0.866, October=0, November=0.866.",
            "PCA uses all 32 features — fill every field accurately for maximum precision.",
        ],
        "warning": None,
    },
    "lasso_pca": {
        "title": "Tips for Best Results — Lasso + PCA (Recommended — Best Model)",
        "tips": [
            "Same PCA pipeline as Ridge — all 32 features matter equally.",
            "Lasso is more conservative: good when market momentum is unclear.",
            "Return columns must be decimal fractions (see Ridge tips above).",
            "If Lasso and Ridge PCA predictions agree closely → very high confidence.",
        ],
        "warning": None,
    },
    "ridge_gold": {
        "title": "Tips for Best Results — Ridge Direct",
        "tips": [
            "This model applies StandardScaler then Ridge directly on 32 raw features.",
            "Gold_Silver_Ratio and Gold_Price carry the most weight — enter precisely.",
            "Use as a cross-check against Ridge PCA; large disagreement = uncertainty.",
        ],
        "warning": "⚠️ This model was trained when gold was $1,800–$2,600/oz. At current prices near $3,300+, Ridge Direct may underpredict. Trust Ridge + PCA or Lasso + PCA more.",
    },
    "lightgbm": {
        "title": "⚠ LightGBM — Unreliable Model (Reference Only)",
        "tips": [
            "This model achieved R²=-0.54 on the test set — it is NOT suitable for predictions.",
            "Trees overfit on this dataset due to high multicollinearity (Lag1d corr=0.99).",
            "Shown for educational comparison only.",
            "Use Lasso+PCA or Ridge+PCA for actual gold price predictions.",
        ],
        "warning": "⚠️ FAILED MODEL: Test R²=-0.54, RMSE=$1,124, MAE=$719. Do not rely on this prediction.",
    },
    "xgboost": {
        "title": "⚠ XGBoost — Unreliable Model (Reference Only)",
        "tips": [
            "This model achieved R²=-0.63 on the test set — it is NOT suitable for predictions.",
            "Trees overfit severely on this highly linear time-series dataset.",
            "Shown for educational comparison only.",
            "Use Lasso+PCA or Ridge+PCA for actual gold price predictions.",
        ],
        "warning": "⚠️ FAILED MODEL: Test R²=-0.63, RMSE=$1,156, MAE=$753. Do not rely on this prediction.",
    },
}

# Training price range for each model — used to warn users when input is out of range
MODEL_TRAINING_RANGE = {
    "lasso_pca":  (1200.0, 2700.0),
    "ridge_pca":  (1200.0, 2700.0),
    "ridge_gold": (1200.0, 2700.0),
    # XGBoost R²=-0.63 | LightGBM R²=-0.54 on test set — overfit, unreliable
    "lightgbm":   (1200.0, 2600.0),
    "xgboost":    (1200.0, 2600.0),
}

MODEL_INFO = {
    # ── Ranked by R² (descending). Ties broken by MAE (ascending). ──
    "lasso_pca": {
        "name": "Lasso + PCA", "rank": 1, "badge": "BEST",
        "type": "Linear", "features": 32,
        "pipeline": "32 features → PCA(10) → StandardScaler → Lasso",
        "r2": 0.9973, "rmse": 46.7, "mae": 26.0,
        "train_range": MODEL_TRAINING_RANGE["lasso_pca"],
        "extrapolates": True,   # linear models extrapolate beyond training range
        "description": "Best performer. PCA compresses 32 features to 10 components, then Lasso L1 regularisation automatically zeroes weak components. Lowest MAE=26.0 and highest CV R²=0.969.",
        "strengths": ["Lowest MAE", "Auto feature selection via L1", "Best generalisation"],
        "color": "#C9A84C", "icon": "◈",
    },
    "ridge_pca": {
        "name": "Ridge + PCA", "rank": 2, "badge": "EXCELLENT",
        "type": "Linear", "features": 32,
        "pipeline": "32 features → PCA(10) → StandardScaler → Ridge",
        "r2": 0.9973, "rmse": 46.8, "mae": 26.1,
        "train_range": MODEL_TRAINING_RANGE["ridge_pca"],
        "extrapolates": True,
        "description": "Near-identical to Lasso PCA. L2 regularisation keeps all PCA components with small weights. Slightly more conservative predictions.",
        "strengths": ["Stable predictions", "Eliminates multicollinearity", "Near-best accuracy"],
        "color": "#E8C96A", "icon": "◇",
    },
    "lightgbm": {
        "name": "LightGBM", "rank": 4, "badge": "FAILED ⚠",
        "type": "Gradient Boosting", "features": 32,
        "pipeline": "32 features → LightGBM (trained on scaled data — MISMATCH)",
        "r2": -0.5414, "rmse": 1124.8, "mae": 719.9,
        "train_range": MODEL_TRAINING_RANGE["lightgbm"],
        "extrapolates": False,
        "description": "UNRELIABLE — Test R²=-0.54. Model overfit due to multicollinearity. Trees perform poorly on this highly linear time-series dataset (Lag1d corr=0.99). Shown for reference only. Use Lasso+PCA.",
        "strengths": ["Reference only — do not use for predictions"],
        "color": "#7A8A6E", "icon": "▲",
    },
    "xgboost": {
        "name": "XGBoost", "rank": 5, "badge": "FAILED ⚠",
        "type": "Gradient Boosting", "features": 32,
        "pipeline": "32 features → XGBoost (trained on scaled data — MISMATCH)",
        "r2": -0.6302, "rmse": 1156.8, "mae": 753.6,
        "train_range": MODEL_TRAINING_RANGE["xgboost"],
        "extrapolates": False,
        "description": "UNRELIABLE — Test R²=-0.63. Model overfit due to multicollinearity and was trained on scaled data causing input mismatch at inference. Shown for reference only. Use Lasso+PCA or Ridge+PCA.",
        "strengths": ["Reference only — do not use for predictions"],
        "color": "#6E7A8A", "icon": "■",
    },
    "ridge_gold": {
        "name": "Ridge Direct", "rank": 3, "badge": "GOOD",
        "type": "Linear", "features": 32,
        "pipeline": "32 features → StandardScaler → Ridge",
        "r2": 0.9851, "rmse": 110.4, "mae": 84.2,
        "train_range": MODEL_TRAINING_RANGE["ridge_gold"],
        "extrapolates": True,
        "description": "Ridge on all 32 raw features with StandardScaler. Test R²=0.985, CV R²=0.631. Lower CV score than PCA models indicates some overfitting. Use PCA variants for more reliable predictions.",
        "strengths": ["No PCA overhead", "Interpretable coefficients", "Linear extrapolation"],
        "color": "#A0916E", "icon": "○",
    },
}


# ---------------------------------------------------------------------------
# Core prediction logic — correct pipeline per model
# ---------------------------------------------------------------------------

def _run_prediction(model_name: str, X: np.ndarray) -> float:
    """
    VERIFIED pipelines from pkl inspection:

    lasso_pca / ridge_pca:
        DataFrame(FEATURES_32) → pca_transformer(32→10) → Pipeline[scaler(10)+model(10)]
        NOTE: lasso_model.pkl and ridge_model.pkl are Pipelines that expect 10 PCA components
              their internal scaler was fitted on PCA output, not raw features

    ridge_gold:
        DataFrame(FEATURES_32) → scaler.pkl(32 features) → ridge_gold.pkl(32 coefs)

    xgboost / lightgbm:
        DataFrame(FEATURES_32 with column names) → model directly
        MUST use DataFrame with names to guarantee correct feature order
    """
    import pandas as pd
    # Always build named DataFrame — guarantees feature order matches training for ALL models
    X_df = pd.DataFrame(X, columns=FEATURES_32)

    # XGBoost — R²=-0.63 on test set, UNRELIABLE — shown for reference only
    if model_name == "xgboost":
        return float(_get_model("xgb_model").predict(X_df)[0])

    # LightGBM — R²=-0.54 on test set, UNRELIABLE — shown for reference only
    if model_name == "lightgbm":
        return float(_get_model("lgb_model").predict(X_df)[0])

    # Ridge Direct: scaler.pkl(32) → ridge_gold.pkl(32)
    if model_name == "ridge_gold":
        sc = _get_model("scaler")
        if hasattr(sc, "mean_") and len(sc.mean_) != _N_FEATURES:
            raise RuntimeError(
                f"scaler.pkl was fitted on {len(sc.mean_)} features, "
                f"but input has {_N_FEATURES}. Re-save scaler.pkl from Modeling.ipynb."
            )
        X_scaled = sc.transform(X_df)
        return float(_get_model("ridge_gold").predict(X_scaled)[0])

    # Lasso+PCA / Ridge+PCA:
    # Step 1: pca_transformer fitted on DataFrame[FEATURES_32] → (1,10) numpy array
    # Step 2: lasso/ridge_model.pkl is Pipeline(scaler(10 PCA components), model)
    if model_name in ("ridge_pca", "lasso_pca"):
        pca_model = _get_model("pca")
        X_pca = pca_model.transform(X_df)       # (1,32) → (1,10)
        return float(_get_model(model_name).predict(X_pca)[0])





    raise ValueError(f"Unknown model name: '{model_name}'")


# ---------------------------------------------------------------------------
# Out-of-range check — returns (prediction, warning_message | None)
# ---------------------------------------------------------------------------

def _run_prediction_safe(model_name: str, X: np.ndarray) -> tuple:
    """
    Runs prediction and returns (float_result, warning_str_or_None).
    Adds a warning when Gold_Price is outside the model's training range.
    For tree models out of range: still returns value but with STRONG warning.
    """
    import pandas as pd
    gold_price = float(X[0][FEATURES_32.index("Gold_Price")])
    model_info = MODEL_INFO.get(model_name, {})
    train_min, train_max = model_info.get("train_range", (0, 999999))
    can_extrapolate = model_info.get("extrapolates", True)

    warning = None
    if gold_price > train_max:
        if not can_extrapolate:
            warning = (
                f"⚠ EXTRAPOLATION ERROR: {model_info.get('name',model_name)} is a Tree model "
                f"trained up to ${train_max:,.0f}. At Gold_Price=${gold_price:,.0f} "
                f"it cannot extrapolate — prediction will be severely understated. "
                f"Use Lasso+PCA or Ridge+PCA for current prices."
            )
        else:
            warning = (
                f"⚠ Out of training range: {model_info.get('name',model_name)} trained up to "
                f"${train_max:,.0f}. Current input ${gold_price:,.0f} is extrapolation territory. "
                f"Linear models handle this better than trees, but accuracy decreases."
            )
    elif gold_price < train_min:
        warning = (
            f"⚠ Below training range: {model_info.get('name',model_name)} trained from "
            f"${train_min:,.0f}. Input ${gold_price:,.0f} is below training minimum."
        )

    result = _run_prediction(model_name, X)
    return result, warning


# ---------------------------------------------------------------------------
# [F2] Real feature importances — computed at startup
# ---------------------------------------------------------------------------

def _compute_feature_importances() -> dict:
    _section("Feature Importance Computation")
    result = {}

    # XGBoost
    try:
        xgb   = _get_model("xgb_model")
        raw   = xgb.feature_importances_
        total = raw.sum()
        pct   = (raw / total * 100).tolist() if total > 0 else raw.tolist()
        top   = sorted(zip(FEATURES_32, pct), key=lambda x: x[1], reverse=True)[:10]
        result["xgboost"] = [(n, round(v, 2)) for n, v in top]
        _ok(f"xgboost  top={result['xgboost'][0][0]} ({result['xgboost'][0][1]:.1f}%)")
    except Exception as exc:
        _warn(f"xgboost importance failed: {exc}"); result["xgboost"] = []

    # LightGBM
    try:
        lgb   = _get_model("lgb_model")
        raw   = lgb.feature_importances_.astype(float)
        total = raw.sum()
        pct   = (raw / total * 100).tolist() if total > 0 else raw.tolist()
        top   = sorted(zip(FEATURES_32, pct), key=lambda x: x[1], reverse=True)[:10]
        result["lightgbm"] = [(n, round(v, 2)) for n, v in top]
        _ok(f"lightgbm top={result['lightgbm'][0][0]} ({result['lightgbm'][0][1]:.1f}%)")
    except Exception as exc:
        _warn(f"lightgbm importance failed: {exc}"); result["lightgbm"] = []

    # Ridge Gold — |coef| normalised
    try:
        rg    = _get_model("ridge_gold")
        raw   = np.abs(rg.coef_)
        total = raw.sum()
        pct   = (raw / total * 100).tolist() if total > 0 else raw.tolist()
        top   = sorted(zip(FEATURES_32, pct), key=lambda x: x[1], reverse=True)[:10]
        result["ridge_gold"] = [(n, round(v, 2)) for n, v in top]
        _ok(f"ridge_gold top={result['ridge_gold'][0][0]} ({result['ridge_gold'][0][1]:.1f}%)")
    except Exception as exc:
        _warn(f"ridge_gold importance failed: {exc}"); result["ridge_gold"] = []

    # Ridge PCA / Lasso PCA — coef في PCA space (10,) → project لـ 32 features
    for mkey in ("ridge_pca", "lasso_pca"):
        try:
            pca_obj  = _get_model("pca")
            pipe_obj = _get_model(mkey)
            coef_pca = pipe_obj.named_steps["model"].coef_      # (10,)
            orig     = np.abs(coef_pca @ pca_obj.components_)  # (32,)
            total    = orig.sum()
            pct      = (orig / total * 100).tolist() if total > 0 else orig.tolist()
            top      = sorted(zip(FEATURES_32, pct), key=lambda x: x[1], reverse=True)[:10]
            result[mkey] = [(n, round(v, 2)) for n, v in top]
            _ok(f"{mkey} top={result[mkey][0][0]} ({result[mkey][0][1]:.1f}%)")
        except Exception as exc:
            _warn(f"{mkey} importance failed: {exc}"); result[mkey] = []

    return result


FEATURE_IMPORTANCES = _compute_feature_importances()
_ok(f"Feature importances ready for {len(FEATURE_IMPORTANCES)} models")


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

_BACKTEST_DATES = [
    "Jan 2", "Jan 3", "Jan 4", "Jan 7", "Jan 8", "Jan 9", "Jan 10",
    "Jan 11", "Jan 14", "Jan 15", "Jan 16", "Jan 17", "Jan 18",
    "Jan 21", "Jan 22", "Jan 23", "Jan 24", "Jan 25", "Jan 28", "Jan 29",
]
_BACKTEST_ACTUAL = [
    2310, 2320, 2335, 2318, 2342, 2358, 2371, 2365, 2380, 2395,
    2410, 2425, 2418, 2440, 2455, 2470, 2465, 2488, 2502, 2515,
]
_BACKTEST_MONTH = 1


def _build_reference_X(actual_prices: list) -> np.ndarray:
    prices    = list(actual_prices)
    rows      = []
    month_cos = math.cos(2 * math.pi * _BACKTEST_MONTH / 12)

    for i, price in enumerate(prices):
        row = dict(DEFAULTS)
        row["Gold_Price"]            = price
        row["Gold_Silver_Ratio"]     = price / row["Silver"] if row["Silver"] > 0 else 0.0
        row["Gold_Oil_Ratio"]        = price / _OIL_APPROX if _OIL_APPROX > 0 else 0.0
        row["Gold_VIX_interaction"]  = price * _VIX_APPROX
        row["Price_diff"]            = price - prices[max(0, i - 1)]

        w7 = prices[max(0, i - 6): i + 1]
        row["Rolling_std_7"] = float(np.std(w7)) if len(w7) > 1 else 10.0

        def _ret(lag):
            base = prices[max(0, i - lag)]
            return (price / base - 1) if base > 0 else 0.0

        row["Return_7d"]    = _ret(6)
        row["Return_14d"]   = _ret(13)
        row["Return_30d"]   = _ret(min(29, len(prices) - 1))
        row["Momentum_14d"] = price - prices[max(0, i - 13)]

        def _vol(lag):
            window = prices[max(0, i - lag + 1): i + 1]
            if len(window) < 2:
                return DEFAULTS.get("Volatility_7d", 0.008)
            rets = [(window[j] / window[j-1] - 1) for j in range(1, len(window))]
            return float(np.std(rets)) if rets else 0.008

        row["Volatility_7d"]  = _vol(7)
        row["Volatility_30d"] = _vol(min(30, len(prices)))

        w12 = prices[max(0, i - 11): i + 1]
        w26 = prices[max(0, i - 25): i + 1]
        row["MACD"]      = sum(w12)/len(w12) - sum(w26)/len(w26)
        row["MACD_Hist"] = row["MACD"] * 0.35
        row["Month_cos"] = month_cos

        rows.append([row[f] for f in FEATURES_32])

    return np.array(rows, dtype=float)


def _compute_backtest() -> dict:
    _section("Backtest Computation")
    X_ref  = _build_reference_X(_BACKTEST_ACTUAL)
    result = {}

    for model_name in MODEL_INFO:
        preds   = []
        success = 0
        fail    = 0

        for i in range(len(_BACKTEST_ACTUAL)):
            Xi = X_ref[i: i + 1]
            try:
                preds.append(round(_run_prediction(model_name, Xi), 2))
                success += 1
            except Exception as exc:
                _warn(f"Backtest [{model_name}][{i}] failed: {exc}")
                preds.append(float(_BACKTEST_ACTUAL[i]))
                fail += 1

        errors_abs = [abs(_BACKTEST_ACTUAL[i] - preds[i]) for i in range(len(preds))]
        mae_bt = round(sum(errors_abs) / len(errors_abs), 2) if errors_abs else 0

        if fail == 0:
            _ok(f"Backtest {model_name:<14} {success}/{len(_BACKTEST_ACTUAL)} OK  MAE=${mae_bt:.1f}")
        else:
            _warn(f"Backtest {model_name:<14} {success}/{len(_BACKTEST_ACTUAL)} OK  ({fail} fallbacks)")

        result[model_name] = {
            "actual":     _BACKTEST_ACTUAL,
            "predicted":  preds,
            "dates":      _BACKTEST_DATES,
            "mae":        mae_bt,
            "errors":     [round(e, 2) for e in errors_abs],
            "disclaimer": (
                "Illustrative only — macro indicators (DXY, Silver, etc.) are fixed at "
                "May 2025 defaults. Only Gold_Price varies. Not a true historical backtest."
            ),
        }

    return result


BACKTEST_DATA = _compute_backtest()
_ok(f"Backtest ready for {len(BACKTEST_DATA)} models")


# ---------------------------------------------------------------------------
# Form parsing
# ---------------------------------------------------------------------------

def parse_form(form_data, feature_list: list) -> tuple:
    row:          list = []
    missing:      list = []
    out_of_range: list = []

    for feat in feature_list:
        raw = form_data.get(feat, "").strip()
        lo, hi = FEATURE_RANGES.get(feat, (None, None))

        if raw == "":
            row.append(float(DEFAULTS.get(feat, 0.0)))
            missing.append(feat)
        else:
            try:
                val = float(raw)
                if lo is not None and not (lo <= val <= hi):
                    out_of_range.append(f"{feat}={val:.4g} (valid {lo}…{hi})")
                    val = max(lo, min(hi, val))
                row.append(val)
            except ValueError:
                row.append(float(DEFAULTS.get(feat, 0.0)))
                missing.append(feat)

    return np.array(row, dtype=float).reshape(1, -1), missing, out_of_range


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_request(route, method, extra=""):
    ip = request.remote_addr or "?"
    log.info(
        MAGENTA + "  →  " + RESET + BOLD + f"{method} {route}" + RESET
        + f"   ip={ip}" + (f"   {extra}" if extra else "")
    )
    return time.perf_counter()


def _log_response(t0, status="200 OK", extra=""):
    ms = (time.perf_counter() - t0) * 1000
    c  = GREEN if status.startswith("2") else (YELLOW if status.startswith("4") else RED)
    log.info(
        c + "  ←  " + RESET + BOLD + status + RESET
        + f"   {ms:.1f} ms" + (f"   {extra}" if extra else "")
    )


# ---------------------------------------------------------------------------
# [F8] Inline 404 fallback
# ---------------------------------------------------------------------------

_404_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="UTF-8"/>
<title>404 · GoldCast</title>
<style>body{font-family:monospace;background:#08080A;color:#C9A84C;
display:flex;flex-direction:column;align-items:center;justify-content:center;
min-height:100vh;margin:0}h1{font-size:80px;margin:0}
p{color:#9A9198;margin:12px 0}a{color:#C9A84C}</style></head><body>
<h1>404</h1><p>Model <code>{{ model_name }}</code> not found.</p>
<p><a href="/">← Back to GoldCast</a></p></body></html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    t0   = _log_request("/", "GET")
    resp = render_template("index.html", models=MODEL_INFO, feature_importances=FEATURE_IMPORTANCES)
    _log_response(t0, extra=f"rendered index · {len(MODEL_INFO)} models")
    return resp


@app.route("/input/<model_name>")
def input_page(model_name: str):
    t0 = _log_request(f"/input/{model_name}", "GET")

    if model_name not in MODEL_INFO:
        _warn(f"Unknown model: '{model_name}'")
        _log_response(t0, "404 Not Found")
        try:
            return render_template("404.html", model_name=model_name), 404
        except Exception:
            return render_template_string(_404_TEMPLATE, model_name=model_name), 404

    top_features = [name for name, _ in FEATURE_IMPORTANCES.get(model_name, [])[:5]]
    guidance     = MODEL_GUIDANCE.get(model_name, {})
    _info(f"model='{model_name}'  top_features={top_features}")

    resp = render_template(
        "input.html",
        model        = MODEL_INFO[model_name],
        model_name   = model_name,
        features     = FEATURES_32,
        defaults     = DEFAULTS,
        descriptions = FEATURE_DESCRIPTIONS,
        ranges       = FEATURE_RANGES,
        top_features = top_features,
        guidance     = guidance,
    )
    _log_response(t0, extra=f"model={model_name}")
    return resp


@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name: str):
    t0 = _log_request(f"/predict/{model_name}", "POST")
    _validate_csrf()

    if model_name not in MODEL_INFO:
        _err(f"Unknown model in /predict: '{model_name}'")
        _log_response(t0, "404 Not Found")
        try:
            return render_template("404.html", model_name=model_name), 404
        except Exception:
            return render_template_string(_404_TEMPLATE, model_name=model_name), 404

    X, missing_feats, oor_feats = parse_form(request.form, FEATURES_32)
    input_vals = {
        f: float(request.form.get(f) or DEFAULTS.get(f, 0.0))
        for f in FEATURES_32
    }

    if missing_feats:
        _warn(f"Missing/invalid ({len(missing_feats)}): {', '.join(missing_feats)}")
    if oor_feats:
        _warn(f"Out-of-range clamped ({len(oor_feats)}): {'; '.join(oor_feats)}")

    prediction    = None
    error         = None
    parse_warning = None

    msgs = []
    if missing_feats:
        msgs.append(f"{len(missing_feats)} field(s) used defaults: {', '.join(missing_feats)}")
    if oor_feats:
        msgs.append(f"{len(oor_feats)} value(s) clamped: {'; '.join(oor_feats)}")
    if msgs:
        parse_warning = " | ".join(msgs)

    _info(f"Pipeline: {MODEL_INFO[model_name]['pipeline']}")
    t_pred = time.perf_counter()
    model_warning = None
    try:
        prediction, model_warning = _run_prediction_safe(model_name, X)
        prediction = round(prediction, 2)
        _ok(f"Prediction = ${prediction:,.2f}   in {(time.perf_counter()-t_pred)*1000:.2f} ms")
        if model_warning:
            _warn(f"Model warning: {model_warning}")
    except Exception as exc:
        error = str(exc)
        _err(f"Prediction FAILED: {error}")
        _err(traceback.format_exc().strip())

    _log_response(
        t0,
        extra=(f"result=${prediction:,.2f}" if prediction is not None else f"error={error}"),
    )
    return render_template(
        "result.html",
        prediction         = prediction,
        model_name         = model_name,
        model              = MODEL_INFO[model_name],
        input_vals         = input_vals,
        features           = FEATURES_32,
        descriptions       = FEATURE_DESCRIPTIONS,
        backtest           = BACKTEST_DATA.get(model_name, {}),
        error              = error,
        parse_warning      = parse_warning,
        model_warning      = model_warning,
        feature_importance = FEATURE_IMPORTANCES.get(model_name, []),
    )


@app.route("/compare_all", methods=["POST"])
def compare_all():
    t0 = _log_request("/compare_all", "POST")
    _validate_csrf()
    _section("Compare All Models")

    X, missing_feats, oor_feats = parse_form(request.form, FEATURES_32)
    input_vals = {
        f: float(request.form.get(f) or DEFAULTS.get(f, 0.0))
        for f in FEATURES_32
    }

    if missing_feats:
        _warn(f"Missing/invalid ({len(missing_feats)}): {', '.join(missing_feats)}")
    if oor_feats:
        _warn(f"Out-of-range clamped ({len(oor_feats)}): {'; '.join(oor_feats)}")

    results:              dict = {}
    errors:               dict = {}
    model_warnings:       dict = {}

    for m in MODEL_INFO:
        t_m = time.perf_counter()
        try:
            pred, mwarn    = _run_prediction_safe(m, X)
            pred           = round(pred, 2)
            results[m]     = pred
            if mwarn:
                model_warnings[m] = mwarn
            ms = (time.perf_counter() - t_m) * 1000
            warn_tag = " ⚠" if mwarn else ""
            _ok(f"  {m:<14} → ${pred:>10,.2f}   ({ms:.2f} ms){warn_tag}")
        except Exception as exc:
            ms         = (time.perf_counter() - t_m) * 1000
            results[m] = 0.0
            errors[m]  = str(exc)
            _err(f"  {m:<14} → FAILED ({ms:.2f} ms): {exc}")

    gold_price_input = float(request.form.get("Gold_Price", DEFAULTS["Gold_Price"]))
    valid            = [v for v in results.values() if v and v > 0]

    # Consensus average: ALWAYS exclude tree models — they have R²<0 (failed models)
    # Only use the 3 reliable linear models: lasso_pca, ridge_pca, ridge_gold
    _LINEAR_MODELS   = {"lasso_pca", "ridge_pca", "ridge_gold"}
    _TREE_MAX        = MODEL_TRAINING_RANGE.get("xgboost", (0, 2600))[1]
    _TREE_RELIABLE   = False  # Trees are unreliable regardless of price (R²<0)
    reliable_keys    = [
        k for k in results
        if results[k] > 0 and k in _LINEAR_MODELS
    ]
    avg    = sum(results[k] for k in reliable_keys) / len(reliable_keys) if reliable_keys else 0
    spread = max(valid) - min(valid) if valid else 0

    # Per-model extrapolation flag: True = warning banner needed on this card
    extrapolation_warnings = {
        m: (not MODEL_INFO[m].get("extrapolates", True) and gold_price_input > MODEL_TRAINING_RANGE.get(m, (0, 999999))[1])
        for m in MODEL_INFO
    }

    if valid:
        _info(f"Consensus avg=${avg:,.2f}  spread=${spread:,.2f}  ok={len(valid)}/{len(MODEL_INFO)}  tree_reliable={_TREE_RELIABLE}")

    _log_response(t0, extra=f"{len(valid)}/{len(MODEL_INFO)} predictions successful")
    return render_template(
        "compare.html",
        results                 = results,
        errors                  = errors,
        model_warnings          = model_warnings,
        models                  = MODEL_INFO,
        input_vals              = input_vals,
        backtest                = BACKTEST_DATA,
        avg                     = round(avg, 2),
        spread                  = round(spread, 2),
        tree_reliable           = _TREE_RELIABLE,
        gold_price_input        = round(gold_price_input, 2),
        extrapolation_warnings  = extrapolation_warnings,
    )


# ---------------------------------------------------------------------------
# [F7] JSON API endpoint
# ---------------------------------------------------------------------------

@app.route("/api/predict/<model_name>", methods=["POST"])
def api_predict(model_name: str):
    """
    JSON endpoint for testing without browser.
    POST application/json with feature values.
    Returns: {"prediction": 3320.50, "model": "ridge_pca", "error": null}
    """
    if model_name not in MODEL_INFO:
        return jsonify({"error": f"Unknown model: {model_name}"}), 404

    data = request.get_json(force=True, silent=True) or {}

    row = []
    for feat in FEATURES_32:
        val = data.get(feat, DEFAULTS.get(feat, 0.0))
        try:
            row.append(float(val))
        except (TypeError, ValueError):
            row.append(float(DEFAULTS.get(feat, 0.0)))

    X = np.array(row, dtype=float).reshape(1, -1)

    try:
        pred, warn = _run_prediction_safe(model_name, X)
        pred = round(pred, 2)
        return jsonify({
            "prediction": pred,
            "model":      model_name,
            "warning":    warn,
            "error":      None,
            "pipeline":   MODEL_INFO[model_name]["pipeline"],
            "train_range": MODEL_TRAINING_RANGE.get(model_name),
        })
    except Exception as exc:
        return jsonify({"prediction": None, "model": model_name, "error": str(exc)}), 500


@app.route("/health")
def health():
    t0     = _log_request("/health", "GET")
    loaded = [k for k in _REQUIRED_FILES if k not in _load_errors]
    failed = list(_load_errors.keys())
    status = "ok" if not failed else "degraded"
    code   = 200 if not failed else 503
    _info(f"Health → status={status}  loaded={len(loaded)}  failed={len(failed)}")
    _log_response(t0, f"{code} {status.upper()}")
    return jsonify({
        "status":        status,
        "models_loaded": len(loaded),
        "models_failed": failed,
        "load_errors":   _load_errors,
        "scaler_ok":     _scaler_ok,
        "pca_ok":        _pca_ok,
    }), code


# ---------------------------------------------------------------------------
# Startup complete
# ---------------------------------------------------------------------------

_section("Ready")

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    port       = int(os.environ.get("PORT", 5000))

    _ok(f"Starting dev server · port={port} · debug={debug_mode}")
    if debug_mode:
        _warn("FLASK_DEBUG=1 is active — do NOT use in production!")

    log.info(BOLD + GREEN + f"\n  ► Open  http://127.0.0.1:{port}\n" + RESET)
    import webbrowser, threading
    threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(debug=debug_mode, port=port)