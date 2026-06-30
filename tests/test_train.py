"""Tests for the trained model artifacts produced by src/models/train.py.

These assert on the artifacts (feature_names.json, risk_classifier.pkl) that
`python src/models/train.py` writes. They load local files only — no API calls.
Run train.py before running this suite.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "src" / "models" / "risk_classifier.pkl"
FEATURE_NAMES_PATH = ROOT / "src" / "models" / "feature_names.json"

EXPECTED_FEATURES = {
    "debt_to_equity",
    "return_on_assets",
    "interest_coverage",
    "revenue_growth",
    "net_margin",
    "fedfunds",
    "t10y2y",
}


def test_feature_names_json_has_expected_seven_features():
    assert FEATURE_NAMES_PATH.exists()
    names = json.loads(FEATURE_NAMES_PATH.read_text())
    assert len(names) == 7
    assert set(names) == EXPECTED_FEATURES
    assert "altman_z" not in names


def test_risk_classifier_pkl_exists():
    assert MODEL_PATH.exists()


def test_loaded_model_predict_proba_returns_one_by_three():
    if not MODEL_PATH.exists():
        pytest.skip("risk_classifier.pkl not present — run src/models/train.py")

    model = joblib.load(MODEL_PATH)
    names = json.loads(FEATURE_NAMES_PATH.read_text())

    # Single row with the correct feature columns/order.
    row = pd.DataFrame([[0.0] * len(names)], columns=names)
    proba = model.predict_proba(row)

    assert proba.shape == (1, 3)
