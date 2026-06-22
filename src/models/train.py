"""Train the FinSight credit-risk classifier.

Reads the processed feature frame (src/utils/build_features.py output), derives
a peer-relative tertile risk label, trains a StandardScaler + RandomForest
pipeline, evaluates on a stratified holdout, and persists the pipeline and the
ordered feature list. Run as a script:

    python src/utils/build_features.py   # writes data/processed/features.csv
    python src/models/train.py           # trains + saves the model

Labeling: all five focus banks are Altman-distress in absolute terms (Z < 1.81),
so the documented absolute-zone rule collapses to a single 'high' class. We
instead label by peer-relative tertiles of altman_z (lowest third -> 'high',
middle -> 'moderate', top -> 'low'), consistent with CLAUDE.md's guidance to
read bank Z-Scores relative to peers rather than against absolute zones.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = ROOT / "data" / "processed" / "features.csv"
MODEL_PATH = ROOT / "src" / "models" / "risk_classifier.pkl"
FEATURE_NAMES_PATH = ROOT / "src" / "models" / "feature_names.json"

# Dropped for banks (unclassified balance sheets -> structurally NaN).
DROP_COLS = ["current_ratio", "working_capital"]
NON_FEATURE_COLS = ["ticker", "year", "risk_label"]
ACCURACY_THRESHOLD = 0.60


def main() -> None:
    """Run the full train/evaluate/save flow."""
    # --- Step 1: load processed features ----------------------------------
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"{FEATURES_CSV} not found. Run `python src/utils/build_features.py` first."
        )
    df = pd.read_csv(FEATURES_CSV)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # --- Step 3 (pre-label cleanup): drop NaN rows ------------------------
    # revenue_growth is NaN for each ticker's first year; altman_z/coverage are
    # NaN where EBIT is unavailable (e.g. BNY 2023). Tertile labels must be
    # computed on the rows we actually train on, so drop NaN before labeling.
    print("Shape before dropna:", df.shape)
    df = df.dropna().reset_index(drop=True)
    print("Shape after dropna :", df.shape)

    # --- Step 2: peer-relative tertile labels -----------------------------
    df["risk_label"] = pd.qcut(
        df["altman_z"], q=3, labels=["high", "moderate", "low"]
    ).astype(str)
    print("\nLabel distribution (peer-relative altman_z tertiles):")
    print(df["risk_label"].value_counts())

    # --- Step 3 (feature list): persist ordered features ------------------
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    FEATURE_NAMES_PATH.write_text(json.dumps(feature_cols, indent=2) + "\n")
    print(f"\nFeatures ({len(feature_cols)}):", feature_cols)

    X = df[feature_cols]
    y = df["risk_label"]

    # --- Step 4: stratified 80/20 split -----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Step 5: train pipeline -------------------------------------------
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)

    # --- Step 6: evaluate on holdout --------------------------------------
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nHoldout accuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred), labels sorted:")
    print(confusion_matrix(y_test, y_pred))
    if accuracy < ACCURACY_THRESHOLD:
        print("WARNING: accuracy below 60% threshold — do not merge model")

    # --- Step 7: persist pipeline (joblib) + feature list (json) ----------
    # feature_names.json stays JSON text (consumed by risk_tools via json.load);
    # only the fitted pipeline is joblib-serialized.
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved pipeline -> {MODEL_PATH}")
    print(f"Saved feature list -> {FEATURE_NAMES_PATH}")


if __name__ == "__main__":
    main()
