# tuning.py
"""
Hyperparameter tuning script for Forest Cover project.

Fix: make XGBoost compatible with non-zero-based labels by mapping labels to 0..(K-1)
during XGBoost training, and wrap saved XGBoost pipeline so predictions are returned
in the original label space.

Usage:
    python tuning.py
"""

import os
import json
import joblib
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import get_scorer

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

# ---------------------------
# User-configurable params
# ---------------------------
DATA_PATH = "data/train.csv"
TARGET_COL = "Cover_Type"      # change if your label column has a different name
DROP_COLS = ["Id"]             # columns to drop if present
N_ITER_RF = 16                 # number of RandomizedSearch iterations for RF
N_ITER_XGB = 12                # number of RandomizedSearch iterations for XGBoost (if used)
CV_FOLDS = 3
RANDOM_STATE = 42
OUT_DIR = "artifacts"
BEST_MODEL_NAME = "best_model.joblib"
SAVE_RESULTS_JSON = True

# ---------------------------
# Helpers
# ---------------------------
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_csv(path)
    return df

def detect_cols(df, drop_cols=None, target_col=None):
    drop_cols = drop_cols or []
    cols = [c for c in df.columns if c not in drop_cols and c != target_col]
    X = df[cols].copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, cat_cols

def build_preprocessor(num_cols, cat_cols, numeric_impute="mean", scale_numeric=True, cat_impute="most_frequent"):
    # numeric pipeline
    num_steps = []
    if numeric_impute:
        num_steps.append(("imputer", SimpleImputer(strategy=numeric_impute)))
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipeline = Pipeline(num_steps) if num_steps else "passthrough"

    # categorical pipeline
    cat_steps = []
    if cat_impute:
        if cat_impute == "most_frequent":
            cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        else:
            cat_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="missing")))
    # handle scikit-learn versions: use sparse_output if available, else use sparse
    ohe_kwargs = {"handle_unknown": "ignore"}
    try:
        OneHotEncoder(sparse_output=False)  # test if arg exists
        ohe_kwargs["sparse_output"] = False
    except TypeError:
        ohe_kwargs["sparse"] = False

    cat_steps.append(("ohe", OneHotEncoder(**ohe_kwargs)))
    cat_pipeline = Pipeline(cat_steps)

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)
    return preprocessor

def ensure_out_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# Wrapper to remap XGBoost predictions back to original labels
class XGBLabelWrapper:
    """
    Wraps a fitted sklearn Pipeline that contains an XGBoost estimator trained
    on 0-based labels. This wrapper will call internal_pipeline.predict(X) and
    add label_offset to restore original label values.
    """
    def __init__(self, internal_pipeline, label_offset):
        self.pipeline = internal_pipeline
        self.label_offset = int(label_offset)

    def predict(self, X):
        preds = self.pipeline.predict(X)
        return preds + self.label_offset

    def predict_proba(self, X):
        # probabilities don't need offset; returns same shape as pipeline.predict_proba
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)
        raise AttributeError("Underlying pipeline has no predict_proba")

    def __getstate__(self):
        # for joblib compatibility
        return {"pipeline": self.pipeline, "label_offset": self.label_offset}

    def __setstate__(self, state):
        self.pipeline = state["pipeline"]
        self.label_offset = state["label_offset"]

# ---------------------------
# Main tuning routine
# ---------------------------
def main():
    print("Loading data from:", DATA_PATH)
    df = load_data(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data columns: {df.columns.tolist()[:10]}...")

    # drop cols if present
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Determine task (classification vs regression)
    is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 30
    if is_regression:
        print("Task inferred: regression")
    else:
        print("Task inferred: classification")

    num_cols, cat_cols = detect_cols(df, drop_cols=DROP_COLS, target_col=TARGET_COL)
    print(f"Detected {len(num_cols)} numeric cols and {len(cat_cols)} categorical cols")

    preprocessor = build_preprocessor(num_cols, cat_cols, numeric_impute="mean", scale_numeric=True, cat_impute="most_frequent")

    results_summary = {}

    # ---------------------------
    # RandomForest tuning
    # ---------------------------
    print("\n=== RandomForest tuning ===")
    if is_regression:
        base_rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        scoring = "r2"
    else:
        base_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        scoring = "accuracy"

    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", base_rf)
    ])

    rf_param_dist = {
        "estimator__n_estimators": [200, 300, 500, 700],
        "estimator__max_depth": [None, 10, 20, 30],
        "estimator__min_samples_split": [2, 5, 10],
        "estimator__min_samples_leaf": [1, 2, 4],
        "estimator__bootstrap": [True, False] if not is_regression else [True]
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE) if not is_regression else KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rf_search = RandomizedSearchCV(
        rf_pipeline,
        rf_param_dist,
        n_iter=N_ITER_RF,
        scoring=scoring,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    print("Fitting RandomizedSearchCV for RandomForest (this can take time)...")
    rf_search.fit(X, y)
    print("RandomForest best CV score:", rf_search.best_score_)
    print("RandomForest best params:", rf_search.best_params_)
    results_summary["RandomForest"] = {
        "best_score": float(rf_search.best_score_),
        "best_params": rf_search.best_params_
    }

    # ---------------------------
    # XGBoost tuning (optional) with label mapping
    # ---------------------------
    xgb_search = None
    xgb_label_info = None  # will store mapping info if used
    if HAS_XGBOOST:
        try:
            print("\n=== XGBoost tuning ===")
            # If classification with non-zero-based labels, remap to 0..K-1 for XGBoost
            if not is_regression:
                label_min = int(y.min())
                if label_min != 0:
                    print(f"Detected non-zero label min ({label_min}). Mapping labels to 0-based for XGBoost.")
                    y_for_xgb = y - label_min
                    xgb_label_info = {"label_min": label_min}
                else:
                    y_for_xgb = y.copy()
                    xgb_label_info = {"label_min": 0}
            else:
                # regression: no remapping
                y_for_xgb = y.copy()
                xgb_label_info = {"label_min": None}

            if is_regression:
                base_xgb = XGBRegressor(random_state=RANDOM_STATE, verbosity=0, objective="reg:squarederror", tree_method="hist")
                xgb_scoring = "r2"
            else:
                base_xgb = XGBClassifier(use_label_encoder=False, random_state=RANDOM_STATE, eval_metric="mlogloss", tree_method="hist")
                xgb_scoring = "accuracy"

            xgb_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("estimator", base_xgb)
            ])

            xgb_param_dist = {
                "estimator__n_estimators": [200, 300, 500],
                "estimator__max_depth": [4, 6, 8],
                "estimator__learning_rate": [0.05, 0.1, 0.2],
                "estimator__subsample": [0.7, 0.9, 1.0],
            }

            xgb_search = RandomizedSearchCV(
                xgb_pipeline,
                xgb_param_dist,
                n_iter=N_ITER_XGB,
                scoring=xgb_scoring,
                cv=cv,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=1
            )
            print("Fitting RandomizedSearchCV for XGBoost (this can take time)...")
            # Note: fit using y_for_xgb (0-based if needed)
            xgb_search.fit(X, y_for_xgb)
            print("XGBoost best CV score:", xgb_search.best_score_)
            print("XGBoost best params:", xgb_search.best_params_)
            results_summary["XGBoost"] = {
                "best_score": float(xgb_search.best_score_),
                "best_params": xgb_search.best_params_
            }

            # If labels were remapped, wrap the best estimator so it returns original labels
            if not is_regression and xgb_label_info and xgb_label_info.get("label_min", 0) != 0:
                label_min_used = xgb_label_info["label_min"]
                wrapped = XGBLabelWrapper(xgb_search.best_estimator_, label_offset=label_min_used)
                # replace the search.best_estimator_ for later comparison & saving
                # but keep xgb_search object intact (we'll use wrapped for saving)
                xgb_search.best_estimator_wrapped_ = wrapped
            else:
                xgb_search.best_estimator_wrapped_ = xgb_search.best_estimator_

        except Exception as e:
            print("XGBoost tuning failed:", str(e))
            traceback.print_exc()
            xgb_search = None
    else:
        print("\nXGBoost not installed. Skipping XGBoost tuning.")

    # ---------------------------
    # Compare & save best
    # ---------------------------
    # Choose best by CV score
    best_model = None
    best_name = None
    best_score = -np.inf

    # Consider RF
    rf_score = rf_search.best_score_
    best_model = rf_search.best_estimator_
    best_name = "RandomForest"
    best_score = rf_score

    # Consider XGB if present
    if xgb_search is not None:
        xgb_score = xgb_search.best_score_
        # use wrapped estimator if present when saving/using predictions
        candidate_est = getattr(xgb_search, "best_estimator_wrapped_", xgb_search.best_estimator_)
        if xgb_score > best_score:
            best_model = candidate_est
            best_name = "XGBoost"
            best_score = xgb_score

    ensure_out_dir(OUT_DIR)
    best_path = os.path.join(OUT_DIR, BEST_MODEL_NAME)
    joblib.dump(best_model, best_path)
    print(f"\n✅ Best model ({best_name}) saved to: {best_path}  (CV score: {best_score:.4f})")

    # Save results summary (JSON) including XGBoost label mapping info if used
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "data_path": DATA_PATH,
        "target_col": TARGET_COL,
        "drop_cols": DROP_COLS,
        "is_regression": bool(is_regression),
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "results": results_summary,
        "selected_best": {"name": best_name, "cv_score": float(best_score)}
    }
    if xgb_label_info is not None:
        summary["xgboost_label_info"] = xgb_label_info

    if SAVE_RESULTS_JSON:
        out_json = os.path.join(OUT_DIR, "tuning_summary.json")
        save_json(summary, out_json)
        print("Saved tuning summary to:", out_json)

    print("\nTuning finished.")

if __name__ == "__main__":
    main()
