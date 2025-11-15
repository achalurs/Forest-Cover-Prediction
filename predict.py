# predict.py
"""
Command-line predictor:
Usage:
    python predict.py input.csv output.csv [--model artifacts/best_model.joblib] [--force]

Validates input columns against model's expected features (if available).
Handles XGBLabelWrapper if present.
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np
import sys

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)

def get_expected_features(pipeline):
    try:
        if hasattr(pipeline, "pipeline"):  # wrapper
            inner = pipeline.pipeline
        else:
            inner = pipeline
        if hasattr(inner, "named_steps") and "preprocessor" in inner.named_steps:
            pre = inner.named_steps["preprocessor"]
            if hasattr(pre, "transformers_"):
                cols = []
                for name, trans, cols_list in pre.transformers_:
                    cols.extend(list(cols_list))
                return cols
    except Exception:
        pass
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", help="Input CSV file with feature columns")
    p.add_argument("output_csv", help="Output CSV path to write predictions")
    p.add_argument("--model", default="artifacts/best_model.joblib", help="Path to joblib pipeline")
    p.add_argument("--force", action="store_true", help="Force prediction even if required columns missing")
    args = p.parse_args()

    if not os.path.exists(args.input_csv):
        print("Input CSV not found:", args.input_csv); sys.exit(1)

    print("Loading model:", args.model)
    model = load_model(args.model)
    expected = get_expected_features(model)
    df = pd.read_csv(args.input_csv)

    if expected is None:
        print("Model does not expose expected input features. Proceeding without strict validation.")
        safe_df = df.copy()
    else:
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        if missing:
            print(f"Missing {len(missing)} required columns: {missing[:10]}{(' ...' if len(missing)>10 else '')}")
            if not args.force:
                print("Use --force to override and proceed (missing columns will be filled with NaN).")
                sys.exit(1)
            else:
                print("Proceeding with --force; missing columns will be filled with NaN.")
        if extra:
            print(f"Extra columns in input (will be ignored): {extra[:10]}{(' ...' if len(extra)>10 else '')}")

        # Build safe_df with expected columns in order (fill missing with NaN)
        safe_df = pd.DataFrame()
        for c in expected:
            if c in df.columns:
                safe_df[c] = df[c]
            else:
                safe_df[c] = np.nan

    print("Running predictions on", safe_df.shape[0], "rows...")
    preds = model.predict(safe_df)
    out = df.copy()
    out["_prediction"] = preds

    # add probability if available and binary
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(safe_df)
            if proba.ndim == 2 and proba.shape[1] == 2:
                out["_prob_positive"] = proba[:, 1]
        except Exception:
            pass

    out.to_csv(args.output_csv, index=False)
    print("Saved predictions to", args.output_csv)

if __name__ == "__main__":
    main()
