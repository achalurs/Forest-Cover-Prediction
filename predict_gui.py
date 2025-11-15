# predict_gui.py
"""
Simple GUI for running predictions using artifacts/best_model.joblib

Usage:
    python predict_gui.py

Features:
 - Pick model file (defaults to artifacts/best_model.joblib)
 - Pick input CSV
 - Pick output CSV path
 - Force-predict (fill missing features with NaN)
 - Preview expected features (button)
 - Handles XGBLabelWrapper saved by tuning.py
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# If your model used the XGBoost wrapper, define it here so joblib can unpickle
class XGBLabelWrapper:
    def __init__(self, internal_pipeline, label_offset):
        self.pipeline = internal_pipeline
        self.label_offset = int(label_offset)
    def predict(self, X):
        preds = self.pipeline.predict(X)
        try:
            return preds + self.label_offset
        except Exception:
            return (np.array(preds) + self.label_offset)
    def predict_proba(self, X):
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)
        raise AttributeError("Underlying pipeline has no predict_proba")
    def __getstate__(self):
        return {"pipeline": self.pipeline, "label_offset": self.label_offset}
    def __setstate__(self, state):
        self.pipeline = state["pipeline"]
        self.label_offset = state["label_offset"]

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

def get_expected_features(pipeline):
    """
    Return list of expected input features if the pipeline exposes a preprocessor with transformers_.
    Returns None if not discoverable.
    """
    try:
        # wrapper case
        if hasattr(pipeline, "pipeline"):
            inner = pipeline.pipeline
        else:
            inner = pipeline
        if hasattr(inner, "named_steps") and "preprocessor" in inner.named_steps:
            pre = inner.named_steps["preprocessor"]
            # transformers_ is created after fitting; for joblib-loaded pipeline it should exist
            if hasattr(pre, "transformers_"):
                cols = []
                for name, trans, cols_list in pre.transformers_:
                    # cols_list can be slice or array-like of names; convert safely
                    try:
                        cols.extend(list(cols_list))
                    except Exception:
                        # fallback: ignore non-iterable
                        pass
                return cols
    except Exception:
        pass
    return None

def run_prediction(model_path, input_csv, output_csv, force=False):
    # Load model
    model = load_model(model_path)
    expected = get_expected_features(model)
    df = pd.read_csv(input_csv)

    if expected is None:
        # proceed
        safe_df = df.copy()
    else:
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        if missing and not force:
            raise ValueError(f"Missing required columns: {missing[:10]}{'...' if len(missing)>10 else ''}")
        # build safe_df
        safe_df = pd.DataFrame()
        for c in expected:
            if c in df.columns:
                safe_df[c] = df[c]
            else:
                safe_df[c] = np.nan

    # predict
    preds = model.predict(safe_df)
    out = df.copy()
    out["_prediction"] = preds

    # optional proba for binary
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(safe_df)
            if proba.ndim == 2 and proba.shape[1] == 2:
                out["_prob_positive"] = proba[:, 1]
        except Exception:
            pass

    out.to_csv(output_csv, index=False)
    return output_csv

# ---------------- GUI ----------------
class PredictGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Predict — Forest Cover")
        self.geometry("720x380")
        self.resizable(False, False)

        self.model_var = tk.StringVar(value=os.path.join("artifacts","best_model.joblib"))
        self.input_var = tk.StringVar(value="")
        self.output_var = tk.StringVar(value="predictions.csv")
        self.force_var = tk.BooleanVar(value=False)

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        # Model row
        ttk.Label(frm, text="Model (.joblib):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.model_var, width=64).grid(row=0, column=1, padx=6, columnspan=2)
        ttk.Button(frm, text="Browse", command=self.browse_model).grid(row=0, column=3, padx=4)

        # Input row
        ttk.Label(frm, text="Input CSV:").grid(row=1, column=0, sticky="w", pady=(10,0))
        ttk.Entry(frm, textvariable=self.input_var, width=64).grid(row=1, column=1, padx=6, columnspan=2, pady=(10,0))
        ttk.Button(frm, text="Browse", command=self.browse_input).grid(row=1, column=3, padx=4, pady=(10,0))

        # Output row
        ttk.Label(frm, text="Output CSV:").grid(row=2, column=0, sticky="w", pady=(10,0))
        ttk.Entry(frm, textvariable=self.output_var, width=64).grid(row=2, column=1, padx=6, columnspan=2, pady=(10,0))
        ttk.Button(frm, text="Browse", command=self.browse_output).grid(row=2, column=3, padx=4, pady=(10,0))

        # Force checkbox
        ttk.Checkbutton(frm, text="Force predict (fill missing cols with NaN)", variable=self.force_var).grid(row=3, column=1, sticky="w", pady=(12,0))

        # Buttons: Run + Preview features
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=4, column=1, columnspan=3, pady=(18,0), sticky="w")
        self.run_btn = ttk.Button(btn_frame, text="Run Predictions", command=self.on_run)
        self.run_btn.grid(row=0, column=0, padx=(0,8))
        self.preview_btn = ttk.Button(btn_frame, text="Preview expected features", command=self.on_preview)
        self.preview_btn.grid(row=0, column=1)

        self.status_lbl = ttk.Label(frm, text="", foreground="green")
        self.status_lbl.grid(row=5, column=0, columnspan=4, pady=(12,0))

        # Padding adjustments
        for i in range(4):
            frm.grid_columnconfigure(i, weight=0)

    def browse_model(self):
        p = filedialog.askopenfilename(title="Select model (.joblib)", filetypes=[("Joblib files","*.joblib"),("All files","*.*")], initialdir=os.getcwd())
        if p:
            self.model_var.set(p)

    def browse_input(self):
        p = filedialog.askopenfilename(title="Select input CSV", filetypes=[("CSV files","*.csv")], initialdir=os.path.join(os.getcwd(),"data"))
        if p:
            self.input_var.set(p)

    def browse_output(self):
        p = filedialog.asksaveasfilename(title="Save output CSV as", defaultextension=".csv", filetypes=[("CSV files","*.csv")], initialdir=os.getcwd())
        if p:
            self.output_var.set(p)

    def on_preview(self):
        model_path = self.model_var.get().strip()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Model not found", f"Model file not found:\n{model_path}")
            return
        try:
            model = load_model(model_path)
            features = get_expected_features(model)
            if features is None:
                messagebox.showinfo("Expected features", "Model does not expose expected input feature names.\nIf you trained without a ColumnTransformer with named feature lists, the GUI cannot preview them.")
            else:
                # show features in a scrollable popup if many
                self._show_features_popup(features, title=f"Expected features ({len(features)})")
        except Exception as e:
            tb = traceback.format_exc()
            messagebox.showerror("Preview failed", f"Could not preview features:\n{e}\n\nDetails:\n{tb}")

    def _show_features_popup(self, features, title="Expected features"):
        popup = tk.Toplevel(self)
        popup.title(title)
        popup.geometry("640x420")
        popup.transient(self)
        popup.grab_set()

        frm = ttk.Frame(popup, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        lbl = ttk.Label(frm, text=f"Expected input features ({len(features)}):", font=("TkDefaultFont", 10, "bold"))
        lbl.pack(anchor="w")

        # Text widget inside a scrollable frame
        text_frame = ttk.Frame(frm)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(6,0))

        scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
        text = tk.Text(text_frame, wrap="none", yscrollcommand=scrollbar.set)
        scrollbar.config(command=text.yview)
        scrollbar.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        # insert features
        for i, f in enumerate(features, start=1):
            text.insert("end", f"{i}. {f}\n")
        text.config(state="disabled")

        btn = ttk.Button(frm, text="Close", command=popup.destroy)
        btn.pack(pady=(8,0))

    def on_run(self):
        model_path = self.model_var.get().strip()
        input_csv = self.input_var.get().strip()
        output_csv = self.output_var.get().strip()
        force = self.force_var.get()

        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Model not found", f"Model file not found:\n{model_path}")
            return
        if not input_csv or not os.path.exists(input_csv):
            messagebox.showerror("Input CSV not found", f"Input CSV not found:\n{input_csv}")
            return
        if not output_csv:
            messagebox.showerror("Output path", "Please choose an output CSV file path.")
            return

        self.run_btn.config(state="disabled")
        self.preview_btn.config(state="disabled")
        self.status_lbl.config(text="Running predictions...", foreground="blue")
        self.update_idletasks()

        try:
            out_path = run_prediction(model_path, input_csv, output_csv, force=force)
            self.status_lbl.config(text=f"Done — saved to: {out_path}", foreground="green")
            messagebox.showinfo("Success", f"Predictions saved to:\n{out_path}")
        except Exception as e:
            tb = traceback.format_exc()
            self.status_lbl.config(text="Failed", foreground="red")
            messagebox.showerror("Prediction failed", f"{str(e)}\n\nDetails:\n{tb}")
        finally:
            self.run_btn.config(state="normal")
            self.preview_btn.config(state="normal")

if __name__ == "__main__":
    app = PredictGUI()
    app.mainloop()
