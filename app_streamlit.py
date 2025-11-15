# app.py
"""
Forest Cover — Predict & Dashboard (cleaned)
- Auto-reload / page refresh features removed entirely
- Tabs: Predict | Dashboard | Evaluate | Settings
- Dynamic badge controls, persisted paths, XGBLabelWrapper compatibility
- PDF export (reportlab + matplotlib optional)
- Quick Evaluate: confusion matrix, classification report, feature importance
"""

import os
import json
import joblib
import platform
import subprocess
from datetime import datetime
from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

# Optional features for PDF export
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# ---------------- Config ----------------
ARTIFACT_DIR = Path("artifacts")
LAST_PATHS = ARTIFACT_DIR / ".last_paths.json"
TUNING_SUMMARY = ARTIFACT_DIR / "tuning_summary.json"
DEFAULT_MODEL = ARTIFACT_DIR / "best_model.joblib"
DATA_PATH = Path("data") / "train.csv"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------------- XGB wrapper (for unpickle) ----------------
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

# ---------------- Helpers ----------------
def find_latest_pipeline(out_dir=ARTIFACT_DIR):
    files = sorted(Path(out_dir).glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None

def load_pipeline(path):
    return joblib.load(path)

def model_info(obj):
    info = {"type": type(obj).__name__, "is_wrapper": False, "label_offset": None, "estimator_type": None, "expected_input_features": None, "has_proba": hasattr(obj, "predict_proba")}
    try:
        if isinstance(obj, XGBLabelWrapper):
            info["is_wrapper"] = True
            info["label_offset"] = int(obj.label_offset)
            inner = obj.pipeline
            info["type"] = "XGBLabelWrapper"
            if isinstance(inner, Pipeline):
                est = inner.named_steps.get("estimator")
                info["estimator_type"] = type(est).__name__ if est is not None else None
                pre = inner.named_steps.get("preprocessor")
                if pre is not None and hasattr(pre, "transformers_"):
                    cols = []
                    for name, trans, cols_list in pre.transformers_:
                        try:
                            cols.extend(list(cols_list))
                        except Exception:
                            pass
                    info["expected_input_features"] = cols
            return info
        if isinstance(obj, Pipeline):
            info["estimator_type"] = type(obj.named_steps.get("estimator")).__name__ if obj.named_steps.get("estimator") is not None else None
            pre = obj.named_steps.get("preprocessor")
            if pre is not None and hasattr(pre, "transformers_"):
                cols = []
                for name, trans, cols_list in pre.transformers_:
                    try:
                        cols.extend(list(cols_list))
                    except Exception:
                        pass
                info["expected_input_features"] = cols
            return info
    except Exception:
        pass
    return info

def read_csv(path_or_buffer):
    if isinstance(path_or_buffer, (str, Path)):
        return pd.read_csv(path_or_buffer)
    else:
        return pd.read_csv(path_or_buffer)

def open_file(path):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        pass

def load_last_paths():
    if LAST_PATHS.exists():
        try:
            return json.loads(LAST_PATHS.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_last_paths(d):
    try:
        LAST_PATHS.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass

def load_tuning_summary():
    if TUNING_SUMMARY.exists():
        try:
            return json.loads(TUNING_SUMMARY.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def export_dashboard_pdf(out_pdf_path, summary):
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab or matplotlib not available. Install: pip install reportlab matplotlib")
    # small chart
    rf = summary.get("results", {}).get("RandomForest", {}).get("best_score")
    xgb = summary.get("results", {}).get("XGBoost", {}).get("best_score")
    labels = []; scores = []
    if rf is not None:
        labels.append("RandomForest"); scores.append(float(rf))
    if xgb is not None:
        labels.append("XGBoost"); scores.append(float(xgb))
    chart_buf = None
    if scores:
        fig, ax = plt.subplots(figsize=(4,1.6))
        ax.bar(labels, scores, color=["#1f6feb","#0f9d58"][:len(labels)])
        ax.set_ylim(0,1)
        ax.set_ylabel("CV score")
        ax.set_title("Model CV scores")
        plt.tight_layout()
        chart_buf = io.BytesIO()
        fig.savefig(chart_buf, format="png", dpi=150)
        plt.close(fig)
        chart_buf.seek(0)
    doc = SimpleDocTemplate(out_pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Forest Cover — Dashboard Report", styles["Title"]))
    story.append(Spacer(1,6))
    t = f"Generated: {datetime.utcnow().isoformat()}"
    story.append(Paragraph(t, styles["Normal"]))
    story.append(Spacer(1,8))
    story.append(Paragraph("Tuning Summary", styles["Heading2"]))
    story.append(Paragraph("<pre>" + (json.dumps(summary, indent=2)[:4000].replace("\n","<br/>")) + "</pre>", styles["Code"]))
    story.append(Spacer(1,8))
    if chart_buf:
        img_path = ARTIFACT_DIR / "temp_chart.png"
        with open(img_path, "wb") as f:
            f.write(chart_buf.read())
        story.append(Image(str(img_path), width=400, height=140))
    doc.build(story)
    return out_pdf_path

# ---------------- Dynamic base CSS ----------------
BASE_CSS = """
<style>
#MainMenu, footer, header { visibility: hidden; }
body { font-family: "Segoe UI", Roboto, Arial, sans-serif; }
.compact-hide { display:none !important; }
.badge { display:inline-grid; place-items:center; color:#fff; font-weight:800; position:relative; z-index:2; }
</style>
"""
st.set_page_config(page_title="Forest Cover — Clean (No Reload)", layout="wide")
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ---------------- Sidebar: settings (no reload options) ----------------
with st.sidebar:
    st.header("Controls")
    use_server_csv = st.checkbox("Use server CSV for preview (data/train.csv)", value=True)
    upload_dataset = st.file_uploader("Upload dataset (optional)", type=["csv"])
    st.markdown("---")
    st.header("Model")
    upload_model = st.file_uploader("Upload pipeline (.joblib)", type=["joblib","pkl"])
    model_path_persist = st.text_input("Model path (persisted)", value=load_last_paths().get("model_path", str(DEFAULT_MODEL if DEFAULT_MODEL.exists() else "")))
    st.markdown("---")
    st.header("Predict paths")
    input_path_persist = st.text_input("Input CSV path (persisted)", value=load_last_paths().get("input_path",""))
    output_path_persist = st.text_input("Output CSV path (persisted)", value=load_last_paths().get("output_path", str(ARTIFACT_DIR / "predictions.csv")))
    if st.button("Save paths"):
        save_last_paths({"model_path": model_path_persist, "input_path": input_path_persist, "output_path": output_path_persist})
        st.success("Paths saved")

    st.markdown("---")
    st.header("Badge & Theme")
    badge_style_sb = st.selectbox("Badge style", ["Hexagon","Pill","Square"], index=0)
    badge_c1_sb = st.text_input("Badge color 1", value="#7b61ff")
    badge_c2_sb = st.text_input("Badge color 2", value="#ff6b9a")
    st.markdown("---")
    st.caption("Auto-reload and refresh controls have been removed.")

# ---------------- Top tabs ----------------
tabs = st.tabs(["Predict", "Dashboard", "Evaluate", "Settings"])

def auto_load_pipeline(upload_file, model_path_field):
    pipeline_obj = None
    pipeline_path = None
    if upload_file:
        try:
            upload_file.seek(0)
            pipeline_obj = joblib.load(upload_file)
            pipeline_path = "uploaded model"
            return pipeline_obj, pipeline_path
        except Exception as e:
            st.sidebar.error(f"Failed to load uploaded pipeline: {e}")
    if model_path_field and Path(model_path_field).exists():
        try:
            pipeline_obj = joblib.load(model_path_field)
            pipeline_path = model_path_field
            return pipeline_obj, pipeline_path
        except Exception as e:
            st.sidebar.warning(f"Failed to load pipeline from specified path: {e}")
    latest = find_latest_pipeline()
    if latest:
        try:
            pipeline_obj = joblib.load(latest)
            pipeline_path = latest
            return pipeline_obj, pipeline_path
        except Exception:
            pass
    return None, None

pipeline, pipeline_src = auto_load_pipeline(upload_model, model_path_persist)

# ---------- Predict tab ----------
with tabs[0]:
    st.header("Predict")
    colL, colR = st.columns([3,5])
    with colL:
        st.info("Upload CSV with features (ID/target columns will be ignored).")
        uploaded_csv = st.file_uploader("Upload CSV for prediction", type=["csv"])
        provided_input_path = st.text_input("Or enter input CSV path", value=input_path_persist)
        output_path_box = st.text_input("Output CSV path", value=output_path_persist)
        force_predict = st.checkbox("Force predict (fill missing cols with NaN)", value=False)
        run_button = st.button("Run predictions", key="run_preds")
    with colR:
        st.subheader("Pipeline status")
        if pipeline is not None:
            info = model_info(pipeline)
            st.success(f"Auto-loaded: {pipeline_src}")
            if info.get("is_wrapper"):
                st.markdown(f"**XGBoost wrapper** — label offset: +{info.get('label_offset')}")
            st.write("Estimator:", info.get("estimator_type") or "N/A")
            if info.get("expected_input_features"):
                st.write("Expected features:", len(info.get("expected_input_features")))
                if st.button("Preview expected features"):
                    st.session_state["_preview_expected"] = True
        else:
            st.warning("No pipeline loaded. Upload or save to artifacts/")

    # handle preview expected features
    if st.session_state.get("_preview_expected", False):
        if pipeline is not None:
            feats = model_info(pipeline).get("expected_input_features")
            if feats:
                with st.expander(f"Expected features ({len(feats)})", expanded=True):
                    st.write(feats)
            else:
                st.info("Model does not expose expected input feature names.")
        else:
            st.info("No pipeline loaded.")
        st.session_state["_preview_expected"] = False

    # Run predictions
    if run_button:
        if uploaded_csv:
            tmp = ARTIFACT_DIR / f"tmp_in_{int(datetime.utcnow().timestamp())}.csv"
            with open(tmp, "wb") as f:
                f.write(uploaded_csv.getbuffer())
            input_path = str(tmp)
        elif provided_input_path and Path(provided_input_path).exists():
            input_path = provided_input_path
        else:
            st.error("No input CSV provided or path invalid.")
            st.stop()

        if pipeline is None:
            st.error("No pipeline available to run predictions.")
            st.stop()

        try:
            df_in = read_csv(input_path)
        except Exception as e:
            st.error(f"Failed to read input CSV: {e}")
            st.stop()

        info = model_info(pipeline)
        expected = info.get("expected_input_features")
        missing = []
        if expected:
            missing = [c for c in expected if c not in df_in.columns]
        if missing and not force_predict:
            st.error(f"Missing {len(missing)} required columns. Enable Force predict or provide correct CSV.")
            st.write("Missing (first 10):", missing[:10])
            st.stop()

        if expected:
            safe_df = pd.DataFrame({c: df_in[c] if c in df_in.columns else np.nan for c in expected})
        else:
            safe_df = df_in.select_dtypes(include=[np.number]).copy()
            if safe_df.empty:
                safe_df = df_in.copy()

        try:
            preds = pipeline.predict(safe_df)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        out = df_in.copy()
        out["_prediction"] = preds
        try:
            if hasattr(pipeline, "predict_proba"):
                pr = pipeline.predict_proba(safe_df)
                if pr.ndim == 2 and pr.shape[1] == 2:
                    out["_prob_positive"] = pr[:,1]
        except Exception:
            pass

        out_path = output_path_box or str(ARTIFACT_DIR / f"pred_{int(datetime.utcnow().timestamp())}.csv")
        out.to_csv(out_path, index=False)
        st.success(f"Saved predictions to: {out_path}")
        st.dataframe(out.head(200))
        st.download_button("Download predictions", data=out.to_csv(index=False).encode("utf-8"), file_name=Path(out_path).name)
        if st.checkbox("Open file now", value=False):
            open_file(out_path)

# ---------- Dashboard tab ----------
with tabs[1]:
    st.header("Dashboard")
    st.subheader("Tuning summary")
    tuning = load_tuning_summary()
    if tuning:
        st.write("Selected best:", tuning.get("selected_best",{}).get("name","—"))
        st.write("Best CV score:", tuning.get("selected_best",{}).get("cv_score","—"))
        st.write("Timestamp:", tuning.get("timestamp","—"))
        st.json(tuning, expanded=False)
        if st.button("Export dashboard to PDF (one-page)"):
            if not HAS_REPORTLAB:
                st.error("Install reportlab & matplotlib: pip install reportlab matplotlib")
            else:
                pdfp = ARTIFACT_DIR / f"dashboard_{int(datetime.utcnow().timestamp())}.pdf"
                def export_simple(pdf_path, summary):
                    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
                    styles = getSampleStyleSheet()
                    story = [Paragraph("Dashboard", styles["Title"]), Spacer(1,6), Paragraph("Summary:", styles["Normal"]), Paragraph("<pre>"+json.dumps(summary, indent=2)[:4000].replace("\n","<br/>")+"</pre>", styles["Code"])]
                    doc.build(story)
                try:
                    export_simple(str(pdfp), tuning)
                    st.success(f"Exported: {pdfp}")
                except Exception as e:
                    st.error("PDF export failed: " + str(e))
    else:
        st.info("No tuning_summary.json found in artifacts/")

    st.markdown("---")
    st.subheader("Pipeline / Artifact")
    if pipeline is not None:
        info = model_info(pipeline)
        st.success("Pipeline loaded")
        st.markdown(f"- Source: `{pipeline_src}`")
        st.markdown(f"- Type: **{info.get('type')}**   Estimator: **{info.get('estimator_type') or '—'}**")
        if info.get("is_wrapper"):
            st.markdown(f"- XGBoost label wrapper: offset +{info.get('label_offset')}")
        if info.get("expected_input_features"):
            st.markdown(f"- Expected features: {len(info.get('expected_input_features'))}")
    else:
        st.warning("No pipeline loaded (upload or add to artifacts/)")

# ---------- Evaluate tab ----------
with tabs[2]:
    st.header("Evaluate")
    st.write("Quick evaluation using dataset with `Cover_Type` (if available).")
    ds = None
    if upload_dataset:
        try:
            ds = read_csv(upload_dataset)
            st.info("Using uploaded dataset for evaluation")
        except Exception as e:
            st.error("Failed to read uploaded dataset: " + str(e))
    elif use_server_csv and DATA_PATH.exists():
        try:
            ds = read_csv(DATA_PATH)
            st.info("Using server CSV: data/train.csv")
        except Exception as e:
            st.error("Failed to read server CSV: " + str(e))

    if ds is None:
        st.info("No dataset available for Evaluate. Upload one or enable server CSV in sidebar.")
    else:
        st.write(f"Dataset shape: {ds.shape}")
        if "Cover_Type" not in ds.columns and "CoverType" not in ds.columns:
            st.warning("Dataset does not contain 'Cover_Type' or 'CoverType' target column; quick eval unavailable.")
        else:
            target_col = "Cover_Type" if "Cover_Type" in ds.columns else "CoverType"
            test_size = st.slider("Test size (%)", min_value=5, max_value=50, value=20)
            rand_seed = st.number_input("Random seed", value=42)
            run_eval = st.button("Run quick evaluation")
            if run_eval:
                X = ds.drop(columns=[target_col], errors='ignore')
                y = ds[target_col]
                info = model_info(pipeline) if pipeline is not None else {}
                expected = info.get("expected_input_features") if info else None
                if expected:
                    Xp = pd.DataFrame({c: X[c] if c in X.columns else np.nan for c in expected})
                else:
                    Xp = X.select_dtypes(include=[np.number]).copy()
                    if Xp.empty:
                        Xp = X.copy()

                X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=test_size/100.0, random_state=int(rand_seed), stratify=y if len(np.unique(y))>1 else None)
                if pipeline is None:
                    st.error("No pipeline loaded to evaluate.")
                else:
                    try:
                        preds = pipeline.predict(X_test)
                    except Exception as e:
                        st.error("Prediction on test split failed: " + str(e))
                        preds = None
                    if preds is not None:
                        acc = accuracy_score(y_test, preds)
                        st.metric("Accuracy", f"{acc:.4f}")
                        st.text("Classification report:")
                        st.text(classification_report(y_test, preds, zero_division=0))
                        cm = confusion_matrix(y_test, preds, labels=np.unique(y_test))
                        fig, ax = plt.subplots(figsize=(5,4))
                        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                        ax.set_title("Confusion matrix")
                        tick_labels = np.unique(y_test)
                        ax.set_xticks(range(len(tick_labels))); ax.set_xticklabels(tick_labels, rotation=45)
                        ax.set_yticks(range(len(tick_labels))); ax.set_yticklabels(tick_labels)
                        plt.colorbar(im, ax=ax)
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
                        st.pyplot(fig)
                        plt.close(fig)
                        estimator = None
                        if isinstance(pipeline, XGBLabelWrapper):
                            inner = pipeline.pipeline
                            if isinstance(inner, Pipeline):
                                estimator = inner.named_steps.get("estimator")
                        elif isinstance(pipeline, Pipeline):
                            estimator = pipeline.named_steps.get("estimator")
                        if estimator is not None and hasattr(estimator, "feature_importances_"):
                            try:
                                fi = estimator.feature_importances_
                                feat_names = expected if expected else Xp.columns.tolist()
                                fi_series = pd.Series(fi, index=feat_names).sort_values(ascending=False).head(40)
                                st.subheader("Feature importance (top 40)")
                                fig2, ax2 = plt.subplots(figsize=(6, max(3, 0.12 * len(fi_series))))
                                fi_series.plot.barh(ax=ax2)
                                ax2.invert_yaxis()
                                st.pyplot(fig2)
                                plt.close(fig2)
                            except Exception as e:
                                st.warning("Could not compute feature importance: " + str(e))
                        else:
                            st.info("Estimator does not expose feature_importances_ (not a tree-based model)")

# ---------- Settings tab ----------
with tabs[3]:
    st.header("Settings")
    st.subheader("Badge customization")
    bstyle = st.selectbox("Badge style", ["Hexagon","Pill","Square"], index=0)
    c1 = st.text_input("Primary color (hex)", value="#7b61ff")
    c2 = st.text_input("Secondary color (hex)", value="#ff6b9a")
    width_px = st.slider("Badge width (px)", min_value=160, max_value=420, value=260)
    height_px = st.slider("Badge height (px)", min_value=48, max_value=120, value=72)
    anim_speed = st.slider("Animation speed (1s..10s)", min_value=1.0, max_value=10.0, value=6.0)
    st.markdown("---")
    st.subheader("Theme")
    theme = st.radio("Theme", ["Dark","Light"], index=0)
    st.markdown("---")
    st.subheader("Persisted paths")
    mp = st.text_input("Model path", value=model_path_persist)
    ip = st.text_input("Input path", value=input_path_persist)
    op = st.text_input("Output path", value=output_path_persist)
    if st.button("Save persisted paths (Settings)"):
        save_last_paths({"model_path": mp, "input_path": ip, "output_path": op})
        st.success("Saved")

    st.markdown("---")
    st.caption("Preview of badge using current settings:")
    def clean_hex(h):
        if not isinstance(h, str): return "#7b61ff"
        h = h.strip()
        if not h.startswith("#"): h = "#" + h
        if len(h) not in (4,7): return "#7b61ff"
        return h
    c1 = clean_hex(c1); c2 = clean_hex(c2)
    anim_ms = int(anim_speed * 1000)
    if bstyle == "Pill":
        badge_css = f"""
        <style>
        .preview {{ display:inline-grid; place-items:center; width:{width_px}px; height:{height_px}px; border-radius:999px;
          background:linear-gradient(90deg,{c1},{c2}); color:white; font-weight:800; font-size:18px;
          animation: pillanim {anim_ms}ms ease-in-out infinite; box-shadow:0 10px 30px rgba(0,0,0,0.2); }}
        @keyframes pillanim {{ 0%{{transform:scale(1)}} 50%{{transform:scale(1.04)}} 100%{{transform:scale(1)}}}}
        </style>
        <div class='preview'>Pipeline ready</div>
        """
    elif bstyle == "Square":
        badge_css = f"""
        <style>
        .preview {{ display:inline-grid; place-items:center; width:{width_px}px; height:{height_px}px; border-radius:12px;
          background:linear-gradient(90deg,{c1},{c2}); color:white; font-weight:800; font-size:18px; box-shadow:0 12px 36px rgba(0,0,0,0.18); }}
        </style><div class='preview'>Pipeline ready</div>
        """
    else:
        badge_css = f"""
        <style>
        .preview {{ display:inline-grid; place-items:center; width:{width_px}px; height:{height_px}px; background:linear-gradient(120deg,{c1},{c2});
          clip-path: polygon(25% 6%, 75% 6%, 100% 50%, 75% 94%, 25% 94%, 0% 50%); color:white; font-weight:800; font-size:18px; box-shadow:0 12px 48px rgba(0,0,0,0.18); animation:rot {anim_ms}ms linear infinite; }}
        @keyframes rot {{ 0%{{transform:rotate(0deg)}} 50%{{transform:rotate(3deg)}} 100%{{transform:rotate(0deg)}}}}
        </style><div class='preview'>Pipeline ready</div>
        """
    st.markdown(badge_css, unsafe_allow_html=True)

# End of app
st.caption("Auto-reload / refresh features removed. The app will only update when you interact or manually refresh the browser.")
