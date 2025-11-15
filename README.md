📘 Forest Cover Type Prediction — Streamlit App

A complete Machine Learning + Streamlit application to predict Forest Cover Types using a trained RandomForest/XGBoost pipeline.
The app includes:

✅ Prediction Interface
✅ Dashboard (Model Metrics, Tuning Summary)
✅ Evaluation Tab (Confusion Matrix, Classification Report)
✅ Feature Importance
✅ Support for XGBoost label-shift wrapper
✅ Downloadable Predictions
✅ Optional PDF Export

🚀 Live Demo (Streamlit Cloud)

(Add your Streamlit app link here)

https://your-username-your-repo-name.streamlit.app

📂 Project Structure
.
├── app.py                     # Main Streamlit app
├── requirements.txt           # Project dependencies
├── artifacts/
│   ├── best_model.joblib      # Saved ML model (committed or downloaded at runtime)
│   └── tuning_summary.json    # Model tuning summary
├── data/
│   └── train.csv              # Dataset (optional)
├── README.md
└── .gitignore

⚙️ Features
🧠 Model

Supports RandomForest, XGBoost, or any sklearn-compatible pipeline

Handles wrapper models (XGBLabelWrapper) with label offset automatically

Automatically detects expected input features

Supports predict & predict_proba

🖥 Dashboard

Shows model tuning summary (RandomizedSearchCV results)

Shows model metadata

One-click PDF Export (if reportlab is installed)

📊 Evaluate

Train/test split from dataset with Cover_Type

Confusion Matrix

Classification Report

Accuracy Score

Feature Importance (for tree-based models)
☁️ Deploying on Streamlit Cloud

Push your code to GitHub

Go to
👉 https://share.streamlit.io

Click New App

Select:

Repo

Branch (main)

App file → app.py

Click Deploy

Your app will be live in seconds.

📝 Requirements

This app uses:

Streamlit

Pandas, NumPy

Scikit-learn

XGBoost (optional)

Reportlab (optional, for PDF export)

See requirements.txt for exact versions.

📮 Feedback & Contributions

Pull requests and issues are welcome!
If you'd like to add new features (training page, SHAP plots, multi-page UI), feel free to open an issue.

⭐ Support

If you find this project useful, consider giving it a star ⭐ on GitHub!