import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

st.set_page_config(page_title="Breast Cancer Model Explorer", page_icon="🧬", layout="wide")
st.title("🧬 Breast Cancer Classification – Model Explorer")
st.write(
    "Upload a labelled CSV test file, choose one of the five trained classifiers, "
    "and inspect its evaluation metrics and confusion matrix."
)
st.info("Target convention: 0 = malignant, 1 = benign. The uploaded CSV must contain all 30 features and a 'target' column.")

@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()}

models = load_models()
model_name = st.selectbox("Select classification model", list(MODEL_FILES.keys()))
uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded is None:
    st.caption("You can use the included test_data.csv to try the application.")
    if st.button("Load included test data"):
        st.session_state["use_sample"] = True
else:
    st.session_state["use_sample"] = False

if uploaded is not None:
    test_data = pd.read_csv(uploaded)
elif st.session_state.get("use_sample", False):
    test_data = pd.read_csv(ROOT / "test_data.csv")
else:
    test_data = None

if test_data is not None:
    missing = [col for col in FEATURES + ["target"] if col not in test_data.columns]
    if missing:
        st.error("The CSV is missing required columns: " + ", ".join(missing))
    else:
        X = test_data[FEATURES]
        y = test_data["target"].astype(int)
        model = models[model_name]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        scores = {
            "Accuracy": accuracy_score(y, predictions),
            "AUC": roc_auc_score(y, probabilities),
            "Precision": precision_score(y, predictions, zero_division=0),
            "Recall": recall_score(y, predictions, zero_division=0),
            "F1 Score": f1_score(y, predictions, zero_division=0),
            "MCC": matthews_corrcoef(y, predictions),
        }
        st.subheader(f"Evaluation results – {model_name}")
        columns = st.columns(6)
        for col, (label, value) in zip(columns, scores.items()):
            col.metric(label, f"{value:.4f}")

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion matrix")
            matrix = confusion_matrix(y, predictions, labels=[0, 1])
            fig, ax = plt.subplots()
            sns.heatmap(
                matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Malignant (0)", "Benign (1)"],
                yticklabels=["Malignant (0)", "Benign (1)"], ax=ax
            )
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("Actual label")
            st.pyplot(fig)
            plt.close(fig)
        with right:
            st.subheader("Classification report")
            report = classification_report(
                y, predictions, target_names=["Malignant", "Benign"],
                output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

        st.subheader("Prediction preview")
        preview = test_data.copy()
        preview["predicted_target"] = predictions
        preview["predicted_class"] = preview["predicted_target"].map({0: "Malignant", 1: "Benign"})
        st.dataframe(preview.head(20), use_container_width=True)
