"""
Train the five required classifiers and save model artifacts plus metrics.
Run: python train_models.py
"""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

dataset = load_breast_cancer(as_frame=True)
data = dataset.data.copy()
data["target"] = dataset.target
train_df, test_df = train_test_split(
    data, test_size=0.25, random_state=42, stratify=data["target"]
)
train_df.to_csv(ROOT / "train_data.csv", index=False)
test_df.to_csv(ROOT / "test_data.csv", index=False)

X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]
X_test, y_test = test_df.drop(columns=["target"]), test_df["target"]

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
    ])
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "AUC": float(roc_auc_score(y_test, probabilities)),
        "Precision": float(precision_score(y_test, predictions, zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, zero_division=0)),
        "F1": float(f1_score(y_test, predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, predictions))
    }
    filename = name.lower().replace("-", "").replace(" ", "_") + ".joblib"
    joblib.dump(model, MODEL_DIR / filename)

with open(MODEL_DIR / "metrics.json", "w") as file:
    json.dump(results, file, indent=2)

print(pd.DataFrame(results).T.round(4))
print("Saved models and test_data.csv.")
