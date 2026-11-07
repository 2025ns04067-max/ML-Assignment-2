# Breast Cancer Classification – Machine Learning Assignment 2

## a. Problem statement
Build and compare five supervised classification algorithms on one dataset, evaluate them using six required metrics, and provide a Streamlit interface where a user can upload labelled test data and inspect model performance.

## b. Dataset description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic), provided by `sklearn.datasets.load_breast_cancer`.
- **Task:** Binary classification of tumour diagnosis.
- **Instances:** 569.
- **Input features:** 30 numeric features derived from digitized images of fine needle aspirates.
- **Target:** `0 = malignant`, `1 = benign`.
- **Split:** 75% training and 25% testing, stratified by target, with `random_state=42`.
- **Test file:** `test_data.csv` contains the held-out test partition and target labels.
- **Source:** UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic) dataset; accessed through scikit-learn's bundled dataset loader.

## c. GitHub Repository Link
**GitHub repository URL :** `https://github.com/2025ns04067-max/ML-Assignment-2`

## d. Models used

The following five classifiers are trained on the same training partition:
1. Logistic Regression (with StandardScaler)
2. Decision Tree Classifier (maximum depth 5)
3. K-Nearest Neighbors (k=5, with StandardScaler)
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel, with StandardScaler)

### Evaluation metrics

All metrics below are computed on the same held-out test set. AUC uses the predicted probability for class 1 (benign).

|                        |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9860 | 0.9977 |      0.9889 |   0.9889 | 0.9889 | 0.9700 |
| Decision Tree          |     0.9371 | 0.9186 |      0.9551 |   0.9444 | 0.9497 | 0.8657 |
| K-Nearest Neighbors    |     0.9790 | 0.9845 |      0.9677 |   1.0000 | 0.9836 | 0.9555 |
| Gaussian Naive Bayes   |     0.9371 | 0.9893 |      0.9263 |   0.9778 | 0.9514 | 0.8650 |
| Support Vector Machine |     0.9790 | 0.9969 |      0.9888 |   0.9778 | 0.9832 | 0.9553 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved 98.60% accuracy, 0.9977 AUC and 0.9889 F1. It is the strongest model on this test split by accuracy, AUC and F1, indicating excellent separation of the two classes. |
| Decision Tree | Achieved 93.71% accuracy and 0.9497 F1. It is comparatively weaker on this split, although the depth-limited tree is easy to interpret and does not require feature scaling. |
| K-Nearest Neighbors | Achieved 97.90% accuracy and 0.9836 F1, with 100% recall for the benign class. Standardization is important for this distance-based model. |
| Gaussian Naive Bayes | Achieved 93.71% accuracy and 0.9514 F1, but its AUC is 0.9893. This suggests good ranking ability despite a lower accuracy at the chosen classification threshold. |
| Support Vector Machine | Achieved 97.90% accuracy, 0.9969 AUC and 0.9832 F1. It performs strongly and is close to Logistic Regression, while using an RBF kernel to model nonlinear boundaries. |
| **Overall winner for this dataset** | **Logistic Regression** achieved the highest F1 score on this particular held-out split. Consider all metrics and the application context before declaring a universally best model. |

## Running the project locally

1. Install Python 3.10 or later.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. To regenerate the split, train all models and save metrics: `python train_models.py`
5. Start the web app: `streamlit run app.py`
6. In the app, upload `test_data.csv`, select a model, and inspect the metrics, confusion matrix and classification report.

## Streamlit app features
- CSV test-data upload.
- Dropdown to select one of the five models.
- Accuracy, AUC, precision, recall, F1 and MCC display.
- Confusion matrix and classification report.
- Prediction preview.

## Deployment
Deploy `app.py` from this GitHub repository using Streamlit Community Cloud. Public app URL here: `https://ml-assignment-2-msc37hm8jfapshzphgaesp.streamlit.app/`.

## BITS Virtual Lab evidence
Run the training script or notebook in BITS Virtual Lab and insert your own screenshot here in the final submission PDF. The screenshot must be captured from your actual BITS Virtual Lab session.

## Important notes
- Do not submit placeholder GitHub or Streamlit URLs.
- The GitHub URL, live app URL, and BITS Virtual Lab screenshot must be supplied by you after completing those external steps.
- The uploaded CSV must contain all 30 feature columns and a `target` column with labels 0 or 1. This allows the app to calculate evaluation metrics.
