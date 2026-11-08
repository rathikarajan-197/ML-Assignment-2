import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

st.title("Heart Disease Prediction System")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

model_choice = st.selectbox(
    "Select Model",
    [
        "logistic_regression",
        "decision_tree",
        "knn",
        "naive_bayes",
        "svm"
    ]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    X = df.drop("target", axis=1)
    y = df["target"]

    model = joblib.load(
        f"{model_choice}.pkl"
    )

    y_pred = model.predict(X)

    y_prob = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Model Metrics")

    st.write("Accuracy:", round(accuracy,4))
    st.write("AUC:", round(auc,4))
    st.write("Precision:", round(precision,4))
    st.write("Recall:", round(recall,4))
    st.write("F1 Score:", round(f1,4))
    st.write("MCC:", round(mcc,4))

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    st.write(cm)
