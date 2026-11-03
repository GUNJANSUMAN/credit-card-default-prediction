# ============================================================
# CREDIT CARD DEFAULT PREDICTION SYSTEM
# MODULE 6: STREAMLIT APPLICATION
# ============================================================


# STEP 1: IMPORT LIBRARIES

import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# STEP 2: CONFIGURE APPLICATION

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)


# STEP 3: DEFINE PROJECT FILE LOCATION

# All model files are stored in the same folder as app.py.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# STEP 4: DEFINE MODEL FILES

models = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbours": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "Support Vector Machine": "svm.joblib"
}


# STEP 5: APPLICATION TITLE

st.title("Credit Card Default Prediction System")

st.write(
    "An interactive machine learning application "
    "to evaluate credit card default prediction models."
)

st.divider()


# STEP 6: MODEL SELECTION

st.sidebar.header("Model Configuration")

model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)


# STEP 7: UPLOAD TEST DATASET

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        # Read uploaded CSV file
        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        # Display selected model
        st.write("Selected Model:", model_name)

        # Display dataset information
        col1, col2 = st.columns(2)

        col1.metric("Total Records", df.shape[0])
        col2.metric("Total Columns", df.shape[1])

        # Display dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Target variable
        target = "default.payment.next.month"

        if target not in df.columns:

            st.error(
                "Target column not found. "
                "Please upload a dataset containing "
                "'default.payment.next.month'."
            )

        else:

            st.success("Target column found.")

            # Separate actual target and input features
            y_test = df[target]

            X_test = df.drop(columns=[target])

            # Remove ID if present
            if "ID" in X_test.columns:
                X_test = X_test.drop(columns=["ID"])

            # STEP 8: LOAD SELECTED TRAINED MODEL

            model_path = os.path.join(
                BASE_DIR,
                models[model_name]
            )

            if not os.path.exists(model_path):

                st.error(
                    f"Model file not found: {models[model_name]}"
                )

            else:

                model = joblib.load(model_path)

                st.success(
                    f"{model_name} loaded successfully."
                )

                # STEP 9: GENERATE PREDICTIONS

                predictions = model.predict(X_test)

                st.success("Predictions generated successfully.")

                # STEP 10: CALCULATE PERFORMANCE METRICS

                accuracy = accuracy_score(
                    y_test, predictions
                )

                precision = precision_score(
                    y_test, predictions, zero_division=0
                )

                recall = recall_score(
                    y_test, predictions, zero_division=0
                )

                f1 = f1_score(
                    y_test, predictions, zero_division=0
                )

                mcc = matthews_corrcoef(
                    y_test, predictions
                )

                # Calculate AUC
                try:

                    if hasattr(model, "predict_proba"):

                        probabilities = model.predict_proba(X_test)[:, 1]

                    else:

                        probabilities = model.decision_function(X_test)

                    auc = roc_auc_score(
                        y_test, probabilities
                    )

                except Exception:

                    auc = None

                # STEP 11: DISPLAY PERFORMANCE METRICS

                st.divider()

                st.header("Model Performance Results")

                col1, col2, col3 = st.columns(3)

                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric(
                    "AUC Score",
                    f"{auc:.4f}" if auc is not None else "N/A"
                )
                col3.metric("Precision", f"{precision:.4f}")

                col4, col5, col6 = st.columns(3)

                col4.metric("Recall", f"{recall:.4f}")
                col5.metric("F1 Score", f"{f1:.4f}")
                col6.metric("MCC Score", f"{mcc:.4f}")

                # STEP 12: CONFUSION MATRIX

                st.divider()

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_test,
                    predictions,
                    labels=[0, 1]
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual No Default", "Actual Default"],
                    columns=["Predicted No Default", "Predicted Default"]
                )

                st.dataframe(cm_df)

                # STEP 13: CLASSIFICATION REPORT

                st.subheader("Classification Report")

                report = classification_report(
                    y_test,
                    predictions,
                    labels=[0, 1],
                    target_names=["No Default", "Default"],
                    output_dict=True,
                    zero_division=0
                )

                report_df = pd.DataFrame(report).transpose()

                st.dataframe(report_df.round(4))

                # STEP 14: PREDICTION PREVIEW

                st.subheader("Prediction Preview")

                prediction_df = pd.DataFrame({
                    "Actual Default": y_test.values,
                    "Predicted Default": predictions
                })

                st.dataframe(prediction_df.head(10))

    except Exception as e:

        st.error(f"An error occurred: {e}")

else:

    st.info("Please upload your test dataset to continue.")