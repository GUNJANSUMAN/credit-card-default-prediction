# Credit Card Default Prediction System

## 1. Project Overview

This project implements a machine learning-based Credit Card Default Prediction System using the UCI Default of Credit Card Clients dataset.

The objective is to develop and compare different classification algorithms for predicting whether a credit card customer is likely to default on payment.

An interactive web application has also been developed using Streamlit to allow users to upload test data, select a trained classification model and evaluate its predictive performance.

## 2. Dataset

**Dataset:** Default of Credit Card Clients

**Source:** UCI Machine Learning Repository

**Observations:** 30,000

**Predictor Features:** 23

**Target Variable:** `default.payment.next.month`

## 3. Machine Learning Models

The following five classification algorithms have been implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbours
4. Gaussian Naive Bayes
5. Support Vector Machine

## 4. Model Evaluation

The models are evaluated using the following performance metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## 5. Streamlit Application

The application provides the following features:

- CSV test dataset upload
- Classification model selection
- Dataset preview
- Prediction generation using saved trained models
- Model performance metrics
- Confusion matrix
- Classification report
- Prediction preview

## 6. Project Structure

```text
ML ASSIGNMENT 2/
│
├── app.py
├── CODE.ipynb
├── requirements.txt
├── README.md
├── test_data.csv
├── model_metrics.csv
├── model_ranking.csv
│
└── model/
    ├── [saved model files]
```

## 7. How to Run the Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in a local web browser.

## 8. Conclusion

The project demonstrates the implementation, evaluation and comparison of five machine learning classification algorithms, along with their integration into an interactive Streamlit application.