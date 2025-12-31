# Breast Cancer Prediction API (FastAPI + ML)

This project demonstrates how to deploy a trained Machine Learning model
as a REST API using **FastAPI**.  
The model predicts whether a breast tumor is **malignant (cancerous)** or
**benign (non-cancerous)** based on medical features.

This project focuses on **ML system design**, not just model training.

---

## 🚀 Project Overview

- Trained a classification model using the Breast Cancer dataset
- Built an end-to-end ML pipeline (preprocessing + model)
- Saved the trained model using `joblib`
- Served predictions using **FastAPI**
- Implemented proper input validation using **Pydantic**
- Handled real-world deployment issues like feature name mismatches

---

## 🧠 Problem Statement

Given numeric features extracted from breast mass images,
predict whether the tumor is:
- **Malignant (0)** → Cancerous
- **Benign (1)** → Non-cancerous

---

## 📊 Dataset

- Breast Cancer Wisconsin Dataset (sklearn)
- 30 numerical features:
  - Mean features
  - Error features
  - Worst features
- Target:
  - `0 = malignant`
  - `1 = benign`

---

## 🏗️ Model Details

- Algorithm: Logistic Regression / Random Forest (pipeline-based)
- Preprocessing:
  - Feature scaling
- Evaluation (offline):
  - Confusion Matrix
  - Precision
  - Recall
  - ROC-AUC

⚠️ Evaluation metrics are computed **offline** during training,
not inside the API.

---

## 🔌 API Design

### Health Check
