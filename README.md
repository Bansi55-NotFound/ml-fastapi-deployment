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

--------------------------------------------------------------------------

## 🏗️ System Design & Architecture

This project follows a clear separation of concerns between
model training, inference, and data engineering workflows.

### 1️⃣ Model Training (Offline)
- Data preprocessing and feature engineering performed offline
- ML pipeline created using scikit-learn
- Model evaluated using metrics like confusion matrix, precision, recall, and ROC-AUC
- Trained pipeline saved as a serialized artifact (`joblib`)

### 2️⃣ Real-Time Inference (FastAPI)
- FastAPI service loads the trained model once at startup
- Input data is validated using Pydantic schemas
- Feature name mapping is applied to ensure compatibility with training schema
- API returns prediction, label, and confidence score
- Designed for low-latency, single-record predictions

### 3️⃣ Batch Inference (Data Engineering Workflow)
- Batch job reads input data from CSV
- Input schema and null checks are performed before prediction
- Feature names are mapped to training format
- Model is reused for batch scoring
- Predictions and probabilities are written back to an output CSV
- Structured logging is used for observability and debugging

### 4️⃣ Production Considerations
- Separation of training and inference
- Fail-fast validation to prevent silent data issues
- Logging instead of print statements
- Batch and real-time inference supported using the same model artifact

## ⚙️ Inference Strategies: API vs Batch vs Spark

This project demonstrates multiple inference strategies depending on
data volume and latency requirements.

### 🔹 1. Real-Time Inference (FastAPI)
**Use when:**
- Low-latency predictions are required
- Single record or small payloads
- Integration with applications or services

**Example:**
- User submits data via API
- Immediate prediction response

**Tech:**
- FastAPI
- Pydantic validation
- scikit-learn pipeline

---

### 🔹 2. Batch Inference (Pandas)
**Use when:**
- Periodic batch predictions (daily/weekly)
- Data fits in memory
- Simpler batch workflows

**Example:**
- CSV file with multiple records
- Output predictions written to CSV

**Tech:**
- Pandas
- scikit-learn
- Structured logging
- Schema & null validation

---

### 🔹 3. Scalable Batch Inference (Spark + Pandas)
**Use when:**
- Large datasets
- Distributed ingestion and validation required
- ML model is built using scikit-learn

**Design Choice:**
- Spark is used for data loading, validation, and scaling
- Data is converted to Pandas for ML inference since scikit-learn
  does not operate on Spark DataFrames

**Tech:**
- PySpark
- Pandas
- scikit-learn
- Shared model artifact

---

### 🧠 Key Design Principle
> Use Spark for distributed data processing  
> Use Pandas / scikit-learn for ML inference

This approach balances scalability and model flexibility without
rewriting the ML pipeline.
