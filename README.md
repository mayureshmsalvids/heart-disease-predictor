# 💓 Heart Disease Prediction using Machine Learning

## 📘 Overview

This project aims to build a machine learning model that predicts the **probability of a person having heart disease** based on medical and lifestyle factors. The model is developed as part of AIHealth’s mission to leverage data science in solving pressing healthcare problems.

---

## 🧠 Problem Statement

Heart disease is the **leading cause of death in the United States**, with major risk factors including high blood pressure, high cholesterol, smoking, poor diet, and diabetes. Early prediction can save lives and reduce medical costs. 

The objective is to:
- Train the best binary classification model to detect the risk of heart disease.
- Identify key contributing factors using explainability techniques.
- Deploy the model with an interactive user interface.

---

## 📊 Dataset

**Target Variable:**  
`HeartDiseaseorAttack` — Indicates whether the person has had heart disease or a heart attack (1: Yes, 0: No)

**Features Include:**
- Medical Indicators: `HighBP`, `HighChol`, `BMI`, `Diabetes`, `PhysHlth`, `MentHlth`
- Lifestyle: `Smoker`, `Fruits`, `Veggies`, `PhysActivity`, `HvyAlcoholConsump`
- Demographics: `Sex`, `Age`, `Education`, `Income`

Source: Open-source public health dataset.

---

## 📈 Workflow

### 1. Data Preprocessing
- Handled missing values using **mean imputation**.
- Scaled numerical features using **StandardScaler**.
- Performed **train-test split with stratification** to maintain class balance.

### 2. Class Imbalance Handling
- The dataset had significant imbalance (only 9.4% positive cases).
- Performance metrics focus on **ROC AUC** and **recall** in addition to accuracy.

### 3. Models Trained
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ Support Vector Machine (SVM)
- ✅ XGBoost

### 4. Evaluation
- Metrics: Accuracy, ROC AUC, Confusion Matrix, Classification Report
- **Precision-Recall curve** analysis for threshold tuning
- Plotted **ROC curves** for model comparison

### 5. Explainability
- Used **Feature Importance** (Random Forest)
- Applied **SHAP (SHapley Additive Explanations)** to identify key features affecting prediction

---

## 🏆 Best Model

| Model               | Accuracy | ROC AUC |
|--------------------|----------|---------|
| Logistic Regression| 0.894    | 0.789   |
| Random Forest      | 0.891    | 0.765   |
| XGBoost            | 0.879    | 0.747   |
| SVM                | 0.894    | 0.634   |

✅ **Final Model:** `Logistic Regression`  
📦 Saved as: `Logistic_Regression_model.pkl`

---

## 🔍 Key Drivers of Heart Disease (from SHAP & Feature Importance)

- Smoking
- Age
- Cholesterol
- High Blood Pressure
- BMI
- Mental and Physical Health

---

## 🚀 Streamlit App

An interactive frontend (`heart_app.py`) is created using **Streamlit** for real-time prediction.

### ▶ Features:
- User inputs basic health parameters
- Model returns probability and risk status
- Easy UI for non-technical users

### 🔧 Backend Tools:
- `Random_Forest_model.pkl` (optional switch)
- `scaler.pkl`
- `imputer.pkl`

### 💡 To Run:
```bash
streamlit run heart_app.py
```

---

## 🗃️ Files Included

| File                    | Description                                      |
|-------------------------|--------------------------------------------------|
| `Capstone.ipynb`        | Jupyter notebook with complete EDA + modeling    |
| `heart_app.py`          | Streamlit app code                               |
| `scaler.pkl`            | Standard Scaler used for preprocessing           |
| `imputer.pkl`           | SimpleImputer for handling missing values        |
| `Logistic_Regression_model.pkl` | Final trained model                    |

---

## 🙋 Author

**Mayuresh M. Salvi**  
*Data Scientist | Healthcare ML Enthusiast*

---

## 📌 Future Improvements

- Implement ensemble stacking of models
- Add external validation datasets
- Deploy via cloud (Heroku, AWS, etc.)
- Integrate alert system for high-risk patients
