# Customer-churn-Prediction
# 📦 Telco Customer Churn Prediction

## 📚 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Problem Statement](#problem-statement)
- [Approach](#approach)
- [Modeling](#modeling)
- [Results](#results)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 📖 Overview
This project aims to develop machine learning models to predict customer churn using the Telco Customer Churn dataset. The project follows a full ML pipeline: data cleaning, exploratory data analysis (EDA), feature engineering, model building, hyperparameter tuning, and evaluation.

---

## 📂 Dataset
- **Dataset Name**: Telco Customer Churn
- **Source**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Features include:
- Customer demographics (gender, senior citizen status, etc.)
- Services signed up for (internet, online security, etc.)
- Account information (contract type, payment method, monthly charges, etc.)
- Target variable: `Churn` (Yes/No)

---

## 📛 Problem Statement
**Business Problem**: Predict whether a customer will churn based on their demographics and services subscribed, helping the business retain customers.

---

## 🛠 Approach
1. **Data Preprocessing**
   - Handled missing values.
   - Encoded categorical variables.
   - Scaled numerical features.
   - Dropped irrelevant features like `customerID`.

2. **Exploratory Data Analysis (EDA)**
   - Visualized churn rates.
   - Analyzed feature distributions.
   - Generated correlation heatmaps.

3. **Model Building**
   - Trained baseline Logistic Regression model.
   - Built Random Forest, XGBoost, and LightGBM models.
   - Compared models using accuracy and ROC-AUC scores.

4. **Hyperparameter Tuning**
   - Optimized Random Forest using RandomizedSearchCV.

5. **Evaluation**
   - Used confusion matrix, classification report, accuracy score, and ROC-AUC score.

---

## 🤖 Modeling
| Model | Accuracy | ROC-AUC |
|:------|:--------|:--------|
| Logistic Regression | Baseline | Baseline |
| Random Forest | Improved | Improved |
| XGBoost | High | High |
| LightGBM | High | High |

Random Forest performance improved significantly after hyperparameter tuning.

---

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/telco-customer-churn.git
cd telco-customer-churn
```

2. Install required libraries:
```bash
pip install -r requirements.txt
```

Main libraries used:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm

---

## 🚀 How to Run

1. Open and run the Jupyter Notebook (`Telco-Customer-Churn-Prediction.ipynb`).
2. Follow the notebook sequentially through preprocessing, EDA, modeling, and evaluation.
3. Modify hyperparameters or models to improve performance further if desired.

---

## 🌟 Future Improvements
- Feature selection using SHAP values.
- Model deployment using Flask and Docker.
- Real-time churn prediction API.
- Integrate Explainable AI (XAI) techniques for model transparency.

---

## 📜 License
This project is licensed under the MIT License.

---

# ✨ Final Note
This project demonstrates:
- Full ML lifecycle (data cleaning, feature engineering, modeling, evaluation)
- Business-driven problem solving with machine learning
- Best practices for professional machine learning projects

Feel free to fork, clone, or contribute!


