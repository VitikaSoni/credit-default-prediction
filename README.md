# Credit Default Prediction

Predicting whether a credit card client will default on payment next month, using the UCI/Kaggle "Default of Credit Card Clients" dataset (30,000 clients, Taiwan, 2005).

## Problem
Given a client's credit limit, demographics, repayment history, and bill/payment amounts, predict the binary outcome: will they default next month?

## Approach
- Cleaned undocumented category codes in `EDUCATION` and `MARRIAGE`
- Trained and compared three models: Logistic Regression, Random Forest, and XGBoost
- Addressed class imbalance (22% default rate) using `class_weight` / `scale_pos_weight`
- Evaluated using AUC and classification metrics (precision, recall, F1)
- Used SHAP to interpret feature importance and identify key default risk drivers

## Results

| Model | AUC |
|---|---|
| Logistic Regression | 0.7157 |
| Random Forest | 0.7755 |
| XGBoost | 0.7755 |

**Key finding:** Recent repayment status (`PAY_0`) is the strongest predictor of default risk, ahead of credit limit, bill amounts, and demographic variables. This aligns with standard credit risk practice, where recent payment behavior is generally more predictive than static client attributes.

## Tools
Python, pandas, scikit-learn, XGBoost, SHAP

## Dataset
[UCI Machine Learning Repository / Kaggle — Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
