# California Housing Price Prediction: A High-Performance Approach

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-red.svg)](https://xgboost.ai/)

This project implements an optimized machine learning pipeline to predict housing prices in California. By combining advanced Feature Engineering with high-performance algorithms like XGBoost, we achieved an **R² Score of 0.8352**.

## 🚀 Performance Summary

| Model | R² Score | RMSE (USD) |
| :--- | :---: | :---: |
| **XGBoost Regressor** | **0.8352** | **$39,949.45** |
| Random Forest | 0.8062 | $43,321.56 |

## 🧠 The Strategy: Engineering Over Raw Data

The success of this model lies in the balance between socio-economic indicators and intrinsic property value.

### 1. Data Cleaning (Handling the "Censored" Ceiling)
The original dataset has a price cap at **$500,001**. This creates a horizontal artifact that misleads regression models. We removed these censored values to ensure the model learns a true, continuous price distribution.

### 2. Advanced Feature Engineering
*   **Geographic Peer Benchmarking:** We used K-Means to create **20 hyper-local clusters**. A new feature, `avg_price_proximity`, was engineered to provide a historical price anchor for each micro-region.
*   **Property Ratios:** Instead of raw counts, we used:
    *   `rooms_per_household`: Proxy for house size.
    *   `bedrooms_per_room`: Indicator of property type and density.
    *   `population_per_household`: Measure of neighborhood density.
*   **The Power of `median_income`:** Reintroduced as a proxy for infrastructure quality and neighborhood prestige, which are not directly measured in the physical dataset.

### 3. Model Optimization (The "Boost")
We tuned the XGBoost Regressor with an aggressive configuration:
*   `n_estimators=1000` with a slow `learning_rate=0.05`.
*   `max_depth=7` to capture non-linear interactions.
*   `subsample` and `colsample_bytree` at 0.8 to ensure robust generalization and prevent overfitting.

## 🛠️ Tech Stack
*   **Language:** Python
*   **Data Handling:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Machine Learning:** Scikit-Learn, XGBoost

## 📈 Key Insights
The `median_income` and our engineered `avg_price_proximity` emerged as the most influential features. This proves that while property size and age matter, the **local economic context** and **geographic benchmarking** are the true drivers of real estate value in California.

---
*Developed for Portfolio Showcase (Kaggle & GitHub)*
