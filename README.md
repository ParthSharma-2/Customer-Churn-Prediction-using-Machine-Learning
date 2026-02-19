# 📉 Customer Churn Prediction using Machine Learning

End-to-End Machine Learning project to predict customer churn using advanced modeling, hyperparameter tuning, threshold optimization, and SHAP-based explainability.

---

## 📌 Project Highlights

- ✅ Final Model: **Tuned Gradient Boosting Classifier**
- ✅ ROC-AUC: **0.845**
- ✅ Recall (Churn Class): **0.63** (Improved from 0.54 baseline)
- ✅ Threshold Optimized to **0.4**
- ✅ SHAP-based model explainability implemented
- ✅ Identified key business drivers of churn

---

## 🎯 Business Problem

Customer churn directly impacts revenue and profitability. Acquiring new customers is significantly more expensive than retaining existing ones.

The goal of this project is to:

- Identify customers at high risk of churn
- Enable proactive retention strategies
- Reduce revenue loss
- Improve customer lifetime value (CLV)

---

## 🧠 Machine Learning Formulation

- **Task:** Supervised Binary Classification  
- **Target Variable:** Churn (0 = Retained, 1 = Churned)  
- **Dataset:** Telco Customer Churn dataset (~7,000 customers)  
- **Churn Rate:** ~26%

The model predicts the probability that a customer will discontinue services.

---

## ⚙️ Project Workflow

1. Dataset Understanding  
2. Exploratory Data Analysis (EDA)  
3. Data Cleaning & Preprocessing  
4. Feature Engineering  
5. Baseline Modeling (Logistic Regression)  
6. Advanced Models (Random Forest, Gradient Boosting)  
7. Hyperparameter Tuning (GridSearchCV)  
8. Threshold Optimization (Business-Aligned)  
9. SHAP Explainability & Business Insight Extraction  

---

## 📊 Model Performance

| Model | ROC-AUC | Recall (Churn) |
|--------|----------|----------------|
| Logistic Regression | 0.84 | 0.54 |
| Random Forest (Optimized Threshold) | 0.83 | 0.62 |
| **Tuned Gradient Boosting (Final)** | **0.845** | **0.63** |

### 🔍 Threshold Optimization

By adjusting the classification threshold from 0.5 to 0.4:

- Recall improved from **54% → 63%**
- Identified **32 additional churners** compared to baseline
- Improved business impact without significant precision loss

---

## 🔎 Explainability using SHAP

SHAP (SHapley Additive exPlanations) was implemented to provide both:

- Global feature importance
- Individual customer-level prediction explanations

### 🔑 Top Drivers of Churn

- Short tenure
- Fiber optic internet service
- Month-to-month contracts
- Electronic check payment method
- Higher monthly charges

SHAP analysis confirmed that:

- Long-term contracts and higher tenure significantly reduce churn probability.
- Short tenure and premium service plans increase churn risk.

---

## 🏆 Key Business Insights

- New customers are at highest churn risk → Improve onboarding strategy.
- Long-term contracts significantly reduce churn → Incentivize yearly subscriptions.
- Fiber optic customers show higher churn tendency → Improve service quality & support.
- Threshold tuning can significantly improve retention targeting effectiveness.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- SHAP
- Joblib

---

## 📁 Repository Structure

```
customer-churn-prediction/
│
├── notebooks/
│   └── customer_churn_master.ipynb
│
├── models/
│   ├── final_model.pkl
│   └── scaler.pkl
│
├── requirements.txt
└── README.md
```

---

## 🚀 Future Improvements

- Cost-sensitive learning
- Real-time deployment (Streamlit / FastAPI)
- Business ROI simulation
- Cross-dataset validation
- Automated ML pipeline structuring

---

## 📌 Conclusion

This project demonstrates:

- End-to-end ML pipeline development
- Structured experimentation & model comparison
- Hyperparameter tuning with cross-validation
- Business-aligned threshold optimization
- Advanced explainability using SHAP
- Actionable strategic insights

The final solution balances performance, interpretability, and business value.

---

⭐ If you found this project useful, feel free to star the repository.
