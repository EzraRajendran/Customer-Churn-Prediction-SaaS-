# 🔮 Customer Churn Prediction — SaaS Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **An end-to-end machine learning project** that predicts whether a SaaS customer will churn, with an interactive Streamlit web app for real-time inference.

**Built by [Ezra Rajendran](https://www.linkedin.com/in/ezra-rajendran788380218/)** | Data Analyst · Mumbai, India

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app)

---

## 📌 Problem Statement

Customer churn is one of the most expensive problems for SaaS businesses. Acquiring a new customer costs **5–7x more** than retaining an existing one. This project builds a classification model to **predict which customers are likely to churn**, enabling proactive retention strategies.

---

## 🗂️ Project Structure

```
customer-churn-prediction/
│
├── app.py                    ← Streamlit web app (entry point)
├── requirements.txt          ← Python dependencies
├── .streamlit/
│   └── config.toml           ← Streamlit theme config
│
├── data/
│   ├── generate_data.py      ← Synthetic dataset generator
│   └── churn_data.csv        ← Generated dataset (2,000 rows)
│
├── src/
│   └── train_model.py        ← Model training + evaluation pipeline
│
├── models/
│   ├── churn_model.pkl       ← Saved trained model pipeline
│   └── model_meta.json       ← Model metadata & comparison results
│
└── notebooks/
    ├── eda.py                ← Exploratory Data Analysis script
    ├── eda_dashboard.png     ← EDA visualizations
    ├── model_evaluation.png  ← ROC curve, confusion matrix, comparison
    └── feature_importance.png← Top feature importances
```

---

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| `tenure` | int | Months customer has been active |
| `monthly_charges` | float | Monthly subscription charge |
| `total_charges` | float | Total amount paid |
| `num_products` | int | Number of products subscribed |
| `support_tickets` | int | Tickets raised in last 3 months |
| `login_frequency` | int | App logins per month |
| `contract_type` | categorical | Month-to-month / 1yr / 2yr |
| `payment_method` | categorical | Electronic check / Card / etc. |
| `senior_citizen` | binary | 0 = No, 1 = Yes |
| `tech_support` | binary | Has tech support add-on? |
| `online_backup` | binary | Has online backup add-on? |
| **`churn`** | **target** | **0 = Retained, 1 = Churned** |

- **2,000 rows** | **20.3% churn rate** | **No missing values**

---

## 🤖 ML Pipeline

```
Raw CSV
  └─► Feature Engineering
        ├─ Numerical: StandardScaler
        └─ Categorical: OneHotEncoder
              └─► Model Training (3 algorithms compared)
                    ├─ Logistic Regression   → AUC: 0.7807 ✅ Best
                    ├─ Random Forest         → AUC: 0.7702
                    └─ Gradient Boosting     → AUC: 0.7677
                          └─► Saved Pipeline (.pkl)
                                └─► Streamlit App (real-time inference)
```

### Model Performance

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| **Logistic Regression** | **0.7807** | **0.7522** |
| Random Forest | 0.7702 | — |
| Gradient Boosting | 0.7677 | — |

---

## ⚙️ Setup & Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate dataset & train model
```bash
python data/generate_data.py    # Creates data/churn_data.csv
python src/train_model.py       # Trains model, saves to models/
python notebooks/eda.py         # Optional: generates EDA visuals
```

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this entire folder to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch: `main`, main file: `app.py`
4. Click **Deploy** — live in ~2 minutes!

> ⚠️ Make sure `models/churn_model.pkl` and `models/model_meta.json` are committed to GitHub (they are small files, ~200KB each).

---

## 📸 Screenshots

| Input Form | Prediction Result |
|---|---|
| *(screenshot)* | *(screenshot)* |

---

## 🔬 Key Insights from EDA

- **Month-to-month contracts** have the highest churn rate (~35%)
- **Customers with <6 months tenure** are 3x more likely to churn
- **Support tickets > 3** is a strong churn signal
- **Low login frequency (<5/month)** correlates strongly with churn
- Customers subscribed to **3+ products** have lower churn (bundling effect)

---

## 🗺️ Future Improvements

- [ ] Add SHAP explainability (feature-level per-customer explanation)
- [ ] Connect to live database (PostgreSQL / Snowflake)
- [ ] Add batch prediction (CSV upload)
- [ ] Add email alert for high-risk customers
- [ ] Retrain pipeline with real-world Telco churn dataset

---

## 👤 About the Author

**Ezra Rajendran** — Data Analyst | MIS Executive | ML Practitioner

- 📍 Mumbai, Maharashtra, India
- 💼 [LinkedIn](https://www.linkedin.com/in/ezra-rajendran788380218/)
- 📝 [Blog](https://ezraraj06.blogspot.com)
- 📧 ezraraj15@gmail.com

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.
