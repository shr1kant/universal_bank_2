# 🏦 Universal Bank – Personal Loan Analytics Dashboard

A fully interactive **Streamlit** analytics dashboard analysing which customers are most likely to accept a Personal Loan offer, built on the Universal Bank dataset (5,000 customers, 14 features).

## 🌐 Live Demo
Deploy to [Streamlit Cloud](https://streamlit.io/cloud) in minutes – see deployment instructions below.

---

## 📊 Dashboard Sections

| Tab | Type | Contents |
|-----|------|----------|
| 📊 Descriptive | Who are our customers? | Distributions of age, income, education, family size, CC spend; loan acceptance donut |
| 🔍 Diagnostic | What drives acceptance? | Feature comparisons (accepted vs rejected), banking services analysis, correlation heatmap, key driver ranking |
| 🤖 Predictive | Who will accept? | 4 ML models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting), ROC curves, confusion matrix, live predictor |
| 🎯 Prescriptive | How should the bank act? | Segment profiler, strategic recommendations, campaign ROI estimator |
| 📈 Drill-Down | Interactive deep dives | Sunburst charts, treemap, parallel coordinates |

---

## 🚀 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/universal-bank-dashboard.git
cd universal-bank-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

> ⚠️ Make sure `UniversalBank.csv` is in the **same directory** as `app.py`.

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub (make sure `UniversalBank.csv` is included).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch (`main`) and set **Main file path** to `app.py`.
4. Click **Deploy** – done! 🎉

---

## 📁 File Structure

```
universal-bank-dashboard/
├── app.py                          # Main Streamlit dashboard
├── requirements.txt                # Python dependencies
├── UniversalBank.csv               # Dataset (5,000 rows)
└── README.md                       # This file
```

---

## 🔑 Key Findings

- Only **9.6%** of customers accepted a personal loan — a significant targeting opportunity.
- **Income** and **CC Average Spend** are the strongest predictors of acceptance.
- Customers with a **CD Account** convert at ~3× the base rate.
- **Advanced/Professional degree** holders have the highest acceptance rates.
- The best ML model (Random Forest / Gradient Boosting) achieves **~98% ROC-AUC**.

---

## 🛠️ Tech Stack

- **Streamlit** – dashboard framework  
- **Plotly** – interactive charts  
- **scikit-learn** – machine learning models  
- **pandas / numpy** – data processing  

---

## 📄 Dataset

The Universal Bank dataset contains 5,000 customer records with features including demographics (Age, Experience, Income, Family), financial behaviour (CCAvg, Mortgage), and banking service usage (Securities Account, CD Account, Online, CreditCard). The target variable is **Personal Loan** (0 = Rejected, 1 = Accepted).
