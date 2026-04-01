<div align="center">

<!-- BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20&height=200&section=header&text=ChurnLens%20AI&fontSize=70&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Customer%20Churn%20Prediction%20for%20Subscription%20Businesses&descAlignY=58&descAlign=50" width="100%"/>

<!-- BADGES ROW 1 -->
<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Enabled-189AB4?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/LightGBM-Enabled-00B96B?style=for-the-badge"/>
</p>

<!-- BADGES ROW 2 -->
<p>
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Optuna-Bayesian%20Tuning-0081FF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SMOTE-Imbalance%20Handled-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ROC--AUC-0.9421-00E396?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<h3>
  A production-grade, end-to-end Machine Learning system that predicts which subscription customers will churn —
  <em>before they do</em> — and explains exactly why.
</h3>

<br/>

[🔍 The Problem](#-the-problem) &nbsp;·&nbsp;
[🏗️ What I Built](#%EF%B8%8F-what-i-built) &nbsp;·&nbsp;
[✅ Problems It Resolves](#-problems-it-resolves) &nbsp;·&nbsp;
[🧱 System Architecture](#-system-architecture) &nbsp;·&nbsp;
[⚙️ Tech Stack](#%EF%B8%8F-tech-stack) &nbsp;·&nbsp;
[🚀 Quick Start](#-quick-start) &nbsp;·&nbsp;
[🔮 Future Improvements](#-future-improvements)

</div>

---

<br/>

## 🔍 The Problem

> **"You can't save a customer you didn't know was leaving."**

Every subscription business — SaaS, telecom, fintech, streaming, e-commerce — bleeds revenue silently through **customer churn**. The numbers are brutal:

```
📉  Acquiring a new customer costs 5–25× more than retaining an existing one
💸  A 5% reduction in churn increases profit by 25–95%  (Harvard Business Review)
🚨  Most churners give no warning — they just stop paying
🎯  Retention teams act too late — after the customer has already decided to leave
```

### The Core Pain Points

| # | Pain Point | Business Impact |
|---|-----------|----------------|
| 1 | **No early warning system** — churn detected only when the invoice fails | Revenue already lost by the time action is possible |
| 2 | **Reactive, not proactive** — CS teams call customers who already churned | Wasted effort, near-zero recovery rate |
| 3 | **No prioritisation** — every at-risk customer gets the same attention | High-value churners get lost in the noise |
| 4 | **Black-box guesswork** — nobody knows *why* specific customers are leaving | Interventions are generic, not personalised |
| 5 | **Manual analysis is slow** — weekly spreadsheet reviews miss real-time signals | Weeks of lag between signal and action |
| 6 | **Class imbalance** — churners are a minority (~26%), so naive models predict "No Churn" for everyone | 0% recall — the model misses all actual churners |

### Who Suffers From This

```
👔  VP of Customer Success     →  Reports lag. No real-time risk dashboard.
🧑‍💼  Customer Success Managers  →  Triage 500 accounts manually every single week.
📊  Data Analysts              →  Build static reports that are stale in 48 hours.
💰  Finance Teams              →  Churn surprises destroy quarterly forecasts.
🧑‍💻  Product Teams              →  No feedback loop on which features drive retention.
```

---

<br/>

## 🏗️ What I Built

**ChurnLens AI** is a complete, production-ready machine learning pipeline that solves the churn prediction problem from raw data all the way to an interactive deployed dashboard. This is not a notebook experiment — it is a deployable system built across three files with clear separation of concerns.

### Three Files, One System

```
churn_model.py   →  The Brain   (14-step ML pipeline: data → features → train → evaluate → save)
ui_styling.py    →  The Face    (All CSS, Plotly charts, UI components — cleanly separated)
app.py           →  The Product (5-tab Streamlit dashboard — the deployed tool)
```

### What the System Delivers

```
┌─────────────────────────────────────────────────────────────┐
│                   INPUT (Customer Profile)                   │
│     20 raw features: tenure · charges · NPS · contract ...  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           FEATURE ENGINEERING  (14 new signals)             │
│  billing_stress · engagement_score · satisfaction_risk ...  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         PREPROCESSING  (zero data leakage design)           │
│    KNN Imputation → Yeo-Johnson Transform → StandardScale   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         SMOTE OVERSAMPLING  (fixes class imbalance)         │
│              Minority 26%  →  Balanced 50 / 50              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              STACKING ENSEMBLE  (best model)                │
│  Base: Random Forest + XGBoost + LightGBM + Grad. Boosting  │
│  Meta: Logistic Regression   |   Optuna-tuned params        │
└───────┬───────────────────────────────────────┬─────────────┘
        │                                       │
        ▼                                       ▼
┌───────────────┐                   ┌───────────────────────┐
│   PREDICT     │                   │   EXPLAIN (SHAP)      │
│  0–100% prob  │                   │   Which features      │
│  Risk tier    │                   │   drove the score     │
│  Recommendation│                  │   per customer        │
└───────────────┘                   └───────────────────────┘
```

### Dashboard — 5 Tabs Built

| Tab | What It Does |
|-----|-------------|
| 🔮 **Predict** | Enter any customer profile → instant churn probability gauge + risk tier + SHAP waterfall + personalised retention playbook |
| 📊 **Dashboard** | Dataset analytics — churn by contract type, feature distributions, Pearson correlation heatmap |
| 🤖 **Model Intel** | Model comparison chart, feature importance ranking, ROC/PR curves, confusion matrix, SHAP plots |
| 📦 **Batch Score** | Upload CSV → score thousands of customers at once → download results with risk tier per row |
| 📚 **About** | Full project documentation embedded in the app itself |

---

<br/>

## ✅ Problems It Resolves

Every pain point from the problem section is directly addressed with a concrete solution.

---

### ❌ Problem 1 → ✅ Resolved: No Early Warning System

**Before:** Churn discovered when invoice fails — too late to act.

**After:** Score every customer with a probability from 0–100%. Customers above the threshold get flagged *weeks before* their renewal date.

```python
result = predict_churn(customer_data)
# → {"churn_probability": 0.783, "risk_tier": "High"}
# → Flagged 30 days before renewal. CSM intervenes. Customer retained.
```

---

### ❌ Problem 2 → ✅ Resolved: Reactive Teams

**Before:** Customer Success reacts to cancellations.

**After:** Adjustable decision threshold in the sidebar — set to 0.35 for maximum recall, catching at-risk customers 4–6 weeks earlier than the billing cycle.

```
Threshold 0.35  →  High Recall   →  Catch 89% of churners before they leave
Threshold 0.50  →  Balanced      →  Default setting
Threshold 0.70  →  High Precision →  Only flag near-certain churners
```

---

### ❌ Problem 3 → ✅ Resolved: No Prioritisation

**Before:** All at-risk customers treated equally. Retention budget wasted.

**After:** Three-tier risk system backed by the `high_value_at_risk` engineered feature:

```
🔴 High Risk   (prob > 65%)  →  Immediate CSM outreach + retention offer
🟡 Medium Risk (35–65%)      →  Automated email sequence + check-in call
🟢 Low Risk    (prob < 35%)  →  Standard engagement. No spend needed.

high_value_at_risk flag = (monthly_charges > 75th percentile) AND (nps < 5)
→ Ensures top-revenue accounts get priority even within the High tier
```

---

### ❌ Problem 4 → ✅ Resolved: Black-Box Guesswork

**Before:** "We think they might leave." No reason. Generic retention script.

**After:** SHAP (SHapley Additive exPlanations) tells you exactly why each customer scored the way they did:

```
Customer #4821 — Churn Probability: 78%

Feature contributions:
  contract_type (Month-to-Month)      +18%  ← biggest driver
  support_tickets_6mo  (8 tickets)    +14%  ← frustrated
  nps_score            (2 / 10)       +12%  ← dissatisfied
  tenure_months        (3 months)     + 9%  ← no loyalty yet
  feature_usage_score  (15 / 100)     + 8%  ← not engaged

Personalised Actions Generated:
  → Offer annual contract at 20% discount  (targets contract_type)
  → Escalate tickets to senior CSM         (targets support_tickets)
  → Send personalised onboarding sequence  (targets feature_usage)
```

---

### ❌ Problem 5 → ✅ Resolved: Slow Manual Analysis

**Before:** Analysts spend 8 hours/week on churn spreadsheets.

**After:** Batch prediction tab — upload a 10,000-row CSV, get risk scores in seconds:

```
Step 1 → Upload CSV  (10,000 customer rows)
Step 2 → Click "Score All Customers"          ← 4 seconds
Step 3 → Download churn_predictions.csv

Output:
  CustomerID   | Churn_Probability | Churn_Prediction | Risk_Tier
  CUST-00421   | 0.8234            | 1                | High
  CUST-00892   | 0.1821            | 0                | Low
  CUST-01334   | 0.5102            | 1                | Medium
  ...
```

---

<br/>

## 🧱 System Architecture

### Full Pipeline Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                       CHURNLENS AI — ARCHITECTURE                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   ┌─────────────────────────────────────────────────────────────────┐   ║
║   │                        DATA LAYER                               │   ║
║   │   churn_data.csv  →  10,000 customers · 20 features · 26% churn│   ║
║   │   3% missingness in nps_score, feature_usage, response_time     │   ║
║   └────────────────────────┬────────────────────────────────────────┘   ║
║                            │                                             ║
║                            ▼                                             ║
║   ┌─────────────────────────────────────────────────────────────────┐   ║
║   │                 FEATURE ENGINEERING LAYER                       │   ║
║   │                                                                  │   ║
║   │   Raw (20)              Engineered (14)         Flags (4)       │   ║
║   │   ─────────             ──────────────          ──────────      │   ║
║   │   tenure                billing_stress_index    is_month_to_m   │   ║
║   │   monthly_charges       engagement_score        is_senior       │   ║
║   │   nps_score             satisfaction_risk       uses_fiber      │   ║
║   │   support_tickets       charge_per_product      no_internet     │   ║
║   │   login_frequency       nps_x_usage                             │   ║
║   │   ...                   high_value_at_risk ★                    │   ║
║   └────────────────────────┬────────────────────────────────────────┘   ║
║                            │                                             ║
║                            ▼                                             ║
║   ┌─────────────────────────────────────────────────────────────────┐   ║
║   │             PREPROCESSING LAYER  (ColumnTransformer)            │   ║
║   │                                                                  │   ║
║   │   Numerical Branch                  Categorical Branch          │   ║
║   │   ────────────────                  ─────────────────           │   ║
║   │   KNNImputer (k=5)                  SimpleImputer (mode)        │   ║
║   │         ↓                                 ↓                     │   ║
║   │   PowerTransformer (Yeo-Johnson)    OrdinalEncoder (unk=-1)     │   ║
║   │         ↓                                                        │   ║
║   │   StandardScaler                                                 │   ║
║   │                                                                  │   ║
║   │   ★ Fit ONLY on train split — transform applied to val & test   │   ║
║   └────────────────────────┬────────────────────────────────────────┘   ║
║                            │                                             ║
║               ┌────────────▼────────────┐                               ║
║               │   SMOTE  (train only)   │                               ║
║               │   26% → 50% minority    │                               ║
║               └────────────┬────────────┘                               ║
║                            │                                             ║
║                            ▼                                             ║
║   ┌─────────────────────────────────────────────────────────────────┐   ║
║   │                      MODEL LAYER                                │   ║
║   │                                                                  │   ║
║   │   LEVEL 0 — Base Models  (5-Fold CV · Optuna Bayesian Tuning)  │   ║
║   │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │   ║
║   │   │  Random  │ │ XGBoost  │ │ LightGBM │ │ Gradient Boosting│ │   ║
║   │   │  Forest  │ │          │ │          │ │                  │ │   ║
║   │   │  0.9201  │ │  0.9289  │ │  0.9318  │ │     0.9134       │ │   ║
║   │   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘ │   ║
║   │        └────────────┴────────────┴─────────────────┘           │   ║
║   │                            │                                    │   ║
║   │           OOF Predictions + Original Features (passthrough=True)│   ║
║   │                            │                                    │   ║
║   │   LEVEL 1 — Meta Learner                                       │   ║
║   │                   ┌────────────────────────┐                   │   ║
║   │                   │   Logistic Regression  │                   │   ║
║   │                   │   ROC-AUC = 0.9421  ★  │                   │   ║
║   │                   └────────────────────────┘                   │   ║
║   └────────────────────────┬────────────────────────────────────────┘   ║
║                            │                                             ║
║                            ▼                                             ║
║   ┌─────────────────────────────────────────────────────────────────┐   ║
║   │            EVALUATION & EXPLAINABILITY LAYER                    │   ║
║   │   ROC-AUC · PR-AUC · F1 · Confusion Matrix · SHAP Explainer    │   ║
║   └────────────────────────┬────────────────────────────────────────┘   ║
║                            │                                             ║
║                            ▼                                             ║
║   ┌─────────────────────────────────────────────────────────────────┐   ║
║   │                   PERSISTENCE LAYER                             │   ║
║   │   model_artefacts/                                              │   ║
║   │     preprocessor.joblib   stacking_model.joblib   rf_model.pkl  │   ║
║   │     feature_meta.pkl      model_card.pkl / .txt                 │   ║
║   └────────────────────────┬────────────────────────────────────────┘   ║
║                            │                                             ║
║                            ▼                                             ║
║   ┌─────────────────────────────────────────────────────────────────┐   ║
║   │             DEPLOYMENT LAYER  (Streamlit + ui_styling.py)       │   ║
║   │   ┌──────────┬──────────┬───────────┬──────────┬─────────────┐ │   ║
║   │   │🔮 Predict│📊 Dashbrd│🤖 Model   │📦 Batch │📚 About     │ │   ║
║   │   └──────────┴──────────┴───────────┴──────────┴─────────────┘ │   ║
║   └─────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

<br/>

### Data Flow — Single Customer Prediction

```
User fills 20-field form in app.py
          │
          ▼
14 engineered features computed automatically from raw inputs
          │
          ▼
pd.DataFrame reindexed to exact training column order (prevents feature mismatch)
          │
          ▼
preprocessor.transform()
  → KNN fills any missing values using 5 nearest neighbours
  → Yeo-Johnson corrects skewed distributions
  → StandardScaler normalises to zero mean, unit variance
          │
          ▼
stacking_model.predict_proba()
  → RF, XGB, LightGBM, GB each produce a probability
  → Meta-learner (LR) combines them using learned weights
  → Output: [0.217, 0.783]
          │
          ├──►  churn_probability = 0.783
          ├──►  threshold check → prediction = 1 (Churn)
          ├──►  risk tier → "High" (prob > 0.65)
          └──►  SHAP approximation → feature contribution waterfall
```

<br/>

### File Structure

```
CHURN_PREDICTION/
│
├── 🧠 churn_model.py            ← Full 14-step ML pipeline (791 lines)
│     ├── STEP 01  Imports & reproducibility seed
│     ├── STEP 02  Synthetic dataset — 10,000 rows, logistic DGP
│     ├── STEP 03  EDA — distributions + Pearson correlation heatmap
│     ├── STEP 04  Feature engineering — 14 derived signals
│     ├── STEP 05  Preprocessing pipeline — no data leakage
│     ├── STEP 06  Stratified 70 / 15 / 15 split
│     ├── STEP 07  SMOTE balancing on train split only
│     ├── STEP 08  5-model cross-validation zoo
│     ├── STEP 09  Optuna Bayesian tuning — 30 TPE trials
│     ├── STEP 10  Stacking ensemble — 4 base + 1 meta
│     ├── STEP 11  Full evaluation suite — 6 metrics
│     ├── STEP 12  SHAP TreeExplainer — global + local
│     ├── STEP 13  Pickle + joblib dual persistence
│     └── STEP 14  Inference helper — predict_churn()
│
├── 🎨 ui_styling.py             ← Separated UI layer (630 lines)
│     ├── PALETTE & RISK_COLORS design tokens
│     ├── inject_css()           dark-theme CSS (Google Fonts, glassmorphism)
│     ├── Component functions    hero, metric card, risk badge, result card
│     └── Plotly chart builders  gauge, donut, bar, heatmap, waterfall
│
├── 🌐 app.py                    ← Streamlit deployment (654 lines)
│     ├── @st.cache_resource     load_artefacts() — loads once, reuses
│     ├── @st.cache_data         load_dataset() + load_model_comparison()
│     ├── predict_customer()     full inference pipeline
│     └── 5 tabs                 Predict · Dashboard · Model Intel · Batch · About
│
├── 📋 requirements.txt
├── 📖 README.md
│
├── 📊 churn_data.csv
├── 📈 model_comparison.csv
├── 🖼️ eda_distributions.png
├── 🖼️ eda_correlation.png
├── 🖼️ model_roc_pr.png
├── 🖼️ confusion_matrix.png
├── 🖼️ shap_summary.png
└── 🖼️ shap_bar.png
│
└── 📦 model_artefacts/
      ├── preprocessor.joblib / .pkl
      ├── stacking_model.joblib / .pkl
      ├── rf_model.joblib / .pkl
      ├── feature_meta.pkl
      ├── model_card.pkl
      └── model_card.txt
```

---

<br/>

## ⚙️ Tech Stack

### Core ML & Data Science

| Library | Version | Role in This Project |
|---------|---------|---------------------|
| **Python** | 3.9+ | Core language |
| **NumPy** | 1.24+ | Numerical computing, synthetic data generation, random seeding |
| **Pandas** | 2.0+ | Data manipulation, feature engineering, DataFrame pipelines |
| **Scikit-Learn** | 1.3+ | ColumnTransformer, KNNImputer, PowerTransformer, StratifiedKFold, StackingClassifier, all metrics |

### Advanced ML

| Library | Version | Role in This Project |
|---------|---------|---------------------|
| **XGBoost** | 1.7+ | Base model — extreme gradient boosting with regularisation |
| **LightGBM** | 4.0+ | Base model — leaf-wise boosting, fastest on large data |
| **imbalanced-learn** | 0.11+ | SMOTE — generates synthetic minority class examples |
| **Optuna** | 3.3+ | Bayesian hyperparameter search — TPE sampler, 30 trials |
| **SHAP** | 0.43+ | TreeExplainer — exact Shapley values for feature attribution |

### Visualisation & Deployment

| Library | Version | Role in This Project |
|---------|---------|---------------------|
| **Streamlit** | 1.28+ | Full dashboard — 5 tabs, sidebar, caching, file upload |
| **Plotly** | 5.15+ | Interactive charts — gauge, donut, grouped bar, heatmap, waterfall |
| **Matplotlib** | 3.7+ | Static EDA plots saved as PNG |
| **Seaborn** | 0.12+ | Styled correlation heatmap |

### Persistence

| Library | Role |
|---------|------|
| **Joblib** | Fast serialisation for large numpy arrays — preferred for sklearn objects |
| **Pickle** | Universal Python serialisation — maximum cross-environment portability |

<br/>


---

<br/>

### Train the Model

```bash
python churn_model.py
```

### Launch the Dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser 🎉

---

<br/>

## 📊 Results

### Model Comparison (Test Set)

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|-------|:-------:|:------:|:--:|:---------:|:------:|
| ⭐ **Stacking Ensemble** | **0.9421** | **0.8934** | **0.8283** | 0.9124 | 0.7584 |
| LightGBM | 0.9318 | 0.8812 | 0.8156 | 0.8987 | 0.7481 |
| XGBoost | 0.9289 | 0.8779 | 0.8098 | 0.8901 | 0.7422 |
| Random Forest | 0.9201 | 0.8654 | 0.7992 | 0.8856 | 0.7298 |
| Gradient Boosting | 0.9134 | 0.8589 | 0.7934 | 0.8823 | 0.7198 |
| Logistic Regression | 0.8876 | 0.8201 | 0.7621 | 0.8512 | 0.6901 |

### Top Churn Drivers (Global SHAP)

```
Feature                     Mean |SHAP Value|    Direction
────────────────────────────────────────────────────────────
contract_type               ████████████████  0.312   ↑ churn
tenure_months               ███████████████   0.287   ↓ churn (longer = safer)
satisfaction_risk           ██████████████    0.261   ↑ churn
nps_score                   █████████████     0.234   ↓ churn (higher = safer)
engagement_score            ████████████      0.198   ↓ churn
monthly_charges             ███████████       0.187   ↑ churn
billing_stress_index        █████████         0.156   ↑ churn
feature_usage_score         ████████          0.143   ↓ churn
login_frequency_weekly      ███████           0.121   ↓ churn
support_tickets_6mo         ███████           0.118   ↑ churn
```

---

<br/>

## 🔮 Future Improvements

The current system is production-ready. The roadmap below shows the natural evolution from single-model dashboard to enterprise MLOps platform.

---

### 🔵 Short Term — 1 to 3 Months

| # | Improvement | Why It Matters |
|---|------------|----------------|
| 1 | **Real customer data integration** — Replace synthetic data with CRM/billing exports | Synthetic data is a stand-in; real data reveals distribution shifts and new feature opportunities |
| 2 | **Per-segment thresholds** — Different thresholds for SMB vs Enterprise vs Consumer | One threshold doesn't fit all — a 35% threshold for a $10k/month account vs 65% for a $10/month |
| 3 | **Automated feature selection** — Add Boruta or RFECV to prune weak features | Fewer features = faster inference, less overfitting, simpler maintenance in production |
| 4 | **Real SHAP in prediction** — Replace approximation with actual TreeExplainer values | The current SHAP waterfall in the dashboard is proportional approximation; exact Shapley adds precision |
| 5 | **CatBoost in model zoo** — Add as a fifth base model | Handles high-cardinality categoricals natively — no OrdinalEncoding needed |

---

### 🟡 Medium Term — 3 to 6 Months

| # | Improvement | Why It Matters |
|---|------------|----------------|
| 6 | **MLflow experiment tracking** — Log every trial's params, metrics, artefacts | No current audit trail of what was tried; MLflow makes experiments reproducible and comparable |
| 7 | **Automated weekly retraining** — Cron job that retrains on latest data | Models decay as customer behaviour shifts; automated retraining keeps performance stable without manual effort |
| 8 | **Survival analysis (Cox / Kaplan-Meier)** — Predict not just *if* but *when* | Gives CSMs a churn deadline — "this customer will likely leave within 14 days" — enabling better timing of interventions |
| 9 | **A/B testing framework** — Track whether interventions actually reduce churn | Without A/B testing you cannot prove the model creates measurable business value |
| 10 | **Email / Slack alert integration** — Auto-notify CSMs when customer crosses threshold | The model is only valuable if CSMs act on it; removing notification friction is the biggest adoption driver |

---

### 🔴 Long Term — 6 to 12 Months

| # | Improvement | Why It Matters |
|---|------------|----------------|
| 11 | **FastAPI REST endpoint** — Replace Streamlit batch CSV with a real-time API | Production systems need millisecond inference via API; file upload workflow doesn't scale |
| 12 | **Neural network baseline (TabNet / FT-Transformer)** — Test deep learning on tabular data | Transformers are showing strong tabular results — worth benchmarking against tree ensembles |
| 13 | **Customer Lifetime Value integration** — Weight predictions by predicted revenue impact | Prioritise retention on highest-value customers, not just highest-probability churners |
| 14 | **Drift detection (Evidently AI)** — Monitor feature and prediction distributions | Models silently degrade when data distributions shift; drift detection triggers automated retraining alerts |
| 15 | **Docker + CI/CD pipeline** — Containerise with automated testing and deployment | Environment-independent deployment; eliminates "works on my machine" issues for team collaboration |
| 16 | **Multi-tenant SaaS version** — Separate model instance per client | Each business has unique churn patterns; one global model is always inferior to client-specific models |

<br/>


##  **Demo video and screenshot**

<img width="1885" height="912" alt="Screenshot 2026-03-31 102709" src="https://github.com/user-attachments/assets/edaaeaa8-8859-43c9-b23e-63d7b06aeaf5" />
<img width="1609" height="909" alt="Screenshot 2026-03-31 102748" src="https://github.com/user-attachments/assets/56eecb9d-159f-49d0-83cd-9da8b62ce76a" />
<img width="1503" height="873" alt="Screenshot 2026-03-31 102856" src="https://github.com/user-attachments/assets/b6197fa3-2db2-4af8-b317-b894bb02b368" />
<img width="1533" height="835" alt="Screenshot 2026-03-31 102922" src="https://github.com/user-attachments/assets/e15ac653-898b-4cf5-957d-61520e3527b0" />
![demo](https://github.com/user-attachments/assets/65dc699b-b468-4dfb-bb88-088f732e8ea0)


## 📄 License

```
MIT License — free to use, modify, and distribute with attribution.
```

---

<br/>

<div align="center">

**Built with ❤️ · Python · Scikit-Learn · XGBoost · LightGBM · Optuna · SHAP · Streamlit**

<br/>

⭐ **Star this repo if it helped you — it helps others find it.**

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20&height=100&section=footer" width="100%"/>

</div>
