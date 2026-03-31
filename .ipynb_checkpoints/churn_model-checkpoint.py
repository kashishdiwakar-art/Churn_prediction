
import warnings
warnings.filterwarnings("ignore")

import os, pickle, joblib, json
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from datetime import datetime

# Sklearn – preprocessing & pipeline
from sklearn.model_selection  import (train_test_split, StratifiedKFold,
                                       cross_val_score)
from sklearn.preprocessing    import (StandardScaler, MinMaxScaler,
                                       LabelEncoder, OrdinalEncoder,
                                       PowerTransformer)
from sklearn.impute            import SimpleImputer, KNNImputer
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer

# Sklearn – models
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import (RandomForestClassifier,
                                       GradientBoostingClassifier,
                                       StackingClassifier)
from sklearn.svm              import SVC

# Sklearn – metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)

# Advanced libraries
try:
    from xgboost  import XGBClassifier
    from lightgbm import LGBMClassifier
    BOOSTING_AVAILABLE = True
except ImportError:
    BOOSTING_AVAILABLE = False
    print("[WARN] XGBoost / LightGBM not installed — falling back to sklearn boosting.")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline      import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARN] imbalanced-learn not installed — skipping SMOTE.")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARN] Optuna not installed — skipping Bayesian hyper-param search.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] SHAP not installed — skipping explainability plots.")


SEED = 42
np.random.seed(SEED)

print("=" * 70)
print("  CUSTOMER CHURN PREDICTION — ADVANCED ML PIPELINE")
print(f"  Started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)


print("\n[STEP 2] Generating synthetic subscription dataset …")

N = 10_000 

rng = np.random.default_rng(SEED)

tenure_months          = rng.integers(1,  72,  N)
monthly_charges        = rng.uniform(20, 120, N)
total_charges          = monthly_charges * tenure_months + rng.normal(0, 50, N)
total_charges          = np.clip(total_charges, 0, None)

num_products           = rng.integers(1, 6, N)
support_tickets_6mo    = rng.integers(0, 15, N)
avg_response_time_hrs  = rng.exponential(4, N)
login_frequency_weekly = rng.integers(0, 20, N)
feature_usage_score    = rng.uniform(0, 100, N)   
payment_delay_days     = rng.integers(0, 30, N)
nps_score              = rng.integers(0, 11, N)   
contract_type          = rng.choice(["Month-to-Month", "One-Year", "Two-Year"],
                                     N, p=[0.55, 0.25, 0.20])
payment_method         = rng.choice(["Credit Card", "Bank Transfer",
                                      "PayPal", "Check"], N)
internet_service       = rng.choice(["Fiber Optic", "DSL", "None"], N, p=[0.45, 0.40, 0.15])
gender                 = rng.choice(["Male", "Female"], N)
senior_citizen         = rng.integers(0, 2, N)
partner                = rng.choice(["Yes", "No"], N)
dependents             = rng.choice(["Yes", "No"], N)
plan_tier              = rng.choice(["Basic", "Standard", "Premium"], N, p=[0.40, 0.35, 0.25])
referral_source        = rng.choice(["Organic", "Paid", "Referral", "Social"], N)

log_odds = (
    -3.0
    - 0.04  * tenure_months          
    + 0.015 * monthly_charges        
    + 0.12  * support_tickets_6mo   
    + 0.08  * payment_delay_days     
    - 0.025 * nps_score              
    - 0.018 * feature_usage_score    
    - 0.10  * login_frequency_weekly 
    + 0.50  * (contract_type == "Month-to-Month")
    - 0.40  * (contract_type == "Two-Year")
    + 0.30  * (internet_service == "Fiber Optic")
    - 0.20  * (plan_tier == "Premium")
    + 0.25  * senior_citizen
    + rng.normal(0, 0.5, N)          
)
churn_prob = 1 / (1 + np.exp(-log_odds))
churn      = (rng.uniform(0, 1, N) < churn_prob).astype(int)


df = pd.DataFrame({
    "CustomerID"            : [f"CUST-{i:05d}" for i in range(N)],
    "gender"                : gender,
    "SeniorCitizen"         : senior_citizen,
    "Partner"               : partner,
    "Dependents"            : dependents,
    "tenure_months"         : tenure_months,
    "num_products"          : num_products,
    "monthly_charges"       : monthly_charges.round(2),
    "total_charges"         : total_charges.round(2),
    "support_tickets_6mo"   : support_tickets_6mo,
    "avg_response_time_hrs" : avg_response_time_hrs.round(2),
    "login_frequency_weekly": login_frequency_weekly,
    "feature_usage_score"   : feature_usage_score.round(2),
    "payment_delay_days"    : payment_delay_days,
    "nps_score"             : nps_score,
    "contract_type"         : contract_type,
    "payment_method"        : payment_method,
    "internet_service"      : internet_service,
    "plan_tier"             : plan_tier,
    "referral_source"       : referral_source,
    "Churn"                 : churn,
})


for col in ["avg_response_time_hrs", "nps_score", "feature_usage_score"]:
    mask = rng.random(N) < 0.03
    df.loc[mask, col] = np.nan

print(f"   Dataset shape : {df.shape}")
print(f"   Churn rate    : {df['Churn'].mean():.2%}")
df.to_csv("churn_data.csv", index=False)
print("   Saved → churn_data.csv")


print("\n[STEP 3] EDA …")

print("\n── Summary Statistics ──")
print(df.describe(include="all").T.to_string())
print(f"\n── Missing Values ──\n{df.isnull().sum()[df.isnull().sum() > 0]}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("EDA — Key Feature Distributions (coloured by Churn)", fontsize=16, fontweight="bold")

numeric_eda = ["tenure_months", "monthly_charges", "support_tickets_6mo",
               "nps_score", "feature_usage_score", "login_frequency_weekly"]
for ax, col in zip(axes.flatten(), numeric_eda):
    for label, grp in df.groupby("Churn"):
        ax.hist(grp[col].dropna(), bins=30, alpha=0.55,
                label=f"Churn={label}", density=True)
    ax.set_title(col)
    ax.legend(fontsize=8)
    ax.set_xlabel(col)
    ax.set_ylabel("Density")

plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=120)
plt.close()
print("   Saved → eda_distributions.png")


num_cols = df.select_dtypes(include=np.number).columns.tolist()
corr     = df[num_cols].corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, annot_kws={"size": 7})
plt.title("Pearson Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=120)
plt.close()
print("   Saved → eda_correlation.png")


print("\n[STEP 4] Advanced Feature Engineering …")

df_fe = df.copy()


df_fe["charge_per_product"]     = df_fe["monthly_charges"] / df_fe["num_products"].replace(0, 1)
df_fe["avg_monthly_total"]      = df_fe["total_charges"] / df_fe["tenure_months"].replace(0, 1)
df_fe["billing_stress_index"]   = df_fe["payment_delay_days"] * df_fe["monthly_charges"] / 100


df_fe["engagement_score"]       = (
    df_fe["feature_usage_score"].fillna(df_fe["feature_usage_score"].median()) * 0.4
    + df_fe["login_frequency_weekly"] * 2.0
)
df_fe["satisfaction_risk"]      = (
    df_fe["support_tickets_6mo"] * 3
    - df_fe["nps_score"].fillna(df_fe["nps_score"].median())
)


bins   = [0, 6, 12, 24, 48, 72]
labels = ["0-6mo", "6-12mo", "12-24mo", "24-48mo", "48+mo"]
df_fe["tenure_bucket"] = pd.cut(df_fe["tenure_months"], bins=bins, labels=labels)


df_fe["ticket_per_month"]       = df_fe["support_tickets_6mo"] / 6
df_fe["nps_x_usage"]            = (
    df_fe["nps_score"].fillna(0) * df_fe["feature_usage_score"].fillna(0)
)
df_fe["high_value_at_risk"]     = (
    (df_fe["monthly_charges"] > df_fe["monthly_charges"].quantile(0.75)) &
    (df_fe["nps_score"].fillna(5) < 5)
).astype(int)

for col in ["total_charges", "avg_response_time_hrs"]:
    df_fe[f"log_{col}"] = np.log1p(df_fe[col].fillna(0))


df_fe["is_month_to_month"]      = (df_fe["contract_type"] == "Month-to-Month").astype(int)
df_fe["is_senior"]              = df_fe["SeniorCitizen"]
df_fe["uses_fiber"]             = (df_fe["internet_service"] == "Fiber Optic").astype(int)
df_fe["no_internet"]            = (df_fe["internet_service"] == "None").astype(int)

print(f"   Features after engineering : {df_fe.shape[1]}")

print("\n[STEP 5] Building Preprocessing Pipeline …")

TARGET = "Churn"
DROP   = ["CustomerID", "Churn"]

feature_df = df_fe.drop(columns=DROP)

numerical_cols   = feature_df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"   Numerical features   : {len(numerical_cols)}")
print(f"   Categorical features : {len(categorical_cols)}")

numerical_pipeline = Pipeline([
    ("knn_imputer", KNNImputer(n_neighbors=5)),
    ("power_transform", PowerTransformer(method="yeo-johnson", standardize=True)),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline,   numerical_cols),
    ("cat", categorical_pipeline, categorical_cols),
], remainder="drop")

feature_meta = {
    "numerical_cols"  : numerical_cols,
    "categorical_cols": categorical_cols,
    "target"          : TARGET,
}

X = feature_df
y = df_fe[TARGET]

print("\n[STEP 6] Stratified 70 / 15 / 15 split …")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=SEED)

print(f"   Train : {X_train.shape[0]} rows  |  churn rate {y_train.mean():.2%}")
print(f"   Val   : {X_val.shape[0]}  rows  |  churn rate {y_val.mean():.2%}")
print(f"   Test  : {X_test.shape[0]}  rows  |  churn rate {y_test.mean():.2%}")


print("\n[STEP 7] Handling class imbalance …")

X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc   = preprocessor.transform(X_val)
X_test_proc  = preprocessor.transform(X_test)

if SMOTE_AVAILABLE:
    sm = SMOTE(random_state=SEED, k_neighbors=5)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_proc, y_train)
    print(f"   After SMOTE — Train : {X_train_bal.shape[0]} rows")
else:
    X_train_bal, y_train_bal = X_train_proc, y_train
    print("   SMOTE unavailable — using original imbalanced data.")

print("\n[STEP 8] Training candidate models …")

if BOOSTING_AVAILABLE:
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
        eval_metric="logloss", random_state=SEED, n_jobs=-1
    )
    lgbm = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED,
        n_jobs=-1, verbose=-1
    )
else:
    xgb  = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                       max_depth=5, random_state=SEED)
    lgbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                       max_depth=4, random_state=SEED)

candidate_models = {
    "Logistic Regression": LogisticRegression(
        C=0.5, max_iter=1000, class_weight="balanced", random_state=SEED),
    "Random Forest"       : RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced",
        min_samples_leaf=5, random_state=SEED, n_jobs=-1),
    "XGBoost"             : xgb,
    "LightGBM"            : lgbm,
    "Gradient Boosting"   : GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5, random_state=SEED),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = {}

for name, model in candidate_models.items():
    scores = cross_val_score(
        model, X_train_bal, y_train_bal,
        cv=cv, scoring="roc_auc", n_jobs=-1
    )
    cv_results[name] = scores
    print(f"   {name:25s}  CV AUC = {scores.mean():.4f} ± {scores.std():.4f}")


print("\n[STEP 9] Bayesian hyper-parameter search with Optuna …")

if OPTUNA_AVAILABLE:
    def rf_objective(trial):
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators",  100, 600, step=50),
            "max_depth"       : trial.suggest_int("max_depth",       4,  20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1,  20),
            "max_features"    : trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "class_weight"    : "balanced",
            "random_state"    : SEED,
            "n_jobs"          : -1,
        }
        model  = RandomForestClassifier(**params)
        scores = cross_val_score(model, X_train_bal, y_train_bal,
                                 cv=3, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(rf_objective, n_trials=30, show_progress_bar=False)

    best_rf_params = study.best_params
    best_rf_params.update({"class_weight": "balanced",
                            "random_state": SEED, "n_jobs": -1})
    print(f"   Best RF params : {best_rf_params}")
    print(f"   Best CV AUC    : {study.best_value:.4f}")

    tuned_rf = RandomForestClassifier(**best_rf_params)
else:
    print("   Optuna unavailable — using default Random Forest params.")
    tuned_rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED, n_jobs=-1
    )

print("\n[STEP 10] Building Stacking Ensemble …")

base_estimators = [
    ("rf"   , tuned_rf),
    ("xgb"  , candidate_models["XGBoost"]),
    ("lgbm" , candidate_models["LightGBM"]),
    ("gb"   , candidate_models["Gradient Boosting"]),
]

meta_learner = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)

stacking_model = StackingClassifier(
    estimators        = base_estimators,
    final_estimator   = meta_learner,
    cv                = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    stack_method      = "predict_proba",
    passthrough       = True,    
    n_jobs            = -1,
)

print("   Fitting stacking ensemble (this may take ~30–90 seconds) …")
stacking_model.fit(X_train_bal, y_train_bal)
print("   ✓ Stacking ensemble trained.")

print("\n[STEP 11] Evaluating on held-out TEST set …")

def evaluate_model(model, X_proc, y_true, model_name="Model", threshold=0.5):
    """Return a dict of evaluation metrics and print a summary."""
    y_prob = model.predict_proba(X_proc)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "Accuracy"          : accuracy_score(y_true, y_pred),
        "Precision"         : precision_score(y_true, y_pred, zero_division=0),
        "Recall"            : recall_score(y_true, y_pred, zero_division=0),
        "F1"                : f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC"           : roc_auc_score(y_true, y_prob),
        "PR-AUC"            : average_precision_score(y_true, y_prob),
    }
    print(f"\n── {model_name} ──")
    for k, v in metrics.items():
        print(f"   {k:<15s}: {v:.4f}")
    print("\n" + classification_report(y_true, y_pred,
                                       target_names=["No Churn", "Churn"]))
    return metrics, y_prob, y_pred

all_results   = {}
all_probas    = {}
models_to_eval = {**candidate_models, "★ Stacking Ensemble": stacking_model}

for name, model in models_to_eval.items():
   
    if name != "★ Stacking Ensemble":
        model.fit(X_train_bal, y_train_bal)
    metrics, y_prob, _ = evaluate_model(model, X_test_proc, y_test, name)
    all_results[name]  = metrics
    all_probas[name]   = y_prob


results_df = pd.DataFrame(all_results).T.sort_values("ROC-AUC", ascending=False)
print("\n═" * 35)
print("MODEL COMPARISON (sorted by ROC-AUC)")
print("═" * 35)
print(results_df.to_string(float_format="{:.4f}".format))
results_df.to_csv("model_comparison.csv")
print("\n   Saved → model_comparison.csv")


fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Model Evaluation — ROC & Precision-Recall Curves",
             fontsize=14, fontweight="bold")

for name, y_prob in all_probas.items():
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0], name=name)
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1], name=name)

axes[0].set_title("ROC Curves")
axes[1].set_title("Precision-Recall Curves")
for ax in axes:
    ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig("model_roc_pr.png", dpi=120)
plt.close()
print("   Saved → model_roc_pr.png")


best_y_pred = (all_probas["★ Stacking Ensemble"] >= 0.5).astype(int)
cm = confusion_matrix(y_test, best_y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"])
ax.set_title("Confusion Matrix — Stacking Ensemble", fontweight="bold")
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.close()
print("   Saved → confusion_matrix.png")

print("\n[STEP 12] SHAP Explainability …")

if SHAP_AVAILABLE:
    
    rf_model  = candidate_models["Random Forest"]
    explainer = shap.TreeExplainer(rf_model)

    sample_idx = np.random.choice(len(X_test_proc), size=min(500, len(X_test_proc)), replace=False)
    X_sample   = X_test_proc[sample_idx]

   
    feature_names = (
        numerical_cols
        + categorical_cols
    )
   
    n_proc_features = X_sample.shape[1]
    if len(feature_names) < n_proc_features:
        feature_names += [f"feat_{i}" for i in range(n_proc_features - len(feature_names))]
    feature_names = feature_names[:n_proc_features]

    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=feature_names,
                      show=False, max_display=20)
    plt.title("SHAP Feature Importance — Global (Churn=1)", fontweight="bold")
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("   Saved → shap_summary.png")

   
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=feature_names,
                      plot_type="bar", show=False, max_display=15)
    plt.title("SHAP Mean |Feature Impact|", fontweight="bold")
    plt.tight_layout()
    plt.savefig("shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("   Saved → shap_bar.png")
else:
    print("   SHAP not available — skipping.")


print("\n[STEP 13] Saving model artefacts …")

os.makedirs("model_artefacts", exist_ok=True)


joblib.dump(preprocessor,    "model_artefacts/preprocessor.joblib")
joblib.dump(stacking_model,  "model_artefacts/stacking_model.joblib")
joblib.dump(candidate_models["Random Forest"], "model_artefacts/rf_model.joblib")


with open("model_artefacts/preprocessor.pkl", "wb")   as f:
    pickle.dump(preprocessor, f, protocol=pickle.HIGHEST_PROTOCOL)
with open("model_artefacts/stacking_model.pkl", "wb") as f:
    pickle.dump(stacking_model, f, protocol=pickle.HIGHEST_PROTOCOL)
with open("model_artefacts/rf_model.pkl", "wb")       as f:
    pickle.dump(candidate_models["Random Forest"], f, protocol=pickle.HIGHEST_PROTOCOL)


with open("model_artefacts/feature_meta.pkl", "wb") as f:
    pickle.dump(feature_meta, f)

model_card = {
    "model_name"         : "Customer Churn Stacking Ensemble",
    "trained_at"         : datetime.now().isoformat(),
    "train_samples"      : int(X_train_bal.shape[0]),
    "test_roc_auc"       : round(all_results["★ Stacking Ensemble"]["ROC-AUC"], 4),
    "test_f1"            : round(all_results["★ Stacking Ensemble"]["F1"], 4),
    "numerical_features" : numerical_cols,
    "categorical_features": categorical_cols,
    "base_models"        : ["Random Forest", "XGBoost", "LightGBM", "Gradient Boosting"],
    "meta_learner"       : "Logistic Regression",
}
with open("model_artefacts/model_card.pkl", "wb") as f:
    pickle.dump(model_card, f)

with open("model_artefacts/model_card.txt", "w") as f:
    for k, v in model_card.items():
        f.write(f"{k}: {v}\n")

print("   Saved to model_artefacts/")
for fname in os.listdir("model_artefacts"):
    size = os.path.getsize(f"model_artefacts/{fname}") / 1024
    print(f"     {fname:<35s}  {size:8.1f} KB")


def predict_churn(customer_dict: dict, threshold: float = 0.5) -> dict:
    """
    Predict churn probability for a single customer.

    Parameters
    ----------
    customer_dict : dict
        Raw feature values (same columns as training, excluding CustomerID & Churn).
    threshold : float
        Decision threshold for binary classification (default 0.5).

    Returns
    -------
    dict with keys:
        churn_probability  : float (0–1)
        churn_prediction   : int   (0 = Stay, 1 = Churn)
        risk_tier          : str   ("Low" / "Medium" / "High")
        confidence         : float
    """
    
    _preprocessor = joblib.load("model_artefacts/preprocessor.joblib")
    _model        = joblib.load("model_artefacts/stacking_model.joblib")
    _meta         = pickle.load(open("model_artefacts/feature_meta.pkl", "rb"))

   
    all_cols = _meta["numerical_cols"] + _meta["categorical_cols"]
    row_df   = pd.DataFrame([customer_dict]).reindex(columns=all_cols)

    X_proc   = _preprocessor.transform(row_df)
    prob     = float(_model.predict_proba(X_proc)[0, 1])
    pred     = int(prob >= threshold)

    if prob < 0.35:
        risk = "Low"
    elif prob < 0.65:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction" : pred,
        "risk_tier"        : risk,
        "confidence"       : round(max(prob, 1 - prob), 4),
    }

print("\n[STEP 14] Demo Inference …")

sample_customer = {
    "gender"                : "Male",
    "SeniorCitizen"         : 0,
    "Partner"               : "No",
    "Dependents"            : "No",
    "tenure_months"         : 3,
    "num_products"          : 1,
    "monthly_charges"       : 95.0,
    "total_charges"         : 285.0,
    "support_tickets_6mo"   : 8,
    "avg_response_time_hrs" : 12.0,
    "login_frequency_weekly": 1,
    "feature_usage_score"   : 15.0,
    "payment_delay_days"    : 20,
    "nps_score"             : 2,
    "contract_type"         : "Month-to-Month",
    "payment_method"        : "Check",
    "internet_service"      : "Fiber Optic",
    "plan_tier"             : "Basic",
    "referral_source"       : "Paid",
    "tenure_bucket"         : "0-6mo",
    "charge_per_product"    : 95.0,
    "avg_monthly_total"     : 95.0,
    "billing_stress_index"  : 19.0,
    "engagement_score"      : 8.0,
    "satisfaction_risk"     : 22.0,
    "ticket_per_month"      : 1.33,
    "nps_x_usage"           : 30.0,
    "high_value_at_risk"    : 1,
    "log_total_charges"     : np.log1p(285.0),
    "log_avg_response_time_hrs": np.log1p(12.0),
    "is_month_to_month"     : 1,
    "is_senior"             : 0,
    "uses_fiber"            : 1,
    "no_internet"           : 0,
}

result = predict_churn(sample_customer)
print(f"   Churn Probability : {result['churn_probability']:.2%}")
print(f"   Risk Tier         : {result['risk_tier']}")
print(f"   Prediction        : {'⚠ CHURN' if result['churn_prediction'] else '✓ STAY'}")

print("\n" + "=" * 70)
print("  PIPELINE COMPLETE — All artefacts saved to model_artefacts/")
print(f"  Finished at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)