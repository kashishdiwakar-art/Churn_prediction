import os, pickle, warnings, io
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

from ui_styling import (
    inject_css, PALETTE, RISK_COLORS,
    render_hero, render_metric_card, render_risk_badge,
    render_section_title, render_info_box, render_result_card,
    render_sidebar_header, render_sidebar_stats,
    gauge_chart, feature_importance_bar, churn_distribution_donut,
    model_comparison_bar, risk_breakdown_bar, correlation_heatmap,
    probability_histogram, shap_waterfall_mock,
)

st.set_page_config(
    page_title     = "ChurnLens AI",
    page_icon      = "🔮",
    layout         = "wide",
    initial_sidebar_state = "expanded",
)

inject_css()

@st.cache_resource(show_spinner="Loading model artefacts …")
def load_artefacts():
    """Load all saved model artefacts. Returns dict or raises FileNotFoundError."""
    art_dir = "model_artefacts"
    if not os.path.exists(art_dir):
        return None

    preprocessor  = joblib.load(f"{art_dir}/preprocessor.joblib")
    model         = joblib.load(f"{art_dir}/stacking_model.joblib")
    rf_model      = joblib.load(f"{art_dir}/rf_model.joblib")
    with open(f"{art_dir}/feature_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    with open(f"{art_dir}/model_card.pkl", "rb") as f:
        card = pickle.load(f)

    return {
        "preprocessor": preprocessor,
        "model"       : model,
        "rf_model"    : rf_model,
        "meta"        : meta,
        "card"        : card,
    }


@st.cache_data(show_spinner="Loading dataset …")
def load_dataset():
    """Load the training dataset if available."""
    if os.path.exists("churn_data.csv"):
        return pd.read_csv("churn_data.csv")
    return None


@st.cache_data(show_spinner="Loading model comparison …")
def load_model_comparison():
    if os.path.exists("model_comparison.csv"):
        df = pd.read_csv("model_comparison.csv", index_col=0)
        return df
    return None

def predict_customer(artefacts: dict, customer_dict: dict,
                     threshold: float = 0.5) -> dict:
    """Score a single customer dict through the full pipeline."""
    meta  = artefacts["meta"]
    all_cols = meta["numerical_cols"] + meta["categorical_cols"]

    row_df = pd.DataFrame([customer_dict]).reindex(columns=all_cols)
    X_proc = artefacts["preprocessor"].transform(row_df)
    prob   = float(artefacts["model"].predict_proba(X_proc)[0, 1])
    pred   = int(prob >= threshold)

    risk   = "Low" if prob < 0.35 else ("Medium" if prob < 0.65 else "High")

    if hasattr(artefacts["rf_model"], "feature_importances_"):
        n = len(artefacts["rf_model"].feature_importances_)
        direction = np.sign(X_proc[0, :n]) * artefacts["rf_model"].feature_importances_
        shap_like  = (direction / (np.abs(direction).sum() + 1e-9)) * prob
    else:
        shap_like  = np.zeros(len(all_cols))

    return {
        "probability" : prob,
        "prediction"  : pred,
        "risk"        : risk,
        "confidence"  : round(max(prob, 1 - prob), 4),
        "shap_approx" : shap_like.tolist(),
    }

artefacts = load_artefacts()
df_data   = load_dataset()
df_comp   = load_model_comparison()

MODELS_READY = artefacts is not None

render_sidebar_header()

if MODELS_READY:
    card = artefacts["card"]
    render_sidebar_stats(
        roc_auc    = card.get("test_roc_auc", 0),
        f1         = card.get("test_f1",      0),
        train_size = card.get("train_samples",0),
    )
else:
    st.sidebar.error("⚠ Model artefacts not found.\nRun `churn_model.py` first.")

st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "🎯 Decision Threshold",
    min_value=0.10, max_value=0.90, value=0.50, step=0.01,
    help="Lower threshold → more sensitive (catches more churners, higher false positives)."
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="font-size:0.72rem; color:{PALETTE['text_muted']}; line-height:1.6;">
<b>Built with</b><br>
🧠 Stacking Ensemble<br>
⚡ Optuna Bayesian Tuning<br>
🔁 SMOTE Oversampling<br>
🌿 SHAP Explainability<br>
🚀 Streamlit
</div>
""", unsafe_allow_html=True)

render_hero(
    "🔮 ChurnLens AI",
    "Advanced Customer Churn Prediction · Stacking Ensemble · Explainable AI"
)

if MODELS_READY and df_data is not None:
    churn_rate = df_data["Churn"].mean()
    n_customers = len(df_data)
    n_churned   = df_data["Churn"].sum()
    avg_charge  = df_data["monthly_charges"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Customers", f"{n_customers:,}",
                            color=PALETTE["primary"])
    with c2:
        render_metric_card("Churn Rate", f"{churn_rate:.1%}",
                            delta="Dataset avg", color=PALETTE["danger"])
    with c3:
        render_metric_card("Churned Customers", f"{n_churned:,}",
                            color=PALETTE["secondary"])
    with c4:
        render_metric_card("Avg Monthly Charge", f"${avg_charge:.0f}",
                            color=PALETTE["accent"])

    st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predict",
    "📊 Dashboard",
    "🤖 Model Intel",
    "📦 Batch Predict",
    "📚 About",
])

with tab1:
    render_section_title("Enter Customer Profile")

    if not MODELS_READY:
        st.error("Model not loaded. Please run `churn_model.py` first.")
    else:
        col_l, col_r = st.columns([1.4, 1])

        with col_l:
            st.markdown("**Demographics**")
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                gender     = st.selectbox("Gender", ["Male","Female"])
                senior     = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            with dc2:
                partner    = st.selectbox("Partner",    ["Yes","No"])
                dependents = st.selectbox("Dependents", ["Yes","No"])
            with dc3:
                plan_tier  = st.selectbox("Plan Tier",  ["Basic","Standard","Premium"])
                internet   = st.selectbox("Internet",   ["DSL","Fiber Optic","None"])

            st.markdown("**Subscription Details**")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                tenure    = st.slider("Tenure (months)", 1, 72, 12)
                n_products= st.slider("Num Products",    1,  6,  2)
            with sc2:
                monthly   = st.slider("Monthly Charge ($)", 20.0, 120.0, 65.0, 0.5)
                contract  = st.selectbox("Contract Type",
                                         ["Month-to-Month","One-Year","Two-Year"])
            with sc3:
                payment   = st.selectbox("Payment Method",
                                          ["Credit Card","Bank Transfer","PayPal","Check"])
                referral  = st.selectbox("Referral Source",
                                          ["Organic","Paid","Referral","Social"])

            st.markdown("**Engagement & Satisfaction**")
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                tickets   = st.slider("Support Tickets (6mo)", 0, 14, 2)
                nps       = st.slider("NPS Score",             0, 10, 7)
            with ec2:
                login_freq= st.slider("Login Freq (weekly)",   0, 20, 8)
                usage     = st.slider("Feature Usage Score",   0.0, 100.0, 60.0)
            with ec3:
                pay_delay = st.slider("Payment Delay (days)",  0, 30, 3)
                resp_time = st.slider("Avg Response Time (hrs)",0.0, 48.0, 4.0)

        total_chg   = monthly * tenure
        tenure_map  = {(0,6):"0-6mo",(6,12):"6-12mo",(12,24):"12-24mo",
                       (24,48):"24-48mo",(48,100):"48+mo"}
        t_bucket    = next(v for (lo,hi),v in tenure_map.items() if lo<=tenure<hi)

        customer = {
            "gender"                : gender,
            "SeniorCitizen"         : senior,
            "Partner"               : partner,
            "Dependents"            : dependents,
            "tenure_months"         : tenure,
            "num_products"          : n_products,
            "monthly_charges"       : monthly,
            "total_charges"         : total_chg,
            "support_tickets_6mo"   : tickets,
            "avg_response_time_hrs" : resp_time,
            "login_frequency_weekly": login_freq,
            "feature_usage_score"   : usage,
            "payment_delay_days"    : pay_delay,
            "nps_score"             : nps,
            "contract_type"         : contract,
            "payment_method"        : payment,
            "internet_service"      : internet,
            "plan_tier"             : plan_tier,
            "referral_source"       : referral,
            "tenure_bucket"         : t_bucket,
            "charge_per_product"    : monthly / max(n_products, 1),
            "avg_monthly_total"     : total_chg / max(tenure, 1),
            "billing_stress_index"  : pay_delay * monthly / 100,
            "engagement_score"      : usage * 0.4 + login_freq * 2,
            "satisfaction_risk"     : tickets * 3 - nps,
            "ticket_per_month"      : tickets / 6,
            "nps_x_usage"           : nps * usage,
            "high_value_at_risk"    : int(monthly > 90 and nps < 5),
            "log_total_charges"     : np.log1p(total_chg),
            "log_avg_response_time_hrs": np.log1p(resp_time),
            "is_month_to_month"     : int(contract == "Month-to-Month"),
            "is_senior"             : senior,
            "uses_fiber"            : int(internet == "Fiber Optic"),
            "no_internet"           : int(internet == "None"),
        }

        with col_r:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            if st.button("⚡ Run Churn Prediction", key="predict_btn"):
                with st.spinner("Scoring customer …"):
                    result = predict_customer(artefacts, customer, threshold)

                st.plotly_chart(gauge_chart(result["probability"], result["risk"]),
                                use_container_width=True, config={"displayModeBar": False})

                
                render_result_card(result["probability"], result["risk"], result["prediction"])

                st.markdown("&nbsp;", unsafe_allow_html=True)
                render_section_title("Feature Contributions")
                meta       = artefacts["meta"]
                feat_names = meta["numerical_cols"] + meta["categorical_cols"]
                shap_approx= result["shap_approx"]
                n          = min(len(feat_names), len(shap_approx))

                st.plotly_chart(
                    shap_waterfall_mock(feat_names[:n], shap_approx[:n],
                                        base_value=0.5),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

                render_section_title("💡 Recommended Retention Actions")
                recs = []
                if result["probability"] > 0.35:
                    if contract == "Month-to-Month":
                        recs.append("📄 Offer annual/bi-annual contract discount (saves ~25 %)")
                    if tickets > 5:
                        recs.append("🎧 Priority support upgrade — assign dedicated CSM")
                    if nps < 6:
                        recs.append("📞 Schedule proactive check-in call within 48 hours")
                    if usage < 30:
                        recs.append("🎓 Send personalised onboarding / feature-discovery email")
                    if pay_delay > 10:
                        recs.append("💳 Offer flexible billing cycle or payment plan")
                    if not recs:
                        recs.append("🎁 Loyalty reward or personalised upsell offer")

                    for r in recs:
                        st.markdown(f"- {r}")
                else:
                    st.success("✅ Customer appears healthy — standard engagement is sufficient.")
            else:
                render_info_box("Fill in the customer profile on the left and click "
                                "<b>Run Churn Prediction</b> to get an instant risk score.")

with tab2:
    if df_data is None:
        st.warning("Dataset not found. Run `churn_model.py` to generate churn_data.csv.")
    else:
        render_section_title("Churn Overview")
        d1, d2 = st.columns([1, 1.5])
        with d1:
            churned = int(df_data["Churn"].sum())
            stayed  = int((df_data["Churn"] == 0).sum())
            st.plotly_chart(churn_distribution_donut(churned, stayed),
                            use_container_width=True,
                            config={"displayModeBar": False})
        with d2:
            render_section_title("Churn Rate by Contract Type")
            by_contract = (df_data.groupby("contract_type")["Churn"]
                           .mean().reset_index()
                           .rename(columns={"Churn":"Churn Rate"}))
            fig_c = go.Figure(go.Bar(
                x=by_contract["contract_type"],
                y=by_contract["Churn Rate"],
                text=[f"{v:.1%}" for v in by_contract["Churn Rate"]],
                textposition="outside",
                marker_color=[PALETTE["danger"], PALETTE["warning"], PALETTE["success"]],
            ))
            fig_c.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color=PALETTE["text_primary"]),
                margin=dict(t=20,b=20,l=20,r=20), height=280,
                xaxis=dict(gridcolor=PALETTE["card_border"]),
                yaxis=dict(gridcolor=PALETTE["card_border"], tickformat=".0%"),
            )
            st.plotly_chart(fig_c, use_container_width=True,
                            config={"displayModeBar": False})

        render_section_title("Feature Distribution by Churn")
        feat_sel = st.selectbox("Select Feature",
                                ["monthly_charges","tenure_months",
                                 "support_tickets_6mo","nps_score",
                                 "feature_usage_score","login_frequency_weekly"])
        fig_dist = go.Figure()
        for label, color in [(0, PALETTE["success"]), (1, PALETTE["danger"])]:
            sub = df_data[df_data["Churn"] == label][feat_sel].dropna()
            fig_dist.add_trace(go.Histogram(
                x=sub, name="Stayed" if label==0 else "Churned",
                opacity=0.65, nbinsx=40, marker_color=color,
            ))
        fig_dist.update_layout(
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=PALETTE["text_primary"]),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20,b=20,l=20,r=20), height=300,
            xaxis=dict(gridcolor=PALETTE["card_border"]),
            yaxis=dict(gridcolor=PALETTE["card_border"]),
        )
        st.plotly_chart(fig_dist, use_container_width=True,
                        config={"displayModeBar": False})

        render_section_title("Correlation Matrix")
        num_cols = df_data.select_dtypes(include=np.number).columns.tolist()[:12]
        st.plotly_chart(correlation_heatmap(df_data, num_cols),
                        use_container_width=True,
                        config={"displayModeBar": False})


with tab3:
    render_section_title("Model Comparison")
    if df_comp is not None:
        st.plotly_chart(model_comparison_bar(df_comp),
                        use_container_width=True,
                        config={"displayModeBar": False})

        st.dataframe(
            df_comp.style
                   .background_gradient(cmap="RdYlGn", subset=["ROC-AUC","F1"])
                   .format("{:.4f}"),
            use_container_width=True
        )
    else:
        render_info_box("Run <b>churn_model.py</b> to generate model comparison data.")

    render_section_title("Feature Importance (Random Forest)")
    if MODELS_READY:
        rf   = artefacts["rf_model"]
        meta = artefacts["meta"]
        feat_names_all = meta["numerical_cols"] + meta["categorical_cols"]
        importances    = rf.feature_importances_
        n              = min(len(feat_names_all), len(importances))

        st.plotly_chart(
            feature_importance_bar(feat_names_all[:n], importances[:n].tolist(), top_n=20),
            use_container_width=True,
            config={"displayModeBar": False}
        )
    else:
        render_info_box("Model not loaded.")

    render_section_title("Model Architecture")
    if MODELS_READY:
        card = artefacts["card"]
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**Base Models (Level 0)**")
            for bm in card.get("base_models", []):
                st.markdown(f"- {bm}")
        with mc2:
            st.markdown("**Meta Learner (Level 1)**")
            st.markdown(f"- {card.get('meta_learner','Logistic Regression')}")
            st.markdown(f"**Trained:** {card.get('trained_at','N/A')[:10]}")
            st.markdown(f"**Test ROC-AUC:** {card.get('test_roc_auc','N/A')}")
            st.markdown(f"**Test F1:** {card.get('test_f1','N/A')}")

    render_section_title("📸 Saved Plots")
    plot_files = {
        "ROC / PR Curves": "model_roc_pr.png",
        "Confusion Matrix": "confusion_matrix.png",
        "SHAP Summary"    : "shap_summary.png",
        "EDA Distributions": "eda_distributions.png",
    }
    cols = st.columns(2)
    for i, (label, fname) in enumerate(plot_files.items()):
        if os.path.exists(fname):
            with cols[i % 2]:
                st.image(fname, caption=label, use_column_width=True)

with tab4:
    render_section_title("Batch Customer Scoring")
    render_info_box("Upload a CSV with the same columns as the training data. "
                    "The app will score every row and add a <b>Churn_Probability</b> "
                    "and <b>Risk_Tier</b> column.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded and MODELS_READY:
        batch_df   = pd.read_csv(uploaded)
        meta       = artefacts["meta"]
        all_cols   = meta["numerical_cols"] + meta["categorical_cols"]

        st.markdown(f"**Rows uploaded:** {len(batch_df):,}")
        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("⚡ Score All Customers"):
            with st.spinner(f"Scoring {len(batch_df):,} customers …"):
                batch_proc  = batch_df.reindex(columns=all_cols)
                X_proc      = artefacts["preprocessor"].transform(batch_proc)
                probs       = artefacts["model"].predict_proba(X_proc)[:, 1]
                preds       = (probs >= threshold).astype(int)
                risk_tiers  = ["Low" if p < 0.35 else ("Medium" if p < 0.65 else "High")
                               for p in probs]

                result_df   = batch_df.copy()
                result_df["Churn_Probability"] = probs.round(4)
                result_df["Churn_Prediction"]  = preds
                result_df["Risk_Tier"]         = risk_tiers

            st.success(f"✅ Scored {len(result_df):,} customers successfully!")

            risk_counts = result_df["Risk_Tier"].value_counts().to_dict()
            r1, r2, r3  = st.columns(3)
            with r1: render_metric_card("🟢 Low Risk",    str(risk_counts.get("Low",0)),    color=PALETTE["success"])
            with r2: render_metric_card("🟡 Medium Risk", str(risk_counts.get("Medium",0)), color=PALETTE["warning"])
            with r3: render_metric_card("🔴 High Risk",   str(risk_counts.get("High",0)),   color=PALETTE["danger"])

            st.markdown("<br>", unsafe_allow_html=True)

            st.plotly_chart(risk_breakdown_bar(risk_counts),
                            use_container_width=True,
                            config={"displayModeBar": False})


            st.plotly_chart(
                probability_histogram(probs, preds),
                use_container_width=True,
                config={"displayModeBar": False}
            )

            st.dataframe(result_df.head(20), use_container_width=True)
            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label     = "⬇ Download Full Results CSV",
                data      = csv_bytes,
                file_name = "churn_predictions.csv",
                mime      = "text/csv",
            )
    elif not MODELS_READY:
        st.error("Model not loaded. Run `churn_model.py` first.")

with tab5:
    st.markdown(f"""
    ## 🔮 ChurnLens AI — Project Documentation

    ### Problem Statement
    Subscription businesses lose significant revenue when customers churn unexpectedly.
    This project builds a **production-grade churn prediction system** that identifies
    at-risk customers early, explains *why* they are at risk, and recommends targeted
    retention actions.

    ---

    ### Pipeline Architecture

    | Step | Description |
    |------|-------------|
    | **1. Data Ingestion** | Synthetic 10,000-customer dataset with realistic correlations |
    | **2. EDA** | Distribution plots, correlation heatmap, class balance analysis |
    | **3. Feature Engineering** | 14 derived features: billing stress, engagement score, interaction terms, log transforms |
    | **4. Preprocessing** | KNN imputation → Yeo-Johnson power transform → StandardScaler (all via ColumnTransformer) |
    | **5. Class Imbalance** | SMOTE oversampling on the minority (churn) class |
    | **6. Model Zoo** | Logistic Regression · Random Forest · XGBoost · LightGBM · Gradient Boosting |
    | **7. Hyper-param Tuning** | Optuna Bayesian optimisation (TPE sampler, 30 trials) |
    | **8. Stacking Ensemble** | Base: RF+XGB+LGBM+GB → Meta: Logistic Regression (passthrough=True) |
    | **9. Evaluation** | ROC-AUC · PR-AUC · F1 · Precision · Recall · Confusion Matrix |
    | **10. Explainability** | SHAP TreeExplainer — global summary + local waterfall |
    | **11. Persistence** | joblib + pickle (preprocessor, model, metadata, model card) |
    | **12. Deployment** | Streamlit dashboard with single predict, batch scoring, and analytics |

    ---

    ### Key Engineered Features

    | Feature | Formula | Signal |
    |---------|---------|--------|
    | `billing_stress_index` | `payment_delay × monthly_charge / 100` | Financial pressure |
    | `engagement_score` | `usage×0.4 + login_freq×2` | Platform stickiness |
    | `satisfaction_risk` | `tickets×3 − nps` | Frustration level |
    | `charge_per_product` | `monthly / num_products` | Value perception |
    | `nps_x_usage` | `nps × feature_usage` | Loyalty amplifier |
    | `high_value_at_risk` | `monthly > P75 and nps < 5` | Priority flag |
    | `log_total_charges` | `log1p(total)` | Skewness reduction |
    | `is_month_to_month` | Binary flag | Contract volatility |

    ---

    ### Files
    ```
    churn_model.py      ← Full ML pipeline (train + save)
    ui_styling.py       ← CSS, colours, Plotly chart builders
    app.py              ← Streamlit deployment app
    model_artefacts/    ← Saved models + metadata
      preprocessor.joblib / .pkl
      stacking_model.joblib / .pkl
      rf_model.joblib / .pkl
      feature_meta.pkl
      model_card.pkl / .txt
    churn_data.csv      ← Training dataset
    model_comparison.csv← Evaluation results
    eda_*.png           ← EDA plots
    shap_*.png          ← SHAP plots
    ```

    ---

    ### How to Run
    ```bash
    # Step 1 — Install dependencies
    pip install numpy pandas scikit-learn xgboost lightgbm imbalanced-learn optuna shap streamlit plotly joblib

    # Step 2 — Train the model (generates all artefacts)
    python churn_model.py

    # Step 3 — Launch the dashboard
    streamlit run app.py
    ```

    ---

    <div style="text-align:center; color:{PALETTE['text_muted']}; font-size:0.8rem; margin-top:2rem;">
        Built with ❤️ · Stacking Ensemble · Optuna · SHAP · Streamlit
    </div>
    """, unsafe_allow_html=True)