"""
app.py  --  Customer Churn Intelligence.   Run:  streamlit run app.py

Option 1  Deep Learning (ANN)
Option 2  Machine Learning models (Logistic Regression / Random Forest / Gradient Boosting)
Option 3  Compare the ANN with the ML models
"""
import json

import pandas as pd
import streamlit as st

import charts
from churn_utils import (ACTIONABLE, ANN_NAME, BASE_DIR, FEATURES, LABELS, ML_NAMES, TARGET,
                         churn_by_segment, evidence_text, explain_customer, load_data, load_store,
                         risk_level, simulate_fix, suggest_action)

st.set_page_config(page_title="Churn Intelligence", page_icon="📉", layout="wide")

# ------------------------------------------------------------------ look and feel
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {background: linear-gradient(90deg, #264653, #2A9D8F); color: white;
       padding: 1.2rem 1.6rem; border-radius: 12px; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 1.8rem; color: white;}
.hero p {margin: .2rem 0 0; opacity: .9;}
.card {background: #F7F9FB; border: 1px solid #E3E8EE; border-radius: 12px;
       padding: .9rem 1.1rem; margin-bottom: .6rem; color: #1F2933;}
.card .label {font-size: .8rem; color: #52606D; text-transform: uppercase; letter-spacing: .04em;}
.card .value {font-size: 1.7rem; font-weight: 700;}
.badge {display: inline-block; padding: .25rem .8rem; border-radius: 999px; color: white; font-weight: 600;}
</style>
<div class="hero"><h1>📉 Customer Churn Intelligence</h1>
<p>Not just <i>who</i> will leave — but <i>why</i>, and what to do about it.</p></div>
""", unsafe_allow_html=True)


def card(label, value):
    st.markdown(f'<div class="card"><div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ load once, keep in session
if "store" not in st.session_state:
    try:
        st.session_state["store"] = load_store()
        st.session_state["data"] = load_data()
        st.session_state["metrics"] = json.loads((BASE_DIR / "models" / "metrics.json").read_text())
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

store = st.session_state["store"]
data = st.session_state["data"]
metrics = st.session_state["metrics"]

# ------------------------------------------------------------------ sidebar
st.sidebar.header("1. Choose approach")
mode = st.sidebar.radio("Approach", ["Option 1: Deep Learning (ANN)",
                                     "Option 2: Machine Learning models",
                                     "Option 3: Compare ANN vs ML"], label_visibility="collapsed")
if mode.startswith("Option 1"):
    model_name = ANN_NAME
elif mode.startswith("Option 2"):
    model_name = st.sidebar.selectbox("ML model", ML_NAMES, index=2)
else:
    model_name = None

st.sidebar.header("2. Customer profile")
with st.sidebar.form("customer_form"):
    countries = store["categories"]["Geography"]
    geography = st.selectbox("Country", countries, index=countries.index("Germany"))
    gender = st.selectbox("Gender", store["categories"]["Gender"])
    age = st.slider("Age", 18, 92, 45)
    credit_score = st.slider("Credit score", 350, 850, 600)
    tenure = st.slider("Tenure (years)", 0, 10, 3)
    balance = st.number_input("Account balance", min_value=0.0, max_value=1_000_000.0,
                              value=120_000.0, step=5_000.0)
    salary = st.number_input("Estimated salary", min_value=0.0, max_value=1_000_000.0,
                             value=80_000.0, step=5_000.0)
    products = st.slider("Number of products", 1, 4, 1)
    has_card = st.radio("Has credit card?", ["Yes", "No"], horizontal=True)
    is_active = st.radio("Active member?", ["Yes", "No"], index=1, horizontal=True)
    st.form_submit_button("Analyse customer", type="primary", width="stretch")

customer = {
    "CreditScore": credit_score, "Geography": geography, "Gender": gender, "Age": age,
    "Tenure": tenure, "Balance": balance, "NumOfProducts": products,
    "HasCrCard": 1 if has_card == "Yes" else 0,
    "IsActiveMember": 1 if is_active == "Yes" else 0,
    "EstimatedSalary": salary,
}


# ------------------------------------------------------------------ page sections
def show_prediction(name):
    """Probability, reasons, actions and what-if for one model."""
    threshold = store["thresholds"][name]
    prob, reasons = explain_customer(store, name, customer)
    level = risk_level(prob, threshold)
    colour = {"High": "#E4572E", "Medium": "#E9A23B", "Low": "#2A9D8F"}[level]
    verdict = {"High": "Likely to leave", "Medium": "Watch closely", "Low": "Likely to stay"}[level]

    left, right = st.columns([1, 1.4])
    with left:
        st.plotly_chart(charts.gauge(prob, threshold), width="stretch")
        st.markdown(f'<div style="text-align:center"><span class="badge" style="background:{colour}">'
                    f'{level} risk · {verdict}</span></div>', unsafe_allow_html=True)
        st.caption(f"Model: {name}. Red line = decision threshold ({threshold:.0%}), "
                   "tuned for the best balance between catching churners and false alarms.")
    with right:
        st.plotly_chart(charts.reasons_bar(reasons), width="stretch")

    drivers = reasons[reasons["impact"] >= 1].head(4)        # ignore effects under 1 point
    st.subheader("Why is this customer at risk?")
    if drivers.empty:
        st.success("No strong risk factor found. This customer looks like our loyal customers.")
        return

    for _, d in drivers.iterrows():
        st.markdown(f"- {evidence_text(data, d['feature'], d['your_value'])}  \n"
                    f"  &nbsp;&nbsp;↳ adds **{d['impact']:.1f} points** to the churn probability")

    st.subheader("What can we do about it?")
    for _, d in drivers.iterrows():
        st.markdown(f"- **{LABELS[d['feature']]}:** {suggest_action(d['feature'], d['your_value'])}")

    fixable = [f for f in drivers["feature"] if f in ACTIONABLE]
    if fixable:
        new_prob = simulate_fix(store, name, customer, fixable)
        c1, c2, c3 = st.columns(3)
        with c1:
            card("Current churn risk", f"{prob:.0%}")
        with c2:
            card("Risk after acting on " + " & ".join(LABELS[f].lower() for f in fixable),
                 f"{new_prob:.0%}")
        with c3:
            card("Possible reduction", f"{(prob - new_prob) * 100:.0f} pts")
        st.caption("What-if: same customer, but these factors moved to the typical loyal-customer value.")
    else:
        st.info("The main drivers (e.g. age, country) cannot be changed by the bank — use them "
                "to decide who to target, not what to fix.")

    with st.expander("How are the reasons calculated?"):
        st.write("For every feature we swap this customer's value with the value of a typical customer "
                 "who stayed and measure how much the churn probability falls. A big fall means that "
                 "feature is a major reason for the risk. Features are tested one at a time, so "
                 "combined effects can be bigger than the sum.")


def show_insights(name):
    """Graphs about the data. name=None means option 3 (show all models' importance)."""
    churned = data[data[TARGET] == 1]
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        card("Customers", f"{len(data):,}")
    with k2:
        card("Churn rate", f"{data[TARGET].mean():.1%}")
    with k3:
        card("Customers lost", f"{len(churned):,}")
    with k4:
        card("Balance walking out", f"{churned['Balance'].sum() / 1e6:,.0f} M")

    st.subheader("Where is churn highest?")
    parts = []
    for f in FEATURES:                                       # check every feature
        table = churn_by_segment(data, f)
        table["feature"] = LABELS[f]
        parts.append(table)
    risky = pd.concat(parts)
    risky = risky[risky["customers"] >= 100].sort_values("churn_rate", ascending=False).head(6)
    risky["churn_rate"] = (risky["churn_rate"] * 100).round(1)
    risky = risky[["feature", "segment", "customers", "churn_rate"]]
    risky.columns = ["Factor", "Group", "Customers", "Churn rate (%)"]
    st.dataframe(risky, hide_index=True, width="stretch")

    st.subheader("Explore a factor")
    c1, c2 = st.columns(2)
    with c1:
        factor = st.selectbox("Churn rate by...", FEATURES, index=FEATURES.index("NumOfProducts"),
                              format_func=LABELS.get)
        st.plotly_chart(charts.segment_bar(data, factor), width="stretch")
    with c2:
        number = st.selectbox("Compare stayed vs left on...",
                              ["Age", "CreditScore", "Balance", "EstimatedSalary", "Tenure"],
                              format_func=LABELS.get)
        st.plotly_chart(charts.distribution(data, number), width="stretch")

    st.subheader("Risky combinations")
    h1, h2 = st.columns(2)
    row_feature = h1.selectbox("Rows", FEATURES, index=FEATURES.index("Age"), format_func=LABELS.get)
    col_feature = h2.selectbox("Columns", FEATURES, index=FEATURES.index("NumOfProducts"),
                               format_func=LABELS.get)
    if row_feature == col_feature:
        st.warning("Pick two different factors.")
    else:
        st.plotly_chart(charts.heatmap(data, row_feature, col_feature), width="stretch")

    g1, g2 = st.columns(2)
    with g1:
        if name:
            st.plotly_chart(charts.importance_bar(store["importance"][name], name), width="stretch")
        else:
            st.plotly_chart(charts.compare_importance(store["importance"]), width="stretch")
    with g2:
        st.plotly_chart(charts.correlation_heatmap(data), width="stretch")


def show_performance(name):
    """How good is one model?"""
    result = metrics[name]
    s = result["scores"]
    st.markdown(f"**Model:** {name}. Numbers are on a separate test set the model never saw.")
    cols = st.columns(5)
    items = [("ROC-AUC", "roc_auc"), ("Accuracy", "accuracy"), ("Precision", "precision"),
             ("Recall", "recall"), ("F1 score", "f1")]
    for col, (label, key) in zip(cols, items):
        with col:
            card(label, f"{s[key]:.2f}")
    st.caption("Accuracy alone can mislead: predicting 'nobody leaves' already scores ~80%. "
               "Recall = share of real churners we catch. Precision = share of our alerts that are right.")

    p1, p2 = st.columns(2)
    with p1:
        st.plotly_chart(charts.roc_curve_fig(result["roc"], s["roc_auc"]), width="stretch")
    with p2:
        st.plotly_chart(charts.confusion_fig(result["confusion_matrix"]), width="stretch")


def show_model_comparison():
    """Option 3, part 1: Deep Learning vs Machine Learning on the test data."""
    st.markdown("**Deep Learning:** Neural Network (ANN)  |  "
                "**Machine Learning:** Logistic Regression, Random Forest, Gradient Boosting")

    rows = []
    for name in metrics:
        s = metrics[name]["scores"]
        kind = "Deep Learning" if name == ANN_NAME else "Machine Learning"
        rows.append([name, kind, s["roc_auc"], s["accuracy"], s["precision"], s["recall"], s["f1"]])
    table = pd.DataFrame(rows, columns=["Model", "Type", "ROC-AUC", "Accuracy", "Precision",
                                        "Recall", "F1"])

    best_auc = table.loc[table["ROC-AUC"].idxmax(), "Model"]
    best_recall = table.loc[table["Recall"].idxmax(), "Model"]
    st.success(f"Best ROC-AUC: **{best_auc}**   |   Catches the most churners (recall): **{best_recall}**")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.compare_metrics_bar(metrics), width="stretch")
    with c2:
        st.plotly_chart(charts.compare_roc(metrics), width="stretch")
    st.dataframe(table.round(3), hide_index=True, width="stretch")
    st.caption("On this table-style (tabular) data, tree-based ML models usually match or beat neural "
               "networks, so a close result between the ANN and Gradient Boosting is normal.")


def show_customer_comparison():
    """Option 3, part 2: what does each model say about the customer in the sidebar?"""
    names, probs, rows = [], [], []
    for name in metrics:
        prob, reasons = explain_customer(store, name, customer)
        threshold = store["thresholds"][name]
        top = reasons[reasons["impact"] >= 1].head(3)
        top_text = ", ".join(f"{r['name']} (+{r['impact']:.0f})" for _, r in top.iterrows())
        names.append(name)
        probs.append(prob)
        rows.append([name, f"{prob:.0%}", risk_level(prob, threshold), top_text or "None"])

    st.plotly_chart(charts.compare_probabilities(names, probs, store["thresholds"]), width="stretch")
    st.subheader("Top reasons according to each model")
    table = pd.DataFrame(rows, columns=["Model", "Churn probability", "Risk", "Top reasons (points)"])
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("If different models give the same top reasons, we can trust those reasons more.")


# ------------------------------------------------------------------ tabs
if model_name:
    tab1, tab2, tab3 = st.tabs(["🔍 Predict & Explain", "📊 Customer Insights", "🎯 Model Performance"])
    with tab1:
        show_prediction(model_name)
    with tab2:
        show_insights(model_name)
    with tab3:
        show_performance(model_name)
else:
    tab1, tab2, tab3 = st.tabs(["⚖️ Compare Models", "👤 Same Customer, All Models",
                                "📊 Customer Insights"])
    with tab1:
        show_model_comparison()
    with tab2:
        show_customer_comparison()
    with tab3:
        show_insights(None)
