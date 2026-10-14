from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    art = joblib.load(ROOT / "models" / "churn_model.joblib")
    pipe = art["pipeline"]
    sample = pd.read_csv(ROOT / "data" / "telco_churn.csv").drop(columns=["customerID", "Churn"])
    sample["TotalCharges"] = pd.to_numeric(sample["TotalCharges"], errors="coerce").fillna(0)
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    bg = pd.DataFrame(prep.transform(sample.sample(200, random_state=1)),
                      columns=prep.get_feature_names_out())
    return art, sample, shap.Explainer(clf, bg)


art, sample, explainer = load_artifacts()
pipe = art["pipeline"]

st.title("📉 Customer Churn Predictor")
st.caption(f"Model: {art['name']}  |  Enter customer details to estimate churn risk and see why.")

opts = lambda c: sorted(sample[c].unique().tolist())

with st.sidebar:
    st.header("Customer details")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", opts("Contract"))
    internet = st.selectbox("Internet service", opts("InternetService"))
    payment = st.selectbox("Payment method", opts("PaymentMethod"))
    monthly = st.slider("Monthly charges", 18.0, 120.0, 70.0)
    tech = st.selectbox("Tech support", opts("TechSupport"))
    senior = st.selectbox("Senior citizen", [0, 1])
    paperless = st.selectbox("Paperless billing", opts("PaperlessBilling"))

# Start from a typical customer (mode/median), then override with user inputs
row = {c: (sample[c].median() if pd.api.types.is_numeric_dtype(sample[c]) else sample[c].mode()[0])
       for c in art["columns"]}
row.update(tenure=tenure, Contract=contract, InternetService=internet, PaymentMethod=payment,
           MonthlyCharges=monthly, TechSupport=tech, SeniorCitizen=senior,
           PaperlessBilling=paperless, TotalCharges=monthly * tenure)
X = pd.DataFrame([row])[art["columns"]]

prob = float(pipe.predict_proba(X)[0, 1])
risk = "High" if prob >= 0.6 else "Medium" if prob >= 0.3 else "Low"

c1, c2 = st.columns([1, 2])
with c1:
    st.metric("Churn probability", f"{prob:.1%}")
    st.metric("Risk level", risk)
    st.subheader("Suggested action")
    if risk == "High":
        st.error("Offer a discounted 1-2 year contract and a proactive retention call.")
    elif risk == "Medium":
        st.warning("Send a loyalty offer or bundle tech support for free for 3 months.")
    else:
        st.success("No action needed. Keep monitoring.")

with c2:
    st.subheader("Why this prediction?")
    prep = pipe.named_steps["prep"]
    Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out())
    sv = explainer(Xt)
    if sv.values.ndim == 3:
        sv = sv[:, :, 1]
    contrib = pd.Series(sv.values[0], index=Xt.columns)
    contrib.index = [c.split("__", 1)[-1] for c in contrib.index]
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:10])[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top.values, color=["#ff0051" if v > 0 else "#008bfb" for v in top.values])
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Impact on churn risk (SHAP value)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Red bars push churn risk up, blue bars push it down.")