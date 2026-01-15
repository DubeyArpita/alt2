import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Analytics Dashboard", layout="wide")

DATA_FILE = "data/dataset.csv"
MODELS_DIR = "models"

@st.cache_resource
def load_models():
    lr = joblib.load(os.path.join(MODELS_DIR, "logistic_model.pkl"))
    xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
    rf = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    rf_cols = joblib.load(os.path.join(MODELS_DIR, "rf_columns.pkl"))
    if not isinstance(rf_cols, (list, tuple)):
        raise ValueError("rf_columns.pkl must be a plain list/tuple of column names (strings).")
    return lr, xgb, rf, list(rf_cols)

def predict_all(input_df: pd.DataFrame, lr_model, xgb_model, rf_model, rf_columns):
    # LR pipeline
    lr_risk = lr_model.predict(input_df)[0]

    # XGB pipeline
    xgb_score = float(np.clip(xgb_model.predict(input_df)[0], 0, 100))

    # RF dummy+align
    rf_encoded = pd.get_dummies(input_df, drop_first=True)
    rf_encoded = rf_encoded.reindex(columns=rf_columns, fill_value=0)
    rf_score = float(np.clip(rf_model.predict(rf_encoded)[0], 0, 100))

    return lr_risk, xgb_score, rf_score

def apply_status_color(val):
    if val in ("High Risk", "High"):
        return "background-color: #ff6b6b; color: white; font-weight: bold;"
    if val in ("Medium Risk", "Medium"):
        return "background-color: #ffd93d; color: black; font-weight: bold;"
    if val in ("Low Risk", "Low"):
        return "background-color: #6bcf7f; color: white; font-weight: bold;"
    return ""

# ----- CSS (keep it simpler; you can paste your long CSS if you want) -----
st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ----- Sidebar -----
with st.sidebar:
    st.markdown("<h2 style='text-align:center;color:#00D1FF;'>ALTSCORE</h2>", unsafe_allow_html=True)
    st.write("---")
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
    if st.button("📊 Dashboard", use_container_width=True):
        st.rerun()
    if st.button("➕ New Registration", use_container_width=True):
        st.switch_page("pages/Add_user_page.py")
    st.write("---")

    if st.button("🗑️ Delete Last Entry", use_container_width=True):
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            if len(df) > 0:
                deleted_user = df.iloc[-1].get("user_id", "Unknown")
                df = df.iloc[:-1]
                df.to_csv(DATA_FILE, index=False)
                st.success(f"✅ Deleted: {deleted_user}")
                st.rerun()
            else:
                st.warning("No entries to delete.")
        else:
            st.warning("No dataset found.")

st.markdown("<h1>📊 Credit Analytics Dashboard</h1>", unsafe_allow_html=True)

# ----- Ensure dataset exists -----
if not os.path.exists("data"):
    os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA_FILE):
    st.warning("📭 Dataset file not found. Please add users first.")
    st.stop()

df_raw = pd.read_csv(DATA_FILE)
if df_raw.empty:
    st.warning("📭 Dataset is empty. Please register some users first.")
    st.stop()

# ----- Load models -----
try:
    lr_model, xgb_model, rf_model, rf_columns = load_models()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ----- Metrics -----
col1, col2, col3, col4 = st.columns(4)
overall_avg = df_raw["alt_credit_score"].mean() if "alt_credit_score" in df_raw.columns else 0.0
with col1:
    st.metric("Avg Score / 100", f"{overall_avg:.1f}")
with col2:
    st.metric("Total Users", f"{len(df_raw)}")
with col3:
    good_users = int((df_raw["alt_credit_score"] >= 70).sum())
    st.metric("Good (≥70)", f"{good_users}")
with col4:
    risky_users = int((df_raw["alt_credit_score"] < 50).sum())
    st.metric("Risky (<50)", f"{risky_users}")

st.write("")

# ----- Registered users (last 10) -----
st.subheader("📋 Registered Users (Last 10)")
display_df = df_raw.tail(10).copy()
display_df.index = range(1, len(display_df) + 1)
cols = [c for c in ["user_id","employment_type","income_range","monthly_income","upi_txn_count","alt_credit_score"] if c in display_df.columns]
st.dataframe(display_df[cols], use_container_width=True)

st.write("")

# ----- AI Predictions for latest 5 -----
st.subheader("🤖 AI Model Predictions (Latest 5 Users)")
df_predict = df_raw.tail(5).copy()
pred_rows = []

for idx, row in df_predict.iterrows():
    input_data = {
        "employment_type": row.get("employment_type", "salaried"),
        "income_range": row.get("income_range", "10000-30000"),
        "city_tier": int(row.get("city_tier", 2)),
        "bank_account_age_months": int(row.get("bank_account_age_months", 24)),
        "num_bank_accounts": int(row.get("num_bank_accounts", 1)),
        "monthly_income": float(row.get("monthly_income", 30000)),
        "rent_paid_on_time": float(row.get("rent_paid_on_time", 1.0)),
        "utility_delay_days": float(row.get("utility_delay_days", 0.0)),
        "upi_txn_count": float(row.get("upi_txn_count", 20.0)),
        "avg_month_end_balance": float(row.get("avg_month_end_balance", 5000.0)),
        "overdraft_event": int(row.get("overdraft_event", 0)),
    }
    input_df = pd.DataFrame([input_data])

    try:
        lr_risk, xgb_score, rf_score = predict_all(input_df, lr_model, xgb_model, rf_model, rf_columns)
    except Exception as e:
        lr_risk, xgb_score, rf_score = "Error", np.nan, np.nan

    pred_rows.append({
        "User ID": row.get("user_id", f"User_{idx}"),
        "Actual Score": row.get("alt_credit_score", np.nan),
        "LR Risk": lr_risk,
        "XGB Score": (f"{xgb_score:.1f}" if np.isfinite(xgb_score) else "Error"),
        "RF Score": (f"{rf_score:.1f}" if np.isfinite(rf_score) else "Error"),
    })

pred_df = pd.DataFrame(pred_rows)
pred_df.index = range(1, len(pred_df) + 1)

if "LR Risk" in pred_df.columns:
    st.dataframe(pred_df.style.map(apply_status_color, subset=["LR Risk"]), use_container_width=True)
else:
    st.dataframe(pred_df, use_container_width=True)
