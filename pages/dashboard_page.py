import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Analytics Dashboard", layout="wide")

DATA_FILE = "data/dataset.csv"
MODELS_DIR = "models"


# -----------------------------
# Model loading (cached)
# -----------------------------
@st.cache_resource
def load_models():
    lr = joblib.load(os.path.join(MODELS_DIR, "logistic_model.pkl"))  # Pipeline (classifier)
    xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))      # Pipeline (regressor)
    rf = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))        # Pipeline (regressor)
    return lr, xgb, rf


# -----------------------------
# Helpers
# -----------------------------
def compute_risk_level(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 70:
        return "Low"
    elif score >= 40:
        return "Medium"
    else:
        return "High"


def color_risk(val):
    if val == "Low":
        return "background-color: #6bcf7f; color: white; font-weight: bold;"
    elif val == "Medium":
        return "background-color: #ffd93d; color: black; font-weight: bold;"
    elif val == "High":
        return "background-color: #ff6b6b; color: white; font-weight: bold;"
    return ""


def color_lr_risk(val):
    if val in ("High Risk", "High"):
        return "background-color: #ff6b6b; color: white; font-weight: bold;"
    if val in ("Medium Risk", "Medium"):
        return "background-color: #ffd93d; color: black; font-weight: bold;"
    if val in ("Low Risk", "Low"):
        return "background-color: #6bcf7f; color: white; font-weight: bold;"
    return ""


def predict_all(input_df: pd.DataFrame, lr_model, xgb_model, rf_model):
    # Normalize categorical text (match training)
    for c in ["employment_type", "income_range"]:
        if c in input_df.columns:
            input_df[c] = input_df[c].astype(str).str.strip().str.lower()

    # LR pipeline -> risk label
    lr_risk = lr_model.predict(input_df)[0]

    # XGB pipeline -> score
    xgb_score = float(np.clip(xgb_model.predict(input_df)[0], 0, 100))

    # RF pipeline -> score (NO rf_columns needed)
    rf_score = float(np.clip(rf_model.predict(input_df)[0], 0, 100))

    return lr_risk, xgb_score, rf_score


# -----------------------------
# Minimal CSS + hide default nav
# -----------------------------
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Sidebar Navigation + Actions
# -----------------------------
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
            try:
                df_del = pd.read_csv(DATA_FILE)
                if not df_del.empty:
                    deleted_user = df_del.iloc[-1].get("user_id", "Unknown")
                    df_del = df_del.iloc[:-1]
                    df_del.to_csv(DATA_FILE, index=False)
                    st.success(f"✅ Deleted: {deleted_user}")
                    st.rerun()
                else:
                    st.warning("No entries to delete.")
            except Exception as e:
                st.error(f"Error deleting entry: {e}")
        else:
            st.warning("No dataset found.")


# -----------------------------
# Main Header
# -----------------------------
st.markdown("<h1>📊 Credit Analytics Dashboard</h1>", unsafe_allow_html=True)
st.caption("Sorted by Credit Score (DESC) so Low Risk users appear on top.")


# -----------------------------
# Load dataset
# -----------------------------
os.makedirs("data", exist_ok=True)

if not os.path.exists(DATA_FILE):
    st.warning("📭 Dataset file not found. Please add users first.")
    st.stop()

df_raw = pd.read_csv(DATA_FILE)

if df_raw.empty:
    st.warning("📭 Dataset is empty. Please register some users first.")
    st.stop()

# Rename column alt_credit_score -> credit_score
if "alt_credit_score" in df_raw.columns and "credit_score" not in df_raw.columns:
    df_raw = df_raw.rename(columns={"alt_credit_score": "credit_score"})

# Ensure credit_score exists
if "credit_score" not in df_raw.columns:
    st.error("❌ Column 'credit_score' not found (or 'alt_credit_score' missing).")
    st.stop()

# Convert + sort DESC
df_raw["credit_score"] = pd.to_numeric(df_raw["credit_score"], errors="coerce")
df_raw = df_raw.sort_values(by="credit_score", ascending=False, na_position="last").reset_index(drop=True)

# Add risk_level column
df_raw["risk_level"] = df_raw["credit_score"].apply(compute_risk_level)


# -----------------------------
# Load models
# -----------------------------
try:
    lr_model, xgb_model, rf_model = load_models()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

overall_avg = float(df_raw["credit_score"].mean(skipna=True))
total_users = len(df_raw)
good_users = int((df_raw["credit_score"] >= 70).sum())
high_risk_users = int((df_raw["credit_score"] < 40).sum())

with col1:
    st.metric("Avg Credit Score / 100", f"{overall_avg:.1f}")
with col2:
    st.metric("Total Users", f"{total_users}")
with col3:
    st.metric("Low Risk (≥70)", f"{good_users}")
with col4:
    st.metric("High Risk (<40)", f"{high_risk_users}")

st.write("")


# -----------------------------
# Top 10 Users (sorted)
# -----------------------------
st.subheader("✅ Top Users (Low Risk on Top) — Sorted by Credit Score")

display_df = df_raw.head(10).copy()
display_df.index = range(1, len(display_df) + 1)

cols = [
    "user_id",
    "employment_type",
    "income_range",
    "city_tier",
    "monthly_income",
    "upi_txn_count",
    "credit_score",
    "risk_level"
]
cols = [c for c in cols if c in display_df.columns]

st.dataframe(
    display_df[cols].style.map(color_risk, subset=["risk_level"]),
    use_container_width=True
)

st.write("")


# -----------------------------
# Predictions for Top 5 Users (by score)
# -----------------------------
st.subheader("🤖 AI Model Predictions (Top 5 Users by Credit Score)")

df_predict = df_raw.head(5).copy()
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
        lr_risk, xgb_score, rf_score = predict_all(input_df, lr_model, xgb_model, rf_model)
        xgb_score_s = f"{xgb_score:.1f}"
        rf_score_s = f"{rf_score:.1f}"
    except Exception:
        lr_risk, xgb_score_s, rf_score_s = "Error", "Error", "Error"

    pred_rows.append({
        "User ID": row.get("user_id", f"User_{idx}"),
        "Credit Score": row.get("credit_score", np.nan),
        "Risk Level": row.get("risk_level", "Unknown"),
        "LR Risk": lr_risk,
        "XGB Score": xgb_score_s,
        "RF Score": rf_score_s,
    })

pred_df = pd.DataFrame(pred_rows)
pred_df.index = range(1, len(pred_df) + 1)

styled = pred_df.style.map(color_risk, subset=["Risk Level"]).map(color_lr_risk, subset=["LR Risk"])
st.dataframe(styled, use_container_width=True)
