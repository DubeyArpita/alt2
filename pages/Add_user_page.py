import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Register User", layout="centered")

DATA_FILE = "data/dataset.csv"
MODELS_DIR = "models"

REQUIRED_COLUMNS = [
    "user_id", "employment_type", "income_range", "city_tier",
    "bank_account_age_months", "num_bank_accounts", "monthly_income",
    "rent_paid_on_time", "utility_delay_days", "upi_txn_count",
    "avg_month_end_balance", "overdraft_event", "alt_credit_score"
]

# ---------- Helpers ----------
@st.cache_resource
def load_models():
    lr = joblib.load(os.path.join(MODELS_DIR, "logistic_model.pkl"))  # Pipeline
    xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))      # Pipeline
    rf = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))        # Pipeline (recommended)

    # If your RF is NOT a pipeline and you still use rf_columns.pkl, keep these two lines:
    # rf_cols = joblib.load(os.path.join(MODELS_DIR, "rf_columns.pkl"))
    # return lr, xgb, rf, list(rf_cols)

    return lr, xgb, rf


def ensure_dataset_file():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(DATA_FILE, index=False)


def get_dropdown_options_from_dataset():
    # Use dataset.csv categories if data exists; else fallback.
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            if not df.empty:
                emp = sorted(df["employment_type"].dropna().unique().tolist()) if "employment_type" in df.columns else []
                inc = sorted(df["income_range"].dropna().unique().tolist()) if "income_range" in df.columns else []
                tiers_raw = df["city_tier"].dropna().unique().tolist() if "city_tier" in df.columns else []
                tiers = []
                for t in tiers_raw:
                    try:
                        tiers.append(int(t))
                    except Exception:
                        pass
                tiers = sorted(list(set(tiers)))
                if emp and inc and tiers:
                    return emp, inc, tiers
        except Exception:
            pass

    # Fallback (must match training categories)
    employment_options = ["gig", "salaried", "self_employed", "contract", "student", "unemployed"]
    income_options = ["0-15000", "10000-30000", "15000-30000", "20000-100000", "25000-80000", "30000-50000"]
    city_tier_options = [1, 2, 3]
    return employment_options, income_options, city_tier_options


def generate_user_id():
    """
    Generates USER_0001, USER_0002... based on last entry in dataset.csv.
    Works even if file is empty or last id is malformed.
    """
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            if not df.empty and "user_id" in df.columns:
                last_id = str(df.iloc[-1]["user_id"])
                m = re.search(r"(\d+)$", last_id)
                if m:
                    n = int(m.group(1))
                    return f"USER_{n+1:04d}"
        except Exception:
            pass
    return "USER_0001"


def predict_all(input_data: dict, lr_model, xgb_model, rf_model):
    input_df = pd.DataFrame([input_data])

    # Normalize categorical text exactly like training (recommended)
    for c in ["employment_type", "income_range"]:
        if c in input_df.columns:
            input_df[c] = input_df[c].astype(str).str.strip().str.lower()

    # Logistic Regression (classification pipeline)
    lr_risk = lr_model.predict(input_df)[0]
    lr_probs_arr = lr_model.predict_proba(input_df)[0]
    lr_probs = dict(zip(lr_model.classes_, lr_probs_arr))

    # Convert risk -> score
    risk_to_score = {"Low Risk": 85, "Medium Risk": 55, "High Risk": 25}
    lr_score = int(risk_to_score.get(lr_risk, 50))

    # XGBoost (regression pipeline)
    xgb_score = float(np.clip(xgb_model.predict(input_df)[0], 0, 100))

    # Random Forest (PIPELINE) ✅
    rf_score = float(np.clip(rf_model.predict(input_df)[0], 0, 100))

    final_score = int(round((lr_score + xgb_score + rf_score) / 3))

    if final_score >= 70:
        eligibility = "✅ ELIGIBLE"
        risk_level = "Low Risk"
    elif final_score >= 40:
        eligibility = "⚠️ CONDITIONAL"
        risk_level = "Medium Risk"
    else:
        eligibility = "❌ RISKY"
        risk_level = "High Risk"

    return {
        "lr_risk": lr_risk,
        "lr_probs": lr_probs,
        "lr_score": lr_score,
        "xgb_score": xgb_score,
        "rf_score": rf_score,
        "final_score": final_score,
        "eligibility": eligibility,
        "risk_level": risk_level,
    }


# ---------- Load models + dataset ----------
ensure_dataset_file()

try:
    lr_model, xgb_model, rf_model = load_models()
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

employment_options, income_options, city_tier_options = get_dropdown_options_from_dataset()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00D1FF;'>ALTSCORE</h2>", unsafe_allow_html=True)
    st.write("---")
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
    if st.button("📊 Dashboard", use_container_width=True):
        st.switch_page("pages/dashboard_page.py")
    st.write("---")

st.markdown("<h1>📝 User Registration</h1>", unsafe_allow_html=True)

# ---------- Form ----------
with st.form("user_registration_form"):
    col1, col2 = st.columns(2)

    with col1:
        employment_type = st.selectbox("Employment Type *", employment_options)
        income_range = st.selectbox("Income Range (Monthly) *", income_options)
        city_tier = st.selectbox("City Tier *", city_tier_options)
        bank_account_age_months = st.number_input(
            "Bank Account Age (Months) *",
            min_value=0, max_value=240, value=24, step=1
        )

    with col2:
        num_bank_accounts = st.number_input("Number of Bank Accounts *", min_value=1, max_value=15, value=1, step=1)
        monthly_income = st.number_input("Monthly Income (₹) *", min_value=0, value=30000, step=1000)
        rent_paid_on_time = st.slider(
            "Rent Paid On Time (0 to 1) *",
            min_value=0.0, max_value=1.0, value=1.0, step=0.1
        )
        utility_delay_days = st.number_input("Utility Delay Days *", min_value=0.0, value=0.0, step=1.0)
        upi_txn_count = st.number_input("Monthly UPI Transaction Count *", min_value=0.0, value=20.0, step=1.0)

    col3, col4 = st.columns(2)
    with col3:
        avg_month_end_balance = st.number_input(
            "Average Month-End Balance (₹) *",
            min_value=0.0, value=5000.0, step=100.0
        )
    with col4:
        overdraft_event = st.selectbox("Overdraft Availed? *", ["No", "Yes"])

    submitted = st.form_submit_button("💾 Save User & Generate Score 🚀", use_container_width=True)

# ---------- On submit ----------
if submitted:
    user_id = generate_user_id()
    st.info(f"🆔 Generated User ID: {user_id}")

    input_data = {
        "employment_type": str(employment_type).strip().lower(),
        "income_range": str(income_range).strip().lower(),
        "city_tier": int(city_tier),
        "bank_account_age_months": int(bank_account_age_months),
        "num_bank_accounts": int(num_bank_accounts),
        "monthly_income": float(monthly_income),
        "rent_paid_on_time": float(rent_paid_on_time),
        "utility_delay_days": float(utility_delay_days),
        "upi_txn_count": float(upi_txn_count),
        "avg_month_end_balance": float(avg_month_end_balance),
        "overdraft_event": 1 if overdraft_event == "Yes" else 0,
    }

    with st.spinner("Processing user data and generating score..."):
        try:
            out = predict_all(input_data, lr_model, xgb_model, rf_model)

            # Save report data
            st.session_state["report_data"] = {
                "user_id": user_id,
                "lr": out["lr_score"],
                "xgb": out["xgb_score"],
                "rf": out["rf_score"],
                "final": out["final_score"],
                "lr_risk": out["lr_risk"],
                "lr_probs": out["lr_probs"],
                "eligibility": out["eligibility"],
                "risk_level": out["risk_level"],
            }

            # Append to CSV
            new_entry = {
                "user_id": user_id,
                "employment_type": input_data["employment_type"],
                "income_range": input_data["income_range"],
                "city_tier": input_data["city_tier"],
                "bank_account_age_months": input_data["bank_account_age_months"],
                "num_bank_accounts": input_data["num_bank_accounts"],
                "monthly_income": input_data["monthly_income"],
                "rent_paid_on_time": input_data["rent_paid_on_time"],
                "utility_delay_days": input_data["utility_delay_days"],
                "upi_txn_count": input_data["upi_txn_count"],
                "avg_month_end_balance": input_data["avg_month_end_balance"],
                "overdraft_event": input_data["overdraft_event"],
                "alt_credit_score": out["final_score"],  # keep original CSV name; dashboard renames to credit_score
            }

            df_csv = pd.read_csv(DATA_FILE)
            df_csv = pd.concat([df_csv, pd.DataFrame([new_entry])], ignore_index=True)
            df_csv.to_csv(DATA_FILE, index=False)

            st.success(f"✅ User {user_id} registered successfully!")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Logistic Regression", f"{out['lr_score']}", out["lr_risk"])
            with c2:
                st.metric("XGBoost", f"{out['xgb_score']:.1f}")
            with c3:
                st.metric("Random Forest", f"{out['rf_score']:.1f}")
            with c4:
                st.metric("Final Score", f"{out['final_score']}", out["risk_level"])

            st.switch_page("pages/user_report_page.py")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            import traceback
            st.code(traceback.format_exc())
