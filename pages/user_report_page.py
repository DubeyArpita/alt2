import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Analysis Report", layout="centered")

DATA_FILE = "data/dataset.csv"

# --- CSS ---
st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none !important; }
.stApp {
    background: linear-gradient(rgba(0,0,0,0.82), rgba(0,0,0,0.82)),
                url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1744&q=80");
    background-size: cover;
    background-attachment: fixed;
}
.main-white-box {
    background-color: white !important;
    padding: 36px !important;
    border-radius: 15px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25) !important;
    margin-top: 18px !important;
    margin-bottom: 30px !important;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00D1FF;'>User Credit Analysis</h2>", unsafe_allow_html=True)
    st.write("---")
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
    if st.button("📊 Dashboard", use_container_width=True):
        st.switch_page("pages/dashboard_page.py")
    if st.button("➕ New Registration", use_container_width=True):
        st.switch_page("pages/Add_user_page.py")
    st.write("---")

# --- Get report data ---
data = st.session_state.get("report_data", None)

# Fallback: if session_state missing, use last row from dataset.csv (best-effort)
if data is None and os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    if not df.empty:
        last = df.iloc[-1]
        data = {
            "user_id": last.get("user_id", "Unknown"),
            "lr": None,
            "xgb": None,
            "rf": None,
            "final": int(last.get("alt_credit_score", 0)),
            "lr_risk": None,
            "eligibility": "✅ ELIGIBLE" if float(last.get("alt_credit_score", 0)) >= 70
                          else "⚠️ CONDITIONAL" if float(last.get("alt_credit_score", 0)) >= 40
                          else "❌ RISKY",
            "risk_level": "Low Risk" if float(last.get("alt_credit_score", 0)) >= 70
                         else "Medium Risk" if float(last.get("alt_credit_score", 0)) >= 40
                         else "High Risk",
        }

if data is None:
    st.error("No data found! Please register first.")
    st.stop()

# --- Balloons once ---
if "show_balloons" not in st.session_state:
    st.balloons()
    st.session_state.show_balloons = True

st.markdown("<h1 style='text-align:center;color:#00D1FF;'>Personalized Credit Report</h1>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align:center;color:white;'>User ID: {data['user_id']}</h3>", unsafe_allow_html=True)

final_score = int(data.get("final", 0))

# LR display
lr_display = "—"
if data.get("lr_risk"):
    emoji = "🟢" if data["lr_risk"] == "Low Risk" else "🟡" if data["lr_risk"] == "Medium Risk" else "🔴"
    lr_display = f"{emoji} {data['lr_risk']}"
elif data.get("lr") is not None:
    lr_val = float(data["lr"])
    lr_display = "🔴 High Risk" if lr_val < 40 else "🟡 Medium Risk" if lr_val < 70 else "🟢 Low Risk"

report_df = pd.DataFrame({
    "Analysis Model": ["Logistic Regression", "Random Forest Score", "XGBoost Score", "FINAL SCORE"],
    "Result": [
        lr_display,
        (f"{data['rf']:.1f}/100" if isinstance(data.get("rf"), (int, float)) else "—"),
        (f"{data['xgb']:.1f}/100" if isinstance(data.get("xgb"), (int, float)) else "—"),
        f"{final_score}/100"
    ],
    "Verdict": [
        (f"Score: {data['lr']}" if data.get("lr") is not None else "LR Prediction"),
        "RF Prediction",
        "XGB Prediction",
        "✅ ELIGIBLE" if final_score >= 70 else "CONDITIONAL APPROVAL" if final_score >= 40 else "❌ RISKY"
    ]
})
report_df.index = report_df.index + 1

st.markdown('<div class="main-white-box">', unsafe_allow_html=True)
st.table(report_df)
st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
st.markdown("<h4 style='color: white; text-align: center;'>Overall Financial Health</h4>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.progress(final_score / 100)
    st.markdown(f"<p style='color:white;text-align:center;font-size:24px;font-weight:800;'>{final_score}/100</p>", unsafe_allow_html=True)

if final_score >= 70:
    st.markdown("""
    <div style='background-color:#d4edda;padding:20px;border-radius:10px;border-left:5px solid #28a745;margin-top:20px;'>
      <h4 style='color:#155724;margin:0;'>🎉 Congratulations!</h4>
      <p style='color:#155724;margin:10px 0 0 0;'>This user has a strong profile and is eligible for credit facilities.</p>
    </div>
    """, unsafe_allow_html=True)
elif final_score >= 40:
    st.markdown("""
    <div style='background-color:#fff3cd;padding:20px;border-radius:10px;border-left:5px solid #ffc107;margin-top:20px;'>
      <h4 style='color:#856404;margin:0;'>⚠️ Conditional Approval</h4>
      <p style='color:#856404;margin:10px 0 0 0;'>Moderate risk. Additional verification or limited credit is recommended.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='background-color:#f8d7da;padding:20px;border-radius:10px;border-left:5px solid #dc3545;margin-top:20px;'>
      <h4 style='color:#721c24;margin:0;'>❌ High Risk</h4>
      <p style='color:#721c24;margin:10px 0 0 0;'>High risk. Credit extension is not recommended.</p>
    </div>
    """, unsafe_allow_html=True)
