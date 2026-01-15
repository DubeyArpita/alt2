import streamlit as st

st.set_page_config(page_title="AltScore India | Home", layout="wide")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<h2 style='text-align:center;color:#00D1FF;'>ALTSCORE</h2>", unsafe_allow_html=True)
    st.write("---")
    if st.button("🏠 Home", use_container_width=True):
        st.rerun()
    if st.button("📊 Dashboard", use_container_width=True):
        st.switch_page("pages/dashboard_page.py")
    if st.button("➕ New Registration", use_container_width=True):
        st.switch_page("pages/Add_user_page.py")
    st.write("---")

# ---------- CSS ----------
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] { display: none; }
    [data-testid="stSidebar"] { background-color: #0b0b0b !important; border-right: 1px solid #333; }

    [data-testid="stSidebar"] .stButton button {
        background-color: #00D1FF !important;
        color: #000 !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        border: none !important;
        transition: 0.2s all ease;
        margin-bottom: 8px !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #fff !important;
        transform: scale(1.02);
    }

    .stApp {
        background: linear-gradient(rgba(0,0,0,0.72), rgba(0,0,0,0.72)),
                    url("https://images.unsplash.com/photo-1554224155-6726b3ff858f?auto=format&fit=crop&w=1772&q=80");
        background-size: cover;
        background-attachment: fixed;
    }

    .hero-container { text-align: center; padding-top: 60px; }
    .app-name { font-size: 50px; font-weight: 900; letter-spacing: 6px; color: #00D1FF; margin-bottom: 16px; }
    .hero-headline { font-size: 56px; font-weight: 800; color: #fff; margin-bottom: 10px; }
    .hero-subtext { font-size: 22px; color: #e0e0e0; max-width: 900px; margin: 0 auto; line-height: 1.6; }

    .feature-card {
        background-color: #fff;
        padding: 28px;
        border-radius: 16px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.32);
        min-height: 240px;
        width: 90%;
        margin-top: 110px;
        word-wrap: break-word;
    }
    .feature-card h4 { color: #1E1E1E; font-size: 24px; font-weight: 800; margin-bottom: 14px; }
    .feature-card p { color: #444; font-size: 16px; line-height: 1.6; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero-container">
        <div class="app-name">ALTSCORE INDIA</div>
        <div class="hero-headline">Credit Identity for the Next Billion.</div>
        <p class="hero-subtext">Redefining creditworthiness by unlocking the power of alternative data.</p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3, gap="large")
with col1:
    st.markdown(
        '<div class="feature-card"><h4>Alternative Data</h4>'
        '<p>Analyze rent, utilities, and UPI patterns to build a robust financial profile without needing traditional credit history.</p></div>',
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        '<div class="feature-card"><h4>AI-Driven Insights</h4>'
        '<p>Uses Logistic Regression (risk class) + XGBoost & Random Forest (score) to generate a fair and explainable score.</p></div>',
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        '<div class="feature-card"><h4>Financial Inclusion</h4>'
        '<p>Empowering students, gig workers, and underserved segments with responsible access to credit.</p></div>',
        unsafe_allow_html=True
    )

st.write("")
c1, c2, c3 = st.columns([1, 1.2, 1])
with c2:
    if st.button("🚀 Get Started (Register User)", use_container_width=True):
        st.switch_page("pages/Add_user_page.py")
