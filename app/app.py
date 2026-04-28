import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Sentilytics AI",
    page_icon="📊",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# =========================
# SESSION STATE
# =========================
if "df" not in st.session_state:
    st.session_state.df = None

# =========================
# SAAS UI STYLING (FIXED)
# =========================
st.markdown("""
<style>

/* MAIN BACKGROUND */
.main {
    background: #0a0f1a;
    color: #e5e7eb;
}

/* HERO SECTION */
.hero {
    padding: 40px;
    border-radius: 16px;
    background: linear-gradient(135deg,#0f172a,#111827);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
}

.title {
    font-size: 40px;
    font-weight: 800;
    background: linear-gradient(90deg,#60a5fa,#a78bfa,#34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #94a3b8;
    font-size: 14px;
}

/* ================= SIDEBAR FIX ================= */
section[data-testid="stSidebar"] {
    background: #0b1220 !important;
}

/* ALL SIDEBAR TEXT WHITE */
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* NAVIGATION */
.stRadio label {
    color: #ffffff !important;
    font-size: 15px;
    font-weight: 500;
}

/* HOVER EFFECT */
.stRadio div:hover label {
    color: #60a5fa !important;
}

/* HEADINGS */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #ffffff !important;
}

/* METRICS */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    padding: 10px;
    border-radius: 10px;
}

/* GENERAL TEXT */
p, span, label {
    color:black, navy blue !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HERO SECTION
# =========================
st.markdown("""
<div class="hero">
    <div class="title">Sentilytics AI</div>
    <div class="subtitle">
        Social Media Sentiment Intelligence Dashboard (NLP + ML)
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📤 Upload Data", "📊 Analytics"]
)

# =========================
# HOME PAGE
# =========================
if page == "🏠 Home":

    st.subheader("🚀 AI-Powered Sentiment Analysis System")

    col1, col2, col3 = st.columns(3)

    col1.metric("⚡ Speed", "Real-Time")
    col2.metric("🤖 Model", "TF-IDF + ML")
    col3.metric("📊 Output", "Business Insights")

    st.info("Upload your dataset to start sentiment analysis.")

# =========================
# UPLOAD PAGE
# =========================
elif page == "📤 Upload Data":

    st.subheader("📤 Upload Dataset")

    file = st.file_uploader("Upload CSV (must contain 'text' column)")

    if file:

        df = pd.read_csv(file)

        df["prediction"] = model.predict(vectorizer.transform(df["text"]))

        st.session_state.df = df

        st.success("Analysis Completed Successfully")

        st.dataframe(df.head())

# =========================
# ANALYTICS PAGE
# =========================
elif page == "📊 Analytics":

    st.subheader("📊 Sentiment Analytics Dashboard")

    df = st.session_state.df

    if df is None:
        st.warning("Please upload dataset first")

    else:

        # ================= KPI =================
        total = len(df)
        pos = len(df[df["prediction"] == "positive"])
        neg = len(df[df["prediction"] == "negative"])
        neu = len(df[df["prediction"] == "neutral"])

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Posts", total)
        c2.metric("Positive %", f"{pos/total*100:.1f}%")
        c3.metric("Negative %", f"{neg/total*100:.1f}%")
        c4.metric("Neutral %", f"{neu/total*100:.1f}%")

        st.divider()

        # ================= PIE CHART =================
        st.markdown("### 🧿 Sentiment Distribution")

        fig1 = px.pie(df, names="prediction", hole=0.55)

        st.plotly_chart(fig1, use_container_width=True)

        # ================= BAR CHART =================
        st.markdown("### 📊 Sentiment Volume")

        bar = df["prediction"].value_counts().reset_index()
        bar.columns = ["sentiment", "count"]

        fig2 = px.bar(bar, x="sentiment", y="count", color="sentiment")

        st.plotly_chart(fig2, use_container_width=True)

        # ================= TREND =================
        st.markdown("### 📈 Sentiment Trend Over Time")

        df = df.copy()
        df["time"] = range(len(df))

        mapping = {"positive": 1, "neutral": 0, "negative": -1}
        df["score"] = df["prediction"].map(mapping)

        df["trend"] = df["score"].rolling(5, 1).mean()

        fig3 = px.line(df, x="time", y="trend", markers=True)

        st.plotly_chart(fig3, use_container_width=True)

        # ================= PLATFORM ANALYSIS =================
        st.markdown("### 🌍 Platform Analysis")

        if "platform" in df.columns:

            platform_data = df.groupby(["platform", "prediction"]).size().reset_index(name="count")

            fig4 = px.bar(
                platform_data,
                x="platform",
                y="count",
                color="prediction",
                barmode="group"
            )

            st.plotly_chart(fig4, use_container_width=True)

        else:
            st.info("No platform column found (optional feature)")

        # ================= INSIGHT =================
        st.markdown("### 🧠 Business Insight")

        score = (pos - neg) / total

        st.metric("Sentiment Score", round(score, 2))

        if score > 0.3:
            st.success("Strong Positive Brand Image")
        elif score < -0.3:
            st.error("Brand Risk Detected")
        else:
            st.warning("Neutral Public Sentiment")