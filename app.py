
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Telecom Customer Intelligence",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD DATA ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    customers = pd.read_csv(os.path.join(BASE_DIR, "customer_data.csv"))
    models = pd.read_csv(os.path.join(BASE_DIR, "model_comparison.csv"))
    segments = pd.read_csv(os.path.join(BASE_DIR, "segment_profile.csv"))
    risk = pd.read_csv(os.path.join(BASE_DIR, "revenue_risk.csv"))
    risk_summary = pd.read_csv(os.path.join(BASE_DIR, "risk_summary.csv"))
    features = pd.read_csv(os.path.join(BASE_DIR, "feature_importance.csv"))
    return customers, models, segments, risk, risk_summary, features

df, results, segment_profile, risk_analysis, risk_summary, feature_importance = load_data()

# ---------------- SIDEBAR ----------------

st.sidebar.title("📊 Telecom Intelligence")
st.sidebar.write("Customer Analytics & Churn Prediction")

page = st.sidebar.radio(
    "Navigate Dashboard",
    [
        "Overview",
        "Churn Analysis",
        "ML Performance",
        "Customer Segmentation",
        "Revenue Risk"
    ]
)

# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.title("📊 Telecom Customer Intelligence Dashboard")
    st.caption("Churn Prediction | Customer Segmentation | Revenue Risk")

    total_customers = len(df)
    churned = (df["Churn"] == "Yes").sum()
    retained = (df["Churn"] == "No").sum()
    churn_rate = churned / total_customers * 100

    rf_row = results[results["Model"] == "Random Forest"]

    accuracy = (
        rf_row["Accuracy"].iloc[0] * 100
        if not rf_row.empty else 0
    )

    revenue_at_risk = risk_analysis["Expected_Revenue_At_Risk"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Churn Rate", f"{churn_rate:.2f}%")
    c4.metric("RF Accuracy", f"{accuracy:.2f}%")
    c5.metric("Expected Monthly Revenue at Risk", f"{revenue_at_risk:,.2f}")

    st.divider()

    col1, col2 = st.columns(2)

    churn_data = df["Churn"].value_counts().reset_index()
    churn_data.columns = ["Churn Status", "Customers"]

    fig1 = px.pie(
        churn_data,
        names="Churn Status",
        values="Customers",
        title="Customer Churn Distribution",
        hole=0.45
    )
    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(
        df,
        x="Contract",
        color="Churn",
        barmode="group",
        title="Churn by Contract Type"
    )
    col2.plotly_chart(fig2, use_container_width=True)

# ---------------- CHURN ANALYSIS ----------------

elif page == "Churn Analysis":

    st.title("Customer Churn Analysis")

    col1, col2 = st.columns(2)

    fig1 = px.histogram(
        df,
        x="tenure",
        color="Churn",
        nbins=20,
        title="Churn by Customer Tenure"
    )
    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(
        df,
        x="PaymentMethod",
        color="Churn",
        barmode="group",
        title="Churn by Payment Method"
    )
    col2.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(
        df,
        x="InternetService",
        color="Churn",
        barmode="group",
        title="Churn by Internet Service"
    )
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges by Churn Status"
    )
    st.plotly_chart(fig4, use_container_width=True)

# ---------------- ML PERFORMANCE ----------------

elif page == "ML Performance":

    st.title("Machine Learning Model Performance")

    st.subheader("Model Comparison")
    st.dataframe(results, use_container_width=True)

    metric_columns = [
        col for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        if col in results.columns
    ]

    model_chart = results.melt(
        id_vars="Model",
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Score"
    )

    fig1 = px.bar(
        model_chart,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Logistic Regression vs Random Forest"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Important Churn Factors")

    fig2 = px.bar(
        feature_importance.head(15).sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Features Influencing Churn"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- CUSTOMER SEGMENTATION ----------------

elif page == "Customer Segmentation":

    st.title("Customer Segmentation Analysis")

    st.subheader("Customer Segment Profiles")
    st.dataframe(segment_profile, use_container_width=True)

    segment_data = df["CustomerSegment"].value_counts().reset_index()
    segment_data.columns = ["Customer Segment", "Customers"]

    col1, col2 = st.columns(2)

    fig1 = px.pie(
        segment_data,
        names="Customer Segment",
        values="Customers",
        title="Customer Segment Distribution"
    )
    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        df,
        x="tenure",
        y="MonthlyCharges",
        color="CustomerSegment",
        title="Customer Segments by Tenure and Monthly Charges"
    )
    col2.plotly_chart(fig2, use_container_width=True)

# ---------------- REVENUE RISK ----------------

elif page == "Revenue Risk":

    st.title("Revenue at Risk Analysis")

    st.metric(
        "Expected Monthly Revenue at Risk",
        f"{risk_analysis['Expected_Revenue_At_Risk'].sum():,.2f}"
    )

    if "Risk_Level" in risk_analysis.columns:

        risk_counts = risk_analysis["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Customers"]

        col1, col2 = st.columns(2)

        fig1 = px.bar(
            risk_counts,
            x="Risk Level",
            y="Customers",
            color="Risk Level",
            title="Customers by Risk Level"
        )
        col1.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(
            risk_summary,
            x="Risk_Level",
            y="Expected_Revenue_At_Risk",
            color="Risk_Level",
            title="Expected Revenue at Risk by Category"
        )
        col2.plotly_chart(fig2, use_container_width=True)

    st.subheader("Customer Risk Details")

    if "Risk_Level" in risk_analysis.columns:
        selected_risk = st.selectbox(
            "Filter by Risk Level",
            ["All"] + list(risk_analysis["Risk_Level"].dropna().unique())
        )

        filtered = risk_analysis.copy()

        if selected_risk != "All":
            filtered = filtered[
                filtered["Risk_Level"] == selected_risk
            ]

        st.dataframe(filtered, use_container_width=True)
    else:
        st.warning("Risk_Level column was not found in the saved risk data.")
        st.dataframe(risk_analysis, use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("Telecom Customer Intelligence System")
st.sidebar.caption("Academic Hackathon Project")
