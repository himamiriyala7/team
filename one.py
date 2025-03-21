import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import numpy as np
import base64
import os

# ✅ Load Data (Filtered by City & Case-Insensitive)
@st.cache_data
def load_data(city):
    conn = sqlite3.connect("hospital_data.db")
    query = f"SELECT * FROM hospital_data WHERE LOWER(CITY) = LOWER('{city}')"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ✅ Convert Image to Base64
def get_base64(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ✅ User Authentication
def authenticate(username, password):
    if username.endswith("_admin"):
        city_name = username.replace("_admin", "").lower()
        expected_password = f"{city_name}@123"
        if password == expected_password:
            return city_name.capitalize()
    return None

# ✅ Login Page
def login():
    st.title("🔐 City Admin Login")
    username = st.text_input("🆔 Username (City)_admin")
    password = st.text_input("🔑 Password", type="password")

    if st.button("🚀 Login"):
        city = authenticate(username, password)
        if city:
            st.session_state.logged_in = True
            st.session_state.city = city
            st.session_state.menu = "Dashboard"
            st.success(f"✅ Welcome, {city} Admin!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password. Please try again.")

# ✅ Settings Page
def settings_page():
    st.title("⚙️ Settings")
    theme_options = ["Light", "Dark", "Blue", "Green", "High Contrast"]
    selected_theme = st.selectbox("Dashboard Theme", options=theme_options)
    st.session_state.theme = selected_theme
    st.success(f"✅ Theme updated to {selected_theme}")

# ✅ Dashboard Page
def dashboard_page(city):
    st.sidebar.header("Filters")

    # ✅ Load and validate data
    data = load_data(city)
    if data.empty:
        st.warning("⚠️ No data available for this city.")
        return

    # ✅ Get Unique Hospital List
    hospital_list = sorted(data["HOSPITAL_NAME"].dropna().str.lower().unique().tolist())
    hospitals = ["All"] + hospital_list  # Add "All" option at the top

    # ✅ Ensure session state exists
    if "selected_hospital" not in st.session_state:
        st.session_state.selected_hospital = "All"

    # ✅ Selectbox for hospital filtering
    selected_hospital = st.sidebar.selectbox(
        "🏥 Select Hospital",
        hospitals,
        index=hospitals.index(st.session_state.selected_hospital) if st.session_state.selected_hospital in hospitals else 0,
        key="hospital_select_dashboard"
    )

    # ✅ Update session state and refresh if selection changes
    if selected_hospital != st.session_state.selected_hospital:
        st.session_state.selected_hospital = selected_hospital
        st.rerun()

    # ✅ Display title after selected_hospital is defined
    st.title(f"{selected_hospital.capitalize()} in {city} - Bed Utilization Dashboard")

    # ✅ Additional Filters (State, City, Procedure)
    states = ["All"] + sorted(data["STATE"].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("🌎 Select State", states, key="state_select_dashboard")

    if selected_state != "All":
        cities = ["All"] + sorted(data[data["STATE"] == selected_state]["CITY"].dropna().unique().tolist())
    else:
        cities = ["All"] + sorted(data["CITY"].dropna().unique().tolist())

    selected_city = st.sidebar.selectbox("🏙️ Select City", cities, key="city_select_dashboard")

    procedures = ["All"] + sorted(data["PROCEDURE_DESCRIPTION"].dropna().unique().tolist())
    selected_procedure = st.sidebar.selectbox("🩺 Select Procedure", procedures, key="procedure_select_dashboard")

    start_date, end_date = st.sidebar.date_input("📅 Select Date Range",
                                                 value=[datetime(2000, 1, 1), datetime.today()])

    # ✅ Apply Filters to Data
    data["START"] = pd.to_datetime(data["START"]).dt.tz_localize(None)
    filtered_data = data[(data["START"] >= pd.to_datetime(start_date)) & 
                         (data["START"] <= pd.to_datetime(end_date))]

    if selected_hospital.lower() != "all":
        filtered_data = filtered_data[filtered_data["HOSPITAL_NAME"].str.lower() == selected_hospital]
    if selected_city != "All":
        filtered_data = filtered_data[filtered_data["CITY"] == selected_city]
    if selected_state != "All":
        filtered_data = filtered_data[filtered_data["STATE"] == selected_state]
    if selected_procedure != "All":
        filtered_data = filtered_data[filtered_data["PROCEDURE_DESCRIPTION"] == selected_procedure]

    # ✅ Display Dashboard Metrics
    st.metric("Total Patients", len(filtered_data))
    st.metric("Average LOS (Days)", round(filtered_data["LOS"].mean(), 2) if not filtered_data.empty else "N/A")

    # ✅ LOS Trend Graph
    if not filtered_data.empty:
        los_trend = filtered_data.groupby(filtered_data["START"].dt.date)["LOS"].mean().reset_index()
        fig1 = px.line(los_trend, x="START", y="LOS", labels={"START": "Date", "LOS": "Avg LOS"})
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("⚠️ No data available for this selection.")

# ✅ Predictions Page using Random Forest with Weekly & Daily Aggregation
def predictions_page(city):
    st.title("🔮 Predictions Page (Random Forest)")
    data = load_data(city)
    if data.empty:
        st.warning("⚠️ No data available for predictions.")
        return

    st.sidebar.header("Filters")

    # ✅ Filters
    hospitals = ["All"] + sorted(data["HOSPITAL_NAME"].str.lower().unique().tolist())
    selected_hospital = st.sidebar.selectbox("🏥 Select Hospital", hospitals, key="hospital_select_predictions")

    states = ["All"] + sorted(data["STATE"].unique().tolist())
    selected_state = st.sidebar.selectbox("🌎 Select State", states, key="state_select_predictions")

    cities = ["All"] + sorted(data[data["STATE"] == selected_state]["CITY"].unique().tolist()) if selected_state != "All" else ["All"]
    selected_city = st.sidebar.selectbox("🏙️ Select City", cities, key="city_select_predictions")

    # ✅ Filter Data Based on Selection
    filtered_data = data.copy()

    if selected_hospital.lower() != "all":
        filtered_data = filtered_data[filtered_data["HOSPITAL_NAME"].str.lower() == selected_hospital]
    if selected_city != "All":
        filtered_data = filtered_data[filtered_data["CITY"] == selected_city]
    if selected_state != "All":
        filtered_data = filtered_data[filtered_data["STATE"] == selected_state]

    if filtered_data.empty:
        st.warning(f"⚠️ No data available for {selected_hospital if selected_hospital != 'All' else city}.")
        return

    st.info(f"✅ Running Prediction for {selected_hospital.capitalize()} in {city}...")

    # ✅ Data Preprocessing (Weekly & Daily Aggregation)
    filtered_data["START"] = pd.to_datetime(filtered_data["START"])
    filtered_data.set_index("START", inplace=True)

    weekly_admissions = filtered_data.resample("W").size().reset_index()
    weekly_admissions.columns = ["week", "admissions"]

    daily_admissions = filtered_data.resample("D").size().reset_index()
    daily_admissions.columns = ["date", "admissions"]

    if len(weekly_admissions) < 5 or len(daily_admissions) < 0:
        st.warning("⚠️ Not enough data for prediction.")
        return

    # ✅ Create lag features (past 4 weeks & past 7 days)
    for lag in range(1, 5):
        weekly_admissions[f"lag_{lag}"] = weekly_admissions["admissions"].shift(lag)
    for lag in range(1, 8):
        daily_admissions[f"lag_{lag}"] = daily_admissions["admissions"].shift(lag)

    weekly_admissions.dropna(inplace=True)
    daily_admissions.dropna(inplace=True)

    # ✅ Train Random Forest Model (Weekly)
    X_weekly = weekly_admissions.drop(columns=["week", "admissions"])
    y_weekly = weekly_admissions["admissions"]

    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_weekly, y_weekly, test_size=0.2, shuffle=False)
    model_weekly = RandomForestRegressor(n_estimators=100, random_state=42)
    model_weekly.fit(X_train_w, y_train_w)

    # ✅ Train Random Forest Model (Daily)
    X_daily = daily_admissions.drop(columns=["date", "admissions"])
    y_daily = daily_admissions["admissions"]

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_daily, y_daily, test_size=0.2, shuffle=False)
    model_daily = RandomForestRegressor(n_estimators=100, random_state=42)
    model_daily.fit(X_train_d, y_train_d)

    # ✅ Get Past 10 Weeks & Future Predictions (Weekly)
    past_10_weeks = weekly_admissions.tail(10)
    future_weeks = [weekly_admissions["week"].max() + timedelta(weeks=i) for i in range(1, 11)]
    future_preds_weekly = model_weekly.predict(X_weekly.iloc[-1:].values.repeat(10, axis=0))

    # ✅ Get Past 30 Days & Future Predictions (Daily)
    past_30_days = daily_admissions.tail(30)
    future_days = [daily_admissions["date"].max() + timedelta(days=i) for i in range(1, 31)]
    future_preds_daily = model_daily.predict(X_daily.iloc[-1:].values.repeat(30, axis=0))

    # ✅ Plot Weekly Prediction
    st.subheader("📊 Weekly Bed Demand Prediction")
    fig_weekly = go.Figure()
    fig_weekly.add_trace(go.Scatter(x=past_10_weeks["week"], y=past_10_weeks["admissions"], mode='lines', name="Past 10 Weeks"))
    fig_weekly.add_trace(go.Scatter(x=future_weeks, y=future_preds_weekly, mode='lines', name="Predicted (Next 10 Weeks)", line=dict(color='red', dash="dot")))
    fig_weekly.update_layout(title=f"📊 Weekly Forecast - {selected_hospital.capitalize()} in {city}",
                             xaxis_title="Week", yaxis_title="Number of Patients")
    st.plotly_chart(fig_weekly, use_container_width=True)

    # ✅ Plot Daily Prediction
    st.subheader("📊 Daily Bed Demand Prediction")
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(x=past_30_days["date"], y=past_30_days["admissions"], mode='lines', name="Past 30 Days"))
    fig_daily.add_trace(go.Scatter(x=future_days, y=future_preds_daily, mode='lines', name="Predicted (Next 30 Days)", line=dict(color='blue', dash="dot")))
    fig_daily.update_layout(title=f"📊 Daily Forecast - {selected_hospital.capitalize()} in {city}",
                            xaxis_title="Date", yaxis_title="Number of Patients")
    st.plotly_chart(fig_daily, use_container_width=True)


# ✅ Navigation Bar
def top_navbar():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📊 Dashboard"):
            st.session_state.menu = "Dashboard"
            st.rerun()
    with col2:
        if st.button("⚙️ Settings"):
            st.session_state.menu = "Settings"
            st.rerun()
    with col3:
        if st.button("🔮 Predictions"):
            st.session_state.menu = "Predictions"
            st.rerun()
    with col4:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.city = None
            st.session_state.menu = "Login"
            st.rerun()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "city" not in st.session_state:
    st.session_state.city = None
if "menu" not in st.session_state:
    st.session_state.menu = "Login"

if not st.session_state.logged_in:
    login()
else:
    top_navbar()
    city = st.session_state.city
    if st.session_state.menu == "Settings":
        settings_page()
    elif st.session_state.menu == "Predictions":
        predictions_page(city)
    else:
        dashboard_page(city)