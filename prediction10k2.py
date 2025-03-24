import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import numpy as np
import base64
import os

# ✅ Assign Hospital Group (A-H, I-P, Q-Z)
def assign_group(hospital_name):
    if hospital_name and hospital_name[0].upper() <= 'H':
        return 'HospitalAH'
    elif hospital_name[0].upper() <= 'P':
        return 'HospitalIP'
    else:
        return 'HospitalQZ'

# ✅ Load Data (With Hospital Grouping)
@st.cache_data
def load_data(group):
    conn = sqlite3.connect("hospital10_data.db")
    df = pd.read_sql("SELECT * FROM hospital10_data", conn)
    conn.close()
    df["HOSPITAL_GROUP"] = df["HOSPITAL_NAME"].apply(assign_group)
    return df[df["HOSPITAL_GROUP"] == group]

# ✅ User Authentication
def authenticate(username, password):
    valid_creds = {
        "HospitalAH_admin": "HospitalAH@123",
        "HospitalIP_admin": "HospitalIP@123",
        "HospitalQZ_admin": "HospitalQZ@123"
    }
    if username in valid_creds and password == valid_creds[username]:
        return username.replace("_admin", "")
    return None

# ✅ Login Page
def login():
    st.title("🔐 Hospital Admin Login")
    username = st.text_input("🆔 Username (e.g. HospitalAH_admin)", key="login_username")
    password = st.text_input("🔑 Password", type="password", key="login_password")

    if st.button("🚀 Login", key="login_button"):
        group = authenticate(username, password)
        if group:
            st.session_state.logged_in = True
            st.session_state.group = group
            st.session_state.menu = "Dashboard"
            st.success(f"✅ Welcome, {group} Admin!")
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
def dashboard_page(group):
    # Initialize session state variables if they don't exist
    if "selected_state" not in st.session_state:
        st.session_state.selected_state = "All"
    if "selected_city" not in st.session_state:
        st.session_state.selected_city = "All"
    if "selected_hospital" not in st.session_state:
        st.session_state.selected_hospital = "All"

    st.sidebar.header("Filters")
    data = load_data(group)
    if data.empty:
        st.warning("⚠️ No data available for this group.")
        return

    # Filter by State
    states = ["All"] + sorted(data["STATE"].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("🌎 Select State", states, index=states.index(st.session_state.selected_state) if st.session_state.selected_state in states else 0, key="state_select_dashboard")

    # Update the session state for the selected state
    st.session_state.selected_state = selected_state

    # Filter cities based on selected state
    if selected_state != "All":
        cities = ["All"] + sorted(data[data["STATE"] == selected_state]["CITY"].dropna().unique().tolist())
    else:
        cities = ["All"] + sorted(data["CITY"].dropna().unique().tolist())
    
    # Select city filter based on the selected state
    selected_city = st.sidebar.selectbox("🏙️ Select City", cities, index=cities.index(st.session_state.selected_city) if st.session_state.selected_city in cities else 0, key="city_select_dashboard")

    # Update the session state for the selected city
    st.session_state.selected_city = selected_city

    # If a city is selected, show hospitals in that city
    if selected_city != "All":
        hospitals = ["All"] + sorted(data[data["CITY"] == selected_city]["HOSPITAL_NAME"].dropna().unique().tolist())
    else:
        hospitals = ["All"] + sorted(data["HOSPITAL_NAME"].dropna().unique().tolist())

    # Select hospital filter based on selected city or other filters
    selected_hospital = st.sidebar.selectbox("🏥 Select Hospital", hospitals, index=hospitals.index(st.session_state.selected_hospital) if st.session_state.selected_hospital in hospitals else 0, key="hospital_select_dashboard")

    # Update session state for selected hospital
    st.session_state.selected_hospital = selected_hospital

    # Now filter the data based on the selections
    st.title(f"Bed Utilization Dashboard")

    # Date range filter
    start_date, end_date = st.sidebar.date_input(
        "📅 Select Date Range",
        value=[
            st.session_state.get("selected_start_date", datetime(1990, 10, 1)),
            st.session_state.get("selected_end_date", datetime.today())
        ]
    )
    
    # Save date range to session state
    st.session_state.selected_start_date = start_date
    st.session_state.selected_end_date = end_date

    # Filtering the data based on the selected filters
    data["START"] = pd.to_datetime(data["START"]).dt.tz_localize(None)
    filtered_data = data[(data["START"] >= pd.to_datetime(start_date)) & (data["START"] <= pd.to_datetime(end_date))]

    if selected_hospital != "All":
        filtered_data = filtered_data[filtered_data["HOSPITAL_NAME"] == selected_hospital]
    if selected_city != "All":
        filtered_data = filtered_data[filtered_data["CITY"] == selected_city]
    if selected_state != "All":
        filtered_data = filtered_data[filtered_data["STATE"] == selected_state]

    total_patients = filtered_data["ENCOUNTER"].nunique()
    avg_los = round(filtered_data["LOS"].mean(), 2)

    # Display metrics
    col1, col2 = st.columns(2)
    col1.metric("Total Patients", total_patients)
    col2.metric("Average LOS (Days)", avg_los)

    # Outlier detection
    q1 = filtered_data["LOS"].quantile(0.25)
    q3 = filtered_data["LOS"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_data.loc[:, "Outlier"] = filtered_data["LOS"].apply(lambda x: "🔴 Outlier" if x < lower_bound or x > upper_bound else "🟢 Normal")

    # Show filtered data
    report = filtered_data.drop_duplicates(subset=["ENCOUNTER"])[["START", "ENCOUNTER", "PATIENT", "HOSPITAL_NAME", "PROCEDURE_DESCRIPTION", "CITY", "STATE", "LOS", "Outlier"]]

    def highlight_outliers(val):
        return "background-color: red; color: white" if val == "🔴 Outlier" else "background-color: lightgreen; color: black"

    st.dataframe(report.head(2000).style.applymap(highlight_outliers, subset=["Outlier"]))

    # LOS trend over time
    filtered_data['Date'] = filtered_data['START'].dt.date
    los_trend = filtered_data.groupby('Date')["LOS"].mean().reset_index()
    fig = px.bar(los_trend, x='Date', y='LOS', title='Average LOS Over Time', labels={'Date': 'Date', 'LOS': 'Average LOS'}, color='LOS', color_continuous_scale='Viridis')
    st.plotly_chart(fig)

    # Procedure-wise LOS
    procedure_los = filtered_data.groupby("PROCEDURE_DESCRIPTION")["LOS"].mean().reset_index().sort_values(by="LOS", ascending=False)
    fig2 = px.bar(procedure_los, x="LOS", y="PROCEDURE_DESCRIPTION", orientation="h", title="Average LOS per Procedure")
    st.plotly_chart(fig2)

    
# ✅ Recursive Forecasting
def recursive_forecast(model, admissions, X, horizon, pandas_freq, time_cols):
    last_row = X.iloc[-1].tolist()
    predictions = []
    future_dates = [admissions["timestamp"].max() + i * pd.tseries.frequencies.to_offset(pandas_freq) for i in range(1, horizon + 1)]

    for i in range(horizon):
        input_row = np.array(last_row).reshape(1, -1)
        pred = model.predict(input_row)[0]
        predictions.append(pred)

        # Shift lag features
        new_lags = [pred] + last_row[:-len(time_cols)-1 if time_cols else None]

        # Create updated time features for the forecast date
        updated_time = []
        forecast_date = future_dates[i]
        if 'day_of_week' in time_cols:
            updated_time.append(forecast_date.dayofweek)
        if 'is_weekend' in time_cols:
            updated_time.append(int(forecast_date.dayofweek in [5, 6]))
        if 'week' in time_cols:
            updated_time.append(forecast_date.isocalendar().week)
        if 'month' in time_cols:
            updated_time.append(forecast_date.month)
        if 'year' in time_cols:
            updated_time.append(forecast_date.year)

        last_row = new_lags + updated_time
    return future_dates, predictions

def predictions_page(group):
    st.title("\U0001F52E Predictions Page (Random Forest)")
    data = load_data(group)
    if data.empty:
        st.warning("⚠️ No data available for predictions.")
        return

    # ✅ Apply filters from session state
    selected_hospital = st.session_state.get("selected_hospital", "All")
    selected_state = st.session_state.get("selected_state", "All")
    selected_city = st.session_state.get("selected_city", "All")

    frequency = st.radio("\U0001F4C5 Select Forecast Frequency", ["Daily", "Weekly", "Monthly", "Yearly"], index=1)

    filtered_data = data.copy()
    if selected_hospital.lower() != "all":
        filtered_data = filtered_data[filtered_data["HOSPITAL_NAME"].str.lower() == selected_hospital.lower()]
    if selected_city != "All":
        filtered_data = filtered_data[filtered_data["CITY"] == selected_city]
    if selected_state != "All":
        filtered_data = filtered_data[filtered_data["STATE"] == selected_state]

    # ✅ If no data remains after filtering, show a warning and stop execution
    if filtered_data.empty:
        st.warning("⚠️ No data available for the selected filters.")
        return

    st.info(f"✅ Running {frequency} Prediction for {selected_hospital.capitalize()} in {group}...")
    filtered_data["START"] = pd.to_datetime(filtered_data["START"])
    filtered_data.set_index("START", inplace=True)
    filtered_data.index = filtered_data.index.tz_localize(None)  # ✅ Remove timezone info

    start_date = st.session_state.get("selected_start_date", datetime(2000, 1, 1))
    end_date = st.session_state.get("selected_end_date", datetime.today())
    
    # ✅ Filter the data based on the selected date range
    filtered_data = filtered_data[
        (filtered_data.index >= pd.to_datetime(start_date)) &
        (filtered_data.index <= pd.to_datetime(end_date))
    ]

    # Map frequencies to pandas offset alias and horizon (minimum data points required)
    freq_map = {
        "Daily": ("D", 5),
        "Weekly": ("W", 4),
        "Monthly": ("M", 3),
        "Yearly": ("Y", 3)
    }

    pandas_freq, horizon = freq_map[frequency]

    # ✅ Resample the data based on the selected frequency
    admissions = filtered_data.resample(pandas_freq).size().reset_index()
    admissions.columns = ["timestamp", "admissions"]

    # ✅ Only keep records with admissions (i.e., filter out zero-admission periods)
    admissions = admissions[admissions["admissions"] > 0]

    # ✅ Check if enough data points are available for prediction
    if len(admissions) < horizon:
        st.warning(f"⚠️ Only {len(admissions)} valid {frequency.lower()} data points found. Need at least {horizon} to predict.")
        return

    # ✅ Create lag features for predictions
    max_lag = min(7, len(admissions) - 1)
    for lag in range(1, max_lag + 1):
        admissions[f"lag_{lag}"] = admissions["admissions"].shift(lag)


    time_cols = []
    if frequency == "Daily":
        admissions["day_of_week"] = admissions["timestamp"].dt.dayofweek
        admissions["is_weekend"] = admissions["day_of_week"].isin([5, 6]).astype(int)
        time_cols = ["day_of_week", "is_weekend"]
    elif frequency == "Weekly":
        admissions["week"] = admissions["timestamp"].dt.isocalendar().week
        admissions["month"] = admissions["timestamp"].dt.month
        admissions["year"] = admissions["timestamp"].dt.year
        time_cols = ["week", "month", "year"]
    elif frequency == "Monthly":
        admissions["month"] = admissions["timestamp"].dt.month
        admissions["year"] = admissions["timestamp"].dt.year
        time_cols = ["month", "year"]
    elif frequency == "Yearly":
        admissions["year"] = admissions["timestamp"].dt.year
        time_cols = ["year"]

    admissions.dropna(inplace=True)
    if len(admissions) < 1:
        st.warning("⚠️ Not enough data after feature generation.")
        return

    features = [col for col in admissions.columns if col.startswith("lag_")] + time_cols
    X = admissions[features]
    y = admissions["admissions"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    future_dates, predictions = recursive_forecast(model, admissions, X, horizon, pandas_freq, time_cols)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=admissions.tail(horizon)["timestamp"], y=admissions.tail(horizon)["admissions"], mode="lines+markers", name=f"Past {frequency}"))
    fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode="lines+markers", name=f"Predicted ({frequency})", line=dict(dash="dot", color="red")))
    fig.update_layout(title=f"\U0001F4CA {frequency} Bed Demand Forecast - {selected_hospital.capitalize()} in {group}", xaxis_title=frequency, yaxis_title="Number of Patients")
    st.plotly_chart(fig, use_container_width=True)

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
            st.session_state.clear()
            st.rerun()

# ✅ Initialize Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "group" not in st.session_state:
    st.session_state.group = None
if "menu" not in st.session_state:
    st.session_state.menu = "Login"

# ✅ App Navigation Logic
if not st.session_state.logged_in:
    login()
else:
    top_navbar()
    group = st.session_state.group
    if st.session_state.menu == "Settings":
        settings_page()
    elif st.session_state.menu == "Predictions":
        predictions_page(group)
    else:
        dashboard_page(group)