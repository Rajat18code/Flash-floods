import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Flash Flood AI",
    page_icon="🌊",
    layout="wide"
)


# ---------------- TITLE ----------------

st.title("🌊Flash Flood ")
st.caption("Flash Flood Risk Monitoring for Hilly Regions")

st.divider()


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/himachal_features.csv")


df = load_data()


# ---------------- SIDEBAR ----------------

st.sidebar.header("⚙️ Monitoring Controls")

selected_month = st.sidebar.selectbox(
    "Select Month",
    sorted(df["MO"].unique())
)

filtered_df = df[df["MO"] == selected_month]


# ---------------- METRICS ----------------

st.subheader("📊 Current Weather Overview")

col1, col2, col3, col4 = st.columns(4)

avg_rain = filtered_df["PRECTOTCORR"].mean()
avg_temp = filtered_df["T2M"].mean()
avg_humidity = filtered_df["RH2M"].mean()
avg_wind = filtered_df["WS2M"].mean()

col1.metric("🌧️ Avg Rainfall", f"{avg_rain:.2f}")
col2.metric("🌡️ Temperature", f"{avg_temp:.2f} °C")
col3.metric("💧 Humidity", f"{avg_humidity:.1f} %")
col4.metric("💨 Wind Speed", f"{avg_wind:.2f} m/s")


st.divider()


# ---------------- CHARTS ----------------

left, right = st.columns(2)


with left:

    st.subheader("🌧️ Rainfall Trend")

    fig_rain = px.line(
        filtered_df,
        x="DY",
        y="PRECTOTCORR",
        title="Daily Precipitation"
    )

    st.plotly_chart(
        fig_rain,
        use_container_width=True
    )


with right:

    st.subheader("🌡️ Temperature Trend")

    fig_temp = px.line(
        filtered_df,
        x="DY",
        y="T2M",
        title="Daily Temperature"
    )

    st.plotly_chart(
        fig_temp,
        use_container_width=True
    )


# ---------------- RISK ASSESSMENT ----------------

st.divider()

st.subheader("⚠️ Flash Flood Risk Assessment")

recent_rain = filtered_df["RAIN_INTENSITY"].mean()

if recent_rain > 10:
    risk = "HIGH"
    risk_value = 85
elif recent_rain > 5:
    risk = "MODERATE"
    risk_value = 55
else:
    risk = "LOW"
    recent_rain = filtered_df["RAIN_INTENSITY"].mean()
avg_rain_3day = filtered_df["RAIN_3DAY"].mean()
avg_humidity = filtered_df["TEMP_HUMIDITY"].mean()
avg_wind_rain = filtered_df["WIND_RAIN"].mean()

rain_score = min(recent_rain * 5, 40)
rain_3day_score = min(avg_rain_3day * 2, 30)
humidity_score = min(avg_humidity * 0.2, 20)
wind_score = min(avg_wind_rain * 2, 10)

risk_value = min(
    rain_score +
    rain_3day_score +
    humidity_score +
    wind_score,
    100
)

if risk_value < 30:
    risk = "LOW"
elif risk_value < 60:
    risk = "MODERATE"
elif risk_value < 80:
    risk = "HIGH"
else:
    risk = "SEVERE"


gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=risk_value,
        title={"text": f"Current Risk: {risk}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red"},
        },
    )
)

st.plotly_chart(
    gauge,
    use_container_width=True
)


# ---------------- DATA PREVIEW ----------------

st.divider()

st.subheader("🔎 Processed Data Preview")

st.dataframe(
    filtered_df.tail(10),
    use_container_width=True
)