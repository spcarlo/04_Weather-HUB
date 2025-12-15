import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from datetime import date, timedelta

# -------------------------------
# Page
# -------------------------------
st.title("Weather")
st.caption("local weather analytics")

LOCATION_NAME = st.text_input("Location", value="Zürich")
DAYS_BACK = st.slider("Days back", min_value=5, max_value=400, value=30)
TIMEZONE = "Europe/Zurich"

sns.set_theme(style="whitegrid")

# -------------------------------
# Data helpers
# -------------------------------
def get_coords(name: str) -> tuple[float, float]:
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": name, "count": 1, "language": "de", "format": "json"},
        timeout=30,
    )
    geo.raise_for_status()
    top = geo.json()["results"][0]
    return top["latitude"], top["longitude"]


def fetch_daily(lat: float, lon: float, days_back: int, timezone: str) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=days_back)

    wx = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": timezone,
        },
        timeout=30,
    )
    wx.raise_for_status()
    j = wx.json()

    return pd.DataFrame({
        "date": j["daily"]["time"],
        "tmax": j["daily"]["temperature_2m_max"],
        "tmin": j["daily"]["temperature_2m_min"],
    })


@st.cache_data(ttl=60 * 60)
def load_data(name: str, days_back: int, timezone: str) -> pd.DataFrame:
    lat, lon = get_coords(name)
    return fetch_daily(lat, lon, days_back, timezone)

# -------------------------------
# Metrics
# -------------------------------
def show_metrics(df: pd.DataFrame):
    col1, col2, col3 = st.columns(3)

    col1.metric("Min temp (°C)", f"{df['tmin'].min():.1f}")
    col2.metric("Max temp (°C)", f"{df['tmax'].max():.1f}")
    col3.metric("Avg daily range (°C)", f"{(df['tmax'] - df['tmin']).mean():.1f}")

# -------------------------------
# Plots
# -------------------------------
def plot_daily(df: pd.DataFrame):
    fig, ax = plt.subplots()

    sns.lineplot(data=df, x="date", y="tmax", label="Max temp", ax=ax)
    sns.lineplot(data=df, x="date", y="tmin", label="Min temp", ax=ax)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    ax.set_xlabel("")
    ax.set_ylabel("°C")
    sns.despine(ax=ax)

    st.pyplot(fig)


def plot_daily_range(df: pd.DataFrame):
    fig, ax = plt.subplots()

    sns.lineplot(data=df, x="date", y="range", label="Daily range", ax=ax)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    ax.set_xlabel("")
    ax.set_ylabel("°C")
    sns.despine(ax=ax)

    st.pyplot(fig)

# -------------------------------
# Run
# -------------------------------
if st.button("Refresh data"):
    st.cache_data.clear()

df = load_data(LOCATION_NAME, DAYS_BACK, TIMEZONE)
df["date"] = pd.to_datetime(df["date"])
df["range"] = df["tmax"] - df["tmin"]

show_metrics(df)
plot_daily(df)
plot_daily_range(df)

