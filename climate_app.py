import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from datetime import date, timedelta

import plotly.graph_objects as go


# -------------------------------
# Page
# -------------------------------
st.title("Weather")
st.caption("local weather analytics")


TIMEZONE = "Europe/Zurich"


def render_controls():
    location = st.text_input("Location", value="Zürich")
    location_placeholder = st.empty()
    map_placeholder = st.empty()

    days_back = st.slider("Days back", min_value=5, max_value=400, value=60)

    current_year = date.today().year
    year_options = list(range(current_year, current_year - 80, -1))
    year = st.selectbox("Compare year", year_options, index=0)


    return location, days_back, year, map_placeholder, location_placeholder


# -------------------------------
# Data helpers
# -------------------------------
@st.cache_data(ttl=24 * 60 * 60)
def get_location(name: str) -> dict:
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": name, "count": 1, "language": "de", "format": "json"},
        timeout=30,
    )
    geo.raise_for_status()
    return geo.json()["results"][0]


def format_location(loc: dict) -> str:
    parts = [
        loc.get("name"),
        loc.get("admin1"),
        loc.get("country"),
    ]
    label = ", ".join(p for p in parts if p)

    elevation = loc.get("elevation")
    if elevation is not None:
        label = f"{label} | {int(elevation)} m ASL"

    return label


def location_df(loc: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat": [loc["latitude"]],
            "lon": [loc["longitude"]],
        }
    )


def year_window(days_back: int, year: int) -> tuple[date, date]:
    today = date.today()
    try:
        end = date(year, today.month, today.day)
    except ValueError:
        # handles Feb 29 on non-leap years
        end = date(year, today.month, 28)
    start = end - timedelta(days=days_back)
    return start, end


def fetch_daily(lat: float, lon: float, start: date, end: date, timezone: str) -> pd.DataFrame:
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
def load_data(name: str, days_back: int, year: int, timezone: str):
    loc = get_location(name)
    start, end = year_window(days_back, year)
    df = fetch_daily(loc["latitude"], loc["longitude"], start, end, timezone)
    return df, loc


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
def plot_daily(df_now: pd.DataFrame, df_year: pd.DataFrame | None, year: int):
    fig = go.Figure()

    now_year = df_now["date"].dt.year.iloc[-1]

    colors = {
        "max": "#AEC7E8",
        "min": "#1F77B4",
    }

    def add_line(x, y, name, color, dash):
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            line=dict(color=color, dash=dash),
        ))

    add_line(df_now["date"], df_now["tmax"], f"Max {now_year}", colors["max"], "solid")
    add_line(df_now["date"], df_now["tmin"], f"Min {now_year}", colors["min"], "solid")

    if df_year is not None:
        df_year = df_year.sort_values("date").reset_index(drop=True)
        df_now = df_now.sort_values("date").reset_index(drop=True)

        add_line(df_now["date"], df_year["tmax"], f"Max {year}", colors["max"], "dot")
        add_line(df_now["date"], df_year["tmin"], f"Min {year}", colors["min"], "dot")

    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode="x unified",
        yaxis_title="°C",
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    st.plotly_chart(fig, width="stretch")





# -------------------------------
# Prep
# -------------------------------
def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["range"] = df["tmax"] - df["tmin"]
    return df


# -------------------------------
# Run
# -------------------------------
def main():
    location, days_back, compare_year, map_placeholder, location_placeholder = render_controls()
    current_year = date.today().year

    df_now, loc = load_data(location, days_back, current_year, TIMEZONE)
    df_now = prepare_df(df_now)

    df_year = None
    if compare_year != current_year:
        df_year, _ = load_data(location, days_back, compare_year, TIMEZONE)
        df_year = prepare_df(df_year)

    location_placeholder.caption(format_location(loc))
    map_placeholder.map(location_df(loc), zoom=9, height=180)

    show_metrics(df_now)

    plot_daily(df_now, df_year, compare_year)


if __name__ == "__main__":
    main()