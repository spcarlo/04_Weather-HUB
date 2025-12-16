import requests
import pandas as pd
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
    return location, days_back, location_placeholder, map_placeholder



# -------------------------------
# Data helpers
# -------------------------------
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

def map_df(loc: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {"lat": [loc["latitude"]], "lon": [loc["longitude"]]}
    )


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
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "hourly": "snow_depth",
            "timezone": timezone,
        },
        timeout=30,
    )
    wx.raise_for_status()
    j = wx.json()

    df_daily = pd.DataFrame({
        "date": j["daily"]["time"],
        "tmax": j["daily"]["temperature_2m_max"],
        "tmin": j["daily"]["temperature_2m_min"],
        "precip": j["daily"]["precipitation_sum"],
    })

    df_hourly = pd.DataFrame({
        "time": j["hourly"]["time"],
        "snow_depth": j["hourly"]["snow_depth"],
    })
    df_hourly["time"] = pd.to_datetime(df_hourly["time"])
    df_hourly["date"] = df_hourly["time"].dt.date.astype(str)

    snow_daily = df_hourly.groupby("date", as_index=False).agg(
        snow_depth=("snow_depth", "max")
    )

    return df_daily.merge(snow_daily, on="date", how="left")



@st.cache_data(ttl=24 * 60 * 60)
def load_location(name: str, v: int = 1) -> dict:
    try:
        return get_location(name)
    except Exception:
        return None

@st.cache_data(ttl=60 * 60)
def load_daily(lat: float, lon: float, days_back: int, timezone: str, v: int = 1) -> pd.DataFrame:
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
def apply_layout(fig, df: pd.DataFrame, y_title: str, t: int):
    fig.update_layout(
        margin=dict(l=10, r=10, t=t, b=10),
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title=y_title,
        showlegend=False,
    )

    fig.update_xaxes(
        showgrid=True,
        range=[df["date"].min(), df["date"].max()],
        tickmode="linear",
        dtick="M1",
        tickformat="%b %Y",
    )
    fig.update_yaxes(showgrid=True)

def plot_daily(df: pd.DataFrame, view: str):
    fig = go.Figure()

    if view == "Min / Max":
        fig.add_trace(go.Scatter(x=df["date"], y=df["tmax"], mode="lines"))
        fig.add_trace(go.Scatter(x=df["date"], y=df["tmin"], mode="lines"))
    else:
        fig.add_trace(go.Scatter(x=df["date"], y=df["tavg"], mode="lines"))

    apply_layout(fig, df, "°C", t=20)
    st.plotly_chart(fig, width="stretch")



def bin_time_series(df: pd.DataFrame, bins: int = 30) -> pd.DataFrame:
    d = df[["date", "precip"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    n = min(bins, len(d))
    d["bin"] = pd.cut(d["date"], bins=n, include_lowest=True)
    return d.groupby("bin", observed=True, as_index=False).agg(
        date=("date", "min"),
        precip=("precip", "sum"),
        )

def plot_precip(df: pd.DataFrame):
    agg = bin_time_series(df, bins=30)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["date"], y=agg["precip"], opacity=0.7))

    apply_layout(fig, df, "mm", t=10)
    st.plotly_chart(fig, width="stretch")

def plot_snow_depth(df: pd.DataFrame):
    if "snow_depth" not in df.columns or df["snow_depth"].isna().all():
        st.caption("No snow depth data")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["snow_depth"], mode="lines"))

    apply_layout(fig, df, "cm", t=10)
    st.plotly_chart(fig, width="stretch")




# -------------------------------
# Prep
# -------------------------------
def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["range"] = df["tmax"] - df["tmin"]
    df["tavg"] = (df["tmax"] + df["tmin"]) / 2
    return df


# -------------------------------
# Run
# -------------------------------
def main():
    location, days_back, location_placeholder, map_placeholder = render_controls()

    loc = load_location(location, v=2)

    if loc is None:
        location_placeholder.error("Location not found")
        return

    location_placeholder.caption(format_location(loc))
    map_placeholder.map(map_df(loc), zoom=9, height=180)

    df = load_daily(loc["latitude"], loc["longitude"], days_back, TIMEZONE, v=2)
    df = prepare_df(df)

    show_metrics(df)

    temp_view = st.radio(
        "temperature view",
        ["Min / Max", "Average"],
        horizontal=True,
        label_visibility="collapsed",
    )

    plot_daily(df, temp_view)
    plot_precip(df)
    plot_snow_depth(df)



if __name__ == "__main__":
    main()
