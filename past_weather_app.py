import requests
import pandas as pd
import streamlit as st
from datetime import date, timedelta
import plotly.graph_objects as go


# -------------------------------
# Page
# -------------------------------
st.set_page_config(
    page_title="Past Weather",
    layout="centered")

st.title("Weather")
st.caption("local weather analytics")

TIMEZONE = "Europe/Zurich"
TEMP_VIEWS = ("Min / Max", "Average")
PRECIP_VIEWS = ("Rain", "Snow")


# -------------------------------
# Controls
# -------------------------------
def render_controls():
    with st.sidebar:
        location = st.text_input("Location", value="Zürich")
        days_back = st.slider("Days back", min_value=5, max_value=90, value=14)
    return location, days_back

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
    parts = [loc.get("name"), loc.get("admin1"), loc.get("country")]
    label = ", ".join(p for p in parts if p)

    elevation = loc.get("elevation")
    if elevation is not None:
        label = f"{label} | {int(elevation)} m ASL"

    return label

def map_df(loc: dict) -> pd.DataFrame:
    return pd.DataFrame({"lat": [loc["latitude"]], "lon": [loc["longitude"]]})


def archive_get(params: dict) -> dict:
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def daily_df_from_json(j: dict) -> pd.DataFrame:
    d = j["daily"]
    return pd.DataFrame(
        {
            "date": d["time"],
            "tmax": d["temperature_2m_max"],
            "tmin": d["temperature_2m_min"],
            "precip": d["precipitation_sum"],
            "rain": d["rain_sum"],
            "snowfall": d["snowfall_sum"],
        }
    )


def snow_depth_daily_df(j: dict) -> pd.DataFrame | None:
    hourly = (j.get("hourly") or {})
    times = hourly.get("time")
    depths = hourly.get("snow_depth")
    if times is None or depths is None:
        return None

    h = pd.DataFrame({"time": times, "snow_depth": depths})
    h["date"] = pd.to_datetime(h["time"]).dt.date

    out = h.groupby("date", as_index=False).agg(snow_depth=("snow_depth", "max"))
    out["date"] = out["date"].astype(str)
    return out


def fetch_daily(lat: float, lon: float, days_back: int, timezone: str) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=days_back)

    base = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": timezone,
    }

    params_daily = {
        **base,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum",
    }
    out = daily_df_from_json(archive_get(params_daily))

    params_snow = {
        **base,
        "hourly": "snow_depth",
    }
    snow_daily = snow_depth_daily_df(archive_get(params_snow))

    if snow_daily is None:
        out["snow_depth"] = pd.NA
    else:
        out = out.merge(snow_daily, on="date", how="left")

    out["snow_depth"] = out["snow_depth"] * 100
    return out


@st.cache_data(ttl=24 * 60 * 60)
def load_location(name: str):
    try:
        return get_location(name)
    except Exception:
        return None


@st.cache_data(ttl=60 * 60)
def load_daily(lat: float, lon: float, days_back: int, timezone: str):
    return fetch_daily(lat, lon, days_back, timezone)


# -------------------------------
# UI blocks
# -------------------------------
def render_location_header(location: str):
    loc_ph = st.empty()
    map_ph = st.empty()

    loc = load_location(location)
    if loc is None:
        loc_ph.error("Location not found")
        return None

    loc_ph.caption(format_location(loc))
    map_ph.map(map_df(loc), zoom=9, height=180)
    return loc


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["tavg"] = (df["tmax"] + df["tmin"]) / 2
    return df


def load_weather_df(loc: dict, days_back: int) -> pd.DataFrame:
    df = load_daily(loc["latitude"], loc["longitude"], days_back, TIMEZONE)
    return prepare_df(df)


def show_metrics(df: pd.DataFrame):
    col1, col2, col3 = st.columns(3)
    col1.metric("Min temp (°C)", f"{df['tmin'].min():.1f}")
    col2.metric("Max temp (°C)", f"{df['tmax'].max():.1f}")
    col3.metric("Avg precip (mm/day)", f"{df['precip'].mean():.1f}")


# -------------------------------
# Plots
# -------------------------------
def apply_layout(fig, x_min, x_max, y_title: str, t: int):
    fig.update_layout(
        margin=dict(l=10, r=10, t=t, b=10),
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title=y_title,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, range=[x_min, x_max])
    fig.update_yaxes(showgrid=True)


def plot_daily(df: pd.DataFrame, view: str):
    fig = go.Figure()

    if view == "Min / Max":
        fig.add_trace(go.Scatter(x=df["date"], y=df["tmax"], mode="lines"))
        fig.add_trace(go.Scatter(x=df["date"], y=df["tmin"], mode="lines"))
    else:
        fig.add_trace(go.Scatter(x=df["date"], y=df["tavg"], mode="lines"))

    apply_layout(fig, df["date"].min(), df["date"].max(), "°C", t=20)
    st.plotly_chart(fig, width="stretch")


def bin_time_series(df: pd.DataFrame, bins: int = 30) -> pd.DataFrame:
    d = df[["date", "rain", "snowfall", "snow_depth"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    n = min(bins, len(d))
    d["bin"] = pd.cut(d["date"], bins=n, include_lowest=True)

    return d.groupby("bin", observed=True, as_index=False).agg(
        date=("date", "min"),
        rain=("rain", "sum"),
        snowfall=("snowfall", "sum"),
        snow_depth=("snow_depth", "max"),
    )


def plot_precip(df: pd.DataFrame, view: str):
    agg = bin_time_series(df, bins=30)
    fig = go.Figure()

    if view == "Rain":
        fig.add_trace(go.Bar(x=agg["date"], y=agg["rain"], name="Rain (mm)", opacity=0.7))
        apply_layout(fig, df["date"].min(), df["date"].max(), "mm", t=10)
        st.plotly_chart(fig, width="stretch")
        return

    fig.add_trace(go.Bar(x=agg["date"], y=agg["snowfall"], name="Snowfall (cm)", opacity=0.7))
    fig.add_trace(
        go.Scatter(
            x=agg["date"],
            y=agg["snow_depth"],
            mode="lines",
            name="Snow depth (cm)",
            line=dict(color="white", width=3, shape="spline"),
        )
    )

    apply_layout(fig, df["date"].min(), df["date"].max(), "cm", t=10)
    st.plotly_chart(fig, width="stretch")


def render_charts(df: pd.DataFrame):
    temp_view = st.radio(
        "temperature view",
        list(TEMP_VIEWS),
        horizontal=True,
        label_visibility="collapsed",
    )
    plot_daily(df, temp_view)

    precip_view = st.radio(
        "precip view",
        list(PRECIP_VIEWS),
        horizontal=True,
        label_visibility="collapsed",
    )
    plot_precip(df, precip_view)


# -------------------------------
# Run
# -------------------------------
def main():
    location, days_back = render_controls()

    loc = render_location_header(location)
    if loc is None:
        return

    df = load_weather_df(loc, days_back)

    show_metrics(df)
    render_charts(df)


if __name__ == "__main__":
    main()

