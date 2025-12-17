import requests
import pandas as pd
import streamlit as st
from datetime import date
import plotly.graph_objects as go


# -------------------------------
# Page
# -------------------------------
st.title("Forecast Accuracy")
st.caption("compare previous forecasts to actuals (Open Meteo previous runs vs archive)")

TIMEZONE = "Europe/Zurich"


# -------------------------------
# Controls
# -------------------------------
def render_controls():
    with st.sidebar:
        location = st.text_input("Location", value="Zürich")
        horizon_days = st.slider("Forecast horizon (days ahead)", min_value=1, max_value=7, value=3)
        past_days = st.slider("Verify past days", min_value=7, max_value=90, value=30)
    return location, horizon_days, past_days


# -------------------------------
# Location helpers
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


@st.cache_data(ttl=24 * 60 * 60)
def load_location(name: str):
    try:
        return get_location(name)
    except Exception:
        return None


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


# -------------------------------
# Data helpers
# -------------------------------
def previous_runs_get(params: dict) -> dict:
    r = requests.get(
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        params=params,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def archive_get(params: dict) -> dict:
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def hourly_to_daily_tmin_tmax(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    d = df[["time", value_col]].copy()
    d["date"] = d["time"].dt.date
    out = d.groupby("date", as_index=False).agg(
        tmin=(value_col, "min"),
        tmax=(value_col, "max"),
    )
    out["date"] = pd.to_datetime(out["date"])
    return out


def fetch_previous_runs_temp(lat: float, lon: float, horizon_days: int, past_days: int, timezone: str) -> pd.DataFrame:
    pred_col = f"temperature_2m_previous_day{horizon_days}"

    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "past_days": past_days,
        "forecast_days": 0,
        "hourly": pred_col,
    }

    j = previous_runs_get(params)
    h = j["hourly"]

    return pd.DataFrame(
        {
            "time": pd.to_datetime(h["time"]),
            "temp_pred": h[pred_col],
        }
    )


def build_daily_pred(hourly_pred: pd.DataFrame) -> pd.DataFrame:
    return hourly_to_daily_tmin_tmax(hourly_pred, "temp_pred").rename(
        columns={"tmin": "tmin_pred", "tmax": "tmax_pred"}
    )


def fetch_actual_daily_tmin_tmax(
    lat: float,
    lon: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    timezone: str,
) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": timezone,
    }

    j = archive_get(params)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(j["daily"]["time"]),
            "tmin_actual": j["daily"]["temperature_2m_min"],
            "tmax_actual": j["daily"]["temperature_2m_max"],
        }
    )


def score_daily_forecast(pred_daily: pd.DataFrame, actual_daily: pd.DataFrame) -> pd.DataFrame:
    df = pred_daily.merge(actual_daily, on="date", how="inner")
    df["tmin_error"] = df["tmin_pred"] - df["tmin_actual"]
    df["tmax_error"] = df["tmax_pred"] - df["tmax_actual"]
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    return {
        "tmin_mae": float(df["tmin_error"].abs().mean()),
        "tmax_mae": float(df["tmax_error"].abs().mean()),
        "tmin_bias": float(df["tmin_error"].mean()),
        "tmax_bias": float(df["tmax_error"].mean()),
    }


@st.cache_data(ttl=60 * 60)
def load_scored(lat: float, lon: float, horizon_days: int, past_days: int, timezone: str) -> pd.DataFrame:
    hourly_pred = fetch_previous_runs_temp(lat, lon, horizon_days, past_days, timezone)
    pred_daily = build_daily_pred(hourly_pred)

    start = pred_daily["date"].min()
    end = pred_daily["date"].max()

    actual_daily = fetch_actual_daily_tmin_tmax(lat, lon, start, end, timezone)
    scored = score_daily_forecast(pred_daily, actual_daily)

    return scored.sort_values("date").reset_index(drop=True)


# -------------------------------
# Plots
# -------------------------------
def apply_layout(fig, x_min, x_max, y_title: str, t: int):
    fig.update_layout(
        margin=dict(l=10, r=10, t=t, b=10),
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title=y_title,
        showlegend=True,
    )
    fig.update_xaxes(showgrid=True, range=[x_min, x_max])
    fig.update_yaxes(showgrid=True)


def plot_errors(scored: pd.DataFrame, which: str):
    fig = go.Figure()
    if which == "Tmin error":
        fig.add_trace(go.Scatter(x=scored["date"], y=scored["tmin_error"], mode="lines+markers", name="tmin_error"))
        apply_layout(fig, scored["date"].min(), scored["date"].max(), "°C", t=10)
    else:
        fig.add_trace(go.Scatter(x=scored["date"], y=scored["tmax_error"], mode="lines+markers", name="tmax_error"))
        apply_layout(fig, scored["date"].min(), scored["date"].max(), "°C", t=10)

    st.plotly_chart(fig, width="stretch")


# -------------------------------
# Run
# -------------------------------
def main():
    location, horizon_days, past_days = render_controls()

    loc = render_location_header(location)
    if loc is None:
        return

    scored = load_scored(loc["latitude"], loc["longitude"], horizon_days, past_days, TIMEZONE)
    if scored.empty:
        st.warning("No data returned for this selection.")
        return

    m = compute_metrics(scored)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tmin MAE (°C)", f"{m['tmin_mae']:.2f}")
    c2.metric("Tmax MAE (°C)", f"{m['tmax_mae']:.2f}")
    c3.metric("Tmin bias (°C)", f"{m['tmin_bias']:.2f}")
    c4.metric("Tmax bias (°C)", f"{m['tmax_bias']:.2f}")

    view = st.radio("View", ["Tmin error", "Tmax error"], horizontal=True, label_visibility="collapsed")
    plot_errors(scored, view)

    with st.expander("Data"):
        st.dataframe(scored, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()

