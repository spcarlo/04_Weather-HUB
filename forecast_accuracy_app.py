import time

import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from forecast_accuracy_calculations import (
    actual_daily_temperature_fields,
    actual_daily_temperature_frame,
    build_daily_pred,
    compute_metrics,
    previous_run_hourly_temperature_frame,
    previous_run_temperature_column,
    score_daily_forecast,
)


# -------------------------------
# Page
# -------------------------------
st.set_page_config(
    page_title="Forecast Accuracy",
    layout="centered")

st.title("Forecast Accuracy")
st.caption("compare previous forecasts to actuals (Open Meteo previous runs vs archive)")

TIMEZONE = "Europe/Zurich"
API_RETRIES = 3
API_RETRY_DELAY_SECONDS = 1
HORIZON_DAYS = range(1, 8)
TEMP_VIEWS = {
    "Mean": "mean",
    "Min": "min",
    "Max": "max",
}

# -------------------------------
# Controls
# -------------------------------
def render_controls() -> tuple[str, int, int, str]:
    with st.sidebar:
        location = st.text_input("Location", value="Zürich")
        horizon_days = st.slider(
            "Forecast horizon (days ahead)",
            min_value=min(HORIZON_DAYS),
            max_value=max(HORIZON_DAYS),
            value=3,
        )
        past_days = st.slider("Verify past days", min_value=7, max_value=90, value=30)
        temp_view = st.selectbox("Temperature view", options=list(TEMP_VIEWS))
    return location, horizon_days, past_days, TEMP_VIEWS[temp_view]


# -------------------------------
# Location helpers
# -------------------------------
def api_get(url: str, params: dict) -> dict:
    last_error = None

    for attempt in range(API_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_error = e
            if attempt < API_RETRIES - 1:
                time.sleep(API_RETRY_DELAY_SECONDS * (attempt + 1))

    if last_error is not None:
        raise last_error

    raise RuntimeError("API request failed")


def get_location(name: str) -> dict:
    j = api_get(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": name, "count": 1, "language": "de", "format": "json"},
    )
    return j["results"][0]


@st.cache_data(ttl=24 * 60 * 60)
def load_location(name: str):
    try:
        return get_location(name)
    except Exception:
        return None


def get_location_cached_by_name(name: str):
    if st.session_state.get("location_name") != name:
        st.session_state.location_name = name
        st.session_state.location_data = load_location(name)
    return st.session_state.location_data


def format_location(loc: dict) -> str:
    parts = [loc.get("name"), loc.get("admin1"), loc.get("country")]
    label = ", ".join(p for p in parts if p)

    elevation = loc.get("elevation")
    if elevation is not None:
        label = f"{label} | {int(elevation)} m ASL"

    return label


def map_df(loc: dict) -> pd.DataFrame:
    return pd.DataFrame({"lat": [loc["latitude"]], "lon": [loc["longitude"]]})


def render_location_header(location: str):
    loc = get_location_cached_by_name(location)
    if loc is None:
        st.error("Location not found")
        return None

    st.caption(format_location(loc))
    st.map(map_df(loc), zoom=9, height=180)
    return loc


def fetch_previous_runs_temp(lat: float, lon: float, past_days: int, timezone: str) -> dict:
    pred_cols = [previous_run_temperature_column(horizon_days) for horizon_days in HORIZON_DAYS]
    j = api_get(
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        {
            "latitude": lat,
            "longitude": lon,
            "timezone": timezone,
            "past_days": past_days,
            "forecast_days": 0,
            "hourly": ",".join(pred_cols),
        },
    )
    return j["hourly"]


def build_daily_pred_for_all_horizons(hourly_data: dict) -> pd.DataFrame:
    daily_frames = []

    for horizon_days in HORIZON_DAYS:
        hourly_pred = previous_run_hourly_temperature_frame(hourly_data, horizon_days)
        pred_daily = build_daily_pred(hourly_pred)
        pred_daily["horizon_days"] = horizon_days
        daily_frames.append(pred_daily)

    return pd.concat(daily_frames, ignore_index=True)


def fetch_actual_daily_temp(
    lat: float,
    lon: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    timezone: str,
) -> pd.DataFrame:
    j = api_get(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
            "daily": actual_daily_temperature_fields(),
            "timezone": timezone,
        },
    )
    return actual_daily_temperature_frame(j["daily"])

@st.cache_data(ttl=60 * 60)
def load_scored_max_window(lat: float, lon: float, timezone: str) -> pd.DataFrame:
    max_days = 90

    hourly_data = fetch_previous_runs_temp(lat, lon, max_days, timezone)
    pred_daily = build_daily_pred_for_all_horizons(hourly_data)

    start = pred_daily["date"].min()
    end = pred_daily["date"].max()

    actual_daily = fetch_actual_daily_temp(lat, lon, start, end, timezone)
    scored = score_daily_forecast(pred_daily, actual_daily)

    return scored.sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=60 * 60)
def load_future_daily_temp(lat: float, lon: float, days: int, timezone: str) -> pd.DataFrame:
    j = api_get(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": lat,
            "longitude": lon,
            "daily": actual_daily_temperature_fields(),
            "forecast_days": days,
            "timezone": timezone,
        },
    )
    d = j.get("daily", {})
    if not d:
        return empty_future_daily_temp()

    return pd.DataFrame(
        {
            "date": pd.to_datetime(d["time"]),
            "temp_future_mean": d["temperature_2m_mean"],
            "temp_future_min": d["temperature_2m_min"],
            "temp_future_max": d["temperature_2m_max"],
        }
    )


def empty_future_daily_temp() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "temp_future_mean", "temp_future_min", "temp_future_max"])


def load_scored_for_location(loc: dict, horizon_days: int, past_days: int, timezone: str) -> pd.DataFrame:
    scored_all = load_scored_max_window(loc["latitude"], loc["longitude"], timezone)
    scored_horizon = scored_all[scored_all["horizon_days"] == horizon_days]

    end = scored_horizon["date"].max()
    start = end - pd.Timedelta(days=past_days - 1)

    return scored_horizon[scored_horizon["date"] >= start].reset_index(drop=True)



# -------------------------------
# Plots
# -------------------------------
def apply_layout(fig, x_min, x_max, y_title: str, t: int) -> None:
    fig.update_layout(
        margin=dict(l=10, r=10, t=t, b=10),
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title=y_title,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        range=[x_min, x_max],
        tickformat="%d.%m",
    )

    # dynamic 10-degree steps
    all_y = []
    for trace in fig.data:
        if hasattr(trace, "y") and trace.y is not None:
            all_y.extend(trace.y)

    y_values = pd.to_numeric(pd.Series(all_y), errors="coerce").dropna()
    if y_values.empty:
        return

    y_min = y_values.min()
    y_max = y_values.max()

    y_floor = int((y_min // 10) * 10)
    y_ceil = int(((y_max + 9) // 10) * 10)

    fig.update_yaxes(
        showgrid=True,
        range=[y_floor, y_ceil],
        tick0=y_floor,
        dtick=2,
    )



def x_range(scored: pd.DataFrame, future: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    x_min = scored["date"].min()
    x_max = scored["date"].max()
    if not future.empty:
        x_min = min(x_min, future["date"].min())
        x_max = max(x_max, future["date"].max())
    return x_min, x_max


def plot_pred_vs_actual(scored: pd.DataFrame, future: pd.DataFrame, temp_metric: str) -> None:
    pred_col = f"temp_pred_{temp_metric}"
    actual_col = f"temp_actual_{temp_metric}"
    future_col = f"temp_future_{temp_metric}"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=scored["date"],
            y=scored[pred_col],
            mode="lines",
            name="old forecast",
            line=dict(color="#F4A261", shape="spline"),
            hovertemplate="%{y:.1f} °C<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=scored["date"],
            y=scored[actual_col],
            mode="lines",
            name="actual",
            line=dict(color="#4CAF50", shape="spline"),
        )
    )


    if not future.empty:
        last_row = scored.iloc[-1]

        future_plot = pd.concat(
            [
                pd.DataFrame(
                    {
                        "date": [last_row["date"]],
                        future_col: [last_row[actual_col]],
                    }
                ),
                future,
            ],
            ignore_index=True,
        )
        
        future_plot = future_plot.drop_duplicates(subset=["date"], keep="first")

        fig.add_trace(
            go.Scatter(
                x=future_plot["date"],
                y=future_plot[future_col],
                mode="lines",
                name="forecast",
                line=dict(color="#5B8DB8", shape="spline", dash="dot"),
            )
        )

    x_min, x_max = x_range(scored, future)
    apply_layout(fig, x_min, x_max, "°C", t=10)

    st.plotly_chart(fig, width="stretch")


# -------------------------------
# UI
# -------------------------------
def format_metric_value(value: float | None) -> str:
    if value is None:
        return "n/a"

    return f"{value:.2f}"


def render_metrics(scored: pd.DataFrame) -> None:
    m = compute_metrics(scored)
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean MAE (°C)", format_metric_value(m["mean_mae"]))
    c2.metric("Min MAE (°C)", format_metric_value(m["min_mae"]))
    c3.metric("Max MAE (°C)", format_metric_value(m["max_mae"]))


def render_data(scored: pd.DataFrame) -> None:
    with st.expander("Data"):
        visible_cols = [
            "date",
            "temp_pred_mean",
            "temp_actual_mean",
            "temp_error_mean",
            "temp_pred_min",
            "temp_actual_min",
            "temp_error_min",
            "temp_pred_max",
            "temp_actual_max",
            "temp_error_max",
        ]
        st.dataframe(scored[visible_cols], width="stretch", hide_index=True)


# -------------------------------
# Run
# -------------------------------
def main() -> None:
    location, horizon_days, past_days, temp_metric = render_controls()

    loc = render_location_header(location)
    if loc is None:
        return

    load_status = st.empty()

    with load_status.status("Loading forecast data...", expanded=False) as status:
        status.write("Fetching all forecast horizons once, then filtering locally.")

        try:
            scored = load_scored_for_location(loc, horizon_days, past_days, TIMEZONE)
        except requests.RequestException as e:
            status.update(label="Forecast accuracy data unavailable", state="error")
            st.error("Could not load forecast accuracy data. Please try again in a moment.")
            st.caption(str(e))
            return

        if scored.empty:
            status.update(label="No forecast accuracy data returned", state="error")
            st.warning("No data returned for this selection.")
            return

        status.write("Loading the current forecast line for the selected temperature view.")

        try:
            future = load_future_daily_temp(loc["latitude"], loc["longitude"], days=7, timezone=TIMEZONE)
        except requests.RequestException:
            st.warning("Future forecast is temporarily unavailable.")
            future = empty_future_daily_temp()

        status.update(label="Forecast data ready", state="complete")

    load_status.empty()

    render_metrics(scored)
    plot_pred_vs_actual(scored, future=future, temp_metric=temp_metric)
    render_data(scored)


if __name__ == "__main__":
    main()

