import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# -------------------------------
# Page
# -------------------------------
st.set_page_config(
    page_title="Forecast Accuracy",
    layout="centered")

st.title("Forecast Accuracy")
st.caption("compare previous forecasts to actuals (Open Meteo previous runs vs archive)")

TIMEZONE = "Europe/Zurich"

# -------------------------------
# Controls
# -------------------------------
def render_controls() -> tuple[str, int, int]:
    with st.sidebar:
        location = st.text_input("Location", value="Zürich")
        horizon_days = st.slider("Forecast horizon (days ahead)", min_value=1, max_value=7, value=3)
        past_days = st.slider("Verify past days", min_value=7, max_value=90, value=30)
    return location, horizon_days, past_days


# -------------------------------
# Location helpers
# -------------------------------
def api_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


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


# -------------------------------
# Data helpers
# -------------------------------
def hourly_to_daily_mean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    d = df[["time", value_col]].copy()
    d["date"] = d["time"].dt.date
    out = d.groupby("date", as_index=False).agg(temp_mean=(value_col, "mean"))
    out["date"] = pd.to_datetime(out["date"])
    return out


def fetch_previous_runs_temp(lat: float, lon: float, horizon_days: int, past_days: int, timezone: str) -> pd.DataFrame:
    pred_col = f"temperature_2m_previous_day{horizon_days}"
    j = api_get(
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        {
            "latitude": lat,
            "longitude": lon,
            "timezone": timezone,
            "past_days": past_days,
            "forecast_days": 0,
            "hourly": pred_col,
        },
    )
    h = j["hourly"]
    return pd.DataFrame({"time": pd.to_datetime(h["time"]), "temp_pred": h[pred_col]})


def build_daily_pred(hourly_pred: pd.DataFrame) -> pd.DataFrame:
    return hourly_to_daily_mean(hourly_pred, "temp_pred").rename(columns={"temp_mean": "temp_pred"})


def fetch_actual_daily_mean(
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
            "daily": "temperature_2m_mean",
            "timezone": timezone,
        },
    )
    return pd.DataFrame(
        {
            "date": pd.to_datetime(j["daily"]["time"]),
            "temp_actual": j["daily"]["temperature_2m_mean"],
        }
    )


def score_daily_forecast(pred_daily: pd.DataFrame, actual_daily: pd.DataFrame) -> pd.DataFrame:
    df = pred_daily.merge(actual_daily, on="date", how="inner")
    df["temp_error"] = df["temp_pred"] - df["temp_actual"]
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    return {"mae": float(df["temp_error"].abs().mean()), "bias": float(df["temp_error"].mean())}


@st.cache_data(ttl=60 * 60)
def load_scored(lat: float, lon: float, horizon_days: int, past_days: int, timezone: str) -> pd.DataFrame:
    hourly_pred = fetch_previous_runs_temp(lat, lon, horizon_days, past_days, timezone)
    pred_daily = build_daily_pred(hourly_pred)

    start = pred_daily["date"].min()
    end = pred_daily["date"].max()

    actual_daily = fetch_actual_daily_mean(lat, lon, start, end, timezone)
    scored = score_daily_forecast(pred_daily, actual_daily)
    return scored.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=60 * 60)
def load_future_daily_mean(lat: float, lon: float, days: int, timezone: str) -> pd.DataFrame:
    j = api_get(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_mean",
            "forecast_days": days,
            "timezone": timezone,
        },
    )
    d = j.get("daily", {})
    if not d:
        return pd.DataFrame(columns=["date", "temp_future"])
    return pd.DataFrame({"date": pd.to_datetime(d["time"]), "temp_future": d["temperature_2m_mean"]})


def load_scored_for_location(loc: dict, horizon_days: int, past_days: int, timezone: str) -> pd.DataFrame:
    return load_scored(loc["latitude"], loc["longitude"], horizon_days, past_days, timezone)


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

    y_min = min(all_y)
    y_max = max(all_y)

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


def plot_pred_vs_actual(scored: pd.DataFrame, future: pd.DataFrame) -> None:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=scored["date"],
            y=scored["temp_pred"],
            mode="lines",
            name="old forecast",
            line=dict(color="#F4A261", shape="spline"),
            hovertemplate="%{y:.1f} °C<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=scored["date"],
            y=scored["temp_actual"],
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
                        "temp_future": [last_row["temp_actual"]],
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
                y=future_plot["temp_future"],
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
def render_metrics(scored: pd.DataFrame) -> None:
    m = compute_metrics(scored)
    c1, c2 = st.columns(2)
    c1.metric("Temp MAE (°C)", f"{m['mae']:.2f}")
    c2.metric("Temp bias (°C)", f"{m['bias']:.2f}")


def render_data(scored: pd.DataFrame) -> None:
    with st.expander("Data"):
        st.dataframe(scored, width="stretch", hide_index=True)


# -------------------------------
# Run
# -------------------------------
def main() -> None:
    location, horizon_days, past_days = render_controls()

    loc = render_location_header(location)
    if loc is None:
        return

    scored = load_scored_for_location(loc, horizon_days, past_days, TIMEZONE)
    if scored.empty:
        st.warning("No data returned for this selection.")
        return

    future = load_future_daily_mean(loc["latitude"], loc["longitude"], days=7, timezone=TIMEZONE)

    render_metrics(scored)
    plot_pred_vs_actual(scored, future=future)
    render_data(scored)


if __name__ == "__main__":
    main()

