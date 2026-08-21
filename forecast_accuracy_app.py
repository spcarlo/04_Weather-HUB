import html
import math
import time

import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
    layout="centered",
    initial_sidebar_state="collapsed",
)


def apply_page_styles() -> None:
    # Keep title, metrics, and captions close in size so type doesn't shout
    st.markdown(
        """
        <style>
        h1 {
            font-size: 1.65rem !important;
            font-weight: 600 !important;
        }
        [data-testid="stHeaderActionElements"] {
            display: none !important;
        }
        [data-testid="stCaptionContainer"] {
            font-size: 0.95rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.95rem !important;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 0.95rem !important;
        }

        /* Keep sidebar + menu buttons; only remove the opaque bar over the title */
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stAppToolbar"] {
            background: transparent !important;
            backdrop-filter: none !important;
            box-shadow: none !important;
            border-bottom: none !important;
        }
        [data-testid="stDecoration"] {
            display: none !important;
        }

        /* Default Streamlit top padding is 6rem; keep a tight gap */
        .block-container,
        [data-testid="stMainBlockContainer"] {
            padding-top: 1.5rem !important;
        }

        /* Sidebar chevron: keep Streamlit's own show/hide, only restyle the icon */
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapseButton"] {
            align-items: center !important;
            width: auto !important;
            overflow: visible !important;
        }
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapseButton"] button {
            color: var(--text-color) !important;
            background: transparent !important;
            border: none !important;
            min-width: 2.1rem !important;
            min-height: 2.1rem !important;
            padding: 0 !important;
        }
        [data-testid="stExpandSidebarButton"] svg,
        [data-testid="stSidebarCollapseButton"] svg {
            width: 1.55rem !important;
            height: 1.55rem !important;
            padding: 0.22rem !important;
            color: var(--text-color) !important;
            fill: var(--text-color) !important;
            stroke: var(--text-color) !important;
            opacity: 1 !important;
            background: rgba(128, 128, 128, 0.22) !important;
            border: 1px solid rgba(128, 128, 128, 0.45) !important;
            border-radius: 0.45rem !important;
            box-sizing: content-box !important;
        }
        .location-row {
            display: flex;
            align-items: center;
            position: relative;
            min-height: 2.1rem;
            margin: 0.1rem 0 0.35rem 0;
            overflow: visible;
        }
        .sidebar-slot {
            display: inline-flex;
            align-items: center;
            position: absolute;
            right: calc(100% + 0.85rem);
            top: 50%;
            transform: translateY(-50%);
            width: 2.2rem;
            height: 2.2rem;
        }
        .location-label {
            font-size: 0.95rem;
            opacity: 0.6;
        }
        [data-testid="stCustomComponentV1"] {
            height: 0 !important;
            width: 0 !important;
            position: absolute !important;
            overflow: hidden !important;
        }

        /* Phones: native hamburger + enough top gap so the title is not covered */
        @media (max-width: 767px) {
            .block-container,
            [data-testid="stMainBlockContainer"] {
                padding-top: 4.5rem !important;
            }
            .sidebar-slot {
                display: none !important;
            }
            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="stAppToolbar"] {
                background: var(--background-color) !important;
                backdrop-filter: none !important;
                box-shadow: none !important;
                border-bottom: none !important;
            }

            /* Keep Mean / Min / Max MAE in one compact row */
            [data-testid="stHorizontalBlock"] {
                flex-direction: row !important;
                flex-wrap: nowrap !important;
                gap: 0.35rem !important;
            }
            [data-testid="stHorizontalBlock"] > div,
            [data-testid="column"] {
                min-width: 0 !important;
                flex: 1 1 0 !important;
            }
            [data-testid="stMetric"] {
                padding: 0 !important;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.05rem !important;
            }
            [data-testid="stMetricLabel"],
            [data-testid="stMetricLabel"] p {
                font-size: 0.7rem !important;
                line-height: 1.15 !important;
            }

            /* 30% slimmer than Streamlit 1.62's 300px default (210px).
               Only override width so collapsed max-width: 0 still hides it. */
            section[data-testid="stSidebar"],
            section.stSidebar,
            [data-testid="stSidebar"] {
                width: 210px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_page_styles()

st.title("Forecast Accuracy")
st.caption("Compare past forecasts with measured weather")

TIMEZONE = "Europe/Zurich"
API_RETRIES = 3
API_RETRY_DELAY_SECONDS = 1
HORIZON_DAYS = range(1, 8)
MAX_PAST_DAYS = 90
PLOT_VIEWS = {
    "Min/max": "range",
    "Mean": "mean",
}
CHART_HEIGHT = 315  # 30% shorter than Plotly's default 450
PLOT_CONFIG = {
    "displayModeBar": False,
    "staticPlot": True,
}

# -------------------------------
# Controls
# -------------------------------
def render_location_control() -> str:
    with st.sidebar:
        return st.text_input("Location", value="Zürich")


def render_forecast_controls() -> tuple[int, int, str, bool]:
    with st.sidebar:
        horizon_days = st.slider(
            "Forecast horizon",
            min_value=min(HORIZON_DAYS),
            max_value=max(HORIZON_DAYS),
            value=3,
        )
        past_days = st.slider("Past days", min_value=7, max_value=MAX_PAST_DAYS, value=30)
        plot_view = st.selectbox("Plot view", options=list(PLOT_VIEWS))
        show_future = st.toggle("Show future", value=False)
    return horizon_days, past_days, PLOT_VIEWS[plot_view], show_future


# -------------------------------
# Data helpers
# -------------------------------
def api_get(url: str, params: dict) -> dict:
    last_error = None

    # Open-Meteo can be briefly unavailable, so retry a few times
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
    loc = load_location(location)
    if loc is None:
        st.error("Location not found")
        return None

    st.markdown(
        f'<div class="location-row">'
        f'<span class="location-label">{html.escape(format_location(loc))}</span>'
        f'<span class="sidebar-slot"></span>'
        f"</div>",
        unsafe_allow_html=True,
    )
    place_sidebar_toggle()
    st.map(map_df(loc), zoom=9, height=180)
    return loc


def place_sidebar_toggle() -> None:
    # Park the open-arrow over the slot next to Zürich. Do not touch the close button.
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            let placing = false;

            function collapseSidebarOnFirstLoad() {
                if (window.parent.__hubDidInitialCollapse) return;
                const expand = doc.querySelector("[data-testid='stExpandSidebarButton']");
                if (expand) {
                    window.parent.__hubDidInitialCollapse = true;
                    return;
                }
                const collapse =
                    doc.querySelector("[data-testid='stSidebarCollapseButton'] button") ||
                    doc.querySelector("[data-testid='stSidebarCollapseButton']");
                if (!collapse) return;
                window.parent.__hubDidInitialCollapse = true;
                collapse.click();
            }

            function resetExpand(el) {
                el.style.position = "";
                el.style.left = "";
                el.style.top = "";
                el.style.zIndex = "";
                el.style.display = "";
                el.style.visibility = "";
                el.style.opacity = "";
                el.style.margin = "";
            }

            function parkExpand() {
                const expandButtons = [...doc.querySelectorAll("[data-testid='stExpandSidebarButton']")];

                // Phones: leave Streamlit's native header hamburger in place
                if (window.parent.innerWidth < 768) {
                    expandButtons.forEach(resetExpand);
                    return;
                }

                const slot = doc.querySelector(".sidebar-slot");
                const active = expandButtons.length ? expandButtons[expandButtons.length - 1] : null;
                expandButtons.forEach((el) => {
                    if (el !== active) el.style.display = "none";
                });
                if (!slot || !active) return;

                const r = slot.getBoundingClientRect();
                active.style.position = "fixed";
                active.style.left = Math.max(8, r.left) + "px";
                active.style.top = r.top + Math.max(0, (r.height - 34) / 2) + "px";
                active.style.zIndex = "1002";
                active.style.display = "inline-flex";
                active.style.visibility = "visible";
                active.style.opacity = "1";
                active.style.margin = "0";
            }

            function place() {
                if (placing) return;
                placing = true;
                try {
                    collapseSidebarOnFirstLoad();
                    parkExpand();
                } finally {
                    placing = false;
                }
            }

            place();
            window.parent.__hubParkExpand = parkExpand;
            if (!window.parent.__hubSidebarToggleListeners) {
                window.parent.__hubSidebarToggleListeners = true;
                window.parent.addEventListener("resize", function() {
                    if (window.parent.__hubParkExpand) window.parent.__hubParkExpand();
                });
                window.parent.addEventListener("scroll", function() {
                    if (window.parent.__hubParkExpand) window.parent.__hubParkExpand();
                }, true);
            }
            if (window.parent.__hubSidebarToggleObserver) {
                window.parent.__hubSidebarToggleObserver.disconnect();
            }
            window.parent.__hubSidebarToggleObserver = new MutationObserver(place);
            window.parent.__hubSidebarToggleObserver.observe(doc.body, {
                childList: true,
                subtree: true,
            });
        })();
        </script>
        """,
        height=0,
    )


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


def empty_future_daily_temp() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "temp_future_mean", "temp_future_min", "temp_future_max"])


@st.cache_data(ttl=60 * 60)
def load_scored_max_window(lat: float, lon: float, timezone: str) -> pd.DataFrame:
    # Fetch the full window once, then filter horizon and days locally
    hourly_data = fetch_previous_runs_temp(lat, lon, MAX_PAST_DAYS, timezone)
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


def location_data_key(loc: dict, timezone: str) -> tuple:
    return (loc["latitude"], loc["longitude"], timezone)


def filter_scored(scored_all: pd.DataFrame, horizon_days: int, past_days: int) -> pd.DataFrame:
    scored_horizon = scored_all[scored_all["horizon_days"] == horizon_days]
    if scored_horizon.empty:
        return scored_horizon.reset_index(drop=True)

    end = scored_horizon["date"].max()
    start = end - pd.Timedelta(days=past_days - 1)  # inclusive window

    return scored_horizon[scored_horizon["date"] >= start].reset_index(drop=True)


# -------------------------------
# Plots
# -------------------------------
def apply_layout(fig, x_min, x_max, top_margin: int, y_values_source=None) -> None:
    fig.update_layout(
        margin=dict(l=10, r=10, t=top_margin, b=10),
        hovermode=False,
        dragmode=False,
        xaxis_title=None,
        yaxis_title=None,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            itemclick=False,
            itemdoubleclick=False,
        ),
    )
    fig.update_traces(hoverinfo="skip", hovertemplate=None)

    fig.update_xaxes(
        showgrid=True,
        range=[x_min, x_max],
        tickformat="%d.%m",
        fixedrange=True,
    )

    # Snap ticks to 5 °C. Min/max values keep mean and range views on the same scale.
    if y_values_source is None:
        all_y = []
        for trace in fig.data:
            if hasattr(trace, "y") and trace.y is not None:
                all_y.extend(trace.y)
    else:
        all_y = y_values_source

    y_values = pd.to_numeric(pd.Series(all_y), errors="coerce").dropna()
    if y_values.empty:
        return

    y_min = y_values.min()
    y_max = y_values.max()

    y_step = 5
    y_padding = 0.5
    y_floor = int(math.floor((y_min - y_padding) / y_step) * y_step)
    y_ceil = int(math.ceil((y_max + y_padding) / y_step) * y_step)

    if y_floor == y_ceil:
        y_ceil = y_floor + y_step

    fig.update_yaxes(
        showgrid=True,
        range=[y_floor, y_ceil],
        tick0=y_floor,
        dtick=y_step,
        fixedrange=True,
    )


def x_range(scored: pd.DataFrame, future: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    x_min = scored["date"].min()
    x_max = scored["date"].max()
    if not future.empty:
        x_min = min(x_min, future["date"].min())
        x_max = max(x_max, future["date"].max())
    return x_min, x_max


def temperature_axis_values(scored: pd.DataFrame, future: pd.DataFrame) -> list:
    values = []

    for col in ["temp_actual_min", "temp_actual_max", "temp_pred_min", "temp_pred_max"]:
        values.extend(scored[col])

    if not future.empty:
        for col in ["temp_future_min", "temp_future_max"]:
            values.extend(future[col])

    return values


def connect_future_to_last_actual(scored: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    # Start the future line on the last measured day so the plot connects
    last_row = scored.iloc[-1]
    bridge = pd.DataFrame(
        {
            "date": [last_row["date"]],
            "temp_future_mean": [last_row["temp_actual_mean"]],
            "temp_future_min": [last_row["temp_actual_min"]],
            "temp_future_max": [last_row["temp_actual_max"]],
        }
    )
    connected = pd.concat([bridge, future], ignore_index=True)
    return connected.drop_duplicates(subset=["date"], keep="first")


def add_temperature_range_lines(
    fig,
    df: pd.DataFrame,
    min_col: str,
    max_col: str,
    name: str,
    color: str,
    dash: str | None = None,
) -> None:
    line = dict(color=color, shape="spline", width=2)
    if dash is not None:
        line["dash"] = dash

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[min_col],
            mode="lines",
            name=name,
            legendgroup=name,
            showlegend=True,
            line=line,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[max_col],
            mode="lines",
            name=name,
            legendgroup=name,
            showlegend=False,
            line=line,
            hoverinfo="skip",
        )
    )


def add_temperature_line(
    fig,
    df: pd.DataFrame,
    value_col: str,
    name: str,
    color: str,
    dash: str | None = None,
) -> None:
    line = dict(color=color, shape="spline", width=2)
    if dash is not None:
        line["dash"] = dash

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[value_col],
            mode="lines",
            name=name,
            line=line,
            hoverinfo="skip",
        )
    )


def plot_mean_forecast(scored: pd.DataFrame, future: pd.DataFrame) -> None:
    fig = go.Figure()

    add_temperature_line(fig, scored, "temp_pred_mean", "forecast", "#F4A261")
    add_temperature_line(fig, scored, "temp_actual_mean", "measured", "#4CAF50")

    if not future.empty:
        future_plot = connect_future_to_last_actual(scored, future)
        add_temperature_line(fig, future_plot, "temp_future_mean", "future", "#5B8DB8", dash="dot")

    x_min, x_max = x_range(scored, future)
    apply_layout(fig, x_min, x_max, top_margin=10, y_values_source=temperature_axis_values(scored, future))

    st.plotly_chart(fig, width="stretch", height=CHART_HEIGHT, config=PLOT_CONFIG)


def plot_min_max_forecast(scored: pd.DataFrame, future: pd.DataFrame) -> None:
    fig = go.Figure()

    add_temperature_range_lines(
        fig,
        scored,
        min_col="temp_actual_min",
        max_col="temp_actual_max",
        name="measured",
        color="#4CAF50",
    )
    add_temperature_range_lines(
        fig,
        scored,
        min_col="temp_pred_min",
        max_col="temp_pred_max",
        name="forecast",
        color="#F4A261",
    )

    if not future.empty:
        future_plot = connect_future_to_last_actual(scored, future)
        add_temperature_range_lines(
            fig,
            future_plot,
            min_col="temp_future_min",
            max_col="temp_future_max",
            name="future",
            color="#5B8DB8",
            dash="dot",
        )

    x_min, x_max = x_range(scored, future)
    apply_layout(fig, x_min, x_max, top_margin=10, y_values_source=temperature_axis_values(scored, future))

    st.plotly_chart(fig, width="stretch", height=CHART_HEIGHT, config=PLOT_CONFIG)


def plot_pred_vs_actual(scored: pd.DataFrame, future: pd.DataFrame, plot_view: str) -> None:
    if plot_view == "mean":
        plot_mean_forecast(scored, future)
        return

    plot_min_max_forecast(scored, future)


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


def get_cached_location_data(data_key: tuple):
    cached = st.session_state.get("forecast_accuracy_data")
    if cached is None or cached.get("key") != data_key:
        return None

    return cached["scored_all"], cached["future"]


def set_cached_location_data(data_key: tuple, scored_all: pd.DataFrame, future: pd.DataFrame) -> None:
    st.session_state.forecast_accuracy_data = {
        "key": data_key,
        "scored_all": scored_all,
        "future": future,
    }


def load_location_forecast_data(loc: dict, timezone: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_all = load_scored_max_window(loc["latitude"], loc["longitude"], timezone)

    try:
        future = load_future_daily_temp(loc["latitude"], loc["longitude"], days=7, timezone=timezone)
    except requests.RequestException:
        st.warning("Future forecast is temporarily unavailable.")
        future = empty_future_daily_temp()

    return scored_all, future


def ensure_location_forecast_data(loc: dict, timezone: str):
    # Keep the full window in session so slider changes only filter locally
    data_key = location_data_key(loc, timezone)
    cached = get_cached_location_data(data_key)
    if cached is not None:
        return cached

    try:
        with st.spinner("Loading forecast data..."):
            scored_all, future = load_location_forecast_data(loc, timezone)
    except requests.RequestException as e:
        st.error("Could not load forecast accuracy data. Please try again in a moment.")
        st.caption(str(e))
        return None

    if scored_all.empty:
        st.warning("No data returned for this selection.")
        return None

    set_cached_location_data(data_key, scored_all, future)
    return scored_all, future


@st.fragment
def render_forecast_panel(loc: dict) -> None:
    horizon_days, past_days, plot_view, show_future = render_forecast_controls()
    data = ensure_location_forecast_data(loc, TIMEZONE)
    if data is None:
        return

    scored_all, future = data
    scored = filter_scored(scored_all, horizon_days, past_days)
    if scored.empty:
        st.warning("No data returned for this selection.")
        return

    plot_future = future if show_future else empty_future_daily_temp()
    render_metrics(scored)
    plot_pred_vs_actual(scored, future=plot_future, plot_view=plot_view)
    render_data(scored)


# -------------------------------
# Run
# -------------------------------
def main() -> None:
    location = render_location_control()

    loc = render_location_header(location)
    if loc is None:
        return

    render_forecast_panel(loc)


if __name__ == "__main__":
    main()

