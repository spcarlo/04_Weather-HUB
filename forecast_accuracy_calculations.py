import pandas as pd


# -------------------------------
# Columns
# -------------------------------
TEMPERATURE_METRICS = ("mean", "min", "max")
ACTUAL_DAILY_TEMPERATURE_FIELDS = (
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
)


# -------------------------------
# Forecast helpers
# -------------------------------
def previous_run_temperature_column(horizon_days: int) -> str:
    # Open-Meteo name for the forecast issued N days before the target hour
    if horizon_days < 1 or horizon_days > 7:
        raise ValueError("horizon_days must be between 1 and 7")

    return f"temperature_2m_previous_day{horizon_days}"


def previous_run_hourly_temperature_frame(hourly_data: dict, horizon_days: int) -> pd.DataFrame:
    pred_col = previous_run_temperature_column(horizon_days)
    return pd.DataFrame(
        {
            "time": pd.to_datetime(hourly_data["time"]),
            "temp_pred": hourly_data[pred_col],
        }
    )


def hourly_to_daily_temperature_stats(
    df: pd.DataFrame,
    value_col: str,
    output_prefix: str,
) -> pd.DataFrame:
    # One mean/min/max row per calendar day
    output_cols = [
        "date",
        f"{output_prefix}_mean",
        f"{output_prefix}_min",
        f"{output_prefix}_max",
    ]
    if df.empty:
        return pd.DataFrame(columns=output_cols)

    d = df[["time", value_col]].copy()
    d["date"] = d["time"].dt.date

    out = (
        d.groupby("date", as_index=False)
        .agg(
            **{
                f"{output_prefix}_mean": (value_col, "mean"),
                f"{output_prefix}_min": (value_col, "min"),
                f"{output_prefix}_max": (value_col, "max"),
            }
        )
        .loc[:, output_cols]
    )
    out["date"] = pd.to_datetime(out["date"])
    return out


def build_daily_pred(hourly_pred: pd.DataFrame) -> pd.DataFrame:
    return hourly_to_daily_temperature_stats(hourly_pred, "temp_pred", "temp_pred")


# -------------------------------
# Actual helpers
# -------------------------------
def actual_daily_temperature_fields() -> str:
    return ",".join(ACTUAL_DAILY_TEMPERATURE_FIELDS)


def actual_daily_temperature_frame(daily_data: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(daily_data["time"]),
            "temp_actual_mean": daily_data["temperature_2m_mean"],
            "temp_actual_min": daily_data["temperature_2m_min"],
            "temp_actual_max": daily_data["temperature_2m_max"],
        }
    )


# -------------------------------
# Scoring
# -------------------------------
def score_daily_forecast(pred_daily: pd.DataFrame, actual_daily: pd.DataFrame) -> pd.DataFrame:
    df = pred_daily.merge(actual_daily, on="date", how="inner")

    # Positive error means the forecast was too warm
    for metric in TEMPERATURE_METRICS:
        df[f"temp_error_{metric}"] = df[f"temp_pred_{metric}"] - df[f"temp_actual_{metric}"]

    return df


def _mean_or_none(series: pd.Series) -> float | None:
    valid_values = pd.to_numeric(series, errors="coerce").dropna()
    if valid_values.empty:
        return None

    return float(valid_values.mean())


def _mae_or_none(series: pd.Series) -> float | None:
    return _mean_or_none(pd.to_numeric(series, errors="coerce").abs())


def compute_metrics(df: pd.DataFrame) -> dict:
    # MAE = typical size of the miss; bias = average signed error
    metrics = {}
    for metric in TEMPERATURE_METRICS:
        error = df[f"temp_error_{metric}"]
        metrics[f"{metric}_mae"] = _mae_or_none(error)
        metrics[f"{metric}_bias"] = _mean_or_none(error)
    return metrics
