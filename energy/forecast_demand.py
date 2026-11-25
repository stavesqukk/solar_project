from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


DAILY_LAG_HOURS = 24
WEEKLY_LAG_HOURS = 24 * 7
DRY_SEASON_MONTHS = (12, 1, 2, 3, 4)
EVENING_HOURS = (18, 19, 20, 21)


def prepare_hourly_demand(demand: pd.Series) -> pd.Series:
    if not isinstance(demand.index, pd.DatetimeIndex):
        raise TypeError("Demand must have a DatetimeIndex.")
    if demand.index.has_duplicates:
        raise ValueError("Demand timestamps must be unique.")
    if demand.empty:
        raise ValueError("Demand series is empty.")

    demand = pd.to_numeric(demand, errors="coerce").sort_index()
    return demand.asfreq("h")


def create_forecasts(
    demand: pd.Series, train_fraction: float = 0.8
) -> tuple[pd.DataFrame, pd.Timestamp]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    hourly = prepare_hourly_demand(demand)
    observed = hourly.dropna()
    split_position = int(len(observed) * train_fraction)
    if split_position == 0 or split_position >= len(observed):
        raise ValueError("Not enough observed hours for the requested split.")

    split_at = observed.index[split_position]
    test = hourly.loc[hourly.index >= split_at]
    forecasts = pd.DataFrame(
        {
            "actual_demand_mw": test,
            "naive_24h_forecast_mw": hourly.shift(DAILY_LAG_HOURS).loc[test.index],
            "naive_168h_forecast_mw": hourly.shift(WEEKLY_LAG_HOURS).loc[test.index],
        }
    )
    forecasts = forecasts.dropna()
    if forecasts.empty:
        raise ValueError(
            "No holdout hours have both daily and weekly historical lags available."
        )

    forecasts.index.name = "timestamp"
    return forecasts, split_at


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    valid = actual.notna() & predicted.notna()
    actual_values = actual.loc[valid].to_numpy(dtype=float)
    predicted_values = predicted.loc[valid].to_numpy(dtype=float)
    if not len(actual_values):
        raise ValueError("Cannot calculate metrics without forecast/actual pairs.")

    errors = predicted_values - actual_values
    mape = (
        float(np.mean(np.abs(errors / actual_values)) * 100)
        if np.all(np.abs(actual_values) > 1)
        else float("nan")
    )
    return {
        "n": len(actual_values),
        "mae_mw": float(np.mean(np.abs(errors))),
        "rmse_mw": float(np.sqrt(np.mean(np.square(errors)))),
        "mape_pct": mape,
    }


def build_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "all_holdout": pd.Series(True, index=forecasts.index),
        "dry_season": pd.Series(
            forecasts.index.month.isin(DRY_SEASON_MONTHS), index=forecasts.index
        ),
        "evening_18_21": pd.Series(
            forecasts.index.hour.isin(EVENING_HOURS), index=forecasts.index
        ),
    }
    model_columns = {
        "same_hour_yesterday": "naive_24h_forecast_mw",
        "same_hour_last_week": "naive_168h_forecast_mw",
    }

    rows = []
    for period, mask in periods.items():
        subset = forecasts.loc[mask]
        for model, column in model_columns.items():
            rows.append(
                {
                    "period": period,
                    "model": model,
                    **calculate_metrics(
                        subset["actual_demand_mw"], subset[column]
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_forecasts(
    demand: pd.Series,
    forecasts: pd.DataFrame,
    split_at: pd.Timestamp,
    output_path: Path,
) -> None:
    hourly_window = None
    for _, segment in forecasts.groupby(
        forecasts.index.to_series().diff().ne(pd.Timedelta(hours=1)).cumsum()
    ):
        if len(segment) >= WEEKLY_LAG_HOURS:
            midnight_starts = np.flatnonzero(segment.index.hour == 0)
            for start in midnight_starts:
                if start + WEEKLY_LAG_HOURS <= len(segment):
                    hourly_window = segment.iloc[start : start + WEEKLY_LAG_HOURS]
                    break
            if hourly_window is not None:
                break
    if hourly_window is None:
        raise ValueError(
            "No complete midnight-aligned seven-day holdout window is available."
        )

    fig, (overview_ax, comparison_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [1, 1.2]},
    )

    missing_positions = np.flatnonzero(demand.isna().to_numpy())
    if missing_positions.size:
        missing_runs = np.split(
            missing_positions,
            np.flatnonzero(np.diff(missing_positions) > 1) + 1,
        )
        longest_gap = max(missing_runs, key=len)
        for missing_run in missing_runs:
            if len(missing_run) >= 24:
                overview_ax.axvspan(
                    demand.index[missing_run[0]],
                    demand.index[missing_run[-1]] + pd.Timedelta(hours=1),
                    color="lightgrey",
                    alpha=0.7,
                )
        if len(longest_gap) > 24:
            gap_start = demand.index[longest_gap[0]]
            gap_end = demand.index[longest_gap[-1]] + pd.Timedelta(hours=1)
            overview_ax.text(
                gap_start + (gap_end - gap_start) / 2,
                0.5,
                "NO SOURCE DATA\n(Jestha-Shrawan)\nnot zero demand",
                transform=overview_ax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=9,
            )

    overview_ax.plot(
        demand.index,
        demand,
        color="steelblue",
        linewidth=0.7,
        label="Cleaned hourly demand",
    )
    overview_ax.axvline(
        split_at,
        color="darkorange",
        linestyle="--",
        label=f"Evaluation holdout begins ({split_at:%d %b %Y})",
    )
    overview_ax.set_title(
        "Historical demand over the available date range (hourly, MW)"
    )
    overview_ax.set_ylabel("Demand (MW)")
    overview_ax.set_xlabel("Date (Asia/Kathmandu)")
    handles, labels = overview_ax.get_legend_handles_labels()
    if missing_positions.size:
        handles.append(Patch(facecolor="lightgrey", alpha=0.7))
        labels.append("Missing source period")
    overview_ax.legend(handles, labels, loc="upper left")
    overview_ax.grid(alpha=0.25)

    comparison_ax.plot(
        hourly_window.index,
        hourly_window["actual_demand_mw"],
        color="black",
        linewidth=1.5,
        label="Observed demand (historical actual)",
    )
    comparison_ax.plot(
        hourly_window.index,
        hourly_window["naive_24h_forecast_mw"],
        color="tab:blue",
        linewidth=1.1,
        label="Forecast: same hour yesterday (t-24 h)",
    )
    comparison_ax.plot(
        hourly_window.index,
        hourly_window["naive_168h_forecast_mw"],
        color="tab:orange",
        linewidth=1.1,
        label="Forecast: same hour last week (t-168 h)",
    )
    first_day = hourly_window.index[0].strftime("%d %b")
    last_day = hourly_window.index[-1].strftime("%d %b %Y")
    comparison_ax.set_title(
        f"Hourly one-step-ahead comparison: {first_day} to {last_day} "
        "(one calendar week)"
    )
    comparison_ax.set_ylabel("Demand (MW)")
    comparison_ax.set_xlabel(
        "Date and time (Asia/Kathmandu; each point is one hour)"
    )
    comparison_ax.legend()
    comparison_ax.grid(alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate daily and weekly seasonal-naive demand forecasts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/demand_hourly.csv"),
        help="Hourly demand CSV with timestamp and demand_mw columns.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Chronological share of observed hours before the holdout.",
    )
    args = parser.parse_args()

    source = pd.read_csv(args.input, parse_dates=["timestamp"])
    required_columns = {"timestamp", "demand_mw"}
    if not required_columns.issubset(source.columns):
        raise ValueError(
            f"{args.input} must contain columns {sorted(required_columns)}."
        )

    demand = source.set_index("timestamp")["demand_mw"]
    forecasts, split_at = create_forecasts(demand, args.train_fraction)
    metrics = build_metrics(forecasts)

    forecast_path = Path("energy/demand_forecast.csv")
    metrics_path = Path("report/demand_baseline_metrics.csv")
    plot_path = Path("report/demand_baseline_comparison.png")
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(forecast_path)
    metrics.to_csv(metrics_path, index=False)
    hourly_demand = prepare_hourly_demand(demand)
    plot_forecasts(hourly_demand, forecasts, split_at, plot_path)

    print("Chronological holdout starts:", split_at)
    print("Scored holdout hours:", len(forecasts))
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("Saved:", forecast_path)
    print("Saved:", metrics_path)
    print("Saved:", plot_path)


if __name__ == "__main__":
    main()
