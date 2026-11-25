from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PROJECT_DRY_MONTHS = (12, 1, 2, 3, 4)
MONSOON_MONTHS = (6, 7, 8, 9)

SEASON_LABELS = {
    "project_dry": "Project dry season (Dec-Apr)",
    "monsoon": "Monsoon season (Jun-Sep)",
    "transition": "Transition months (May, Oct-Nov)",
}


def classify_season(month: int) -> str:
    if month in PROJECT_DRY_MONTHS:
        return "project_dry"
    if month in MONSOON_MONTHS:
        return "monsoon"
    return "transition"


def read_hourly_demand(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "demand_mw"}
    if not required.issubset(source.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}.")

    timestamps = pd.DatetimeIndex(source["timestamp"], name="timestamp")
    if timestamps.tz is None:
        raise ValueError("Demand timestamps must include the Asia/Kathmandu timezone.")
    if timestamps.has_duplicates:
        raise ValueError("Demand timestamps must be unique.")

    hourly = source.drop(columns="timestamp")
    hourly.index = timestamps
    return hourly.sort_index().asfreq("h")


def add_calendar_and_lag_features(hourly: pd.DataFrame) -> pd.DataFrame:
    features = hourly.copy()
    timestamps = features.index
    features["month"] = timestamps.strftime("%Y-%m")
    features["month_number"] = timestamps.month
    features["hour_of_day"] = timestamps.hour
    features["day_of_week"] = timestamps.day_name()
    features["is_nepal_weekend"] = timestamps.day_of_week == 5
    features["season_type"] = [
        classify_season(month) for month in timestamps.month
    ]
    features["demand_lag_24h_mw"] = features["demand_mw"].shift(24)
    features["demand_lag_168h_mw"] = features["demand_mw"].shift(168)
    features["previous_24h_mean_mw"] = (
        features["demand_mw"].shift(1).rolling(24, min_periods=24).mean()
    )
    return features


def build_monthly_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, month_data in features.groupby("month", sort=True):
        observed = month_data["demand_mw"].dropna()
        peak_timestamp = observed.idxmax() if not observed.empty else pd.NaT
        expected_hours = pd.Period(month, freq="M").days_in_month * 24
        rows.append(
            {
                "month": month,
                "season_type": classify_season(int(month[-2:])),
                "expected_hours": expected_hours,
                "observed_hours": len(observed),
                "missing_hours": expected_hours - len(observed),
                "coverage_pct": len(observed) / expected_hours * 100,
                "mean_demand_mw": observed.mean(),
                "peak_demand_mw": observed.max(),
                "peak_timestamp": peak_timestamp,
                "peak_hour_of_day": (
                    peak_timestamp.hour if not pd.isna(peak_timestamp) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def build_hourly_profiles(
    features: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    return (
        features.groupby(group_columns + ["hour_of_day"], sort=True)["demand_mw"]
        .agg(
            mean_demand_mw="mean",
            observed_hours="count",
            timestamp_slots_in_series="size",
        )
        .reset_index()
    )


def plot_hourly_overview(features: pd.DataFrame, output_path: Path):
    demand = features["demand_mw"]
    missing_positions = np.flatnonzero(demand.isna().to_numpy())
    fig, ax = plt.subplots(figsize=(13, 5))

    if missing_positions.size:
        missing_runs = np.split(
            missing_positions,
            np.flatnonzero(np.diff(missing_positions) > 1) + 1,
        )
        for missing_run in missing_runs:
            if len(missing_run) >= 24:
                ax.axvspan(
                    features.index[missing_run[0]],
                    features.index[missing_run[-1]] + pd.Timedelta(hours=1),
                    color="lightgrey",
                    alpha=0.7,
                )
        longest_gap = max(missing_runs, key=len)
        if len(longest_gap) > 24:
            gap_start = features.index[longest_gap[0]]
            gap_end = features.index[longest_gap[-1]] + pd.Timedelta(hours=1)
            ax.text(
                gap_start + (gap_end - gap_start) / 2,
                0.5,
                "No source data\n(not zero demand)",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
            )

    ax.plot(
        features.index,
        demand,
        color="steelblue",
        linewidth=0.7,
        label="Cleaned demand",
    )
    ax.set_title("Historical hourly demand (source: BS 2073; x-axis: Gregorian AD)")
    ax.set_xlabel("Date (Asia/Kathmandu, Gregorian)")
    ax.set_ylabel("Demand (MW)")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if missing_positions.size:
        handles.append(Patch(facecolor="lightgrey", alpha=0.7))
        labels.append("Missing source hours (not zero demand)")
    ax.legend(handles, labels)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    return fig


def plot_monthly_profiles(
    profiles: pd.DataFrame, monthly_summary: pd.DataFrame, output_path: Path
):
    coverage = monthly_summary.set_index("month")["coverage_pct"]
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.get_cmap("tab20")

    for position, (month, month_data) in enumerate(
        profiles.groupby("month", sort=True)
    ):
        if month_data["observed_hours"].sum() == 0:
            continue
        month_period = pd.Period(month, freq="M")
        partial = coverage.loc[month] < 100
        label = f"{month_period:%b %Y}{'*' if partial else ''}"
        ax.plot(
            month_data["hour_of_day"],
            month_data["mean_demand_mw"],
            marker="o",
            markersize=3,
            linewidth=1.3,
            color=colors(position % 20),
            label=label,
        )

    ax.set_title("Average demand for each hour of the day, by month")
    ax.set_xlabel("Hour of day (Asia/Kathmandu)")
    ax.set_ylabel("Mean demand (MW)")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{hour:02d}:00" for hour in range(24)], rotation=45)
    ax.grid(alpha=0.25)
    ax.legend(title="Month (* = partial source coverage)", ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    return fig


def plot_seasonal_profiles(profiles: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for season, season_data in profiles.groupby("season_type", sort=True):
        if season_data["observed_hours"].sum() == 0:
            continue
        profile = (
            season_data.groupby("hour_of_day")["mean_demand_mw"].mean().sort_index()
        )
        ax.plot(
            profile.index,
            profile.values,
            marker="o",
            linewidth=1.6,
            label=SEASON_LABELS[season],
        )

    ax.set_title(
        "Average hourly demand profile by season category\n"
        "Calendar categories only; not a measure of hydrological conditions"
    )
    ax.set_xlabel("Hour of day (Asia/Kathmandu)")
    ax.set_ylabel("Mean demand (MW)")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{hour:02d}:00" for hour in range(24)], rotation=45)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create demand tables, hourly profiles, and plots."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/demand_hourly.csv"),
        help="Cleaned hourly demand CSV.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the generated plots in interactive windows.",
    )
    args = parser.parse_args()

    hourly = read_hourly_demand(args.input)
    features = add_calendar_and_lag_features(hourly)
    monthly_summary = build_monthly_summary(features)
    month_profiles = build_hourly_profiles(features, ["month", "season_type"])
    season_profiles = build_hourly_profiles(features, ["season_type"])

    feature_path = Path("data/demand_hourly_features.csv")
    monthly_summary_path = Path("report/demand_monthly_summary.csv")
    month_profiles_path = Path("report/demand_profile_by_month_hour.csv")
    season_profiles_path = Path("report/demand_profile_by_season_hour.csv")
    overview_plot_path = Path("report/demand_hourly_overview.png")
    month_plot_path = Path("report/demand_monthly_profiles.png")
    season_plot_path = Path("report/demand_seasonal_profiles.png")

    for path in (
        feature_path,
        monthly_summary_path,
        month_profiles_path,
        season_profiles_path,
        overview_plot_path,
        month_plot_path,
        season_plot_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(feature_path, index_label="timestamp")
    monthly_summary.to_csv(monthly_summary_path, index=False)
    month_profiles.to_csv(month_profiles_path, index=False)
    season_profiles.to_csv(season_profiles_path, index=False)

    figures = [
        plot_hourly_overview(features, overview_plot_path),
        plot_monthly_profiles(
            month_profiles, monthly_summary, month_plot_path
        ),
        plot_seasonal_profiles(season_profiles, season_plot_path),
    ]

    print("Input rows:", len(hourly))
    print("Observed demand hours:", int(hourly["demand_mw"].notna().sum()))
    print("Missing demand hours:", int(hourly["demand_mw"].isna().sum()))
    print("\nMonthly summary:")
    print(
        monthly_summary.to_string(
            index=False, float_format=lambda value: f"{value:.2f}"
        )
    )
    print("\nSaved tables and plots:")
    for path in (
        feature_path,
        monthly_summary_path,
        month_profiles_path,
        season_profiles_path,
        overview_plot_path,
        month_plot_path,
        season_plot_path,
    ):
        print(path)

    if args.show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()
