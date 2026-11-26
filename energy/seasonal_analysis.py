from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy.load_demand import (
    MONTH_DAYS_2073,
    MONTH_STARTS_2073,
    load_season_months,
)


TIMEZONE = "Asia/Kathmandu"
MIDDAY_HOURS = (11, 12, 13, 14)
EVENING_HOURS = (18, 19, 20, 21)
SEASON_ORDER = ("dry", "shoulder", "wet")
MONTH_LABELS = {
    month: month.title()
    for month in MONTH_STARTS_2073
}


def read_hourly_demand(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "demand_mw", "season"}
    if not required.issubset(source.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}.")

    timestamps = pd.DatetimeIndex(source.pop("timestamp"), name="timestamp")
    if timestamps.tz is None:
        raise ValueError("Demand timestamps must include the local timezone.")
    if timestamps.has_duplicates:
        raise ValueError("Demand timestamps must be unique.")

    source.index = timestamps.tz_convert(TIMEZONE)
    return source.sort_index()


def add_bs_month(hourly: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(hourly.index, pd.DatetimeIndex) or hourly.index.tz is None:
        raise ValueError("Hourly demand must have timezone-aware timestamps.")

    month_boundaries = sorted(
        (pd.Timestamp(start), month)
        for month, start in MONTH_STARTS_2073.items()
    )
    next_baisakh = (
        pd.Timestamp(MONTH_STARTS_2073["chaitra"])
        + pd.Timedelta(days=MONTH_DAYS_2073["chaitra"])
    )
    month_boundaries.append((next_baisakh, "baisakh"))

    local_timestamps = hourly.index.tz_convert(TIMEZONE).tz_localize(None)
    bs_month = pd.Series(index=hourly.index, dtype="object", name="bs_month")
    for position, (start, month) in enumerate(month_boundaries):
        start = start.normalize()
        if position + 1 < len(month_boundaries):
            end = month_boundaries[position + 1][0].normalize()
            matches = (local_timestamps >= start) & (local_timestamps < end)
        else:
            matches = local_timestamps >= start
        bs_month.loc[matches] = month

    if bs_month.isna().any():
        raise ValueError("Could not assign a BS month to every timestamp.")

    result = hourly.copy()
    result["bs_month"] = bs_month
    result["hour_of_day"] = result.index.hour
    return result


def select_bs_2073(hourly: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # Assumption: 0:00 is the next day's midnight, so the final timestamp at
    # Baisakh 1, 2074 belongs to that next BS year and is excluded from 2073.
    year_end = (
        pd.Timestamp(MONTH_STARTS_2073["chaitra"])
        + pd.Timedelta(days=MONTH_DAYS_2073["chaitra"])
    ).tz_localize(TIMEZONE)
    in_year = hourly.index < year_end
    return hourly.loc[in_year].copy(), int((~in_year).sum())


def build_daily_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(MONTH_STARTS_2073["baisakh"]).tz_localize(TIMEZONE)
    end = (
        pd.Timestamp(MONTH_STARTS_2073["chaitra"])
        + pd.Timedelta(days=MONTH_DAYS_2073["chaitra"])
    ).tz_localize(TIMEZONE)
    calendar_days = pd.date_range(
        start, end - pd.Timedelta(days=1), freq="D", tz=TIMEZONE
    )

    rows = []
    for day_start in calendar_days:
        day_end = day_start + pd.Timedelta(days=1)
        day = hourly.loc[
            (hourly.index >= day_start) & (hourly.index < day_end)
        ]
        observed = day["demand_mw"].dropna()
        minimum_time = observed.idxmin() if not observed.empty else pd.NaT
        maximum_time = observed.idxmax() if not observed.empty else pd.NaT
        by_hour = day.set_index(day.index.hour)["demand_mw"]

        midday = by_hour.reindex(MIDDAY_HOURS)
        evening = by_hour.reindex(EVENING_HOURS)
        at_17 = by_hour.get(17, np.nan)
        at_18 = by_hour.get(18, np.nan)
        complete = len(observed) == 24

        # Assumption: hourly MW values are treated as average MW for one hour.
        # Energy is an observed-hours estimate; missing hours are not filled.
        rows.append(
            {
                "date": day_start.strftime("%Y-%m-%d"),
                "bs_month": day["bs_month"].iloc[0],
                "season": day["season"].iloc[0],
                "weekday": day_start.day_name(),
                # Assumption (user-provided): Saturday is Nepal's weekend day.
                "is_nepal_weekend": day_start.day_of_week == 5,
                "expected_hours": 24,
                "observed_hours": len(observed),
                "complete_day": complete,
                "mean_demand_mw": observed.mean(),
                "minimum_demand_mw": observed.min(),
                "minimum_hour": minimum_time.hour if pd.notna(minimum_time) else np.nan,
                "maximum_demand_mw": observed.max(),
                "maximum_hour": maximum_time.hour if pd.notna(maximum_time) else np.nan,
                "observed_energy_gwh": observed.sum() / 1000,
                "mean_midday_11_14_mw": midday.mean(),
                "peak_evening_18_21_mw": evening.max(),
                "peak_evening_hour": (
                    evening.idxmax() if evening.notna().any() else np.nan
                ),
                "ramp_17_to_18_mw": (
                    at_18 - at_17
                    if pd.notna(at_17) and pd.notna(at_18)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _mode_hour(values: pd.Series) -> float:
    observed = values.dropna().astype(int)
    if observed.empty:
        return np.nan
    counts = observed.value_counts()
    return float(counts[counts == counts.max()].index.min())


def _summarize_group(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    period_type: str,
    period: str,
    expected_days: int,
) -> dict:
    if period_type == "month":
        hourly_group = hourly.loc[hourly["bs_month"] == period.lower()]
        daily_group = daily.loc[daily["bs_month"] == period.lower()]
    else:
        hourly_group = hourly.loc[hourly["season"] == period]
        daily_group = daily.loc[daily["season"] == period]

    observed = hourly_group["demand_mw"].dropna()
    hourly_profile = hourly_group.groupby("hour_of_day")["demand_mw"].mean()
    midday_mean = observed.loc[
        hourly_group.loc[observed.index, "hour_of_day"].isin(MIDDAY_HOURS)
    ].mean()
    evening_profile = hourly_profile.reindex(EVENING_HOURS).dropna()
    evening_peak = evening_profile.max() if not evening_profile.empty else np.nan
    complete_days = daily_group.loc[daily_group["complete_day"]]

    # Assumption: daily extrema and their typical hours use complete days only;
    # partial days remain in the daily output with their observed-hour counts.
    weekday_daily_mean = complete_days.loc[
        ~complete_days["is_nepal_weekend"], "mean_demand_mw"
    ]
    saturday_daily_mean = complete_days.loc[
        complete_days["is_nepal_weekend"], "mean_demand_mw"
    ]

    return {
        "period_type": period_type,
        "period": period,
        "expected_days": expected_days,
        "expected_hours": expected_days * 24,
        "observed_hours": len(observed),
        "missing_hours": expected_days * 24 - len(observed),
        "coverage_pct": len(observed) / (expected_days * 24) * 100,
        "mean_demand_mw": observed.mean(),
        "minimum_demand_mw": observed.min(),
        "maximum_demand_mw": observed.max(),
        "observed_energy_gwh": observed.sum() / 1000,
        "days_with_observations": int(daily_group["observed_hours"].gt(0).sum()),
        "complete_days": len(complete_days),
        "mean_daily_minimum_mw": complete_days["minimum_demand_mw"].mean(),
        "hour_of_daily_minimum_mode": _mode_hour(complete_days["minimum_hour"]),
        "mean_daily_maximum_mw": complete_days["maximum_demand_mw"].mean(),
        "hour_of_daily_maximum_mode": _mode_hour(complete_days["maximum_hour"]),
        "mean_midday_11_14_mw": midday_mean,
        "peak_mean_evening_18_21_mw": evening_peak,
        "peak_mean_evening_hour": (
            evening_profile.idxmax() if not evening_profile.empty else np.nan
        ),
        "evening_ratio_peak_to_midday": (
            evening_peak / midday_mean
            if pd.notna(evening_peak) and pd.notna(midday_mean) and midday_mean
            else np.nan
        ),
        "mean_ramp_17_to_18_mw": daily_group["ramp_17_to_18_mw"].mean(),
        "days_with_ramp": int(daily_group["ramp_17_to_18_mw"].notna().sum()),
        "weekday_mean_daily_demand_mw": weekday_daily_mean.mean(),
        "saturday_mean_daily_demand_mw": saturday_daily_mean.mean(),
        "saturday_minus_weekday_mw": (
            saturday_daily_mean.mean() - weekday_daily_mean.mean()
            if weekday_daily_mean.notna().any()
            and saturday_daily_mean.notna().any()
            else np.nan
        ),
        "complete_weekdays": len(weekday_daily_mean),
        "complete_saturdays": len(saturday_daily_mean),
        "minimum_mean_profile_mw": hourly_profile.min(),
        "minimum_mean_profile_hour": (
            hourly_profile.idxmin() if hourly_profile.notna().any() else np.nan
        ),
        "maximum_mean_profile_mw": hourly_profile.max(),
        "maximum_mean_profile_hour": (
            hourly_profile.idxmax() if hourly_profile.notna().any() else np.nan
        ),
    }


def build_group_summary(
    hourly: pd.DataFrame, daily: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for month, days in MONTH_DAYS_2073.items():
        rows.append(
            _summarize_group(
                hourly, daily, "month", MONTH_LABELS[month], days
            )
        )

    season_by_month = load_season_months()
    for season in SEASON_ORDER:
        months = [month for month, label in season_by_month.items() if label == season]
        expected_days = sum(MONTH_DAYS_2073[month] for month in months)
        rows.append(
            _summarize_group(hourly, daily, "season", season, expected_days)
        )
    return pd.DataFrame(rows)


def _hourly_profiles(
    hourly: pd.DataFrame, group_column: str
) -> pd.DataFrame:
    return (
        hourly.groupby([group_column, "hour_of_day"])["demand_mw"]
        .mean()
        .rename("mean_demand_mw")
        .reset_index()
    )


def plot_season_profiles(profiles: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for season in SEASON_ORDER:
        data = profiles.loc[profiles["season"] == season].sort_values("hour_of_day")
        ax.plot(
            data["hour_of_day"],
            data["mean_demand_mw"],
            marker="o",
            linewidth=1.8,
            label=season.title(),
        )
    ax.set_title(
        "2073 BS mean hourly demand profile by season\n"
        "Historical average shape; not a forecast"
    )
    ax.set_xlabel("Hour of day (Asia/Kathmandu)")
    ax.set_ylabel("Mean demand (MW)")
    ax.set_xticks(range(24), [f"{hour:02d}:00" for hour in range(24)], rotation=45)
    ax.grid(alpha=0.25)
    ax.legend(title="User-defined season")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    return fig


def plot_month_profiles(profiles: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.get_cmap("tab20")
    month_order = list(MONTH_STARTS_2073)
    for position, month in enumerate(month_order):
        label = MONTH_LABELS[month]
        data = profiles.loc[profiles["bs_month"] == month].sort_values(
            "hour_of_day"
        )
        ax.plot(
            data["hour_of_day"],
            data["mean_demand_mw"],
            marker="o",
            markersize=3,
            linewidth=1.3,
            color=colors(position % 20),
            label=label,
        )
    ax.set_title(
        "2073 BS mean hourly demand profile by month\n"
        "Historical average shape; not a forecast"
    )
    ax.set_xlabel("Hour of day (Asia/Kathmandu)")
    ax.set_ylabel("Mean demand (MW)")
    ax.set_xticks(range(24), [f"{hour:02d}:00" for hour in range(24)], rotation=45)
    ax.grid(alpha=0.25)
    ax.legend(title="BS month", ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize observed 2073 BS demand by day, month, and season."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/demand_hourly.csv"),
        help="Hourly demand CSV produced by energy/load_demand.py.",
    )
    args = parser.parse_args()

    hourly = add_bs_month(read_hourly_demand(args.input))
    hourly, excluded_boundary_rows = select_bs_2073(hourly)
    daily = build_daily_summary(hourly)
    summary = build_group_summary(hourly, daily)
    season_profiles = _hourly_profiles(hourly, "season")
    month_profiles = _hourly_profiles(hourly, "bs_month")

    daily_path = Path("report/demand_daily_summary_2073.csv")
    summary_path = Path("report/seasonal_demand_summary_2073.csv")
    season_plot_path = Path("report/demand_mean_profile_by_season_2073.png")
    month_plot_path = Path("report/demand_mean_profiles_by_month_2073.png")
    for path in (daily_path, summary_path, season_plot_path, month_plot_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    daily.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    season_figure = plot_season_profiles(season_profiles, season_plot_path)
    month_figure = plot_month_profiles(month_profiles, month_plot_path)
    plt.close(season_figure)
    plt.close(month_figure)

    print("Hourly slots analyzed:", len(hourly))
    print("Observed demand hours:", int(hourly["demand_mw"].notna().sum()))
    print("Missing demand hours:", int(hourly["demand_mw"].isna().sum()))
    print("Excluded next-BS-year boundary rows:", excluded_boundary_rows)
    print("\nMonthly and seasonal summary:")
    print(
        summary.to_string(
            index=False, float_format=lambda value: f"{value:.2f}"
        )
    )
    print("\nSaved:")
    for path in (daily_path, summary_path, season_plot_path, month_plot_path):
        print(path)


if __name__ == "__main__":
    main()
