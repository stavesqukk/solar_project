
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MONTH_STARTS_2073 = {
    "baisakh": "2016-04-13",
    "jestha": "2016-05-14",
    "ashadh": "2016-06-15",
    "shrawan": "2016-07-16",
    "bhadra": "2016-08-17",
    "ashoj": "2016-09-17",
    "kartik": "2016-10-17",
    "mangsir": "2016-11-16",
    "paush": "2016-12-16",
    "magh": "2017-01-14",
    "falgun": "2017-02-12",
    "chaitra": "2017-03-14",
}

MONTH_DAYS_2073 = {
    "baisakh": 31,
    "jestha": 32,
    "ashadh": 31,
    "shrawan": 32,
    "bhadra": 31,
    "ashoj": 30,
    "kartik": 30,
    "mangsir": 30,
    "paush": 29,
    "magh": 29,
    "falgun": 30,
    "chaitra": 31,
}


def parse_time(value: str) -> tuple[int, int]:
    normalized = value.replace(" ", "")
    if ":" not in normalized:
        fraction_of_day = float(normalized)
        if not 0 <= fraction_of_day < 1:
            raise ValueError(f"Invalid fractional-day time: {value!r}")
        half_hour_steps = round(fraction_of_day * 48)
        if half_hour_steps >= 48:
            raise ValueError(f"Invalid fractional-day time: {value!r}")
        hour, half_hour = divmod(half_hour_steps, 2)
        return hour, half_hour * 30

    hour_text, minute_text = normalized.split(":")
    hour = int(hour_text)
    minute = int(minute_text)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"Invalid hourly time: {value!r}")
    return hour, minute


def load_month(path: Path) -> tuple[pd.DataFrame, list[int]]:
    month = path.stem.lower()
    if month not in MONTH_STARTS_2073:
        raise ValueError(
            f"Unknown month '{month}'. Add its BS-to-AD mapping to MONTH_STARTS_2073."
        )

    month_start = pd.Timestamp(MONTH_STARTS_2073[month])
    source = pd.read_csv(path, header=None, dtype=str)
    header_positions = [
        position
        for position, value in enumerate(source.iloc[:, 0])
        if str(value).strip().lower() == "day"
    ]
    if not header_positions:
        raise ValueError(f"Could not find a 'Day' header row in {path}.")
    header_position = header_positions[0]
    headers = source.iloc[header_position].tolist()
    raw = source.iloc[header_position + 1 :].copy()
    raw.columns = headers
    raw = raw.loc[:, [column for column in raw.columns if pd.notna(column)]]
    raw = raw.loc[:, [str(column).strip() for column in raw.columns]]
    raw.columns = [str(column).strip() for column in raw.columns]
    raw = raw[pd.to_numeric(raw["Day"], errors="coerce").notna()].copy()
    raw["Day"] = pd.to_numeric(raw["Day"], errors="coerce").astype(int)
    valid_day = raw["Day"].between(1, MONTH_DAYS_2073[month])
    invalid_days = raw.loc[~valid_day, "Day"].tolist()
    raw = raw.loc[valid_day].copy()

    rows: list[dict] = []
    for _, row in raw.iterrows():
        for column in raw.columns:
            if column == "Day":
                continue

            hour, minute = parse_time(str(column))
            date = month_start + pd.Timedelta(days=int(row["Day"]) - 1)
            timestamp = pd.Timestamp(
                date.year, date.month, date.day, hour, minute
            )
            if hour == 0 and minute == 0:
                timestamp += pd.Timedelta(days=1)

            rows.append(
                {
                    "timestamp": timestamp,
                    "demand_mw": pd.to_numeric(row[column], errors="coerce"),
                }
            )

    result = pd.DataFrame(rows).sort_values("timestamp")
    result = result.set_index("timestamp")
    return result, invalid_days


def clean_readings(readings: pd.DataFrame) -> pd.DataFrame:
    readings = readings.copy()
    readings["demand_mw"] = pd.to_numeric(readings["demand_mw"], errors="coerce")

    readings["flagged"] = (
        readings["demand_mw"].isna()
        | (readings["demand_mw"] < 400)
        | (readings["demand_mw"] > 1800)
    )
    readings["demand_clean_mw"] = readings["demand_mw"].where(
        ~readings["flagged"]
    )
    segments = readings.index.to_series().diff().gt(pd.Timedelta(hours=2)).cumsum()
    readings["demand_clean_mw"] = readings.groupby(
        segments, sort=False
    )["demand_clean_mw"].transform(
        lambda values: values.interpolate(method="time", limit_direction="both")
    )
    return readings


def build_hourly(readings: pd.DataFrame) -> pd.DataFrame:
    hourly = readings["demand_clean_mw"].resample("h").mean()
    hourly.name = "demand_mw"
    result = hourly.to_frame()
    result["readings_in_hour"] = readings["demand_mw"].resample("h").count()
    return result


def plot_week(hourly: pd.DataFrame, output_path: Path) -> None:
    week = hourly.iloc[: 7 * 24]
    if week.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(week.index, week["demand_mw"], color="steelblue", linewidth=1.2)
    ax.set_title("Demand - First Available Week")
    ax.set_xlabel("Asia/Kathmandu time")
    ax.set_ylabel("Demand (MW)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean the NEA Nepali-year 2073 demand CSV files."
    )
    parser.add_argument("raw_folder", type=Path)
    args = parser.parse_args()

    csv_files = sorted(args.raw_folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.raw_folder}")

    monthly_readings = []
    ignored_day_rows = []
    for csv_file in csv_files:
        month_readings, invalid_days = load_month(csv_file)
        monthly_readings.append(month_readings)
        if invalid_days:
            ignored_day_rows.append((csv_file.name, invalid_days))

    readings = pd.concat(monthly_readings, axis="index", sort=False).sort_index()
    if readings.index.has_duplicates:
        duplicate_times = readings.index[readings.index.duplicated()].unique()
        raise ValueError(
            "Monthly files contain overlapping timestamps: "
            f"{duplicate_times[:5].tolist()}"
        )

    readings = clean_readings(readings)

    year_start = pd.Timestamp(MONTH_STARTS_2073["baisakh"]) + pd.Timedelta(hours=1)
    year_end = pd.Timestamp(MONTH_STARTS_2073["chaitra"]) + pd.Timedelta(
        days=MONTH_DAYS_2073["chaitra"]
    )
    year_index = pd.date_range(year_start, year_end, freq="h", name="timestamp")
    hourly = build_hourly(readings).reindex(year_index)
    hourly.index = hourly.index.tz_localize(
        "Asia/Kathmandu", nonexistent="shift_forward", ambiguous="NaT"
    )

    readings.index = readings.index.tz_localize(
        "Asia/Kathmandu", nonexistent="shift_forward", ambiguous="NaT"
    )

    data_folder = Path("data")
    report_folder = Path("report")
    data_folder.mkdir(exist_ok=True)
    report_folder.mkdir(exist_ok=True)

    readings.to_csv(data_folder / "demand_all_readings.csv")
    hourly.to_csv(data_folder / "demand_hourly.csv")
    plot_week(hourly, report_folder / "demand_first_week.png")

    print("Loaded CSV files:", len(csv_files))
    print("Out-of-range BS day rows ignored:", sum(len(days) for _, days in ignored_day_rows))
    for file_name, days in ignored_day_rows:
        print(f"  {file_name}: invalid day labels {days}")
    print("All-reading rows:", len(readings))
    print("Hourly rows:", len(hourly))
    print("Flagged readings:", int(readings["flagged"].sum()))
    print("Missing hourly periods:", int(hourly["demand_mw"].isna().sum()))
    print("Date range:", hourly.index.min(), "to", hourly.index.max())
    print("Saved:", data_folder / "demand_hourly.csv")
    print("Saved:", data_folder / "demand_all_readings.csv")
    print("Saved:", report_folder / "demand_first_week.png")


if __name__ == "__main__":
    main()
