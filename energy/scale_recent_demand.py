from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from energy.load_demand import (
    BS_MONTH_ALIASES,
    MONTH_DAYS_2073,
    MONTH_STARTS_2073,
    load_season_months,
)
from energy.seasonal_analysis import add_bs_month, read_hourly_demand


TIMEZONE = "Asia/Kathmandu"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "assumptions.yaml"
BALANCE_SECTION = "fy_2025_26_dry_monthly_balance"
BALANCE_FIELDS = (
    "domestic_generation_gwh",
    "import_gwh",
    "export_gwh",
    "national_demand_gwh",
)


def load_monthly_balance(
    path: Path = CONFIG_PATH,
) -> dict[str, dict[str, float | str]]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    section = config.get(BALANCE_SECTION) if isinstance(config, dict) else None
    if not isinstance(section, dict) or section.get("status") != "unverified_user_provided":
        raise ValueError(
            f"{path} must mark {BALANCE_SECTION} as unverified_user_provided."
        )
    if section.get("units") != "GWh":
        raise ValueError(f"{path} must specify GWh for {BALANCE_SECTION}.")

    raw_months = section.get("months")
    if not isinstance(raw_months, dict):
        raise ValueError(f"{path} must define monthly balances in {BALANCE_SECTION}.")

    season_by_month = load_season_months(path)
    expected_months = {
        month for month, season in season_by_month.items() if season == "dry"
    }
    balances: dict[str, dict[str, float | str]] = {}
    for label, raw_values in raw_months.items():
        if not isinstance(label, str) or not isinstance(raw_values, dict):
            raise ValueError(f"Invalid month entry in {BALANCE_SECTION}: {label!r}.")
        month = BS_MONTH_ALIASES.get(label.strip().lower(), label.strip().lower())
        if month not in expected_months:
            raise ValueError(f"{label!r} is not a configured dry-season month.")
        values: dict[str, float | str] = {
            "month_label": label,
            "source_status": section["status"],
        }
        for field in BALANCE_FIELDS:
            value = raw_values.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must provide numeric {field}.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} {field} must be finite and nonnegative.")
            values[field] = float(value)
        if values["national_demand_gwh"] <= 0:
            raise ValueError(f"{label} national_demand_gwh must be greater than zero.")
        if month in balances:
            raise ValueError(f"Duplicate balance for BS month {label!r}.")
        balances[month] = values

    if set(balances) != expected_months:
        missing = sorted(expected_months - set(balances))
        raise ValueError(f"{path} is missing dry-month balances: {missing}.")
    return balances


def scale_dry_months(
    hourly: pd.DataFrame,
    monthly_balance: dict[str, dict[str, float | str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"demand_mw", "bs_month"}
    if not required.issubset(hourly.columns):
        raise ValueError(f"Hourly data must contain columns {sorted(required)}.")
    if not isinstance(hourly.index, pd.DatetimeIndex) or hourly.index.tz is None:
        raise ValueError("Hourly data must use timezone-aware timestamps.")
    if hourly.index.has_duplicates:
        raise ValueError("Hourly timestamps must be unique.")

    # Assumption: use the user-provided FY 2025/26 demand totals to scale the
    # observed, cleaned 2073 hourly shape; this does not create missing hours.
    year_end = (
        pd.Timestamp(MONTH_STARTS_2073["chaitra"])
        + pd.Timedelta(days=MONTH_DAYS_2073["chaitra"])
    ).tz_localize(TIMEZONE)
    in_2073 = hourly.index < year_end
    scaled = hourly.copy()
    scaled["demand_scaled_mw"] = np.nan
    rows = []

    for month, values in monthly_balance.items():
        mask = in_2073 & hourly["bs_month"].eq(month).to_numpy()
        observed = hourly.loc[mask, "demand_mw"]
        observed_energy_gwh = observed.sum(min_count=1) / 1000
        if not np.isfinite(observed_energy_gwh) or observed_energy_gwh <= 0:
            raise ValueError(f"No positive observed 2073 demand energy for {month}.")

        target_energy_gwh = float(values["national_demand_gwh"])
        multiplier = target_energy_gwh / observed_energy_gwh
        scaled.loc[mask, "demand_scaled_mw"] = observed * multiplier
        hydro_share = (
            float(values["domestic_generation_gwh"])
            - float(values["export_gwh"])
        ) / target_energy_gwh
        import_share = float(values["import_gwh"]) / target_energy_gwh

        rows.append(
            {
                "month": values["month_label"],
                "balance_status": values["source_status"],
                "domestic_generation_gwh": values["domestic_generation_gwh"],
                "import_gwh": values["import_gwh"],
                "export_gwh": values["export_gwh"],
                "national_demand_gwh": target_energy_gwh,
                "hydro_share_pct": hydro_share * 100,
                "import_share_pct": import_share * 100,
                "gap_share_pct": (1 - hydro_share) * 100,
                "2073_observed_energy_gwh": observed_energy_gwh,
                "scale_multiplier": multiplier,
                "scaled_observed_energy_gwh": (
                    scaled.loc[mask, "demand_scaled_mw"].sum(min_count=1) / 1000
                ),
            }
        )

    return scaled, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale dry-month 2073 demand profiles to provided FY 2025/26 totals."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/demand_hourly.csv"),
        help="Cleaned hourly demand CSV.",
    )
    parser.add_argument(
        "--hourly-output",
        type=Path,
        default=Path("data/demand_hourly_scaled_2025_26.csv"),
        help="Output CSV retaining demand_mw and adding demand_scaled_mw.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("report/dry_month_scaling_2025_26.csv"),
        help="Output CSV for monthly balances, derived shares, and multipliers.",
    )
    args = parser.parse_args()

    hourly = add_bs_month(read_hourly_demand(args.input))
    scaled, summary = scale_dry_months(hourly, load_monthly_balance())
    args.hourly_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    scaled.to_csv(args.hourly_output, index_label="timestamp")
    summary.to_csv(args.summary_output, index=False)

    print("Dry-month scaling summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved:")
    print(args.hourly_output)
    print(args.summary_output)


if __name__ == "__main__":
    main()
