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
)
from energy.scale_recent_demand import (
    CONFIG_PATH,
    load_monthly_balance,
)
from energy.seasonal_analysis import read_hourly_demand


TIMEZONE = "Asia/Kathmandu"
SCENARIO_SECTION = "hydro_gap_scenarios"
STORAGE_HOURS = frozenset(range(18, 23))
GAP_WINDOW_HOURS = frozenset(range(17, 23))
PEAK_COMPARISON_HOURS = frozenset(range(18, 23))


def load_scenario_config(path: Path = CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    section = config.get(SCENARIO_SECTION) if isinstance(config, dict) else None
    if not isinstance(section, dict) or section.get("status") != "user_assumptions":
        raise ValueError(f"{path} must define {SCENARIO_SECTION} as user_assumptions.")

    raw_shift_values = section.get("storage_shift_percentages")
    if (
        not isinstance(raw_shift_values, list)
        or not raw_shift_values
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value < 100
            for value in raw_shift_values
        )
        or len(set(raw_shift_values)) != len(raw_shift_values)
    ):
        raise ValueError("Storage shift percentages must be unique values from 0 to <100.")
    shift_values: list[float] = [float(value) for value in raw_shift_values]

    raw_base_shift = section.get("storage_shift_base_percent")
    if (
        isinstance(raw_base_shift, bool)
        or not isinstance(raw_base_shift, (int, float))
        or not math.isfinite(raw_base_shift)
    ):
        raise ValueError("The base storage shift must be numeric.")
    base_shift = float(raw_base_shift)
    if base_shift not in shift_values:
        raise ValueError("The base storage shift must be included in the sweep.")

    sensitivity = section.get("hydro_share_sensitivity_percentage_points")
    if (
        isinstance(sensitivity, bool)
        or not isinstance(sensitivity, (int, float))
        or not math.isfinite(sensitivity)
        or sensitivity < 0
    ):
        raise ValueError("Hydro-share sensitivity must be a nonnegative percentage.")

    bounds = section.get("hydro_share_bounds_percent")
    if bounds != [0, 100]:
        raise ValueError("Hydro-share bounds must be [0, 100] percent.")

    expectation = section.get("nea_import_expectation")
    if (
        not isinstance(expectation, dict)
        or expectation.get("status") != "user_reported_unverified"
    ):
        raise ValueError("NEA import expectation must be marked user_reported_unverified.")
    for key in ("peak_hours_mw", "around_the_clock_mw"):
        interval = expectation.get(key)
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in interval
            )
            or interval[0] > interval[1]
        ):
            raise ValueError(f"{key} must be a valid two-value MW range.")

    raw_comparison_months = expectation.get("comparison_months")
    if not isinstance(raw_comparison_months, list) or not raw_comparison_months:
        raise ValueError("NEA comparison months must be a non-empty list.")
    normalized_months: list[str] = []
    for month in raw_comparison_months:
        if not isinstance(month, str):
            raise ValueError("NEA comparison months must be text month labels.")
        normalized_month = month.strip().lower()
        normalized_months.append(
            BS_MONTH_ALIASES.get(normalized_month, normalized_month)
        )
    if any(month not in MONTH_DAYS_2073 for month in normalized_months):
        raise ValueError("NEA comparison months contain an unknown BS month.")

    return {
        "storage_shift_percentages": shift_values,
        "storage_shift_base_percent": base_shift,
        "hydro_share_sensitivity_percentage_points": float(sensitivity),
        "comparison_months": normalized_months,
        "nea_peak_range_mw": [float(value) for value in expectation["peak_hours_mw"]],
        "nea_around_clock_range_mw": [
            float(value) for value in expectation["around_the_clock_mw"]
        ],
        "nea_status": expectation["status"],
    }


def _hydro_supply_for_share(
    index: pd.DatetimeIndex,
    monthly_mean_demand_mw: float,
    hydro_share: float,
    shift_percent: float | None = None,
) -> pd.Series:
    flat_supply = monthly_mean_demand_mw * hydro_share
    if shift_percent is None:
        return pd.Series(flat_supply, index=index, dtype=float)

    # User assumption: move the selected share of total daily hydro energy from
    # the other 19 hours proportionally and split it evenly across 18:00-22:00.
    shift_fraction = shift_percent / 100
    peak_hour_supply = flat_supply * (1 + 24 * shift_fraction / len(STORAGE_HOURS))
    other_hour_supply = flat_supply * (1 - 24 * shift_fraction / (24 - len(STORAGE_HOURS)))
    values = np.where(index.hour.isin(STORAGE_HOURS), peak_hour_supply, other_hour_supply)
    return pd.Series(values, index=index, dtype=float)


def build_hydro_gap_table(
    hourly: pd.DataFrame,
    monthly_balance: dict[str, dict[str, float | str]],
    scenario_config: dict,
) -> pd.DataFrame:
    required = {"demand_scaled_mw", "bs_month"}
    if not required.issubset(hourly.columns):
        raise ValueError(f"Hourly data must contain columns {sorted(required)}.")
    if not isinstance(hourly.index, pd.DatetimeIndex) or hourly.index.tz is None:
        raise ValueError("Hourly data must use timezone-aware timestamps.")
    if hourly.index.has_duplicates:
        raise ValueError("Hourly timestamps must be unique.")

    year_end = (
        pd.Timestamp(MONTH_STARTS_2073["chaitra"])
        + pd.Timedelta(days=MONTH_DAYS_2073["chaitra"])
    ).tz_localize(TIMEZONE)
    in_2073 = hourly.index < year_end
    rows = []

    def append_scenario(
        month_data: pd.DataFrame,
        month: str,
        month_label: str,
        mean_demand_mw: float,
        hydro_share: float,
        scenario: str,
        family: str,
        base_case: bool,
        shift_percent: float | None,
        note: str,
    ) -> None:
        month_index = pd.DatetimeIndex(month_data.index)
        hydro = _hydro_supply_for_share(
            month_index, mean_demand_mw, hydro_share, shift_percent
        )
        demand = month_data["demand_scaled_mw"]
        gap = (demand - hydro).clip(lower=0)
        scenario_rows = pd.DataFrame(
            {
                "timestamp": month_index,
                "bs_month": month,
                "month_label": month_label,
                "demand_scaled_mw": demand.to_numpy(),
                "scenario": scenario,
                "scenario_family": family,
                "is_base_case": base_case,
                "shift_percent": shift_percent,
                "hydro_share_pct": hydro_share * 100,
                "monthly_mean_demand_mw": mean_demand_mw,
                "hydro_supply_mw": hydro.to_numpy(),
                "gap_mw": gap.to_numpy(),
                "scenario_note": note,
            }
        )
        rows.extend(scenario_rows.to_dict("records"))

    for month, values in monthly_balance.items():
        month_mask = in_2073 & hourly["bs_month"].eq(month).to_numpy()
        month_data = hourly.loc[month_mask]
        expected_hours = MONTH_DAYS_2073[month] * 24
        if len(month_data) != expected_hours:
            raise ValueError(
                f"{month} has {len(month_data)} hourly rows; expected {expected_hours}."
            )

        target_demand_gwh = float(values["national_demand_gwh"])
        monthly_mean_demand_mw = target_demand_gwh * 1000 / expected_hours
        hydro_share = (
            float(values["domestic_generation_gwh"]) - float(values["export_gwh"])
        ) / target_demand_gwh
        month_label = str(values["month_label"])

        append_scenario(
            month_data,
            month,
            month_label,
            monthly_mean_demand_mw,
            hydro_share,
            "A_flat_base",
            "A",
            True,
            None,
            "Flat hydro supply at hydro share times monthly mean demand.",
        )
        for shift_percent in scenario_config["storage_shift_percentages"]:
            base_case = shift_percent == scenario_config["storage_shift_base_percent"]
            suffix = f"{shift_percent:g}pct"
            if base_case:
                suffix += "_base"
            elif shift_percent >= 20:
                suffix += "_optimistic"
            append_scenario(
                month_data,
                month,
                month_label,
                monthly_mean_demand_mw,
                hydro_share,
                f"B_storage_{suffix}",
                "B",
                base_case,
                shift_percent,
                (
                    "User-estimated 10% base shift; not a measured Nepal figure."
                    if base_case
                    else "Storage shift sensitivity; not a measured Nepal figure."
                ),
            )

        sensitivity = (
            scenario_config["hydro_share_sensitivity_percentage_points"] / 100
        )
        for direction, adjusted_share in (
            ("minus", max(0.0, hydro_share - sensitivity)),
            ("plus", min(1.0, hydro_share + sensitivity)),
        ):
            append_scenario(
                month_data,
                month,
                month_label,
                monthly_mean_demand_mw,
                adjusted_share,
                f"C_hydro_share_{direction}_10pp",
                "C",
                False,
                None,
                "Flat hydro sensitivity; hydro share bounded to 0-100%.",
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No dry-month hourly data was available for scenarios.")
    return result.sort_values(["timestamp", "scenario"]).reset_index(drop=True)


def _summarize_period(
    data: pd.DataFrame,
    period_type: str,
    period: str,
    nea_peak_range: list[float],
    nea_around_clock_range: list[float],
) -> list[dict]:
    rows = []
    for scenario, group in data.groupby("scenario", sort=False):
        valid = group.loc[group["gap_mw"].notna()]
        window = valid.loc[valid["timestamp"].dt.hour.isin(GAP_WINDOW_HOURS)]
        peak_window = valid.loc[
            valid["timestamp"].dt.hour.isin(PEAK_COMPARISON_HOURS)
        ]
        gap_energy_gwh = valid["gap_mw"].sum() / 1000
        window_energy_gwh = window["gap_mw"].sum() / 1000
        mean_gap_mw = valid["gap_mw"].mean()
        peak_gap_mw = valid["gap_mw"].max()
        peak_window_gap_mw = window["gap_mw"].max()
        peak_comparison_gap_mw = peak_window["gap_mw"].max()
        share_in_window = (
            window_energy_gwh / gap_energy_gwh * 100
            if gap_energy_gwh > 0
            else np.nan
        )

        peak_comparison = ""
        average_comparison = ""
        note = ""
        peak_low = peak_high = average_low = average_high = np.nan
        if period_type == "comparison_period":
            peak_low, peak_high = nea_peak_range
            average_low, average_high = nea_around_clock_range
            peak_comparison = _range_comparison(
                peak_comparison_gap_mw, peak_low, peak_high
            )
            average_comparison = _range_comparison(
                mean_gap_mw, average_low, average_high
            )
            note = (
                f"Maximum 18:00-22:00 gap is {peak_comparison} the user-reported "
                f"{peak_low:g}-{peak_high:g} MW peak-import range; average gap "
                f"is {average_comparison} the user-reported "
                f"{average_low:g}-{average_high:g} MW around-the-clock range. "
                "Demand-minus-hydro gap is not identical to actual imports."
            )

        first = group.iloc[0]
        monthly_shares = group.drop_duplicates("bs_month")
        share_weights = (
            monthly_shares["monthly_mean_demand_mw"]
            * monthly_shares["bs_month"].map(MONTH_DAYS_2073)
            * 24
        )
        weighted_hydro_share_pct = np.average(
            monthly_shares["hydro_share_pct"], weights=share_weights
        )
        rows.append(
            {
                "period_type": period_type,
                "period": period,
                "scenario": scenario,
                "scenario_family": first["scenario_family"],
                "is_base_case": first["is_base_case"],
                "shift_percent": first["shift_percent"],
                "hydro_share_pct": weighted_hydro_share_pct,
                "demand_observed_hours": len(valid),
                "gap_peak_mw": peak_gap_mw,
                "gap_peak_17_22_mw": peak_window_gap_mw,
                "gap_peak_18_22_mw": peak_comparison_gap_mw,
                "gap_energy_gwh": gap_energy_gwh,
                "gap_17_22_energy_gwh": window_energy_gwh,
                "gap_17_22_share_pct": share_in_window,
                "mean_hourly_gap_mw": mean_gap_mw,
                "nea_peak_import_min_mw": peak_low,
                "nea_peak_import_max_mw": peak_high,
                "nea_round_clock_import_min_mw": average_low,
                "nea_round_clock_import_max_mw": average_high,
                "peak_18_22_gap_comparison": peak_comparison,
                "mean_gap_comparison": average_comparison,
                "nea_comparison_note": note,
            }
        )
    return rows


def _range_comparison(value: float, lower: float, upper: float) -> str:
    if value < lower:
        return "below"
    if value > upper:
        return "above"
    return "within"


def summarize_hydro_gaps(
    gap_table: pd.DataFrame,
    scenario_config: dict,
    monthly_balance: dict[str, dict[str, float | str]],
) -> pd.DataFrame:
    rows = []
    for month in monthly_balance:
        month_data = gap_table.loc[gap_table["bs_month"] == month]
        month_label = str(monthly_balance[month]["month_label"])
        rows.extend(
            _summarize_period(
                month_data,
                "month",
                month_label,
                scenario_config["nea_peak_range_mw"],
                scenario_config["nea_around_clock_range_mw"],
            )
        )

    all_dry = gap_table
    rows.extend(
        _summarize_period(
            all_dry,
            "dry_season",
            "Mangsir-Chaitra",
            scenario_config["nea_peak_range_mw"],
            scenario_config["nea_around_clock_range_mw"],
        )
    )
    comparison_months = scenario_config["comparison_months"]
    comparison = gap_table.loc[gap_table["bs_month"].isin(comparison_months)]
    comparison_label = "Magh-Falgun (mid-Jan to mid-Mar)"
    rows.extend(
        _summarize_period(
            comparison,
            "comparison_period",
            comparison_label,
            scenario_config["nea_peak_range_mw"],
            scenario_config["nea_around_clock_range_mw"],
        )
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dry-season hydro and demand-minus-hydro gap scenarios."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/demand_hourly_scaled_2025_26.csv"),
        help="Step 4 hourly output containing demand_scaled_mw and bs_month.",
    )
    parser.add_argument(
        "--gap-output",
        type=Path,
        default=Path("data/dry_season_hydro_gap_2025_26.csv"),
        help="Long-form hourly hydro/gap scenario table.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("report/hydro_gap_scenario_summary_2025_26.csv"),
        help="Monthly, dry-season, and comparison-period summary.",
    )
    args = parser.parse_args()

    hourly = read_hourly_demand(args.input)
    monthly_balance = load_monthly_balance()
    scenario_config = load_scenario_config()
    gap_table = build_hydro_gap_table(hourly, monthly_balance, scenario_config)
    summary = summarize_hydro_gaps(gap_table, scenario_config, monthly_balance)
    args.gap_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    gap_table.to_csv(args.gap_output, index=False)
    summary.to_csv(args.summary_output, index=False)

    print("Gap comparison for Magh-Falgun:")
    comparison = summary.loc[summary["period_type"] == "comparison_period"]
    print(
        comparison[
            [
                "scenario",
                "gap_peak_mw",
                "gap_peak_18_22_mw",
                "mean_hourly_gap_mw",
                "gap_energy_gwh",
                "gap_17_22_share_pct",
                "peak_18_22_gap_comparison",
                "mean_gap_comparison",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )
    print("\nSaved:")
    print(args.gap_output)
    print(args.summary_output)


if __name__ == "__main__":
    main()
