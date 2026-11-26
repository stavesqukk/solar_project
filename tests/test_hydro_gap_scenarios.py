import unittest

import numpy as np
import pandas as pd

from energy.hydro_gap_scenarios import (
    build_hydro_gap_table,
    load_scenario_config,
    summarize_hydro_gaps,
)
from energy.scale_recent_demand import load_monthly_balance
from energy.load_demand import MONTH_DAYS_2073


class HydroGapScenarioTests(unittest.TestCase):
    def test_scenario_config_marks_10_percent_base_and_user_reported_ranges(self):
        config = load_scenario_config()

        self.assertEqual(config["storage_shift_percentages"], [0, 5, 10, 20, 30])
        self.assertEqual(config["storage_shift_base_percent"], 10)
        self.assertEqual(config["nea_peak_range_mw"], [800, 900])
        self.assertEqual(config["nea_around_clock_range_mw"], [450, 500])

    def test_storage_preserves_daily_energy_and_gap_keeps_missing_hours_missing(self):
        index = pd.date_range(
            "2016-11-16 00:00",
            periods=sum(MONTH_DAYS_2073[month] for month in (
                "mangsir",
                "paush",
                "magh",
                "falgun",
                "chaitra",
            )) * 24,
            freq="h",
            tz="Asia/Kathmandu",
            name="timestamp",
        )
        demand = np.full(len(index), 2000.0)
        demand[18] = np.nan
        month_labels = [
            month
            for month in ("mangsir", "paush", "magh", "falgun", "chaitra")
            for _ in range(MONTH_DAYS_2073[month] * 24)
        ]
        hourly = pd.DataFrame(
            {
                "demand_scaled_mw": demand,
                "bs_month": month_labels,
            },
            index=index,
        )
        balances = load_monthly_balance()
        config = load_scenario_config()

        gap_table = build_hydro_gap_table(hourly, balances, config)
        first_day = gap_table.loc[
            (gap_table["timestamp"].dt.date == index[0].date())
        ]
        scenario_hydro = first_day.pivot(
            index="timestamp", columns="scenario", values="hydro_supply_mw"
        )
        pd.testing.assert_series_equal(
            scenario_hydro["A_flat_base"],
            scenario_hydro["B_storage_0pct"],
            check_names=False,
        )
        base_daily_energy = scenario_hydro["A_flat_base"].sum()
        for shift in (0, 5, 10, 20, 30):
            scenario = f"B_storage_{shift}pct"
            if shift == 10:
                scenario += "_base"
            elif shift >= 20:
                scenario += "_optimistic"
            self.assertAlmostEqual(
                scenario_hydro[scenario].sum(), base_daily_energy, places=8
            )

        daily_hydro_energy = (
            gap_table.assign(local_date=gap_table["timestamp"].dt.date)
            .groupby(["local_date", "scenario"])["hydro_supply_mw"]
            .sum()
            .unstack()
        )
        for shift in (0, 5, 10, 20, 30):
            scenario = f"B_storage_{shift}pct"
            if shift == 10:
                scenario += "_base"
            elif shift >= 20:
                scenario += "_optimistic"
            np.testing.assert_allclose(
                daily_hydro_energy[scenario],
                daily_hydro_energy["A_flat_base"],
                rtol=0,
                atol=1e-8,
            )

        self.assertEqual(len(gap_table), len(index) * 8)
        missing_hour = gap_table.loc[
            (gap_table["timestamp"] == index[18])
        ]
        self.assertTrue(missing_hour["gap_mw"].isna().all())
        self.assertTrue(missing_hour["hydro_supply_mw"].notna().all())
        high_share = gap_table.loc[
            gap_table["scenario"] == "C_hydro_share_plus_10pp", "hydro_share_pct"
        ]
        self.assertLessEqual(high_share.max(), 100.0)

        summary = summarize_hydro_gaps(gap_table, config, balances)
        comparison = summary.loc[
            summary["period_type"] == "comparison_period"
        ]
        self.assertEqual(len(comparison), 8)
        expected_weighted_share = (
            ((864 - 1) / 1125 * 1125) + ((801 - 13) / 1067 * 1067)
        ) / (1125 + 1067) * 100
        flat_comparison = comparison.loc[
            comparison["scenario"] == "A_flat_base"
        ].iloc[0]
        self.assertAlmostEqual(
            flat_comparison["hydro_share_pct"], expected_weighted_share
        )
        self.assertTrue(
            comparison["nea_comparison_note"]
            .str.contains("not identical to actual imports")
            .all()
        )


if __name__ == "__main__":
    unittest.main()
