import unittest

import numpy as np
import pandas as pd

from energy.seasonal_analysis import (
    add_bs_month,
    build_daily_summary,
    build_group_summary,
    select_bs_2073,
)


class SeasonalAnalysisTests(unittest.TestCase):
    def test_bs_month_changes_at_local_midnight_boundary(self):
        index = pd.DatetimeIndex(
            [
                "2016-05-13 23:00",
                "2016-05-14 00:00",
                "2017-04-14 00:00",
            ],
            tz="Asia/Kathmandu",
            name="timestamp",
        )
        hourly = pd.DataFrame(
            {"demand_mw": [700.0, 710.0, 720.0], "season": ["shoulder", "wet", "shoulder"]},
            index=index,
        )

        mapped = add_bs_month(hourly)
        year_data, excluded = select_bs_2073(mapped)

        self.assertEqual(mapped["bs_month"].tolist(), ["baisakh", "jestha", "baisakh"])
        self.assertEqual(len(year_data), 2)
        self.assertEqual(excluded, 1)

    def test_daily_and_group_metrics_use_observed_hours_and_complete_days(self):
        full_index = pd.date_range(
            "2016-04-13 01:00",
            "2017-04-14 00:00",
            freq="h",
            tz="Asia/Kathmandu",
            name="timestamp",
        )
        values = np.full(len(full_index), np.nan)
        first_day = full_index.get_loc(pd.Timestamp("2016-04-16 00:00", tz="Asia/Kathmandu"))
        values[first_day : first_day + 24] = 700.0
        values[first_day + 24 : first_day + 48] = 700.0
        values[first_day + 3] = 500.0
        values[first_day + 17] = 800.0
        values[first_day + 18] = 1000.0
        values[first_day + 19] = 900.0
        values[first_day + 20] = 850.0
        values[first_day + 21] = 800.0
        values[first_day + 24 + 3] = 600.0
        values[first_day + 24 + 17] = 750.0
        values[first_day + 24 + 18] = 950.0
        values[first_day + 24 + 19] = 900.0
        values[first_day + 24 + 20] = 850.0
        values[first_day + 24 + 21] = 800.0
        hourly = pd.DataFrame(
            {
                "demand_mw": values,
                "season": ["shoulder"] * len(full_index),
                "bs_month": ["baisakh"] * len(full_index),
                "hour_of_day": full_index.hour,
            },
            index=full_index,
        )

        daily = build_daily_summary(hourly)
        summary = build_group_summary(hourly, daily)
        month = summary.loc[
            (summary["period_type"] == "month")
            & (summary["period"] == "Baisakh")
        ].iloc[0]

        day_index = daily.index[daily["date"] == "2016-04-16"][0]
        self.assertEqual(len(daily), 366)
        self.assertEqual(daily.loc[day_index, "observed_hours"], 24)
        self.assertEqual(daily.loc[day_index, "minimum_hour"], 3)
        self.assertEqual(daily.loc[day_index, "maximum_hour"], 18)
        self.assertEqual(month["observed_hours"], 48)
        self.assertAlmostEqual(
            month["observed_energy_gwh"], np.nansum(values) / 1000
        )
        self.assertAlmostEqual(month["mean_midday_11_14_mw"], 700.0)
        self.assertAlmostEqual(month["peak_mean_evening_18_21_mw"], 975.0)
        self.assertAlmostEqual(
            month["evening_ratio_peak_to_midday"], 975.0 / 700.0
        )
        self.assertAlmostEqual(month["mean_ramp_17_to_18_mw"], 200.0)
        self.assertEqual(month["complete_saturdays"], 1)
        self.assertEqual(month["complete_weekdays"], 1)

    def test_daily_summary_keeps_partial_days_visible(self):
        index = pd.date_range(
            "2016-04-13 01:00",
            "2017-04-14 00:00",
            freq="h",
            tz="Asia/Kathmandu",
            name="timestamp",
        )
        values = np.full(len(index), np.nan)
        first_hour = index.get_loc(
            pd.Timestamp("2016-04-16 01:00", tz="Asia/Kathmandu")
        )
        values[first_hour : first_hour + 23] = 700.0
        hourly = pd.DataFrame(
            {
                "demand_mw": values,
                "season": ["shoulder"] * len(index),
                "bs_month": ["baisakh"] * len(index),
                "hour_of_day": index.hour,
            },
            index=index,
        )

        daily = build_daily_summary(hourly)
        day_index = daily.index[daily["date"] == "2016-04-16"][0]

        self.assertEqual(daily.loc[day_index, "observed_hours"], 23)
        self.assertFalse(daily.loc[day_index, "complete_day"])
        self.assertTrue(pd.notna(daily.loc[day_index, "minimum_hour"]))


if __name__ == "__main__":
    unittest.main()
