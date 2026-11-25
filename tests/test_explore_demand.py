import unittest

import numpy as np
import pandas as pd

from energy.explore_demand import (
    add_calendar_and_lag_features,
    build_hourly_profiles,
    build_monthly_summary,
    classify_season,
)


class DemandExplorationTests(unittest.TestCase):
    def test_season_categories_distinguish_project_dry_and_monsoon(self):
        self.assertEqual(classify_season(3), "project_dry")
        self.assertEqual(classify_season(7), "monsoon")
        self.assertEqual(classify_season(5), "transition")
        self.assertEqual(classify_season(10), "transition")

    def test_feature_table_builds_time_and_past_demand_features(self):
        index = pd.date_range(
            "2016-05-27", periods=48, freq="h", tz="Asia/Kathmandu"
        )
        hourly = pd.DataFrame(
            {"demand_mw": np.arange(48, dtype=float)},
            index=index,
        )

        features = add_calendar_and_lag_features(hourly)

        self.assertTrue(features["is_nepal_weekend"].loc["2016-05-28"].all())
        self.assertEqual(features["demand_lag_24h_mw"].iloc[24], 0.0)
        self.assertEqual(features["demand_lag_168h_mw"].notna().sum(), 0)
        self.assertEqual(features["previous_24h_mean_mw"].iloc[24], 11.5)

    def test_monthly_tables_retain_missing_hour_slots(self):
        index = pd.date_range(
            "2016-06-01", periods=48, freq="h", tz="Asia/Kathmandu"
        )
        hourly = pd.DataFrame({"demand_mw": [np.nan] * 48}, index=index)
        features = add_calendar_and_lag_features(hourly)

        summary = build_monthly_summary(features)
        profiles = build_hourly_profiles(
            features, ["month", "season_type"]
        )

        self.assertEqual(summary.loc[0, "observed_hours"], 0)
        self.assertEqual(summary.loc[0, "expected_hours"], 720)
        self.assertEqual(summary.loc[0, "missing_hours"], 720)
        self.assertEqual(len(profiles), 24)
        self.assertTrue(profiles["mean_demand_mw"].isna().all())
        self.assertTrue((profiles["timestamp_slots_in_series"] == 2).all())

    def test_monthly_coverage_counts_dates_outside_the_dataset_span(self):
        index = pd.date_range(
            "2016-04-13", periods=48, freq="h", tz="Asia/Kathmandu"
        )
        hourly = pd.DataFrame({"demand_mw": np.ones(48)}, index=index)

        summary = build_monthly_summary(add_calendar_and_lag_features(hourly))

        self.assertEqual(summary.loc[0, "expected_hours"], 720)
        self.assertEqual(summary.loc[0, "observed_hours"], 48)
        self.assertAlmostEqual(summary.loc[0, "coverage_pct"], 48 / 720 * 100)


if __name__ == "__main__":
    unittest.main()
