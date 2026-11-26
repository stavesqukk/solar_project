import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from energy.load_demand import (
    add_season_column,
    build_hourly,
    clean_readings,
    load_month,
    load_season_months,
)


class DemandLoaderTests(unittest.TestCase):
    def test_month_loader_excludes_day_labels_outside_bs_calendar_month(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ashoj.csv"
            path.write_text(
                "A,B\nDay,1:00\n30,700\n31,710\n",
                encoding="utf-8",
            )

            readings, invalid_days = load_month(path)

        self.assertEqual(invalid_days, [31])
        self.assertEqual(
            readings.index.min(), pd.Timestamp("2016-10-16 01:00")
        )

    def test_month_loader_accepts_header_first_decimal_clock_values(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "SHRAWAN.csv"
            path.write_text(
                "Day,0.04,0.96,0.00\n"
                "1.00,700,800,900\n",
                encoding="utf-8",
            )

            readings, invalid_days = load_month(path)

        self.assertEqual(invalid_days, [])
        self.assertEqual(
            readings.index.tolist(),
            [
                pd.Timestamp("2016-07-16 01:00"),
                pd.Timestamp("2016-07-16 23:00"),
                pd.Timestamp("2016-07-17 00:00"),
            ],
        )

    def test_month_loader_uses_full_jestha_calendar_length(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "jestha.csv"
            path.write_text(
                "Day,1:00\n1,700\n32,800\n",
                encoding="utf-8",
            )

            readings, invalid_days = load_month(path)

        self.assertEqual(invalid_days, [])
        self.assertEqual(
            readings.index.tolist(),
            [
                pd.Timestamp("2016-05-14 01:00"),
                pd.Timestamp("2016-06-14 01:00"),
            ],
        )

    def test_hourly_resampling_preserves_missing_calendar_hours(self):
        index = pd.DatetimeIndex(
            ["2016-04-13 00:00", "2016-04-13 03:00"], name="timestamp"
        )
        readings = pd.DataFrame(
            {
                "demand_mw": [700.0, 800.0],
                "flagged": [False, False],
                "demand_clean_mw": [700.0, 800.0],
            },
            index=index,
        )

        hourly = build_hourly(readings)

        self.assertEqual(len(hourly), 4)
        self.assertTrue(hourly["demand_mw"].iloc[1:3].isna().all())
        self.assertEqual(hourly["readings_in_hour"].iloc[1], 0)

    def test_outlier_interpolation_does_not_cross_long_data_gap(self):
        index = pd.DatetimeIndex(
            [
                "2016-04-13 00:00",
                "2016-08-17 00:00",
                "2016-12-16 00:00",
            ],
            name="timestamp",
        )
        readings = pd.DataFrame({"demand_mw": [700.0, 100.0, 800.0]}, index=index)

        cleaned = clean_readings(readings)

        self.assertTrue(cleaned["flagged"].iloc[1])
        self.assertTrue(pd.isna(cleaned["demand_clean_mw"].iloc[1]))

    def test_config_defines_each_bs_month_in_exactly_one_season(self):
        season_by_month = load_season_months()

        self.assertEqual(len(season_by_month), 12)
        self.assertEqual(
            {month for month, season in season_by_month.items() if season == "dry"},
            {"mangsir", "paush", "magh", "falgun", "chaitra"},
        )
        self.assertEqual(
            {
                month
                for month, season in season_by_month.items()
                if season == "shoulder"
            },
            {"baisakh", "kartik"},
        )
        self.assertEqual(
            {month for month, season in season_by_month.items() if season == "wet"},
            {"jestha", "ashadh", "shrawan", "bhadra", "ashoj"},
        )

    def test_season_labels_follow_bs_month_at_midnight_boundaries(self):
        index = pd.DatetimeIndex(
            [
                "2016-05-13 23:00",
                "2016-05-14 00:00",
                "2016-11-15 23:00",
                "2016-11-16 00:00",
                "2017-04-13 23:00",
                "2017-04-14 00:00",
            ],
            tz="Asia/Kathmandu",
            name="timestamp",
        )
        hourly = pd.DataFrame({"demand_mw": range(len(index))}, index=index)

        result = add_season_column(hourly, load_season_months())

        self.assertEqual(
            result["season"].tolist(),
            ["shoulder", "wet", "shoulder", "dry", "dry", "shoulder"],
        )


if __name__ == "__main__":
    unittest.main()
