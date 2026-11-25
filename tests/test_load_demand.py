import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from energy.load_demand import build_hourly, clean_readings, load_month


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


if __name__ == "__main__":
    unittest.main()
