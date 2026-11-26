import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from energy.forecast_demand import (
    build_metrics,
    build_error_breakdown,
    calculate_metrics,
    create_forecasts,
    plot_forecast_error_analysis,
)


class DemandForecastTests(unittest.TestCase):
    def test_forecasts_use_prior_day_and_prior_week_on_chronological_holdout(self):
        index = pd.date_range("2017-01-01", periods=24 * 50, freq="h", tz="Asia/Kathmandu")
        demand = pd.Series(np.arange(len(index), dtype=float), index=index)

        forecasts, split_at = create_forecasts(demand)

        self.assertEqual(split_at, index[int(len(index) * 0.8)])
        self.assertTrue((forecasts.index >= split_at).all())
        timestamp = forecasts.index[0]
        self.assertEqual(
            forecasts.loc[timestamp, "naive_24h_forecast_mw"],
            demand.loc[timestamp - pd.Timedelta(hours=24)],
        )
        self.assertEqual(
            forecasts.loc[timestamp, "naive_168h_forecast_mw"],
            demand.loc[timestamp - pd.Timedelta(hours=168)],
        )

    def test_weekly_forecast_does_not_bridge_missing_calendar_hours(self):
        index = pd.date_range("2017-01-01", periods=1000, freq="h")
        demand = pd.Series(np.arange(len(index), dtype=float), index=index)
        demand.iloc[800:820] = np.nan

        forecasts, _ = create_forecasts(demand)

        self.assertTrue(
            ((forecasts.index >= index[800]) & (forecasts.index < index[968])).any()
        )
        self.assertFalse(
            ((forecasts.index >= index[968]) & (forecasts.index < index[988])).any()
        )
        self.assertIn(index[988], forecasts.index)

    def test_metrics_report_mae_rmse_and_mape(self):
        actual = pd.Series([100.0, 200.0])
        predicted = pd.Series([110.0, 220.0])

        metrics = calculate_metrics(actual, predicted)

        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["mae_mw"], 15.0)
        self.assertAlmostEqual(metrics["rmse_mw"], np.sqrt(250))
        self.assertAlmostEqual(metrics["mape_pct"], 10.0)

    def test_dry_season_metrics_follow_configured_bs_months(self):
        index = pd.DatetimeIndex(
            [
                "2016-04-13 18:00",
                "2016-05-14 18:00",
                "2016-11-16 18:00",
            ],
            tz="Asia/Kathmandu",
            name="timestamp",
        )
        forecasts = pd.DataFrame(
            {
                "actual_demand_mw": [1000.0, 1000.0, 1000.0],
                "naive_24h_forecast_mw": [900.0, 900.0, 900.0],
                "naive_168h_forecast_mw": [800.0, 800.0, 800.0],
            },
            index=index,
        )

        metrics = build_metrics(forecasts)
        dry_metrics = metrics.loc[metrics["period"] == "dry_season"]

        self.assertEqual(dry_metrics["n"].tolist(), [1, 1])

    def test_error_breakdown_summarizes_month_hour_and_bias_and_plots(self):
        index = pd.DatetimeIndex(
            [
                "2016-04-13 18:00",
                "2016-05-14 18:00",
                "2016-11-16 19:00",
            ],
            tz="Asia/Kathmandu",
            name="timestamp",
        )
        forecasts = pd.DataFrame(
            {
                "actual_demand_mw": [1000.0, 1000.0, 1000.0],
                "naive_24h_forecast_mw": [900.0, 1100.0, 1000.0],
                "naive_168h_forecast_mw": [800.0, 900.0, 1000.0],
            },
            index=index,
        )

        breakdown = build_error_breakdown(forecasts)
        baisakh_yesterday = breakdown.loc[
            (breakdown["period_type"] == "bs_month")
            & (breakdown["period"] == "baisakh")
            & (breakdown["model"] == "same_hour_yesterday")
        ].iloc[0]
        hour_18_yesterday = breakdown.loc[
            (breakdown["period_type"] == "hour_of_day")
            & (breakdown["period"] == 18)
            & (breakdown["model"] == "same_hour_yesterday")
        ].iloc[0]

        self.assertEqual(baisakh_yesterday["n"], 1)
        self.assertEqual(baisakh_yesterday["mean_error_mw"], -100.0)
        self.assertEqual(hour_18_yesterday["n"], 2)
        self.assertEqual(hour_18_yesterday["mae_mw"], 100.0)
        self.assertEqual(hour_18_yesterday["mean_error_mw"], 0.0)

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "forecast_errors.png"
            plot_forecast_error_analysis(breakdown, output_path)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
