import unittest

import numpy as np
import pandas as pd

from energy.forecast_demand import calculate_metrics, create_forecasts


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


if __name__ == "__main__":
    unittest.main()
