import unittest

import numpy as np
import pandas as pd

from energy.scale_recent_demand import load_monthly_balance, scale_dry_months


class RecentDemandScalingTests(unittest.TestCase):
    def test_config_has_unverified_user_balances_and_derived_shares(self):
        balances = load_monthly_balance()

        self.assertEqual(
            set(balances),
            {"mangsir", "paush", "magh", "falgun", "chaitra"},
        )
        self.assertEqual(balances["paush"]["month_label"], "Poush")
        self.assertEqual(balances["mangsir"]["national_demand_gwh"], 1065.0)

    def test_scale_keeps_original_values_and_does_not_fill_missing_or_other_months(self):
        index = pd.DatetimeIndex(
            [
                "2016-11-20 01:00",
                "2016-11-20 02:00",
                "2016-11-20 03:00",
                "2016-04-13 01:00",
                "2017-04-14 00:00",
            ],
            tz="Asia/Kathmandu",
            name="timestamp",
        )
        hourly = pd.DataFrame(
            {
                "demand_mw": [500.0, np.nan, 1000.0, 750.0, 900.0],
                "bs_month": ["mangsir", "mangsir", "mangsir", "baisakh", "baisakh"],
            },
            index=index,
        )
        balance = {
            "mangsir": {
                "month_label": "Mangsir",
                "domestic_generation_gwh": 2.5,
                "import_gwh": 0.75,
                "export_gwh": 1.0,
                "national_demand_gwh": 3.0,
                "source_status": "unverified_user_provided",
            }
        }

        scaled, summary = scale_dry_months(hourly, balance)

        self.assertTrue(scaled["demand_mw"].equals(hourly["demand_mw"]))
        self.assertEqual(scaled["demand_scaled_mw"].iloc[0], 1000.0)
        self.assertTrue(pd.isna(scaled["demand_scaled_mw"].iloc[1]))
        self.assertEqual(scaled["demand_scaled_mw"].iloc[2], 2000.0)
        self.assertTrue(scaled["demand_scaled_mw"].iloc[3:].isna().all())
        self.assertAlmostEqual(summary.loc[0, "scale_multiplier"], 2.0)
        self.assertEqual(summary.loc[0, "balance_status"], "unverified_user_provided")
        self.assertAlmostEqual(summary.loc[0, "hydro_share_pct"], 50.0)
        self.assertAlmostEqual(summary.loc[0, "import_share_pct"], 25.0)
        self.assertAlmostEqual(summary.loc[0, "gap_share_pct"], 50.0)
        self.assertAlmostEqual(summary.loc[0, "scaled_observed_energy_gwh"], 3.0)


if __name__ == "__main__":
    unittest.main()
