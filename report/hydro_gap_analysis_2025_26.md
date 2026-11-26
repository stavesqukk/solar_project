# Dry-season demand-minus-hydro gap analysis

## What this analysis estimates

This is a scenario study, not a record of actual electricity shortages or
imports. It compares an hourly demand profile shaped from the cleaned 2073 BS
load data and scaled to the user-provided FY 2025/26 monthly demand balances
with modeled hydro supply. The balances are unverified and should be checked
against the NEA annual report.

For each observed hour, the modeled residual is
`max(scaled demand - modeled hydro supply, 0)`. Reported gap energy sums those
hourly residuals; missing demand readings are not filled. A positive residual
does not prove that the system was short by that amount: this simplified
comparison does not model all supply sources, dispatch, network constraints,
or actual imports.

## Magh-Falgun comparison

Magh-Falgun is used as an approximation of the reported mid-January to
mid-March period. The summary contains 1,416 observed hourly demand values.
The import ranges sometimes compared with these results (800-900 MW at peak
and 450-500 MW around the clock) are user-reported, unverified figures. The
comparison is illustrative only because a modeled demand-minus-hydro residual
is not the same measure as actual imports.

| Scenario | Maximum gap, any hour (MW) | Maximum gap, 18:00-22:00 (MW) | Mean hourly gap (MW) | Gap energy (GWh) | Gap energy share, 17:00-22:00 |
|---|---:|---:|---:|---:|---:|
| A - flat hydro base | 1,220.89 | 1,220.89 | 388.14 | 549.61 | 43.77% |
| B - 0% shift | 1,220.89 | 1,220.89 | 388.14 | 549.61 | 43.77% |
| B - 5% shift | 1,269.18 | 958.22 | 385.28 | 545.55 | 30.19% |
| B - 10% shift (base) | 1,347.49 | 695.55 | 399.94 | 566.31 | 19.18% |
| B - 20% shift (optimistic) | 1,504.12 | 170.22 | 464.66 | 657.96 | 6.78% |
| B - 30% shift (optimistic) | 1,660.74 | 0.00 | 579.83 | 821.04 | 6.25% |
| C -10 pp hydro share | 1,378.30 | 1,378.30 | 537.33 | 760.87 | 38.81% |
| C +10 pp hydro share | 1,072.69 | 1,072.69 | 263.33 | 372.87 | 49.97% |

### Reading the base case

Compared with Scenario A, the 10% storage-shift base case lowers the
18:00-22:00 maximum gap from 1,220.89 MW to 695.55 MW (about 43%). But its
maximum gap across all hours rises to 1,347.49 MW, occurring at 08:00 on
2017-02-01 in the historical-shaped series. Its estimated gap energy also
rises by 16.70 GWh, from 549.61 to 566.31 GWh, and the mean hourly gap rises
from 388.14 to 399.94 MW.

This is the trade-off represented by the model: Scenario B moves the selected
share of each day's hydro energy into 18:00-22:00 and removes it
proportionally from the other 19 hours, preserving daily hydro energy. The
evening residual can fall while residuals at other hours grow. Since gaps are
clipped at zero, extra modeled hydro in an hour with no residual cannot
mathematically cancel a residual in another hour. The 10% shift is the user's
rough estimate, not a measured operating pattern.

The 20% and 30% cases are optimistic sensitivities, not forecasts. In the
model, they reduce the Magh-Falgun evening maximum further, to 170.22 MW and
zero, while increasing the maximum gap at any hour and the total residual
energy. This is a warning against interpreting an evening-only reduction as
an improvement across the whole day.

## Comparison with the reported import ranges

- Scenario A's 18:00-22:00 maximum gap (1,220.89 MW) is above the
  user-reported 800-900 MW peak-import range; its mean hourly gap
  (388.14 MW) is below the reported 450-500 MW around-the-clock range.
- The 10% base's evening maximum (695.55 MW) is below that peak-import range;
  its mean hourly gap (399.94 MW) is below the reported around-the-clock range.
- These comparisons do not validate either the model or the reported import
  figures. Imports are only one part of a real system balance, while this
  estimate is a simplified residual based on assumed hydro supply.

## Figures and reproducibility

- [Hourly demand, modeled hydro supply, and gap profiles](hydro_demand_gap_profiles_2025_26.png)
  compares Scenarios A and B's 10% base across the dry season.
- [Monthly scenario heatmaps](hydro_gap_monthly_scenarios_2025_26.png) show
  maximum hourly gap, maximum 18:00-22:00 gap, and observed gap energy for
  each dry month and scenario.
- [Scenario summary data](hydro_gap_scenario_summary_2025_26.csv) contains
  monthly, dry-season, and Magh-Falgun metrics.
- Recreate the data and figures from the project root with
  `.\.venv\Scripts\python.exe -m energy.hydro_gap_scenarios`.

The hourly shape comes from historical 2073 BS demand and is scaled using
user-provided, unverified FY 2025/26 monthly balances. Scenario hydro output
is modeled, not measured current supply. See [ASSUMPTIONS.md](../ASSUMPTIONS.md)
for the input data, model definitions, missing-data handling, and assumptions.
