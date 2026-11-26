# Assumptions

- The NEA 2073 monthly files now cover all twelve Nepali months in the repository.
- BS month starts and lengths were cross-checked against two published 2073 BS
  calendar references: [UseMiti](https://usemiti.com/month-dates/2073) and
  [Calendaw](https://www.calendaw.com/bs/2073). Baisakh 1 = 2016-04-13,
  Jestha 1 = 2016-05-14, Asadh 1 = 2016-06-15, and Shrawan 1 = 2016-07-16.
  The full mapping is in `energy/load_demand.py`; the NEA file's own date
  semantics have not been independently confirmed.
- The source CSVs include Asoj Day 31 and Poush Day 30, which exceed those
  calendar months' lengths; these rows are excluded.
- Existing source files have an extra spreadsheet header row; the newly added
  Jestha, Asadh, and Shrawan files do not. Shrawan encodes times as fractions of
  a day rounded to the nearest half hour; the loader normalizes these to clock
  times. The source data is missing Jestha Day 32, Mangsir Day 30, and Chaitra
  Day 31, leaving 72 hourly slots missing; these remain missing, not zero-filled.
  Month files should be checked for missing readings rather than assuming
  calendar coverage guarantees every hourly observation exists.
- The user confirmed that a `0:00` reading is the midnight timestamp at the
  start of the following Gregorian day.
- Demand values below 400 MW or above 1800 MW are flagged and then interpolated
  within continuous data segments. Interpolation does not cross gaps longer than
  two hours.
- The cleaned data is interpreted as national demand in MW, although the source
  files do not state units.
- All timestamps are localized to `Asia/Kathmandu` (`UTC+05:45`) and resampled
  to whole hourly values. Missing calendar periods remain present as missing
  values rather than being dropped or filled.
- All twelve months are now included in the regenerated hourly series; the
  former three-month gap is gone. Forecasts are still scored only where both
  required historical lags are available.
- The demand baseline uses a chronological 80/20 split of observed hourly values
  and a rolling one-step-ahead evaluation. The two seasonal-naive methods use
  the same hour 24 hours and 168 hours earlier, respectively.
- Forecast dry-season metrics use the configured Nepali BS month-to-season
  mapping in `config/assumptions.yaml`, not Gregorian month numbers.
- `demand_hourly.season` uses the user-provided groups in
  `config/assumptions.yaml`: dry = Mangsir, Poush, Magh, Falgun, Chaitra;
  shoulder = Baisakh, Kartik; wet = Jestha, Asadh, Shrawan, Bhadra, Ashoj.
  These are assumptions, not season measurements. The user based them on NEA
  saying domestic supply falls below demand from mid-December to mid-April;
  that statement has not been independently verified in this step.
- Season assignment follows the BS month containing each localized timestamp.
  Because `0:00` is assigned to the following Gregorian day, midnight readings
  on a BS month boundary receive the season of the new BS month.
- The Step 3 daily and monthly tables use the observed cleaned 2073 BS demand
  series only. The midnight observation at the start of Baisakh 2074 is excluded
  from 2073 summaries; missing 2073 hours are not filled.
- Reported energy is an observed-hours estimate calculated as the sum of
  hourly MW values divided by 1,000. This assumes each hourly value represents
  the mean MW during that hour; missing hours are not extrapolated.
- Daily minima, maxima, and their typical hours are summarized using complete
  24-observation days only. Partial days remain visible in the daily CSV with
  their observed-hour count. Saturday alone is treated as Nepal's weekend day,
  per the user's instruction.
- Midday means hours 11:00-14:00 inclusive; the evening peak is the highest
  mean hourly demand among 18:00-21:00 inclusive. The evening ratio is that
  peak divided by the midday mean. The evening ramp is the daily 18:00 minus
  17:00 change, averaged where both observations exist.
- The Step 3 charts show empirical average hourly shapes from the historical
  2073 readings, not a current forecast. Any hourly shape used in a future
  projection must be identified as modelled; Step 3 fits no forecast model.
- The FY 2025/26 dry-month domestic generation, import, export, and national
  demand balances in `config/assumptions.yaml` are user-provided and unverified;
  check them against the NEA annual report before treating them as measured
  facts. Hydro share is calculated as (domestic generation - export) / demand,
  import share as import / demand, and gap share as 1 - hydro share.
- Step 4 scales each dry month's observed cleaned 2073 demand profile by
  FY 2025/26 monthly national demand divided by that month's observed 2073
  energy estimate. Missing hours are not filled; scaled values are stored in
  `demand_scaled_mw` beside the original `demand_mw` in
  `data/demand_hourly_scaled_2025_26.csv`. Non-dry months and timestamps
  outside BS 2073 have no scaled value. The resulting multiplier is an
  assumption-based rescaling, not a forecast or an independently verified
  estimate of current hourly demand.
- Run Step 4 from the project root with
  `.\.venv\Scripts\python.exe -m energy.scale_recent_demand`. The script saves
  the monthly shares and scaling multipliers to
  `report/dry_month_scaling_2025_26.csv`.
- Step 5's 10% daily hydro-energy shift into 18:00-22:00 is the user's rough
  estimate, not a measured Nepal figure. The sweep uses 0%, 5%, 10%, 20%, and
  30%; the user identifies 10% as the base and 20-30% as optimistic
  sensitivities. The user's rough calculation is about 20% PRoR share times
  (about 80% of PRoR output in the evening minus the flat profile's 5/24 share),
  or about 12%; lower PRoR shares and limited pondage motivate the 8-10% base
  estimate. The user reports conflicting PRoR estimates (about 20% in a newer
  thesis versus about 50% in an older assessment), short peaking windows, and
  limited dry-season pondage/storage. The shared references
  ([thesis](https://elibrary.tucl.edu.np/JQ99OgQIizUxyjI9nB0on9OyLkqsGIf4/api/core/bitstreams/0c3e1019-5998-42f4-9064-0dcc55814bba/content)
  and
  [renewables assessment](https://fennerschool.anu.edu.au/files/Nepal%20State%20of%20Knowledge%20Report%20-%20Renewables%20&%20PSH.pdf))
  and the other cited claims have not been independently checked here.
  For each day, the selected share of total flat-profile hydro energy is removed
  proportionally from the other 19 hours and added evenly to the five evening
  hours, preserving daily energy.
- Scenario A uses flat hydro supply equal to the monthly hydro share times the
  FY 2025/26 monthly demand divided by the BS month's full calendar hours.
  Scenario C applies plus/minus 10 percentage points to the hydro share on a
  flat profile, bounded to 0-100%. The hourly gap is
  `max(demand_scaled_mw - hydro_supply_mw, 0)`. Missing demand readings remain
  missing gaps; reported gap energy sums observed hourly gaps only.
- The Step 5 comparison uses Magh and Falgun as the approximate mid-January to
  mid-March period. The 800-900 MW peak and 450-500 MW around-the-clock import
  expectation is user-reported and unverified. The peak comparison uses the
  18:00-22:00 gap; the reported 17:00-22:00 gap-energy share includes hourly
  labels 17 through 22 and is the window's observed gap energy divided by total
  observed gap energy. The 24-hour mean gap excludes missing demand hours. The
  demand-minus-hydro gap is a simplified residual and is not identical to
  actual imports.
- Run Step 5 from the project root with
  `.\.venv\Scripts\python.exe -m energy.hydro_gap_scenarios`. Hourly scenario
  data is saved to `data/dry_season_hydro_gap_2025_26.csv`; monthly, dry-season,
  and Magh-Falgun metrics are saved to
  `report/hydro_gap_scenario_summary_2025_26.csv`. The demand/supply/gap hourly
  profiles for Scenarios A and B's 10% base are plotted in
  `report/hydro_demand_gap_profiles_2025_26.png`; monthly gap peaks and energy
  for all scenarios are compared in
  `report/hydro_gap_monthly_scenarios_2025_26.png`.
- `energy/explore_demand.py` reads the cleaned hourly CSV, adds local calendar
  fields and past-demand features, and writes monthly, month-by-hour, and
  season-by-hour tables with plots. The seasonal labels describe calendar
  categories only; demand data alone does not establish actual hydrology.
- Monthly coverage is measured against every hour in the full Gregorian
  calendar month, so partial first/last months and missing source months are
  visible as incomplete rather than appearing fully covered.
- In the exploration feature table, Saturday is marked as Nepal's weekend day.
  The 24-hour and 168-hour lags refer to those exact earlier local timestamps;
  missing source observations remain missing rather than being bridged.

## Forecast and plot interpretation

- Run the demand exploration from the project root with
  `.\.venv\Scripts\python.exe energy\explore_demand.py`. Add `--show` to open
  the generated plots interactively.
- The exploration writes the timestamped feature table to
  `data/demand_hourly_features.csv`; monthly statistics and hour-of-day profiles
  go to `report/demand_monthly_summary.csv`,
  `report/demand_profile_by_month_hour.csv`, and
  `report/demand_profile_by_season_hour.csv`. Its overview, monthly profile, and
  seasonal profile plots are saved under `report/`.
- "Actual" in the forecast output means the observed historical demand for that
  target timestamp; it is not today's live demand.
- "Holdout" is the later chronological portion kept aside to evaluate forecasts.
  The current split begins on 2017-01-30 at 10:00 Nepal time.
- The forecast comparison uses one calendar week of hourly points. The monthly
  profile plot averages demand separately for each hour of each available month.
- Both plots keep the underlying measurements hourly. The monthly profile
  averages each hour of the day within each month. Missing source periods are
  shown as gaps and are not interpreted as zero demand.
- Run the chronological forecast backtest with
  `.\.venv\Scripts\python.exe -m energy.forecast_demand`. The output includes a
  breakdown by BS month and hour of day at
  `report/demand_forecast_error_breakdown.csv`, plus a chart of monthly MAE and
  hourly forecast bias at `report/demand_forecast_error_analysis.png`.
  Forecast bias is forecast minus actual: positive values indicate
  overforecasting. Breakdown results describe only the holdout dates present in
  this historical dataset; they are not current-demand estimates.
