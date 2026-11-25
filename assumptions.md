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
- The `0:00` reading is assigned to the end of the same BS day, represented as
  midnight at the start of the following Gregorian day. This time-column
  interpretation still needs confirmation from the NEA source documentation.
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
- The project dry-season comparison uses December-April. This is a project
  convention, not the same as the DHM meteorological monsoon definition
  (June-September; see the [DHM 2025 monsoon report](https://dhm.gov.np/uploads/dhm/climateService/Monsoon_seasonal_2025.pdf));
  May and October-November are treated as transition months.
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
