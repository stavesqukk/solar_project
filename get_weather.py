import os
import pvlib
import pandas as pd
import matplotlib.pyplot as plt

# ---- PART 1: download (yours, unchanged) ----
data, meta = pvlib.iotools.get_nasa_power(
    latitude=27.67,
    longitude=85.32,
    start="2016-04-01",
    end="2017-04-30",
    parameters=["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI",
                "ALLSKY_SFC_SW_DIFF", "T2M", "WS10M"],
)

# ---- PART 2: checks (yours, unchanged) ----
data = data.tz_convert("Asia/Kathmandu")
print(data.index[:3])
print("duplicates:", data.index.duplicated().sum())
full = pd.date_range(data.index.min(), data.index.max(), freq="h")
print("missing hours:", len(full) - len(data))
print(data.isna().sum())

# ---- PART 3: move from :45 to :00 ----
target = pd.date_range(data.index.min().ceil("h"),
                       data.index.max().floor("h"), freq="h")
hourly = (data.reindex(data.index.union(target))
              .interpolate("time")
              .loc[target])
print(hourly.head(3))
print("rows before:", len(data))
print("rows after:", len(hourly))
print("first:", hourly.index[0])
print("last:", hourly.index[-1])

# ---- PART 4: save ----
os.makedirs("data", exist_ok=True)
hourly.to_csv("data/weather_hourly.csv", index_label="time")

# ---- PART 5: two plots ----
hourly.iloc[0:24]["ghi"].plot(marker="o", title="GHI, first day")
plt.show()

hourly["ghi"].resample("D").mean().plot(title="Daily mean GHI, whole period")
plt.show()