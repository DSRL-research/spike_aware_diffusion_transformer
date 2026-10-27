import pandas as pd

from configs.base import ROOT
from src.features import build_table

RAW = ROOT / "data" / "raw" / "heatDemand.csv"
OUT = ROOT / "data" / "processed" / "heat.parquet"
TZ = "Europe/Brussels"
LAT, LON = 50.85, 4.35
FLAT_HOURS = 6
WEATHER = {"Temperature": "temp", "Humidity": "humidity", "Wind Speed": "wind", "Cloud Cover": "cloud",
           "Rainfall": "precip", "Solar Irradiation": "solar"}


def main():
    df = pd.read_csv(RAW, parse_dates=["Time"]).set_index("Time")
    utc = df.index.tz_localize("UTC")
    local = utc.tz_convert(TZ)
    weather = df[list(WEATHER)].rename(columns=WEATHER)
    table = build_table(df["Power"], weather, local, utc, lat=LAT, lon=LON, flat_hours=FLAT_HOURS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT)
    print(f"{OUT}: {table.shape}, {int((~table['target_valid']).sum())} invalid hours")


if __name__ == "__main__":
    main()
