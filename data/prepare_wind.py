import numpy as np
import pandas as pd

from configs.base import ROOT
from src.features import target_features

RAW = ROOT / "data" / "raw" / "Vestas_V52_preprocessed_history_data.csv"
OUT = ROOT / "data" / "processed" / "wind.parquet"
START, END = "2012-02-01 00:00", "2018-09-30 13:00"
TO_KW = 1000.0


def circular_mean_deg(x):
    r = np.radians(x.dropna())
    if len(r) == 0:
        return np.nan
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def main():
    df = pd.read_csv(RAW, parse_dates=["Timestamp"]).set_index("Timestamp").sort_index()
    df = df.rename(columns={"WindDirection": "WindDirAbs", "ActivePower": "Power"})
    df[["Power", "TheoreticalPower"]] *= TO_KW
    h = df.resample("1h").agg({"WindSpeed": "mean", "StdDevWindSpeed": "mean", "WindDirAbs": circular_mean_deg,
                               "Power": "mean", "TheoreticalPower": "mean", "Pitch": "mean", "GenRPM": "mean",
                               "EnvirTemp": "mean"})
    h = h.loc[START:END]
    valid = h["WindSpeed"].notna() & h["Power"].notna()
    power = h["Power"].clip(lower=0)
    filled = power.ffill().bfill()
    obs = h.drop(columns=["Power"]).ffill().bfill()
    obs["TheoreticalPower"] = obs["TheoreticalPower"].clip(lower=0)
    wd = np.radians(obs["WindDirAbs"].to_numpy())
    hour, doy = h.index.hour.to_numpy(), h.index.dayofyear.to_numpy()
    table = pd.DataFrame({
        "target": power.where(valid).astype(np.float64), "target_valid": valid.to_numpy(), "target_filled": filled,
        "cal__hour_sin": np.sin(2 * np.pi * hour / 24), "cal__hour_cos": np.cos(2 * np.pi * hour / 24),
        "cal__doy_sin": np.sin(2 * np.pi * doy / 365), "cal__doy_cos": np.cos(2 * np.pi * doy / 365),
        "obs__wind_speed": obs["WindSpeed"], "obs__wind_speed_std": obs["StdDevWindSpeed"],
        "obs__wind_dir_sin": np.sin(wd), "obs__wind_dir_cos": np.cos(wd),
        "obs__theoretical_power": obs["TheoreticalPower"], "obs__pitch": obs["Pitch"], "obs__gen_rpm": obs["GenRPM"],
        "obs__temp": obs["EnvirTemp"],
    }, index=h.index)
    table["target"] = table["target"].fillna(table["target_filled"])
    table = pd.concat([table, target_features(filled)], axis=1)
    table.index.name = "Time"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT)
    print(f"{OUT}: {table.shape}, {int(valid.sum())} hours with data")


if __name__ == "__main__":
    main()
