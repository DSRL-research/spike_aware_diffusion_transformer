import numpy as np
import pandas as pd

from src.calendars import public_holidays_belgium, solar_elevation

GROUP_PREFIXES = {"cal": ("cal__",), "obs": ("obs__",), "tgt": ("tgt_level__", "tgt_scale__")}
PHYSICAL_RANGE = dict(temp=(-40, 50), humidity=(1, 100), wind=(0, 200), cloud=(0, 100), precip=(0, 200),
                      solar=(0, 1500))


def _run_lengths(y: pd.Series) -> tuple:
    run_id = (y.diff() != 0).cumsum()
    total = y.groupby(run_id).transform("size")
    so_far = y.groupby(run_id).cumcount() + 1
    return total.to_numpy(), so_far.to_numpy()


def outage_masks(y: pd.Series, flat_hours: int) -> tuple:
    total, so_far = _run_lengths(y)
    base = (y.isna() | (y <= 0)).to_numpy()
    invalid = base | (total >= flat_hours)
    causal = base | (so_far >= flat_hours)
    after = lambda m: m | np.concatenate([[False], m[:-1]])
    return after(invalid), after(causal)


def causal_fill(y: pd.Series, causal_invalid: np.ndarray) -> pd.Series:
    v = y.to_numpy(dtype=np.float64).copy()
    last_good = np.nan
    for t in range(len(v)):
        if causal_invalid[t]:
            v[t] = v[t - 24] if t >= 24 else last_good
        else:
            last_good = v[t]
    if np.isnan(v).any():
        raise ValueError("Outage at the start of the series cannot be filled causally")
    return pd.Series(v, index=y.index)


def calendar_features(local: pd.DatetimeIndex, utc: pd.DatetimeIndex, lat: float, lon: float) -> pd.DataFrame:
    hour, dow = local.hour.to_numpy(), local.dayofweek.to_numpy()
    year_len = np.where(local.is_leap_year, 366, 365)
    doy = (local.dayofyear.to_numpy() - 1) / year_len
    holidays = public_holidays_belgium(sorted(set(local.year)))
    is_holiday = np.array([d in holidays for d in local.date])
    sun = solar_elevation(utc + pd.Timedelta(minutes=30), lat, lon)
    f = {
        "cal__hour_sin": np.sin(2 * np.pi * hour / 24), "cal__hour_cos": np.cos(2 * np.pi * hour / 24),
        "cal__dow_sin": np.sin(2 * np.pi * dow / 7), "cal__dow_cos": np.cos(2 * np.pi * dow / 7),
        "cal__doy_sin": np.sin(2 * np.pi * doy), "cal__doy_cos": np.cos(2 * np.pi * doy),
        "cal__weekend": (dow >= 5).astype(float),
        "cal__holiday": is_holiday.astype(float),
        "cal__sun_elev": np.sin(np.deg2rad(sun)),
    }
    return pd.DataFrame(f, index=local)


def weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.copy()
    for col in w.columns:
        lo, hi = PHYSICAL_RANGE[col]
        bad = w[col].isna() | (w[col] < lo) | (w[col] > hi)
        w[col] = w[col].mask(bad).ffill().bfill()
    f = {f"obs__{c}": w[c] for c in w.columns if c != "precip"}
    f["obs__precip_log"] = np.log1p(w["precip"])
    f["obs__temp_mean_24h"] = w["temp"].rolling(24, min_periods=1).mean()
    f["obs__heating_degree"] = np.maximum(0.0, 18.0 - w["temp"])
    f["obs__cooling_degree"] = np.maximum(0.0, w["temp"] - 22.0)
    return pd.DataFrame(f, index=weather.index)


def _lag(y: np.ndarray, k: int) -> np.ndarray:
    out = np.empty_like(y)
    for t in range(len(y)):
        if t >= k:
            out[t] = y[t - k]
        else:
            j = t // 24
            out[t] = y[t - 24 * j] if j > 0 else y[t]
    return out


def target_features(filled: pd.Series) -> pd.DataFrame:
    y = filled.to_numpy(dtype=np.float64)
    s = filled.astype(float)
    lags = {k: _lag(y, 24 * k) for k in range(1, 8)}
    f = {
        "tgt_level__lag_24": lags[1],
        "tgt_level__lag_168": lags[7],
        "tgt_level__same_hour_mean_7d": np.mean([lags[k] for k in range(1, 8)], axis=0),
        "tgt_level__mean_6h": s.rolling(6, min_periods=1).mean().to_numpy(),
        "tgt_level__mean_24h": s.rolling(24, min_periods=1).mean().to_numpy(),
        "tgt_level__max_24h": s.rolling(24, min_periods=1).max().to_numpy(),
        "tgt_level__min_24h": s.rolling(24, min_periods=1).min().to_numpy(),
        "tgt_scale__std_6h": s.rolling(6, min_periods=1).std(ddof=0).to_numpy(),
        "tgt_scale__std_24h": s.rolling(24, min_periods=1).std(ddof=0).to_numpy(),
        "tgt_scale__diff_1": np.concatenate([[0.0], np.diff(y)]),
        "tgt_scale__diff_24": y - lags[1],
    }
    return pd.DataFrame(f, index=filled.index)


def build_table(target: pd.Series, weather: pd.DataFrame, local: pd.DatetimeIndex, utc: pd.DatetimeIndex, lat: float,
                lon: float, flat_hours: int) -> pd.DataFrame:
    invalid, causal_invalid = outage_masks(target, flat_hours)
    filled = causal_fill(target, causal_invalid)
    cal = calendar_features(local, utc, lat, lon)
    obs = weather_features(weather)
    tgt = target_features(filled)
    for part in (cal, obs, tgt):
        part.index = target.index
    return pd.concat([
        pd.DataFrame({"target": target.astype(np.float64), "target_valid": ~invalid, "target_filled": filled},
                     index=target.index),
        cal, obs, tgt,
    ], axis=1)


def feature_columns(columns, groups) -> list:
    missing = [g for g in groups if not any(c.startswith(GROUP_PREFIXES[g]) for c in columns)]
    if missing:
        raise ValueError(f"The processed table has no {', '.join(missing)} feature columns")
    prefixes = tuple(p for g in groups for p in GROUP_PREFIXES[g])
    return [c for c in columns if c.startswith(prefixes)]
