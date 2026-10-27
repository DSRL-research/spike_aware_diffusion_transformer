import datetime as dt

import numpy as np
import pandas as pd
from dateutil.easter import EASTER_WESTERN, easter


def public_holidays_belgium(years) -> set:
    days = set()
    for y in years:
        d = dt.date
        e = easter(y, EASTER_WESTERN)
        days |= {d(y, 1, 1), d(y, 5, 1), d(y, 7, 21), d(y, 8, 15), d(y, 11, 1), d(y, 11, 11), d(y, 12, 25)}
        days |= {e, e + dt.timedelta(days=1), e + dt.timedelta(days=39), e + dt.timedelta(days=49),
                 e + dt.timedelta(days=50)}
    return days


def solar_elevation(utc_index: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    doy = utc_index.dayofyear.to_numpy()
    hour = utc_index.hour.to_numpy() + utc_index.minute.to_numpy() / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    true_solar_minutes = hour * 60 + eqtime + 4 * lon
    hour_angle = np.deg2rad(true_solar_minutes / 4 - 180)
    phi = np.deg2rad(lat)
    cos_zenith = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    return 90 - np.rad2deg(np.arccos(np.clip(cos_zenith, -1, 1)))
