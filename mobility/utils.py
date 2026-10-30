"""Small, dependency-free helpers shared across the package."""

import re
import warnings

import numpy as np
import pandas as pd


def is_numeric_col(series):
    return (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    )


def is_text_col(series):
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ) and not pd.api.types.is_numeric_dtype(series)


def numeric_columns(df):
    return [c for c in df.columns if is_numeric_col(df[c])]


def sample_for_plot(data, limit=50_000):
    """Random sample for plotting. Returns (data, was_sampled)."""
    if len(data) > limit:
        return data.sample(limit, random_state=42).sort_index(), True
    return data, False


def name_tokens(name):
    """'roadID' / 'road_id' / 'Road ID' -> ['road', 'id']."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def compact_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def parse_datetime(series):
    """Robust datetime parsing. Returns naive datetimes (UTC if tz-aware).

    * numeric epoch seconds / milliseconds are recognised
    * mixed date formats are handled
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        if getattr(series.dt, "tz", None) is not None:
            return series.dt.tz_convert("UTC").dt.tz_localize(None)
        return series

    if is_numeric_col(series):
        median = series.dropna().abs().median()
        if pd.isna(median) or median < 1e8:
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        unit = "ms" if median > 1e11 else "s"
        return pd.to_datetime(series, unit=unit, errors="coerce")

    def convert(**kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(series, errors="coerce", utc=True, **kwargs)

    parsed = convert()
    present = int(series.notna().sum())

    if present and (present - int(parsed.notna().sum())) / present > 0.05:
        try:
            alternative = convert(format="mixed")
            if alternative.notna().sum() > parsed.notna().sum():
                parsed = alternative
        except (TypeError, ValueError):
            pass

    return parsed.dt.tz_localize(None)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (vectorised)."""
    radius = 6_371_008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def infer_speed_unit(column_name):
    """Guess the speed unit from a column name, or None."""
    name = str(column_name).lower()
    if "mph" in name:
        return "mph"
    if any(k in name for k in ("kmh", "kph", "km_h", "km/h")):
        return "km/h"
    if re.search(r"(^|[^a-z])m_?s($|[^a-z])", name) or "mps" in name:
        return "m/s"
    if "knot" in name:
        return "knots"
    return None
