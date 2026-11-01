"""Data-quality profiling and cleaning operations (pure pandas)."""

import numpy as np
import pandas as pd

from .config import SPEED_UNITS
from .detection import find_identifier_columns
from .utils import is_numeric_col, is_text_col, parse_datetime


def speed_factor(mapping):
    """Multiplier that converts the mapped speed unit to km/h."""
    return SPEED_UNITS.get((mapping or {}).get("speed_unit", "km/h"), 1.0)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def coordinate_invalid_mask(df, lat_col, lon_col):
    """Rows where a coordinate is present but the pair is unusable.

    Invalid = unparseable, out of range, only one of the two present,
    or exactly (0, 0), which is almost always a GPS error.
    """
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")

    present = df[lat_col].notna() | df[lon_col].notna()
    ok = (
        lat.between(-90, 90)
        & lon.between(-180, 180)
        & ~((lat == 0) & (lon == 0))
    )
    return present & ~ok


def iqr_outlier_mask(values):
    valid = values.dropna()
    if len(valid) < 8:
        return pd.Series(False, index=values.index)

    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=values.index)

    return (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)


def impossible_mask(series, kind, factor=1.0):
    """Physically impossible values: negative, or speed above 250 km/h.

    ``factor`` converts speed to km/h before the 250 km/h screening rule.
    """
    values = pd.to_numeric(series, errors="coerce")
    if kind == "speed":
        mask = (values < 0) | (values * factor > 250)
    else:
        mask = values < 0
    return mask.fillna(False)


def unusual_mask(series, kind, factor=1.0, include_iqr=True):
    """Impossible values, plus (optionally) statistical IQR outliers.

    IQR outliers in traffic data are often *real* (rush-hour peaks), so the
    IQR part is a screening aid, not proof of an error.
    """
    mask = impossible_mask(series, kind, factor)
    if include_iqr:
        values = pd.to_numeric(series, errors="coerce")
        mask = mask | iqr_outlier_mask(values).fillna(False)
    return mask


def column_profile(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(t) for t in df.dtypes],
        "Missing": df.isna().sum().to_numpy(),
        "Missing %": (df.isna().mean() * 100).round(2).to_numpy(),
        "Unique": df.nunique(dropna=True).to_numpy(),
    })


def profile_data(df, mapping):
    rows, cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    findings = {
        "rows": rows,
        "columns": cols,
        "missing_cells": missing_cells,
        "missing_pct": missing_cells / max(rows * cols, 1) * 100,
        "duplicate_rows": duplicates,
        "invalid_coordinates": 0,
        "invalid_timestamps": 0,
        "impossible_speed": 0,
        "impossible_volume": 0,
        "outlier_speed": 0,      # statistical (IQR) only, not impossible
        "outlier_volume": 0,
    }

    lat_col, lon_col = mapping.get("latitude"), mapping.get("longitude")
    if lat_col and lon_col:
        findings["invalid_coordinates"] = int(
            coordinate_invalid_mask(df, lat_col, lon_col).sum())

    ts_col = mapping.get("timestamp")
    if ts_col:
        present = df[ts_col].notna()
        if present.any():
            parsed = parse_datetime(df.loc[present, ts_col])
            findings["invalid_timestamps"] = int(parsed.isna().sum())

    for kind in ("speed", "volume"):
        col = mapping.get(kind)
        if not col:
            continue
        factor = speed_factor(mapping) if kind == "speed" else 1.0
        impossible = impossible_mask(df[col], kind, factor)
        everything = unusual_mask(df[col], kind, factor, include_iqr=True)
        findings[f"impossible_{kind}"] = int(impossible.sum())
        findings[f"outlier_{kind}"] = int((everything & ~impossible).sum())

    denom = max(rows, 1)
    penalty = (
        min(findings["missing_pct"] * 0.35, 35)
        + min(duplicates / denom * 100 * 0.20, 20)
        + min(findings["invalid_coordinates"] / denom * 100 * 0.20, 20)
        + min(findings["invalid_timestamps"] / denom * 100 * 0.10, 10)
        + min((findings["impossible_speed"] + findings["impossible_volume"])
              / denom * 100 * 0.15, 15)
        + min((findings["outlier_speed"] + findings["outlier_volume"])
              / denom * 100 * 0.02, 5)
    )
    score = max(0.0, 100 - penalty)
    # Never show a perfect 100 while known problems remain.
    has_problems = any(findings[k] for k in (
        "missing_cells", "duplicate_rows", "invalid_coordinates", "invalid_timestamps",
        "impossible_speed", "impossible_volume"))
    findings["quality_score"] = round(min(score, 99.9) if has_problems else score, 1)
    return findings


def fill_candidates(df, mapping):
    """Split columns with missing values into fillable vs skipped (+reason)."""
    protected = {
        mapping.get(k) for k in (
            "latitude", "longitude", "timestamp", "road_id", "vehicle_id",
            "origin_lat", "origin_lon", "dest_lat", "dest_lon")
    } - {None}
    ids = find_identifier_columns(
        df, exclude={mapping.get("speed"), mapping.get("volume")})

    eligible, skipped = [], []
    for col in df.columns:
        share = df[col].isna().mean()
        if share == 0:
            continue
        if col in protected:
            skipped.append((col, "mapped coordinate/time/ID field"))
        elif col in ids:
            skipped.append((col, "looks like an identifier"))
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            skipped.append((col, "datetime column"))
        elif share > 0.5:
            skipped.append((col, "more than 50% missing"))
        else:
            eligible.append(col)
    return eligible, skipped


# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------

def make_log_entry(operation, column, rows_before, df_after, details=""):
    return {
        "operation": operation,
        "column": column,
        "details": details,
        "rows_before": int(rows_before),
        "rows_after": int(len(df_after)),
        "missing_after": int(df_after.isna().sum().sum()),
    }


def run_smart_cleaning(df, actions, mapping):
    """Apply approved smart-cleaning actions to a copy of ``df``.

    Returns (cleaned_df, log_entries, total_changes).
    """
    work = df.copy(deep=True)
    entries = []
    changes = 0

    if actions.get("duplicates"):
        before = len(work)
        work = work.drop_duplicates()
        removed = before - len(work)
        changes += removed
        entries.append(make_log_entry(
            "Smart: remove duplicate rows", "(all)", before, work,
            f"Removed {removed:,} exact duplicate rows"))

    if actions.get("coordinates"):
        lat_col, lon_col = mapping["latitude"], mapping["longitude"]
        bad = coordinate_invalid_mask(work, lat_col, lon_col)
        count = int(bad.sum())
        if count:
            work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce").mask(bad)
            work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce").mask(bad)
        changes += count
        entries.append(make_log_entry(
            "Smart: invalid coordinates", f"{lat_col}, {lon_col}", len(work), work,
            f"Set {count:,} invalid coordinate pairs to missing"))

    for action, include_iqr, label in (
        ("impossible", False, "impossible values (negative / >250 km/h)"),
        ("outliers", True, "statistical (IQR) outliers"),
    ):
        if not actions.get(action):
            continue
        changed = []
        for kind in ("speed", "volume"):
            col = mapping.get(kind)
            if not col:
                continue
            factor = speed_factor(mapping) if kind == "speed" else 1.0
            mask = unusual_mask(work[col], kind, factor, include_iqr=include_iqr)
            count = int(mask.sum())
            if count:
                work[col] = pd.to_numeric(work[col], errors="coerce").mask(mask)
                changed.append(f"{col}: {count:,}")
                changes += count

        entries.append(make_log_entry(
            f"Smart: {label}", "speed / volume", len(work), work,
            f"Set {label} to missing — "
            + ("; ".join(changed) if changed else "none found")))

    if actions.get("missing"):
        eligible, _ = fill_candidates(work, mapping)
        filled = []
        for col in eligible:
            n_missing = int(work[col].isna().sum())
            if is_numeric_col(work[col]):
                value = work[col].median()
            else:
                modes = work[col].mode(dropna=True)
                value = modes.iloc[0] if not modes.empty else np.nan

            if pd.notna(value):
                work[col] = work[col].fillna(value)
                filled.append(f"{col} ({n_missing:,})")
                changes += n_missing

        entries.append(make_log_entry(
            "Smart: fill missing values", "multiple", len(work), work,
            "Filled: " + (", ".join(filled) if filled else "nothing eligible")))

    return work, entries, changes


OPERATIONS = [
    "Remove duplicate rows",
    "Drop rows with missing values",
    "Remove rows with missing values in a selected column",
    "Fill missing values in one column",
    "Convert column to numeric",
    "Convert column to datetime",
    "Remove rows outside a numeric range",
    "Trim whitespace from text",
    "Standardize text to lowercase",
    "Remove outliers using IQR",
    "Clip a numeric column to a valid range",
]


def parse_bound(text):
    text = text.strip()
    if not text:
        return None, None
    try:
        return float(text), None
    except ValueError:
        return None, f"'{text}' is not a number."


def apply_operation(df, operation, col, opts):
    """Run one cleaning operation on a copy. Returns (new_df, details)."""
    work = df.copy(deep=True)
    details = ""

    if operation == "Remove duplicate rows":
        work = work.drop_duplicates()
        details = f"Removed {len(df) - len(work):,} duplicate rows"

    elif operation == "Drop rows with missing values":
        work = work.dropna()
        details = f"Removed {len(df) - len(work):,} rows"

    elif operation == "Remove rows with missing values in a selected column":
        work = work.dropna(subset=[col])
        details = f"Removed {len(df) - len(work):,} rows"

    elif operation == "Fill missing values in one column":
        method = opts["fill_method"]
        series = work[col]
        missing_before = int(series.isna().sum())

        if method in ("Median", "Mean", "Zero"):
            if not is_numeric_col(series):
                raise ValueError(
                    f"{method} fill needs a numeric column. Convert '{col}' to "
                    "numeric first, or use Mode / Forward fill / Backward fill.")
            value = {"Median": series.median(), "Mean": series.mean(), "Zero": 0}[method]
            if pd.isna(value):
                raise ValueError("The column has no valid values to compute a fill value.")
            work[col] = series.fillna(value)
        elif method == "Mode":
            modes = series.mode(dropna=True)
            if modes.empty:
                raise ValueError("The column has no values to compute a mode.")
            work[col] = series.fillna(modes.iloc[0])
        elif method == "Forward fill":
            work[col] = series.ffill()
        else:
            work[col] = series.bfill()

        filled = missing_before - int(work[col].isna().sum())
        details = f"Filled {filled:,} of {missing_before:,} missing values ({method.lower()})"

    elif operation == "Convert column to numeric":
        valid_before = int(work[col].notna().sum())
        converted = pd.to_numeric(work[col], errors="coerce")
        lost = valid_before - int(converted.notna().sum())
        work[col] = converted
        details = f"{lost:,} values could not be converted and became missing"

    elif operation == "Convert column to datetime":
        valid_before = int(work[col].notna().sum())
        converted = parse_datetime(work[col])
        lost = valid_before - int(converted.notna().sum())
        work[col] = converted
        details = f"{lost:,} values could not be parsed and became missing"

    elif operation == "Remove rows outside a numeric range":
        numeric = pd.to_numeric(work[col], errors="coerce")
        outside = pd.Series(False, index=work.index)
        if opts["lower"] is not None:
            outside |= numeric < opts["lower"]
        if opts["upper"] is not None:
            outside |= numeric > opts["upper"]
        if opts.get("drop_missing"):
            outside |= numeric.isna()
        work = work.loc[~outside]
        details = f"Removed {len(df) - len(work):,} rows outside the range"

    elif operation in ("Trim whitespace from text", "Standardize text to lowercase"):
        if opts.get("text_scope") == "All text columns":
            targets = [c for c in work.columns if is_text_col(work[c])]
        else:
            if not is_text_col(work[col]):
                raise ValueError("Choose a text column, or apply to all text columns.")
            targets = [col]

        func = (
            (lambda v: v.strip() if isinstance(v, str) else v)
            if operation.startswith("Trim")
            else (lambda v: v.lower() if isinstance(v, str) else v)
        )
        changed = 0
        for target in targets:
            old = work[target]
            new = old.map(func)
            changed += int((new.astype("string") != old.astype("string")).fillna(False).sum())
            work[target] = new
        details = f"Changed {changed:,} values in {len(targets)} column(s)"

    elif operation == "Remove outliers using IQR":
        numeric = pd.to_numeric(work[col], errors="coerce")
        q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr):
            raise ValueError("The column has no valid numeric values.")
        outside = (numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)
        work = work.loc[~outside.fillna(False)]
        details = f"Removed {len(df) - len(work):,} outlier rows (missing values kept)"

    elif operation == "Clip a numeric column to a valid range":
        numeric = pd.to_numeric(work[col], errors="coerce")
        clipped = numeric.clip(lower=opts["lower"], upper=opts["upper"])
        changed = int(((clipped != numeric) & numeric.notna()).sum())
        work[col] = clipped
        details = f"Clipped {changed:,} values"

    else:
        raise ValueError(f"Unknown operation: {operation}")

    return work, details
