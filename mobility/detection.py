"""Column detection: names first (config-driven), then a value sanity check."""

import pandas as pd

from .config import ID_TOKENS, load_field_rules
from .utils import (compact_name, infer_speed_unit, is_numeric_col,
                    name_tokens, parse_datetime)


def _numeric_ratio(series, sample=200):
    values = series.dropna().head(sample)
    if values.empty:
        return 0.0
    return float(pd.to_numeric(values, errors="coerce").notna().mean())


def _datetime_ratio(series, sample=50):
    if pd.api.types.is_datetime64_any_dtype(series):
        return 1.0
    values = series.dropna().head(sample)
    if values.empty:
        return 0.0
    return float(parse_datetime(values).notna().mean())


def _plausible(series, field):
    if field == "timestamp":
        return _datetime_ratio(series) >= 0.8
    if field in ("latitude", "longitude", "origin_lat", "origin_lon",
                 "dest_lat", "dest_lon"):
        return _numeric_ratio(series) >= 0.8
    if field in ("speed", "volume"):
        return _numeric_ratio(series) >= 0.5
    return True


def detect_columns(df, rules=None):
    """Map transport fields to columns. Returns {field: column or None}."""
    rules = rules or load_field_rules()
    columns = list(df.columns)
    compact = {c: compact_name(c) for c in columns}
    tokens = {c: set(name_tokens(c)) for c in columns}

    detected = {}
    used = set()

    for field, rule in rules.items():
        exact, contains = set(rule["exact"]), tuple(rule["contains"])
        token_set, exclude = set(rule["tokens"]), set(rule["exclude"])
        chosen = None

        for tier in ("exact", "contains", "tokens"):
            hits = []
            for col in columns:
                if col in used or tokens[col] & exclude:
                    continue

                if tier == "exact":
                    matched = compact[col] in exact
                elif tier == "contains":
                    matched = any(k in compact[col] for k in contains)
                else:
                    matched = bool(tokens[col] & token_set)

                if matched and _plausible(df[col], field):
                    hits.append(col)

            if hits:
                if field == "timestamp":
                    hits.sort(key=lambda c: not pd.api.types.is_datetime64_any_dtype(df[c]))
                elif field not in ("road_id", "vehicle_id"):
                    hits.sort(key=lambda c: not is_numeric_col(df[c]))
                chosen = hits[0]
                break

        detected[field] = chosen
        if chosen:
            used.add(chosen)

    return detected


def detect_speed_unit(df, speed_col):
    """Guess the speed unit from the column name; defaults to km/h."""
    return (infer_speed_unit(speed_col) if speed_col else None) or "km/h"


def looks_like_identifier(series, name):
    if set(name_tokens(name)) & ID_TOKENS:
        return True
    if (
        pd.api.types.is_integer_dtype(series)
        and len(series) >= 50
        and series.nunique() / len(series) > 0.95
    ):
        return True
    return False


def find_identifier_columns(df, exclude=()):
    """Columns that look like IDs (never filled, never used as features)."""
    exclude = set(exclude)
    return {
        c for c in df.columns
        if c not in exclude and looks_like_identifier(df[c], c)
    }
