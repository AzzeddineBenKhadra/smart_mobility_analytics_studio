"""Mobility insights from speed / volume time series (pure pandas).

Definitions
-----------
free-flow speed     high percentile (default 85th) of a segment's observed
                    speeds — a standard proxy when no speed limit is known.
congestion index    CI = 1 - speed / free-flow speed, clipped to [0, 1].
planning time index free-flow speed / 5th-percentile speed: how much extra
                    time a traveller should budget to arrive on time 95% of
                    the time (a reliability measure).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import CONGESTION_LEVELS, DAY_NAMES, HEAVY_THRESHOLD
from .quality import speed_factor
from .utils import parse_datetime

LEVEL_EDGES = [lo for _, lo, _ in CONGESTION_LEVELS] + [CONGESTION_LEVELS[-1][2]]
LEVEL_LABELS = [name for name, _, _ in CONGESTION_LEVELS]


# ---------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------

def prepare_panel(df, m):
    """Tidy view: ts, road, speed (km/h), volume, lat, lon. None if unusable."""
    ts_col = m.get("timestamp")
    if not ts_col or ts_col not in df.columns:
        return None
    if not (m.get("speed") or m.get("volume")):
        return None

    panel = pd.DataFrame(index=df.index)
    panel["ts"] = parse_datetime(df[ts_col])

    if m.get("speed"):
        panel["speed"] = pd.to_numeric(df[m["speed"]], errors="coerce") * speed_factor(m)
    if m.get("volume"):
        panel["volume"] = pd.to_numeric(df[m["volume"]], errors="coerce")

    if m.get("road_id"):
        panel["road"] = df[m["road_id"]].astype("string").fillna("(unknown)").astype(object)
    else:
        panel["road"] = pd.Series(["All"] * len(df), index=df.index, dtype=object)

    for key, col in (("lat", m.get("latitude")), ("lon", m.get("longitude"))):
        if col:
            panel[key] = pd.to_numeric(df[col], errors="coerce")

    panel = panel.dropna(subset=["ts"])
    panel = panel.sort_values("ts", kind="stable").reset_index(drop=True)
    return panel if len(panel) else None


def infer_interval_minutes(panel, default=15.0):
    """Typical spacing between consecutive observations of one road."""
    steps = []
    for _, group in list(panel.groupby("road"))[:20]:
        ts = np.sort(group["ts"].dropna().unique())
        if len(ts) > 2:
            diff = np.diff(ts).astype("timedelta64[s]").astype(float)
            diff = diff[diff > 0]
            if len(diff):
                steps.append(np.median(diff))
    if not steps:
        return float(default)
    return max(float(np.median(steps)) / 60.0, 1 / 60)


def add_congestion(panel, free_flow_q=0.85, min_points=20):
    """Add free_flow, ci and level columns. Returns a new frame."""
    out = panel.copy()
    if "speed" not in out.columns:
        return out

    per_road = out.groupby("road")["speed"].quantile(free_flow_q)
    counts = out.groupby("road")["speed"].count()
    global_ff = out["speed"].quantile(free_flow_q)
    free_flow = per_road.where(counts >= min_points, global_ff)
    free_flow = free_flow.where(free_flow > 0)

    out["free_flow"] = out["road"].map(free_flow).astype(float)
    out["ci"] = (1 - out["speed"] / out["free_flow"]).clip(0, 1)
    out["level"] = pd.cut(out["ci"], bins=LEVEL_EDGES, labels=LEVEL_LABELS,
                          right=False)
    return out


def is_weekend(ts, weekend_days):
    return ts.dt.dayofweek.isin(list(weekend_days))


# ---------------------------------------------------------------------------
# time patterns
# ---------------------------------------------------------------------------

def time_profile(panel, value, weekend_days=(5, 6)):
    """Average of ``value`` by hour of day, split weekday / weekend."""
    work = panel[["ts", value]].dropna()
    work = work.assign(
        hour=work["ts"].dt.hour,
        day_type=np.where(is_weekend(work["ts"], weekend_days), "Weekend", "Weekday"),
    )
    return (work.groupby(["day_type", "hour"])[value]
            .agg(mean="mean", count="count").reset_index())


def heat_table(panel, value):
    """Mean of ``value`` by day of week (rows) and hour (columns)."""
    work = panel[["ts", value]].dropna()
    table = (work.groupby([work["ts"].dt.dayofweek, work["ts"].dt.hour])[value]
             .mean().unstack())
    table = table.reindex(index=range(7), columns=range(24))
    table.index = DAY_NAMES
    return table


def peak_periods(panel, value, weekend_days=(5, 6), mode="max"):
    """Morning (<12h) and afternoon/evening (>=12h) weekday peaks."""
    prof = time_profile(panel, value, weekend_days)
    prof = prof[prof["day_type"] == "Weekday"].set_index("hour")["mean"]
    if prof.empty:
        return None

    pick = (lambda s: s.idxmax()) if mode == "max" else (lambda s: s.idxmin())
    result = {}
    for label, part in (("am", prof[prof.index < 12]), ("pm", prof[prof.index >= 12])):
        if part.empty:
            result[label] = None
        else:
            hour = int(pick(part))
            result[label] = (hour, float(part.loc[hour]))
    return result


def worst_time_slots(panel, n=10):
    """Day-of-week/hour combinations with the highest mean congestion."""
    if "ci" not in panel.columns:
        return pd.DataFrame()
    work = panel.dropna(subset=["ci"])
    grouped = (work.groupby([work["ts"].dt.dayofweek.rename("dow"),
                             work["ts"].dt.hour.rename("hour")])["ci"]
               .agg(mean_ci="mean", records="count").reset_index())
    grouped["Time slot"] = [
        f"{DAY_NAMES[int(d)]} {int(h):02d}:00" for d, h in zip(grouped["dow"], grouped["hour"])]
    grouped = grouped.sort_values("mean_ci", ascending=False).head(n)
    return grouped[["Time slot", "mean_ci", "records"]].reset_index(drop=True)


def level_distribution(panel):
    if "level" not in panel.columns:
        return pd.DataFrame()
    counts = panel["level"].value_counts().reindex(LEVEL_LABELS).fillna(0)
    total = counts.sum()
    return pd.DataFrame({
        "Level": counts.index.astype(str),
        "Records": counts.to_numpy().astype(int),
        "Share %": (counts / total * 100 if total else counts).round(1).to_numpy(),
    })


# ---------------------------------------------------------------------------
# segments
# ---------------------------------------------------------------------------

def segment_ranking(panel):
    """One row per road, most congested first."""
    if "ci" not in panel.columns:
        return pd.DataFrame()

    work = panel.dropna(subset=["ci"])
    if work.empty:
        return pd.DataFrame()

    grouped = work.groupby("road")
    table = pd.DataFrame({
        "records": grouped.size(),
        "mean_speed": grouped["speed"].mean(),
        "free_flow": grouped["free_flow"].first(),
        "mean_ci": grouped["ci"].mean(),
        "pct_heavy": (work["ci"] > HEAVY_THRESHOLD).groupby(work["road"]).mean() * 100,
        "speed_p5": grouped["speed"].quantile(0.05),
    })
    table["planning_time_index"] = (table["free_flow"] / table["speed_p5"]).replace(
        [np.inf, -np.inf], np.nan)

    if "volume" in work.columns:
        table["mean_volume"] = grouped["volume"].mean()

    hourly = (work.groupby(["road", work["ts"].dt.hour.rename("hour")])["ci"]
              .mean().reset_index())
    worst = hourly.sort_values("ci").groupby("road").tail(1).set_index("road")["hour"]
    table["worst_hour"] = worst

    for col in ("lat", "lon"):
        if col in work.columns:
            table[col] = grouped[col].median()

    table = table.drop(columns=["speed_p5"]).sort_values("mean_ci", ascending=False)
    return table.reset_index()


def capacity_estimate(panel):
    """Per-road empirical capacity (95th percentile volume) and speed at it."""
    if not {"speed", "volume"}.issubset(panel.columns):
        return pd.DataFrame()
    rows = []
    for road, group in panel.dropna(subset=["speed", "volume"]).groupby("road"):
        if len(group) < 20:
            continue
        q95 = group["volume"].quantile(0.95)
        near_cap = group[group["volume"] >= group["volume"].quantile(0.90)]
        rows.append({
            "road": road,
            "capacity_q95": float(q95),
            "speed_near_capacity": float(near_cap["speed"].median()),
            "free_flow": float(group["speed"].quantile(0.85)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# anomalies / incidents
# ---------------------------------------------------------------------------

def _seasonal_z(work, value, min_rel_scale=0.03):
    """Robust z-score of ``value`` against its (road, weekday, hour) median.

    The noise scale is estimated per (road, day type, hour) so that busy hours,
    which are naturally more variable, are not over-flagged. A small relative
    floor stops tiny scales from inflating the score.
    """
    dow = work["ts"].dt.dayofweek
    hour = work["ts"].dt.hour
    grouped = work.groupby([work["road"], dow, hour])[value]
    expected = grouped.transform("median")
    support = grouped.transform("count")
    road_median = work.groupby("road")[value].transform("median")
    expected = expected.where(support >= 3, road_median)

    resid = work[value] - expected
    day_type = (dow >= 5).astype(int)
    keys = [work["road"], day_type, hour]
    r_med = resid.groupby(keys).transform("median")
    mad = (resid - r_med).abs().groupby(keys).transform("median") * 1.4826
    std = resid.groupby(work["road"]).transform("std")

    scale = mad.where(mad > 0, std)
    scale = np.maximum(scale, min_rel_scale * expected.abs())
    scale = scale.where(scale > 0)
    return expected, resid / scale


def detect_anomalies(panel, metric="speed", method="seasonal",
                     z_threshold=4.0, contamination=0.01, direction="drops"):
    """Find unusual observations.

    method="seasonal"  robust z-score against the typical value for that road,
                       weekday and hour (rush hours are therefore NOT flagged).
    method="isolation" Isolation Forest on the seasonal z-scores of speed and
                       volume together (catches unusual *combinations*).

    Returns (flagged, full) — ``full`` has expected/z for every observation.
    """
    if metric not in panel.columns:
        raise ValueError(f"The data has no '{metric}' column.")

    work = panel.dropna(subset=[metric]).copy().reset_index(drop=True)
    if len(work) < 30:
        raise ValueError("At least 30 observations are needed to detect anomalies.")

    work["expected"], work["z"] = _seasonal_z(work, metric)
    work["score"] = np.nan

    if method == "isolation":
        features = pd.DataFrame({"z_" + metric: work["z"]})
        for other in ("speed", "volume"):
            if other != metric and other in work.columns:
                _, z_other = _seasonal_z(
                    work.dropna(subset=[other]).reindex(work.index), other)
                features["z_" + other] = z_other
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-20, 20)

        forest = IsolationForest(n_estimators=200, contamination=contamination,
                                 random_state=42, n_jobs=-1)
        flag = forest.fit_predict(features) == -1
        work["score"] = -forest.decision_function(features)
    else:
        if direction == "drops":
            flag = work["z"] < -z_threshold
        elif direction == "spikes":
            flag = work["z"] > z_threshold
        else:
            flag = work["z"].abs() > z_threshold

    work["flag"] = np.asarray(flag.fillna(False) if hasattr(flag, "fillna") else flag)
    work["direction"] = np.where(work["z"] < 0, "below expected", "above expected")
    flagged = work[work["flag"]].copy()
    return flagged, work


def group_incidents(flagged, interval_minutes, gap_factor=2.0, min_points=1):
    """Merge consecutive anomalous observations on a road into incidents.

    ``min_points`` drops isolated single-observation blips (likely noise).
    """
    if flagged.empty:
        return pd.DataFrame(columns=[
            "road", "start", "end", "duration_min", "points", "worst_z",
            "lowest_value", "expected"])

    data = flagged.sort_values(["road", "ts"]).reset_index(drop=True)
    gap = pd.Timedelta(minutes=interval_minutes * gap_factor)
    new_group = (data["road"] != data["road"].shift()) | (data["ts"].diff() > gap)
    data["incident"] = np.cumsum(np.asarray(new_group.fillna(True), dtype=bool))

    value_col = next(c for c in ("speed", "volume") if c in data.columns)
    out = data.groupby("incident").agg(
        road=("road", "first"),
        start=("ts", "min"),
        end=("ts", "max"),
        points=("ts", "size"),
        worst_z=("z", lambda s: s.iloc[np.argmax(np.abs(s.to_numpy()))]),
        lowest_value=(value_col, "min"),
        expected=("expected", "mean"),
    ).reset_index(drop=True)
    out = out[out["points"] >= min_points].copy()
    out["duration_min"] = (
        (out["end"] - out["start"]).dt.total_seconds() / 60 + interval_minutes)
    out = out.sort_values("worst_z", key=lambda s: s.abs(), ascending=False)
    return out[["road", "start", "end", "duration_min", "points", "worst_z",
                "lowest_value", "expected"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# narrative
# ---------------------------------------------------------------------------

def network_summary(panel):
    out = {
        "records": int(len(panel)),
        "roads": int(panel["road"].nunique()),
        "start": panel["ts"].min(),
        "end": panel["ts"].max(),
    }
    if "speed" in panel.columns:
        out["mean_speed"] = float(panel["speed"].mean())
    if "ci" in panel.columns:
        out["mean_ci"] = float(panel["ci"].mean())
        out["pct_heavy"] = float((panel["ci"].dropna() > HEAVY_THRESHOLD).mean() * 100)
    if "volume" in panel.columns:
        out["mean_volume"] = float(panel["volume"].mean())
    return out


def generate_findings(panel, weekend_days=(5, 6), n_incidents=None):
    """Plain-language findings. Every number comes from the data."""
    findings = []
    summary = network_summary(panel)
    days = max((summary["end"] - summary["start"]).total_seconds() / 86400, 0)

    findings.append(
        f"The data covers {days:.0f} days ({summary['start']:%d %b %Y} – "
        f"{summary['end']:%d %b %Y}), {summary['roads']} road segment(s) and "
        f"{summary['records']:,} observations.")

    if "ci" in panel.columns and panel["ci"].notna().any():
        ci_peaks = peak_periods(panel, "ci", weekend_days, "max")
        if ci_peaks and ci_peaks["am"] and ci_peaks["pm"]:
            findings.append(
                f"Weekday congestion peaks at {ci_peaks['am'][0]:02d}:00 "
                f"(average congestion index {ci_peaks['am'][1]:.2f}) in the morning and "
                f"{ci_peaks['pm'][0]:02d}:00 ({ci_peaks['pm'][1]:.2f}) later in the day.")

        ranking = segment_ranking(panel)
        if not ranking.empty:
            top = ranking.iloc[0]
            findings.append(
                f"{top['road']} is the most congested segment: average speed "
                f"{top['mean_speed']:.0f} km/h against a free-flow speed of "
                f"{top['free_flow']:.0f} km/h, with heavy congestion in "
                f"{top['pct_heavy']:.0f}% of observations (typically worst around "
                f"{int(top['worst_hour']):02d}:00).")
            if len(ranking) > 1:
                best = ranking.iloc[-1]
                findings.append(
                    f"{best['road']} is the least congested "
                    f"(average congestion index {best['mean_ci']:.2f}).")
            risky = ranking.dropna(subset=["planning_time_index"]).sort_values(
                "planning_time_index", ascending=False)
            if not risky.empty and risky.iloc[0]["planning_time_index"] > 1.3:
                findings.append(
                    f"Travel-time reliability is weakest on {risky.iloc[0]['road']}: "
                    f"its planning time index is {risky.iloc[0]['planning_time_index']:.2f}, "
                    "i.e. travellers should budget that much more than free-flow "
                    "time to be on time 95% of the time.")

        slots = worst_time_slots(panel, 1)
        if not slots.empty:
            findings.append(
                f"The single worst time slot network-wide is {slots.iloc[0]['Time slot']} "
                f"(average congestion index {slots.iloc[0]['mean_ci']:.2f}).")

        wk = is_weekend(panel["ts"], weekend_days)
        ci_week, ci_wend = panel.loc[~wk, "ci"].mean(), panel.loc[wk, "ci"].mean()
        if pd.notna(ci_week) and pd.notna(ci_wend) and ci_wend > 0:
            ratio = ci_week / ci_wend
            findings.append(
                f"Weekday congestion is {ratio:.1f}× the weekend level "
                f"({ci_week:.2f} vs {ci_wend:.2f})." if ratio >= 1 else
                f"Weekend congestion is {1 / ratio:.1f}× the weekday level "
                f"({ci_wend:.2f} vs {ci_week:.2f}).")

        night = panel[panel["ts"].dt.hour < 5]["speed"].mean()
        if "speed" in panel.columns and pd.notna(night) and ci_peaks and ci_peaks["pm"]:
            pm_hour = ci_peaks["pm"][0]
            pm_speed = panel[(panel["ts"].dt.hour == pm_hour)
                             & ~is_weekend(panel["ts"], weekend_days)]["speed"].mean()
            if pd.notna(pm_speed) and night > 0:
                findings.append(
                    f"At the weekday {pm_hour:02d}:00 peak, speeds are "
                    f"{(1 - pm_speed / night) * 100:.0f}% below overnight levels.")

    if "volume" in panel.columns and panel["volume"].notna().any():
        vol_peaks = peak_periods(panel, "volume", weekend_days, "max")
        if vol_peaks and vol_peaks["am"] and vol_peaks["pm"]:
            findings.append(
                f"Weekday traffic volume peaks at {vol_peaks['am'][0]:02d}:00 and "
                f"{vol_peaks['pm'][0]:02d}:00 (mean "
                f"{vol_peaks['am'][1]:.0f} and {vol_peaks['pm'][1]:.0f} per record).")

    if n_incidents is not None:
        findings.append(
            f"{n_incidents:,} possible incident(s) (abnormal speed drops) were detected.")

    return findings
