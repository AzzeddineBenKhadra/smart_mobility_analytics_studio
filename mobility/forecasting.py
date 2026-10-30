"""Traffic forecasting: lag features, honest baselines, time-series CV, intervals.

Design choices
--------------
* The series is aggregated to a regular grid (e.g. 15 min) per road and every
  feature is built with *past* information only (``shift`` / ``rolling``), so
  there is no look-ahead leakage.
* Forecasting from origin time ``t`` to ``t + horizon``: features are the value
  at ``t``, its lags, rolling statistics, and calendar features of the *target*
  time (which are known in advance).
* The test period is always the **latest** part of the data, and training rows
  whose target would overlap the test period are purged (gap = horizon).
* Four baselines are reported next to the model: persistence (value now),
  seasonal naive (same time last week/day), typical profile (mean for that
  road/weekday/hour in training) and the training mean. The model is only
  useful if it beats the *best* of them.
* Prediction intervals are empirical: residuals on a held-out, chronologically
  later calibration slice of the training data give a constant-width interval
  around the prediction. Coverage on the test set is reported, not assumed.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MAX_TRAIN_ROWS

GB_NAME = "Gradient Boosting (HistGB)"
RF_NAME = "Random Forest"
RIDGE_NAME = "Ridge Regression"
LGBM_NAME = "LightGBM"

BASELINES = ("persistence", "seasonal_naive", "typical_profile", "train_mean")
BASELINE_LABELS = {
    "persistence": "Persistence (value now)",
    "seasonal_naive": "Seasonal naive (same time last week/day)",
    "typical_profile": "Typical profile (road × weekday × hour)",
    "train_mean": "Training mean",
}


def available_models():
    names = [GB_NAME, RF_NAME, RIDGE_NAME]
    try:
        import lightgbm  # noqa: F401
        names.insert(1, LGBM_NAME)
    except ImportError:
        pass
    return names


def _estimator(name, n_estimators):
    if name == GB_NAME:
        return [("model", HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=0.1, random_state=42))]
    if name == LGBM_NAME:
        from lightgbm import LGBMRegressor
        return [("model", LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=42, verbose=-1))]
    if name == RF_NAME:
        return [("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=n_estimators, min_samples_leaf=3, max_features=0.6,
                    n_jobs=-1, random_state=42))]
    if name == RIDGE_NAME:
        return [("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0))]
    raise ValueError(f"Unknown model: {name}")


def build_model(name, n_estimators=150, columns=None):
    """Pipeline that first keeps only ``columns`` (those with data).

    Columns that are entirely missing in the training window (e.g. a one-week
    lag in the first week of data) would break some estimators, so they are
    dropped inside the pipeline and remembered in ``usable_columns_``.
    """
    steps = _estimator(name, n_estimators)
    if columns is not None:
        select = ColumnTransformer([("keep", "passthrough", list(columns))],
                                   remainder="drop")
        steps = [("select", select)] + steps
    pipe = Pipeline(steps)
    pipe.usable_columns_ = list(columns) if columns is not None else None
    return pipe


# ---------------------------------------------------------------------------
# dataset construction
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series                 # target value at t + horizon (NaN if unknown)
    meta: pd.DataFrame           # ts, road, ts_target, dow_t, hour_t, baselines
    feature_names: list
    interval_min: float
    horizon_min: float
    horizon_steps: int
    season_steps: int            # 0 when no seasonal baseline is possible
    target: str
    dropped_roads: int = 0
    notes: list = field(default_factory=list)


def default_lags(interval_min, span_days):
    """Sensible lag/rolling windows (minutes) for the data's resolution."""
    candidates = [interval_min * k for k in (1, 2, 4)] + [60, 120, 180]
    if span_days >= 3:
        candidates.append(1440)
    if span_days >= 14:
        candidates.append(10080)
    lags = sorted({int(round(c)) for c in candidates if c >= interval_min})
    rolls = sorted({w for w in (60, 180, 1440)
                    if w >= 2 * interval_min and (w <= 1440 or span_days >= 3)})
    return lags, rolls


def horizon_options(interval_min, span_days):
    """Candidate forecast horizons (minutes) for the data's resolution."""
    options = [15, 30, 60, 180, 360, 1440]
    return [h for h in options
            if h >= interval_min and h * 3 <= span_days * 1440]


def make_wide(panel, target, interval_min, agg="mean", max_roads=30):
    """Time × road matrix on a regular grid. Returns (wide, dropped_roads)."""
    secs = int(round(interval_min * 60))
    work = panel[["ts", "road", target]].dropna(subset=[target])
    if work.empty:
        raise ValueError(f"There are no '{target}' values to forecast.")

    dropped = 0
    counts = work["road"].value_counts()
    if len(counts) > max_roads:
        keep = counts.head(max_roads).index
        dropped = len(counts) - max_roads
        work = work[work["road"].isin(keep)]

    work = work.assign(bin=work["ts"].dt.floor(f"{secs}s"))
    grouped = work.groupby(["bin", "road"])[target].agg(agg).unstack("road")
    full = pd.date_range(grouped.index.min(), grouped.index.max(), freq=f"{secs}s")
    return grouped.reindex(full), dropped


def _steps(minutes, interval_min):
    return int(round(minutes / interval_min))


def build_dataset(wide, interval_min, horizon_min, lag_minutes, roll_minutes,
                  target="speed", weekend_days=(5, 6), dropped_roads=0):
    """Create the supervised dataset for one forecast horizon."""
    n_t, n_r = wide.shape
    if n_t < 30:
        raise ValueError("Too few time steps to build a forecasting dataset.")

    h = max(1, _steps(horizon_min, interval_min))
    span_days = n_t * interval_min / 1440
    notes = []

    feats = {"y_now": wide}
    for lm in lag_minutes:
        k = _steps(lm, interval_min)
        if 1 <= k < n_t // 2:
            feats[f"lag_{int(lm)}m"] = wide.shift(k)
    for wm in roll_minutes:
        w = _steps(wm, interval_min)
        if 2 <= w < n_t // 2:
            roll = wide.rolling(w, min_periods=max(2, w // 2))
            feats[f"roll_mean_{int(wm)}m"] = roll.mean()
            feats[f"roll_std_{int(wm)}m"] = roll.std()

    # seasonal-naive baseline: value at (t + h - S), S = week, else day
    season = 0
    for steps_per_season, min_days in ((_steps(10080, interval_min), 14),
                                       (_steps(1440, interval_min), 2)):
        if steps_per_season >= 1 and span_days >= min_days:
            season = steps_per_season
            break
    if season:
        season_eff = season * int(np.ceil(h / season))
        seasonal = wide.shift(season_eff - h)
    else:
        seasonal = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns)
        notes.append("Too little history for a seasonal-naive baseline (needs ≥ 2 days).")

    def flat(frame):
        return frame.to_numpy(dtype=float).ravel()  # time-major, roads fastest

    times = wide.index.to_numpy()
    roads = np.array(list(wide.columns), dtype=object)
    ts = np.repeat(times, n_r)
    ts_target = ts + np.timedelta64(int(round(h * interval_min * 60)), "s")
    ts_target_idx = pd.DatetimeIndex(ts_target)

    meta = pd.DataFrame({
        "ts": ts,
        "road": np.tile(roads, n_t),
        "ts_target": ts_target,
        "dow_t": ts_target_idx.dayofweek.to_numpy(),
        "hour_t": ts_target_idx.hour.to_numpy(),
        "persistence": flat(wide),
        "seasonal_naive": flat(seasonal),
    })
    y = pd.Series(flat(wide.shift(-h)), name="y")

    X = pd.DataFrame({name: flat(frame) for name, frame in feats.items()})

    tod = (ts_target_idx.hour + ts_target_idx.minute / 60).to_numpy()
    dow = ts_target_idx.dayofweek.to_numpy()
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    X["is_weekend"] = np.isin(dow, list(weekend_days)).astype(float)

    if n_r > 1:
        if n_r <= 30:
            for road in roads:
                X[f"road_{road}"] = (meta["road"].to_numpy() == road).astype(float)
        else:
            codes = pd.Series(meta["road"]).astype("category").cat.codes
            X["road_code"] = codes.to_numpy().astype(float)

    keep = meta["persistence"].notna().to_numpy()   # need the current value
    X, y, meta = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True), \
        meta[keep].reset_index(drop=True)

    return Dataset(X=X, y=y, meta=meta, feature_names=list(X.columns),
                   interval_min=float(interval_min), horizon_min=float(h * interval_min),
                   horizon_steps=h, season_steps=int(season), target=target,
                   dropped_roads=dropped_roads, notes=notes)


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

def time_split(ds, test_pct=20):
    """Chronological split with a purge gap of one horizon. Returns index arrays."""
    has_target = ds.y.notna().to_numpy()
    times = np.sort(ds.meta.loc[has_target, "ts"].unique())
    if len(times) < 20:
        raise ValueError("Not enough time steps with a known target (need ≥ 20).")

    cut = int(len(times) * (1 - test_pct / 100))
    cut = min(max(cut, 10), len(times) - 1)
    test_start = times[cut]
    train_end = times[max(cut - ds.horizon_steps, 0)]

    ts = ds.meta["ts"].to_numpy()
    train_idx = np.where(has_target & (ts <= train_end))[0]
    test_idx = np.where(has_target & (ts >= test_start))[0]
    if len(train_idx) < 50 or len(test_idx) < 10:
        raise ValueError(
            "The split leaves too few rows (train ≥ 50 and test ≥ 10 needed). "
            "Use more data or a smaller test share.")
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# baselines & metrics
# ---------------------------------------------------------------------------

def profile_baseline(ds, train_idx, test_idx):
    """Mean target by (road, weekday, hour) of the target time, from training."""
    train = ds.meta.iloc[train_idx].assign(y=ds.y.iloc[train_idx].to_numpy())
    profile = train.groupby(["road", "dow_t", "hour_t"])["y"].mean()
    test = ds.meta.iloc[test_idx]
    keys = pd.MultiIndex.from_arrays([test["road"], test["dow_t"], test["hour_t"]])
    values = profile.reindex(keys).to_numpy()

    road_mean = train.groupby("road")["y"].mean()
    fallback = test["road"].map(road_mean).to_numpy(dtype=float)
    values = np.where(np.isnan(values), fallback, values)
    return np.where(np.isnan(values), float(train["y"].mean()), values)


def baseline_predictions(ds, train_idx, test_idx):
    meta = ds.meta.iloc[test_idx]
    return {
        "persistence": meta["persistence"].to_numpy(),
        "seasonal_naive": meta["seasonal_naive"].to_numpy(),
        "typical_profile": profile_baseline(ds, train_idx, test_idx),
        "train_mean": np.full(len(test_idx), float(ds.y.iloc[train_idx].mean())),
    }


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 2:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "wape": np.nan, "n": int(ok.sum())}
    yt, yp = y_true[ok], y_pred[ok]
    denom = np.abs(yt).sum()
    r2 = r2_score(yt, yp) if np.ptp(yt) > 0 else np.nan
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2),
        "wape": float(np.abs(yt - yp).sum() / denom) if denom > 0 else np.nan,
        "n": int(ok.sum()),
    }


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def fit_with_interval(model_name, X_train, y_train, ts_train, n_estimators=150,
                      coverage=0.8, calib_frac=0.15, max_rows=MAX_TRAIN_ROWS):
    """Fit a model; calibrate an empirical prediction interval.

    The last ``calib_frac`` of the training period (by time) is held out of the
    fit and only used to measure residuals.
    """
    order = np.argsort(np.asarray(ts_train), kind="stable")
    X_train, y_train = X_train.iloc[order], y_train.iloc[order]

    cut = int(len(X_train) * (1 - calib_frac))
    cut = min(max(cut, 20), len(X_train) - 10)
    X_fit, y_fit = X_train.iloc[:cut], y_train.iloc[:cut]
    X_cal, y_cal = X_train.iloc[cut:], y_train.iloc[cut:]

    if len(X_fit) > max_rows:
        pick = np.sort(np.random.default_rng(42).choice(len(X_fit), max_rows, replace=False))
        X_fit, y_fit = X_fit.iloc[pick], y_fit.iloc[pick]

    columns = [c for c in X_fit.columns if X_fit[c].notna().any()]
    if not columns:
        raise ValueError("None of the features has data in the training period.")
    model = build_model(model_name, n_estimators, columns)
    model.fit(X_fit, y_fit)

    residuals = y_cal.to_numpy() - model.predict(X_cal)
    alpha = (1 - coverage) / 2
    low_q, high_q = np.quantile(residuals, [alpha, 1 - alpha])
    return model, float(low_q), float(high_q)


def predict_interval(model, X, low_q, high_q, nonneg=True):
    pred = model.predict(X)
    low, high = pred + low_q, pred + high_q
    if nonneg:
        pred, low, high = (np.clip(a, 0, None) for a in (pred, low, high))
    return pred, low, high


def feature_importance(model, model_name, X_test, y_test, max_rows=1500):
    """Normalised importance (sums to 1)."""
    inner = model.named_steps["model"]
    names = getattr(model, "usable_columns_", None) or list(X_test.columns)

    if hasattr(inner, "feature_importances_"):
        raw = np.asarray(inner.feature_importances_, dtype=float)
        label = "Importance (model-based)"
    elif hasattr(inner, "coef_"):
        raw = np.abs(np.asarray(inner.coef_, dtype=float))
        label = "|standardised coefficient|"
    else:
        n = min(max_rows, len(X_test))
        pick = np.random.default_rng(42).choice(len(X_test), n, replace=False)
        result = permutation_importance(
            model, X_test.iloc[pick], y_test.iloc[pick], n_repeats=3,
            random_state=42, scoring="neg_mean_absolute_error", n_jobs=1)
        raw = np.clip(result.importances_mean, 0, None)
        label = "Permutation importance (MAE increase)"

    total = raw.sum()
    share = raw / total if total > 0 else raw
    table = pd.DataFrame({"Feature": names, "Importance": share})
    return table.sort_values("Importance", ascending=False).reset_index(drop=True), label


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def best_baseline(baseline_metrics):
    scored = {k: v["mae"] for k, v in baseline_metrics.items() if np.isfinite(v["mae"])}
    if not scored:
        return None, np.nan
    key = min(scored, key=scored.get)
    return key, scored[key]


def evaluate(ds, model_name, test_pct=20, n_estimators=150, coverage=0.8,
             refit_for_forecast=True):
    """Train on the past, test on the latest period, compare with baselines."""
    train_idx, test_idx = time_split(ds, test_pct)
    X_train, y_train = ds.X.iloc[train_idx], ds.y.iloc[train_idx]
    X_test, y_test = ds.X.iloc[test_idx], ds.y.iloc[test_idx]
    ts_train = ds.meta["ts"].iloc[train_idx].to_numpy()

    model, low_q, high_q = fit_with_interval(
        model_name, X_train, y_train, ts_train, n_estimators, coverage)
    pred, low, high = predict_interval(model, X_test, low_q, high_q)

    baselines = baseline_predictions(ds, train_idx, test_idx)

    # score every method on the same rows (where all baselines exist)
    comparable = np.ones(len(test_idx), dtype=bool)
    for name in ("persistence", "seasonal_naive") if ds.season_steps else ("persistence",):
        comparable &= np.isfinite(baselines[name])

    model_metrics = regression_metrics(y_test.to_numpy()[comparable], pred[comparable])
    baseline_metrics = {
        k: regression_metrics(y_test.to_numpy()[comparable], v[comparable])
        for k, v in baselines.items()
        if k != "seasonal_naive" or ds.season_steps
    }
    best_key, best_mae = best_baseline(baseline_metrics)
    skill = (1 - model_metrics["mae"] / best_mae) if best_mae and np.isfinite(best_mae) else np.nan

    inside = (y_test.to_numpy() >= low) & (y_test.to_numpy() <= high)
    importance, importance_label = feature_importance(model, model_name, X_test, y_test)

    results = ds.meta.iloc[test_idx][["ts", "road", "ts_target"]].reset_index(drop=True)
    results["actual"] = y_test.to_numpy()
    results["predicted"] = pred
    results["low"], results["high"] = low, high
    for key, values in baselines.items():
        results[key] = values
    results["residual"] = results["actual"] - results["predicted"]

    forecast = latest_forecast(
        ds, model_name, model, low_q, high_q, n_estimators, coverage, refit_for_forecast)

    return {
        "model_name": model_name,
        "target": ds.target,
        "horizon_min": ds.horizon_min,
        "interval_min": ds.interval_min,
        "features": ds.feature_names,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "comparable_rows": int(comparable.sum()),
        "test_start": ds.meta["ts"].iloc[test_idx].min(),
        "test_end": ds.meta["ts"].iloc[test_idx].max(),
        "metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "best_baseline": best_key,
        "skill": float(skill) if np.isfinite(skill) else None,
        "coverage_target": coverage,
        "coverage_actual": float(inside.mean()),
        "interval_width": float(np.mean(high - low)),
        "results": results,
        "importance": importance,
        "importance_label": importance_label,
        "forecast": forecast,
        "notes": list(ds.notes) + ([f"{ds.dropped_roads} low-volume road(s) were left out "
                                    "to keep the model small."] if ds.dropped_roads else []),
    }


def latest_forecast(ds, model_name, fitted_model, low_q, high_q, n_estimators,
                    coverage, refit=True):
    """Forecast ``horizon`` ahead from the most recent observation per road."""
    last_ts = ds.meta["ts"].max()
    rows = np.where(ds.meta["ts"].to_numpy() == last_ts)[0]
    if len(rows) == 0:
        return pd.DataFrame()

    model = fitted_model
    if refit:
        known = np.where(ds.y.notna().to_numpy())[0]
        if len(known) >= 100:
            model, low_q, high_q = fit_with_interval(
                model_name, ds.X.iloc[known], ds.y.iloc[known],
                ds.meta["ts"].iloc[known].to_numpy(), n_estimators, coverage)

    pred, low, high = predict_interval(model, ds.X.iloc[rows], low_q, high_q)
    meta = ds.meta.iloc[rows]
    return pd.DataFrame({
        "road": meta["road"].to_numpy(),
        "last_observed_at": meta["ts"].to_numpy(),
        "last_value": meta["persistence"].to_numpy(),
        "forecast_for": meta["ts_target"].to_numpy(),
        "forecast": pred, "low": low, "high": high,
    }).reset_index(drop=True)


def time_series_cv(ds, model_name, n_splits=4, n_estimators=100, coverage=0.8):
    """Expanding-window CV; every fold tests on a later block than it trains on."""
    has_target = ds.y.notna().to_numpy()
    times = np.sort(ds.meta.loc[has_target, "ts"].unique())
    block = len(times) // (n_splits + 1)
    if block < 10:
        raise ValueError("Not enough history for this many folds; use fewer splits.")

    ts = ds.meta["ts"].to_numpy()
    rows = []
    for fold in range(1, n_splits + 1):
        test_lo = times[fold * block]
        test_hi = times[min((fold + 1) * block, len(times)) - 1] if fold < n_splits \
            else times[-1]
        train_hi = times[max(fold * block - ds.horizon_steps, 0)]

        train_idx = np.where(has_target & (ts <= train_hi))[0]
        test_idx = np.where(has_target & (ts >= test_lo) & (ts <= test_hi))[0]
        if len(train_idx) < 50 or len(test_idx) < 10:
            continue

        model, lq, hq = fit_with_interval(
            model_name, ds.X.iloc[train_idx], ds.y.iloc[train_idx], ts[train_idx],
            n_estimators, coverage)
        pred, low, high = predict_interval(model, ds.X.iloc[test_idx], lq, hq)
        y_test = ds.y.iloc[test_idx].to_numpy()

        baselines = baseline_predictions(ds, train_idx, test_idx)
        ok = np.isfinite(baselines["persistence"])
        if ds.season_steps:
            ok &= np.isfinite(baselines["seasonal_naive"])

        row = {
            "fold": fold,
            "train_rows": int(len(train_idx)),
            "test_rows": int(ok.sum()),
            "test_from": pd.Timestamp(test_lo),
            "test_to": pd.Timestamp(test_hi),
            "model_mae": regression_metrics(y_test[ok], pred[ok])["mae"],
            "coverage": float(((y_test >= low) & (y_test <= high)).mean()),
        }
        for key in BASELINES:
            if key == "seasonal_naive" and not ds.season_steps:
                continue
            row[f"{key}_mae"] = regression_metrics(y_test[ok], baselines[key][ok])["mae"]
        rows.append(row)

    if not rows:
        raise ValueError("No fold had enough data. Use fewer splits or more history.")
    out = pd.DataFrame(rows)
    base_cols = [c for c in out.columns if c.endswith("_mae") and c != "model_mae"]
    out["best_baseline_mae"] = out[base_cols].min(axis=1)
    out["skill"] = 1 - out["model_mae"] / out["best_baseline_mae"]
    return out


def horizon_sweep(wide, interval_min, horizons_min, model_name, lag_minutes,
                  roll_minutes, target, weekend_days=(5, 6), test_pct=20,
                  n_estimators=100, progress=None):
    """Train one model per horizon and compare MAE with the best baseline."""
    rows = []
    for i, horizon in enumerate(horizons_min):
        if progress:
            progress(i / max(len(horizons_min), 1), f"Horizon {horizon} min")
        try:
            ds = build_dataset(wide, interval_min, horizon, lag_minutes, roll_minutes,
                               target, weekend_days)
            result = evaluate(ds, model_name, test_pct, n_estimators,
                              refit_for_forecast=False)
        except ValueError as exc:
            rows.append({"horizon_min": horizon, "error": str(exc)})
            continue

        row = {
            "horizon_min": horizon,
            "model_mae": result["metrics"]["mae"],
            "best_baseline": result["best_baseline"],
            "best_baseline_mae": min(
                (v["mae"] for v in result["baseline_metrics"].values()
                 if np.isfinite(v["mae"])), default=np.nan),
            "skill": result["skill"],
            "coverage": result["coverage_actual"],
        }
        for key, metrics in result["baseline_metrics"].items():
            row[f"{key}_mae"] = metrics["mae"]
        rows.append(row)

    if progress:
        progress(1.0, "Done")
    return pd.DataFrame(rows)
