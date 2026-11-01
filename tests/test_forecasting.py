import numpy as np

from mobility import forecasting as F, insights as I


def _wide(traffic, traffic_map):
    panel = I.prepare_panel(traffic, traffic_map)
    wide, dropped = F.make_wide(panel, "speed", 15)
    return wide, dropped


def _dataset(traffic, traffic_map, horizon=60):
    wide, dropped = _wide(traffic, traffic_map)
    lags, rolls = F.default_lags(15, 28)
    return F.build_dataset(wide, 15, horizon, lags, rolls, "speed", (5, 6), dropped)


def test_split_is_chronological_with_purge_gap(traffic, traffic_map):
    ds = _dataset(traffic, traffic_map, horizon=60)
    train, test = F.time_split(ds, 20)
    ts = ds.meta["ts"].to_numpy()
    assert ts[train].max() < ts[test].min()
    gap_min = (ts[test].min() - ts[train].max()) / np.timedelta64(1, "m")
    assert gap_min >= 60


def test_evaluate_reports_baselines_and_interval(traffic, traffic_map):
    ds = _dataset(traffic, traffic_map)
    res = F.evaluate(ds, F.GB_NAME, 20, 50, 0.8)
    assert set(res["baseline_metrics"]) >= {"persistence", "seasonal_naive"}
    assert res["metrics"]["mae"] > 0
    assert 0.5 < res["coverage_actual"] <= 1.0


def test_model_not_worse_than_mean_baseline(traffic, traffic_map):
    ds = _dataset(traffic, traffic_map)
    res = F.evaluate(ds, F.GB_NAME, 20, 50, 0.8)
    assert res["metrics"]["mae"] <= res["baseline_metrics"]["train_mean"]["mae"]


def test_cross_validation_runs(traffic, traffic_map):
    ds = _dataset(traffic, traffic_map)
    table = F.time_series_cv(ds, F.RIDGE_NAME, 3, 50, 0.8)
    assert len(table) >= 2
