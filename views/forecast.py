"""Forecasting page: lag-feature models vs honest baselines, CV, horizons, intervals."""

import json
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mobility import forecasting as F
from mobility import insights as I
from mobility.state import cached
from mobility.ui import fmt_minutes, metric_row, safe_section, show_df, show_plot
from .common import get_panel, mapping_key

TARGET_LABELS = {"speed": "Speed (km/h)", "volume": "Traffic volume"}
INTERVAL_OPTIONS = [1, 5, 10, 15, 30, 60]


def _clean(value):
    """JSON-safe copy (NaN → None, numpy → python)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


def _fmt(v, digits=3):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def _result_json(res):
    keep = ("model_name", "target", "horizon_min", "interval_min", "features",
            "train_rows", "test_rows", "comparable_rows", "test_start", "test_end",
            "metrics", "baseline_metrics", "best_baseline", "skill",
            "coverage_target", "coverage_actual", "interval_width")
    return json.dumps(_clean({k: res[k] for k in keep}), indent=2)


def _show_result(res, df_version):
    st.subheader("Evaluation on the most recent period")
    if res["df_version"] != df_version:
        st.warning("The dataset changed after this model was trained. Retrain to update.")
    for note in res["notes"]:
        st.info(note)

    mm = res["metrics"]
    best = res["best_baseline"]
    best_label = F.BASELINE_LABELS.get(best, "n/a")
    skill = res["skill"]

    metric_row([
        ("MAE", _fmt(mm["mae"])),
        ("RMSE", _fmt(mm["rmse"])),
        ("R²", _fmt(mm["r2"])),
        ("Skill vs best baseline", "n/a" if skill is None else f"{skill * 100:+.1f}%"),
        (f"{int(res['coverage_target'] * 100)}% interval coverage",
         f"{res['coverage_actual'] * 100:.0f}%"),
    ])

    if skill is None:
        st.info("No baseline could be computed for comparison.")
    elif skill > 0.05:
        st.success(f"The model's average error is {skill * 100:.0f}% lower than the best "
                   f"baseline ({best_label}).")
    elif skill > 0:
        st.info(f"The model is only marginally better ({skill * 100:.1f}%) than the best "
                f"baseline ({best_label}). A simple rule is almost as good here.")
    else:
        st.warning(f"The model does **not** beat the best baseline ({best_label}). "
                   "Try another model, more history, or different lags.")

    rows = [{"Method": f"{res['model_name']} (model)", **{k.upper(): v for k, v in mm.items()
                                                      if k != "n"}}]
    for key, metrics in res["baseline_metrics"].items():
        rows.append({"Method": F.BASELINE_LABELS[key],
                     **{k.upper(): v for k, v in metrics.items() if k != "n"}})
    table = pd.DataFrame(rows).rename(columns={"WAPE": "WAPE (error ÷ actual)"})
    show_df(table, hide_index=True)
    st.caption(
        f"Horizon: {fmt_minutes(res['horizon_min'])} ahead · resolution "
        f"{fmt_minutes(res['interval_min'])} · trained on {res['train_rows']:,} rows · "
        f"tested on {res['comparable_rows']:,} later rows "
        f"({res['test_start']:%d %b} – {res['test_end']:%d %b %Y}). Every method is "
        "scored on exactly the same rows. Lower error is better.")

    results = res["results"]
    roads = sorted(results["road"].unique())
    c1, c2 = st.columns(2)
    road = c1.selectbox("Road", roads, key="fc_road")
    days = c2.slider("Days shown", 1, 14, 3, key="fc_days")
    sub = results[results["road"] == road].sort_values("ts_target")
    sub = sub[sub["ts_target"] >= sub["ts_target"].max() - pd.Timedelta(days=days)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["ts_target"], y=sub["high"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sub["ts_target"], y=sub["low"], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(99,110,250,0.2)",
                             name=f"{int(res['coverage_target'] * 100)}% interval"))
    fig.add_trace(go.Scatter(x=sub["ts_target"], y=sub["actual"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=sub["ts_target"], y=sub["predicted"], mode="lines",
                             name="Predicted"))
    if "seasonal_naive" in res["baseline_metrics"]:
        fig.add_trace(go.Scatter(x=sub["ts_target"], y=sub["seasonal_naive"], mode="lines",
                                 name="Seasonal naive", line=dict(dash="dot", width=1)))
    fig.update_layout(title=f"{road}: actual vs forecast", xaxis_title="Time",
                      yaxis_title=TARGET_LABELS.get(res["target"], res["target"]))
    show_plot(fig, key="fc_line")

    left, right = st.columns(2)
    with left:
        by_hour = (results.assign(abs_err=results["residual"].abs(),
                                  hour=results["ts_target"].dt.hour)
                   .groupby("hour")["abs_err"].mean().reset_index())
        show_plot(px.bar(by_hour, x="hour", y="abs_err",
                         labels={"hour": "Hour of day", "abs_err": "Mean absolute error"},
                         title="When is the model least accurate?"), key="fc_hour")
    with right:
        imp = res["importance"].head(15).sort_values("Importance")
        show_plot(px.bar(imp, x="Importance", y="Feature", orientation="h",
                         title=res["importance_label"]), key="fc_imp")

    forecast = res["forecast"]
    if not forecast.empty:
        st.subheader("Latest forecast")
        st.caption(f"What the model predicts for {fmt_minutes(res['horizon_min'])} after the "
                   "last observation in the data (model refit on all history).")
        fig = go.Figure(go.Bar(
            x=forecast["road"], y=forecast["forecast"], name="Forecast",
            error_y=dict(type="data", symmetric=False,
                         array=(forecast["high"] - forecast["forecast"]).to_numpy(),
                         arrayminus=(forecast["forecast"] - forecast["low"]).to_numpy())))
        fig.add_trace(go.Scatter(x=forecast["road"], y=forecast["last_value"],
                                 mode="markers", name="Last observed",
                                 marker=dict(size=10, symbol="diamond")))
        fig.update_layout(yaxis_title=TARGET_LABELS.get(res["target"], res["target"]))
        show_plot(fig, key="fc_latest")
        show_df(forecast, hide_index=True)

    st.download_button("Download test predictions (CSV)",
                       results.to_csv(index=False).encode("utf-8"),
                       "forecast_test_predictions.csv", "text/csv")
    st.download_button("Download model summary (JSON)", _result_json(res),
                       "forecast_summary.json", "application/json")


def _show_cv(cv):
    st.subheader("Time-series cross-validation")
    st.caption("Each fold trains on an earlier block and tests on the next one, so the "
               "model is always tested on the future relative to its training data.")
    cols = ["fold", "train_rows", "test_rows", "test_from", "test_to", "model_mae",
            "best_baseline_mae", "skill", "coverage"]
    show_df(cv[cols].rename(columns={
        "model_mae": "Model MAE", "best_baseline_mae": "Best baseline MAE",
        "skill": "Skill", "coverage": "Interval coverage"}), hide_index=True)
    metric_row([("Mean skill vs best baseline", f"{cv['skill'].mean() * 100:+.1f}%"),
                ("Folds beating baseline", f"{int((cv['skill'] > 0).sum())}/{len(cv)}"),
                ("Mean interval coverage", f"{cv['coverage'].mean() * 100:.0f}%")])
    long = cv.melt(id_vars="fold", value_vars=["model_mae", "best_baseline_mae"],
                   var_name="Method", value_name="MAE")
    long["Method"] = long["Method"].map({"model_mae": "Model", "best_baseline_mae": "Best baseline"})
    show_plot(px.bar(long, x="fold", y="MAE", color="Method", barmode="group",
                     title="Error by fold"), key="fc_cv_chart")


def _show_sweep(sweep):
    st.subheader("Accuracy vs forecast horizon")
    ok = sweep.drop(columns=["error"], errors="ignore").dropna(subset=["model_mae"])
    if ok.empty:
        st.warning("No horizon could be evaluated with the current data.")
        return
    ok = ok.assign(Horizon=ok["horizon_min"].map(fmt_minutes))
    mae_cols = [c for c in ok.columns if c.endswith("_mae") and c != "best_baseline_mae"]
    long = ok.melt(id_vars=["horizon_min", "Horizon"], value_vars=mae_cols,
                   var_name="Method", value_name="MAE")
    long["Method"] = long["Method"].replace(
        {"model_mae": "Model", **{f"{k}_mae": F.BASELINE_LABELS[k] for k in F.BASELINES}})
    fig = px.line(long, x="horizon_min", y="MAE", color="Method", markers=True,
                  labels={"horizon_min": "Forecast horizon (minutes)"},
                  title="Mean absolute error by forecast horizon")
    fig.update_xaxes(type="log")
    show_plot(fig, key="fc_sweep")
    st.caption("Persistence is hard to beat very close in time; seasonal and profile "
               "baselines matter further ahead. The gap between the model and the best "
               "baseline is what the model adds.")
    show_df(ok[["Horizon", "model_mae", "best_baseline", "best_baseline_mae", "skill",
                "coverage"]].rename(columns={
                    "model_mae": "Model MAE", "best_baseline": "Best baseline",
                    "best_baseline_mae": "Best baseline MAE", "skill": "Skill",
                    "coverage": "Interval coverage"}), hide_index=True)


def render(df, m):
    st.header("Traffic Forecasting")
    st.write(
        "Predict speed or volume a chosen time ahead from recent values (lags), rolling "
        "statistics and calendar features. Every model is compared with simple baselines "
        "on the **latest** part of the data, so you can see whether it adds real value.")

    if not m.get("timestamp") or not (m.get("speed") or m.get("volume")):
        st.info("Forecasting needs a timestamp and a speed and/or volume column "
                "(sidebar → Column mapping).")
        return

    panel = get_panel(df, m)
    if panel is None:
        st.warning("No usable rows: the timestamp column could not be parsed.")
        return

    targets = [k for k in ("speed", "volume") if k in panel.columns]
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Target", targets, format_func=TARGET_LABELS.get, key="fc_target")

    inferred = cached("interval", lambda: I.infer_interval_minutes(panel), mapping_key(m))
    index = next((i for i, v in enumerate(INTERVAL_OPTIONS) if v >= inferred - 1e-9),
                 len(INTERVAL_OPTIONS) - 1)
    interval = c2.selectbox("Time resolution (minutes)", INTERVAL_OPTIONS, index=index,
                            key="fc_interval",
                            help=f"The data's typical spacing is about {inferred:g} minutes.")

    span_days = (panel["ts"].max() - panel["ts"].min()).total_seconds() / 86400
    horizons = F.horizon_options(interval, span_days)
    if not horizons:
        st.warning("The data span is too short for any forecast horizon at this resolution.")
        return
    horizon = c3.selectbox("Forecast horizon", horizons, format_func=fmt_minutes,
                           index=min(2, len(horizons) - 1), key="fc_horizon")

    c4, c5 = st.columns(2)
    models = F.available_models()
    model_name = c4.selectbox("Model", models, key="fc_model")
    test_pct = c5.slider("Test share (latest part of the data)", 10, 40, 20, step=5,
                         key="fc_test")

    with st.expander("Advanced settings"):
        lag_default, roll_default = F.default_lags(interval, span_days)
        lag_options = sorted(set(lag_default) | {15, 30, 60, 120, 180, 360, 1440, 10080})
        lag_options = [v for v in lag_options if v >= interval]
        lags = st.multiselect("Lag features (minutes ago)", lag_options, lag_default,
                              format_func=fmt_minutes, key="fc_lags")
        roll_options = sorted(set(roll_default) | {60, 180, 360, 1440})
        rolls = st.multiselect("Rolling-window features (minutes)", roll_options, roll_default,
                               format_func=fmt_minutes, key="fc_rolls")
        a1, a2, a3 = st.columns(3)
        n_estimators = a1.slider("Trees (Random Forest)", 50, 300, 150, step=50, key="fc_trees")
        coverage = a2.slider("Prediction-interval coverage", 0.5, 0.95, 0.8, 0.05, key="fc_cov")
        max_roads = a3.slider("Maximum roads modelled", 5, 100, 30, 5, key="fc_roads")
        agg = "mean"
        if target == "volume":
            agg = st.radio("Combine several records in one time step by",
                           ["mean", "sum"], horizontal=True, key="fc_agg")

    weekend = tuple(m.get("weekend_days", (5, 6)))
    wide_key = (mapping_key(m), target, interval, agg, max_roads)
    with st.spinner("Preparing time series..."):
        wide, dropped = cached("fc_wide", lambda: F.make_wide(
            panel, target, interval, agg, max_roads), wide_key)
    st.caption(f"{wide.shape[0]:,} time steps × {wide.shape[1]} road(s) at "
               f"{fmt_minutes(interval)} resolution · {wide.notna().mean().mean() * 100:.0f}% "
               "of cells observed.")

    b1, b2, b3 = st.columns(3)
    train = b1.button("Train and evaluate", type="primary", key="fc_train")
    run_cv = b2.button("Run time-series cross-validation", key="fc_run_cv")
    run_sweep = b3.button("Compare horizons", key="fc_run_sweep")

    store = st.session_state.forecast
    version = st.session_state.df_version

    if train:
        with safe_section("Training the forecasting model"), st.spinner("Training..."):
            ds = F.build_dataset(wide, interval, horizon, lags, rolls, target, weekend, dropped)
            res = F.evaluate(ds, model_name, test_pct, n_estimators, coverage)
            res["df_version"] = version
            store["result"] = res

    if run_cv:
        with safe_section("Cross-validation"), st.spinner("Running folds..."):
            ds = F.build_dataset(wide, interval, horizon, lags, rolls, target, weekend, dropped)
            store["cv"] = {"table": F.time_series_cv(ds, model_name, 4, 100, coverage),
                           "df_version": version}

    if run_sweep:
        with safe_section("Horizon comparison"):
            bar = st.progress(0.0, text="Starting...")
            sweep = F.horizon_sweep(
                wide, interval, horizons, model_name, lags, rolls, target, weekend, test_pct,
                100, progress=lambda f, text: bar.progress(min(f, 1.0), text=text))
            bar.empty()
            store["sweep"] = {"table": sweep, "df_version": version}

    if "result" in store:
        st.divider()
        with safe_section("Showing results"):
            _show_result(store["result"], version)
    if "cv" in store:
        st.divider()
        with safe_section("Showing cross-validation"):
            if store["cv"]["df_version"] != version:
                st.warning("Cross-validation results are from an earlier version of the data.")
            _show_cv(store["cv"]["table"])
    if "sweep" in store:
        st.divider()
        with safe_section("Showing horizon comparison"):
            if store["sweep"]["df_version"] != version:
                st.warning("Horizon results are from an earlier version of the data.")
            _show_sweep(store["sweep"]["table"])

    if not store:
        st.info("Choose settings and press **Train and evaluate**.")
