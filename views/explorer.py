"""Data Explorer page."""

import plotly.express as px
import streamlit as st

from mobility.config import MAX_PLOT_POINTS
from mobility.ui import metric_row, safe_section, show_df, show_plot
from mobility.utils import is_numeric_col, numeric_columns, sample_for_plot


def render(df, m):
    st.header("Data Explorer")

    st.subheader("Preview")
    n_rows = st.number_input("Rows to preview", min_value=10, max_value=5000,
                             value=100, step=10)
    show_df(df.head(int(n_rows)))
    st.caption(f"Showing up to {int(n_rows):,} of {len(df):,} rows.")

    st.subheader("Column statistics")
    column = st.selectbox("Select a column", list(df.columns))
    series = df[column]
    metric_row([
        ("Data type", str(series.dtype)),
        ("Missing values", f"{int(series.isna().sum()):,}"),
        ("Unique values", f"{int(series.nunique(dropna=True)):,}"),
    ])

    if is_numeric_col(series):
        show_df(series.describe().to_frame("Statistics"))
        plot_data, sampled = sample_for_plot(series.dropna().to_frame(), MAX_PLOT_POINTS)
        show_plot(px.histogram(plot_data, x=column, title=f"Distribution of {column}"),
                  key="ex_hist")
        if sampled:
            st.caption(f"Histogram uses a random sample of {MAX_PLOT_POINTS:,} values.")
    else:
        counts = (series.astype("string").fillna("(Missing)").value_counts().head(30)
                  .rename_axis("Value").reset_index(name="Count"))
        show_df(counts, hide_index=True)
        if not counts.empty:
            show_plot(px.bar(counts, x="Value", y="Count", title=f"Top values: {column}"),
                      key="ex_top")

    st.subheader("Custom chart")
    numbers = numeric_columns(df)
    if not numbers:
        st.info("No numeric columns are available for charting.")
        return

    chart_type = st.selectbox(
        "Chart type", ["Scatter", "Line", "Bar (mean of Y by X)", "Box", "Histogram"])
    x_col = st.selectbox("X-axis", list(df.columns), key="explorer_x")
    y_col = None
    if chart_type != "Histogram":
        y_col = st.selectbox("Numeric Y-axis", numbers, key="explorer_y")

    with safe_section("Creating the chart", show_details=False):
        cols = [x_col] + ([y_col] if y_col and y_col != x_col else [])
        data = df[cols].dropna()
        sampled = False

        if chart_type == "Bar (mean of Y by X)":
            grouped = data.groupby(x_col)[y_col].mean().reset_index()
            if len(grouped) > 100:
                st.warning("More than 100 groups — showing the first 100.")
                grouped = grouped.head(100)
            fig = px.bar(grouped, x=x_col, y=y_col)
        else:
            data, sampled = sample_for_plot(data, MAX_PLOT_POINTS)
            if chart_type == "Scatter":
                fig = px.scatter(data, x=x_col, y=y_col)
            elif chart_type == "Line":
                fig = px.line(data.sort_values(x_col), x=x_col, y=y_col)
            elif chart_type == "Box":
                fig = px.box(data, x=x_col if data[x_col].nunique() <= 30 else None, y=y_col)
            else:
                fig = px.histogram(data, x=x_col)

        show_plot(fig, key="explorer_custom_chart")
        if sampled:
            st.caption(f"Showing a random sample of {MAX_PLOT_POINTS:,} points.")
