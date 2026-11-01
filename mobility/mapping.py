"""Folium map builders. No Streamlit imports — each function returns a Map."""

import copy
import html

import folium
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap
from folium.plugins import HeatMap, MarkerCluster, TimestampedGeoJson

from .geo import blend

CONGESTION_STOPS = ("#1a9850", "#fee08b", "#d73027")      # green → yellow → red
SPEED_STOPS = ("#d73027", "#fee08b", "#1a9850")            # slow red → fast green
SEQUENTIAL_STOPS = ("#2c7bb6", "#ffffbf", "#d7191c")


def _colormap(stops, vmin, vmax, caption):
    if not vmax > vmin:
        vmax = vmin + 1
    return LinearColormap(list(stops), vmin=float(vmin), vmax=float(vmax), caption=caption)


def base_map(lats, lons, tiles="OpenStreetMap"):
    """Map fitted to the given coordinates."""
    lats, lons = np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)
    fmap = folium.Map(tiles=tiles, control_scale=True, prefer_canvas=True)
    if len(lats) == 0:
        return fmap
    if np.isclose(lats.min(), lats.max()) and np.isclose(lons.min(), lons.max()):
        fmap.location = [float(lats[0]), float(lons[0])]
        fmap.options["zoom"] = 14
    else:
        fmap.fit_bounds([[float(lats.min()), float(lons.min())],
                         [float(lats.max()), float(lons.max())]])
    return fmap


def popup_html(record):
    lines = [f"<b>{html.escape(str(k))}</b>: {html.escape(str(v))}"
             for k, v in record.items() if pd.notna(v)]
    return "<br>".join(lines) or "Record"


# ---------------------------------------------------------------------------
# raw points
# ---------------------------------------------------------------------------

def points_map(df, lat_col, lon_col, mode="Points", color_col=None,
               weight_col=None, max_points=2000, popup_cols=None):
    """Points / clusters / heatmap of valid coordinates.

    Returns (map, legend_html, shown, total_valid).
    """
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    valid = lat.between(-90, 90) & lon.between(-180, 180)

    data = df.loc[valid].copy()
    data[lat_col], data[lon_col] = lat[valid], lon[valid]
    total = len(data)
    if total == 0:
        raise ValueError("No valid coordinate pairs were found.")
    if total > max_points:
        data = data.sample(max_points, random_state=42)

    lats, lons = data[lat_col].to_numpy(), data[lon_col].to_numpy()
    fmap = base_map(lats, lons)
    legend_html = ""

    if mode == "Heatmap":
        points = data[[lat_col, lon_col]].to_numpy().tolist()
        if weight_col:
            w = pd.to_numeric(data[weight_col], errors="coerce").fillna(0)
            w = w - w.min()
            w = w / w.max() if w.max() > 0 else w + 1
            points = [p + [float(x)] for p, x in zip(points, w)]
        HeatMap(points, radius=12, blur=15).add_to(fmap)
        return fmap, legend_html, len(data), total

    cols = list(popup_cols) if popup_cols else list(df.columns[:8])
    if color_col and color_col not in cols:
        cols.append(color_col)
    records = data[cols].to_dict("records")

    colors = ["#3388ff"] * len(data)
    if color_col:
        raw = data[color_col]
        numeric = pd.to_numeric(raw, errors="coerce")
        if pd.api.types.is_numeric_dtype(raw) and numeric.notna().any():
            vmin, vmax = numeric.quantile(0.02), numeric.quantile(0.98)
            if not vmax > vmin:
                vmin, vmax = numeric.min(), numeric.max()
            _colormap(SEQUENTIAL_STOPS, vmin, vmax, color_col).add_to(fmap)
            colors = [blend(v, vmin, vmax, SEQUENTIAL_STOPS) if pd.notna(v) else "#999999"
                      for v in numeric]
        else:
            labels = raw.astype("string").fillna("(Missing)")
            top = list(labels.value_counts().head(10).index)
            palette = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                       "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]
            lookup = {cat: palette[i % len(palette)] for i, cat in enumerate(top)}
            colors = [lookup.get(v, "#999999") for v in labels]
            items = [(c, lookup[c]) for c in top]
            if labels.nunique() > len(top):
                items.append(("Other", "#999999"))
            legend_html = " &nbsp; ".join(
                f"<span style='color:{col}'>●</span> {html.escape(str(name))}"
                for name, col in items)

    target = MarkerCluster().add_to(fmap) if mode == "Clusters" else fmap
    for la, lo, record, color in zip(lats, lons, records, colors):
        folium.CircleMarker(
            location=[float(la), float(lo)], radius=4, color=color, fill=True,
            fill_color=color, fill_opacity=0.75,
            popup=folium.Popup(popup_html(record), max_width=300),
        ).add_to(target)
    return fmap, legend_html, len(data), total


# ---------------------------------------------------------------------------
# congestion
# ---------------------------------------------------------------------------

def segment_map(ranking):
    """Road segments (points) coloured by mean congestion index."""
    data = ranking.dropna(subset=["lat", "lon"])
    if data.empty:
        raise ValueError("Segments need latitude/longitude columns to be mapped.")

    fmap = base_map(data["lat"], data["lon"])
    vmax = max(0.5, float(data["mean_ci"].max()))
    _colormap(CONGESTION_STOPS, 0, vmax, "Mean congestion index").add_to(fmap)

    volume = data["mean_volume"] if "mean_volume" in data else None
    for i, row in enumerate(data.itertuples(index=False)):
        radius = 10
        if volume is not None and volume.max() > 0:
            radius = 7 + 13 * float(row.mean_volume) / float(volume.max())
        text = (
            f"<b>{html.escape(str(row.road))}</b><br>"
            f"Mean speed: {row.mean_speed:.0f} km/h<br>"
            f"Free-flow: {row.free_flow:.0f} km/h<br>"
            f"Congestion index: {row.mean_ci:.2f}<br>"
            f"Heavy congestion: {row.pct_heavy:.0f}% of time<br>"
            f"Worst hour: {int(row.worst_hour):02d}:00")
        color = blend(row.mean_ci, 0, vmax, CONGESTION_STOPS)
        folium.CircleMarker(
            location=[float(row.lat), float(row.lon)], radius=radius, color="#333333",
            weight=1, fill=True, fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(text, max_width=280),
            tooltip=f"{row.road}: CI {row.mean_ci:.2f}",
        ).add_to(fmap)
    return fmap


def congestion_time_map(panel):
    """Animated map: average congestion by hour of day for each segment."""
    needed = {"lat", "lon", "ci", "road", "ts"}
    if not needed.issubset(panel.columns):
        raise ValueError("The animation needs timestamp, speed and coordinates.")

    data = panel.dropna(subset=["lat", "lon", "ci"])
    if data.empty:
        raise ValueError("No observations with valid coordinates and speed.")

    hourly = (data.groupby(["road", data["ts"].dt.hour.rename("hour")])
              .agg(ci=("ci", "mean"), speed=("speed", "mean"),
                   lat=("lat", "median"), lon=("lon", "median"))
              .reset_index())
    if len(hourly["road"].unique()) > 1500:
        raise ValueError("Too many segments for an animation (limit 1,500).")

    fmap = base_map(hourly["lat"], hourly["lon"])
    vmax = max(0.5, float(hourly["ci"].max()))
    _colormap(CONGESTION_STOPS, 0, vmax, "Average congestion index").add_to(fmap)

    features = []
    for row in hourly.itertuples(index=False):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row.lon), float(row.lat)]},
            "properties": {
                "time": f"2000-01-01T{int(row.hour):02d}:00:00",
                "popup": (f"<b>{html.escape(str(row.road))}</b><br>"
                          f"{int(row.hour):02d}:00 — CI {row.ci:.2f}, "
                          f"{row.speed:.0f} km/h"),
                "icon": "circle",
                "iconstyle": {
                    "fillColor": blend(row.ci, 0, vmax, CONGESTION_STOPS),
                    "fillOpacity": 0.85, "stroke": "true", "color": "#333333",
                    "radius": 9,
                },
            },
        })

    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period="PT1H", duration="PT1H", add_last_point=False, auto_play=False,
        loop=True, loop_button=True, max_speed=3, date_options="HH:mm",
        time_slider_drag_update=True,
    ).add_to(fmap)
    return fmap


# ---------------------------------------------------------------------------
# zones
# ---------------------------------------------------------------------------

def grid_map(cells, value_label="value"):
    """Square zones coloured by value."""
    if cells.empty:
        raise ValueError("No grid cells to draw.")
    fmap = base_map(
        np.r_[cells["lat_min"], cells["lat_max"]], np.r_[cells["lon_min"], cells["lon_max"]])

    values = cells["value"].astype(float)
    vmin, vmax = values.quantile(0.02), values.quantile(0.98)
    if not vmax > vmin:
        vmin, vmax = values.min(), values.max()
    _colormap(SEQUENTIAL_STOPS, vmin, vmax, value_label).add_to(fmap)

    for row in cells.itertuples(index=False):
        color = blend(row.value, vmin, vmax, SEQUENTIAL_STOPS)
        folium.Rectangle(
            bounds=[[row.lat_min, row.lon_min], [row.lat_max, row.lon_max]],
            color=color, weight=1, fill=True, fill_color=color, fill_opacity=0.6,
            tooltip=f"{value_label}: {row.value:,.2f} ({int(row.count):,} records)",
        ).add_to(fmap)
    return fmap


def choropleth_map(geojson, zone_table, value_label="value"):
    """Polygon zones coloured by aggregated value (grey = no data)."""
    shapes = copy.deepcopy(geojson)
    values = zone_table["value"].astype(float)
    finite = values[np.isfinite(values) & (zone_table["count"] > 0)]
    vmin = float(finite.min()) if len(finite) else 0.0
    vmax = float(finite.max()) if len(finite) else 1.0

    colors = []
    for feature, row in zip(shapes["features"], zone_table.itertuples(index=False)):
        has_data = row.count > 0 and np.isfinite(row.value)
        colors.append(blend(row.value, vmin, vmax, SEQUENTIAL_STOPS) if has_data else "#cccccc")
        feature["properties"] = {
            "_zone": str(row.name),
            "_records": f"{int(row.count):,}",
            "_value": f"{row.value:,.2f}" if has_data else "no data",
        }

    lats, lons = [], []
    for feature in shapes["features"]:
        geometry = feature["geometry"]
        polys = ([geometry["coordinates"]] if geometry["type"] == "Polygon"
                 else geometry["coordinates"])
        for poly in polys:
            ring = np.asarray(poly[0], dtype=float)
            lons.extend(ring[:, 0])
            lats.extend(ring[:, 1])

    fmap = base_map(lats, lons)
    if len(finite):
        _colormap(SEQUENTIAL_STOPS, vmin, vmax, value_label).add_to(fmap)

    def style(feature):
        index = shapes["features"].index(feature)
        return {"fillColor": colors[index], "color": "#444444", "weight": 1,
                "fillOpacity": 0.65}

    folium.GeoJson(
        shapes, style_function=style,
        tooltip=folium.GeoJsonTooltip(
            fields=["_zone", "_records", "_value"],
            aliases=["Zone", "Records", value_label]),
    ).add_to(fmap)
    return fmap


# ---------------------------------------------------------------------------
# trips, stops, OD
# ---------------------------------------------------------------------------

def od_map(flows, top_n=40, min_trips=1):
    """Desire lines between zone centres; width ∝ number of trips."""
    data = flows[(flows["trips"] >= min_trips)].head(top_n)
    inter = data[~data["intrazonal"]]
    if data.empty:
        raise ValueError("No flows to draw with the current filters.")

    fmap = base_map(np.r_[data["o_lat"], data["d_lat"]], np.r_[data["o_lon"], data["d_lon"]])
    top = float(data["trips"].max())

    for row in inter.itertuples(index=False):
        weight = 1.5 + 9 * row.trips / top
        folium.PolyLine(
            [[row.o_lat, row.o_lon], [row.d_lat, row.d_lon]],
            weight=weight, color=blend(row.trips, 0, top, SEQUENTIAL_STOPS), opacity=0.6,
            tooltip=f"{row.origin} → {row.destination}: {int(row.trips):,} trips",
        ).add_to(fmap)

    activity = pd.concat([
        data.groupby(["origin", "o_lat", "o_lon"])["trips"].sum().reset_index()
            .rename(columns={"origin": "zone", "o_lat": "lat", "o_lon": "lon"}),
        data.groupby(["destination", "d_lat", "d_lon"])["trips"].sum().reset_index()
            .rename(columns={"destination": "zone", "d_lat": "lat", "d_lon": "lon"}),
    ]).groupby(["zone", "lat", "lon"])["trips"].sum().reset_index()

    for row in activity.itertuples(index=False):
        folium.CircleMarker(
            [row.lat, row.lon], radius=4 + 10 * np.sqrt(row.trips / activity["trips"].max()),
            color="#333333", weight=1, fill=True, fill_color="#ffffff", fill_opacity=0.9,
            tooltip=f"{row.zone}: {int(row.trips):,} trip ends/starts",
        ).add_to(fmap)
    return fmap


def trips_map(traj, trips, max_trips=150, max_points=80):
    """Sampled trip paths coloured by average speed (red = slow)."""
    sample = trips.sample(min(max_trips, len(trips)), random_state=42) if len(trips) else trips
    if sample.empty:
        raise ValueError("There are no trips to draw.")

    points = traj[traj["trip_id"].isin(sample["trip_id"])]
    fmap = base_map(points["lat"], points["lon"])
    vmax = float(sample["avg_speed_kmh"].quantile(0.98))
    _colormap(SPEED_STOPS, 0, vmax, "Trip average speed (km/h)").add_to(fmap)

    speeds = sample.set_index("trip_id")["avg_speed_kmh"]
    for trip_id, group in points.groupby("trip_id"):
        step = max(len(group) // max_points, 1)
        path = group[["lat", "lon"]].iloc[::step].to_numpy().tolist()
        speed = float(speeds.loc[trip_id])
        folium.PolyLine(
            path, weight=3, opacity=0.75, color=blend(speed, 0, vmax, SPEED_STOPS),
            tooltip=f"Trip {int(trip_id)}: {speed:.0f} km/h avg",
        ).add_to(fmap)
    return fmap


def stops_map(stops, max_points=1500):
    data = stops.head(max_points) if len(stops) <= max_points else stops.sample(
        max_points, random_state=42)
    if data.empty:
        raise ValueError("There are no stops to draw.")

    fmap = base_map(data["lat"], data["lon"])
    cluster = MarkerCluster().add_to(fmap)
    for row in data.itertuples(index=False):
        folium.CircleMarker(
            [row.lat, row.lon], radius=5, color="#444444", weight=1, fill=True,
            fill_color="#e15759", fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>{html.escape(str(row.vehicle))}</b><br>"
                f"{row.start_ts:%Y-%m-%d %H:%M} → {row.end_ts:%H:%M}<br>"
                f"Duration: {row.duration_min:.0f} min", max_width=260),
        ).add_to(cluster)
    return fmap
