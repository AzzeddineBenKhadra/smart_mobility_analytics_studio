"""Geometry helpers: grid zones, GeoJSON point-in-polygon, no GIS libraries."""

import json

import numpy as np
import pandas as pd

METERS_PER_DEGREE = 111_320.0


# ---------------------------------------------------------------------------
# square grid zones
# ---------------------------------------------------------------------------

def grid_steps(cell_m, ref_lat):
    """(lat step, lon step) in degrees for a square cell of ``cell_m`` metres."""
    dlat = cell_m / METERS_PER_DEGREE
    dlon = cell_m / (METERS_PER_DEGREE * max(np.cos(np.radians(ref_lat)), 0.01))
    return dlat, dlon


def assign_grid(lat, lon, cell_m, ref_lat=None):
    """Integer grid cell indices (iy, ix) for coordinates."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ref = float(np.nanmedian(lat)) if ref_lat is None else ref_lat
    dlat, dlon = grid_steps(cell_m, ref)
    iy = np.floor(lat / dlat)
    ix = np.floor(lon / dlon)
    return iy, ix, (dlat, dlon)


def cell_bounds(iy, ix, steps):
    dlat, dlon = steps
    return iy * dlat, (iy + 1) * dlat, ix * dlon, (ix + 1) * dlon


def cell_label(iy, ix):
    return f"{int(iy)}_{int(ix)}"


def aggregate_grid(df, lat_col, lon_col, cell_m, value_col=None, agg="mean"):
    """Aggregate points into grid cells.

    Returns a DataFrame: cell, count, value, lat_min/max, lon_min/max,
    center_lat, center_lon.
    """
    data = df[[lat_col, lon_col] + ([value_col] if value_col else [])].copy()
    data[lat_col] = pd.to_numeric(data[lat_col], errors="coerce")
    data[lon_col] = pd.to_numeric(data[lon_col], errors="coerce")
    data = data.dropna(subset=[lat_col, lon_col])
    data = data[data[lat_col].between(-90, 90) & data[lon_col].between(-180, 180)]
    if data.empty:
        return pd.DataFrame()

    iy, ix, steps = assign_grid(data[lat_col], data[lon_col], cell_m)
    data["iy"], data["ix"] = iy, ix

    if value_col and agg != "count":
        data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
        grouped = data.groupby(["iy", "ix"])[value_col].agg(
            value=agg, count="count").reset_index()
        sizes = data.groupby(["iy", "ix"]).size().rename("n").reset_index()
        grouped = grouped.merge(sizes, on=["iy", "ix"])
        grouped["count"] = grouped["n"]
        grouped = grouped.drop(columns="n")
    else:
        grouped = data.groupby(["iy", "ix"]).size().rename("count").reset_index()
        grouped["value"] = grouped["count"]

    lat_min, lat_max, lon_min, lon_max = cell_bounds(grouped["iy"], grouped["ix"], steps)
    grouped["lat_min"], grouped["lat_max"] = lat_min, lat_max
    grouped["lon_min"], grouped["lon_max"] = lon_min, lon_max
    grouped["center_lat"] = (lat_min + lat_max) / 2
    grouped["center_lon"] = (lon_min + lon_max) / 2
    grouped["cell"] = [cell_label(a, b) for a, b in zip(grouped["iy"], grouped["ix"])]
    return grouped.drop(columns=["iy", "ix"])


# ---------------------------------------------------------------------------
# GeoJSON polygons
# ---------------------------------------------------------------------------

def parse_geojson(data):
    """Accept bytes / str / dict and return a FeatureCollection dict."""
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8-sig")
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise ValueError("The file is not a valid GeoJSON object.")

    kind = data.get("type")
    if kind == "FeatureCollection":
        features = data.get("features", [])
    elif kind == "Feature":
        features = [data]
    elif kind in ("Polygon", "MultiPolygon"):
        features = [{"type": "Feature", "properties": {}, "geometry": data}]
    else:
        raise ValueError("GeoJSON must be a FeatureCollection, Feature or Polygon.")

    features = [
        f for f in features
        if (f.get("geometry") or {}).get("type") in ("Polygon", "MultiPolygon")
    ]
    if not features:
        raise ValueError("The GeoJSON contains no polygon features.")
    return {"type": "FeatureCollection", "features": features}


def property_names(geojson):
    names = []
    for feature in geojson["features"]:
        for key in (feature.get("properties") or {}):
            if key not in names:
                names.append(key)
    return names


def _ring_contains(px, py, ring):
    """Vectorised ray casting. ring is an (n, 2) array of (lon, lat)."""
    x, y = ring[:, 0], ring[:, 1]
    inside = np.zeros(len(px), dtype=bool)
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi, xj, yj = x[i], y[i], x[j], y[j]
        crosses = (yi > py) != (yj > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (xj - xi) * (py - yi) / (yj - yi) + xi
        inside ^= crosses & (px < x_at)
        j = i
    return inside


def _polygon_contains(px, py, polygon):
    """polygon = [exterior, hole1, hole2, ...] in GeoJSON coordinate order."""
    exterior = np.asarray(polygon[0], dtype=float)[:, :2]
    inside = _ring_contains(px, py, exterior)
    for hole in polygon[1:]:
        inside &= ~_ring_contains(px, py, np.asarray(hole, dtype=float)[:, :2])
    return inside


def assign_points_to_polygons(lat, lon, geojson):
    """Index of the first containing feature for every point, -1 if none."""
    px = np.asarray(lon, dtype=float)
    py = np.asarray(lat, dtype=float)
    result = np.full(len(px), -1, dtype=int)
    valid = np.isfinite(px) & np.isfinite(py)

    for index, feature in enumerate(geojson["features"]):
        geometry = feature["geometry"]
        polygons = ([geometry["coordinates"]] if geometry["type"] == "Polygon"
                    else geometry["coordinates"])

        todo = valid & (result == -1)
        if not todo.any():
            break

        inside = np.zeros(len(px), dtype=bool)
        for polygon in polygons:
            ring = np.asarray(polygon[0], dtype=float)[:, :2]
            xmin, ymin = ring.min(axis=0)
            xmax, ymax = ring.max(axis=0)
            box = todo & (px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)
            if box.any():
                sub = np.where(box)[0]
                inside[sub] |= _polygon_contains(px[sub], py[sub], polygon)

        result[todo & inside] = index

    return result


def aggregate_by_polygon(df, lat_col, lon_col, geojson, name_property=None,
                         value_col=None, agg="mean"):
    """Aggregate point data into polygon features.

    Returns a DataFrame: feature (index), name, count, value. Features with no
    points are included with count 0 so they can be drawn grey.
    """
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy()
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy()
    assigned = assign_points_to_polygons(lat, lon, geojson)

    names = []
    for i, feature in enumerate(geojson["features"]):
        props = feature.get("properties") or {}
        names.append(str(props.get(name_property, f"Zone {i + 1}")) if name_property
                     else f"Zone {i + 1}")

    work = pd.DataFrame({"feature": assigned})
    if value_col and agg != "count":
        work["v"] = pd.to_numeric(df[value_col], errors="coerce").to_numpy()
    work = work[work["feature"] >= 0]

    base = pd.DataFrame({"feature": range(len(names)), "name": names})
    if work.empty:
        base["count"], base["value"] = 0, np.nan
        return base

    counts = work.groupby("feature").size().rename("count")
    if value_col and agg != "count":
        values = work.groupby("feature")["v"].agg(agg).rename("value")
    else:
        values = counts.rename("value")

    out = base.merge(counts, left_on="feature", right_index=True, how="left")
    out = out.merge(values, left_on="feature", right_index=True, how="left")
    out["count"] = out["count"].fillna(0).astype(int)
    return out


# ---------------------------------------------------------------------------
# colours (pure, so they can be tested)
# ---------------------------------------------------------------------------

def blend(value, vmin, vmax, stops=("#2c7bb6", "#ffffbf", "#d7191c")):
    """Colour for ``value`` on a multi-stop scale; grey for NaN."""
    if value is None or not np.isfinite(value):
        return "#999999"
    if not vmax > vmin:
        t = 0.5
    else:
        t = float(np.clip((value - vmin) / (vmax - vmin), 0, 1))

    rgb = [tuple(int(s[i:i + 2], 16) for i in (1, 3, 5)) for s in stops]
    pos = t * (len(rgb) - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, len(rgb) - 1)
    frac = pos - lo
    mixed = [round(rgb[lo][k] + (rgb[hi][k] - rgb[lo][k]) * frac) for k in range(3)]
    return "#{:02x}{:02x}{:02x}".format(*mixed)
