"""Synthetic sample datasets (clearly artificial — for demos and tests).

* ``generate_sample_traffic``      – 8 road segments, 15-minute speed/volume
                                      for 4 weeks, with rush hours, weekends,
                                      incidents and a few deliberate data
                                      problems so the cleaning tools have
                                      something to find.
* ``generate_sample_trajectories`` – GPS traces of 25 vehicles over one week
                                      (trips, stops, GPS glitches).

Locations are approximate positions around Dubai chosen for illustration;
the numbers are simulated and do not describe real traffic.
"""

import numpy as np
import pandas as pd

from .utils import haversine_m

SEGMENTS = [
    # id, name, lat, lon, free-flow km/h, capacity (vehicles / 15 min), load
    ("SEG01", "Sheikh Zayed Road", 25.2048, 55.2708, 100, 1500, 1.00),
    ("SEG02", "Al Khail Road", 25.1500, 55.3100, 95, 1350, 0.95),
    ("SEG03", "Emirates Road", 25.0900, 55.3700, 110, 1600, 0.90),
    ("SEG04", "Al Ittihad Road", 25.2850, 55.3500, 90, 1300, 1.05),
    ("SEG05", "Jumeirah Beach Road", 25.2000, 55.2400, 70, 900, 0.85),
    ("SEG06", "Al Qudra Road", 24.9700, 55.3900, 110, 800, 0.55),
    ("SEG07", "Dubai-Al Ain Road", 25.1000, 55.4200, 100, 1100, 0.80),
    ("SEG08", "Garhoud Bridge", 25.2370, 55.3380, 70, 1000, 1.10),
]

HUBS = {
    "Dubai Marina": (25.0805, 55.1403),
    "Downtown": (25.1972, 55.2744),
    "Deira": (25.2697, 55.3095),
    "Airport": (25.2532, 55.3657),
    "Jebel Ali": (25.0050, 55.0600),
    "Investment Park": (24.9800, 55.1700),
    "Al Barsha": (25.1120, 55.2000),
    "Mirdif": (25.2200, 55.4200),
    "Silicon Oasis": (25.1190, 55.3800),
    "Academic City": (25.1260, 55.4140),
    "Al Quoz": (25.1400, 55.2300),
    "Al Qusais": (25.2900, 55.3800),
}


def _bump(h, centre, width):
    return np.exp(-(((h - centre) / width) ** 2))


def _demand(hour, dow):
    """Relative demand 0..~1.1 for fractional hour and day of week (0=Mon)."""
    if dow >= 5:  # weekend (Sat/Sun)
        base = 0.06 + 0.55 * _bump(hour, 13.5, 3.0) + 0.65 * _bump(hour, 20.0, 2.2)
    else:
        pm = 1.0
        if dow == 4:  # Friday: lighter evening peak
            pm = 0.8
        base = (0.08 + 0.95 * _bump(hour, 7.6, 1.2) + pm * _bump(hour, 17.7, 1.6)
                + 0.45 * _bump(hour, 12.8, 2.5))
        if dow == 4:  # Friday prayer dip
            base = base * (1 - 0.35 * _bump(hour, 12.8, 0.8))
    return base


def generate_sample_traffic(seed=7, days=28, start="2026-02-02", dirty=True):
    """Return a DataFrame of 15-minute traffic observations."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=days * 96, freq="15min")
    hours = (times.hour + times.minute / 60).to_numpy()
    dows = times.dayofweek.to_numpy()
    demand = np.array([_demand(h, d) for h, d in zip(hours, dows)])

    frames = []
    for seg_id, name, lat, lon, free_flow, capacity, load in SEGMENTS:
        util = np.clip(demand * load * (1 + 0.05 * rng.standard_normal(len(times))),
                       0.02, 1.2)
        volume = capacity * util * (1 + 0.05 * rng.standard_normal(len(times)))
        speed = free_flow / (1 + 1.2 * util ** 6)
        speed = speed * (1 + 0.03 * rng.standard_normal(len(times)))

        frames.append(pd.DataFrame({
            "timestamp": times,
            "segment_id": seg_id,
            "road_name": name,
            "latitude": lat,
            "longitude": lon,
            "speed_kmh": speed,
            "vehicle_count": np.clip(volume, 0, None),
        }))

    data = pd.concat(frames, ignore_index=True)

    # Incidents: speed collapses for 1–2.5 hours on one segment.
    for _ in range(max(4, days // 7)):
        seg = rng.integers(0, len(SEGMENTS))
        day = rng.integers(0, days)
        start_slot = rng.integers(6 * 4, 22 * 4)
        length = rng.integers(4, 11)
        first = day * 96 + start_slot
        rows = seg * len(times) + np.arange(first, min(first + length, len(times)))
        data.loc[rows, "speed_kmh"] *= 0.35
        data.loc[rows, "vehicle_count"] *= 0.6

    data["vehicle_count"] = data["vehicle_count"].round().astype(int)
    data["speed_kmh"] = data["speed_kmh"].round(1)
    data = data.sort_values(["timestamp", "segment_id"]).reset_index(drop=True)
    data["timestamp"] = data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    if dirty:
        n = len(data)
        pick = lambda k: rng.choice(n, size=k, replace=False)  # noqa: E731

        data.loc[pick(max(3, n // 400)), "speed_kmh"] = np.nan
        data.loc[pick(max(2, n // 800)), "vehicle_count"] = np.nan
        data.loc[pick(3), "latitude"] = 999.0           # invalid coordinate
        data.loc[pick(2), "speed_kmh"] = -5.0           # impossible speed
        data.loc[pick(2), "speed_kmh"] = 420.0          # sensor glitch
        dup = data.iloc[pick(5)]
        data = pd.concat([data, dup], ignore_index=True)

    return data


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------

def _drive(rng, a, b, t0, step_s=30):
    """Simulate one drive from a to b. Returns (times, lat, lon, speed_kmh)."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    direction = b - a
    perp = np.array([-direction[1], direction[0]])
    mid = (a + b) / 2 + perp * rng.uniform(-0.15, 0.15)
    poly = np.vstack([a, mid, b])

    seg_len = haversine_m(poly[:-1, 0], poly[:-1, 1], poly[1:, 0], poly[1:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]

    cruise = rng.uniform(45, 85) / 3.6
    pos, t = 0.0, 0
    times, positions, speeds = [t0], [0.0], [0.0]
    hold = 0

    while pos < total and len(times) < 600:
        if hold > 0:
            v = 0.0
            hold -= 1
        else:
            ramp = min(1.0, pos / 600, (total - pos) / 600)
            v = cruise * max(ramp, 0.15) * (1 + 0.12 * rng.standard_normal())
            v = max(v, 0.5)
            if rng.random() < 0.012:
                hold = int(rng.integers(2, 5))  # traffic light / queue
        pos = min(total, pos + v * step_s)
        t += step_s
        times.append(t0 + pd.Timedelta(seconds=t))
        positions.append(pos)
        speeds.append(v * 3.6)

    positions = np.array(positions)
    lat = np.interp(positions, cum, poly[:, 0])
    lon = np.interp(positions, cum, poly[:, 1])
    return times, lat, lon, np.array(speeds)


def _park(rng, loc, t0, minutes, step_s=120):
    n = max(int(minutes * 60 // step_s), 1)
    times = [t0 + pd.Timedelta(seconds=step_s * i) for i in range(1, n + 1)]
    return times, np.full(n, loc[0]), np.full(n, loc[1]), np.zeros(n)


def generate_sample_trajectories(seed=11, vehicles=25, start="2026-02-02", days=7):
    """GPS traces with trips, stops and a few glitch points."""
    rng = np.random.default_rng(seed)
    hub_names = list(HUBS)
    day0 = pd.Timestamp(start)

    records = []

    def emit(vid, times, lat, lon, speed):
        for tt, la, lo, sp in zip(times, lat, lon, speed):
            records.append((vid, tt, la, lo, sp))

    for v in range(vehicles):
        vid = f"VEH{v + 1:03d}"
        home, work = rng.choice(hub_names, size=2, replace=False)
        home_xy, work_xy = HUBS[home], HUBS[work]

        for d in range(days):
            day = day0 + pd.Timedelta(days=d)
            weekday = day.dayofweek < 5
            location = home_xy
            clock = day + pd.Timedelta(hours=float(rng.uniform(6.8, 8.5)) if weekday
                                       else float(rng.uniform(9, 14)))

            def go(dest_xy, depart, current):
                times, la, lo, sp = _drive(rng, current, dest_xy, depart)
                emit(vid, times, la, lo, sp)
                return times[-1]

            if weekday:
                if rng.random() < 0.9:  # some days off
                    arrive = go(work_xy, clock, location)
                    location = work_xy

                    if rng.random() < 0.3:  # lunch trip
                        t, la, lo, sp = _park(rng, location, arrive, rng.uniform(180, 240))
                        emit(vid, t, la, lo, sp)
                        lunch = HUBS[rng.choice(hub_names)]
                        arrive = go(lunch, t[-1], location)
                        t, la, lo, sp = _park(rng, lunch, arrive, rng.uniform(30, 60))
                        emit(vid, t, la, lo, sp)
                        arrive = go(work_xy, t[-1], lunch)

                    stay = rng.uniform(150, 240)
                    t, la, lo, sp = _park(rng, location, arrive, stay)
                    emit(vid, t, la, lo, sp)
                    depart = max(t[-1], day + pd.Timedelta(hours=float(rng.uniform(16.5, 18.3))))
                    arrive = go(home_xy, depart, location)
                    location = home_xy
                    t, la, lo, sp = _park(rng, location, arrive, rng.uniform(20, 90))
                    emit(vid, t, la, lo, sp)

                    if rng.random() < 0.25:  # evening errand
                        mall = HUBS[rng.choice(hub_names)]
                        arrive = go(mall, t[-1] + pd.Timedelta(minutes=30), location)
                        t, la, lo, sp = _park(rng, mall, arrive, rng.uniform(40, 120))
                        emit(vid, t, la, lo, sp)
                        arrive = go(home_xy, t[-1], mall)
                        t, la, lo, sp = _park(rng, home_xy, arrive, 30)
                        emit(vid, t, la, lo, sp)
            else:
                for _ in range(int(rng.integers(0, 3))):
                    dest = HUBS[rng.choice(hub_names)]
                    arrive = go(dest, clock, location)
                    t, la, lo, sp = _park(rng, dest, arrive, rng.uniform(60, 180))
                    emit(vid, t, la, lo, sp)
                    arrive = go(home_xy, t[-1], dest)
                    t, la, lo, sp = _park(rng, home_xy, arrive, rng.uniform(60, 120))
                    emit(vid, t, la, lo, sp)
                    clock = t[-1] + pd.Timedelta(minutes=30)

    data = pd.DataFrame(records, columns=["vehicle_id", "timestamp", "latitude",
                                          "longitude", "speed_kmh"])
    data = data.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)

    # GPS noise (~5 m) and a few outlier jumps.
    noise = 5 / 111_320
    data["latitude"] = data["latitude"] + rng.normal(0, noise, len(data))
    data["longitude"] = data["longitude"] + rng.normal(0, noise, len(data))
    data["speed_kmh"] = np.clip(data["speed_kmh"] + rng.normal(0, 1.5, len(data)), 0, None)

    glitch = rng.choice(len(data), size=max(3, len(data) // 800), replace=False)
    data.loc[glitch, "latitude"] += rng.choice([-0.2, 0.2], size=len(glitch))

    data["speed_kmh"] = data["speed_kmh"].round(1)
    data["latitude"] = data["latitude"].round(6)
    data["longitude"] = data["longitude"].round(6)
    data["timestamp"] = data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return data
