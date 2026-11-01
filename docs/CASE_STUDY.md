# Where and when is Dubai congested?

**A mobility case study using Smart Mobility Analytics Studio**

> **Read this first — what this case study is and is not.**
> Part 1 is a **desk study**: it summarises *published* traffic reports and a TomTom-based news summary. I did not measure these numbers myself.
> Part 2 is the **data analysis you run yourself** in the app on real Dubai data (Dubai Police incident records from Dubai Pulse). Its result boxes are left empty on purpose: fill them only with what *your* screenshots show.
> The app's built-in samples are **simulated** and must never be presented as Dubai measurements.

---

## 1. Question

1. **When** during the day and week is Dubai's road network most congested?
2. **Where** are the recurring hotspots?
3. How reliable are conclusions from open data, and what would be needed to go further?

## 2. Desk study: what published sources say

### 2.1 When — timing

| Finding | Value | Source |
|---|---|---|
| Typical weekday morning window | 6–10 am, worst about 7–9 am | Blacklane guide |
| Typical weekday evening window | 4–8 pm, heaviest about 5–7 pm | Blacklane guide |
| Average time to drive 10 km, 2025 | 19.1 min (13.7 min in 2024) | TomTom via Khaleej Times |
| Morning vs evening, 10 km | 18.4 min vs **26.3 min** | TomTom via Khaleej Times |
| Morning vs evening average speed | 32.1 km/h vs **22.6 km/h** | TomTom via Khaleej Times |
| Time lost in traffic, 2025 | 72 h per driver (TomTom); about 45 h (INRIX) | Khaleej Times |
| Worst recorded moment | 5 pm, 11 Nov 2025 (168% congestion) | Khaleej Times |

**Takeaway:** the **evening peak is clearly worse** than the morning peak (about 43% longer per 10 km). Different providers measure differently (TomTom 72 h vs INRIX 45 h), so compare trends, not absolute numbers across providers.

![Morning vs evening](images/fig01_tomtom_morning_vs_evening.png)
*Figure 1 — Drawn from the TomTom figures reported by Khaleej Times (see Sources).*

> Caveat: the Khaleej Times article uses the number 26.3 both for evening minutes per 10 km and for rush-hour average speed in km/h without explanation. I used only the morning/evening pair that is internally consistent (18.4 vs 26.3 minutes; 32.1 vs 22.6 km/h). Check against TomTom's own page before publishing.

### 2.2 Where — recurring hotspots

Gulf News live traffic reports (18 June 2026, morning; 8 September 2026, evening) and the Blacklane guide repeatedly name the same corridors. These are **single-day reports**, not statistics, so treat them as indicative.

| Corridor | Where it is slow | Peak |
|---|---|---|
| **Sheikh Zayed Road (E11)** | Sharjah → Dubai near the Airport/Garhoud; Business Bay to Interchange 3; Marina–Al Barsha–Business Bay toward Defence Roundabout | Morning and evening |
| **Sheikh Rashid Road (D75) / Karama–Bur Dubai** | Karama toward Bur Dubai, near junction with E11 | Morning and evening |
| **Al Khail Road (E44)** | Ras Al Khor ↔ Meydan; Al Quoz; Dubai Hills ↔ Meydan | Morning and evening |
| **Sheikh Mohammed bin Zayed Rd (E311)** | Sharjah → Dubai around Muhaisnah (am); Nad Al Sheba → Muhaisnah (pm) | Morning (inbound), evening (outbound) |
| **Al Safa Street (D71) / Umm Suqeim Street (D63)** | City Walk, Dubai Mall area; Al Sufouh–Al Barsha | Includes school drop-off/pick-up |
| **Business Bay Crossing** | Toward Deira (am), toward Bur Dubai (pm) | Peaks |
| **Emirates Road (E611)** | Mostly smooth; pockets near Tilal / Al Badia interchange | Evening |

**Pattern:** congestion concentrates on the **radial commuter corridors between Sharjah/Ajman and central Dubai** (E11, E311, D75) and the **inner east–west spine** (E11 and Al Khail). Emirates Road is relatively free-flowing in these reports.

### 2.3 Context: supply keeps growing

Per the same report: active registered vehicles reached 4.65 million (Q3 2025) and population passed 4 million in 2025, while average rush-hour speeds fell ~1 km/h year on year and highway speeds ~2.1 km/h. New capacity (e.g. World Trade Centre roundabout bridges, Al Hessa Street lanes, Mall of the Emirates bridge) was opened in 2025.

## 3. Data analysis you run in the app

### 3.1 Data

**Dataset:** *Traffic Incidents* from Dubai Police, published on **Dubai Pulse** (https://www.dubaipulse.gov.ae/data/dp-traffic/dp_traffic_incidents-open).

| Fields | Notes |
|---|---|
| `acci_id`, `acci_time`, `acci_name`, `acci_x`, `acci_y` | Time, Arabic category, coordinates. CSV, about 7 MB. |

- **Access:** not a direct download. Click **Request Permission**, accept the terms, wait for approval (the page says up to 14 days). The page states no licence ("not specified") — say so in your write-up.
- **What it measures:** *incident reports*, **not speed or flow**. It shows where and when incidents are *reported*, a proxy for stress on the network, not congestion itself. Reporting itself varies by area and time.
- **Note:** the page labels the coordinate columns "latitude" and "longitude" but names them `acci_x`/`acci_y`. Check which is which in the app's column mapping (Dubai latitude ≈ 25.0–25.4, longitude ≈ 55.0–55.5).

### 3.2 Steps in the app

1. `streamlit run app.py`, upload the CSV.
2. **Column mapping** (sidebar): timestamp = `acci_time`; latitude/longitude = the two coordinate columns (swap if needed); set weekend to **Sat–Sun**.
3. **Data Quality → Data Cleaning**: run smart cleaning (removes duplicates and impossible coordinates). Record what it removed.
4. **Data Explorer**: histogram of `acci_time` hour and weekday. → *Screenshot A*.
5. **Maps → Heatmap** and **Grid zones** (try 1 km cells): where incidents cluster. → *Screenshots B, C*.
6. Compare with Part 2.2: do the hotspots sit on E11, E44, E311, D75?
7. **Report & Export → Build report**, download the HTML.

### 3.3 Your results (fill in)

| Screenshot | Save as | What to say about it |
|---|---|---|
| A. Incidents by hour/weekday | `docs/images/figA_incidents_by_hour.png` | _Peak hours? Same as 7–9 am / 5–7 pm?_ |
| B. Incident heatmap | `docs/images/figB_heatmap.png` | _Which areas glow?_ |
| C. Grid-zone map | `docs/images/figC_grid.png` | _Top 3 cells and their areas_ |

Embed with:
```markdown
![Incidents by hour](images/figA_incidents_by_hour.png)
```

Write 3–5 sentences: do incident peaks match the published peaks? Do hotspots match the corridors in 2.2? Where do they disagree and why (reporting bias, exposure — busy roads have more incidents simply because they have more cars)?

## 4. Limitations (keep these — they make the study credible)

- Part 2.1/2.2 come from news reports and a traffic-index summary, not raw measurements. Live reports describe single days.
- Incident counts are not congestion and are biased towards busier roads.
- No open Dubai speed/flow dataset was used. A stronger study would use road-sensor or probe-speed data (e.g. from RTA if available) and run Mobility Insights (congestion index, planning-time index, incident detection) and Forecasting.
- Data licence is unspecified.

## 5. Next steps

- Obtain speed/flow or probe data and rerun the app's Mobility Insights and Forecasting pages.
- Add weather and event calendars (e.g. Ramadan, school terms) as forecast features.
- Normalise incidents by traffic volume to separate "dangerous" from "just busy".

## Sources

- [Dubai motorists lose nearly 4 days to traffic in 2025 as congestion rises — Khaleej Times](https://www.khaleejtimes.com/uae/transport/dubai-commute-time-increase-2025-more-cars)
- [Dubai morning traffic update: Where roads are busiest during rush hour — Gulf News, 18 Jun 2026](https://gulfnews.com/uae/transport/heading-to-work-dubai-morning-traffic-is-slowing-these-major-roads-1.500578222)
- [Dubai–Sharjah traffic: Where motorists face the worst evening rush-hour delays — Gulf News, 8 Sep 2026](https://gulfnews.com/uae/transport/dubai-sharjah-traffic-where-motorists-face-the-worst-evening-rush-hour-delays-1.500667440)
- [When is Rush Hour in Dubai? — Blacklane](https://www.blacklane.com/en/blog/travel/rush-hour-dubai/)
- [TomTom Traffic Index — Dubai](https://www.tomtom.com/traffic-index/dubai-traffic/)
- [Traffic Incidents — Dubai Pulse / Dubai Police](https://www.dubaipulse.gov.ae/data/dp-traffic/dp_traffic_incidents-open)
