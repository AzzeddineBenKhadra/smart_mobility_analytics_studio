# Dataset ideas (check each licence and find the current download page yourself)

- Road sensors: Caltrans PeMS; METR-LA / PEMS-BAY (common forecasting benchmarks); city open-data traffic counters.
- Trips / OD: NYC TLC trip records; city bike-share trip data.
- GPS traces: T-Drive (Beijing taxis), GeoLife, Porto taxi.
- Transit schedules: any agency's GTFS feed (future extension: stop-level headways).

Tips: upload Parquet when possible; map columns in the sidebar if auto-detection misses; add unusual column names to `config/field_aliases.json`.
