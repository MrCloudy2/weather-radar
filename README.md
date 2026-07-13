# Slovenia Weather Radar

A self-hosted, live weather-radar dashboard for Slovenia, built from **raw EUMETNET
OPERA ODIM_H5 polar volumes** (via the open `openradar-24h` S3 bucket) — not from
pre-made consumer maps. Pure Python (NumPy + Pillow, no Py-ART), FastAPI backend,
vanilla-JS + Leaflet frontend.

## What it does

- Ingests 5-minute polar volumes from the Slovenian radars **Lisca** (`silis`) and
  **Pasja Ravan** (`sipas`) plus nearby Croatian polarimetric radars
  **Puntijarka** (`hrpun`) and **Bilogora** (`hrbil`).
- Georeferences gates with the standard 4/3-effective-earth beam model and renders
  onto a shared **Web Mercator** grid, so overlays align pixel-perfectly under
  Leaflet (validated against the official ARSO SIRAD product).
- Multi-panel (1/2/4) synced maps, timeline playback, and a hover inspector.

## Products (13)

Reflectivity (DBZH), Rainfall rate (mm/h, Marshall-Palmer Z-R), Echo tops, VIL,
Total power, Velocity (dealiased), Azimuthal shear (rotation), ρHV particle
uniformity + ZDR (Croatian radars only), plus multi-radar **composites**
(max-reflectivity + nearest-radar velocity), **column-max (CMAX)**, storm-motion
vectors + **nowcast**, rainfall **accumulation**, and **dual-Doppler wind**.

The flagship **composite CMAX reflectivity** is pre-rendered on every new frame and
kept for **24 hours** on disk (`archive/`), giving a full-day loop (the "24h" toggle)
and instant loads.

## Run (Docker)

```bash
docker compose up -d --build       # serves on :8099
```

Behind Nginx at a subpath, set `RADAR_ROOT_PATH=/radar` (already in compose) and
proxy `location /radar/ { proxy_pass http://127.0.0.1:8099/; }`. The container is
capped (`mem_limit`, `cpus`) and renders are concurrency-limited so it can't
overload a shared host.

## Key endpoints

| Endpoint | Purpose |
|----------|---------|
| `/api/sources` | radar sources + per-source products |
| `/api/timeline?node=` | live 90-min frame list |
| `/api/archive` | 24h flagship-reflectivity frame list |
| `/api/tile.png?quantity=&time=&node=|nodes=&cmax=&sweep=` | rendered Web-Mercator tile |
| `/api/value?...&lat=&lon=` | point value (hover inspector) |
| `/api/legend?quantity=` | colour-bar spec |
| `/api/motion`, `/api/windvectors` | storm-motion / dual-Doppler arrows |

## Layout

- `radar/` — backend: `fetcher` (S3), `odim` (h5py reader), `geo`, `colormaps`,
  `process`, `render`, `composite`, `analysis`, `store` (worker + cache + archive),
  `api` (FastAPI).
- `radar/static/` — Leaflet frontend.
- `scripts/` — manual validation/inspection tools.
