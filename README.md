# 🚁 Drone-Mounted LiDAR Agro-Topographic Mapping System

**An end-to-end pipeline that turns raw drone LiDAR point clouds into actionable precision-agriculture intelligence** — terrain models, slope & drainage analysis, and variable-rate irrigation zoning.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

---

## 📌 Overview

Precision agriculture increasingly relies on drone-mounted LiDAR to survey fields at centimeter-level accuracy — far beyond what satellite imagery or manual surveying can provide. This project implements the **full processing chain** a real agro-topographic mapping system needs, from raw point cloud to farm-ready decision layers:

```
Raw LiDAR Point Cloud
        │
        ▼
 ① Noise Filtering (statistical outlier removal)
        │
        ▼
 ② Ground / Canopy Classification (progressive morphological filter)
        │
        ▼
 ③ Surface Generation → DTM (bare earth) · DSM (surface) · CHM (canopy height)
        │
        ▼
 ④ Topographic Analysis → Slope · Aspect · Curvature · Flow Accumulation · TWI
        │
        ▼
 ⑤ Agronomic Interpretation → Erosion Risk · Drainage Risk · Irrigation Zones
        │
        ▼
   Field Report (JSON) + Map Outputs (PNG)
```

> **Note on data:** No physical drone was flown for this project. `src/lidar_simulator.py` generates a physically-plausible synthetic point cloud (rolling terrain, a carved drainage channel, and crop-row canopy structure) sampled the way a real LiDAR sensor captures a survey flight. Every downstream module — filtering, classification, DTM generation, topographic and agronomic analysis — operates on that point cloud exactly as it would on a real `.las`/`.laz` flight export, so the pipeline is a drop-in fit for real flight data.

---

## ✨ Features

| Module | What it does |
|---|---|
| **Flight Simulation** | Generates a realistic multi-swath drone survey point cloud with ground + vegetation returns, sensor noise, and intensity values |
| **Point Cloud Cleaning** | Statistical outlier removal (k-NN distance filtering) |
| **Ground Classification** | Simplified Progressive Morphological Filter (PMF), the same family of algorithm used in PDAL / lasground |
| **Surface Modeling** | Interpolates DTM, DSM, and CHM raster surfaces from classified points |
| **Topographic Analysis** | Slope, aspect, curvature, D8 flow accumulation, Topographic Wetness Index (TWI) |
| **Agronomic Metrics** | Erosion risk classing, drainage/waterlogging risk, variable-rate irrigation zoning, canopy height / vigor statistics |
| **Visualization** | 8 map/figure outputs including a 3D terrain reconstruction |

---

## 🗺️ Sample Outputs

| Digital Terrain Model | Predicted Drainage Network |
|---|---|
| ![DTM](outputs/02_dtm.png) | ![Drainage](outputs/06_drainage_network.png) |

| Slope Map | Variable-Rate Irrigation Zones |
|---|---|
| ![Slope](outputs/04_slope.png) | ![Irrigation Zones](outputs/07_irrigation_zones.png) |

| Canopy Height Model | 3D Terrain Reconstruction |
|---|---|
| ![CHM](outputs/03_chm.png) | ![3D Terrain](outputs/08_3d_terrain.png) |

Full set of 8 generated maps lives in [`/outputs`](outputs), and the raw classified point cloud + DTM + JSON field report are in [`/data/sample_output`](data/sample_output).

### Example field summary output

```json
{
  "field_area_ha": 3.02,
  "elevation_range_m": [148.27, 155.73],
  "mean_slope_deg": 2.77,
  "max_slope_deg": 10.42,
  "erosion_risk_pct": { "low": 73.8, "moderate": 21.7, "high": 4.4, "severe": 0.0 },
  "drainage_risk_pct": { "well_drained": 33.0, "moderate": 33.0, "poorly_drained": 24.0, "waterlogging_risk": 10.0 },
  "crop_canopy": { "mean_height_m": 0.20, "max_height_m": 1.43 }
}
```

---

## 🧠 Methodology Notes

- **Ground classification** uses an increasing-window morphological erosion filter with a slope-adaptive elevation threshold — a simplified version of the Progressive Morphological Filter (Zhang et al.), the standard approach in commercial LiDAR ground-classification software.
- **Flow accumulation** implements single-flow-direction D8 routing (each cell drains to its steepest downhill neighbor), the same algorithm underlying ArcGIS Hydrology and QGIS's `r.watershed` tools.
- **TWI (Topographic Wetness Index)**, `ln(contributing area / tan(slope))`, is a well-established proxy for soil moisture and waterlogging risk used widely in digital soil mapping and precision irrigation research.
- **Irrigation zoning** combines normalized slope and TWI into a water-need index, then bins the field into contiguous management zones — mirroring how commercial variable-rate irrigation (VRI) prescriptions are built in practice.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- `numpy` / `scipy` — array processing, spatial KD-trees, interpolation, morphological filtering
- `pandas` — tabular point cloud I/O
- `matplotlib` — 2D/3D geospatial visualization

No proprietary GIS software or paid SDKs required — everything runs from open-source scientific Python.

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/drone-lidar-agro-mapping.git
cd drone-lidar-agro-mapping

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python scripts/run_pipeline.py
```

This will simulate a survey flight, run the full processing chain, and write:
- Figures → `outputs/*.png`
- Field summary report → `data/sample_output/field_summary_report.json`
- Classified point cloud (CSV) and DTM raster (CSV) → `data/sample_output/`

### Using your own flight data

Swap `lidar_simulator.simulate_flight()` for a loader that reads your `.las`/`.laz` file (e.g. via [`laspy`](https://laspy.readthedocs.io/)) into the same `{x, y, z, intensity, classification}` dict format — every downstream module works unchanged.

---

## 📂 Project Structure

```
drone-lidar-agro-mapping/
├── src/
│   ├── lidar_simulator.py        # synthetic drone LiDAR flight generator
│   ├── point_cloud_processor.py  # outlier removal + ground classification
│   ├── dtm_generator.py          # DTM / DSM / CHM raster generation
│   ├── topographic_analysis.py   # slope, aspect, curvature, flow, TWI
│   ├── agro_metrics.py           # erosion/drainage risk, irrigation zoning
│   └── visualize.py              # all figure generation
├── scripts/
│   └── run_pipeline.py           # end-to-end pipeline entry point
├── data/sample_output/           # generated point cloud, DTM, and JSON report
├── outputs/                      # generated maps and figures
├── docs/
│   └── technical_report.md       # detailed write-up of methods & results
├── requirements.txt
└── LICENSE
```

---

## 🔭 Future Work

- Real flight data ingestion via `laspy` / `.las`-`.laz` support
- GeoTIFF export with real-world CRS georeferencing (`rasterio`)
- Multi-flight temporal comparison (crop growth over a season)
- Web dashboard for interactive zone editing (Leaflet/Mapbox front-end)
- Integration with soil sensor data for multi-source irrigation prescriptions

---

## 📄 License

Released under the [MIT License](LICENSE).

## 🌐 Portfolio Page

A standalone visual write-up of this project (with embedded output images) lives at [`docs/portfolio.html`](docs/portfolio.html) — open it directly in a browser, or view it live via GitHub Pages once enabled.

## 👤 Author

Built by **[Your Name]** — [GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-profile>)
