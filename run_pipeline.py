"""
run_pipeline.py
-----------------
End-to-end pipeline for the Drone-Mounted LiDAR Agro-Topographic Mapping
System.

    Simulated flight -> point cloud cleaning -> ground classification
    -> DTM/DSM/CHM generation -> topographic analysis -> agro metrics
    -> figures + JSON report

Usage
-----
    python scripts/run_pipeline.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from lidar_simulator import simulate_flight
from point_cloud_processor import process
from dtm_generator import generate_surfaces
from topographic_analysis import analyze
from agro_metrics import field_summary, generate_irrigation_zones
import visualize as viz

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_FIGS = os.path.join(ROOT, "outputs")
OUT_DATA = os.path.join(ROOT, "data", "sample_output")
RESOLUTION = 1.5  # meters per grid cell


def main():
    os.makedirs(OUT_FIGS, exist_ok=True)
    os.makedirs(OUT_DATA, exist_ok=True)

    print("[1/6] Simulating drone LiDAR survey flight...")
    raw_cloud = simulate_flight()
    print(f"      -> {len(raw_cloud['x']):,} raw returns")

    print("[2/6] Cleaning point cloud (outlier removal + ground classification)...")
    cloud = process(raw_cloud)
    n_ground = int((cloud["classification"] == 2).sum())
    print(f"      -> {len(cloud['x']):,} points retained, {n_ground:,} classified as ground")

    print("[3/6] Generating DTM / DSM / CHM raster surfaces...")
    surfaces = generate_surfaces(cloud, resolution=RESOLUTION)
    print(f"      -> raster grid shape {surfaces['dtm'].shape}")

    print("[4/6] Running topographic analysis (slope, aspect, flow, TWI)...")
    topo = analyze(surfaces["dtm"], resolution=RESOLUTION)

    print("[5/6] Computing agronomic metrics and irrigation zones...")
    summary = field_summary(
        surfaces["dtm"], topo["slope"], topo["aspect"], topo["twi"], surfaces["chm"],
        resolution=RESOLUTION,
    )
    zones, water_need_index = generate_irrigation_zones(topo["slope"], topo["twi"])

    with open(os.path.join(OUT_DATA, "field_summary_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    sample_idx = np.random.default_rng(0).choice(
        len(cloud["x"]), size=min(20000, len(cloud["x"])), replace=False
    )
    pd.DataFrame({
        "x": cloud["x"][sample_idx], "y": cloud["y"][sample_idx], "z": cloud["z"][sample_idx],
        "intensity": cloud["intensity"][sample_idx], "classification": cloud["classification"][sample_idx],
    }).to_csv(os.path.join(OUT_DATA, "classified_point_cloud_sample.csv"), index=False)

    np.savetxt(os.path.join(OUT_DATA, "dtm_raster.csv"), surfaces["dtm"], delimiter=",")

    print("[6/6] Rendering figures to /outputs ...")
    viz.plot_point_cloud(cloud, os.path.join(OUT_FIGS, "01_point_cloud.png"))
    viz.plot_dtm(surfaces, os.path.join(OUT_FIGS, "02_dtm.png"))
    viz.plot_chm(surfaces, os.path.join(OUT_FIGS, "03_chm.png"))
    viz.plot_slope(surfaces["x"], surfaces["y"], topo["slope"], os.path.join(OUT_FIGS, "04_slope.png"))
    viz.plot_aspect(surfaces["x"], surfaces["y"], topo["aspect"], os.path.join(OUT_FIGS, "05_aspect.png"))
    viz.plot_drainage(surfaces["x"], surfaces["y"], surfaces["dtm"], topo["flow_accumulation"],
                       os.path.join(OUT_FIGS, "06_drainage_network.png"))
    viz.plot_irrigation_zones(surfaces["x"], surfaces["y"], zones,
                               os.path.join(OUT_FIGS, "07_irrigation_zones.png"))
    viz.plot_3d_terrain(surfaces, os.path.join(OUT_FIGS, "08_3d_terrain.png"))

    print("\nDone. Field summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nFigures saved to: {OUT_FIGS}")
    print(f"Data saved to: {OUT_DATA}")


if __name__ == "__main__":
    main()
