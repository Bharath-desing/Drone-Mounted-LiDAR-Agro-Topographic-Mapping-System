"""
point_cloud_processor.py
-------------------------
Cleans and classifies a raw LiDAR point cloud.

Implements two standard photogrammetry/LiDAR processing steps:

1. Statistical outlier removal (SOR) - removes sensor noise / stray returns
   by rejecting points whose distance to their k nearest neighbors is an
   outlier relative to the local point cloud density.

2. Simplified Progressive Morphological Filter (PMF) ground classification -
   separates bare-earth ("ground") returns from vegetation/canopy returns
   by applying an increasing-window minimum filter over the point cloud
   and testing elevation difference against a slope-based threshold.
   This is a simplified version of the algorithm used in tools like
   PDAL / lasground.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import grey_erosion


def remove_statistical_outliers(cloud, k=8, std_ratio=2.5):
    """Remove points whose mean distance to k nearest neighbors is an outlier."""
    xyz = np.column_stack([cloud["x"], cloud["y"], cloud["z"]])
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k + 1)  # includes self at index 0
    mean_dists = dists[:, 1:].mean(axis=1)

    mu, sigma = mean_dists.mean(), mean_dists.std()
    keep = mean_dists < mu + std_ratio * sigma

    return {key: val[keep] for key, val in cloud.items()}


def classify_ground_pmf(cloud, cell_size=1.0, max_window=12, slope=0.15, initial_dh=0.3):
    """Reclassify points as ground (2) or non-ground (5) using a simplified PMF.

    Bins points into a grid, tracks the running minimum surface, then grows
    the filter window and re-checks elevation difference against a
    slope-adaptive threshold - points that stay close to the eroded minimum
    surface are classified as ground.
    """
    x, y, z = cloud["x"], cloud["y"], cloud["z"]
    nx = int(np.ceil((x.max() - x.min()) / cell_size)) + 1
    ny = int(np.ceil((y.max() - y.min()) / cell_size)) + 1

    col = ((x - x.min()) / cell_size).astype(int)
    row = ((y - y.min()) / cell_size).astype(int)

    min_surface = np.full((ny, nx), np.nan)
    for r, c, zi in zip(row, col, z):
        if np.isnan(min_surface[r, c]) or zi < min_surface[r, c]:
            min_surface[r, c] = zi

    fill_value = np.nanmax(min_surface)
    filled = np.where(np.isnan(min_surface), fill_value, min_surface)

    window = 3
    surface = filled.copy()
    while window <= max_window:
        eroded = grey_erosion(surface, size=(window, window))
        dh_threshold = slope * window * cell_size + initial_dh
        surface = np.where((surface - eroded) > dh_threshold, eroded, surface)
        window += 2

    point_ground_elev = surface[row, col]
    classification = np.where(
        (z - point_ground_elev) < initial_dh, 2, 5
    ).astype(np.uint8)

    out = dict(cloud)
    out["classification"] = classification
    return out


def process(raw_cloud):
    """Full cleaning pipeline: outlier removal -> ground classification."""
    cleaned = remove_statistical_outliers(raw_cloud)
    classified = classify_ground_pmf(cleaned)
    return classified


if __name__ == "__main__":
    from lidar_simulator import simulate_flight

    raw = simulate_flight()
    result = process(raw)
    print(f"Input points: {len(raw['x']):,} -> after cleaning: {len(result['x']):,}")
    print(f"Ground: {(result['classification'] == 2).sum():,}  "
          f"Vegetation: {(result['classification'] == 5).sum():,}")
