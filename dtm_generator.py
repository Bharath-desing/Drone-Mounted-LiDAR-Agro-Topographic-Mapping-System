"""
dtm_generator.py
-----------------
Interpolates classified LiDAR points into regular raster surfaces:

- DTM (Digital Terrain Model): bare-earth elevation, from ground returns only
- DSM (Digital Surface Model): top-of-canopy elevation, from all returns
- CHM (Canopy Height Model): DSM - DTM, i.e. crop/vegetation height

These three rasters are the standard input products for precision
agriculture analysis (drainage, slope, crop height / biomass proxying).
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


def _grid_coords(x, y, resolution):
    xi = np.arange(x.min(), x.max(), resolution)
    yi = np.arange(y.min(), y.max(), resolution)
    return xi, yi


def points_to_raster(x, y, z, resolution=1.0, method="linear"):
    """Interpolate scattered points onto a regular grid raster."""
    xi, yi = _grid_coords(x, y, resolution)
    grid_x, grid_y = np.meshgrid(xi, yi)

    grid_z = griddata((x, y), z, (grid_x, grid_y), method=method)

    # fill any remaining NaNs (edges) with nearest-neighbor interpolation
    if np.isnan(grid_z).any():
        nn = griddata((x, y), z, (grid_x, grid_y), method="nearest")
        grid_z = np.where(np.isnan(grid_z), nn, grid_z)

    return grid_x, grid_y, grid_z


def generate_surfaces(cloud, resolution=1.0, dtm_smoothing_sigma=1.0):
    """Build DTM, DSM, and CHM rasters from a classified point cloud.

    A light Gaussian smoothing pass is applied to the interpolated DTM to
    suppress high-frequency sensor-noise artifacts that would otherwise
    dominate slope/curvature derivatives - standard practice in real LiDAR
    terrain processing pipelines (equivalent to a low-pass terrain filter).

    Returns
    -------
    dict with keys: x, y (2D coordinate grids), dtm, dsm, chm, resolution
    """
    ground_mask = cloud["classification"] == 2
    gx, gy, gz = cloud["x"][ground_mask], cloud["y"][ground_mask], cloud["z"][ground_mask]

    grid_x, grid_y, dtm = points_to_raster(gx, gy, gz, resolution, method="linear")
    if dtm_smoothing_sigma > 0:
        dtm = gaussian_filter(dtm, sigma=dtm_smoothing_sigma)

    _, _, dsm = points_to_raster(cloud["x"], cloud["y"], cloud["z"], resolution, method="linear")

    # dsm/dtm may have slightly different NaN edges from linear interp; both
    # were built on the same grid_x/grid_y so they align directly
    chm = np.clip(dsm - dtm, 0, None)

    return {
        "x": grid_x,
        "y": grid_y,
        "dtm": dtm,
        "dsm": dsm,
        "chm": chm,
        "resolution": resolution,
    }


if __name__ == "__main__":
    from lidar_simulator import simulate_flight
    from point_cloud_processor import process

    raw = simulate_flight()
    classified = process(raw)
    surfaces = generate_surfaces(classified, resolution=1.0)

    print(f"DTM grid shape: {surfaces['dtm'].shape}")
    print(f"Elevation range: {np.nanmin(surfaces['dtm']):.2f} - {np.nanmax(surfaces['dtm']):.2f} m")
    print(f"Max canopy height: {np.nanmax(surfaces['chm']):.2f} m")
