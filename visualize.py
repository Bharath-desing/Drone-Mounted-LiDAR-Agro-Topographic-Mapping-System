"""
visualize.py
-------------
Generates all figure outputs for the mapping system: elevation/DTM maps,
slope & aspect maps, drainage/flow accumulation overlays, canopy height
maps, irrigation zone maps, and a 3D terrain rendering, plus the raw
point cloud scatter for the report.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_point_cloud(cloud, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ground = cloud["classification"] == 2
    ax.scatter(cloud["x"][~ground], cloud["y"][~ground], s=1, c="#5a8f3c", alpha=0.4, label="Vegetation returns")
    ax.scatter(cloud["x"][ground], cloud["y"][ground], s=1, c="#7a5230", alpha=0.6, label="Ground returns")
    ax.set_title("Classified LiDAR Point Cloud (top-down)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(markerscale=8, loc="upper right")
    ax.set_aspect("equal")
    _save(fig, path)


def plot_raster(grid_x, grid_y, values, title, cmap, path, cbar_label, contour=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(grid_x, grid_y, values, cmap=cmap, shading="auto")
    if contour:
        cs = ax.contour(grid_x, grid_y, values, colors="black", linewidths=0.4, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    _save(fig, path)


def plot_dtm(surfaces, path):
    plot_raster(surfaces["x"], surfaces["y"], surfaces["dtm"],
                "Digital Terrain Model (bare-earth elevation)", "terrain",
                path, "Elevation (m)", contour=True)


def plot_chm(surfaces, path):
    plot_raster(surfaces["x"], surfaces["y"], surfaces["chm"],
                "Canopy Height Model (crop height)", "YlGn",
                path, "Height (m)")


def plot_slope(grid_x, grid_y, slope, path):
    plot_raster(grid_x, grid_y, slope, "Slope Map", "inferno", path, "Slope (deg)")


def plot_aspect(grid_x, grid_y, aspect, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(grid_x, grid_y, aspect, cmap="hsv", shading="auto", vmin=0, vmax=360)
    fig.colorbar(im, ax=ax, label="Aspect (deg from North)")
    ax.set_title("Aspect Map")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    _save(fig, path)


def plot_drainage(grid_x, grid_y, dtm, flow_accumulation, path, threshold_pct=95):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(grid_x, grid_y, dtm, cmap="Greys", shading="auto", alpha=0.6)
    fig.colorbar(im, ax=ax, label="Elevation (m)")

    threshold = np.nanpercentile(flow_accumulation, threshold_pct)
    channels = np.where(flow_accumulation >= threshold, flow_accumulation, np.nan)
    ax.pcolormesh(grid_x, grid_y, channels, cmap="Blues", shading="auto")

    ax.set_title("Predicted Drainage Network (blue = concentrated flow)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    _save(fig, path)


def plot_irrigation_zones(grid_x, grid_y, zones, path):
    n_zones = int(np.nanmax(zones))
    cmap = plt.get_cmap("RdYlBu", n_zones)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(grid_x, grid_y, zones, cmap=cmap, shading="auto", vmin=0.5, vmax=n_zones + 0.5)
    cbar = fig.colorbar(im, ax=ax, ticks=range(1, n_zones + 1))
    cbar.set_label("Irrigation Zone (1=needs less water, higher=needs more)")
    ax.set_title("Variable-Rate Irrigation Zones")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    _save(fig, path)


def plot_3d_terrain(surfaces, path):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        surfaces["x"], surfaces["y"], surfaces["dtm"],
        cmap="terrain", linewidth=0, antialiased=True, rstride=1, cstride=1,
    )
    fig.colorbar(surf, ax=ax, shrink=0.6, label="Elevation (m)")
    ax.set_title("3D Terrain Reconstruction from LiDAR")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Elevation (m)")
    ax.view_init(elev=45, azim=-60)
    _save(fig, path)
