"""
lidar_simulator.py
-------------------
Simulates a raw LiDAR point cloud collected by a drone flying a lawnmower
survey pattern over an agricultural field.

Real drone LiDAR units (e.g. DJI Zenmuse L2, Livox, Velodyne Puck) return
discrete points with (x, y, z, intensity, return_number, classification).
Since no physical flight data is available for this project, this module
generates a physically-plausible synthetic terrain (rolling topography,
a drainage channel, crop rows, and scattered vegetation/canopy returns)
and samples it the way a real LiDAR sensor would during a survey flight.

This lets the rest of the pipeline (ground filtering, DTM generation,
topographic analysis) run on realistic data and be swapped for real
.las/.laz flight data later with no changes downstream.
"""

import numpy as np


def _terrain_elevation(x, y):
    """Ground truth elevation surface (meters) for the synthetic field.

    Combines: gentle regional slope, rolling hills (multi-frequency sine),
    and a carved drainage channel running diagonally across the field.
    """
    regional_slope = 0.015 * x + 0.008 * y
    hills = (
        1.2 * np.sin(x / 40.0) * np.cos(y / 55.0)
        + 0.6 * np.sin(x / 15.0 + 1.3) * np.sin(y / 20.0)
    )
    # Drainage channel: a shallow trench following a diagonal sinusoidal path
    channel_center = 0.35 * x + 20 * np.sin(x / 60.0)
    dist_to_channel = np.abs(y - channel_center)
    channel = -1.8 * np.exp(-(dist_to_channel ** 2) / (2 * 8.0 ** 2))
    return 150.0 + regional_slope + hills + channel


def _canopy_height(x, y, rng):
    """Crop/vegetation height above ground (meters), simulating crop rows."""
    row_spacing = 3.0
    row_pattern = 0.5 * (1 + np.sin(2 * np.pi * y / row_spacing)) ** 2
    base_height = 0.9 * row_pattern
    patchiness = rng.normal(0, 0.08, size=x.shape)
    return np.clip(base_height + patchiness, 0, None)


def simulate_flight(
    field_width=200.0,
    field_length=150.0,
    swath_spacing=25.0,
    points_per_sq_m=8,
    ground_return_ratio=0.55,
    noise_std=0.03,
    seed=42,
):
    """Simulate a drone LiDAR survey and return a raw point cloud.

    Parameters
    ----------
    field_width, field_length : float
        Dimensions of the surveyed field in meters.
    swath_spacing : float
        Distance between parallel flight lines (m) - mimics real flight planning.
    points_per_sq_m : float
        Average point density, typical of drone LiDAR (5-50 pts/m^2).
    ground_return_ratio : float
        Fraction of pulses that penetrate canopy and hit bare ground directly
        (multi-return LiDAR characteristic).
    noise_std : float
        Sensor ranging noise (m), typical survey-grade LiDAR is 2-5 cm.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict of np.ndarray with keys: x, y, z, intensity, classification
        classification: 2 = ground, 5 = vegetation/canopy (ASPRS LAS convention)
    """
    rng = np.random.default_rng(seed)

    n_lines = int(field_length / swath_spacing) + 1
    total_points = int(field_width * field_length * points_per_sq_m)
    points_per_line = total_points // n_lines

    xs, ys = [], []
    swath_width = swath_spacing * 1.3  # sensor field-of-view gives overlapping swaths
    for i in range(n_lines):
        line_y_center = i * swath_spacing
        x_line = rng.uniform(0, field_width, points_per_line)
        # across-track scan angle distribution: denser near nadir (line center),
        # sparser at swath edges - approximates a real oscillating-mirror scanner
        y_line = line_y_center + rng.triangular(
            -swath_width / 2, 0, swath_width / 2, points_per_line
        )
        xs.append(x_line)
        ys.append(y_line)

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    mask = (y >= 0) & (y <= field_length)
    x, y = x[mask], y[mask]

    ground_z = _terrain_elevation(x, y)
    canopy_h = _canopy_height(x, y, rng)

    is_ground_return = rng.random(x.shape[0]) < ground_return_ratio
    z = np.where(is_ground_return, ground_z, ground_z + canopy_h)
    z += rng.normal(0, noise_std, size=z.shape)

    intensity = np.where(
        is_ground_return,
        rng.normal(180, 20, size=z.shape),
        rng.normal(90, 25, size=z.shape),
    )
    intensity = np.clip(intensity, 0, 255)

    classification = np.where(is_ground_return, 2, 5).astype(np.uint8)

    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "z": z.astype(np.float32),
        "intensity": intensity.astype(np.float32),
        "classification": classification,
    }


if __name__ == "__main__":
    cloud = simulate_flight()
    print(f"Simulated {len(cloud['x']):,} LiDAR returns")
    print(f"Ground returns: {(cloud['classification'] == 2).sum():,}")
    print(f"Vegetation returns: {(cloud['classification'] == 5).sum():,}")
