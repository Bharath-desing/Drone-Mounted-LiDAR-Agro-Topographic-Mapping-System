"""
topographic_analysis.py
-------------------------
Derives standard topographic attributes from a DTM raster, used across
precision agriculture, hydrology, and soil science:

- Slope (degrees)      - steepness, drives erosion risk & machinery access
- Aspect (degrees)     - compass direction of steepest descent, affects
                          sun exposure and micro-climate
- Curvature            - concave/convex terrain, affects water accumulation
- Flow direction (D8)  - single-flow-direction routing, standard hydrology algorithm
- Flow accumulation    - upstream contributing area, identifies drainage lines
- TWI (Topographic Wetness Index) - ln(accumulation / tan(slope)), a widely
                          used proxy for soil moisture and waterlogging risk
"""

import numpy as np

# D8 neighbor offsets and their direction codes (ESRI convention)
_D8_OFFSETS = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
]
_D8_DISTANCE_FACTOR = [1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2)]


def compute_slope_aspect(dtm, resolution=1.0):
    """Compute slope (degrees) and aspect (degrees, 0=N clockwise) via gradient."""
    dz_dy, dz_dx = np.gradient(dtm, resolution)

    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    aspect_deg = (90.0 - np.degrees(aspect_rad)) % 360.0

    return slope_deg, aspect_deg


def compute_curvature(dtm, resolution=1.0):
    """Compute profile curvature (2nd derivative) - positive = convex (ridge),
    negative = concave (valley/depression, water-accumulating)."""
    dy, dx = np.gradient(dtm, resolution)
    dyy, dyx = np.gradient(dy, resolution)
    dxy, dxx = np.gradient(dx, resolution)
    curvature = dxx + dyy
    return curvature


def compute_flow_accumulation(dtm):
    """Simplified D8 flow accumulation.

    Each cell drains fully to its steepest downhill neighbor. Accumulation
    is computed by processing cells from highest to lowest elevation,
    passing each cell's accumulated flow (1 + inflow) to its downhill target.
    This is the standard approach used in GIS tools like ArcHydro/QGIS,
    simplified here to single-flow-direction (no partitioning).
    """
    ny, nx = dtm.shape
    accumulation = np.ones((ny, nx), dtype=np.float64)
    flow_to = np.full((ny, nx, 2), -1, dtype=int)

    for r in range(ny):
        for c in range(nx):
            best_drop = 0.0
            target = None
            for (dr, dc), dist_factor in zip(_D8_OFFSETS, _D8_DISTANCE_FACTOR):
                rr, cc = r + dr, c + dc
                if 0 <= rr < ny and 0 <= cc < nx:
                    drop = (dtm[r, c] - dtm[rr, cc]) / dist_factor
                    if drop > best_drop:
                        best_drop = drop
                        target = (rr, cc)
            if target is not None:
                flow_to[r, c] = target

    order = np.dstack(np.unravel_index(np.argsort(-dtm, axis=None), dtm.shape))[0]
    for r, c in order:
        tr, tc = flow_to[r, c]
        if tr != -1:
            accumulation[tr, tc] += accumulation[r, c]

    return accumulation


def compute_twi(slope_deg, flow_accumulation, cell_area=1.0):
    """Topographic Wetness Index: ln(specific catchment area / tan(slope)).

    High TWI = low-lying, flat, high-contributing-area cells that tend to
    stay wet (drainage/waterlogging risk). Low TWI = well-drained upland.
    """
    slope_rad = np.radians(np.clip(slope_deg, 0.1, None))  # avoid div-by-zero
    specific_catchment_area = flow_accumulation * cell_area
    twi = np.log(specific_catchment_area / np.tan(slope_rad))
    return twi


def analyze(dtm, resolution=1.0):
    """Run the full topographic analysis suite on a DTM."""
    slope, aspect = compute_slope_aspect(dtm, resolution)
    curvature = compute_curvature(dtm, resolution)
    flow_acc = compute_flow_accumulation(dtm)
    twi = compute_twi(slope, flow_acc, cell_area=resolution ** 2)

    return {
        "slope": slope,
        "aspect": aspect,
        "curvature": curvature,
        "flow_accumulation": flow_acc,
        "twi": twi,
    }


if __name__ == "__main__":
    from lidar_simulator import simulate_flight
    from point_cloud_processor import process
    from dtm_generator import generate_surfaces

    raw = simulate_flight()
    classified = process(raw)
    surfaces = generate_surfaces(classified, resolution=2.0)
    topo = analyze(surfaces["dtm"], resolution=2.0)

    print(f"Mean slope: {np.nanmean(topo['slope']):.2f} deg")
    print(f"Max flow accumulation: {np.nanmax(topo['flow_accumulation']):.0f} cells")
    print(f"TWI range: {np.nanmin(topo['twi']):.2f} to {np.nanmax(topo['twi']):.2f}")
