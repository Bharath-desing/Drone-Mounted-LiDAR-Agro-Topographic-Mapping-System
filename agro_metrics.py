"""
agro_metrics.py
-----------------
Translates raw topographic rasters into actionable agronomic outputs -
the layer that turns a "topographic survey" into an "agro-topographic
mapping system" for farm management decisions.

Outputs:
- Erosion risk classification (from slope)
- Drainage / waterlogging risk classification (from TWI)
- Variable-rate irrigation zones (from combined slope + TWI + curvature)
- Crop height / biomass summary statistics (from CHM)
- A field-level summary report (dict, JSON-serializable)
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def classify_erosion_risk(slope_deg):
    """Bins slope into standard USDA-style erosion risk classes."""
    risk = np.full(slope_deg.shape, "low", dtype=object)
    risk = np.where(slope_deg >= 3, "moderate", risk)
    risk = np.where(slope_deg >= 8, "high", risk)
    risk = np.where(slope_deg >= 15, "severe", risk)
    return risk


def classify_drainage_risk(twi):
    """Bins TWI into waterlogging / drainage risk classes.

    Thresholds are relative to the field's own TWI distribution since TWI
    is not universally calibrated across sites.
    """
    p33, p66, p90 = np.nanpercentile(twi, [33, 66, 90])
    risk = np.full(twi.shape, "well_drained", dtype=object)
    risk = np.where(twi >= p33, "moderate", risk)
    risk = np.where(twi >= p66, "poorly_drained", risk)
    risk = np.where(twi >= p90, "waterlogging_risk", risk)
    return risk


def generate_irrigation_zones(slope_deg, twi, n_zones=3, smoothing_sigma=2.0):
    """Clusters cells into variable-rate irrigation zones using slope + TWI.

    Uses a simple k-means-free quantile approach on a combined water-need
    index: flatter + higher TWI (wetter) cells need less irrigation;
    steeper + lower TWI (drier, faster runoff) cells need more.

    A Gaussian smoothing pass is applied first since real variable-rate
    irrigation prescriptions use contiguous management zones (a rate
    controller can't practically change every 1.5m), not raw pixel-level
    noise.
    """
    slope_s = gaussian_filter(slope_deg, sigma=smoothing_sigma)
    twi_s = gaussian_filter(twi, sigma=smoothing_sigma)

    norm_slope = (slope_s - np.nanmin(slope_s)) / (np.nanmax(slope_s) - np.nanmin(slope_s) + 1e-9)
    norm_twi = (twi_s - np.nanmin(twi_s)) / (np.nanmax(twi_s) - np.nanmin(twi_s) + 1e-9)

    water_need_index = norm_slope - norm_twi  # high = dry/steep, low = wet/flat

    edges = np.nanpercentile(water_need_index, np.linspace(0, 100, n_zones + 1))
    zones = np.digitize(water_need_index, edges[1:-1]) + 1  # zones 1..n_zones
    return zones, water_need_index


def crop_height_summary(chm):
    """Summary statistics for the Canopy Height Model - a proxy for crop
    growth stage / biomass and useful for identifying stunted patches."""
    valid = chm[~np.isnan(chm)]
    return {
        "mean_height_m": float(np.mean(valid)),
        "std_height_m": float(np.std(valid)),
        "max_height_m": float(np.max(valid)),
        "min_height_m": float(np.min(valid)),
        "low_vigor_area_pct": float(np.mean(valid < (np.mean(valid) - np.std(valid))) * 100),
    }


def field_summary(dtm, slope, aspect, twi, chm, resolution=1.0):
    """Builds a single JSON-serializable summary report for the whole field."""
    erosion = classify_erosion_risk(slope)
    drainage = classify_drainage_risk(twi)

    cell_area_ha = (resolution ** 2) / 10000.0
    total_area_ha = dtm.size * cell_area_ha

    erosion_breakdown = {
        cls: float(np.mean(erosion == cls) * 100)
        for cls in ["low", "moderate", "high", "severe"]
    }
    drainage_breakdown = {
        cls: float(np.mean(drainage == cls) * 100)
        for cls in ["well_drained", "moderate", "poorly_drained", "waterlogging_risk"]
    }

    return {
        "field_area_ha": round(total_area_ha, 2),
        "elevation_range_m": [round(float(np.nanmin(dtm)), 2), round(float(np.nanmax(dtm)), 2)],
        "mean_slope_deg": round(float(np.nanmean(slope)), 2),
        "max_slope_deg": round(float(np.nanmax(slope)), 2),
        "dominant_aspect_deg": round(float(np.nanmedian(aspect)), 1),
        "erosion_risk_pct": {k: round(v, 1) for k, v in erosion_breakdown.items()},
        "drainage_risk_pct": {k: round(v, 1) for k, v in drainage_breakdown.items()},
        "crop_canopy": crop_height_summary(chm),
    }


if __name__ == "__main__":
    from lidar_simulator import simulate_flight
    from point_cloud_processor import process
    from dtm_generator import generate_surfaces
    from topographic_analysis import analyze
    import json

    raw = simulate_flight()
    classified = process(raw)
    surfaces = generate_surfaces(classified, resolution=2.0)
    topo = analyze(surfaces["dtm"], resolution=2.0)

    summary = field_summary(
        surfaces["dtm"], topo["slope"], topo["aspect"], topo["twi"], surfaces["chm"],
        resolution=2.0,
    )
    print(json.dumps(summary, indent=2))
