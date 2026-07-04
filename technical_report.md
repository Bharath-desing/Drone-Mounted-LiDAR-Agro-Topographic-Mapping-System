# Technical Report: Drone-Mounted LiDAR Agro-Topographic Mapping System

## 1. Problem Statement

Farm management decisions — irrigation scheduling, drainage remediation, erosion control, variable-rate input application — depend on accurate, high-resolution knowledge of field topography. Traditional ground surveying is slow and sparse; satellite-derived elevation data (e.g. SRTM, 30m resolution) is far too coarse to resolve field-scale drainage patterns or crop-row-level detail. Drone-mounted LiDAR closes this gap, delivering centimeter-accurate 3D point clouds over tens to hundreds of hectares in a single flight.

This project implements the complete processing chain required to convert a raw LiDAR point cloud into farm-ready topographic and agronomic products.

## 2. Data

No physical drone flight was available for this project. A synthetic point cloud generator (`src/lidar_simulator.py`) produces a physically-plausible stand-in dataset:

- **Terrain**: a multi-frequency elevation surface (regional slope + rolling hills) with an explicitly carved diagonal drainage channel, so downstream drainage-network extraction has a real feature to recover.
- **Canopy**: a periodic crop-row height pattern with random patchiness, giving the Canopy Height Model something realistic to characterize.
- **Sensor model**: multi-swath flight lines with overlapping triangular across-track point density (denser near nadir, as in a real oscillating-mirror or rotating-polygon scanner), ~55% direct ground penetration (multi-return LiDAR characteristic), and 3cm ranging noise (typical of survey-grade drone LiDAR).

Every module downstream of the simulator consumes a plain `{x, y, z, intensity, classification}` array dictionary — the same shape a `.las`/`.laz` file loader (e.g. via `laspy`) would produce from a real flight — so the simulator is a transparent, swappable stand-in.

## 3. Methods

### 3.1 Point Cloud Cleaning

Statistical Outlier Removal (SOR): for each point, the mean distance to its *k* nearest neighbors (k=8) is computed via a KD-tree. Points whose mean neighbor distance exceeds `mean + 2.5·std` of the global distribution are discarded as sensor noise or stray returns.

### 3.2 Ground Classification

A simplified Progressive Morphological Filter (PMF), following the general approach of Zhang et al. (2003):

1. Bin points into a regular grid, keeping the minimum elevation per cell.
2. Apply a grayscale morphological erosion with an increasing window size.
3. At each window size, compare the surface to its eroded version; where the elevation difference exceeds a slope-adaptive threshold (`slope × window_size × cell_size + initial_dh`), the surface is replaced with the eroded (lower) value.
4. Points within `initial_dh` of the final surface are classified as ground (LAS class 2); all others as non-ground/vegetation (class 5).

This is the same family of algorithm used in production tools such as PDAL's `filters.pmf` and commercial `lasground`.

### 3.3 Surface Generation

- **DTM** (bare-earth): linear interpolation (`scipy.interpolate.griddata`) of ground-classified points onto a regular grid, followed by light Gaussian smoothing (σ=1 cell) to suppress interpolation/sensor-noise artifacts before derivative-based analysis.
- **DSM** (first-surface): same interpolation over *all* classified points (ground + canopy).
- **CHM** (canopy height): `DSM − DTM`, clipped to ≥0.

### 3.4 Topographic Analysis

- **Slope & Aspect**: computed from the DTM gradient (`np.gradient`); slope as `arctan(|∇z|)`, aspect as compass bearing of steepest descent.
- **Curvature**: sum of second derivatives (`∂²z/∂x² + ∂²z/∂y²`); negative values indicate concave, water-accumulating terrain.
- **Flow Accumulation**: single-flow-direction D8 routing. Each cell drains fully to whichever of its 8 neighbors has the steepest downhill gradient; accumulation is propagated from highest to lowest elevation. This is the same core algorithm behind ArcGIS Hydrology and GRASS `r.watershed`.
- **Topographic Wetness Index (TWI)**: `ln(specific catchment area / tan(slope))`, a standard proxy for soil moisture and waterlogging risk in digital soil mapping (Beven & Kirkby, 1979).

### 3.5 Agronomic Interpretation

- **Erosion risk**: slope binned into USDA-style classes (low <3°, moderate 3–8°, high 8–15°, severe >15°).
- **Drainage risk**: TWI binned into field-relative percentile classes (well-drained, moderate, poorly-drained, waterlogging risk), since TWI is not universally calibrated across sites and must be interpreted relative to the field's own distribution.
- **Irrigation zoning**: a smoothed (σ=2 cell Gaussian) composite water-need index (`normalized slope − normalized TWI`) is binned into 3 contiguous management zones — dry/steep areas needing more water vs. flat/wet areas needing less — matching how commercial variable-rate irrigation (VRI) prescriptions are typically constructed.

## 4. Results (Synthetic Field, 3.02 ha)

| Metric | Value |
|---|---|
| Elevation range | 148.3 – 155.7 m |
| Mean slope | 2.77° |
| Max slope | 10.4° |
| Dominant aspect | 236° (SW-facing) |
| Erosion risk (low / moderate / high / severe) | 73.8% / 21.7% / 4.4% / 0.0% |
| Drainage risk (well / moderate / poor / waterlogging) | 33% / 33% / 24% / 10% |
| Mean canopy height | 0.20 m |

The drainage network extraction (Figure 6) correctly recovers the dendritic channel pattern converging on the synthetically-carved trench, validating the D8 flow routing implementation. The irrigation zone map (Figure 7) shows spatial coherence with the underlying slope/TWI fields rather than pixel-level noise, confirming the zoning smoothing step is effective.

## 5. Limitations

- Synthetic data cannot fully capture real-world LiDAR artifacts (multipath returns, GPS/IMU drift, strip misalignment between flight lines).
- D8 flow routing is a simplification; multi-flow-direction algorithms (e.g. D-infinity) better represent divergent flow on convex terrain.
- No absolute georeferencing (CRS/UTM) is applied — outputs are in a local field-relative coordinate frame.

## 6. References

- Zhang, K. et al. (2003). *A progressive morphological filter for removing non-ground measurements from airborne LiDAR data.* IEEE TGRS.
- Beven, K.J. & Kirkby, M.J. (1979). *A physically based, variable contributing area model of basin hydrology.* Hydrological Sciences Bulletin.
- Tarboton, D.G. (1997). *A new method for the determination of flow directions and upslope areas in grid digital elevation models.* Water Resources Research (D-infinity, referenced as future-work alternative to D8).
