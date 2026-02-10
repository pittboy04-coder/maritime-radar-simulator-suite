# Terrain-Occluded Marine Radar Simulator — System Reference

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture and Data Flow](#2-architecture-and-data-flow)
3. [Component Breakdown](#3-component-breakdown)
   - 3.1 [HeightMap and TerrainConfig](#31-heightmap-and-terrainconfig)
   - 3.2 [OcclusionEngine](#32-occlusionengine)
   - 3.3 [Modified DetectionEngine](#33-modified-detectionengine)
   - 3.4 [Modified Simulation](#34-modified-simulation)
   - 3.5 [Inherited Radar Chain](#35-inherited-radar-chain)
4. [How a Sweep Is Produced](#4-how-a-sweep-is-produced)
5. [Relation to Research Goals](#5-relation-to-research-goals)
   - 5.1 [Goal 1 — SRTM/NASADEM Integration and Literature Baseline Validation](#51-goal-1--srtmnasadem-integration-and-literature-baseline-validation)
   - 5.2 [Goal 2 — Configurable X-Band Parameters and Scenario Editor GUI](#52-goal-2--configurable-x-band-parameters-and-scenario-editor-gui)
   - 5.3 [Goal 3 — Validation Against 4–5 Real Radar Scenarios (<20% Mean Geometric Error)](#53-goal-3--validation-against-45-real-radar-scenarios-20-mean-geometric-error)
   - 5.4 [Goal 4 — >10,000 Ray-Terrain Intersections/Second via C++ Acceleration](#54-goal-4--10000-ray-terrain-intersectionssecond-via-c-acceleration)
   - 5.5 [Deliverables Mapping](#55-deliverables-mapping)
6. [Current Limitations](#6-current-limitations)
7. [Proposed Changes and Improvements](#7-proposed-changes-and-improvements)

---

## 1. System Overview

This simulator extends a full marine radar simulation chain (antenna rotation, beam patterns, target detection, sea/rain clutter, coastline returns, weather effects) with two new capabilities:

- **Terrain height maps** — grid-based elevation surfaces that represent islands, ridges, or any landform with actual height above sea level.
- **Occlusion engine** — a ray-march algorithm that uses those height maps to determine what the radar can and cannot see, producing both shadow zones behind terrain and radar returns from the terrain surfaces themselves.

The original radar chain (copied from `furuno_radar_ppi`) handles everything a flat-water radar simulation needs: antenna patterns, R^4 signal falloff, RCS-based target strength, range/bearing quantisation, sea clutter, rain clutter, thermal noise, and 2-D coastline polygon returns. This project adds the vertical dimension. Terrain is no longer just a 2-D outline that produces flat returns — it has height, it blocks line-of-sight, and it casts radar shadows.

---

## 2. Architecture and Data Flow

```
                         ┌─────────────────┐
                         │   HeightMap(s)   │  numpy float32 grids
                         │   terrain.py     │  with bilinear interp
                         └────────┬────────┘
                                  │
                                  ▼
┌──────────┐           ┌─────────────────────┐
│  Vessel  │           │   OcclusionEngine   │
│  objects │           │   occlusion.py      │
└────┬─────┘           │                     │
     │                 │  • is_target_occluded()     ─── LOS check
     │                 │  • generate_terrain_returns() ── terrain echoes
     │                 │  • get_occlusion_mask()      ─── shadow map
     │                 │  • compute_occlusion_profile()─ angle profile
     │                 └──────┬──────────────┘
     │                        │
     ▼                        ▼
┌─────────────────────────────────────────┐
│            DetectionEngine              │
│            detection.py                 │
│                                         │
│  For each target:                       │
│    1. Range/bearing check               │
│    2. ── NEW ── occlusion_engine        │
│       .is_target_occluded() ───────►skip│
│    3. Radar equation (R^4, RCS, gain)   │
│    4. Measurement noise                 │
│    5. Emit Detection / sweep bin        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│              Simulation                 │
│              simulation.py              │
│                                         │
│  get_radar_sweep_data(bearing):         │
│    1. Target returns  (from RadarSystem)│
│    2. Coastline returns (polygon fill)  │
│    3. ── NEW ── Terrain returns         │
│       (OcclusionEngine                  │
│        .generate_terrain_returns())     │
│    4. Weather (sea clutter, rain, noise)│
│                                         │
│  Each layer merged via max(existing,new)│
└─────────────────────────────────────────┘
```

The key integration points are:

- **DetectionEngine** now holds an optional reference to `OcclusionEngine`. When set, every target is checked with `is_target_occluded()` before signal strength is computed. Occluded targets are skipped entirely — they produce no detection and no sweep bin energy.
- **Simulation.get_radar_sweep_data()** now has a terrain-returns stage between coastline returns and weather effects. The occlusion engine's `generate_terrain_returns()` produces range-bin intensities that are merged into the sweep using `max()`, the same way coastline returns are merged.
- **Simulation.add_terrain()** triggers `_rebuild_occlusion_engine()`, which constructs a new `OcclusionEngine` from all current height maps and passes the antenna height from `RadarParameters.antenna_height_m`. It also wires the engine into `DetectionEngine.occlusion_engine`.

---

## 3. Component Breakdown

### 3.1 HeightMap and TerrainConfig

**File:** `radar_sim/environment/terrain.py`

**TerrainConfig** is a dataclass that pins a grid to the world coordinate system:

| Field | Default | Purpose |
|-------|---------|---------|
| `origin_x` | 0.0 | World X coordinate of grid cell (0, 0) |
| `origin_y` | 0.0 | World Y coordinate of grid cell (0, 0) |
| `cell_size` | 50.0 m | Distance in meters between adjacent grid cells |
| `reflectivity` | 0.85 | Radar reflectivity of terrain surface (0–1) |
| `roughness` | 0.3 | Surface roughness factor (reserved for future use) |

**HeightMap** wraps a 2-D numpy float32 array where each cell holds an elevation in meters above sea level. Row index increases with world-Y, column index increases with world-X.

**`get_elevation(wx, wy)`** is the core query method. It converts world coordinates to fractional grid coordinates, clamps to the grid boundary (returning 0.0 outside), then performs bilinear interpolation across the four surrounding cells:

```
elevation = e00*(1-dr)*(1-dc) + e01*(1-dr)*dc + e10*dr*(1-dc) + e11*dr*dc
```

This gives smooth elevation values at any world position without staircase artifacts from the grid discretisation.

**Factory methods:**

- `HeightMap.from_array(elevations, origin_x, origin_y, cell_size)` — takes any numpy array directly. This is the intended entry point for real-world data: load an SRTM HGT file or NASADEM GeoTIFF into numpy, crop to the area of interest, and pass it here.
- `HeightMap.from_generator(rows, cols, generator_fn, ...)` — takes a `callable(row, col) -> elevation` for procedural terrain.

**Pre-built factories:**

- `create_island_terrain(center_x, center_y, radius, peak_height, ...)` — produces a conical island with a smooth cosine falloff from peak to sea level at the given radius, plus small Gaussian noise for surface texture. Uses a fixed random seed (42) for reproducibility.
- `create_ridge_terrain(start_x, start_y, end_x, end_y, width, peak_height, ...)` — produces a linear ridge between two endpoints with a cosine cross-section profile. Useful for simulating mountainous coastlines or headlands.

### 3.2 OcclusionEngine

**File:** `radar_sim/environment/occlusion.py`

Constructed with a list of `HeightMap` objects and the antenna height in meters. The helper `_sample_elevation(wx, wy)` queries every height map and returns the maximum elevation at that point — this allows overlapping terrain features (e.g. an island on top of a broader shelf).

**`is_target_occluded(origin_x, origin_y, target_x, target_y, target_height_m, step_m)`**

This is the line-of-sight (LOS) check. The algorithm:

1. Compute the range and bearing from antenna to target.
2. Compute the elevation angle from the antenna to the *top* of the target: `atan2(target_height - antenna_height, range)`.
3. Ray-march from the antenna toward the target in steps of `step_m` (default 50 m).
4. At each step, sample the terrain elevation and compute the elevation angle from the antenna to that terrain point: `atan2(terrain_elev - antenna_height, distance)`.
5. Track the running maximum terrain elevation angle.
6. If the maximum terrain angle exceeds the target angle at any point before reaching the target, the target is occluded.

The step size of 50 m matches the default `cell_size` of the height map, giving one sample per grid cell. This is sufficient for terrain features at the scale of islands and ridges but can be reduced for finer resolution.

**`generate_terrain_returns(origin_x, origin_y, bearing_deg, beamwidth_deg, max_range_m, num_bins)`**

Generates radar return intensities from terrain surfaces. The algorithm:

1. Cast 5 sub-rays evenly spread across the antenna beamwidth (from -half_beam to +half_beam).
2. Each ray is weighted by beam pattern: `weight = 1.0 - 0.3 * |offset| / half_beam` (centre rays are stronger).
3. For each ray, march outward bin-by-bin. At each bin, sample terrain elevation.
4. Track a running maximum elevation angle. If the current terrain point's angle equals or exceeds the max angle, it is *visible* — it produces a return. Otherwise it is shadowed by nearer, higher terrain.
5. Visible terrain intensity = `reflectivity * ray_weight * min(1.0, elevation / 50.0)`. Higher terrain produces stronger returns.
6. Returns are merged across sub-rays using `max()`.

The shadow-casting logic means terrain behind a tall peak produces no return — the radar cannot see it, just as in reality.

**`get_occlusion_mask(origin_x, origin_y, bearing_deg, max_range_m, num_bins)`**

Returns a boolean list where `True` means the range bin is in radar shadow. The algorithm tracks the running max elevation angle along the bearing. Once terrain establishes a shadow line, every subsequent bin whose elevation is below `antenna_height + distance * tan(max_angle)` is marked as shadowed.

**`compute_occlusion_profile(origin_x, origin_y, bearing_deg, max_range_m, step_m)`**

Returns the running maximum elevation angle at each step along a bearing. This is the raw data behind shadow masks and occlusion checks — useful for visualising the elevation-angle envelope along a specific bearing direction.

### 3.3 Modified DetectionEngine

**File:** `radar_sim/radar/detection.py`

The `DetectionEngine` class gains one new attribute: `self.occlusion_engine: Optional[OcclusionEngine] = None`. When this is set (by `Simulation._rebuild_occlusion_engine()`), two methods gain occlusion filtering:

**`detect_targets()`** — after checking that a target is active and within radar range, but *before* computing received power, the engine calls `occlusion_engine.is_target_occluded()`. If the target is occluded, it is skipped entirely and produces no `Detection` object. This means occluded vessels do not appear in the detection list, tracking, or any downstream processing.

**`generate_sweep_data()`** — same check inserted after the beam-pattern gate. Occluded targets produce no energy in any range bin. This is the per-sweep equivalent: even if the antenna is pointed at an occluded target, the sweep shows nothing at that range.

Both methods pass `target.height` as the `target_height_m` parameter, so taller targets are harder to fully occlude — a 25 m cargo ship is visible over terrain that would hide an 8 m fishing boat.

### 3.4 Modified Simulation

**File:** `radar_sim/core/simulation.py`

New attributes:
- `self.terrain_maps: List[HeightMap]` — all loaded terrain height maps.
- `self.occlusion_engine: Optional[OcclusionEngine]` — the engine built from those maps.

New methods:
- `add_terrain(terrain: HeightMap)` — appends a terrain map and calls `_rebuild_occlusion_engine()`.
- `clear_terrain()` — removes all terrain maps, sets `occlusion_engine` to `None`, and disconnects it from `DetectionEngine`.
- `_rebuild_occlusion_engine()` — constructs a new `OcclusionEngine` with the current list of terrain maps and the antenna height from `RadarParameters.antenna_height_m`, then wires it into `self.radar.detection_engine.occlusion_engine`.

Modified method — `get_radar_sweep_data(bearing)`:

The sweep assembly pipeline is now four stages:

1. **Target returns** — `self.radar.get_sweep_at_bearing(bearing)` retrieves the pre-computed sweep buffer, which already has occlusion-filtered target data.
2. **Coastline returns** — existing behaviour, unchanged. Each coastline polygon's returns are merged via `max()`.
3. **Terrain returns** — NEW. `self.occlusion_engine.generate_terrain_returns()` produces terrain surface echoes with shadow casting. Merged via `max()`, scaled by `self.radar.params.gain`.
4. **Weather effects** — sea clutter, rain clutter, and thermal noise applied last.

The `reset()` method now also calls `clear_terrain()`.

### 3.5 Inherited Radar Chain

These modules are copied directly from `furuno_radar_ppi` without modification:

| Module | Key Classes | Role |
|--------|------------|------|
| `radar/parameters.py` | `RadarParameters` | All radar config: frequency (9.41 GHz X-band), power (25 kW), beamwidth (1.2 deg horizontal, 22 deg vertical), rotation (24 RPM), range scales, gain/clutter knobs |
| `radar/antenna.py` | `Antenna` | Rotation simulation, sinc-squared beam pattern, bearing tracking |
| `radar/system.py` | `RadarSystem` | Integrates antenna + detection engine, manages 360x512 sweep buffer, detection persistence |
| `core/world.py` | `World` | Container for vessels, time tracking, position updates |
| `core/range_bearing.py` | (functions) | Navigation math: range, bearing, CPA, polar/cartesian conversion |
| `objects/vessel.py` | `Vessel`, `VesselType` | Vessel state: position, course, speed, dimensions, RCS estimation |
| `environment/coastline.py` | `Coastline` | 2-D polygon coastlines with ray-casting filled returns |
| `environment/clutter.py` | `SeaClutter`, `RainClutter` | Range-dependent clutter with wind/sea-state coupling |
| `environment/noise.py` | `NoiseGenerator` | Gaussian thermal noise |
| `environment/weather.py` | `WeatherEffects` | Combines clutter + noise into sweep-level weather effects |

---

## 4. How a Sweep Is Produced

Walking through what happens when `sim.get_radar_sweep_data(bearing=0)` is called for a bearing that crosses the island in the demo scenario:

**Step 1 — Target returns.** The `RadarSystem` sweep buffer already contains data from the most recent `sim.update()`. During that update, `DetectionEngine.generate_sweep_data()` was called for the antenna's current bearing. For bearing 0 (north), the hidden cargo vessel at (0, 5000) would normally produce a strong return — but the occlusion check inside `generate_sweep_data()` called `is_target_occluded(0, 0, 0, 5000, target_height_m=25)`, which ray-marched north and found that the island (peak 120 m at 3000 m range) subtends an elevation angle of roughly `atan2(120 - 15, 3000) ≈ 2.0 deg`, while the target at 5000 m subtends only `atan2(25 - 15, 5000) ≈ 0.11 deg`. Since 2.0 > 0.11, the target is occluded. No energy is written into the sweep buffer for this target at this bearing.

**Step 2 — Coastline returns.** No coastlines are loaded in the demo scenario, so this stage produces nothing.

**Step 3 — Terrain returns.** `OcclusionEngine.generate_terrain_returns()` casts 5 sub-rays across the 1.2 degree beamwidth. Each ray marches outward. At range bins corresponding to ~2200–3800 m, terrain elevation rises above zero. The first visible terrain points produce returns with intensity proportional to `0.85 * ray_weight * min(1.0, elev/50)`. At the peak (120 m, clipped to 1.0 by the min), intensity reaches `0.85 * 1.0 * 1.0 = 0.85` before gain scaling. The far side of the island is in shadow — its elevation angle is lower than the peak's, so it produces no return. Beyond the island, the terrain drops to zero, and the shadow persists. This matches real radar behaviour: you see the near face of an island but not the far side.

**Step 4 — Weather effects.** Sea clutter (sea state 1, minimal), rain clutter (0 mm/h), and thermal noise are applied. At sea state 1, close-range clutter is faint. Thermal noise adds a ~0.02 standard deviation Gaussian floor across all bins.

**Result:** The sweep shows a terrain echo cluster at 2.5–3.5 km (the island), noise elsewhere, and no target signature at 5 km. For bearing 90 (east toward the visible tanker), no terrain exists, so the target return from the tanker appears normally in the sweep.

---

## 5. Relation to Research Goals

The four formal research goals and the project deliverables are listed below, each mapped to what the current codebase already provides and what specific work remains.

### 5.1 Goal 1 — SRTM/NASADEM Integration and Literature Baseline Validation

> *Integrate SRTM/NASADEM terrain data with ray-casting and replicate documented scenarios from literature for baseline validation.*

**What exists now.**
The `HeightMap` class stores elevation as a numpy float32 grid and exposes `get_elevation(wx, wy)` with bilinear interpolation. The factory `HeightMap.from_array(elevations, origin_x, origin_y, cell_size)` accepts any numpy array without assumptions about its source — this is the integration point for SRTM and NASADEM data. The `OcclusionEngine` performs ray-terrain intersection via ray-marching, sampling `get_elevation()` at each step and tracking the running maximum elevation angle. This is functionally a discrete ray-cast against a height-field surface.

The demo (`main.py`) already performs a basic literature-style validation pattern: place a radar at a known position, place terrain between the radar and a target, and confirm that the LOS check correctly reports the target as occluded. The elevation-angle comparison (`max_terrain_angle > target_angle`) is the standard propagation-geometry test described in radar textbook treatments of terrain masking (e.g. Skolnik, *Introduction to Radar Systems*, Chapter 12; Barton, *Modern Radar System Analysis*, Section 2.5).

**What needs to be built.**

| Task | Detail |
|------|--------|
| **SRTM loader** | A function `HeightMap.from_srtm(filepath, radar_lat, radar_lon)` that reads a `.hgt` file (1201x1201 or 3601x3601 big-endian int16), converts to float32 metres, computes origin in local East/North metres using equirectangular projection relative to the radar position. Also a NASADEM variant using `rasterio` for GeoTIFF. |
| **Earth curvature** | Replace flat-earth `atan2` with the 4/3 effective Earth radius model: `adjusted_elev = terrain_elev - range^2 / (2 * R_eff)` where `R_eff = 6371000 * 4/3`. Without this, LOS accuracy degrades beyond ~10 NM. |
| **Literature scenario replication** | Select 2–3 published terrain-masking scenarios with known geometry and expected results (e.g. a VTS radar coverage analysis for a specific port, an ITU-R P.526 propagation example, or a coastal surveillance study with documented shadow sectors). Load the corresponding SRTM tile, configure the antenna position and height, and compare `get_occlusion_mask()` output against the published shadow boundaries. Document the comparison quantitatively. |

**How this gets the project to Goal 1.**
Once the SRTM loader and Earth curvature correction are in place, the system can ingest real terrain for any location worldwide and produce occlusion results that are directly comparable to published analyses. The ray-march + elevation-angle algorithm is already the correct physical model — the remaining work is connecting it to real data sources and validating against known benchmarks.

### 5.2 Goal 2 — Configurable X-Band Parameters and Scenario Editor GUI

> *Develop configurable radar parameters matching our X-band radar specifications with scenario editor GUI enabling non-programmer configuration.*

**What exists now.**
`RadarParameters` (in `radar/parameters.py`) is a dataclass with every relevant X-band parameter exposed as a named field:

| Parameter | Current default | Configurable |
|-----------|----------------|-------------|
| `frequency_ghz` | 9.41 | Yes |
| `peak_power_kw` | 25.0 | Yes |
| `horizontal_beamwidth_deg` | 1.2 | Yes |
| `vertical_beamwidth_deg` | 22.0 | Yes |
| `antenna_height_m` | 15.0 | Yes |
| `rotation_rpm` | 24.0 | Yes |
| `pulse_lengths_us` | [0.07, 0.15, 0.5, 0.8, 1.2] | Yes |
| `range_scales_nm` | [0.25 ... 96] | Yes |
| `gain` | 0.5 | Yes |
| `sea_clutter` / `rain_clutter` | 0.3 | Yes |

All fields can be set at construction or modified at runtime. The `Simulation` class provides `add_terrain()`, `add_coastline()`, `world.add_vessel()`, and `weather.set_conditions()` to compose a full scenario from code.

The `Vessel` dataclass allows setting position (x, y), course, speed, dimensions (length, beam, height), and optionally RCS. Vessel types are enumerated (`CARGO`, `TANKER`, `FISHING`, etc.).

**What needs to be built.**

| Task | Detail |
|------|--------|
| **JSON/YAML scenario files** | Define a schema for scenario configuration: radar parameters, terrain file paths, vessel placements, weather conditions, coastline definitions. Load with a `Scenario.from_file(path)` method. This is the prerequisite for a GUI — the GUI writes scenario files, the engine reads them. |
| **GUI scenario editor** | A desktop application (likely Tkinter or PyQt, matching the existing Furuno project's use of pygame) with panels for: (a) radar parameter sliders/fields matching all `RadarParameters` fields; (b) a map view showing terrain contours and vessel positions with drag-to-place; (c) a terrain file browser for loading SRTM tiles; (d) a weather conditions panel; (e) a "Run" button that executes the simulation and shows results. The GUI writes scenario JSON and calls `Simulation` methods. |
| **Parameter presets** | Pre-configured parameter sets for specific radar models (e.g. Furuno FAR-2127, FAR-2837S, DRS25A) that a non-programmer can select from a dropdown. |

**How this gets the project to Goal 2.**
The data model for all radar parameters already exists and is fully configurable. The missing piece is the non-programmer interface layer. A JSON scenario format provides the serialization boundary, and the GUI provides the interaction layer. The simulation engine itself requires no changes for this goal — only a UI wrapper.

### 5.3 Goal 3 — Validation Against 4–5 Real Radar Scenarios (<20% Mean Geometric Error)

> *Validate simulator against 4–5 real radar scenarios spanning coastal, open water, and harbor environments, achieving <20% mean geometric error relative to measured returns (comparable to automotive radar simulation benchmarks), with documented error metrics and sensitivity analysis.*

**What exists now.**
The simulation produces per-bearing sweep data as a list of 512 intensity values (range bins), the same format used by the existing `maritime_radar_sim` CSV export pipeline. The `get_radar_sweep_data(bearing)` method composites target returns, coastline returns, terrain returns, and weather effects into a single sweep. This output is directly comparable to recorded real-radar sweep data in the same bin/intensity format.

The `DetectionEngine.calculate_received_power()` implements a simplified radar equation: signal is proportional to `antenna_gain * (R_max/R)^4 * RCS/1000 * gain`. This captures the dominant R^4 range dependence and RCS scaling.

**What needs to be built.**

| Task | Detail |
|------|--------|
| **Real radar data collection** | Record PPI sweep data from the actual X-band radar at 4–5 locations: at least one coastal environment with terrain (island/headland), one open water scenario, one harbor approach, one with mixed terrain and vessel traffic, and optionally one with heavy weather. Export as CSV using the existing recording tools. |
| **Scenario reconstruction** | For each recorded scenario, build the matching simulation: load the SRTM terrain for that location, place vessels at their recorded positions (from AIS data or manual observation notes), set weather conditions to match, and configure radar parameters to match the specific radar installation. |
| **Error metric framework** | Implement a comparison module that aligns simulated and measured sweeps and computes: (a) mean geometric error — the average difference in range to terrain/target returns, expressed as a percentage of true range; (b) intensity RMSE — root-mean-square error of return intensity across range bins; (c) shadow boundary error — displacement of predicted vs. actual terrain shadow edges in range bins. The <20% target applies to geometric error (range to returns). |
| **Sensitivity analysis** | Systematically vary key parameters (antenna height ±2 m, terrain cell size 30 m vs. 90 m, reflectivity ±0.1, sea state ±1) and report how each affects the error metrics. This identifies which parameters the validation is most sensitive to and which need the most careful calibration. |

**How the current system supports this.**
The sweep data pipeline already produces output in the right format for comparison. The terrain returns and occlusion are the new elements being validated — the rest of the radar chain (targets, clutter, noise) was already functional in the Furuno project. The <20% geometric error target is achievable because the dominant source of geometric error in terrain scenarios is the LOS calculation, and the elevation-angle ray-march is a well-understood, physically correct method. The main sources of error will be: (a) SRTM resolution (30 m or 90 m) vs. actual terrain at finer scales, (b) flat-earth vs. curved-earth at longer ranges (fixed by the 4/3 Earth radius correction), and (c) reflectivity model simplifications.

### 5.4 Goal 4 — >10,000 Ray-Terrain Intersections/Second via C++ Acceleration

> *Demonstrate >10,000 ray-terrain intersections/second via C++ acceleration.*

**What exists now.**
The ray-terrain intersection is implemented in pure Python in `OcclusionEngine`. The core loop in `is_target_occluded()` marches along a ray in 50 m steps, calling `_sample_elevation()` (which queries each HeightMap's `get_elevation()` with bilinear interpolation) at each step. For a 10 km range at 50 m steps, that is 200 elevation lookups per ray.

A rough benchmark of the current Python implementation: a single `is_target_occluded()` call at 5 km range with 50 m steps takes approximately 0.5–1.0 ms on a modern CPU (100 steps × ~5–10 μs per bilinear lookup with numpy). This gives approximately 1,000–2,000 LOS checks per second in Python — below the 10,000 target.

For `generate_terrain_returns()`, which casts 5 sub-rays × 512 bins = 2,560 elevation lookups per bearing, a full 360-degree sweep requires 360 × 2,560 = 921,600 lookups. In Python this would take roughly 5–10 seconds. The 10,000 intersections/second target requires this to be at least 100× faster.

**What needs to be built.**

| Task | Detail |
|------|--------|
| **C++ ray-march kernel** | Write a C++ function that takes: a pointer to the elevation grid (float32), grid dimensions, origin, cell size, antenna height, ray origin, ray direction, max range, and step size. It performs the same ray-march loop with bilinear interpolation and returns the max elevation angle (for occlusion profiles) or a boolean (for LOS checks). The bilinear interpolation is 4 multiplies and 3 adds — trivially fast in C++. |
| **pybind11 / ctypes binding** | Expose the C++ kernel to Python. pybind11 is the standard choice for numpy interoperability. The binding accepts numpy arrays for the grid and scalar parameters for the ray, returning results directly. |
| **Batch ray-cast API** | Expose a function that takes an array of N ray directions and returns N results in a single call, avoiding Python-to-C++ call overhead per ray. For a 360-degree sweep, pass all 360 bearings at once. |
| **Benchmark harness** | A script that runs N ray-terrain intersection calls, measures wall-clock time, and reports intersections/second. Run on the target hardware and document results. The 10,000 target should be easily exceeded — a single C++ bilinear lookup takes ~10–50 ns, so 200 lookups per ray takes ~2–10 μs per ray, giving 100,000–500,000 rays/second for single-threaded C++. With OpenMP threading across bearings, millions/second is feasible. |
| **CMake build** | A `CMakeLists.txt` that compiles the C++ extension and installs it into the Python package. Fallback to the pure-Python implementation if the C++ extension is not built. |

**How this gets the project to Goal 4.**
The algorithmic structure is already correct and the Python prototype works. The C++ port is a direct translation of the inner loop — no algorithmic changes needed. The 10,000 intersections/second target is conservative for C++; the actual achievable throughput is likely 100,000+ intersections/second on a single core, well above the requirement. The key deliverable is the benchmark demonstration showing the measured throughput.

### 5.5 Deliverables Mapping

| Deliverable | Current status | Remaining work |
|-------------|---------------|----------------|
| **Open-source Python/C++ simulator on GitHub under MIT license** | Repository created at `github.com/pittboy04-coder/Validated-Terrain-Occluded-Radar-Simulation`. Python simulator is functional with terrain occlusion. No license file yet. | Add MIT LICENSE file. Build C++ acceleration module. Add documented API (docstrings exist; add a formal API reference page or use Sphinx/pdoc). |
| **Documented API** | All public classes and methods have docstrings. This SYSTEM_REFERENCE.md documents architecture and internals. | Generate API reference docs (Sphinx autodoc or pdoc3). Add usage examples for each major class. |
| **Validation dataset: simulated vs. measured returns with full error analysis** | Sweep data export pipeline exists (inherited from furuno_radar_ppi). Comparison framework not yet built. | Collect real radar data at 4–5 scenarios. Build comparison/error-metric module. Run validation and produce error analysis report. |
| **Poster presentation at USC Summer Research Symposium** | N/A | Prepare poster with: system architecture diagram, validation results (simulated vs. measured PPI images side-by-side), error metrics table, performance benchmark (C++ throughput), and shadow-zone map for a real terrain scenario. |

---

## 6. Current Limitations

**Flat-earth geometry.** The LOS check uses `atan2(height_diff, slant_range)` without Earth curvature. For ranges beyond ~20 km this introduces meaningful error — the radar horizon due to Earth curvature alone is approximately `d_km = 4.12 * sqrt(h_m)`, which gives ~16 km for a 15 m antenna. At shorter ranges (under 10 NM / 18.5 km), the flat-earth approximation is reasonable.

**No diffraction.** The model treats occlusion as a hard binary — a target is either visible or fully blocked. In reality, radar energy diffracts around terrain edges, so targets just behind a ridge may still be weakly detected. The current model will report them as fully occluded.

**Fixed step size.** The ray-march step of 50 m means narrow terrain features (walls, cliffs less than 50 m wide) might be stepped over entirely. The step size is configurable per-call but there is no adaptive stepping.

**No multipath.** Radar signals can reflect off water surfaces and terrain faces to reach targets via indirect paths. The model only considers the direct line-of-sight path.

**Single-reflectivity terrain.** All terrain surfaces use a fixed reflectivity of 0.85. Real terrain reflectivity varies significantly: bare rock, vegetation, wet soil, and urban structures all have different radar cross-sections.

**No Doppler or MTI.** Moving targets on or near terrain are not differentiated from static terrain clutter. A real radar would use Moving Target Indication to separate them.

**2-D coastlines are independent of 3-D terrain.** The coastline polygon system and the height-map terrain system are separate layers. A coastline polygon at the same location as a terrain island will produce two overlapping sets of returns merged via `max()`. For consistency, terrain features near the coast should replace the corresponding coastline polygons, or the coastline system should be extended to query terrain elevation.

---

## 7. Proposed Changes and Improvements

Changes are grouped by which research goal they serve. Items marked **(critical path)** must be completed to meet the stated goal; others improve quality or reduce error.

### Goal 1 path — SRTM/NASADEM + Literature Validation

#### 7.1 SRTM / NASADEM Loader **(critical path)**

Add `HeightMap.from_srtm(filepath, radar_lat, radar_lon)`:
1. Read `.hgt` file: 1201x1201 (3 arc-second) or 3601x3601 (1 arc-second) big-endian int16. Convert to float32 metres.
2. Parse tile name (e.g. `N33W118.hgt`) for SW corner lat/lon.
3. Convert lat/lon grid to local East/North metres relative to `(radar_lat, radar_lon)` using equirectangular projection: `dx = (lon - lon0) * 111320 * cos(lat0)`, `dy = (lat - lat0) * 110540`.
4. Return a HeightMap with correct origin and cell size (~30 m for 1 arc-second, ~90 m for 3 arc-second).

Add a NASADEM/GeoTIFF variant using `rasterio` for reading.

**Dependencies:** `numpy` (already required), `rasterio` (new optional dependency for GeoTIFF).

#### 7.2 Earth Curvature and Refraction Correction **(critical path)**

Replace all flat-earth `atan2(height_diff, range)` calls in `OcclusionEngine` with the standard 4/3 effective Earth radius model:

```
R_eff = 6371000 * (4/3)  # standard atmospheric refraction
earth_drop = range^2 / (2 * R_eff)
adjusted_elev = terrain_elev - earth_drop
angle = atan2(adjusted_elev - antenna_height, range)
```

Apply in: `is_target_occluded()`, `compute_occlusion_profile()`, `generate_terrain_returns()`, `get_occlusion_mask()`.

Without this, LOS predictions at ranges beyond ~10 NM will be wrong. With it, the model correctly accounts for radar horizon and atmospheric refraction, matching the propagation geometry used in ITU-R P.526 and standard VTS coverage analyses.

#### 7.3 Literature Scenario Replication

Select 2–3 published terrain-masking scenarios with documented geometry and expected shadow sectors. Candidates:
- An ITU-R P.526 worked example (knife-edge diffraction over a ridge with specified height, distance, and frequency).
- A VTS (Vessel Traffic Service) radar coverage study for a real port with published shadow maps.
- A coastal surveillance paper comparing measured vs. predicted radar coverage with terrain.

For each: load the SRTM tile, configure antenna position and height, run `get_occlusion_mask()` for all bearings, and compare quantitatively against the published result. Document per-bearing shadow boundary error in metres.

#### 7.4 Knife-Edge Diffraction (improves validation accuracy)

Replace the binary occluded/visible result with a diffraction attenuation factor using the Fresnel-Kirchhoff single-knife-edge model:

```
v = sqrt(2) * clearance / sqrt(wavelength * d1 * d2 / (d1 + d2))
loss_dB = J(v)  # Fresnel integral lookup
```

`is_target_occluded()` becomes `get_occlusion_loss(...)` returning 0.0 (clear) to 1.0 (fully blocked), with intermediate values in the penumbra. The detection engine multiplies received power by `(1 - loss)`. This reduces geometric error at shadow boundaries where the binary model overestimates blockage.

---

### Goal 2 path — Configurable Parameters + GUI

#### 7.5 JSON/YAML Scenario File Format **(critical path)**

Define a schema for scenario configuration:

```json
{
  "radar": {
    "frequency_ghz": 9.41,
    "peak_power_kw": 25.0,
    "beamwidth_deg": 1.2,
    "antenna_height_m": 15.0,
    "rotation_rpm": 24.0,
    "range_scale_nm": 6.0,
    "gain": 0.5
  },
  "terrain": [
    {"type": "srtm", "file": "N33W118.hgt", "radar_lat": 33.72, "radar_lon": -118.28}
  ],
  "vessels": [
    {"id": "target_1", "type": "cargo", "x": 3000, "y": 5000, "course": 225, "speed": 12, "length": 150, "height": 25}
  ],
  "weather": {
    "sea_state": 3, "wind_speed_knots": 15, "wind_direction": 45, "rain_rate_mmh": 0
  }
}
```

Add `Simulation.from_scenario_file(path)` that reads this and constructs a fully configured simulation. Add `Simulation.to_scenario_file(path)` for round-tripping.

#### 7.6 Radar Parameter Presets

Pre-built parameter sets for specific radar models:

| Preset name | Frequency | Power | Beamwidth | Antenna height |
|-------------|-----------|-------|-----------|----------------|
| `furuno_far2127` | 9.41 GHz | 25 kW | 1.2 deg | 15 m |
| `furuno_far2837s` | 9.41 GHz | 25 kW | 0.95 deg | 20 m |
| `furuno_drs25a` | 9.41 GHz | 25 kW | 1.8 deg | 12 m |

Loaded by name: `RadarParameters.from_preset("furuno_far2127")`.

#### 7.7 Scenario Editor GUI **(critical path)**

A desktop application (Tkinter or PyQt5) with:
- **Radar panel:** Sliders/entries for all `RadarParameters` fields. Preset dropdown.
- **Map panel:** Top-down view showing terrain contour overlay (from loaded SRTM), vessel icons with drag-to-place, coastline polygon editor.
- **Weather panel:** Sea state slider, wind direction/speed, rain rate.
- **Terrain panel:** File browser for SRTM tiles. Display loaded terrain extents on map.
- **Run panel:** "Simulate" button runs a full 360-degree sweep and displays a PPI image and shadow map. Export to CSV.

The GUI reads/writes the scenario JSON format from 7.5. A non-programmer can configure everything through the interface without touching Python code.

---

### Goal 3 path — Validation (<20% Mean Geometric Error)

#### 7.8 Real Radar Data Collection Protocol

For each of the 4–5 validation scenarios:
1. Record full 360-degree PPI sweep data using the X-band radar's built-in recording or the existing CSV export tool.
2. Record concurrent AIS data for vessel positions (ground truth for target returns).
3. Document: antenna position (lat/lon, height above sea level), radar settings (range scale, gain, clutter suppression), weather conditions (sea state, wind, rain), date/time.
4. Photograph or sketch the scenario for reference.

Target scenarios:
- **Coastal with terrain:** Harbour approach with islands or headlands blocking sectors.
- **Open water:** No terrain, vessels at various ranges (baseline without occlusion).
- **Harbour:** Close-range terrain on multiple sides, multiple vessels.
- **Mixed:** Partial terrain with some vessels in shadow and some visible.
- **Weather (optional):** Same location under different sea states.

#### 7.9 Scenario Reconstruction Pipeline

For each recorded scenario:
1. Load SRTM tile matching the location.
2. Place vessels at their AIS-recorded positions.
3. Set radar parameters to match the installation.
4. Set weather to match recorded conditions.
5. Run full 360-degree sweep simulation.
6. Export simulated sweep data to CSV.

#### 7.10 Error Metric Framework **(critical path)**

Implement a comparison module (`validation/compare.py`) that:

1. **Aligns** simulated and measured sweeps by bearing (accounting for any bearing offset between recording and simulation).
2. **Computes geometric error** per bearing: find the range of the first significant return (above a threshold) in both simulated and measured sweeps. The geometric error is `|R_sim - R_measured| / R_measured * 100%`. The mean geometric error across all bearings with terrain returns is the primary metric. Target: <20%.
3. **Computes intensity RMSE:** RMS of intensity differences across all range bins per bearing.
4. **Computes shadow boundary error:** For bearings at the edge of terrain shadows (transition from returns to no-returns), measure the angular width of the transition zone and the range offset of the shadow boundary.
5. **Produces a validation report** per scenario: tables, plots, pass/fail against the 20% threshold.

#### 7.11 Sensitivity Analysis

For each validated scenario, systematically vary one parameter at a time and re-run the comparison:

| Parameter | Variation range | Expected impact |
|-----------|----------------|-----------------|
| Antenna height | ±2 m from recorded | High — changes all shadow boundaries |
| Terrain cell size | 30 m vs. 90 m SRTM | Medium — affects fine terrain features |
| Reflectivity | 0.7 to 0.95 | Medium — affects terrain return intensity |
| Sea state | ±1 from recorded | Low for geometric, high for intensity |
| Ray-march step | 25 m, 50 m, 100 m | Low if terrain is smooth, high near cliffs |

Report the sensitivity as `Δerror / Δparameter` for each combination. Identify which parameters require the most careful calibration and which have negligible effect.

#### 7.12 Variable Terrain Reflectivity (improves intensity validation)

Add a per-cell reflectivity array to `HeightMap` (second numpy channel) using land-cover classification data:

| Surface | Reflectivity |
|---------|-------------|
| Bare rock / cliffs | 0.9 |
| Dense vegetation | 0.4 |
| Urban / buildings | 0.95 |
| Sand / beach | 0.3 |
| Wet soil | 0.7 |

This reduces intensity RMSE by matching the actual radar response of different terrain surfaces rather than treating all land as uniform 0.85.

---

### Goal 4 path — C++ Acceleration (>10,000 intersections/sec)

#### 7.13 C++ Ray-March Kernel **(critical path)**

Write `src/ray_march.cpp` with:

```cpp
struct RayResult {
    float max_elevation_angle;
    bool target_occluded;
};

RayResult ray_march_los(
    const float* grid, int rows, int cols,
    float origin_x, float origin_y, float cell_size,
    float antenna_height,
    float ray_ox, float ray_oy, float ray_dx, float ray_dy,
    float max_range, float step,
    float target_height, float target_range
);
```

The inner loop is: advance position by `(dx*step, dy*step)`, compute grid indices, bilinear interpolate (4 muls + 3 adds), compute elevation angle, update max. This is ~10 floating-point operations per step. At 200 steps per ray, ~2,000 flops per ray. A modern CPU at ~10 GFlops/s single-threaded gives ~5,000,000 rays/second — well above the 10,000 target.

#### 7.14 Batch API for Full-Sweep Ray-Casting

```cpp
void batch_ray_march(
    const float* grid, int rows, int cols,
    float origin_x, float origin_y, float cell_size,
    float antenna_height,
    float ray_ox, float ray_oy,
    const float* bearings, int num_bearings,
    float max_range, float step,
    bool* occlusion_mask_out,  // [num_bearings x num_bins]
    float* terrain_returns_out // [num_bearings x num_bins]
);
```

Process all bearings in a single call. Add `#pragma omp parallel for` over the bearing loop for multi-core acceleration.

#### 7.15 pybind11 Binding **(critical path)**

Expose the C++ functions to Python:

```python
from radar_sim._cpp_accel import ray_march_los, batch_ray_march

# Single LOS check
occluded = ray_march_los(grid_array, config, antenna_pos, target_pos)

# Full sweep
masks, returns = batch_ray_march(grid_array, config, antenna_pos, bearings_array)
```

The binding passes numpy arrays directly to C++ without copying (pybind11 `py::array_t<float>`).

#### 7.16 CMake Build and Fallback

```
src/
  ray_march.cpp
  bindings.cpp
  CMakeLists.txt
```

`CMakeLists.txt` uses pybind11 to build `_cpp_accel.so`. The Python `OcclusionEngine` checks at import time whether the C++ extension is available and uses it if present, otherwise falls back to the pure-Python implementation. The package installs and runs on any platform without requiring C++ compilation — the C++ module is an optional performance upgrade.

#### 7.17 Benchmark Harness **(critical path)**

`benchmarks/ray_throughput.py`:
1. Load a 1201x1201 SRTM grid (or generate a synthetic one).
2. Run 10,000 `ray_march_los()` calls with random bearings.
3. Measure wall-clock time.
4. Report: intersections/second, comparison to pure-Python, speedup factor.

Expected results:
- Python: ~1,000–2,000 intersections/sec
- C++ single-threaded: ~500,000+ intersections/sec
- C++ with OpenMP (4 cores): ~2,000,000+ intersections/sec

The 10,000 intersections/second threshold will be exceeded by C++ single-threaded by ~50×.

---

### Additional improvements (not on critical path)

#### 7.18 Terrain + Coastline Unification

Derive coastline outlines from the terrain height map (zero-elevation contour) instead of maintaining separate polygon coastlines. Eliminates inconsistent overlapping returns. Single data source drives both 2-D return shape and 3-D occlusion.

#### 7.19 Full 360-Degree Shadow Map Export

Add `OcclusionEngine.compute_full_shadow_map(origin_x, origin_y, max_range_m, num_bins, num_bearings=360)` returning a 2-D numpy array. Export to CSV and render to polar PNG. Directly produces "blind zone maps" for the poster presentation.

#### 7.20 CSV Sweep Export with Terrain Metadata

Re-integrate the `RadarDataExporter` from `furuno_radar_ppi`. Extend CSV headers with terrain parameters (number of maps, antenna height, occlusion active). Enables the existing `maritime_radar_sim` validation tools to ingest terrain-occluded data.

#### 7.21 Multipath Reflection Model

Add sea-surface reflection path for the "4-ray" interference pattern. Significant for low-altitude targets at medium range. Extends `is_target_occluded()` to check reflected-path geometry alongside direct path.

#### 7.22 Adaptive Ray-March Step Size

Reduce step size near terrain edges where elevation gradient is steep. Halve step when consecutive elevation change exceeds a threshold. Improves accuracy around cliffs without increasing computation over flat areas.

---

## 8. Implementation Priority Order

Sequencing the critical-path items to build toward all four goals:

| Phase | Items | Goals served | Outcome |
|-------|-------|-------------|---------|
| **Phase 1** | 7.1 SRTM loader, 7.2 Earth curvature | Goal 1 | Can load real terrain and get physically correct LOS results |
| **Phase 2** | 7.5 Scenario JSON format, 7.6 Presets | Goal 2 | Scenarios are serialisable and shareable |
| **Phase 3** | 7.3 Literature replication | Goal 1 | Baseline validation against published results |
| **Phase 4** | 7.13 C++ kernel, 7.15 pybind11, 7.17 Benchmark | Goal 4 | Performance target demonstrated |
| **Phase 5** | 7.8–7.11 Real data collection + validation | Goal 3 | <20% error demonstrated across 4–5 scenarios |
| **Phase 6** | 7.7 GUI editor | Goal 2 | Non-programmer scenario configuration |
| **Phase 7** | 7.4 Diffraction, 7.12 Reflectivity, 7.19 Shadow map export | Goals 1, 3 | Improved accuracy and visualisation for poster |

Phases 1–4 are prerequisite for the validation campaign (Phase 5). The GUI (Phase 6) can be developed in parallel. Phase 7 polishes results for the symposium poster.

---

*This document describes the system as of the initial commit. Update it as features from Section 7 are implemented.*
