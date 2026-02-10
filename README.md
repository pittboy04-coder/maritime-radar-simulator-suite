# Maritime Radar Simulation

A simplified geometric simulation for maritime radar with interactive visualization, object detection, and environmental effects.

## Features

- **Realistic PPI Display**: Classic circular Plan Position Indicator with rotating sweep and fading trails
- **Vessel Detection**: Geometric radar detection based on range, bearing, and radar cross-section
- **Environmental Effects**:
  - Sea clutter (randomized, depends on sea state)
  - Rain attenuation and precipitation clutter
  - Atmospheric effects
- **Interactive Controls**: Real-time adjustment of radar parameters
- **Multiple Vessel Types**: Yacht, patrol boat, fishing vessel, cargo ship, tanker, container ship

## Installation

1. Install Python dependencies:
```bash
cd maritime_radar_sim
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run with default settings:
```bash
python src/main.py
```

### Command Line Options

```bash
python src/main.py --range 30000 --gain 35 --rpm 30 --sea-state 5 --rain 15
```

Options:
- `--range METERS`: Maximum radar range in meters (default: 20000)
- `--gain DB`: Receiver gain in dB (default: 30.0)
- `--rpm RPM`: Antenna rotation rate in RPM (default: 24.0)
- `--sea-state 0-9`: Sea state Beaufort scale (default: 3)
- `--rain MM/HR`: Rain rate in mm/hr (default: 0.0)
- `--random N`: Create random scenario with N vessels

### Random Scenario

Create a simulation with 15 random vessels:
```bash
python src/main.py --random 15
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| **ESC** | Quit simulation |
| **SPACE** | Pause/Resume |
| **+/-** | Zoom in/out (adjust range) |
| **↑↓** | Increase/decrease gain |
| **←→** | Decrease/increase rotation rate |
| **0-9** | Set sea state (0=calm, 9=phenomenal) |
| **R** | Toggle rain (0 or 25 mm/hr) |
| **C** | Clear all vessels |
| **H** | Show help |

## Understanding the Display

### PPI Display Elements

- **Center**: Radar position (origin)
- **Green Rings**: Range rings (distance markers)
- **Radial Lines**: Bearing markers (N, E, S, W)
- **Rotating Line**: Radar antenna sweep
- **Bright Dots**: Detected vessels
- **Fading Trails**: Detection history (persistence)

### Right Panel

Displays current radar status:
- Range setting
- Receiver gain
- Antenna rotation rate (RPM)
- Sea state
- Rain rate
- Number of detected vessels
- Frames per second

## Testing Different Conditions

### 1. Test Detection Range

- Press `+` multiple times to zoom in (reduce range)
- Press `-` to zoom out (increase range)
- Observe vessels appearing/disappearing as range changes

### 2. Test Sea Clutter Effects

- Press `0` for calm seas (minimal clutter)
- Press `5` for rough seas (moderate clutter)
- Press `9` for phenomenal seas (extreme clutter)
- Notice increased "noise" and false alarms at higher sea states

### 3. Test Weather Impact

- Start with clear weather
- Press `R` to enable heavy rain (25 mm/hr)
- Observe reduced detection range
- Notice precipitation clutter returns

### 4. Test Gain Adjustment

- Press `↓` repeatedly to reduce gain
- Vessels at longer ranges disappear
- Press `↑` to increase gain
- Vessels reappear but more clutter

### 5. Test Rotation Rate

- Press `←` to slow rotation (12 RPM)
- Slower sweep, but better integration time
- Press `→` for faster rotation (60 RPM)
- Faster updates, but may miss weak targets

## Technical Details

### Coordinate System

- Origin (0, 0) at radar position
- North = +Y axis (0°)
- East = +X axis (90°)
- Bearing convention: 0° = North, clockwise

### Radar Equation

Simplified radar equation used for detection:

```
SNR = (Pt * G² * λ² * σ) / ((4π)³ * R⁴ * kTB * NF)
```

Where:
- Pt: Transmit power
- G: Antenna gain
- λ: Wavelength
- σ: Radar cross-section
- R: Range
- k: Boltzmann constant
- T: System temperature
- B: Bandwidth
- NF: Noise figure

Detection threshold: 13 dB (Pd ≈ 0.9, Pfa ≈ 10⁻⁶)

### Vessel Types

| Type | RCS (m²) | Typical Speed (m/s) | Length (m) |
|------|----------|---------------------|------------|
| Yacht | 10 | 5 | 15 |
| Patrol | 40 | 15 | 25 |
| Fishing | 80 | 8 | 30 |
| Cargo | 200 | 12 | 150 |
| Tanker | 500 | 10 | 250 |
| Container | 1000 | 13 | 300 |

### Sea State Scale

| State | Description | Wave Height |
|-------|-------------|-------------|
| 0 | Calm (glassy) | 0 m |
| 3 | Slight | 0.5-1.25 m |
| 5 | Rough | 2.5-4 m |
| 7 | High | 6-9 m |
| 9 | Phenomenal | >14 m |

## Project Structure

```
maritime_radar_sim/
├── src/
│   ├── geometry/       # Range/bearing calculations
│   ├── radar/          # Radar system components
│   ├── objects/        # Vessel classes and motion models
│   ├── environment/    # Sea clutter, weather effects
│   ├── display/        # PPI visualization
│   ├── core/           # Simulation orchestrator
│   └── main.py         # Entry point
├── config/             # Configuration files (future)
├── tests/              # Unit tests (future)
└── requirements.txt    # Python dependencies
```

## Future Enhancements

Planned features for future development:

- **UI Panel**: Sliders for real-time parameter adjustment
- **Scenario Manager**: Load/save preset scenarios
- **Mouse Interaction**: Click to select vessels, add new vessels
- **Statistics**: Track detection rates, false alarms
- **Advanced Features**: Target tracking, ARPA functions, guard zones

## License

This is a simulation for educational and research purposes.

## Acknowledgments

Based on maritime radar principles and geometric simulation techniques.
