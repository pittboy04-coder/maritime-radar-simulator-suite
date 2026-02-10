"""
Radar parameters and configuration.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class RadarParameters:
    """
    Configuration parameters for radar system.

    All units in SI unless otherwise specified.
    """

    # Range parameters
    max_range: float = 20000.0  # Maximum detection range (meters)
    min_range: float = 50.0  # Minimum range / blind zone (meters)

    # RF parameters
    frequency: float = 9.4  # Radar frequency (GHz) - X-band typical
    transmit_power: float = 25.0  # Transmit power (kW)
    pulse_width: float = 0.08  # Pulse duration (microseconds)

    # Antenna parameters
    rotation_rate: float = 24.0  # Antenna rotation rate (RPM)
    beam_width: float = 1.2  # Horizontal beam width (degrees)
    antenna_gain: float = 30.0  # Antenna gain (dB)

    # Receiver parameters
    gain: float = 30.0  # Receiver gain (dB) - adjustable
    noise_figure: float = 3.0  # Receiver noise figure (dB)

    # Detection parameters
    detection_threshold: float = 13.0  # SNR threshold for detection (dB)
    # Threshold of 13 dB gives Pd ≈ 0.9, Pfa ≈ 10^-6

    # Physical parameters
    antenna_height: float = 20.0  # Antenna height above sea level (meters)

    def __post_init__(self):
        """Calculate derived parameters."""
        # Wavelength from frequency
        c = 3e8  # Speed of light (m/s)
        self.wavelength = c / (self.frequency * 1e9)  # meters

        # Rotation period
        self.rotation_period = 60.0 / self.rotation_rate  # seconds per rotation

        # Angular velocity
        self.angular_velocity = 360.0 / self.rotation_period  # degrees per second

    def get_wavelength(self) -> float:
        """Get radar wavelength in meters."""
        return self.wavelength

    def get_rotation_period(self) -> float:
        """Get rotation period in seconds."""
        return self.rotation_period

    def get_angular_velocity(self) -> float:
        """Get angular velocity in degrees per second."""
        return self.angular_velocity


@dataclass
class EnvironmentParameters:
    """
    Environmental condition parameters.
    """

    # Sea state (Beaufort scale 0-9)
    sea_state: int = 3  # 0=calm, 3=moderate (1-1.5m waves), 9=phenomenal

    # Wind
    wind_speed: float = 15.0  # Wind speed (knots)
    wind_direction: float = 0.0  # Wind direction (degrees, 0=North)

    # Precipitation
    rain_rate: float = 0.0  # Rain rate (mm/hr), 0=no rain, 25=heavy rain, 100=extreme

    # Atmospheric conditions
    temperature: float = 15.0  # Temperature (Celsius)
    humidity: float = 75.0  # Relative humidity (%)
    atmospheric_pressure: float = 1013.25  # Atmospheric pressure (hPa)

    def get_sea_state_description(self) -> str:
        """Get descriptive text for sea state."""
        descriptions = {
            0: "Calm (glassy)",
            1: "Calm (rippled)",
            2: "Smooth (0.1-0.5m)",
            3: "Slight (0.5-1.25m)",
            4: "Moderate (1.25-2.5m)",
            5: "Rough (2.5-4m)",
            6: "Very Rough (4-6m)",
            7: "High (6-9m)",
            8: "Very High (9-14m)",
            9: "Phenomenal (>14m)",
        }
        return descriptions.get(self.sea_state, "Unknown")

    def get_rain_description(self) -> str:
        """Get descriptive text for rain rate."""
        if self.rain_rate == 0:
            return "No rain"
        elif self.rain_rate < 2.5:
            return "Light rain"
        elif self.rain_rate < 10:
            return "Moderate rain"
        elif self.rain_rate < 50:
            return "Heavy rain"
        else:
            return "Violent rain"


@dataclass
class DisplayParameters:
    """
    Display and visualization parameters.
    """

    # Window size
    window_width: int = 1200
    window_height: int = 1000

    # PPI display size (circular radar display)
    ppi_size: int = 800  # pixels (diameter)

    # Range rings
    range_ring_interval: float = 2000.0  # meters between range rings
    show_range_rings: bool = True

    # Bearing markers
    bearing_marker_interval: float = 30.0  # degrees between markers
    show_bearing_markers: bool = True

    # Trail persistence
    trail_persistence: float = 20.0  # seconds
    trail_fade_rate: float = 0.02  # fade factor per frame (0.02 = 98% retention)

    # Colors (RGB)
    background_color: tuple = (0, 0, 0)  # Black
    range_ring_color: tuple = (0, 80, 0)  # Dim green
    bearing_marker_color: tuple = (0, 80, 0)  # Dim green
    sweep_color: tuple = (0, 255, 0)  # Bright green
    detection_color: tuple = (0, 255, 0)  # Bright green
    ui_text_color: tuple = (0, 255, 0)  # Bright green

    # Frame rate
    target_fps: int = 60

    def get_ppi_center(self) -> tuple:
        """Get center position of PPI display."""
        # PPI display is on the left side
        center_x = self.ppi_size // 2 + 50  # 50 pixel margin
        center_y = self.window_height // 2
        return (center_x, center_y)
