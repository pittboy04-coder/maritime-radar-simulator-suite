"""
Radar antenna model with rotation and beam pattern.
"""

import numpy as np
from src.radar.parameters import RadarParameters
from src.geometry.range_bearing import normalize_bearing, bearing_difference


class Antenna:
    """
    Radar antenna with rotation and beam pattern characteristics.

    Tracks antenna azimuth and determines which targets are in the beam.
    """

    def __init__(self, params: RadarParameters):
        """
        Initialize antenna.

        Args:
            params: Radar parameters
        """
        self.params = params
        self.current_azimuth = 0.0  # degrees [0, 360)
        self.beam_width = params.beam_width  # degrees
        self.gain_db = params.antenna_gain  # dB

    def update(self, dt: float) -> None:
        """
        Update antenna rotation.

        Args:
            dt: Time step in seconds
        """
        # Rotate antenna based on angular velocity
        delta_angle = self.params.get_angular_velocity() * dt
        self.current_azimuth = (self.current_azimuth + delta_angle) % 360

    def set_azimuth(self, azimuth: float) -> None:
        """
        Set antenna azimuth directly (for testing).

        Args:
            azimuth: Azimuth angle in degrees
        """
        self.current_azimuth = normalize_bearing(azimuth)

    def get_azimuth(self) -> float:
        """Get current antenna azimuth in degrees."""
        return self.current_azimuth

    def is_target_in_beam(self, target_bearing: float) -> bool:
        """
        Check if a target at given bearing is within the antenna beam.

        Args:
            target_bearing: Bearing to target in degrees

        Returns:
            True if target is within beam width, False otherwise
        """
        diff = abs(bearing_difference(target_bearing, self.current_azimuth))
        return diff <= self.beam_width / 2.0

    def get_antenna_gain_at_angle(self, angle_off_boresight: float) -> float:
        """
        Get antenna gain at a specific angle off boresight.

        Simplified beam pattern:
        - Maximum gain at boresight (0°)
        - Decreases to -3dB at ±beam_width/2
        - Sidelobes approximated as constant -20dB beyond main beam

        Args:
            angle_off_boresight: Angle from beam center in degrees

        Returns:
            Gain in dB
        """
        abs_angle = abs(angle_off_boresight)

        # Main lobe: Gaussian-like pattern
        if abs_angle <= self.beam_width / 2.0:
            # -3dB at half-power beamwidth
            # Gaussian approximation: G(θ) = G_max * exp(-2.77 * (θ/θ_3dB)^2)
            factor = (abs_angle / (self.beam_width / 2.0)) ** 2
            gain_db = self.gain_db - 2.77 * factor

        # First sidelobes
        elif abs_angle <= self.beam_width * 2:
            gain_db = self.gain_db - 20.0  # -20dB sidelobes

        # Far sidelobes
        else:
            gain_db = self.gain_db - 30.0  # -30dB far sidelobes

        return gain_db

    def get_gain_factor_at_bearing(self, target_bearing: float) -> float:
        """
        Get antenna gain factor (linear, not dB) for target at given bearing.

        Args:
            target_bearing: Bearing to target in degrees

        Returns:
            Gain factor (linear)
        """
        angle_off = bearing_difference(target_bearing, self.current_azimuth)
        gain_db = self.get_antenna_gain_at_angle(angle_off)

        # Convert dB to linear
        return 10 ** (gain_db / 10.0)

    def get_rotation_rate(self) -> float:
        """Get antenna rotation rate in RPM."""
        return self.params.rotation_rate

    def set_rotation_rate(self, rpm: float) -> None:
        """
        Set antenna rotation rate.

        Args:
            rpm: Rotation rate in revolutions per minute
        """
        self.params.rotation_rate = max(1.0, min(60.0, rpm))  # Clamp to [1, 60]
        # Recalculate derived parameters
        self.params.__post_init__()

    def __repr__(self) -> str:
        return (
            f"Antenna(azimuth={self.current_azimuth:.1f}°, "
            f"beam_width={self.beam_width:.1f}°, "
            f"rotation_rate={self.params.rotation_rate:.0f} RPM)"
        )
