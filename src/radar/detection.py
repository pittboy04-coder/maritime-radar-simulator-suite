"""
Radar detection engine using simplified radar equation.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from src.radar.parameters import RadarParameters, EnvironmentParameters
from src.radar.antenna import Antenna
from src.objects.vessel import Vessel
from src.geometry.range_bearing import calculate_range, calculate_bearing, check_horizon


@dataclass
class Detection:
    """Represents a radar detection."""

    vessel_id: int
    range_m: float  # Range in meters
    bearing_deg: float  # Bearing in degrees
    snr_db: float  # Signal-to-noise ratio in dB
    timestamp: float  # Simulation time of detection


class DetectionEngine:
    """
    Radar detection engine implementing simplified radar equation.

    Radar equation (simplified):
    SNR = (Pt * G^2 * λ^2 * σ) / ((4π)^3 * R^4 * k * T * B * NF)

    Where:
    - Pt: Transmit power (W)
    - G: Antenna gain (linear)
    - λ: Wavelength (m)
    - σ: Target radar cross-section (m²)
    - R: Range (m)
    - k: Boltzmann constant (1.38e-23 J/K)
    - T: System noise temperature (K)
    - B: Receiver bandwidth (Hz)
    - NF: Noise figure (linear)
    """

    def __init__(self, params: RadarParameters, env_params: EnvironmentParameters):
        """
        Initialize detection engine.

        Args:
            params: Radar parameters
            env_params: Environment parameters
        """
        self.params = params
        self.env_params = env_params

        # Physical constants
        self.k_boltzmann = 1.38e-23  # J/K
        self.system_temp = 290.0  # K (typical ~17°C)

        # Estimate receiver bandwidth from pulse width
        # B ≈ 1 / pulse_width
        self.bandwidth = 1.0 / (params.pulse_width * 1e-6)  # Hz

    def calculate_snr(
        self, vessel: Vessel, range_m: float, antenna: Antenna, bearing_deg: float
    ) -> float:
        """
        Calculate signal-to-noise ratio for a vessel.

        Args:
            vessel: Target vessel
            range_m: Range to vessel in meters
            antenna: Antenna object
            bearing_deg: Bearing to vessel in degrees

        Returns:
            SNR in dB
        """
        # Convert parameters to SI units
        pt_watts = self.params.transmit_power * 1000  # kW to W

        # Get antenna gain at target bearing (linear)
        g_linear = antenna.get_gain_factor_at_bearing(bearing_deg)

        # Wavelength
        wavelength = self.params.get_wavelength()

        # Target RCS
        sigma = vessel.rcs

        # Calculate received signal power (radar equation)
        numerator = pt_watts * (g_linear**2) * (wavelength**2) * sigma
        denominator = ((4 * np.pi) ** 3) * (range_m**4)

        signal_power = numerator / denominator

        # Calculate noise power
        # P_noise = k * T * B * NF
        nf_linear = 10 ** (self.params.noise_figure / 10.0)
        noise_power = (
            self.k_boltzmann * self.system_temp * self.bandwidth * nf_linear
        )

        # SNR (linear)
        if noise_power > 0:
            snr_linear = signal_power / noise_power
        else:
            snr_linear = 0.0

        # Convert to dB
        if snr_linear > 0:
            snr_db = 10 * np.log10(snr_linear)
        else:
            snr_db = -np.inf

        # Apply receiver gain
        snr_db += self.params.gain

        return snr_db

    def check_detection(
        self,
        vessel: Vessel,
        radar_pos: Tuple[float, float],
        antenna: Antenna,
        simulation_time: float,
        clutter_db: float = 0.0,
        weather_loss_db: float = 0.0,
    ) -> Optional[Detection]:
        """
        Check if a vessel is detected by the radar.

        Args:
            vessel: Target vessel
            radar_pos: Radar position (x, y) in meters
            antenna: Antenna object
            simulation_time: Current simulation time in seconds
            clutter_db: Clutter power in dB (optional)
            weather_loss_db: Weather attenuation in dB (optional)

        Returns:
            Detection object if detected, None otherwise
        """
        # Calculate range and bearing to vessel
        vessel_pos = vessel.get_position()
        range_m = calculate_range(radar_pos, vessel_pos)
        bearing_deg = calculate_bearing(radar_pos, vessel_pos)

        # Check if in range limits
        if range_m < self.params.min_range or range_m > self.params.max_range:
            return None

        # Check if target is in antenna beam
        if not antenna.is_target_in_beam(bearing_deg):
            return None

        # Check radar horizon
        if not check_horizon(range_m, self.params.antenna_height, vessel.height):
            return None

        # Calculate SNR
        snr_db = self.calculate_snr(vessel, range_m, antenna, bearing_deg)

        # Apply weather losses
        snr_db -= weather_loss_db

        # Account for clutter (add clutter to noise floor)
        # SNR_effective = Signal / (Noise + Clutter)
        # In dB: SNR_eff = SNR - 10*log10(1 + 10^(C/N / 10))
        if clutter_db > -np.inf:
            # Clutter-to-noise ratio
            cnr_db = clutter_db
            cnr_linear = 10 ** (cnr_db / 10.0)
            snr_degradation = 10 * np.log10(1 + cnr_linear)
            snr_db -= snr_degradation

        # Check detection threshold
        if snr_db >= self.params.detection_threshold:
            return Detection(
                vessel_id=vessel.id,
                range_m=range_m,
                bearing_deg=bearing_deg,
                snr_db=snr_db,
                timestamp=simulation_time,
            )

        return None

    def set_environment(self, env_params: EnvironmentParameters) -> None:
        """
        Update environment parameters.

        Args:
            env_params: New environment parameters
        """
        self.env_params = env_params

    def __repr__(self) -> str:
        return (
            f"DetectionEngine(threshold={self.params.detection_threshold:.1f} dB, "
            f"bandwidth={self.bandwidth / 1e6:.1f} MHz)"
        )
