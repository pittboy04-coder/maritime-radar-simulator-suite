"""
Weather effects for radar simulation (rain, fog, atmospheric attenuation).
"""

import numpy as np
from src.radar.parameters import EnvironmentParameters, RadarParameters
from src.environment.noise import NoiseGenerator


class WeatherEffects:
    """
    Models weather-induced radar effects.

    Primary effects:
    - Rain attenuation (frequency-dependent signal loss)
    - Precipitation clutter (volumetric returns from rain)
    - Fog effects (minimal at X-band frequencies)
    """

    def __init__(
        self, radar_params: RadarParameters, env_params: EnvironmentParameters
    ):
        """
        Initialize weather effects.

        Args:
            radar_params: Radar parameters
            env_params: Environment parameters
        """
        self.radar_params = radar_params
        self.env_params = env_params
        self.noise_gen = NoiseGenerator()

    def calculate_rain_attenuation(self, range_m: float) -> float:
        """
        Calculate two-way rain attenuation using ITU-R model.

        Attenuation: α = k * R^β (dB/km)
        where R is rain rate in mm/hr

        For X-band (9.4 GHz):
        k ≈ 0.0308, β ≈ 1.165

        Args:
            range_m: Range in meters (two-way path)

        Returns:
            Two-way attenuation in dB
        """
        rain_rate = self.env_params.rain_rate  # mm/hr

        if rain_rate <= 0:
            return 0.0

        # ITU-R parameters for X-band (~9-10 GHz)
        k = 0.0308
        beta = 1.165

        # Specific attenuation (dB/km)
        alpha_db_km = k * (rain_rate**beta)

        # Two-way path length
        path_length_km = (2 * range_m) / 1000.0

        # Total attenuation
        attenuation_db = alpha_db_km * path_length_km

        return attenuation_db

    def generate_rain_clutter(self, range_m: float) -> float:
        """
        Generate precipitation clutter return.

        Rain creates volumetric scattering that appears as clutter.

        Args:
            range_m: Range in meters

        Returns:
            Rain clutter power in dB
        """
        rain_rate = self.env_params.rain_rate

        if rain_rate <= 0:
            return -np.inf

        # Rain clutter depends on rain rate and range resolution
        # Simplified model: clutter increases with rain rate
        # Base clutter level
        base_clutter_db = 10 * np.log10(rain_rate) if rain_rate > 0 else -np.inf

        # Add randomness (Rayleigh distributed)
        noise_amplitude = self.noise_gen.generate_rayleigh_noise(scale=1.0, size=1)[0]
        noise_db = 20 * np.log10(noise_amplitude) if noise_amplitude > 0 else -np.inf

        clutter_db = base_clutter_db + noise_db

        # Rain clutter is relatively range-independent (volumetric)
        # but may decrease slightly at longer ranges
        if range_m > 10000:
            range_factor = 10000.0 / range_m
            clutter_db += 10 * np.log10(range_factor)

        return clutter_db

    def calculate_atmospheric_attenuation(self, range_m: float) -> float:
        """
        Calculate atmospheric attenuation (gases, water vapor).

        At X-band, atmospheric attenuation is typically small (~0.01 dB/km).

        Args:
            range_m: Range in meters

        Returns:
            Two-way atmospheric attenuation in dB
        """
        # Typical atmospheric attenuation at X-band
        # Depends on humidity, temperature, pressure
        # Simplified model: ~0.01-0.02 dB/km

        humidity_factor = self.env_params.humidity / 100.0  # 0-1
        attenuation_db_km = 0.01 + 0.01 * humidity_factor

        # Two-way path
        path_length_km = (2 * range_m) / 1000.0

        attenuation_db = attenuation_db_km * path_length_km

        return attenuation_db

    def calculate_fog_attenuation(self, range_m: float) -> float:
        """
        Calculate fog attenuation.

        At X-band (9.4 GHz), fog has minimal effect.
        Fog attenuation is significant only at millimeter wavelengths.

        Args:
            range_m: Range in meters

        Returns:
            Fog attenuation in dB (typically negligible)
        """
        # For X-band, fog attenuation is negligible
        # Typical: < 0.001 dB/km for heavy fog
        return 0.0

    def get_total_weather_loss(self, range_m: float) -> float:
        """
        Get total weather-induced signal loss.

        Args:
            range_m: Range in meters

        Returns:
            Total loss in dB
        """
        rain_loss = self.calculate_rain_attenuation(range_m)
        atmos_loss = self.calculate_atmospheric_attenuation(range_m)
        fog_loss = self.calculate_fog_attenuation(range_m)

        total_loss = rain_loss + atmos_loss + fog_loss

        return total_loss

    def get_precipitation_clutter(self, range_m: float) -> float:
        """
        Get precipitation clutter level.

        Args:
            range_m: Range in meters

        Returns:
            Clutter power in dB
        """
        return self.generate_rain_clutter(range_m)

    def update_environment(self, env_params: EnvironmentParameters) -> None:
        """
        Update environment parameters.

        Args:
            env_params: New environment parameters
        """
        self.env_params = env_params

    def __repr__(self) -> str:
        return (
            f"WeatherEffects(rain={self.env_params.rain_rate:.1f}mm/hr, "
            f"temp={self.env_params.temperature:.1f}°C, "
            f"humidity={self.env_params.humidity:.0f}%)"
        )
