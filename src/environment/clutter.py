"""
Sea clutter generation for maritime radar simulation.
"""

import numpy as np
from scipy import ndimage
from src.radar.parameters import EnvironmentParameters
from src.environment.noise import NoiseGenerator


class SeaClutterGenerator:
    """
    Generates realistic sea clutter for maritime radar.

    Sea clutter characteristics:
    - Range-dependent (stronger at close range)
    - Depends on sea state (Beaufort scale)
    - Angular dependence (stronger upwind/downwind)
    - Statistical distribution (Rayleigh or K-distribution)
    - Spatially correlated
    """

    def __init__(self, env_params: EnvironmentParameters):
        """
        Initialize sea clutter generator.

        Args:
            env_params: Environment parameters
        """
        self.env_params = env_params
        self.noise_gen = NoiseGenerator()

    def get_clutter_scale(self, range_m: float) -> float:
        """
        Get clutter scale factor based on range and sea state.

        Clutter power varies as approximately 1/R³ for distributed clutter.

        Args:
            range_m: Range in meters

        Returns:
            Clutter scale factor
        """
        # Base scale from sea state (0-9)
        # Sea state 0: minimal clutter
        # Sea state 9: extreme clutter
        base_scale = 0.1 * (self.env_params.sea_state + 1)

        # Range dependence: clutter decreases with range
        # Use inverse cube law for distributed clutter
        if range_m > 100:
            range_factor = 1000.0 / range_m  # Normalize to 1km reference
        else:
            range_factor = 10.0  # Very high at close range

        scale = base_scale * range_factor

        return scale

    def generate_clutter_return(
        self, range_m: float, bearing_deg: float
    ) -> float:
        """
        Generate a single clutter return at given range and bearing.

        Uses Rayleigh distribution for amplitude.

        Args:
            range_m: Range in meters
            bearing_deg: Bearing in degrees

        Returns:
            Clutter power in dB relative to noise
        """
        # Get scale based on range and sea state
        scale = self.get_clutter_scale(range_m)

        # Angular dependence (simplified)
        # Clutter is typically stronger upwind/downwind
        wind_dir = self.env_params.wind_direction
        relative_bearing = abs(bearing_deg - wind_dir) % 180

        # Maximum at 0° and 180° (upwind/downwind)
        # Minimum at 90° (crosswind)
        angular_factor = 1.0 + 0.5 * np.cos(np.radians(2 * relative_bearing))

        scale *= angular_factor

        # Generate Rayleigh-distributed amplitude
        if scale > 0:
            amplitude = self.noise_gen.generate_rayleigh_noise(scale=scale, size=1)[0]

            # Convert to dB (relative to noise level = 0 dB)
            if amplitude > 0:
                clutter_db = 20 * np.log10(amplitude)
            else:
                clutter_db = -np.inf
        else:
            clutter_db = -np.inf

        return clutter_db

    def generate_clutter_field(
        self, shape: tuple, range_scale: float = 1000.0
    ) -> np.ndarray:
        """
        Generate a 2D spatially-correlated clutter field.

        Useful for visualization.

        Args:
            shape: Shape of output array (height, width)
            range_scale: Scale factor for range calculation

        Returns:
            2D array of clutter values
        """
        # Generate base Rayleigh noise
        base_scale = 0.1 * (self.env_params.sea_state + 1)
        clutter = self.noise_gen.generate_rayleigh_noise(
            scale=base_scale, size=shape[0] * shape[1]
        ).reshape(shape)

        # Apply spatial correlation
        correlation_length = 2.0 + self.env_params.sea_state * 0.5
        clutter = ndimage.gaussian_filter(clutter, sigma=correlation_length)

        # Apply range dependence (assuming center is close range)
        center_y, center_x = shape[0] // 2, shape[1] // 2
        y, x = np.ogrid[: shape[0], : shape[1]]
        distance_pixels = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # Convert pixel distance to range factor
        range_factor = 1.0 / (1.0 + distance_pixels / 50.0)
        clutter *= range_factor

        return clutter

    def get_average_clutter_power(self, range_m: float) -> float:
        """
        Get average clutter power at a given range.

        Args:
            range_m: Range in meters

        Returns:
            Average clutter power in dB
        """
        scale = self.get_clutter_scale(range_m)

        # For Rayleigh distribution: mean = scale * sqrt(π/2)
        mean_amplitude = scale * np.sqrt(np.pi / 2)

        if mean_amplitude > 0:
            clutter_db = 20 * np.log10(mean_amplitude)
        else:
            clutter_db = -np.inf

        return clutter_db

    def update_environment(self, env_params: EnvironmentParameters) -> None:
        """
        Update environment parameters.

        Args:
            env_params: New environment parameters
        """
        self.env_params = env_params

    def __repr__(self) -> str:
        return (
            f"SeaClutterGenerator(sea_state={self.env_params.sea_state}, "
            f"wind={self.env_params.wind_speed:.0f}kt @ {self.env_params.wind_direction:.0f}°)"
        )
