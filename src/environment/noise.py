"""
Noise generation for radar simulation.
"""

import numpy as np
from scipy import ndimage


class NoiseGenerator:
    """
    Generates various types of radar noise.
    """

    def __init__(self, seed: int = None):
        """
        Initialize noise generator.

        Args:
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            np.random.seed(seed)

    def generate_thermal_noise(self, size: int = 1) -> np.ndarray:
        """
        Generate thermal noise (Gaussian/normal distribution).

        Args:
            size: Number of samples to generate

        Returns:
            Array of noise samples
        """
        return np.random.normal(0.0, 1.0, size)

    def generate_rayleigh_noise(self, scale: float = 1.0, size: int = 1) -> np.ndarray:
        """
        Generate Rayleigh-distributed noise (typical for sea clutter).

        Args:
            scale: Scale parameter (related to average amplitude)
            size: Number of samples

        Returns:
            Array of Rayleigh-distributed samples
        """
        return np.random.rayleigh(scale, size)

    def generate_noise_field(
        self, shape: tuple, correlation_length: float = 0.0
    ) -> np.ndarray:
        """
        Generate a 2D spatially-correlated noise field.

        Args:
            shape: Shape of output array (height, width)
            correlation_length: Spatial correlation length in pixels (0 = uncorrelated)

        Returns:
            2D array of spatially-correlated noise
        """
        # Generate white noise
        noise = np.random.normal(0.0, 1.0, shape)

        # Apply spatial correlation if requested
        if correlation_length > 0:
            # Use Gaussian filter for spatial correlation
            sigma = correlation_length
            noise = ndimage.gaussian_filter(noise, sigma)

        return noise

    def generate_speckle(self, mean: float = 0.0, variance: float = 1.0, size: int = 1) -> np.ndarray:
        """
        Generate speckle noise (multiplicative noise).

        Args:
            mean: Mean value
            variance: Variance
            size: Number of samples

        Returns:
            Array of speckle samples
        """
        noise = np.random.gamma(mean**2 / variance, variance / mean, size)
        return noise
