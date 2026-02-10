"""
Main radar system integrating antenna and detection engine.
"""

from typing import List, Tuple
from src.radar.parameters import RadarParameters, EnvironmentParameters
from src.radar.antenna import Antenna
from src.radar.detection import DetectionEngine, Detection
from src.core.world import World


class RadarSystem:
    """
    Complete radar system.

    Integrates antenna, detection engine, and provides scanning interface.
    """

    def __init__(
        self,
        params: RadarParameters = None,
        env_params: EnvironmentParameters = None,
        position: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        Initialize radar system.

        Args:
            params: Radar parameters (uses defaults if None)
            env_params: Environment parameters (uses defaults if None)
            position: Radar position (x, y) in meters
        """
        self.params = params if params is not None else RadarParameters()
        self.env_params = (
            env_params if env_params is not None else EnvironmentParameters()
        )
        self.position = position

        # Initialize components
        self.antenna = Antenna(self.params)
        self.detection_engine = DetectionEngine(self.params, self.env_params)

        # Environmental effects (will be set by environment modules)
        self.current_clutter_db = 0.0
        self.current_weather_loss_db = 0.0

    def update(self, dt: float) -> None:
        """
        Update radar system (primarily antenna rotation).

        Args:
            dt: Time step in seconds
        """
        self.antenna.update(dt)

    def scan(self, world: World) -> List[Detection]:
        """
        Perform radar scan on all vessels in the world.

        Returns detections for vessels currently in the antenna beam.

        Args:
            world: World containing vessels

        Returns:
            List of Detection objects
        """
        detections = []

        for vessel in world.get_all_vessels():
            detection = self.detection_engine.check_detection(
                vessel=vessel,
                radar_pos=self.position,
                antenna=self.antenna,
                simulation_time=world.simulation_time,
                clutter_db=self.current_clutter_db,
                weather_loss_db=self.current_weather_loss_db,
            )

            if detection is not None:
                detections.append(detection)

        return detections

    def set_environmental_effects(
        self, clutter_db: float = 0.0, weather_loss_db: float = 0.0
    ) -> None:
        """
        Set current environmental effects.

        Args:
            clutter_db: Clutter power in dB relative to noise
            weather_loss_db: Weather attenuation loss in dB
        """
        self.current_clutter_db = clutter_db
        self.current_weather_loss_db = weather_loss_db

    # Parameter adjustment methods (for UI controls)

    def set_max_range(self, range_m: float) -> None:
        """Set maximum detection range."""
        self.params.max_range = max(1000.0, min(50000.0, range_m))

    def get_max_range(self) -> float:
        """Get maximum detection range in meters."""
        return self.params.max_range

    def set_gain(self, gain_db: float) -> None:
        """Set receiver gain in dB."""
        self.params.gain = max(0.0, min(40.0, gain_db))

    def get_gain(self) -> float:
        """Get receiver gain in dB."""
        return self.params.gain

    def set_rotation_rate(self, rpm: float) -> None:
        """Set antenna rotation rate in RPM."""
        self.antenna.set_rotation_rate(rpm)

    def get_rotation_rate(self) -> float:
        """Get antenna rotation rate in RPM."""
        return self.antenna.get_rotation_rate()

    def get_azimuth(self) -> float:
        """Get current antenna azimuth in degrees."""
        return self.antenna.get_azimuth()

    def set_sea_state(self, sea_state: int) -> None:
        """
        Set sea state (0-9 Beaufort scale).

        Args:
            sea_state: Sea state value
        """
        self.env_params.sea_state = max(0, min(9, sea_state))

    def get_sea_state(self) -> int:
        """Get current sea state."""
        return self.env_params.sea_state

    def set_rain_rate(self, rain_mm_hr: float) -> None:
        """
        Set rain rate.

        Args:
            rain_mm_hr: Rain rate in mm/hr
        """
        self.env_params.rain_rate = max(0.0, min(100.0, rain_mm_hr))

    def get_rain_rate(self) -> float:
        """Get current rain rate in mm/hr."""
        return self.env_params.rain_rate

    def get_environment_params(self) -> EnvironmentParameters:
        """Get environment parameters."""
        return self.env_params

    def set_environment_params(self, env_params: EnvironmentParameters) -> None:
        """Set environment parameters."""
        self.env_params = env_params
        self.detection_engine.set_environment(env_params)

    def __repr__(self) -> str:
        return (
            f"RadarSystem(range={self.params.max_range / 1000:.1f}km, "
            f"azimuth={self.antenna.get_azimuth():.1f}°, "
            f"rpm={self.antenna.get_rotation_rate():.0f}, "
            f"sea_state={self.env_params.sea_state})"
        )
