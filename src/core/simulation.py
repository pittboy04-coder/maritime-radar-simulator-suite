"""
Main simulation orchestrator for maritime radar simulation.
"""

import pygame
from typing import Optional
from src.core.world import World
from src.radar.radar_system import RadarSystem
from src.radar.parameters import RadarParameters, EnvironmentParameters, DisplayParameters
from src.display.ppi_display import PPIDisplay
from src.environment.clutter import SeaClutterGenerator
from src.environment.weather import WeatherEffects


class RadarSimulation:
    """
    Main simulation class orchestrating all components.
    """

    def __init__(
        self,
        world: World = None,
        radar_params: RadarParameters = None,
        env_params: EnvironmentParameters = None,
        display_params: DisplayParameters = None,
    ):
        """
        Initialize radar simulation.

        Args:
            world: World object with vessels (creates empty if None)
            radar_params: Radar parameters (uses defaults if None)
            env_params: Environment parameters (uses defaults if None)
            display_params: Display parameters (uses defaults if None)
        """
        # Initialize parameters
        self.radar_params = radar_params if radar_params else RadarParameters()
        self.env_params = env_params if env_params else EnvironmentParameters()
        self.display_params = display_params if display_params else DisplayParameters()

        # Initialize world
        self.world = world if world else World()

        # Initialize radar system
        self.radar = RadarSystem(self.radar_params, self.env_params)

        # Initialize environmental effects
        self.clutter_gen = SeaClutterGenerator(self.env_params)
        self.weather = WeatherEffects(self.radar_params, self.env_params)

        # Initialize display
        self.display = PPIDisplay(self.radar_params, self.display_params)

        # Simulation state
        self.running = False
        self.paused = False
        self.time_scale = 1.0  # 1x, 2x, 5x, 10x speed

        # Physics time step
        self.physics_dt = 0.1  # seconds

    def handle_input(self) -> bool:
        """
        Handle user input.

        Returns:
            False if quit requested, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                # Quit
                if event.key == pygame.K_ESCAPE:
                    return False

                # Pause/Resume
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                # Range adjustment (zoom)
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.display.handle_zoom(zoom_in=True)
                    self.radar.set_max_range(self.display.radar_params.max_range)
                elif event.key == pygame.K_MINUS:
                    self.display.handle_zoom(zoom_in=False)
                    self.radar.set_max_range(self.display.radar_params.max_range)

                # Gain adjustment
                elif event.key == pygame.K_UP:
                    new_gain = self.radar.get_gain() + 1.0
                    self.radar.set_gain(new_gain)
                elif event.key == pygame.K_DOWN:
                    new_gain = self.radar.get_gain() - 1.0
                    self.radar.set_gain(new_gain)

                # Rotation rate adjustment
                elif event.key == pygame.K_LEFT:
                    new_rpm = self.radar.get_rotation_rate() - 3
                    self.radar.set_rotation_rate(new_rpm)
                elif event.key == pygame.K_RIGHT:
                    new_rpm = self.radar.get_rotation_rate() + 3
                    self.radar.set_rotation_rate(new_rpm)

                # Sea state adjustment
                elif event.key == pygame.K_0:
                    self.radar.set_sea_state(0)
                    self.clutter_gen.update_environment(self.radar.env_params)
                elif event.key == pygame.K_1:
                    self.radar.set_sea_state(1)
                    self.clutter_gen.update_environment(self.radar.env_params)
                elif event.key == pygame.K_2:
                    self.radar.set_sea_state(2)
                    self.clutter_gen.update_environment(self.radar.env_params)
                elif event.key == pygame.K_3:
                    self.radar.set_sea_state(3)
                    self.clutter_gen.update_environment(self.radar.env_params)
                elif event.key == pygame.K_4:
                    self.radar.set_sea_state(4)
                    self.clutter_gen.update_environment(self.radar.env_params)
                elif event.key == pygame.K_5:
                    self.radar.set_sea_state(5)
                    self.clutter_gen.update_environment(self.radar.env_params)
                elif event.key == pygame.K_6:
                    self.radar.set_sea_state(6)
                    self.clutter_gen.update_environment(self.radar.env_params)
                elif event.key == pygame.K_7:
                    self.radar.set_sea_state(7)
                    self.clutter_gen.update_environment(self.radar.env_params)
                elif event.key == pygame.K_8:
                    self.radar.set_sea_state(8)
                    self.clutter_gen.update_environment(self.radar.env_params)
                elif event.key == pygame.K_9:
                    self.radar.set_sea_state(9)
                    self.clutter_gen.update_environment(self.radar.env_params)

                # Rain toggle
                elif event.key == pygame.K_r:
                    if self.radar.get_rain_rate() > 0:
                        self.radar.set_rain_rate(0.0)
                    else:
                        self.radar.set_rain_rate(25.0)  # Heavy rain
                    self.weather.update_environment(self.radar.env_params)

                # Clear vessels
                elif event.key == pygame.K_c:
                    self.world.clear_vessels()

                # Help (to be implemented)
                elif event.key == pygame.K_h:
                    print("Keyboard controls:")
                    print("  ESC: Quit")
                    print("  SPACE: Pause/Resume")
                    print("  +/-: Zoom in/out")
                    print("  Up/Down: Adjust gain")
                    print("  Left/Right: Adjust rotation rate")
                    print("  0-9: Set sea state")
                    print("  R: Toggle rain")
                    print("  C: Clear all vessels")
                    print("  H: Show this help")

        return True

    def update(self, dt: float):
        """
        Update simulation state.

        Args:
            dt: Time step in seconds
        """
        if self.paused:
            return

        # Apply time scale
        scaled_dt = dt * self.time_scale

        # Update world (vessel positions)
        self.world.update(scaled_dt)

        # Update radar (antenna rotation)
        self.radar.update(scaled_dt)

        # Calculate environmental effects
        # For simplicity, use average effects across display range
        avg_range = self.radar.get_max_range() / 2.0
        clutter_db = self.clutter_gen.get_average_clutter_power(avg_range)
        weather_loss_db = self.weather.get_total_weather_loss(avg_range)

        # Update radar with environmental effects
        self.radar.set_environmental_effects(clutter_db, weather_loss_db)

    def render(self):
        """Render current frame."""
        # Perform radar scan
        detections = self.radar.scan(self.world)

        # Prepare statistics
        stats = {
            "vessel_count": self.world.get_vessel_count(),
            "fps": self.display.get_fps(),
            "sea_state": self.radar.get_sea_state(),
            "rain_rate": self.radar.get_rain_rate(),
        }

        # Render display
        self.display.render(
            detections=detections,
            antenna_azimuth=self.radar.get_azimuth(),
            stats=stats,
        )

    def run(self):
        """Run main simulation loop."""
        self.running = True
        clock = pygame.time.Clock()

        print("Maritime Radar Simulation started.")
        print("Press H for help.")

        while self.running:
            # Handle input
            if not self.handle_input():
                break

            # Update (fixed time step)
            self.update(self.physics_dt)

            # Render
            self.render()

            # Maintain frame rate
            clock.tick(self.display_params.target_fps)

        # Cleanup
        self.display.cleanup()
        print("Simulation ended.")

    def add_vessel(self, vessel):
        """Add a vessel to the simulation."""
        self.world.add_vessel(vessel)

    def set_time_scale(self, scale: float):
        """Set simulation time scale (speed multiplier)."""
        self.time_scale = max(0.1, min(10.0, scale))
