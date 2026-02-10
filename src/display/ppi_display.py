"""
PPI (Plan Position Indicator) display for maritime radar simulation.
"""

import pygame
import numpy as np
from typing import List, Tuple
from src.radar.parameters import RadarParameters, DisplayParameters
from src.radar.detection import Detection
from src.display.colors import RadarColors
from src.geometry.range_bearing import polar_to_cartesian


class PPIDisplay:
    """
    Plan Position Indicator (PPI) radar display.

    Circular radar display showing range and bearing to targets.
    """

    def __init__(
        self,
        radar_params: RadarParameters,
        display_params: DisplayParameters,
        window_size: Tuple[int, int] = None,
    ):
        """
        Initialize PPI display.

        Args:
            radar_params: Radar parameters
            display_params: Display parameters
            window_size: Window size (width, height) - overrides display_params if provided
        """
        pygame.init()

        self.radar_params = radar_params
        self.display_params = display_params

        # Window setup
        if window_size:
            self.window_width, self.window_height = window_size
        else:
            self.window_width = display_params.window_width
            self.window_height = display_params.window_height

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Maritime Radar Simulation")

        # PPI display parameters
        self.ppi_size = display_params.ppi_size
        self.ppi_center = display_params.get_ppi_center()
        self.ppi_radius = self.ppi_size // 2

        # Create surfaces
        self.main_surface = pygame.Surface((self.window_width, self.window_height))
        self.persistence_surface = pygame.Surface(
            (self.ppi_size, self.ppi_size), pygame.SRCALPHA
        )

        # Colors
        self.colors = RadarColors

        # Font for UI text
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

        # Clock for FPS
        self.clock = pygame.time.Clock()

        # Scale factor (meters per pixel)
        self.update_scale()

    def update_scale(self):
        """Update scale factor based on current max range."""
        # Scale: max_range should fit in radius
        self.scale = self.radar_params.max_range / self.ppi_radius

    def meters_to_pixels(self, range_m: float) -> float:
        """Convert range in meters to pixels."""
        return range_m / self.scale

    def polar_to_screen(self, range_m: float, bearing_deg: float) -> Tuple[int, int]:
        """
        Convert polar coordinates (range, bearing) to screen coordinates.

        Args:
            range_m: Range in meters
            bearing_deg: Bearing in degrees (0=North, 90=East)

        Returns:
            Screen coordinates (x, y)
        """
        # Convert to Cartesian (x=East, y=North)
        x, y = polar_to_cartesian(range_m, bearing_deg)

        # Convert to pixels
        x_pixels = x / self.scale
        y_pixels = -y / self.scale  # Negative because screen y increases downward

        # Translate to screen coordinates
        screen_x = int(self.ppi_center[0] + x_pixels)
        screen_y = int(self.ppi_center[1] + y_pixels)

        return (screen_x, screen_y)

    def draw_range_rings(self):
        """Draw range rings on the display."""
        if not self.display_params.show_range_rings:
            return

        interval = self.display_params.range_ring_interval
        max_range = self.radar_params.max_range

        current_range = interval
        while current_range <= max_range:
            radius_pixels = int(self.meters_to_pixels(current_range))

            # Draw ring
            pygame.draw.circle(
                self.main_surface,
                self.colors.RANGE_RING,
                self.ppi_center,
                radius_pixels,
                1,
            )

            # Draw range label
            label_text = f"{current_range / 1000:.1f}km"
            label = self.font_small.render(label_text, True, self.colors.RANGE_RING)
            label_pos = (self.ppi_center[0] + 5, self.ppi_center[1] - radius_pixels - 15)
            self.main_surface.blit(label, label_pos)

            current_range += interval

    def draw_bearing_markers(self):
        """Draw bearing markers (radial lines) on the display."""
        if not self.display_params.show_bearing_markers:
            return

        interval = self.display_params.bearing_marker_interval

        for bearing in range(0, 360, int(interval)):
            # Calculate end point of bearing line
            end_x, end_y = self.polar_to_screen(self.radar_params.max_range, bearing)

            # Draw line from center to edge
            pygame.draw.line(
                self.main_surface,
                self.colors.BEARING_MARKER,
                self.ppi_center,
                (end_x, end_y),
                1,
            )

            # Draw bearing label
            if bearing % 90 == 0:  # Only label cardinal directions
                labels = {0: "N", 90: "E", 180: "S", 270: "W"}
                label_text = labels.get(bearing, str(bearing))
                label = self.font.render(label_text, True, self.colors.UI_TEXT)

                # Position label slightly outside circle
                label_range = self.radar_params.max_range * 1.1
                label_x, label_y = self.polar_to_screen(label_range, bearing)
                label_rect = label.get_rect(center=(label_x, label_y))
                self.main_surface.blit(label, label_rect)

    def draw_sweep_line(self, azimuth: float):
        """
        Draw rotating sweep line.

        Args:
            azimuth: Current antenna azimuth in degrees
        """
        # Main sweep line
        end_x, end_y = self.polar_to_screen(self.radar_params.max_range, azimuth)

        pygame.draw.line(
            self.main_surface, self.colors.SWEEP, self.ppi_center, (end_x, end_y), 2
        )

        # Glow effect (draw additional lines with lower alpha nearby)
        for offset in [-0.5, 0.5]:
            glow_azimuth = azimuth + offset
            glow_x, glow_y = self.polar_to_screen(
                self.radar_params.max_range, glow_azimuth
            )
            glow_surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
            pygame.draw.line(
                glow_surface,
                (*self.colors.SWEEP_GLOW, 128),
                self.ppi_center,
                (glow_x, glow_y),
                1,
            )
            self.main_surface.blit(glow_surface, (0, 0))

    def update_persistence(self, detections: List[Detection]):
        """
        Update persistence surface with new detections.

        Args:
            detections: List of current detections
        """
        # Fade existing persistence
        fade_alpha = int(255 * self.display_params.trail_fade_rate)
        fade_surface = pygame.Surface((self.ppi_size, self.ppi_size), pygame.SRCALPHA)
        fade_surface.fill((0, 0, 0, fade_alpha))
        self.persistence_surface.blit(fade_surface, (0, 0))

        # Draw new detections on persistence surface
        for detection in detections:
            # Convert to screen coordinates relative to PPI surface
            screen_pos = self.polar_to_screen(detection.range_m, detection.bearing_deg)

            # Convert to persistence surface coordinates
            persist_x = screen_pos[0] - (self.ppi_center[0] - self.ppi_radius)
            persist_y = screen_pos[1] - (self.ppi_center[1] - self.ppi_radius)

            # Draw bright pixel
            if 0 <= persist_x < self.ppi_size and 0 <= persist_y < self.ppi_size:
                # Draw small circle for better visibility
                pygame.draw.circle(
                    self.persistence_surface,
                    self.colors.DETECTION,
                    (int(persist_x), int(persist_y)),
                    2,
                )

    def draw_ui_overlay(self, stats: dict = None):
        """
        Draw UI overlay with statistics and parameters.

        Args:
            stats: Dictionary of statistics to display
        """
        if stats is None:
            stats = {}

        # UI panel background (right side)
        ui_x = self.ppi_center[0] + self.ppi_radius + 20
        ui_width = self.window_width - ui_x - 20
        ui_panel = pygame.Rect(ui_x, 20, ui_width, 300)
        pygame.draw.rect(self.main_surface, self.colors.UI_BACKGROUND, ui_panel)
        pygame.draw.rect(self.main_surface, self.colors.UI_BORDER, ui_panel, 2)

        # Draw statistics
        y_offset = 40
        line_height = 25

        # Title
        title = self.font.render("RADAR STATUS", True, self.colors.UI_TEXT)
        self.main_surface.blit(title, (ui_x + 10, y_offset))
        y_offset += line_height + 10

        # Parameters and stats
        info_lines = [
            f"Range: {self.radar_params.max_range / 1000:.1f} km",
            f"Gain: {self.radar_params.gain:.0f} dB",
            f"RPM: {self.radar_params.rotation_rate:.0f}",
            f"Sea State: {stats.get('sea_state', 0)}",
            f"Rain: {stats.get('rain_rate', 0):.1f} mm/hr",
            "",
            f"Vessels: {stats.get('vessel_count', 0)}",
            f"FPS: {stats.get('fps', 0):.0f}",
        ]

        for line in info_lines:
            if line:  # Skip empty lines
                text = self.font_small.render(line, True, self.colors.UI_TEXT)
                self.main_surface.blit(text, (ui_x + 10, y_offset))
            y_offset += line_height

    def render(
        self,
        detections: List[Detection],
        antenna_azimuth: float,
        stats: dict = None,
    ):
        """
        Render complete frame.

        Args:
            detections: List of current detections
            antenna_azimuth: Current antenna azimuth in degrees
            stats: Statistics dictionary
        """
        # Clear main surface
        self.main_surface.fill(self.colors.BACKGROUND)

        # Draw range rings and bearing markers
        self.draw_range_rings()
        self.draw_bearing_markers()

        # Update and draw persistence (trails)
        self.update_persistence(detections)

        # Blit persistence surface onto main surface
        persist_pos = (
            self.ppi_center[0] - self.ppi_radius,
            self.ppi_center[1] - self.ppi_radius,
        )
        self.main_surface.blit(self.persistence_surface, persist_pos)

        # Draw sweep line on top
        self.draw_sweep_line(antenna_azimuth)

        # Draw UI overlay
        self.draw_ui_overlay(stats)

        # Blit to screen
        self.screen.blit(self.main_surface, (0, 0))
        pygame.display.flip()

        # Maintain frame rate
        self.clock.tick(self.display_params.target_fps)

    def get_fps(self) -> float:
        """Get current frames per second."""
        return self.clock.get_fps()

    def handle_zoom(self, zoom_in: bool):
        """
        Handle zoom in/out.

        Args:
            zoom_in: True to zoom in, False to zoom out
        """
        if zoom_in:
            self.radar_params.max_range *= 0.8
        else:
            self.radar_params.max_range *= 1.25

        # Clamp range
        self.radar_params.max_range = max(
            1000.0, min(50000.0, self.radar_params.max_range)
        )

        self.update_scale()

    def cleanup(self):
        """Clean up Pygame resources."""
        pygame.quit()
