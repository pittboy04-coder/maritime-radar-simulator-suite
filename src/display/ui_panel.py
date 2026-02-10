"""
Interactive control panel for radar simulation.
"""

import pygame
from typing import Optional
from src.display.widgets import Slider, Button, Dropdown, Label
from src.display.colors import RadarColors
from src.radar.radar_system import RadarSystem
from src.core.world import World


class UIPanel:
    """
    Interactive UI control panel.

    Provides sliders and controls for real-time parameter adjustment.
    """

    def __init__(
        self,
        position: tuple,
        width: int,
        height: int,
        radar: RadarSystem,
        world: World,
    ):
        """
        Initialize UI panel.

        Args:
            position: (x, y) position of panel
            width: Panel width
            height: Panel height
            radar: Radar system to control
            world: World object
        """
        self.position = position
        self.width = width
        self.height = height
        self.radar = radar
        self.world = world

        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

        # Create widgets
        self.widgets = []
        self._create_widgets()

        # Statistics labels
        self.vessel_count_label = None
        self.fps_label = None

    def _create_widgets(self):
        """Create all UI widgets."""
        x = self.position[0] + 10
        y_start = self.position[1] + 50
        y = y_start
        slider_width = self.width - 100
        slider_height = 20
        y_spacing = 50

        # Range slider
        range_slider = Slider(
            rect=pygame.Rect(x, y, slider_width, slider_height),
            min_value=1000,
            max_value=50000,
            initial_value=self.radar.get_max_range(),
            label="Range (m)",
            callback=lambda v: self.radar.set_max_range(v),
        )
        self.widgets.append(range_slider)
        y += y_spacing

        # Gain slider
        gain_slider = Slider(
            rect=pygame.Rect(x, y, slider_width, slider_height),
            min_value=0,
            max_value=40,
            initial_value=self.radar.get_gain(),
            label="Gain (dB)",
            callback=lambda v: self.radar.set_gain(v),
        )
        self.widgets.append(gain_slider)
        y += y_spacing

        # Rotation rate slider
        rpm_slider = Slider(
            rect=pygame.Rect(x, y, slider_width, slider_height),
            min_value=12,
            max_value=60,
            initial_value=self.radar.get_rotation_rate(),
            label="RPM",
            callback=lambda v: self.radar.set_rotation_rate(v),
        )
        self.widgets.append(rpm_slider)
        y += y_spacing

        # Sea state slider
        sea_state_slider = Slider(
            rect=pygame.Rect(x, y, slider_width, slider_height),
            min_value=0,
            max_value=9,
            initial_value=self.radar.get_sea_state(),
            label="Sea State",
            callback=lambda v: self.radar.set_sea_state(int(v)),
        )
        self.widgets.append(sea_state_slider)
        y += y_spacing

        # Rain slider
        rain_slider = Slider(
            rect=pygame.Rect(x, y, slider_width, slider_height),
            min_value=0,
            max_value=100,
            initial_value=self.radar.get_rain_rate(),
            label="Rain (mm/hr)",
            callback=lambda v: self.radar.set_rain_rate(v),
        )
        self.widgets.append(rain_slider)
        y += y_spacing + 20

        # Buttons section
        button_width = (self.width - 40) // 2
        button_height = 30

        # Add vessel button
        add_vessel_btn = Button(
            rect=pygame.Rect(x, y, button_width - 5, button_height),
            text="Add Vessel",
            callback=self._add_random_vessel,
        )
        self.widgets.append(add_vessel_btn)

        # Clear vessels button
        clear_btn = Button(
            rect=pygame.Rect(x + button_width + 5, y, button_width - 5, button_height),
            text="Clear All",
            callback=self._clear_vessels,
        )
        self.widgets.append(clear_btn)
        y += button_height + 10

        # Pause button (will be added separately in simulation)
        pause_btn = Button(
            rect=pygame.Rect(x, y, button_width - 5, button_height),
            text="Pause",
            callback=None,  # Set by simulation
        )
        self.widgets.append(pause_btn)
        self.pause_button = pause_btn

        # Reset button
        reset_btn = Button(
            rect=pygame.Rect(x + button_width + 5, y, button_width - 5, button_height),
            text="Reset",
            callback=self._reset_parameters,
        )
        self.widgets.append(reset_btn)
        y += button_height + 30

        # Statistics section
        stats_y = y
        self.vessel_count_label = Label(
            rect=pygame.Rect(x, stats_y, self.width - 20, 20),
            text="Vessels: 0",
            font_size=18,
        )
        self.widgets.append(self.vessel_count_label)
        stats_y += 25

        self.fps_label = Label(
            rect=pygame.Rect(x, stats_y, self.width - 20, 20),
            text="FPS: 0",
            font_size=18,
        )
        self.widgets.append(self.fps_label)

    def _add_random_vessel(self):
        """Add a random vessel to the world."""
        import numpy as np
        from src.objects.vessel import Vessel, VESSEL_TYPES

        # Random position within radar range
        angle = np.random.uniform(0, 360)
        distance = np.random.uniform(2000, self.radar.get_max_range() * 0.8)
        x = distance * np.sin(np.radians(angle))
        y = distance * np.cos(np.radians(angle))

        # Random velocity
        speed = np.random.uniform(5, 20)
        heading = np.random.uniform(0, 360)

        # Random vessel type
        vessel_type = np.random.choice(list(VESSEL_TYPES.keys()))
        rcs = VESSEL_TYPES[vessel_type].typical_rcs

        vessel = Vessel(
            position=(x, y),
            velocity=(speed, heading),
            rcs=rcs,
            vessel_type=vessel_type,
        )
        self.world.add_vessel(vessel)
        print(f"Added {vessel_type} at ({x:.0f}, {y:.0f})")

    def _clear_vessels(self):
        """Clear all vessels from the world."""
        self.world.clear_vessels()
        print("Cleared all vessels")

    def _reset_parameters(self):
        """Reset radar parameters to defaults."""
        from src.radar.parameters import RadarParameters

        defaults = RadarParameters()
        self.radar.set_max_range(defaults.max_range)
        self.radar.set_gain(defaults.gain)
        self.radar.set_rotation_rate(defaults.rotation_rate)
        self.radar.set_sea_state(3)
        self.radar.set_rain_rate(0.0)

        # Update sliders
        for widget in self.widgets:
            if isinstance(widget, Slider):
                if widget.label == "Range (m)":
                    widget.set_value(defaults.max_range)
                elif widget.label == "Gain (dB)":
                    widget.set_value(defaults.gain)
                elif widget.label == "RPM":
                    widget.set_value(defaults.rotation_rate)
                elif widget.label == "Sea State":
                    widget.set_value(3)
                elif widget.label == "Rain (mm/hr)":
                    widget.set_value(0.0)

        print("Reset parameters to defaults")

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame event.

        Args:
            event: Pygame event

        Returns:
            True if event was handled by a widget
        """
        for widget in self.widgets:
            if widget.handle_event(event):
                return True
        return False

    def update(self):
        """Update panel state."""
        for widget in self.widgets:
            widget.update()

    def update_statistics(self, vessel_count: int, fps: float):
        """
        Update statistics display.

        Args:
            vessel_count: Number of vessels
            fps: Frames per second
        """
        if self.vessel_count_label:
            self.vessel_count_label.set_text(f"Vessels: {vessel_count}")
        if self.fps_label:
            self.fps_label.set_text(f"FPS: {fps:.0f}")

    def draw(self, surface: pygame.Surface):
        """
        Draw panel.

        Args:
            surface: Surface to draw on
        """
        # Draw panel background
        panel_rect = pygame.Rect(self.position[0], self.position[1], self.width, self.height)
        pygame.draw.rect(surface, RadarColors.UI_BACKGROUND, panel_rect)
        pygame.draw.rect(surface, RadarColors.UI_BORDER, panel_rect, 2)

        # Draw title
        title = self.font.render("CONTROLS", True, RadarColors.UI_TEXT)
        title_rect = title.get_rect(centerx=self.position[0] + self.width // 2, top=self.position[1] + 10)
        surface.blit(title, title_rect)

        # Draw statistics section separator
        separator_y = self.position[1] + 400
        pygame.draw.line(
            surface,
            RadarColors.UI_BORDER,
            (self.position[0] + 10, separator_y),
            (self.position[0] + self.width - 10, separator_y),
            1,
        )

        stats_title = self.font_small.render("STATISTICS", True, RadarColors.UI_TEXT)
        surface.blit(stats_title, (self.position[0] + 10, separator_y + 5))

        # Draw all widgets
        for widget in self.widgets:
            widget.draw(surface)

    def get_panel_rect(self) -> pygame.Rect:
        """Get panel rectangle."""
        return pygame.Rect(self.position[0], self.position[1], self.width, self.height)
