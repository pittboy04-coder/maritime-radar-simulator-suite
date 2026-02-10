"""
Reusable UI widgets for the radar simulation interface.
"""

import pygame
from typing import Callable, Optional, List, Tuple
from src.display.colors import RadarColors


class Widget:
    """Base widget class."""

    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.visible = True
        self.enabled = True

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame event.

        Returns:
            True if event was handled, False otherwise
        """
        return False

    def update(self):
        """Update widget state."""
        pass

    def draw(self, surface: pygame.Surface):
        """Draw widget on surface."""
        pass


class Slider(Widget):
    """Horizontal slider widget."""

    def __init__(
        self,
        rect: pygame.Rect,
        min_value: float,
        max_value: float,
        initial_value: float,
        label: str = "",
        callback: Optional[Callable[[float], None]] = None,
    ):
        """
        Initialize slider.

        Args:
            rect: Slider rectangle
            min_value: Minimum value
            max_value: Maximum value
            initial_value: Initial value
            label: Label text
            callback: Function called when value changes
        """
        super().__init__(rect)
        self.min_value = min_value
        self.max_value = max_value
        self.value = initial_value
        self.label = label
        self.callback = callback

        self.dragging = False
        self.hover = False

        self.font = pygame.font.Font(None, 18)

    def set_value(self, value: float):
        """Set slider value."""
        self.value = max(self.min_value, min(self.max_value, value))
        if self.callback:
            self.callback(self.value)

    def get_value(self) -> float:
        """Get current value."""
        return self.value

    def _get_handle_x(self) -> int:
        """Get handle x position."""
        # Calculate normalized position (0-1)
        if self.max_value > self.min_value:
            normalized = (self.value - self.min_value) / (self.max_value - self.min_value)
        else:
            normalized = 0

        # Convert to pixel position
        track_width = self.rect.width - 20
        x = self.rect.x + 10 + int(normalized * track_width)
        return x

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse events."""
        if not self.enabled or not self.visible:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check if clicked on slider
            handle_x = self._get_handle_x()
            handle_rect = pygame.Rect(handle_x - 5, self.rect.y, 10, self.rect.height)

            if handle_rect.collidepoint(event.pos) or self.rect.collidepoint(event.pos):
                self.dragging = True
                # Update value based on click position
                self._update_value_from_mouse(event.pos[0])
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self._update_value_from_mouse(event.pos[0])
                return True
            else:
                # Check hover
                self.hover = self.rect.collidepoint(event.pos)

        return False

    def _update_value_from_mouse(self, mouse_x: int):
        """Update value based on mouse x position."""
        # Calculate normalized position
        track_width = self.rect.width - 20
        relative_x = mouse_x - (self.rect.x + 10)
        normalized = max(0, min(1, relative_x / track_width))

        # Calculate value
        new_value = self.min_value + normalized * (self.max_value - self.min_value)
        self.set_value(new_value)

    def draw(self, surface: pygame.Surface):
        """Draw slider."""
        if not self.visible:
            return

        # Draw label
        if self.label:
            label_text = self.font.render(self.label, True, RadarColors.UI_TEXT)
            surface.blit(label_text, (self.rect.x, self.rect.y - 20))

        # Draw track
        track_rect = pygame.Rect(
            self.rect.x + 10, self.rect.y + self.rect.height // 2 - 2,
            self.rect.width - 20, 4
        )
        pygame.draw.rect(surface, RadarColors.SLIDER_BG, track_rect)

        # Draw handle
        handle_x = self._get_handle_x()
        handle_color = RadarColors.BUTTON_HOVER if self.hover or self.dragging else RadarColors.SLIDER_HANDLE
        pygame.draw.circle(
            surface, handle_color, (handle_x, self.rect.y + self.rect.height // 2), 6
        )

        # Draw value
        value_text = f"{self.value:.1f}"
        value_surface = self.font.render(value_text, True, RadarColors.UI_TEXT)
        surface.blit(value_surface, (self.rect.x + self.rect.width + 10, self.rect.y))


class Button(Widget):
    """Button widget."""

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        callback: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize button.

        Args:
            rect: Button rectangle
            text: Button text
            callback: Function called when clicked
        """
        super().__init__(rect)
        self.text = text
        self.callback = callback

        self.hover = False
        self.pressed = False

        self.font = pygame.font.Font(None, 20)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse events."""
        if not self.enabled or not self.visible:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.pressed and self.rect.collidepoint(event.pos):
                self.pressed = False
                if self.callback:
                    self.callback()
                return True
            self.pressed = False

        elif event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)

        return False

    def draw(self, surface: pygame.Surface):
        """Draw button."""
        if not self.visible:
            return

        # Choose color based on state
        if self.pressed:
            color = RadarColors.BUTTON_ACTIVE
        elif self.hover:
            color = RadarColors.BUTTON_HOVER
        else:
            color = RadarColors.BUTTON_NORMAL

        # Draw button background
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, RadarColors.UI_BORDER, self.rect, 2)

        # Draw text
        text_surface = self.font.render(self.text, True, RadarColors.UI_TEXT)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


class Dropdown(Widget):
    """Dropdown menu widget."""

    def __init__(
        self,
        rect: pygame.Rect,
        options: List[str],
        initial_index: int = 0,
        label: str = "",
        callback: Optional[Callable[[int, str], None]] = None,
    ):
        """
        Initialize dropdown.

        Args:
            rect: Dropdown rectangle
            options: List of option strings
            initial_index: Initial selected index
            label: Label text
            callback: Function called when selection changes (index, text)
        """
        super().__init__(rect)
        self.options = options
        self.selected_index = initial_index
        self.label = label
        self.callback = callback

        self.expanded = False
        self.hover = False
        self.hover_index = -1

        self.font = pygame.font.Font(None, 18)

    def get_selected(self) -> Tuple[int, str]:
        """Get selected option as (index, text)."""
        if 0 <= self.selected_index < len(self.options):
            return (self.selected_index, self.options[self.selected_index])
        return (0, "")

    def set_selected(self, index: int):
        """Set selected index."""
        if 0 <= index < len(self.options):
            self.selected_index = index
            if self.callback:
                self.callback(index, self.options[index])

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse events."""
        if not self.enabled or not self.visible:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                # Toggle expanded
                self.expanded = not self.expanded
                return True
            elif self.expanded:
                # Check if clicked on option
                for i, option_rect in enumerate(self._get_option_rects()):
                    if option_rect.collidepoint(event.pos):
                        self.set_selected(i)
                        self.expanded = False
                        return True
                # Clicked outside - collapse
                self.expanded = False
                return True

        elif event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)

            # Check hover on options
            if self.expanded:
                self.hover_index = -1
                for i, option_rect in enumerate(self._get_option_rects()):
                    if option_rect.collidepoint(event.pos):
                        self.hover_index = i
                        break

        return False

    def _get_option_rects(self) -> List[pygame.Rect]:
        """Get rectangles for dropdown options."""
        rects = []
        for i in range(len(self.options)):
            rect = pygame.Rect(
                self.rect.x,
                self.rect.y + self.rect.height * (i + 1),
                self.rect.width,
                self.rect.height,
            )
            rects.append(rect)
        return rects

    def draw(self, surface: pygame.Surface):
        """Draw dropdown."""
        if not self.visible:
            return

        # Draw label
        if self.label:
            label_text = self.font.render(self.label, True, RadarColors.UI_TEXT)
            surface.blit(label_text, (self.rect.x, self.rect.y - 20))

        # Draw main box
        color = RadarColors.BUTTON_HOVER if self.hover else RadarColors.BUTTON_NORMAL
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, RadarColors.UI_BORDER, self.rect, 2)

        # Draw selected option
        if 0 <= self.selected_index < len(self.options):
            text = self.options[self.selected_index]
            text_surface = self.font.render(text, True, RadarColors.UI_TEXT)
            text_rect = text_surface.get_rect(
                midleft=(self.rect.x + 5, self.rect.centery)
            )
            surface.blit(text_surface, text_rect)

        # Draw arrow
        arrow = "▼" if not self.expanded else "▲"
        arrow_surface = self.font.render(arrow, True, RadarColors.UI_TEXT)
        arrow_rect = arrow_surface.get_rect(
            midright=(self.rect.right - 5, self.rect.centery)
        )
        surface.blit(arrow_surface, arrow_rect)

        # Draw expanded options
        if self.expanded:
            for i, (option, option_rect) in enumerate(
                zip(self.options, self._get_option_rects())
            ):
                # Background
                bg_color = (
                    RadarColors.BUTTON_HOVER
                    if i == self.hover_index
                    else RadarColors.UI_BACKGROUND
                )
                pygame.draw.rect(surface, bg_color, option_rect)
                pygame.draw.rect(surface, RadarColors.UI_BORDER, option_rect, 1)

                # Text
                option_text = self.font.render(option, True, RadarColors.UI_TEXT)
                option_text_rect = option_text.get_rect(
                    midleft=(option_rect.x + 5, option_rect.centery)
                )
                surface.blit(option_text, option_text_rect)


class Label(Widget):
    """Text label widget."""

    def __init__(
        self, rect: pygame.Rect, text: str, font_size: int = 20, color: Tuple[int, int, int] = None
    ):
        """
        Initialize label.

        Args:
            rect: Label rectangle
            text: Label text
            font_size: Font size
            color: Text color (defaults to UI_TEXT)
        """
        super().__init__(rect)
        self.text = text
        self.font = pygame.font.Font(None, font_size)
        self.color = color if color else RadarColors.UI_TEXT

    def set_text(self, text: str):
        """Update label text."""
        self.text = text

    def draw(self, surface: pygame.Surface):
        """Draw label."""
        if not self.visible:
            return

        text_surface = self.font.render(self.text, True, self.color)
        text_rect = text_surface.get_rect(topleft=self.rect.topleft)
        surface.blit(text_surface, text_rect)
