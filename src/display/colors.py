"""
Color schemes and palettes for radar display.
"""

from typing import Dict, Tuple


class RadarColors:
    """Classic green radar color scheme."""

    # Background
    BACKGROUND = (0, 0, 0)  # Black

    # Range rings and bearing markers
    RANGE_RING = (0, 80, 0)  # Dim green
    BEARING_MARKER = (0, 80, 0)  # Dim green
    GRID = (0, 60, 0)  # Very dim green

    # Sweep and detections
    SWEEP = (0, 255, 0)  # Bright green
    SWEEP_GLOW = (0, 180, 0)  # Medium green
    DETECTION = (0, 255, 0)  # Bright green
    DETECTION_BRIGHT = (100, 255, 100)  # Very bright green

    # UI elements
    UI_TEXT = (0, 255, 0)  # Bright green
    UI_BACKGROUND = (0, 30, 0)  # Very dark green
    UI_BORDER = (0, 120, 0)  # Medium green

    # Buttons and controls
    BUTTON_NORMAL = (0, 100, 0)
    BUTTON_HOVER = (0, 150, 0)
    BUTTON_ACTIVE = (0, 200, 0)
    SLIDER_BG = (0, 60, 0)
    SLIDER_HANDLE = (0, 200, 0)

    # Status indicators
    GOOD = (0, 255, 0)  # Green
    WARNING = (255, 200, 0)  # Yellow
    ALERT = (255, 0, 0)  # Red


class AlternativeColors:
    """Alternative color schemes."""

    # Blue scheme (modern radar)
    BLUE_BACKGROUND = (0, 0, 20)
    BLUE_GRID = (0, 80, 120)
    BLUE_SWEEP = (100, 200, 255)
    BLUE_DETECTION = (150, 220, 255)
    BLUE_UI_TEXT = (200, 230, 255)

    # Amber scheme (military/aviation)
    AMBER_BACKGROUND = (10, 5, 0)
    AMBER_GRID = (80, 60, 0)
    AMBER_SWEEP = (255, 200, 0)
    AMBER_DETECTION = (255, 220, 100)
    AMBER_UI_TEXT = (255, 200, 0)


def get_color_scheme(scheme: str = "classic_green") -> Dict[str, Tuple[int, int, int]]:
    """
    Get a complete color scheme.

    Args:
        scheme: Color scheme name ("classic_green", "blue", "amber")

    Returns:
        Dictionary of color names to RGB tuples
    """
    if scheme == "classic_green":
        return {
            "background": RadarColors.BACKGROUND,
            "range_ring": RadarColors.RANGE_RING,
            "bearing_marker": RadarColors.BEARING_MARKER,
            "sweep": RadarColors.SWEEP,
            "detection": RadarColors.DETECTION,
            "ui_text": RadarColors.UI_TEXT,
            "ui_background": RadarColors.UI_BACKGROUND,
        }
    elif scheme == "blue":
        return {
            "background": AlternativeColors.BLUE_BACKGROUND,
            "range_ring": AlternativeColors.BLUE_GRID,
            "bearing_marker": AlternativeColors.BLUE_GRID,
            "sweep": AlternativeColors.BLUE_SWEEP,
            "detection": AlternativeColors.BLUE_DETECTION,
            "ui_text": AlternativeColors.BLUE_UI_TEXT,
            "ui_background": (0, 20, 40),
        }
    elif scheme == "amber":
        return {
            "background": AlternativeColors.AMBER_BACKGROUND,
            "range_ring": AlternativeColors.AMBER_GRID,
            "bearing_marker": AlternativeColors.AMBER_GRID,
            "sweep": AlternativeColors.AMBER_SWEEP,
            "detection": AlternativeColors.AMBER_DETECTION,
            "ui_text": AlternativeColors.AMBER_UI_TEXT,
            "ui_background": (20, 10, 0),
        }
    else:
        # Default to classic green
        return get_color_scheme("classic_green")
