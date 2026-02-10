"""
Range and bearing calculation utilities for maritime radar simulation.

Coordinate system:
- Cartesian (x, y) in meters with origin at radar position
- North = +y axis, East = +x axis (nautical/aviation standard)
- Bearing: 0° = North, 90° = East, clockwise rotation
"""

import numpy as np
from typing import Tuple


def calculate_range(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two positions.

    Args:
        pos1: First position as (x, y) in meters
        pos2: Second position as (x, y) in meters

    Returns:
        Range in meters
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return np.sqrt(dx**2 + dy**2)


def calculate_bearing(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """
    Calculate bearing from pos1 to pos2.

    Convention: 0° = North, 90° = East, clockwise rotation

    Args:
        pos1: From position as (x, y) in meters
        pos2: To position as (x, y) in meters

    Returns:
        Bearing in degrees [0, 360)
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]

    # atan2(x, y) NOT atan2(y, x) for our convention (North=0°)
    bearing_rad = np.arctan2(dx, dy)
    bearing_deg = np.degrees(bearing_rad)

    # Normalize to [0, 360)
    return bearing_deg % 360


def cartesian_to_polar(x: float, y: float) -> Tuple[float, float]:
    """
    Convert Cartesian coordinates to polar (range, bearing).

    Args:
        x: East coordinate in meters
        y: North coordinate in meters

    Returns:
        Tuple of (range in meters, bearing in degrees)
    """
    range_m = np.sqrt(x**2 + y**2)
    bearing_deg = np.degrees(np.arctan2(x, y)) % 360
    return range_m, bearing_deg


def polar_to_cartesian(range_m: float, bearing_deg: float) -> Tuple[float, float]:
    """
    Convert polar coordinates (range, bearing) to Cartesian.

    Args:
        range_m: Range in meters
        bearing_deg: Bearing in degrees (0° = North, 90° = East)

    Returns:
        Tuple of (x, y) in meters where x=East, y=North
    """
    bearing_rad = np.radians(bearing_deg)
    x = range_m * np.sin(bearing_rad)
    y = range_m * np.cos(bearing_rad)
    return x, y


def check_horizon(range_m: float, antenna_height_m: float, target_height_m: float = 5.0) -> bool:
    """
    Check if target is within radar horizon (line of sight).

    Uses radar horizon formula: d = 2.21 * sqrt(h) nautical miles
    where h is antenna height in meters above sea level.

    For two-way (radar + target height):
    d_total = 2.21 * (sqrt(h_radar) + sqrt(h_target))

    Args:
        range_m: Range to target in meters
        antenna_height_m: Radar antenna height in meters
        target_height_m: Target height in meters (default 5m for typical vessel)

    Returns:
        True if target is within horizon (visible), False otherwise
    """
    # Radar horizon formula in nautical miles
    d_radar_nm = 2.21 * np.sqrt(antenna_height_m)
    d_target_nm = 2.21 * np.sqrt(target_height_m)
    horizon_nm = d_radar_nm + d_target_nm

    # Convert to meters (1 nautical mile = 1852 meters)
    horizon_m = horizon_nm * 1852.0

    return range_m <= horizon_m


def normalize_bearing(bearing_deg: float) -> float:
    """
    Normalize bearing to [0, 360) range.

    Args:
        bearing_deg: Bearing in degrees (can be negative or > 360)

    Returns:
        Normalized bearing in [0, 360)
    """
    return bearing_deg % 360


def bearing_difference(bearing1_deg: float, bearing2_deg: float) -> float:
    """
    Calculate smallest angular difference between two bearings.

    Args:
        bearing1_deg: First bearing in degrees
        bearing2_deg: Second bearing in degrees

    Returns:
        Difference in degrees [-180, 180]
    """
    diff = (bearing2_deg - bearing1_deg + 180) % 360 - 180
    return diff


def is_bearing_in_sector(bearing_deg: float, sector_center_deg: float, sector_width_deg: float) -> bool:
    """
    Check if bearing falls within a sector.

    Args:
        bearing_deg: Bearing to test in degrees
        sector_center_deg: Center of sector in degrees
        sector_width_deg: Width of sector in degrees (total, not ±)

    Returns:
        True if bearing is within sector, False otherwise
    """
    diff = abs(bearing_difference(bearing_deg, sector_center_deg))
    return diff <= sector_width_deg / 2.0
