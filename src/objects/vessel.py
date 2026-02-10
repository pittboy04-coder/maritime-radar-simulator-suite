"""
Vessel class representing ships and boats in the maritime radar simulation.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class VesselType:
    """Vessel type specifications."""
    name: str
    typical_rcs: float  # Radar cross-section in m²
    typical_speed: float  # Typical speed in m/s
    length: float  # Length in meters
    width: float  # Width in meters


# Predefined vessel types
VESSEL_TYPES = {
    "yacht": VesselType("Yacht", 10.0, 5.0, 15.0, 4.0),
    "patrol": VesselType("Patrol Boat", 40.0, 15.0, 25.0, 6.0),
    "fishing": VesselType("Fishing Vessel", 80.0, 8.0, 30.0, 8.0),
    "cargo": VesselType("Cargo Ship", 200.0, 12.0, 150.0, 25.0),
    "tanker": VesselType("Tanker", 500.0, 10.0, 250.0, 40.0),
    "container": VesselType("Container Ship", 1000.0, 13.0, 300.0, 48.0),
}


class Vessel:
    """
    Represents a vessel (ship/boat) in the simulation.

    Attributes:
        position: Current (x, y) position in meters
        velocity: Tuple of (speed m/s, heading degrees)
        rcs: Radar cross-section in m²
        vessel_type: Type of vessel
        id: Unique identifier
    """

    _id_counter = 0

    def __init__(
        self,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        rcs: float,
        vessel_type: str = "cargo",
        height: float = 5.0
    ):
        """
        Initialize a vessel.

        Args:
            position: Initial (x, y) position in meters
            velocity: Tuple of (speed m/s, heading degrees where 0=North, 90=East)
            rcs: Radar cross-section in m²
            vessel_type: Type of vessel (yacht, patrol, fishing, cargo, tanker, container)
            height: Height above water in meters (for horizon calculations)
        """
        self.position = np.array(position, dtype=float)
        self.speed = velocity[0]  # m/s
        self.heading = velocity[1] % 360  # degrees [0, 360)
        self.rcs = rcs
        self.vessel_type = vessel_type
        self.height = height

        # Assign unique ID
        self.id = Vessel._id_counter
        Vessel._id_counter += 1

        # Get vessel type specifications
        self.type_spec = VESSEL_TYPES.get(vessel_type, VESSEL_TYPES["cargo"])

    def update_position(self, dt: float):
        """
        Update vessel position based on current velocity.

        Args:
            dt: Time step in seconds
        """
        # Convert heading to radians (0° = North = +y, 90° = East = +x)
        heading_rad = np.radians(self.heading)

        # Calculate velocity components
        vx = self.speed * np.sin(heading_rad)
        vy = self.speed * np.cos(heading_rad)

        # Update position
        self.position[0] += vx * dt
        self.position[1] += vy * dt

    def set_velocity(self, speed: float, heading: float):
        """
        Set vessel velocity.

        Args:
            speed: Speed in m/s
            heading: Heading in degrees (0=North, 90=East)
        """
        self.speed = max(0.0, speed)  # Ensure non-negative
        self.heading = heading % 360

    def get_position(self) -> Tuple[float, float]:
        """Get current position as tuple."""
        return tuple(self.position)

    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity as (speed, heading) tuple."""
        return (self.speed, self.heading)

    def __repr__(self) -> str:
        return (
            f"Vessel(id={self.id}, type={self.vessel_type}, "
            f"pos=({self.position[0]:.0f}, {self.position[1]:.0f}), "
            f"speed={self.speed:.1f}m/s, heading={self.heading:.0f}°, "
            f"rcs={self.rcs:.0f}m²)"
        )


class ConstantVelocityMotion:
    """Simple constant velocity motion model."""

    def update(self, vessel: Vessel, dt: float):
        """Update vessel position with constant velocity."""
        vessel.update_position(dt)


class WaypointMotion:
    """Motion model that navigates between waypoints."""

    def __init__(self, waypoints: list[Tuple[float, float]], speed: float = 10.0):
        """
        Initialize waypoint motion.

        Args:
            waypoints: List of (x, y) waypoints in meters
            speed: Speed in m/s
        """
        self.waypoints = waypoints
        self.speed = speed
        self.current_waypoint_index = 0
        self.waypoint_threshold = 50.0  # meters - distance to consider waypoint reached

    def update(self, vessel: Vessel, dt: float):
        """Update vessel position toward next waypoint."""
        if not self.waypoints:
            return

        # Get current target waypoint
        target = self.waypoints[self.current_waypoint_index]

        # Calculate direction to waypoint
        dx = target[0] - vessel.position[0]
        dy = target[1] - vessel.position[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Check if waypoint reached
        if distance < self.waypoint_threshold:
            # Move to next waypoint (wrap around)
            self.current_waypoint_index = (
                self.current_waypoint_index + 1
            ) % len(self.waypoints)
            target = self.waypoints[self.current_waypoint_index]
            dx = target[0] - vessel.position[0]
            dy = target[1] - vessel.position[1]

        # Calculate heading to waypoint
        heading = np.degrees(np.arctan2(dx, dy)) % 360

        # Set velocity and update
        vessel.set_velocity(self.speed, heading)
        vessel.update_position(dt)


class CircularMotion:
    """Motion model for circular patrol patterns."""

    def __init__(
        self, center: Tuple[float, float], radius: float, angular_speed: float = 0.1
    ):
        """
        Initialize circular motion.

        Args:
            center: Center of circle (x, y) in meters
            radius: Radius in meters
            angular_speed: Angular speed in radians per second
        """
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.angular_speed = angular_speed
        self.angle = 0.0  # Current angle in radians

    def update(self, vessel: Vessel, dt: float):
        """Update vessel position in circular pattern."""
        # Update angle
        self.angle += self.angular_speed * dt

        # Calculate position on circle
        x = self.center[0] + self.radius * np.cos(self.angle)
        y = self.center[1] + self.radius * np.sin(self.angle)

        # Calculate heading (tangent to circle)
        heading = np.degrees(self.angle + np.pi / 2) % 360

        # Calculate speed based on angular speed and radius
        speed = self.angular_speed * self.radius

        # Update vessel
        vessel.position = np.array([x, y])
        vessel.set_velocity(speed, heading)
