"""
World container class for managing all simulated entities.
"""

from typing import List, Optional
from src.objects.vessel import Vessel


class World:
    """
    Container for all simulated entities and environment state.

    Manages vessels and provides spatial queries.
    """

    def __init__(self):
        """Initialize an empty world."""
        self.vessels: List[Vessel] = []
        self.radar_position = (0.0, 0.0)  # Radar at origin
        self.simulation_time = 0.0  # seconds

    def add_vessel(self, vessel: Vessel) -> None:
        """
        Add a vessel to the world.

        Args:
            vessel: Vessel to add
        """
        self.vessels.append(vessel)

    def remove_vessel(self, vessel: Vessel) -> None:
        """
        Remove a vessel from the world.

        Args:
            vessel: Vessel to remove
        """
        if vessel in self.vessels:
            self.vessels.remove(vessel)

    def remove_vessel_by_id(self, vessel_id: int) -> bool:
        """
        Remove a vessel by its ID.

        Args:
            vessel_id: ID of vessel to remove

        Returns:
            True if vessel was found and removed, False otherwise
        """
        for vessel in self.vessels:
            if vessel.id == vessel_id:
                self.vessels.remove(vessel)
                return True
        return False

    def get_vessel_by_id(self, vessel_id: int) -> Optional[Vessel]:
        """
        Get a vessel by its ID.

        Args:
            vessel_id: ID of vessel to find

        Returns:
            Vessel if found, None otherwise
        """
        for vessel in self.vessels:
            if vessel.id == vessel_id:
                return vessel
        return None

    def get_all_vessels(self) -> List[Vessel]:
        """
        Get all vessels in the world.

        Returns:
            List of all vessels
        """
        return self.vessels.copy()

    def clear_vessels(self) -> None:
        """Remove all vessels from the world."""
        self.vessels.clear()

    def get_vessels_in_range(self, position: tuple, max_range: float) -> List[Vessel]:
        """
        Get all vessels within a certain range of a position.

        Args:
            position: Center position as (x, y) in meters
            max_range: Maximum range in meters

        Returns:
            List of vessels within range
        """
        import numpy as np

        vessels_in_range = []
        for vessel in self.vessels:
            dx = vessel.position[0] - position[0]
            dy = vessel.position[1] - position[1]
            distance = np.sqrt(dx**2 + dy**2)

            if distance <= max_range:
                vessels_in_range.append(vessel)

        return vessels_in_range

    def update(self, dt: float) -> None:
        """
        Update all vessels in the world.

        Args:
            dt: Time step in seconds
        """
        for vessel in self.vessels:
            vessel.update_position(dt)

        self.simulation_time += dt

    def get_vessel_count(self) -> int:
        """Get number of vessels in world."""
        return len(self.vessels)

    def __repr__(self) -> str:
        return (
            f"World(vessels={len(self.vessels)}, "
            f"time={self.simulation_time:.1f}s)"
        )
