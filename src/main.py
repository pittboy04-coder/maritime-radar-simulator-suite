"""
Main entry point for maritime radar simulation.
"""

import sys
import argparse
from src.core.simulation import RadarSimulation
from src.core.world import World
from src.objects.vessel import Vessel, VESSEL_TYPES
from src.radar.parameters import RadarParameters, EnvironmentParameters, DisplayParameters
import numpy as np


def create_default_scenario() -> World:
    """
    Create a default scenario with several vessels.

    Returns:
        World object with vessels
    """
    world = World()

    # Add various vessels at different positions and velocities
    vessels = [
        # Cargo ship heading Northeast
        Vessel(
            position=(5000, 3000),
            velocity=(10, 45),
            rcs=VESSEL_TYPES["cargo"].typical_rcs,
            vessel_type="cargo",
        ),
        # Tanker heading South
        Vessel(
            position=(-3000, 8000),
            velocity=(12, 180),
            rcs=VESSEL_TYPES["tanker"].typical_rcs,
            vessel_type="tanker",
        ),
        # Patrol boat heading East
        Vessel(
            position=(7000, -2000),
            velocity=(20, 90),
            rcs=VESSEL_TYPES["patrol"].typical_rcs,
            vessel_type="patrol",
        ),
        # Fishing vessel heading Northwest
        Vessel(
            position=(-5000, -4000),
            velocity=(8, 315),
            rcs=VESSEL_TYPES["fishing"].typical_rcs,
            vessel_type="fishing",
        ),
        # Yacht heading West
        Vessel(
            position=(2000, 6000),
            velocity=(6, 270),
            rcs=VESSEL_TYPES["yacht"].typical_rcs,
            vessel_type="yacht",
        ),
        # Container ship heading North
        Vessel(
            position=(0, -7000),
            velocity=(14, 0),
            rcs=VESSEL_TYPES["container"].typical_rcs,
            vessel_type="container",
        ),
        # Another cargo ship
        Vessel(
            position=(-6000, 2000),
            velocity=(11, 135),
            rcs=VESSEL_TYPES["cargo"].typical_rcs,
            vessel_type="cargo",
        ),
    ]

    for vessel in vessels:
        world.add_vessel(vessel)

    print(f"Created scenario with {len(vessels)} vessels")
    return world


def create_random_scenario(num_vessels: int = 10) -> World:
    """
    Create a scenario with random vessels.

    Args:
        num_vessels: Number of vessels to create

    Returns:
        World object with random vessels
    """
    world = World()

    vessel_type_names = list(VESSEL_TYPES.keys())

    for _ in range(num_vessels):
        # Random position within 15km radius
        angle = np.random.uniform(0, 360)
        distance = np.random.uniform(2000, 15000)
        x = distance * np.sin(np.radians(angle))
        y = distance * np.cos(np.radians(angle))

        # Random velocity
        speed = np.random.uniform(5, 20)
        heading = np.random.uniform(0, 360)

        # Random vessel type
        vessel_type = np.random.choice(vessel_type_names)
        rcs = VESSEL_TYPES[vessel_type].typical_rcs

        vessel = Vessel(
            position=(x, y),
            velocity=(speed, heading),
            rcs=rcs,
            vessel_type=vessel_type,
        )
        world.add_vessel(vessel)

    print(f"Created random scenario with {num_vessels} vessels")
    return world


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Maritime Radar Simulation")
    parser.add_argument(
        "--range", type=float, default=20000, help="Maximum radar range in meters"
    )
    parser.add_argument(
        "--gain", type=float, default=30.0, help="Receiver gain in dB"
    )
    parser.add_argument(
        "--rpm", type=float, default=24.0, help="Antenna rotation rate in RPM"
    )
    parser.add_argument(
        "--sea-state", type=int, default=3, help="Sea state (0-9 Beaufort scale)"
    )
    parser.add_argument(
        "--rain", type=float, default=0.0, help="Rain rate in mm/hr"
    )
    parser.add_argument(
        "--random", type=int, default=0, help="Create random scenario with N vessels"
    )

    args = parser.parse_args()

    # Create parameters
    radar_params = RadarParameters(
        max_range=args.range,
        gain=args.gain,
        rotation_rate=args.rpm,
    )

    env_params = EnvironmentParameters(
        sea_state=args.sea_state,
        rain_rate=args.rain,
    )

    display_params = DisplayParameters()

    # Create world
    if args.random > 0:
        world = create_random_scenario(args.random)
    else:
        world = create_default_scenario()

    # Create and run simulation
    sim = RadarSimulation(
        world=world,
        radar_params=radar_params,
        env_params=env_params,
        display_params=display_params,
    )

    try:
        sim.run()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
