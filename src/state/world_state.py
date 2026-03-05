"""
World state initialisation utilities.
"""

from core.data_structures import WorldState


def create_world_state(
    entities: dict = None,
    object_states: dict = None,
    constraints: list = None,
) -> WorldState:
    """
    Construct a WorldState with explicit initial values.

    Parameters
    ----------
    entities : dict, optional
        Mapping of entity name to attribute dict.
    object_states : dict, optional
        Mapping of object name to current state label.
    constraints : list, optional
        Immutable narrative rules.

    Returns
    -------
    WorldState
    """
    world = WorldState()
    world.entities = entities or {}
    world.object_states = object_states or {}
    world.constraints = constraints or []
    return world


def create_initial_world_state() -> WorldState:
    """
    Create a default world state with no entities or constraints.

    Returns
    -------
    WorldState
    """
    return create_world_state()