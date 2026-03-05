"""
Character state initialisation utilities.
"""

from core.data_structures import (
    BeliefNode,
    CharacterState,
    EmotionState,
    RelationshipState,
    TraitState,
)


def create_character_state(
    traits: dict = None,
    trait_plasticity: float = 0.05,
    beliefs: dict = None,
    relationships: dict = None,
    valence: float = 0.0,
    arousal: float = 0.5,
) -> CharacterState:
    """
    Construct a CharacterState with explicit initial values.

    Parameters
    ----------
    traits : dict, optional
        Mapping of trait name to intensity.
    trait_plasticity : float
        Plasticity for the trait node.
    beliefs : dict, optional
        Mapping of proposition string to BeliefNode.
    relationships : dict, optional
        Mapping of entity name to RelationshipState.
    valence : float
        Initial valence in [-1, 1].
    arousal : float
        Initial arousal in [0, 1].

    Returns
    -------
    CharacterState
    """
    state = CharacterState()
    state.traits = TraitState(traits or {}, plasticity=trait_plasticity)
    state.emotions = EmotionState(valence=valence, arousal=arousal)
    state.beliefs = beliefs or {}
    state.relationships = relationships or {}
    return state


def create_initial_character_state() -> CharacterState:
    """
    Create a default character state with neutral values.

    Returns
    -------
    CharacterState
    """
    return create_character_state()