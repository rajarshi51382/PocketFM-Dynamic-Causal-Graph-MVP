"""
State propagation module.

Responsible for propagating belief updates and event effects
into emotional state, relationships, and intentions.

This module operates on the CharacterState defined in
core.data_structures and modifies internal variables
based on the incoming event frame.

All functions modify CharacterState in place and return the updated
state for convenience.
"""

import math

from core.data_structures import CharacterState, EventFrame

_EMOTION_TONE_MAP = {
    "joy": ("valence", +0.3),
    "anger": ("valence", -0.3),
    "fear": ("valence", -0.2),
    "sadness": ("valence", -0.25),
    "surprise": ("arousal", +0.3),
    "disgust": ("valence", -0.15),
    "trust": ("valence", +0.1),
    "anticipation": ("arousal", +0.2),
}

_TRUST_TONE_MAP = {
    "joy": +0.05,
    "trust": +0.1,
    "anger": -0.1,
    "disgust": -0.08,
    "fear": -0.05,
    "sadness": -0.03,
}


def update_emotional_state(state: CharacterState, event: EventFrame) -> CharacterState:
    """
    Update the character's emotional state based on the event tone.

    Applies a plasticity-weighted shift to valence and arousal. Discrete
    emotion tag intensities are updated additively and clamped to [0, 1].

    Preconditions
    -------------
    state : CharacterState
        Current internal character state
    event : EventFrame
        Structured representation of the incoming dialogue event

    Procedure
    ---------
    1. Read emotional tone (τt) from the event frame
    2. Adjust valence and arousal accordingly
    3. Update emotion tags to reflect detected emotions
    4. Apply emotional plasticity to smooth updates

    Postconditions
    --------------
    state.emotions updated

    Parameters
    ----------
    state : CharacterState
    event : EventFrame

    Returns
    -------
    CharacterState
        The same object, modified in place.
    """
    tone = event.emotional_tone
    if tone is None:
        return state

    tone_lower = tone.strip().lower()
    alpha = state.emotions.plasticity

    if tone_lower in _EMOTION_TONE_MAP:
        dimension, delta = _EMOTION_TONE_MAP[tone_lower]
        if dimension == "valence":
            state.emotions.valence = _clamp(
                state.emotions.valence + alpha * delta, -1.0, 1.0
            )
        else:
            state.emotions.arousal = _clamp(
                state.emotions.arousal + alpha * delta, 0.0, 1.0
            )

    prev = state.emotions.emotion_tags.get(tone_lower, 0.0)
    state.emotions.emotion_tags[tone_lower] = _clamp(prev + alpha * 0.3, 0.0, 1.0)

    return state


def update_relationship_state(state: CharacterState, event: EventFrame) -> CharacterState:
    """
    Update relationship values between the character and event entities.

    Trust shifts are driven by the event's emotional tone. Only entities
    that already have a relationship entry are updated.

    Preconditions
    -------------
    state : CharacterState
    event : EventFrame

    Procedure
    ---------
    1. Identify interacting entities from event.entities
    2. Locate relationship nodes in state.relationships
    3. Adjust trust, affection, or respect based on event tone
    4. Apply relationship decay or reinforcement

    Postconditions
    --------------
    state.relationships updated

    Parameters
    ----------
    state : CharacterState
    event : EventFrame

    Returns
    -------
    CharacterState
        The same object, modified in place.
    """
    tone = (event.emotional_tone or "").strip().lower()
    delta_trust = _TRUST_TONE_MAP.get(tone, 0.0)

    for entity in event.entities:
        rel = state.relationships.get(entity)
        if rel is not None:
            rel.trust = _clamp(rel.trust + delta_trust * event.confidence, 0.0, 1.0)

    return state


def update_intentions(state: CharacterState) -> CharacterState:
    """
    Derive character intentions from current beliefs and emotional state.

    High-confidence positive beliefs generate approach intentions.
    High-confidence negative beliefs generate avoidance intentions.
    High arousal promotes assertive or reactive intentions.

    Preconditions
    -------------
    state : CharacterState

    Procedure
    ---------
    1. Analyze current belief nodes
    2. Consider emotional state
    3. Derive likely goals or motivations
    4. Update intention representation

    Postconditions
    --------------
    state.intentions updated

    Parameters
    ----------
    state : CharacterState

    Returns
    -------
    CharacterState
        The same object, modified in place.
    """
    intentions = []

    for key, belief in state.beliefs.items():
        prob = 1.0 / (1.0 + math.exp(-belief.log_odds))
        if prob > 0.75:
            intentions.append(f"assert_{key}")
        elif prob < 0.25:
            intentions.append(f"deny_{key}")

    if state.emotions.arousal > 0.7:
        intentions.append("respond_urgently")

    state.intentions = intentions
    return state


def propagate_state_updates(state: CharacterState, event: EventFrame) -> CharacterState:
    """
    Execute the full internal state update pipeline for one turn.

    Steps:
    1. Update emotional state from event tone.
    2. Update relationship state for referenced entities.
    3. Derive updated intentions from beliefs and emotions.
    4. Increment the timeline index.

    Preconditions
    -------------
    state : CharacterState
    event : EventFrame

    Procedure
    ---------
    1. Update emotional state
    2. Update relationship state
    3. Update intentions
    4. Increment timeline index

    Postconditions
    --------------
    Updated CharacterState returned

    Parameters
    ----------
    state : CharacterState
    event : EventFrame

    Returns
    -------
    CharacterState
        The same object, modified in place.
    """
    update_emotional_state(state, event)
    update_relationship_state(state, event)
    update_intentions(state)
    state.timeline_index += 1
    return state


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))