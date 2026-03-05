"""
Offline demo conversations for testing belief updates and character
state consistency without a live LLM connection.

Run from the repository root with:
    PYTHONPATH=src python demo/demo_conversations.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.data_structures import BeliefNode, EventFrame, RelationshipState
from reasoning.belief_update import apply_belief_updates
from reasoning.state_update import propagate_state_updates
from simulation.simulation_loop import simulation_turn
from state.character_state import create_character_state
from state.world_state import create_initial_world_state


def _log_odds_to_prob(log_odds: float) -> float:
    return 1.0 / (1.0 + math.exp(-log_odds))


def _print_beliefs(beliefs: dict) -> None:
    if not beliefs:
        print("  (no beliefs)")
        return
    for prop, node in beliefs.items():
        prob = _log_odds_to_prob(node.log_odds)
        print(f"  {prop}: log_odds={node.log_odds:.3f}  p={prob:.3f}")


def run_neutral_conversation() -> None:
    """
    Demonstrate a routine dialogue turn that does not shift beliefs
    significantly (narrative_importance = 1.0, low-trust source).
    """
    print("=== Neutral Conversation Demo ===\n")

    state = create_character_state(
        traits={"bravery": 0.6, "honesty": 0.8},
        beliefs={
            "door_is_locked": BeliefNode("door_is_locked", log_odds=1.0),
        },
        relationships={
            "guard": RelationshipState(trust=0.4, affection=0.3, respect=0.5),
        },
    )

    event = EventFrame(
        propositions=["door_is_locked"],
        entities=["guard"],
        speaker="guard",
        emotional_tone="trust",
        confidence=0.8,
    )

    print("Before update:")
    _print_beliefs(state.beliefs)

    apply_belief_updates(state, event, lambda_base=0.5, narrative_importance=1.0)
    propagate_state_updates(state, event)

    print("\nAfter neutral event (guard says door is locked):")
    _print_beliefs(state.beliefs)
    print(f"  emotions: valence={state.emotions.valence:.3f}, "
          f"arousal={state.emotions.arousal:.3f}")
    print(f"  intentions: {state.intentions}")


def run_major_event_conversation() -> None:
    """
    Demonstrate a high-importance narrative event that causes a large
    belief shift (narrative_importance >> 1.0, contradicting evidence).
    """
    print("\n=== Major Event Conversation Demo ===\n")

    world_state = create_initial_world_state()
    state = create_character_state(
        beliefs={
            "ally_is_trustworthy": BeliefNode("ally_is_trustworthy", log_odds=2.0),
            "not_ally_is_trustworthy": BeliefNode("not_ally_is_trustworthy", log_odds=-2.0),
        },
        relationships={
            "narrator": RelationshipState(trust=0.9, affection=0.5, respect=0.8),
        },
    )

    print("Before major event:")
    _print_beliefs(state.beliefs)

    betrayal_event = EventFrame(
        propositions=["not_ally_is_trustworthy"],
        entities=["ally"],
        speaker="narrator",
        emotional_tone="anger",
        confidence=0.95,
    )

    apply_belief_updates(state, betrayal_event, lambda_base=0.5, narrative_importance=3.0)
    propagate_state_updates(state, betrayal_event)

    print("\nAfter betrayal event (high narrative importance):")
    _print_beliefs(state.beliefs)
    print(f"  emotions: valence={state.emotions.valence:.3f}, "
          f"arousal={state.emotions.arousal:.3f}")
    print(f"  intentions: {state.intentions}")

    print("\nSimulation turn (LLM stub):")
    response = simulation_turn(
        "I can't believe the ally betrayed us.",
        state,
        world_state,
        lambda_base=0.5,
        narrative_importance=2.0,
    )
    print(f"  Character: {response}")


if __name__ == "__main__":
    run_neutral_conversation()
    run_major_event_conversation()