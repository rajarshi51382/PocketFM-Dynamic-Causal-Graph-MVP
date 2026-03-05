"""
Main simulation pipeline.

Connects all modules:

    message
    -> event extraction
    -> belief update
    -> state propagation
    -> dialogue generation
"""

from typing import List, Tuple

from core.data_structures import CharacterState, WorldState
from extraction.event_extraction import extract_event, validate_event
from generation.dialogue_generation import produce_dialogue
from reasoning.belief_update import apply_belief_updates
from reasoning.state_update import propagate_state_updates


def simulation_turn(
    user_message: str,
    character_state: CharacterState,
    world_state: WorldState,
    lambda_base: float = 0.5,
    narrative_importance: float = 1.0,
) -> str:
    """
    Execute one conversation turn.

    Steps:
    1. Extract and validate a structured event frame from the message.
    2. Apply belief updates to the character state.
    3. Propagate downstream state changes (emotions, relationships,
       intentions).
    4. Generate and return the character's dialogue response.

    Parameters
    ----------
    user_message : str
    character_state : CharacterState
        Modified in place.
    world_state : WorldState
    lambda_base : float
        Base learning rate for belief updates.
    narrative_importance : float
        Shock-gated plasticity factor for this turn.

    Returns
    -------
    str
        Generated dialogue response.
    """
    event = extract_event(user_message)
    event = validate_event(event, user_message)

    if event.confidence > 0.0:
        apply_belief_updates(
            character_state,
            event,
            lambda_base=lambda_base,
            narrative_importance=narrative_importance,
        )
        propagate_state_updates(character_state, event)

    world_state.timeline_index = character_state.timeline_index

    return produce_dialogue(character_state, user_message)


def run_simulation(
    initial_character_state: CharacterState,
    world_state: WorldState,
) -> List[Tuple[str, str]]:
    """
    Run an interactive simulation in the terminal.

    Reads user input from stdin, calls simulation_turn, and prints the
    response. Returns the full conversation history.

    Parameters
    ----------
    initial_character_state : CharacterState
    world_state : WorldState

    Returns
    -------
    List[Tuple[str, str]]
        List of (user_message, character_response) pairs.
    """
    history: List[Tuple[str, str]] = []
    character_state = initial_character_state

    print("Simulation started. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in {"quit", "exit"}:
            break

        response = simulation_turn(user_input, character_state, world_state)
        print(f"Character: {response}\n")
        history.append((user_input, response))

    return history