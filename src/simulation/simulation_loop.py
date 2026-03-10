"""
Main simulation pipeline.

Connects all modules:

    message
    -> event extraction
    -> belief update
    -> causal propagation
    -> state propagation
    -> dialogue generation
"""

from typing import List, Tuple

from core.data_structures import CharacterState, WorldState, EventFrame
from extraction.event_extraction import extract_event, validate_event
from generation.dialogue_generation import produce_dialogue
from reasoning.belief_update import apply_belief_updates, DIRECT_OBSERVATION
from reasoning.state_update import propagate_state_updates
from reasoning.causal_propagation import propagate_causal_effects
from reasoning.verifier import verify_dialogue

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
    1. Synchronize timeline (Character follows World).
    2. Perceive World: Character sees facts in their vicinity.
    3. Extract Dialogue: Extract event from user message.
    4. Epistemic Filter: Check knowledge boundaries.
    5. Update: Apply beliefs, causal effects, and state propagation.
    6. Generate: Return character response.
    """
    # 0. Sync and Perceive
    character_state.timeline_index = world_state.timeline_index
    
    # Simple location tracking (assuming Sir_Galahad for now or generic)
    loc = world_state.entities.get(character_state.character_id, {}).get("location")
    obs = world_state.perceive(character_location=loc)
    
    # 1. Direct Observation Update
    if obs["visible_objects"]:
        for obj, state in obs["visible_objects"].items():
            prop = f"{obj}_is_{state}"
            obs_event = EventFrame(
                propositions=[prop],
                speaker=DIRECT_OBSERVATION,
                confidence=1.0
            )
            apply_belief_updates(character_state, obs_event)

    # 2. Extract and validate user event
    event = extract_event(user_message)
    event = validate_event(
        event,
        user_message,
        allowed_predicates=character_state.belief_schema,
    )

    # 3. Epistemic Filtering & Update
    if event.confidence > 0.0:
        # Check if character is allowed to know about this event/time
        if character_state.can_know_event_at(world_state.timeline_index):
            # Direct evidence update
            apply_belief_updates(
                character_state,
                event,
                lambda_base=lambda_base,
                narrative_importance=narrative_importance,
            )
            
            # Causal inference (internal reasoning)
            propagate_causal_effects(character_state)
            
            # Emotional/Social state update
            propagate_state_updates(character_state, event)
        else:
            # Epistemic block: character is "out of the loop" or event is too recent/far
            # Still update emotions slightly based on tone, even if facts aren't believed
            from reasoning.state_update import update_emotional_state
            update_emotional_state(character_state, event)

    response = produce_dialogue(character_state, user_message)

    is_valid, violations = verify_dialogue(response, character_state)
    if not is_valid:
        reason = violations[0] if violations else "unknown_violation"
        response = f"I'm not sure that would be consistent with what I know. [{reason}]"
    
    return response


def run_simulation(
    initial_character_state: CharacterState,
    world_state: WorldState,
) -> List[Tuple[str, str]]:
    """
    Run an interactive simulation in the terminal.

    Reads user input from stdin, calls simulation_turn, and prints the
    response. Returns the full conversation history.

    Preconditions
    -------------
    initial_character_state : CharacterState
    world_state : WorldState

    Procedure
    ---------
    1. Receive user input
    2. Call simulation_turn
    3. Update timeline index
    4. Repeat

    Postconditions
    --------------
    Returns conversation history

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
        
        if not user_input:
            continue

        response = simulation_turn(user_input, character_state, world_state)
        print(f"Character: {response}\n")
        history.append((user_input, response))

    return history