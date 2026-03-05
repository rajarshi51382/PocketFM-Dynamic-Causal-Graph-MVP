"""
Dialogue generation module.

Generates character responses conditioned on the updated CharacterState.

In production this module builds a structured prompt from the character's
beliefs, emotions, and intentions, then calls an LLM to produce a response.
The stubs below implement a deterministic template-based fallback used in
tests and offline demos.
"""

import math
from typing import Optional

from core.data_structures import CharacterState


def build_generation_prompt(state: CharacterState, user_message: str) -> str:
    """
    Construct an LLM prompt from the character state.

    Serialises the highest-confidence beliefs, current emotional valence,
    and active intentions into a structured prompt template.

    Preconditions
    -------------
    state : CharacterState
    user_message : str

    Procedure
    ---------
    1. Extract relevant beliefs, emotions, and intentions
    2. Format them into a prompt template

    Postconditions
    --------------
    Returns prompt string

    Parameters
    ----------
    state : CharacterState
    user_message : str

    Returns
    -------
    str
    """
    belief_lines = []
    for prop, node in state.beliefs.items():
        prob = 1.0 / (1.0 + math.exp(-node.log_odds))
        belief_lines.append(f"  {prop}: {prob:.2f}")

    beliefs_text = "\n".join(belief_lines) if belief_lines else "  (none)"
    intentions_text = ", ".join(state.intentions) if state.intentions else "(none)"
    valence = state.emotions.valence

    prompt = (
        f"Character state:\n"
        f"  valence={valence:.2f}, arousal={state.emotions.arousal:.2f}\n"
        f"  intentions: {intentions_text}\n"
        f"Beliefs:\n{beliefs_text}\n\n"
        f"User: {user_message}\n"
        f"Character:"
    )
    return prompt


def generate_response(prompt: str) -> Optional[str]:
    """
    Generate dialogue using the language model.

    In production: sends the prompt to the configured LLM endpoint and
    returns the generated text.

    This stub returns None to indicate no LLM is connected. Callers
    should handle None gracefully.

    Preconditions
    -------------
    prompt : str

    Procedure
    ---------
    1. Send prompt to language model

    Postconditions
    --------------
    Returns generated response text

    Parameters
    ----------
    prompt : str

    Returns
    -------
    Optional[str]
        Generated response text, or None if no LLM is available.
    """
    return None


def produce_dialogue(state: CharacterState, user_message: str) -> str:
    """
    Main generation pipeline.

    Builds the prompt and calls the language model. Falls back to an
    acknowledgement string if no LLM response is available.

    Preconditions
    -------------
    state : CharacterState
    user_message : str

    Procedure
    ---------
    1. Build prompt
    2. Generate response

    Postconditions
    --------------
    Returns dialogue response

    Parameters
    ----------
    state : CharacterState
    user_message : str

    Returns
    -------
    str
        The character's dialogue response.
    """
    prompt = build_generation_prompt(state, user_message)
    response = generate_response(prompt)
    if response is None:
        return "[LLM not connected -- state updated internally]"
    return response