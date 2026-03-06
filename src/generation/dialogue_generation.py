"""
Dialogue generation module.

Generates character responses conditioned on the updated CharacterState.

In production this module builds a structured prompt from the character's
beliefs, emotions, and intentions, then calls an LLM to produce a response.
The stubs below implement a deterministic template-based fallback used in
tests and offline demos.
"""

import math
import re
from typing import Optional, List

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
    # Use probability for easier reading in the prompt
    for prop, node in state.beliefs.items():
        belief_lines.append(f"  {prop}: {node.probability:.2f}")

    beliefs_text = "\n".join(belief_lines) if belief_lines else "  (none)"
    intentions_text = ", ".join(state.intentions) if state.intentions else "(none)"
    valence = state.emotions.valence
    arousal = state.emotions.arousal
    dominant = state.emotions.dominant_emotion() or "neutral"

    prompt = (
        f"Character ID: {state.character_id}\n"
        f"Current Emotion: {dominant} (valence={valence:.2f}, arousal={arousal:.2f})\n"
        f"Intentions: {intentions_text}\n"
        f"Beliefs (proposition: probability):\n{beliefs_text}\n\n"
        f"User: {user_message}\n"
        f"Character response style: {state.traits}\n"
        f"Character:"
    )
    return prompt


def generate_response(prompt: str) -> Optional[str]:
    """
    Generate dialogue using a rule-based fallback (simulating an LLM).

    In production: sends the prompt to the configured LLM endpoint.
    This MVP implementation parses the prompt to generate a response
    consistent with the character state.

    Preconditions
    -------------
    prompt : str

    Procedure
    ---------
    1. Parse emotional state and intentions from the prompt.
    2. Select a response template based on valence and arousal.
    3. Incorporate intentions if present.

    Postconditions
    --------------
    Returns generated response text

    Parameters
    ----------
    prompt : str

    Returns
    -------
    Optional[str]
        Generated response text.
    """
    # Parse valence and arousal
    v_match = re.search(r'valence=([-\d.]+)', prompt)
    a_match = re.search(r'arousal=([-\d.]+)', prompt)
    i_match = re.search(r'Intentions: (.+)', prompt)
    
    valence = float(v_match.group(1)) if v_match else 0.0
    arousal = float(a_match.group(1)) if a_match else 0.5
    intentions = i_match.group(1) if i_match else "(none)"

    # Response selection logic
    if valence > 0.3:
        if arousal > 0.6:
            base = "I am thrilled by this! "
        else:
            base = "I am pleased to hear that. "
    elif valence < -0.3:
        if arousal > 0.6:
            base = "This is unacceptable! "
        else:
            base = "This is quite disappointing. "
    else:
        if arousal > 0.6:
            base = "I am listening intently. "
        else:
            base = "I see. "

    if intentions != "(none)":
        base += f"Regarding my goal of {intentions}, I think we should proceed carefully."
    else:
        base += "What else can you tell me?"

    return base


def produce_dialogue(state: CharacterState, user_message: str) -> str:
    """
    Main generation pipeline.

    Builds the prompt and calls the generation function.
    Ensures a response is always returned.

    Preconditions
    -------------
    state : CharacterState
    user_message : str

    Procedure
    ---------
    1. Build prompt
    2. Generate response
    3. Fallback if generation fails

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
        # Emergency fallback
        return "I'm not sure how to respond to that right now."
        
    return response