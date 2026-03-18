"""
Dialogue generation module.

Generates character responses conditioned on the updated CharacterState.

In production this module builds a structured prompt from the character's
beliefs, emotions, and intentions, then calls an LLM to produce a response.
The stubs below implement a sophisticated template-based fallback used in
tests, offline demos, and free deployment.
"""

import math
import re
import random
import logging
from typing import Optional, List

from core.data_structures import CharacterState
from core import llm_client

logger = logging.getLogger(__name__)

# Rich response templates for engaging rule-based dialogue
RESPONSE_TEMPLATES = {
    "positive_high_arousal": [
        "By the stars, this fills me with hope! {context}",
        "Excellent news! My heart races with anticipation. {context}",
        "This is wonderful to hear! {context}",
        "I feel invigorated by these tidings! {context}",
        "Such fortune! This changes everything. {context}",
    ],
    "positive_low_arousal": [
        "That is good to know. {context}",
        "I am pleased by this development. {context}",
        "This brings me some comfort. {context}",
        "Indeed, this is reassuring news. {context}",
        "I appreciate you sharing this. {context}",
    ],
    "negative_high_arousal": [
        "What treachery is this?! {context}",
        "This cannot stand! {context}",
        "By all that is sacred, this is dire news! {context}",
        "I am outraged! {context}",
        "This demands immediate action! {context}",
    ],
    "negative_low_arousal": [
        "This troubles me deeply... {context}",
        "Alas, I feared as much. {context}",
        "This is... most unfortunate. {context}",
        "A heavy burden to bear. {context}",
        "I must think on this carefully. {context}",
    ],
    "neutral_high_arousal": [
        "I listen with keen interest! {context}",
        "You have my full attention. {context}",
        "Intriguing... tell me more. {context}",
        "This is noteworthy indeed. {context}",
        "I am eager to understand this better. {context}",
    ],
    "neutral_low_arousal": [
        "I see. {context}",
        "Very well. {context}",
        "That is noted. {context}",
        "I understand. {context}",
        "Please, continue. {context}",
    ],
}

BELIEF_CONTEXT = {
    "castle_is_safe": {
        True: "The castle remains our sanctuary.",
        False: "If the castle is no longer safe, we must find shelter elsewhere.",
    },
    "forest_is_dangerous": {
        True: "The forest paths are treacherous indeed.",
        False: "Perhaps the forest is not as perilous as we thought.",
    },
    "king_is_wise": {
        True: "The king's wisdom guides us well.",
        False: "I begin to question the king's judgment.",
    },
}

TRAIT_MODIFIERS = {
    "bravery": {
        "high": "I shall face whatever comes with courage.",
        "low": "We must proceed with utmost caution.",
    },
    "honesty": {
        "high": "I speak only the truth as I know it.",
        "low": "Some truths are best left unspoken.",
    },
    "trusting": {
        "high": "I believe in the good intentions of others.",
        "low": "Experience has taught me to be wary.",
    },
}

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

    traits_text = ", ".join([f"{k} ({v:.2f})" for k,v in state.traits.traits.items()]) or "neutral"

    prompt = (
        f"Roleplay as the following character:\n"
        f"Character ID: {state.character_id}\n"
        f"Traits: {traits_text}\n"
        f"Current Emotion: {dominant} (valence={valence:.2f}, arousal={arousal:.2f})\n"
        f"Intentions: {intentions_text}\n"
        f"Beliefs (proposition: probability):\n{beliefs_text}\n\n"
        f"Context:\nUser says: \"{user_message}\"\n\n"
        f"Instructions:\n"
        f"Respond to the user in character. Reflect your traits, current emotion, and beliefs.\n"
        f"Keep the response concise (1-3 sentences).\n"
        f"Do not output internal monologue, just the dialogue.\n\n"
        f"Character response:"
    )
    return prompt


def generate_response(prompt: str) -> Optional[str]:
    """
    Generate dialogue using LLM conditioned on embeddings graph, 
    falling back to rule-based logic.
    """
    if llm_client.is_llm_available():
        text = llm_client.generate_text(prompt)
        if text:
            return text
            
    return _generate_response_rules(prompt)


def _generate_response_rules(prompt: str, state: Optional[CharacterState] = None) -> Optional[str]:
    """
    Sophisticated rule-based dialogue generation.
    
    Generates contextually aware responses based on character state,
    beliefs, emotions, and traits.
    """
    # Parse valence and arousal from the prompt string
    v_match = re.search(r'valence=([-\d.]+)', prompt)
    a_match = re.search(r'arousal=([-\d.]+)', prompt)
    i_match = re.search(r'Intentions: (.+)', prompt)
    char_match = re.search(r'Character ID: (.+)', prompt)
    
    valence = float(v_match.group(1)) if v_match else 0.0
    arousal = float(a_match.group(1)) if a_match else 0.5
    intentions = i_match.group(1) if i_match else "(none)"
    char_name = char_match.group(1) if char_match else "I"
    
    # Parse beliefs from prompt
    beliefs_section = re.search(r'Beliefs \(proposition: probability\):\n(.*?)\n\nContext:', prompt, re.DOTALL)
    belief_probs = {}
    if beliefs_section:
        for line in beliefs_section.group(1).strip().split('\n'):
            match = re.match(r'\s+(\w+): ([\d.]+)', line)
            if match:
                belief_probs[match.group(1)] = float(match.group(2))
    
    # Parse traits from prompt
    traits_match = re.search(r'Traits: (.+)', prompt)
    traits = {}
    if traits_match:
        for item in traits_match.group(1).split(', '):
            match = re.match(r'(\w+) \(([-\d.]+)\)', item)
            if match:
                traits[match.group(1)] = float(match.group(2))
    
    # Select response category based on emotional state
    if valence > 0.3:
        category = "positive_high_arousal" if arousal > 0.6 else "positive_low_arousal"
    elif valence < -0.3:
        category = "negative_high_arousal" if arousal > 0.6 else "negative_low_arousal"
    else:
        category = "neutral_high_arousal" if arousal > 0.6 else "neutral_low_arousal"
    
    # Build context from beliefs
    context_parts = []
    for belief, prob in belief_probs.items():
        if belief in BELIEF_CONTEXT:
            is_high = prob > 0.5
            context_parts.append(BELIEF_CONTEXT[belief][is_high])
    
    # Add trait-based modifier
    for trait, value in traits.items():
        if trait in TRAIT_MODIFIERS:
            level = "high" if value > 0.5 else "low"
            if random.random() > 0.5:  # Don't always add trait commentary
                context_parts.append(TRAIT_MODIFIERS[trait][level])
    
    # Add intention if present
    if intentions not in ["(none)", "None", ""]:
        first_intention = intentions.split(',')[0].strip()
        context_parts.append(f"Regarding {first_intention}, I must act decisively.")
    
    # Select and format response
    templates = RESPONSE_TEMPLATES.get(category, RESPONSE_TEMPLATES["neutral_low_arousal"])
    template = random.choice(templates)
    
    context = " ".join(context_parts[:2]) if context_parts else "What else would you have me know?"
    
    response = template.format(context=context)
    
    return response


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