"""
Event extraction pipeline.

Responsible for converting raw user dialogue into structured event frames
used by the update phase.

Converts raw user dialogue into structured EventFrame objects.
In production this module calls an LLM extraction pass followed by a
validation pass. The stubs below provide a rule-based fallback used
in tests and offline demos.
"""

import re
import json
import logging
from typing import Optional, List, Dict, Any

from core.data_structures import EventFrame
from core import llm_client

logger = logging.getLogger(__name__)

# Fallback tone map used if LLM fails
_TONE_MAP = {
    "joy": ["happy", "good", "great", "joy", "wonderful", "excellent", "thanks", "thank"],
    "anger": ["angry", "mad", "hate", "furious", "annoyed", "stop"],
    "sadness": ["sad", "bad", "unfortunate", "sorry", "regret", "pity"],
    "fear": ["scared", "fear", "afraid", "terrified", "worry", "worried"],
    "trust": ["trust", "believe", "agree", "sure", "certain"],
    "disgust": ["gross", "disgust", "hate", "eww"],
    "surprise": ["wow", "really", "unexpected", "surprise"],
    "anticipation": ["hope", "expect", "wait", "looking forward"]
}

def _normalize_base_prop(prop: str) -> str:
    p = prop.strip().lower()
    if p.startswith("not_"):
        return p[4:]
    if p.startswith("~"):
        return p[1:]
    return p

def _canonicalize_proposition(prop: str, allowed_predicates: set[str]) -> str | None:
    """
    Map extracted proposition into the existing belief schema.
    Returns canonical proposition string, usually:
      - king_is_wise
      - not_king_is_wise
    Returns None if no safe mapping is found.
    """
    p = prop.strip().lower()

    if not p:
        return None

    # already in canonical positive form
    base = _normalize_base_prop(p)
    if base in allowed_predicates:
        if p.startswith("not_") or p.startswith("~"):
            return f"not_{base}"
        return base

    # lightweight antonym / alias mapping for current MVP
    alias_to_canonical = {
        "king_is_evil": "not_king_is_wise",
        "king_is_bad": "not_king_is_wise",
        "king_is_foolish": "not_king_is_wise",
        "king_is_liar": "not_king_is_wise",
        
        "king_evil": "not_king_is_wise",
        "king_bad": "not_king_is_wise",
        "king_foolish": "not_king_is_wise",
        "king_liar": "not_king_is_wise",
        
        "castle_is_dangerous": "not_castle_is_safe",
        "castle_is_unsafe": "not_castle_is_safe",
        "fortress_is_crumbling": "not_castle_is_safe",
        
        "castle_dangerous": "not_castle_is_safe",
        "castle_unsafe": "not_castle_is_safe",
        "fortress_crumbling": "not_castle_is_safe",
    }

    if p in alias_to_canonical:
        mapped = alias_to_canonical[p]
        mapped_base = _normalize_base_prop(mapped)
        if mapped_base in allowed_predicates:
            return mapped

    return None

def extract_event(user_message: str) -> EventFrame:
    """
    Convert raw dialogue into a structured event frame.

    Attempts to use the configured LLM client. Falls back to a rule-based
    heuristic if the LLM is unavailable or fails.

    Preconditions
    -------------
    user_message : str

    Procedure
    ---------
    1. Check for configured LLM client.
    2. If available, prompt for structured extraction.
    3. If unavailable/fails, use regex/heuristic fallback.

    Postconditions
    --------------
    Returns EventFrame representing the dialogue event

    Parameters
    ----------
    user_message : str

    Returns
    -------
    EventFrame
    """
    # Attempt LLM extraction first
    if llm_client.configure_client():
        try:
            return _extract_event_llm(user_message)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}. Falling back to rules.")
    
    # Fallback to rule-based extraction
    return _extract_event_rules(user_message)


def _extract_event_llm(user_message: str) -> EventFrame:
    """Helper for LLM-based extraction."""
    schema = {
        "type": "object",
        "properties": {
            "propositions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of factual assertions in snake_case (e.g. 'door_is_locked', 'not_safe'). Use 'not_' prefix for negation."
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of proper nouns or entities mentioned."
            },
            "emotional_tone": {
                "type": "string",
                "enum": ["joy", "anger", "sadness", "fear", "trust", "disgust", "surprise", "anticipation", "neutral"],
                "description": "Dominant emotional tone of the speaker."
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in the extraction (0.0 to 1.0)."
            }
        },
        "required": ["propositions", "entities", "emotional_tone", "confidence"]
    }

    prompt = (
        f"Analyze the following dialogue line from a user in a roleplay scenario.\n"
        f"Extract factual propositions (snake_case), mentioned entities, and emotional tone.\n"
        f"User Message: \"{user_message}\"\n"
    )

    result = llm_client.generate_structured(prompt, schema)
    
    if not result:
        raise ValueError("LLM returned empty result")

    return EventFrame(
        propositions=result.get("propositions", []),
        entities=result.get("entities", []),
        speaker="user",
        emotional_tone=result.get("emotional_tone", "neutral"),
        confidence=float(result.get("confidence", 0.5)),
        source_text=user_message
    )


def _extract_event_rules(user_message: str) -> EventFrame:
    """Original rule-based fallback logic."""
    # 1. Basic cleaning and splitting
    message = user_message.strip()
    clauses = re.split(r'[.!?;,]', message)
    clauses = [c.strip() for c in clauses if c.strip()]

    propositions = []
    for clause in clauses:
        # 2. Normalize and handle negation
        words = clause.lower().split()
        is_negated = any(neg in words for neg in ["not", "never", "no", "neither", "nor"]) or "~" in clause
        
        # Remove negation words for the base proposition
        clean_words = [w for w in words if w not in ["not", "never", "no", "a", "an", "the", "is", "are", "was", "were", "has", "have"]]
        prop = "_".join(clean_words).replace("~", "")
        
        if prop:
            if is_negated:
                propositions.append(f"not_{prop}")
            else:
                propositions.append(prop)

    # 3. Entity extraction - improved to catch Proper Nouns and snake_case_entities
    entities = list(set(re.findall(r'\b(?:[A-Z][a-z]+(?:_[A-Z][a-z]+)*|[A-Z]{2,})\b', message)))
    
    # 4. Tone detection
    detected_tone = "neutral"
    message_lower = message.lower()
    for tone, keywords in _TONE_MAP.items():
        if any(word in message_lower for word in keywords):
            detected_tone = tone
            break

    return EventFrame(
        propositions=propositions,
        entities=entities,
        speaker="user",
        emotional_tone=detected_tone,
        confidence=1.0 if propositions else 0.5,
        source_text=user_message
    )


def validate_event(
    event: EventFrame,
    user_message: str,
    allowed_predicates: set[str] | None = None,
) -> EventFrame:
    """
    Validate extraction output and adjust confidence.

    Checks that the event frame has at least one proposition and
    that the confidence is within bounds. Downstream callers should
    treat a returned confidence of 0.0 as an extraction failure.

    Preconditions
    -------------
    event : EventFrame
    user_message : str

    Procedure
    ---------
    1. Check if propositions were extracted
    2. Adjust confidence based on message length vs extraction
    3. Clamp confidence to [0, 1]

    Postconditions
    --------------
    Returns corrected EventFrame

    Parameters
    ----------
    event : EventFrame
    user_message : str

    Returns
    -------
    EventFrame
        Corrected or unchanged EventFrame.
    """
    # If no propositions and no entities, it's a weak extraction
    if not event.propositions and not event.entities:
        event.confidence *= 0.1
    
    # If the message is very long but only one short prop was extracted, lower confidence
    if len(user_message.split()) > 10 and len(event.propositions) == 1:
        event.confidence *= 0.8

    if allowed_predicates is not None:
        canonical_props = []
        for prop in event.propositions:
            mapped = _canonicalize_proposition(prop, allowed_predicates)
            if mapped is not None:
                canonical_props.append(mapped)
        
        # dedupe while preserving order
        seen = set()
        event.propositions = [
            p for p in canonical_props
            if not (p in seen or seen.add(p))
        ]
    
    # Final clamping
    event.confidence = max(0.0, min(1.0, event.confidence))
    
    # If confidence is extremely low, mark as 0
    if event.confidence < 0.1:
        event.confidence = 0.0
        
    return event