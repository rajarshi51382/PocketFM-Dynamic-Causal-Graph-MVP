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
from core.data_structures import EventFrame


def extract_event(user_message: str) -> EventFrame:
    """
    Convert raw dialogue into a structured event frame.

    In production: sends user_message to a schema-constrained LLM
    extraction prompt and parses the structured JSON response.

    This implementation provides a rule-based fallback:
    - Splits message into clauses by punctuation.
    - Normalises propositions into snake_case.
    - Handles basic negation (not, never, ~).
    - Extracts capitalized words as potential entities.
    - Infers emotional tone from keywords.

    Preconditions
    -------------
    user_message : str

    Procedure
    ---------
    1. Pre-process text and split into clauses.
    2. For each clause, extract proposition and handle negation.
    3. Extract entities using regex.
    4. Map keywords to emotional tone labels.
    5. Construct and return EventFrame.

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
    # 1. Basic cleaning and splitting
    message = user_message.strip()
    clauses = re.split(r'[.!?;,]', message)
    clauses = [c.strip() for c in clauses if c.strip()]

    propositions = []
    for clause in clauses:
        # 2. Normalize and handle negation
        # "The door is not locked" -> "not_door_is_locked"
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

    # 3. Entity extraction (Capitalized words, excluding start of sentence if not Proper Noun)
    # Simple heuristic: any word starting with Capital letter that isn't the first word, 
    # OR first word if it looks like a name.
    entities = list(set(re.findall(r'\b[A-Z][a-z]+\b', message)))

    # 4. Tone detection
    tone_map = {
        "joy": ["happy", "good", "great", "joy", "wonderful", "excellent", "thanks", "thank"],
        "anger": ["angry", "mad", "hate", "furious", "annoyed", "stop"],
        "sadness": ["sad", "bad", "unfortunate", "sorry", "regret", "pity"],
        "fear": ["scared", "fear", "afraid", "terrified", "worry", "worried"],
        "trust": ["trust", "believe", "agree", "sure", "certain"],
        "disgust": ["gross", "disgust", "hate", "eww"],
        "surprise": ["wow", "really", "unexpected", "surprise"],
        "anticipation": ["hope", "expect", "wait", "looking forward"]
    }
    
    detected_tone = "neutral"
    message_lower = message.lower()
    for tone, keywords in tone_map.items():
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


def validate_event(event: EventFrame, user_message: str) -> EventFrame:
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

    # Final clamping
    event.confidence = max(0.0, min(1.0, event.confidence))
    
    # If confidence is extremely low, mark as 0
    if event.confidence < 0.1:
        event.confidence = 0.0
        
    return event