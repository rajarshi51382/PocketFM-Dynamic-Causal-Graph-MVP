"""
Event extraction pipeline.

Responsible for converting raw user dialogue into structured event frames
used by the update phase.

Converts raw user dialogue into structured EventFrame objects.
In production this module calls an LLM extraction pass followed by a
validation pass. The stubs below provide a rule-based fallback used
in tests and offline demos.
"""

from core.data_structures import EventFrame


def extract_event(user_message: str) -> EventFrame:
    """
    Convert raw dialogue into a structured event frame.

    In production: sends user_message to a schema-constrained LLM
    extraction prompt and parses the structured JSON response.

    This implementation provides a lightweight rule-based fallback:
    each sentence-delimited clause is treated as a proposition and
    confidence is set to 1.0.

    Preconditions
    -------------
    user_message : str

    Procedure
    ---------
    1. Send message to LLM extraction prompt
    2. Parse structured JSON output
    3. Construct EventFrame

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
    propositions = [
        clause.strip().lower().replace(" ", "_")
        for clause in user_message.split(".")
        if clause.strip()
    ]
    return EventFrame(
        propositions=propositions,
        entities=[],
        speaker=None,
        emotional_tone=None,
        confidence=1.0,
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
    1. Check schema validity
    2. Verify alignment with source text
    3. Adjust confidence score

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
    if not event.propositions:
        event.confidence = 0.0
    event.confidence = max(0.0, min(1.0, event.confidence))
    return event