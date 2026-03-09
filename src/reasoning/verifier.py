"""
This is a lightweight MVP verifier.

Current checks:
- contradiction with high-confidence beliefs
- explicit world-constraint violations
- simple temporal leakage markers

This is not yet a full semantic verifier or planner-critic loop.
"""

from core.data_structures import CharacterState


def _positive_and_negative_forms(prop: str) -> tuple[str, str]:
    """
    Convert a proposition into rough natural-language positive and negative forms.

    Examples
    --------
    door_is_locked -> ("door is locked", "door is not locked")
    king_is_wise   -> ("king is wise", "king is not wise")
    """
    prop = prop.lower().strip()

    if "_is_" in prop:
        subject, predicate = prop.split("_is_", 1)
        subject_text = subject.replace("_", " ")
        predicate_text = predicate.replace("_", " ")
        positive = f"{subject_text} is {predicate_text}"
        negative = f"{subject_text} is not {predicate_text}"
        return positive, negative

    positive = prop.replace("_", " ")
    negative = f"not {positive}"
    return positive, negative


def verify_dialogue(response: str, state: CharacterState) -> tuple[bool, list[str]]:
    """
    Verify whether a generated response is consistent with the
    character's current state.

    Returns
    -------
    (is_valid, violations)
        is_valid : bool
            True if no violations were found.
        violations : list[str]
            List of reasons the response failed verification.
    """
    snapshot = state.verifier_snapshot()
    violations = []
    response_lower = response.lower()

    # 1. Check explicit world constraints
    for constraint in snapshot["world_constraints"]:
        if str(constraint).lower() in response_lower:
            violations.append(f"violates_world_constraint:{constraint}")

    # 2. Check contradictions with high-confidence beliefs
    for belief in snapshot["high_confidence_beliefs"]:
        prop = belief["proposition"].lower()

        if prop.startswith("not_"):
            base_prop = prop[4:]
            base_positive, _ = _positive_and_negative_forms(base_prop)
            if base_positive in response_lower:
                violations.append(f"contradicts_belief:{prop}")
        else:
            _, negative_text = _positive_and_negative_forms(prop)
            if negative_text in response_lower:
                violations.append(f"contradicts_belief:{prop}")

    # 3. Very lightweight temporal leakage heuristic
    kb = snapshot["knowledge_boundary"]
    if kb <= state.timeline_index:
        future_markers = ["tomorrow", "next week", "next year", "in the future"]
        if any(marker in response_lower for marker in future_markers):
            violations.append("possible_temporal_leakage")

    return len(violations) == 0, violations