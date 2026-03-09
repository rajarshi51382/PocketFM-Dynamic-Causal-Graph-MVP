from core.data_structures import CharacterState, BeliefNode
from reasoning.verifier import verify_dialogue


def test_verifier_allows_consistent_response():
    state = CharacterState()
    state.add_belief(BeliefNode("king_is_wise", log_odds=3.0))

    response = "The king is wise."

    is_valid, violations = verify_dialogue(response, state)

    assert is_valid is True
    assert violations == []


def test_verifier_flags_contradiction_for_is_proposition():
    state = CharacterState()
    state.add_belief(BeliefNode("king_is_wise", log_odds=3.0))

    response = "The king is not wise."

    is_valid, violations = verify_dialogue(response, state)

    assert is_valid is False
    assert "contradicts_belief:king_is_wise" in violations


def test_verifier_flags_world_constraint_violation():
    state = CharacterState(world_constraints=["forbidden_spell"])

    response = "I will use the forbidden_spell now."

    is_valid, violations = verify_dialogue(response, state)

    assert is_valid is False
    assert "violates_world_constraint:forbidden_spell" in violations


def test_verifier_flags_possible_temporal_leakage():
    state = CharacterState()
    state.timeline_index = 5
    state.knowledge_boundary = 5

    response = "Tomorrow, the king will return."

    is_valid, violations = verify_dialogue(response, state)

    assert is_valid is False
    assert "possible_temporal_leakage" in violations