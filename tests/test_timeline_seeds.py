import pytest

from state.timeline_seeds import TIMELINE_SEEDS, create_character_state_for_seed


def test_seed_keys_are_defined():
    assert "baseline" in TIMELINE_SEEDS
    assert "after_betrayal" in TIMELINE_SEEDS
    assert "after_peace" in TIMELINE_SEEDS


def test_seed_populates_beliefs_and_relationships():
    state = create_character_state_for_seed("after_betrayal")
    assert state.timeline_index == TIMELINE_SEEDS["after_betrayal"]["timeline_index"]
    belief = state.get_belief("king_is_wise")
    assert belief is not None
    assert belief.log_odds == pytest.approx(-0.6)
    assert "king" in state.relationships
    assert state.relationships["king"].trust == pytest.approx(0.2)


def test_unknown_seed_raises():
    with pytest.raises(ValueError):
        create_character_state_for_seed("missing_seed")
