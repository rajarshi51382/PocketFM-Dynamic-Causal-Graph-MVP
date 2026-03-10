"""
Tests for src/reasoning/belief_update.py

Covers:
- directional_alignment
- compute_source_credibility
- update_belief_log_odds
- resolve_belief_conflicts
- apply_belief_updates
"""

import math
import pytest

from core.data_structures import BeliefNode, CharacterState, EventFrame, RelationshipState
from reasoning.belief_update import (
    DIRECT_OBSERVATION,
    _log_odds_to_prob,
    _negation_of,
    _prob_to_log_odds,
    apply_belief_updates,
    compute_source_credibility,
    directional_alignment,
    resolve_belief_conflicts,
    update_belief_log_odds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(beliefs=None, relationships=None):
    state = CharacterState()
    state.beliefs = beliefs or {}
    state.relationships = relationships or {}
    return state


def make_event(propositions=None, speaker=None, confidence=1.0, emotional_tone=None):
    return EventFrame(
        propositions=propositions or [],
        entities=[],
        speaker=speaker,
        emotional_tone=emotional_tone,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# directional_alignment
# ---------------------------------------------------------------------------

class TestDirectionalAlignment:
    def test_supports_belief(self):
        event = make_event(["door_is_locked"])
        belief = BeliefNode("door_is_locked", 0.0)
        assert directional_alignment(event, belief) == +1

    def test_contradicts_belief_with_not_prefix(self):
        event = make_event(["not_door_is_locked"])
        belief = BeliefNode("door_is_locked", 0.0)
        assert directional_alignment(event, belief) == -1

    def test_contradicts_positive_from_negated_belief(self):
        event = make_event(["door_is_locked"])
        belief = BeliefNode("not_door_is_locked", 0.0)
        assert directional_alignment(event, belief) == -1

    def test_unrelated_event_returns_zero(self):
        event = make_event(["sky_is_blue"])
        belief = BeliefNode("door_is_locked", 0.0)
        assert directional_alignment(event, belief) == 0

    def test_empty_propositions_returns_zero(self):
        event = make_event([])
        belief = BeliefNode("door_is_locked", 0.0)
        assert directional_alignment(event, belief) == 0

    def test_tilde_negation_syntax(self):
        event = make_event(["~door_is_locked"])
        belief = BeliefNode("door_is_locked", 0.0)
        assert directional_alignment(event, belief) == -1

    def test_case_insensitive_match(self):
        event = make_event(["DOOR_IS_LOCKED"])
        belief = BeliefNode("door_is_locked", 0.0)
        assert directional_alignment(event, belief) == +1


# ---------------------------------------------------------------------------
# compute_source_credibility
# ---------------------------------------------------------------------------

class TestComputeSourceCredibility:
    def test_direct_observation_returns_one(self):
        event = make_event(speaker=DIRECT_OBSERVATION)
        state = make_state()
        assert compute_source_credibility(event, state) == 1.0

    def test_none_speaker_returns_one(self):
        event = make_event(speaker=None)
        state = make_state()
        assert compute_source_credibility(event, state) == 1.0

    def test_known_speaker_uses_trust(self):
        rel = RelationshipState(trust=0.7)
        state = make_state(relationships={"ally": rel})
        event = make_event(speaker="ally")
        assert compute_source_credibility(event, state) == pytest.approx(0.7)

    def test_unknown_speaker_returns_half(self):
        state = make_state()
        event = make_event(speaker="stranger")
        assert compute_source_credibility(event, state) == pytest.approx(0.5)

    def test_zero_trust_speaker(self):
        rel = RelationshipState(trust=0.0)
        state = make_state(relationships={"enemy": rel})
        event = make_event(speaker="enemy")
        assert compute_source_credibility(event, state) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# update_belief_log_odds
# ---------------------------------------------------------------------------

class TestUpdateBeliefLogOdds:
    def test_positive_alignment_increases_log_odds(self):
        belief = BeliefNode("door_is_locked", 0.0)
        event = make_event(["door_is_locked"], confidence=1.0)
        update_belief_log_odds(belief, event, credibility=1.0, lambda_base=1.0)
        assert belief.log_odds > 0.0

    def test_negative_alignment_decreases_log_odds(self):
        belief = BeliefNode("door_is_locked", 0.0)
        event = make_event(["not_door_is_locked"], confidence=1.0)
        update_belief_log_odds(belief, event, credibility=1.0, lambda_base=1.0)
        assert belief.log_odds < 0.0

    def test_unrelated_event_no_change(self):
        belief = BeliefNode("door_is_locked", 1.5)
        event = make_event(["sky_is_blue"], confidence=1.0)
        update_belief_log_odds(belief, event, credibility=1.0, lambda_base=1.0)
        assert belief.log_odds == pytest.approx(1.5)

    def test_credibility_zero_no_change(self):
        belief = BeliefNode("door_is_locked", 0.0)
        event = make_event(["door_is_locked"], confidence=1.0)
        update_belief_log_odds(belief, event, credibility=0.0, lambda_base=1.0)
        assert belief.log_odds == pytest.approx(0.0)

    def test_confidence_zero_no_change(self):
        belief = BeliefNode("door_is_locked", 0.0)
        event = make_event(["door_is_locked"], confidence=0.0)
        update_belief_log_odds(belief, event, credibility=1.0, lambda_base=1.0)
        assert belief.log_odds == pytest.approx(0.0)

    def test_update_magnitude_formula(self):
        belief = BeliefNode("door_is_locked", 0.0)
        event = make_event(["door_is_locked"], confidence=0.8)
        update_belief_log_odds(
            belief, event, credibility=0.7, lambda_base=0.5, narrative_importance=2.0
        )
        expected = 0.5 * 2.0 * 0.7 * 1 * 0.8
        assert belief.log_odds == pytest.approx(expected)

    def test_evidence_source_appended(self):
        belief = BeliefNode("door_is_locked", 0.0)
        event = make_event(["door_is_locked"], speaker="guard")
        update_belief_log_odds(belief, event, credibility=1.0, lambda_base=1.0)
        assert "guard" in belief.evidence_sources

    def test_no_evidence_source_on_no_update(self):
        belief = BeliefNode("door_is_locked", 0.0)
        event = make_event(["sky_is_blue"], speaker="guard")
        update_belief_log_odds(belief, event, credibility=1.0, lambda_base=1.0)
        assert len(belief.evidence_sources) == 0


# ---------------------------------------------------------------------------
# resolve_belief_conflicts
# ---------------------------------------------------------------------------

class TestResolveBeliefConflicts:
    def test_conflicting_pair_normalised(self):
        # Start with equal confidence so both should end at 0 (p=0.5)
        beliefs = {
            "door_is_locked": BeliefNode("door_is_locked", 0.0),
            "not_door_is_locked": BeliefNode("not_door_is_locked", 0.0),
        }
        resolve_belief_conflicts(beliefs)
        # Equal probs -> each stays at 0.5 -> log-odds remain ~0
        assert beliefs["door_is_locked"].log_odds == pytest.approx(0.0, abs=1e-6)
        assert beliefs["not_door_is_locked"].log_odds == pytest.approx(0.0, abs=1e-6)

    def test_stronger_belief_wins_after_normalisation(self):
        beliefs = {
            "door_is_locked": BeliefNode("door_is_locked", 2.0),
            "not_door_is_locked": BeliefNode("not_door_is_locked", -2.0),
        }
        resolve_belief_conflicts(beliefs)
        assert beliefs["door_is_locked"].log_odds > beliefs["not_door_is_locked"].log_odds

    def test_non_conflicting_beliefs_unchanged(self):
        beliefs = {
            "door_is_locked": BeliefNode("door_is_locked", 1.0),
            "sky_is_blue": BeliefNode("sky_is_blue", 0.5),
        }
        resolve_belief_conflicts(beliefs)
        assert beliefs["door_is_locked"].log_odds == pytest.approx(1.0)
        assert beliefs["sky_is_blue"].log_odds == pytest.approx(0.5)

    def test_probabilities_sum_to_one_after_resolution(self):
        beliefs = {
            "x": BeliefNode("x", 1.5),
            "not_x": BeliefNode("not_x", 0.3),
        }
        resolve_belief_conflicts(beliefs)
        p_x = _log_odds_to_prob(beliefs["x"].log_odds)
        p_not_x = _log_odds_to_prob(beliefs["not_x"].log_odds)
        assert p_x + p_not_x == pytest.approx(1.0, abs=1e-6)

    def test_empty_beliefs_no_error(self):
        resolve_belief_conflicts({})

    def test_returns_same_dict(self):
        beliefs = {"a": BeliefNode("a", 0.0)}
        result = resolve_belief_conflicts(beliefs)
        assert result is beliefs


# ---------------------------------------------------------------------------
# apply_belief_updates
# ---------------------------------------------------------------------------

class TestApplyBeliefUpdates:
    def test_supporting_event_increases_belief(self):
        state = make_state(
            beliefs={"door_is_locked": BeliefNode("door_is_locked", 0.0)}
        )
        event = make_event(["door_is_locked"], confidence=1.0)
        apply_belief_updates(state, event, lambda_base=0.5)
        assert state.beliefs["door_is_locked"].log_odds > 0.0

    def test_contradicting_event_decreases_belief(self):
        state = make_state(
            beliefs={"door_is_locked": BeliefNode("door_is_locked", 1.0)}
        )
        event = make_event(["not_door_is_locked"], confidence=1.0)
        apply_belief_updates(state, event, lambda_base=0.5)
        assert state.beliefs["door_is_locked"].log_odds < 1.0

    def test_credibility_propagates_from_relationships(self):
        rel = RelationshipState(trust=0.1)
        state = make_state(
            beliefs={"x": BeliefNode("x", 0.0)},
            relationships={"liar": rel},
        )
        event = make_event(["x"], speaker="liar", confidence=1.0)
        apply_belief_updates(state, event, lambda_base=1.0)
        assert state.beliefs["x"].log_odds == pytest.approx(0.1)

    def test_conflict_resolution_runs_after_update(self):
        state = make_state(
            beliefs={
                "door_is_locked": BeliefNode("door_is_locked", 0.5),
                "not_door_is_locked": BeliefNode("not_door_is_locked", -0.5),
            }
        )
        event = make_event(["door_is_locked"], confidence=1.0)
        apply_belief_updates(state, event, lambda_base=0.5)

        p_pos = _log_odds_to_prob(state.beliefs["door_is_locked"].log_odds)
        p_neg = _log_odds_to_prob(state.beliefs["not_door_is_locked"].log_odds)
        assert p_pos + p_neg == pytest.approx(1.0, abs=1e-6)

    def test_no_beliefs_no_error(self):
        state = make_state()
        event = make_event(["x"])
        apply_belief_updates(state, event)

    def test_self_generated_dialogue_blocked(self):
        state = make_state(
            beliefs={"x": BeliefNode("x", 0.0)},
        )
        # Source with zero trust should produce no significant update
        rel = RelationshipState(trust=0.0)
        state.relationships["self"] = rel
        event = make_event(["x"], speaker="self", confidence=1.0)
        apply_belief_updates(state, event, lambda_base=1.0)
        assert state.beliefs["x"].log_odds == pytest.approx(0.0)
    
    def test_schema_canonicalization_allows_negative_evidence_to_hit_existing_belief(self):
        from extraction.event_extraction import validate_event
        
        state = CharacterState()
        state.add_belief(BeliefNode("king_is_wise", 1.0))
        state.add_belief(BeliefNode("castle_is_safe", 1.0))
        
        raw_event = EventFrame(
            propositions=["king_is_evil"],
            entities=[],
            speaker="user",
            confidence=1.0,
        )
        
        event = validate_event(
            raw_event,
            "The king is evil.",
            allowed_predicates=state.belief_schema,
        )
        
        assert "not_king_is_wise" in event.propositions
        
        before = state.get_belief("king_is_wise").log_odds
        apply_belief_updates(state, event, lambda_base=0.5)
        after = state.get_belief("king_is_wise").log_odds
        
        assert after < before
    
    def test_counterfactual_parent_intervention_reduces_inflation(self):
        from reasoning.causal_propagation import propagate_causal_effects
        
        state = CharacterState()
        state.add_belief(BeliefNode("castle_is_safe", 2.0))
        state.add_belief(BeliefNode("king_is_wise", 0.0))
        state.add_causal_link("castle_is_safe", "king_is_wise", weight=1.0)
        
        factual = state.copy()
        propagate_causal_effects(factual)
        factual_score = factual.get_belief("king_is_wise").log_odds
        
        counterfactual = state.copy()
        counterfactual.get_belief("castle_is_safe").log_odds = 0.0
        propagate_causal_effects(counterfactual)
        counterfactual_score = counterfactual.get_belief("king_is_wise").log_odds
        
        assert factual_score > counterfactual_score

# ---------------------------------------------------------------------------
# Internal utility functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_log_odds_to_prob_zero(self):
        assert _log_odds_to_prob(0.0) == pytest.approx(0.5)

    def test_log_odds_to_prob_positive(self):
        assert _log_odds_to_prob(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_log_odds_to_prob_negative(self):
        assert _log_odds_to_prob(-100.0) == pytest.approx(0.0, abs=1e-6)

    def test_prob_to_log_odds_half(self):
        assert _prob_to_log_odds(0.5) == pytest.approx(0.0, abs=1e-6)

    def test_roundtrip(self):
        for lo in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            assert _log_odds_to_prob(_prob_to_log_odds(_log_odds_to_prob(lo))) == pytest.approx(
                _log_odds_to_prob(lo), abs=1e-6
            )

    def test_negation_of_plain(self):
        assert _negation_of("door_is_locked") == "not_door_is_locked"

    def test_negation_of_prefixed(self):
        assert _negation_of("not_door_is_locked") == "door_is_locked"

    def test_negation_of_tilde(self):
        assert _negation_of("~door_is_locked") == "door_is_locked"
