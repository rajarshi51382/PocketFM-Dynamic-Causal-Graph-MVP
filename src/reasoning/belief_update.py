"""
Belief update module.

Implements the log-odds belief revision policy described in the
Dynamic Causal Character Graphs paper (Section 3.3).

The central update rule is:

    l_{t,k} = l_{t-1,k} + lambda_base * sigma(e_t) * rho(src) * delta(e_t, phi_k) * c_t

where:
    lambda_base     -- base learning rate
    sigma(e_t)      -- narrative importance (shock-gated plasticity, >= 1)
    rho(src)        -- source credibility in [0, 1]
    delta(e_t, phi) -- directional alignment in {-1, 0, +1}
    c_t             -- extraction confidence in [0, 1]
"""

import math
from typing import Dict

from core.data_structures import BeliefNode, CharacterState, EventFrame

DIRECT_OBSERVATION = "DirectObservation"


def directional_alignment(event: EventFrame, belief: BeliefNode) -> int:
    """
    Determine whether an event supports or contradicts a belief.

    Compares each proposition in the event frame against the belief
    proposition using a deterministic predicate comparison. Negations
    are expressed with the prefix "not_" or the tilde prefix "~".

    Parameters
    ----------
    event : EventFrame
    belief : BeliefNode

    Returns
    -------
    int
        +1 if the event asserts the belief proposition.
        -1 if the event asserts its negation.
         0 if unrelated.
    """
    proposition = belief.proposition.strip().lower()
    negation = _negation_of(proposition)

    for p in event.propositions:
        p_norm = p.strip().lower()
        if p_norm == proposition:
            return +1
        if p_norm == negation or _negation_of(p_norm) == proposition:
            return -1

    return 0


def compute_source_credibility(event: EventFrame, state: CharacterState) -> float:
    """
    Estimate the credibility of the event's information source.

    Returns 1.0 for direct observations. For character or user sources,
    the trust value is retrieved from the character's relationship graph.
    Falls back to 0.5 for unknown sources.

    Parameters
    ----------
    event : EventFrame
    state : CharacterState

    Returns
    -------
    float
        Credibility value in [0, 1].
    """
    if event.speaker is None or event.speaker == DIRECT_OBSERVATION:
        return 1.0

    rel = state.relationships.get(event.speaker)
    if rel is not None:
        return float(rel.trust)

    return 0.5


def update_belief_log_odds(
    belief: BeliefNode,
    event: EventFrame,
    credibility: float,
    lambda_base: float,
    narrative_importance: float = 1.0,
) -> None:
    """
    Apply the log-odds belief update rule in place.

    Update rule:
        l_{t,k} = l_{t-1,k}
                  + lambda_base * narrative_importance * credibility
                    * delta(e_t, phi_k) * c_t

    No-op when the event is directionally unrelated to the belief.
    The event speaker is appended to belief.evidence_sources on update.

    Parameters
    ----------
    belief : BeliefNode
        Modified in place.
    event : EventFrame
    credibility : float
        Source credibility rho(src) in [0, 1].
    lambda_base : float
        Base learning rate.
    narrative_importance : float
        Shock-gated plasticity factor sigma(e_t), default 1.0.
    """
    delta = directional_alignment(event, belief)
    if delta == 0:
        return

    update = lambda_base * narrative_importance * credibility * delta * event.confidence
    belief.log_odds += update

    source = event.speaker if event.speaker is not None else DIRECT_OBSERVATION
    belief.evidence_sources.append(source)


def resolve_belief_conflicts(beliefs: Dict[str, BeliefNode]) -> Dict[str, BeliefNode]:
    """
    Resolve mutually exclusive belief pairs via pairwise normalisation.

    Two beliefs conflict when one proposition is the negation of the
    other. For a conflicting pair (b_i, b_j) the probabilities are
    renormalised:

        p'_i = p_i / (p_i + p_j),  p'_j = p_j / (p_i + p_j)

    The log-odds are then updated to reflect the renormalised values.

    Parameters
    ----------
    beliefs : Dict[str, BeliefNode]
        Keyed by normalised proposition string.

    Returns
    -------
    Dict[str, BeliefNode]
        The same dictionary with conflicting pairs resolved.
    """
    resolved: set = set()

    for prop in list(beliefs.keys()):
        negation = _negation_of(prop)
        pair = frozenset({prop, negation})

        if negation in beliefs and pair not in resolved:
            b_pos = beliefs[prop]
            b_neg = beliefs[negation]

            prob_pos = _log_odds_to_prob(b_pos.log_odds)
            prob_neg = _log_odds_to_prob(b_neg.log_odds)
            denom = prob_pos + prob_neg

            if denom > 0:
                b_pos.log_odds = _prob_to_log_odds(prob_pos / denom)
                b_neg.log_odds = _prob_to_log_odds(prob_neg / denom)
            else:
                b_pos.log_odds = 0.0
                b_neg.log_odds = 0.0

            resolved.add(pair)

    return beliefs


def apply_belief_updates(
    state: CharacterState,
    event: EventFrame,
    lambda_base: float = 0.5,
    narrative_importance: float = 1.0,
) -> None:
    """
    Update all relevant beliefs in the character state.

    Pipeline:
    1. Compute source credibility from the event speaker.
    2. Apply the log-odds update to every belief that the event touches.
    3. Resolve any resulting contradictions via pairwise normalisation.

    Parameters
    ----------
    state : CharacterState
        Modified in place.
    event : EventFrame
    lambda_base : float
        Base learning rate, default 0.5.
    narrative_importance : float
        Shock-gated plasticity factor sigma(e_t), default 1.0.
    """
    credibility = compute_source_credibility(event, state)

    for belief in state.beliefs.values():
        update_belief_log_odds(
            belief,
            event,
            credibility,
            lambda_base,
            narrative_importance,
        )

    resolve_belief_conflicts(state.beliefs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _negation_of(proposition: str) -> str:
    """Return the canonical negation of a proposition string."""
    p = proposition.strip().lower()
    if p.startswith("not_"):
        return p[4:]
    if p.startswith("~"):
        return p[1:]
    return "not_" + p


def _log_odds_to_prob(log_odds: float) -> float:
    """Convert log-odds to probability via the sigmoid function."""
    return 1.0 / (1.0 + math.exp(-log_odds))


def _prob_to_log_odds(prob: float, eps: float = 1e-9) -> float:
    """Convert probability to log-odds, clamped away from 0 and 1."""
    prob = max(eps, min(1.0 - eps, prob))
    return math.log(prob / (1.0 - prob))