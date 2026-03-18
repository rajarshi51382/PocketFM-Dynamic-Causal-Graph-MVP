"""
Seeded character states for timeline-based testing.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from core.data_structures import BeliefNode, CharacterState, RelationshipState, TraitState


def _default_seeds() -> Dict[str, dict]:
    return {
        "baseline": {
            "label": "Baseline",
            "timeline_index": 0,
            "beliefs": {
                "castle_is_safe": 1.5,
                "forest_is_dangerous": 1.0,
                "king_is_wise": 0.5,
            },
            "relationships": {},
        },
        "after_betrayal": {
            "label": "After betrayal",
            "timeline_index": 3,
            "beliefs": {
                "castle_is_safe": 0.3,
                "forest_is_dangerous": 1.2,
                "king_is_wise": -0.6,
            },
            "relationships": {
                "king": {"trust": 0.2, "affection": 0.2, "respect": 0.3},
            },
        },
        "after_peace": {
            "label": "After peace",
            "timeline_index": 6,
            "beliefs": {
                "castle_is_safe": 1.2,
                "forest_is_dangerous": 0.2,
                "king_is_wise": 0.8,
            },
            "relationships": {
                "king": {"trust": 0.7, "affection": 0.6, "respect": 0.7},
            },
        },
    }


def load_timeline_seeds(path: Optional[str] = None) -> Dict[str, dict]:
    """
    Load timeline seeds from JSON, falling back to defaults if missing.
    """
    if path is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        path = os.path.join(root, "data", "timeline_seeds.json")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Timeline seeds JSON must be a dict")
        return data
    except FileNotFoundError:
        return _default_seeds()


TIMELINE_SEEDS: Dict[str, dict] = load_timeline_seeds()


def create_character_state_for_seed(
    seed_key: str,
    seeds: Optional[Dict[str, dict]] = None,
) -> CharacterState:
    """
    Create a CharacterState from a timeline seed.
    """
    seeds = seeds or TIMELINE_SEEDS
    if seed_key not in seeds:
        raise ValueError(f"Unknown timeline seed: {seed_key}")

    seed = seeds[seed_key]
    traits_data = seed.get("traits", {
        "bravery": 0.8,
        "honesty": 0.6,
        "neuroticism": 0.4,
        "trusting": 0.2,
    })
    traits = TraitState(traits=traits_data)

    beliefs = {
        key: BeliefNode(key, log_odds=value)
        for key, value in seed.get("beliefs", {}).items()
    }
    relationships = {
        entity: RelationshipState(entity_id=entity, **values)
        for entity, values in seed.get("relationships", {}).items()
    }

    character_id = seed.get("character_id", "Sir_Galahad")

    state = CharacterState(
        character_id=character_id,
        traits=traits,
        beliefs=beliefs,
        relationships=relationships,
        timeline_index=seed.get("timeline_index", 0),
        knowledge_boundary=seed.get("knowledge_boundary", seed.get("timeline_index", 0)),
    )
    
    causal_links = seed.get("causal_links")
    if causal_links is not None:
        for link in causal_links:
            state.add_causal_link(link["antecedent"], link["consequent"], weight=link.get("weight", 1.0))
    else:
        state.add_causal_link("castle_is_safe", "king_is_wise", weight=0.8)
        state.add_causal_link("forest_is_dangerous", "castle_is_safe", weight=0.5)
        state.add_causal_link("not_castle_is_safe", "not_king_is_wise", weight=0.8)

    world_constraints = seed.get("world_constraints", [])
    state.world_constraints = world_constraints

    return state
