"""
Defines the shared schemas used throughout the system.

These structures represent the causal character graph state,
world state, and structured event frames as defined in the
Dynamic Causal Character Graphs paper.
"""

from __future__ import annotations

import math
import copy
from typing import Dict, List, Optional, Any, Set

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a scalar to [lo, hi]."""
    return max(lo, min(hi, float(value)))


def _validate_unit(value: float, name: str) -> float:
    """Validate and clamp a value to [0, 1]."""
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    return _clamp(v, 0.0, 1.0)


def _validate_signed_unit(value: float, name: str) -> float:
    """Validate and clamp a value to [-1, 1]."""
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    return _clamp(v, -1.0, 1.0)

# ===================================================================
# TraitState
# ===================================================================

class TraitState:
    """
    Represents stable personality traits.
    
    τ_j ∈ [-1, 1] or [0, 1].
    These nodes have low plasticity and evolve slowly over time.
    Ex: bravery, honesty, curiosity, risk-aversion.

    Attributes
    ----------
    traits : dict[str, float]
        Mapping of trait name → intensity (clamped to [-1, 1]).
    plasticity : float
        Base plasticity α ∈ [0, 1] — should be small for traits.
    """

    def __init__(self, traits: Dict[str, float], plasticity: float = 0.05):
        if not isinstance(traits, dict):
            raise TypeError("traits must be a dict")
        # Store each trait clamped to [-1, 1]
        self.traits: Dict[str, float] = {
            str(k): _validate_signed_unit(v, f"trait '{k}'")
            for k, v in traits.items()
        }
        self.plasticity: float = _validate_unit(plasticity, "plasticity")

    # -- accessors ---------------------------------------------------------

    def get(self, name: str, default: float = 0.0) -> float:
        """Return trait intensity, or *default* if the trait is absent."""
        return self.traits.get(name, default)

    def set_trait(self, name: str, value: float) -> None:
        """Set a single trait (clamped)."""
        self.traits[str(name)] = _validate_signed_unit(value, f"trait '{name}'")

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {"traits": dict(self.traits), "plasticity": self.plasticity}

    @classmethod
    def from_dict(cls, d: dict) -> "TraitState":
        return cls(traits=d["traits"], plasticity=d.get("plasticity", 0.05))

    def __repr__(self) -> str:
        return f"TraitState(traits={self.traits}, α={self.plasticity})"

# ===================================================================
# EmotionState
# ===================================================================

class EmotionState:
    """
    Low-dimensional affective state.
    --------------
    e_t = (v_t, a_t)  where v = valence, a = arousal.
    Discrete emotional tags (fear, anger, joy …) are intensity
    values in [0, 1].  Emotions are FAST nodes (large α).

    Attributes
    ----------
    valence : float
        Positive/negative affect in [-1, 1].
    arousal : float
        Activation level in [0, 1].
    emotion_tags : Dict[str, float]
        Named emotion intensities in [0, 1].
    plasticity : float
        Rate at which emotions update. Fast nodes have high plasticity.
    """

    def __init__(
        self,
        valence: float = 0.0,
        arousal: float = 0.0,
        emotion_tags: Optional[Dict[str, float]] = None,
        plasticity: float = 0.8,
    ):
        self.valence: float = _validate_signed_unit(valence, "valence")
        self.arousal: float = _validate_unit(arousal, "arousal")
        self.emotion_tags: Dict[str, float] = {}
        if emotion_tags:
            for k, v in emotion_tags.items():
                self.emotion_tags[str(k)] = _validate_unit(v, f"emotion_tag '{k}'")
        self.plasticity: float = _validate_unit(plasticity, "plasticity")

    # -- convenience -------------------------------------------------------

    def dominant_emotion(self) -> Optional[str]:
        """Return the tag with highest intensity, or None."""
        if not self.emotion_tags:
            return None
        return max(self.emotion_tags, key=self.emotion_tags.get)

    def set_tag(self, tag: str, intensity: float) -> None:
        self.emotion_tags[str(tag)] = _validate_unit(intensity, f"emotion_tag '{tag}'")

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "emotion_tags": dict(self.emotion_tags),
            "plasticity": self.plasticity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmotionState":
        return cls(
            valence=d.get("valence", 0.0),
            arousal=d.get("arousal", 0.0),
            emotion_tags=d.get("emotion_tags"),
            plasticity=d.get("plasticity", 0.8),
        )

    def __repr__(self) -> str:
        dom = self.dominant_emotion()
        return (
            f"EmotionState(v={self.valence:.2f}, a={self.arousal:.2f}, "
            f"dominant={dom}, α={self.plasticity})"
        )

# ===================================================================
# RelationshipState
# ===================================================================

class RelationshipState:
    """
    Represents relationship values with another entity.

    --------------
    R_t(x) = (trust, affection, respect) ∈ [0, 1]³.
    Semi-stable nodes — moderate plasticity.

    Attributes
    ----------
    entity_id : str       who this relationship is with
    trust     : float     ∈ [0, 1]
    affection : float     ∈ [0, 1]
    respect   : float     ∈ [0, 1]
    plasticity: float     ∈ [0, 1]
    """

    def __init__(
        self,
        entity_id: str = "",
        trust: float = 0.5,
        affection: float = 0.5,
        respect: float = 0.5,
        plasticity: float = 0.3,
    ):
        self.entity_id: str = str(entity_id)
        self.trust: float = _validate_unit(trust, "trust")
        self.affection: float = _validate_unit(affection, "affection")
        self.respect: float = _validate_unit(respect, "respect")
        self.plasticity: float = _validate_unit(plasticity, "plasticity")

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "trust": self.trust,
            "affection": self.affection,
            "respect": self.respect,
            "plasticity": self.plasticity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RelationshipState":
        return cls(**d)

    def __repr__(self) -> str:
        return (
            f"RelationshipState(entity={self.entity_id!r}, "
            f"trust={self.trust:.2f}, aff={self.affection:.2f}, "
            f"resp={self.respect:.2f}, α={self.plasticity})"
        )

# ===================================================================
# BeliefNode
# ===================================================================

class BeliefNode:
    """
    Represents a belief proposition using log-odds encoding.

    b_k = (φ_k, ℓ_{t,k}, S_k)
      φ_k  : propositional content (e.g. "Player_has_Item(Key)")
      ℓ     : epistemic confidence as log-odds (logit)
              ℓ = ln(p / (1-p));  ℓ = 0 → p = 0.5 (max uncertainty)
      S_k   : provenance metadata (source tracking)

    A value of 0 represents maximum uncertainty (p = 0.5).
    Beliefs are SEMI-STABLE nodes (moderate plasticity).

    Attributes
    ----------
    proposition      : str    the propositional content φ_k
    log_odds         : float  ℓ_{t,k} — unbounded real
    evidence_sources : list   provenance identifiers S_k
    plasticity       : float  ∈ [0, 1]
    """

    def __init__(
        self,
        proposition: str,
        log_odds: float = 0.0,
        evidence_sources: Optional[List[str]] = None,
        plasticity: float = 0.4,
    ):
        if not proposition:
            raise ValueError("proposition must be a non-empty string")
        self.proposition: str = str(proposition)
        if not math.isfinite(float(log_odds)):
            raise ValueError(f"log_odds must be finite, got {log_odds}")
        self.log_odds: float = float(log_odds)
        self.evidence_sources: List[str] = list(evidence_sources or [])
        self.plasticity: float = _validate_unit(plasticity, "plasticity")

    # -- probability conversion -----------------------------

    @property
    def probability(self) -> float:
        """Convert log-odds → probability via sigmoid: p = 1/(1+exp(-ℓ))."""
        # Numerically stable sigmoid
        if self.log_odds >= 0:
            z = math.exp(-self.log_odds)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(self.log_odds)
            return z / (1.0 + z)

    @probability.setter
    def probability(self, p: float) -> None:
        """Set confidence via probability (auto-converts to log-odds)."""
        p = _clamp(float(p), 1e-9, 1.0 - 1e-9)  # avoid log(0)
        self.log_odds = math.log(p / (1.0 - p))

    # -- provenance --------------------------------------------------------

    def add_evidence(self, source_id: str) -> None:
        """Append a provenance identifier."""
        if source_id not in self.evidence_sources:
            self.evidence_sources.append(source_id)

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "proposition": self.proposition,
            "log_odds": self.log_odds,
            "probability": round(self.probability, 6),
            "evidence_sources": list(self.evidence_sources),
            "plasticity": self.plasticity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BeliefNode":
        return cls(
            proposition=d["proposition"],
            log_odds=d.get("log_odds", 0.0),
            evidence_sources=d.get("evidence_sources"),
            plasticity=d.get("plasticity", 0.4),
        )

    def __repr__(self) -> str:
        return (
            f"BeliefNode('{self.proposition}', "
            f"p={self.probability:.3f}, ℓ={self.log_odds:.3f}, "
            f"sources={len(self.evidence_sources)})"
        )




class CharacterState:
    """
    Full internal character state in the Dynamic SCM.

    --------------
    Nodes: traits, emotions, beliefs, relationships, intentions,
           timeline state, world constraints.
    Canonical edge ordering:
      Traits → Beliefs → Emotions → Intentions → Dialogue
    (with feedback across time steps via the dynamic causal model).

    The graph is intentionally small (tens of nodes) to ensure
    stable long-horizon behavior.

    Attributes
    ----------
    character_id        : str
    traits              : TraitState
    emotions            : EmotionState
    beliefs             : dict[str, BeliefNode]   keyed by proposition
    relationships       : dict[str, RelationshipState]  keyed by entity_id
    intentions          : list[str]               current behavioral goals
    timeline_index      : int                     T_t ∈ ℤ
    knowledge_boundary  : int                     max accessible time index
    world_constraints   : list[str]               hard narrative rules
    """

    def __init__(
        self,
        character_id: str = "default",
        traits: Optional[TraitState] = None,
        emotions: Optional[EmotionState] = None,
        beliefs: Optional[Dict[str, BeliefNode]] = None,
        relationships: Optional[Dict[str, RelationshipState]] = None,
        intentions: Optional[List[str]] = None,
        timeline_index: int = 0,
        knowledge_boundary: int = 0,
        world_constraints: Optional[List[str]] = None,
    ):
        self.character_id: str = str(character_id)
        self.traits: TraitState = traits or TraitState(traits={})
        self.emotions: EmotionState = emotions or EmotionState()
        self.beliefs: Dict[str, BeliefNode] = dict(beliefs or {})
        self.relationships: Dict[str, RelationshipState] = dict(relationships or {})
        self.intentions: List[str] = list(intentions or [])
        self.timeline_index: int = int(timeline_index)
        self.knowledge_boundary: int = int(knowledge_boundary)
        self.world_constraints: List[str] = list(world_constraints or [])

    # -- belief helpers ----------------------------------------------------

    def add_belief(self, belief: BeliefNode) -> None:
        """Insert or overwrite a belief keyed by its proposition."""
        self.beliefs[belief.proposition] = belief

    def get_belief(self, proposition: str) -> Optional[BeliefNode]:
        return self.beliefs.get(proposition)

    def high_confidence_beliefs(self, threshold: float = 0.7) -> List[BeliefNode]:
        """Return beliefs with probability above *threshold*."""
        return [b for b in self.beliefs.values() if b.probability >= threshold]

    # -- relationship helpers ----------------------------------------------

    def add_relationship(self, rel: RelationshipState) -> None:
        self.relationships[rel.entity_id] = rel

    def get_relationship(self, entity_id: str) -> Optional[RelationshipState]:
        return self.relationships.get(entity_id)

    # -- temporal helpers (paper §3.1.1 Timeline State) --------------------

    def advance_timeline(self, steps: int = 1) -> None:
        self.timeline_index += steps

    def can_know_event_at(self, event_time: int) -> bool:
        """Return True if event_time ≤ knowledge_boundary (no leakage)."""
        return event_time <= self.knowledge_boundary

    # -- snapshot for verifier (paper §3.2.3 Verification) -----------------

    def verifier_snapshot(self) -> dict:
        """
        Read-only snapshot sent to the verification stage.
        Contains ONLY what the verifier is allowed to see:
          - high-confidence beliefs
          - knowledge boundary
          - world constraints
        """
        return {
            "high_confidence_beliefs": [
                b.to_dict() for b in self.high_confidence_beliefs()
            ],
            "knowledge_boundary": self.knowledge_boundary,
            "world_constraints": list(self.world_constraints),
        }

    # -- deep copy for counterfactual queries ------------------------------

    def copy(self) -> "CharacterState":
        """Return a deep copy (used for counterfactual do(·) queries)."""
        return copy.deepcopy(self)

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "character_id": self.character_id,
            "traits": self.traits.to_dict(),
            "emotions": self.emotions.to_dict(),
            "beliefs": {k: v.to_dict() for k, v in self.beliefs.items()},
            "relationships": {k: v.to_dict() for k, v in self.relationships.items()},
            "intentions": list(self.intentions),
            "timeline_index": self.timeline_index,
            "knowledge_boundary": self.knowledge_boundary,
            "world_constraints": list(self.world_constraints),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CharacterState":
        return cls(
            character_id=d.get("character_id", "default"),
            traits=TraitState.from_dict(d["traits"]),
            emotions=EmotionState.from_dict(d["emotions"]),
            beliefs={
                k: BeliefNode.from_dict(v) for k, v in d.get("beliefs", {}).items()
            },
            relationships={
                k: RelationshipState.from_dict(v)
                for k, v in d.get("relationships", {}).items()
            },
            intentions=d.get("intentions", []),
            timeline_index=d.get("timeline_index", 0),
            knowledge_boundary=d.get("knowledge_boundary", 0),
            world_constraints=d.get("world_constraints", []),
        )

    def __repr__(self) -> str:
        return (
            f"CharacterState(id={self.character_id!r}, "
            f"traits={len(self.traits.traits)}, "
            f"beliefs={len(self.beliefs)}, "
            f"rels={len(self.relationships)}, "
            f"t={self.timeline_index})"
        )



class WorldState:
    """
    Represents the objective world state (canonical narrative truth).

    Unlike CharacterState, this graph may contain facts unknown to
    any single character.
    
    The world graph encodes objective narrative facts and timeline
    state.  Characters NEVER condition directly on the world graph;
    belief updates are driven by observations derived from it.

    W_t contains:
      - entity locations
      - object states
      - global events
      - timeline progression
      - hard constraints (forbidden abilities, physical limits, etc.)

    Attributes
    ----------
    entities      : dict[str, dict]   entity_id → properties
    object_states : dict[str, Any]    object_id → state
    global_events : list[dict]        chronological event log
    constraints   : list[str]         immutable narrative rules
    timeline_index: int               canonical story time T
    """

    def __init__(
        self,
        entities: Optional[Dict[str, dict]] = None,
        object_states: Optional[Dict[str, Any]] = None,
        global_events: Optional[List[dict]] = None,
        constraints: Optional[List[str]] = None,
        timeline_index: int = 0,
    ):
        self.entities: Dict[str, dict] = dict(entities or {})
        self.object_states: Dict[str, Any] = dict(object_states or {})
        self.global_events: List[dict] = list(global_events or [])
        self.constraints: List[str] = list(constraints or [])
        self.timeline_index: int = int(timeline_index)

    # -- mutation helpers --------------------------------------------------

    def add_entity(self, entity_id: str, properties: dict) -> None:
        self.entities[entity_id] = dict(properties)

    def update_object(self, object_id: str, state: Any) -> None:
        self.object_states[object_id] = state

    def record_event(self, event: dict) -> None:
        """Append a global event and advance the timeline."""
        evt = dict(event)
        evt.setdefault("time", self.timeline_index)
        self.global_events.append(evt)

    def advance_timeline(self, steps: int = 1) -> None:
        self.timeline_index += steps

    # -- observation model (paper §3.2.2) ----------------------------------

    def perceive(
        self,
        character_location: Optional[str] = None,
        sensory_access: Optional[Set[str]] = None,
    ) -> dict:
        """
        Generate an observation O_t = Perception(W_t, State_c).

        Returns a dict of world facts visible to the character
        given their location and sensory access. This enforces
        the epistemic separation between W_t and B_t^c.
        """
        sensory_access = sensory_access or set()

        visible_entities = {}
        for eid, props in self.entities.items():
            entity_loc = props.get("location")
            # Entity is visible if same location or in sensory set
            if entity_loc == character_location or eid in sensory_access:
                visible_entities[eid] = dict(props)

        visible_objects = {}
        for oid, state in self.object_states.items():
            if oid in sensory_access or character_location is not None:
                visible_objects[oid] = state

        return {
            "visible_entities": visible_entities,
            "visible_objects": visible_objects,
            "current_time": self.timeline_index,
        }

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "entities": {k: dict(v) for k, v in self.entities.items()},
            "object_states": dict(self.object_states),
            "global_events": [dict(e) for e in self.global_events],
            "constraints": list(self.constraints),
            "timeline_index": self.timeline_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorldState":
        return cls(
            entities=d.get("entities"),
            object_states=d.get("object_states"),
            global_events=d.get("global_events"),
            constraints=d.get("constraints"),
            timeline_index=d.get("timeline_index", 0),
        )

    def __repr__(self) -> str:
        return (
            f"WorldState(entities={len(self.entities)}, "
            f"objects={len(self.object_states)}, "
            f"events={len(self.global_events)}, t={self.timeline_index})"
        )




class EventFrame:
    """
    Structured event representation extracted from user dialogue.

    Follows the schema:
    e_t = (P_t, E_t, a_t, τ_t, c_t)
      P_t : set of asserted propositions
      E_t : set of referenced entities
      a_t : speaker identity
      τ_t : inferred emotional tone
      c_t : extraction confidence ∈ [0, 1]

    Attributes
    ----------
    propositions : list[str]   P_t — asserted propositions
    entities     : list[str]   E_t — referenced entity IDs
    speaker      : str         a_t — who is speaking
    emotional_tone: str        τ_t — emotional tone label
    confidence   : float       c_t ∈ [0, 1]
    turn_index   : int         which dialogue turn this came from
    source_text  : str         raw input (for provenance / debugging)
    """

    def __init__(
        self,
        propositions: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        speaker: str = "unknown",
        emotional_tone: str = "neutral",
        confidence: float = 1.0,
        turn_index: int = 0,
        source_text: str = "",
    ):
        self.propositions: List[str] = list(propositions or [])
        self.entities: List[str] = list(entities or [])
        self.speaker: str = str(speaker)
        self.emotional_tone: str = str(emotional_tone)
        self.confidence: float = _validate_unit(confidence, "confidence")
        self.turn_index: int = int(turn_index)
        self.source_text: str = str(source_text)

    # -- query helpers used by belief update (Member 3) --------------------

    def asserts(self, proposition: str) -> bool:
        """True if this event explicitly asserts *proposition*."""
        return proposition in self.propositions

    def denies(self, proposition: str) -> bool:
        """True if event asserts the negation (prefixed with NOT:)."""
        return f"NOT:{proposition}" in self.propositions

    def references_entity(self, entity_id: str) -> bool:
        return entity_id in self.entities

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "propositions": list(self.propositions),
            "entities": list(self.entities),
            "speaker": self.speaker,
            "emotional_tone": self.emotional_tone,
            "confidence": self.confidence,
            "turn_index": self.turn_index,
            "source_text": self.source_text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EventFrame":
        return cls(
            propositions=d.get("propositions"),
            entities=d.get("entities"),
            speaker=d.get("speaker", "unknown"),
            emotional_tone=d.get("emotional_tone") or d.get("tone") or "neutral",
            confidence=d.get("confidence", 1.0),
            turn_index=d.get("turn_index", 0),
            source_text=d.get("source_text", ""),
        )

    def __repr__(self) -> str:
        return (
            f"EventFrame(speaker={self.speaker!r}, "
            f"props={len(self.propositions)}, "
            f"entities={self.entities}, "
            f"tone={self.emotional_tone!r}, c={self.confidence:.2f})"
        )

