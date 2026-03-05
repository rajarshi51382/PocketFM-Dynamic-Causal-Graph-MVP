"""
Defines the shared schemas used throughout the system.

These structures represent the causal character graph state,
world state, and structured event frames as defined in the
Dynamic Causal Character Graphs paper.
"""

from typing import Dict, List, Optional


class TraitState:
    """
    Represents stable personality traits.

    Attributes
    ----------
    traits : Dict[str, float]
        Mapping of trait name to intensity in [-1, 1] or [0, 1].
    plasticity : float
        Governs how slowly traits evolve. Low values resist change.
    """

    def __init__(self, traits: Dict[str, float], plasticity: float = 0.05):
        self.traits: Dict[str, float] = traits
        self.plasticity: float = plasticity


class EmotionState:
    """
    Represents the emotional state of a character.

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
        arousal: float = 0.5,
        emotion_tags: Optional[Dict[str, float]] = None,
        plasticity: float = 0.8,
    ):
        self.valence: float = valence
        self.arousal: float = arousal
        self.emotion_tags: Dict[str, float] = emotion_tags if emotion_tags is not None else {}
        self.plasticity: float = plasticity


class RelationshipState:
    """
    Represents relationship values with another entity.

    All values are in [0, 1].

    Attributes
    ----------
    trust : float
    affection : float
    respect : float
    """

    def __init__(
        self,
        trust: float = 0.5,
        affection: float = 0.5,
        respect: float = 0.5,
    ):
        self.trust: float = trust
        self.affection: float = affection
        self.respect: float = respect


class BeliefNode:
    """
    Represents a belief proposition using log-odds encoding.

    The log-odds representation enables additive evidence updates:
        l_t = ln(p_t / (1 - p_t))

    A value of 0 represents maximum uncertainty (p = 0.5).

    Attributes
    ----------
    proposition : str
        The propositional content, e.g. "player_has_key".
    log_odds : float
        Epistemic confidence expressed as log-odds.
    evidence_sources : List[str]
        Provenance metadata tracking the source of each update.
    """

    def __init__(self, proposition: str, log_odds: float = 0.0):
        self.proposition: str = proposition
        self.log_odds: float = log_odds
        self.evidence_sources: List[str] = []


class CharacterState:
    """
    Full internal character state in the Dynamic SCM.

    Attributes
    ----------
    traits : TraitState
    emotions : EmotionState
    beliefs : Dict[str, BeliefNode]
        Keyed by normalised proposition string.
    relationships : Dict[str, RelationshipState]
        Keyed by entity name.
    intentions : List[str]
        Current behavioural goals.
    timeline_index : int
        Current position on the character's knowledge timeline.
    """

    def __init__(self):
        self.traits: TraitState = TraitState({}, plasticity=0.05)
        self.emotions: EmotionState = EmotionState()
        self.beliefs: Dict[str, BeliefNode] = {}
        self.relationships: Dict[str, RelationshipState] = {}
        self.intentions: List[str] = []
        self.timeline_index: int = 0


class WorldState:
    """
    Represents the objective world state (canonical narrative truth).

    Unlike CharacterState, this graph may contain facts unknown to
    any single character.

    Attributes
    ----------
    entities : Dict[str, dict]
        Entity attribute maps.
    object_states : Dict[str, str]
        Current state label for each object.
    constraints : List[str]
        Immutable narrative rules (forbidden abilities, etc.).
    timeline_index : int
    """

    def __init__(self):
        self.entities: Dict[str, dict] = {}
        self.object_states: Dict[str, str] = {}
        self.constraints: List[str] = []
        self.timeline_index: int = 0


class EventFrame:
    """
    Structured event representation extracted from user dialogue.

    Follows the schema:
        e_t = (P_t, E_t, a_t, tau_t, c_t)

    Attributes
    ----------
    propositions : List[str]
        Asserted propositions P_t.
    entities : List[str]
        Referenced entities E_t.
    speaker : Optional[str]
        Speaker identity a_t. None or "DirectObservation" for
        first-person observations.
    emotional_tone : Optional[str]
        Inferred emotional tone tau_t (e.g. "anger", "joy").
    confidence : float
        Extraction confidence c_t in [0, 1].
    """

    def __init__(
        self,
        propositions: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        speaker: Optional[str] = None,
        emotional_tone: Optional[str] = None,
        confidence: float = 1.0,
    ):
        self.propositions: List[str] = propositions if propositions is not None else []
        self.entities: List[str] = entities if entities is not None else []
        self.speaker: Optional[str] = speaker
        self.emotional_tone: Optional[str] = emotional_tone
        self.confidence: float = confidence