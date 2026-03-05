# Dynamic Causal Character Graphs

A lightweight structured-state framework for persistent fictional agents in long-form dialogue systems.

## Overview

Large language models lack an explicit representation of persistent psychological state, causing personality drift and motivational inconsistency in extended interactions. This project implements Dynamic Causal Character Graphs (DCCGs), a structured causal framework that models a character's core traits, beliefs, and world constraints as a directed acyclic graph.

Each dialogue turn runs in two phases:

1. **Update phase** - structured state transitions via causal propagation (belief revision, emotional update, relationship update).
2. **Act phase** - conditioned dialogue generation based on the updated character state.

The belief revision system uses a log-odds framework for numerically stable, additive evidence integration. See `pocketfm_shikhar (10).pdf` for the full technical specification.

## Repository Structure

```
src/
  core/
    data_structures.py      - Shared schemas: TraitState, EmotionState,
                              RelationshipState, BeliefNode, CharacterState,
                              WorldState, EventFrame
  extraction/
    event_extraction.py     - Converts raw dialogue into structured EventFrame
  reasoning/
    belief_update.py        - Log-odds belief revision (core module)
    state_update.py         - Emotional, relationship, and intention propagation
  generation/
    dialogue_generation.py  - Prompt construction and LLM response generation
  simulation/
    simulation_loop.py      - Full two-phase dialogue loop
  state/
    character_state.py      - CharacterState factory utilities
    world_state.py          - WorldState factory utilities

tests/
  test_belief_update.py     - Unit tests for the belief update module

demo/
  demo_conversations.py     - Offline demo showing belief state evolution
```

## Belief Update Logic

The central update rule (Section 3.3.2 of the paper):

```
l_{t,k} = l_{t-1,k} + lambda_base * sigma(e_t) * rho(src) * delta(e_t, phi_k) * c_t
```

| Term | Description |
|---|---|
| `lambda_base` | Base learning rate |
| `sigma(e_t)` | Narrative importance / shock-gated plasticity |
| `rho(src)` | Source credibility from the relationship graph |
| `delta(e_t, phi_k)` | Directional alignment: +1, -1, or 0 |
| `c_t` | Extraction confidence |

Conflicting beliefs (e.g., `door_is_locked` and `not_door_is_locked`) are resolved via pairwise probability normalisation after each update.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10 or later is required.

## Running Tests

```bash
pytest
```

## Running the Demo

```bash
PYTHONPATH=src python demo/demo_conversations.py
```

The demo runs fully offline (no LLM connection required) and prints belief state before and after narrative events.

## LLM Integration

`src/extraction/event_extraction.py` and `src/generation/dialogue_generation.py` contain documented stubs for the LLM extraction and generation passes. Replace the stub bodies with calls to your preferred model endpoint. All other modules are LLM-independent.
