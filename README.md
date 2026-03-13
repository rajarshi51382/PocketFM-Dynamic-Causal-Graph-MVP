# Dynamic Causal Character Graphs

[![CI](https://github.com/itsloganmann/PocketFM-Dynamic-Causal-Graph-MVP/actions/workflows/deploy.yml/badge.svg)](https://github.com/itsloganmann/PocketFM-Dynamic-Causal-Graph-MVP/actions/workflows/deploy.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://causal-character-graphs.streamlit.app)

A lightweight structured-state framework for persistent fictional agents in long-form dialogue systems.

## Live Demo

**[https://causal-character-graphs.streamlit.app](https://causal-character-graphs.streamlit.app)**

## Test Scenarios and Grounded Context

- Scenario checklist: `docs/scenarios.md`
- Grounded context and assumptions: `docs/context.md`

### No API Key Required

- **Belief revision** - log-odds updates as you chat
- **Causal propagation** - A to B belief graph updates in real time
- **Emotional state** - valence and arousal bars update after each turn
- **Relationship tracking** - trust, affection, respect per entity
- **Smart dialogue generation** - contextually-aware character responses
- **Scenario presets** - one-click narrative events
- **Save, load, download** - persist and restore the full simulation state

> **Optional**: Add a Gemini API key for enhanced LLM-powered responses

## Overview

Large language models lack an explicit representation of persistent psychological state, causing personality drift and motivational inconsistency in extended interactions. This project implements Dynamic Causal Character Graphs (DCCGs), a structured causal framework that models a character's core traits, beliefs, and world constraints as a directed acyclic graph.

Each dialogue turn runs in two phases:

1. **Update phase** - structured state transitions via causal propagation (belief revision, emotional update, relationship update).
2. **Act phase** - conditioned dialogue generation based on the updated character state.

The belief revision system uses a log-odds framework for numerically stable, additive evidence integration.
## Repository Structure

```
streamlit_app.py          - Web demo (Streamlit Community Cloud entry point)
.streamlit/
  config.toml             - Streamlit theme & server configuration

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

docs/
  context.md                - Grounded context and assumptions
  scenarios.md              - Scenario test checklist
  deploy.md                 - Deployment guide
  huggingface.md            - Hugging Face deployment notes
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

## Running the Web Demo Locally

```bash
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

## Deploying on Streamlit Community Cloud (Free, Permanent)

1. Fork or push this repository to your GitHub account.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **"New app"** → select your repository and branch.
4. Set the **Main file path** to `streamlit_app.py`.
5. **Set a custom subdomain** (e.g., `causal-character-graphs`) in Advanced settings for a clean URL.
6. Click **Deploy** - your app will be live at a permanent `*.streamlit.app` URL within minutes.

> **Note**: No API keys or secrets are required. All features work out of the box!

## Running Tests

```bash
pytest
```

## Running the CLI Demo

```bash
PYTHONPATH=src python demo/demo_conversations.py
```

The demo runs fully offline (no LLM connection required) and prints belief state before and after narrative events.

## LLM Integration

`src/extraction/event_extraction.py` and `src/generation/dialogue_generation.py` contain documented stubs for the LLM extraction and generation passes. Replace the stub bodies with calls to your preferred model endpoint. All other modules are LLM-independent.
