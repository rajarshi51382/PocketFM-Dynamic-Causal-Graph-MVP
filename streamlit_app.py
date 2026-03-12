"""
Dynamic Causal Character Graph — Streamlit Web Demo

Features:
- LLM-driven event extraction and dialogue generation (Gemini API)
- Causal belief propagation (A -> B dependencies)
- Trait-modulated emotional updates
- Dynamic relationship discovery
- Interactive belief / emotion / causal-graph visualisation
- State persistence (save / load JSON)

Deploy for free on Streamlit Community Cloud:
  https://share.streamlit.io
"""

import json
import math
import os
import sys
import copy
import subprocess

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — works whether the app is run from the repo root or via
# `streamlit run streamlit_app.py`
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.data_structures import (
    BeliefNode,
    CharacterState,
    RelationshipState,
    TraitState,
    WorldState,
)
from core.llm_client import configure_client
from reasoning.belief_update import apply_belief_updates
from reasoning.causal_propagation import propagate_causal_effects
from reasoning.state_update import propagate_state_updates
from simulation.simulation_loop import simulation_turn
from state.persistence import save_simulation_state, load_simulation_state

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dynamic Causal Character Graphs",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_odds_to_prob(log_odds: float) -> float:
    return 1.0 / (1.0 + math.exp(-log_odds))


def _emotion_color(valence: float) -> str:
    if valence > 0.3:
        return "green"
    if valence < -0.3:
        return "red"
    return "orange"


def _get_commit_label() -> str:
    """Best-effort short commit label for UI display."""
    env_sha = os.getenv("STREAMLIT_GIT_COMMIT") or os.getenv("GITHUB_SHA")
    if env_sha:
        return env_sha[:7]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def _create_default_character() -> CharacterState:
    traits = TraitState(
        traits={
            "bravery": 0.8,
            "honesty": 0.6,
            "neuroticism": 0.4,
            "trusting": 0.2,
        }
    )
    beliefs = {
        "castle_is_safe": BeliefNode("castle_is_safe", log_odds=1.5),
        "forest_is_dangerous": BeliefNode("forest_is_dangerous", log_odds=1.0),
        "king_is_wise": BeliefNode("king_is_wise", log_odds=0.5),
    }
    state = CharacterState(
        character_id="Sir_Galahad",
        traits=traits,
        beliefs=beliefs,
    )
    state.add_causal_link(antecedent="castle_is_safe", consequent="king_is_wise", weight=0.8)
    state.add_causal_link(antecedent="forest_is_dangerous", consequent="castle_is_safe", weight=0.5)
    state.add_causal_link(antecedent="not_castle_is_safe", consequent="not_king_is_wise", weight=0.8)
    return state


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _apply_api_key() -> None:
    """Sync the session-state API key into the process environment."""
    if st.session_state.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = st.session_state.gemini_api_key


def _init_session():
    if "character" not in st.session_state:
        st.session_state.character = _create_default_character()
    if "world" not in st.session_state:
        st.session_state.world = WorldState()
    if "history" not in st.session_state:
        st.session_state.history = []   # list of (user_msg, char_response)
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")


_init_session()

# ---------------------------------------------------------------------------
# Sidebar — configuration & controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")

    # ── API Key ──────────────────────────────────────────────────────────────
    st.subheader("🔑 LLM Configuration")
    st.markdown("**All features work without an API key!**")
    
    with st.expander("Optional: Add Gemini API Key for enhanced responses"):
        api_key_input = st.text_input(
            "GEMINI_API_KEY",
            value=st.session_state.gemini_api_key,
            type="password",
            help="Enter your Google Gemini API key for LLM-powered responses. "
                 "Leave blank to use the built-in intelligent response system.",
        )
        if api_key_input != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = api_key_input

    if st.session_state.gemini_api_key:
        st.success("✅ LLM mode (Gemini API)")
    else:
        st.success("✅ Smart rule-based mode (no API needed)")

    st.divider()

    # ── Character setup ───────────────────────────────────────────────────
    st.subheader("🧑‍🦱 Character Setup")
    char_id = st.text_input(
        "Character name",
        value=st.session_state.character.character_id,
    )

    st.markdown("**Personality Traits** (–1 to +1)")
    col_t1, col_t2 = st.columns(2)
    bravery = col_t1.slider("Bravery", -1.0, 1.0,
                             float(st.session_state.character.traits.get("bravery", 0.8)), 0.05)
    honesty = col_t2.slider("Honesty", -1.0, 1.0,
                             float(st.session_state.character.traits.get("honesty", 0.6)), 0.05)
    neuroticism = col_t1.slider("Neuroticism", -1.0, 1.0,
                                 float(st.session_state.character.traits.get("neuroticism", 0.4)), 0.05)
    trusting = col_t2.slider("Trusting", -1.0, 1.0,
                              float(st.session_state.character.traits.get("trusting", 0.2)), 0.05)

    if st.button("🔄 Reset / Apply Character"):
        new_traits = TraitState(
            traits={
                "bravery": bravery,
                "honesty": honesty,
                "neuroticism": neuroticism,
                "trusting": trusting,
            }
        )
        new_char = _create_default_character()
        new_char.character_id = char_id
        new_char.traits = new_traits
        st.session_state.character = new_char
        st.session_state.world = WorldState()
        st.session_state.history = []
        st.success("Character reset!")
        st.rerun()

    st.divider()

    # ── Simulation parameters ─────────────────────────────────────────────
    st.subheader("🔧 Simulation Parameters")
    lambda_base = st.slider("λ base (learning rate)", 0.0, 2.0, 0.5, 0.05)
    narrative_importance = st.slider("σ narrative importance", 0.5, 5.0, 1.0, 0.25)

    st.divider()

    # ── Save / Load ───────────────────────────────────────────────────────
    st.subheader("💾 Save / Load State")
    save_path = st.text_input("Save file", "savegame.json")
    if st.button("Save"):
        ok = save_simulation_state(
            st.session_state.character, st.session_state.world, save_path
        )
        if ok:
            st.success(f"Saved → {save_path}")
        else:
            st.error("Save failed.")

    load_path = st.text_input("Load file", "savegame.json", key="load_path")
    if st.button("Load"):
        try:
            c, w = load_simulation_state(load_path)
            st.session_state.character = c
            st.session_state.world = w
            st.success(f"Loaded ← {load_path}")
            st.rerun()
        except Exception as ex:
            st.error(f"Load failed: {ex}")

    # Download current state as JSON
    state_json = json.dumps(
        {
            "character_state": st.session_state.character.to_dict(),
            "world_state": st.session_state.world.to_dict(),
        },
        indent=2,
    )
    st.download_button(
        "⬇️ Download state as JSON",
        data=state_json,
        file_name="dccg_state.json",
        mime="application/json",
    )

    st.divider()
    st.caption(f"App version: `{_get_commit_label()}`")

# ---------------------------------------------------------------------------
# Main layout — two-column: chat on the left, state panels on the right
# ---------------------------------------------------------------------------

st.title("🧠 Dynamic Causal Character Graphs")
st.markdown(
    "An interactive demo of the **Dynamic Causal Character Graph (DCCG)** system — "
    "belief revision, causal propagation, and conditioned dialogue generation. "
    "**All features work instantly — no API keys required!**"
)
st.caption(f"Deployed version: `{_get_commit_label()}`")

left_col, right_col = st.columns([3, 2], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT — chat interface
# ─────────────────────────────────────────────────────────────────────────────
with left_col:
    st.subheader(f"💬 Conversation with *{st.session_state.character.character_id}*")

    # Render conversation history
    chat_container = st.container(height=420, border=True)
    with chat_container:
        if not st.session_state.history:
            st.markdown(
                "_The stage is set. Send a message to begin the interaction…_"
            )
        for user_msg, char_resp in st.session_state.history:
            with st.chat_message("user"):
                st.write(user_msg)
            with st.chat_message("assistant", avatar="🧑‍🦱"):
                st.write(char_resp)

    # Message input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your message",
            placeholder="E.g. 'The king has betrayed the castle!'",
            label_visibility="collapsed",
        )
        send_col, clear_col = st.columns([4, 1])
        submitted = send_col.form_submit_button("Send ➤", use_container_width=True)
        cleared = clear_col.form_submit_button("Clear", use_container_width=True)

    if cleared:
        st.session_state.history = []
        st.rerun()

    if submitted and user_input.strip():
        _apply_api_key()

        with st.spinner("Thinking…"):
            response = simulation_turn(
                user_input.strip(),
                st.session_state.character,
                st.session_state.world,
                lambda_base=lambda_base,
                narrative_importance=narrative_importance,
            )

        st.session_state.history.append((user_input.strip(), response))
        st.rerun()

    # Scenario presets
    st.subheader("🎭 Scenario Presets")
    st.markdown("Click a preset to send it as a message:")
    preset_cols = st.columns(2)
    presets = [
        ("🏰 Castle is unsafe", "I heard the castle walls are crumbling and it's no longer safe!"),
        ("👑 King betrayed us", "The king has betrayed the entire kingdom — he's a liar!"),
        ("🌲 Forest now safe", "Actually, the forest has been cleared; it's perfectly safe now."),
        ("🤝 Ally is trustworthy", "I want you to know that your ally has proven completely trustworthy."),
        ("⚔️ Enemy approaching", "An enemy army is approaching — we must act fast!"),
        ("🕊️ Peace declared", "The war is over; peace has been declared throughout the land."),
    ]
    for i, (label, text) in enumerate(presets):
        col = preset_cols[i % 2]
        if col.button(label, key=f"preset_{i}", use_container_width=True):
            _apply_api_key()
            with st.spinner("Thinking…"):
                response = simulation_turn(
                    text,
                    st.session_state.character,
                    st.session_state.world,
                    lambda_base=lambda_base,
                    narrative_importance=narrative_importance,
                )
            st.session_state.history.append((text, response))
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT — internal state visualisation
# ─────────────────────────────────────────────────────────────────────────────
with right_col:
    char = st.session_state.character

    # ── Emotional State ─────────────────────────────────────────────────────
    st.subheader("😊 Emotional State")
    v = char.emotions.valence
    a = char.emotions.arousal
    dom = char.emotions.dominant_emotion() or "neutral"
    emo_color = _emotion_color(v)
    st.markdown(
        f"**Dominant:** :{emo_color}[{dom}]  "
        f"| Valence: `{v:+.2f}`  | Arousal: `{a:.2f}`"
    )
    # Valence bar
    st.progress(
        max(0.0, min(1.0, (v + 1.0) / 2.0)),
        text=f"Valence  {v:+.2f}  (–1 = very negative … +1 = very positive)",
    )
    st.progress(
        max(0.0, min(1.0, a)),
        text=f"Arousal  {a:.2f}  (0 = calm … 1 = excited)",
    )
    if char.emotions.emotion_tags:
        with st.expander("Emotion tags"):
            for tag, intensity in sorted(
                char.emotions.emotion_tags.items(), key=lambda x: -x[1]
            ):
                st.progress(intensity, text=f"{tag}  {intensity:.2f}")

    st.divider()

    # ── Beliefs ─────────────────────────────────────────────────────────────
    st.subheader("💡 Beliefs")
    if char.beliefs:
        sorted_beliefs = sorted(
            char.beliefs.items(), key=lambda x: -abs(x[1].log_odds)
        )
        for prop, node in sorted_beliefs:
            prob = node.probability
            st.markdown(f"**{prop}**  `p={prob:.2f}`  `ℓ={node.log_odds:+.2f}`")
            st.progress(prob)
    else:
        st.info("No beliefs tracked yet.")

    st.divider()

    # ── Traits ─────────────────────────────────────────────────────────────
    st.subheader("🎭 Personality Traits")
    for trait, val in char.traits.traits.items():
        # Map [-1,1] → [0,1] for the progress bar
        st.progress(
            max(0.0, min(1.0, (val + 1.0) / 2.0)),
            text=f"{trait}  {val:+.2f}",
        )

    st.divider()

    # ── Causal Links ────────────────────────────────────────────────────────
    st.subheader("🔗 Causal Graph Links")
    if char.causal_links:
        for link in char.causal_links:
            ant = link.get("antecedent", "?")
            con = link.get("consequent", "?")
            w = link.get("weight", 1.0)
            ant_belief = char.get_belief(ant)
            ant_prob = f"{ant_belief.probability:.2f}" if ant_belief else "?"
            st.markdown(
                f"**{ant}** `p={ant_prob}` → **{con}** &nbsp; _(w={w:.2f})_"
            )
    else:
        st.info("No causal links defined.")

    st.divider()

    # ── Relationships ───────────────────────────────────────────────────────
    st.subheader("🤝 Relationships")
    if char.relationships:
        for entity, rel in char.relationships.items():
            with st.expander(f"{entity}"):
                st.progress(rel.trust, text=f"Trust  {rel.trust:.2f}")
                st.progress(rel.affection, text=f"Affection  {rel.affection:.2f}")
                st.progress(rel.respect, text=f"Respect  {rel.respect:.2f}")
    else:
        st.info("No relationships tracked yet.")

    # ── Intentions ─────────────────────────────────────────────────────────
    if char.intentions:
        st.divider()
        st.subheader("🎯 Active Intentions")
        for intention in char.intentions:
            st.markdown(f"- {intention}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    """
    <div style='text-align:center; color: grey; font-size: 0.85em;'>
    Dynamic Causal Character Graphs · 
    <a href='https://github.com/rajarshi51382/PocketFM-Dynamic-Causal-Graph-MVP' target='_blank'>GitHub</a> ·
    Powered by <a href='https://streamlit.io' target='_blank'>Streamlit Community Cloud</a>
    </div>
    """,
    unsafe_allow_html=True,
)
