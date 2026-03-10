"""
Interactive demo showcasing the full Dynamic Causal Character Graph system.

Features:
- LLM-driven event extraction and dialogue generation
- Causal belief propagation (A -> B dependencies)
- Trait-modulated emotional updates
- Dynamic relationship discovery
- State persistence (save/load)

Usage:
    export GEMINI_API_KEY=your_key_here
    PYTHONPATH=src python demo/interactive_demo.py
"""

import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.data_structures import CharacterState, WorldState, TraitState, BeliefNode
from simulation.simulation_loop import simulation_turn
from state.persistence import save_simulation_state, load_simulation_state
from core.llm_client import configure_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_demo_character() -> CharacterState:
    """Create a character with rich initial state for the demo."""
    # 1. Traits
    traits = TraitState(
        traits={
            "bravery": 0.8,
            "honesty": 0.6,
            "neuroticism": 0.4,  # Moderate emotional volatility
            "trusting": 0.2,     # Slightly suspicious initially
        }
    )
    
    # 2. Initial Beliefs
    beliefs = {
        "castle_is_safe": BeliefNode("castle_is_safe", log_odds=1.5),
        "forest_is_dangerous": BeliefNode("forest_is_dangerous", log_odds=1.0),
        "king_is_wise": BeliefNode("king_is_wise", log_odds=0.5),
    }
    
    # 3. Create State
    state = CharacterState(
        character_id="Sir_Galahad",
        traits=traits,
        beliefs=beliefs
    )
    
    # 4. Add Causal Links
    # If the castle is safe, the king is likely wise (correlation)
    state.add_causal_link(antecedent="castle_is_safe", consequent="king_is_wise", weight=0.8)
    
    # If the forest is dangerous, the castle is even safer (contrast/refuge)
    state.add_causal_link(antecedent="forest_is_dangerous", consequent="castle_is_safe", weight=0.5)

    # If the castle is not safe, the king is likely not wise (correlation)
    state.add_causal_link(antecedent="not_castle_is_safe", consequent="not_king_is_wise", weight=0.8)
    
    return state

def main():
    print("=== Dynamic Causal Character Graph: Interactive Demo ===")
    
    # Check LLM configuration
    if not configure_client():
        print("WARNING: GEMINI_API_KEY not found. Running in fallback mode (rule-based).")
        print("Set the environment variable to enable LLM features.\n")
    else:
        print("LLM Client Configured. Using Gemini model.\n")

    # Initialize
    character = create_demo_character()
    world = WorldState()
    
    print(f"Character Initialized: {character.character_id}")
    print(f"Traits: {character.traits.traits}")
    print("Commands: /save [file], /load [file], /quit\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
            
        # Handle commands
        if user_input.lower().startswith("/save"):
            parts = user_input.split()
            filename = parts[1] if len(parts) > 1 else "savegame.json"
            save_simulation_state(character, world, filename)
            continue
            
        if user_input.lower().startswith("/load"):
            parts = user_input.split()
            filename = parts[1] if len(parts) > 1 else "savegame.json"
            try:
                character, world = load_simulation_state(filename)
                print(f"Loaded state from {filename}")
            except Exception as e:
                print(f"Error loading: {e}")
            continue

        if user_input.lower() in {"/quit", "/exit", "quit", "exit"}:
            break

        # Run simulation turn
        response = simulation_turn(user_input, character, world)
        
        # Display response and internal state summary
        print(f"\nCharacter: {response}")
        
        # Debug output for demo purposes
        print(f"\n[Internal State]")
        dom_emotion = character.emotions.dominant_emotion()
        print(f"  Emotion: {dom_emotion} (v={character.emotions.valence:.2f}, a={character.emotions.arousal:.2f})")
        print(f"  Top Intentions: {character.intentions}")
        print(f"  Active Beliefs:")
        for b in character.beliefs.values():
            print(f"    - {b.proposition}: {b.probability:.2f}")
        print("-" * 40 + "\n")

if __name__ == "__main__":
    main()
