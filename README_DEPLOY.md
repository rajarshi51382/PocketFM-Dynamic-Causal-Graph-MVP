# Deploy Dynamic Causal Character Graphs for Free

This document explains how to deploy the demo for free on multiple platforms.

## Option 1: Streamlit Community Cloud (Recommended)

**Result:** `https://your-app-name.streamlit.app`

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select this repository (`rajarshi51382/PocketFM-Dynamic-Causal-Graph-MVP`)
5. Set:
   - Branch: `main`
   - Main file path: `streamlit_app.py`
6. In "Advanced settings":
   - Set custom subdomain: `causal-character-graphs` (or your preferred name)
7. Click Deploy

Your app will be live at `https://causal-character-graphs.streamlit.app` within 2-3 minutes.

## Option 2: Hugging Face Spaces

**Result:** `https://huggingface.co/spaces/your-username/causal-character-graphs`

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Name it `causal-character-graphs`
4. Select "Streamlit" as the SDK
5. Clone/upload this repository's files to the space

Or use the Hugging Face CLI:
```bash
# Install huggingface_hub
pip install huggingface_hub

# Login
huggingface-cli login

# Create and push to space
huggingface-cli repo create causal-character-graphs --type space --space_sdk streamlit
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/causal-character-graphs
git push hf main
```

## Option 3: Render

**Result:** `https://causal-character-graphs.onrender.com`

1. Go to [render.com](https://render.com)
2. Create a free account
3. Click "New Web Service"
4. Connect your GitHub repository
5. Set:
   - Name: `causal-character-graphs`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
6. Deploy

## Features That Work Without API Keys

All core features work out of the box:

✅ **Belief Revision** — Log-odds updates as you chat  
✅ **Causal Propagation** — A→B belief graph updates in real time  
✅ **Emotional State** — Valence/arousal bars update after each turn  
✅ **Relationship Tracking** — Trust, affection, respect per entity  
✅ **Smart Dialogue** — Rule-based contextually-aware character responses  
✅ **Scenario Presets** — One-click narrative events  
✅ **Save/Load/Download** — Persist and restore the full simulation state  

## Optional: Enhanced LLM Responses

To enable LLM-powered responses (optional):
1. Get a free Gemini API key from [ai.google.dev](https://ai.google.dev)
2. Enter it in the sidebar of the app

The app works perfectly without this—the rule-based system generates engaging, contextually-aware responses.
