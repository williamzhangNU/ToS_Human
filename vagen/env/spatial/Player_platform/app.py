import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from vagen.env.spatial.Player_platform.env_adapter import (
    get_user_session_state,
    set_user_id,
    get_base_user_id,
)

st.set_page_config(page_title="Spatial Exploration Dashboard", layout="wide", page_icon="🧭")

st.title("🧭 Spatial Exploration Dashboard")

if "participant_input" not in st.session_state:
    st.session_state.participant_input = ""

# Keep the raw input separate; the session ID adds the timestamp silently.
typed_id = st.text_input(
    "Enter your participant ID",
    value=st.session_state.participant_input or "",
    placeholder="e.g., user123"
)
st.session_state.participant_input = typed_id
base_id, session_id = set_user_id(typed_id)

if not session_id:
    st.warning("⚠️ Please enter your ID before proceeding to Play.")
else:
    get_user_session_state(session_id)  # bootstrap per-user bucket
    st.success(f"Using session ID: {session_id}")
    st.caption(f"Participant ID: {base_id}")

st.markdown("""
Welcome! Use the **Play** page to interact with the environment turn by turn.
- Type an action → press **Submit** (or hit Enter)
- See the new observation, reward, and whether the episode is done
- **Trajectory** page shows a turn-by-turn log
- **Analytics** gives quick summaries
- **Admin** lets you set seed, max turns, and reset
""")

st.markdown("---")
st.subheader("Quick Tips")
st.markdown("""
- You can reset the episode any time from the **Admin** page.
- Your session is isolated—no server-side persistence unless you add it.
""")
