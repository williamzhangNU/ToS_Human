# pages/03_📈_Analytics.py
import streamlit as st
import pandas as pd
from vagen.env.spatial.Player_platform.env_adapter import (
    get_user_session_state,
    require_user_id,
    get_base_user_id,
)

st.set_page_config(page_title="Analytics", layout="wide", page_icon="📈")
st.title("📈 Analytics")

session_id = require_user_id("Set your participant ID on the Home page before viewing analytics.")
base_id = get_base_user_id()
st.caption(f"Participant ID: {base_id} · Session ID: {session_id}")

# Share the same per-user bucket the Play page uses.
state = get_user_session_state(session_id)
env = state.get("env")
if not env:
    st.info("No environment yet.")
else:
    cfg = env.cfg.to_dict()
    st.subheader("Environment Config")
    st.json(cfg)
    summary = env.get_env_summary()
    st.subheader("Exploration Summary")
    st.json(summary["exploration_summary"])
    st.subheader("Evaluation Summary")
    st.json(summary["evaluation_summary"])
    st.subheader("Turn Logs")
    st.json(summary["env_turn_logs"])
    correct_answers = env.get_eval_answers()
    st.subheader("Evaluation Answers")
    st.write(correct_answers)


hist = state.get("history", [])
if not hist:
    st.info("No data yet.")
else:
    df = pd.DataFrame([{"t": h.t, "reward": h.reward} for h in hist])
    st.subheader("Reward per turn")
    st.line_chart(df.set_index("t"))
    st.subheader("Cumulative reward")
    df["cum_reward"] = df["reward"].cumsum()
    st.line_chart(df.set_index("t")[["cum_reward"]])
