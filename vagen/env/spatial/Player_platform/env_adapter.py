from __future__ import annotations
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import json
import yaml
import re
import time
from omegaconf import OmegaConf
try:
    import streamlit as st
except Exception:  # Streamlit is only available in the web app runtime.
    st = None
from vagen.env.spatial.Base.tos_base.utils.utils import format_llm_output
from vagen.env.spatial.env import SpatialGym                  
from vagen.env.spatial.env_config import SpatialGymConfig

_USER_STATE_KEY = "_player_sessions"
_USER_BASE_FIELD = "_player_id_base"
_USER_FULL_FIELD = "_player_id_full"


def _require_streamlit() -> "st":
    if not st:
        raise RuntimeError("Streamlit is required for session helpers.")
    return st


def _session_store() -> Dict[str, Dict[str, Any]]:
    """Return the shared session bucket that holds per-user dictionaries."""
    _require_streamlit()
    return st.session_state.setdefault(_USER_STATE_KEY, {})


def _sanitize_user_id(value: Optional[str]) -> str:
    """Normalize IDs so every helper speaks the same language."""
    value = (value or "").strip()
    return value or ""


def set_user_id(raw_id: str, force_new: bool = False) -> Tuple[str, str]:
    """
    Persist both the human-entered ID and its unique timestamped variant.
    Returns (base_id, session_id).
    """
    _require_streamlit()
    base = _sanitize_user_id(raw_id)
    if not base:
        st.session_state.pop(_USER_BASE_FIELD, None)
        st.session_state.pop(_USER_FULL_FIELD, None)
        st.session_state.pop("user_id", None)
        return "", ""

    stored_base = st.session_state.get(_USER_BASE_FIELD)
    session_id = st.session_state.get(_USER_FULL_FIELD)
    if force_new or base != stored_base or not session_id:
        session_id = f"{base}-{int(time.time() * 1000)}"

    st.session_state[_USER_BASE_FIELD] = base
    st.session_state[_USER_FULL_FIELD] = session_id
    st.session_state["user_id"] = session_id  # main ID now includes timestamp
    return base, session_id


def require_user_id(message: str = "Set your participant ID on the Home page.") -> str:
    """
    Stop the page early unless the participant has typed an ID.
    Returns the unique session ID (base + timestamp).
    """
    _require_streamlit()
    session_id = get_full_user_id("")
    if not session_id:
        st.warning(message)
        st.stop()
    return session_id


def get_user_session_state(session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Keep user-scoped objects (env, history, etc.) under one key so
    concurrent participants never overwrite one another.
    """
    store = _session_store()
    key = (session_id or get_full_user_id("")).strip()
    if not key:
        raise RuntimeError("No active participant session. Call set_user_id first.")
    return store.setdefault(key, {})


def get_base_user_id(default: str = "") -> str:
    """Return the human-entered participant ID."""
    if not st:
        return default
    return (st.session_state.get(_USER_BASE_FIELD) or "").strip() or default


def get_full_user_id(default: str = "anon") -> str:
    """Return the unique ID (base+timestamp) used for storage and logging."""
    if not st:
        return default
    return (st.session_state.get(_USER_FULL_FIELD) or "").strip() or default


def bind_model_name_to_user(cfg: SpatialGymConfig, user_id: str) -> SpatialGymConfig:
    """
    Ensure model_name (used for history directories) contains a user-specific suffix.
    """
    user_tag = re.sub(r"[^A-Za-z0-9._-]", "_", _sanitize_user_id(user_id)) or "anon"
    cfg.kwargs = cfg.kwargs or {}
    model_cfg = dict(cfg.kwargs.get("model_config") or {})
    base_name = model_cfg.get("model_name") or "human_player"
    model_cfg["model_name"] = f"{base_name}-{user_tag}"
    cfg.kwargs["model_config"] = model_cfg
    return cfg
@dataclass
class TurnRecord:
    t: int
    action: str
    obs_text: str
    reward: float
    done: bool
    info: Dict[str, Any]
    obs_raw: Dict[str, Any] = None 

def load_cfg_from_yaml(path: str) -> SpatialGymConfig:
    """
    Load SpatialGymConfig from a YAML file.
    Converts eval_task_counts to eval_tasks format.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    del raw['seed-range']
    # Convert eval_task_counts to eval_tasks if present
    if "eval_task_counts" in raw and not raw.get("eval_tasks"):
        eval_tasks = []
        for task_type, count in raw["eval_task_counts"].items():
            eval_tasks.append({
                "task_type": task_type,
                "num": int(count)
            })
        raw["eval_tasks"] = eval_tasks
        # Remove eval_task_counts to avoid confusion
        del raw["eval_task_counts"]

    # OmegaConf to ensure compatibility with ListConfig, etc.
    conf = OmegaConf.create(raw)

    # Convert into dataclass
    return SpatialGymConfig(**OmegaConf.to_container(conf, resolve=True))

def _wrap_user_action_for_env(user_text: str, enable_think: bool = True) -> str:
    """
    Convert human input into the expected LLM-style format (header-based).
    """
    action = user_text.strip()
    think_content = "Placeholder" if enable_think else ""
    # Case 1: single letter answer (A, B, C, etc.)
    if re.fullmatch(r"[A-Za-z]", action):
        return format_llm_output(think_content, action, enable_think=enable_think)

    # Case 2: action sequence (contains "()")
    if "(" in action and ")" in action:
        return format_llm_output(think_content, f"Actions: [{action}]", enable_think=enable_think)
    return format_llm_output(think_content, action, enable_think=enable_think)

def save_episode(user_id: str, episode_id: int, trajectory: list, analytics: dict, correct_answers: dict = None):
    log_dir = os.path.join("vagen/env/spatial/Player_platform/logs", user_id)
    os.makedirs(log_dir, exist_ok=True)

    out_path = os.path.join(log_dir, f"episode_{episode_id}.json")

    clean_traj = []
    for r in trajectory:
        # If it's a dataclass or object with __dict__, copy it
        if hasattr(r, "__dict__"):
            d = r.__dict__.copy()
        else:
            d = dict(r)  # already a dict

        #Drop obs_raw (contains PIL images)
        d.pop("obs_raw", None)
        clean_traj.append(d)

    payload = {
        "user_id": user_id,
        "episode_id": episode_id,
        "trajectory": clean_traj,
        "analytics": analytics,
        "correct_answers": correct_answers,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return out_path


class SpatialEnvAdapter:
    """
    Thin wrapper exposing a stable API to the Streamlit pages.
    """
    def __init__(
        self,
        cfg: SpatialGymConfig,
    ):
        self.cfg = cfg
        self.env = SpatialGym(cfg)
        self.turn = 0
        self._last_obs = None
        self._last_info = {}

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        obs, info = self.env.reset(seed=seed)
        if isinstance(obs, str):
            obs = {"obs_str": obs}
        self.turn = 0
        self._last_obs = obs
        self._last_info = info or {}
        return obs

    def step(self, user_action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Streamlit calls this with human action text. We wrap it for the env.
        Your env.step returns: (obs: dict, reward: float, done: bool, step_info: dict)
        """
        enable_think = bool(self.cfg.prompt_config.get("enable_think", True))
        llm_response = _wrap_user_action_for_env(user_action, enable_think=enable_think)

        obs, reward, done, step_info = self.env.step(llm_response)
        self.turn += 1
        self._last_obs = obs
        self._last_info = step_info or {}
        return obs, reward, done, step_info

    def get_eval_answers(self):
        answers = []
        for task in self.env.evaluation_manager.tasks:
            answers.append(task.answer)
        return answers

    def get_room_objects_str(self):
        return ";".join(sorted(self.env.exploration_manager.observed_items))

    def max_turn(self) -> int:
        return int(self.cfg.max_exp_steps or 20)

    # ---- optional helpers surfaced to UI pages ----
    def get_env_summary(self) -> Dict[str, Any]:
        env_summary = self.env.get_env_summary()
        env_summary["exploration_summary"] = self.env.get_exp_summary()
        env_summary["evaluation_summary"] = self.env.get_eval_summary()
        return env_summary

    def render_cache(self) -> Dict[str, Any]:
        return self.env.render() 

def format_obs(obs: Dict[str, Any]) -> str:
    """Pick the text to display to the user (your env places FORMAT_PROMPT at the end already)."""
    return obs.get("obs_str", "").strip()

def summarize_turn(t: int, action: str, obs: Dict[str, Any], reward: float, done: bool, info: Dict[str, Any]) -> TurnRecord:
    return TurnRecord(
        t=t,
        action=action,
        obs_text=format_obs(obs),   
        reward=reward,
        done=done,
        info=info,
        obs_raw=obs                 
    )

