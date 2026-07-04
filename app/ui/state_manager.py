"""
ui/state_manager.py — Single source of truth for all st.session_state keys.
Prevents key typos across UI components.
"""

import streamlit as st

# ── Key constants ─────────────────────────────────────────────────────────────
SESSION_ID      = "te_session_id"
LLM_PROVIDER    = "te_llm_provider"
CHAT_HISTORY    = "te_chat_history"
LAST_PROVIDER   = "te_last_provider"
LAST_FALLBACK   = "te_last_fallback"

# Per-source upload state
SOURCE_A_FILE   = "te_source_a_file"
SOURCE_B_FILE   = "te_source_b_file"
SOURCE_C_FILE   = "te_source_c_file"
SOURCE_A_CHUNKS = "te_source_a_chunks"
SOURCE_B_CHUNKS = "te_source_b_chunks"
SOURCE_C_CHUNKS = "te_source_c_chunks"
SOURCE_A_TYPE   = "te_source_a_type"
SOURCE_B_TYPE   = "te_source_b_type"
SOURCE_C_TYPE   = "te_source_c_type"
SOURCES_LOADED  = "te_sources_loaded"   # list of loaded labels e.g. ['A', 'B']


def init_state() -> None:
    defaults = {
        SESSION_ID:      None,
        LLM_PROVIDER:    "gemini",
        CHAT_HISTORY:    [],
        LAST_PROVIDER:   None,
        LAST_FALLBACK:   False,
        SOURCE_A_FILE:   None,
        SOURCE_B_FILE:   None,
        SOURCE_C_FILE:   None,
        SOURCE_A_CHUNKS: 0,
        SOURCE_B_CHUNKS: 0,
        SOURCE_C_CHUNKS: 0,
        SOURCE_A_TYPE:   None,
        SOURCE_B_TYPE:   None,
        SOURCE_C_TYPE:   None,
        SOURCES_LOADED:  [],
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def mark_source_loaded(label: str, filename: str, chunk_count: int, file_type: str) -> None:
    """Update state after a successful ingest for a source."""
    file_key   = {"A": SOURCE_A_FILE,   "B": SOURCE_B_FILE,   "C": SOURCE_C_FILE  }.get(label)
    chunk_key  = {"A": SOURCE_A_CHUNKS, "B": SOURCE_B_CHUNKS, "C": SOURCE_C_CHUNKS}.get(label)
    type_key   = {"A": SOURCE_A_TYPE,   "B": SOURCE_B_TYPE,   "C": SOURCE_C_TYPE  }.get(label)
    if file_key:
        st.session_state[file_key]  = filename
        st.session_state[chunk_key] = chunk_count
        st.session_state[type_key]  = file_type
    # Update loaded list
    loaded = st.session_state.get(SOURCES_LOADED, [])
    if label not in loaded:
        loaded = loaded + [label]
    st.session_state[SOURCES_LOADED] = loaded


def reset_source(label: str) -> None:
    file_key  = {"A": SOURCE_A_FILE,   "B": SOURCE_B_FILE,   "C": SOURCE_C_FILE  }.get(label)
    chunk_key = {"A": SOURCE_A_CHUNKS, "B": SOURCE_B_CHUNKS, "C": SOURCE_C_CHUNKS}.get(label)
    type_key  = {"A": SOURCE_A_TYPE,   "B": SOURCE_B_TYPE,   "C": SOURCE_C_TYPE  }.get(label)
    if file_key:
        st.session_state[file_key]  = None
        st.session_state[chunk_key] = 0
        st.session_state[type_key]  = None
    loaded = [l for l in st.session_state.get(SOURCES_LOADED, []) if l != label]
    st.session_state[SOURCES_LOADED] = loaded


def add_message(role: str, content: str, metadata: dict | None = None) -> None:
    entry = {"role": role, "content": content}
    if metadata:
        entry["metadata"] = metadata
    st.session_state[CHAT_HISTORY].append(entry)


def get_chat_history() -> list[dict]:
    return st.session_state.get(CHAT_HISTORY, [])


def clear_chat() -> None:
    st.session_state[CHAT_HISTORY] = []
