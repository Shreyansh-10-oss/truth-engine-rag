"""
ui/upload_panel.py — Three-source upload panel.
Renders separate upload zones for Source A, B, and C.
Each calls POST /ingest/{label} and shares the same session_id.
No business logic — UI only.
"""

import streamlit as st
import requests

from app.ui import state_manager as sm

API_BASE = "http://localhost:8000"

SUPPORTED_TYPES = ["pdf", "json", "csv", "md", "txt"]

SOURCE_CONFIG = {
    "A": {
        "icon": "🏆",
        "title": "Source A — Golden Truth",
        "desc": "The most authoritative source. Highest priority in answers.",
        "color": "#1a472a",
        "badge": "GOLDEN",
    },
    "B": {
        "icon": "🔧",
        "title": "Source B — Real-World Fixes",
        "desc": "Support logs, real-world data. Experimental — may not be authoritative.",
        "color": "#1c3a5e",
        "badge": "REAL-WORLD",
    },
    "C": {
        "icon": "📜",
        "title": "Source C — Legacy Wiki",
        "desc": "Older documentation. May be deprecated or incorrect.",
        "color": "#5e3a1c",
        "badge": "LEGACY",
    },
}


def render_upload_panel() -> None:
    """Render three upload zones side by side."""
    st.markdown("### 📂 Upload Sources")
    st.caption(
        "Upload files for each source. At least one source is required to chat. "
        "Supported formats: **PDF · JSON · CSV · Markdown · TXT**"
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        _render_source_uploader("A")
    with col_b:
        _render_source_uploader("B")
    with col_c:
        _render_source_uploader("C")


def _render_source_uploader(label: str) -> None:
    cfg = SOURCE_CONFIG[label]
    file_key   = {"A": sm.SOURCE_A_FILE,   "B": sm.SOURCE_B_FILE,   "C": sm.SOURCE_C_FILE  }[label]
    chunk_key  = {"A": sm.SOURCE_A_CHUNKS, "B": sm.SOURCE_B_CHUNKS, "C": sm.SOURCE_C_CHUNKS}[label]
    type_key   = {"A": sm.SOURCE_A_TYPE,   "B": sm.SOURCE_B_TYPE,   "C": sm.SOURCE_C_TYPE  }[label]

    current_file   = st.session_state.get(file_key)
    current_chunks = st.session_state.get(chunk_key, 0)
    current_type   = st.session_state.get(type_key)

    # Header
    st.markdown(f"**{cfg['icon']} {cfg['title']}**")
    st.caption(cfg["desc"])

    # Status badge
    if current_file:
        st.success(f"✓ **{current_file}**  \n`{current_chunks} chunks` · `{current_type}`")
        if st.button(f"Remove Source {label}", key=f"remove_{label}", use_container_width=True):
            sm.reset_source(label)
            st.rerun()
        return

    # Upload widget
    uploaded = st.file_uploader(
        label=f"Source {label} file",
        type=SUPPORTED_TYPES,
        key=f"uploader_{label}",
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info(f"No file for Source {label}")
        return

    # Trigger ingest
    with st.spinner(f"Indexing Source {label}: **{uploaded.name}**…"):
        session_id = st.session_state.get(sm.SESSION_ID)
        try:
            params = {}
            if session_id:
                params["session_id"] = session_id

            response = requests.post(
                f"{API_BASE}/ingest/{label}",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")},
                params=params,
                timeout=120,
            )
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach backend. Make sure `main.py` is running.")
            return
        except requests.exceptions.Timeout:
            st.error("❌ Ingestion timed out.")
            return

    if response.status_code == 200:
        data = response.json()
        # Store the session_id from the first ingest response
        if not session_id:
            st.session_state[sm.SESSION_ID] = data["session_id"]
        sm.mark_source_loaded(
            label=label,
            filename=data["filename"],
            chunk_count=data["chunk_count"],
            file_type=data["file_type"],
        )
        st.success(f"✓ Source {label} indexed: {data['chunk_count']} chunks")
        st.rerun()

    elif response.status_code == 422:
        detail = response.json().get("detail", "Unprocessable file.")
        st.error(f"❌ {detail}")
    elif response.status_code == 413:
        st.error("❌ File exceeds 20 MB limit.")
    else:
        detail = response.json().get("detail", "Unknown error.")
        st.error(f"❌ Ingestion failed (HTTP {response.status_code}): {detail}")
