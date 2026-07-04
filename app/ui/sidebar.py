"""
ui/sidebar.py — Sidebar: LLM provider selector + sources status panel.
No business logic — UI only.
"""

import streamlit as st

from app.ui import state_manager as sm

PROVIDER_LABELS = {
    "gemini": "✦ Gemini (Primary · Best Quality)",
    "groq":   "⚡ Groq (Secondary · Fastest)",
    "ollama": "🖥 Ollama (Tertiary · Local Fallback)",
}
PROVIDER_KEYS = list(PROVIDER_LABELS.keys())

SOURCE_META = {
    "A": {"icon": "🏆", "label": "Golden Truth",   "color": "green"},
    "B": {"icon": "🔧", "label": "Real-World Fixes","color": "blue"},
    "C": {"icon": "📜", "label": "Legacy Wiki",     "color": "orange"},
}


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.divider()

        # ── LLM Provider selector ─────────────────────────────────────────────
        current  = st.session_state.get(sm.LLM_PROVIDER, "gemini")
        cur_idx  = PROVIDER_KEYS.index(current) if current in PROVIDER_KEYS else 0

        selected_label = st.selectbox(
            "LLM Provider",
            options=list(PROVIDER_LABELS.values()),
            index=cur_idx,
            help="Primary LLM. Falls back automatically: Gemini → Groq → Ollama.",
        )
        selected_key = PROVIDER_KEYS[list(PROVIDER_LABELS.values()).index(selected_label)]
        st.session_state[sm.LLM_PROVIDER] = selected_key

        st.caption("**Fallback chain:**")
        chain = []
        for k in PROVIDER_KEYS:
            chain.append(f"**{k.capitalize()} ←**" if k == selected_key else k.capitalize())
        st.caption(" → ".join(chain))

        st.divider()

        # ── Sources status ────────────────────────────────────────────────────
        st.markdown("## 📚 Sources Status")

        file_keys   = {"A": sm.SOURCE_A_FILE,   "B": sm.SOURCE_B_FILE,   "C": sm.SOURCE_C_FILE}
        chunk_keys  = {"A": sm.SOURCE_A_CHUNKS, "B": sm.SOURCE_B_CHUNKS, "C": sm.SOURCE_C_CHUNKS}
        type_keys   = {"A": sm.SOURCE_A_TYPE,   "B": sm.SOURCE_B_TYPE,   "C": sm.SOURCE_C_TYPE}

        for label, meta in SOURCE_META.items():
            fname  = st.session_state.get(file_keys[label])
            chunks = st.session_state.get(chunk_keys[label], 0)
            ftype  = st.session_state.get(type_keys[label])

            if fname:
                st.success(
                    f"{meta['icon']} **Source {label}** · {meta['label']}\n\n"
                    f"`{fname}` · {chunks} chunks · {ftype}"
                )
            else:
                st.warning(f"{meta['icon']} **Source {label}** · {meta['label']}\n\nNot loaded")

        loaded = st.session_state.get(sm.SOURCES_LOADED, [])
        if loaded:
            st.caption(f"Sources loaded: **{', '.join(loaded)}**")
        else:
            st.caption("No sources loaded yet.")

        st.divider()

        # ── Session info ──────────────────────────────────────────────────────
        session_id = st.session_state.get(sm.SESSION_ID)
        if session_id:
            st.markdown("## 🔑 Session")
            st.code(session_id[:18] + "…", language=None)

        # ── Last request metadata ──────────────────────────────────────────────
        last_provider = st.session_state.get(sm.LAST_PROVIDER)
        if last_provider:
            st.markdown("## 🔍 Last Request")
            st.caption(f"Provider used: **{last_provider}**")
            if st.session_state.get(sm.LAST_FALLBACK):
                st.warning("⚠️ Fallback was triggered on last query.")

        st.divider()
        st.caption("Truth Engine v1.0 · Tier 1 · Multi-source RAG")
