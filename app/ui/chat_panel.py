"""
ui/chat_panel.py — Tier 2 chat interface.
Shows: confidence labels, conflict panel with BOTH A and C versions side by side.
No nested expanders.
"""

import streamlit as st
import requests

from app.ui import state_manager as sm

API_BASE = "http://localhost:8000"

SOURCE_ICONS  = {"A": "🏆", "B": "🔧", "C": "📜"}
SOURCE_LABELS = {
    "A": "Source A — Golden Truth",
    "B": "Source B — Real-World Fixes",
    "C": "Source C — Legacy Wiki",
}

CONFIDENCE_COLORS = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
CONFLICT_SEVERITY_COLORS = {"high": "🔴", "medium": "🟡", "low": "🔵"}


def render_chat_panel() -> None:
    loaded = st.session_state.get(sm.SOURCES_LOADED, [])
    if not loaded:
        st.info("👆 Upload at least one source above to start querying.")
        return

    provider = st.session_state.get(sm.LLM_PROVIDER, "gemini")
    st.markdown("### 💬 Query the Truth Engine")
    st.caption(
        f"Sources loaded: **{', '.join(loaded)}** · "
        f"Provider: `{provider}` · Tier 2: Hybrid Search + Re-Ranking + Confidence"
    )

    history = sm.get_chat_history()

    if not history:
        st.markdown(
            "<div style='text-align:center;color:#888;padding:24px 0;'>"
            "Ask a question — the engine queries all sources with hybrid search and re-ranks results."
            "</div>",
            unsafe_allow_html=True,
        )

    for msg in history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                _render_structured_answer(msg.get("metadata", {}), fallback=msg["content"])

    query = st.chat_input("Ask anything about your sources…")
    if not query:
        return

    sm.add_message("user", query)
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Hybrid search + re-ranking…"):
            result = _call_chat_api(query, st.session_state[sm.SESSION_ID], provider)
        if result is None:
            return
        _render_structured_answer(result)

    sm.add_message("assistant", result.get("conclusion", ""), metadata=result)
    st.session_state[sm.LAST_PROVIDER] = result.get("provider_used", provider)
    st.session_state[sm.LAST_FALLBACK] = result.get("fallback_triggered", False)
    st.rerun()


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render_structured_answer(data: dict, fallback: str = "") -> None:
    if not data:
        st.markdown(fallback)
        return

    source_keys    = [k for k in ["source_a", "source_b", "source_c"] if data.get(k)]
    label_map      = {"source_a": "A", "source_b": "B", "source_c": "C"}
    active_sources = [k for k in source_keys if data[k] and data[k].get("loaded")]

    conflict = data.get("conflict", {})
    conflict_detected = conflict.get("conflict_detected", False)

    # ── Conflict banner ───────────────────────────────────────────────────────
    if conflict_detected:
        severity = conflict.get("severity", "low")
        sev_icon = CONFLICT_SEVERITY_COLORS.get(severity, "⚠️")
        st.warning(
            f"{sev_icon} **Conflict detected** between Source A and Source C  "
            f"(Severity: **{severity.upper()}**)\n\n"
            f"{conflict.get('explanation', '')}"
        )

    # ── Per-source answers ────────────────────────────────────────────────────
    for key in active_sources:
        src     = data[key]
        label   = label_map[key]
        icon    = SOURCE_ICONS.get(label, "📄")
        heading = SOURCE_LABELS.get(label, f"Source {label}")
        summary = src.get("summary", "")
        chunks  = src.get("chunks", [])
        conf    = src.get("confidence")

        # Confidence badge
        conf_str = ""
        if conf:
            conf_icon  = CONFIDENCE_COLORS.get(conf.get("label", ""), "⚪")
            conf_label = conf.get("label", "")
            conf_score = conf.get("score", 0)
            conf_str   = f"  {conf_icon} Confidence: **{conf_label}** ({conf_score:.0%})"

        with st.expander(f"{icon} **{heading}**{conf_str}", expanded=True):
            if summary:
                st.markdown(summary)
            else:
                st.caption("_(No specific answer from this source)_")

            if chunks:
                st.markdown("---")
                st.caption(f"📎 {len(chunks)} chunk(s) after re-ranking:")
                for i, chunk in enumerate(chunks, start=1):
                    c_label = chunk.get("confidence_label", "")
                    c_icon  = CONFIDENCE_COLORS.get(c_label, "⚪")
                    c_val   = chunk.get("confidence", 0)
                    st.caption(
                        f"Chunk {chunk.get('chunk_id', i)} · "
                        f"Section ~{chunk.get('page_hint', '?')} · "
                        f"{c_icon} {c_label} ({c_val:.0%})"
                    )
                    excerpt = chunk.get("text", "")
                    st.caption(excerpt[:250] + ("…" if len(excerpt) > 250 else ""))
                    if i < len(chunks):
                        st.markdown("---")

    # ── Conflict comparison panel (show BOTH versions side by side) ───────────
    if conflict_detected:
        st.markdown("---")
        st.markdown("### ⚖️ Conflict Comparison — Both Versions")
        st.caption("Review both versions and decide which to trust.")

        col_a, col_c = st.columns(2)

        with col_a:
            st.markdown("**🏆 Source A says (Golden Truth):**")
            a_excerpt = conflict.get("source_a_excerpt", "")
            if a_excerpt:
                st.info(a_excerpt[:400] + ("…" if len(a_excerpt) > 400 else ""))
            else:
                st.caption("_(No excerpt available)_")

        with col_c:
            st.markdown("**📜 Source C says (Legacy):**")
            c_excerpt = conflict.get("source_c_excerpt", "")
            if c_excerpt:
                st.warning(c_excerpt[:400] + ("…" if len(c_excerpt) > 400 else ""))
            else:
                st.caption("_(No excerpt available)_")

        numeric_conflicts = conflict.get("numeric_conflicts", [])
        if numeric_conflicts:
            st.caption(f"Conflicting values detected: `{'`, `'.join(numeric_conflicts)}`")

    # ── Conclusion ────────────────────────────────────────────────────────────
    conclusion = data.get("conclusion", "")
    if conclusion:
        st.markdown("---")
        st.markdown("### 🎯 Conclusion")
        st.markdown(conclusion)

    # ── Provider badge ────────────────────────────────────────────────────────
    provider_used      = data.get("provider_used", "")
    fallback_triggered = data.get("fallback_triggered", False)
    if provider_used:
        icons = {"gemini": "✦", "groq": "⚡", "ollama": "🖥"}
        badge = f"{icons.get(provider_used, '🤖')} `{provider_used}`"
        if fallback_triggered:
            badge += " ⚠️ *(fallback)*"
        st.caption(f"Answered by {badge} · Tier 2: Hybrid Search + Re-Ranking")


def _call_chat_api(query: str, session_id: str, llm_provider: str) -> dict | None:
    try:
        response = requests.post(
            f"{API_BASE}/chat",
            json={"query": query, "session_id": session_id, "llm_provider": llm_provider},
            timeout=90,
        )
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the backend API. Make sure `main.py` is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out.")
        return None

    if response.status_code == 200:
        return response.json()

    detail = response.json().get("detail", "Unknown error.")
    if response.status_code == 404:
        st.error("❌ Session expired. Please re-upload your documents.")
        st.session_state[sm.SESSION_ID] = None
        st.session_state[sm.SOURCES_LOADED] = []
        st.rerun()
    elif response.status_code == 503:
        st.error(f"❌ All LLM providers unavailable: {detail}")
    elif response.status_code == 409:
        st.error(f"❌ {detail}")
    else:
        st.error(f"❌ Query failed (HTTP {response.status_code}): {detail}")
    return None
