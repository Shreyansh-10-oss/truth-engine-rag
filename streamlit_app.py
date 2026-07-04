"""
streamlit_app.py — Truth Engine Tier 2 Streamlit entry point.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st

from app.ui import state_manager as sm
from app.ui.chat_panel import render_chat_panel
from app.ui.sidebar import render_sidebar
from app.ui.upload_panel import render_upload_panel

st.set_page_config(
    page_title="Truth Engine — Tier 2",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
[data-testid="stFileUploader"] {
    border: 1.5px dashed #555;
    border-radius: 8px;
    padding: 0.5rem;
}
[data-testid="stChatMessage"] { padding: 0.6rem 0; }
.stAlert { border-radius: 6px; }
.streamlit-expanderHeader { font-size: 0.9rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

sm.init_state()
render_sidebar()

st.title("⚖️ Truth Engine — Tier 2")
st.caption(
    "Hybrid Search (BM25 + FAISS) · Cross-Encoder Re-Ranking · "
    "Confidence Scoring · Conflict Detection"
)

st.divider()
render_upload_panel()
st.divider()
render_chat_panel()

if sm.get_chat_history():
    st.markdown("---")
    if st.button("🗑 Clear chat history"):
        sm.clear_chat()
        st.rerun()
