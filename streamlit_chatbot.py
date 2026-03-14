import os
import re
import streamlit as st
from dotenv import load_dotenv
from md_parser import load_all_episodes
from search_index import build_index
from rag_pipeline import PodcastRAG

load_dotenv()

# ─────────────────────────────────────────────
# 1. Page Config & Modern Dark UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="🎙️ Podcast Chatbot", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .stApp { background-color: #0d0d0d; color: #f0ece4; font-family: 'Syne', sans-serif; }
    
    /* Header */
    .chatbot-header { text-align: center; padding: 1.5rem 0; }
    .chatbot-header h1 {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #f0ece4 0%, #ff6b35 60%, #f0ece4 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

   /* Chat Bubbles - Updated for Clickability */
    .user-bubble {
        background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 16px 16px 4px 16px;
        padding: 1rem; margin: 0.5rem 0 0.5rem 15%; color: #f0ece4;
        position: relative;
    }
    .bot-bubble {
        background: #141414; border: 1px solid #ff6b35; border-left: 4px solid #ff6b35;
        border-radius: 4px 16px 16px 16px; padding: 1rem; margin: 0.5rem 15% 0.5rem 0;
        color: #f0ece4; line-height: 1.7;
        
        /* Forces this bubble to handle mouse clicks */
        position: relative;
        z-index: 999999 !important;
        pointer-events: auto !important; 
    }

    /* Hyperlinks - Updated for Hover and Pointer */
    .bot-bubble a {
        color: #ff6b35 !important;
        text-decoration: underline !important;
        font-weight: bold !important;
        
        /* Crucial for hover/click */
        cursor: pointer !important;
        pointer-events: auto !important;
        position: relative;
        z-index: 1000000 !important;
    }
    
    .bot-bubble a:hover {
        color: #ffffff !important;
        background-color: rgba(255, 107, 53, 0.2);
    }

    .label-user { font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #555; text-align: right; margin-right: 0.3rem; }
    .label-bot { font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #ff6b35; margin-left: 0.3rem; }
    
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 2. Helpers & Resource Loading
# ─────────────────────────────────────────────

def linkify_content(text):
    """
    Converts raw URLs into actual HTML anchor tags so they are 
    guaranteed to be clickable inside the custom bot-bubble div.
    """
    # 1. Clean up any weird brackets from the LLM
    text = text.replace('[[', '').replace(']]', '').replace('[', '').replace(']', '')
    
    # 2. Regex to find URLs
    url_pattern = r'(https?://\S+)'
    
    def replace_with_html_link(match):
        url = match.group(0)
        # Strip trailing punctuation
        clean_url = url.rstrip('.,)]')
        punctuation = url[len(clean_url):]
        # Create a REAL HTML link
        return f'<a href="{clean_url}" target="_blank">{clean_url}</a>{punctuation}'
    
    return re.sub(url_pattern, replace_with_html_link, text)

@st.cache_resource(show_spinner=False)
def load_resources():
    episodes_dir = os.environ.get("EPISODES_DIR", "data")
    try:
        segments = load_all_episodes(episodes_dir)
        index = build_index(segments)
        rag = PodcastRAG(index=index, segments=segments)
        return rag, None
    except Exception as e:
        return None, str(e)

rag_system, error_msg = load_resources()

# ─────────────────────────────────────────────
# 3. Session State & Input Handling
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

def handle_input():
    user_query = st.session_state.user_input_key.strip()
    if user_query:
        # 1. Store user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # 2. Process RAG response
        with st.spinner("Searching transcripts..."):
            raw_response = rag_system.ask(user_query)
            # 3. Apply the robust link cleaner
            formatted_response = linkify_content(raw_response)
            
            st.session_state.messages.append({"role": "assistant", "content": formatted_response})
        
        # 4. Clear input field
        st.session_state.user_input_key = ""

# ─────────────────────────────────────────────
# 4. Main UI Rendering
# ─────────────────────────────────────────────
st.markdown("<div class='chatbot-header'><h1>🎙️ Podcast Chatbot</h1></div>", unsafe_allow_html=True)

if error_msg:
    st.error(f"Data Load Error: {error_msg}. Ensure your podcast markdown files are in '/data'.")
    st.stop()

# Render History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='label-user'>YOU</div><div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        # Bot bubbles use unsafe_allow_html=True to render the Markdown hyperlinks
        st.markdown(f"<div class='label-bot'>BOT</div><div class='bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

st.markdown("---")
st.text_input(
    label="Ask a question",
    placeholder="Ask me anything about the episodes...",
    key="user_input_key",
    on_change=handle_input,
    label_visibility="collapsed"
)

with st.sidebar:
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()