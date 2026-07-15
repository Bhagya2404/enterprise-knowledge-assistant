import os
import re
import time
import random
 
import streamlit as st
 
from dotenv import load_dotenv
from openai import AzureOpenAI
 
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
 
from orchestrator import route_query
 
load_dotenv()
 
 
def get_secret(key):
    """Reads from Streamlit secrets first (for Streamlit Cloud), falls back to .env locally."""
    return st.secrets.get(key, os.getenv(key))
 
 
# ==================================================
# GRATITUDE DETECTION
# A short "thank you" style message gets a warm,
# instant reply instead of going through the RAG
# pipeline.
# ==================================================
GRATITUDE_PHRASES = {
    "thank you", "thanks", "thank u", "thankyou", "thanks a lot",
    "thank you so much", "thanks so much", "many thanks",
    "thank you very much", "ty", "tysm", "appreciate it",
    "appreciated", "much appreciated"
}
 
GRATITUDE_REPLIES = [
    "You're most welcome! 🌻 Happy to assist you anytime.",
    "Anytime! Glad I could help — let me know if anything else comes up. 😊",
    "You're welcome! Always happy to help with your workplace questions. 🌿",
    "My pleasure! Feel free to ask if anything else comes to mind. ☀️",
]
 
 
def is_gratitude(text: str) -> bool:
    cleaned = re.sub(r"[^a-z\s]", "", text.lower()).strip()
    if cleaned in GRATITUDE_PHRASES:
        return True
    return len(cleaned.split()) <= 6 and any(
        phrase in cleaned for phrase in ["thank you", "thanks", "thankyou", "ty"]
    )
 
 
# ==================================================
# GREETING DETECTION
# ==================================================
GREETING_PHRASES = {
    "hi", "hii", "hello", "hey",
    "good morning", "good afternoon", "good evening", "hola"
}
 
 
def is_greeting(text: str) -> bool:
    cleaned = re.sub(r"[^a-z\s]", "", text.lower()).strip()
    return cleaned in GREETING_PHRASES
 
 
GREETING_REPLY = """
### 👋 Welcome to Enterprise AI Assistant
 
I can help you with:
 
✅ Employee Policies
✅ Leave & Attendance Information
✅ Work From Home Guidelines
✅ Travel Reimbursements
✅ IT & VPN Support
✅ Ticket Assistance
✅ Enterprise Knowledge Search
 
How can I help you today?
"""
 
 
# ==================================================
# TYPEWRITER REVEAL HELPERS
# Renders text progressively into a placeholder so
# responses feel like they're being typed live.
# ==================================================
def typewriter_markdown(placeholder, text, delay=0.028):
    words = text.split(" ")
    shown = ""
    for i, word in enumerate(words):
        shown += word + (" " if i < len(words) - 1 else "")
        cursor = '<span class="type-caret">&nbsp;</span>' if i < len(words) - 1 else ""
        placeholder.markdown(shown + cursor, unsafe_allow_html=True)
        time.sleep(delay)
    placeholder.markdown(text)
 
 
def typewriter_alert(placeholder, text, kind="info", delay=0.028):
    renderer = {"success": placeholder.success, "info": placeholder.info, "error": placeholder.error}[kind]
    words = text.split(" ")
    shown = ""
    for word in words:
        shown += word + " "
        renderer(shown.strip())
        time.sleep(delay)
 
 
# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ==================================================
# LOAD CUSTOM CSS
# ==================================================
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
 
# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
 
    st.markdown(
        """
<div class="brand-block">
<div class="brand-seal">EA</div>
<h2>Enterprise AI</h2>
<p>Knowledge Intelligence Platform</p>
</div>
        """,
        unsafe_allow_html=True
    )
 
    st.markdown("---")
 
    st.markdown('<div class="side-label">Model Configuration</div>', unsafe_allow_html=True)
 
    model_name = st.selectbox(
        "Select AI Model",
        ["gpt-4.1-mini", "gpt-5.1"],
        label_visibility="collapsed"
    )
 
    st.markdown("---")
 
    st.markdown('<div class="side-label">System Status</div>', unsafe_allow_html=True)
 
    st.markdown(
        """
<div class="status-plate">
<div class="status-row">
<span class="label">AI Services</span>
<span class="value">
<span class="pulse-bars">
<span></span><span></span><span></span>
</span>
<span class="dot dot-live"></span>Online
</span>
</div>
<div class="status-row">
<span class="label">Knowledge Base</span>
<span class="value"><span class="dot dot-live"></span>Connected</span>
</div>
<div class="status-row">
<span class="label">Security</span>
<span class="value"><span class="dot dot-linked"></span>Governed</span>
</div>
</div>
        """,
        unsafe_allow_html=True
    )
 
    st.markdown("---")
 
    st.markdown('<div class="side-label">Capabilities</div>', unsafe_allow_html=True)
 
    st.markdown(
        """
<ul class="capability-list">
<li>Policy intelligence</li>
<li>HR knowledge search</li>
<li>Attendance support</li>
<li>Leave management</li>
<li>Reimbursement queries</li>
<li>IT support</li>
<li>Ticket assistance</li>
<li>Governance controls</li>
</ul>
        """,
        unsafe_allow_html=True
    )
 
# ==================================================
# HERO SECTION
# ==================================================
st.markdown(
    """
<div class="hero">
<div>
<div class="hero-eyebrow">Enterprise AI Assistant</div>
<h1>Enterprise Knowledge Assistant</h1>
<p>
                Ask about company policy, leave, reimbursement, or IT access.
                Every answer is grounded in your organization's own knowledge
                base, and sensitive requests are routed through your
                approval workflow.
</p>
</div>
<div class="hero-side">
<div class="stamp-badge">
<span class="stamp-text">Verified<br>AI Online</span>
</div>
<div class="hero-activity">
<div class="pulse-bars">
<span></span><span></span><span></span><span></span><span></span>
</div>
                Live
</div>
</div>
</div>
    """,
    unsafe_allow_html=True
)
 
# ==================================================
# DASHBOARD METRICS
# ==================================================
st.markdown("## Workspace Overview")
 
c1, c2, c3, c4 = st.columns(4)
 
with c1:
    st.metric("Knowledge Assets", "15")
with c2:
    st.metric("AI Assistant", "Online")
with c3:
    st.metric("AI Models", "2")
with c4:
    st.metric("Security", "Governed")
 
# ==================================================
# SESSION MEMORY
# ==================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
 
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
 
# ==================================================
# WELCOME SCREEN
# ==================================================
if len(st.session_state.messages) == 0:
 
    st.markdown(
        """
<div class="welcome-card">
<h2>Welcome to Enterprise AI Assistant</h2>
<p>
                Ask questions about your organization's policies, processes,
                and support services. Answers come from your internal
                knowledge base — not the open internet.
</p>
<div class="feature-grid">
<div>📄<b>Policies</b>HR rules and guidelines</div>
<div>🏠<b>Work From Home</b>Remote work policies</div>
<div>💳<b>Expenses</b>Reimbursement help</div>
<div>🔐<b>Security</b>VPN and access</div>
</div>
</div>
        """,
        unsafe_allow_html=True
    )
 
# ==================================================
# SUGGESTED QUERIES — clickable, functional
# ==================================================
st.markdown('<div class="section-label">Suggested Queries</div>', unsafe_allow_html=True)
 
suggestions = [
    "What is the WFH policy?",
    "How many casual leaves are available?",
    "What is the travel reimbursement policy?",
    "Explain VPN access process"
]
 
cols = st.columns(4)
 
for i, item in enumerate(suggestions):
    with cols[i]:
        if st.button(item, key=f"suggestion_{i}", use_container_width=True):
            st.session_state.pending_prompt = item
 
# ==================================================
# CHAT HISTORY
# ==================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
# ==================================================
# AZURE OPENAI INITIALIZATION
# ==================================================
client = AzureOpenAI(
    api_key=get_secret("AZURE_OPENAI_API_KEY"),
    azure_endpoint=get_secret("AZURE_OPENAI_ENDPOINT"),
    api_version=get_secret("API_VERSION")
)
 
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=get_secret("AZURE_OPENAI_ENDPOINT"),
    api_key=get_secret("AZURE_OPENAI_API_KEY"),
    api_version=get_secret("API_VERSION"),
    azure_deployment=get_secret("EMBEDDING_DEPLOYMENT")
)
 
vectorstore = FAISS.load_local(
    "vectordb",
    embeddings,
    allow_dangerous_deserialization=True
)
 
# ==================================================
# CHAT INPUT
# ==================================================
typed_prompt = st.chat_input("Ask anything about your enterprise...")
 
# A suggestion click fills the prompt just like typing would.
prompt = typed_prompt or st.session_state.pending_prompt
st.session_state.pending_prompt = None
 
# ==================================================
# QUERY PROCESSING
# ==================================================
if prompt:
 
    st.session_state.messages.append({"role": "user", "content": prompt})
 
    with st.chat_message("user"):
        st.markdown(prompt)
 
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            '<div class="typing-dots"><span></span><span></span><span></span></div>'
            '<span class="typing-label">Assistant is typing…</span>',
            unsafe_allow_html=True
        )
 
        response_text = ""
 
        # -------------------------------
        # GREETING
        # -------------------------------
        if is_greeting(prompt):
 
            response_text = GREETING_REPLY
            time.sleep(0.5)
            typewriter_markdown(placeholder, response_text, delay=0.03)
 
        # -------------------------------
        # GRATITUDE — instant warm reply, skips the pipeline
        # -------------------------------
        elif is_gratitude(prompt):
 
            response_text = random.choice(GRATITUDE_REPLIES)
            time.sleep(0.5)
            typewriter_markdown(placeholder, response_text, delay=0.05)
 
        else:
            result = route_query(prompt)
 
            # -------------------------------
            # GOVERNANCE
            # -------------------------------
            if result["type"] == "approval":
 
                response_text = (
                    "Approval required. This request needs authorization "
                    "through your organization's approval workflow before "
                    "it can proceed."
                )
                typewriter_alert(placeholder, response_text, kind="error")
 
            # -------------------------------
            # CALCULATOR
            # -------------------------------
            elif result["type"] == "calculator":
 
                response_text = result["response"]
                typewriter_alert(placeholder, response_text, kind="success")
 
            # -------------------------------
            # TICKET
            # -------------------------------
            elif result["type"] == "ticket":
 
                response_text = result["response"]
                typewriter_alert(placeholder, response_text, kind="info")
 
            # -------------------------------
            # RAG
            # -------------------------------
            else:
 
                docs = vectorstore.similarity_search(prompt, k=3)
 
                context = "\n\n".join([doc.page_content for doc in docs])
 
                llm_prompt = f"""
You are an Enterprise Knowledge Assistant.
 
Answer only using the context provided.
 
If information is unavailable say:
"I could not find that information in the knowledge base."
 
Context:
{context}
 
Question:
{prompt}
 
Answer:
"""
 
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": llm_prompt}]
                )
 
                response_text = response.choices[0].message.content
                typewriter_markdown(placeholder, response_text)
 
    st.session_state.messages.append({"role": "assistant", "content": response_text})
 
# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
 
st.markdown(
    """
<div class="footer">
        Enterprise Knowledge Assistant &nbsp;·&nbsp;
        Secure Internal AI Workspace &nbsp;·&nbsp; Version 2.0
</div>
    """,
    unsafe_allow_html=True
)