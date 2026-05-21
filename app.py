"""
💬 AI Chatbot — Multi-provider conversational interface
Supports OpenAI, Gemini, Groq with LangSmith tracing
"""
import os
import time
from dotenv import load_dotenv
load_dotenv()

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "ai-chatbot")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

import streamlit as st

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@700;800&display=swap');

:root {
    --bg:      #0A0E1A;
    --card:    #111827;
    --border:  #1E2A3A;
    --cyan:    #00D4FF;
    --violet:  #7C3AED;
    --green:   #10B981;
    --rose:    #F43F5E;
    --amber:   #F59E0B;
    --muted:   #4B5563;
    --text:    #E2E8F0;
    --subtext: #94A3B8;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid var(--border);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Scrollable chat window */
.chat-window {
    height: 62vh;
    overflow-y: auto;
    padding: 1.2rem 0.5rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
}
.chat-window::-webkit-scrollbar { width: 4px; }
.chat-window::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* Message bubbles */
.msg-row { display: flex; align-items: flex-end; gap: 10px; animation: fadeUp .25s ease; }
.msg-row.user { flex-direction: row-reverse; }

@keyframes fadeUp {
    from { opacity:0; transform: translateY(10px); }
    to   { opacity:1; transform: translateY(0); }
}

.avatar {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.avatar-user { background: linear-gradient(135deg, var(--violet), var(--cyan)); }
.avatar-ai   { background: linear-gradient(135deg, #1C2333, #2D3748); border: 1px solid var(--border); }

.bubble {
    max-width: 72%;
    padding: 0.85rem 1.1rem;
    border-radius: 16px;
    font-size: 0.875rem;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
}
.bubble-user {
    background: linear-gradient(135deg, #3B1F8C, #1E3A5F);
    border: 1px solid #4C1D95;
    border-bottom-right-radius: 4px;
    color: #E9D5FF;
}
.bubble-ai {
    background: var(--card);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
    color: var(--text);
}

.msg-meta {
    font-size: 0.65rem;
    color: var(--muted);
    margin-top: 4px;
    padding: 0 4px;
}
.msg-meta.user { text-align: right; }

/* Typing indicator */
.typing-dots { display: flex; gap: 5px; align-items: center; padding: 0.5rem 0; }
.typing-dots span {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--cyan); opacity: 0.4;
    animation: blink 1.2s infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%,80%,100%{opacity:.4} 40%{opacity:1} }

/* Input bar */
.input-bar {
    display: flex;
    gap: 10px;
    align-items: flex-end;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.6rem 0.8rem;
    margin-top: 0.8rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--violet), var(--cyan)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity .2s, transform .1s !important;
}
.stButton > button:hover { opacity: .85 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* Inputs */
.stTextInput input, .stTextArea textarea {
    background: #0D1117 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.875rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--violet) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,.25) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #0D1117 !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }

/* Provider badge */
.provider-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1C2333; border: 1px solid var(--border);
    border-radius: 20px; padding: 3px 10px;
    font-size: 0.68rem; color: var(--subtext);
}
.provider-dot { width: 7px; height: 7px; border-radius: 50%; }

/* Stats bar */
.stats-bar {
    display: flex; gap: 1.5rem; align-items: center;
    padding: 0.6rem 1rem;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; margin-bottom: 0.8rem;
    font-size: 0.72rem; color: var(--subtext);
}
.stat-item { display: flex; gap: 6px; align-items: center; }
.stat-val { color: var(--cyan); font-weight: 600; }

/* Greeting card */
.greeting {
    text-align: center; padding: 3rem 1rem;
    color: var(--muted); font-size: 0.85rem;
}
.greeting-icon { font-size: 3rem; margin-bottom: 1rem; }
.greeting-title {
    font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800;
    background: linear-gradient(135deg, var(--cyan), var(--violet));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--subtext) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Polished app shell */
[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(145deg, rgba(20,184,166,.10), transparent 28%),
        linear-gradient(215deg, rgba(244,63,94,.08), transparent 32%),
        linear-gradient(180deg, #08111f 0%, #0b1020 48%, #090b12 100%) !important;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: linear-gradient(to bottom, rgba(0,0,0,.85), transparent 78%);
}

.block-container {
    max-width: 1180px;
    padding-top: 1.4rem !important;
    padding-bottom: 1rem !important;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(16,185,129,.10), transparent 24%),
        #080d16 !important;
    box-shadow: 12px 0 40px rgba(0,0,0,.22);
}

[data-testid="stSidebar"] * {
    letter-spacing: 0 !important;
}

.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 1rem 1.1rem;
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 18px;
    background:
        linear-gradient(135deg, rgba(20,184,166,.08), rgba(124,58,237,.07)),
        rgba(15,23,42,.76);
    box-shadow: 0 18px 50px rgba(0,0,0,.24);
    backdrop-filter: blur(14px);
    margin-bottom: .9rem;
}

.title-stack { display: flex; align-items: center; gap: .8rem; }
.title-mark {
    width: 44px; height: 44px; border-radius: 14px;
    display: grid; place-items: center;
    background: linear-gradient(135deg, #14B8A6, #7C3AED 55%, #F43F5E);
    box-shadow: 0 12px 28px rgba(20,184,166,.20);
    font-size: 1.3rem;
    border: 1px solid rgba(255,255,255,.14);
}
.title-copy h1 {
    margin: 0;
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    line-height: 1;
    letter-spacing: 0;
    color: #F8FAFC;
}
.title-copy p {
    margin: .35rem 0 0;
    color: #94A3B8;
    font-size: .76rem;
}

.provider-badge {
    background: rgba(8,13,22,.78);
    border: 1px solid rgba(148,163,184,.22);
    border-radius: 999px;
    padding: .48rem .72rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.06);
}

.stats-bar {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: .8rem;
    padding: 0;
    background: transparent;
    border: 0;
    border-radius: 0;
    margin-bottom: 1rem;
}
.stat-item {
    min-height: 70px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    gap: .24rem;
    padding: .85rem 1rem;
    border: 1px solid rgba(148,163,184,.16);
    border-radius: 14px;
    background:
        linear-gradient(180deg, rgba(255,255,255,.035), transparent),
        rgba(15,23,42,.58);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
}
.stat-label { color: #94A3B8; font-size: .68rem; }
.stat-val { color: #F8FAFC; font-size: 1.05rem; font-weight: 700; }

.chat-window {
    height: 58vh;
    padding: 1.1rem;
    border: 1px solid rgba(148,163,184,.16);
    border-radius: 18px;
    background: rgba(2,6,23,.42);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
}

.avatar {
    width: 38px;
    height: 38px;
    box-shadow: 0 10px 24px rgba(0,0,0,.22);
}
.bubble {
    max-width: min(760px, 74%);
    border-radius: 18px;
    box-shadow: 0 12px 32px rgba(0,0,0,.20);
}
.bubble-user {
    background: linear-gradient(135deg, #0F766E, #4C1D95);
    border-color: rgba(45,212,191,.32);
    color: #F8FAFC;
}
.bubble-ai {
    background: linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.96));
    border-color: rgba(148,163,184,.18);
}

.greeting {
    min-height: 46vh;
    display: grid;
    place-items: center;
    padding: 2rem 1rem;
    border: 1px solid rgba(148,163,184,.16);
    border-radius: 18px;
    background:
        linear-gradient(135deg, rgba(20,184,166,.12), transparent 44%),
        linear-gradient(225deg, rgba(244,63,94,.10), transparent 46%),
        rgba(2,6,23,.42);
    position: relative;
    overflow: hidden;
}
.greeting::after {
    content: "";
    position: absolute;
    inset: 1px;
    border-radius: 17px;
    pointer-events: none;
    background:
        radial-gradient(circle at 50% 0%, rgba(255,255,255,.10), transparent 30%),
        linear-gradient(90deg, transparent, rgba(255,255,255,.06), transparent);
    opacity: .75;
}
.greeting > div {
    position: relative;
    z-index: 1;
}
.greeting-icon {
    width: 68px;
    height: 68px;
    margin: 0 auto 1rem;
    border-radius: 20px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #14B8A6, #7C3AED 55%, #F43F5E);
    box-shadow: 0 20px 50px rgba(20,184,166,.16);
}
.greeting-title {
    color: #F8FAFC;
    background: none;
    -webkit-text-fill-color: #F8FAFC;
    font-size: 1.75rem;
}
.greeting-copy {
    max-width: 520px;
    color: #94A3B8;
    font-size: .86rem;
    line-height: 1.65;
}

.stButton > button, .stFormSubmitButton > button {
    min-height: 2.65rem;
    background: linear-gradient(135deg, #14B8A6, #7C3AED) !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 28px rgba(20,184,166,.16) !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    filter: brightness(1.08);
    opacity: 1 !important;
}
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div {
    background: rgba(2,6,23,.72) !important;
    border-color: rgba(148,163,184,.18) !important;
    border-radius: 12px !important;
}
.stTextInput input {
    min-height: 2.65rem;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: rgba(148,163,184,.66) !important;
}
[data-testid="stSidebar"] .stMarkdown hr {
    border-color: rgba(148,163,184,.14);
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: .55rem;
}

@media (max-width: 760px) {
    .app-header { align-items: flex-start; flex-direction: column; }
    .stats-bar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .bubble { max-width: 84%; }
    .title-copy h1 { font-size: 1.35rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PROVIDERS = {
    "OpenAI (GPT-4o)": {
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "color":  "#10B981",
        "dot":    "#10B981",
        "env":    "OPENAI_API_KEY",
        "skey":   "openai_key",
    },
    "Gemini (Google)": {
        "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
        "color":  "#00D4FF",
        "dot":    "#00D4FF",
        "env":    "GOOGLE_API_KEY",
        "skey":   "gemini_key",
    },
    "Groq (LLaMA-3)": {
        "models": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
        "color":  "#F59E0B",
        "dot":    "#F59E0B",
        "env":    "GROQ_API_KEY",
        "skey":   "groq_key",
    },
}

# ── Session state defaults ────────────────────────────────────────────────────
def _init():
    defaults = {
        "messages":      [],       # {role, content, provider, model, latency, tokens, timestamp}
        "total_tokens":  0,
        "total_latency": 0.0,
        "msg_count":     0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── LLM builder ───────────────────────────────────────────────────────────────
def build_llm(provider: str, model: str, temperature: float, max_tokens: int):
    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["env"], "")

    if provider == "OpenAI (GPT-4o)":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature,
                          max_tokens=max_tokens, api_key=api_key)

    elif provider == "Gemini (Google)":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature,
                                      max_output_tokens=max_tokens, google_api_key=api_key)

    elif provider == "Groq (LLaMA-3)":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature,
                        max_tokens=max_tokens, groq_api_key=api_key)

    raise ValueError(f"Unknown provider: {provider}")


def call_llm(provider, model, system_prompt, history, temperature, max_tokens):
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    llm = build_llm(provider, model, temperature, max_tokens)

    lc_messages = [SystemMessage(content=system_prompt)]
    for m in history:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    t0 = time.time()
    try:
        resp    = llm.invoke(lc_messages)
        latency = round(time.time() - t0, 2)
        usage   = {}
        if hasattr(resp, "response_metadata"):
            usage = (resp.response_metadata.get("token_usage")
                     or resp.response_metadata.get("usage_metadata") or {})
        tokens = (usage.get("total_tokens")
                  or usage.get("totalTokenCount")
                  or usage.get("output_tokens", 0))
        return {"ok": True,  "content": resp.content, "latency": latency, "tokens": int(tokens or 0)}
    except Exception as e:
        return {"ok": False, "content": f"❌ {e}", "latency": round(time.time()-t0,2), "tokens": 0}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style='text-align:center;padding:1.2rem 0 0.8rem;'>
      <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;
                  background:linear-gradient(135deg,#00D4FF,#7C3AED);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        ⚡ AI Chatbot
      </div>
      <div style='color:#4B5563;font-size:.62rem;letter-spacing:0;margin-top:3px;'>
        MULTI-PROVIDER
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Provider & Model
    st.markdown("<div style='color:#94A3B8;font-size:.72rem;letter-spacing:0;margin-bottom:6px;'>🤖 MODEL</div>", unsafe_allow_html=True)
    provider = st.selectbox("Provider", list(PROVIDERS.keys()), label_visibility="collapsed", key="sb_provider")
    model    = st.selectbox("Model",    PROVIDERS[provider]["models"], label_visibility="collapsed", key="sb_model")

    st.markdown("<br>", unsafe_allow_html=True)

    # Parameters
    st.markdown("<div style='color:#94A3B8;font-size:.72rem;letter-spacing:0;margin-bottom:6px;'>⚙️ PARAMETERS</div>", unsafe_allow_html=True)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens  = st.slider("Max Tokens",  128, 4096, 1024, 128)

    st.markdown("<br>", unsafe_allow_html=True)

    # System prompt
    st.markdown("<div style='color:#94A3B8;font-size:.72rem;letter-spacing:0;margin-bottom:6px;'>📋 SYSTEM PROMPT</div>", unsafe_allow_html=True)
    system_prompt = st.text_area(
        "", label_visibility="collapsed",
        value="You are a helpful, knowledgeable, and friendly AI assistant. Be concise and clear.",
        height=110, key="sb_system"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Provider status
    st.markdown("<div style='color:#94A3B8;font-size:.72rem;letter-spacing:0;margin-bottom:8px;'>📡 STATUS</div>", unsafe_allow_html=True)
    for pname, pcfg in PROVIDERS.items():
        short = pname.split(" ")[0]
        ok    = bool(os.getenv(pcfg["env"]))
        color = pcfg["color"] if ok else "#4B5563"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:.75rem;'>"
            f"<div style='width:7px;height:7px;border-radius:50%;background:{color};'></div>"
            f"<span style='color:{color};'>{short}</span>"
            f"<span style='color:#4B5563;margin-left:auto;'>{'●' if ok else '○'}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Clear button
    if st.button("🗑️  Clear Chat", use_container_width=True):
        st.session_state["messages"]      = []
        st.session_state["total_tokens"]  = 0
        st.session_state["total_latency"] = 0.0
        st.session_state["msg_count"]     = 0
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
# Header
pcfg  = PROVIDERS[provider]
pcolor = pcfg["color"]
st.markdown(f"""
<div class='app-header'>
  <div class='title-stack'>
    <div class='title-mark'>💬</div>
    <div class='title-copy'>
      <h1>AI Chat Studio</h1>
      <p>Fast conversations across OpenAI, Gemini, and Groq.</p>
    </div>
  </div>
  <div class='provider-badge'>
    <div class='provider-dot' style='background:{pcolor};'></div>
    <span style='color:{pcolor};'>{provider}</span>
    <span style='color:#4B5563;'>·</span>
    <span>{model}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Stats bar
n   = st.session_state["msg_count"]
tok = st.session_state["total_tokens"]
lat = round(st.session_state["total_latency"] / n, 2) if n else 0

st.markdown(f"""
<div class='stats-bar'>
  <div class='stat-item'><span class='stat-label'>Messages</span><span class='stat-val'>{n}</span></div>
  <div class='stat-item'><span class='stat-label'>Tokens</span><span class='stat-val'>{tok:,}</span></div>
  <div class='stat-item'><span class='stat-label'>Avg latency</span><span class='stat-val'>{lat}s</span></div>
  <div class='stat-item'>
    <span class='stat-label'>Active model</span>
    <span class='stat-val' style='color:{pcolor};font-size:.92rem;'>{provider.split(" ")[0]}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Chat window ───────────────────────────────────────────────────────────────
messages = st.session_state["messages"]

if not messages:
    # Greeting
    st.markdown("""
    <div class='greeting'>
      <div>
        <div class='greeting-icon'>✦</div>
        <div class='greeting-title'>What are we building today?</div>
        <div class='greeting-copy'>Ask anything below and your selected model will pick it up from here.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Render message history
    chat_html = "<div class='chat-window' id='chat-bottom'>"
    for msg in messages:
        role    = msg["role"]
        content = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
        ts      = msg.get("timestamp", "")

        if role == "user":
            chat_html += f"""
            <div class='msg-row user'>
              <div class='avatar avatar-user'>👤</div>
              <div>
                <div class='bubble bubble-user'>{content}</div>
                <div class='msg-meta user'>{ts}</div>
              </div>
            </div>"""
        else:
            prov    = msg.get("provider", provider).split(" ")[0]
            mdl     = msg.get("model", model)
            lat_msg = msg.get("latency", "")
            tok_msg = msg.get("tokens", "")
            meta    = f"{prov} · {mdl} · {lat_msg}s · {tok_msg} tokens"
            chat_html += f"""
            <div class='msg-row ai'>
              <div class='avatar avatar-ai'>🤖</div>
              <div>
                <div class='bubble bubble-ai'>{content}</div>
                <div class='msg-meta'>{meta} &nbsp;·&nbsp; {ts}</div>
              </div>
            </div>"""

    chat_html += "</div>"
    chat_html += "<script>document.getElementById('chat-bottom').scrollTop=99999;</script>"
    st.markdown(chat_html, unsafe_allow_html=True)

# ── Input area ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([6, 1])

    with col_in:
        user_input = st.text_input(
            "", placeholder="Type a message…  (Enter to send)",
            label_visibility="collapsed", key="chat_input"
        )
    with col_btn:
        send_btn = st.form_submit_button("Send ➤", use_container_width=True)

# Optional: multi-line toggle
with st.expander("⌨️  Multi-line input"):
    with st.form("multiline_form", clear_on_submit=True):
        ml_input = st.text_area("", placeholder="Paste long text here…", height=120,
                                label_visibility="collapsed", key="ml_input")
        ml_send_btn = st.form_submit_button("Send multi-line ➤")

# ── Handle send ───────────────────────────────────────────────────────────────
def handle_send(text: str):
    if not text.strip():
        return

    ts = time.strftime("%H:%M")
    st.session_state["messages"].append({"role": "user", "content": text, "timestamp": ts})

    with st.spinner(""):
        result = call_llm(
            provider, model, system_prompt,
            st.session_state["messages"],
            temperature, max_tokens
        )

    st.session_state["messages"].append({
        "role":      "assistant",
        "content":   result["content"],
        "provider":  provider,
        "model":     model,
        "latency":   result["latency"],
        "tokens":    result["tokens"],
        "timestamp": time.strftime("%H:%M"),
    })

    st.session_state["total_tokens"]  += result["tokens"]
    st.session_state["total_latency"] += result["latency"]
    st.session_state["msg_count"]     += 1
    st.rerun()


if send_btn and user_input.strip():
    handle_send(user_input)
elif ml_send_btn and ml_input.strip():
    handle_send(ml_input)
