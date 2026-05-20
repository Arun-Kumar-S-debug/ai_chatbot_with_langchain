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

STARTER_PROMPTS = [
    "✍️  Write a Python function to reverse a linked list",
    "🌍  Explain climate change in simple terms",
    "🧠  What is the difference between AI, ML, and DL?",
    "📧  Draft a professional email requesting a meeting",
    "🔢  Solve: if 2x + 5 = 17, find x",
    "🍕  Give me a quick pasta recipe",
]

# ── Session state defaults ────────────────────────────────────────────────────
def _init():
    defaults = {
        "messages":      [],       # {role, content, provider, model, latency, tokens, timestamp}
        "total_tokens":  0,
        "total_latency": 0.0,
        "msg_count":     0,
        "openai_key":    os.getenv("OPENAI_API_KEY", ""),
        "gemini_key":    os.getenv("GOOGLE_API_KEY", ""),
        "groq_key":      os.getenv("GROQ_API_KEY", ""),
        "langsmith_key": os.getenv("LANGCHAIN_API_KEY", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── LLM builder ───────────────────────────────────────────────────────────────
def build_llm(provider: str, model: str, temperature: float, max_tokens: int):
    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["env"]) or st.session_state.get(cfg["skey"], "")

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
      <div style='color:#4B5563;font-size:.62rem;letter-spacing:.18em;margin-top:3px;'>
        MULTI-PROVIDER
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Provider & Model
    st.markdown("<div style='color:#94A3B8;font-size:.72rem;letter-spacing:.1em;margin-bottom:6px;'>🤖 MODEL</div>", unsafe_allow_html=True)
    provider = st.selectbox("Provider", list(PROVIDERS.keys()), label_visibility="collapsed", key="sb_provider")
    model    = st.selectbox("Model",    PROVIDERS[provider]["models"], label_visibility="collapsed", key="sb_model")

    st.markdown("<br>", unsafe_allow_html=True)

    # Parameters
    st.markdown("<div style='color:#94A3B8;font-size:.72rem;letter-spacing:.1em;margin-bottom:6px;'>⚙️ PARAMETERS</div>", unsafe_allow_html=True)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens  = st.slider("Max Tokens",  128, 4096, 1024, 128)

    st.markdown("<br>", unsafe_allow_html=True)

    # System prompt
    st.markdown("<div style='color:#94A3B8;font-size:.72rem;letter-spacing:.1em;margin-bottom:6px;'>📋 SYSTEM PROMPT</div>", unsafe_allow_html=True)
    system_prompt = st.text_area(
        "", label_visibility="collapsed",
        value="You are a helpful, knowledgeable, and friendly AI assistant. Be concise and clear.",
        height=110, key="sb_system"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # API Keys
    st.markdown("<div style='color:#94A3B8;font-size:.72rem;letter-spacing:.1em;margin-bottom:6px;'>🔑 API KEYS</div>", unsafe_allow_html=True)
    with st.expander("Configure", expanded=not bool(os.getenv("OPENAI_API_KEY"))):
        for label, env_var, skey, ph in [
            ("OpenAI",    "OPENAI_API_KEY",    "openai_key",    "sk-…"),
            ("Gemini",    "GOOGLE_API_KEY",    "gemini_key",    "AIza…"),
            ("Groq",      "GROQ_API_KEY",      "groq_key",      "gsk_…"),
            ("LangSmith", "LANGCHAIN_API_KEY", "langsmith_key", "ls__…"),
        ]:
            val = st.text_input(label, type="password",
                                value=os.getenv(env_var, ""),
                                placeholder=ph, key=f"key_{skey}")
            if val:
                st.session_state[skey] = val
                os.environ[env_var] = val
        if st.session_state.get("langsmith_key"):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"]    = "ai-chatbot"

    st.markdown("<br>", unsafe_allow_html=True)

    # Provider status
    st.markdown("<div style='color:#94A3B8;font-size:.72rem;letter-spacing:.1em;margin-bottom:8px;'>📡 STATUS</div>", unsafe_allow_html=True)
    for pname, pcfg in PROVIDERS.items():
        short = pname.split(" ")[0]
        ok    = bool(os.getenv(pcfg["env"]) or st.session_state.get(pcfg["skey"]))
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
<div style='display:flex;align-items:center;justify-content:space-between;
            padding:1rem 0 0.8rem;border-bottom:1px solid #1E2A3A;margin-bottom:1rem;'>
  <div>
    <span style='font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;
      background:linear-gradient(135deg,#00D4FF,#7C3AED);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
      💬 Chat
    </span>
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
  <div class='stat-item'>💬 Messages <span class='stat-val'>{n}</span></div>
  <div class='stat-item'>🪙 Tokens <span class='stat-val'>{tok:,}</span></div>
  <div class='stat-item'>⏱ Avg latency <span class='stat-val'>{lat}s</span></div>
  <div class='stat-item' style='margin-left:auto;'>
    <div style='width:7px;height:7px;border-radius:50%;background:{pcolor};'></div>
    <span style='color:{pcolor};font-size:.7rem;'>{provider.split(" ")[0]}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Chat window ───────────────────────────────────────────────────────────────
messages = st.session_state["messages"]

if not messages:
    # Greeting / starter prompts
    st.markdown("""
    <div class='greeting'>
      <div class='greeting-icon'>🤖</div>
      <div class='greeting-title'>How can I help you today?</div>
      <div>Pick a starter or type your own message below</div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    for i, prompt in enumerate(STARTER_PROMPTS):
        if cols[i % 2].button(prompt, key=f"starter_{i}", use_container_width=True):
            clean = prompt.split("  ", 1)[-1]   # strip emoji prefix
            st.session_state["messages"].append({"role": "user", "content": clean})
            st.rerun()

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
