import streamlit as st
from groq import Groq
from transformers import pipeline
import time

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="Sentiment-Aware Intelligent Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #F4F6F9 !important; }
#MainMenu, footer, header { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: #1E293B !important;
    border-right: 1px solid #334155 !important;
}
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: #2563EB !important; color: #fff !important;
    border: none !important; border-radius: 8px !important; font-weight: 600 !important;
}
.sb-title { font-size: 1.1rem; font-weight: 700; color: #F8FAFC !important; }
.sb-sub   { font-size: 0.75rem; color: #94A3B8 !important; margin-top: 2px; margin-bottom: 18px; }
.sb-label { font-size: 0.65rem; font-weight: 600; color: #64748B !important; text-transform: uppercase; letter-spacing: 1px; margin: 16px 0 8px; }
.sb-model-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:6px 0; border-bottom:1px solid #334155;
    font-size:0.79rem; color:#CBD5E1 !important;
}
.sb-model-row:last-child { border-bottom:none; }
.sb-mono { font-family:monospace !important; font-size:0.69rem; color:#64748B !important; }
.stat-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin-top:4px; }
.stat-box  { background:#0F172A; border:1px solid #334155; border-radius:8px; padding:10px 6px; text-align:center; }
.stat-num  { font-size:1.2rem; font-weight:700; color:#F8FAFC !important; }
.stat-lbl  { font-size:0.6rem; color:#64748B !important; text-transform:uppercase; letter-spacing:0.5px; margin-top:2px; }
.page-hdr  { background:#ffffff; border:1px solid #E2E8F0; border-radius:12px; padding:18px 24px; margin-bottom:18px; }
.page-hdr-title { font-size:1.25rem; font-weight:700; color:#1E293B; letter-spacing:-0.3px; }
.page-hdr-sub   { font-size:0.82rem; color:#64748B; margin-top:3px; }
.dot { width:9px; height:9px; background:#22C55E; border-radius:50%; display:inline-block; margin-right:8px; box-shadow:0 0 0 3px rgba(34,197,94,0.15); }
.panel { background:#ffffff; border:1px solid #E2E8F0; border-radius:12px; padding:20px; }
.panel-title {
    font-size:0.7rem; font-weight:600; color:#94A3B8;
    text-transform:uppercase; letter-spacing:0.9px;
    padding-bottom:12px; border-bottom:1px solid #F1F5F9; margin-bottom:16px;
}
.user-wrap { display:flex; flex-direction:column; align-items:flex-end;   margin:8px 0; }
.bot-wrap  { display:flex; flex-direction:column; align-items:flex-start; margin:8px 0; }
.user-bubble {
    background:#2563EB; color:#ffffff;
    padding:10px 16px; border-radius:14px 14px 4px 14px;
    max-width:72%; font-size:0.88rem; line-height:1.55; word-wrap:break-word;
}
.bot-bubble {
    background:#F8FAFC; color:#1E293B;
    padding:10px 16px; border-radius:14px 14px 14px 4px;
    max-width:72%; font-size:0.88rem; line-height:1.55;
    border:1px solid #E2E8F0; word-wrap:break-word;
}
.ts { font-size:0.67rem; color:#CBD5E1; margin-top:4px; }
.empty-state { text-align:center; padding:48px 20px; color:#94A3B8; font-size:0.85rem; }
.stTextInput > div > div > input {
    background:#ffffff !important; border:1px solid #CBD5E1 !important;
    border-radius:8px !important; color:#1E293B !important; font-size:0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color:#2563EB !important; box-shadow:0 0 0 3px rgba(37,99,235,0.1) !important;
}
.stButton > button {
    background:#2563EB !important; color:#ffffff !important;
    border:none !important; border-radius:8px !important; font-weight:600 !important;
}
.stButton > button:hover { background:#1D4ED8 !important; }
.acard { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:14px; margin-bottom:10px; }
.acard-label { font-size:0.67rem; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:0.9px; margin-bottom:10px; }
.badge { display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-pos { background:#DCFCE7; color:#166534; }
.badge-neg { background:#FEE2E2; color:#991B1B; }
.badge-neu { background:#FEF9C3; color:#854D0E; }
.prog-bg   { background:#E2E8F0; border-radius:4px; height:4px; margin-top:5px; overflow:hidden; }
.prog-fill { height:100%; border-radius:4px; }
.mrow {
    display:flex; justify-content:space-between; align-items:center;
    font-size:0.81rem; color:#334155; padding:5px 0; border-bottom:1px solid #F1F5F9;
}
.mrow:last-child { border-bottom:none; }
.mval { font-weight:600; color:#1E293B; }
.ner-chip { display:inline-block; padding:3px 9px; border-radius:5px; font-size:0.72rem; font-weight:500; margin:2px; }
.ner-PER  { background:#EDE9FE; color:#6D28D9; }
.ner-ORG  { background:#DCFCE7; color:#166534; }
.ner-LOC  { background:#DBEAFE; color:#1D4ED8; }
.ner-MISC { background:#FEE2E2; color:#991B1B; }
.summary-box {
    background:#EFF6FF; border-left:3px solid #2563EB;
    border-radius:0 6px 6px 0; padding:10px 14px;
    font-size:0.82rem; color:#1E40AF; line-height:1.6; font-style:italic;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_pipelines():
    return {
        "sentiment": pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        ),
        "emotion": pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            top_k=None
        ),
        "ner": pipeline(
            "ner",
            model="elastic/distilbert-base-uncased-finetuned-conll03-english",
            aggregation_strategy="simple"
        ),
        "summarizer": pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            max_length=100,
            min_length=20,
        ),
        "zero_shot": pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli"
        ),
    }


def get_sentiment(pipe, text):
    try:
        r = pipe(text[:512])[0]
        return r["label"], round(r["score"], 4)
    except:
        return "NEUTRAL", 0.5

def get_emotions(pipe, text):
    try:
        return sorted(pipe(text[:512])[0], key=lambda x: x["score"], reverse=True)[:4]
    except:
        return []

def get_entities(pipe, text):
    try:
        return pipe(text[:512])
    except:
        return []

def get_summary(pipe, text):
    try:
        if len(text.split()) < 30:
            return None
        r = pipe(text[:1024], truncation=True)
        return r[0]["summary_text"]
    except:
        return None

def get_intent(pipe, text):
    try:
        labels = ["question", "complaint", "request", "opinion", "greeting", "technical"]
        r = pipe(text[:512], candidate_labels=labels)
        return list(zip(r["labels"][:3], r["scores"][:3]))
    except:
        return []


def get_reply(history, user_msg, sentiment_label):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        tone = {
            "POSITIVE": "The user is in a positive mood. Match their energy.",
            "NEGATIVE": "The user seems frustrated. Be calm, empathetic, and supportive.",
            "NEUTRAL":  "The user is neutral. Be clear and helpful.",
        }.get(sentiment_label, "")
        messages = [{
            "role": "system",
            "content": f"You are a professional and helpful AI assistant. {tone} Keep responses concise and clear."
        }]
        for m in history[-8:]:
            messages.append({
                "role": "user" if m["role"] == "user" else "assistant",
                "content": m["content"]
            })
        messages.append({"role": "user", "content": user_msg})
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


defaults = {"messages": [], "analyses": [], "pos": 0, "neg": 0, "neu": 0, "count": 0}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.spinner("Loading transformer models... please wait."):
    pipes = load_pipelines()


with st.sidebar:
    st.markdown('<div class="sb-title">Sentiment-Aware Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Groq API &nbsp;·&nbsp; HuggingFace Transformers</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Transformer Models</div>', unsafe_allow_html=True)
    for name, tag in [
        ("Sentiment Analysis",       "distilbert-sst2"),
        ("Emotion Detection",        "distilbert-emotion"),
        ("Named Entity Recognition", "distilbert-conll03"),
        ("Summarization",            "bart-large-cnn"),
        ("Intent Classification",    "distilbert-mnli"),
    ]:
        st.markdown(
            f'<div class="sb-model-row"><span>{name}</span>'
            f'<span class="sb-mono">{tag}</span></div>',
            unsafe_allow_html=True
        )
    st.markdown('<div class="sb-label">Session Overview</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-box"><div class="stat-num">{st.session_state.count}</div><div class="stat-lbl">Total</div></div>
        <div class="stat-box"><div class="stat-num" style="color:#4ADE80">{st.session_state.pos}</div><div class="stat-lbl">Positive</div></div>
        <div class="stat-box"><div class="stat-num" style="color:#F87171">{st.session_state.neg}</div><div class="stat-lbl">Negative</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear Conversation", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()
    st.markdown("""
    <div style="font-size:0.72rem; color:#475569; margin-top:28px; line-height:1.8;">
        HEMANTHSELVA A K<br>Data Science Intern<br>Sourcesys Technologies
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="page-hdr">
    <div class="page-hdr-title"><span class="dot"></span>Sentiment-Aware Intelligent Chatbot</div>
    <div class="page-hdr-sub">Powered by Groq (LLaMA 3) &nbsp;·&nbsp; 5 HuggingFace Transformer models running in real-time</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Conversation</div>', unsafe_allow_html=True)
if not st.session_state.messages:
    st.markdown('<div class="empty-state">No messages yet. Type a message below to start.</div>', unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        ts = msg.get("time", "")
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-wrap">
                <div class="user-bubble">{msg["content"]}</div>
                <div class="ts">{ts} &nbsp;·&nbsp; You</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-wrap">
                <div class="ts">Assistant &nbsp;·&nbsp; {ts}</div>
                <div class="bot-bubble">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    c1, c2 = st.columns([6, 1])
    with c1:
        user_input = st.text_input("msg", placeholder="Type your message here...", label_visibility="collapsed")
    with c2:
        send = st.form_submit_button("Send", use_container_width=True)

if send and user_input.strip():
    ts = time.strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": user_input, "time": ts})
    st.session_state.count += 1
    with st.spinner("Analyzing message..."):
        s_label, s_score = get_sentiment(pipes["sentiment"], user_input)
        emotions          = get_emotions(pipes["emotion"],   user_input)
        entities          = get_entities(pipes["ner"],       user_input)
        summary           = get_summary(pipes["summarizer"], user_input)
        intent            = get_intent(pipes["zero_shot"],   user_input)
    if   s_label == "POSITIVE": st.session_state.pos += 1
    elif s_label == "NEGATIVE": st.session_state.neg += 1
    else:                       st.session_state.neu += 1
    with st.spinner("Generating response..."):
        reply = get_reply(st.session_state.messages[:-1], user_input, s_label)
    st.session_state.messages.append({"role": "assistant", "content": reply, "time": time.strftime("%H:%M")})
    st.session_state.analyses.append({
        "s_label": s_label, "s_score": s_score,
        "emotions": emotions, "entities": entities,
        "summary": summary,   "intent": intent,
    })
    st.rerun()


st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Real-Time Analysis</div>', unsafe_allow_html=True)

if not st.session_state.analyses:
    st.markdown('<div class="empty-state">Analysis results will appear here after you send a message.</div>', unsafe_allow_html=True)
else:
    a = st.session_state.analyses[-1]
    col1, col2, col3 = st.columns(3)

    with col1:
        lbl, score = a["s_label"], a["s_score"]
        bcls = "badge-pos" if lbl == "POSITIVE" else "badge-neg" if lbl == "NEGATIVE" else "badge-neu"
        clr  = "#22C55E"   if lbl == "POSITIVE" else "#EF4444"   if lbl == "NEGATIVE" else "#F59E0B"
        st.markdown(f"""
        <div class="acard">
            <div class="acard-label">Sentiment Analysis &nbsp;<span class="sb-mono">distilbert-sst2</span></div>
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <span class="badge {bcls}">{lbl}</span>
                <span style="font-size:0.8rem;color:#64748B;">{score:.1%} confidence</span>
            </div>
            <div class="prog-bg"><div class="prog-fill" style="width:{score*100:.0f}%;background:{clr};"></div></div>
        </div>""", unsafe_allow_html=True)

        if a["emotions"]:
            rows = "".join([
                f'<div class="mrow"><span>{e["label"].capitalize()}</span>'
                f'<span class="mval">{e["score"]:.1%}</span></div>'
                f'<div class="prog-bg"><div class="prog-fill" style="width:{e["score"]*100:.0f}%;background:#2563EB;"></div></div>'
                for e in a["emotions"]
            ])
            st.markdown(f"""
            <div class="acard">
                <div class="acard-label">Emotion Detection &nbsp;<span class="sb-mono">distilbert-emotion</span></div>
                {rows}
            </div>""", unsafe_allow_html=True)

    with col2:
        if a["intent"]:
            rows = "".join([
                f'<div class="mrow"><span>{l.capitalize()}</span>'
                f'<span class="mval">{s:.1%}</span></div>'
                f'<div class="prog-bg"><div class="prog-fill" style="width:{s*100:.0f}%;background:#7C3AED;"></div></div>'
                for l, s in a["intent"]
            ])
            st.markdown(f"""
            <div class="acard">
                <div class="acard-label">Intent Classification &nbsp;<span class="sb-mono">distilbert-mnli</span></div>
                {rows}
            </div>""", unsafe_allow_html=True)

        chips = "".join([
            f'<span class="ner-chip ner-{ent.get("entity_group","MISC")}">'
            f'{ent.get("word","")} <small style="opacity:0.6;">{ent.get("entity_group","")}</small></span>'
            for ent in a["entities"]
        ]) if a["entities"] else '<span style="font-size:0.8rem;color:#94A3B8;">No named entities detected.</span>'
        st.markdown(f"""
        <div class="acard">
            <div class="acard-label">Named Entity Recognition &nbsp;<span class="sb-mono">distilbert-conll03</span></div>
            <div style="line-height:2.4;">{chips}</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        if a["summary"]:
            st.markdown(f"""
            <div class="acard">
                <div class="acard-label">Auto Summary &nbsp;<span class="sb-mono">bart-large-cnn</span></div>
                <div class="summary-box">{a["summary"]}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="acard">
                <div class="acard-label">Auto Summary &nbsp;<span class="sb-mono">bart-large-cnn</span></div>
                <span style="font-size:0.8rem;color:#94A3B8;">Send a longer message (30+ words) to trigger summarization.</span>
            </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)