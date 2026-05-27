import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import numpy as np
import cv2
import io
import base64
from streamlit_drawable_canvas import st_canvas
import joblib
import requests

# ─────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recognition & Vocalization",
    page_icon="🔢",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────
# Custom CSS — dark premium UI
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}
header[data-testid="stHeader"] { background: transparent; }

.hero-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    letter-spacing: -1px;
}
.hero-sub {
    text-align: center;
    color: rgba(255,255,255,0.5);
    font-size: 1.05rem;
    margin-bottom: 2rem;
    font-weight: 300;
}
.result-badge {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, rgba(167,139,250,0.25), rgba(96,165,250,0.15));
    border: 2px solid rgba(167,139,250,0.4);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 0 40px rgba(167,139,250,0.15);
}
.result-number {
    font-size: 5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.result-label {
    font-size: 1.5rem;
    font-weight: 600;
    color: rgba(255,255,255,0.85);
    margin-top: 0.4rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.result-conf {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.45);
    margin-top: 0.3rem;
}
.prob-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    height: 8px;
    width: 100%;
    margin: 4px 0;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
}
.stButton > button {
    background: linear-gradient(135deg, #a78bfa, #60a5fa) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 25px rgba(167,139,250,0.3) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 35px rgba(167,139,250,0.5) !important;
}
.section-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.info-box {
    background: rgba(52, 211, 153, 0.08);
    border-left: 3px solid #34d399;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: rgba(255,255,255,0.65);
    font-size: 0.9rem;
    margin: 1rem 0;
}
audio { width: 100%; border-radius: 12px; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
MODEL_PATH = "digit_model.pkl"
LABELS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}

# ─────────────────────────────────────────────────────────
# Model Loading (no training on cloud — model is in repo)
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "❌ `digit_model.pkl` not found. "
            "Please run `python train_local.py` locally and commit the file."
        )
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ─────────────────────────────────────────────────────────
# Audio (gTTS)
# ─────────────────────────────────────────────────────────
def text_to_audio_b64(text: str) -> str:
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return ""

# ─────────────────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────────────────
def predict_digit(canvas_data: np.ndarray):
    """
    canvas_data: RGBA uint8 array (H x W x 4) from st_canvas
    Returns (digit_int, label_str, probabilities_array)
    """
    gray = cv2.cvtColor(canvas_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None, None, None

    x, y, w, h = cv2.boundingRect(coords)
    pad = 20
    x1 = max(x - pad, 0);  y1 = max(y - pad, 0)
    x2 = min(x + w + pad, thresh.shape[1]);  y2 = min(y + h + pad, thresh.shape[0])
    crop = thresh[y1:y2, x1:x2]

    resized = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
    flat = resized.astype("float32").reshape(1, -1) / 255.0

    probs = model.predict_proba(flat)[0]
    digit = int(np.argmax(probs))
    return digit, LABELS[digit], probs

# ─────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🔢 Digit Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Draw a digit · AI recognises it · Hear it spoken aloud</p>', unsafe_allow_html=True)

col_canvas, col_result = st.columns([1.1, 0.9], gap="large")

with col_canvas:
    st.markdown('<div class="section-label">✏️ Draw a Digit (0–9)</div>', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#1a1a2e",
        height=320,
        width=320,
        drawing_mode="freedraw",
        key="canvas",
    )
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        predict_btn = st.button("🔍 Predict", use_container_width=True)
    with btn_col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    if clear_btn:
        st.rerun()

with col_result:
    st.markdown('<div class="section-label">🎯 Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn and canvas_result.image_data is not None:
        digit, label, probs = predict_digit(canvas_result.image_data)

        if digit is None:
            st.warning("Canvas is empty — please draw a digit first!")
        else:
            confidence = float(probs[digit]) * 100

            st.markdown(f"""
            <div class="result-badge">
                <div class="result-number">{digit}</div>
                <div class="result-label">{label}</div>
                <div class="result-conf">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label" style="margin-top:1rem">📊 All Probabilities</div>', unsafe_allow_html=True)
            for i, p in enumerate(probs):
                pct = float(p) * 100
                is_top = (i == digit)
                color = "#a78bfa" if is_top else "rgba(255,255,255,0.3)"
                bar_bg = "linear-gradient(90deg,#a78bfa,#60a5fa)" if is_top else "rgba(255,255,255,0.2)"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin:3px 0;">
                    <span style="color:{color};font-weight:{'700' if is_top else '400'};
                          font-size:0.85rem;width:52px;">{i} · {LABELS[i][:3]}</span>
                    <div class="prob-bar-bg" style="flex:1;">
                        <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{bar_bg};"></div>
                    </div>
                    <span style="color:rgba(255,255,255,0.45);font-size:0.75rem;width:40px;">{pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-label" style="margin-top:1rem">🔊 Vocalization</div>', unsafe_allow_html=True)
            audio_b64 = text_to_audio_b64(label)
            if audio_b64:
                st.markdown(f"""
                <audio autoplay controls>
                    <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                </audio>
                """, unsafe_allow_html=True)
            else:
                st.info("🔇 Audio unavailable (network required for gTTS)")
    else:
        st.markdown("""
        <div class="result-badge" style="min-height:180px;opacity:0.5;">
            <div class="result-number" style="font-size:3rem;">?</div>
            <div class="result-label">Awaiting Input</div>
            <div class="result-conf">Draw a digit and click Predict</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div style="text-align:center;padding:1rem;">
        <div style="font-size:2rem;">✏️</div>
        <div style="color:rgba(255,255,255,0.85);font-weight:600;margin:0.5rem 0;">Draw</div>
        <div style="color:rgba(255,255,255,0.4);font-size:0.85rem;">Use mouse or touch to draw any digit (0–9)</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div style="text-align:center;padding:1rem;">
        <div style="font-size:2rem;">🧠</div>
        <div style="color:rgba(255,255,255,0.85);font-weight:600;margin:0.5rem 0;">Recognise</div>
        <div style="color:rgba(255,255,255,0.4);font-size:0.85rem;">MLP trained on 60,000 MNIST samples (~97% accuracy)</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div style="text-align:center;padding:1rem;">
        <div style="font-size:2rem;">🔊</div>
        <div style="color:rgba(255,255,255,0.85);font-weight:600;margin:0.5rem 0;">Vocalize</div>
        <div style="color:rgba(255,255,255,0.4);font-size:0.85rem;">Digit is spoken aloud via Google Text-to-Speech</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:rgba(255,255,255,0.2);font-size:0.75rem;margin-top:2rem;">
    Digit Recognition & Vocalization · sklearn MLP · MNIST · Streamlit
</div>
""", unsafe_allow_html=True)
