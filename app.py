import os
import streamlit as st
import numpy as np
import cv2
import io
import base64
from streamlit_drawable_canvas import st_canvas

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
audio { width: 100%; border-radius: 12px; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Pure-numpy MLP inference (no sklearn/tensorflow needed)
# ─────────────────────────────────────────────────────────
LABELS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}

@st.cache_resource(show_spinner=False)
def load_weights():
    weights_path = "digit_weights.npy"
    if not os.path.exists(weights_path):
        st.error("digit_weights.npy not found in repository.")
        st.stop()
    return np.load(weights_path, allow_pickle=True).item()

def mlp_predict(X: np.ndarray, weights: dict) -> np.ndarray:
    """
    Pure numpy forward pass through the MLP.
    Activation: ReLU on hidden layers, Softmax on output.
    """
    coefs      = weights["coefs"]
    intercepts = weights["intercepts"]

    a = X
    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        z = a @ W + b
        if i < len(coefs) - 1:          # hidden layers — ReLU
            a = np.maximum(0, z)
        else:                            # output layer — Softmax
            z -= np.max(z, axis=1, keepdims=True)  # numerical stability
            exp_z = np.exp(z)
            a = exp_z / exp_z.sum(axis=1, keepdims=True)
    return a   # shape (1, 10) — class probabilities

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
# Prediction helper — MNIST-matching preprocessing
# ─────────────────────────────────────────────────────────
def _mnist_preprocess(thresh: np.ndarray) -> np.ndarray:
    """
    Converts a thresholded canvas crop into a 28×28 float array
    that matches the MNIST format as closely as possible:
      1. Square crop (preserve aspect ratio)
      2. Resize digit to 20×20
      3. Embed in 28×28 with 4-px border
      4. Re-centre by centre-of-mass (same trick MNIST uses)
      5. Normalise to [0, 1]
    """
    h, w = thresh.shape
    # 1. Pad to square
    size   = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    y_off  = (size - h) // 2
    x_off  = (size - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = thresh

    # 2. Resize digit to 20×20
    digit20 = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)

    # 3. Embed in 28×28 (4-px border on each side)
    img28 = np.zeros((28, 28), dtype=np.uint8)
    img28[4:24, 4:24] = digit20

    # 4. Re-centre by centre-of-mass
    M = cv2.moments(img28)
    if M["m00"] != 0:
        cx     = int(M["m10"] / M["m00"])
        cy     = int(M["m01"] / M["m00"])
        sh_x   = 14 - cx
        sh_y   = 14 - cy
        mat    = np.float32([[1, 0, sh_x], [0, 1, sh_y]])
        img28  = cv2.warpAffine(img28, mat, (28, 28),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 5. Normalise
    return img28.astype("float64") / 255.0


def predict_digit(canvas_data: np.ndarray, weights: dict):
    # Slight blur first to smooth canvas anti-aliasing
    gray  = cv2.cvtColor(canvas_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    gray  = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None, None, None

    x, y, w, h = cv2.boundingRect(coords)
    pad = 15
    x1, y1 = max(x - pad, 0), max(y - pad, 0)
    x2, y2 = min(x + w + pad, thresh.shape[1]), min(y + h + pad, thresh.shape[0])
    crop   = thresh[y1:y2, x1:x2]

    flat   = _mnist_preprocess(crop).reshape(1, -1)

    probs  = mlp_predict(flat, weights)[0]
    digit  = int(np.argmax(probs))
    return digit, LABELS[digit], probs

# ─────────────────────────────────────────────────────────
# Load model weights
# ─────────────────────────────────────────────────────────
weights = load_weights()

# ─────────────────────────────────────────────────────────
# Session state — persist prediction across reruns
# ─────────────────────────────────────────────────────────
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0          # increment to reset canvas
if "prediction" not in st.session_state:
    st.session_state.prediction = None       # stores last result dict

# ─────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🔢 Digit Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Draw a digit &middot; AI recognises it &middot; Hear it spoken aloud</p>',
            unsafe_allow_html=True)

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
        key=f"canvas_{st.session_state.canvas_key}",  # new key = fresh canvas
    )
    b1, b2 = st.columns(2)
    with b1:
        predict_btn = st.button("🔍 Predict", use_container_width=True)
    with b2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    if clear_btn:
        st.session_state.canvas_key += 1     # reset canvas
        st.session_state.prediction = None   # reset prediction output too
        st.rerun()

with col_result:
    st.markdown('<div class="section-label">🎯 Prediction Result</div>', unsafe_allow_html=True)

    # Run prediction and save to session_state
    if predict_btn and canvas_result.image_data is not None:
        digit, label, probs = predict_digit(canvas_result.image_data, weights)
        if digit is None:
            st.warning("Canvas is empty — please draw a digit first!")
        else:
            st.session_state.prediction = {
                "digit": digit, "label": label, "probs": probs
            }

    # Always display from session_state (persists after Clear)
    pred = st.session_state.prediction
    if pred:
        digit      = pred["digit"]
        label      = pred["label"]
        probs      = pred["probs"]
        confidence = float(probs[digit]) * 100

        st.markdown(f"""
        <div class="result-badge">
            <div class="result-number">{digit}</div>
            <div class="result-label">{label}</div>
            <div class="result-conf">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1rem">📊 All Probabilities</div>',
                    unsafe_allow_html=True)
        for i, p in enumerate(probs):
            pct   = float(p) * 100
            top   = (i == digit)
            color = "#a78bfa" if top else "rgba(255,255,255,0.3)"
            bar   = "linear-gradient(90deg,#a78bfa,#60a5fa)" if top else "rgba(255,255,255,0.18)"
            fw    = "700" if top else "400"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin:3px 0;">
              <span style="color:{color};font-weight:{fw};font-size:.85rem;width:52px;">
                {i} · {LABELS[i][:3]}</span>
              <div class="prob-bar-bg" style="flex:1;">
                <div style="height:100%;border-radius:8px;width:{pct:.1f}%;background:{bar};"></div>
              </div>
              <span style="color:rgba(255,255,255,0.45);font-size:.75rem;width:40px;">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1rem">🔊 Vocalization</div>',
                    unsafe_allow_html=True)
        # Only play audio when Predict was just clicked
        if predict_btn:
            audio_b64 = text_to_audio_b64(label)
            if audio_b64:
                st.markdown(f"""
                <audio autoplay controls>
                  <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                </audio>""", unsafe_allow_html=True)
            else:
                st.info("Audio unavailable (needs internet for gTTS)")
        else:
            st.markdown(f"<p style='color:rgba(255,255,255,0.4);font-size:.9rem;'>Last spoken: <b style='color:#a78bfa'>{label}</b></p>",
                        unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-badge" style="min-height:180px;opacity:0.5;">
          <div class="result-number" style="font-size:3rem;">?</div>
          <div class="result-label">Awaiting Input</div>
          <div class="result-conf">Draw a digit and click Predict</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3 = st.columns(3)
for col, icon, title, desc in [
    (c1, "✏️", "Draw",      "Use mouse or touch to draw any digit (0–9) on the canvas"),
    (c2, "🧠", "Recognise", "MLP trained on 60,000 MNIST samples — 98% accuracy"),
    (c3, "🔊", "Vocalize",  "Digit is spoken aloud via Google Text-to-Speech"),
]:
    with col:
        st.markdown(f"""
        <div style="text-align:center;padding:1rem;">
          <div style="font-size:2rem;">{icon}</div>
          <div style="color:rgba(255,255,255,0.85);font-weight:600;margin:.5rem 0;">{title}</div>
          <div style="color:rgba(255,255,255,0.4);font-size:.85rem;">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:rgba(255,255,255,0.2);font-size:.75rem;margin-top:2rem;">
  Digit Recognition &amp; Vocalization &middot; Pure NumPy MLP &middot; MNIST &middot; Streamlit
</div>""", unsafe_allow_html=True)
