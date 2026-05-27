import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import base64
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
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

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Hide default Streamlit header */
header[data-testid="stHeader"] { background: transparent; }

/* Main container */
.main-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 24px;
    padding: 2.5rem;
    margin: 1rem 0;
    box-shadow: 0 25px 50px rgba(0,0,0,0.4);
}

/* Title */
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

/* Result badge */
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
    transition: all 0.4s ease;
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

/* Probability bars */
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
    transition: width 0.6s ease;
}

/* Buttons */
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

/* Section labels */
.section-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Info box */
.info-box {
    background: rgba(52, 211, 153, 0.08);
    border-left: 3px solid #34d399;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: rgba(255,255,255,0.65);
    font-size: 0.9rem;
    margin: 1rem 0;
}

/* Audio player styling */
audio {
    width: 100%;
    border-radius: 12px;
    margin-top: 0.5rem;
}

/* Canvas container */
.canvas-container {
    border-radius: 16px;
    overflow: hidden;
    border: 2px solid rgba(255,255,255,0.15);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

/* Training spinner */
.stSpinner > div { border-top-color: #a78bfa !important; }

/* Streamlit element colors */
.stProgress > div > div > div { background: linear-gradient(90deg, #a78bfa, #60a5fa); }

/* Toast / alert */
[data-testid="stNotification"] { background: rgba(167,139,250,0.15) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Model Training / Loading
# ─────────────────────────────────────────────────────────
MODEL_PATH = "bestmodel.keras"

LABELS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}


def is_valid_model(path: str) -> bool:
    """Check if a Keras model file is valid (not a stub)."""
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    return size > 10_000  # A real model is at least ~10 KB


def build_and_train_model() -> keras.Model:
    """Build CNN and train it on MNIST, save to MODEL_PATH."""
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Normalise & reshape
    X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    X_test  = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    Y_train = to_categorical(Y_train, 10)
    Y_test  = to_categorical(Y_test, 10)

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPool2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),
        Flatten(),
        Dropout(0.25),
        Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss=keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )

    es = EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=4, verbose=0)
    mc = ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", verbose=0, save_best_only=True)

    model.fit(
        X_train, Y_train,
        epochs=50,
        validation_split=0.2,
        batch_size=128,
        callbacks=[es, mc],
        verbose=0,
    )
    return keras.models.load_model(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_model_cached():
    if is_valid_model(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH)
    return None  # Will trigger training below


# ─────────────────────────────────────────────────────────
# Audio generation via gTTS
# ─────────────────────────────────────────────────────────
def text_to_audio_b64(text: str) -> str:
    """Generate speech with gTTS and return base64-encoded MP3."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return b64
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────
# Digit prediction helper
# ─────────────────────────────────────────────────────────
def predict_digit(model, canvas_data: np.ndarray):
    """
    canvas_data: RGBA uint8 array from st_canvas (H x W x 4)
    Returns (digit_int, label_str, probabilities_array)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(canvas_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

    # Threshold & invert (black background, white digit → MNIST style)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Find bounding box of drawn content
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None, None, None

    x, y, w, h = cv2.boundingRect(coords)
    pad = 20
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, thresh.shape[1])
    y2 = min(y + h + pad, thresh.shape[0])
    digit_crop = thresh[y1:y2, x1:x2]

    # Resize to 28×28, normalise
    digit_resized = cv2.resize(digit_crop, (28, 28), interpolation=cv2.INTER_AREA)
    digit_norm = digit_resized.astype("float32") / 255.0
    digit_input = digit_norm.reshape(1, 28, 28, 1)

    preds = model.predict(digit_input, verbose=0)[0]
    digit = int(np.argmax(preds))
    return digit, LABELS[digit], preds


# ─────────────────────────────────────────────────────────
# UI — Header
# ─────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🔢 Digit Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Draw a digit · AI recognises it · Hear it spoken aloud</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Model Loading / Training
# ─────────────────────────────────────────────────────────
model = load_model_cached()

if model is None:
    st.markdown("""
    <div class="info-box">
    ⚙️ <strong>First Launch:</strong> Training the CNN model on MNIST dataset. This takes ~1–2 minutes and only happens once.
    </div>
    """, unsafe_allow_html=True)
    with st.spinner("Training model on MNIST (CNN · Adam · ~99% accuracy)..."):
        try:
            model = build_and_train_model()
            st.cache_resource.clear()
            st.success("✅ Model trained and saved! Refreshing...")
            st.rerun()
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

# ─────────────────────────────────────────────────────────
# Main layout — two columns
# ─────────────────────────────────────────────────────────
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
        digit, label, probs = predict_digit(model, canvas_result.image_data)

        if digit is None:
            st.warning("Canvas is empty — please draw a digit first!")
        else:
            confidence = float(probs[digit]) * 100

            # Result badge
            st.markdown(f"""
            <div class="result-badge">
                <div class="result-number">{digit}</div>
                <div class="result-label">{label}</div>
                <div class="result-conf">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars for all digits
            st.markdown('<div class="section-label" style="margin-top:1rem">📊 All Probabilities</div>', unsafe_allow_html=True)
            for i, p in enumerate(probs):
                pct = float(p) * 100
                color = "#a78bfa" if i == digit else "rgba(255,255,255,0.3)"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin:3px 0;">
                    <span style="color:{color};font-weight:{'700' if i==digit else '400'};
                          font-size:0.85rem;width:52px;">{i} · {LABELS[i][:3]}</span>
                    <div class="prob-bar-bg" style="flex:1;">
                        <div class="prob-bar-fill" style="width:{pct:.1f}%;
                             background:{'linear-gradient(90deg,#a78bfa,#60a5fa)' if i==digit else 'rgba(255,255,255,0.2)'};">
                        </div>
                    </div>
                    <span style="color:rgba(255,255,255,0.45);font-size:0.75rem;width:40px;">{pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

            # 🔊 Audio vocalization
            st.markdown('<div class="section-label" style="margin-top:1rem">🔊 Vocalization</div>', unsafe_allow_html=True)
            audio_b64 = text_to_audio_b64(label)
            if audio_b64:
                audio_html = f"""
                <audio autoplay controls>
                    <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
            else:
                st.info("Audio unavailable (network required for gTTS)")
    else:
        st.markdown("""
        <div class="result-badge" style="min-height:180px;opacity:0.5;">
            <div class="result-number" style="font-size:3rem;">?</div>
            <div class="result-label">Awaiting Input</div>
            <div class="result-conf">Draw a digit and click Predict</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# How it works section
# ─────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div style="text-align:center;padding:1rem;">
        <div style="font-size:2rem;">✏️</div>
        <div style="color:rgba(255,255,255,0.85);font-weight:600;margin:0.5rem 0;">Draw</div>
        <div style="color:rgba(255,255,255,0.4);font-size:0.85rem;">Use your mouse or touchscreen to draw any digit (0–9) on the canvas</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div style="text-align:center;padding:1rem;">
        <div style="font-size:2rem;">🧠</div>
        <div style="color:rgba(255,255,255,0.85);font-weight:600;margin:0.5rem 0;">Recognise</div>
        <div style="color:rgba(255,255,255,0.4);font-size:0.85rem;">CNN trained on 60,000 MNIST samples classifies your digit with ~99% accuracy</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div style="text-align:center;padding:1rem;">
        <div style="font-size:2rem;">🔊</div>
        <div style="color:rgba(255,255,255,0.85);font-weight:600;margin:0.5rem 0;">Vocalize</div>
        <div style="color:rgba(255,255,255,0.4);font-size:0.85rem;">The recognized digit is spoken aloud using Google Text-to-Speech</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:rgba(255,255,255,0.2);font-size:0.75rem;margin-top:2rem;">
    Digit Recognition & Vocalization · CNN · MNIST · Streamlit
</div>
""", unsafe_allow_html=True)
