import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import gdown
import os

# Google Drive file IDs
MODEL_FILE_ID = '1b0ny-ePb8v4kuSLgIbRmWiyhovRSihWr'
ENCODER_FILE_ID = '10FmmSOvJZsFsyMK0zhUh3y2TKEdMpEuA'

# Local filenames
MODEL_PATH = 'model.h5'
ENCODER_PATH = 'label_encoder.pkl'

# Download model if not exists
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(f'https://drive.google.com/uc?id={MODEL_FILE_ID}', MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Download label encoder if not exists
@st.cache_resource
def load_encoder():
    if not os.path.exists(ENCODER_PATH):
        with st.spinner("Downloading label encoder..."):
            gdown.download(f'https://drive.google.com/uc?id={ENCODER_FILE_ID}', ENCODER_PATH, quiet=False)
    with open(ENCODER_PATH, 'rb') as f:
        return pickle.load(f)

# Feature extraction
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.resize(mel_db, (128, 128))
    return mel_db

# Streamlit UI
st.set_page_config(page_title="Baby Cry Detector", layout="centered")
st.title("👶 Baby Cry Sound Detector")

uploaded_file = st.file_uploader("Upload a baby cry sound (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    try:
        model = load_model()
        encoder = load_encoder()

        with st.spinner("Analyzing cry..."):
            features = extract_features(uploaded_file)
            features = np.expand_dims(features, axis=(0, -1))  # Shape: (1, 128, 128, 1)

            prediction = model.predict(features)
            predicted_index = np.argmax(prediction)
            predicted_label = encoder.inverse_transform([predicted_index])[0]

        st.success(f"🍼 Predicted Cry Type: **{predicted_label}**")
    except Exception as e:
        st.error(f"Error: {e}")
