import streamlit as st
import numpy as np
import librosa
import joblib
import gdown
import tensorflow as tf
import os

# Google Drive File Links
MODEL_URL = "https://drive.google.com/uc?id=16mHv7vT3FFh7J-BBM0Dn_L30w22UBq79"
ENCODER_URL = "https://drive.google.com/uc?id=1Fu-HKsq6_6gnMPgUa9cCV-sebmRxoGCE"

# Download files if not present
def download_files():
    if not os.path.exists("model.h5"):
        gdown.download(MODEL_URL, "model.h5", quiet=False)
    if not os.path.exists("label_encoder.pkl"):
        gdown.download(ENCODER_URL, "label_encoder.pkl", quiet=False)

# Load model and encoder
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("model.h5")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

# Feature extraction
def extract_features(file_path, max_len=216):
    y, sr = librosa.load(file_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    if log_mel.shape[1] < max_len:
        pad_width = max_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :max_len]
    return log_mel

# Prediction function
def predict(file):
    with open("temp.wav", "wb") as f:
        f.write(file.read())
    features = extract_features("temp.wav")
    features = np.expand_dims(features, axis=-1)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    predicted_label = encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# UI
st.title("👶 Baby Cry Detector")
st.markdown("Upload a baby's cry audio (.wav) to detect the reason.")

# Download required files
download_files()

# Load model and encoder
model, encoder = load_model_and_encoder()

uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    if st.button("Predict"):
        with st.spinner("Analyzing cry..."):
            result = predict(uploaded_file)
        st.success(f"Prediction: **{result}**")
