import os
import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model
model = tf.keras.models.load_model("best_model.h5")

# Load the label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder: LabelEncoder = pickle.load(f)

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db[..., np.newaxis]
    mel_db = tf.image.resize(mel_db, [128, 128])
    return np.expand_dims(mel_db, axis=0)

# Streamlit UI
st.title("👶 Baby Cry Detector")
st.write("Upload a baby cry audio file (.wav) to detect the reason.")

uploaded_file = st.file_uploader("Upload WAV File", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    features = extract_features("temp.wav")
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    st.success(f"Predicted Cry Type: **{predicted_label}**")
