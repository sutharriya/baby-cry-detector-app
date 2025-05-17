import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model("best_model.h5")

# Load encoder
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db[..., np.newaxis]
    mel_db = tf.image.resize(mel_db, [128, 128])
    return np.expand_dims(mel_db, axis=0)

st.title("👶 Baby Cry Detector")
st.write("Upload a baby cry audio file (.wav) to detect the reason.")

file = st.file_uploader("Upload WAV file", type=["wav"])
if file:
    st.audio(file, format="audio/wav")
    with open("temp.wav", "wb") as f:
        f.write(file.read())

    features = extract_features("temp.wav")
    pred = model.predict(features)
    label = encoder.inverse_transform([np.argmax(pred)])[0]

    st.success(f"Predicted Cry Type: **{label}**")
