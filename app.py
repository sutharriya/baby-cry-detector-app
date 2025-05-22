import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import librosa
import joblib
import tensorflow as tf
import streamlit as st

def load_model_anyway(model_path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return np.expand_dims(np.expand_dims(mel_db, -1), 0)
    except Exception as e:
        st.error(f"Audio processing failed: {str(e)}")
        return None

st.set_page_config(page_title="Baby Cry Detector", page_icon="👶", layout="centered")

@st.cache_resource
def load_all():
    model = load_model_anyway("best_model.h5")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_all()

st.title("👶 Baby Cry Analyzer")
st.write("Upload a WAV file (2-5 seconds) of a baby crying")

uploaded_file = st.file_uploader("Choose file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    features = extract_features("temp.wav")
    if features is not None:
        pred = model.predict(features, verbose=0)[0]
        top_idx = np.argmax(pred)
        st.success(f"Most likely: {encoder.classes_[top_idx]} ({pred[top_idx]*100:.1f}%)")
    
    if os.path.exists("temp.wav"):
        os.remove("temp.wav")
