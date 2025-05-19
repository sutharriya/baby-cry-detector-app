import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress ALL TensorFlow logs
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import librosa
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st

# ================== NUCLEAR-OPTION COMPATIBILITY ==================
def load_model_anyway(model_path):
    """Try every possible way to load the model"""
    try:
        # Method 1: Normal load
        try:
            return load_model(model_path, compile=False)
        except:
            # Method 2: With custom objects
            try:
                return load_model(
                    model_path,
                    custom_objects={
                        'LSTM': tf.keras.layers.LSTM,
                        'Bidirectional': tf.keras.layers.Bidirectional
                    },
                    compile=False
                )
            except:
                # Method 3: Legacy format
                return tf.keras.models.load_model(
                    model_path,
                    custom_objects={
                        'LSTM': tf.compat.v1.keras.layers.LSTM,
                        'Bidirectional': tf.compat.v1.keras.layers.Bidirectional
                    },
                    compile=False
                )
    except Exception as e:
        st.error(f"FATAL ERROR: Could not load model. Technical details: {str(e)}")
        st.stop()

# ================== SIMPLIFIED AUDIO PROCESSING ==================
def extract_features_simple(file_path):
    """More reliable feature extraction"""
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=4)
        if len(y) < 22050 * 4:
            y = np.pad(y, (0, 22050 * 4 - len(y)))
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel)
        return np.expand_dims(np.expand_dims(mel_db, -1), 0)
    except Exception as e:
        st.error(f"Audio processing failed: {str(e)}")
        return None

# ================== STREAMLIT APP ==================
st.set_page_config(
    page_title="Baby Cry Detector",
    page_icon="👶",
    layout="centered"
)

@st.cache_resource
def load_all():
    model = load_model_anyway("best_model.h5")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_all()

st.title("👶 Baby Cry Analyzer")
st.write("Upload a short WAV file (2-5 seconds) of a baby crying")

uploaded_file = st.file_uploader("Choose file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)
    
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        features = extract_features_simple("temp.wav")
        if features is not None:
            pred = model.predict(features, verbose=0)[0]
            top_idx = np.argmax(pred)
            st.success(f"Most likely: {encoder.classes_[top_idx]} ({pred[top_idx]*100:.1f}%)")
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
    finally:
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")
