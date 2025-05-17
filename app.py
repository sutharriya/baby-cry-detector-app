import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import os
from tensorflow.keras.layers import Bidirectional, LSTM

# Set TensorFlow logging to avoid unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Custom object dictionary for model loading
CUSTOM_OBJECTS = {
    'Bidirectional': Bidirectional,
    'LSTM': LSTM
}

@st.cache_resource
def load_components():
    """Load model and encoder with caching"""
    try:
        # Load model with custom objects
        model = tf.keras.models.load_model(
            "best_model.h5",
            custom_objects=CUSTOM_OBJECTS,
            compile=False
        )
        
        # Load encoder
        with open("label_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
            
        return model, encoder
    except Exception as e:
        st.error(f"Error loading model/encoder: {str(e)}")
        st.stop()

# Load components
model, encoder = load_components()

def extract_features(file_path):
    """Extract audio features with error handling"""
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = mel_db[..., np.newaxis]
        mel_db = tf.image.resize(mel_db, [128, 128])
        return np.expand_dims(mel_db, axis=0)
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

# UI Components
st.title("👶 Baby Cry Detector")
st.write("Upload a baby cry audio file (.wav) to detect the reason.")

file = st.file_uploader("Upload WAV file", type=["wav"])
if file:
    st.audio(file, format="audio/wav")
    
    try:
        # Save temp file
        with open("temp.wav", "wb") as f:
            f.write(file.read())
        
        # Extract features
        features = extract_features("temp.wav")
        if features is not None:
            # Make prediction
            pred = model.predict(features)
            label = encoder.inverse_transform([np.argmax(pred)])[0]
            
            st.success(f"Predicted Cry Type: **{label}**")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")
