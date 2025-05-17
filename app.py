import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import librosa
import joblib
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional
import streamlit as st

# Custom LSTM layer to handle version incompatibility
class CompatibleLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # Remove problematic argument
        super().__init__(*args, **kwargs)

# Define custom objects for model loading
CUSTOM_OBJECTS = {
    'LSTM': CompatibleLSTM,
    'Bidirectional': Bidirectional
}

@st.cache_resource
def load_components():
    """Load model and encoder with compatibility fixes"""
    try:
        # Load model with custom objects
        model = tf.keras.models.load_model(
            "best_model.h5",
            custom_objects=CUSTOM_OBJECTS,
            compile=False
        )
        
        # Load label encoder
        label_encoder = joblib.load("label_encoder.pkl")
        
        return model, label_encoder
    except Exception as e:
        st.error(f"❌ Error loading model components: {str(e)}")
        st.stop()

# Feature extraction function
def extract_features(file_path, duration=4, sr=22050, n_mels=128):
    """Extract mel-spectrogram features from audio file"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=sr)
        
        # Ensure consistent length
        samples = sr * duration
        if len(y) < samples:
            y = np.pad(y, (0, samples - len(y)))
        else:
            y = y[:samples]
        
        # Extract mel-spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize and add channel dimension
        mel_db = mel_db / np.max(np.abs(mel_db))
        mel_db = mel_db[..., np.newaxis]
        
        return mel_db.astype(np.float32)
    except Exception as e:
        st.error(f"❌ Error processing audio file: {str(e)}")
        return None

# Streamlit UI setup
st.set_page_config(
    page_title="Baby Cry Detector",
    page_icon="👶",
    layout="centered"
)

# App title and description
st.title("👶 Baby Cry Detector")
st.markdown("""
Upload a baby cry audio file (.wav) to predict the reason for crying.
The model will show the top 3 most likely reasons.
""")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a WAV file",
    type=["wav"],
    accept_multiple_files=False
)

if uploaded_file:
    # Display audio player
    st.audio(uploaded_file, format="audio/wav")
    
    # Temporary file handling
    temp_file = "temp_audio.wav"
    try:
        # Save uploaded file temporarily
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Show loading spinner while processing
        with st.spinner("Analyzing the cry..."):
            # Extract features
            features = extract_features(temp_file)
            
            if features is not None:
                # Prepare input shape (1, 128, time, 1)
                features = np.expand_dims(features, axis=-1)
                features = np.expand_dims(features, axis=0)
                
                # Make prediction
                preds = model.predict(features, verbose=0)[0]
                
                # Get top 3 predictions
                top_indices = preds.argsort()[-3:][::-1]
                labels = label_encoder.inverse_transform(top_indices)
                probs = preds[top_indices] * 100  # Convert to percentage
                
                # Display results
                st.subheader("🔊 Prediction Results")
                
                # Show predictions as progress bars
                for label, prob in zip(labels, probs):
                    st.write(f"**{label}**")
                    st.progress(int(prob))
                    st.write(f"{prob:.2f}% confidence")
                    st.write("---")
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

# Add footer
st.markdown("---")
st.caption("""
Note: This is a demo application for baby cry classification. 
For medical concerns, please consult a pediatrician.
""")
