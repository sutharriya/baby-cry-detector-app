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

# =============================================
# COMPATIBILITY FIXES
# =============================================
class FixedLSTM(tf.keras.layers.LSTM):
    """LSTM layer with removed problematic arguments for TF 2.16+ compatibility"""
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        kwargs.pop('unroll', None)
        kwargs.pop('implementation', None)
        super().__init__(*args, **kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.pop('time_major', None)
        return config

CUSTOM_OBJECTS = {
    'LSTM': FixedLSTM,
    'Bidirectional': Bidirectional
}

# =============================================
# MODEL LOADING WITH CACHING
# =============================================
@st.cache_resource
def load_model_and_encoder():
    """Load and cache model components with error handling"""
    try:
        model = tf.keras.models.load_model(
            "best_model.h5",
            custom_objects=CUSTOM_OBJECTS,
            compile=False
        )
        with open("label_encoder.pkl", "rb") as f:
            encoder = joblib.load(f)
        return model, encoder
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()

# =============================================
# AUDIO PROCESSING
# =============================================
def process_audio(file_path, duration=4, sr=22050, n_mels=128):
    """Extract normalized mel-spectrogram features"""
    try:
        # Load and pad/trim audio
        y, sr = librosa.load(file_path, sr=sr)
        samples = sr * duration
        y = y[:samples] if len(y) >= samples else np.pad(y, (0, samples - len(y)))
        
        # Extract features
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = mel_db / np.max(np.abs(mel_db))  # Normalize
        
        # Reshape for model (1, 128, time, 1)
        return np.expand_dims(mel_db[..., np.newaxis], axis=0).astype(np.float32)
    except Exception as e:
        st.error(f"❌ Audio processing failed: {str(e)}")
        return None

# =============================================
# STREAMLIT UI
# =============================================
# Configure page
st.set_page_config(
    page_title="Baby Cry Analyzer",
    page_icon="👶",
    layout="centered"
)

# Load model (cached)
model, encoder = load_model_and_encoder()

# App header
st.title("👶 Baby Cry Analyzer")
st.markdown("""
Upload a baby cry recording (.wav) to identify potential needs.
Audio should be 2-5 seconds long for best results.
""")

# File upload
uploaded_file = st.file_uploader(
    "Choose WAV file", 
    type=["wav"],
    accept_multiple_files=False
)

if uploaded_file:
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')
    
    # Temporary file handling
    temp_path = "temp_audio.wav"
    try:
        # Save uploaded file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process with loading indicator
        with st.spinner("Analyzing cry pattern..."):
            # Extract features
            features = process_audio(temp_path)
            
            if features is not None:
                # Make prediction
                predictions = model.predict(features, verbose=0)[0]
                
                # Get top 3 results
                top_indices = predictions.argsort()[-3:][::-1]
                results = {
                    encoder.classes_[i]: f"{predictions[i]*100:.1f}%"
                    for i in top_indices
                }
                
                # Display results
                st.success("🔊 Prediction Results")
                for label, confidence in results.items():
                    st.progress(float(confidence[:-1])/100)
                    st.write(f"**{label}** ({confidence} confidence)")
    
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
    
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Footer
st.markdown("---")
st.caption("""
Note: This tool provides suggestions only. 
Always consult a pediatrician for medical concerns.
""")
