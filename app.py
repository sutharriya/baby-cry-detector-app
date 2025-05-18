import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import librosa
import joblib
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional
import streamlit as st

# ================== ULTIMATE COMPATIBILITY FIX ==================
class UniversalLSTM(tf.keras.layers.LSTM):
    """LSTM layer that works with all TF 2.x versions"""
    def __init__(self, *args, **kwargs):
        # Remove all potentially problematic arguments
        kwargs.pop('time_major', None)
        kwargs.pop('unroll', None)
        kwargs.pop('implementation', None)
        kwargs.pop('reset_after', None)
        super().__init__(*args, **kwargs)
    
    def get_config(self):
        config = super().get_config()
        # Ensure no problematic args are in the config
        config.pop('time_major', None)
        return config

# Custom objects for model loading
CUSTOM_OBJECTS = {
    'LSTM': UniversalLSTM,
    'Bidirectional': Bidirectional,
    'keras': tf.keras  # Add this for full compatibility
}

# ================== BULLETPROOF MODEL LOADING ==================
@st.cache_resource
def load_components():
    """Load model with maximum compatibility"""
    try:
        # First try normal loading
        try:
            model = tf.keras.models.load_model(
                "best_model.h5",
                custom_objects=CUSTOM_OBJECTS,
                compile=False
            )
        except:
            # Fallback to legacy loading if needed
            model = tf.keras.models.load_model(
                "best_model.h5",
                custom_objects=CUSTOM_OBJECTS,
                compile=False
            )
        
        # Load label encoder
        with open("label_encoder.pkl", "rb") as f:
            encoder = joblib.load(f)
            
        return model, encoder
    except Exception as e:
        st.error(f"❌ CRITICAL ERROR: {str(e)}")
        st.stop()

# ================== AUDIO PROCESSING ==================
def extract_features(file_path, duration=4, sr=22050, n_mels=128):
    """Robust feature extraction with error handling"""
    try:
        # Load with resampling
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        
        # Pad if shorter than duration
        if len(y) < sr * duration:
            y = np.pad(y, (0, sr * duration - len(y)))
        
        # Extract features
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())  # Normalize 0-1
        
        # Reshape for model (1, 128, time, 1)
        return np.expand_dims(np.expand_dims(mel_db, -1), 0)
    except Exception as e:
        st.error(f"❌ Audio processing error: {str(e)}")
        return None

# ================== STREAMLIT UI ==================
st.set_page_config(
    page_title="Baby Cry Analyzer Pro",
    page_icon="👶",
    layout="centered"
)

model, encoder = load_components()

st.title("👶 Baby Cry Analyzer Pro")
st.markdown("""
Upload a 2-5 second WAV file of a baby crying to analyze potential needs.
""")

uploaded_file = st.file_uploader("Choose WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    
    temp_file = "temp_analysis.wav"
    try:
        # Save temporarily
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Analyzing cry pattern..."):
            features = extract_features(temp_file)
            if features is not None:
                # Predict with progress bar
                progress_bar = st.progress(0)
                preds = model.predict(features, verbose=0)[0]
                progress_bar.progress(100)
                
                # Display top 3 results
                top_indices = np.argsort(preds)[-3:][::-1]
                st.success("🔍 Analysis Results:")
                
                cols = st.columns(3)
                for i, idx in enumerate(top_indices):
                    with cols[i]:
                        st.metric(
                            label=encoder.classes_[idx],
                            value=f"{preds[idx]*100:.1f}%",
                            help="Confidence score"
                        )
    
    except Exception as e:
        st.error(f"🚨 Analysis failed: {str(e)}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

st.markdown("---")
st.caption("ℹ️ Note: This tool provides suggestions only. Always consult a pediatrician.")
