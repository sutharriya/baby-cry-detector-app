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

# SOLUTION: Custom LSTM class that completely handles the time_major issue
class FixedLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        # Remove all problematic arguments
        kwargs.pop('time_major', None)
        kwargs.pop('unroll', None)
        kwargs.pop('implementation', None)
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        # Ensure no problematic args are saved
        config.pop('time_major', None)
        config.pop('unroll', None)
        config.pop('implementation', None)
        return config

# Custom objects for model loading
CUSTOM_OBJECTS = {
    'LSTM': FixedLSTM,
    'Bidirectional': Bidirectional
}

@st.cache_resource
def load_components():
    """Load model and encoder with comprehensive fixes"""
    try:
        # SOLUTION: Use tf.keras.models.load_model with custom_objects
        model = tf.keras.models.load_model(
            "best_model.h5",
            custom_objects=CUSTOM_OBJECTS,
            compile=False
        )
        
        # Load label encoder
        label_encoder = joblib.load("label_encoder.pkl")
        
        return model, label_encoder
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

# Load components
model, label_encoder = load_components()

def extract_features(file_path, duration=4, sr=22050, n_mels=128):
    """Audio feature extraction with error handling"""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        samples = sr * duration
        y = y[:samples] if len(y) >= samples else np.pad(y, (0, samples - len(y)))
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = mel_db / np.max(np.abs(mel_db))  # Normalize
        return mel_db.astype(np.float32)
    except Exception as e:
        st.error(f"❌ Audio processing error: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="Baby Cry Detector", page_icon="👶")
st.title("👶 Baby Cry Detector")
st.write("Upload a baby cry audio file (.wav) for analysis")

uploaded_file = st.file_uploader("Choose WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    temp_file = "temp.wav"
    
    try:
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Analyzing..."):
            features = extract_features(temp_file)
            if features is not None:
                # Prepare input tensor
                features = np.expand_dims(features[..., np.newaxis], axis=0)
                
                # Predict
                preds = model.predict(features, verbose=0)[0]
                top_3 = preds.argsort()[-3:][::-1]
                results = {
                    label_encoder.classes_[i]: f"{preds[i]*100:.1f}%"
                    for i in top_3
                }
                
                st.success("Top Predictions:")
                for label, prob in results.items():
                    st.write(f"- {label}: {prob}")
    
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

st.markdown("---")
st.caption("Note: For medical concerns, consult a pediatrician.")
