import os
# Reduce TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
# Ignore warnings
warnings.filterwarnings('ignore')

import numpy as np
import librosa
import joblib
import tensorflow as tf
import streamlit as st

# Function to load the model
def load_model_anyway(model_path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

# Function to extract features from audio
def extract_features(file_path):
    try:
        # Load WAV file at 22050 Hz sample rate
        y, sr = librosa.load(file_path, sr=22050)

        n_fft = 2048
        hop_length = 512
        target_frames = 173
        target_samples = (target_frames - 1) * hop_length + n_fft
        
        # Pad or truncate the audio to the required number of samples
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)), 'constant')
        elif len(y) > target_samples:
            y = y[:target_samples]

        # Extract Mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=n_fft, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        current_frames = mel_db.shape[1]
        if current_frames < target_frames:
            padding_needed = target_frames - current_frames
            mel_db = np.pad(mel_db, ((0, 0), (0, padding_needed)), 'constant')
        elif current_frames > target_frames:
            mel_db = mel_db[:, :target_frames]

        return np.expand_dims(mel_db, 0) 

    except Exception as e:
        st.error(f"Audio processing failed: {str(e)}")
        return None

# Streamlit page configuration
st.set_page_config(page_title="Baby Cry Detector", page_icon="👶", layout="centered")

# Cache the model and encoder to prevent reloading on every app rerun
@st.cache_resource
def load_all():
    model = load_model_anyway("best_model.h5")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

# Load model and encoder
model, encoder = load_all()

# App UI
st.title("👶 Baby Cry Analyzer")
st.write("Upload a WAV file (2-5 seconds) of a baby crying")

# File uploader widget
uploaded_file = st.file_uploader("Choose file", type=["wav"])

# If a file is uploaded
if uploaded_file:
    st.audio(uploaded_file)
    
    # Temporarily save the file so librosa can read it
    temp_file_path = "temp.wav"
    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        st.error(f"Error saving temporary file: {str(e)}")
        temp_file_path = None

    if temp_file_path:
        features = extract_features(temp_file_path)
        
        if features is not None:
            # Check input shape compatibility
            if model.input_shape and \
               model.input_shape[1] == features.shape[1] and \
               model.input_shape[2] == features.shape[2]:
                
                # Make prediction
                pred = model.predict(features, verbose=0)[0]
                
                # Get indices of top predictions (e.g., top 3)
                top_n_indices = np.argsort(pred)[::-1][:3]

                st.subheader("Prediction Results:")
                for i in top_n_indices:
                    label = encoder.classes_[i]
                    probability = pred[i] * 100
                    st.write(f"- **{label.replace('_', ' ').title()}**: {probability:.1f}%")
                
            else:
                st.error(f"Input shape mismatch! Model expects {model.input_shape} but received {features.shape}. "
                         f"Please ensure audio length (2-5 seconds recommended) and processing match model's training.")
        
    # Remove the temporary file
    if os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
        except Exception as e:
            st.warning(f"Could not remove temporary file: {str(e)}") # Changed to warning as it's not critical
