import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import librosa
import joblib
import tensorflow as tf
import streamlit as st
import time # Import time module for sleep

# --- Constants ---
MAX_FILE_SIZE_MB = 10 # Set a maximum file size for upload to prevent OOM
TARGET_SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
TARGET_FRAMES = 173
TARGET_SAMPLES = (TARGET_FRAMES - 1) * HOP_LENGTH + N_FFT

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
        st.write(f"DEBUG: [EXTRACT_FEATURES] 1. Starting feature extraction for file: {file_path}") # More specific debug
        
        y, sr = librosa.load(file_path, sr=TARGET_SR)
        st.write(f"DEBUG: [EXTRACT_FEATURES] 2. Audio loaded. Shape: {y.shape}, Sample Rate: {sr}")

        if len(y) < TARGET_SAMPLES:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)), 'constant')
            st.write("DEBUG: [EXTRACT_FEATURES] 3. Audio padded.")
        elif len(y) > TARGET_SAMPLES:
            y = y[:TARGET_SAMPLES]
            st.write("DEBUG: [EXTRACT_FEATURES] 4. Audio truncated.")
        
        st.write(f"DEBUG: [EXTRACT_FEATURES] 5. Audio length after adjustment: {len(y)} samples.")

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=N_FFT, hop_length=HOP_LENGTH)
        st.write(f"DEBUG: [EXTRACT_FEATURES] 6. Mel spectrogram computed. Shape: {mel.shape}")
        mel_db = librosa.power_to_db(mel, ref=np.max)
        st.write(f"DEBUG: [EXTRACT_FEATURES] 7. Mel_db converted to dB. Shape: {mel_db.shape}")

        current_frames = mel_db.shape[1]
        if current_frames < TARGET_FRAMES:
            padding_needed = TARGET_FRAMES - current_frames
            mel_db = np.pad(mel_db, ((0, 0), (0, padding_needed)), 'constant')
            st.write("DEBUG: [EXTRACT_FEATURES] 8. Mel_db padded to target frames.")
        elif current_frames > TARGET_FRAMES:
            mel_db = mel_db[:, :TARGET_FRAMES]
            st.write("DEBUG: [EXTRACT_FEATURES] 9. Mel_db truncated to target frames.")

        st.write(f"DEBUG: [EXTRACT_FEATURES] 10. Final mel_db shape before expanding dims: {mel_db.shape}")
        return np.expand_dims(mel_db, 0) 

    except Exception as e:
        st.error(f"Audio processing failed: {str(e)}")
        st.exception(e) # This will print the full traceback to the UI/logs
        st.write(f"DEBUG: [EXTRACT_FEATURES] ERROR caught during feature extraction: {str(e)}")
        return None

# Streamlit page configuration
st.set_page_config(page_title="Baby Cry Detector", page_icon="👶", layout="centered")

# Cache the model and encoder to prevent reloading on every app rerun
@st.cache_resource
def load_all():
    st.write("DEBUG: [LOAD_ALL] 0. Loading model and encoder...")
    model = load_model_anyway("best_model.h5")
    encoder = joblib.load("label_encoder.pkl")
    if model:
        st.write(f"DEBUG: [LOAD_ALL] Model loaded. Expected input shape: {model.input_shape}")
    st.write(f"DEBUG: [LOAD_ALL] Encoder loaded. Class names: {encoder.classes_}")
    return model, encoder

# Load model and encoder
model, encoder = load_all()

st.title("👶 Baby Cry Analyzer")
st.write(f"Upload a WAV file (2-5 seconds recommended, max {MAX_FILE_SIZE_MB}MB) of a baby crying.")

# File uploader widget
uploaded_file = st.file_uploader("Choose file", type=["wav"])

# If a file is uploaded
if uploaded_file:
    st.write("DEBUG: [MAIN] File uploaded. Attempting to process.")
    
    # --- File Size Check ---
    file_size_bytes = uploaded_file.getbuffer().nbytes
    file_size_mb = file_size_bytes / (1024 * 1024)
    st.write(f"DEBUG: [MAIN] Uploaded file size: {file_size_mb:.2f} MB")

    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large! Please upload a WAV file smaller than {MAX_FILE_SIZE_MB}MB.")
        st.write("DEBUG: [MAIN] File size exceeded limit. Stopping processing.")
        st.stop() # Stop processing if file is too big
    # --- End File Size Check ---

    st.audio(uploaded_file)
    
    # Temporarily save the file so librosa can read it
    temp_file_path = "temp.wav"
    try:
        st.write("DEBUG: [MAIN] Attempting to save temporary file.")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"DEBUG: [MAIN] Temporary file saved at {temp_file_path}")
    except Exception as e:
        st.error(f"Error saving temporary file: {str(e)}")
        st.exception(e) # Print traceback
        st.write(f"DEBUG: [MAIN] ERROR saving temp file: {str(e)}")
        temp_file_path = None

    if temp_file_path:
        # Add a short delay to allow UI to update and logs to flush
        st.write("DEBUG: [MAIN] Adding a short delay before feature extraction...")
        time.sleep(0.5) # Wait for 0.5 seconds
        
        features = extract_features(temp_file_path)
        
        if features is not None:
            st.write(f"DEBUG: [MAIN] Extracted features shape: {features.shape}")

            if model.input_shape and \
               model.input_shape[1] == features.shape[1] and \
               model.input_shape[2] == features.shape[2]:
                
                st.write("DEBUG: [MAIN] Making prediction...")
                pred = model.predict(features, verbose=0)[0]
                st.write(f"DEBUG: [MAIN] Prediction made. Raw prediction: {pred}")
                
                top_n_indices = np.argsort(pred)[::-1][:3]
                st.write(f"DEBUG: [MAIN] Top N indices: {top_n_indices}")

                st.subheader("Prediction Results:")
                for i in top_n_indices:
                    label = encoder.classes_[i]
                    probability = pred[i] * 100
                    st.write(f"- **{label.replace('_', ' ').title()}**: {probability:.1f}%")
                st.write("DEBUG: [MAIN] Prediction complete.")
                
            else:
                st.error(f"Input shape mismatch! Model expects {model.input_shape} but received {features.shape}. "
                         f"Please ensure audio length (2-5 seconds recommended) and processing match model's training.")
                st.exception(Exception("Shape mismatch error")) # Print traceback for this custom error
                st.write(f"DEBUG: [MAIN] ERROR - Input shape mismatch. Expected {model.input_shape}, got {features.shape}")
                st.stop()
        else:
            st.write("DEBUG: [MAIN] Feature extraction skipped due to previous error.")
    
    if os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
            st.write("DEBUG: [MAIN] Temporary file removed.")
        except Exception as e:
            st.warning(f"Could not remove temporary file: {str(e)}")
            st.write(f"DEBUG: [MAIN] ERROR removing temp file: {str(e)}")
