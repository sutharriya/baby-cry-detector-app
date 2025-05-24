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
# compile=False is used to prevent the model from attempting to recompile,
# which can sometimes cause issues due to version mismatches.
def load_model_anyway(model_path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop() # Stop the app if model fails to load

# Function to extract features from audio
def extract_features(file_path):
    try:
        st.write(f"DEBUG: 1. Starting feature extraction for file: {file_path}") # DEBUG POINT 1
        
        # Load WAV file at 22050 Hz sample rate
        y, sr = librosa.load(file_path, sr=22050)
        st.write(f"DEBUG: 2. Audio loaded. Shape: {y.shape}, Sample Rate: {sr}") # DEBUG POINT 2

        n_fft = 2048
        hop_length = 512
        target_frames = 173
        
        # Calculate the required number of samples
        target_samples = (target_frames - 1) * hop_length + n_fft
        
        # Pad or truncate the audio to the required number of samples
        if len(y) < target_samples:
            # Pad if audio is shorter than expected duration
            y = np.pad(y, (0, target_samples - len(y)), 'constant')
            st.write("DEBUG: 3. Audio padded.") # DEBUG POINT 3
        elif len(y) > target_samples:
            # Truncate if audio is longer than expected duration
            y = y[:target_samples]
            st.write("DEBUG: 4. Audio truncated.") # DEBUG POINT 4
        
        st.write(f"DEBUG: 5. Audio length after adjustment: {len(y)} samples.") # DEBUG POINT 5

        # Extract Mel spectrogram
        # n_mels=128 matches the model's 128 Mel bands
        # Explicitly specify n_fft and hop_length for consistency
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=n_fft, hop_length=hop_length)
        st.write(f"DEBUG: 6. Mel spectrogram computed. Shape: {mel.shape}") # DEBUG POINT 6
        mel_db = librosa.power_to_db(mel, ref=np.max)
        st.write(f"DEBUG: 7. Mel_db converted to dB. Shape: {mel_db.shape}") # DEBUG POINT 7

        current_frames = mel_db.shape[1]

        if current_frames < target_frames:
            # Pad if there are fewer frames
            padding_needed = target_frames - current_frames
            mel_db = np.pad(mel_db, ((0, 0), (0, padding_needed)), 'constant')
            st.write("DEBUG: 8. Mel_db padded to target frames.") # DEBUG POINT 8
        elif current_frames > target_frames:
            # Truncate if there are more frames
            mel_db = mel_db[:, :target_frames]
            st.write("DEBUG: 9. Mel_db truncated to target frames.") # DEBUG POINT 9

        st.write(f"DEBUG: 10. Final mel_db shape before expanding dims: {mel_db.shape}") # DEBUG POINT 10
        return np.expand_dims(mel_db, 0) 

    except Exception as e:
        st.error(f"ऑडियो प्रोसेसिंग में विफल: {str(e)}")
        st.write(f"DEBUG: ERROR caught during feature extraction: {str(e)}") # DEBUG POINT 11 - This will also go to logs
        return None

# Streamlit page configuration
st.set_page_config(page_title="Baby Cry Detector", page_icon="👶", layout="centered")

# Cache the model and encoder to prevent reloading on every app rerun
@st.cache_resource
def load_all():
    st.write("DEBUG: 0. Loading model and encoder...") # DEBUG POINT 0 - This appears when the app starts
    model = load_model_anyway("best_model.h5") # Ensure this path is correct
    encoder = joblib.load("label_encoder.pkl") # Ensure this path is correct
    if model:
        # Log expected input shape for debugging
        st.write(f"DEBUG: Model loaded. Expected input shape: {model.input_shape}")
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
    st.write("DEBUG: File uploaded. Attempting to process.") # DEBUG POINT File Uploaded
    st.audio(uploaded_file) # Play the uploaded audio
    
    # Temporarily save the file so librosa can read it
    temp_file_path = "temp.wav"
    try: # Added try-except around file writing for robustness
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"DEBUG: Temporary file saved at {temp_file_path}") # DEBUG POINT Temp file saved
    except Exception as e:
        st.error(f"Error saving temporary file: {str(e)}")
        st.write(f"DEBUG: ERROR saving temp file: {str(e)}") # DEBUG POINT Temp file error
        temp_file_path = None # Set to None to prevent further processing if save fails

    if temp_file_path: # Only proceed if temp file was saved successfully
        # Extract features
        features = extract_features(temp_file_path)
        
        if features is not None:
            # Log extracted features shape for debugging
            st.write(f"DEBUG: Extracted features shape: {features.shape}") # DEBUG POINT Features Extracted

            # Check input shape compatibility
            if model.input_shape and \
               model.input_shape[1] == features.shape[1] and \
               model.input_shape[2] == features.shape[2]:
                
                st.write("DEBUG: Making prediction...") # DEBUG POINT Prediction Start
                pred = model.predict(features, verbose=0)[0]
                
                # Get indices of top predictions (e.g., top 3)
                top_n_indices = np.argsort(pred)[::-1][:3] # Get top 3 indices

                st.subheader("Prediction Results:")
                for i in top_n_indices:
                    label = encoder.classes_[i]
                    probability = pred[i] * 100
                    st.write(f"- **{label.replace('_', ' ').title()}**: {probability:.1f}%") # Format output
                st.write("DEBUG: Prediction complete.") # DEBUG POINT Prediction End
                
            else:
                # If shapes do not match, show error and stop
                st.error(f"Input shape mismatch! Model expects {model.input_shape} but received {features.shape}. "
                         f"Please ensure audio length (2-5 seconds recommended) and processing match model's training.")
                st.write(f"DEBUG: ERROR - Input shape mismatch. Expected {model.input_shape}, got {features.shape}") # DEBUG POINT Shape Mismatch
                st.stop() # Stop the app if shape is incorrect
        else:
            st.write("DEBUG: Feature extraction skipped due to temp file error.") # DEBUG POINT No Features
    
    # Remove the temporary file
    if os.path.exists(temp_file_path):
        try: # Added try-except around file removal for robustness
            os.remove(temp_file_path)
            st.write("DEBUG: Temporary file removed.") # DEBUG POINT Temp file removed
        except Exception as e:
            st.write(f"DEBUG: ERROR removing temp file: {str(e)}") # DEBUG POINT Temp file removal error
