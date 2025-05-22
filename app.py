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
        # Load WAV file at 22050 Hz sample rate
        y, sr = librosa.load(file_path, sr=22050)

        # Model expects input shape of (None, 128, 173).
        # Based on librosa's default n_fft (2048) and hop_length (512),
        # calculate the required number of samples for 173 frames.
        n_fft = 2048
        hop_length = 512
        target_frames = 173
        
        # Calculate the required number of samples
        target_samples = (target_frames - 1) * hop_length + n_fft
        
        # Pad or truncate the audio to the required number of samples
        if len(y) < target_samples:
            # Pad if audio is shorter than expected duration
            y = np.pad(y, (0, target_samples - len(y)), 'constant')
        elif len(y) > target_samples:
            # Truncate if audio is longer than expected duration
            y = y[:target_samples]

        # Extract Mel spectrogram
        # n_mels=128 matches the model's 128 Mel bands
        # Explicitly specify n_fft and hop_length for consistency
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=n_fft, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Ensure the number of frames (second dimension of mel_db) is exactly target_frames (173).
        current_frames = mel_db.shape[1]

        if current_frames < target_frames:
            # Pad if there are fewer frames
            padding_needed = target_frames - current_frames
            mel_db = np.pad(mel_db, ((0, 0), (0, padding_needed)), 'constant')
        elif current_frames > target_frames:
            # Truncate if there are more frames
            mel_db = mel_db[:, :target_frames]

        # Model expects a 3D input of shape (1, 128, 173).
        # np.expand_dims(mel_db, 0) adds the batch dimension (at axis=0).
        return np.expand_dims(mel_db, 0) 

    except Exception as e:
        st.error(f"Audio processing failed: {str(e)}")
        return None

# Streamlit page configuration
st.set_page_config(page_title="Baby Cry Detector", page_icon="👶", layout="centered")

# Cache the model and encoder to prevent reloading on every app rerun
@st.cache_resource
def load_all():
    model = load_model_anyway("best_model.h5") # Ensure this path is correct
    encoder = joblib.load("label_encoder.pkl") # Ensure this path is correct
    if model:
        # Log expected input shape for debugging
        st.write(f"Model loaded. Expected input shape: {model.input_shape}")
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
    st.audio(uploaded_file) # Play the uploaded audio
    
    # Temporarily save the file so librosa can read it
    temp_file_path = "temp.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract features
    features = extract_features(temp_file_path)
    
    if features is not None:
        # Log extracted features shape for debugging
        st.write(f"Extracted features shape: {features.shape}")

        # Check input shape compatibility
        # The first element of model.input_shape (None) is batch size,
        # so we don't compare it directly, but compare the other dimensions.
        # features.shape will have 1 at batch dimension because of np.expand_dims(mel_db, 0).
        # So, we only compare the last two dimensions (128 and 173).
        if model.input_shape and \
           model.input_shape[1] == features.shape[1] and \
           model.input_shape[2] == features.shape[2]:
            
            # If shapes are compatible, make prediction
            pred = model.predict(features, verbose=0)[0]
            
            # Get indices of top predictions (e.g., top 3)
            # np.argsort returns indices that would sort an array.
            # [::-1] reverses it to get descending order (highest probabilities first).
            top_n_indices = np.argsort(pred)[::-1][:3] # Get top 3 indices

            st.subheader("Prediction Results:")
            for i in top_n_indices:
                label = encoder.classes_[i]
                probability = pred[i] * 100
                st.write(f"- **{label.replace('_', ' ').title()}**: {probability:.1f}%") # Format output like "Sleepy: 45.0%"
            
        else:
            # If shapes do not match, show error and stop
            st.error(f"Input shape mismatch! Model expects {model.input_shape} but received {features.shape}. "
                     f"Please ensure audio length (2-5 seconds recommended) and processing match model's training.")
            st.stop() # Stop the app if shape is incorrect
    
    # Remove the temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

