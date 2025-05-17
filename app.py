import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import os
from tensorflow.keras.layers import Bidirectional, LSTM

# Configuration to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Custom objects for model loading
CUSTOM_OBJECTS = {
    'Bidirectional': Bidirectional,
    'LSTM': LSTM
}

@st.cache_resource
def load_model_and_encoder():
    """Load and cache the model and label encoder"""
    try:
        # Load model with custom objects
        model = tf.keras.models.load_model(
            "best_model.h5",
            custom_objects=CUSTOM_OBJECTS,
            compile=True
        )
        
        # Load label encoder
        with open("label_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
            
        return model, encoder
    except Exception as e:
        st.error(f"❌ Error loading model components: {str(e)}")
        st.stop()

# Load components
model, encoder = load_model_and_encoder()

def extract_features(file_path):
    """Extract mel-spectrogram features from audio file"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)
        
        # Extract mel-spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Add channel dimension and resize
        mel_db = mel_db[..., np.newaxis]
        mel_db = tf.image.resize(mel_db, [128, 128])
        
        return np.expand_dims(mel_db, axis=0)
    except Exception as e:
        st.error(f"❌ Error processing audio file: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="Baby Cry Detector", page_icon="👶")
st.title("👶 Baby Cry Detector")
st.write("Upload a baby cry audio file (.wav) to identify the reason for crying.")

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
                # Make prediction
                prediction = model.predict(features, verbose=0)
                predicted_class = encoder.inverse_transform([np.argmax(prediction)])[0]
                
                # Display result
                st.success(f"**Predicted Cry Type:** {predicted_class}")
                
                # Optional: Show confidence scores
                with st.expander("See detailed probabilities"):
                    st.write("Class probabilities:")
                    for i, prob in enumerate(prediction[0]):
                        st.write(f"{encoder.classes_[i]}: {prob:.2%}")
    
    except Exception as e:
        st.error(f"❌ An error occurred during processing: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

# Add footer
st.markdown("---")
st.caption("Note: This is a demo application for baby cry classification. For medical concerns, please consult a pediatrician.")
