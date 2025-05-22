import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import librosa
import joblib
import tensorflow as tf
import streamlit as st

def load_model_anyway(model_path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        # Assuming your model was trained on a fixed duration, e.g., 3 seconds or 5 seconds.
        # Let's start by assuming the model expects a fixed 3-second audio input for its features.
        # This will ensure 'y' always has the same number of samples.
        fixed_duration_samples = sr * 3 # Adjust this '3' if your model expects a different fixed duration (e.g., 5 seconds)
        
        if len(y) < fixed_duration_samples:
            # Pad if audio is shorter than expected duration
            y = np.pad(y, (0, fixed_duration_samples - len(y)), 'constant')
        elif len(y) > fixed_duration_samples:
            # Truncate if audio is longer than expected duration
            y = y[:fixed_duration_samples]

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return np.expand_dims(np.expand_dims(mel_db, -1), 0)
    except Exception as e:
        st.error(f"Audio processing failed: {str(e)}")
        return None

st.set_page_config(page_title="Baby Cry Detector", page_icon="👶", layout="centered")

@st.cache_resource
def load_all():
    model = load_model_anyway("best_model.h5")
    encoder = joblib.load("label_encoder.pkl")
    if model:
        # मॉडल के अपेक्षित इनपुट शेप को लॉग करें
        st.write(f"Model loaded. Expected input shape: {model.input_shape}")
    return model, encoder

model, encoder = load_all()

st.title("👶 Baby Cry Analyzer")
st.write("Upload a WAV file (2-5 seconds) of a baby crying")

uploaded_file = st.file_uploader("Choose file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    features = extract_features("temp.wav")
    if features is not None:
        # एक्सट्रैक्टेड फीचर्स के शेप को लॉग करें
        st.write(f"Extracted features shape: {features.shape}")

        # यहाँ हम एक बेसिक शेप कम्पेटिबिलिटी चेक भी जोड़ रहे हैं
        if model.input_shape and len(model.input_shape) == len(features.shape) and \
           all(s1 == s2 or s1 is None for s1, s2 in zip(model.input_shape[1:], features.shape[1:])):
            # शेप संगत हैं, आगे बढ़ें
            pass
        else:
            # अगर शेप मैच नहीं करते तो एरर दिखाएं और रुकें
            st.error(f"Input shape mismatch! Model expects {model.input_shape} but received {features.shape}. "
                     f"Please ensure audio length (2-5 seconds recommended) and processing match model's training.")
            st.stop() # यह ऐप को आगे बढ़ने से रोकेगा अगर शेप गलत है

        pred = model.predict(features, verbose=0)[0]
        top_idx = np.argmax(pred)
        st.success(f"Most likely: {encoder.classes_[top_idx]} ({pred[top_idx]*100:.1f}%)")
    
    if os.path.exists("temp.wav"):
        os.remove("temp.wav")
