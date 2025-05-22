import os
# TensorFlow के लॉग्स को कम करने के लिए (कम verbose आउटपुट)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
# चेतावनियों को अनदेखा करने के लिए
warnings.filterwarnings('ignore')

import numpy as np
import librosa
import joblib
import tensorflow as tf
import streamlit as st

# मॉडल को लोड करने का फंक्शन
# compile=False का उपयोग किया जाता है ताकि मॉडल को फिर से कंपाइल करने की कोशिश न हो,
# जो कभी-कभी वर्जन मिसमैच के कारण समस्याएँ पैदा कर सकता है।
def load_model_anyway(model_path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"मॉडल लोड करने में विफल: {str(e)}")
        st.stop() # ऐप को रोकें यदि मॉडल लोड नहीं हो पाता

# ऑडियो से फीचर्स निकालने का फंक्शन
def extract_features(file_path):
    try:
        # WAV फ़ाइल को 22050 Hz सैंपल रेट पर लोड करें
        y, sr = librosa.load(file_path, sr=22050)

        # मॉडल को (None, 128, 173) आकार का इनपुट चाहिए।
        # librosa के डिफ़ॉल्ट n_fft (2048) और hop_length (512) के आधार पर,
        # 173 फ्रेम्स के लिए आवश्यक सैंपल की संख्या की गणना करें।
        n_fft = 2048
        hop_length = 512
        target_frames = 173
        
        # आवश्यक सैंपल की संख्या की गणना करें
        # (फ्रेम्स की संख्या - 1) * हॉप_लेंथ + n_fft
        target_samples = (target_frames - 1) * hop_length + n_fft
        
        # ऑडियो को आवश्यक सैंपल की संख्या तक पैड (pad) या ट्रंकेट (truncate) करें
        if len(y) < target_samples:
            # यदि ऑडियो अपेक्षित अवधि से छोटा है, तो पैड करें
            y = np.pad(y, (0, target_samples - len(y)), 'constant')
        elif len(y) > target_samples:
            # यदि ऑडियो अपेक्षित अवधि से लंबा है, तो ट्रंकेट करें
            y = y[:target_samples]

        # मेल स्पेक्ट्रोग्राम निकालें
        # n_mels=128 मॉडल के 128 मेल बैंड्स से मेल खाता है
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=n_fft, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # मॉडल को (1, 128, 173) आकार का 3D इनपुट चाहिए।
        # np.expand_dims(mel_db, 0) बैच डायमेंशन (axis=0 पर) जोड़ता है।
        return np.expand_dims(mel_db, 0) 

    except Exception as e:
        st.error(f"ऑडियो प्रोसेसिंग में विफल: {str(e)}")
        return None

# Streamlit पेज कॉन्फ़िगरेशन
st.set_page_config(page_title="बेबी क्राई डिटेक्टर", page_icon="👶", layout="centered")

# मॉडल और एनकोडर को कैश करें ताकि हर बार ऐप लोड होने पर उन्हें फिर से लोड न करना पड़े
@st.cache_resource
def load_all():
    model = load_model_anyway("best_model.h5") # सुनिश्चित करें कि यह पथ सही है
    encoder = joblib.load("label_encoder.pkl") # सुनिश्चित करें कि यह पथ सही है
    if model:
        # मॉडल के अपेक्षित इनपुट शेप को लॉग करें (डीबगिंग के लिए)
        st.write(f"मॉडल लोड हो गया है। अपेक्षित इनपुट शेप: {model.input_shape}")
    return model, encoder

# मॉडल और एनकोडर लोड करें
model, encoder = load_all()

# ऐप का UI
st.title("👶 बेबी क्राई एनालाइज़र")
st.write("एक बच्चे के रोने की WAV फ़ाइल (2-5 सेकंड) अपलोड करें")

# फ़ाइल अपलोडर विजेट
uploaded_file = st.file_uploader("फ़ाइल चुनें", type=["wav"])

# यदि कोई फ़ाइल अपलोड की गई है
if uploaded_file:
    st.audio(uploaded_file) # अपलोड की गई ऑडियो चलाएं
    
    # अस्थायी रूप से फ़ाइल को सेव करें ताकि librosa उसे पढ़ सके
    temp_file_path = "temp.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # फीचर्स निकालें
    features = extract_features(temp_file_path)
    
    if features is not None:
        # निकाले गए फीचर्स के शेप को लॉग करें (डीबगिंग के लिए)
        st.write(f"निकाले गए फीचर्स का शेप: {features.shape}")

        # इनपुट शेप की संगतता जांचें
        # मॉडल के अपेक्षित इनपुट शेप का पहला तत्व (None) बैच साइज़ है,
        # इसलिए हम इसकी तुलना नहीं करते, बल्कि बाकी आयामों की तुलना करते हैं।
        if model.input_shape and len(model.input_shape) == len(features.shape) and \
           all(s1 == s2 or s1 is None for s1, s2 in zip(model.input_shape[1:], features.shape[1:])):
            # यदि शेप संगत हैं, तो प्रेडिक्शन करें
            pred = model.predict(features, verbose=0)[0]
            top_idx = np.argmax(pred)
            st.success(f"सबसे संभावित: {encoder.classes_[top_idx]} ({pred[top_idx]*100:.1f}%)")
        else:
            # यदि शेप मैच नहीं करते तो एरर दिखाएं और रुकें
            st.error(f"इनपुट शेप मिसमैच! मॉडल को {model.input_shape} चाहिए लेकिन {features.shape} मिला। "
                     f"कृपया सुनिश्चित करें कि ऑडियो की लंबाई (2-5 सेकंड अनुशंसित) और प्रोसेसिंग मॉडल की ट्रेनिंग से मेल खाती है।")
            st.stop() # ऐप को आगे बढ़ने से रोकेगा यदि शेप गलत है
    
    # अस्थायी फ़ाइल को हटा दें
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

