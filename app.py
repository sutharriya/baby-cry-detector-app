import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
from flask import Flask, request, render_template_string

# ✅ Google Drive se model & label encoder download
import gdown

MODEL_ID = "16mHv7vT3FFh7J-BBM0Dn_L30w22UBq79"
ENCODER_ID = "1Fu-HKsq6_6gnMPgUa9cCV-sebmRxoGCE"
MODEL_PATH = "best_model.h5"
ENCODER_PATH = "label_encoder.pkl"

if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)

if not os.path.exists(ENCODER_PATH):
    print("🔽 Downloading label encoder...")
    gdown.download(f"https://drive.google.com/uc?id={ENCODER_ID}", ENCODER_PATH, quiet=False)

# ✅ Load Model & Label Encoder
model = tf.keras.models.load_model("best_model.h5")
label_encoder = joblib.load("label_encoder.pkl")

# ✅ Flask App Setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Feature Extraction Function
def extract_features(file_path, duration=4, sr=22050, n_mels=128):
    samples = sr * duration
    y, sr = librosa.load(file_path, sr=sr)
    if len(y) < samples:
        y = np.pad(y, (0, samples - len(y)))
    else:
        y = y[:samples]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db / np.max(np.abs(mel_db))  # Normalize
    return mel_db.astype(np.float32)

# ✅ Updated HTML Template (Top 3 Labels ke liye)
HTML_TEMPLATE = '''
<!doctype html>
<title>Baby Cry Predictor</title>
<h2>Upload a Baby Cry .wav file</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file accept=".wav">
  <input type=submit value=Predict>
</form>

{% if predictions %}
  <h3>🔊 Top 3 Predictions:</h3>
  <ul>
    {% for label, prob in predictions.items() %}
      <li><strong>{{ label }}</strong>: {{ prob }}%</li>
    {% endfor %}
  </ul>
{% endif %}
'''

# ✅ Routes
@app.route('/', methods=['GET', 'POST'])
def predict():
    predictions = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.wav'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            features = extract_features(filepath)
            features = np.expand_dims(features, axis=-1)  # (128, time, 1)
            features = np.expand_dims(features, axis=0)   # (1, 128, time, 1)

            preds = model.predict(features)[0]  # Get first element

            # Top 3 predictions nikalna
            top_indices = preds.argsort()[-3:][::-1]
            labels = label_encoder.inverse_transform(top_indices)
            probs = preds[top_indices] * 100  # % me convert

            # Label aur % ek dictionary me store
            predictions = {label: round(prob, 2) for label, prob in zip(labels, probs)}

    return render_template_string(HTML_TEMPLATE, predictions=predictions)

# ✅ Run App
if __name__ == '__main__':
    app.run(debug=True)
