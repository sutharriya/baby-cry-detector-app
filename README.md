# Baby Cry Detector App

## Description
A web application that uses a Convolutional Neural Network (CNN) to detect and analyze baby cries from audio files. It aims to identify the reason behind a baby's cry, such as hunger, discomfort, or tiredness.

## Features
- Analyzes WAV audio files (recommended 2-5 seconds long).
- Predicts the type of baby cry based on the trained model (e.g., hungry, pain, tired).
- Provides a simple and intuitive web interface for uploading audio and viewing results.
- Displays the top predicted cry reasons with their probabilities.

## Requirements
- Python 3.x
- Key libraries: Streamlit, TensorFlow (CPU version), Librosa, NumPy, Scikit-learn, Joblib.
- For a full list of dependencies, please see the `requirements.txt` file.

## How to Run
1.  **Clone the repository** (if you haven't already):
    ```bash
    # git clone <repository-url> # Replace <repository-url> with the actual URL
    # cd baby-cry-detector-app
    ```
    (Note: If you are operating within a pre-cloned environment, you can skip this step and ensure you are in the project's root directory.)

2.  **Install dependencies**:
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```
    This will start the application, and your browser should open to the app's URL automatically. If not, the console will display the URL to use (typically `http://localhost:8501`).

## How to Use
1.  Once the app is running, open the provided URL in your web browser.
2.  Click on the "Choose file" button to upload a WAV audio file.
    - The audio file should ideally be 2-5 seconds long.
    - Ensure the audio primarily contains a baby's cry for best results.
3.  After uploading, the app will automatically process the audio.
4.  The prediction results, indicating the most likely reasons for the cry along with their probabilities, will be displayed.

---
*This application is for informational purposes and should not replace professional medical advice or parental intuition.*