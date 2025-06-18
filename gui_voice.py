"""
🎤 Refactored GUI for Voice-based Age and Emotion Detection
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import joblib
import librosa
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Constants
MAX_PAD_LEN = 174
N_MFCC = 40
SAMPLE_RATE = 22050

# Paths
MODEL_DIR = "saved_models"
GENDER_MODEL_PATH = os.path.join(MODEL_DIR, 'model_gender_voice.joblib')
GENDER_SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_gender_voice.joblib')
AGE_MODEL_PATH = os.path.join(MODEL_DIR, 'model_age_voice.joblib')
AGE_SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_age_voice.joblib')
EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, 'model_emotion_voice.joblib')
EMOTION_SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_emotion_voice.joblib')

# Label maps
gender_map_gui = {0: 'Male', 1: 'Female'}
age_map_gui = {0: 'Youth (0-18)', 1: 'Adult (19-60)', 2: 'Senior (61+)'}
emotion_map_gui = {0: 'Neutral', 1: 'Calm', 2: 'Happy', 3: 'Sad', 4: 'Angry', 5: 'Fearful', 6: 'Disgust', 7: 'Surprised'}
SENIOR_AGE_LABEL = 'Senior (61+)'

# Load models
try:
    print("Loading models and scalers...")
    model_gender_voice = joblib.load(GENDER_MODEL_PATH)
    scaler_gender_voice = joblib.load(GENDER_SCALER_PATH)
    model_age_voice = joblib.load(AGE_MODEL_PATH)
    scaler_age_voice = joblib.load(AGE_SCALER_PATH)
    model_emotion_voice = joblib.load(EMOTION_MODEL_PATH)
    scaler_emotion_voice = joblib.load(EMOTION_SCALER_PATH)
    print("Models and scalers loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading models/scalers: {e}")
    exit()
except Exception as e:
    print(f"Unexpected error: {e}")
    exit()

# MFCC Feature Extractor
def generate_mfcc_features(file_path, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN, sr=SAMPLE_RATE):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', sr=sr, duration=10)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs_padded = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs_padded = mfccs[:, :max_pad_len]
        return mfccs_padded.flatten()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Inference
def run_voice_prediction_audio_file(file_path):
    status_label.config(text="Processing audio...")
    root.update_idletasks()

    features = generate_mfcc_features(file_path)
    if features is None:
        status_label.config(text="Error: Could not extract features.")
        return

    try:
        features = features.reshape(1, -1)
        gender_encoded = model_gender_voice.predict(scaler_gender_voice.transform(features))[0]
        gender_label = gender_map_gui.get(gender_encoded, f"Unknown Gender ({gender_encoded})")

        if gender_label == 'Female':
            status_label.config(text="Upload male voice.")
            return

        age_encoded = model_age_voice.predict(scaler_age_voice.transform(features))[0]
        age_label = age_map_gui.get(age_encoded, f"Unknown Age Group ({age_encoded})")

        result = f"Gender: {gender_label}\nAge Group: {age_label}\n"

        if age_label == SENIOR_AGE_LABEL:
            emotion_encoded = model_emotion_voice.predict(scaler_emotion_voice.transform(features))[0]
            emotion_label = emotion_map_gui.get(emotion_encoded, f"Unknown Emotion ({emotion_encoded})")
            result += f"Status: Senior Citizen\nEmotion: {emotion_label}"
        else:
            result += "Status: Not Senior Citizen"

        status_label.config(text=result)
    except Exception as e:
        status_label.config(text=f"Prediction error: {e}")
        print(f"Debug error: {e}")

# File uploader
def upload_action():
    file_path = filedialog.askopenfilename(title="Select Audio File", filetypes=(("Audio Files", "*.wav *.mp3"),))
    if file_path:
        run_voice_prediction_audio_file(file_path)
    else:
        print("No file selected.")

# GUI Setup
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Age & Emotion Detector (Male Voices)")
    root.geometry("450x300")

    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill=tk.BOTH)

    title_label = ttk.Label(main_frame, text="Voice Analysis", font=("Helvetica", 16, "bold"))
    title_label.pack(pady=(0, 15))

    upload_button = ttk.Button(main_frame, text="Upload Audio File (.wav, .mp3)", command=upload_action, width=30)
    upload_button.pack(pady=10)

    status_label = ttk.Label(main_frame, text="Upload an audio file to begin analysis.",
                             justify=tk.LEFT, wraplength=400, padding=(10, 10),
                             relief=tk.SUNKEN, borderwidth=1, anchor='nw')
    status_label.pack(pady=10, fill=tk.BOTH, expand=True)

    root.mainloop()
