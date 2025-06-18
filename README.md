# 🎤 Voice-Based Age and Emotion Detection (Male Voices Only)

This project uses machine learning to detect a speaker's **age group** and **emotion** from a voice clip — only if the speaker is **male**. Female voices are automatically rejected. The application includes a clean, user-friendly GUI built using Tkinter.

## ✅ Features

- 🎯 Detects **gender** from audio
- 🚫 Rejects **female voice** with a message
- 👦👨👴 Predicts **age group**:
  - Youth (0–18)
  - Adult (19–60)
  - Senior (61+)
- 😄😡😭 If senior, predicts **emotion** from 8 categories
- 🖥️ Tkinter GUI for selecting `.mp3` or `.wav` audio files

## 🗂️ Project Structure

voice-age-emotion-detector-gui/
├── gui_voice_refactored_fixed.py        # Main GUI app  
├── train_models_cleaned.ipynb           # Model training notebook  
├── saved_models/                        # Trained model & scaler files  
├── sample_audio/                        # Test audio clips (optional)  
├── requirements.txt                     # Python dependencies  
└── README.md                            # Project overview

## 🚀 How to Run the Project

###  ✅ Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```
### ✅ Step 2: Train the Models (Optional)
You can run the training notebook to generate the required model files:

```bash
train_models_cleaned.ipynb
```

This will create .joblib model and scaler files in the saved_models/ directory:

model_gender_voice.joblib

model_age_voice.joblib

model_emotion_voice.joblib

And their respective scalers

You can skip this step if the saved models are already included.

### ✅ Step 3: Run the GUI App
Use the following command to start the application:
```bash
python gui_voice.py
```
This will open the GUI window where you can upload a voice clip for prediction.

### ✅ Step 4: Upload a Voice File
Supported formats: .wav or .mp3

✅ If male voice:

The system will predict age group

If the person is a senior (age > 60), it will also predict emotion

❌ If female voice:

The system will reject the input and display:
"Upload male voice."



