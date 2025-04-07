# grammar-scoring-engine



# 🗣️ Grammar Scoring Engine for Voice Samples

A Streamlit-based web application that predicts the grammar quality of spoken English from `.wav` audio files. It uses audio features like MFCCs and Spectral Contrast, and a trained Random Forest Regressor to output a score between **0 and 5**.

## 🚀 Features

- 🎤 Upload `.wav` voice samples and get an instant grammar score.
- 📘 Audio-based grammar feedback (Beginner → Advanced).
- 📊 Visual insights: waveform + MFCC heatmap.
- 📁 Planned: Batch scoring of multiple audio files.

## 🧠 How It Works

- Extracts audio features using [Librosa](https://librosa.org/):  
  - MFCCs (Mel-frequency cepstral coefficients)  
  - Spectral Contrast  
  - Zero Crossing Rate (ZCR)
- Predicts grammar score using a trained **Random Forest Regressor**.
- Built using:
  - `Python`
  - `Streamlit`
  - `Scikit-learn`
  - `Librosa`

## 📷 Screenshots

![Screenshot (2119)](https://github.com/user-attachments/assets/d91c54bc-6a18-45c5-9151-01c820ea9b11)

![Screenshot (2120)](https://github.com/user-attachments/assets/d78d3a07-85b8-4a3b-add8-465921a9b1cd)




