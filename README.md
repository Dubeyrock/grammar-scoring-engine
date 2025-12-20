# grammar-scoring-engine

## 🗣️ Grammar Scoring Engine for Voice Samples

A Streamlit-based web application that predicts the grammar quality of spoken English from `.wav` audio files. It uses audio features like MFCCs and Spectral Contrast, and a trained Random Forest Regressor to output a score between **0 and 5**.


## overview 


The objective of this competition is to develop a Grammar Scoring Engine for spoken data samples. You are provided with an audio dataset where each file is between 45 to 60 seconds long. The ground truth labels are MOS Likert Grammar Scores for each audio instance (see rubric below). Your task is to build a model that takes an audio file as input and outputs a continuous score ranging from 0 to 5.

Your submission will be assessed based on your ability to preprocess the audio data, select an appropriate methodology to solve the problem, and evaluate its performance using relevant metrics.

## Training: The training dataset consists of 444 samples.


## Testing (Evaluation): The testing dataset consists of 195 samples.

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
## deployment (streamlit ):

https://grammar-scoring-engine-emsgbqhgtkwehwehpv4h6f.streamlit.app/


## 📷 Screenshots

![Screenshot (2119)](https://github.com/user-attachments/assets/d91c54bc-6a18-45c5-9151-01c820ea9b11)

![Screenshot (2120)](https://github.com/user-attachments/assets/d78d3a07-85b8-4a3b-add8-465921a9b1cd)

## 🎥 Demo

[![Watch the Demo](https://img.youtube.com/vi/bMhJ5Ro0k78/0.jpg)](https://www.youtube.com/watch?v=bMhJ5Ro0k78)




built by shivam dubey
