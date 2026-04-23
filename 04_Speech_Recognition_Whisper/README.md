# Project 4: Speech Recognition with OpenAI Whisper 🎙️📜

Adding an audio processing component rounds out your AI portfolio perfectly. Now you have covered Tabular Data, Computer Vision, Text/LLMs, and **Sound!**

## The Goal
To build a system that listens to an audio recording of a human speaking and accurately transcribes it into text using Deep Learning (Automatic Speech Recognition or ASR).

## Skills Demonstrated
*   **Frameworks:** `transformers` (Hugging Face), `pytorch`
*   **Audio Processing Model:** `openai/whisper-tiny`
*   **Concepts:** Automatic Speech Recognition (ASR), Model Pipelines, Downloading external data streams.

## 📁 Files Explained
1.  **`download_sample.py`**: Just a quick script that automatically grabs a public domain `.wav` file of an American speaker reading random sentences.
2.  **`transcribe_audio.py`**: The core AI logic. It uses Hugging Face's pipeline tool to load the open-source "Whisper" model by OpenAI and convert the audio directly to text on your local machine.

## 🚀 How to Run (Beginner Friendly)

**Step 1:** Install checking the required libraries:
```powershell
pip install transformers torch soundfile librosa
```
*(Note: `soundfile` and `librosa` are standard Python libraries used to read audio data.)*

**Step 2:** Download the sample audio:
```powershell
python download_sample.py
```

**Step 3:** Run the AI Transcriber:
```powershell
python transcribe_audio.py
```

## Bonus Challenge
Open the "Voice Recorder" app on Windows, record yourself saying something like *"Hello, I am testing my new machine learning model,"* save it as `sample_audio.wav` in this folder, and run the script again!
