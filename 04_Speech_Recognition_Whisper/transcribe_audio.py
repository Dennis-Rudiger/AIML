import os
from transformers import pipeline

def transcribe_audio():
    """
    Uses OpenAI's open-source Whisper model to listen to an audio file
    and convert the speech into text locally on your machine.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(script_dir, "sample_audio.wav")
    
    if not os.path.exists(audio_path):
        print("Error: Could not find 'sample_audio.wav'. Please run download_sample.py first!")
        return

    print("====================================")
    print("1. LOADING THE AI MODEL (Whisper)")
    print("====================================")
    print("Loading the 'openai/whisper-tiny' model (this may take a minute the first time as it downloads the weights)...")
    
    # We use Hugging Face's pipeline for Automatic Speech Recognition (ASR)
    # The "tiny" model is extremely fast and runs perfectly on a standard CPU.
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    
    print("\nModel loaded successfully! Listening to the audio file...")

    print("\n====================================")
    print("2. TRANSCRIBING AUDIO to TEXT")
    print("====================================")
    
    # Pass the audio file to the transcriber
    result = transcriber(audio_path)
    
    print("\n🤖 AI Transcription Result:\n")
    print(f'"{result["text"].strip()}"')
    
    print("\n====================================")
    print("Process Complete! Try recording your own voice using Voice Recorder")
    print("and replace 'sample_audio.wav' to see it transcribe your own speech.")

if __name__ == "__main__":
    transcribe_audio()
