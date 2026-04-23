import urllib.request
import os

def download_audio_sample():
    """
    Downloads a sample spoken sentence (.wav file) so we have something to transcribe.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "sample_audio.wav")
    
    print("Downloading a sample audio file...")
    
    # A public domain voice sample from Harvard's open dataset
    url = "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav"
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ Success! Saved to {output_path}")
        print("This file contains a few random spoken sentences.")
    except Exception as e:
        print(f"Failed to download audio. Error: {e}")
        print("Please ensure you are connected to the internet.")

if __name__ == "__main__":
    download_audio_sample()
