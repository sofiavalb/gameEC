import requests
import base64

def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")

    url = "http://localhost:11434/api/generate"
    data = {
        "model": "whisper",
        "audio": audio_base64
    }
    response = requests.post(url, json=data)
    return response.json()["response"]

transcribed_text = transcribe_audio("path/to/your/audio.wav")
print("Transcription:", transcribed_text)
