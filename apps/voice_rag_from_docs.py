# voice_rag_from_docs.py
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import pyttsx3

from backend.rag.pipeline import answer_query  # <-- IMPORTANT: use same brain

RECORD_SECONDS = 5
SAMPLE_RATE = 16000

# ---- Voice
stt = whisper.load_model("base")
tts = pyttsx3.init()
tts.setProperty("rate", 175)

def speak(text: str):
    tts.say(text)
    tts.runAndWait()

def record_wav(path="query.wav", seconds=RECORD_SECONDS):
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()
    write(path, SAMPLE_RATE, audio)
    return path

def transcribe(path="query.wav"):
    r = stt.transcribe(path)
    return (r.get("text") or "").strip()


print("✅ Voice Bank Support RAG Chatbot ready. Say 'exit' to stop.")
speak("Hello. Ask your banking question.")

while True:
    print("\n🎙️ Speak now...")
    wav = record_wav()
    query = transcribe(wav)
    print("You said:", query)

    if not query:
        speak("I didn't catch that. Please repeat.")
        continue

    if query.lower().strip() == "exit":
        speak("Bye!")
        break

    result = answer_query(query)  # <-- one line does RAG + LLM
    answer = result["answer"]

    print("\nAnswer:", answer)
    speak(answer)
