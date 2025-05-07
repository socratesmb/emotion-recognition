import os
import whisper

from transformers import pipeline
from openai import OpenAI

# 0. Configurar tu clave de OpenAI
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("API-KEY"),
)

# 1. Transcripción de audio.
def transcribe_audio(audio_path: str) -> str:
    model = whisper.load_model("small")
    result = model.transcribe(audio_path, task="translate")
    return result["text"].strip()

# 2. Análisis de emoción en texto con Transformers (BERT).
def analyze_text_emotion(text: str) -> dict:
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    scores = classifier(text)[0]
    return {item['label']: item['score'] for item in scores}

# 3. Validación de refuerzo con LLM.
def combine_with_llm(text: str, text_emotion: dict) -> str:
    prompt = (
        "Eres muy bueno en percepción emocional.\n"
        f"Texto transcrito traducido a ingles:\n{text}\n"
        f"Resultado de análisis de emociones con BERT de Google: {text_emotion}\n"
        "Analiza bien el texto transcrito, genera un analisis de emociones y comparalo con el analisis de BERT, para que al final me indiques la emoción predominante. Las emociones que determines quiero que me la menciones junto con un emogi de la emoción que detectaste. "
    )
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="Eres un experto en análisis emocional de personas.",
        input=prompt,
    )
    return response.output_text

# Flujo principal
def main(audio_path: str):
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Archivo no encontrado: {audio_path}")

    print("1. Transcribiendo audio...")
    texto = transcribe_audio(audio_path)
    print(f"Texto: {texto}\n")

    print("2. Analizando emoción en texto...")
    text_emotions = analyze_text_emotion(texto)
    print(f"Emociones: {text_emotions}\n")

    print("3. Combinando con LLM para validación final...")
    emotion = combine_with_llm(texto, text_emotions)
    print(f"Emoción detectada: {emotion}")

if __name__ == "__main__":
    audio = "audio_data/alegria.wav"  # Cambia esto a la ruta de tu archivo de audio
    main(audio)
