import whisper
import os

audio_path = "audio_data/sorpresa.wav"
# Función para transcribir audio
def transcribe_audio(audio_path):
    # Carga el modelo; opciones: tiny, base, small, medium, large
    model = whisper.load_model("small")
    # Procesar el audio, transcribirlo y traducirlo a ingles
    result = model.transcribe(audio_path, task="translate")
    print(result)
    # Retornar el resultado
    return result
# Ejecutar la función de transcripción y guardar el resultado en la variable text
text = transcribe_audio(audio_path)

# Guardar la transcripción en un archivo .txt
output_path = os.path.join("text_data", "tristeza.txt")
with open(output_path, "w", encoding="utf-8") as file:
    file.write(text["text"])

# Imprimir el resultado
print("Texto transcrito: ", text["text"].strip())