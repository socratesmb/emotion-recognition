from transformers import pipeline

# Función para analizar emociones en texto
def analyze_text_emotion(text: str) -> dict:
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    scores = classifier(text)[0]
    return {item['label']: item['score'] for item in scores}

# Leer el contenido de un archivo .txt
def read_text_from_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Ruta del archivo .txt
file_path = "text_data/tristeza_trans.txt" 

# Leer el texto del archivo
texto = read_text_from_file(file_path)

# Analizar emociones en el texto leído
print("Analizando emoción en texto...")
text_emotions = analyze_text_emotion(texto)
print(f"Emociones (texto): {text_emotions}\n")