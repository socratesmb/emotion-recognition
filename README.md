# Emotion Recognition

Una solución en Python para detectar la emoción predominante en un fragmento de audio de menos de 60 segundos combinando:

* **Transcripción automática** con Whisper (OpenAI) y traducción a inglés.
* **Análisis de emociones en texto** usando un modelo BERT preentrenado.
* **Validación y refuerzo** con un LLM (ChatGPT) para decidir la emoción final e incluir un emoji representativo.

---

## 📋 Estructura del proyecto

```
EMOTION-RECOGNITION/
├── audio_data/           # Audios de ejemplo (.wav)
│   ├── alegria.wav
│   ├── asco.wav
│   └── ...              
├── text_data/            # Transcripciones de audio (.txt)
│   ├── alegria.txt
│   ├── asco.txt
│   └── ...
├── transcribe.py         # Script de transcripción con Whisper
├── BERT.py               # Análisis de emociones de texto con Transformers
├── prosodic.py           # (Opcional) extracción de características prosódicas
├── main.py               # Orquestador: transcribe → analiza texto → refuerza con LLM
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Documentación (este archivo)
```

---

## 🔧 Requisitos

* Python 3.8 o superior
* ffmpeg (para que Whisper procese audio)
* Variables de entorno:

  * `API_KEY`: tu clave de OpenAI para acceder al servicio de ChatGPT.

### Dependencias Python

Se incluye un `requirements.txt`; instala con:

```bash
pip install -r requirements.txt
```

**Principales librerías**:

* `openai-whisper`  : Transcripción y traducción de audio.
* `transformers`    : Modelos BERT para clasificación de texto.
* `openai`         : Cliente para invocar el LLM (ChatGPT).
* `pyAudioAnalysis` : (Opcional) extracción de características acústicas.
* `pydub`          : Manejo de archivos de audio.

---

## 🚀 Instalación

1. Clona el repositorio:

   ```bash
   git clone https://github.com/socratesmb/emotion-recognition.git 
   cd emotion-recognition
   ```

2. Crea y activa un entorno virtual (recomendado):
   
    ```bash
    python -m venv venv
    source venv/bin/activate   # Linux/macOS
    venv\\Scripts\\activate  # Windows
    ```

3. Instala dependencias:

   ```bash
   pip install -r requirements.txt
   ```


4. Asegúrate de tener `ffmpeg` en tu PATH:
   ```bash
    # Linux/macOS
    sudo apt-get install ffmpeg
    # Windows
    Descargar desde https://ffmpeg.org/download.html y configurar variable PATH
    ```

---

## 🖥️ Uso

### 1. Transcripción (`transcribe.py`)

```bash
python transcribe.py
```

* Edita `audio_path` en `transcribe.py` para apuntar al archivo `.wav` que quieras procesar.
* Genera en `text_data/` un `.txt` con la transcripción traducida a inglés.

### 2. Análisis de texto (`BERT.py`)

```bash
python BERT.py
```

* Edita `file_path` dentro de `BERT.py` para apuntar al `.txt` deseado.
* Muestra en consola un diccionario con puntuaciones de emociones.

### 3. Pipeline completo (`main.py`)

```bash
python main.py
```

* Modifica la variable `audio` al inicio de `main.py` para seleccionar tu audio.
* El script:

  1. Transcribe y traduce.
  2. Analiza emociones de texto.
  3. Envía un prompt al LLM.
  4. Imprime la emoción predominante con emoji.

---

## 📂 Estructura detallada

* **`audio_data/`**: ejemplos de archivos `.wav` con cada emoción.
* **`text_data/`**: transcripciones resultantes.
* **`transcribe.py`**: función `transcribe_audio` basada en whisper.
* **`BERT.py`**: función `analyze_text_emotion` y lector de archivos.
* **`prosodic.py`**: extracción de características acústicas (opcional).
* **`main.py`**: orquestador principal combinando todo.
* **`requirements.txt`**: lista de paquetes pip.

---

## 🤝 Contribuciones

¡Bienvenidas! Para aportar:

1. Haz un *fork* del proyecto.
2. Crea una rama con tu mejora: `git checkout -b feature/mi-mejora`.
3. Realiza tus cambios y haz *commit*: `git commit -m "Agrega ..."`.
4. Empuja tu rama: `git push origin feature/mi-mejora`.
5. Abre un *Pull Request*.

---

## ⚖️ Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
