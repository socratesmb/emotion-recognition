from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO
import numpy as np
import os

def analyze_prosody(file_path: str) -> dict:
    # Verificar si el archivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
    
    # Leer el archivo de audio
    [Fs, x] = audioBasicIO.read_audio_file(file_path)
    if x is None:
        raise ValueError(f"No se pudo leer el archivo de audio: {file_path}")
    
    # Convertir a mono si es estéreo
    x = audioBasicIO.stereo_to_mono(x)
    
    # Definir parámetros de ventana corta y paso corto (en segundos)
    short_window = 0.05  # 50 ms
    short_step = 0.025   # 25 ms
    
    # Extraer características de término medio (e.g., energía, cero cruzamientos, MFCCs)
    mt_feats, _, _ = aF.mid_feature_extraction(x, Fs, 1.0 * Fs, 0.5 * Fs, short_window * Fs, short_step * Fs)
    
    # Promediar cada característica
    feat_means = np.mean(mt_feats, axis=1)
    
    # Mapear características a emociones (ejemplo heurístico)
    emotions = {
        'energy': float(feat_means[0]),
        'zero_crossing_rate': float(feat_means[1]),
        # Puedes agregar más características aquí
    }
    
    # Heurística para emociones básicas
    if emotions['energy'] > 0.7 and emotions['zero_crossing_rate'] > 0.2:
        emotions['predicted_emotion'] = "Alegría"
    elif emotions['energy'] < 0.2 and emotions['zero_crossing_rate'] < 0.1:
        emotions['predicted_emotion'] = "Tristeza"
    elif emotions['energy'] > 0.5 and emotions['zero_crossing_rate'] < 0.1:
        emotions['predicted_emotion'] = "Enojo"
    elif emotions['energy'] < 0.3 and emotions['zero_crossing_rate'] > 0.2:
        emotions['predicted_emotion'] = "Miedo"
    elif emotions['energy'] > 0.6 and 0.1 < emotions['zero_crossing_rate'] < 0.2:
        emotions['predicted_emotion'] = "Sorpresa"
    else:
        emotions['predicted_emotion'] = "Neutral"
    
    return emotions

# Ruta del archivo de audio
audio_path = "audio_data/ira.wav"

# Analizar características acústicas
try:
    print("Analizando características acústicas...")
    prosody = analyze_prosody(audio_path)
    print(f"Prosody: {prosody}\n")
except (FileNotFoundError, ValueError) as e:
    print(f"Error: {e}")