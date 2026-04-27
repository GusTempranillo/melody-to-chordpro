"""
audio_loader.py
===============
Estación 1 de la pipeline: carga el MP3 y lo normaliza.

Analogía: es el técnico de sonido que recibe la cinta bruta,
la convierte al formato estándar del estudio y ajusta el volumen
antes de pasársela al resto del equipo.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


# Constantes del estudio
TARGET_SR = 22050   # Sample rate estándar (Hz) — lo que usa librosa internamente
TARGET_CHANNELS = 1  # Mono: una sola pista (la voz)


class AudioLoader:
    """
    Carga un MP3 (o WAV/OGG/FLAC) y devuelve:
      - y:  array numpy con las muestras de audio
      - sr: sample rate
      - duration: duración en segundos
    """

    def __init__(self, target_sr: int = TARGET_SR):
        self.target_sr = target_sr

    def load(self, file_path: str | Path) -> dict:
        """
        Carga el archivo de audio y lo normaliza.

        Returns:
            dict con claves:
              - 'y':        np.ndarray  — muestras de audio normalizadas
              - 'sr':       int         — sample rate
              - 'duration': float       — duración en segundos
              - 'path':     Path        — ruta original
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {path}")

        supported = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
        if path.suffix.lower() not in supported:
            raise ValueError(
                f"Formato no soportado: {path.suffix}. "
                f"Usa uno de: {', '.join(supported)}"
            )

        # librosa convierte automáticamente a mono y al SR deseado
        # Es como pedirle al técnico que saque UNA pista limpia a 22kHz
        y, sr = librosa.load(str(path), sr=self.target_sr, mono=True)

        # Normalización de amplitud: llevamos el pico a 0 dB
        # (sin esto, una voz susurrada y una a todo volumen tendrían
        # comportamientos muy distintos en el detector de pitch)
        y = self._normalize(y)

        duration = librosa.get_duration(y=y, sr=sr)

        print(f"✓ Audio cargado: {path.name}")
        print(f"  Duración: {duration:.2f}s | SR: {sr}Hz | Shape: {y.shape}")

        return {
            "y": y,
            "sr": sr,
            "duration": duration,
            "path": path,
        }

    def save_wav(self, audio: dict, output_path: str | Path) -> Path:
        """
        Guarda el audio normalizado como WAV.
        Útil para debuggear o pasar a otros módulos que prefieren WAV.
        """
        output_path = Path(output_path)
        sf.write(str(output_path), audio["y"], audio["sr"])
        print(f"✓ WAV guardado en: {output_path}")
        return output_path

    def _normalize(self, y: np.ndarray) -> np.ndarray:
        """
        Normalización peak: escala el audio para que el sample
        máximo sea exactamente 1.0 (o -1.0 si es negativo).

        Analogía: es como subir el volumen hasta que el techo
        acústico esté justo al borde sin distorsión.
        """
        peak = np.max(np.abs(y))
        if peak == 0:
            raise ValueError("El audio está en silencio (todos los samples son 0).")
        return y / peak

    def get_info(self, file_path: str | Path) -> dict:
        """
        Devuelve metadatos del archivo sin cargarlo completamente.
        Útil para validaciones rápidas antes de procesar.
        """
        path = Path(file_path)
        duration = librosa.get_duration(path=str(path))
        return {
            "name": path.name,
            "size_mb": round(path.stat().st_size / 1_048_576, 2),
            "duration_s": round(duration, 2),
            "format": path.suffix.lower(),
        }
