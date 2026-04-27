"""
pitch_detector.py
=================
Estación 2: detecta el pitch (frecuencia fundamental) frame a frame
y lo convierte a notas musicales con tiempos.

Algoritmo: pyin (Probabilistic YIN)
  - Más preciso que YIN clásico para voz monofónica
  - Devuelve probabilidades de confianza por frame
  - Analogía: es como un afinador cromático profesional que,
    en vez de solo decirte "estás entre La y La#",
    te dice "hay un 87% de probabilidad de que sea La4"

Limitación clave:
  - Funciona bien con voz limpia y estable
  - Vibrato excesivo, glissandos o ruido de fondo generan "notas fantasma"
  - Silabas muy cortas (<50ms) pueden no detectarse
"""

import numpy as np
import librosa
from dataclasses import dataclass, field


@dataclass
class Note:
    """Representa una nota musical detectada en el audio."""
    midi_number: int          # Número MIDI (60 = C4 = Do4)
    name: str                 # Nombre de la nota (ej: "A4", "C#3")
    frequency_hz: float       # Frecuencia en Hz
    start_time: float         # Tiempo de inicio en segundos
    end_time: float           # Tiempo de fin en segundos
    confidence: float         # Confianza del detector (0.0 a 1.0)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __str__(self):
        return (
            f"{self.name:4s} | "
            f"{self.start_time:.2f}s–{self.end_time:.2f}s | "
            f"conf: {self.confidence:.0%}"
        )


class PitchDetector:
    """
    Detecta el pitch de una señal de audio monofónica (voz sola)
    y devuelve una lista de objetos Note con tiempo y confianza.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,  # Descarta frames poco confiables
        min_note_duration: float = 0.08,      # Notas < 80ms se ignoran (ruido)
        frame_length: int = 2048,
        hop_length: int = 512,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_note_duration = min_note_duration
        self.frame_length = frame_length
        self.hop_length = hop_length

    def detect(self, y: np.ndarray, sr: int) -> list[Note]:
        """
        Detecta notas en el audio.

        Args:
            y:  array de muestras de audio (mono, normalizado)
            sr: sample rate

        Returns:
            Lista de Note ordenada por tiempo de inicio
        """
        print("🎵 Detectando pitch con pyin...")

        # pyin devuelve tres arrays:
        #   f0:            frecuencia estimada por frame (Hz), None si no detectó
        #   voiced_flag:   True si el frame tiene voz
        #   voiced_probs:  probabilidad de que el frame tenga voz
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),   # 65 Hz — límite inferior razonable para voz
            fmax=librosa.note_to_hz("C7"),   # 2093 Hz — límite superior para soprano
            sr=sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )

        # Convertir índices de frames a tiempos en segundos
        times = librosa.frames_to_time(
            np.arange(len(f0)),
            sr=sr,
            hop_length=self.hop_length,
        )

        # Agrupar frames consecutivos en "notas"
        notes = self._frames_to_notes(f0, voiced_flag, voiced_probs, times)

        print(f"✓ {len(notes)} notas detectadas")
        return notes

    def _frames_to_notes(
        self,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
        voiced_probs: np.ndarray,
        times: np.ndarray,
    ) -> list[Note]:
        """
        Convierte arrays de frames a lista de notas agrupadas.

        Analogía: imagina que tienes una cinta de papel donde cada
        centímetro representa 23ms de audio, y en cada centímetro
        hay escrita una frecuencia. Este método agrupa los centímetros
        consecutivos con la misma nota en "palabras" (notas con duración).
        """
        notes = []
        current_note = None
        current_frames = []

        for i, (freq, voiced, prob) in enumerate(
            zip(f0, voiced_flag, voiced_probs)
        ):
            # ¿Es este frame un pitch confiable?
            is_valid = (
                voiced
                and prob >= self.confidence_threshold
                and freq is not None
                and not np.isnan(freq)
            )

            if is_valid:
                midi = self._freq_to_midi(freq)

                # ¿Es la misma nota que el frame anterior?
                if current_note is not None and midi == current_note:
                    current_frames.append((i, freq, prob))
                else:
                    # Guardar nota anterior si era válida
                    if current_note is not None and current_frames:
                        note = self._build_note(current_note, current_frames, times)
                        if note:
                            notes.append(note)

                    # Empezar nueva nota
                    current_note = midi
                    current_frames = [(i, freq, prob)]
            else:
                # Frame sin voz: cerrar nota actual
                if current_note is not None and current_frames:
                    note = self._build_note(current_note, current_frames, times)
                    if note:
                        notes.append(note)
                current_note = None
                current_frames = []

        # Última nota
        if current_note is not None and current_frames:
            note = self._build_note(current_note, current_frames, times)
            if note:
                notes.append(note)

        return notes

    def _build_note(
        self,
        midi_number: int,
        frames: list[tuple],
        times: np.ndarray,
    ) -> Note | None:
        """Construye un objeto Note a partir de frames agrupados."""
        if not frames:
            return None

        start_idx = frames[0][0]
        end_idx = frames[-1][0]
        start_time = times[start_idx]
        end_time = times[min(end_idx + 1, len(times) - 1)]

        # Filtrar notas demasiado cortas (probablemente ruido)
        if (end_time - start_time) < self.min_note_duration:
            return None

        freqs = [f[1] for f in frames]
        probs = [f[2] for f in frames]
        avg_freq = np.mean(freqs)
        avg_conf = np.mean(probs)

        return Note(
            midi_number=midi_number,
            name=librosa.midi_to_note(midi_number),
            frequency_hz=round(avg_freq, 2),
            start_time=round(start_time, 3),
            end_time=round(end_time, 3),
            confidence=round(avg_conf, 3),
        )

    def _freq_to_midi(self, freq: float) -> int:
        """
        Convierte frecuencia Hz a número MIDI redondeado al semitono más cercano.
        MIDI 69 = A4 = 440 Hz (el "la de afinación" universal).
        """
        return int(round(librosa.hz_to_midi(freq)))

    def print_notes(self, notes: list[Note]) -> None:
        """Imprime la lista de notas en formato legible."""
        print(f"\n{'Nota':<5} {'Inicio':>8} {'Fin':>8} {'Dur':>7} {'Conf':>6}")
        print("-" * 42)
        for note in notes:
            print(
                f"{note.name:<5} "
                f"{note.start_time:>7.2f}s "
                f"{note.end_time:>7.2f}s "
                f"{note.duration:>6.2f}s "
                f"{note.confidence:>5.0%}"
            )
