"""
harmonizer.py
=============
Estación 4: propone acordes que encajan con la melodía.

Analogía: es como un acompañante de jazz que escucha la melodía
y elige de memoria qué acordes del cancionero funcionan bien.
No inventa acordes nuevos — elige los mejores de un catálogo conocido.

Estrategia:
  1. Dividir la melodía en "ventanas" de tiempo (ej: cada 2s)
  2. Para cada ventana, identificar las notas predominantes
  3. Mapear esas notas a grados de la escala (I, II, III...)
  4. Asignar el acorde funcional más coherente
  5. Suavizar transiciones (evitar saltos de acorde raros)

Limitación: es armonía funcional básica (común-práctica occidental).
No funciona bien con flamenco, modos griegos, música microtonal, etc.
"""

from dataclasses import dataclass, field
from collections import Counter
import numpy as np
from music21 import chord as m21chord, key, pitch, roman, harmony

from .pitch_detector import Note as DetectedNote
from .tonal_analyzer import TonalResult


# Acordes por grado para modo mayor (cifrado anglosajón)
# Ej: en Do mayor → I=C, II=Dm, III=Em, IV=F, V=G, VI=Am, VII=Bdim
MAJOR_CHORDS = {
    0: ("I",   "maj"),   # Tónica
    1: ("bII", "maj"),   # Napolitano (raro, pero posible)
    2: ("II",  "min"),   # Supertónica
    3: ("III", "min"),   # Mediante
    4: ("IV",  "maj"),   # Subdominante
    5: ("V",   "maj"),   # Dominante (el más importante)
    6: ("VII", "dim"),   # Sensible disminuido
    7: ("I",   "maj"),   # Octava → misma que tónica
    8: ("VI",  "min"),   # Submediante relativa menor
    9: ("II",  "min"),   # Supertónica
    10: ("bVII","maj"),  # Séptimo grado mayor (común en pop/rock)
    11: ("VII", "dim"),  # Sensible
}

# Acordes por grado para modo menor natural
MINOR_CHORDS = {
    0: ("I",   "min"),   # Tónica menor
    1: ("bII", "maj"),   # Napolitano
    2: ("II",  "dim"),   # Supertónica disminuida
    3: ("bIII","maj"),   # Mediante mayor relativa
    4: ("IV",  "min"),   # Subdominante menor
    5: ("V",   "maj"),   # Dominante (usando sensible armónica)
    6: ("bVI", "maj"),   # Submediante mayor
    7: ("I",   "min"),   # Octava
    8: ("bVI", "maj"),   # Submediante
    9: ("IV",  "min"),   # Subdominante
    10: ("bVII","maj"),  # Séptimo mayor
    11: ("VII", "dim"),  # Sensible
}


@dataclass
class ChordSuggestion:
    """Un acorde sugerido para un segmento de la melodía."""
    chord_name: str        # Ej: "Am", "G", "F#dim"
    root: str              # Ej: "A", "G", "F#"
    quality: str           # "maj", "min", "dim", "aug"
    start_time: float      # Segundo de inicio
    end_time: float        # Segundo de fin
    degree: str            # Ej: "I", "IV", "V"
    notes_covered: list[str] = field(default_factory=list)  # Notas de la melodía que cubre

    @property
    def display_name(self) -> str:
        """Nombre para mostrar en ChordPro (sin 'maj' explícito)."""
        if self.quality == "maj":
            return self.root
        elif self.quality == "min":
            return self.root + "m"
        elif self.quality == "dim":
            return self.root + "dim"
        elif self.quality == "aug":
            return self.root + "aug"
        return self.chord_name

    def __str__(self):
        return (
            f"[{self.display_name}] "
            f"{self.start_time:.1f}s–{self.end_time:.1f}s "
            f"(grado {self.degree})"
        )


class Harmonizer:
    """
    Sugiere acordes para acompañar una melodía monofónica.
    """

    def __init__(self, window_size: float = 2.0):
        """
        Args:
            window_size: duración de cada "ventana" de análisis en segundos.
                         2s es un buen balance: cambia con suficiente frecuencia
                         sin resultar errático. Puedes bajar a 1s para canciones
                         rápidas o subir a 4s para baladas lentas.
        """
        self.window_size = window_size

    def suggest_chords(
        self,
        notes: list[DetectedNote],
        tonal: TonalResult,
        duration: float,
    ) -> list[ChordSuggestion]:
        """
        Genera sugerencias de acordes para toda la canción.

        Args:
            notes:    lista de notas detectadas
            tonal:    resultado del análisis tonal
            duration: duración total del audio en segundos

        Returns:
            Lista de ChordSuggestion ordenada por tiempo
        """
        print(f"🎸 Armonizando en {tonal.key_name}...")

        # Crear ventanas de tiempo
        windows = self._create_windows(duration)

        # Mapear notas a cada ventana
        chords = []
        for start, end in windows:
            window_notes = [
                n for n in notes
                if n.start_time < end and n.end_time > start
            ]

            if not window_notes:
                # Ventana vacía: repetir el último acorde o usar tónica
                if chords:
                    prev = chords[-1]
                    chords.append(ChordSuggestion(
                        chord_name=prev.chord_name,
                        root=prev.root,
                        quality=prev.quality,
                        start_time=start,
                        end_time=end,
                        degree=prev.degree,
                    ))
                else:
                    chords.append(self._tonic_chord(tonal, start, end))
                continue

            chord = self._choose_chord(window_notes, tonal, start, end)
            chords.append(chord)

        # Suavizar: eliminar cambios de acorde redundantes consecutivos
        chords = self._smooth_chords(chords)

        print(f"✓ {len(chords)} acordes sugeridos")
        for ch in chords:
            print(f"  {ch}")

        return chords

    def _create_windows(self, duration: float) -> list[tuple[float, float]]:
        """Divide la canción en ventanas de window_size segundos."""
        windows = []
        t = 0.0
        while t < duration:
            end = min(t + self.window_size, duration)
            windows.append((round(t, 3), round(end, 3)))
            t += self.window_size
        return windows

    def _choose_chord(
        self,
        window_notes: list[DetectedNote],
        tonal: TonalResult,
        start: float,
        end: float,
    ) -> ChordSuggestion:
        """
        Elige el acorde más adecuado para una ventana de notas.

        Método: weighted pitch class profile
          - Cada nota vota por su clase de pitch, ponderada por duración
          - La clase de pitch más "votada" es la nota predominante
          - Se mapea al acorde que mejor contiene esa nota en la tonalidad
        """
        # Perfil de clases de pitch ponderado por duración
        pitch_weights = Counter()
        for n in window_notes:
            pc = n.midi_number % 12  # Clase de pitch (0=C, 1=C#, ..., 11=B)
            weight = n.duration * n.confidence
            pitch_weights[pc] += weight

        # Tónica de la tonalidad (0-11)
        tonic_pc = pitch.Pitch(tonal.root).pitchClass

        # Nota predominante en la ventana
        dominant_pc = pitch_weights.most_common(1)[0][0]

        # Intervalo desde la tónica
        interval = (dominant_pc - tonic_pc) % 12

        # Seleccionar acorde según modo y grado
        chord_map = MINOR_CHORDS if tonal.mode == "minor" else MAJOR_CHORDS
        degree_name, quality = chord_map.get(interval, ("I", "maj" if tonal.mode == "major" else "min"))

        # Calcular la nota raíz del acorde propuesto
        root_midi = tonic_pc  # simplificación: usamos la tónica como raíz base
        # Ajustar para grados especiales
        root_offset = self._degree_to_semitone(degree_name)
        chord_root_pc = (tonic_pc + root_offset) % 12
        chord_root_name = pitch.Pitch(chord_root_pc).name.replace("-", "b")

        # Notas de la melodía cubiertas por este acorde
        chord_pcs = self._get_chord_pcs(chord_root_pc, quality)
        covered = [
            n.name for n in window_notes
            if (n.midi_number % 12) in chord_pcs
        ]

        chord_display = chord_root_name + ("m" if quality == "min" else
                                           "dim" if quality == "dim" else "")

        return ChordSuggestion(
            chord_name=chord_display,
            root=chord_root_name,
            quality=quality,
            start_time=start,
            end_time=end,
            degree=degree_name,
            notes_covered=covered,
        )

    def _degree_to_semitone(self, degree: str) -> int:
        """Convierte un grado romano a semitonos desde la tónica."""
        MAP = {
            "I": 0, "bII": 1, "II": 2, "bIII": 3, "III": 4,
            "IV": 5, "bV": 6, "V": 7, "bVI": 8, "VI": 9,
            "bVII": 10, "VII": 11,
        }
        return MAP.get(degree, 0)

    def _get_chord_pcs(self, root_pc: int, quality: str) -> set[int]:
        """Devuelve las clases de pitch de un acorde (tríada)."""
        if quality == "maj":
            return {root_pc % 12, (root_pc + 4) % 12, (root_pc + 7) % 12}
        elif quality == "min":
            return {root_pc % 12, (root_pc + 3) % 12, (root_pc + 7) % 12}
        elif quality == "dim":
            return {root_pc % 12, (root_pc + 3) % 12, (root_pc + 6) % 12}
        return {root_pc % 12}

    def _tonic_chord(
        self, tonal: TonalResult, start: float, end: float
    ) -> ChordSuggestion:
        """Devuelve el acorde de tónica (para ventanas vacías)."""
        quality = "min" if tonal.mode == "minor" else "maj"
        root = tonal.root.replace("-", "b")
        return ChordSuggestion(
            chord_name=root + ("m" if quality == "min" else ""),
            root=root,
            quality=quality,
            start_time=start,
            end_time=end,
            degree="I",
        )

    def _smooth_chords(
        self, chords: list[ChordSuggestion]
    ) -> list[ChordSuggestion]:
        """
        Elimina repeticiones consecutivas del mismo acorde,
        extendiendo la duración del primero.

        Analogía: si el guitarrista toca el mismo acorde dos veces
        seguidas, simplemente lo sostiene más tiempo.
        """
        if not chords:
            return []

        smoothed = [chords[0]]
        for ch in chords[1:]:
            if ch.chord_name == smoothed[-1].chord_name:
                # Extender el acorde anterior
                smoothed[-1].end_time = ch.end_time
            else:
                smoothed.append(ch)

        return smoothed
