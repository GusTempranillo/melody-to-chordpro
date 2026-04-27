"""
tonal_analyzer.py
=================
Estación 3: analiza las notas detectadas y determina la tonalidad.

Analogía: es como un sommelier que, después de identificar los
ingredientes de un vino (notas), infiere la región de origen (tonalidad).
No necesita ver la etiqueta — lo deduce del perfil de sabores.

Método: Krumhansl-Schmuckler Key-Finding Algorithm
  - Compara la distribución de clases de pitch (pitch class profile)
    con los perfiles de correlación de cada tonalidad mayor y menor
  - Elige la tonalidad cuyo perfil tiene mayor correlación de Pearson
  - Implementado nativamente en music21
"""

from dataclasses import dataclass
import numpy as np
from music21 import stream, note, pitch, key, analysis
from .pitch_detector import Note as DetectedNote


@dataclass
class TonalResult:
    """Resultado del análisis tonal."""
    key_name: str          # Ej: "D minor", "G major"
    root: str              # Ej: "D", "G"
    mode: str              # "major" o "minor"
    confidence: float      # Correlación de Pearson (0 a 1)
    scale_notes: list[str] # Notas de la escala detectada
    alternative_keys: list[dict]  # Otras tonalidades probables


class TonalAnalyzer:
    """
    Determina la tonalidad de una melodía a partir de su lista de notas.
    """

    def analyze(self, notes: list[DetectedNote]) -> TonalResult:
        """
        Analiza la tonalidad de la melodía.

        Args:
            notes: lista de notas detectadas por PitchDetector

        Returns:
            TonalResult con la tonalidad más probable y alternativas
        """
        if not notes:
            raise ValueError("No hay notas para analizar.")

        print("🎼 Analizando tonalidad...")

        # Convertir nuestras notas al formato de music21
        m21_stream = self._to_music21_stream(notes)

        # Análisis de tonalidad con Krumhansl-Schmuckler
        detected_key = m21_stream.analyze("key")

        # Calcular alternativas (las siguientes 3 más probables)
        alternatives = self._get_alternative_keys(m21_stream, detected_key)

        # Notas de la escala
        scale_notes = self._get_scale_notes(detected_key)

        result = TonalResult(
            key_name=str(detected_key),
            root=detected_key.tonic.name,
            mode=detected_key.mode,
            confidence=round(float(detected_key.correlationCoefficient), 3),
            scale_notes=scale_notes,
            alternative_keys=alternatives,
        )

        self._print_result(result)
        return result

    def _to_music21_stream(self, notes: list[DetectedNote]) -> stream.Stream:
        """
        Convierte nuestra lista de DetectedNote al formato Stream de music21.

        Analogía: es como traducir una lista de ingredientes (notas con tiempo)
        a una receta formal que el chef (music21) puede leer.
        """
        s = stream.Stream()

        for detected in notes:
            try:
                # Crear nota music21 desde número MIDI
                m21_note = note.Note()
                m21_note.pitch = pitch.Pitch(midi=detected.midi_number)
                # Duración en quarter notes (negras): 1 quarter = ~0.5s a 120bpm
                # Aproximación razonable para MVP
                m21_note.quarterLength = max(0.25, detected.duration * 2)
                s.append(m21_note)
            except Exception:
                # Si una nota falla, la saltamos (robustez ante notas inválidas)
                continue

        return s

    def _get_alternative_keys(
        self,
        s: stream.Stream,
        primary_key: key.Key,
        n: int = 3,
    ) -> list[dict]:
        """
        Obtiene las N tonalidades alternativas más probables.
        Usa .alternateInterpretations de music21 v9+.
        """
        alternatives = []
        for alt_key in primary_key.alternateInterpretations[:n]:
            alternatives.append({
                "key": str(alt_key),
                "correlation": round(float(alt_key.correlationCoefficient), 3),
            })
        return alternatives

    def _get_scale_notes(self, detected_key: key.Key) -> list[str]:
        """
        Devuelve los 7 grados de la escala detectada.

        Ej: D menor → ['D', 'E', 'F', 'G', 'A', 'B♭', 'C']
        """
        scale = detected_key.getScale()
        return [p.name for p in scale.getPitches("C1", "C2")[:-1]]

    def _print_result(self, result: TonalResult) -> None:
        print(f"\n✓ Tonalidad detectada: {result.key_name}")
        print(f"  Confianza (correlación): {result.confidence:.1%}")
        print(f"  Escala: {' - '.join(result.scale_notes)}")
        if result.alternative_keys:
            alts = ", ".join(
                f"{a['key']} ({a['correlation']:.2f})"
                for a in result.alternative_keys
            )
            print(f"  Alternativas: {alts}")

        # Advertencia si la confianza es baja
        if result.confidence < 0.7:
            print(
                "\n  ⚠️  Confianza baja — posibles causas:\n"
                "     - Melodía cromática o modal\n"
                "     - Muchas notas de adorno o poco sustain\n"
                "     - Tonalidad ambigua (ej: relativas mayor/menor)"
            )
