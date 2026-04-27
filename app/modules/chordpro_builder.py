"""
chordpro_builder.py
===================
Estación 5 (final): ensambla todo en un archivo ChordPro editable.

¿Qué es ChordPro?
  Formato de texto plano donde los acordes van entre corchetes,
  justo antes de la sílaba donde se tocan.
  Ej:  [Am]Ayer soñé [F]contigo
  Es el "markdown de las canciones" — simple, editable, universal.

Analogía de esta estación: es el editor que toma las notas del
musicólogo (acordes), las del transcriptor (letra) y los del
reloj (tiempos) y los ensambla en un documento coherente.

Sin Whisper (MVP sin letra):
  Genera la estructura con acordes y notas, marcando con "???"
  las sílabas donde no hay letra disponible.
"""

from dataclasses import dataclass
from pathlib import Path

from .pitch_detector import Note as DetectedNote
from .tonal_analyzer import TonalResult
from .harmonizer import ChordSuggestion


@dataclass
class ChordProResult:
    """Resultado final completo."""
    chordpro_text: str
    key_name: str
    tempo_bpm: float
    notes: list[DetectedNote]
    chords: list[ChordSuggestion]
    lyrics: str | None  # None si Whisper no está disponible


class ChordProBuilder:
    """
    Construye el documento ChordPro final a partir de todos los módulos.
    """

    def build(
        self,
        notes: list[DetectedNote],
        tonal: TonalResult,
        chords: list[ChordSuggestion],
        duration: float,
        lyrics: str | None = None,
        title: str = "Mi Canción",
        artist: str = "",
        tempo_bpm: float = 0,
    ) -> ChordProResult:
        """
        Genera el documento ChordPro.

        Args:
            notes:      lista de notas detectadas
            tonal:      resultado del análisis tonal
            chords:     acordes sugeridos
            duration:   duración total en segundos
            lyrics:     letra transcrita (opcional — requiere Whisper)
            title:      título de la canción
            artist:     artista/intérprete
            tempo_bpm:  tempo estimado (0 = no detectado)

        Returns:
            ChordProResult con el texto ChordPro y metadatos
        """
        print("📄 Generando ChordPro...")

        lines = []

        # === CABECERA (directivas ChordPro) ===
        lines.append(f"{{title: {title}}}")
        if artist:
            lines.append(f"{{artist: {artist}}}")
        lines.append(f"{{key: {tonal.key_name}}}")
        if tempo_bpm:
            lines.append(f"{{tempo: {int(tempo_bpm)}}}")
        lines.append(f"{{capo: 0}}")
        lines.append("")
        lines.append(f"# Generado por melody-to-chordpro")
        lines.append(f"# Tonalidad: {tonal.key_name} (confianza: {tonal.confidence:.0%})")
        lines.append(f"# Duración: {duration:.1f}s")
        lines.append(f"# Notas detectadas: {len(notes)}")
        lines.append(f"# Escala: {' - '.join(tonal.scale_notes)}")
        lines.append("")

        # === MELODÍA (lista de notas) ===
        lines.append("{comment: === MELODÍA DETECTADA ===}")
        lines.append("{start_of_verse: Notas}")
        melody_line = self._build_melody_line(notes)
        lines.append(melody_line)
        lines.append("{end_of_verse}")
        lines.append("")

        # === SECCIÓN PRINCIPAL (letra + acordes) ===
        lines.append("{comment: === LETRA Y ACORDES ===}")
        lines.append("{start_of_verse: Verso 1}")

        if lyrics:
            # Con letra: alinear acordes con el texto
            chord_lyric_lines = self._align_chords_with_lyrics(
                lyrics, chords, duration
            )
            lines.extend(chord_lyric_lines)
        else:
            # Sin letra: mostrar acordes con marcadores de tiempo
            lines.append("# Sin letra — instala Whisper para transcripción automática:")
            lines.append("# pip install openai-whisper")
            lines.append("")
            chord_lines = self._build_chord_only_lines(chords)
            lines.extend(chord_lines)

        lines.append("{end_of_verse}")
        lines.append("")

        # === ACORDES SUGERIDOS (referencia) ===
        lines.append("{comment: === ACORDES SUGERIDOS ===}")
        unique_chords = list({ch.display_name for ch in chords})
        lines.append(f"# {' - '.join(unique_chords)}")
        lines.append("")

        # === DIAGRAMA DE PROGRESIÓN ===
        lines.append("{comment: === PROGRESIÓN ===}")
        progression = self._summarize_progression(chords)
        lines.append(f"# {progression}")
        lines.append("")

        chordpro_text = "\n".join(lines)

        print(f"✓ ChordPro generado ({len(lines)} líneas)")

        return ChordProResult(
            chordpro_text=chordpro_text,
            key_name=tonal.key_name,
            tempo_bpm=tempo_bpm,
            notes=notes,
            chords=chords,
            lyrics=lyrics,
        )

    def save(self, result: ChordProResult, output_path: str | Path) -> Path:
        """Guarda el ChordPro en disco."""
        output_path = Path(output_path)
        output_path.write_text(result.chordpro_text, encoding="utf-8")
        print(f"✓ Guardado en: {output_path}")
        return output_path

    def _build_melody_line(self, notes: list[DetectedNote]) -> str:
        """
        Convierte la lista de notas a una línea de texto legible.
        Ej: C4(0.5s) E4(0.3s) G4(0.8s) ...
        """
        parts = []
        for n in notes:
            parts.append(f"{n.name}({n.duration:.1f}s)")
        return " ".join(parts)

    def _align_chords_with_lyrics(
        self,
        lyrics: str,
        chords: list[ChordSuggestion],
        duration: float,
    ) -> list[str]:
        """
        Alinea acordes con la letra usando timestamps.

        Analogía: es como un editor de karaoke que pone el acorde
        justo encima de la sílaba donde cae el cambio armónico.

        Sin timestamps de Whisper precisos, hacemos una distribución
        proporcional al tiempo.
        """
        words = lyrics.split()
        if not words:
            return ["(sin letra detectada)"]

        # Distribuir palabras en el tiempo proporcional
        word_duration = duration / len(words)
        lines_out = []
        current_line_chords = ""
        current_line_words = ""
        current_chord_idx = 0

        for i, word in enumerate(words):
            word_time = i * word_duration

            # ¿Hay un cambio de acorde aquí?
            if current_chord_idx < len(chords):
                chord = chords[current_chord_idx]
                if word_time >= chord.start_time:
                    current_line_chords += f"[{chord.display_name}]"
                    current_line_words += " " * max(0, len(f"[{chord.display_name}]") - 1)
                    current_chord_idx += 1

            current_line_words += word + " "
            current_line_chords += " " * (len(word) + 1)

            # Nueva línea cada ~8 palabras
            if (i + 1) % 8 == 0:
                lines_out.append(current_line_chords.rstrip())
                lines_out.append(current_line_words.rstrip())
                lines_out.append("")
                current_line_chords = ""
                current_line_words = ""

        # Última línea
        if current_line_words.strip():
            lines_out.append(current_line_chords.rstrip())
            lines_out.append(current_line_words.rstrip())

        return lines_out

    def _build_chord_only_lines(
        self, chords: list[ChordSuggestion]
    ) -> list[str]:
        """
        Genera líneas ChordPro sin letra, mostrando acordes con
        indicaciones de tiempo.

        Formato:
          [Am]          [F]           [C]           [G]
          (0.0s–2.0s)   (2.0s–4.0s)   (4.0s–6.0s)  ...
        """
        chord_row = ""
        time_row = ""
        lines = []

        for i, ch in enumerate(chords):
            ch_tag = f"[{ch.display_name}]"
            time_tag = f"({ch.start_time:.1f}s)"
            col_width = max(len(ch_tag), len(time_tag)) + 2

            chord_row += ch_tag.ljust(col_width)
            time_row += time_tag.ljust(col_width)

            # Nueva línea cada 4 acordes
            if (i + 1) % 4 == 0:
                lines.append(chord_row.rstrip())
                lines.append(time_row.rstrip())
                lines.append("")
                chord_row = ""
                time_row = ""

        if chord_row.strip():
            lines.append(chord_row.rstrip())
            lines.append(time_row.rstrip())

        return lines

    def _summarize_progression(self, chords: list[ChordSuggestion]) -> str:
        """
        Devuelve la progresión de acordes únicos por grado.
        Ej: I - VI - IV - V (la "four chord song" de Axis of Awesome)
        """
        seen = []
        for ch in chords:
            if not seen or ch.degree != seen[-1]:
                seen.append(ch.degree)
        return " - ".join(seen)
