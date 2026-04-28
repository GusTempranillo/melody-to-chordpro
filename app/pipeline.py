"""
pipeline.py
===========
El director de orquesta: llama a cada módulo en el orden correcto
y pasa el resultado de uno al siguiente.

Analogía: es como una cadena de montaje donde cada operario
hace una sola cosa bien y pasa la pieza al siguiente.
Este módulo es el capataz que supervisa todo el proceso.
"""

from pathlib import Path
from dataclasses import dataclass

# Caché de modelos Whisper para no recargarlos entre peticiones
_WHISPER_CACHE: dict = {}

from app.modules.audio_loader import AudioLoader
from app.modules.pitch_detector import PitchDetector, Note
from app.modules.tonal_analyzer import TonalAnalyzer, TonalResult
from app.modules.harmonizer import Harmonizer, ChordSuggestion
from app.modules.chordpro_builder import ChordProBuilder, ChordProResult


@dataclass
class PipelineConfig:
    """Configuración global del pipeline."""
    # Pitch
    confidence_threshold: float = 0.65
    min_note_duration: float = 0.08
    # Armonización
    chord_window_size: float = 2.0
    # Whisper (opcional)
    use_whisper: bool = False
    whisper_model: str = "small"   # tiny | base | small | medium | large
    whisper_language: str | None = "es"  # None = autodetectar
    # Output
    output_dir: str = "outputs"


class MelodyPipeline:
    """
    Pipeline completo: MP3 → ChordPro
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

        # Inicializar todos los módulos
        self.loader = AudioLoader()
        self.pitch_detector = PitchDetector(
            confidence_threshold=self.config.confidence_threshold,
            min_note_duration=self.config.min_note_duration,
        )
        self.tonal_analyzer = TonalAnalyzer()
        self.harmonizer = Harmonizer(
            window_size=self.config.chord_window_size
        )
        self.builder = ChordProBuilder()

        # Whisper se carga solo si está disponible y configurado
        self._whisper_model = None

    def run(
        self,
        audio_path: str | Path,
        title: str = "Mi Canción",
        artist: str = "",
    ) -> ChordProResult:
        """
        Ejecuta el pipeline completo.

        Args:
            audio_path: ruta al archivo MP3 (o WAV, OGG, FLAC)
            title:      título para el ChordPro
            artist:     intérprete para el ChordPro

        Returns:
            ChordProResult con todo el análisis y el texto ChordPro
        """
        print("\n" + "═" * 50)
        print(f"🎤 MELODY TO CHORDPRO")
        print(f"   Archivo: {Path(audio_path).name}")
        print("═" * 50 + "\n")

        # ── Estación 1: Cargar audio ────────────────────────
        print("[ 1/5 ] Cargando audio...")
        audio = self.loader.load(audio_path)

        # ── Estación 2: Transcribir letra (opcional) ────────
        lyrics = None
        lyrics_segments = None
        if self.config.use_whisper:
            print("\n[ 2/5 ] Transcribiendo letra con Whisper...")
            lyrics, lyrics_segments = self._transcribe(audio_path)
        else:
            print("\n[ 2/5 ] Whisper desactivado — omitiendo transcripción de letra")

        # ── Estación 3: Detectar pitch ──────────────────────
        print("\n[ 3/5 ] Detectando melodía (notas)...")
        notes = self.pitch_detector.detect(audio["y"], audio["sr"])

        if not notes:
            raise ValueError(
                "No se detectaron notas. Verifica que el audio contenga "
                "voz clara y sin mucho ruido de fondo."
            )

        self.pitch_detector.print_notes(notes)

        # ── Estación 4: Analizar tonalidad ──────────────────
        print("\n[ 4/5 ] Analizando tonalidad...")
        tonal = self.tonal_analyzer.analyze(notes)

        # ── Estación 5a: Sugerir acordes ────────────────────
        print("\n[ 5/5 ] Armonizando...")
        chords = self.harmonizer.suggest_chords(
            notes, tonal, audio["duration"]
        )

        # ── Estación 5b: Ensamblar ChordPro ─────────────────
        result = self.builder.build(
            notes=notes,
            tonal=tonal,
            chords=chords,
            duration=audio["duration"],
            lyrics=lyrics,
            lyrics_segments=lyrics_segments,
            title=title,
            artist=artist,
        )

        # Guardar resultado
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        stem = Path(audio_path).stem
        output_path = output_dir / f"{stem}.cho"
        self.builder.save(result, output_path)

        print("\n" + "═" * 50)
        print("✅ PIPELINE COMPLETADO")
        print(f"   Resultado: {output_path}")
        print("═" * 50 + "\n")

        return result

    def _transcribe(self, audio_path: str | Path) -> tuple[str | None, list | None]:
        """
        Transcribe la letra con faster-whisper (4× más rápido que openai-whisper).
        Devuelve (texto_completo, segmentos_con_timestamps_por_palabra).
        """
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError:
            print("  ⚠️  faster-whisper no instalado. Instala con:")
            print("       pip install faster-whisper")
            return None, None

        model_name = self.config.whisper_model
        if model_name not in _WHISPER_CACHE:
            print(f"  Cargando modelo '{model_name}' (primera vez, se cachea)...")
            _WHISPER_CACHE[model_name] = WhisperModel(
                model_name, device="cpu", compute_type="int8"
            )
        else:
            print(f"  Modelo '{model_name}' ya cargado (caché).")

        model = _WHISPER_CACHE[model_name]
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=self.config.whisper_language,
            task="transcribe",
            word_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.9,
        )

        segments = []
        full_text = ""
        for seg in segments_iter:
            words = []
            if seg.words:
                for w in seg.words:
                    words.append({"word": w.word, "start": w.start, "end": w.end})
            segments.append({"start": seg.start, "end": seg.end,
                              "text": seg.text, "words": words})
            full_text += seg.text

        lyrics = full_text.strip()
        print(f"  Segmentos: {len(segments)} | Letra: {lyrics[:100]}{'...' if len(lyrics) > 100 else ''}")
        return lyrics, segments
