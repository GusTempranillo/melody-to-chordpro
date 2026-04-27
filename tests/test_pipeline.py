"""
test_pipeline.py
================
Prueba de integración completa usando audio sintético.

En vez de necesitar un MP3 real, generamos una melodía sintética
de prueba: una escala de La menor (Am) con frecuencias puras.

Analogía: es como probar un detector de metales usando monedas
conocidas antes de llevarlo al campo. Si lo conocido funciona,
el desconocido también debería.

Escala de prueba:
  Am:  A4(440Hz) - B4(494Hz) - C5(523Hz) - D5(587Hz)
       E5(659Hz) - F5(698Hz) - G5(784Hz) - A5(880Hz)
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# Aseguramos que el proyecto raíz esté en el path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.modules.audio_loader import AudioLoader
from app.modules.pitch_detector import PitchDetector
from app.modules.tonal_analyzer import TonalAnalyzer
from app.modules.harmonizer import Harmonizer
from app.modules.chordpro_builder import ChordProBuilder


def generate_test_melody(output_path: str = "/tmp/test_melody.wav") -> str:
    """
    Genera un WAV sintético: escala de La menor ascendente y descendente.

    Cada nota dura 0.5 segundos, con un pequeño fade in/out
    para evitar clicks (artefactos de corte abrupto).
    """
    sr = 22050
    note_duration = 0.6   # segundos por nota
    amplitude = 0.8

    # Escala de La menor: A4 B4 C5 D5 E5 F5 G5 A5 (ascendente + descendente)
    # Frecuencias en Hz
    am_scale_hz = [
        440.0,  # A4
        493.9,  # B4
        523.3,  # C5
        587.3,  # D5
        659.3,  # E5
        698.5,  # F5
        784.0,  # G5
        880.0,  # A5
        784.0,  # G5 (descendente)
        698.5,  # F5
        659.3,  # E5
        587.3,  # D5
        523.3,  # C5
        493.9,  # B4
        440.0,  # A4
    ]

    samples_per_note = int(sr * note_duration)
    t = np.linspace(0, note_duration, samples_per_note)

    audio_chunks = []
    for freq in am_scale_hz:
        # Onda sinusoidal pura + segundo armónico (suena más a "voz")
        chunk = amplitude * (
            0.7 * np.sin(2 * np.pi * freq * t) +
            0.3 * np.sin(2 * np.pi * freq * 2 * t)
        )

        # Fade in/out (10% de la nota)
        fade = int(samples_per_note * 0.1)
        chunk[:fade] *= np.linspace(0, 1, fade)
        chunk[-fade:] *= np.linspace(1, 0, fade)

        audio_chunks.append(chunk)

    # Pequeño silencio entre notas (50ms)
    silence = np.zeros(int(sr * 0.05))
    full_audio = np.concatenate(
        [val for pair in zip(audio_chunks, [silence] * len(audio_chunks))
         for val in pair]
    )

    sf.write(output_path, full_audio, sr)
    duration = len(full_audio) / sr
    print(f"✓ Melodía de prueba generada: {output_path} ({duration:.1f}s)")
    return output_path


def test_audio_loader(wav_path: str):
    print("\n" + "─" * 40)
    print("TEST 1: AudioLoader")
    print("─" * 40)
    loader = AudioLoader()
    audio = loader.load(wav_path)
    assert audio["y"] is not None
    assert audio["sr"] == 22050
    assert audio["duration"] > 0
    print(f"  ✅ Cargado OK — {audio['duration']:.2f}s")
    return audio


def test_pitch_detector(audio: dict):
    print("\n" + "─" * 40)
    print("TEST 2: PitchDetector")
    print("─" * 40)
    detector = PitchDetector(confidence_threshold=0.5, min_note_duration=0.05)
    notes = detector.detect(audio["y"], audio["sr"])
    assert len(notes) > 0, "No se detectaron notas"
    detector.print_notes(notes)
    print(f"  ✅ {len(notes)} notas detectadas")
    return notes


def test_tonal_analyzer(notes):
    print("\n" + "─" * 40)
    print("TEST 3: TonalAnalyzer")
    print("─" * 40)
    analyzer = TonalAnalyzer()
    result = analyzer.analyze(notes)
    assert result.key_name is not None
    print(f"  ✅ Tonalidad: {result.key_name} (conf: {result.confidence:.0%})")
    return result


def test_harmonizer(notes, tonal, duration):
    print("\n" + "─" * 40)
    print("TEST 4: Harmonizer")
    print("─" * 40)
    harmonizer = Harmonizer(window_size=1.5)
    chords = harmonizer.suggest_chords(notes, tonal, duration)
    assert len(chords) > 0
    print(f"  ✅ {len(chords)} acordes sugeridos")
    return chords


def test_chordpro_builder(notes, tonal, chords, duration):
    print("\n" + "─" * 40)
    print("TEST 5: ChordProBuilder")
    print("─" * 40)
    builder = ChordProBuilder()
    result = builder.build(
        notes=notes,
        tonal=tonal,
        chords=chords,
        duration=duration,
        lyrics=None,
        title="Test Am Scale",
        artist="Synthetic Voice",
    )
    output_path = Path("/tmp/test_output.cho")
    builder.save(result, output_path)

    print("\n--- CHORDPRO GENERADO ---")
    print(result.chordpro_text[:800])
    if len(result.chordpro_text) > 800:
        print("... (truncado)")
    print("─" * 40)
    print(f"  ✅ ChordPro guardado en: {output_path}")
    return result


def main():
    print("=" * 50)
    print("🧪 TEST DE INTEGRACIÓN — Melody to ChordPro")
    print("=" * 50)

    # Generar audio de prueba
    wav_path = generate_test_melody()

    try:
        audio   = test_audio_loader(wav_path)
        notes   = test_pitch_detector(audio)
        tonal   = test_tonal_analyzer(notes)
        chords  = test_harmonizer(notes, tonal, audio["duration"])
        result  = test_chordpro_builder(notes, tonal, chords, audio["duration"])

        print("\n" + "=" * 50)
        print("✅ TODOS LOS TESTS PASARON")
        print(f"   Tonalidad: {result.key_name}")
        print(f"   Acordes únicos: {list({ch.display_name for ch in result.chords})}")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
