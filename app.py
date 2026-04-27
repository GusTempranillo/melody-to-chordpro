"""
app.py — Gradio UI for melody-to-chordpro
Hugging Face Spaces entry point.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import os
import tempfile
from pathlib import Path

# Garantizar que ffmpeg (instalado por winget) está en el PATH en Windows
_FFMPEG_BIN = (
    r"C:\Users\Usuario\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin"
)
if os.path.isdir(_FFMPEG_BIN) and _FFMPEG_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

import gradio as gr

from app.pipeline import MelodyPipeline, PipelineConfig


def process_audio(
    audio_path: str,
    title: str,
    artist: str,
    use_whisper: bool,
    whisper_model: str,
    whisper_language: str,
    window_size: float,
    confidence: float,
) -> tuple[str, str, str]:
    """Run the pipeline and return (chordpro_text, summary, download_path)."""
    if not audio_path:
        return "", "⚠️ Sube un archivo de audio para comenzar.", None

    title = title.strip() or Path(audio_path).stem
    artist = artist.strip()

    lang = None if whisper_language == "auto" else whisper_language

    config = PipelineConfig(
        chord_window_size=window_size,
        confidence_threshold=confidence,
        use_whisper=use_whisper,
        whisper_model=whisper_model,
        whisper_language=lang,
    )
    pipeline = MelodyPipeline(config)

    try:
        result = pipeline.run(audio_path, title=title, artist=artist)
    except Exception as e:
        return "", f"❌ Error: {e}", None

    lyrics_status = "con letra (Whisper)" if result.lyrics else "sin letra"
    summary_lines = [
        f"**Tonalidad:** {result.key_name}",
        f"**Notas detectadas:** {len(result.notes)}",
        f"**Acordes sugeridos:** {len(result.chords)}",
        f"**Progresión:** {' → '.join(ch.display_name for ch in result.chords)}",
        f"**Letra:** {lyrics_status}",
    ]

    tmp = tempfile.NamedTemporaryFile(
        suffix=".cho", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(result.chordpro_text)
    tmp.close()

    return result.chordpro_text, "\n".join(summary_lines), tmp.name


# ── UI layout ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Melody → ChordPro") as demo:
    gr.Markdown(
        """
        # 🎵 Melody → ChordPro
        Sube una grabación de voz o melodía y obtén los acordes en formato **ChordPro**
        listo para usar en OpenSong, ChordU, Ultimate Guitar y similares.

        > **Funciona mejor con:** voz limpia, una sola melodía, sin instrumentos de fondo.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="🎤 Audio (MP3 / WAV / OGG / FLAC)",
                type="filepath",
                sources=["upload", "microphone"],
            )
            title_input = gr.Textbox(
                label="Título de la canción",
                placeholder="Mi Canción",
                max_lines=1,
            )
            artist_input = gr.Textbox(
                label="Artista / Intérprete",
                placeholder="(opcional)",
                max_lines=1,
            )

            with gr.Accordion("🎙️ Transcripción de letra (Whisper)", open=True):
                use_whisper_check = gr.Checkbox(
                    label="Transcribir letra automáticamente",
                    value=False,
                )
                with gr.Row():
                    whisper_model_radio = gr.Radio(
                        choices=["tiny", "base", "small", "medium"],
                        value="small",
                        label="Modelo",
                        info="tiny=rápido, small=equilibrado, medium=preciso",
                    )
                    whisper_lang_dropdown = gr.Dropdown(
                        choices=["auto", "es", "en", "fr", "pt", "de", "it"],
                        value="es",
                        label="Idioma",
                    )

            with gr.Accordion("⚙️ Opciones avanzadas", open=False):
                window_slider = gr.Slider(
                    minimum=0.5,
                    maximum=4.0,
                    value=2.0,
                    step=0.5,
                    label="Ventana de acordes (segundos)",
                    info="Más bajo = más cambios. Más alto = progresión más simple.",
                )
                confidence_slider = gr.Slider(
                    minimum=0.3,
                    maximum=0.9,
                    value=0.65,
                    step=0.05,
                    label="Umbral de confianza del pitch",
                    info="Más bajo = más notas (y más ruido). Más alto = solo notas claras.",
                )

            run_btn = gr.Button("▶ Analizar", variant="primary", size="lg")

        with gr.Column(scale=2):
            summary_output = gr.Markdown(label="Resumen")
            chordpro_output = gr.Textbox(
                label="📄 Resultado ChordPro",
                lines=20,
                max_lines=40,
            )
            download_output = gr.File(label="⬇ Descargar .cho")

    run_btn.click(
        fn=process_audio,
        inputs=[
            audio_input, title_input, artist_input,
            use_whisper_check, whisper_model_radio, whisper_lang_dropdown,
            window_slider, confidence_slider,
        ],
        outputs=[chordpro_output, summary_output, download_output],
    )

    gr.Markdown(
        """
        ---
        ### Cómo interpretar el resultado
        - Los **acordes** aparecen entre corchetes: `[Am]`, `[G]`, `[F]`
        - Los **tiempos** indican cuándo cambia cada acorde
        - Con Whisper activo, la letra se alinea automáticamente con los acordes
        - La primera vez que uses Whisper descargará el modelo (~150MB para "small")

        ### Formato ChordPro
        Compatible con [OpenSong](https://opensong.org), [Chordpro.org](https://www.chordpro.org),
        [ChordU](https://chordu.com) y cualquier editor de hojas de canciones.

        **Proyecto:** [GusTempranillo/melody-to-chordpro](https://github.com/GusTempranillo/melody-to-chordpro)
        """
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
