"""
app.py — Gradio UI for melody-to-chordpro
Hugging Face Spaces entry point.
"""

import os
import tempfile
from pathlib import Path

import gradio as gr

from app.pipeline import MelodyPipeline, PipelineConfig


def process_audio(
    audio_path: str,
    title: str,
    artist: str,
    window_size: float,
    confidence: float,
) -> tuple[str, str, str]:
    """Run the pipeline and return (chordpro_text, summary, download_path)."""
    if not audio_path:
        return "", "⚠️ Sube un archivo de audio para comenzar.", None

    title = title.strip() or Path(audio_path).stem
    artist = artist.strip()

    config = PipelineConfig(
        chord_window_size=window_size,
        confidence_threshold=confidence,
        use_whisper=False,
    )
    pipeline = MelodyPipeline(config)

    try:
        result = pipeline.run(audio_path, title=title, artist=artist)
    except ValueError as e:
        return "", f"❌ Error: {e}", None

    summary_lines = [
        f"**Tonalidad:** {result.key_name}",
        f"**Notas detectadas:** {len(result.notes)}",
        f"**Acordes sugeridos:** {len(result.chords)}",
        f"**Progresión:** {' → '.join(ch.display_name for ch in result.chords)}",
    ]

    # Write .cho file to a temp location for download
    tmp = tempfile.NamedTemporaryFile(
        suffix=".cho", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(result.chordpro_text)
    tmp.close()

    return result.chordpro_text, "\n".join(summary_lines), tmp.name


# ── UI layout ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Melody → ChordPro", theme=gr.themes.Soft()) as demo:
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
            with gr.Accordion("⚙️ Opciones avanzadas", open=False):
                window_slider = gr.Slider(
                    minimum=0.5,
                    maximum=4.0,
                    value=2.0,
                    step=0.5,
                    label="Ventana de acordes (segundos)",
                    info="Más bajo = más cambios de acorde. Más alto = progresión más simple.",
                )
                confidence_slider = gr.Slider(
                    minimum=0.3,
                    maximum=0.9,
                    value=0.65,
                    step=0.05,
                    label="Umbral de confianza del pitch",
                    info="Más bajo = más notas detectadas (pero más ruido). Más alto = solo notas claras.",
                )
            run_btn = gr.Button("▶ Analizar", variant="primary", size="lg")

        with gr.Column(scale=2):
            summary_output = gr.Markdown(label="Resumen")
            chordpro_output = gr.Textbox(
                label="📄 Resultado ChordPro",
                lines=20,
                max_lines=40,
                show_copy_button=True,
            )
            download_output = gr.File(label="⬇ Descargar .cho")

    run_btn.click(
        fn=process_audio,
        inputs=[audio_input, title_input, artist_input, window_slider, confidence_slider],
        outputs=[chordpro_output, summary_output, download_output],
    )

    gr.Markdown(
        """
        ---
        ### Cómo interpretar el resultado
        - Los **acordes** aparecen entre corchetes: `[Am]`, `[G]`, `[F]`
        - Los **tiempos** indican cuándo cambia cada acorde
        - Sin Whisper instalado, la letra aparece como marcadores de posición
        - Para añadir transcripción automática de letra, activa Whisper en el servidor

        ### Formato ChordPro
        Compatible con [OpenSong](https://opensong.org), [Chordpro.org](https://www.chordpro.org),
        [ChordU](https://chordu.com) y cualquier editor de hojas de canciones.

        **Proyecto:** [GusTempranillo/melody-to-chordpro](https://github.com/GusTempranillo/melody-to-chordpro)
        """
    )


if __name__ == "__main__":
    demo.launch()
