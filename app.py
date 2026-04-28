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
from app.modules.chordpro_builder import ChordProResult


# Paleta de colores para los acordes (cíclica)
CHORD_COLORS = [
    "#4f46e5", "#0891b2", "#059669", "#d97706",
    "#dc2626", "#7c3aed", "#db2777", "#0284c7",
]


def chord_pill(name: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;background:{color};color:#fff;'
        f'font-weight:700;font-size:0.95em;padding:4px 13px;border-radius:20px;'
        f'margin:2px 4px;letter-spacing:0.03em;">{name}</span>'
    )


def build_visual_html(result: ChordProResult, title: str, artist: str) -> str:
    """Genera una hoja de canción visual en HTML."""

    # ── Mapa de color por acorde único ─────────────────────────────────────
    unique_chords = list(dict.fromkeys(ch.display_name for ch in result.chords))
    color_map = {ch: CHORD_COLORS[i % len(CHORD_COLORS)] for i, ch in enumerate(unique_chords)}

    # ── Cabecera ────────────────────────────────────────────────────────────
    artist_line = f'<p style="color:#6b7280;font-size:1.1em;margin:4px 0 0;">{artist}</p>' if artist else ""
    key_parts = result.key_name.split()
    key_display = f"{key_parts[0]} {'menor' if 'minor' in result.key_name else 'mayor'}" if len(key_parts) >= 2 else result.key_name

    header = f"""
    <div style="text-align:center;padding:24px 20px 16px;border-bottom:2px solid #e5e7eb;margin-bottom:20px;">
      <h1 style="font-family:Georgia,serif;font-size:2em;margin:0;color:#111827;">{title}</h1>
      {artist_line}
      <div style="margin-top:12px;display:flex;justify-content:center;gap:10px;flex-wrap:wrap;">
        <span style="background:#f3f4f6;border:1px solid #d1d5db;border-radius:8px;padding:5px 14px;font-size:0.9em;color:#374151;">
          🎼 Tonalidad: <strong>{key_display}</strong>
        </span>
        <span style="background:#f3f4f6;border:1px solid #d1d5db;border-radius:8px;padding:5px 14px;font-size:0.9em;color:#374151;">
          🎵 {len(result.notes)} notas detectadas
        </span>
        <span style="background:#f3f4f6;border:1px solid #d1d5db;border-radius:8px;padding:5px 14px;font-size:0.9em;color:#374151;">
          🎸 {len(unique_chords)} acordes
        </span>
      </div>
    </div>
    """

    # ── Progresión de acordes ───────────────────────────────────────────────
    prev = None
    prog_pills = []
    for ch in result.chords:
        if ch.display_name != prev:
            prog_pills.append(chord_pill(ch.display_name, color_map[ch.display_name]))
            if prev is not None:
                prog_pills.insert(-1, '<span style="color:#9ca3af;font-size:1.2em;margin:0 2px;">→</span>')
            prev = ch.display_name

    # Reconstruir con flechas intercaladas
    prog_pills = []
    prev = None
    for ch in result.chords:
        if ch.display_name != prev:
            if prev is not None:
                prog_pills.append('<span style="color:#9ca3af;font-size:1.2em;margin:0 2px;">→</span>')
            prog_pills.append(chord_pill(ch.display_name, color_map[ch.display_name]))
            prev = ch.display_name

    progression_html = f"""
    <div style="background:#f9fafb;border-radius:12px;padding:16px 20px;margin-bottom:24px;">
      <p style="margin:0 0 10px;font-size:0.8em;font-weight:600;text-transform:uppercase;
                letter-spacing:0.08em;color:#6b7280;">Progresión de acordes</p>
      <div style="display:flex;flex-wrap:wrap;align-items:center;gap:4px;">
        {''.join(prog_pills)}
      </div>
    </div>
    """

    # ── Leyenda de acordes ──────────────────────────────────────────────────
    legend_items = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
        f'{chord_pill(ch, color_map[ch])}'
        f'<span style="color:#6b7280;font-size:0.85em;">grado {next(c.degree for c in result.chords if c.display_name == ch)}</span>'
        f'</div>'
        for ch in unique_chords
    )
    legend_html = f"""
    <div style="background:#f9fafb;border-radius:12px;padding:16px 20px;margin-bottom:24px;">
      <p style="margin:0 0 10px;font-size:0.8em;font-weight:600;text-transform:uppercase;
                letter-spacing:0.08em;color:#6b7280;">Referencia de acordes</p>
      {legend_items}
    </div>
    """

    # ── Letra con acordes ───────────────────────────────────────────────────
    if result.lyrics:
        lyrics_html = _render_lyrics_with_chords(result, color_map)
    else:
        lyrics_html = _render_chord_timeline(result, color_map)

    # ── Ensamblado final ────────────────────────────────────────────────────
    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                max-width:680px;margin:0 auto;padding:10px;">
      {header}
      {progression_html}
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;">
        <div>{legend_html}</div>
        <div>{_render_stats(result)}</div>
      </div>
      {lyrics_html}
      <p style="text-align:center;color:#d1d5db;font-size:0.75em;margin-top:32px;">
        generado con melody-to-chordpro
      </p>
    </div>
    """


def _render_lyrics_with_chords(result: ChordProResult, color_map: dict) -> str:
    """Renderiza la letra con los acordes encima de cada bloque."""

    # Construir lista de (palabra, tiempo_inicio) desde timestamps reales de Whisper
    word_times: list[tuple[str, float]] = []
    if result.lyrics_segments:
        for seg in result.lyrics_segments:
            for w in seg.get("words", []):
                text = w.get("word", "").strip()
                if text:
                    word_times.append((text, float(w.get("start", 0.0))))

    # Fallback: distribución proporcional si no hay timestamps
    if not word_times and result.lyrics:
        words = result.lyrics.split()
        duration = result.chords[-1].end_time if result.chords else 1
        step = duration / max(len(words), 1)
        word_times = [(w, i * step) for i, w in enumerate(words)]

    if not word_times:
        return ""

    def chord_at(t):
        for ch in result.chords:
            if ch.start_time <= t < ch.end_time:
                return ch
        return result.chords[-1] if result.chords else None

    lines_out = []
    chunk = []
    current_chord = None

    for i, (word, t) in enumerate(word_times):
        ch = chord_at(t)
        ch_name = ch.display_name if ch else None
        is_new = ch_name != current_chord
        chunk.append((word, ch_name if is_new else None))
        current_chord = ch_name

        if (i + 1) % 8 == 0:
            lines_out.append(chunk)
            chunk = []
    if chunk:
        lines_out.append(chunk)

    html_lines = []
    for line in lines_out:
        chord_row = ""
        word_row = ""
        for word, ch_name in line:
            cell_w = max(len(word), len(ch_name) if ch_name else 0) + 2
            if ch_name:
                color = color_map.get(ch_name, "#4f46e5")
                chord_row += (
                    f'<span style="color:{color};font-weight:700;'
                    f'min-width:{cell_w}ch;display:inline-block;">{ch_name}</span>'
                )
            else:
                chord_row += f'<span style="min-width:{cell_w}ch;display:inline-block;"> </span>'
            word_row += f'<span style="min-width:{cell_w}ch;display:inline-block;color:#1f2937;">{word} </span>'

        html_lines.append(f"""
        <div style="margin-bottom:18px;">
          <div style="font-family:monospace;font-size:1em;line-height:1.6;">{chord_row}</div>
          <div style="font-size:1.05em;line-height:1.6;color:#1f2937;">{word_row}</div>
        </div>
        """)

    return f"""
    <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:20px 24px;margin-bottom:24px;">
      <p style="margin:0 0 16px;font-size:0.8em;font-weight:600;text-transform:uppercase;
                letter-spacing:0.08em;color:#6b7280;">Letra y acordes</p>
      {''.join(html_lines)}
    </div>
    """


def _render_chord_timeline(result: ChordProResult, color_map: dict) -> str:
    """Cuando no hay letra: timeline visual de los acordes."""
    total = result.chords[-1].end_time if result.chords else 1
    bars = ""
    for ch in result.chords:
        pct = round((ch.end_time - ch.start_time) / total * 100, 1)
        color = color_map.get(ch.display_name, "#4f46e5")
        bars += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
          {chord_pill(ch.display_name, color)}
          <div style="flex:1;background:#f3f4f6;border-radius:4px;height:10px;">
            <div style="width:{pct}%;background:{color};height:100%;border-radius:4px;opacity:0.7;"></div>
          </div>
          <span style="color:#6b7280;font-size:0.8em;min-width:80px;">
            {ch.start_time:.1f}s – {ch.end_time:.1f}s
          </span>
        </div>
        """

    return f"""
    <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:20px 24px;margin-bottom:24px;">
      <p style="margin:0 0 16px;font-size:0.8em;font-weight:600;text-transform:uppercase;
                letter-spacing:0.08em;color:#6b7280;">Timeline de acordes</p>
      <p style="color:#9ca3af;font-size:0.85em;margin:0 0 16px;">
        Activa Whisper para ver la letra alineada con los acordes.
      </p>
      {bars}
    </div>
    """


def _render_stats(result: ChordProResult) -> str:
    """Panel de estadísticas técnicas."""
    key_parts = result.key_name.split()
    key_display = f"{key_parts[0]} {'menor' if 'minor' in result.key_name else 'mayor'}" if len(key_parts) >= 2 else result.key_name
    duration = result.chords[-1].end_time if result.chords else 0
    rows = [
        ("Tonalidad", key_display),
        ("Duración", f"{duration:.1f} s"),
        ("Notas", str(len(result.notes))),
        ("Acordes distintos", str(len({c.display_name for c in result.chords}))),
        ("Letra", "sí (Whisper)" if result.lyrics else "no"),
    ]
    table_rows = "".join(
        f'<tr><td style="color:#6b7280;padding:5px 10px 5px 0;font-size:0.9em;">{k}</td>'
        f'<td style="font-weight:600;font-size:0.9em;color:#111827;">{v}</td></tr>'
        for k, v in rows
    )
    return f"""
    <div style="background:#f9fafb;border-radius:12px;padding:16px 20px;margin-bottom:24px;height:100%;">
      <p style="margin:0 0 10px;font-size:0.8em;font-weight:600;text-transform:uppercase;
                letter-spacing:0.08em;color:#6b7280;">Análisis</p>
      <table style="border-collapse:collapse;width:100%;">{table_rows}</table>
    </div>
    """


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
    if not audio_path:
        empty_html = '<p style="color:#9ca3af;text-align:center;padding:40px;">Sube un audio y pulsa Analizar.</p>'
        return empty_html, "", None

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
        err_html = f'<p style="color:#dc2626;padding:20px;">❌ Error: {e}</p>'
        return err_html, "", None

    visual_html = build_visual_html(result, title, artist)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".cho", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(result.chordpro_text)
    tmp.close()

    return visual_html, result.chordpro_text, tmp.name


# ── UI layout ────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1100px !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Melody → ChordPro", css=CSS) as demo:
    gr.Markdown(
        """
        # 🎵 Melody → ChordPro
        Sube una grabación de voz o melodía y obtén los acordes y la letra alineada.
        > **Funciona mejor con:** voz limpia sin música de fondo.
        """
    )

    with gr.Row():
        # ── Panel izquierdo: controles ───────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            audio_input = gr.Audio(
                label="Audio (MP3 / WAV / OGG / FLAC)",
                type="filepath",
                sources=["upload", "microphone"],
            )
            title_input = gr.Textbox(
                label="Título",
                placeholder="Mi Canción",
                max_lines=1,
            )
            artist_input = gr.Textbox(
                label="Artista",
                placeholder="(opcional)",
                max_lines=1,
            )

            with gr.Accordion("🎙️ Letra con Whisper", open=True):
                use_whisper_check = gr.Checkbox(
                    label="Transcribir letra automáticamente",
                    value=True,
                )
                whisper_model_radio = gr.Radio(
                    choices=["tiny", "base", "small", "medium"],
                    value="small",
                    label="Modelo",
                    info="tiny=~15s | base=~30s | small=~1min | medium=~4min",
                )
                whisper_lang_dropdown = gr.Dropdown(
                    choices=["auto", "es", "en", "fr", "pt", "de", "it"],
                    value="es",
                    label="Idioma",
                )

            with gr.Accordion("⚙️ Ajustes avanzados", open=False):
                window_slider = gr.Slider(
                    minimum=0.5, maximum=4.0, value=2.0, step=0.5,
                    label="Ventana de acordes (s)",
                    info="Menor = más cambios de acorde",
                )
                confidence_slider = gr.Slider(
                    minimum=0.3, maximum=0.9, value=0.65, step=0.05,
                    label="Confianza mínima de pitch",
                )

            run_btn = gr.Button("▶  Analizar", variant="primary", size="lg")

        # ── Panel derecho: resultados ────────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("🎼 Vista de canción"):
                    visual_output = gr.HTML(
                        value='<p style="color:#9ca3af;text-align:center;padding:60px 20px;">'
                              'Sube un audio y pulsa <strong>Analizar</strong>.</p>'
                    )

                with gr.Tab("📄 ChordPro (texto)"):
                    chordpro_output = gr.Textbox(
                        label="",
                        lines=25,
                        max_lines=50,
                        placeholder="El texto ChordPro aparecerá aquí...",
                    )

            download_output = gr.File(label="⬇ Descargar archivo .cho")

    run_btn.click(
        fn=process_audio,
        inputs=[
            audio_input, title_input, artist_input,
            use_whisper_check, whisper_model_radio, whisper_lang_dropdown,
            window_slider, confidence_slider,
        ],
        outputs=[visual_output, chordpro_output, download_output],
    )


if __name__ == "__main__":
    demo.launch()
