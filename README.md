---
title: Melody to ChordPro
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
short_description: Convierte melodías de voz en archivos ChordPro con acordes
---

# 🎵 Melody → ChordPro

Sube una grabación de voz o melodía (MP3, WAV, OGG, FLAC) y obtén los acordes en formato **ChordPro** listo para usar en OpenSong, ChordU, Ultimate Guitar y similares.

## Cómo funciona

```
Audio → Carga → Detección de pitch (pyin) → Análisis tonal → Armonización → ChordPro
```

1. **AudioLoader** — normaliza y convierte a 22kHz mono
2. **PitchDetector** — detecta notas con el algoritmo pyin de librosa
3. **TonalAnalyzer** — determina la tonalidad con Krumhansl-Schmuckler
4. **Harmonizer** — propone acordes funcionales por ventanas de tiempo
5. **ChordProBuilder** — ensambla el documento `.cho` final

## Uso local

```bash
git clone https://github.com/GusTempranillo/melody-to-chordpro
cd melody-to-chordpro
pip install -r requirements.txt
python app.py
```

## Notas

- Funciona mejor con voz limpia sin instrumentos de fondo
- La transcripción de letra requiere `openai-whisper` (no incluido por tamaño)
- Armonía funcional occidental — no óptimo para flamenco o modos griegos
