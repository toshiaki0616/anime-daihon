# Subtitle Library UI - Anime Whisper Step 1

This project now uses **faster-whisper + Anime Whisper** instead of openai-whisper for transcription.

## What changed

- Switched transcription to aster-whisper
- Default model directory is models/anime-whisper-ct2/
- CPU / int8 execution is now the default path
- Old saved values like ase, small, and medium are mapped to Anime Whisper for compatibility
- initial_prompt is kept in the UI for now, but Anime Whisper does not use it

## Install dependencies

`powershell
cd C:\work\codex\project
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
`

## Prepare the Anime Whisper model

Run this once to download and convert the model into CTranslate2 format:

`powershell
cd C:\work\codex\project
.venv\Scripts\Activate.ps1
.\scripts\setup_anime_whisper.ps1
`

The converted model will be stored here:

`	ext
models/
  anime-whisper-ct2/
`

If the model has not been prepared yet, the app will show a clear error message telling you to run scripts\setup_anime_whisper.ps1.

## Run the app

`powershell
cd C:\work\codex\project
.venv\Scripts\Activate.ps1
python app.py
`
