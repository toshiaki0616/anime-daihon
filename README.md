# Subtitle Library UI

This project uses **faster-whisper + Anime Whisper** for transcription and now includes a dedicated Phase 1 preprocessing stage before later ASR and diarization steps.

## Phase 1 overview

Phase 1 now does the following before transcription:

- normalize the selected media into mono 16kHz wav
- run Silero VAD speech segmentation on the normalized wav
- save debug output to `data/debug/debug_vad_segments.json`
- keep the existing UI mostly unchanged while moving media logic out of `app.py`

This phase does not redesign ASR, diarization, or speaker naming yet.

## Install dependencies

```powershell
cd C:\work\codex\project
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dependencies

- `ffmpeg`
  - required for mp4/mp3/wav normalization and clip extraction
  - must be available on `PATH`
- `silero-vad`
  - used for Phase 1 VAD-based speech segmentation
- Anime Whisper CTranslate2 model
  - used by the existing transcription stage after preprocessing

## Prepare the Anime Whisper model

Run this once to download and convert the model into CTranslate2 format:

```powershell
cd C:\work\codex\project
.\.venv\Scripts\Activate.ps1
.\scripts\setup_anime_whisper.ps1
```

The converted model will be stored here:

```text
models/
  anime-whisper-ct2/
```

If the model has not been prepared yet, the app will show a clear error message telling you to run `scripts\setup_anime_whisper.ps1`.

## Debug output

Each preprocessing run writes:

```text
data/debug/debug_vad_segments.json
```

The debug file includes:

- source file path
- normalized wav path
- processed range
- final VAD segments
- fallback flag if VAD had to fall back to a single full-range segment

## Run the app

```powershell
cd C:\work\codex\project
.\.venv\Scripts\Activate.ps1
python app.py
```
