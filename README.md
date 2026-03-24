# Subtitle Library UI

This project uses **faster-whisper + Anime Whisper** for transcription and now includes dedicated preprocessing and segmented ASR stages before later diarization work.

## Phase 1 and Phase 2 overview

The current pipeline now does the following:

- normalize the selected media into mono 16kHz wav
- run Silero VAD speech segmentation on the normalized wav
- run Anime Whisper per VAD speech segment instead of transcribing the whole file as one chunk
- build structured subtitle segments from segmented ASR results
- save debug output to `data/debug/debug_vad_segments.json`
- save ASR debug output to `data/debug/debug_asr_segments.json`
- keep the existing UI mostly unchanged while moving media logic out of `app.py`

This stage still does not redesign diarization, speaker clustering, or voiceprint behavior yet.

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
  - used for VAD-based speech segmentation
- Anime Whisper CTranslate2 model
  - used for per-segment ASR after preprocessing

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

Each subtitle generation run writes:

```text
data/debug/debug_vad_segments.json
data/debug/debug_asr_segments.json
```

The debug files include:

- source file path
- normalized wav path
- processed range
- final VAD segments
- fallback flag if VAD had to fall back to a single full-range segment
- raw ASR text per VAD segment
- output subtitle segments generated from segmented ASR

## Run the app

```powershell
cd C:\work\codex\project
.\.venv\Scripts\Activate.ps1
python app.py
```
