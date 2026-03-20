from __future__ import annotations

from dataclasses import dataclass
from os import getenv
from pathlib import Path
from typing import Any


class TranscriptionError(Exception):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


DEFAULT_MODEL_NAME = getenv("WHISPER_MODEL", "base")


def transcribe_wav(wav_path: str, model_name: str = DEFAULT_MODEL_NAME) -> list[TranscriptionSegment]:
    source = Path(wav_path)
    if not source.exists() or not source.is_file():
        raise TranscriptionError("字幕の作成に失敗しました")

    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise TranscriptionError("字幕の作成に失敗しました") from exc

    try:
        model = whisper.load_model(model_name)
        result: dict[str, Any] = model.transcribe(str(source), verbose=False, language="ja")
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError("字幕の作成に失敗しました") from exc

    segments = []
    for item in result.get("segments", []):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            TranscriptionSegment(
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                text=text,
            )
        )

    if not segments:
        raise TranscriptionError("字幕の作成に失敗しました")
    return segments
