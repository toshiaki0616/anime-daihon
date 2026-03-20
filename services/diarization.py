from __future__ import annotations

from dataclasses import dataclass
from os import getenv
from pathlib import Path


class DiarizationError(Exception):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str


DEFAULT_DIARIZATION_MODEL = getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
DEFAULT_AUTH_TOKEN = getenv("HF_TOKEN") or getenv("HUGGINGFACE_TOKEN")


def diarize_wav(
    wav_path: str,
    model_name: str = DEFAULT_DIARIZATION_MODEL,
    auth_token: str | None = DEFAULT_AUTH_TOKEN,
) -> list[DiarizationSegment]:
    source = Path(wav_path)
    if not source.exists() or not source.is_file():
        raise DiarizationError("話者の分割に失敗しました")

    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError as exc:
        raise DiarizationError("話者の分割に失敗しました") from exc

    try:
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=auth_token)
        diarization = pipeline(str(source))
    except Exception as exc:  # noqa: BLE001
        raise DiarizationError("話者の分割に失敗しました") from exc

    segments: list[DiarizationSegment] = []
    try:
        iterator = diarization.itertracks(yield_label=True)
    except Exception as exc:  # noqa: BLE001
        raise DiarizationError("話者の分割に失敗しました") from exc

    for segment, _, speaker in iterator:
        start = float(getattr(segment, "start", 0.0))
        end = float(getattr(segment, "end", 0.0))
        if end <= start:
            continue
        segments.append(DiarizationSegment(start=start, end=end, speaker=str(speaker)))

    if not segments:
        raise DiarizationError("話者の分割に失敗しました")
    return segments
