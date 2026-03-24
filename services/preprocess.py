from __future__ import annotations

import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from .audio_segmentation import AudioSegmentationError, normalize_audio


CLIP_SOURCE_MISSING_ERROR = "\u97f3\u58f0\u30af\u30ea\u30c3\u30d7\u306e\u5143\u30d5\u30a1\u30a4\u30eb\u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093"
CLIP_RANGE_ERROR = "\u97f3\u58f0\u30af\u30ea\u30c3\u30d7\u306e\u6642\u9593\u7bc4\u56f2\u304c\u4e0d\u6b63\u3067\u3059"
CLIP_EXTRACT_ERROR = "\u97f3\u58f0\u30af\u30ea\u30c3\u30d7\u306e\u62bd\u51fa\u306b\u5931\u6557\u3057\u307e\u3057\u305f"


class MediaPreprocessError(Exception):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class PreprocessResult:
    source_path: str
    wav_path: str
    range_start: str
    range_end: str


def preprocess_media(
    file_path: str,
    range_start: str,
    range_end: str,
    data_dir: str | Path,
) -> PreprocessResult:
    try:
        result = normalize_audio(
            input_path=file_path,
            output_dir=Path(data_dir) / "processed",
            range_start=range_start,
            range_end=range_end,
        )
    except AudioSegmentationError as exc:
        raise MediaPreprocessError(exc.user_message) from exc

    return PreprocessResult(
        source_path=result.source_path,
        wav_path=result.normalized_wav_path,
        range_start=range_start,
        range_end=range_end,
    )


def extract_audio_clip(
    file_path: str,
    clip_start: float,
    clip_end: float,
    data_dir: str | Path,
) -> str:
    source = Path(file_path)
    if not source.exists() or not source.is_file():
        raise MediaPreprocessError(CLIP_SOURCE_MISSING_ERROR)
    if clip_end <= clip_start:
        raise MediaPreprocessError(CLIP_RANGE_ERROR)

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise MediaPreprocessError(CLIP_EXTRACT_ERROR)

    output_dir = Path(data_dir) / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source.stem}_clip_{uuid.uuid4().hex[:8]}.wav"
    command = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{clip_start:.3f}",
        "-to",
        f"{clip_end:.3f}",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(output_path),
    ]
    _run_ffmpeg(command, CLIP_EXTRACT_ERROR)
    return str(output_path)


def _run_ffmpeg(command: list[str], user_message: str) -> None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise MediaPreprocessError(user_message) from exc

    if completed.returncode != 0:
        raise MediaPreprocessError(user_message)
