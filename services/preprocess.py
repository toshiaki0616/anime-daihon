from __future__ import annotations

import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".mp4", ".wav", ".mp3"}


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
    source = Path(file_path)
    if not source.exists() or not source.is_file():
        raise MediaPreprocessError("ファイルを読み込めませんでした")

    extension = source.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise MediaPreprocessError("ファイルを読み込めませんでした")

    output_dir = Path(data_dir) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_wav_path = _ensure_wav(source, output_dir)
    final_wav_path = _slice_range(base_wav_path, range_start, range_end, output_dir)

    return PreprocessResult(
        source_path=str(source),
        wav_path=str(final_wav_path),
        range_start=range_start,
        range_end=range_end,
    )


def _ensure_wav(source: Path, output_dir: Path) -> Path:
    if source.suffix.lower() == ".wav":
        return source

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise MediaPreprocessError("音声の取り出しに失敗しました")

    output_path = output_dir / f"{source.stem}_{uuid.uuid4().hex[:8]}.wav"
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(output_path),
    ]
    _run_ffmpeg(command, "音声の取り出しに失敗しました")
    return output_path


def _slice_range(source_wav: Path, range_start: str, range_end: str, output_dir: Path) -> Path:
    if not range_start and not range_end:
        return source_wav

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise MediaPreprocessError("音声の取り出しに失敗しました")

    output_path = output_dir / f"{source_wav.stem}_slice_{uuid.uuid4().hex[:8]}.wav"
    command = [ffmpeg_path, "-y"]
    if range_start:
        command.extend(["-ss", range_start])
    command.extend(["-i", str(source_wav)])
    if range_end:
        command.extend(["-to", range_end])
    command.extend(
        [
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(output_path),
        ]
    )
    _run_ffmpeg(command, "音声の取り出しに失敗しました")
    return output_path


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
