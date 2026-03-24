from __future__ import annotations

import shutil
import subprocess
import uuid
import wave
from pathlib import Path

from models.state import PreprocessingResult, VadSegment


SUPPORTED_EXTENSIONS = {".mp4", ".mp3", ".wav"}
DEFAULT_MIN_SEGMENT_SECONDS = 1.0
DEFAULT_MERGE_GAP_SECONDS = 0.35
DEFAULT_MAX_SEGMENT_SECONDS = 8.5
DEFAULT_SPLIT_GAP_SECONDS = 0.15
FILE_READ_ERROR = "\u30d5\u30a1\u30a4\u30eb\u3092\u8aad\u307f\u8fbc\u3081\u307e\u305b\u3093\u3067\u3057\u305f"
INVALID_TIME_RANGE_ERROR = "\u7d42\u4e86\u6642\u523b\u306f\u958b\u59cb\u6642\u523b\u3088\u308a\u5f8c\u306b\u3057\u3066\u304f\u3060\u3055\u3044"
AUDIO_EXTRACT_ERROR = "\u97f3\u58f0\u306e\u53d6\u308a\u51fa\u3057\u306b\u5931\u6557\u3057\u307e\u3057\u305f"
VAD_ERROR = "\u767a\u8a71\u533a\u9593\u306e\u62bd\u51fa\u306b\u5931\u6557\u3057\u307e\u3057\u305f"
TIME_FORMAT_ERROR = "\u6642\u523b\u306f 00:00:00 \u306e\u5f62\u5f0f\u3067\u5165\u529b\u3057\u3066\u304f\u3060\u3055\u3044"


class AudioSegmentationError(Exception):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


def normalize_audio(
    input_path: str,
    output_dir: str | Path,
    range_start: str | None = None,
    range_end: str | None = None,
) -> PreprocessingResult:
    source = Path(input_path)
    if not source.exists() or not source.is_file():
        raise AudioSegmentationError(FILE_READ_ERROR)
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioSegmentationError(FILE_READ_ERROR)

    start_seconds = _parse_time_to_seconds(range_start) or 0.0
    end_seconds = _parse_time_to_seconds(range_end)
    if end_seconds is not None and end_seconds <= start_seconds:
        raise AudioSegmentationError(INVALID_TIME_RANGE_ERROR)

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise AudioSegmentationError(AUDIO_EXTRACT_ERROR)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{source.stem}_normalized_{uuid.uuid4().hex[:8]}.wav"

    command = [ffmpeg_path, "-y"]
    if start_seconds > 0:
        command.extend(["-ss", _format_ffmpeg_seconds(start_seconds)])
    command.extend(["-i", str(source)])
    if end_seconds is not None:
        command.extend(["-to", _format_ffmpeg_seconds(end_seconds)])
    command.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(output_path),
        ]
    )
    _run_ffmpeg(command, AUDIO_EXTRACT_ERROR)

    normalized_duration = _get_wav_duration(output_path)
    processed_end = end_seconds if end_seconds is not None else start_seconds + normalized_duration

    return PreprocessingResult(
        source_path=str(source),
        normalized_wav_path=str(output_path),
        processed_range_start=start_seconds,
        processed_range_end=max(start_seconds, processed_end),
        vad_segments=[],
        debug_paths={},
        fallback_used=False,
    )


def segment_speech_with_vad(wav_path: str | Path) -> list[VadSegment]:
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio  # type: ignore
    except ImportError as exc:
        raise AudioSegmentationError(VAD_ERROR) from exc

    try:
        audio = read_audio(str(wav_path), sampling_rate=16000)
        model = load_silero_vad()
        raw_segments = get_speech_timestamps(
            audio,
            model,
            sampling_rate=16000,
            min_silence_duration_ms=250,
            min_speech_duration_ms=180,
            speech_pad_ms=120,
            return_seconds=True,
        )
    except Exception as exc:
        raise AudioSegmentationError(VAD_ERROR) from exc

    return [
        VadSegment(
            start=float(item.get("start", 0.0) or 0.0),
            end=float(item.get("end", 0.0) or 0.0),
        )
        for item in raw_segments
        if float(item.get("end", 0.0) or 0.0) > float(item.get("start", 0.0) or 0.0)
    ]


def merge_short_vad_segments(
    segments: list[VadSegment],
    min_duration: float = DEFAULT_MIN_SEGMENT_SECONDS,
    max_gap: float = DEFAULT_MERGE_GAP_SECONDS,
) -> list[VadSegment]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    merged: list[VadSegment] = [VadSegment(start=ordered[0].start, end=ordered[0].end)]
    for segment in ordered[1:]:
        previous = merged[-1]
        previous_duration = previous.end - previous.start
        current_duration = segment.end - segment.start
        gap = max(0.0, segment.start - previous.end)
        should_merge = gap <= max_gap or previous_duration < min_duration or current_duration < min_duration
        if should_merge:
            previous.end = max(previous.end, segment.end)
            continue
        merged.append(VadSegment(start=segment.start, end=segment.end))
    return merged


def split_long_vad_segments(
    segments: list[VadSegment],
    max_duration: float = DEFAULT_MAX_SEGMENT_SECONDS,
    split_gap: float = DEFAULT_SPLIT_GAP_SECONDS,
) -> list[VadSegment]:
    if not segments:
        return []

    split_segments: list[VadSegment] = []
    for segment in segments:
        duration = max(0.0, segment.end - segment.start)
        if duration <= max_duration:
            split_segments.append(VadSegment(start=segment.start, end=segment.end))
            continue

        cursor = segment.start
        while cursor < segment.end:
            chunk_end = min(cursor + max_duration, segment.end)
            split_segments.append(VadSegment(start=cursor, end=chunk_end))
            if chunk_end >= segment.end:
                break
            cursor = chunk_end - split_gap
    return split_segments


def get_audio_duration_seconds(wav_path: str | Path) -> float:
    return _get_wav_duration(Path(wav_path))


def _parse_time_to_seconds(value: str | None) -> float | None:
    text = (value or "").strip()
    if not text:
        return None

    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return (hours * 3600) + (minutes * 60) + seconds
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return (minutes * 60) + seconds
        return float(text)
    except ValueError as exc:
        raise AudioSegmentationError(TIME_FORMAT_ERROR) from exc


def _format_ffmpeg_seconds(seconds: float) -> str:
    return f"{seconds:.3f}"


def _run_ffmpeg(command: list[str], user_message: str) -> None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise AudioSegmentationError(user_message) from exc

    if completed.returncode != 0:
        raise AudioSegmentationError(user_message)


def _get_wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate() or 16000
            frame_count = handle.getnframes()
        return frame_count / float(frame_rate)
    except (wave.Error, OSError) as exc:
        raise AudioSegmentationError(FILE_READ_ERROR) from exc
