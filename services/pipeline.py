from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from models.state import PreprocessingResult, VadSegment

from .audio_segmentation import (
    AudioSegmentationError,
    get_audio_duration_seconds,
    merge_short_vad_segments,
    normalize_audio,
    segment_speech_with_vad,
    split_long_vad_segments,
)


ProgressCallback = Callable[[float, str], None]
EXTRACTING_AUDIO_MESSAGE = "\u97f3\u58f0\u3092\u53d6\u308a\u51fa\u3057\u3066\u3044\u307e\u3059..."
RUNNING_VAD_MESSAGE = "\u767a\u8a71\u533a\u9593\u3092\u62bd\u51fa\u3057\u3066\u3044\u307e\u3059..."
PREPROCESS_DONE_MESSAGE = "\u524d\u51e6\u7406\u304c\u5b8c\u4e86\u3057\u307e\u3057\u305f..."
PREPROCESS_SAVE_ERROR = "\u524d\u51e6\u7406\u7d50\u679c\u306e\u4fdd\u5b58\u306b\u5931\u6557\u3057\u307e\u3057\u305f"


def run_preprocessing_pipeline(
    input_path: str,
    range_start: str,
    range_end: str,
    data_dir: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> PreprocessingResult:
    output_root = Path(data_dir)
    processed_dir = output_root / "processed"
    debug_dir = output_root / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress_callback, 0.12, EXTRACTING_AUDIO_MESSAGE)
    result = normalize_audio(
        input_path=input_path,
        output_dir=processed_dir,
        range_start=range_start,
        range_end=range_end,
    )

    fallback_used = False
    fallback_reason = ""
    _notify(progress_callback, 0.24, RUNNING_VAD_MESSAGE)
    try:
        vad_segments = segment_speech_with_vad(result.normalized_wav_path)
        vad_segments = merge_short_vad_segments(vad_segments)
        vad_segments = split_long_vad_segments(vad_segments)
        if not vad_segments:
            fallback_used = True
            fallback_reason = "no_segments"
            vad_segments = _build_full_range_fallback(result.normalized_wav_path)
    except AudioSegmentationError:
        fallback_used = True
        fallback_reason = "vad_error"
        vad_segments = _build_full_range_fallback(result.normalized_wav_path)

    result.vad_segments = vad_segments
    result.fallback_used = fallback_used
    debug_path = debug_dir / "debug_vad_segments.json"
    result.debug_paths = {"vad_segments": str(debug_path)}
    _write_vad_debug_output(result, debug_path, fallback_reason)
    _notify(progress_callback, 0.32, PREPROCESS_DONE_MESSAGE)
    return result


def _build_full_range_fallback(wav_path: str) -> list[VadSegment]:
    duration = max(0.0, get_audio_duration_seconds(wav_path))
    return [VadSegment(start=0.0, end=duration)]


def _write_vad_debug_output(
    result: PreprocessingResult,
    output_path: Path,
    fallback_reason: str,
) -> None:
    payload = {
        "source_file_path": result.source_path,
        "normalized_wav_path": result.normalized_wav_path,
        "processed_range_start": result.processed_range_start,
        "processed_range_end": result.processed_range_end,
        "processed_duration": max(0.0, result.processed_range_end - result.processed_range_start),
        "fallback_used": result.fallback_used,
        "fallback_reason": fallback_reason,
        "final_vad_segments": [segment.to_dict() for segment in result.vad_segments],
        "debug_paths": dict(result.debug_paths),
    }
    try:
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        raise AudioSegmentationError(PREPROCESS_SAVE_ERROR) from exc


def _notify(progress_callback: ProgressCallback | None, value: float, message: str) -> None:
    if progress_callback is not None:
        progress_callback(value, message)
