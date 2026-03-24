from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from models.state import (
    DiarizationSegment,
    PreprocessingResult,
    SpeakerAssignmentResult,
    TranscriptionResult,
    VadSegment,
)

from .audio_segmentation import (
    AudioSegmentationError,
    get_audio_duration_seconds,
    merge_short_vad_segments,
    normalize_audio,
    segment_speech_with_vad,
    split_long_vad_segments,
)
from .diarization import DiarizationError, diarize_wav_full
from .speaker_assignment import assign_dominant_speaker_to_segments
from .transcription import TranscriptionError, transcribe_segments


ProgressCallback = Callable[[float, str], None]
EXTRACTING_AUDIO_MESSAGE = "\u97f3\u58f0\u3092\u53d6\u308a\u51fa\u3057\u3066\u3044\u307e\u3059..."
RUNNING_VAD_MESSAGE = "\u767a\u8a71\u533a\u9593\u3092\u62bd\u51fa\u3057\u3066\u3044\u307e\u3059..."
PREPROCESS_DONE_MESSAGE = "\u524d\u51e6\u7406\u304c\u5b8c\u4e86\u3057\u307e\u3057\u305f..."
PREPROCESS_SAVE_ERROR = "\u524d\u51e6\u7406\u7d50\u679c\u306e\u4fdd\u5b58\u306b\u5931\u6557\u3057\u307e\u3057\u305f"
RUNNING_ASR_MESSAGE = "\u5b57\u5e55\u3092\u4f5c\u6210\u3057\u3066\u3044\u307e\u3059..."
ASR_SAVE_ERROR = "\u5b57\u5e55\u306e\u4f5c\u6210\u306b\u5931\u6557\u3057\u307e\u3057\u305f"
RUNNING_DIARIZATION_MESSAGE = "\u8a71\u8005\u3092\u5206\u5272\u3057\u3066\u3044\u307e\u3059..."
DIARIZATION_SAVE_ERROR = "\u8a71\u8005\u306e\u5206\u5272\u306b\u5931\u6557\u3057\u307e\u3057\u305f"


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


def run_transcription_pipeline(
    preprocessing_result: PreprocessingResult,
    model_name: str,
    initial_prompt: str,
    data_dir: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> TranscriptionResult:
    debug_dir = Path(data_dir) / "debug"
    temp_dir = Path(data_dir) / "processed" / "asr_segments"
    debug_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress_callback, 0.44, RUNNING_ASR_MESSAGE)
    raw_segments, failed_segments = transcribe_segments(
        wav_path=preprocessing_result.normalized_wav_path,
        vad_segments=preprocessing_result.vad_segments,
        model_name=model_name,
        initial_prompt=initial_prompt,
        temp_dir=temp_dir,
    )

    result = TranscriptionResult(
        source_path=preprocessing_result.source_path,
        wav_path=preprocessing_result.normalized_wav_path,
        vad_segments=list(preprocessing_result.vad_segments),
        raw_transcript_segments=raw_segments,
        debug_paths={},
        failed_segments=failed_segments,
    )
    debug_path = debug_dir / "debug_asr_segments.json"
    result.debug_paths = {"asr_segments": str(debug_path)}
    _write_asr_debug_output(result, debug_path)
    return result


def run_diarization_pipeline(
    preprocessing_result: PreprocessingResult,
    transcription_result: TranscriptionResult,
    data_dir: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> SpeakerAssignmentResult:
    debug_dir = Path(data_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    diarization_segments: list[DiarizationSegment] = []
    fallback_used = False
    error_message = ""
    _notify(progress_callback, 0.72, RUNNING_DIARIZATION_MESSAGE)
    try:
        diarization_segments = diarize_wav_full(preprocessing_result.normalized_wav_path)
    except DiarizationError as exc:
        fallback_used = True
        error_message = exc.user_message
        diarization_segments = []

    assignment_result = assign_dominant_speaker_to_segments(
        transcript_segments=transcription_result.raw_transcript_segments,
        diarization_segments=diarization_segments,
    )
    assignment_result.fallback_used = fallback_used
    assignment_result.error_message = error_message

    diarization_debug_path = debug_dir / "debug_diarization_segments.json"
    final_debug_path = debug_dir / "debug_final_segments.json"
    assignment_result.debug_paths = {
        "diarization_segments": str(diarization_debug_path),
        "final_segments": str(final_debug_path),
    }
    _write_diarization_debug_output(
        preprocessing_result.normalized_wav_path,
        diarization_segments,
        assignment_result.ui_speaker_map,
        assignment_result.fallback_used,
        assignment_result.error_message,
        diarization_debug_path,
    )
    _write_final_segments_debug_output(assignment_result, final_debug_path)
    return assignment_result


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


def _write_asr_debug_output(result: TranscriptionResult, output_path: Path) -> None:
    payload = {
        "source_file_path": result.source_path,
        "normalized_wav_path": result.wav_path,
        "vad_segments_used": [segment.to_dict() for segment in result.vad_segments],
        "raw_asr_text_per_segment": [segment.to_dict() for segment in result.raw_transcript_segments],
        "output_subtitle_segments": [segment.to_dict() for segment in result.raw_transcript_segments],
        "failed_segments": [dict(item) for item in result.failed_segments],
        "debug_paths": dict(result.debug_paths),
    }
    try:
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        raise TranscriptionError(ASR_SAVE_ERROR) from exc


def _write_diarization_debug_output(
    wav_path: str,
    diarization_segments: list[DiarizationSegment],
    ui_speaker_map: dict[str, str],
    fallback_used: bool,
    error_message: str,
    output_path: Path,
) -> None:
    payload = {
        "wav_path": wav_path,
        "raw_diarization_output": [segment.to_dict() for segment in diarization_segments],
        "normalized_speaker_mapping": dict(ui_speaker_map),
        "fallback_used": fallback_used,
        "error_message": error_message,
    }
    try:
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        raise DiarizationError(DIARIZATION_SAVE_ERROR) from exc


def _write_final_segments_debug_output(
    result: SpeakerAssignmentResult,
    output_path: Path,
) -> None:
    payload = {
        "final_subtitle_rows": [
            {
                "start": segment.start,
                "end": segment.end,
                "speaker_id": segment.speaker_id,
                "raw_label": segment.raw_label,
                "display_name": segment.display_name,
                "original_text": segment.original_text,
                "edited_text": segment.edited_text,
            }
            for segment in result.assigned_subtitle_segments
        ],
        "ui_speaker_map": dict(result.ui_speaker_map),
        "fallback_used": result.fallback_used,
        "error_message": result.error_message,
    }
    try:
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        raise DiarizationError(DIARIZATION_SAVE_ERROR) from exc


def _notify(progress_callback: ProgressCallback | None, value: float, message: str) -> None:
    if progress_callback is not None:
        progress_callback(value, message)
