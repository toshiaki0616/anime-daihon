from __future__ import annotations

import uuid
import wave
from dataclasses import dataclass
from os import getenv
from pathlib import Path
from typing import Any

from models.state import RawTranscriptSegment, VadSegment


CREATE_SUBTITLE_ERROR = "\u5b57\u5e55\u306e\u4f5c\u6210\u306b\u5931\u6557\u3057\u307e\u3057\u305f"
TRANSCRIPTION_DEPENDENCY_ERROR = (
    "\u6587\u5b57\u8d77\u3053\u3057\u306e\u6e96\u5099\u306b\u5931\u6557\u3057\u307e\u3057\u305f\u3002"
    " faster-whisper \u3092\u30a4\u30f3\u30b9\u30c8\u30fc\u30eb\u3057\u3066\u304f\u3060\u3055\u3044"
)
MISSING_AUDIO_FILE_ERROR = (
    "\u6587\u5b57\u8d77\u3053\u3057\u306e\u6e96\u5099\u306b\u5931\u6557\u3057\u307e\u3057\u305f\u3002"
    "\u97f3\u58f0\u30d5\u30a1\u30a4\u30eb\u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093"
)
MODEL_NOT_FOUND_ERROR = (
    "Anime Whisper \u30e2\u30c7\u30eb\u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093\u3067\u3057\u305f\u3002"
    " scripts\\setup_anime_whisper.ps1 \u3092\u5b9f\u884c\u3057\u3066"
    " models\\anime-whisper-ct2 \u3092\u6e96\u5099\u3057\u3066\u304f\u3060\u3055\u3044"
)
MODEL_INCOMPLETE_ERROR_PREFIX = (
    "Anime Whisper \u30e2\u30c7\u30eb\u306f\u3042\u308a\u307e\u3059\u304c\u3001"
    " CTranslate2 \u5909\u63db\u5f8c\u306e\u5fc5\u8981\u30d5\u30a1\u30a4\u30eb\u304c\u4e0d\u8db3\u3057\u3066\u3044\u307e\u3059\u3002"
)
MODEL_LOAD_ERROR = (
    "Anime Whisper \u30e2\u30c7\u30eb\u306e\u8aad\u307f\u8fbc\u307f\u306b\u5931\u6557\u3057\u307e\u3057\u305f\u3002"
    " scripts\\setup_anime_whisper.ps1 \u3092\u518d\u5b9f\u884c\u3057\u3066"
    " models\\anime-whisper-ct2 \u3092\u4f5c\u308a\u76f4\u3057\u3066\u304f\u3060\u3055\u3044"
)
EMPTY_TRANSCRIPTION_ERROR = (
    "\u5b57\u5e55\u306e\u4f5c\u6210\u306b\u5931\u6557\u3057\u307e\u3057\u305f\u3002"
    "\u97f3\u58f0\u304b\u3089\u30c6\u30ad\u30b9\u30c8\u3092\u53d6\u5f97\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f"
)
PRIMARY_SPLIT_PUNCTUATION = "\u3002\uff01\uff1f!?"
SECONDARY_SPLIT_PUNCTUATION = "\u3001\uff0c,/\u30fb"
TEMP_SPEAKER_ID = "A"
TEMP_SPEAKER_LABEL = "\u8a71\u8005A"
MAX_SUBTITLE_CHARS = 28
MIN_SUBTITLE_CHARS = 4


class TranscriptionError(Exception):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    source_start: float | None = None
    source_end: float | None = None


DEFAULT_MODEL_NAME = "anime_whisper"
LEGACY_MODEL_NAMES = {"", "base", "small", "medium", "anime-whisper-ct2", DEFAULT_MODEL_NAME}
DEFAULT_MODEL_DIR = Path(
    getenv(
        "ANIME_WHISPER_MODEL_DIR",
        str(Path(__file__).resolve().parent.parent / "models" / "anime-whisper-ct2"),
    )
)
CPU_THREADS = int(getenv("ANIME_WHISPER_CPU_THREADS", "8"))
MODEL_OPTIONS = [
    ("Anime Whisper (CPU int8)", DEFAULT_MODEL_NAME),
]
_MODEL_CACHE: dict[str, Any] = {}
REQUIRED_MODEL_FILES = ("config.json", "model.bin", "tokenizer.json")


def normalize_model_selection(model_name: str | None) -> str:
    candidate = (model_name or "").strip()
    if candidate in LEGACY_MODEL_NAMES:
        return DEFAULT_MODEL_NAME
    return candidate


def resolve_model_dir(model_name: str | None) -> Path:
    normalized = normalize_model_selection(model_name)
    if normalized == DEFAULT_MODEL_NAME:
        return DEFAULT_MODEL_DIR
    return Path(normalized)


def find_missing_model_files(model_dir: Path) -> list[str]:
    return [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).exists()]


def transcribe_segment(
    wav_path: str,
    start_sec: float,
    end_sec: float,
    model_name: str = DEFAULT_MODEL_NAME,
    initial_prompt: str = "",
    temp_dir: str | Path | None = None,
    segment_prefix: str = "vad",
) -> list[RawTranscriptSegment]:
    source = Path(wav_path)
    if not source.exists() or not source.is_file():
        raise TranscriptionError(MISSING_AUDIO_FILE_ERROR)
    if end_sec <= start_sec:
        return []

    clip_dir = Path(temp_dir) if temp_dir is not None else source.parent
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = _slice_wav_segment(source, start_sec, end_sec, clip_dir, segment_prefix)

    try:
        model = _load_model(model_name)
        segments_iter, _info = model.transcribe(
            str(clip_path),
            language="ja",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=True,
        )
    except TranscriptionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(CREATE_SUBTITLE_ERROR) from exc
    finally:
        try:
            clip_path.unlink(missing_ok=True)
        except OSError:
            pass

    transcript_segments: list[TranscriptionSegment] = []
    for item in segments_iter:
        text = str(getattr(item, "text", "")).strip()
        if not text:
            continue
        local_start = float(getattr(item, "start", 0.0) or 0.0)
        local_end = float(getattr(item, "end", 0.0) or 0.0)
        absolute_start = max(start_sec, start_sec + local_start)
        absolute_end = min(end_sec, max(absolute_start, start_sec + local_end))
        if absolute_end <= absolute_start:
            absolute_start = start_sec
            absolute_end = end_sec
        transcript_segments.append(
            TranscriptionSegment(
                start=absolute_start,
                end=absolute_end,
                text=text,
                source_start=absolute_start,
                source_end=absolute_end,
            )
        )

    if not transcript_segments:
        return []
    return _to_raw_transcript_segments(split_subtitle_segments(transcript_segments), segment_prefix)


def transcribe_segments(
    wav_path: str,
    vad_segments: list[VadSegment],
    model_name: str = DEFAULT_MODEL_NAME,
    initial_prompt: str = "",
    temp_dir: str | Path | None = None,
) -> tuple[list[RawTranscriptSegment], list[dict[str, Any]]]:
    _load_model(model_name)
    transcript_segments: list[RawTranscriptSegment] = []
    failures: list[dict[str, Any]] = []

    for index, vad_segment in enumerate(vad_segments):
        try:
            items = transcribe_segment(
                wav_path=wav_path,
                start_sec=vad_segment.start,
                end_sec=vad_segment.end,
                model_name=model_name,
                initial_prompt=initial_prompt,
                temp_dir=temp_dir,
                segment_prefix=f"vad_{index + 1:03d}",
            )
        except TranscriptionError as exc:
            failures.append(
                {
                    "vad_index": index,
                    "start": vad_segment.start,
                    "end": vad_segment.end,
                    "error": exc.user_message,
                }
            )
            continue

        if not items:
            failures.append(
                {
                    "vad_index": index,
                    "start": vad_segment.start,
                    "end": vad_segment.end,
                    "error": EMPTY_TRANSCRIPTION_ERROR,
                }
            )
            continue

        transcript_segments.extend(items)

    _renumber_raw_segments(transcript_segments)
    if not transcript_segments:
        raise TranscriptionError(CREATE_SUBTITLE_ERROR)
    return transcript_segments, failures


def transcribe_wav(
    wav_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    initial_prompt: str = "",
) -> list[TranscriptionSegment]:
    duration = _get_wav_duration(Path(wav_path))
    raw_segments, _failures = transcribe_segments(
        wav_path=wav_path,
        vad_segments=[VadSegment(start=0.0, end=duration)],
        model_name=model_name,
        initial_prompt=initial_prompt,
        temp_dir=Path(wav_path).parent,
    )
    return [
        TranscriptionSegment(
            start=item.start,
            end=item.end,
            text=item.original_text,
            source_start=item.source_start,
            source_end=item.source_end,
        )
        for item in raw_segments
    ]


def split_subtitle_segments(segments: list[TranscriptionSegment]) -> list[TranscriptionSegment]:
    expanded: list[TranscriptionSegment] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        primary_chunks = _split_on_punctuation(text, PRIMARY_SPLIT_PUNCTUATION)
        if not primary_chunks:
            primary_chunks = [text]

        final_chunks: list[str] = []
        for chunk in primary_chunks:
            final_chunks.extend(_split_long_chunk(chunk))
        final_chunks = _merge_short_chunks(final_chunks)

        if len(final_chunks) <= 1:
            expanded.append(
                TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    text=text,
                    source_start=segment.source_start if segment.source_start is not None else segment.start,
                    source_end=segment.source_end if segment.source_end is not None else segment.end,
                )
            )
            continue

        total_weight = sum(max(len(chunk.strip()), 1) for chunk in final_chunks)
        duration = max(segment.end - segment.start, 0.0)
        current_start = segment.start

        for index, chunk in enumerate(final_chunks):
            weight = max(len(chunk.strip()), 1)
            if index == len(final_chunks) - 1 or duration <= 0:
                chunk_end = segment.end
            else:
                chunk_duration = duration * (weight / total_weight)
                chunk_end = min(segment.end, current_start + chunk_duration)
            expanded.append(
                TranscriptionSegment(
                    start=current_start,
                    end=chunk_end,
                    text=chunk,
                    source_start=current_start,
                    source_end=chunk_end,
                )
            )
            current_start = chunk_end

    return expanded


def _load_model(model_name: str | None):
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise TranscriptionError(TRANSCRIPTION_DEPENDENCY_ERROR) from exc

    model_dir = resolve_model_dir(model_name)
    if not model_dir.exists() or not model_dir.is_dir():
        raise TranscriptionError(MODEL_NOT_FOUND_ERROR)

    missing_files = find_missing_model_files(model_dir)
    if missing_files:
        missing_list = ", ".join(missing_files)
        raise TranscriptionError(f"{MODEL_INCOMPLETE_ERROR_PREFIX} 不足ファイル: {missing_list}")

    cache_key = str(model_dir.resolve())
    if cache_key not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[cache_key] = WhisperModel(
                cache_key,
                device="cpu",
                compute_type="int8",
                cpu_threads=CPU_THREADS,
            )
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(MODEL_LOAD_ERROR) from exc
    return _MODEL_CACHE[cache_key]


def _split_on_punctuation(text: str, delimiters: str) -> list[str]:
    chunks: list[str] = []
    current = ""
    for char in text:
        current += char
        if char in delimiters:
            cleaned = current.strip()
            if cleaned:
                chunks.append(cleaned)
            current = ""
    cleaned = current.strip()
    if cleaned:
        chunks.append(cleaned)
    return chunks


def _split_long_chunk(text: str, max_chars: int = MAX_SUBTITLE_CHARS) -> list[str]:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return [cleaned] if cleaned else []

    secondary = _split_on_punctuation(cleaned, SECONDARY_SPLIT_PUNCTUATION)
    if len(secondary) > 1:
        output: list[str] = []
        for item in secondary:
            output.extend(_split_long_chunk(item, max_chars=max_chars))
        return output

    output = []
    start = 0
    while start < len(cleaned):
        output.append(cleaned[start : start + max_chars].strip())
        start += max_chars
    return [item for item in output if item]


def _merge_short_chunks(chunks: list[str], min_chars: int = MIN_SUBTITLE_CHARS) -> list[str]:
    if not chunks:
        return []

    merged: list[str] = []
    for chunk in chunks:
        cleaned = chunk.strip()
        if not cleaned:
            continue
        if merged and len(cleaned) < min_chars:
            merged[-1] = f"{merged[-1]}{cleaned}"
            continue
        merged.append(cleaned)
    return merged


def _slice_wav_segment(
    source_path: Path,
    start_sec: float,
    end_sec: float,
    output_dir: Path,
    segment_prefix: str,
) -> Path:
    output_path = output_dir / f"{segment_prefix}_{uuid.uuid4().hex[:8]}.wav"
    try:
        with wave.open(str(source_path), "rb") as reader:
            frame_rate = reader.getframerate()
            sample_width = reader.getsampwidth()
            channel_count = reader.getnchannels()
            start_frame = max(0, int(start_sec * frame_rate))
            end_frame = max(start_frame, int(end_sec * frame_rate))
            frame_count = max(0, end_frame - start_frame)
            reader.setpos(start_frame)
            frames = reader.readframes(frame_count)

        with wave.open(str(output_path), "wb") as writer:
            writer.setnchannels(channel_count)
            writer.setsampwidth(sample_width)
            writer.setframerate(frame_rate)
            writer.writeframes(frames)
    except (wave.Error, OSError) as exc:
        raise TranscriptionError(CREATE_SUBTITLE_ERROR) from exc
    return output_path


def _get_wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate() or 16000
            frame_count = handle.getnframes()
        return frame_count / float(frame_rate)
    except (wave.Error, OSError) as exc:
        raise TranscriptionError(MISSING_AUDIO_FILE_ERROR) from exc


def _to_raw_transcript_segments(
    segments: list[TranscriptionSegment],
    segment_prefix: str,
) -> list[RawTranscriptSegment]:
    items: list[RawTranscriptSegment] = []
    for index, segment in enumerate(segments):
        text = segment.text.strip()
        if not text:
            continue
        items.append(
            RawTranscriptSegment(
                id=f"{segment_prefix}_{index + 1:03d}",
                start=segment.start,
                end=segment.end,
                text=text,
                original_text=text,
                edited_text=text,
                speaker_id=TEMP_SPEAKER_ID,
                raw_label=TEMP_SPEAKER_LABEL,
                display_name=TEMP_SPEAKER_LABEL,
                source_start=segment.source_start if segment.source_start is not None else segment.start,
                source_end=segment.source_end if segment.source_end is not None else segment.end,
            )
        )
    return items


def _renumber_raw_segments(segments: list[RawTranscriptSegment]) -> None:
    for index, segment in enumerate(segments):
        segment.id = f"seg_{index + 1:03d}"
