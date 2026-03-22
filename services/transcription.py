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
SPLIT_PUNCTUATION = "。！？!?…"
SECONDARY_SPLIT_PUNCTUATION = "、，,・/／"
MAX_SUBTITLE_CHARS = 28
MIN_SUBTITLE_CHARS = 4


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


def split_subtitle_segments(segments: list[TranscriptionSegment]) -> list[TranscriptionSegment]:
    expanded: list[TranscriptionSegment] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        primary_chunks = _split_on_punctuation(text, SPLIT_PUNCTUATION)
        if not primary_chunks:
            primary_chunks = [text]

        final_chunks: list[str] = []
        for chunk in primary_chunks:
            final_chunks.extend(_split_long_chunk(chunk))
        final_chunks = _merge_short_chunks(final_chunks)

        if len(final_chunks) <= 1:
            expanded.append(TranscriptionSegment(start=segment.start, end=segment.end, text=text))
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
                )
            )
            current_start = chunk_end

    return expanded


def _load_model(model_name: str | None):
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise TranscriptionError("字幕の作成に失敗しました。faster-whisper をインストールしてください") from exc

    model_dir = resolve_model_dir(model_name)
    if not model_dir.exists() or not model_dir.is_dir():
        raise TranscriptionError(
            "Anime Whisper モデルが見つかりませんでした。"
            "scripts\\setup_anime_whisper.ps1 を実行して、models\\anime-whisper-ct2 を準備してください"
        )

    missing_files = find_missing_model_files(model_dir)
    if missing_files:
        missing_list = ", ".join(missing_files)
        raise TranscriptionError(
            "Anime Whisper モデルはありますが、CTranslate2 形式への変換が完了していません。"
            f"不足ファイル: {missing_list}。scripts\\setup_anime_whisper.ps1 を再実行してください"
        )

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
            raise TranscriptionError(
                "Anime Whisper モデルの読み込みに失敗しました。"
                "scripts\\setup_anime_whisper.ps1 を再実行して、models\\anime-whisper-ct2 を作り直してください"
            ) from exc
    return _MODEL_CACHE[cache_key]


def transcribe_wav(
    wav_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    initial_prompt: str = "",
) -> list[TranscriptionSegment]:
    source = Path(wav_path)
    if not source.exists() or not source.is_file():
        raise TranscriptionError("字幕の作成に失敗しました。音声ファイルが見つかりません")

    try:
        model = _load_model(model_name)
        # Anime Whisper does not use initial_prompt directly.
        segments_iter, _info = model.transcribe(
            str(source),
            language="ja",
            beam_size=5,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
            condition_on_previous_text=True,
            no_repeat_ngram_size=7,
            repetition_penalty=1.1,
        )
    except TranscriptionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError("字幕の作成に失敗しました") from exc

    raw_segments: list[TranscriptionSegment] = []
    for item in segments_iter:
        text = str(getattr(item, "text", "")).strip()
        if not text:
            continue
        raw_segments.append(
            TranscriptionSegment(
                start=float(getattr(item, "start", 0.0)),
                end=float(getattr(item, "end", 0.0)),
                text=text,
            )
        )

    segments = split_subtitle_segments(raw_segments)
    if not segments:
        raise TranscriptionError("字幕の作成に失敗しました。音声から文字を取得できませんでした")
    return segments
