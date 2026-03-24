from __future__ import annotations

from os import getenv
from pathlib import Path

from models.state import DiarizationSegment, SpeakerProfile, SubtitleSegment


DIARIZATION_ERROR = "\u8a71\u8005\u306e\u5206\u5272\u306b\u5931\u6557\u3057\u307e\u3057\u305f"
DEFAULT_DIARIZATION_MODEL = getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
DEFAULT_AUTH_TOKEN = getenv("HF_TOKEN") or getenv("HUGGINGFACE_TOKEN")


class DiarizationError(Exception):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


def diarize_wav_full(
    wav_path: str,
    diarization_config: dict[str, str] | None = None,
) -> list[DiarizationSegment]:
    source = Path(wav_path)
    if not source.exists() or not source.is_file():
        raise DiarizationError(DIARIZATION_ERROR)

    config = diarization_config or {}
    model_name = config.get("model_name", DEFAULT_DIARIZATION_MODEL)
    auth_token = config.get("auth_token", DEFAULT_AUTH_TOKEN)

    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError as exc:
        raise DiarizationError(DIARIZATION_ERROR) from exc

    try:
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=auth_token)
        diarization = pipeline(str(source))
    except Exception as exc:  # noqa: BLE001
        raise DiarizationError(DIARIZATION_ERROR) from exc

    segments: list[DiarizationSegment] = []
    try:
        iterator = diarization.itertracks(yield_label=True)
    except Exception as exc:  # noqa: BLE001
        raise DiarizationError(DIARIZATION_ERROR) from exc

    for segment, _, speaker in iterator:
        start = float(getattr(segment, "start", 0.0) or 0.0)
        end = float(getattr(segment, "end", 0.0) or 0.0)
        if end <= start:
            continue
        segments.append(
            DiarizationSegment(
                start=start,
                end=end,
                raw_speaker_id=str(speaker).strip(),
            )
        )

    if not segments:
        raise DiarizationError(DIARIZATION_ERROR)
    return segments


def diarize_wav(
    wav_path: str,
    model_name: str = DEFAULT_DIARIZATION_MODEL,
    auth_token: str | None = DEFAULT_AUTH_TOKEN,
) -> list[DiarizationSegment]:
    return diarize_wav_full(
        wav_path=wav_path,
        diarization_config={
            "model_name": model_name,
            "auth_token": auth_token or "",
        },
    )


def normalize_raw_speaker_labels(diarization_segments: list[DiarizationSegment]) -> dict[str, str]:
    first_seen: dict[str, float] = {}
    for segment in diarization_segments:
        first_seen.setdefault(segment.raw_speaker_id, segment.start)

    ordered = sorted(first_seen, key=lambda speaker_id: (first_seen[speaker_id], speaker_id))
    speaker_map: dict[str, str] = {}
    for index, raw_speaker_id in enumerate(ordered):
        if index < 26:
            speaker_map[raw_speaker_id] = f"speaker_{chr(ord('a') + index)}"
        else:
            speaker_map[raw_speaker_id] = f"speaker_{index + 1:02d}"
    return speaker_map


def build_speaker_profiles_from_segments(
    segments: list[SubtitleSegment],
) -> list[SpeakerProfile]:
    grouped: dict[str, list[SubtitleSegment]] = {}
    for segment in segments:
        grouped.setdefault(segment.speaker_id, []).append(segment)

    profiles: list[SpeakerProfile] = []
    for speaker_id, items in grouped.items():
        raw_label = items[0].raw_label if items else "\u8a71\u8005A"
        display_name = items[0].display_name if items else raw_label
        profiles.append(
            SpeakerProfile(
                speaker_id=speaker_id,
                raw_label=raw_label,
                display_name=display_name,
                utterance_count=len(items),
                sample_texts=[item.edited_text for item in items[:3]],
            )
        )
    profiles.sort(key=lambda item: item.raw_label)
    return profiles
