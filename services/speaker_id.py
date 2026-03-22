from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from os import getenv
from pathlib import Path
from typing import Any

from models.state import VoiceprintProfile, VoiceprintSample
from services.transcription import TranscriptionSegment


class SpeakerIdentificationError(Exception):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class VoiceprintMatch:
    profile_id: str
    character_name: str
    confidence: float
    sample_count: int


@dataclass
class VoiceprintAssignment:
    profile_id: str
    character_name: str
    confidence: float


DEFAULT_EMBEDDING_MODEL = getenv("VOICEPRINT_EMBEDDING_MODEL", "pyannote/embedding")
DEFAULT_AUTH_TOKEN = getenv("HF_TOKEN") or getenv("HUGGINGFACE_TOKEN")
VOICEPRINT_THRESHOLD = float(getenv("VOICEPRINT_THRESHOLD", "0.45"))
_EMBEDDING_INFERENCE_CACHE: dict[tuple[str, str], Any] = {}


def now_label() -> str:
    return datetime.now().replace(microsecond=0).isoformat(timespec="seconds")


def build_voiceprint_sample(
    *,
    episode_id: str,
    speaker_id: str,
    character_name: str,
    source_wav_path: str,
    clip_start: float,
    clip_end: float,
    transcript_text: str = "",
    embedding: list[float] | None = None,
) -> VoiceprintSample:
    return VoiceprintSample(
        sample_id=f"vp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        episode_id=episode_id,
        speaker_id=speaker_id,
        character_name=character_name.strip(),
        source_wav_path=source_wav_path,
        clip_start=clip_start,
        clip_end=clip_end,
        transcript_text=transcript_text.strip(),
        embedding=[float(value) for value in (embedding or [])],
        created_at=now_label(),
    )


def _normalize_embedding(values: list[float]) -> list[float]:
    if not values:
        return []
    norm = sqrt(sum(value * value for value in values))
    if norm <= 0:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def average_embedding(samples: list[VoiceprintSample]) -> list[float]:
    valid = [sample.embedding for sample in samples if sample.embedding]
    if not valid:
        return []

    size = len(valid[0])
    if any(len(vector) != size for vector in valid):
        raise SpeakerIdentificationError("Voiceprint vector dimensions do not match")

    totals = [0.0] * size
    for vector in valid:
        for index, value in enumerate(vector):
            totals[index] += value
    averaged = [value / len(valid) for value in totals]
    return _normalize_embedding(averaged)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    normalized_left = _normalize_embedding(left)
    normalized_right = _normalize_embedding(right)
    return sum(a * b for a, b in zip(normalized_left, normalized_right))


def upsert_voiceprint_profile(
    *,
    work_id: str,
    character_name: str,
    sample: VoiceprintSample,
    profiles: list[VoiceprintProfile],
    samples: list[VoiceprintSample],
) -> tuple[list[VoiceprintProfile], list[VoiceprintSample], VoiceprintProfile]:
    next_profiles = [VoiceprintProfile.from_dict(profile.to_dict()) for profile in profiles]
    next_samples = [VoiceprintSample.from_dict(existing.to_dict()) for existing in samples]
    next_samples.append(sample)

    target_name = character_name.strip()
    profile = next(
        (
            item
            for item in next_profiles
            if item.work_id == work_id and item.character_name == target_name
        ),
        None,
    )
    timestamp = now_label()
    if profile is None:
        profile = VoiceprintProfile(
            profile_id=f"profile_{len(next_profiles) + 1:03d}",
            work_id=work_id,
            character_name=target_name,
            sample_ids=[],
            sample_count=0,
            average_embedding=[],
            created_at=timestamp,
            updated_at=timestamp,
        )
        next_profiles.append(profile)

    profile.sample_ids.append(sample.sample_id)
    profile.sample_count = len(profile.sample_ids)
    profile.updated_at = timestamp
    profile.average_embedding = average_embedding(
        [item for item in next_samples if item.sample_id in profile.sample_ids]
    )
    return next_profiles, next_samples, profile


def score_voiceprint_profiles(
    embedding: list[float],
    profiles: list[VoiceprintProfile],
) -> list[VoiceprintMatch]:
    matches: list[VoiceprintMatch] = []
    for profile in profiles:
        confidence = cosine_similarity(embedding, profile.average_embedding)
        matches.append(
            VoiceprintMatch(
                profile_id=profile.profile_id,
                character_name=profile.character_name,
                confidence=confidence,
                sample_count=profile.sample_count,
            )
        )
    matches.sort(key=lambda item: (-item.confidence, -item.sample_count, item.character_name))
    return matches


def select_best_voiceprint_match(
    embedding: list[float],
    profiles: list[VoiceprintProfile],
    threshold: float = VOICEPRINT_THRESHOLD,
) -> VoiceprintMatch | None:
    matches = score_voiceprint_profiles(embedding, profiles)
    if not matches:
        return None
    best = matches[0]
    if best.confidence < threshold:
        return None
    return best


def _load_embedding_inference(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    auth_token: str | None = DEFAULT_AUTH_TOKEN,
):
    cache_key = (model_name, auth_token or "")
    if cache_key in _EMBEDDING_INFERENCE_CACHE:
        return _EMBEDDING_INFERENCE_CACHE[cache_key]

    try:
        from pyannote.audio import Inference, Model  # type: ignore
    except ImportError as exc:
        raise SpeakerIdentificationError("pyannote.audio is required for voiceprint extraction") from exc

    try:
        model = Model.from_pretrained(model_name, use_auth_token=auth_token)
        inference = Inference(model, window="whole")
    except Exception as exc:  # noqa: BLE001
        raise SpeakerIdentificationError("Failed to load the voiceprint embedding model") from exc

    _EMBEDDING_INFERENCE_CACHE[cache_key] = inference
    return inference


def extract_voice_embedding(
    wav_path: str,
    clip_start: float,
    clip_end: float,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    auth_token: str | None = DEFAULT_AUTH_TOKEN,
) -> list[float]:
    source = Path(wav_path)
    if not source.exists() or not source.is_file():
        raise SpeakerIdentificationError("Voiceprint extraction failed because the audio file was not found")
    if clip_end <= clip_start:
        raise SpeakerIdentificationError("Voiceprint extraction failed because the time range is invalid")

    inference = _load_embedding_inference(model_name=model_name, auth_token=auth_token)

    try:
        from pyannote.core import Segment  # type: ignore
        import numpy as np
    except ImportError as exc:
        raise SpeakerIdentificationError("pyannote.core and numpy are required for voiceprint extraction") from exc

    try:
        embedding = inference.crop(str(source), Segment(clip_start, clip_end))
        values = np.asarray(embedding, dtype=float).reshape(-1).tolist()
    except Exception as exc:  # noqa: BLE001
        raise SpeakerIdentificationError("Voiceprint extraction failed") from exc

    normalized = _normalize_embedding([float(value) for value in values])
    if not normalized:
        raise SpeakerIdentificationError("Voiceprint extraction returned an empty embedding")
    return normalized


def _apply_continuity_rule(
    assignments: list[VoiceprintAssignment | None],
    segments: list[TranscriptionSegment],
) -> list[VoiceprintAssignment | None]:
    smoothed = list(assignments)
    for index in range(1, len(smoothed)):
        previous = smoothed[index - 1]
        current = smoothed[index]
        if previous is None or current is None:
            continue
        gap = segments[index].start - segments[index - 1].end
        if gap > 0.6:
            continue
        if previous.character_name != current.character_name and current.confidence < previous.confidence + 0.08:
            smoothed[index] = VoiceprintAssignment(
                profile_id=previous.profile_id,
                character_name=previous.character_name,
                confidence=max(current.confidence, previous.confidence * 0.92),
            )
    return smoothed


def assign_voiceprints_to_segments(
    wav_path: str,
    segments: list[TranscriptionSegment],
    profiles: list[VoiceprintProfile],
    threshold: float = VOICEPRINT_THRESHOLD,
) -> list[VoiceprintAssignment | None]:
    if not profiles:
        return [None for _ in segments]

    assignments: list[VoiceprintAssignment | None] = []
    for segment in segments:
        duration = segment.end - segment.start
        if duration <= 0.35:
            assignments.append(None)
            continue
        try:
            embedding = extract_voice_embedding(wav_path, segment.start, segment.end)
            best = select_best_voiceprint_match(embedding, profiles, threshold=threshold)
        except SpeakerIdentificationError:
            assignments.append(None)
            continue

        if best is None:
            assignments.append(None)
            continue

        assignments.append(
            VoiceprintAssignment(
                profile_id=best.profile_id,
                character_name=best.character_name,
                confidence=best.confidence,
            )
        )

    return _apply_continuity_rule(assignments, segments)
