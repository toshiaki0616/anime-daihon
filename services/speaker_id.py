from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from pathlib import Path

from models.state import VoiceprintProfile, VoiceprintSample


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
        raise SpeakerIdentificationError("声紋ベクトルの次元が一致していません")

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
    threshold: float = 0.45,
) -> VoiceprintMatch | None:
    matches = score_voiceprint_profiles(embedding, profiles)
    if not matches:
        return None
    best = matches[0]
    if best.confidence < threshold:
        return None
    return best


def extract_voice_embedding(wav_path: str, clip_start: float, clip_end: float) -> list[float]:
    source = Path(wav_path)
    if not source.exists() or not source.is_file():
        raise SpeakerIdentificationError("声紋登録に失敗しました。音声ファイルが見つかりません")

    raise SpeakerIdentificationError(
        "声紋抽出はまだ未実装です。Step 5 で pyannote の embedding 抽出を接続します"
    )
