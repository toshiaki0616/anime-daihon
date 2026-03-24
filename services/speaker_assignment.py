from __future__ import annotations

from copy import deepcopy

from models.state import (
    DiarizationSegment,
    RawTranscriptSegment,
    SpeakerAssignmentResult,
    SpeakerProfile,
)
from services.diarization import normalize_raw_speaker_labels


FALLBACK_SPEAKER_ID = "speaker_a"
FALLBACK_LABEL = "\u8a71\u8005A"


def compute_overlap_score(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def map_raw_speakers_to_ui_labels(
    diarization_segments: list[DiarizationSegment],
) -> dict[str, str]:
    mapping = normalize_raw_speaker_labels(diarization_segments)
    mapping["__fallback__"] = FALLBACK_SPEAKER_ID
    return mapping


def assign_dominant_speaker_to_segments(
    transcript_segments: list[RawTranscriptSegment],
    diarization_segments: list[DiarizationSegment],
) -> SpeakerAssignmentResult:
    ui_speaker_map = map_raw_speakers_to_ui_labels(diarization_segments)
    assigned_segments: list[RawTranscriptSegment] = []

    for segment in transcript_segments:
        dominant_raw_speaker = _find_dominant_raw_speaker(segment, diarization_segments)
        normalized_id = ui_speaker_map.get(dominant_raw_speaker or "__fallback__", FALLBACK_SPEAKER_ID)
        normalized_label = _label_for_speaker_id(normalized_id)

        updated = deepcopy(segment)
        updated.speaker_id = normalized_id
        updated.raw_label = normalized_label
        updated.display_name = normalized_label
        assigned_segments.append(updated)

    speaker_profiles = _build_profiles_from_raw_segments(assigned_segments)
    return SpeakerAssignmentResult(
        diarization_segments=deepcopy(diarization_segments),
        assigned_subtitle_segments=assigned_segments,
        ui_speaker_map=ui_speaker_map,
        speaker_profiles=speaker_profiles,
        debug_paths={},
        fallback_used=not bool(diarization_segments),
        error_message="",
    )


def _find_dominant_raw_speaker(
    segment: RawTranscriptSegment,
    diarization_segments: list[DiarizationSegment],
) -> str | None:
    if not diarization_segments:
        return None

    overlap_by_speaker: dict[str, float] = {}
    earliest_overlap: dict[str, float] = {}
    for diarization in diarization_segments:
        overlap = compute_overlap_score(segment.start, segment.end, diarization.start, diarization.end)
        if overlap <= 0:
            continue
        overlap_by_speaker[diarization.raw_speaker_id] = overlap_by_speaker.get(diarization.raw_speaker_id, 0.0) + overlap
        earliest_overlap[diarization.raw_speaker_id] = min(
            earliest_overlap.get(diarization.raw_speaker_id, float("inf")),
            max(segment.start, diarization.start),
        )

    if overlap_by_speaker:
        return min(
            overlap_by_speaker,
            key=lambda speaker_id: (
                -overlap_by_speaker[speaker_id],
                earliest_overlap.get(speaker_id, float("inf")),
                speaker_id,
            ),
        )

    midpoint = segment.start + ((segment.end - segment.start) / 2.0)
    nearest: tuple[float, float, str] | None = None
    for diarization in diarization_segments:
        if diarization.start <= midpoint <= diarization.end:
            distance = 0.0
        else:
            distance = min(abs(midpoint - diarization.start), abs(midpoint - diarization.end))
        candidate = (distance, diarization.start, diarization.raw_speaker_id)
        if nearest is None or candidate < nearest:
            nearest = candidate
    if nearest is None or nearest[0] > 1.5:
        return None
    return nearest[2]


def _build_profiles_from_raw_segments(
    segments: list[RawTranscriptSegment],
) -> list[SpeakerProfile]:
    grouped: dict[str, list[RawTranscriptSegment]] = {}
    for item in segments:
        grouped.setdefault(item.speaker_id, []).append(item)

    profiles: list[SpeakerProfile] = []
    for speaker_id, items in grouped.items():
        raw_label = items[0].raw_label if items else FALLBACK_LABEL
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


def _label_for_speaker_id(speaker_id: str) -> str:
    if speaker_id.startswith("speaker_") and len(speaker_id) == 9 and speaker_id[-1].isalpha():
        return f"\u8a71\u8005{speaker_id[-1].upper()}"
    if speaker_id.startswith("speaker_"):
        suffix = speaker_id.split("_", 1)[1]
        if suffix.isdigit():
            return f"\u8a71\u8005{int(suffix)}"
    return FALLBACK_LABEL
