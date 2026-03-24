from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import wave

import numpy as np

from models.state import (
    DiarizationSegment,
    RawTranscriptSegment,
    SpeakerAssignmentResult,
    SpeakerProfile,
)
from services.diarization import normalize_raw_speaker_labels


FALLBACK_SPEAKER_ID = "speaker_a"
FALLBACK_LABEL = "\u8a71\u8005A"
MAX_FALLBACK_SPEAKERS = 4
MIN_CLUSTER_SIMILARITY = 0.83


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
    wav_path: str | None = None,
) -> SpeakerAssignmentResult:
    raw_diarization_segments = list(diarization_segments)
    used_fallback_clustering = False

    if not raw_diarization_segments and wav_path:
        raw_diarization_segments = _build_fallback_diarization_segments(wav_path, transcript_segments)
        used_fallback_clustering = bool(raw_diarization_segments)

    ui_speaker_map = map_raw_speakers_to_ui_labels(raw_diarization_segments)
    assigned_segments: list[RawTranscriptSegment] = []

    for segment in transcript_segments:
        dominant_raw_speaker = _find_dominant_raw_speaker(segment, raw_diarization_segments)
        normalized_id = ui_speaker_map.get(dominant_raw_speaker or "__fallback__", FALLBACK_SPEAKER_ID)
        normalized_label = _label_for_speaker_id(normalized_id)

        updated = deepcopy(segment)
        updated.speaker_id = normalized_id
        updated.raw_label = normalized_label
        updated.display_name = normalized_label
        assigned_segments.append(updated)

    speaker_profiles = _build_profiles_from_raw_segments(assigned_segments)
    return SpeakerAssignmentResult(
        diarization_segments=deepcopy(raw_diarization_segments),
        assigned_subtitle_segments=assigned_segments,
        ui_speaker_map=ui_speaker_map,
        speaker_profiles=speaker_profiles,
        debug_paths={},
        fallback_used=used_fallback_clustering or not bool(diarization_segments),
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


def _build_fallback_diarization_segments(
    wav_path: str,
    transcript_segments: list[RawTranscriptSegment],
) -> list[DiarizationSegment]:
    features = _extract_segment_features(wav_path, transcript_segments)
    if not features:
        return []

    labels = _cluster_feature_vectors(features)
    diarization_segments: list[DiarizationSegment] = []
    for segment, label in zip(transcript_segments, labels):
        diarization_segments.append(
            DiarizationSegment(
                start=segment.start,
                end=segment.end,
                raw_speaker_id=f"fallback_{label}",
            )
        )
    return _merge_adjacent_segments(diarization_segments)


def _extract_segment_features(
    wav_path: str,
    transcript_segments: list[RawTranscriptSegment],
) -> list[np.ndarray]:
    source = Path(wav_path)
    if not source.exists():
        return []

    try:
        with wave.open(str(source), "rb") as reader:
            sample_rate = reader.getframerate()
            sample_width = reader.getsampwidth()
            channel_count = reader.getnchannels()
            frames = reader.readframes(reader.getnframes())
    except OSError:
        return []

    if sample_width != 2:
        return []

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channel_count > 1:
        audio = audio.reshape(-1, channel_count).mean(axis=1)
    if audio.size == 0:
        return []

    audio = audio / 32768.0
    features: list[np.ndarray] = []
    for segment in transcript_segments:
        start_frame = max(0, int(segment.start * sample_rate))
        end_frame = min(audio.shape[0], int(segment.end * sample_rate))
        clip = audio[start_frame:end_frame]
        if clip.size < max(int(sample_rate * 0.2), 1):
            clip = _pad_clip(clip, max(int(sample_rate * 0.2), 1))
        features.append(_build_feature_vector(clip, sample_rate))
    return features


def _pad_clip(clip: np.ndarray, target_size: int) -> np.ndarray:
    if clip.size >= target_size:
        return clip
    padding = np.zeros(target_size - clip.size, dtype=np.float32)
    return np.concatenate([clip, padding])


def _build_feature_vector(clip: np.ndarray, sample_rate: int) -> np.ndarray:
    if clip.size == 0:
        return np.zeros(19, dtype=np.float32)

    centered = clip - float(np.mean(clip))
    window = np.hanning(centered.size)
    spectrum = np.abs(np.fft.rfft(centered * window))
    if spectrum.size == 0:
        spectrum = np.zeros(16, dtype=np.float32)
    log_spectrum = np.log1p(spectrum)
    binned_spectrum = _bin_vector(log_spectrum, 12)

    rms = float(np.sqrt(np.mean(np.square(clip))))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(clip).astype(np.int8)))))
    energy = float(np.mean(np.abs(clip)))
    pitch_like = _estimate_pitch_like(centered, sample_rate)

    return np.concatenate(
        [
            np.array([rms, zcr, energy, pitch_like], dtype=np.float32),
            binned_spectrum.astype(np.float32),
        ]
    )


def _bin_vector(values: np.ndarray, bin_count: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(bin_count, dtype=np.float32)
    edges = np.linspace(0, values.size, bin_count + 1, dtype=int)
    bins = []
    for left, right in zip(edges[:-1], edges[1:]):
        chunk = values[left:right]
        bins.append(float(np.mean(chunk)) if chunk.size else 0.0)
    return np.asarray(bins, dtype=np.float32)


def _estimate_pitch_like(values: np.ndarray, sample_rate: int) -> float:
    if values.size < 32:
        return 0.0
    autocorr = np.correlate(values, values, mode="full")[values.size - 1 :]
    min_lag = max(1, int(sample_rate / 300))
    max_lag = min(autocorr.size - 1, int(sample_rate / 80))
    if max_lag <= min_lag:
        return 0.0
    region = autocorr[min_lag:max_lag]
    if region.size == 0:
        return 0.0
    best_lag = int(np.argmax(region)) + min_lag
    return float(sample_rate / best_lag) if best_lag > 0 else 0.0


def _cluster_feature_vectors(features: list[np.ndarray]) -> list[int]:
    normalized = np.vstack(features).astype(np.float32)
    if normalized.shape[0] == 1:
        return [0]

    mean = normalized.mean(axis=0, keepdims=True)
    std = normalized.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    normalized = (normalized - mean) / std

    centroids: list[np.ndarray] = []
    labels: list[int] = []
    for vector in normalized:
        if not centroids:
            centroids.append(vector.copy())
            labels.append(0)
            continue

        best_index = 0
        best_score = -1.0
        for index, centroid in enumerate(centroids):
            score = _cosine_similarity(vector, centroid)
            if score > best_score:
                best_score = score
                best_index = index

        if best_score >= MIN_CLUSTER_SIMILARITY or len(centroids) >= MAX_FALLBACK_SPEAKERS:
            labels.append(best_index)
            centroids[best_index] = ((centroids[best_index] * (labels.count(best_index))) + vector) / (labels.count(best_index) + 1)
        else:
            centroids.append(vector.copy())
            labels.append(len(centroids) - 1)

    return labels


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _merge_adjacent_segments(
    diarization_segments: list[DiarizationSegment],
) -> list[DiarizationSegment]:
    if not diarization_segments:
        return []
    merged = [deepcopy(diarization_segments[0])]
    for segment in diarization_segments[1:]:
        current = merged[-1]
        if segment.raw_speaker_id == current.raw_speaker_id and segment.start <= current.end + 0.05:
            current.end = max(current.end, segment.end)
            continue
        merged.append(deepcopy(segment))
    return merged
