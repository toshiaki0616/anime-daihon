from __future__ import annotations

from copy import deepcopy

from models.state import DiarizationSegment, RawTranscriptSegment, VadSegment


def refine_segment_timestamps(
    transcript_segments: list[RawTranscriptSegment],
    vad_segments: list[VadSegment],
    diarization_segments: list[DiarizationSegment],
) -> list[RawTranscriptSegment]:
    refined_segments: list[RawTranscriptSegment] = []
    for segment in transcript_segments:
        refined = deepcopy(segment)
        refined.raw_start = segment.raw_start or segment.start
        refined.raw_end = segment.raw_end or segment.end
        refined.refined_start = choose_display_start_time(segment, vad_segments, diarization_segments)
        refined.refined_end = choose_display_end_time(segment, vad_segments, diarization_segments)
        if refined.refined_end <= refined.refined_start:
            refined.refined_start = refined.raw_start
            refined.refined_end = refined.raw_end
        refined_segments.append(refined)
    return refined_segments


def choose_display_start_time(
    segment: RawTranscriptSegment,
    vad_segments: list[VadSegment],
    diarization_segments: list[DiarizationSegment],
) -> float:
    matched_vad = _find_covering_vad_segment(segment, vad_segments)
    raw_start = segment.raw_start or segment.start
    if matched_vad is not None:
        if matched_vad.start <= raw_start <= matched_vad.end:
            return raw_start
        return matched_vad.start

    matched_diarization = _find_dominant_diarization_overlap(segment, diarization_segments)
    if matched_diarization is not None:
        if matched_diarization.start <= raw_start <= matched_diarization.end:
            return raw_start
        return matched_diarization.start

    return raw_start


def choose_display_end_time(
    segment: RawTranscriptSegment,
    vad_segments: list[VadSegment],
    diarization_segments: list[DiarizationSegment],
) -> float:
    matched_vad = _find_covering_vad_segment(segment, vad_segments)
    raw_end = segment.raw_end or segment.end
    if matched_vad is not None:
        if matched_vad.start <= raw_end <= matched_vad.end:
            return raw_end
        return matched_vad.end

    matched_diarization = _find_dominant_diarization_overlap(segment, diarization_segments)
    if matched_diarization is not None:
        if matched_diarization.start <= raw_end <= matched_diarization.end:
            return raw_end
        return matched_diarization.end

    return raw_end


def _find_covering_vad_segment(
    segment: RawTranscriptSegment,
    vad_segments: list[VadSegment],
) -> VadSegment | None:
    best_match: VadSegment | None = None
    best_overlap = 0.0
    for vad_segment in vad_segments:
        overlap = _compute_overlap(segment.start, segment.end, vad_segment.start, vad_segment.end)
        if overlap <= 0:
            continue
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = vad_segment
    return best_match


def _find_dominant_diarization_overlap(
    segment: RawTranscriptSegment,
    diarization_segments: list[DiarizationSegment],
) -> DiarizationSegment | None:
    best_match: DiarizationSegment | None = None
    best_overlap = 0.0
    for diarization_segment in diarization_segments:
        overlap = _compute_overlap(segment.start, segment.end, diarization_segment.start, diarization_segment.end)
        if overlap <= 0:
            continue
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = diarization_segment
    return best_match


def _compute_overlap(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))
