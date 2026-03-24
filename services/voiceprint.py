from models.state import SubtitleSegment, VoiceprintCandidate


MIN_CANDIDATE_DURATION = 2.5
MAX_CANDIDATE_DURATION = 10.0
MIN_CANDIDATE_TEXT_LENGTH = 6
SHORT_ACKNOWLEDGEMENTS = {
    "\u3048",
    "\u3048?",
    "\u3042\u3042",
    "\u3046\u3093",
    "\u306f\u3044",
    "\u304a\u3046",
    "\u3042\u3063",
    "\u3048\u3063",
}


def build_voiceprint_candidates_from_clean_segments_only(
    episode_id: str,
    subtitle_segments: list[SubtitleSegment],
) -> list[VoiceprintCandidate]:
    candidates: list[VoiceprintCandidate] = []
    for segment in subtitle_segments:
        is_good, flags = is_good_voiceprint_candidate(segment)
        if not is_good:
            continue
        candidates.append(
            build_voiceprint_candidate_metadata(
                episode_id=episode_id,
                subtitle_segment=segment,
                candidate_index=len(candidates) + 1,
                confidence_flags=flags,
            )
        )
    return candidates


def is_good_voiceprint_candidate(segment: SubtitleSegment) -> tuple[bool, list[str]]:
    flags: list[str] = []
    clip_start = segment.refined_start or segment.start
    clip_end = segment.refined_end or segment.end
    duration = max(0.0, clip_end - clip_start)
    cleaned_text = "".join((segment.edited_text or "").split())

    if duration < MIN_CANDIDATE_DURATION:
        return False, ["too_short"]
    if duration > MAX_CANDIDATE_DURATION:
        return False, ["too_long"]
    if len(cleaned_text) < MIN_CANDIDATE_TEXT_LENGTH:
        return False, ["text_too_short"]
    if cleaned_text in SHORT_ACKNOWLEDGEMENTS:
        return False, ["acknowledgement"]

    if segment.refined_start and segment.raw_start:
        refinement_shift = abs(segment.refined_start - segment.raw_start)
        if refinement_shift > 1.25:
            flags.append("large_refinement_shift")

    if segment.refined_end and segment.refined_start:
        if (segment.refined_end - segment.refined_start) < 3.5:
            flags.append("short_but_acceptable")

    if (segment.display_name or "").startswith("\u8a71\u8005"):
        flags.append("manual_naming_pending")

    return True, flags


def build_voiceprint_candidate_metadata(
    episode_id: str,
    subtitle_segment: SubtitleSegment,
    candidate_index: int,
    confidence_flags: list[str],
) -> VoiceprintCandidate:
    clip_start = subtitle_segment.refined_start or subtitle_segment.start
    clip_end = subtitle_segment.refined_end or subtitle_segment.end
    return VoiceprintCandidate(
        candidate_id=f"vpc_{candidate_index:02d}",
        episode_id=episode_id,
        source_segment_id=subtitle_segment.id,
        speaker_id=subtitle_segment.speaker_id,
        clip_start=clip_start,
        clip_end=clip_end,
        transcript_text=subtitle_segment.edited_text,
        duration=max(0.0, clip_end - clip_start),
        confidence_flags=list(confidence_flags),
    )
