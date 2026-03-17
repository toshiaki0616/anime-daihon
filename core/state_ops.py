from __future__ import annotations

from typing import Iterable

from models.state import AppState, SpeakerProfile, SubtitleSegment

MAX_SPEAKER_SLOTS = 4


def _rebuild_speakers(
    segments: Iterable[SubtitleSegment],
    existing_names: dict[str, str],
) -> list[SpeakerProfile]:
    speakers_by_id: dict[str, list[SubtitleSegment]] = {}
    for segment in segments:
        speakers_by_id.setdefault(segment.speaker_id, []).append(segment)

    speakers: list[SpeakerProfile] = []
    for speaker_id, speaker_segments in speakers_by_id.items():
        raw_label = speaker_segments[0].raw_speaker_label
        speakers.append(
            SpeakerProfile(
                speaker_id=speaker_id,
                display_name=existing_names.get(speaker_id, raw_label),
                utterance_count=len(speaker_segments),
                sample_texts=[segment.edited_text for segment in speaker_segments[:3]],
            )
        )

    speakers.sort(key=lambda speaker: speaker.speaker_id)
    return speakers


def build_mock_state(
    file_path: str = "",
    range_start: str = "",
    range_end: str = "",
    enhance_audio: bool = False,
) -> AppState:
    speakers = [
        SpeakerProfile("speaker_a", "話者A"),
        SpeakerProfile("speaker_b", "話者B"),
        SpeakerProfile("speaker_c", "話者C"),
    ]
    segments = [
        SubtitleSegment(
            id="seg_001",
            start=12.0,
            end=14.2,
            speaker_id="speaker_a",
            raw_speaker_label="話者A",
            display_speaker_name="話者A",
            original_text="そんなことできるわけないだろ",
            edited_text="そんなことできるわけないだろ",
        ),
        SubtitleSegment(
            id="seg_002",
            start=15.0,
            end=16.4,
            speaker_id="speaker_b",
            raw_speaker_label="話者B",
            display_speaker_name="話者B",
            original_text="……できるさ",
            edited_text="……できるさ",
        ),
        SubtitleSegment(
            id="seg_003",
            start=21.0,
            end=24.0,
            speaker_id="speaker_c",
            raw_speaker_label="話者C",
            display_speaker_name="話者C",
            original_text="今はまだ黙って進むしかない",
            edited_text="今はまだ黙って進むしかない",
        ),
        SubtitleSegment(
            id="seg_004",
            start=27.0,
            end=29.0,
            speaker_id="speaker_a",
            raw_speaker_label="話者A",
            display_speaker_name="話者A",
            original_text="聞こえたら合図してくれ",
            edited_text="聞こえたら合図してくれ",
        ),
        SubtitleSegment(
            id="seg_005",
            start=31.0,
            end=33.0,
            speaker_id="speaker_b",
            raw_speaker_label="話者B",
            display_speaker_name="話者B",
            original_text="わかった、ここで待つ",
            edited_text="わかった、ここで待つ",
        ),
    ]

    return sync_state(
        AppState(
            file_path=file_path,
            range_start=range_start,
            range_end=range_end,
            enhance_audio=enhance_audio,
            subtitle_segments=segments,
            speakers=speakers,
            merge_map={},
        )
    )


def sync_state(state: AppState) -> AppState:
    existing_names = {speaker.speaker_id: speaker.display_name for speaker in state.speakers}
    state.speakers = _rebuild_speakers(state.subtitle_segments, existing_names)

    display_names = {speaker.speaker_id: speaker.display_name for speaker in state.speakers}
    for segment in state.subtitle_segments:
        segment.display_speaker_name = display_names.get(segment.speaker_id, segment.raw_speaker_label)
    return state


def rename_speaker(state: AppState, speaker_id: str, new_name: str) -> AppState:
    fallback_label = next(
        (segment.raw_speaker_label for segment in state.subtitle_segments if segment.speaker_id == speaker_id),
        "",
    )
    next_name = new_name.strip() or fallback_label

    for speaker in state.speakers:
        if speaker.speaker_id == speaker_id:
            speaker.display_name = next_name
            break
    return sync_state(state)


def merge_speakers(state: AppState, source_speaker_id: str, target_speaker_id: str) -> AppState:
    if source_speaker_id == target_speaker_id:
        return state

    for segment in state.subtitle_segments:
        if segment.speaker_id == source_speaker_id:
            segment.speaker_id = target_speaker_id
            state.merge_map[source_speaker_id] = target_speaker_id
    return sync_state(state)


def swap_speakers(state: AppState, left_speaker_id: str, right_speaker_id: str) -> AppState:
    if left_speaker_id == right_speaker_id:
        return state

    left_name = ""
    right_name = ""
    for speaker in state.speakers:
        if speaker.speaker_id == left_speaker_id:
            left_name = speaker.display_name
        if speaker.speaker_id == right_speaker_id:
            right_name = speaker.display_name

    for segment in state.subtitle_segments:
        if segment.speaker_id == left_speaker_id:
            segment.speaker_id = "__swap_tmp__"
        elif segment.speaker_id == right_speaker_id:
            segment.speaker_id = left_speaker_id

    for segment in state.subtitle_segments:
        if segment.speaker_id == "__swap_tmp__":
            segment.speaker_id = right_speaker_id

    synced_state = sync_state(state)
    for speaker in synced_state.speakers:
        if speaker.speaker_id == left_speaker_id:
            speaker.display_name = right_name or speaker.display_name
        elif speaker.speaker_id == right_speaker_id:
            speaker.display_name = left_name or speaker.display_name
    return sync_state(synced_state)


def apply_subtitle_edits(state: AppState, rows: list[list[str]]) -> AppState:
    row_map = {row[0]: row for row in rows if len(row) >= 4}
    for segment in state.subtitle_segments:
        row = row_map.get(segment.id)
        if row:
            segment.edited_text = row[3]
    return sync_state(state)
