from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from string import ascii_uppercase
from typing import Callable, Iterable

from models.state import AppState, Episode, PreprocessingResult, SpeakerProfile, SubtitleSegment, Work
from services.diarization import DiarizationSegment
from services.speaker_id import VoiceprintAssignment
from services.transcription import TranscriptionSegment

MAX_SPEAKER_SLOTS = 8
FALLBACK_SPEAKER_ID = "speaker_a"
FALLBACK_LABEL = "話者A"


def now_label() -> str:
    return datetime.now().replace(microsecond=0).isoformat(timespec="seconds")


def build_mock_app_state() -> AppState:
    work_one = Work(
        work_id="work_001",
        title="ONE PIECE",
        character_names=["ルフィ", "ゾロ", "ナミ"],
        created_at="2026-03-19T18:30:00",
        updated_at="2026-03-19T20:15:00",
        episodes=[
            build_mock_episode("episode_001", "第1話", "作業中", "2026-03-19T20:15:00"),
            build_empty_episode("episode_002", "第2話", "未整理", "2026-03-18T21:00:00"),
        ],
    )
    work_two = Work(
        work_id="work_002",
        title="鬼滅の刃",
        character_names=["炭治郎", "禰豆子", "善逸"],
        created_at="2026-03-15T14:00:00",
        updated_at="2026-03-18T19:20:00",
        episodes=[
            build_mock_episode("episode_003", "第1話", "完了", "2026-03-18T19:20:00"),
        ],
    )
    return AppState(
        works=sorted([work_one, work_two], key=lambda work: work.updated_at, reverse=True),
        current_page="work_list",
    )


def build_empty_episode(episode_id: str, title: str, status: str, updated_at: str) -> Episode:
    return Episode(
        episode_id=episode_id,
        title=title,
        status=status,
        updated_at=updated_at,
        subtitle_segments=[],
        speakers=[],
        merge_map={},
        speaker_label_map={},
    )


def build_mock_episode(episode_id: str, title: str, status: str, updated_at: str) -> Episode:
    episode = Episode(
        episode_id=episode_id,
        title=title,
        status=status,
        updated_at=updated_at,
        file_path="sample.mp4",
        wav_path="sample.wav",
        range_start="00:00:00",
        range_end="00:02:00",
        enhance_audio=False,
        subtitle_segments=[
            SubtitleSegment(
                id="seg_001",
                start=12.0,
                end=14.2,
                speaker_id="speaker_a",
                raw_label="話者A",
                display_name="話者A",
                original_text="そんなことできるわけないだろ",
                edited_text="そんなことできるわけないだろ",
            ),
            SubtitleSegment(
                id="seg_002",
                start=15.0,
                end=16.4,
                speaker_id="speaker_b",
                raw_label="話者B",
                display_name="話者B",
                original_text="……できるさ",
                edited_text="……できるさ",
            ),
            SubtitleSegment(
                id="seg_003",
                start=21.0,
                end=24.0,
                speaker_id="speaker_c",
                raw_label="話者C",
                display_name="話者C",
                original_text="今はまだ黙って進むしかない",
                edited_text="今はまだ黙って進むしかない",
            ),
            SubtitleSegment(
                id="seg_004",
                start=27.0,
                end=29.0,
                speaker_id="speaker_a",
                raw_label="話者A",
                display_name="話者A",
                original_text="聞こえたら合図してくれ",
                edited_text="聞こえたら合図してくれ",
            ),
        ],
        speakers=[],
        merge_map={},
        speaker_label_map={
            "speaker_0": "speaker_a",
            "speaker_1": "speaker_b",
            "speaker_2": "speaker_c",
        },
    )
    return sync_episode(episode)


def get_selected_work(state: AppState) -> Work | None:
    return next((work for work in state.works if work.work_id == state.selected_work_id), None)


def get_selected_episode(state: AppState) -> Episode | None:
    work = get_selected_work(state)
    if work is None:
        return None
    return next((episode for episode in work.episodes if episode.episode_id == state.selected_episode_id), None)


def _speaker_label_for_index(index: int) -> str:
    if 0 <= index < len(ascii_uppercase):
        return f"話者{ascii_uppercase[index]}"
    return f"話者{index + 1}"


def _speaker_id_for_index(index: int) -> str:
    if 0 <= index < len(ascii_uppercase):
        return f"speaker_{ascii_uppercase[index].lower()}"
    return f"speaker_{index + 1:02d}"


def _label_for_speaker_id(speaker_id: str) -> str:
    if speaker_id.startswith("speaker_") and len(speaker_id) == 9 and speaker_id[-1].isalpha():
        suffix = speaker_id[-1].upper()
        return f"話者{suffix}"
    if speaker_id.startswith("speaker_"):
        tail = speaker_id.split("_", 1)[1]
        if tail.isdigit():
            return f"話者{int(tail)}"
    return FALLBACK_LABEL


def _overlap_duration(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _segment_midpoint(start: float, end: float) -> float:
    return start + ((end - start) / 2.0)


def _nearest_speaker_for_segment(item: TranscriptionSegment, diarization_segments: list[DiarizationSegment]) -> str | None:
    if not diarization_segments:
        return None

    midpoint = _segment_midpoint(item.start, item.end)
    nearest: tuple[float, float, str] | None = None
    for diarization in diarization_segments:
        if diarization.start <= midpoint <= diarization.end:
            distance = 0.0
        else:
            distance = min(abs(midpoint - diarization.start), abs(midpoint - diarization.end))
        candidate = (distance, diarization.start, diarization.speaker)
        if nearest is None or candidate < nearest:
            nearest = candidate

    if nearest is None or nearest[0] > 1.5:
        return None
    return nearest[2]


def _assign_speaker_mapping(
    transcription_segments: list[TranscriptionSegment],
    diarization_segments: list[DiarizationSegment],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    assignments: list[dict[str, str]] = []
    speaker_first_seen: dict[str, float] = {}

    for item in transcription_segments:
        overlap_by_speaker: dict[str, float] = {}
        earliest_overlap: dict[str, float] = {}
        for diarization in diarization_segments:
            overlap = _overlap_duration(item.start, item.end, diarization.start, diarization.end)
            if overlap <= 0:
                continue
            overlap_by_speaker[diarization.speaker] = overlap_by_speaker.get(diarization.speaker, 0.0) + overlap
            earliest_overlap[diarization.speaker] = min(
                earliest_overlap.get(diarization.speaker, float("inf")),
                max(item.start, diarization.start),
            )

        dominant_raw = None
        if overlap_by_speaker:
            dominant_raw = min(
                overlap_by_speaker,
                key=lambda speaker: (
                    -overlap_by_speaker[speaker],
                    earliest_overlap.get(speaker, float("inf")),
                    speaker,
                ),
            )
        else:
            dominant_raw = _nearest_speaker_for_segment(item, diarization_segments)

        if dominant_raw is not None:
            speaker_first_seen.setdefault(dominant_raw, item.start)
            assignments.append({"raw_speaker": dominant_raw})
        else:
            assignments.append({"raw_speaker": "__fallback__"})

    ordered_raw_speakers = sorted(speaker_first_seen, key=lambda speaker: (speaker_first_seen[speaker], speaker))
    speaker_label_map = {
        raw_speaker: _speaker_id_for_index(index)
        for index, raw_speaker in enumerate(ordered_raw_speakers)
    }
    speaker_label_map["__fallback__"] = FALLBACK_SPEAKER_ID

    for assignment in assignments:
        normalized_id = speaker_label_map.get(assignment["raw_speaker"], FALLBACK_SPEAKER_ID)
        normalized_label = _label_for_speaker_id(normalized_id)
        assignment["speaker_id"] = normalized_id
        assignment["raw_label"] = normalized_label
        assignment["display_name"] = normalized_label

    normalized_map = {
        raw_speaker: speaker_id
        for raw_speaker, speaker_id in speaker_label_map.items()
        if raw_speaker != "__fallback__"
    }
    return assignments, normalized_map


def _preserve_edited_text(
    previous_segments: list[SubtitleSegment],
    index: int,
    original_text: str,
    start: float,
    end: float,
    default_edited_text: str,
) -> str:
    if index < len(previous_segments):
        previous = previous_segments[index]
        same_text = previous.original_text.strip() == original_text.strip()
        close_range = abs(previous.start - start) < 0.75 and abs(previous.end - end) < 0.75
        if same_text or close_range:
            return previous.edited_text

    for previous in previous_segments:
        if previous.original_text.strip() == original_text.strip():
            return previous.edited_text
    return default_edited_text


def _rebuild_speakers(
    segments: Iterable[SubtitleSegment],
    existing_profiles: dict[str, SpeakerProfile],
) -> list[SpeakerProfile]:
    grouped: dict[str, list[SubtitleSegment]] = {}
    for segment in segments:
        grouped.setdefault(segment.speaker_id, []).append(segment)

    speaker_ids = set(grouped.keys()) | {speaker_id for speaker_id, profile in existing_profiles.items() if profile.utterance_count == 0}
    speakers: list[SpeakerProfile] = []
    for speaker_id in speaker_ids:
        items = grouped.get(speaker_id, [])
        existing = existing_profiles.get(speaker_id)
        raw_label = items[0].raw_label if items else (existing.raw_label if existing else _label_for_speaker_id(speaker_id))
        display_name = existing.display_name if existing else raw_label
        speakers.append(
            SpeakerProfile(
                speaker_id=speaker_id,
                raw_label=raw_label,
                display_name=display_name,
                utterance_count=len(items),
                sample_texts=[item.edited_text for item in items[:3]],
            )
        )
    speakers.sort(key=lambda item: item.raw_label)
    return speakers


def _build_speaker_diagnostics(
    transcription_segments: list[TranscriptionSegment],
    diarization_segments: list[DiarizationSegment],
    subtitle_segments: list[SubtitleSegment],
) -> str:
    if not transcription_segments:
        return "話者分離診断: まだ字幕がありません。"

    lines = [f"話者分離診断: 字幕候補 {len(transcription_segments)}件"]
    if not diarization_segments:
        lines.append("話者候補は検出できず、話者Aへフォールバックしました。")
    else:
        durations: dict[str, float] = {}
        for item in diarization_segments:
            durations[item.speaker] = durations.get(item.speaker, 0.0) + max(0.0, item.end - item.start)
        ordered = sorted(durations.items(), key=lambda pair: (-pair[1], pair[0]))
        lines.append(f"話者候補: {len(ordered)}人")
        lines.extend([f"- {speaker}: {duration:.1f}秒" for speaker, duration in ordered[:5]])

    assigned_counts: dict[str, int] = {}
    for segment in subtitle_segments:
        assigned_counts[segment.raw_label] = assigned_counts.get(segment.raw_label, 0) + 1
    if assigned_counts:
        summary = ' / '.join(f"{label}:{count}件" for label, count in sorted(assigned_counts.items()))
        lines.append(f"字幕への割り当て: {summary}")
        if len(assigned_counts) == 1 and diarization_segments:
            lines.append("複数話者の可能性はありますが、割り当ては1人に寄っています。必要なら手動で話者を移動してください。")

    matched_voiceprints = [
        segment for segment in subtitle_segments
        if segment.voiceprint_character_name and segment.voiceprint_confidence > 0
    ]
    if matched_voiceprints:
        lines.append(f"声紋一致: {len(matched_voiceprints)}件")
        for segment in matched_voiceprints[:5]:
            lines.append(
                f"- [{segment.start:.1f}s] {segment.voiceprint_character_name} ({segment.voiceprint_confidence:.2f})"
            )

    return "`n".join(lines)


def _touch_work_for_episode(state: AppState, episode: Episode) -> None:
    work = get_selected_work(state)
    if work is not None:
        work.updated_at = episode.updated_at


def record_preprocessing_result(state: AppState, result: PreprocessingResult) -> AppState:
    next_state = deepcopy(state)
    next_state.last_preprocessing_source_path = result.source_path
    next_state.last_preprocessing_wav_path = result.normalized_wav_path
    next_state.last_processed_range_start = result.processed_range_start
    next_state.last_processed_range_end = result.processed_range_end
    next_state.last_vad_segments = deepcopy(result.vad_segments)
    next_state.last_preprocessing_status = "fallback" if result.fallback_used else "success"
    next_state.last_preprocessing_error = ""
    next_state.last_debug_output_path = result.debug_paths.get("vad_segments", "")
    next_state.last_vad_fallback_used = result.fallback_used
    return next_state


def record_preprocessing_error(state: AppState, source_path: str, error_message: str) -> AppState:
    next_state = deepcopy(state)
    next_state.last_preprocessing_source_path = source_path
    next_state.last_preprocessing_wav_path = ""
    next_state.last_processed_range_start = 0.0
    next_state.last_processed_range_end = 0.0
    next_state.last_vad_segments = []
    next_state.last_preprocessing_status = "error"
    next_state.last_preprocessing_error = error_message
    next_state.last_debug_output_path = ""
    next_state.last_vad_fallback_used = False
    return next_state


def sync_episode(episode: Episode) -> Episode:
    existing_profiles = {speaker.speaker_id: speaker for speaker in episode.speakers}
    episode.speakers = _rebuild_speakers(episode.subtitle_segments, existing_profiles)
    display_names = {speaker.speaker_id: speaker.display_name for speaker in episode.speakers}
    raw_labels = {speaker.speaker_id: speaker.raw_label for speaker in episode.speakers}
    for segment in episode.subtitle_segments:
        segment.raw_label = raw_labels.get(segment.speaker_id, _label_for_speaker_id(segment.speaker_id))
        segment.display_name = display_names.get(segment.speaker_id, segment.raw_label)
    if episode.subtitle_segments and episode.status == "未整理":
        episode.status = "作業中"
    episode.updated_at = now_label()
    return episode


def _next_speaker_identity(episode: Episode) -> tuple[str, str]:
    used_ids = {speaker.speaker_id for speaker in episode.speakers}
    for index in range(len(ascii_uppercase)):
        speaker_id = _speaker_id_for_index(index)
        if speaker_id not in used_ids:
            return speaker_id, _speaker_label_for_index(index)
    speaker_number = len(episode.speakers) + 1
    return f"speaker_{speaker_number:02d}", f"話者{speaker_number}"


def _resolve_manual_speaker_id(episode: Episode, label_text: str) -> str | None:
    normalized = label_text.strip()
    if not normalized:
        return None

    # Subtitle table shows confidence prefixes for display only.
    # Strip them before resolving speaker identities so inline edits
    # do not create "UNKNOWN | ..." pseudo speakers on every refresh.
    while " | " in normalized:
        prefix, remainder = normalized.split(" | ", 1)
        if prefix.strip() not in {"UNKNOWN", "MEDIUM"}:
            break
        normalized = remainder.strip()

    if not normalized:
        return None

    for speaker in episode.speakers:
        if normalized in {speaker.raw_label, speaker.display_name}:
            return speaker.speaker_id

    if normalized.startswith("話者") and len(normalized) == 3 and normalized[-1].upper() in ascii_uppercase:
        speaker_id = f"speaker_{normalized[-1].lower()}"
        for speaker in episode.speakers:
            if speaker.speaker_id == speaker_id:
                return speaker_id
        episode.speakers.append(
            SpeakerProfile(
                speaker_id=speaker_id,
                raw_label=normalized,
                display_name=normalized,
                utterance_count=0,
                sample_texts=[],
            )
        )
        return speaker_id

    speaker_id, raw_label = _next_speaker_identity(episode)
    episode.speakers.append(
        SpeakerProfile(
            speaker_id=speaker_id,
            raw_label=raw_label,
            display_name=normalized,
            utterance_count=0,
            sample_texts=[],
        )
    )
    return speaker_id


def create_work(state: AppState) -> AppState:
    next_state = deepcopy(state)
    work_number = len(next_state.works) + 1
    timestamp = now_label()
    work = Work(
        work_id=f"work_{work_number:03d}",
        title=f"新しい作品{work_number}",
        character_names=[],
        created_at=timestamp,
        updated_at=timestamp,
        episodes=[],
    )
    next_state.works.insert(0, work)
    next_state.selected_work_id = work.work_id
    next_state.current_page = "work_detail"
    return next_state


def update_work_title(state: AppState, title: str) -> AppState:
    next_state = deepcopy(state)
    work = get_selected_work(next_state)
    if work is None:
        return next_state
    trimmed = title.strip()
    if trimmed:
        work.title = trimmed
        work.updated_at = now_label()
    return next_state


def create_episode(state: AppState) -> AppState:
    next_state = deepcopy(state)
    work = get_selected_work(next_state)
    if work is None:
        return next_state
    episode_number = len(work.episodes) + 1
    episode = build_empty_episode(
        episode_id=f"{work.work_id}_ep_{episode_number:03d}",
        title=f"第{episode_number}話",
        status="未整理",
        updated_at=now_label(),
    )
    work.episodes.insert(0, episode)
    work.updated_at = episode.updated_at
    next_state.selected_episode_id = episode.episode_id
    next_state.current_page = "episode_editor"
    return next_state


def add_speaker_profile(state: AppState) -> AppState:
    next_state = deepcopy(state)
    episode = get_selected_episode(next_state)
    if episode is None:
        return next_state
    speaker_id, raw_label = _next_speaker_identity(episode)
    episode.speakers.append(
        SpeakerProfile(
            speaker_id=speaker_id,
            raw_label=raw_label,
            display_name=raw_label,
            utterance_count=0,
            sample_texts=[],
        )
    )
    next_state.selected_speaker_id = speaker_id
    sync_episode(episode)
    _touch_work_for_episode(next_state, episode)
    return next_state


def apply_transcription_segments(
    state: AppState,
    file_path: str,
    wav_path: str,
    range_start: str,
    range_end: str,
    enhance_audio: bool,
    transcription_segments: list[TranscriptionSegment],
    diarization_segments: list[DiarizationSegment] | None = None,
    text_postprocessor: Callable[[str], str] | None = None,
    voiceprint_assignments: list[VoiceprintAssignment | None] | None = None,
) -> AppState:
    next_state = deepcopy(state)
    episode = get_selected_episode(next_state)
    if episode is None:
        return next_state

    previous_segments = deepcopy(episode.subtitle_segments)
    diarization_segments = diarization_segments or []
    voiceprint_assignments = voiceprint_assignments or []
    assignments, speaker_label_map = _assign_speaker_mapping(transcription_segments, diarization_segments)

    subtitle_segments: list[SubtitleSegment] = []
    for index, item in enumerate(transcription_segments):
        text = item.text.strip()
        if not text:
            continue
        edited_text = text_postprocessor(text) if text_postprocessor else text
        assignment = assignments[index] if index < len(assignments) else {
            "speaker_id": FALLBACK_SPEAKER_ID,
            "raw_label": FALLBACK_LABEL,
            "display_name": FALLBACK_LABEL,
        }
        voiceprint_assignment = voiceprint_assignments[index] if index < len(voiceprint_assignments) else None
        speaker_id = assignment["speaker_id"]
        raw_label = assignment["raw_label"]
        display_name = assignment["display_name"]
        voiceprint_profile_id = ""
        voiceprint_character_name = ""
        voiceprint_confidence = 0.0
        if voiceprint_assignment is not None:
            speaker_id = f"voiceprint_{voiceprint_assignment.profile_id}"
            raw_label = voiceprint_assignment.character_name
            display_name = voiceprint_assignment.character_name
            voiceprint_profile_id = voiceprint_assignment.profile_id
            voiceprint_character_name = voiceprint_assignment.character_name
            voiceprint_confidence = voiceprint_assignment.confidence
        subtitle_segments.append(
            SubtitleSegment(
                id=f"seg_{len(subtitle_segments) + 1:03d}",
                start=item.start,
                end=item.end,
                source_start=item.source_start if item.source_start is not None else item.start,
                source_end=item.source_end if item.source_end is not None else item.end,
                speaker_id=speaker_id,
                raw_label=raw_label,
                display_name=display_name,
                original_text=text,
                edited_text=_preserve_edited_text(
                    previous_segments,
                    index,
                    text,
                    item.start,
                    item.end,
                    edited_text,
                ),
                voiceprint_profile_id=voiceprint_profile_id,
                voiceprint_character_name=voiceprint_character_name,
                voiceprint_confidence=voiceprint_confidence,
            )
        )

    episode.file_path = file_path
    episode.wav_path = wav_path
    episode.range_start = range_start
    episode.range_end = range_end
    episode.enhance_audio = enhance_audio
    episode.subtitle_segments = subtitle_segments
    episode.speaker_label_map = speaker_label_map if subtitle_segments else {}
    episode.speaker_diagnostics = _build_speaker_diagnostics(transcription_segments, diarization_segments, subtitle_segments)
    episode.merge_map = {}
    episode.status = "作業中" if subtitle_segments else episode.status
    sync_episode(episode)
    _touch_work_for_episode(next_state, episode)
    return next_state


def update_character_names(state: AppState, raw_text: str) -> AppState:
    next_state = deepcopy(state)
    work = get_selected_work(next_state)
    if work is None:
        return next_state
    work.character_names = [
        item.strip()
        for item in raw_text.replace("、", ",").replace("\n", ",").split(",")
        if item.strip()
    ]
    work.updated_at = now_label()
    return next_state


def apply_subtitle_edits(state: AppState, rows: list[list[str]]) -> AppState:
    next_state = deepcopy(state)
    episode = get_selected_episode(next_state)
    if episode is None:
        return next_state
    row_map = {row[0]: row for row in rows if len(row) >= 5}
    for segment in episode.subtitle_segments:
        row = row_map.get(segment.id)
        if not row:
            continue
        manual_speaker_id = _resolve_manual_speaker_id(episode, str(row[3]))
        if manual_speaker_id:
            segment.speaker_id = manual_speaker_id
        segment.edited_text = str(row[4])
    sync_episode(episode)
    _touch_work_for_episode(next_state, episode)
    return next_state


def apply_voiceprint_assignments_to_episode(
    state: AppState,
    voiceprint_assignments: list[VoiceprintAssignment | None],
) -> AppState:
    next_state = deepcopy(state)
    episode = get_selected_episode(next_state)
    if episode is None or not episode.subtitle_segments:
        return next_state

    for index, segment in enumerate(episode.subtitle_segments):
        assignment = voiceprint_assignments[index] if index < len(voiceprint_assignments) else None
        if assignment is None:
            segment.voiceprint_profile_id = ""
            segment.voiceprint_character_name = ""
            segment.voiceprint_confidence = 0.0
            continue

        segment.speaker_id = f"voiceprint_{assignment.profile_id}"
        segment.raw_label = assignment.character_name
        segment.display_name = assignment.character_name
        segment.voiceprint_profile_id = assignment.profile_id
        segment.voiceprint_character_name = assignment.character_name
        segment.voiceprint_confidence = assignment.confidence

    sync_episode(episode)
    _touch_work_for_episode(next_state, episode)
    return next_state


def rename_speaker(state: AppState, speaker_id: str, new_name: str) -> AppState:
    next_state = deepcopy(state)
    episode = get_selected_episode(next_state)
    if episode is None:
        return next_state
    target = next((speaker for speaker in episode.speakers if speaker.speaker_id == speaker_id), None)
    if target is None:
        return next_state
    target.display_name = new_name.strip() or target.raw_label
    sync_episode(episode)
    _touch_work_for_episode(next_state, episode)
    return next_state


def merge_speakers(state: AppState, source_speaker_id: str, target_speaker_id: str) -> AppState:
    next_state = deepcopy(state)
    episode = get_selected_episode(next_state)
    if episode is None or source_speaker_id == target_speaker_id:
        return next_state

    source_exists = any(speaker.speaker_id == source_speaker_id for speaker in episode.speakers)
    target_exists = any(speaker.speaker_id == target_speaker_id for speaker in episode.speakers)
    if not source_exists or not target_exists:
        return next_state

    for segment in episode.subtitle_segments:
        if segment.speaker_id == source_speaker_id:
            segment.speaker_id = target_speaker_id
    episode.merge_map[source_speaker_id] = target_speaker_id
    episode.speakers = [speaker for speaker in episode.speakers if speaker.speaker_id != source_speaker_id]
    if next_state.selected_speaker_id == source_speaker_id:
        next_state.selected_speaker_id = target_speaker_id
    sync_episode(episode)
    _touch_work_for_episode(next_state, episode)
    return next_state


def move_segment_to_speaker(state: AppState, segment_id: str, target_speaker_id: str) -> AppState:
    next_state = deepcopy(state)
    episode = get_selected_episode(next_state)
    if episode is None or not segment_id or not target_speaker_id:
        return next_state

    target_exists = any(speaker.speaker_id == target_speaker_id for speaker in episode.speakers)
    if not target_exists:
        return next_state

    target_segment = next((segment for segment in episode.subtitle_segments if segment.id == segment_id), None)
    if target_segment is None:
        return next_state

    target_segment.speaker_id = target_speaker_id
    sync_episode(episode)
    _touch_work_for_episode(next_state, episode)
    return next_state


def update_segment_text(state: AppState, segment_id: str, new_text: str) -> AppState:
    next_state = deepcopy(state)
    episode = get_selected_episode(next_state)
    if episode is None or not segment_id:
        return next_state

    target_segment = next((segment for segment in episode.subtitle_segments if segment.id == segment_id), None)
    if target_segment is None:
        return next_state

    target_segment.edited_text = new_text.strip() if new_text.strip() else target_segment.edited_text
    sync_episode(episode)
    _touch_work_for_episode(next_state, episode)
    return next_state


def delete_speaker_profile(state: AppState, speaker_id: str) -> AppState:
    next_state = deepcopy(state)
    episode = get_selected_episode(next_state)
    if episode is None or not speaker_id:
        return next_state

    target = next((speaker for speaker in episode.speakers if speaker.speaker_id == speaker_id), None)
    if target is None or target.utterance_count > 0:
        return next_state

    episode.speakers = [speaker for speaker in episode.speakers if speaker.speaker_id != speaker_id]
    if next_state.selected_speaker_id == speaker_id:
        next_state.selected_speaker_id = ""
    sync_episode(episode)
    _touch_work_for_episode(next_state, episode)
    return next_state


def swap_speakers(state: AppState, left_speaker_id: str, right_speaker_id: str) -> AppState:
    next_state = deepcopy(state)
    episode = get_selected_episode(next_state)
    if episode is None or left_speaker_id == right_speaker_id:
        return next_state

    left_exists = any(speaker.speaker_id == left_speaker_id for speaker in episode.speakers)
    right_exists = any(speaker.speaker_id == right_speaker_id for speaker in episode.speakers)
    if not left_exists or not right_exists:
        return next_state

    for segment in episode.subtitle_segments:
        if segment.speaker_id == left_speaker_id:
            segment.speaker_id = "__swap_tmp__"
        elif segment.speaker_id == right_speaker_id:
            segment.speaker_id = left_speaker_id
    for segment in episode.subtitle_segments:
        if segment.speaker_id == "__swap_tmp__":
            segment.speaker_id = right_speaker_id

    sync_episode(episode)
    _touch_work_for_episode(next_state, episode)
    return next_state

