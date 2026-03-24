from __future__ import annotations

from typing import Any

from core.state_ops import MAX_SPEAKER_SLOTS, get_selected_episode, get_selected_work
from models.state import AppState


def make_empty_state() -> dict[str, Any]:
    return AppState().to_dict()


def format_seconds(seconds: float) -> str:
    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    centiseconds = int(round((seconds - whole_seconds) * 100))
    if centiseconds >= 100:
        sec += 1
        centiseconds = 0
    if sec >= 60:
        minutes += 1
        sec = 0
    if minutes >= 60:
        hours += 1
        minutes = 0
    return f"{hours:02d}:{minutes:02d}:{sec:02d}.{centiseconds:02d}"


def format_timestamp(value: str) -> str:
    return value.replace("T", " ") if value else "-"


def parse_time_offset_seconds(value: str) -> float:
    text = (value or "").strip()
    if not text:
        return 0.0
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return (hours * 3600) + (minutes * 60) + seconds
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return (minutes * 60) + seconds
        return float(text)
    except ValueError:
        return 0.0


def build_work_rows(state: AppState) -> list[list[str]]:
    return [
        [work.title, str(len(work.episodes)), format_timestamp(work.updated_at)]
        for work in state.works
    ]


def build_episode_rows(state: AppState) -> list[list[str]]:
    work = get_selected_work(state)
    if work is None:
        return []
    return [
        [episode.title, episode.status, format_timestamp(episode.updated_at)]
        for episode in sorted(work.episodes, key=lambda item: item.updated_at, reverse=True)
    ]


def build_subtitle_rows(state: AppState) -> list[list[str]]:
    episode = get_selected_episode(state)
    if episode is None:
        return []
    time_offset = parse_time_offset_seconds(episode.range_start)
    return [
        [
            segment.id,
            f"[{format_seconds(time_offset + _segment_display_start(segment))}]",
            f"{max(0.0, _segment_display_end(segment) - _segment_display_start(segment)):.2f}s",
            _build_segment_speaker_label(segment),
            segment.edited_text,
        ]
        for segment in episode.subtitle_segments
    ]


def _build_segment_speaker_label(segment) -> str:
    label = segment.display_name
    if segment.voiceprint_profile_id:
        if segment.voiceprint_confidence < 0.75:
            return f"MEDIUM | {label}"
        return label
    return label


def _segment_display_start(segment) -> float:
    return float(segment.refined_start or segment.source_start or segment.start or 0.0)


def _segment_display_end(segment) -> float:
    return float(segment.refined_end or segment.source_end or segment.end or 0.0)


def build_speaker_list_payloads(state: AppState) -> list[dict[str, Any]]:
    episode = get_selected_episode(state)
    empty_payload = {
        "visible": False,
        "label": "",
        "speaker_id": None,
        "variant": "secondary",
    }
    if episode is None:
        return [empty_payload.copy() for _ in range(MAX_SPEAKER_SLOTS)]

    payloads: list[dict[str, Any]] = []
    for index in range(MAX_SPEAKER_SLOTS):
        if index >= len(episode.speakers):
            payloads.append(empty_payload.copy())
            continue
        speaker = episode.speakers[index]
        name_label = speaker.display_name or speaker.raw_label
        count_label = f"{name_label} ({speaker.utterance_count}件)"
        payloads.append(
            {
                "visible": True,
                "label": count_label,
                "speaker_id": speaker.speaker_id,
                "variant": "primary" if speaker.speaker_id == state.selected_speaker_id else "secondary",
            }
        )
    return payloads


def build_speaker_detail_payload(state: AppState) -> dict[str, Any]:
    episode = get_selected_episode(state)
    empty_payload = {
        "title": "### 話者を選択してください",
        "raw_label": "",
        "speaker_id": None,
        "display_name": "",
        "utterances_html": "<div class='speaker-empty'>左の話者を選ぶと、ここにセリフ一覧と編集内容が表示されます。</div>",
        "merge_choices": [],
        "swap_choices": [],
        "move_segment_choices": [],
        "move_target_choices": [],
        "utterance_choices": [],
        "can_delete": False,
    }
    if episode is None or not state.selected_speaker_id:
        return empty_payload

    speaker = next((item for item in episode.speakers if item.speaker_id == state.selected_speaker_id), None)
    if speaker is None:
        return empty_payload

    lines = [
        f"<li><span class='speaker-time'>[{format_seconds(parse_time_offset_seconds(episode.range_start) + _segment_display_start(segment))}]</span><span>{segment.edited_text}</span></li>"
        for segment in episode.subtitle_segments
        if segment.speaker_id == speaker.speaker_id
    ]
    sample_lines = "".join(lines) if lines else "<li class='speaker-empty'>まだこの話者にセリフはありません。</li>"
    speaker_choices = [(item.display_name or item.raw_label, item.speaker_id) for item in episode.speakers]
    move_segment_choices = [
        (f"[{format_seconds(parse_time_offset_seconds(episode.range_start) + _segment_display_start(segment))}] {segment.edited_text[:50]}", segment.id)
        for segment in episode.subtitle_segments
        if segment.speaker_id == speaker.speaker_id
    ]
    visible_name = speaker.display_name if speaker.display_name != speaker.raw_label else ""
    return {
        "title": f"### {speaker.raw_label}（{speaker.utterance_count}件）",
        "raw_label": speaker.raw_label,
        "speaker_id": speaker.speaker_id,
        "display_name": visible_name,
        "utterances_html": f"<div class='speaker-sample-list'><div class='speaker-sample-caption'>セリフ一覧</div><ul>{sample_lines}</ul></div>",
        "merge_choices": [choice for choice in speaker_choices if choice[1] != speaker.speaker_id],
        "swap_choices": [choice for choice in speaker_choices if choice[1] != speaker.speaker_id],
        "move_segment_choices": move_segment_choices,
        "move_target_choices": [choice for choice in speaker_choices if choice[1] != speaker.speaker_id],
        "utterance_choices": move_segment_choices,
        "can_delete": speaker.utterance_count == 0,
    }


def format_status_box(message: str, kind: str = "info") -> str:
    colors = {
        "info": "#dbeafe",
        "success": "#bbf7d0",
        "error": "#fecaca",
    }
    border_colors = {
        "info": "#1d4ed8",
        "success": "#15803d",
        "error": "#dc2626",
    }
    return (
        "<div style='padding: 12px 14px; border-radius: 12px; background: #111827; "
        f"border: 1px solid {border_colors.get(kind, border_colors['info'])}; color: {colors.get(kind, colors['info'])};'>{message}</div>"
    )
