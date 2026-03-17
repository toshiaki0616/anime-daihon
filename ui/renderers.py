from __future__ import annotations

from typing import Any

from core.state_ops import MAX_SPEAKER_SLOTS
from models.state import AppState, SpeakerProfile


def make_empty_state() -> dict[str, Any]:
    return AppState().to_dict()


def format_seconds(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def build_subtitle_rows(state: AppState) -> list[list[str]]:
    return [
        [
            segment.id,
            f"[{format_seconds(segment.start)}]",
            segment.display_speaker_name,
            segment.edited_text,
        ]
        for segment in state.subtitle_segments
    ]


def raw_label_for_speaker(speaker: SpeakerProfile) -> str:
    return speaker.display_name if speaker.display_name.startswith("話者") else ""


def build_speaker_slot_updates(state: AppState) -> list[dict[str, Any]]:
    speaker_choices = [(speaker.display_name, speaker.speaker_id) for speaker in state.speakers]
    slots: list[dict[str, Any]] = []

    for index in range(MAX_SPEAKER_SLOTS):
        if index >= len(state.speakers):
            slots.append(
                {
                    "visible": False,
                    "title": "",
                    "speaker_id": None,
                    "display_name": "",
                    "utterances_html": "",
                    "merge_choices": [],
                    "swap_choices": [],
                }
            )
            continue

        speaker = state.speakers[index]
        utterances = [
            segment
            for segment in state.subtitle_segments
            if segment.speaker_id == speaker.speaker_id
        ]
        utterance_lines = "".join(
            f"<li><strong>[{format_seconds(segment.start)}]</strong> {segment.edited_text}</li>"
            for segment in utterances
        )
        slots.append(
            {
                "visible": True,
                "title": f"{speaker.display_name}（{speaker.utterance_count}件）",
                "speaker_id": speaker.speaker_id,
                "display_name": "" if speaker.display_name.startswith("話者") else speaker.display_name,
                "utterances_html": f"<ul>{utterance_lines}</ul>",
                "merge_choices": [
                    choice for choice in speaker_choices if choice[1] != speaker.speaker_id
                ],
                "swap_choices": [
                    choice for choice in speaker_choices if choice[1] != speaker.speaker_id
                ],
            }
        )
    return slots


def format_status_box(message: str, kind: str = "info") -> str:
    colors = {
        "info": "#0f172a",
        "success": "#166534",
        "error": "#b91c1c",
    }
    return (
        "<div style='padding: 12px; border-radius: 10px; background: #f8fafc; "
        f"border: 1px solid #cbd5e1; color: {colors.get(kind, colors['info'])};'>{message}</div>"
    )
