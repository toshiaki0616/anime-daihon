from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SubtitleSegment:
    id: str
    start: float
    end: float
    speaker_id: str
    raw_speaker_label: str
    display_speaker_name: str
    original_text: str
    edited_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubtitleSegment":
        return cls(**data)


@dataclass
class SpeakerProfile:
    speaker_id: str
    display_name: str
    utterance_count: int = 0
    sample_texts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeakerProfile":
        return cls(**data)


@dataclass
class AppState:
    file_path: str = ""
    wav_path: str = ""
    range_start: str = ""
    range_end: str = ""
    enhance_audio: bool = False
    subtitle_segments: list[SubtitleSegment] = field(default_factory=list)
    speakers: list[SpeakerProfile] = field(default_factory=list)
    merge_map: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "wav_path": self.wav_path,
            "range_start": self.range_start,
            "range_end": self.range_end,
            "enhance_audio": self.enhance_audio,
            "subtitle_segments": [segment.to_dict() for segment in self.subtitle_segments],
            "speakers": [speaker.to_dict() for speaker in self.speakers],
            "merge_map": dict(self.merge_map),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppState":
        return cls(
            file_path=data.get("file_path", ""),
            wav_path=data.get("wav_path", ""),
            range_start=data.get("range_start", ""),
            range_end=data.get("range_end", ""),
            enhance_audio=data.get("enhance_audio", False),
            subtitle_segments=[
                SubtitleSegment.from_dict(segment)
                for segment in data.get("subtitle_segments", [])
            ],
            speakers=[
                SpeakerProfile.from_dict(speaker)
                for speaker in data.get("speakers", [])
            ],
            merge_map=data.get("merge_map", {}),
        )
