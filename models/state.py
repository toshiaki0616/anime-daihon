from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SubtitleSegment:
    id: str
    start: float
    end: float
    speaker_id: str
    raw_label: str
    display_name: str
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
    raw_label: str
    display_name: str
    utterance_count: int = 0
    sample_texts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeakerProfile":
        return cls(**data)


@dataclass
class Episode:
    episode_id: str
    title: str
    status: str
    updated_at: str
    file_path: str = ""
    wav_path: str = ""
    range_start: str = ""
    range_end: str = ""
    enhance_audio: bool = False
    subtitle_segments: list[SubtitleSegment] = field(default_factory=list)
    speakers: list[SpeakerProfile] = field(default_factory=list)
    merge_map: dict[str, str] = field(default_factory=dict)
    speaker_label_map: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "title": self.title,
            "status": self.status,
            "updated_at": self.updated_at,
            "file_path": self.file_path,
            "wav_path": self.wav_path,
            "range_start": self.range_start,
            "range_end": self.range_end,
            "enhance_audio": self.enhance_audio,
            "subtitle_segments": [segment.to_dict() for segment in self.subtitle_segments],
            "speakers": [speaker.to_dict() for speaker in self.speakers],
            "merge_map": dict(self.merge_map),
            "speaker_label_map": dict(self.speaker_label_map),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        return cls(
            episode_id=data.get("episode_id", ""),
            title=data.get("title", ""),
            status=data.get("status", "未整理"),
            updated_at=data.get("updated_at", ""),
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
            speaker_label_map=data.get("speaker_label_map", {}),
        )


@dataclass
class Work:
    work_id: str
    title: str
    character_names: list[str]
    created_at: str
    updated_at: str
    episodes: list[Episode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "title": self.title,
            "character_names": list(self.character_names),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Work":
        return cls(
            work_id=data.get("work_id", ""),
            title=data.get("title", ""),
            character_names=data.get("character_names", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            episodes=[Episode.from_dict(episode) for episode in data.get("episodes", [])],
        )


@dataclass
class AppState:
    works: list[Work] = field(default_factory=list)
    current_page: str = "work_list"
    selected_work_id: str = ""
    selected_episode_id: str = ""
    selected_speaker_id: str = ""
    show_character_manager: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "works": [work.to_dict() for work in self.works],
            "current_page": self.current_page,
            "selected_work_id": self.selected_work_id,
            "selected_episode_id": self.selected_episode_id,
            "selected_speaker_id": self.selected_speaker_id,
            "show_character_manager": self.show_character_manager,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppState":
        return cls(
            works=[Work.from_dict(work) for work in data.get("works", [])],
            current_page=data.get("current_page", "work_list"),
            selected_work_id=data.get("selected_work_id", ""),
            selected_episode_id=data.get("selected_episode_id", ""),
            selected_speaker_id=data.get("selected_speaker_id", ""),
            show_character_manager=data.get("show_character_manager", False),
        )
