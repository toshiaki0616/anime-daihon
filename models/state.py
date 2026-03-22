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
    source_start: float = 0.0
    source_end: float = 0.0
    voiceprint_profile_id: str = ""
    voiceprint_character_name: str = ""
    voiceprint_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubtitleSegment":
        payload = dict(data)
        payload["source_start"] = float(payload.get("source_start", payload.get("start", 0.0)) or 0.0)
        payload["source_end"] = float(payload.get("source_end", payload.get("end", 0.0)) or 0.0)
        payload["voiceprint_profile_id"] = str(payload.get("voiceprint_profile_id", "")).strip()
        payload["voiceprint_character_name"] = str(payload.get("voiceprint_character_name", "")).strip()
        payload["voiceprint_confidence"] = float(payload.get("voiceprint_confidence", 0.0) or 0.0)
        return cls(**payload)


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
class VoiceprintSample:
    sample_id: str
    episode_id: str
    speaker_id: str
    character_name: str
    source_wav_path: str
    clip_start: float
    clip_end: float
    transcript_text: str = ""
    embedding: list[float] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceprintSample":
        payload = dict(data)
        payload["embedding"] = [float(value) for value in payload.get("embedding", [])]
        return cls(**payload)


@dataclass
class VoiceprintProfile:
    profile_id: str
    work_id: str
    character_name: str
    sample_ids: list[str] = field(default_factory=list)
    sample_count: int = 0
    average_embedding: list[float] = field(default_factory=list)
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceprintProfile":
        payload = dict(data)
        payload["sample_ids"] = [str(value) for value in payload.get("sample_ids", [])]
        payload["average_embedding"] = [
            float(value)
            for value in payload.get("average_embedding", [])
        ]
        return cls(**payload)


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
    whisper_model: str = "base"
    initial_prompt: str = ""
    speaker_diagnostics: str = ""
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
            "whisper_model": self.whisper_model,
            "initial_prompt": self.initial_prompt,
            "speaker_diagnostics": self.speaker_diagnostics,
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
            whisper_model=data.get("whisper_model", "base"),
            initial_prompt=data.get("initial_prompt", ""),
            speaker_diagnostics=data.get("speaker_diagnostics", ""),
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
class VoiceprintCandidate:
    candidate_id: str
    episode_id: str
    source_segment_id: str
    speaker_id: str
    clip_start: float
    clip_end: float
    transcript_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceprintCandidate":
        payload = dict(data)
        payload["clip_start"] = float(payload.get("clip_start", 0.0) or 0.0)
        payload["clip_end"] = float(payload.get("clip_end", 0.0) or 0.0)
        payload["transcript_text"] = str(payload.get("transcript_text", "")).strip()
        return cls(**payload)


@dataclass
class AppState:
    works: list[Work] = field(default_factory=list)
    current_page: str = "work_list"
    selected_work_id: str = ""
    selected_episode_id: str = ""
    selected_speaker_id: str = ""
    show_character_manager: bool = False
    selected_subtitle_segment_id: str = ""
    selected_subtitle_preview: str = ""
    rerun_candidate_text: str = ""
    rerun_candidate_label: str = ""
    rerun_candidate_range: str = ""
    voiceprint_candidates: list[VoiceprintCandidate] = field(default_factory=list)
    selected_voiceprint_candidate_id: str = ""
    selected_voiceprint_character_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "works": [work.to_dict() for work in self.works],
            "current_page": self.current_page,
            "selected_work_id": self.selected_work_id,
            "selected_episode_id": self.selected_episode_id,
            "selected_speaker_id": self.selected_speaker_id,
            "show_character_manager": self.show_character_manager,
            "selected_subtitle_segment_id": self.selected_subtitle_segment_id,
            "selected_subtitle_preview": self.selected_subtitle_preview,
            "rerun_candidate_text": self.rerun_candidate_text,
            "rerun_candidate_label": self.rerun_candidate_label,
            "rerun_candidate_range": self.rerun_candidate_range,
            "voiceprint_candidates": [
                candidate.to_dict() for candidate in self.voiceprint_candidates
            ],
            "selected_voiceprint_candidate_id": self.selected_voiceprint_candidate_id,
            "selected_voiceprint_character_name": self.selected_voiceprint_character_name,
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
            selected_subtitle_segment_id=data.get("selected_subtitle_segment_id", ""),
            selected_subtitle_preview=data.get("selected_subtitle_preview", ""),
            rerun_candidate_text=data.get("rerun_candidate_text", ""),
            rerun_candidate_label=data.get("rerun_candidate_label", ""),
            rerun_candidate_range=data.get("rerun_candidate_range", ""),
            voiceprint_candidates=[
                VoiceprintCandidate.from_dict(candidate)
                for candidate in data.get("voiceprint_candidates", [])
            ],
            selected_voiceprint_candidate_id=data.get("selected_voiceprint_candidate_id", ""),
            selected_voiceprint_character_name=data.get("selected_voiceprint_character_name", ""),
        )
