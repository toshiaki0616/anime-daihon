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
class VadSegment:
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VadSegment":
        return cls(
            start=float(data.get("start", 0.0) or 0.0),
            end=float(data.get("end", 0.0) or 0.0),
        )


@dataclass
class PreprocessingResult:
    source_path: str
    normalized_wav_path: str
    processed_range_start: float
    processed_range_end: float
    vad_segments: list[VadSegment] = field(default_factory=list)
    debug_paths: dict[str, str] = field(default_factory=dict)
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "normalized_wav_path": self.normalized_wav_path,
            "processed_range_start": self.processed_range_start,
            "processed_range_end": self.processed_range_end,
            "vad_segments": [segment.to_dict() for segment in self.vad_segments],
            "debug_paths": dict(self.debug_paths),
            "fallback_used": self.fallback_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessingResult":
        return cls(
            source_path=str(data.get("source_path", "")).strip(),
            normalized_wav_path=str(data.get("normalized_wav_path", "")).strip(),
            processed_range_start=float(data.get("processed_range_start", 0.0) or 0.0),
            processed_range_end=float(data.get("processed_range_end", 0.0) or 0.0),
            vad_segments=[
                VadSegment.from_dict(segment)
                for segment in data.get("vad_segments", [])
            ],
            debug_paths={
                str(key): str(value)
                for key, value in dict(data.get("debug_paths", {})).items()
            },
            fallback_used=bool(data.get("fallback_used", False)),
        )


@dataclass
class RawTranscriptSegment:
    id: str
    start: float
    end: float
    text: str
    original_text: str
    edited_text: str
    speaker_id: str = "A"
    raw_label: str = "話者A"
    display_name: str = "話者A"
    source_start: float = 0.0
    source_end: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawTranscriptSegment":
        payload = dict(data)
        payload["start"] = float(payload.get("start", 0.0) or 0.0)
        payload["end"] = float(payload.get("end", 0.0) or 0.0)
        payload["source_start"] = float(payload.get("source_start", payload.get("start", 0.0)) or 0.0)
        payload["source_end"] = float(payload.get("source_end", payload.get("end", 0.0)) or 0.0)
        payload["text"] = str(payload.get("text", "")).strip()
        payload["original_text"] = str(payload.get("original_text", payload.get("text", ""))).strip()
        payload["edited_text"] = str(payload.get("edited_text", payload.get("original_text", payload.get("text", "")))).strip()
        payload["speaker_id"] = str(payload.get("speaker_id", "A")).strip() or "A"
        payload["raw_label"] = str(payload.get("raw_label", "話者A")).strip() or "話者A"
        payload["display_name"] = str(payload.get("display_name", payload.get("raw_label", "話者A"))).strip() or "話者A"
        return cls(**payload)


@dataclass
class DiarizationSegment:
    start: float
    end: float
    raw_speaker_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiarizationSegment":
        return cls(
            start=float(data.get("start", 0.0) or 0.0),
            end=float(data.get("end", 0.0) or 0.0),
            raw_speaker_id=str(
                data.get("raw_speaker_id", data.get("speaker", ""))
            ).strip(),
        )


@dataclass
class TranscriptionResult:
    source_path: str
    wav_path: str
    vad_segments: list[VadSegment] = field(default_factory=list)
    raw_transcript_segments: list[RawTranscriptSegment] = field(default_factory=list)
    debug_paths: dict[str, str] = field(default_factory=dict)
    failed_segments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "wav_path": self.wav_path,
            "vad_segments": [segment.to_dict() for segment in self.vad_segments],
            "raw_transcript_segments": [
                segment.to_dict() for segment in self.raw_transcript_segments
            ],
            "debug_paths": dict(self.debug_paths),
            "failed_segments": [dict(item) for item in self.failed_segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionResult":
        return cls(
            source_path=str(data.get("source_path", "")).strip(),
            wav_path=str(data.get("wav_path", "")).strip(),
            vad_segments=[
                VadSegment.from_dict(segment)
                for segment in data.get("vad_segments", [])
            ],
            raw_transcript_segments=[
                RawTranscriptSegment.from_dict(segment)
                for segment in data.get("raw_transcript_segments", [])
            ],
            debug_paths={
                str(key): str(value)
                for key, value in dict(data.get("debug_paths", {})).items()
            },
            failed_segments=[
                dict(item)
                for item in data.get("failed_segments", [])
            ],
        )


@dataclass
class SpeakerAssignmentResult:
    diarization_segments: list[DiarizationSegment] = field(default_factory=list)
    assigned_subtitle_segments: list[RawTranscriptSegment] = field(default_factory=list)
    ui_speaker_map: dict[str, str] = field(default_factory=dict)
    speaker_profiles: list["SpeakerProfile"] = field(default_factory=list)
    debug_paths: dict[str, str] = field(default_factory=dict)
    fallback_used: bool = False
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "diarization_segments": [segment.to_dict() for segment in self.diarization_segments],
            "assigned_subtitle_segments": [
                segment.to_dict() for segment in self.assigned_subtitle_segments
            ],
            "ui_speaker_map": dict(self.ui_speaker_map),
            "speaker_profiles": [profile.to_dict() for profile in self.speaker_profiles],
            "debug_paths": dict(self.debug_paths),
            "fallback_used": self.fallback_used,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeakerAssignmentResult":
        return cls(
            diarization_segments=[
                DiarizationSegment.from_dict(segment)
                for segment in data.get("diarization_segments", [])
            ],
            assigned_subtitle_segments=[
                RawTranscriptSegment.from_dict(segment)
                for segment in data.get("assigned_subtitle_segments", [])
            ],
            ui_speaker_map={
                str(key): str(value)
                for key, value in dict(data.get("ui_speaker_map", {})).items()
            },
            speaker_profiles=[
                SpeakerProfile.from_dict(profile)
                for profile in data.get("speaker_profiles", [])
            ],
            debug_paths={
                str(key): str(value)
                for key, value in dict(data.get("debug_paths", {})).items()
            },
            fallback_used=bool(data.get("fallback_used", False)),
            error_message=str(data.get("error_message", "")).strip(),
        )


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
    last_preprocessing_source_path: str = ""
    last_preprocessing_wav_path: str = ""
    last_processed_range_start: float = 0.0
    last_processed_range_end: float = 0.0
    last_vad_segments: list[VadSegment] = field(default_factory=list)
    last_preprocessing_status: str = ""
    last_preprocessing_error: str = ""
    last_debug_output_path: str = ""
    last_vad_fallback_used: bool = False
    last_raw_transcript_segments: list[RawTranscriptSegment] = field(default_factory=list)
    last_transcription_status: str = ""
    last_transcription_error: str = ""
    last_debug_asr_output_path: str = ""
    last_diarization_segments: list[DiarizationSegment] = field(default_factory=list)
    last_ui_speaker_map: dict[str, str] = field(default_factory=dict)
    last_diarization_status: str = ""
    last_diarization_error: str = ""
    last_debug_diarization_output_path: str = ""
    last_debug_final_output_path: str = ""

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
            "last_preprocessing_source_path": self.last_preprocessing_source_path,
            "last_preprocessing_wav_path": self.last_preprocessing_wav_path,
            "last_processed_range_start": self.last_processed_range_start,
            "last_processed_range_end": self.last_processed_range_end,
            "last_vad_segments": [segment.to_dict() for segment in self.last_vad_segments],
            "last_preprocessing_status": self.last_preprocessing_status,
            "last_preprocessing_error": self.last_preprocessing_error,
            "last_debug_output_path": self.last_debug_output_path,
            "last_vad_fallback_used": self.last_vad_fallback_used,
            "last_raw_transcript_segments": [
                segment.to_dict() for segment in self.last_raw_transcript_segments
            ],
            "last_transcription_status": self.last_transcription_status,
            "last_transcription_error": self.last_transcription_error,
            "last_debug_asr_output_path": self.last_debug_asr_output_path,
            "last_diarization_segments": [
                segment.to_dict() for segment in self.last_diarization_segments
            ],
            "last_ui_speaker_map": dict(self.last_ui_speaker_map),
            "last_diarization_status": self.last_diarization_status,
            "last_diarization_error": self.last_diarization_error,
            "last_debug_diarization_output_path": self.last_debug_diarization_output_path,
            "last_debug_final_output_path": self.last_debug_final_output_path,
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
            last_preprocessing_source_path=data.get("last_preprocessing_source_path", ""),
            last_preprocessing_wav_path=data.get("last_preprocessing_wav_path", ""),
            last_processed_range_start=float(data.get("last_processed_range_start", 0.0) or 0.0),
            last_processed_range_end=float(data.get("last_processed_range_end", 0.0) or 0.0),
            last_vad_segments=[
                VadSegment.from_dict(segment)
                for segment in data.get("last_vad_segments", [])
            ],
            last_preprocessing_status=data.get("last_preprocessing_status", ""),
            last_preprocessing_error=data.get("last_preprocessing_error", ""),
            last_debug_output_path=data.get("last_debug_output_path", ""),
            last_vad_fallback_used=bool(data.get("last_vad_fallback_used", False)),
            last_raw_transcript_segments=[
                RawTranscriptSegment.from_dict(segment)
                for segment in data.get("last_raw_transcript_segments", [])
            ],
            last_transcription_status=data.get("last_transcription_status", ""),
            last_transcription_error=data.get("last_transcription_error", ""),
            last_debug_asr_output_path=data.get("last_debug_asr_output_path", ""),
            last_diarization_segments=[
                DiarizationSegment.from_dict(segment)
                for segment in data.get("last_diarization_segments", [])
            ],
            last_ui_speaker_map={
                str(key): str(value)
                for key, value in dict(data.get("last_ui_speaker_map", {})).items()
            },
            last_diarization_status=data.get("last_diarization_status", ""),
            last_diarization_error=data.get("last_diarization_error", ""),
            last_debug_diarization_output_path=data.get("last_debug_diarization_output_path", ""),
            last_debug_final_output_path=data.get("last_debug_final_output_path", ""),
        )
