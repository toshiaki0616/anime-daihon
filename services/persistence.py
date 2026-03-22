from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from models.state import AppState, Episode, VoiceprintProfile, VoiceprintSample, Work


class PersistenceError(Exception):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


WORKS_DIRNAME = "works"
EPISODES_DIRNAME = "episodes"
EXPORTS_DIRNAME = "exports"
VOICEPRINTS_DIRNAME = "voiceprints"


def _works_dir(data_dir: Path) -> Path:
    return data_dir / WORKS_DIRNAME


def _episodes_dir(data_dir: Path) -> Path:
    return data_dir / EPISODES_DIRNAME


def _exports_dir(data_dir: Path) -> Path:
    return data_dir / EXPORTS_DIRNAME


def _voiceprints_root_dir(data_dir: Path) -> Path:
    return data_dir / VOICEPRINTS_DIRNAME


def _voiceprints_dir(data_dir: Path, work_id: str) -> Path:
    return _voiceprints_root_dir(data_dir) / work_id


def _ensure_storage_dirs(data_dir: Path) -> None:
    _works_dir(data_dir).mkdir(parents=True, exist_ok=True)
    _episodes_dir(data_dir).mkdir(parents=True, exist_ok=True)
    _exports_dir(data_dir).mkdir(parents=True, exist_ok=True)
    _voiceprints_root_dir(data_dir).mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_voiceprint_storage(data_dir: Path, work_id: str) -> Path:
    _ensure_storage_dirs(data_dir)
    target = _voiceprints_dir(data_dir, work_id)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _safe_slug(value: str) -> str:
    normalized = re.sub(r"[\\/:*?\"<>|]+", "_", value.strip())
    normalized = re.sub(r"\s+", "_", normalized)
    return normalized[:80] or "untitled"


def _format_seconds(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def save_library_state(state: AppState, data_dir: Path) -> None:
    _ensure_storage_dirs(data_dir)
    work_ids: set[str] = set()
    episode_ids: set[str] = set()

    try:
        for work in state.works:
            work_ids.add(work.work_id)
            work_payload = {
                "work_id": work.work_id,
                "title": work.title,
                "character_names": list(work.character_names),
                "created_at": work.created_at,
                "updated_at": work.updated_at,
                "episode_ids": [episode.episode_id for episode in work.episodes],
            }
            _json_dump(_works_dir(data_dir) / f"{work.work_id}.json", work_payload)

            for episode in work.episodes:
                episode_ids.add(episode.episode_id)
                episode_payload = episode.to_dict()
                episode_payload["work_id"] = work.work_id
                _json_dump(_episodes_dir(data_dir) / f"{episode.episode_id}.json", episode_payload)

        for path in _works_dir(data_dir).glob("*.json"):
            if path.stem not in work_ids:
                path.unlink(missing_ok=True)
        for path in _episodes_dir(data_dir).glob("*.json"):
            if path.stem not in episode_ids:
                path.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        raise PersistenceError("保存に失敗しました") from exc


def load_library_state(data_dir: Path) -> AppState:
    works_dir = _works_dir(data_dir)
    episodes_dir = _episodes_dir(data_dir)
    if not works_dir.exists() or not any(works_dir.glob("*.json")):
        return AppState()

    try:
        works_by_id: dict[str, Work] = {}
        for work_path in works_dir.glob("*.json"):
            payload = _json_load(work_path)
            work = Work(
                work_id=payload.get("work_id", work_path.stem),
                title=payload.get("title", work_path.stem),
                character_names=payload.get("character_names", []),
                created_at=payload.get("created_at", ""),
                updated_at=payload.get("updated_at", ""),
                episodes=[],
            )
            works_by_id[work.work_id] = work

        for episode_path in episodes_dir.glob("*.json"):
            payload = _json_load(episode_path)
            work_id = payload.pop("work_id", "")
            if work_id not in works_by_id:
                continue
            episode = Episode.from_dict(payload)
            works_by_id[work_id].episodes.append(episode)

        works = list(works_by_id.values())
        for work in works:
            work.episodes.sort(key=lambda item: item.updated_at, reverse=True)
        works.sort(key=lambda item: item.updated_at, reverse=True)
        return AppState(works=works, current_page="work_list")
    except Exception as exc:  # noqa: BLE001
        raise PersistenceError("読み込みに失敗しました") from exc


def save_voiceprint_state(
    data_dir: Path,
    work_id: str,
    profiles: list[VoiceprintProfile],
    samples: list[VoiceprintSample],
) -> None:
    voiceprints_dir = ensure_voiceprint_storage(data_dir, work_id)
    samples_dir = voiceprints_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    profile_ids = {profile.profile_id for profile in profiles}
    sample_ids = {sample.sample_id for sample in samples}

    try:
        profiles_payload = {
            "work_id": work_id,
            "profiles": [profile.to_dict() for profile in profiles],
        }
        _json_dump(voiceprints_dir / "profiles.json", profiles_payload)

        for sample in samples:
            _json_dump(samples_dir / f"{sample.sample_id}.json", sample.to_dict())

        for path in samples_dir.glob("*.json"):
            if path.stem not in sample_ids:
                path.unlink(missing_ok=True)

        # Remove profiles file if there are no profiles and no samples.
        if not profile_ids and not sample_ids:
            (voiceprints_dir / "profiles.json").unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        raise PersistenceError("声紋データの保存に失敗しました") from exc


def load_voiceprint_state(
    data_dir: Path,
    work_id: str,
) -> tuple[list[VoiceprintProfile], list[VoiceprintSample]]:
    voiceprints_dir = _voiceprints_dir(data_dir, work_id)
    profiles_path = voiceprints_dir / "profiles.json"
    samples_dir = voiceprints_dir / "samples"

    if not voiceprints_dir.exists():
        return [], []

    try:
        profiles: list[VoiceprintProfile] = []
        if profiles_path.exists():
            payload = _json_load(profiles_path)
            profiles = [
                VoiceprintProfile.from_dict(item)
                for item in payload.get("profiles", [])
            ]

        samples: list[VoiceprintSample] = []
        if samples_dir.exists():
            for sample_path in sorted(samples_dir.glob("*.json")):
                samples.append(VoiceprintSample.from_dict(_json_load(sample_path)))

        return profiles, samples
    except Exception as exc:  # noqa: BLE001
        raise PersistenceError("声紋データの読み込みに失敗しました") from exc


def export_episode_txt(work: Work, episode: Episode, data_dir: Path) -> str:
    _ensure_storage_dirs(data_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_safe_slug(work.title)}_{_safe_slug(episode.title)}_{timestamp}.txt"
    target = _exports_dir(data_dir) / filename
    try:
        lines = [
            f"[{_format_seconds(segment.start)}] {segment.display_name}：{segment.edited_text}"
            for segment in episode.subtitle_segments
        ]
        target.write_text("\n".join(lines), encoding="utf-8")
        return str(target)
    except Exception as exc:  # noqa: BLE001
        raise PersistenceError("書き出しに失敗しました") from exc


def export_episode_csv(work: Work, episode: Episode, data_dir: Path) -> str:
    _ensure_storage_dirs(data_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_safe_slug(work.title)}_{_safe_slug(episode.title)}_{timestamp}.csv"
    target = _exports_dir(data_dir) / filename
    try:
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["start_time", "display_name", "edited_text"])
            for segment in episode.subtitle_segments:
                writer.writerow([_format_seconds(segment.start), segment.display_name, segment.edited_text])
        return str(target)
    except Exception as exc:  # noqa: BLE001
        raise PersistenceError("書き出しに失敗しました") from exc
