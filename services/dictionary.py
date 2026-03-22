from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


DICTIONARIES_DIRNAME = "dictionaries"


@dataclass
class DictionaryEntry:
    source: str
    target: str
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DictionaryEntry":
        return cls(
            source=str(payload.get("source", "")).strip(),
            target=str(payload.get("target", "")).strip(),
            aliases=[
                str(item).strip()
                for item in payload.get("aliases", [])
                if str(item).strip()
            ],
        )


@dataclass
class WorkDictionary:
    work_id: str
    work_title: str = ""
    updated_at: str = ""
    entries: list[DictionaryEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "work_id": self.work_id,
            "work_title": self.work_title,
            "updated_at": self.updated_at,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkDictionary":
        return cls(
            work_id=str(payload.get("work_id", "")).strip(),
            work_title=str(payload.get("work_title", "")).strip(),
            updated_at=str(payload.get("updated_at", "")).strip(),
            entries=[
                entry
                for entry in (
                    DictionaryEntry.from_dict(item)
                    for item in payload.get("entries", [])
                )
                if entry.source and entry.target
            ],
        )


def _now_label() -> str:
    return datetime.now().replace(microsecond=0).isoformat(timespec="seconds")


def _dictionaries_dir(data_dir: Path) -> Path:
    return data_dir / DICTIONARIES_DIRNAME


def ensure_dictionary_storage(data_dir: Path) -> Path:
    target = _dictionaries_dir(data_dir)
    target.mkdir(parents=True, exist_ok=True)
    return target


def dictionary_path(data_dir: Path, work_id: str) -> Path:
    return ensure_dictionary_storage(data_dir) / f"{work_id}.json"


def load_work_dictionary(data_dir: Path, work_id: str, work_title: str = "") -> WorkDictionary:
    path = dictionary_path(data_dir, work_id)
    if not path.exists():
        return WorkDictionary(work_id=work_id, work_title=work_title, updated_at=_now_label())

    payload = json.loads(path.read_text(encoding="utf-8"))
    loaded = WorkDictionary.from_dict(payload)
    if work_title and not loaded.work_title:
        loaded.work_title = work_title
    if not loaded.updated_at:
        loaded.updated_at = _now_label()
    return loaded


def save_work_dictionary(data_dir: Path, work_dictionary: WorkDictionary) -> None:
    path = dictionary_path(data_dir, work_dictionary.work_id)
    work_dictionary.updated_at = _now_label()
    path.write_text(
        json.dumps(work_dictionary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sync_work_dictionary(
    data_dir: Path,
    work_id: str,
    work_title: str,
    character_names: list[str],
) -> WorkDictionary:
    work_dictionary = load_work_dictionary(data_dir, work_id, work_title)
    work_dictionary.work_title = work_title

    entries_by_source = {
        entry.source: entry
        for entry in work_dictionary.entries
        if entry.source
    }
    for name in character_names:
        normalized = str(name).strip()
        if not normalized:
            continue
        if normalized not in entries_by_source:
            entry = DictionaryEntry(source=normalized, target=normalized)
            work_dictionary.entries.append(entry)
            entries_by_source[normalized] = entry
        elif not entries_by_source[normalized].target.strip():
            entries_by_source[normalized].target = normalized

    work_dictionary.entries.sort(key=lambda entry: (entry.source != entry.target, entry.source))
    save_work_dictionary(data_dir, work_dictionary)
    return work_dictionary


def _replacement_pairs(work_dictionary: WorkDictionary) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen_sources: set[str] = set()
    for entry in work_dictionary.entries:
        source_candidates = [entry.source, *entry.aliases]
        for source in source_candidates:
            normalized_source = source.strip()
            if not normalized_source or normalized_source in seen_sources:
                continue
            seen_sources.add(normalized_source)
            pairs.append((normalized_source, entry.target.strip() or normalized_source))
    pairs.sort(key=lambda item: (-len(item[0]), item[0]))
    return pairs


def apply_dictionary(text: str, work_dictionary: WorkDictionary | None) -> str:
    if work_dictionary is None:
        return text

    updated = text
    for source, target in _replacement_pairs(work_dictionary):
        updated = updated.replace(source, target)
    return updated


def merge_dictionaries(
    base_dictionary: WorkDictionary | None,
    overlay_dictionary: WorkDictionary | None,
) -> WorkDictionary | None:
    if base_dictionary is None and overlay_dictionary is None:
        return None
    if base_dictionary is None:
        return WorkDictionary.from_dict(overlay_dictionary.to_dict()) if overlay_dictionary else None
    if overlay_dictionary is None:
        return WorkDictionary.from_dict(base_dictionary.to_dict())

    merged = WorkDictionary.from_dict(base_dictionary.to_dict())
    entries_by_target = {entry.target: entry for entry in merged.entries if entry.target}
    for overlay_entry in overlay_dictionary.entries:
        target = overlay_entry.target.strip()
        if not target:
            continue
        existing = entries_by_target.get(target)
        if existing is None:
            merged.entries.append(DictionaryEntry.from_dict(overlay_entry.to_dict()))
            entries_by_target[target] = merged.entries[-1]
            continue
        alias_set = {existing.source, *existing.aliases}
        alias_set.update([overlay_entry.source, *overlay_entry.aliases])
        alias_set.discard(target)
        existing.aliases = sorted(item for item in alias_set if item.strip())
    return merged


def _kana_variants(text: str) -> list[str]:
    try:
        from pykakasi import kakasi  # type: ignore
    except ImportError:
        return []

    converter = kakasi()
    fragments = converter.convert(text)
    hira = "".join(fragment.get("hira", "") for fragment in fragments).strip()
    kana = "".join(fragment.get("kana", "") for fragment in fragments).strip()

    variants: list[str] = []
    for item in [hira, kana]:
        cleaned = item.replace(" ", "")
        if cleaned:
            variants.append(cleaned)
            if len(cleaned) >= 2:
                variants.append(cleaned[:2])
            if len(cleaned) >= 3:
                variants.append(cleaned[:3])
            if len(cleaned) >= 2:
                variants.append(cleaned[-2:])
            if len(cleaned) >= 3:
                variants.append(cleaned[-3:])
            variants.append(f"{cleaned}くん")
            variants.append(f"{cleaned}クン")
            variants.append(f"{cleaned}君")
            if len(cleaned) >= 2:
                tail = cleaned[-2:]
                variants.extend([tail, f"{tail}くん", f"{tail}クン", f"{tail}君"])
    return [item for item in variants if item]


def _build_prompt_entry(target: str) -> DictionaryEntry | None:
    cleaned_target = target.strip()
    if len(cleaned_target) < 2:
        return None
    aliases: list[str] = []
    seen: set[str] = {cleaned_target}
    for alias in _kana_variants(cleaned_target):
        normalized = alias.strip()
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        aliases.append(normalized)
    return DictionaryEntry(source=cleaned_target, target=cleaned_target, aliases=aliases)


def build_prompt_dictionary(prompt_text: str) -> WorkDictionary | None:
    text = (prompt_text or "").strip()
    if not text:
        return None

    patterns = [
        r"(?:人物|キャラ|名前)[は:：]\s*([一-龯々ヶヵァ-ヴーー]{2,20})",
        r"([一-龯々ヶヵァ-ヴーー]{2,20})という(?:人物|キャラ|名前|人)",
        r"(?:登場人物|キャラ名|固有名詞)[は:：]\s*([一-龯々ヶヵァ-ヴーー]{2,20})",
    ]
    extracted: list[str] = []
    for pattern in patterns:
        extracted.extend(re.findall(pattern, text))

    if not extracted:
        extracted.extend(re.findall(r"[一-龯々ヶヵ]{2,12}", text))

    entries: list[DictionaryEntry] = []
    seen_targets: set[str] = set()
    for target in extracted:
        entry = _build_prompt_entry(target)
        if entry is None or entry.target in seen_targets:
            continue
        seen_targets.add(entry.target)
        entries.append(entry)

    if not entries:
        return None
    return WorkDictionary(work_id="__prompt__", work_title="prompt", updated_at=_now_label(), entries=entries)
