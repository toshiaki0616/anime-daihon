from __future__ import annotations

from datetime import datetime
import gradio as gr
from pathlib import Path

from core.state_ops import (
    MAX_SPEAKER_SLOTS,
    add_speaker_profile,
    apply_assigned_transcript_segments,
    apply_subtitle_edits,
    apply_voiceprint_assignments_to_episode,
    build_mock_app_state,
    create_episode,
    create_work,
    delete_speaker_profile,
    get_selected_episode,
    get_selected_work,
    merge_speakers,
    move_segment_to_speaker,
    record_preprocessing_error,
    record_preprocessing_result,
    record_diarization_error,
    record_diarization_result,
    record_transcription_error,
    record_transcription_result,
    rename_speaker,
    swap_speakers,
    sync_episode,
    update_character_names,
    update_work_title,
    update_segment_text,
)
from models.state import AppState, VoiceprintCandidate
from services import (
    AudioSegmentationError,
    DictionaryEntry,
    DiarizationError,
    MediaPreprocessError,
    PersistenceError,
    TranscriptionError,
    apply_dictionary,
    build_voiceprint_candidates_from_clean_segments_only,
    assign_voiceprints_to_segments,
    build_voiceprint_sample,
    build_prompt_dictionary,
    ensure_dictionary_storage,
    ensure_voiceprint_storage,
    extract_audio_clip,
    export_episode_csv,
    export_episode_txt,
    extract_voice_embedding,
    load_library_state,
    load_voiceprint_state,
    load_work_dictionary,
    merge_dictionaries,
    preprocess_media,
    run_diarization_pipeline,
    run_preprocessing_pipeline,
    run_transcription_pipeline,
    save_library_state,
    save_voiceprint_state,
    save_work_dictionary,
    SpeakerIdentificationError,
    sync_work_dictionary,
    transcribe_wav,
    TranscriptionSegment,
    upsert_voiceprint_profile,
    DEFAULT_MODEL_NAME,
    MODEL_OPTIONS,
    normalize_model_selection,
)
from ui.renderers import (
    build_episode_rows,
    build_speaker_detail_payload,
    build_speaker_list_payloads,
    build_subtitle_rows,
    build_work_rows,
    format_seconds,
    format_status_box,
    parse_time_offset_seconds,
)

DATA_DIR = Path(__file__).parent / "data"


def now_label() -> str:
    return datetime.now().replace(microsecond=0).isoformat(timespec="seconds")


def load_initial_state() -> tuple[AppState, str, str]:
    try:
        ensure_dictionary_storage(DATA_DIR)
        state = load_library_state(DATA_DIR)
    except PersistenceError:
        return build_mock_app_state(), "読み込みに失敗しました", "error"

    if not state.works:
        return build_mock_app_state(), "モックライブラリを読み込みました", "info"

    for work in state.works:
        sync_work_dictionary(DATA_DIR, work.work_id, work.title, work.character_names)
        ensure_voiceprint_storage(DATA_DIR, work.work_id)
        for episode in work.episodes:
            sync_episode(episode)
    return state, "保存済みライブラリを読み込みました", "info"


INITIAL_STATE, INITIAL_MESSAGE, INITIAL_KIND = load_initial_state()


def parse_state(state_dict: dict | None) -> AppState:
    return AppState.from_dict(state_dict or INITIAL_STATE.to_dict())


def normalize_voiceprint_candidate_state(state: AppState) -> None:
    episode = get_selected_episode(state)
    if episode is None:
        state.voiceprint_candidates = []
        state.selected_voiceprint_candidate_id = ""
        return

    filtered_candidates = [
        candidate
        for candidate in state.voiceprint_candidates
        if candidate.episode_id == episode.episode_id
    ]
    if not filtered_candidates and episode.subtitle_segments:
        filtered_candidates = build_voiceprint_candidates_from_clean_segments_only(
            episode_id=episode.episode_id,
            subtitle_segments=episode.subtitle_segments,
        )
    state.voiceprint_candidates = filtered_candidates
    valid_ids = {candidate.candidate_id for candidate in filtered_candidates}
    if state.selected_voiceprint_candidate_id not in valid_ids:
        state.selected_voiceprint_candidate_id = filtered_candidates[0].candidate_id if filtered_candidates else ""


def _truncate_preview(text: str, limit: int = 48) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit - 1]}…"


def get_display_segment_start(segment) -> float:
    if getattr(segment, "refined_start", 0.0):
        return float(segment.refined_start)
    if getattr(segment, "source_start", 0.0):
        return float(segment.source_start)
    return float(getattr(segment, "start", 0.0) or 0.0)


def get_display_segment_end(segment) -> float:
    if getattr(segment, "refined_end", 0.0):
        return float(segment.refined_end)
    if getattr(segment, "source_end", 0.0):
        return float(segment.source_end)
    return float(getattr(segment, "end", 0.0) or 0.0)


def get_display_segment_duration(segment) -> float:
    return max(0.0, get_display_segment_end(segment) - get_display_segment_start(segment))


def build_voiceprint_candidate_choices(state: AppState):
    normalize_voiceprint_candidate_state(state)
    episode = get_selected_episode(state)
    choices = []
    for candidate in state.voiceprint_candidates:
        duration = candidate.duration or max(0.0, candidate.clip_end - candidate.clip_start)
        flag_suffix = ""
        if candidate.confidence_flags:
            flag_suffix = f" [{' / '.join(candidate.confidence_flags)}]"
        label = (
            f"[{format_episode_seconds(episode, candidate.clip_start)} - {format_episode_seconds(episode, candidate.clip_end)}] "
            f"{duration:.2f}s / {_truncate_preview(candidate.transcript_text)}{flag_suffix}"
        )
        choices.append((label, candidate.candidate_id))
    return choices


def find_selected_voiceprint_candidate(state: AppState) -> VoiceprintCandidate | None:
    normalize_voiceprint_candidate_state(state)
    return next(
        (
            candidate
            for candidate in state.voiceprint_candidates
            if candidate.candidate_id == state.selected_voiceprint_candidate_id
        ),
        None,
    )


def format_episode_seconds(episode, seconds: float) -> str:
    offset = parse_time_offset_seconds(episode.range_start) if episode is not None else 0.0
    return format_seconds(offset + seconds)


def build_selected_subtitle_label(
    start_label: str = "",
    duration_label: str = "",
    speaker_name: str = "",
    preview: str = "",
) -> str:
    if not start_label and not preview:
        return (
            "<div style='padding:12px 14px; border-radius:12px; "
            "border:1px dashed #4b5563; background:#111827; color:#d1d5db;'>"
            "話者欄をクリックして変更するセリフを選択してください"
            "</div>"
        )

    return (
        "<div style='padding:12px 14px; border-radius:12px; "
        "border:2px solid #f97316; background:#1f2937; color:#fff7ed; "
        "box-shadow:0 0 0 1px rgba(249,115,22,0.25) inset;'>"
        "<div style='font-size:12px; font-weight:700; letter-spacing:0.04em; color:#fdba74;'>"
        "SELECTED SEGMENT"
        "</div>"
        f"<div style='margin-top:6px; font-size:14px;'><strong>{start_label}</strong>"
        f" / {duration_label} / {speaker_name}</div>"
        f"<div style='margin-top:6px; font-size:15px; line-height:1.5; color:#ffffff;'>{preview}</div>"
        "</div>"
    )


def get_segment_source_range(episode, segment) -> tuple[str, float, float]:
    if episode is None or segment is None:
        return "", 0.0, 0.0
    display_start = get_display_segment_start(segment)
    display_end = get_display_segment_end(segment)
    if episode.wav_path and Path(episode.wav_path).exists():
        return episode.wav_path, display_start, display_end
    offset = parse_time_offset_seconds(episode.range_start)
    source_path = episode.file_path or episode.wav_path
    return source_path, offset + display_start, offset + display_end


def build_selected_segment_audio_update(episode, segment):
    if episode is None or segment is None:
        return gr.update(value=None, visible=False)
    source_path, clip_start, clip_end = get_segment_source_range(episode, segment)
    if not source_path or clip_end <= clip_start:
        return gr.update(value=None, visible=False)
    try:
        clip_path = extract_audio_clip(source_path, clip_start, clip_end, DATA_DIR)
    except MediaPreprocessError:
        return gr.update(value=None, visible=False)
    return gr.update(value=clip_path, visible=True)


def sorted_episodes_for_selected_work(state: AppState):
    work = get_selected_work(state)
    if work is None:
        return []
    return sorted(work.episodes, key=lambda item: item.updated_at, reverse=True)


def resolve_row_index(index_value) -> int | None:
    if isinstance(index_value, int):
        return index_value
    if isinstance(index_value, tuple):
        return index_value[0] if index_value else None
    if isinstance(index_value, list):
        return index_value[0] if index_value else None
    return None


def resolve_selected_speaker_id(state: AppState) -> str:
    episode = get_selected_episode(state)
    if episode is None or not episode.speakers:
        return ""
    speaker_ids = [speaker.speaker_id for speaker in episode.speakers]
    if state.selected_speaker_id in speaker_ids:
        return state.selected_speaker_id
    return speaker_ids[0]


def build_episode_speaker_choices(state: AppState):
    episode = get_selected_episode(state)
    if episode is None:
        return []
    return [
        (speaker.display_name or speaker.raw_label, speaker.speaker_id)
        for speaker in episode.speakers
    ]


def build_voiceprint_character_choices(state: AppState):
    work = get_selected_work(state)
    if work is None:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(value: str) -> None:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        candidates.append(cleaned)

    for name in work.character_names:
        add_candidate(name)

    episode = get_selected_episode(state)
    if episode is not None:
        for speaker in episode.speakers:
            add_candidate(speaker.display_name or "")
            add_candidate(speaker.raw_label or "")

    try:
        profiles, _samples = load_voiceprint_state(DATA_DIR, work.work_id)
    except PersistenceError:
        profiles = []
    for profile in profiles:
        add_candidate(profile.character_name)

    return [(name, name) for name in candidates]


def build_subtitle_segment_choices(state: AppState):
    episode = get_selected_episode(state)
    if episode is None:
        return []
    return [
        (
            f"{segment.id} | {format_episode_seconds(episode, get_display_segment_start(segment))} | {_truncate_preview(segment.edited_text, 36)}",
            segment.id,
        )
        for segment in episode.subtitle_segments
    ]


def normalize_dropdown_value(value: str | None, choices) -> str | None:
    candidate = (value or "").strip()
    valid_values = {choice[1] for choice in choices}
    if candidate in valid_values:
        return candidate
    return None


def build_voiceprint_summary(state: AppState) -> str:
    work = get_selected_work(state)
    if work is None:
        return "声紋登録: まだありません。"
    ensure_voiceprint_storage(DATA_DIR, work.work_id)
    try:
        profiles, samples = load_voiceprint_state(DATA_DIR, work.work_id)
    except PersistenceError:
        return "声紋登録: 読み込みに失敗しました。"
    if not profiles and not samples:
        return "声紋登録: まだありません。"

    lines = [f"声紋登録: {len(samples)}件 / プロフィール {len(profiles)}件"]
    for profile in profiles:
        lines.append(f"- {profile.character_name}: {profile.sample_count}件")
    return "\n".join(lines)


def build_voiceprint_selection_label(state: AppState) -> str:
    normalize_voiceprint_candidate_state(state)
    episode = get_selected_episode(state)
    selected_candidate = find_selected_voiceprint_candidate(state)
    if selected_candidate is not None:
        duration = selected_candidate.duration or max(0.0, selected_candidate.clip_end - selected_candidate.clip_start)
        flags = ""
        if selected_candidate.confidence_flags:
            flags = f" / {', '.join(selected_candidate.confidence_flags)}"
        return (
            "選択中の声紋候補: "
            f"[{format_episode_seconds(episode, selected_candidate.clip_start)} - {format_episode_seconds(episode, selected_candidate.clip_end)}] "
            f"{duration:.2f}s / {_truncate_preview(selected_candidate.transcript_text, limit=64)}{flags}"
        )

    if episode is None:
        return "高品質な声紋候補は、字幕生成後にここへ表示されます。"
    if state.voiceprint_candidates:
        return f"高品質な声紋候補が {len(state.voiceprint_candidates)} 件あります。候補を選んで登録してください。"
    return "この話では、高品質な声紋候補はまだ見つかっていません。"


def build_dictionary_rows(state: AppState):
    work = get_selected_work(state)
    if work is None:
        return []
    work_dictionary = load_work_dictionary(DATA_DIR, work.work_id, work.title)
    return [
        [entry.source, entry.target, "、".join(entry.aliases)]
        for entry in work_dictionary.entries
    ]


def build_transcription_prompt(state: AppState, user_prompt: str) -> str:
    work = get_selected_work(state)
    prompt_parts = []
    if work is not None and work.title:
        prompt_parts.append(f"作品名: {work.title}")
    if work is not None and work.character_names:
        prompt_parts.append("固有名詞候補: " + "、".join(work.character_names))
    cleaned_prompt = user_prompt.strip()
    if cleaned_prompt:
        prompt_parts.append("補助プロンプト: " + cleaned_prompt)
    return "\n".join(prompt_parts)


def render_all(state: AppState, message: str, kind: str = "info"):
    work = get_selected_work(state)
    episode = get_selected_episode(state)
    normalize_voiceprint_candidate_state(state)
    state.selected_speaker_id = resolve_selected_speaker_id(state)
    if (
        state.selected_voiceprint_character_name
        and work is not None
        and state.selected_voiceprint_character_name not in work.character_names
    ):
        state.selected_voiceprint_character_name = ""
    subtitle_segment_choices = build_subtitle_segment_choices(state)
    selected_segment = None
    if episode is not None and state.selected_subtitle_segment_id:
        selected_segment = next(
            (item for item in episode.subtitle_segments if item.id == state.selected_subtitle_segment_id),
            None,
        )
    selected_speaker_name = ""
    selected_audio_update = gr.update(value=None, visible=False)
    selected_speaker_target_value = None
    if selected_segment is not None:
        selected_speaker_name = selected_segment.display_name
        if selected_segment.voiceprint_profile_id and selected_segment.voiceprint_confidence < 0.75:
            selected_speaker_name = f"MEDIUM | {selected_segment.display_name}"
        selected_audio_update = build_selected_segment_audio_update(episode, selected_segment)
        selected_speaker_target_value = selected_segment.speaker_id
    speaker_detail = build_speaker_detail_payload(state)

    outputs = [
        state.to_dict(),
        format_status_box(message, kind),
        gr.update(visible=state.current_page == "work_list"),
        gr.update(visible=state.current_page == "work_detail"),
        gr.update(visible=state.current_page == "episode_editor"),
        gr.update(visible=state.current_page == "speaker_page"),
        build_work_rows(state),
        gr.update(value=f"## {work.title if work else '作品詳細'}"),
        gr.update(value=work.title if work else ""),
        gr.update(visible=state.show_character_manager),
        gr.update(value="、".join(work.character_names) if work else ""),
        build_dictionary_rows(state),
        build_episode_rows(state),
        gr.update(value=f"## {episode.title if episode else '話数編集'}"),
        episode.range_start if episode else "",
        episode.range_end if episode else "",
        episode.enhance_audio if episode else False,
        normalize_model_selection(episode.whisper_model) if episode else DEFAULT_MODEL_NAME,
        episode.initial_prompt if episode else "",
        build_subtitle_rows(state),
        state.selected_subtitle_segment_id,
        gr.update(
            value=build_selected_subtitle_label(
                f"[{format_episode_seconds(episode, get_display_segment_start(selected_segment))}]" if selected_segment is not None else "",
                f"{get_display_segment_duration(selected_segment):.2f}s" if selected_segment is not None else "",
                selected_speaker_name,
                selected_segment.edited_text[:60] if selected_segment is not None else "",
            )
        ),
        gr.update(
            choices=subtitle_segment_choices,
            value=normalize_dropdown_value(state.selected_subtitle_segment_id, subtitle_segment_choices),
        ),
        gr.update(value=selected_speaker_name),
        selected_audio_update,
        gr.update(choices=build_episode_speaker_choices(state), value=selected_speaker_target_value),
        gr.update(value=episode.speaker_diagnostics if episode and episode.speaker_diagnostics else "話者分離診断: まだありません。"),
        gr.update(value=build_voiceprint_selection_label(state), visible=bool(episode and episode.subtitle_segments)),
        gr.update(
            choices=build_voiceprint_candidate_choices(state),
            value=state.selected_voiceprint_candidate_id or None,
            visible=bool(episode and episode.subtitle_segments),
        ),
        gr.update(choices=build_voiceprint_character_choices(state), value=state.selected_voiceprint_character_name or None),
        gr.update(value=build_voiceprint_summary(state)),
        gr.update(value=state.rerun_candidate_label or "選択した時刻を再読み込みすると候補がここに表示されます。"),
        state.rerun_candidate_range,
        state.rerun_candidate_text,
    ]

    for payload in build_speaker_list_payloads(state):
        outputs.extend(
            [
                gr.update(visible=payload["visible"], value=payload["label"], variant=payload["variant"]),
                payload["speaker_id"],
            ]
        )

    outputs.extend(
        [
            gr.update(value=speaker_detail["title"]),
            gr.update(value=speaker_detail["raw_label"]),
            gr.update(value=speaker_detail["display_name"]),
            gr.update(choices=speaker_detail["utterance_choices"], value=None),
            gr.update(choices=[], value=None, visible=False),
            gr.update(value=""),
            gr.update(choices=speaker_detail["merge_choices"], value=None),
            gr.update(choices=speaker_detail["swap_choices"], value=None),
            gr.update(choices=speaker_detail["move_segment_choices"], value=None),
            gr.update(choices=speaker_detail["move_target_choices"], value=None),
            "",
            gr.update(interactive=speaker_detail["can_delete"]),
            speaker_detail["speaker_id"],
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        ]
    )
    return outputs


def persist_state_or_error(state: AppState) -> tuple[bool, str]:
    try:
        save_library_state(state, DATA_DIR)
        return True, ""
    except PersistenceError as exc:
        return False, exc.user_message


def add_work(state_dict: dict):
    state = create_work(parse_state(state_dict))
    work = get_selected_work(state)
    if work is not None:
        sync_work_dictionary(DATA_DIR, work.work_id, work.title, work.character_names)
        ensure_voiceprint_storage(DATA_DIR, work.work_id)
    return render_all(state, "新しい作品を追加しました", "success")


def reload_library(state_dict: dict | None = None):
    try:
        ensure_dictionary_storage(DATA_DIR)
        state = load_library_state(DATA_DIR)
    except PersistenceError:
        return render_all(parse_state(state_dict), "読み込みに失敗しました", "error")
    if not state.works:
        state = build_mock_app_state()
        return render_all(state, "保存済みデータがないためモックライブラリを表示しています", "info")
    for work in state.works:
        sync_work_dictionary(DATA_DIR, work.work_id, work.title, work.character_names)
        ensure_voiceprint_storage(DATA_DIR, work.work_id)
        for episode in work.episodes:
            sync_episode(episode)
    return render_all(state, "保存済みライブラリを再読込しました", "success")


def open_work(select_data: gr.SelectData, state_dict: dict):
    state = parse_state(state_dict)
    row_index = resolve_row_index(select_data.index)
    if row_index is None or row_index >= len(state.works):
        return render_all(state, "作品を選択してください", "error")
    state.selected_work_id = state.works[row_index].work_id
    state.current_page = "work_detail"
    return render_all(state, "作品詳細を開きました", "info")


def back_to_work_list(state_dict: dict):
    state = parse_state(state_dict)
    state.current_page = "work_list"
    state.show_character_manager = False
    state.voiceprint_candidates = []
    state.selected_voiceprint_candidate_id = ""
    return render_all(state, "作品一覧に戻りました", "info")


def save_work_title(new_title: str, state_dict: dict):
    state = update_work_title(parse_state(state_dict), new_title)
    work = get_selected_work(state)
    if work is not None:
        sync_work_dictionary(DATA_DIR, work.work_id, work.title, work.character_names)
    ok, error = persist_state_or_error(state)
    if not ok:
        return render_all(state, error, "error")
    return render_all(state, "作品名を保存しました", "success")


def toggle_character_manager(state_dict: dict):
    state = parse_state(state_dict)
    state.show_character_manager = not state.show_character_manager
    work = get_selected_work(state)
    message = "キャラ名管理を表示しました" if state.show_character_manager else "キャラ名管理を閉じました"
    return [
        state.to_dict(),
        format_status_box(message, "info"),
        gr.update(visible=state.show_character_manager),
        gr.update(value="、".join(work.character_names) if work else ""),
        build_dictionary_rows(state),
    ]


def save_character_candidates(raw_text: str, state_dict: dict):
    state = update_character_names(parse_state(state_dict), raw_text)
    work = get_selected_work(state)
    if work is not None:
        sync_work_dictionary(DATA_DIR, work.work_id, work.title, work.character_names)
    ok, error = persist_state_or_error(state)
    if not ok:
        return render_all(state, error, "error")
    return render_all(state, "キャラ名候補を保存しました", "success")


def save_dictionary_entries(rows, state_dict: dict):
    state = parse_state(state_dict)
    work = get_selected_work(state)
    if work is None:
        return render_all(state, "作品を選択してください", "error")

    normalized_rows = rows.values.tolist() if hasattr(rows, "values") else rows or []
    entries: list[DictionaryEntry] = []
    for row in normalized_rows:
        source = str(row[0]).strip() if len(row) > 0 else ""
        target = str(row[1]).strip() if len(row) > 1 else ""
        aliases_raw = str(row[2]).strip() if len(row) > 2 else ""
        if not source or not target:
            continue
        aliases = [
            item.strip()
            for item in aliases_raw.replace("、", ",").replace("\n", ",").split(",")
            if item.strip()
        ]
        entries.append(DictionaryEntry(source=source, target=target, aliases=aliases))

    work_dictionary = load_work_dictionary(DATA_DIR, work.work_id, work.title)
    work_dictionary.entries = entries
    save_work_dictionary(DATA_DIR, work_dictionary)
    return render_all(state, "辞書を保存しました", "success")


def add_episode(state_dict: dict):
    state = create_episode(parse_state(state_dict))
    state.voiceprint_candidates = []
    state.selected_voiceprint_candidate_id = ""
    return render_all(state, "新しい話を追加しました", "success")


def open_episode(select_data: gr.SelectData, state_dict: dict):
    state = parse_state(state_dict)
    episodes = sorted_episodes_for_selected_work(state)
    row_index = resolve_row_index(select_data.index)
    if row_index is None or row_index >= len(episodes):
        return render_all(state, "話数を選択してください", "error")
    state.selected_episode_id = episodes[row_index].episode_id
    state.current_page = "episode_editor"
    state.voiceprint_candidates = []
    state.selected_voiceprint_candidate_id = ""
    return render_all(state, "話数編集を開きました", "info")


def back_to_work_detail(state_dict: dict):
    state = parse_state(state_dict)
    state.current_page = "work_detail"
    state.voiceprint_candidates = []
    state.selected_voiceprint_candidate_id = ""
    return render_all(state, "作品詳細に戻りました", "info")


def generate_subtitles(file_path: str | None, start_time: str, end_time: str, enhance_audio: bool, whisper_model: str, initial_prompt: str, state_dict: dict, progress=gr.Progress(track_tqdm=False)):
    state = parse_state(state_dict)
    episode = get_selected_episode(state)
    work = get_selected_work(state)
    if episode is None:
        return render_all(state, "話数を選択してください", "error")
    if not file_path:
        state = record_preprocessing_error(state, "", "ファイルを選んでください")
        return render_all(state, "ファイルを選んでください", "error")

    episode.range_start = start_time.strip()
    episode.range_end = end_time.strip()
    episode.enhance_audio = enhance_audio
    episode.whisper_model = normalize_model_selection(whisper_model)
    episode.initial_prompt = initial_prompt.strip()
    prompt_text = build_transcription_prompt(state, initial_prompt)
    work_dictionary = None
    if work is not None:
        work_dictionary = sync_work_dictionary(DATA_DIR, work.work_id, work.title, work.character_names)
        work_dictionary = merge_dictionaries(work_dictionary, build_prompt_dictionary(initial_prompt))

    diarization_result = None
    try:
        preprocessing_result = run_preprocessing_pipeline(
            input_path=file_path,
            range_start=start_time.strip(),
            range_end=end_time.strip(),
            data_dir=DATA_DIR,
            progress_callback=lambda value, message: progress(value, desc=message),
        )
        state = record_preprocessing_result(state, preprocessing_result)
        transcription_result = run_transcription_pipeline(
            preprocessing_result=preprocessing_result,
            model_name=episode.whisper_model,
            initial_prompt=prompt_text,
            data_dir=DATA_DIR,
            progress_callback=lambda value, message: progress(value, desc=message),
        )
        state = record_transcription_result(state, transcription_result)
        diarization_result = run_diarization_pipeline(
            preprocessing_result=preprocessing_result,
            transcription_result=transcription_result,
            episode_id=episode.episode_id,
            data_dir=DATA_DIR,
            progress_callback=lambda value, message: progress(value, desc=message),
        )
        state = record_diarization_result(state, diarization_result)
    except AudioSegmentationError as exc:
        state = record_preprocessing_error(state, str(file_path or ""), exc.user_message)
        return render_all(state, exc.user_message, "error")
    except TranscriptionError as exc:
        state = record_transcription_error(state, exc.user_message)
        return render_all(state, exc.user_message, "error")
    except DiarizationError as exc:
        state = record_diarization_error(state, exc.user_message)
        return render_all(state, exc.user_message, "error")

    assignment_result = diarization_result
    if assignment_result is None:
        state = record_diarization_error(state, "話者の分割に失敗しました")
        return render_all(state, "話者の分割に失敗しました", "error")

    state = apply_assigned_transcript_segments(
        state=state,
        file_path=preprocessing_result.source_path,
        wav_path=preprocessing_result.normalized_wav_path,
        range_start=start_time.strip(),
        range_end=end_time.strip(),
        enhance_audio=enhance_audio,
        transcript_segments=assignment_result.assigned_subtitle_segments,
        speaker_label_map=assignment_result.ui_speaker_map,
        diarization_segments=assignment_result.diarization_segments,
        text_postprocessor=lambda text: apply_dictionary(text, work_dictionary),
    )
    normalize_voiceprint_candidate_state(state)
    if state.voiceprint_candidates and not state.selected_voiceprint_candidate_id:
        state.selected_voiceprint_candidate_id = state.voiceprint_candidates[0].candidate_id
    progress(1.0, desc="字幕を作成しました")
    if assignment_result.fallback_used and transcription_result.failed_segments and preprocessing_result.fallback_used:
        return render_all(state, "字幕を作成しました（一部の区間はフォールバックや失敗を含みます）", "info")
    if assignment_result.fallback_used and len(get_selected_episode(state).speakers) <= 1 and assignment_result.error_message:
        return render_all(state, "話者の分割に失敗しました", "error")
    if assignment_result.fallback_used:
        return render_all(state, "字幕を作成しました（話者はフォールバック推定です）", "info")
    if transcription_result.failed_segments:
        return render_all(state, "字幕を作成しました（一部の区間は失敗しました）", "info")
    if preprocessing_result.fallback_used:
        return render_all(state, "字幕を作成しました（発話区間はフォールバック）", "info")
    return render_all(state, "字幕を作成しました", "success")


def sync_subtitle_edits(rows, state_dict: dict):
    normalized_rows = rows.values.tolist() if hasattr(rows, "values") else rows or []
    state = apply_subtitle_edits(parse_state(state_dict), normalized_rows)
    return render_all(state, "字幕編集を反映しました", "success")


def sync_subtitle_edits_inline(rows, state_dict: dict):
    normalized_rows = rows.values.tolist() if hasattr(rows, "values") else rows or []
    state = apply_subtitle_edits(parse_state(state_dict), normalized_rows)
    return [
        state.to_dict(),
        format_status_box("字幕編集を一時反映しました", "info"),
    ]


def save_current_episode(state_dict: dict):
    state = parse_state(state_dict)
    work = get_selected_work(state)
    episode = get_selected_episode(state)
    if work is None or episode is None:
        return render_all(state, "話数を選択してください", "error")
    timestamp = now_label()
    episode.updated_at = timestamp
    work.updated_at = timestamp
    ok, error = persist_state_or_error(state)
    if not ok:
        return render_all(state, error, "error")
    return render_all(state, "保存しました", "success")


def export_txt(state_dict: dict):
    state = parse_state(state_dict)
    work = get_selected_work(state)
    episode = get_selected_episode(state)
    if work is None or episode is None:
        rendered = render_all(state, "話数を選択してください", "error")
        rendered[-2] = gr.update(value=None, visible=False)
        rendered[-1] = gr.update(value=None, visible=False)
        return rendered
    try:
        path = export_episode_txt(work, episode, DATA_DIR)
    except PersistenceError as exc:
        rendered = render_all(state, exc.user_message, "error")
        rendered[-2] = gr.update(value=None, visible=False)
        rendered[-1] = gr.update(value=None, visible=False)
        return rendered
    rendered = render_all(state, "テキストを書き出しました", "success")
    rendered[-2] = gr.update(value=path, visible=True)
    rendered[-1] = gr.update(value=None, visible=False)
    return rendered


def export_csv(state_dict: dict):
    state = parse_state(state_dict)
    work = get_selected_work(state)
    episode = get_selected_episode(state)
    if work is None or episode is None:
        rendered = render_all(state, "話数を選択してください", "error")
        rendered[-2] = gr.update(value=None, visible=False)
        rendered[-1] = gr.update(value=None, visible=False)
        return rendered
    try:
        path = export_episode_csv(work, episode, DATA_DIR)
    except PersistenceError as exc:
        rendered = render_all(state, exc.user_message, "error")
        rendered[-2] = gr.update(value=None, visible=False)
        rendered[-1] = gr.update(value=None, visible=False)
        return rendered
    rendered = render_all(state, "CSVを書き出しました", "success")
    rendered[-2] = gr.update(value=None, visible=False)
    rendered[-1] = gr.update(value=path, visible=True)
    return rendered


def open_speaker_page(state_dict: dict):
    state = parse_state(state_dict)
    state.current_page = "speaker_page"
    return render_all(state, "話者一覧を開きました", "info")


def back_to_editor(state_dict: dict):
    state = parse_state(state_dict)
    state.current_page = "episode_editor"
    return render_all(state, "話数編集に戻りました", "info")


def select_speaker(slot_speaker_id: str | None, state_dict: dict):
    state = parse_state(state_dict)
    state.selected_speaker_id = slot_speaker_id or ""
    return render_all(state, "話者を選択しました", "info")


def create_speaker(state_dict: dict):
    state = add_speaker_profile(parse_state(state_dict))
    return render_all(state, "話者を追加しました", "success")


def apply_rename(active_speaker_id: str | None, new_name: str, state_dict: dict):
    state = parse_state(state_dict)
    if not active_speaker_id:
        return render_all(state, "話者を選択してください", "error")
    state.selected_speaker_id = active_speaker_id
    state = rename_speaker(state, active_speaker_id, new_name)
    return render_all(state, "名前を反映しました", "success")


def apply_merge(active_speaker_id: str | None, merge_source_id: str | None, state_dict: dict):
    state = parse_state(state_dict)
    if not active_speaker_id or not merge_source_id:
        return render_all(state, "統合する話者を選んでください", "error")
    state.selected_speaker_id = active_speaker_id
    state = merge_speakers(state, merge_source_id, active_speaker_id)
    return render_all(state, "話者を統合しました", "success")


def apply_swap(active_speaker_id: str | None, swap_target_id: str | None, state_dict: dict):
    state = parse_state(state_dict)
    if not active_speaker_id or not swap_target_id:
        return render_all(state, "入れ替える話者を選んでください", "error")
    state.selected_speaker_id = active_speaker_id
    state = swap_speakers(state, active_speaker_id, swap_target_id)
    return render_all(state, "話者を入れ替えました", "success")


def move_selected_utterance(active_speaker_id: str | None, segment_ids, target_speaker_id: str | None, state_dict: dict):
    state = parse_state(state_dict)
    selected_segment_ids = []
    if isinstance(segment_ids, list):
        selected_segment_ids = [str(segment_id) for segment_id in segment_ids if str(segment_id).strip()]
    elif segment_ids:
        selected_segment_ids = [str(segment_ids)]

    if not active_speaker_id or not selected_segment_ids or not target_speaker_id:
        return render_all(state, "移動するセリフと移動先の話者を選んでください", "error")
    state.selected_speaker_id = active_speaker_id
    for segment_id in selected_segment_ids:
        state = move_segment_to_speaker(state, segment_id, target_speaker_id)
    return render_all(state, f"{len(selected_segment_ids)}件のセリフを他の話者へ移動しました", "success")


def select_speaker_utterance(segment_id: str | None, state_dict: dict):
    state = parse_state(state_dict)
    if not segment_id:
        return "", gr.update(value="")
    episode = get_selected_episode(state)
    if episode is None:
        return "", gr.update(value="")
    segment = next((item for item in episode.subtitle_segments if item.id == segment_id), None)
    if segment is None:
        return "", gr.update(value="")
    return segment.id, gr.update(value=segment.edited_text)


def save_speaker_utterance(active_speaker_id: str | None, segment_id: str | None, new_text: str, state_dict: dict):
    state = parse_state(state_dict)
    if not active_speaker_id or not segment_id:
        return render_all(state, "編集するセリフを選んでください", "error")
    state.selected_speaker_id = active_speaker_id
    state = update_segment_text(state, segment_id, new_text)
    return render_all(state, "セリフを更新しました", "success")


def delete_selected_speaker(active_speaker_id: str | None, state_dict: dict):
    state = parse_state(state_dict)
    if not active_speaker_id:
        return render_all(state, "削除する話者を選んでください", "error")
    episode = get_selected_episode(state)
    if episode is None:
        return render_all(state, "話者を選択してください", "error")
    target = next((speaker for speaker in episode.speakers if speaker.speaker_id == active_speaker_id), None)
    if target is None:
        return render_all(state, "削除する話者を選んでください", "error")
    if target.utterance_count > 0:
        return render_all(state, "話者を削除する前に、セリフを移動または統合してください", "error")
    state = delete_speaker_profile(state, active_speaker_id)
    return render_all(state, "話者を削除しました", "success")


def select_subtitle_segment(rows, state_dict: dict, evt: gr.SelectData):
    state = parse_state(state_dict)
    row_index = resolve_row_index(getattr(evt, "index", None))
    normalized_rows = rows.values.tolist() if hasattr(rows, "values") else rows or []
    speaker_choices = build_episode_speaker_choices(state)
    segment_choices = build_subtitle_segment_choices(state)
    if row_index is None or row_index >= len(normalized_rows):
        state.selected_subtitle_segment_id = ""
        return "", gr.update(value=build_selected_subtitle_label()), gr.update(choices=segment_choices, value=None), gr.update(value=""), gr.update(value=None, visible=False), gr.update(choices=speaker_choices, value=None)

    row = normalized_rows[row_index]
    segment_id = str(row[0]) if row else ""
    start_label = str(row[1]) if len(row) > 1 else ""
    duration_label = str(row[2]) if len(row) > 2 else ""
    speaker_name = str(row[3]) if len(row) > 3 else ""
    preview = str(row[4]) if len(row) > 4 else ""
    episode = get_selected_episode(state)
    current_speaker_id = None
    audio_update = gr.update(value=None, visible=False)
    if episode is not None:
        segment = next((item for item in episode.subtitle_segments if item.id == segment_id), None)
        if segment is not None:
            current_speaker_id = segment.speaker_id
            state.selected_subtitle_segment_id = segment.id
            audio_update = build_selected_segment_audio_update(episode, segment)
    return (
        segment_id,
        gr.update(value=build_selected_subtitle_label(start_label, duration_label, speaker_name, preview[:60])),
        gr.update(choices=segment_choices, value=normalize_dropdown_value(segment_id, segment_choices)),
        gr.update(value=speaker_name),
        audio_update,
        gr.update(choices=speaker_choices, value=current_speaker_id),
    )


def select_subtitle_segment_by_id(segment_id: str | None, state_dict: dict):
    state = parse_state(state_dict)
    episode = get_selected_episode(state)
    speaker_choices = build_episode_speaker_choices(state)
    segment_choices = build_subtitle_segment_choices(state)
    target_segment_id = (segment_id or "").strip()
    if episode is None or not target_segment_id:
        state.selected_subtitle_segment_id = ""
        return "", gr.update(value=build_selected_subtitle_label()), gr.update(choices=segment_choices, value=None), gr.update(value=""), gr.update(value=None, visible=False), gr.update(choices=speaker_choices, value=None)

    segment = next((item for item in episode.subtitle_segments if item.id == target_segment_id), None)
    if segment is None:
        state.selected_subtitle_segment_id = ""
        return "", gr.update(value=build_selected_subtitle_label()), gr.update(choices=segment_choices, value=None), gr.update(value=""), gr.update(value=None, visible=False), gr.update(choices=speaker_choices, value=None)

    state.selected_subtitle_segment_id = segment.id
    speaker_name = segment.display_name
    if segment.voiceprint_profile_id and segment.voiceprint_confidence < 0.75:
        speaker_name = f"MEDIUM | {segment.display_name}"
    return (
        segment.id,
        gr.update(
            value=build_selected_subtitle_label(
                f"[{format_episode_seconds(episode, get_display_segment_start(segment))}]",
                f"{get_display_segment_duration(segment):.2f}s",
                speaker_name,
                segment.edited_text[:60],
            )
        ),
        gr.update(choices=segment_choices, value=normalize_dropdown_value(segment.id, segment_choices)),
        gr.update(value=speaker_name),
        build_selected_segment_audio_update(episode, segment),
        gr.update(choices=speaker_choices, value=segment.speaker_id),
    )


def generate_voiceprint_candidates(state_dict: dict):
    state = parse_state(state_dict)
    episode = get_selected_episode(state)
    if episode is None:
        return render_all(state, "話数を選択してください", "error")
    if not episode.subtitle_segments:
        return render_all(state, "先に字幕を作成してください", "error")

    normalize_voiceprint_candidate_state(state)
    state.selected_voiceprint_candidate_id = state.voiceprint_candidates[0].candidate_id if state.voiceprint_candidates else ""
    if state.voiceprint_candidates:
        selected = state.voiceprint_candidates[0]
        state.selected_subtitle_segment_id = selected.source_segment_id
        state.selected_speaker_id = selected.speaker_id
        return render_all(
            state,
            f"高品質な声紋候補を {len(state.voiceprint_candidates)}件 表示しました",
            "success",
        )
    return render_all(state, "この話では高品質な声紋候補を作れませんでした", "error")


def select_voiceprint_candidate(candidate_id: str | None, state_dict: dict):
    state = parse_state(state_dict)
    normalize_voiceprint_candidate_state(state)
    state.selected_voiceprint_candidate_id = (candidate_id or "").strip()
    selected = find_selected_voiceprint_candidate(state)
    if selected is not None:
        state.selected_subtitle_segment_id = selected.source_segment_id
        state.selected_speaker_id = selected.speaker_id
        return render_all(state, "声紋サンプルを選択しました", "info")
    return render_all(state, "声紋サンプルを選択してください", "error")


def register_voiceprint(candidate_id: str | None, character_name: str | None, state_dict: dict):
    state = parse_state(state_dict)
    work = get_selected_work(state)
    episode = get_selected_episode(state)
    target_name = (character_name or "").strip()

    if work is None or episode is None:
        return render_all(state, "話数を選択してください", "error")
    if not target_name:
        return render_all(state, "登録先のキャラ名を選択してください", "error")
    normalize_voiceprint_candidate_state(state)
    chosen_candidate_id = (candidate_id or state.selected_voiceprint_candidate_id).strip()
    selected_candidate = next(
        (item for item in state.voiceprint_candidates if item.candidate_id == chosen_candidate_id),
        None,
    )
    if selected_candidate is None:
        return render_all(state, "声紋候補を選択してください", "error")
    target_segment_id = selected_candidate.source_segment_id

    segment = next((item for item in episode.subtitle_segments if item.id == target_segment_id), None)
    if segment is None:
        return render_all(state, "選択したセリフが見つかりません", "error")
    original_speaker_id = segment.speaker_id
    state.selected_subtitle_segment_id = segment.id
    state.selected_speaker_id = segment.speaker_id
    state.selected_voiceprint_character_name = target_name
    state.selected_voiceprint_candidate_id = selected_candidate.candidate_id

    if target_name not in work.character_names:
        work.character_names.append(target_name)
        work.character_names.sort()
        sync_work_dictionary(DATA_DIR, work.work_id, work.title, work.character_names)

    if episode.wav_path and Path(episode.wav_path).exists():
        source_wav_path = episode.wav_path
        clip_start = selected_candidate.clip_start
        clip_end = selected_candidate.clip_end
    else:
        source_wav_path, clip_start, clip_end = get_segment_source_range(episode, segment)
    if not source_wav_path:
        return render_all(state, "声紋登録に使う音声パスが見つかりません", "error")

    ensure_voiceprint_storage(DATA_DIR, work.work_id)
    try:
        profiles, samples = load_voiceprint_state(DATA_DIR, work.work_id)
        embedding = extract_voice_embedding(source_wav_path, clip_start, clip_end)
        sample = build_voiceprint_sample(
            episode_id=episode.episode_id,
            speaker_id=segment.speaker_id,
            character_name=target_name,
            source_wav_path=source_wav_path,
            clip_start=clip_start,
            clip_end=clip_end,
            transcript_text=selected_candidate.transcript_text,
            embedding=embedding,
        )
        profiles, samples, profile = upsert_voiceprint_profile(
            work_id=work.work_id,
            character_name=target_name,
            sample=sample,
            profiles=profiles,
            samples=samples,
        )
        save_voiceprint_state(DATA_DIR, work.work_id, profiles, samples)
    except SpeakerIdentificationError as exc:
        return render_all(state, exc.user_message, "error")
    except PersistenceError as exc:
        return render_all(state, exc.user_message, "error")

    if episode.wav_path and Path(episode.wav_path).exists():
        transcription_segments = [
            TranscriptionSegment(
                start=segment.start,
                end=segment.end,
                text=segment.original_text or segment.edited_text,
                source_start=segment.source_start,
                source_end=segment.source_end,
            )
            for segment in episode.subtitle_segments
        ]
        try:
            current_assignments = assign_voiceprints_to_segments(
                episode.wav_path,
                transcription_segments,
                profiles,
            )
            state = apply_voiceprint_assignments_to_episode(state, current_assignments)
        except SpeakerIdentificationError:
            pass

    latest_episode = get_selected_episode(state)
    latest_segment = next(
        (item for item in latest_episode.subtitle_segments if item.id == target_segment_id),
        None,
    ) if latest_episode is not None else None
    if latest_segment is not None and latest_segment.voiceprint_profile_id:
        state.selected_speaker_id = latest_segment.speaker_id
    else:
        state = rename_speaker(state, original_speaker_id, target_name)
    ok, error = persist_state_or_error(state)
    if not ok:
        return render_all(state, error, "error")

    return render_all(state, f"{target_name} に声紋サンプルを登録し、現在の話へ再適用しました（{profile.sample_count}件）", "success")


def apply_episode_speaker_change(segment_id: str | None, target_speaker_id: str | None, state_dict: dict):
    state = parse_state(state_dict)
    if not segment_id:
        return render_all(state, "話者変更するセリフを選択してください", "error")
    if not target_speaker_id:
        return render_all(state, "変更先の話者を選んでください", "error")

    episode = get_selected_episode(state)
    if episode is None:
        return render_all(state, "話数を選択してください", "error")

    for _ in range(MAX_SPEAKER_SLOTS):
        episode = get_selected_episode(state)
        if episode is not None and any(speaker.speaker_id == target_speaker_id for speaker in episode.speakers):
            break
        state = add_speaker_profile(state)
    state = move_segment_to_speaker(state, segment_id, target_speaker_id)
    return render_all(state, "話者を変更しました", "success")


def build_partial_rerun_range(start_seconds: float, end_seconds: float) -> tuple[str, str]:
    padded_start = max(0.0, start_seconds - 1.0)
    padded_end = max(padded_start + 1.0, end_seconds + 1.0)
    return format_seconds(padded_start), format_seconds(padded_end)


def preview_partial_rerun(segment_id: str | None, state_dict: dict, progress=gr.Progress(track_tqdm=False)):
    state = parse_state(state_dict)
    episode = get_selected_episode(state)
    if episode is None:
        return render_all(state, "話数を選択してください", "error")
    if not segment_id:
        return render_all(state, "時刻を選択してから再読み込みしてください", "error")

    segment = next((item for item in episode.subtitle_segments if item.id == segment_id), None)
    if segment is None:
        return render_all(state, "再読み込みするセリフを選択してください", "error")

    source_path, source_range_start, source_range_end = get_segment_source_range(episode, segment)
    if not source_path or not Path(source_path).exists():
        return render_all(state, "再読み込み用の元ファイルが見つかりませんでした", "error")

    range_start, range_end = build_partial_rerun_range(source_range_start, source_range_end)
    prompt_text = build_transcription_prompt(state, episode.initial_prompt)

    progress(0.1, desc="選択区間を切り出しています...")
    try:
        preprocess_result = preprocess_media(
            file_path=source_path,
            range_start=range_start,
            range_end=range_end,
            data_dir=DATA_DIR,
        )
        progress(0.55, desc="候補を文字にしています...")
        transcription_segments = transcribe_wav(
            preprocess_result.wav_path,
            model_name=normalize_model_selection(episode.whisper_model),
            initial_prompt=prompt_text,
        )
    except MediaPreprocessError as exc:
        return render_all(state, exc.user_message, "error")
    except TranscriptionError as exc:
        return render_all(state, exc.user_message, "error")

    candidate_text = "".join(item.text.strip() for item in transcription_segments if item.text.strip()).strip()
    if not candidate_text:
        return render_all(state, "再読み込み候補を取得できませんでした", "error")

    state.selected_subtitle_segment_id = segment.id
    state.selected_subtitle_preview = f"[{format_episode_seconds(episode, get_display_segment_start(segment))}] {segment.edited_text[:40]}"
    state.rerun_candidate_label = "再読み込み候補を取得しました。反映するか確認してください。"
    state.rerun_candidate_range = f"{range_start} - {range_end}"
    state.rerun_candidate_text = candidate_text
    return render_all(state, "再読み込み候補を取得しました", "success")


def apply_partial_rerun_candidate(segment_id: str | None, state_dict: dict):
    state = parse_state(state_dict)
    target_segment_id = segment_id or state.selected_subtitle_segment_id
    if not target_segment_id or not state.rerun_candidate_text.strip():
        return render_all(state, "反映する候補がありません", "error")
    state = update_segment_text(state, target_segment_id, state.rerun_candidate_text)
    state.selected_subtitle_segment_id = target_segment_id
    state.rerun_candidate_label = ""
    state.rerun_candidate_range = ""
    state.rerun_candidate_text = ""
    return render_all(state, "再読み込み候補を反映しました", "success")


def clear_partial_rerun_candidate(state_dict: dict):
    state = parse_state(state_dict)
    state.rerun_candidate_label = ""
    state.rerun_candidate_range = ""
    state.rerun_candidate_text = ""
    return render_all(state, "再読み込み候補を破棄しました", "info")


custom_css = """
body { overflow-x: auto; }
.gradio-container { min-width: 1200px !important; }
.app-shell { max-width: 1180px; margin: 0 auto; width: 100%; overflow-x: auto; }
.list-table { min-height: 300px; }
.subtitle-table { min-height: 420px; }
.speaker-layout { gap: 16px; flex-wrap: nowrap !important; align-items: flex-start; }
.speaker-sidebar { border: 1px solid #30363d; border-radius: 14px; padding: 12px; background: #111827; min-width: 280px; }
.speaker-detail { border: 1px solid #30363d; border-radius: 14px; padding: 18px; background: #111827; min-width: 720px; }
.speaker-action { width: 100%; justify-content: flex-start; margin-bottom: 10px; }
.speaker-add { width: 100%; margin: 8px 0 16px; }
.speaker-samples { min-height: 220px; max-height: 320px; overflow-y: auto; border: 1px solid #30363d; border-radius: 12px; padding: 12px; background: #0f172a; color: #e5e7eb; }
.speaker-samples .wrap { display: grid; gap: 10px; }
.speaker-samples label { border: 1px solid #334155; border-radius: 10px; background: #111827; padding: 10px 12px; }
.speaker-samples label:has(input:checked) { border-color: #60a5fa; background: #172554; }
.speaker-samples .wrap span { color: #e5e7eb; }
.speaker-empty { color: #cbd5e1; }
.detail-header { gap: 12px; align-items: end; }
.export-files { margin-top: 8px; }
"""


with gr.Blocks(title="字幕ライブラリ", css=custom_css) as demo:
    app_state = gr.State(INITIAL_STATE.to_dict())

    speaker_select_buttons = []
    speaker_id_states = []

    gr.Markdown("# 字幕ライブラリ")
    status_box = gr.HTML(format_status_box(INITIAL_MESSAGE, INITIAL_KIND))

    with gr.Group(visible=True) as work_list_page:
        with gr.Column(elem_classes=["app-shell"]):
            with gr.Row():
                add_work_button = gr.Button("＋ 新しい作品を追加", variant="primary")
                reload_button = gr.Button("保存済みを再読込", variant="secondary")
            work_table = gr.Dataframe(
                headers=["作品名", "話数", "更新日"],
                datatype=["str", "str", "str"],
                interactive=False,
                value=[],
                elem_classes=["list-table"],
            )

    with gr.Group(visible=False) as work_detail_page:
        with gr.Column(elem_classes=["app-shell"]):
            with gr.Row():
                back_to_library_button = gr.Button("戻る", variant="secondary")
                work_title_markdown = gr.Markdown("## 作品詳細")
            with gr.Row(elem_classes=["detail-header"]):
                work_title_input = gr.Textbox(label="作品名", placeholder="作品名を入力")
                save_work_title_button = gr.Button("作品名を保存", variant="secondary")
            with gr.Row():
                add_episode_button = gr.Button("＋ 新しい話を追加", variant="primary")
                manage_characters_button = gr.Button("キャラ名を管理", variant="secondary")
            with gr.Column(visible=False) as character_manager_area:
                character_names_input = gr.Textbox(label="キャラ名候補", placeholder="例: ルフィ、ゾロ、ナミ")
                save_character_names_button = gr.Button("候補を保存")
                dictionary_table = gr.Dataframe(
                    headers=["読み", "置換後", "別名候補"],
                    datatype=["str", "str", "str"],
                    interactive=True,
                    value=[],
                    wrap=True,
                )
                save_dictionary_button = gr.Button("辞書を保存", variant="secondary")
            episode_table = gr.Dataframe(
                headers=["話数", "状態", "更新日"],
                datatype=["str", "str", "str"],
                interactive=False,
                value=[],
                elem_classes=["list-table"],
            )

    with gr.Group(visible=False) as episode_editor_page:
        with gr.Column(elem_classes=["app-shell"]):
            with gr.Row():
                back_to_detail_button = gr.Button("戻る", variant="secondary")
                editor_title_markdown = gr.Markdown("## 話数編集")
            with gr.Row():
                file_input = gr.File(label="ファイル選択", file_types=[".mp4", ".wav", ".mp3"], type="filepath")
                with gr.Column():
                    start_input = gr.Textbox(label="開始時刻", placeholder="00:00:00")
                    end_input = gr.Textbox(label="終了時刻", placeholder="00:00:00")
                    enhance_toggle = gr.Checkbox(label="音声を聞き取りやすくする", value=False)
                    whisper_model_input = gr.Dropdown(label="文字起こしエンジン", choices=MODEL_OPTIONS, value=DEFAULT_MODEL_NAME)
                    initial_prompt_input = gr.Textbox(label="補助プロンプト", lines=4, placeholder="例: 固有名詞候補、言い間違えやすい単語、キャラ名")
                    generate_button = gr.Button("字幕を作成", variant="primary")
            with gr.Row():
                save_subtitles_button = gr.Button("字幕編集を保存", variant="secondary")
                save_episode_button = gr.Button("保存", variant="primary")
                export_txt_button = gr.Button("テキストを書き出す", variant="secondary")
                export_csv_button = gr.Button("CSVを書き出す", variant="secondary")
                open_speakers_button = gr.Button("☰ 話者一覧", variant="secondary")
            subtitle_table = gr.Dataframe(
                headers=["segment_id", "開始", "長さ", "話者", "セリフ"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=True,
                value=[],
                wrap=True,
                elem_classes=["subtitle-table"],
            )
            selected_subtitle_segment = gr.State(value="")
            with gr.Row():
                selected_subtitle_label = gr.Markdown("話者欄をクリックして変更するセリフを選択してください")
                selected_subtitle_picker = gr.Dropdown(label="選択セリフ", choices=[], value=None)
                selected_subtitle_current_speaker = gr.Textbox(label="現在の話者", interactive=False, value="")
                selected_subtitle_audio = gr.Audio(label="選択セリフ音声", interactive=False, visible=False, type="filepath")
            with gr.Row():
                speaker_change_target = gr.Dropdown(
                    label="変更先の話者",
                    choices=[],
                    value=None,
                )
                apply_speaker_change_button = gr.Button("選択中のセリフの話者を変更", variant="secondary")
            speaker_diagnostics_box = gr.Textbox(label="話者分離診断", lines=5, interactive=False, value="話者分離診断: まだありません。")
            voiceprint_selection_label = gr.Markdown("選択中のセリフを声紋登録できます。", visible=False)
            create_voiceprint_samples_button = gr.Button("声紋サンプルを作成", variant="secondary", visible=False)
            voiceprint_sample_input = gr.Dropdown(
                label="声紋サンプル候補",
                choices=[],
                value=None,
                visible=False,
            )
            with gr.Row():
                voiceprint_character_input = gr.Dropdown(
                    label="声紋の登録先キャラ",
                    choices=[],
                    value=None,
                    allow_custom_value=True,
                )
                register_voiceprint_button = gr.Button("選択セリフを声紋登録", variant="secondary")
                rerun_selected_button = gr.Button("選択セリフを再読み込み", variant="secondary")
            voiceprint_summary_box = gr.Textbox(label="声紋登録状況", lines=4, interactive=False, value="声紋登録: まだありません。")
            rerun_candidate_label = gr.Markdown("選択した時刻を再読み込みすると候補がここに表示されます。")
            with gr.Row():
                rerun_candidate_range = gr.Textbox(label="再読み込み範囲", interactive=False, value="")
            rerun_candidate_textbox = gr.Textbox(label="再読み込み候補", lines=4, interactive=False, value="")
            with gr.Row():
                apply_rerun_candidate_button = gr.Button("候補を反映", variant="secondary")
                clear_rerun_candidate_button = gr.Button("候補を破棄", variant="secondary")
            with gr.Row(elem_classes=["export-files"]):
                txt_export_file = gr.File(label="TXT書き出し", interactive=False, visible=False)
                csv_export_file = gr.File(label="CSV書き出し", interactive=False, visible=False)

    with gr.Group(visible=False) as speaker_page:
        with gr.Column(elem_classes=["app-shell"]):
            with gr.Row():
                gr.Markdown("# 話者一覧")
            with gr.Row(elem_classes=["speaker-layout"]):
                with gr.Column(scale=1, elem_classes=["speaker-sidebar"]):
                    gr.Markdown("### 左の一覧から話者を選択")
                    add_speaker_button = gr.Button("＋ 話者を追加", variant="primary", elem_classes=["speaker-add"])
                    for index in range(MAX_SPEAKER_SLOTS):
                        speaker_button = gr.Button(value=f"話者{index + 1}", visible=False, elem_classes=["speaker-action"])
                        speaker_select_buttons.append(speaker_button)
                        speaker_id_state = gr.State(value=None)
                        speaker_id_states.append(speaker_id_state)
                    back_to_editor_button = gr.Button("字幕結果に戻る", variant="secondary")
                with gr.Column(scale=2, elem_classes=["speaker-detail"]):
                    speaker_detail_title = gr.Markdown("### 話者を選択してください")
                    gr.Markdown("白い欄は選択中話者のセリフ一覧です。ここから移動・統合・入れ替えを行えます。")
                    active_speaker_id = gr.State(value=None)
                    raw_label_box = gr.Textbox(label="元ラベル", interactive=False)
                    speaker_name_input = gr.Textbox(label="キャラ名", placeholder="例: ルフィ")
                    rename_button = gr.Button("名前を反映")
                    speaker_sample_box = gr.Radio(label="セリフ一覧をクリックして編集", choices=[], value=None, elem_classes=["speaker-samples"])
                    editable_segment_id = gr.State(value="")
                    speaker_edit_selector = gr.Radio(label="旧セリフ選択", choices=[], value=None, visible=False)
                    speaker_edit_input = gr.Textbox(label="選択したセリフを編集", lines=3, placeholder="ここでセリフを修正")
                    save_speaker_edit_button = gr.Button("選択したセリフを保存", variant="secondary")
                    move_segment = gr.Dropdown(label="ほかの話者へ移動するセリフ", choices=[], value=[], multiselect=True)
                    move_target = gr.Dropdown(label="移動先の話者", choices=[], value=None)
                    move_button = gr.Button("選んだセリフを移動")
                    merge_target = gr.Dropdown(label="他の話者をまとめる", choices=[], value=None)
                    merge_button = gr.Button("この話者に統合")
                    swap_target = gr.Dropdown(label="話者を入れ替える", choices=[], value=None)
                    swap_button = gr.Button("入れ替える")
                    delete_speaker_button = gr.Button("話者を削除", variant="stop", interactive=False)

    speaker_list_outputs = []
    for index in range(MAX_SPEAKER_SLOTS):
        speaker_list_outputs.extend([speaker_select_buttons[index], speaker_id_states[index]])

    full_outputs = [
        app_state,
        status_box,
        work_list_page,
        work_detail_page,
        episode_editor_page,
        speaker_page,
        work_table,
        work_title_markdown,
        work_title_input,
        character_manager_area,
        character_names_input,
        dictionary_table,
        episode_table,
        editor_title_markdown,
        start_input,
        end_input,
        enhance_toggle,
        whisper_model_input,
        initial_prompt_input,
        subtitle_table,
        selected_subtitle_segment,
        selected_subtitle_label,
        selected_subtitle_picker,
        selected_subtitle_current_speaker,
        selected_subtitle_audio,
        speaker_change_target,
        speaker_diagnostics_box,
        voiceprint_selection_label,
        voiceprint_sample_input,
        voiceprint_character_input,
        voiceprint_summary_box,
        rerun_candidate_label,
        rerun_candidate_range,
        rerun_candidate_textbox,
        *speaker_list_outputs,
        speaker_detail_title,
        raw_label_box,
        speaker_name_input,
        speaker_sample_box,
        speaker_edit_selector,
        speaker_edit_input,
        merge_target,
        swap_target,
        move_segment,
        move_target,
        editable_segment_id,
        delete_speaker_button,
        active_speaker_id,
        txt_export_file,
        csv_export_file,
    ]

    demo.load(
        fn=lambda state_dict: render_all(parse_state(state_dict), INITIAL_MESSAGE, INITIAL_KIND),
        inputs=[app_state],
        outputs=full_outputs,
    )
    add_work_button.click(fn=add_work, inputs=[app_state], outputs=full_outputs, queue=False)
    reload_button.click(fn=reload_library, inputs=[app_state], outputs=full_outputs, queue=False)
    work_table.select(fn=open_work, inputs=[app_state], outputs=full_outputs, queue=False)
    back_to_library_button.click(fn=back_to_work_list, inputs=[app_state], outputs=full_outputs, queue=False)
    save_work_title_button.click(fn=save_work_title, inputs=[work_title_input, app_state], outputs=full_outputs, queue=False)
    manage_characters_button.click(
        fn=toggle_character_manager,
        inputs=[app_state],
        outputs=[app_state, status_box, character_manager_area, character_names_input, dictionary_table],
        queue=False,
    )
    save_character_names_button.click(fn=save_character_candidates, inputs=[character_names_input, app_state], outputs=full_outputs, queue=False)
    save_dictionary_button.click(fn=save_dictionary_entries, inputs=[dictionary_table, app_state], outputs=full_outputs, queue=False)
    add_episode_button.click(fn=add_episode, inputs=[app_state], outputs=full_outputs, queue=False)
    episode_table.select(fn=open_episode, inputs=[app_state], outputs=full_outputs, queue=False)
    back_to_detail_button.click(fn=back_to_work_detail, inputs=[app_state], outputs=full_outputs, queue=False)
    generate_button.click(fn=generate_subtitles, inputs=[file_input, start_input, end_input, enhance_toggle, whisper_model_input, initial_prompt_input, app_state], outputs=full_outputs, queue=True)
    subtitle_table.change(
        fn=sync_subtitle_edits_inline,
        inputs=[subtitle_table, app_state],
        outputs=[app_state, status_box],
        queue=False,
    )
    subtitle_table.select(
        fn=select_subtitle_segment,
        inputs=[subtitle_table, app_state],
        outputs=[selected_subtitle_segment, selected_subtitle_label, selected_subtitle_picker, selected_subtitle_current_speaker, selected_subtitle_audio, speaker_change_target],
        queue=False,
    )
    selected_subtitle_picker.change(
        fn=select_subtitle_segment_by_id,
        inputs=[selected_subtitle_picker, app_state],
        outputs=[selected_subtitle_segment, selected_subtitle_label, selected_subtitle_picker, selected_subtitle_current_speaker, selected_subtitle_audio, speaker_change_target],
        queue=False,
    )
    apply_speaker_change_button.click(
        fn=apply_episode_speaker_change,
        inputs=[selected_subtitle_segment, speaker_change_target, app_state],
        outputs=full_outputs,
        queue=False,
    )
    create_voiceprint_samples_button.click(
        fn=generate_voiceprint_candidates,
        inputs=[app_state],
        outputs=full_outputs,
        queue=False,
    )
    voiceprint_sample_input.change(
        fn=select_voiceprint_candidate,
        inputs=[voiceprint_sample_input, app_state],
        outputs=full_outputs,
        queue=False,
    )
    register_voiceprint_button.click(
        fn=register_voiceprint,
        inputs=[voiceprint_sample_input, voiceprint_character_input, app_state],
        outputs=full_outputs,
        queue=False,
    )
    rerun_selected_button.click(fn=preview_partial_rerun, inputs=[selected_subtitle_segment, app_state], outputs=full_outputs, queue=True)
    apply_rerun_candidate_button.click(fn=apply_partial_rerun_candidate, inputs=[selected_subtitle_segment, app_state], outputs=full_outputs, queue=False)
    clear_rerun_candidate_button.click(fn=clear_partial_rerun_candidate, inputs=[app_state], outputs=full_outputs, queue=False)
    save_subtitles_button.click(fn=sync_subtitle_edits, inputs=[subtitle_table, app_state], outputs=full_outputs, queue=False)
    save_episode_button.click(fn=save_current_episode, inputs=[app_state], outputs=full_outputs, queue=False)
    export_txt_button.click(fn=export_txt, inputs=[app_state], outputs=full_outputs, queue=False)
    export_csv_button.click(fn=export_csv, inputs=[app_state], outputs=full_outputs, queue=False)
    open_speakers_button.click(fn=open_speaker_page, inputs=[app_state], outputs=full_outputs, queue=False)
    back_to_editor_button.click(fn=back_to_editor, inputs=[app_state], outputs=full_outputs, queue=False)
    add_speaker_button.click(fn=create_speaker, inputs=[app_state], outputs=full_outputs, queue=False)

    for index in range(MAX_SPEAKER_SLOTS):
        speaker_select_buttons[index].click(
            fn=select_speaker,
            inputs=[speaker_id_states[index], app_state],
            outputs=full_outputs,
            queue=False,
        )

    rename_button.click(fn=apply_rename, inputs=[active_speaker_id, speaker_name_input, app_state], outputs=full_outputs, queue=False)
    speaker_sample_box.change(fn=select_speaker_utterance, inputs=[speaker_sample_box, app_state], outputs=[editable_segment_id, speaker_edit_input], queue=False)
    save_speaker_edit_button.click(fn=save_speaker_utterance, inputs=[active_speaker_id, editable_segment_id, speaker_edit_input, app_state], outputs=full_outputs, queue=False)
    move_button.click(fn=move_selected_utterance, inputs=[active_speaker_id, move_segment, move_target, app_state], outputs=full_outputs, queue=False)
    merge_button.click(fn=apply_merge, inputs=[active_speaker_id, merge_target, app_state], outputs=full_outputs, queue=False)
    swap_button.click(fn=apply_swap, inputs=[active_speaker_id, swap_target, app_state], outputs=full_outputs, queue=False)
    delete_speaker_button.click(fn=delete_selected_speaker, inputs=[active_speaker_id, app_state], outputs=full_outputs, queue=False)


if __name__ == "__main__":
    demo.launch()

