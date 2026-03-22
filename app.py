from __future__ import annotations

from datetime import datetime
import gradio as gr
from pathlib import Path

from core.state_ops import (
    MAX_SPEAKER_SLOTS,
    add_speaker_profile,
    apply_subtitle_edits,
    apply_transcription_segments,
    build_mock_app_state,
    create_episode,
    create_work,
    delete_speaker_profile,
    get_selected_episode,
    get_selected_work,
    merge_speakers,
    move_segment_to_speaker,
    rename_speaker,
    swap_speakers,
    sync_episode,
    update_character_names,
    update_work_title,
    update_segment_text,
)
from models.state import AppState
from services import (
    DiarizationError,
    MediaPreprocessError,
    PersistenceError,
    TranscriptionError,
    diarize_wav,
    export_episode_csv,
    export_episode_txt,
    load_library_state,
    preprocess_media,
    save_library_state,
    transcribe_wav,
    MODEL_OPTIONS,
)
from ui.renderers import (
    build_episode_rows,
    build_speaker_detail_payload,
    build_speaker_list_payloads,
    build_subtitle_rows,
    build_work_rows,
    format_seconds,
    format_status_box,
)

DATA_DIR = Path(__file__).parent / "data"


def now_label() -> str:
    return datetime.now().replace(microsecond=0).isoformat(timespec="seconds")


def load_initial_state() -> tuple[AppState, str, str]:
    try:
        state = load_library_state(DATA_DIR)
    except PersistenceError:
        return build_mock_app_state(), "読み込みに失敗しました", "error"

    if not state.works:
        return build_mock_app_state(), "モックライブラリを読み込みました", "info"

    for work in state.works:
        for episode in work.episodes:
            sync_episode(episode)
    return state, "保存済みライブラリを読み込みました", "info"


INITIAL_STATE, INITIAL_MESSAGE, INITIAL_KIND = load_initial_state()


def parse_state(state_dict: dict | None) -> AppState:
    return AppState.from_dict(state_dict or INITIAL_STATE.to_dict())


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
    state.selected_speaker_id = resolve_selected_speaker_id(state)
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
        build_episode_rows(state),
        gr.update(value=f"## {episode.title if episode else '話数編集'}"),
        episode.range_start if episode else "",
        episode.range_end if episode else "",
        episode.enhance_audio if episode else False,
        episode.whisper_model if episode else "base",
        episode.initial_prompt if episode else "",
        build_subtitle_rows(state),
        "",
        gr.update(value="話者欄をクリックして変更するセリフを選択してください"),
        gr.update(value=""),
        gr.update(choices=build_episode_speaker_choices(state), value=None),
        gr.update(value=episode.speaker_diagnostics if episode and episode.speaker_diagnostics else "話者分離診断: まだありません。"),
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
    return render_all(state, "新しい作品を追加しました", "success")


def reload_library(state_dict: dict | None = None):
    try:
        state = load_library_state(DATA_DIR)
    except PersistenceError:
        return render_all(parse_state(state_dict), "読み込みに失敗しました", "error")
    if not state.works:
        state = build_mock_app_state()
        return render_all(state, "保存済みデータがないためモックライブラリを表示しています", "info")
    for work in state.works:
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
    return render_all(state, "作品一覧に戻りました", "info")


def save_work_title(new_title: str, state_dict: dict):
    state = update_work_title(parse_state(state_dict), new_title)
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
    ]


def save_character_candidates(raw_text: str, state_dict: dict):
    state = update_character_names(parse_state(state_dict), raw_text)
    ok, error = persist_state_or_error(state)
    if not ok:
        return render_all(state, error, "error")
    return render_all(state, "キャラ名候補を保存しました", "success")


def add_episode(state_dict: dict):
    state = create_episode(parse_state(state_dict))
    return render_all(state, "新しい話を追加しました", "success")


def open_episode(select_data: gr.SelectData, state_dict: dict):
    state = parse_state(state_dict)
    episodes = sorted_episodes_for_selected_work(state)
    row_index = resolve_row_index(select_data.index)
    if row_index is None or row_index >= len(episodes):
        return render_all(state, "話数を選択してください", "error")
    state.selected_episode_id = episodes[row_index].episode_id
    state.current_page = "episode_editor"
    return render_all(state, "話数編集を開きました", "info")


def back_to_work_detail(state_dict: dict):
    state = parse_state(state_dict)
    state.current_page = "work_detail"
    return render_all(state, "作品詳細に戻りました", "info")


def generate_subtitles(file_path: str | None, start_time: str, end_time: str, enhance_audio: bool, whisper_model: str, initial_prompt: str, state_dict: dict, progress=gr.Progress(track_tqdm=False)):
    state = parse_state(state_dict)
    episode = get_selected_episode(state)
    if episode is None:
        return render_all(state, "話数を選択してください", "error")
    if not file_path:
        return render_all(state, "ファイルを選んでください", "error")

    progress(0.05, desc="音声を取り出しています...")

    episode.range_start = start_time.strip()
    episode.range_end = end_time.strip()
    episode.enhance_audio = enhance_audio
    episode.whisper_model = whisper_model or "base"
    episode.initial_prompt = initial_prompt.strip()
    prompt_text = build_transcription_prompt(state, initial_prompt)

    try:
        preprocess_result = preprocess_media(
            file_path=file_path,
            range_start=start_time.strip(),
            range_end=end_time.strip(),
            data_dir=DATA_DIR,
        )
        progress(0.4, desc="音声を文字にしています...")
        transcription_segments = transcribe_wav(
            preprocess_result.wav_path,
            model_name=episode.whisper_model,
            initial_prompt=prompt_text,
        )
    except MediaPreprocessError as exc:
        return render_all(state, exc.user_message, "error")
    except TranscriptionError:
        return render_all(state, "字幕の作成に失敗しました", "error")

    diarization_segments = []
    diarization_failed = False
    progress(0.72, desc="話者ごとに分けています...")
    try:
        diarization_segments = diarize_wav(preprocess_result.wav_path)
    except DiarizationError:
        diarization_failed = True

    state = apply_transcription_segments(
        state=state,
        file_path=preprocess_result.source_path,
        wav_path=preprocess_result.wav_path,
        range_start=preprocess_result.range_start,
        range_end=preprocess_result.range_end,
        enhance_audio=enhance_audio,
        transcription_segments=transcription_segments,
        diarization_segments=diarization_segments,
    )
    progress(1.0, desc="字幕を作成しました")
    if diarization_failed:
        return render_all(state, "話者の分割に失敗しました", "error")
    return render_all(state, "字幕を作成しました", "success")


def sync_subtitle_edits(rows, state_dict: dict):
    normalized_rows = rows.values.tolist() if hasattr(rows, "values") else rows or []
    state = apply_subtitle_edits(parse_state(state_dict), normalized_rows)
    return render_all(state, "字幕編集を反映しました", "success")


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
    if row_index is None or row_index >= len(normalized_rows):
        return "", gr.update(value="話者欄をクリックして変更するセリフを選択してください"), gr.update(value=""), gr.update(choices=speaker_choices, value=None)

    row = normalized_rows[row_index]
    segment_id = str(row[0]) if row else ""
    timestamp = str(row[1]) if len(row) > 1 else ""
    speaker_name = str(row[2]) if len(row) > 2 else ""
    preview = str(row[3]) if len(row) > 3 else ""
    episode = get_selected_episode(state)
    current_speaker_id = None
    if episode is not None:
        segment = next((item for item in episode.subtitle_segments if item.id == segment_id), None)
        if segment is not None:
            current_speaker_id = segment.speaker_id
    return (
        segment_id,
        gr.update(value=f"選択中: {timestamp} {speaker_name} / {preview[:40]}"),
        gr.update(value=speaker_name),
        gr.update(choices=speaker_choices, value=current_speaker_id),
    )


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

    source_path = episode.file_path or episode.wav_path
    if not source_path or not Path(source_path).exists():
        return render_all(state, "再読み込み用の元ファイルが見つかりませんでした", "error")

    range_start, range_end = build_partial_rerun_range(segment.start, segment.end)
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
            model_name=episode.whisper_model or "base",
            initial_prompt=prompt_text,
        )
    except MediaPreprocessError as exc:
        return render_all(state, exc.user_message, "error")
    except TranscriptionError:
        return render_all(state, "字幕の作成に失敗しました", "error")

    candidate_text = "".join(item.text.strip() for item in transcription_segments if item.text.strip()).strip()
    if not candidate_text:
        return render_all(state, "再読み込み候補を取得できませんでした", "error")

    state.selected_subtitle_segment_id = segment.id
    state.selected_subtitle_preview = f"[{format_seconds(segment.start)}] {segment.edited_text[:40]}"
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
                    whisper_model_input = gr.Dropdown(label="文字起こし精度", choices=MODEL_OPTIONS, value="base")
                    initial_prompt_input = gr.Textbox(label="補助プロンプト", lines=4, placeholder="例: 固有名詞候補、言い間違えやすい単語、キャラ名")
                    generate_button = gr.Button("字幕を作成", variant="primary")
            subtitle_table = gr.Dataframe(
                headers=["segment_id", "時刻", "話者", "セリフ"],
                datatype=["str", "str", "str", "str"],
                interactive=True,
                value=[],
                wrap=True,
                elem_classes=["subtitle-table"],
            )
            selected_subtitle_segment = gr.State(value="")
            with gr.Row():
                selected_subtitle_label = gr.Markdown("話者欄をクリックして変更するセリフを選択してください")
                selected_subtitle_current_speaker = gr.Textbox(label="現在の話者", interactive=False, value="")
            with gr.Row():
                speaker_change_target = gr.Dropdown(
                    label="変更先の話者",
                    choices=[],
                    value=None,
                )
                apply_speaker_change_button = gr.Button("選択中のセリフの話者を変更", variant="secondary")
            speaker_diagnostics_box = gr.Textbox(label="話者分離診断", lines=5, interactive=False, value="話者分離診断: まだありません。")
            rerun_candidate_label = gr.Markdown("選択した時刻を再読み込みすると候補がここに表示されます。")
            with gr.Row():
                rerun_candidate_range = gr.Textbox(label="再読み込み範囲", interactive=False, value="")
                rerun_selected_button = gr.Button("選択した時刻を再読み込み", variant="secondary")
            rerun_candidate_textbox = gr.Textbox(label="再読み込み候補", lines=4, interactive=False, value="")
            with gr.Row():
                apply_rerun_candidate_button = gr.Button("候補を反映", variant="secondary")
                clear_rerun_candidate_button = gr.Button("候補を破棄", variant="secondary")
            with gr.Row():
                save_subtitles_button = gr.Button("字幕編集を保存", variant="secondary")
                save_episode_button = gr.Button("保存", variant="primary")
                export_txt_button = gr.Button("テキストを書き出す", variant="secondary")
                export_csv_button = gr.Button("CSVを書き出す", variant="secondary")
                open_speakers_button = gr.Button("☰ 話者一覧", variant="secondary")
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
        selected_subtitle_current_speaker,
        speaker_change_target,
        speaker_diagnostics_box,
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
        outputs=[app_state, status_box, character_manager_area, character_names_input],
        queue=False,
    )
    save_character_names_button.click(fn=save_character_candidates, inputs=[character_names_input, app_state], outputs=full_outputs, queue=False)
    add_episode_button.click(fn=add_episode, inputs=[app_state], outputs=full_outputs, queue=False)
    episode_table.select(fn=open_episode, inputs=[app_state], outputs=full_outputs, queue=False)
    back_to_detail_button.click(fn=back_to_work_detail, inputs=[app_state], outputs=full_outputs, queue=False)
    generate_button.click(fn=generate_subtitles, inputs=[file_input, start_input, end_input, enhance_toggle, whisper_model_input, initial_prompt_input, app_state], outputs=full_outputs, queue=True)
    subtitle_table.change(fn=sync_subtitle_edits, inputs=[subtitle_table, app_state], outputs=full_outputs, queue=False)
    subtitle_table.select(
        fn=select_subtitle_segment,
        inputs=[subtitle_table, app_state],
        outputs=[selected_subtitle_segment, selected_subtitle_label, selected_subtitle_current_speaker, speaker_change_target],
        queue=False,
    )
    apply_speaker_change_button.click(
        fn=apply_episode_speaker_change,
        inputs=[selected_subtitle_segment, speaker_change_target, app_state],
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

