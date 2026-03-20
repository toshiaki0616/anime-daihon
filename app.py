from __future__ import annotations

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
    get_selected_episode,
    get_selected_work,
    merge_speakers,
    rename_speaker,
    swap_speakers,
    update_character_names,
    update_work_title,
)
from models.state import AppState
from services import (
    DiarizationError,
    MediaPreprocessError,
    TranscriptionError,
    diarize_wav,
    preprocess_media,
    transcribe_wav,
)
from ui.renderers import (
    build_episode_rows,
    build_speaker_detail_payload,
    build_speaker_list_payloads,
    build_subtitle_rows,
    build_work_rows,
    format_status_box,
)

DATA_DIR = Path(__file__).parent / "data"


def parse_state(state_dict: dict | None) -> AppState:
    return AppState.from_dict(state_dict or build_mock_app_state().to_dict())


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
        build_subtitle_rows(state),
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
            gr.update(value=speaker_detail["utterances_html"]),
            gr.update(choices=speaker_detail["merge_choices"], value=None),
            gr.update(choices=speaker_detail["swap_choices"], value=None),
            speaker_detail["speaker_id"],
        ]
    )
    return outputs


def add_work(state_dict: dict):
    state = create_work(parse_state(state_dict))
    return render_all(state, "新しい作品を追加しました", "success")


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
    return render_all(state, "作品名を更新しました", "success")


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
    return render_all(state, "キャラ名候補を更新しました", "success")


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


def generate_subtitles(file_path: str | None, start_time: str, end_time: str, enhance_audio: bool, state_dict: dict, progress=gr.Progress(track_tqdm=False)):
    state = parse_state(state_dict)
    episode = get_selected_episode(state)
    if episode is None:
        return render_all(state, "話数を選択してください", "error")
    if not file_path:
        return render_all(state, "ファイルを選んでください", "error")

    progress(0.05, desc="???????????...")

    try:
        preprocess_result = preprocess_media(
            file_path=file_path,
            range_start=start_time.strip(),
            range_end=end_time.strip(),
            data_dir=DATA_DIR,
        )
        progress(0.4, desc="???????????...")
        transcription_segments = transcribe_wav(preprocess_result.wav_path)
    except MediaPreprocessError as exc:
        return render_all(state, exc.user_message, "error")
    except TranscriptionError:
        return render_all(state, "字幕の作成に失敗しました", "error")

    diarization_segments = []
    diarization_failed = False
    progress(0.72, desc="???????????...")
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
    if diarization_failed:
        return render_all(state, "話者の分割に失敗しました", "error")
    return render_all(state, "字幕を作成しました", "success")


def sync_subtitle_edits(rows, state_dict: dict):
    normalized_rows = rows.values.tolist() if hasattr(rows, "values") else rows or []
    state = apply_subtitle_edits(parse_state(state_dict), normalized_rows)
    return render_all(state, "字幕編集を反映しました", "success")


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
.speaker-samples { min-height: 220px; max-height: 320px; overflow-y: auto; border: 1px solid #30363d; border-radius: 12px; padding: 0; background: #0f172a; color: #e5e7eb; }
.speaker-sample-list { padding: 14px; }
.speaker-sample-list ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }
.speaker-sample-list li { padding: 10px 12px; border: 1px solid #334155; border-radius: 10px; background: #111827; }
.speaker-sample-caption { font-weight: 700; margin-bottom: 12px; color: #f8fafc; }
.speaker-time { display: inline-block; min-width: 88px; color: #93c5fd; font-weight: 700; }
.speaker-empty { color: #cbd5e1; }
.detail-header { gap: 12px; align-items: end; }
"""


with gr.Blocks(title="字幕ライブラリ", css=custom_css) as demo:
    app_state = gr.State(build_mock_app_state().to_dict())

    speaker_select_buttons = []
    speaker_id_states = []

    gr.Markdown("# 字幕ライブラリ")
    status_box = gr.HTML(format_status_box("モックデータで UI を確認できます。"))

    with gr.Group(visible=True) as work_list_page:
        with gr.Column(elem_classes=["app-shell"]):
            add_work_button = gr.Button("＋ 新しい作品を追加", variant="primary")
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
                    generate_button = gr.Button("字幕を作成", variant="primary")
            subtitle_table = gr.Dataframe(
                headers=["segment_id", "時刻", "話者", "セリフ"],
                datatype=["str", "str", "str", "str"],
                interactive=True,
                value=[],
                wrap=True,
                elem_classes=["subtitle-table"],
            )
            with gr.Row():
                save_subtitles_button = gr.Button("字幕編集を保存", variant="secondary")
                open_speakers_button = gr.Button("☰ 話者一覧", variant="secondary")

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
                    gr.Markdown("白い欄だった部分は、選択中話者のセリフ一覧です。見やすいように濃い背景へ変更しています。")
                    active_speaker_id = gr.State(value=None)
                    raw_label_box = gr.Textbox(label="元ラベル", interactive=False)
                    speaker_name_input = gr.Textbox(label="キャラ名", placeholder="例: ルフィ")
                    rename_button = gr.Button("名前を反映")
                    speaker_sample_box = gr.HTML(elem_classes=["speaker-samples"])
                    merge_target = gr.Dropdown(label="他の話者をまとめる", choices=[], value=None)
                    merge_button = gr.Button("この話者に統合")
                    swap_target = gr.Dropdown(label="話者を入れ替える", choices=[], value=None)
                    swap_button = gr.Button("入れ替える")

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
        subtitle_table,
        *speaker_list_outputs,
        speaker_detail_title,
        raw_label_box,
        speaker_name_input,
        speaker_sample_box,
        merge_target,
        swap_target,
        active_speaker_id,
    ]

    demo.load(
        fn=lambda state_dict: render_all(parse_state(state_dict), "モックライブラリを読み込みました", "info"),
        inputs=[app_state],
        outputs=full_outputs,
    )
    add_work_button.click(fn=add_work, inputs=[app_state], outputs=full_outputs, queue=False)
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
    generate_button.click(fn=generate_subtitles, inputs=[file_input, start_input, end_input, enhance_toggle, app_state], outputs=full_outputs, queue=True)
    subtitle_table.change(fn=sync_subtitle_edits, inputs=[subtitle_table, app_state], outputs=full_outputs, queue=False)
    save_subtitles_button.click(fn=sync_subtitle_edits, inputs=[subtitle_table, app_state], outputs=full_outputs, queue=False)
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
    merge_button.click(fn=apply_merge, inputs=[active_speaker_id, merge_target, app_state], outputs=full_outputs, queue=False)
    swap_button.click(fn=apply_swap, inputs=[active_speaker_id, swap_target, app_state], outputs=full_outputs, queue=False)


if __name__ == "__main__":
    demo.launch()
