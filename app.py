from __future__ import annotations

import copy
import re
from pathlib import Path

import gradio as gr

from core.state_ops import (
    MAX_SPEAKER_SLOTS,
    apply_subtitle_edits,
    merge_speakers,
    rename_speaker,
    swap_speakers,
)
from models.state import AppState
from services import MediaPreprocessError, preprocess_media
from services.mock_pipeline import run_mock_pipeline
from ui.renderers import (
    build_speaker_detail,
    build_speaker_slot_updates,
    build_subtitle_rows,
    format_status_box,
    make_empty_state,
)

TIME_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")
DATA_DIR = Path(__file__).parent / "data"


def parse_state(state_dict: dict | None) -> AppState:
    return AppState.from_dict(state_dict or make_empty_state())


def show_main_page():
    return gr.update(visible=True), gr.update(visible=False)


def show_speaker_page():
    return gr.update(visible=False), gr.update(visible=True)


def validate_times(start_time: str, end_time: str) -> str | None:
    if start_time and not TIME_PATTERN.match(start_time):
        return "時刻は 00:00:00 の形式で入力してください"
    if end_time and not TIME_PATTERN.match(end_time):
        return "時刻は 00:00:00 の形式で入力してください"
    if start_time and end_time and start_time >= end_time:
        return "終了時刻は開始時刻より後にしてください"
    return None


def resolve_selected_speaker(state: AppState, selected_speaker_id: str | None) -> str | None:
    speaker_ids = [speaker.speaker_id for speaker in state.speakers]
    if selected_speaker_id in speaker_ids:
        return selected_speaker_id
    return speaker_ids[0] if speaker_ids else None


def render_all(
    state: AppState,
    status_message: str,
    status_kind: str = "info",
    selected_speaker_id: str | None = None,
):
    resolved_speaker_id = resolve_selected_speaker(state, selected_speaker_id)
    detail = build_speaker_detail(state, resolved_speaker_id)

    outputs: list = [
        state.to_dict(),
        build_subtitle_rows(state),
        format_status_box(status_message, status_kind),
        resolved_speaker_id,
    ]

    for slot in build_speaker_slot_updates(state, resolved_speaker_id):
        outputs.extend(
            [
                gr.update(visible=slot["visible"], value=slot["button_label"], variant=slot["variant"]),
                slot["speaker_id"],
            ]
        )

    outputs.extend(
        [
            gr.update(value=detail["title"]),
            gr.update(value=detail["display_name"]),
            gr.update(value=detail["utterances_html"]),
            gr.update(choices=detail["merge_choices"], value=None),
            gr.update(choices=detail["swap_choices"], value=None),
            detail["speaker_id"],
        ]
    )
    return outputs


def render_status_only(state_dict: dict, message: str, kind: str = "info"):
    state = parse_state(state_dict)
    selected = resolve_selected_speaker(state, None)
    return render_all(state, message, kind, selected)


def load_mock_subtitles(file_path: str | None, start_time: str, end_time: str, enhance_audio: bool):
    if not file_path:
        return render_all(AppState(), "ファイルを選んでください", "error")

    start_time = start_time.strip()
    end_time = end_time.strip()
    error_message = validate_times(start_time, end_time)
    if error_message:
        return render_all(AppState(file_path=file_path), error_message, "error")

    try:
        preprocess_result = preprocess_media(
            file_path=file_path,
            range_start=start_time,
            range_end=end_time,
            data_dir=DATA_DIR,
        )
    except MediaPreprocessError as exc:
        return render_all(AppState(file_path=file_path), exc.user_message, "error")

    next_state = run_mock_pipeline(
        file_path=preprocess_result.source_path,
        wav_path=preprocess_result.wav_path,
        start_time=preprocess_result.range_start,
        end_time=preprocess_result.range_end,
        enhance_audio=enhance_audio,
    )
    return render_all(next_state, "字幕を作成しました", "success")


def save_subtitle_edits(rows: list[list[str]], state_dict: dict, selected_speaker_id: str | None):
    state = parse_state(state_dict)
    normalized_rows = rows
    if hasattr(rows, "values"):
        normalized_rows = rows.values.tolist()
    elif rows is None:
        normalized_rows = []
    next_state = apply_subtitle_edits(copy.deepcopy(state), normalized_rows)
    return render_all(next_state, "字幕を更新しました", "success", selected_speaker_id)


def handle_rename(
    active_speaker_id: str | None,
    new_name: str,
    state_dict: dict,
    selected_speaker_id: str | None,
):
    state = parse_state(state_dict)
    if not active_speaker_id:
        return render_all(state, "話者情報がありません", "error", selected_speaker_id)

    next_state = rename_speaker(copy.deepcopy(state), active_speaker_id, new_name)
    return render_all(next_state, "話者名を反映しました", "success", active_speaker_id)


def handle_merge(
    active_speaker_id: str | None,
    merge_source_id: str | None,
    state_dict: dict,
    selected_speaker_id: str | None,
):
    state = parse_state(state_dict)
    if not active_speaker_id or not merge_source_id:
        return render_all(state, "統合する話者を選んでください", "error", selected_speaker_id)

    next_state = merge_speakers(copy.deepcopy(state), merge_source_id, active_speaker_id)
    return render_all(next_state, "話者を統合しました", "success", active_speaker_id)


def handle_swap(
    active_speaker_id: str | None,
    swap_target_id: str | None,
    state_dict: dict,
    selected_speaker_id: str | None,
):
    state = parse_state(state_dict)
    if not active_speaker_id or not swap_target_id:
        return render_all(state, "入れ替える話者を選んでください", "error", selected_speaker_id)

    next_state = swap_speakers(copy.deepcopy(state), active_speaker_id, swap_target_id)
    return render_all(next_state, "話者を入れ替えました", "success", active_speaker_id)


def select_speaker(slot_speaker_id: str | None, state_dict: dict):
    state = parse_state(state_dict)
    return render_all(state, "話者を選択しました", "info", slot_speaker_id)


custom_css = """
.app-shell { max-width: 1180px; margin: 0 auto; }
.toolbar { align-items: end; }
.subtitle-header { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.subtitle-table { min-height: 420px; }
.speaker-sidebar { border: 1px solid #2f3542; border-radius: 14px; padding: 12px; background: #111827; }
.speaker-detail { border: 1px solid #2f3542; border-radius: 14px; padding: 18px; background: #111827; }
.speaker-list { min-height: 120px; max-height: 220px; overflow-y: auto; border: 1px solid #dbe4f0; border-radius: 12px; padding: 12px; background: #ffffff; }
.speaker-action { width: 100%; justify-content: flex-start; margin-bottom: 10px; }
.top-actions { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
"""


with gr.Blocks(title="字幕作成", css=custom_css) as demo:
    app_state = gr.State(make_empty_state())
    selected_speaker_state = gr.State(value=None)

    speaker_select_buttons = []
    speaker_slot_ids = []

    gr.Markdown("# 字幕作成")
    status_box = gr.HTML(format_status_box("ファイルを選んで、モック字幕を表示できます。"))

    with gr.Column(elem_classes=["app-shell"], visible=True) as main_page:
        with gr.Row(elem_classes=["toolbar"]):
            file_input = gr.File(
                label="動画・音声ファイル",
                file_types=[".mp4", ".wav", ".mp3"],
                type="filepath",
            )
            with gr.Column():
                start_input = gr.Textbox(label="開始時刻", placeholder="00:00:00")
                end_input = gr.Textbox(label="終了時刻", placeholder="00:00:00")
                enhance_toggle = gr.Checkbox(label="音声を聞き取りやすくする", value=False)
                run_button = gr.Button("字幕を作成", variant="primary")

        with gr.Row():
            gr.Markdown("## 字幕結果")
            open_speakers_button = gr.Button("☰ 話者一覧", variant="secondary")

        subtitle_table = gr.Dataframe(
            headers=["segment_id", "時刻", "話者", "セリフ"],
            datatype=["str", "str", "str", "str"],
            row_count=(0, "dynamic"),
            col_count=(4, "fixed"),
            value=[],
            interactive=True,
            wrap=True,
            elem_classes=["subtitle-table"],
        )
        save_edits_button = gr.Button("字幕編集を保存")

    with gr.Column(elem_classes=["app-shell"], visible=False) as speaker_page:
        gr.Markdown("# 話者一覧")
        gr.Markdown("話者ごとのセリフを確認して、名前を登録する")

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["speaker-sidebar"]):
                gr.Markdown("### キャラ名一覧")
                for index in range(MAX_SPEAKER_SLOTS):
                    speaker_select_button = gr.Button(
                        value=f"話者{index + 1}",
                        visible=False,
                        elem_classes=["speaker-action"],
                    )
                    speaker_select_buttons.append(speaker_select_button)
                    speaker_slot_id = gr.State(value=None)
                    speaker_slot_ids.append(speaker_slot_id)
                back_button = gr.Button("字幕結果に戻る", variant="secondary")

            with gr.Column(scale=2, elem_classes=["speaker-detail"]):
                speaker_detail_title = gr.Markdown("### 話者を選択してください")
                active_speaker_id = gr.State(value=None)
                speaker_name_input = gr.Textbox(label="キャラ名", placeholder="例: ルフィ")
                rename_button = gr.Button("名前を反映")
                gr.Markdown("セリフ一覧")
                speaker_utterance_box = gr.HTML(elem_classes=["speaker-list"])
                merge_target = gr.Dropdown(label="他の話者をまとめる", choices=[], value=None)
                merge_button = gr.Button("この話者に統合")
                swap_target = gr.Dropdown(label="話者を入れ替える", choices=[], value=None)
                swap_button = gr.Button("入れ替える")

    speaker_list_outputs = []
    for index in range(MAX_SPEAKER_SLOTS):
        speaker_list_outputs.extend([speaker_select_buttons[index], speaker_slot_ids[index]])

    detail_outputs = [
        speaker_detail_title,
        speaker_name_input,
        speaker_utterance_box,
        merge_target,
        swap_target,
        active_speaker_id,
    ]

    full_outputs = [
        app_state,
        subtitle_table,
        status_box,
        selected_speaker_state,
        *speaker_list_outputs,
        *detail_outputs,
    ]

    run_button.click(
        fn=load_mock_subtitles,
        inputs=[file_input, start_input, end_input, enhance_toggle],
        outputs=full_outputs,
        queue=False,
    )
    save_edits_button.click(
        fn=save_subtitle_edits,
        inputs=[subtitle_table, app_state, selected_speaker_state],
        outputs=full_outputs,
        queue=False,
    )
    rename_button.click(
        fn=handle_rename,
        inputs=[active_speaker_id, speaker_name_input, app_state, selected_speaker_state],
        outputs=full_outputs,
        queue=False,
    )
    merge_button.click(
        fn=handle_merge,
        inputs=[active_speaker_id, merge_target, app_state, selected_speaker_state],
        outputs=full_outputs,
        queue=False,
    )
    swap_button.click(
        fn=handle_swap,
        inputs=[active_speaker_id, swap_target, app_state, selected_speaker_state],
        outputs=full_outputs,
        queue=False,
    )

    for index in range(MAX_SPEAKER_SLOTS):
        speaker_select_buttons[index].click(
            fn=select_speaker,
            inputs=[speaker_slot_ids[index], app_state],
            outputs=full_outputs,
            queue=False,
        )

    open_speakers_button.click(fn=show_speaker_page, outputs=[main_page, speaker_page], queue=False)
    back_button.click(fn=show_main_page, outputs=[main_page, speaker_page], queue=False)


if __name__ == "__main__":
    demo.launch()
