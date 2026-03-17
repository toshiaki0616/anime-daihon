from __future__ import annotations

import copy
import re

import gradio as gr

from core.state_ops import (
    MAX_SPEAKER_SLOTS,
    apply_subtitle_edits,
    merge_speakers,
    rename_speaker,
    swap_speakers,
)
from models.state import AppState
from services.mock_pipeline import run_mock_pipeline
from ui.renderers import (
    build_speaker_slot_updates,
    build_subtitle_rows,
    format_status_box,
    make_empty_state,
)

TIME_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")


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


def render_all(state: AppState, status_message: str, status_kind: str = "info"):
    outputs: list = [
        state.to_dict(),
        build_subtitle_rows(state),
        format_status_box(status_message, status_kind),
    ]

    for slot in build_speaker_slot_updates(state):
        outputs.extend(
            [
                gr.update(visible=slot["visible"], label=slot["title"]),
                gr.update(value=slot["display_name"]),
                gr.update(value=slot["utterances_html"]),
                gr.update(choices=slot["merge_choices"], value=None),
                gr.update(choices=slot["swap_choices"], value=None),
                slot["speaker_id"],
            ]
        )
    return outputs


def load_mock_subtitles(file_path: str | None, start_time: str, end_time: str, enhance_audio: bool):
    if not file_path:
        return render_all(AppState(), "ファイルを選んでください", "error")

    start_time = start_time.strip()
    end_time = end_time.strip()
    error_message = validate_times(start_time, end_time)
    if error_message:
        return render_all(AppState(file_path=file_path), error_message, "error")

    next_state = run_mock_pipeline(
        file_path=file_path,
        start_time=start_time,
        end_time=end_time,
        enhance_audio=enhance_audio,
    )
    return render_all(next_state, "字幕を作成しました", "success")


def save_subtitle_edits(rows: list[list[str]], state_dict: dict):
    state = parse_state(state_dict)
    next_state = apply_subtitle_edits(copy.deepcopy(state), rows or [])
    return render_all(next_state, "字幕を更新しました", "success")


def handle_rename(slot_speaker_id: str | None, new_name: str, state_dict: dict):
    state = parse_state(state_dict)
    if not slot_speaker_id:
        return render_all(state, "話者情報がありません", "error")

    next_state = rename_speaker(copy.deepcopy(state), slot_speaker_id, new_name)
    return render_all(next_state, "話者名を反映しました", "success")


def handle_merge(slot_speaker_id: str | None, merge_source_id: str | None, state_dict: dict):
    state = parse_state(state_dict)
    if not slot_speaker_id or not merge_source_id:
        return render_all(state, "統合する話者を選んでください", "error")

    next_state = merge_speakers(copy.deepcopy(state), merge_source_id, slot_speaker_id)
    return render_all(next_state, "話者を統合しました", "success")


def handle_swap(slot_speaker_id: str | None, swap_target_id: str | None, state_dict: dict):
    state = parse_state(state_dict)
    if not slot_speaker_id or not swap_target_id:
        return render_all(state, "入れ替える話者を選んでください", "error")

    next_state = swap_speakers(copy.deepcopy(state), slot_speaker_id, swap_target_id)
    return render_all(next_state, "話者を入れ替えました", "success")


custom_css = """
.app-shell { max-width: 1120px; margin: 0 auto; }
.subtitle-table { min-height: 420px; }
.speaker-list { min-height: 120px; max-height: 180px; overflow-y: auto; border: 1px solid #dbe4f0; border-radius: 12px; padding: 12px; background: #ffffff; }
"""


with gr.Blocks(title="字幕作成", css=custom_css) as demo:
    app_state = gr.State(make_empty_state())

    speaker_cards = []
    speaker_name_inputs = []
    speaker_utterance_boxes = []
    speaker_merge_dropdowns = []
    speaker_swap_dropdowns = []
    speaker_id_states = []
    rename_buttons = []
    merge_buttons = []
    swap_buttons = []

    gr.Markdown("# 字幕作成")
    status_box = gr.HTML(format_status_box("ファイルを選んで、モック字幕を表示できます。"))

    with gr.Column(elem_classes=["app-shell"], visible=True) as main_page:
        with gr.Row():
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

        gr.Markdown("## 字幕結果")
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
        with gr.Row():
            save_edits_button = gr.Button("字幕編集を保存")
            open_speakers_button = gr.Button("話者一覧を開く")

    with gr.Column(elem_classes=["app-shell"], visible=False) as speaker_page:
        gr.Markdown("# 話者一覧")
        gr.Markdown("話者ごとのセリフを確認して、名前を登録する")

        for index in range(MAX_SPEAKER_SLOTS):
            speaker_id_state = gr.State(value=None)
            speaker_id_states.append(speaker_id_state)

            with gr.Accordion(f"話者{index + 1}", open=True, visible=False) as speaker_card:
                speaker_cards.append(speaker_card)

                name_input = gr.Textbox(label="キャラ名", placeholder="例: ルフィ")
                speaker_name_inputs.append(name_input)

                rename_button = gr.Button("名前を反映")
                rename_buttons.append(rename_button)

                gr.Markdown("セリフ一覧")
                utterance_html = gr.HTML(elem_classes=["speaker-list"])
                speaker_utterance_boxes.append(utterance_html)

                merge_target = gr.Dropdown(label="他の話者をまとめる", choices=[], value=None)
                speaker_merge_dropdowns.append(merge_target)
                merge_button = gr.Button("この話者に統合")
                merge_buttons.append(merge_button)

                swap_target = gr.Dropdown(label="話者を入れ替える", choices=[], value=None)
                speaker_swap_dropdowns.append(swap_target)
                swap_button = gr.Button("入れ替える")
                swap_buttons.append(swap_button)

        back_button = gr.Button("字幕結果に戻る")

    speaker_component_outputs = []
    for index in range(MAX_SPEAKER_SLOTS):
        speaker_component_outputs.extend(
            [
                speaker_cards[index],
                speaker_name_inputs[index],
                speaker_utterance_boxes[index],
                speaker_merge_dropdowns[index],
                speaker_swap_dropdowns[index],
                speaker_id_states[index],
            ]
        )

    full_outputs = [app_state, subtitle_table, status_box, *speaker_component_outputs]

    run_button.click(
        fn=load_mock_subtitles,
        inputs=[file_input, start_input, end_input, enhance_toggle],
        outputs=full_outputs,
        queue=False,
    )
    save_edits_button.click(
        fn=save_subtitle_edits,
        inputs=[subtitle_table, app_state],
        outputs=full_outputs,
        queue=False,
    )

    for index in range(MAX_SPEAKER_SLOTS):
        rename_buttons[index].click(
            fn=handle_rename,
            inputs=[speaker_id_states[index], speaker_name_inputs[index], app_state],
            outputs=full_outputs,
            queue=False,
        )
        merge_buttons[index].click(
            fn=handle_merge,
            inputs=[speaker_id_states[index], speaker_merge_dropdowns[index], app_state],
            outputs=full_outputs,
            queue=False,
        )
        swap_buttons[index].click(
            fn=handle_swap,
            inputs=[speaker_id_states[index], speaker_swap_dropdowns[index], app_state],
            outputs=full_outputs,
            queue=False,
        )

    open_speakers_button.click(fn=show_speaker_page, outputs=[main_page, speaker_page], queue=False)
    back_button.click(fn=show_main_page, outputs=[main_page, speaker_page], queue=False)


if __name__ == "__main__":
    demo.launch()
