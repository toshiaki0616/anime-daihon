from __future__ import annotations

from core.state_ops import build_mock_state


def run_mock_pipeline(
    file_path: str,
    wav_path: str,
    start_time: str,
    end_time: str,
    enhance_audio: bool,
):
    state = build_mock_state(
        file_path=file_path,
        range_start=start_time,
        range_end=end_time,
        enhance_audio=enhance_audio,
    )
    state.wav_path = wav_path
    return state
