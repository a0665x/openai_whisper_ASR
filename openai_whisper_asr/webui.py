"""Gradio based Web UI for Whisper ASR."""

from __future__ import annotations

import logging
from typing import Optional

import gradio as gr

from .service import (
    DEFAULT_MODEL_NAME,
    WhisperService,
    is_ffmpeg_available,
    list_microphones,
)


LOGGER = logging.getLogger(__name__)


def _format_microphone_list() -> str:
    microphones = list_microphones()
    if not microphones:
        return "未偵測到麥克風裝置，請確認系統是否允許容器存取音訊硬體。"

    lines = ["偵測到的麥克風裝置："]
    for mic in microphones:
        line = f"#{mic.index} - {mic.name} (channels: {mic.max_input_channels})"
        if mic.default_samplerate:
            line += f" @ {mic.default_samplerate:.0f}Hz"
        lines.append(line)
    return "\n".join(lines)


def build_interface(service: Optional[WhisperService] = None) -> gr.Blocks:
    service = service or WhisperService()

    def switch_model(model_name: str) -> str:
        try:
            service.set_model(model_name)
        except Exception as exc:  # pragma: no cover - UI feedback
            LOGGER.exception("Unable to load model")
            return f"載入模型時發生錯誤：{exc}"
        return service.describe_environment()

    def transcribe(audio_file: str, language: str, temperature: float) -> str:
        if not audio_file:
            return "請先錄製或上傳音訊。"
        result = service.transcribe(
            audio_file,
            language=None if language == "auto" else language,
            temperature=temperature,
        )
        return result or "（沒有辨識到語音內容）"

    model_choices = list(service.available_models())
    environment_text = service.describe_environment()
    microphone_text = _format_microphone_list()

    with gr.Blocks(title="OpenAI Whisper ASR WebUI") as demo:
        gr.Markdown(
            """
            # OpenAI Whisper 即時語音辨識
            使用瀏覽器內建麥克風錄製語音，並透過 OpenAI Whisper 模型進行辨識。
            """
        )

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=model_choices,
                value=service.model_name or DEFAULT_MODEL_NAME,
                label="模型大小",
                info="模型會在首次使用時自動下載。建議在資源有限時使用 small/base。",
            )
            temperature_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.1,
                label="Temperature",
                info="較低的數值可獲得更穩定的結果。",
            )
            language_dropdown = gr.Dropdown(
                choices=["auto", "zh", "en"],
                value="auto",
                label="語言",
                info="若已知語言，可指定以提升速度與準確率。",
            )

        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="錄製或上傳音訊",
        )
        transcript_output = gr.Textbox(
            label="辨識結果",
            lines=5,
            interactive=False,
        )

        with gr.Accordion("系統資訊", open=False):
            environment_box = gr.Markdown(environment_text)
            if not is_ffmpeg_available():
                gr.Markdown("⚠️ 未偵測到 ffmpeg，請先安裝以支援更多音訊格式。")
            gr.Markdown(microphone_text)

        audio_input.change(
            fn=transcribe,
            inputs=[audio_input, language_dropdown, temperature_slider],
            outputs=transcript_output,
        )

        model_dropdown.change(
            fn=switch_model,
            inputs=model_dropdown,
            outputs=environment_box,
        )

    return demo


def main() -> None:  # pragma: no cover - entry point
    interface = build_interface()
    interface.launch()


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
