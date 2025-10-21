"""Service helpers for loading and running Whisper ASR models."""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

import torch

try:  # pragma: no-cover - optional dependency for microphone detection
    import sounddevice as sd

    _SOUNDDEVICE_AVAILABLE = True
except Exception:  # pragma: no-cover - sounddevice is optional
    sd = None
    _SOUNDDEVICE_AVAILABLE = False

import whisper


LOGGER = logging.getLogger(__name__)


WHISPER_MODELS: List[str] = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
]

DEFAULT_MODEL_NAME = "small"


@lru_cache(maxsize=1)
def detect_compute_device() -> str:
    """Detect the best compute device available for Whisper."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def is_ffmpeg_available() -> bool:
    """Return True when ffmpeg binary is discoverable."""

    return shutil.which("ffmpeg") is not None


@dataclasses.dataclass
class MicrophoneInfo:
    """Information about a single microphone device."""

    index: int
    name: str
    max_input_channels: int
    default_samplerate: Optional[float]


def list_microphones() -> List[MicrophoneInfo]:
    """Return a list of available microphone input devices."""

    if not _SOUNDDEVICE_AVAILABLE:
        LOGGER.debug("sounddevice is not available; skipping microphone discovery")
        return []

    try:
        devices = sd.query_devices()  # type: ignore[assignment]
    except Exception as exc:  # pragma: no-cover - hardware dependent
        LOGGER.warning("Unable to query audio devices: %s", exc)
        return []

    microphones: List[MicrophoneInfo] = []
    for index, info in enumerate(devices):
        max_input = info.get("max_input_channels", 0)
        if max_input and max_input > 0:
            microphones.append(
                MicrophoneInfo(
                    index=index,
                    name=info.get("name", f"Device {index}"),
                    max_input_channels=max_input,
                    default_samplerate=info.get("default_samplerate"),
                )
            )

    return microphones


class WhisperService:
    """High level API for loading Whisper models and running inference."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        download_root: Optional[os.PathLike[str]] = None,
    ) -> None:
        self.device = device or detect_compute_device()
        self.download_root = Path(download_root) if download_root else None
        self.model_name = ""
        self._model: Optional[whisper.Whisper] = None
        self.set_model(model_name)

    @property
    def model(self) -> whisper.Whisper:
        if self._model is None:
            raise RuntimeError("Model has not been loaded")
        return self._model

    def set_model(self, model_name: str) -> None:
        if model_name not in WHISPER_MODELS:
            raise ValueError(f"Unsupported model '{model_name}'")

        if model_name == self.model_name and self._model is not None:
            return

        LOGGER.info("Loading Whisper model '%s' on %s", model_name, self.device)
        kwargs = {"device": self.device}
        if self.download_root is not None:
            kwargs["download_root"] = str(self.download_root)

        self._model = whisper.load_model(model_name, **kwargs)
        self.model_name = model_name

    def available_models(self) -> Iterable[str]:
        return WHISPER_MODELS

    def transcribe(
        self,
        audio_path: str,
        *,
        language: Optional[str] = None,
        temperature: float = 0.0,
        use_fp16: Optional[bool] = None,
    ) -> str:
        if use_fp16 is None:
            use_fp16 = self.device != "cpu"

        LOGGER.debug(
            "Running transcription with language=%s temperature=%.2f fp16=%s",
            language,
            temperature,
            use_fp16,
        )

        result = self.model.transcribe(
            audio_path,
            language=language,
            temperature=temperature,
            fp16=use_fp16,
        )
        return result.get("text", "").strip()

    def describe_environment(self) -> str:
        ffmpeg_status = "available" if is_ffmpeg_available() else "missing"
        return (
            f"Model: {self.model_name}\n"
            f"Device: {self.device}\n"
            f"ffmpeg: {ffmpeg_status}"
        )


__all__ = [
    "DEFAULT_MODEL_NAME",
    "MicrophoneInfo",
    "WhisperService",
    "detect_compute_device",
    "is_ffmpeg_available",
    "list_microphones",
]
