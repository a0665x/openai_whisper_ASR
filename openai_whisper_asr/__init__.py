"""Utility package for running Whisper-based ASR demos."""

from .service import WhisperService, detect_compute_device

__all__ = ["WhisperService", "detect_compute_device"]
