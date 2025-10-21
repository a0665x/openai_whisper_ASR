"""Command line interface for Whisper ASR."""

from __future__ import annotations

import argparse

from .service import DEFAULT_MODEL_NAME, WhisperService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Whisper ASR on an audio file")
    parser.add_argument("audio", help="Path to an audio file (wav/mp3/ogg/...)")
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large-v2"],
        default=DEFAULT_MODEL_NAME,
        help="Whisper model size",
    )
    parser.add_argument(
        "--language",
        help="Force a target language (e.g. 'en', 'zh'). Leave empty for auto detection.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--device",
        help="Override compute device (cpu/cuda/mps). Default auto detect.",
    )
    parser.add_argument(
        "--download-root",
        help="Custom directory to store Whisper model weights.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    service = WhisperService(
        model_name=args.model,
        device=args.device,
        download_root=args.download_root,
    )
    text = service.transcribe(
        args.audio,
        language=args.language,
        temperature=args.temperature,
    )
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
