"""
Voice note transcription for the WhatsApp bridge.

Reuses MKAngel's existing :class:`app.voice.VoiceEngine` if available;
otherwise raises a clear :class:`RuntimeError` with install instructions.

WhatsApp voice notes are opus in an ogg container. ``openai-whisper``
handles that natively via ffmpeg, and ``VoiceEngine.transcribe`` is a
thin wrapper around Whisper so it accepts any path Whisper can read.
No new dependencies are introduced by this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_engine: Any = None
_engine_tried: bool = False


def _get_engine() -> Any | None:
    """Instantiate app.voice.VoiceEngine once, or return None if unavailable."""
    global _engine, _engine_tried
    if _engine_tried:
        return _engine
    _engine_tried = True
    try:
        from app.voice import VoiceEngine  # type: ignore
    except Exception:
        _engine = None
        return None
    try:
        _engine = VoiceEngine()
    except Exception:
        _engine = None
    return _engine


def _has_local_whisper() -> bool:
    """Return True if ``openai-whisper`` is importable without loading a model."""
    try:
        import whisper  # noqa: F401
    except Exception:
        return False
    return True


def transcribe(audio_path: Path) -> str:
    """Transcribe a voice note (ogg/opus from WhatsApp) to text.

    Tries, in order:
      1. ``app.voice.VoiceEngine().transcribe(path)`` (which itself wraps
         local Whisper when available).
      2. A direct ``openai-whisper`` call, but only if the library is
         already importable on the host.

    Raises:
        RuntimeError: if no transcription backend is available. The
            caller is responsible for deleting ``audio_path`` afterwards.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise RuntimeError(f"audio file not found: {audio_path}")

    # Attempt 1: MKAngel's own VoiceEngine. It already knows how to drive
    # Whisper and returns a TranscriptionResult with a .text attribute.
    engine = _get_engine()
    if engine is not None and hasattr(engine, "transcribe"):
        try:
            result = engine.transcribe(audio_path)
        except RuntimeError:
            # VoiceEngine raises RuntimeError when no Whisper backend is
            # installed -- fall through to the direct whisper attempt.
            result = None
        except Exception as exc:
            raise RuntimeError(f"VoiceEngine.transcribe failed: {exc}") from exc
        if result is not None:
            text = getattr(result, "text", None)
            if text is None and isinstance(result, str):
                text = result
            if text:
                return text.strip()

    # Attempt 2: direct openai-whisper, only if already installed.
    if _has_local_whisper():
        import whisper  # type: ignore

        try:
            model = whisper.load_model("base")
            out = model.transcribe(str(audio_path))
        except Exception as exc:
            raise RuntimeError(f"whisper transcription failed: {exc}") from exc
        text = (out.get("text") or "").strip()
        if text:
            return text

    raise RuntimeError(
        "no transcription backend; install openai-whisper or wire up "
        "app.voice.VoiceEngine.transcribe"
    )
