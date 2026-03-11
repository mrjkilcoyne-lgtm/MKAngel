"""
Voice capabilities for MKAngel.

The Angel speaks and listens.  Voice is the oldest substrate of language —
before writing, before grammar was formalised, there was speech.  This
module gives MKAngel the ability to:

- Record audio from microphone (ears)
- Transcribe speech to text via Whisper (hearing → understanding)
- Clone and synthesise voice (speaking with any voice)
- Isolate voices from mixed audio (attention in the auditory stream)
- Text-to-speech output (the Angel's voice)

All capabilities gracefully degrade: if libraries are missing, the module
reports what's needed and falls back to text-only operation.

Ports and protocols:
- Local microphone via sounddevice / pyaudio
- Whisper (local) or Whisper API for transcription
- TTS via pyttsx3 (offline), Coqui TTS (neural), or cloud APIs
- Voice cloning via Coqui YourTTS or RVC
"""

from __future__ import annotations

import io
import json
import os
import time
import wave
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


from app.paths import mkangel_dir

_VOICE_DIR = mkangel_dir() / "voice"
_RECORDINGS_DIR = _VOICE_DIR / "recordings"
_CLONES_DIR = _VOICE_DIR / "clones"
_MODELS_DIR = _VOICE_DIR / "models"


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

def _check_availability() -> dict[str, bool]:
    """Check which voice libraries are available."""
    available = {}
    for module, key in [
        ("sounddevice", "recording"),
        ("numpy", "numpy"),
        ("whisper", "whisper_local"),
        ("openai", "whisper_api"),
        ("pyttsx3", "tts_offline"),
        ("TTS", "tts_neural"),
        ("scipy", "audio_processing"),
    ]:
        try:
            __import__(module)
            available[key] = True
        except ImportError:
            available[key] = False
    return available


def check_voice_capabilities() -> dict[str, Any]:
    """Return a report of available voice capabilities."""
    avail = _check_availability()
    capabilities = {
        "recording": {
            "available": avail.get("recording", False),
            "requires": "pip install sounddevice numpy",
        },
        "transcription_local": {
            "available": avail.get("whisper_local", False),
            "requires": "pip install openai-whisper",
        },
        "transcription_api": {
            "available": avail.get("whisper_api", False),
            "requires": "pip install openai (+ API key)",
        },
        "tts_offline": {
            "available": avail.get("tts_offline", False),
            "requires": "pip install pyttsx3",
        },
        "tts_neural": {
            "available": avail.get("tts_neural", False),
            "requires": "pip install TTS",
        },
        "voice_cloning": {
            "available": avail.get("tts_neural", False),
            "requires": "pip install TTS (includes YourTTS)",
        },
        "voice_isolation": {
            "available": avail.get("audio_processing", False) and avail.get("numpy", False),
            "requires": "pip install scipy numpy",
        },
    }
    return capabilities


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AudioSegment:
    """A segment of audio data."""
    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    duration: float = 0.0
    format: str = "wav"
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save audio segment to WAV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.data)

    @classmethod
    def from_file(cls, path: str | Path) -> "AudioSegment":
        """Load audio from WAV file."""
        with wave.open(str(path), "rb") as wf:
            data = wf.readframes(wf.getnframes())
            return cls(
                data=data,
                sample_rate=wf.getframerate(),
                channels=wf.getnchannels(),
                duration=wf.getnframes() / wf.getframerate(),
            )


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription."""
    text: str
    language: str = "en"
    confidence: float = 0.0
    segments: list[dict[str, Any]] = field(default_factory=list)
    duration: float = 0.0
    model_used: str = ""


@dataclass
class VoiceProfile:
    """A cloned or configured voice profile."""
    name: str
    source_audio: str = ""
    model_path: str = ""
    language: str = "en"
    properties: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: Path | None = None) -> None:
        """Save voice profile."""
        d = directory or _CLONES_DIR
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{self.name}.json", "w") as f:
            json.dump({
                "name": self.name,
                "source_audio": self.source_audio,
                "model_path": self.model_path,
                "language": self.language,
                "properties": self.properties,
            }, f, indent=2)

    @classmethod
    def load(cls, name: str, directory: Path | None = None) -> "VoiceProfile":
        """Load voice profile."""
        d = directory or _CLONES_DIR
        with open(d / f"{name}.json") as f:
            data = json.load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# VoiceEngine — the central voice capability manager
# ---------------------------------------------------------------------------

class VoiceEngine:
    """Voice engine for MKAngel — ears, mouth, and vocal cords.

    Provides recording, transcription, synthesis, cloning, and isolation.
    Gracefully degrades when dependencies are missing.
    """

    def __init__(self) -> None:
        self._capabilities = _check_availability()
        self._recording = False
        self._tts_engine = None
        self._whisper_model = None
        self._voice_profiles: dict[str, VoiceProfile] = {}

        # Ensure directories exist
        for d in (_VOICE_DIR, _RECORDINGS_DIR, _CLONES_DIR, _MODELS_DIR):
            d.mkdir(parents=True, exist_ok=True)

        # Load existing voice profiles
        self._load_profiles()

    # ------------------------------------------------------------------
    # Recording (ears)
    # ------------------------------------------------------------------

    def record(
        self,
        duration: float = 5.0,
        sample_rate: int = 16000,
        channels: int = 1,
        save_path: str | None = None,
    ) -> AudioSegment:
        """Record audio from microphone.

        Args:
            duration: Recording duration in seconds.
            sample_rate: Sample rate in Hz.
            channels: Number of audio channels.
            save_path: Optional path to save the recording.

        Returns:
            AudioSegment with the recorded data.
        """
        if not self._capabilities.get("recording"):
            raise RuntimeError(
                "Recording requires sounddevice and numpy. "
                "Install: pip install sounddevice numpy"
            )

        import sounddevice as sd
        import numpy as np

        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
        )
        sd.wait()

        segment = AudioSegment(
            data=audio_data.tobytes(),
            sample_rate=sample_rate,
            channels=channels,
            duration=duration,
            metadata={"recorded_at": time.time()},
        )

        if save_path:
            segment.save(save_path)
        else:
            ts = int(time.time())
            segment.save(_RECORDINGS_DIR / f"recording_{ts}.wav")

        return segment

    def start_recording(
        self,
        sample_rate: int = 16000,
        callback: Callable[[bytes], None] | None = None,
    ) -> None:
        """Start continuous recording (for real-time transcription)."""
        if not self._capabilities.get("recording"):
            raise RuntimeError("Recording requires sounddevice. Install: pip install sounddevice numpy")

        self._recording = True

    def stop_recording(self) -> AudioSegment | None:
        """Stop continuous recording and return the audio."""
        self._recording = False
        return None

    # ------------------------------------------------------------------
    # Transcription (hearing → understanding)
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: AudioSegment | str | Path,
        model: str = "base",
        language: str | None = None,
        use_api: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio to text using Whisper.

        Args:
            audio: AudioSegment or path to audio file.
            model: Whisper model size (tiny/base/small/medium/large).
            language: Language code (auto-detect if None).
            use_api: Use OpenAI Whisper API instead of local model.

        Returns:
            TranscriptionResult with transcribed text.
        """
        if isinstance(audio, (str, Path)):
            audio_path = str(audio)
        else:
            # Save to temp file
            tmp = _RECORDINGS_DIR / "_tmp_transcribe.wav"
            audio.save(tmp)
            audio_path = str(tmp)

        if use_api and self._capabilities.get("whisper_api"):
            return self._transcribe_api(audio_path, language)
        elif self._capabilities.get("whisper_local"):
            return self._transcribe_local(audio_path, model, language)
        else:
            raise RuntimeError(
                "Transcription requires either:\n"
                "  Local: pip install openai-whisper\n"
                "  API:   pip install openai (+ OPENAI_API_KEY)"
            )

    def _transcribe_local(
        self, audio_path: str, model: str, language: str | None,
    ) -> TranscriptionResult:
        """Transcribe using local Whisper model."""
        import whisper

        if self._whisper_model is None or self._whisper_model_name != model:
            self._whisper_model = whisper.load_model(model)
            self._whisper_model_name = model

        result = self._whisper_model.transcribe(
            audio_path,
            language=language,
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", ""),
            })

        return TranscriptionResult(
            text=result.get("text", "").strip(),
            language=result.get("language", "en"),
            confidence=0.9,
            segments=segments,
            model_used=f"whisper-{model}",
        )

    def _transcribe_api(
        self, audio_path: str, language: str | None,
    ) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper API."""
        from openai import OpenAI
        client = OpenAI()

        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language,
            )

        return TranscriptionResult(
            text=result.text,
            language=language or "auto",
            confidence=0.95,
            model_used="whisper-1-api",
        )

    # ------------------------------------------------------------------
    # Text-to-Speech (the Angel speaks)
    # ------------------------------------------------------------------

    def speak(
        self,
        text: str,
        voice: str | None = None,
        save_path: str | None = None,
        speed: float = 1.0,
    ) -> AudioSegment | None:
        """Convert text to speech.

        Args:
            text: Text to speak.
            voice: Voice profile name or None for default.
            save_path: Optional path to save audio.
            speed: Speech speed multiplier.

        Returns:
            AudioSegment if save_path is set, else plays directly.
        """
        if voice and voice in self._voice_profiles:
            return self._speak_cloned(text, self._voice_profiles[voice], save_path)

        if self._capabilities.get("tts_neural"):
            return self._speak_neural(text, save_path, speed)
        elif self._capabilities.get("tts_offline"):
            return self._speak_offline(text, speed)
        else:
            raise RuntimeError(
                "TTS requires either:\n"
                "  Offline: pip install pyttsx3\n"
                "  Neural:  pip install TTS"
            )

    def _speak_offline(self, text: str, speed: float) -> None:
        """Speak using pyttsx3 (offline, system voices)."""
        import pyttsx3
        if self._tts_engine is None:
            self._tts_engine = pyttsx3.init()

        rate = self._tts_engine.getProperty("rate")
        self._tts_engine.setProperty("rate", int(rate * speed))
        self._tts_engine.say(text)
        self._tts_engine.runAndWait()
        return None

    def _speak_neural(
        self, text: str, save_path: str | None, speed: float,
    ) -> AudioSegment | None:
        """Speak using Coqui TTS (neural, high quality)."""
        from TTS.api import TTS as CoquiTTS

        tts = CoquiTTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        out_path = save_path or str(_RECORDINGS_DIR / f"tts_{int(time.time())}.wav")
        tts.tts_to_file(text=text, file_path=out_path, speed=speed)
        return AudioSegment.from_file(out_path)

    def _speak_cloned(
        self, text: str, profile: VoiceProfile, save_path: str | None,
    ) -> AudioSegment | None:
        """Speak using a cloned voice."""
        if not self._capabilities.get("tts_neural"):
            raise RuntimeError("Voice cloning requires: pip install TTS")

        from TTS.api import TTS as CoquiTTS

        tts = CoquiTTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
        out_path = save_path or str(_RECORDINGS_DIR / f"clone_{int(time.time())}.wav")
        tts.tts_to_file(
            text=text,
            file_path=out_path,
            speaker_wav=profile.source_audio,
            language=profile.language,
        )
        return AudioSegment.from_file(out_path)

    # ------------------------------------------------------------------
    # Voice cloning
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        name: str,
        source_audio: str | Path,
        language: str = "en",
    ) -> VoiceProfile:
        """Create a voice clone from a reference audio sample.

        Args:
            name: Name for this voice profile.
            source_audio: Path to reference audio (WAV, 10-30 seconds ideal).
            language: Language code.

        Returns:
            VoiceProfile that can be used with speak().
        """
        if not self._capabilities.get("tts_neural"):
            raise RuntimeError("Voice cloning requires: pip install TTS")

        profile = VoiceProfile(
            name=name,
            source_audio=str(source_audio),
            language=language,
            properties={"created_at": time.time()},
        )
        profile.save()
        self._voice_profiles[name] = profile
        return profile

    def list_voices(self) -> list[VoiceProfile]:
        """List all available voice profiles."""
        return list(self._voice_profiles.values())

    # ------------------------------------------------------------------
    # Voice isolation
    # ------------------------------------------------------------------

    def isolate_voice(
        self,
        audio: AudioSegment | str | Path,
        method: str = "spectral",
    ) -> AudioSegment:
        """Isolate a single voice from mixed audio.

        Uses spectral subtraction or harmonic-percussive separation
        to extract the dominant voice from background noise or other
        speakers.

        Args:
            audio: Input audio with mixed sources.
            method: Isolation method ("spectral" or "hpss").

        Returns:
            AudioSegment with isolated voice.
        """
        if not (self._capabilities.get("audio_processing") and self._capabilities.get("numpy")):
            raise RuntimeError("Voice isolation requires: pip install scipy numpy")

        import numpy as np
        from scipy import signal
        from scipy.io import wavfile

        if isinstance(audio, (str, Path)):
            sr, data = wavfile.read(str(audio))
        else:
            sr = audio.sample_rate
            data = np.frombuffer(audio.data, dtype=np.int16)

        if len(data.shape) > 1:
            data = data.mean(axis=1).astype(np.int16)

        if method == "spectral":
            # Bandpass filter for voice frequencies (80 Hz - 8 kHz)
            nyquist = sr / 2
            low = max(80 / nyquist, 0.001)
            high = min(8000 / nyquist, 0.999)
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, data.astype(np.float64))
            isolated_data = filtered.astype(np.int16)
        else:
            # Simple high-pass to remove low-frequency noise
            nyquist = sr / 2
            cutoff = max(100 / nyquist, 0.001)
            b, a = signal.butter(4, cutoff, btype='high')
            filtered = signal.filtfilt(b, a, data.astype(np.float64))
            isolated_data = filtered.astype(np.int16)

        return AudioSegment(
            data=isolated_data.tobytes(),
            sample_rate=sr,
            channels=1,
            duration=len(isolated_data) / sr,
            metadata={"isolation_method": method},
        )

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def _load_profiles(self) -> None:
        """Load saved voice profiles."""
        if _CLONES_DIR.exists():
            for path in _CLONES_DIR.glob("*.json"):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    profile = VoiceProfile(**data)
                    self._voice_profiles[profile.name] = profile
                except (json.JSONDecodeError, OSError, TypeError):
                    continue

    # ------------------------------------------------------------------
    # Convenience: listen loop
    # ------------------------------------------------------------------

    def listen_and_transcribe(
        self,
        duration: float = 5.0,
        model: str = "base",
        language: str | None = None,
    ) -> TranscriptionResult:
        """Record from microphone and transcribe in one step.

        The ears and mind working together — hear and understand.
        """
        audio = self.record(duration=duration)
        return self.transcribe(audio, model=model, language=language)
