"""
Caption generation module using Whisper.
Generates accurate SRT captions from audio with word-level timing.
"""
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import subprocess

from config.settings import AUDIO_DIR, CAPTIONS_DIR, OPENAI_API_KEY


@dataclass
class CaptionSegment:
    """A single caption segment with timing."""
    index: int
    start_time: float
    end_time: float
    text: str


class CaptionGenerator:
    """Generates SRT captions from audio using Whisper."""

    def __init__(self, use_local: bool = True):
        """
        Initialize caption generator.

        Args:
            use_local: Try local Whisper first, fallback to API
        """
        self.use_local = use_local
        self._local_model = None

    def generate(self, project_id: str) -> Optional[Path]:
        """
        Generate SRT captions for a project.

        Args:
            project_id: Project identifier

        Returns:
            Path to SRT file or None if failed
        """
        audio_file = AUDIO_DIR / f"{project_id}.mp3"
        srt_file = CAPTIONS_DIR / f"{project_id}.srt"

        if not audio_file.exists():
            return None

        # Try local Whisper first
        if self.use_local:
            result = self._transcribe_local(audio_file, srt_file)
            if result:
                return result

        # Fallback to API
        return self._transcribe_api(audio_file, srt_file)

    def _transcribe_local(self, audio: Path, output: Path) -> Optional[Path]:
        """Transcribe using local Whisper model."""
        try:
            import whisper

            if self._local_model is None:
                self._local_model = whisper.load_model("base")

            result = self._local_model.transcribe(
                str(audio),
                word_timestamps=True,
                verbose=False
            )

            # Convert to SRT
            segments = self._whisper_to_segments(result)
            self._write_srt(segments, output)

            return output

        except ImportError:
            return None
        except Exception as e:
            print(f"Local transcription error: {e}")
            return None

    def _transcribe_api(self, audio: Path, output: Path) -> Optional[Path]:
        """Transcribe using OpenAI Whisper API."""
        if not OPENAI_API_KEY:
            return None

        try:
            import openai

            client = openai.OpenAI(api_key=OPENAI_API_KEY)

            with open(audio, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )

            # Convert API response to segments
            segments = []
            for i, seg in enumerate(response.segments, 1):
                segments.append(CaptionSegment(
                    index=i,
                    start_time=seg["start"],
                    end_time=seg["end"],
                    text=seg["text"].strip()
                ))

            self._write_srt(segments, output)
            return output

        except Exception as e:
            print(f"API transcription error: {e}")
            return None

    def _whisper_to_segments(self, result: dict) -> list[CaptionSegment]:
        """Convert Whisper result to caption segments."""
        segments = []

        for i, seg in enumerate(result.get("segments", []), 1):
            # Clean up text
            text = seg["text"].strip()
            if not text:
                continue

            segments.append(CaptionSegment(
                index=i,
                start_time=seg["start"],
                end_time=seg["end"],
                text=text
            ))

        return segments

    def _write_srt(self, segments: list[CaptionSegment], output: Path):
        """Write segments to SRT file."""
        lines = []

        for seg in segments:
            start = self._format_srt_time(seg.start_time)
            end = self._format_srt_time(seg.end_time)

            lines.append(str(seg.index))
            lines.append(f"{start} --> {end}")
            lines.append(seg.text)
            lines.append("")

        output.write_text("\n".join(lines), encoding="utf-8")

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def generate_vtt(self, project_id: str) -> Optional[Path]:
        """
        Generate WebVTT captions (alternative format).

        Args:
            project_id: Project identifier

        Returns:
            Path to VTT file or None
        """
        srt_file = CAPTIONS_DIR / f"{project_id}.srt"
        vtt_file = CAPTIONS_DIR / f"{project_id}.vtt"

        # Generate SRT first if needed
        if not srt_file.exists():
            result = self.generate(project_id)
            if not result:
                return None

        # Convert SRT to VTT
        try:
            srt_content = srt_file.read_text(encoding="utf-8")

            # VTT header
            vtt_lines = ["WEBVTT", ""]

            for line in srt_content.split("\n"):
                # Convert timestamp separator
                if " --> " in line:
                    line = line.replace(",", ".")
                vtt_lines.append(line)

            vtt_file.write_text("\n".join(vtt_lines), encoding="utf-8")
            return vtt_file

        except Exception as e:
            print(f"VTT conversion error: {e}")
            return None


def generate_captions(project_id: str, format: str = "srt") -> Optional[str]:
    """
    Convenience function to generate captions.

    Args:
        project_id: Project identifier
        format: Output format ("srt" or "vtt")

    Returns:
        Path to caption file as string, or None
    """
    generator = CaptionGenerator()

    if format == "vtt":
        result = generator.generate_vtt(project_id)
    else:
        result = generator.generate(project_id)

    return str(result) if result else None
