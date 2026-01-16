"""
Caption generation module using Whisper.
Generates accurate SRT captions with 2-3 words per frame for TikTok/YouTube Shorts style.
"""
import json
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from config.settings import AUDIO_DIR, CAPTIONS_DIR, OPENAI_API_KEY


@dataclass
class CaptionSegment:
    """A single caption segment with timing."""
    index: int
    start_time: float
    end_time: float
    text: str


class CaptionGenerator:
    """Generates SRT captions from audio using Whisper with 2-3 word chunks."""

    def __init__(self, words_per_chunk: int = 3):
        """
        Initialize caption generator.

        Args:
            words_per_chunk: Max words per caption (default 3 for punchy style)
        """
        self.words_per_chunk = words_per_chunk
        self._local_model = None

    def generate(self, project_id: str) -> Optional[Path]:
        """
        Generate SRT captions for a project with 2-3 words per caption.

        Args:
            project_id: Project identifier

        Returns:
            Path to SRT file or None if failed
        """
        audio_file = AUDIO_DIR / f"{project_id}.mp3"
        srt_file = CAPTIONS_DIR / f"{project_id}.srt"

        if not audio_file.exists():
            return None

        # Try local Whisper first (has word-level timestamps)
        result = self._transcribe_local(audio_file)
        if result:
            segments = self._chunk_words(result)
            self._write_srt(segments, srt_file)
            return srt_file

        # Fallback to API
        return self._transcribe_api(audio_file, srt_file)

    def _transcribe_local(self, audio: Path) -> Optional[dict]:
        """Transcribe using local Whisper model with word timestamps."""
        try:
            import whisper

            if self._local_model is None:
                self._local_model = whisper.load_model("base")

            result = self._local_model.transcribe(
                str(audio),
                word_timestamps=True,
                verbose=False
            )
            return result

        except ImportError:
            return None
        except Exception as e:
            print(f"Local transcription error: {e}")
            return None

    def _chunk_words(self, whisper_result: dict) -> list[CaptionSegment]:
        """
        Split transcription into 2-3 word chunks for punchy captions.
        """
        segments = []
        index = 1

        for segment in whisper_result.get("segments", []):
            words = segment.get("words", [])

            if not words:
                # Fallback: split segment text manually
                text = segment.get("text", "").strip()
                word_list = text.split()
                seg_start = segment.get("start", 0)
                seg_end = segment.get("end", seg_start + 1)
                seg_duration = seg_end - seg_start

                # Estimate timing per word
                if word_list:
                    time_per_word = seg_duration / len(word_list)

                    for i in range(0, len(word_list), self.words_per_chunk):
                        chunk_words = word_list[i:i + self.words_per_chunk]
                        chunk_start = seg_start + (i * time_per_word)
                        chunk_end = min(chunk_start + (len(chunk_words) * time_per_word), seg_end)

                        segments.append(CaptionSegment(
                            index=index,
                            start_time=chunk_start,
                            end_time=chunk_end,
                            text=" ".join(chunk_words).upper()
                        ))
                        index += 1
            else:
                # Use word-level timestamps from Whisper
                for i in range(0, len(words), self.words_per_chunk):
                    chunk = words[i:i + self.words_per_chunk]
                    if chunk:
                        chunk_text = " ".join(w.get("word", "").strip() for w in chunk)
                        # Clean up text
                        chunk_text = re.sub(r'[^\w\s\'-]', '', chunk_text).upper()

                        if chunk_text.strip():
                            segments.append(CaptionSegment(
                                index=index,
                                start_time=chunk[0].get("start", 0),
                                end_time=chunk[-1].get("end", chunk[0].get("start", 0) + 0.5),
                                text=chunk_text
                            ))
                            index += 1

        return segments

    def _transcribe_api(self, audio: Path, output: Path) -> Optional[Path]:
        """Transcribe using OpenAI Whisper API with word-level timestamps."""
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
                    timestamp_granularities=["word"]
                )

            # Chunk the words into 2-3 word segments
            segments = []
            words = response.words if hasattr(response, 'words') else []
            index = 1

            for i in range(0, len(words), self.words_per_chunk):
                chunk = words[i:i + self.words_per_chunk]
                if chunk:
                    chunk_text = " ".join(w.word.strip() for w in chunk)
                    chunk_text = re.sub(r'[^\w\s\'-]', '', chunk_text).upper()

                    if chunk_text.strip():
                        segments.append(CaptionSegment(
                            index=index,
                            start_time=chunk[0].start,
                            end_time=chunk[-1].end,
                            text=chunk_text
                        ))
                        index += 1

            self._write_srt(segments, output)
            return output

        except Exception as e:
            print(f"API transcription error: {e}")
            return None

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
        """Generate WebVTT captions (alternative format)."""
        srt_file = CAPTIONS_DIR / f"{project_id}.srt"
        vtt_file = CAPTIONS_DIR / f"{project_id}.vtt"

        if not srt_file.exists():
            result = self.generate(project_id)
            if not result:
                return None

        try:
            srt_content = srt_file.read_text(encoding="utf-8")
            vtt_lines = ["WEBVTT", ""]

            for line in srt_content.split("\n"):
                if " --> " in line:
                    line = line.replace(",", ".")
                vtt_lines.append(line)

            vtt_file.write_text("\n".join(vtt_lines), encoding="utf-8")
            return vtt_file

        except Exception as e:
            print(f"VTT conversion error: {e}")
            return None


def generate_captions(project_id: str, format: str = "srt",
                      words_per_chunk: int = 3) -> Optional[str]:
    """
    Convenience function to generate captions.

    Args:
        project_id: Project identifier
        format: Output format ("srt" or "vtt")
        words_per_chunk: Max words per caption frame (default 3)

    Returns:
        Path to caption file as string, or None
    """
    generator = CaptionGenerator(words_per_chunk=words_per_chunk)

    if format == "vtt":
        result = generator.generate_vtt(project_id)
    else:
        result = generator.generate(project_id)

    return str(result) if result else None
