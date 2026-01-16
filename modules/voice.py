"""
Voice generation module using OpenAI TTS.
Converts script segments to audio with the 'onyx' voice.
"""
import json
import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import openai

from config.settings import (
    OPENAI_API_KEY, TTS_MODEL, TTS_VOICE,
    AUDIO_DIR, RECORDINGS_DIR
)


@dataclass
class AudioSegment:
    """Generated audio segment metadata."""
    segment_index: int
    start_time: float
    end_time: float
    audio_file: str
    duration: float
    text: str


class VoiceGenerator:
    """Generates voice audio from scripts using OpenAI TTS."""

    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def generate_from_script(self, script_file: str) -> dict:
        """
        Generate voice audio from a script file.

        Args:
            script_file: Path to script JSON

        Returns:
            dict with audio file path and segment info
        """
        # Load script
        with open(script_file) as f:
            script_data = json.load(f)

        project_id = script_data["project_id"]
        segments = script_data["segments"]
        hook = script_data.get("hook", "")
        outro = script_data.get("outro", "")

        # Combine all text for single audio generation
        # This ensures consistent pacing and tone
        full_text = self._build_full_narration(hook, segments, outro)

        # Generate audio
        audio_file = AUDIO_DIR / f"{project_id}.mp3"
        self._generate_audio(full_text, audio_file)

        # Calculate expected duration (rough estimate: 150 words/minute)
        word_count = len(full_text.split())
        estimated_duration = (word_count / 150) * 60

        # Save audio metadata
        audio_meta = {
            "project_id": project_id,
            "audio_file": str(audio_file),
            "word_count": word_count,
            "estimated_duration": estimated_duration,
            "voice": TTS_VOICE,
            "model": TTS_MODEL,
            "full_text": full_text
        }

        meta_file = AUDIO_DIR / f"{project_id}_audio_meta.json"
        with open(meta_file, "w") as f:
            json.dump(audio_meta, f, indent=2)

        return audio_meta

    def _build_full_narration(self, hook: str, segments: list[dict],
                               outro: str) -> str:
        """Build complete narration text with natural pauses."""
        parts = []

        # Hook
        if hook:
            parts.append(hook)
            parts.append("...")  # Natural pause

        # Main segments
        for seg in segments:
            text = seg["text"]
            emphasis = seg.get("emphasis", "normal")

            # Add emphasis markers for TTS interpretation
            if emphasis == "excited":
                text = text.rstrip(".!") + "!"
            elif emphasis == "thoughtful":
                text = "..." + text

            parts.append(text)

        # Outro
        if outro:
            parts.append("...")  # Transition pause
            parts.append(outro)

        return " ".join(parts)

    def _generate_audio(self, text: str, output_path: Path):
        """Generate audio file using OpenAI TTS."""
        response = self.client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            response_format="mp3"
        )

        # Stream to file
        response.stream_to_file(str(output_path))

    def generate_segment_audio(self, script_file: str) -> list[AudioSegment]:
        """
        Generate separate audio files for each segment.
        Useful for precise timing control.

        Args:
            script_file: Path to script JSON

        Returns:
            List of AudioSegment with individual file paths
        """
        with open(script_file) as f:
            script_data = json.load(f)

        project_id = script_data["project_id"]
        segments = script_data["segments"]
        hook = script_data.get("hook", "")
        outro = script_data.get("outro", "")

        audio_segments = []
        segment_index = 0

        # Generate hook audio
        if hook:
            hook_file = AUDIO_DIR / f"{project_id}_hook.mp3"
            self._generate_audio(hook, hook_file)
            audio_segments.append(AudioSegment(
                segment_index=segment_index,
                start_time=0,
                end_time=5,  # Estimated
                audio_file=str(hook_file),
                duration=5,
                text=hook
            ))
            segment_index += 1

        # Generate segment audio
        for i, seg in enumerate(segments):
            seg_file = AUDIO_DIR / f"{project_id}_seg{i:03d}.mp3"
            self._generate_audio(seg["text"], seg_file)

            audio_segments.append(AudioSegment(
                segment_index=segment_index,
                start_time=seg["start_time"],
                end_time=seg["end_time"],
                audio_file=str(seg_file),
                duration=seg["end_time"] - seg["start_time"],
                text=seg["text"]
            ))
            segment_index += 1

        # Generate outro audio
        if outro:
            outro_file = AUDIO_DIR / f"{project_id}_outro.mp3"
            self._generate_audio(outro, outro_file)
            audio_segments.append(AudioSegment(
                segment_index=segment_index,
                start_time=segments[-1]["end_time"] if segments else 0,
                end_time=script_data["total_duration"],
                audio_file=str(outro_file),
                duration=10,  # Estimated
                text=outro
            ))

        # Save segments metadata
        meta_file = AUDIO_DIR / f"{project_id}_segments_meta.json"
        with open(meta_file, "w") as f:
            json.dump([{
                "segment_index": s.segment_index,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "audio_file": s.audio_file,
                "duration": s.duration,
                "text": s.text
            } for s in audio_segments], f, indent=2)

        return audio_segments


def generate_voice(script_file: str, segmented: bool = False) -> dict:
    """
    Convenience function to generate voice audio.

    Args:
        script_file: Path to script JSON
        segmented: If True, generate separate files per segment

    Returns:
        Audio metadata dict or list of segments
    """
    generator = VoiceGenerator()

    if segmented:
        segments = generator.generate_segment_audio(script_file)
        return {"segments": [vars(s) for s in segments]}
    else:
        return generator.generate_from_script(script_file)
