"""
Video assembly module using FFmpeg.
Combines recording, audio, and captions into final video.
Features: Bold captions, transitions, professional styling.
"""
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

from config.settings import (
    VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS,
    VIDEO_CODEC, AUDIO_CODEC, AUDIO_SAMPLE_RATE,
    RECORDINGS_DIR, AUDIO_DIR, CAPTIONS_DIR, REVIEW_DIR,
    MIN_AUDIO_VIDEO_SYNC_TOLERANCE
)


@dataclass
class AssemblyResult:
    """Result of video assembly."""
    project_id: str
    output_file: str
    duration: float
    has_captions: bool
    audio_video_sync: float  # Ratio (1.0 = perfect)
    success: bool
    message: str


class VideoAssembler:
    """Assembles final videos from components."""

    def __init__(self):
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Verify FFmpeg is available."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")

    def assemble(self, project_id: str,
                 include_captions: bool = True,
                 include_intro: bool = False,
                 include_outro: bool = False) -> AssemblyResult:
        """
        Assemble final video from components.

        Args:
            project_id: Project identifier
            include_captions: Whether to burn in captions
            include_intro: Whether to prepend intro video
            include_outro: Whether to append outro video

        Returns:
            AssemblyResult with output details
        """
        # Locate files
        recording = RECORDINGS_DIR / f"{project_id}.webm"
        audio = AUDIO_DIR / f"{project_id}.mp3"
        captions = CAPTIONS_DIR / f"{project_id}.srt"
        output = REVIEW_DIR / f"{project_id}.mp4"

        # Validate inputs
        if not recording.exists():
            return AssemblyResult(
                project_id=project_id,
                output_file="",
                duration=0,
                has_captions=False,
                audio_video_sync=0,
                success=False,
                message=f"Recording not found: {recording}"
            )

        if not audio.exists():
            return AssemblyResult(
                project_id=project_id,
                output_file="",
                duration=0,
                has_captions=False,
                audio_video_sync=0,
                success=False,
                message=f"Audio not found: {audio}"
            )

        # Get durations for sync check
        video_duration = self._get_duration(recording)
        audio_duration = self._get_duration(audio)

        if video_duration == 0:
            return AssemblyResult(
                project_id=project_id,
                output_file="",
                duration=0,
                has_captions=False,
                audio_video_sync=0,
                success=False,
                message="Could not determine video duration"
            )

        # Calculate sync ratio
        sync_ratio = audio_duration / video_duration if video_duration > 0 else 0

        # Build FFmpeg command
        cmd = self._build_ffmpeg_command(
            recording=recording,
            audio=audio,
            output=output,
            captions=captions if include_captions and captions.exists() else None,
            target_duration=video_duration
        )

        # Execute FFmpeg
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Get final duration
            final_duration = self._get_duration(output)

            # Create metadata file
            self._create_review_metadata(project_id, output, final_duration)

            return AssemblyResult(
                project_id=project_id,
                output_file=str(output),
                duration=final_duration,
                has_captions=include_captions and captions.exists(),
                audio_video_sync=sync_ratio,
                success=True,
                message="Video assembled successfully"
            )

        except subprocess.CalledProcessError as e:
            return AssemblyResult(
                project_id=project_id,
                output_file="",
                duration=0,
                has_captions=False,
                audio_video_sync=sync_ratio,
                success=False,
                message=f"FFmpeg error: {e.stderr}"
            )

    def _build_ffmpeg_command(self, recording: Path, audio: Path,
                               output: Path, captions: Optional[Path],
                               target_duration: float,
                               with_transitions: bool = True) -> list[str]:
        """Build FFmpeg command for assembly with styled captions."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(recording),
            "-i", str(audio),
        ]

        # Video filters
        vf_filters = [
            f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}",
            f"fps={VIDEO_FPS}"
        ]

        # Add fade in/out transitions
        if with_transitions:
            vf_filters.append("fade=t=in:st=0:d=0.5")
            # Fade out at end (requires knowing duration, applied via separate filter)

        # Add styled captions if available
        # Bold, centered, with shadow/outline for readability
        if captions:
            caption_path = str(captions).replace("\\", "/").replace(":", "\\:")
            # Subtitle styling: bold, white, centered, with black outline
            # Force style overrides for ASS/SRT
            subtitle_filter = (
                f"subtitles='{caption_path}':"
                f"force_style='FontName=Arial Black,"
                f"FontSize=28,"
                f"PrimaryColour=&H00FFFFFF,"  # White
                f"OutlineColour=&H00000000,"  # Black outline
                f"BackColour=&H80000000,"     # Semi-transparent black background
                f"Bold=1,"
                f"Outline=3,"
                f"Shadow=2,"
                f"MarginV=60,"                # Bottom margin
                f"Alignment=2'"               # Center-bottom
            )
            vf_filters.append(subtitle_filter)

        cmd.extend(["-vf", ",".join(vf_filters)])

        # Audio settings
        cmd.extend([
            "-c:v", VIDEO_CODEC,
            "-c:a", AUDIO_CODEC,
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-shortest",
            "-preset", "medium",
            "-crf", "23",
            str(output)
        ])

        return cmd

    def _get_duration(self, file_path: Path) -> float:
        """Get media file duration in seconds."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            return 0

    def _create_review_metadata(self, project_id: str, video_path: Path,
                                 duration: float):
        """Create comprehensive metadata file for YouTube upload."""
        script_file = RECORDINGS_DIR / f"{project_id}_script.json"
        timestamps_file = RECORDINGS_DIR / f"{project_id}_timestamps.json"

        title = f"Tutorial: {project_id}"
        description = ""
        segments = []
        url = ""

        # Load script data
        if script_file.exists():
            with open(script_file) as f:
                script_data = json.load(f)
                title = script_data.get("title", title)
                hook = script_data.get("hook", "")
                outro = script_data.get("outro", "")
                segments = script_data.get("segments", [])

        # Load timestamps for chapter markers
        if timestamps_file.exists():
            with open(timestamps_file) as f:
                ts_data = json.load(f)
                url = ts_data.get("url", "")

        # Generate YouTube chapters from segments
        chapters = self._generate_chapters(segments)

        # Build full description
        description_parts = [
            hook,
            "",
            "TIMESTAMPS:",
            chapters,
            "",
            f"Demo URL: {url}" if url else "",
            "",
            outro,
            "",
            "---",
            "#AI #Tutorial #Technology #Automation"
        ]
        description = "\n".join(filter(None, description_parts))

        # Generate smart tags
        tags = self._generate_tags(title, hook)

        meta = {
            "project_id": project_id,
            "video_file": str(video_path),
            "thumbnail_file": str(REVIEW_DIR / f"{project_id}_thumb.jpg"),
            "duration_seconds": duration,
            "duration_formatted": self._format_duration(duration),
            "title": title,
            "description": description,
            "tags": tags,
            "chapters": chapters,
            "demo_url": url,
            "created_at": datetime.now().isoformat(),
            "status": "pending_review",
            "approved": False,
            "youtube_category": "28",  # Science & Technology
            "youtube_privacy": "private"
        }

        meta_file = REVIEW_DIR / f"{project_id}_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

    def _generate_chapters(self, segments: list) -> str:
        """Generate YouTube chapter timestamps from segments."""
        if not segments:
            return "0:00 Intro"

        chapters = ["0:00 Intro"]
        for i, seg in enumerate(segments):
            start = seg.get("start_time", 0)
            if start > 0:
                mins = int(start // 60)
                secs = int(start % 60)
                # Create chapter title from segment text (first few words)
                text = seg.get("text", f"Part {i+1}")
                chapter_title = " ".join(text.split()[:4])
                if len(chapter_title) > 30:
                    chapter_title = chapter_title[:30] + "..."
                chapters.append(f"{mins}:{secs:02d} {chapter_title}")

        return "\n".join(chapters)

    def _generate_tags(self, title: str, hook: str) -> list:
        """Generate relevant tags from content."""
        base_tags = ["tutorial", "ai", "technology", "how to", "guide"]

        # Extract keywords from title
        skip_words = {"the", "a", "an", "is", "are", "how", "to", "and", "or", "for", "in", "here", "this"}
        title_words = [w.lower() for w in title.split() if w.lower() not in skip_words and len(w) > 2]

        # Add title-based tags
        tags = base_tags + title_words[:5]

        # Add common AI-related tags if relevant
        ai_keywords = ["ai", "gpt", "claude", "llm", "chatgpt", "automation", "coding", "developer"]
        combined_text = (title + " " + hook).lower()
        for kw in ai_keywords:
            if kw in combined_text and kw not in tags:
                tags.append(kw)

        return list(set(tags))[:15]  # YouTube max 15 tags

    def _format_duration(self, seconds: float) -> str:
        """Format duration as MM:SS or HH:MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        if mins >= 60:
            hours = mins // 60
            mins = mins % 60
            return f"{hours}:{mins:02d}:{secs:02d}"
        return f"{mins}:{secs:02d}"

    def generate_captions(self, project_id: str) -> Optional[Path]:
        """
        Generate SRT captions from audio using Whisper.

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
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(str(audio_file), word_timestamps=True)

            # Convert to SRT
            self._write_srt(result, srt_file)
            return srt_file

        except ImportError:
            # Fall back to OpenAI Whisper API
            return self._whisper_api_transcribe(audio_file, srt_file)

    def _write_srt(self, whisper_result: dict, output: Path):
        """Convert Whisper result to SRT format."""
        segments = whisper_result.get("segments", [])
        lines = []

        for i, seg in enumerate(segments, 1):
            start = self._format_srt_time(seg["start"])
            end = self._format_srt_time(seg["end"])
            text = seg["text"].strip()

            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")

        output.write_text("\n".join(lines))

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT timestamp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _whisper_api_transcribe(self, audio: Path, output: Path) -> Optional[Path]:
        """Use OpenAI Whisper API for transcription."""
        try:
            import openai
            from config.settings import OPENAI_API_KEY

            client = openai.OpenAI(api_key=OPENAI_API_KEY)

            with open(audio, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="srt"
                )

            output.write_text(response)
            return output

        except Exception:
            return None


def assemble_video(project_id: str, with_captions: bool = True,
                   generate_thumbnail: bool = True) -> dict:
    """
    Convenience function to assemble a video with all features.

    Args:
        project_id: Project identifier
        with_captions: Include captions (2-3 words per frame)
        generate_thumbnail: Auto-generate YouTube thumbnail

    Returns:
        Assembly result as dict
    """
    from .captioner import CaptionGenerator
    from .thumbnail import ThumbnailGenerator

    assembler = VideoAssembler()

    # Generate captions with 2-3 word chunks
    if with_captions:
        captioner = CaptionGenerator(words_per_chunk=3)
        captioner.generate(project_id)

    # Assemble video
    result = assembler.assemble(project_id, include_captions=with_captions)

    # Generate thumbnail if video succeeded
    thumbnail_path = ""
    if result.success and generate_thumbnail:
        thumb_gen = ThumbnailGenerator()
        thumb_result = thumb_gen.generate(project_id)
        if thumb_result.success:
            thumbnail_path = thumb_result.thumbnail_path

    return {
        "project_id": result.project_id,
        "output_file": result.output_file,
        "thumbnail_file": thumbnail_path,
        "duration": result.duration,
        "has_captions": result.has_captions,
        "audio_video_sync": result.audio_video_sync,
        "success": result.success,
        "message": result.message
    }
