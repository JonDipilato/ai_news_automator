"""
Thumbnail generation module.
Creates YouTube thumbnails with bold text overlay using FFmpeg.
"""
import subprocess
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from config.settings import (
    RECORDINGS_DIR, REVIEW_DIR, ASSETS_DIR,
    VIDEO_WIDTH, VIDEO_HEIGHT
)


@dataclass
class ThumbnailResult:
    """Result of thumbnail generation."""
    project_id: str
    thumbnail_path: str
    title_text: str
    success: bool
    message: str


class ThumbnailGenerator:
    """Generates YouTube thumbnails from video frames with text overlay."""

    def __init__(self):
        self.width = VIDEO_WIDTH
        self.height = VIDEO_HEIGHT
        # Thumbnail dimensions (YouTube recommended)
        self.thumb_width = 1280
        self.thumb_height = 720

    def generate(self, project_id: str, title_text: Optional[str] = None,
                 timestamp: float = 5.0) -> ThumbnailResult:
        """
        Generate thumbnail from video with text overlay.

        Args:
            project_id: Project identifier
            title_text: 2-3 words to display (auto-extracted from script if None)
            timestamp: Video timestamp to capture (default 5 seconds in)

        Returns:
            ThumbnailResult with path and status
        """
        video_file = RECORDINGS_DIR / f"{project_id}.webm"
        script_file = RECORDINGS_DIR / f"{project_id}_script.json"
        thumbnail_path = REVIEW_DIR / f"{project_id}_thumb.jpg"

        if not video_file.exists():
            # Try review folder
            video_file = REVIEW_DIR / f"{project_id}.mp4"
            if not video_file.exists():
                return ThumbnailResult(
                    project_id=project_id,
                    thumbnail_path="",
                    title_text="",
                    success=False,
                    message="Video file not found"
                )

        # Extract title text if not provided
        if not title_text:
            title_text = self._extract_title_words(script_file)

        # Generate thumbnail with FFmpeg
        try:
            self._create_thumbnail(video_file, thumbnail_path, title_text, timestamp)

            return ThumbnailResult(
                project_id=project_id,
                thumbnail_path=str(thumbnail_path),
                title_text=title_text,
                success=True,
                message="Thumbnail generated successfully"
            )

        except Exception as e:
            return ThumbnailResult(
                project_id=project_id,
                thumbnail_path="",
                title_text=title_text,
                success=False,
                message=f"Generation failed: {str(e)}"
            )

    def _extract_title_words(self, script_file: Path) -> str:
        """Extract 2-3 punchy words from script title."""
        if not script_file.exists():
            return "MUST SEE"

        try:
            with open(script_file) as f:
                data = json.load(f)

            title = data.get("title", "")

            # Extract key words (skip common words)
            skip_words = {"the", "a", "an", "is", "are", "how", "to", "and", "or", "for", "in", "on", "with"}
            words = [w for w in title.split() if w.lower() not in skip_words]

            # Take first 2-3 impactful words
            key_words = words[:3] if len(words) >= 3 else words[:2]

            if not key_words:
                key_words = ["MUST", "SEE"]

            return " ".join(key_words).upper()

        except Exception:
            return "MUST SEE"

    def _create_thumbnail(self, video: Path, output: Path,
                          title_text: str, timestamp: float):
        """Create thumbnail using FFmpeg with text overlay."""
        # Escape special characters for FFmpeg
        escaped_text = title_text.replace("'", "'\\''").replace(":", "\\:")

        # FFmpeg filter for thumbnail with text
        # - Extract frame at timestamp
        # - Scale to thumbnail size
        # - Add dark gradient overlay at top
        # - Add bold white text with shadow
        filter_complex = (
            f"scale={self.thumb_width}:{self.thumb_height},"
            # Dark gradient at top for text readability
            f"drawbox=x=0:y=0:w=iw:h=ih/3:color=black@0.5:t=fill,"
            # Main title text - bold, centered at top
            f"drawtext=text='{escaped_text}':"
            f"fontsize=80:"
            f"fontcolor=white:"
            f"font=Arial Black:"
            f"x=(w-text_w)/2:"
            f"y=80:"
            f"shadowcolor=black:"
            f"shadowx=4:"
            f"shadowy=4"
        )

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", str(video),
            "-vframes", "1",
            "-vf", filter_complex,
            "-q:v", "2",
            str(output)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

    def generate_with_screenshot(self, project_id: str, screenshot_path: Path,
                                  title_text: str) -> ThumbnailResult:
        """
        Generate thumbnail from existing screenshot with text overlay.

        Args:
            project_id: Project identifier
            screenshot_path: Path to screenshot image
            title_text: 2-3 words to display

        Returns:
            ThumbnailResult with path and status
        """
        thumbnail_path = REVIEW_DIR / f"{project_id}_thumb.jpg"

        if not screenshot_path.exists():
            return ThumbnailResult(
                project_id=project_id,
                thumbnail_path="",
                title_text=title_text,
                success=False,
                message="Screenshot not found"
            )

        try:
            escaped_text = title_text.replace("'", "'\\''").replace(":", "\\:")

            filter_complex = (
                f"scale={self.thumb_width}:{self.thumb_height},"
                f"drawbox=x=0:y=0:w=iw:h=ih/3:color=black@0.5:t=fill,"
                f"drawtext=text='{escaped_text}':"
                f"fontsize=80:"
                f"fontcolor=white:"
                f"font=Arial Black:"
                f"x=(w-text_w)/2:"
                f"y=80:"
                f"shadowcolor=black:"
                f"shadowx=4:"
                f"shadowy=4"
            )

            cmd = [
                "ffmpeg", "-y",
                "-i", str(screenshot_path),
                "-vf", filter_complex,
                "-q:v", "2",
                str(thumbnail_path)
            ]

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            return ThumbnailResult(
                project_id=project_id,
                thumbnail_path=str(thumbnail_path),
                title_text=title_text,
                success=True,
                message="Thumbnail generated from screenshot"
            )

        except Exception as e:
            return ThumbnailResult(
                project_id=project_id,
                thumbnail_path="",
                title_text=title_text,
                success=False,
                message=f"Generation failed: {str(e)}"
            )


def generate_thumbnail(project_id: str, title_text: Optional[str] = None,
                       timestamp: float = 5.0) -> dict:
    """
    Convenience function to generate thumbnail.

    Args:
        project_id: Project identifier
        title_text: Optional 2-3 word title (auto-extracted if None)
        timestamp: Video timestamp to capture

    Returns:
        Result dict with thumbnail path and status
    """
    generator = ThumbnailGenerator()
    result = generator.generate(project_id, title_text, timestamp)

    return {
        "project_id": result.project_id,
        "thumbnail_path": result.thumbnail_path,
        "title_text": result.title_text,
        "success": result.success,
        "message": result.message
    }
