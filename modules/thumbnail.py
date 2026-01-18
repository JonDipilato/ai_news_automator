"""
Thumbnail generation module.
Creates dramatic YouTube thumbnails with bold text, glow effects, and professional styling.
Inspired by high-performing AI/Tech YouTube thumbnails.
"""
import subprocess
import json
import os
from pathlib import Path
from typing import Optional, List, Tuple
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
    """Generates dramatic YouTube thumbnails with bold styling."""

    # Color schemes for different themes
    THEMES = {
        "fire": {
            "primary": "#FF4444",
            "secondary": "#FF8800",
            "glow": "#FF0000",
            "bg_gradient": ["#1a0000", "#330000", "#000000"],
        },
        "electric": {
            "primary": "#00FFFF",
            "secondary": "#0088FF",
            "glow": "#00FFFF",
            "bg_gradient": ["#000022", "#001144", "#000000"],
        },
        "matrix": {
            "primary": "#00FF00",
            "secondary": "#88FF88",
            "glow": "#00FF00",
            "bg_gradient": ["#001100", "#002200", "#000000"],
        },
        "purple": {
            "primary": "#FF00FF",
            "secondary": "#AA00FF",
            "glow": "#FF00FF",
            "bg_gradient": ["#110022", "#220044", "#000000"],
        },
        "gold": {
            "primary": "#FFD700",
            "secondary": "#FFA500",
            "glow": "#FFFF00",
            "bg_gradient": ["#1a1100", "#332200", "#000000"],
        },
    }

    def __init__(self):
        self.thumb_width = 1280
        self.thumb_height = 720

    def generate_dramatic(
        self,
        project_id: str,
        title_text: str,
        subtitle_text: Optional[str] = None,
        theme: str = "fire",
        background_images: Optional[List[Path]] = None,
        badge_text: Optional[str] = None,
    ) -> ThumbnailResult:
        """
        Generate a dramatic, eye-catching thumbnail.

        Args:
            project_id: Project identifier
            title_text: Main bold text (2-4 words, ALL CAPS recommended)
            subtitle_text: Optional smaller text below title
            theme: Color theme (fire, electric, matrix, purple, gold)
            background_images: Optional list of screenshot paths to composite
            badge_text: Optional badge/label text (e.g., "NEW", "FREE")

        Returns:
            ThumbnailResult with path and status
        """
        thumbnail_path = REVIEW_DIR / f"{project_id}_thumb.jpg"
        temp_dir = RECORDINGS_DIR / "temp_thumb"
        temp_dir.mkdir(exist_ok=True)

        colors = self.THEMES.get(theme, self.THEMES["fire"])

        try:
            # Step 1: Create dramatic gradient background
            bg_path = temp_dir / "bg.png"
            self._create_gradient_background(bg_path, colors)

            # Step 2: Add code/tech pattern overlay
            pattern_path = temp_dir / "pattern.png"
            self._create_tech_pattern(pattern_path)

            # Step 3: Composite background images if provided
            composite_path = temp_dir / "composite.png"
            if background_images and any(p.exists() for p in background_images if p):
                self._composite_screenshots(
                    bg_path, [p for p in background_images if p and p.exists()],
                    composite_path, colors
                )
            else:
                # Just use gradient bg
                subprocess.run([
                    "cp", str(bg_path), str(composite_path)
                ], check=True)

            # Step 4: Add all text layers with glow effects
            self._add_dramatic_text(
                composite_path, thumbnail_path,
                title_text, subtitle_text, badge_text, colors
            )

            # Cleanup temp files
            for f in temp_dir.glob("*"):
                f.unlink()
            temp_dir.rmdir()

            return ThumbnailResult(
                project_id=project_id,
                thumbnail_path=str(thumbnail_path),
                title_text=title_text,
                success=True,
                message="Dramatic thumbnail generated"
            )

        except Exception as e:
            return ThumbnailResult(
                project_id=project_id,
                thumbnail_path="",
                title_text=title_text,
                success=False,
                message=f"Generation failed: {str(e)}"
            )

    def _create_gradient_background(self, output: Path, colors: dict):
        """Create a dramatic dark gradient background."""
        # Create radial gradient with vignette effect
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s={self.thumb_width}x{self.thumb_height}:d=1",
            "-vf", (
                # Base dark gradient
                f"geq=r='clip(128-sqrt((X-{self.thumb_width}/2)^2+(Y-{self.thumb_height}/2)^2)/3,0,255)':"
                f"g='clip(32-sqrt((X-{self.thumb_width}/2)^2+(Y-{self.thumb_height}/2)^2)/4,0,255)':"
                f"b='clip(32-sqrt((X-{self.thumb_width}/2)^2+(Y-{self.thumb_height}/2)^2)/4,0,255)'"
            ),
            "-frames:v", "1",
            str(output)
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    def _create_tech_pattern(self, output: Path):
        """Create subtle tech/code pattern overlay."""
        # For now, skip pattern (can be enhanced later)
        pass

    def _composite_screenshots(
        self, bg_path: Path, screenshots: List[Path],
        output: Path, colors: dict
    ):
        """Composite screenshots onto background with perspective/styling."""
        if not screenshots:
            subprocess.run(["cp", str(bg_path), str(output)], check=True)
            return

        # Take first screenshot and add it with border glow
        screenshot = screenshots[0]

        # Create composite with screenshot on right side, styled
        filter_complex = (
            # Scale screenshot to fit nicely
            f"[1:v]scale=600:-1,format=rgba,"
            # Add colored border/glow effect
            f"pad=w=iw+20:h=ih+20:x=10:y=10:color={colors['glow']}@0.8[img];"
            # Position on background
            f"[0:v][img]overlay=x={self.thumb_width}-650:y={self.thumb_height}/2-oh/2"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(bg_path),
            "-i", str(screenshot),
            "-filter_complex", filter_complex,
            "-frames:v", "1",
            str(output)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback to just background
            subprocess.run(["cp", str(bg_path), str(output)], check=True)

    def _add_dramatic_text(
        self, input_path: Path, output_path: Path,
        title: str, subtitle: Optional[str],
        badge: Optional[str], colors: dict
    ):
        """Add bold text with glow effects."""
        escaped_title = title.replace("'", "'\\''").replace(":", "\\:")

        # Build filter chain for dramatic text
        filters = []

        # Main title with multiple shadow layers for glow effect
        glow_color = colors["glow"].replace("#", "0x")
        primary_color = colors["primary"].replace("#", "")

        # Glow layer 1 (outer)
        filters.append(
            f"drawtext=text='{escaped_title}':"
            f"fontsize=100:"
            f"fontcolor={glow_color}@0.3:"
            f"font=Impact:"
            f"x=60:"
            f"y={self.thumb_height}/2-60:"
            f"shadowcolor={glow_color}@0.5:"
            f"shadowx=0:shadowy=0"
        )

        # Glow layer 2 (mid)
        filters.append(
            f"drawtext=text='{escaped_title}':"
            f"fontsize=100:"
            f"fontcolor={glow_color}@0.5:"
            f"font=Impact:"
            f"x=60:"
            f"y={self.thumb_height}/2-60:"
            f"shadowcolor=black:"
            f"shadowx=4:shadowy=4"
        )

        # Main title text
        filters.append(
            f"drawtext=text='{escaped_title}':"
            f"fontsize=100:"
            f"fontcolor=white:"
            f"borderw=4:"
            f"bordercolor=0x{primary_color}:"
            f"font=Impact:"
            f"x=60:"
            f"y={self.thumb_height}/2-60"
        )

        # Subtitle if provided
        if subtitle:
            escaped_sub = subtitle.replace("'", "'\\''").replace(":", "\\:")
            filters.append(
                f"drawtext=text='{escaped_sub}':"
                f"fontsize=50:"
                f"fontcolor=white@0.9:"
                f"font=Arial:"
                f"x=60:"
                f"y={self.thumb_height}/2+60"
            )

        # Badge if provided (top left corner)
        if badge:
            escaped_badge = badge.replace("'", "'\\''").replace(":", "\\:")
            # Badge background
            filters.append(
                f"drawbox=x=30:y=30:w=150:h=50:color={glow_color}:t=fill"
            )
            # Badge text
            filters.append(
                f"drawtext=text='{escaped_badge}':"
                f"fontsize=36:"
                f"fontcolor=white:"
                f"font=Impact:"
                f"x=55:y=38"
            )

        # Add bottom vignette for depth
        filters.append(
            f"drawbox=x=0:y={self.thumb_height}-100:w={self.thumb_width}:h=100:"
            f"color=black@0.7:t=fill"
        )

        filter_chain = ",".join(filters)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", filter_chain,
            "-q:v", "2",
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

    def generate(self, project_id: str, title_text: Optional[str] = None,
                 timestamp: float = 5.0) -> ThumbnailResult:
        """
        Generate thumbnail from video - now uses dramatic styling.

        Args:
            project_id: Project identifier
            title_text: 2-3 words to display (auto-extracted from script if None)
            timestamp: Video timestamp to capture

        Returns:
            ThumbnailResult with path and status
        """
        video_file = RECORDINGS_DIR / f"{project_id}.webm"
        script_file = RECORDINGS_DIR / f"{project_id}_script.json"
        temp_frame = RECORDINGS_DIR / f"{project_id}_temp_frame.png"

        if not video_file.exists():
            video_file = REVIEW_DIR / f"{project_id}.mp4"
            if not video_file.exists():
                # Try to find any scene video
                scene_videos = list(RECORDINGS_DIR.glob(f"{project_id}_scene*.webm"))
                if scene_videos:
                    video_file = scene_videos[0]
                else:
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

        # Extract a frame from video for background
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", str(video_file),
                "-vframes", "1",
                "-vf", f"scale={self.thumb_width}:{self.thumb_height}",
                str(temp_frame)
            ], capture_output=True, check=True)

            # Generate dramatic thumbnail
            result = self.generate_dramatic(
                project_id=project_id,
                title_text=title_text,
                theme="fire",  # Default theme
                background_images=[temp_frame] if temp_frame.exists() else None
            )

            # Cleanup temp frame
            if temp_frame.exists():
                temp_frame.unlink()

            return result

        except Exception as e:
            return ThumbnailResult(
                project_id=project_id,
                thumbnail_path="",
                title_text=title_text,
                success=False,
                message=f"Generation failed: {str(e)}"
            )

    def _extract_title_words(self, script_file: Path) -> str:
        """Extract punchy words from script title."""
        if not script_file.exists():
            return "MUST SEE"

        try:
            with open(script_file) as f:
                data = json.load(f)

            title = data.get("title", "")

            # Extract key words
            skip_words = {"the", "a", "an", "is", "are", "how", "to", "and", "or",
                         "for", "in", "on", "with", "that", "will", "your"}
            words = [w for w in title.split() if w.lower() not in skip_words]

            # Take impactful words
            key_words = words[:3] if len(words) >= 3 else words[:2]

            if not key_words:
                key_words = ["AI", "TOOLS"]

            return " ".join(key_words).upper()

        except Exception:
            return "AI TOOLS"

    def generate_with_screenshot(self, project_id: str, screenshot_path: Path,
                                  title_text: str, theme: str = "fire") -> ThumbnailResult:
        """
        Generate thumbnail from existing screenshot with dramatic styling.

        Args:
            project_id: Project identifier
            screenshot_path: Path to screenshot image
            title_text: Bold title text
            theme: Color theme

        Returns:
            ThumbnailResult with path and status
        """
        if not screenshot_path.exists():
            return ThumbnailResult(
                project_id=project_id,
                thumbnail_path="",
                title_text=title_text,
                success=False,
                message="Screenshot not found"
            )

        return self.generate_dramatic(
            project_id=project_id,
            title_text=title_text,
            theme=theme,
            background_images=[screenshot_path]
        )


def generate_thumbnail(project_id: str, title_text: Optional[str] = None,
                       timestamp: float = 5.0) -> dict:
    """
    Convenience function to generate thumbnail.

    Args:
        project_id: Project identifier
        title_text: Optional title (auto-extracted if None)
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


def generate_dramatic_thumbnail(
    project_id: str,
    title_text: str,
    subtitle_text: Optional[str] = None,
    theme: str = "fire",
    screenshot_path: Optional[Path] = None
) -> dict:
    """
    Generate a dramatic YouTube thumbnail.

    Args:
        project_id: Project identifier
        title_text: Main bold text (2-4 words recommended)
        subtitle_text: Optional subtitle
        theme: Color theme (fire, electric, matrix, purple, gold)
        screenshot_path: Optional background screenshot

    Returns:
        Result dict with thumbnail path and status
    """
    generator = ThumbnailGenerator()

    background = [screenshot_path] if screenshot_path else None
    result = generator.generate_dramatic(
        project_id=project_id,
        title_text=title_text,
        subtitle_text=subtitle_text,
        theme=theme,
        background_images=background
    )

    return {
        "project_id": result.project_id,
        "thumbnail_path": result.thumbnail_path,
        "title_text": result.title_text,
        "success": result.success,
        "message": result.message
    }
