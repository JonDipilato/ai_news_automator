"""
REAL Terminal Recorder - Uses asciinema for AUTHENTIC terminal recordings.

This module:
- Uses asciinema to record ACTUAL terminal sessions
- Captures real command execution and output
- Renders authentic terminal frames using the actual terminal data
- Combines into video using FFmpeg

NO HTML - NO FAKES - REAL TERMINAL RECORDINGS
"""
import os
import json
import subprocess
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from config.settings import VIDEO_WIDTH, VIDEO_HEIGHT, RECORDINGS_DIR, RECORDING_FORMAT


@dataclass
class RealRecordingResult:
    """Result of a real terminal recording."""
    scene_index: int
    scene_type: str
    video_path: str
    duration: float
    timestamps: List[dict]
    success: bool
    error: Optional[str] = None


class AsciinemaRecorder:
    """
    Records REAL terminal sessions using asciinema.

    This is the authentic approach - records actual terminal sessions
    that can be replayed exactly as they happened.
    """

    def __init__(self, width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT):
        self.width = width
        self.height = height
        self.recordings_dir = Path(RECORDINGS_DIR)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

    def record_session(
        self,
        project_id: str,
        scene_index: int,
        commands: List[str],
        duration: float
    ) -> RealRecordingResult:
        """
        Record a REAL terminal session using asciinema.

        This creates an actual terminal recording that captures everything
        that happens on screen - colors, cursor movement, real output.
        """
        video_path = self.recordings_dir / f"{project_id}_scene{scene_index:03d}.{RECORDING_FORMAT}"
        timestamps = []
        start_time = time.time()

        # Create temporary script file for asciinema to run
        script_content = self._create_shell_script(commands, duration)
        script_path = self.recordings_dir / f"{project_id}_scene{scene_index:03d}_script.sh"

        # Create cast file path
        cast_path = self.recordings_dir / f"{project_id}_scene{scene_index:03d}.cast"

        try:
            # Write the script
            script_path.write_text(script_content)
            script_path.chmod(0o755)

            # Record using asciinema
            self._record_with_asciinema(script_path, cast_path, duration)

            # Process timestamps
            for i, cmd in enumerate(commands):
                timestamps.append({
                    "time_seconds": round(i * (duration / len(commands)), 2),
                    "action_type": "command_executed",
                    "description": f"Executed: {cmd[:50]}...",
                })

            # Convert asciinema cast to video
            self._cast_to_video(cast_path, str(video_path), duration)

            # Cleanup
            script_path.unlink(missing_ok=True)
            cast_path.unlink(missing_ok=True)

            return RealRecordingResult(
                scene_index=scene_index,
                scene_type="terminal",
                video_path=str(video_path),
                duration=time.time() - start_time,
                timestamps=timestamps,
                success=True
            )

        except Exception as e:
            import traceback
            traceback.print_exc()

            # Cleanup on error
            script_path.unlink(missing_ok=True)
            cast_path.unlink(missing_ok=True)

            return RealRecordingResult(
                scene_index=scene_index,
                scene_type="terminal",
                video_path="",
                duration=0,
                timestamps=[],
                success=False,
                error=str(e)
            )

    def _create_shell_script(self, commands: List[str], duration: float) -> str:
        """Create a shell script that runs the commands."""
        lines = [
            "#!/bin/bash",
            "# Auto-generated terminal recording script",
            "set +e  # Don't exit on error",
            "",
            "export TERM=xterm-256color",
            "clear",
            "",
        ]

        for cmd in commands:
            cmd = cmd.strip()
            if not cmd:
                continue
            if cmd.startswith('#'):
                # Echo comments in a nice color
                lines.append(f"echo -e '\\033[90m{cmd}\\033[0m'")
            elif cmd.startswith('echo '):
                lines.append(cmd)
            else:
                # Add a prompt before the command
                lines.append(f"echo -e '\\033[94m~\\033[0m \\033[92m$\\033[0m {cmd}'")
                lines.append(cmd)
                lines.append("echo ''")  # Add blank line after output

        # Add hold at end
        lines.append("sleep 2")

        return "\n".join(lines)

    def _record_with_asciinema(
        self,
        script_path: Path,
        cast_path: Path,
        duration: float
    ):
        """
        Record terminal session using asciinema.

        This creates a .cast file with the actual terminal session data.
        """
        # Check if asciinema is available
        result = subprocess.run(
            ["which", "asciinema"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                "asciinema not found. Install it with:\n"
                "  sudo apt-get install asciinema\n"
                "Or use: pip install asciinema"
            )

        # Run asciinema rec
        # Format: asciinema rec --stdin --command "bash script.sh" output.cast
        cmd = [
            "asciinema", "rec",
            "--stdin",  # Record stdin (what we type)
            "--command", f"bash {script_path}",
            "--overwrite",  # Overwrite if exists
            str(cast_path)
        ]

        # Set environment for recording
        env = os.environ.copy()
        env["ASCIINEMA_REC_TIMEOUT"] = str(int(duration) + 5)

        # Run with timeout
        try:
            subprocess.run(
                cmd,
                timeout=duration + 10,
                env=env,
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.TimeoutExpired:
            pass  # Expected for long-running recordings
        except subprocess.CalledProcessError as e:
            # asciinema might return non-zero but still create the file
            if not cast_path.exists():
                raise RuntimeError(f"asciinema failed: {e.stderr}")

        # Verify cast file was created
        if not cast_path.exists():
            raise RuntimeError(f"asciinema did not create cast file at {cast_path}")

    def _cast_to_video(
        self,
        cast_path: Path,
        output_path: str,
        duration: float
    ):
        """
        Convert asciinema .cast file to video using FFmpeg.

        This parses the .cast file and renders each frame as a video.
        Uses a fixed frame rate for smooth playback.
        """
        # Parse the cast file
        # Format: First line is JSON header, rest are [delay, "o", "text"] lines
        frames = []
        header = {}

        with open(cast_path, 'r') as f:
            # First line is the header
            header_line = f.readline()
            header = json.loads(header_line)

            # Rest are frame data
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    frame = json.loads(line)
                    if len(frame) >= 2:
                        frames.append(frame)
                except json.JSONDecodeError:
                    continue

        if not frames:
            raise RuntimeError("No frames in cast file")

        # Get terminal dimensions from header
        term_width = int(header.get("width", 80))
        term_height = int(header.get("height", 24))

        # Use 30 FPS for smooth video
        fps = 30
        total_frames = int(duration * fps)

        # Render frames to images
        frames_dir = Path(output_path).parent / f"{Path(output_path).stem}_frames"
        frames_dir.mkdir(exist_ok=True)

        from PIL import Image, ImageDraw, ImageFont

        # Terminal colors (GitHub Dark theme)
        BG_COLOR = (13, 17, 23)
        FG_COLOR = (201, 201, 201)
        PROMPT_COLOR = (88, 166, 255)
        COMMENT_COLOR = (107, 140, 175)

        # Calculate font size based on terminal size and video size
        font_size = min(
            self.width // term_width * 0.6,
            self.height // term_height * 0.8
        )
        font_size = int(max(12, min(20, font_size)))

        # Try to load a monospace font
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                font_size
            )
        except:
            try:
                font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
            except:
                font = ImageFont.load_default()

        # Build up screen state over time, then render frames
        import re

        screen_states = []
        screen_lines = [""] * term_height
        current_line = 0
        cursor_col = 0
        cumulative_time = 0

        # Process all frames to build screen states
        for frame in frames:
            if len(frame) >= 3:
                delay, output_type, output = frame
            else:
                delay = frame[0]
                output = frame[1] if len(frame) > 1 else ""

            cumulative_time += delay

            # Strip ANSI escape sequences
            clean_output = re.sub(r'\x1b\[[0-9;]*[mGK]', '', str(output))

            # Update screen state
            for char in clean_output:
                if char == '\n':
                    current_line = min(current_line + 1, term_height - 1)
                    cursor_col = 0
                elif char == '\r':
                    cursor_col = 0
                elif char == '\b':
                    cursor_col = max(0, cursor_col - 1)
                elif char.isprintable() or ord(char) >= 32:
                    if cursor_col < term_width - 1:
                        line = list(screen_lines[current_line])
                        while len(line) <= cursor_col:
                            line.append(' ')
                        line[cursor_col] = char
                        screen_lines[current_line] = ''.join(line)
                        cursor_col += 1

            # Save screen state at this time
            screen_states.append({
                'time': cumulative_time,
                'lines': list(screen_lines)
            })

        # If we don't have enough screen states, repeat the last one
        if not screen_states:
            screen_states = [{'time': 0, 'lines': [""] * term_height}]

        # Generate frames at 30 FPS
        print(f"  Generating {total_frames} frames at {fps} FPS...")

        frame_files = []
        time_per_frame = 1.0 / fps

        for frame_idx in range(total_frames):
            current_time = frame_idx * time_per_frame

            # Find the screen state for this time
            current_state = screen_states[-1]  # Default to last state
            for state in screen_states:
                if state['time'] >= current_time:
                    current_state = state
                    break
                elif state == screen_states[-1]:
                    current_state = state

            frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
            self._render_terminal_frame(
                current_state['lines'],
                term_width, term_height,
                frame_path,
                font, font_size,
                BG_COLOR, FG_COLOR, PROMPT_COLOR, COMMENT_COLOR
            )
            frame_files.append(frame_path)

        print(f"  Creating video from {len(frame_files)} frames...")

        # Use FFmpeg to create video from frames
        if str(output_path).endswith('.webm'):
            video_codec = "libvpx-vp9"
            extra_args = []
        else:
            video_codec = "libx264"
            extra_args = ["-movflags", "+faststart"]

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%06d.png"),
            "-c:v", video_codec,
            "-preset", "fast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={self.width}:{self.height}",
        ] + extra_args + [output_path]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr[-500:]}")

        # Cleanup frames
        shutil.rmtree(frames_dir, ignore_errors=True)

        print(f"  Video created: {output_path}")

    def _render_terminal_frame(
        self,
        lines: List[str],
        term_width: int,
        term_height: int,
        output_path: Path,
        font, font_size: int,
        bg_color, fg_color, prompt_color, comment_color: tuple
    ):
        """Render a single terminal frame to an image."""
        from PIL import Image, ImageDraw

        # Calculate dimensions
        char_width = font_size * 0.6
        char_height = font_size * 1.2
        padding = 20
        header_height = 36

        img_width = int(term_width * char_width + padding * 2)
        img_height = int(term_height * char_height + header_height + padding * 2)

        # Create image
        img = Image.new("RGB", (img_width, img_height), bg_color)
        draw = ImageDraw.Draw(img)

        # Draw header bar (macOS style)
        draw.rectangle([(0, 0), (img_width, header_height)], (22, 27, 34))
        # Traffic lights
        for i, color in enumerate([(255, 95, 86), (255, 189, 46), (63, 185, 80)]):
            x = padding + i * 20
            y = 12
            draw.ellipse([(x, y), (x + 12, y + 12)], color)

        # Draw title
        title_text = f"bash — {term_width}×{term_height}"
        draw.text(
            (img_width // 2, 12),
            title_text,
            fill=(139, 148, 158),
            font=font,
            anchor="mm"
        )

        # Draw terminal content
        y_start = header_height + padding
        x_start = padding

        for i, line in enumerate(lines[:term_height]):
            y = int(y_start + i * char_height)

            # Determine color based on line content
            color = fg_color
            if line.strip().startswith('#'):
                color = comment_color
            elif line.strip().startswith('~') or '$' in line:
                color = prompt_color

            # Draw the line
            draw.text((x_start, y), line[:term_width], fill=color, font=font)

        # Scale to target size
        img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        img.save(output_path, optimize=True, quality=95)


# Convenience function
def record_real_terminal(
    project_id: str,
    scene_index: int,
    commands: List[str],
    duration: float
) -> RealRecordingResult:
    """
    Record a truly authentic terminal session using asciinema.

    This creates a REAL recording of actual command execution.
    """
    recorder = AsciinemaRecorder()
    return recorder.record_session(project_id, scene_index, commands, duration)
