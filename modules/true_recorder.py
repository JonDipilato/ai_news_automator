"""
TrueTerminalRecorder - Authentic terminal recording via PIL frame rendering.

Executes REAL commands via subprocess, captures REAL output,
renders high-fidelity terminal frames with ANSI color support,
and combines them into a video.

Best for headless/WSL environments where screen capture isn't available.
"""
import os
import re
import subprocess
import time
import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from config.settings import VIDEO_WIDTH, VIDEO_HEIGHT, RECORDINGS_DIR, RECORDING_FORMAT

# Optional pyte for terminal emulation (better ANSI parsing)
try:
    import pyte
    PYTE_AVAILABLE = True
except ImportError:
    PYTE_AVAILABLE = False


# GitHub Dark theme colors
COLORS = {
    "background": (13, 17, 23),
    "foreground": (201, 209, 217),
    "prompt": (88, 166, 255),
    "prompt_arrow": (248, 129, 102),
    "prompt_dir": (126, 231, 135),
    "comment": (139, 148, 158),
    "success": (63, 185, 80),
    "error": (248, 81, 73),
    "warning": (210, 153, 34),
    "info": (88, 166, 255),
    "string": (165, 214, 255),
    "keyword": (255, 123, 114),
    "cursor": (88, 166, 255),
    "header": (22, 27, 34),
    "header_text": (139, 148, 158),
}

# ANSI color code mapping
ANSI_COLORS = {
    30: (0, 0, 0),          # Black
    31: (248, 81, 73),      # Red
    32: (63, 185, 80),      # Green
    33: (210, 153, 34),     # Yellow
    34: (88, 166, 255),     # Blue
    35: (188, 140, 255),    # Magenta
    36: (57, 199, 199),     # Cyan
    37: (201, 209, 217),    # White
    90: (139, 148, 158),    # Bright Black (Gray)
    91: (255, 123, 114),    # Bright Red
    92: (126, 231, 135),    # Bright Green
    93: (255, 209, 102),    # Bright Yellow
    94: (165, 214, 255),    # Bright Blue
    95: (214, 171, 255),    # Bright Magenta
    96: (107, 237, 237),    # Bright Cyan
    97: (255, 255, 255),    # Bright White
}


@dataclass
class TrueRecordingResult:
    """Result of a true screen recording."""
    scene_index: int
    scene_type: str
    video_path: str
    duration: float
    timestamps: List[dict]
    success: bool
    error: Optional[str] = None


@dataclass
class StyledChar:
    """A character with styling information."""
    char: str
    fg_color: tuple = None
    bg_color: tuple = None
    bold: bool = False


class TerminalEmulator:
    """
    Terminal emulator for processing command output with ANSI codes.

    Uses pyte if available for accurate terminal emulation,
    otherwise falls back to basic ANSI parsing.
    """

    def __init__(self, cols: int = 100, rows: int = 30):
        self.cols = cols
        self.rows = rows

        if PYTE_AVAILABLE:
            self.screen = pyte.Screen(cols, rows)
            self.stream = pyte.Stream(self.screen)
            self.use_pyte = True
        else:
            self.use_pyte = False
            self.lines: List[List[StyledChar]] = [
                [StyledChar(" ") for _ in range(cols)]
                for _ in range(rows)
            ]
            self.cursor_row = 0
            self.cursor_col = 0

    def feed(self, text: str):
        """Process text with potential ANSI codes."""
        if self.use_pyte:
            self.stream.feed(text)
        else:
            self._process_ansi(text)

    def get_display(self) -> List[List[StyledChar]]:
        """Get the current display as styled characters."""
        if self.use_pyte:
            return self._pyte_to_styled()
        return self.lines

    def _pyte_to_styled(self) -> List[List[StyledChar]]:
        """Convert pyte screen to styled characters."""
        result = []
        for y in range(self.screen.lines):
            row = []
            for x in range(self.screen.columns):
                char_data = self.screen.buffer[y][x]
                char = char_data.data if char_data.data else " "

                # Get colors from pyte
                fg = COLORS["foreground"]
                if char_data.fg and char_data.fg != "default":
                    if isinstance(char_data.fg, str) and char_data.fg.isdigit():
                        fg = ANSI_COLORS.get(int(char_data.fg), fg)

                row.append(StyledChar(
                    char=char,
                    fg_color=fg,
                    bold=char_data.bold
                ))
            result.append(row)
        return result

    def _process_ansi(self, text: str):
        """Basic ANSI processing (fallback when pyte unavailable)."""
        # Remove ANSI codes for basic processing
        clean_text = re.sub(r'\x1b\[[0-9;]*[mGKH]', '', text)

        for char in clean_text:
            if char == '\n':
                self.cursor_row = min(self.cursor_row + 1, self.rows - 1)
                self.cursor_col = 0
            elif char == '\r':
                self.cursor_col = 0
            elif char.isprintable():
                if self.cursor_col < self.cols:
                    self.lines[self.cursor_row][self.cursor_col] = StyledChar(char)
                    self.cursor_col += 1


class TrueTerminalRenderer:
    """
    High-fidelity terminal renderer using PIL.

    Creates visually authentic terminal frames with:
    - macOS-style window chrome
    - Proper font rendering
    - Cursor animation
    - ANSI color support
    """

    def __init__(self, cols: int = 100, rows: int = 30, font_size: int = 14):
        self.cols = cols
        self.rows = rows
        self.font_size = font_size
        self.char_width = int(font_size * 0.6)
        self.char_height = int(font_size * 1.4)
        self.padding = 24
        self.header_height = 40

        # Calculate dimensions
        self.content_width = self.cols * self.char_width + (self.padding * 2)
        self.content_height = self.rows * self.char_height + self.padding
        self.width = max(self.content_width, VIDEO_WIDTH)
        self.height = self.header_height + self.content_height

        # Load fonts
        self.font = self._load_font(font_size)
        self.small_font = self._load_font(font_size - 2)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load a monospace font, trying several options."""
        font_paths = [
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
            # Windows
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/CascadiaCode.ttf",
            # macOS
            "/System/Library/Fonts/Monaco.dfont",
            "/System/Library/Fonts/SFNSMono.ttf",
            # Fallback names
            "DejaVuSansMono.ttf",
            "Consolas.ttf",
        ]

        for path in font_paths:
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue

        return ImageFont.load_default()

    def render_frame(
        self,
        lines: List[List[StyledChar]],
        cursor_pos: Optional[Tuple[int, int]] = None,
        show_cursor: bool = True,
        title: str = "bash"
    ) -> Image.Image:
        """
        Render a terminal frame.

        Args:
            lines: 2D list of StyledChar objects
            cursor_pos: (row, col) position of cursor
            show_cursor: Whether to show the cursor
            title: Window title

        Returns:
            PIL Image of the rendered frame
        """
        # Create image with background
        img = Image.new("RGB", (self.width, self.height), COLORS["background"])
        draw = ImageDraw.Draw(img)

        # Draw window chrome (macOS style)
        self._draw_header(draw, title)

        # Draw terminal content
        y_start = self.header_height + 16
        x_start = self.padding + (self.width - self.content_width) // 2

        for row_idx, row in enumerate(lines[:self.rows]):
            y = y_start + row_idx * self.char_height

            for col_idx, styled_char in enumerate(row[:self.cols]):
                x = x_start + col_idx * self.char_width
                char = styled_char.char if styled_char.char else " "
                color = styled_char.fg_color or COLORS["foreground"]

                if styled_char.bold:
                    # Draw twice with offset for bold effect
                    draw.text((x, y), char, fill=color, font=self.font)
                    draw.text((x + 1, y), char, fill=color, font=self.font)
                else:
                    draw.text((x, y), char, fill=color, font=self.font)

        # Draw cursor
        if show_cursor and cursor_pos:
            row, col = cursor_pos
            if 0 <= row < self.rows and 0 <= col < self.cols:
                cursor_x = x_start + col * self.char_width
                cursor_y = y_start + row * self.char_height
                draw.rectangle(
                    [(cursor_x, cursor_y),
                     (cursor_x + self.char_width - 2, cursor_y + self.char_height - 2)],
                    fill=COLORS["cursor"]
                )

        return img

    def _draw_header(self, draw: ImageDraw.Draw, title: str):
        """Draw macOS-style window header."""
        # Header background
        draw.rectangle([(0, 0), (self.width, self.header_height)], COLORS["header"])

        # Traffic light buttons
        button_y = (self.header_height - 12) // 2
        buttons = [
            (self.padding, (255, 95, 86)),       # Close (red)
            (self.padding + 20, (255, 189, 46)), # Minimize (yellow)
            (self.padding + 40, (39, 201, 63)),  # Maximize (green)
        ]

        for x, color in buttons:
            draw.ellipse([(x, button_y), (x + 12, button_y + 12)], fill=color)
            # Inner highlight
            draw.ellipse([(x + 2, button_y + 2), (x + 5, button_y + 5)],
                        fill=(255, 255, 255, 50))

        # Window title
        try:
            title_text = f"Terminal \u2014 {title}"
            bbox = draw.textbbox((0, 0), title_text, font=self.small_font)
            title_width = bbox[2] - bbox[0]
            title_x = (self.width - title_width) // 2
            title_y = (self.header_height - (bbox[3] - bbox[1])) // 2
            draw.text((title_x, title_y), title_text, fill=COLORS["header_text"],
                     font=self.small_font)
        except Exception:
            pass

    def save_frame(self, img: Image.Image, path: str, quality: int = 95):
        """Save a frame to disk."""
        img.save(path, optimize=True, quality=quality)


class TrueTerminalRecorder:
    """
    Records REAL terminal sessions.

    Commands are ACTUALLY executed.
    Output is REAL.
    Frames are REAL images.
    Video is REAL.
    """

    def __init__(self, width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT):
        self.width = width
        self.height = height
        self.recordings_dir = Path(RECORDINGS_DIR)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.renderer = TrueTerminalRenderer()
        self.emulator = TerminalEmulator()

    def record_session(
        self,
        project_id: str,
        scene_index: int,
        commands: List[str],
        duration: float
    ) -> TrueRecordingResult:
        """
        Record a truly authentic terminal session.

        Args:
            project_id: Project identifier
            scene_index: Scene index for file naming
            commands: List of commands to execute
            duration: Target duration in seconds

        Returns:
            TrueRecordingResult with recording details
        """
        video_path = self.recordings_dir / f"{project_id}_scene{scene_index:03d}.mp4"
        timestamps = []
        start_time = time.time()

        try:
            # Execute commands and capture output
            command_outputs = self._execute_commands(commands, duration, timestamps)

            # Generate frames with typing animation
            frames = self._generate_frames(command_outputs, duration)

            # Create video from frames
            self._create_video(frames, str(video_path), duration)

            return TrueRecordingResult(
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
            return TrueRecordingResult(
                scene_index=scene_index,
                scene_type="terminal",
                video_path="",
                duration=0,
                timestamps=[],
                success=False,
                error=str(e)
            )

    def _execute_commands(
        self,
        commands: List[str],
        duration: float,
        timestamps: List[dict]
    ) -> List[dict]:
        """Execute commands for real and capture their output."""
        results = []

        for i, cmd in enumerate(commands):
            if not cmd.strip():
                continue

            cmd_time = i * (duration / max(len(commands), 1))

            # Skip comments but include them for display
            if cmd.strip().startswith('#'):
                results.append({
                    "command": cmd,
                    "output": "",
                    "exit_code": 0,
                    "is_comment": True
                })
                continue

            # Execute the command
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd=os.path.expanduser("~"),
                    env={**os.environ, "TERM": "xterm-256color"}
                )

                output = result.stdout
                if result.stderr and result.returncode != 0:
                    output = result.stderr

                results.append({
                    "command": cmd,
                    "output": output.strip(),
                    "exit_code": result.returncode,
                    "is_comment": False
                })

                timestamps.append({
                    "time_seconds": round(cmd_time, 2),
                    "action_type": "command_executed",
                    "description": f"Executed: {cmd[:50]}",
                    "output_preview": output[:100] if output else ""
                })

            except subprocess.TimeoutExpired:
                results.append({
                    "command": cmd,
                    "output": "Command timed out",
                    "exit_code": 1,
                    "is_comment": False
                })
            except Exception as e:
                results.append({
                    "command": cmd,
                    "output": str(e),
                    "exit_code": 1,
                    "is_comment": False
                })

        return results

    def _generate_frames(
        self,
        command_outputs: List[dict],
        duration: float
    ) -> List[Image.Image]:
        """Generate frames with typing animation."""
        frames = []
        fps = 30
        total_frames = int(duration * fps)

        # Build the sequence of display states
        display_states = self._build_display_states(command_outputs, duration)

        print(f"  Generating {total_frames} frames at {fps} FPS...")

        for frame_idx in range(total_frames):
            current_time = frame_idx / fps

            # Find the appropriate display state
            state = self._get_state_at_time(display_states, current_time)

            # Render the frame
            cursor_pos = (state.get("cursor_row", 0), state.get("cursor_col", 0))
            show_cursor = (frame_idx % 30) < 15  # Blink every 0.5 seconds

            img = self.renderer.render_frame(
                lines=state["lines"],
                cursor_pos=cursor_pos,
                show_cursor=show_cursor
            )

            frames.append(img)

        return frames

    def _build_display_states(
        self,
        command_outputs: List[dict],
        duration: float
    ) -> List[dict]:
        """Build a sequence of display states with typing animation."""
        states = []
        lines: List[List[StyledChar]] = [
            [StyledChar(" ") for _ in range(100)]
            for _ in range(30)
        ]
        current_row = 0
        current_time = 0

        # Calculate time per command
        time_per_command = duration / max(len(command_outputs), 1)

        for cmd_data in command_outputs:
            cmd = cmd_data["command"]
            output = cmd_data.get("output", "")
            is_comment = cmd_data.get("is_comment", False)

            # Draw prompt
            prompt = "\u276f "  # Arrow prompt
            prompt_color = COLORS["prompt"]

            col = 0
            for char in "~ ":
                lines[current_row][col] = StyledChar(char, fg_color=COLORS["prompt_dir"])
                col += 1

            for char in prompt:
                lines[current_row][col] = StyledChar(char, fg_color=COLORS["prompt_arrow"])
                col += 1

            # Save state before typing
            states.append({
                "time": current_time,
                "lines": [list(row) for row in lines],
                "cursor_row": current_row,
                "cursor_col": col
            })

            # Type command character by character
            cmd_color = COLORS["comment"] if is_comment else COLORS["foreground"]
            typing_time = time_per_command * 0.5

            for i, char in enumerate(cmd):
                lines[current_row][col] = StyledChar(char, fg_color=cmd_color)
                col += 1

                # Add state for typing animation
                char_time = current_time + (i + 1) * (typing_time / max(len(cmd), 1))
                states.append({
                    "time": char_time,
                    "lines": [list(row) for row in lines],
                    "cursor_row": current_row,
                    "cursor_col": col
                })

            current_time += typing_time

            # Show output if any
            if output and not is_comment:
                current_row = min(current_row + 1, 29)

                # Parse and display output
                output_color = COLORS["error"] if cmd_data.get("exit_code", 0) != 0 else COLORS["foreground"]

                for line in output.split('\n')[:10]:  # Limit output lines
                    col = 0
                    for char in line[:95]:
                        lines[current_row][col] = StyledChar(char, fg_color=output_color)
                        col += 1
                    current_row = min(current_row + 1, 29)

                current_time += time_per_command * 0.3

            # Move to next line for next command
            current_row = min(current_row + 1, 29)

            states.append({
                "time": current_time,
                "lines": [list(row) for row in lines],
                "cursor_row": current_row,
                "cursor_col": 0
            })

            current_time += time_per_command * 0.2

        # Final state
        states.append({
            "time": duration,
            "lines": [list(row) for row in lines],
            "cursor_row": current_row,
            "cursor_col": 0
        })

        return states

    def _get_state_at_time(self, states: List[dict], time: float) -> dict:
        """Get the display state at a given time."""
        if not states:
            return {
                "lines": [[StyledChar(" ")] * 100 for _ in range(30)],
                "cursor_row": 0,
                "cursor_col": 0
            }

        # Find the latest state before or at the given time
        for state in reversed(states):
            if state["time"] <= time:
                return state

        return states[0]

    def _create_video(
        self,
        frames: List[Image.Image],
        output_path: str,
        duration: float
    ):
        """Create video from PIL Image frames."""
        frames_dir = Path(output_path).parent / f"{Path(output_path).stem}_frames"
        frames_dir.mkdir(exist_ok=True)

        print(f"  Saving {len(frames)} frame images...")

        # Save frames
        for i, img in enumerate(frames):
            frame_path = frames_dir / f"frame_{i:06d}.png"
            self.renderer.save_frame(img, str(frame_path))

        # Calculate FPS
        fps = max(1, min(30, len(frames) / duration))

        print(f"  Creating video with FFmpeg at {fps} FPS...")

        # Create video
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={self.width}:{self.height}",
            "-movflags", "+faststart",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr[-500:]}")

        # Cleanup frames
        shutil.rmtree(frames_dir, ignore_errors=True)

        print(f"  Video created: {output_path}")


def record_true_terminal(
    project_id: str,
    scene_index: int,
    commands: List[str],
    duration: float
) -> TrueRecordingResult:
    """
    Record a truly authentic terminal session.

    This is the main entry point. Commands are executed for real,
    output is captured, and video is created from actual images.
    """
    recorder = TrueTerminalRecorder()
    return recorder.record_session(project_id, scene_index, commands, duration)
