"""
ScreenCaptureRecorder - Real screen capture using FFmpeg.

Captures actual terminal windows with real mouse movement and typing.
This is the most authentic recording method for environments with a display.

Supported platforms:
- Linux (X11): FFmpeg x11grab
- Windows: FFmpeg gdigrab
- macOS: FFmpeg avfoundation
"""
import os
import sys
import time
import json
import subprocess
import shutil
import platform
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from config.settings import VIDEO_WIDTH, VIDEO_HEIGHT, RECORDINGS_DIR

# Optional pyautogui for keyboard automation
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False


@dataclass
class ScreenCaptureResult:
    """Result of a screen capture recording."""
    scene_index: int
    scene_type: str
    video_path: str
    duration: float
    timestamps: List[dict]
    success: bool
    error: Optional[str] = None


class ScreenCaptureRecorder:
    """
    Records real screen capture of terminal sessions.

    Opens an actual terminal window, types commands via pyautogui,
    and captures the screen region with FFmpeg.

    This provides the most authentic recordings with real cursor movement,
    scrolling, and visual effects.
    """

    def __init__(self, width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT):
        self.width = width
        self.height = height
        self.recordings_dir = Path(RECORDINGS_DIR)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.platform = self._detect_platform()

    def _detect_platform(self) -> str:
        """Detect the current platform for screen capture."""
        system = platform.system().lower()

        if system == "linux":
            # Check if we have a display
            display = os.environ.get("DISPLAY")
            if display:
                return "x11"
            # Check for Wayland
            wayland = os.environ.get("WAYLAND_DISPLAY")
            if wayland:
                return "wayland"
            return "headless"

        elif system == "darwin":
            return "macos"

        elif system == "windows":
            return "windows"

        return "unknown"

    def is_available(self) -> Tuple[bool, str]:
        """Check if screen capture is available on this system."""
        if self.platform == "headless":
            return False, "No display available (headless environment)"

        if self.platform == "wayland":
            return False, "Wayland not yet supported for screen capture"

        if self.platform == "unknown":
            return False, f"Unknown platform: {platform.system()}"

        # Check FFmpeg availability
        if not shutil.which("ffmpeg"):
            return False, "FFmpeg not installed"

        # Check pyautogui for keyboard automation
        if not PYAUTOGUI_AVAILABLE:
            return False, "pyautogui not installed (pip install pyautogui)"

        return True, "Screen capture available"

    def record_session(
        self,
        project_id: str,
        scene_index: int,
        commands: List[str],
        duration: float,
        terminal_app: Optional[str] = None
    ) -> ScreenCaptureResult:
        """
        Record a terminal session by opening a real terminal window
        and capturing the screen.

        Args:
            project_id: Project identifier
            scene_index: Scene index for file naming
            commands: List of commands to execute
            duration: Target duration in seconds
            terminal_app: Optional specific terminal application to use

        Returns:
            ScreenCaptureResult with recording details
        """
        video_path = self.recordings_dir / f"{project_id}_scene{scene_index:03d}.mp4"
        timestamps = []
        start_time = time.time()

        # Check availability
        available, reason = self.is_available()
        if not available:
            return ScreenCaptureResult(
                scene_index=scene_index,
                scene_type="terminal",
                video_path="",
                duration=0,
                timestamps=[],
                success=False,
                error=f"Screen capture not available: {reason}"
            )

        try:
            # Step 1: Open terminal window
            terminal_proc, window_geometry = self._open_terminal_window(terminal_app)
            if not terminal_proc:
                raise RuntimeError("Failed to open terminal window")

            time.sleep(1.5)  # Wait for terminal to be ready

            # Step 2: Start FFmpeg screen capture
            ffmpeg_proc = self._start_screen_capture(
                str(video_path),
                window_geometry,
                duration + 5  # Extra buffer time
            )

            time.sleep(0.5)  # Let capture stabilize

            # Step 3: Type commands with realistic timing
            for i, cmd in enumerate(commands):
                cmd_start = time.time() - start_time

                # Type the command
                self._type_command(cmd)

                timestamps.append({
                    "time_seconds": round(cmd_start, 2),
                    "action_type": "command_executed",
                    "description": f"Executed: {cmd[:50]}",
                })

                # Wait for command to complete
                time.sleep(0.5)

            # Step 4: Hold for remaining duration
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            if remaining > 0:
                time.sleep(remaining)

            # Step 5: Stop capture and cleanup
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=5)

            terminal_proc.terminate()

            return ScreenCaptureResult(
                scene_index=scene_index,
                scene_type="terminal",
                video_path=str(video_path),
                duration=time.time() - start_time,
                timestamps=timestamps,
                success=video_path.exists()
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ScreenCaptureResult(
                scene_index=scene_index,
                scene_type="terminal",
                video_path="",
                duration=0,
                timestamps=[],
                success=False,
                error=str(e)
            )

    def _open_terminal_window(
        self,
        terminal_app: Optional[str] = None
    ) -> Tuple[Optional[subprocess.Popen], dict]:
        """
        Open a terminal window at a specific position and size.

        Returns:
            Tuple of (process, geometry dict with x, y, width, height)
        """
        # Calculate window position (center of screen or specific position)
        x, y = 100, 100
        width, height = self.width, self.height

        geometry = {"x": x, "y": y, "width": width, "height": height}

        if self.platform == "x11":
            return self._open_linux_terminal(terminal_app, geometry), geometry
        elif self.platform == "windows":
            return self._open_windows_terminal(terminal_app, geometry), geometry
        elif self.platform == "macos":
            return self._open_macos_terminal(terminal_app, geometry), geometry

        return None, geometry

    def _open_linux_terminal(
        self,
        terminal_app: Optional[str],
        geometry: dict
    ) -> Optional[subprocess.Popen]:
        """Open a terminal on Linux (X11)."""
        width, height = geometry["width"], geometry["height"]
        x, y = geometry["x"], geometry["y"]

        # Try terminals in order of preference
        terminals = [
            terminal_app,
            "gnome-terminal",
            "konsole",
            "xfce4-terminal",
            "xterm",
        ]

        for term in terminals:
            if term and shutil.which(term):
                try:
                    if term == "gnome-terminal":
                        proc = subprocess.Popen([
                            term,
                            f"--geometry={width}x{height}+{x}+{y}",
                        ])
                    elif term == "konsole":
                        proc = subprocess.Popen([
                            term,
                            "--new-tab",
                        ])
                    elif term == "xterm":
                        proc = subprocess.Popen([
                            term,
                            "-geometry", f"100x30+{x}+{y}",
                            "-fa", "DejaVu Sans Mono",
                            "-fs", "12",
                            "-bg", "#0d1117",
                            "-fg", "#c9d1d9",
                        ])
                    else:
                        proc = subprocess.Popen([term])

                    return proc
                except Exception:
                    continue

        return None

    def _open_windows_terminal(
        self,
        terminal_app: Optional[str],
        geometry: dict
    ) -> Optional[subprocess.Popen]:
        """Open a terminal on Windows."""
        # Try Windows Terminal first, then cmd
        terminals = [
            terminal_app,
            "wt",  # Windows Terminal
            "powershell",
            "cmd",
        ]

        for term in terminals:
            if term and shutil.which(term):
                try:
                    proc = subprocess.Popen([term], creationflags=subprocess.CREATE_NEW_CONSOLE)
                    return proc
                except Exception:
                    continue

        return None

    def _open_macos_terminal(
        self,
        terminal_app: Optional[str],
        geometry: dict
    ) -> Optional[subprocess.Popen]:
        """Open a terminal on macOS."""
        script = '''
        tell application "Terminal"
            activate
            do script ""
        end tell
        '''

        try:
            proc = subprocess.Popen(["osascript", "-e", script])
            return proc
        except Exception:
            return None

    def _start_screen_capture(
        self,
        output_path: str,
        geometry: dict,
        duration: float
    ) -> subprocess.Popen:
        """Start FFmpeg screen capture for the specified region."""
        x, y = geometry["x"], geometry["y"]
        width, height = geometry["width"], geometry["height"]

        if self.platform == "x11":
            cmd = [
                "ffmpeg", "-y",
                "-f", "x11grab",
                "-framerate", "30",
                "-video_size", f"{width}x{height}",
                "-i", f":0.0+{x},{y}",
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                output_path
            ]
        elif self.platform == "windows":
            cmd = [
                "ffmpeg", "-y",
                "-f", "gdigrab",
                "-framerate", "30",
                "-offset_x", str(x),
                "-offset_y", str(y),
                "-video_size", f"{width}x{height}",
                "-i", "desktop",
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                output_path
            ]
        elif self.platform == "macos":
            cmd = [
                "ffmpeg", "-y",
                "-f", "avfoundation",
                "-framerate", "30",
                "-i", "1:",  # Screen capture
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-vf", f"crop={width}:{height}:{x}:{y}",
                output_path
            ]
        else:
            raise RuntimeError(f"Unsupported platform for screen capture: {self.platform}")

        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def _type_command(self, cmd: str, typing_delay: float = 0.04):
        """
        Type a command using pyautogui with realistic delays.

        Args:
            cmd: Command to type
            typing_delay: Base delay between keystrokes
        """
        if not PYAUTOGUI_AVAILABLE:
            return

        import random

        for char in cmd:
            pyautogui.press(char) if len(char) == 1 else pyautogui.hotkey(char)

            # Variable delay for realism
            delay = typing_delay + random.uniform(0, 0.03)

            # Occasional longer pauses
            if char in " ,." and random.random() > 0.7:
                delay += 0.1

            time.sleep(delay)

        # Press Enter to execute
        time.sleep(0.2)
        pyautogui.press("enter")


def record_screen_terminal(
    project_id: str,
    scene_index: int,
    commands: List[str],
    duration: float
) -> ScreenCaptureResult:
    """
    Convenience function to record a terminal session with screen capture.

    This is the most authentic recording method when a display is available.
    """
    recorder = ScreenCaptureRecorder()
    return recorder.record_session(project_id, scene_index, commands, duration)


def is_screen_capture_available() -> Tuple[bool, str]:
    """Check if screen capture recording is available."""
    recorder = ScreenCaptureRecorder()
    return recorder.is_available()
