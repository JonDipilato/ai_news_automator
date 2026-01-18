"""
Multi-scene video assembler with FFmpeg transitions.
Combines multiple scene recordings with professional transitions.
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
    RECORDINGS_DIR, AUDIO_DIR, CAPTIONS_DIR, REVIEW_DIR
)
from .scenes import TransitionType, SceneManifest, load_manifest


@dataclass
class SceneAssemblyResult:
    """Result of multi-scene assembly."""
    project_id: str
    output_file: str
    duration: float
    scene_count: int
    has_captions: bool
    has_transitions: bool
    success: bool
    message: str


class SceneAssembler:
    """Assembles multi-scene videos with transitions."""

    def __init__(self):
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Verify FFmpeg is available."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")

    def assemble_from_manifest(self, project_id: str,
                                include_captions: bool = True) -> SceneAssemblyResult:
        """
        Assemble video from recorded scenes.

        Args:
            project_id: Project identifier
            include_captions: Whether to burn in captions

        Returns:
            SceneAssemblyResult
        """
        scenes_file = RECORDINGS_DIR / f"{project_id}_scenes.json"
        audio_file = AUDIO_DIR / f"{project_id}.mp3"
        captions_file = CAPTIONS_DIR / f"{project_id}.srt"
        output_file = REVIEW_DIR / f"{project_id}.mp4"

        if not scenes_file.exists():
            return SceneAssemblyResult(
                project_id=project_id,
                output_file="",
                duration=0,
                scene_count=0,
                has_captions=False,
                has_transitions=False,
                success=False,
                message=f"Scenes file not found: {scenes_file}"
            )

        with open(scenes_file) as f:
            scenes_data = json.load(f)

        scenes = scenes_data.get("scenes", [])
        manifest_data = scenes_data.get("manifest", {})

        if not scenes:
            return SceneAssemblyResult(
                project_id=project_id,
                output_file="",
                duration=0,
                scene_count=0,
                has_captions=False,
                has_transitions=False,
                success=False,
                message="No scenes found in scenes file"
            )

        # Step 1: Concatenate all scene videos with transitions
        concat_file = RECORDINGS_DIR / f"{project_id}_concat.mp4"
        has_transitions = self._concatenate_with_transitions(
            scenes, manifest_data, concat_file
        )

        if not concat_file.exists():
            return SceneAssemblyResult(
                project_id=project_id,
                output_file="",
                duration=0,
                scene_count=len(scenes),
                has_captions=False,
                has_transitions=False,
                success=False,
                message="Failed to concatenate scenes"
            )

        # Get video duration
        video_duration = self._get_duration(concat_file)

        # Step 2: Add audio
        audio_duration = 0
        if audio_file.exists():
            audio_duration = self._get_duration(audio_file)

        # Step 3: Combine video + audio + captions
        # CRITICAL FIX: Don't use -shortest, instead extend video if audio is longer
        success = self._final_assembly(
            video_file=concat_file,
            audio_file=audio_file if audio_file.exists() else None,
            captions_file=captions_file if include_captions and captions_file.exists() else None,
            output_file=output_file,
            video_duration=video_duration,
            audio_duration=audio_duration
        )

        # Cleanup temp file
        concat_file.unlink(missing_ok=True)

        if not success:
            return SceneAssemblyResult(
                project_id=project_id,
                output_file="",
                duration=0,
                scene_count=len(scenes),
                has_captions=False,
                has_transitions=has_transitions,
                success=False,
                message="Failed to assemble final video"
            )

        final_duration = self._get_duration(output_file)

        # Create metadata
        self._create_metadata(project_id, output_file, final_duration, scenes_data)

        return SceneAssemblyResult(
            project_id=project_id,
            output_file=str(output_file),
            duration=final_duration,
            scene_count=len(scenes),
            has_captions=include_captions and captions_file.exists(),
            has_transitions=has_transitions,
            success=True,
            message="Video assembled successfully"
        )

    def _concatenate_with_transitions(self, scenes: list, manifest_data: dict,
                                       output: Path) -> bool:
        """Concatenate scene videos with transitions between them."""
        # Get valid video files
        video_files = []
        transitions = []

        for scene in scenes:
            if scene.get("success") and scene.get("video_path"):
                video_path = Path(scene["video_path"])
                if video_path.exists():
                    video_files.append(video_path)

                    # Get transition for this scene from manifest
                    scene_idx = scene.get("scene_index", 0)
                    manifest_scenes = manifest_data.get("scenes", [])
                    if scene_idx < len(manifest_scenes):
                        trans = manifest_scenes[scene_idx].get("transition_in", "fade")
                        trans_dur = manifest_scenes[scene_idx].get("transition_duration", 0.5)
                        transitions.append((trans, trans_dur))
                    else:
                        transitions.append(("fade", 0.5))

        if not video_files:
            return False

        if len(video_files) == 1:
            # Single scene, just copy/convert
            self._convert_to_mp4(video_files[0], output)
            return False

        # Build complex filter for transitions
        filter_complex = self._build_transition_filter(video_files, transitions)

        if filter_complex:
            return self._apply_transition_filter(video_files, filter_complex, output)
        else:
            # Fallback: simple concatenation
            return self._simple_concat(video_files, output)

    def _build_transition_filter(self, video_files: list[Path],
                                  transitions: list[tuple]) -> Optional[str]:
        """Build FFmpeg filter_complex for transitions."""
        n = len(video_files)
        if n < 2:
            return None

        filter_parts = []
        current_output = "[0:v]"

        # Normalize all inputs first
        for i in range(n):
            filter_parts.append(
                f"[{i}:v]scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,"
                f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
                f"fps={VIDEO_FPS},format=yuv420p[v{i}]"
            )

        # Apply transitions between consecutive clips
        for i in range(1, n):
            trans_type, trans_dur = transitions[i] if i < len(transitions) else ("fade", 0.5)
            trans_filter = self._get_transition_filter(trans_type, trans_dur)

            if i == 1:
                input1 = f"[v0]"
                input2 = f"[v{i}]"
            else:
                input1 = f"[trans{i-1}]"
                input2 = f"[v{i}]"

            output_label = f"[trans{i}]" if i < n - 1 else "[outv]"

            # xfade transition
            filter_parts.append(
                f"{input1}{input2}xfade=transition={trans_filter}:duration={trans_dur}:offset=0{output_label}"
            )

        return ";".join(filter_parts)

    def _get_transition_filter(self, trans_type: str, duration: float) -> str:
        """Convert our transition type to FFmpeg xfade transition name."""
        mapping = {
            "cut": "fade",  # FFmpeg doesn't have "cut", use very short fade
            "fade": "fade",
            "crossfade": "fade",
            "wipe_left": "wipeleft",
            "wipe_right": "wiperight",
            "wipe_up": "wipeup",
            "wipe_down": "wipedown",
            "zoom_in": "zoomin",
            "zoom_out": "fadeblack",  # No direct zoom out, use fadeblack
            "slide_left": "slideleft",
            "slide_right": "slideright",
            "glitch": "pixelize",  # Closest to glitch effect
        }
        return mapping.get(trans_type, "fade")

    def _apply_transition_filter(self, video_files: list[Path],
                                  filter_complex: str, output: Path) -> bool:
        """Apply the transition filter to create output."""
        # Build input arguments
        inputs = []
        for vf in video_files:
            inputs.extend(["-i", str(vf)])

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", VIDEO_CODEC,
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(output)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return output.exists()
        except Exception:
            # Fallback to simple concat on error
            return self._simple_concat(video_files, output)

    def _simple_concat(self, video_files: list[Path], output: Path) -> bool:
        """Simple concatenation without transitions."""
        # Create file list
        list_file = RECORDINGS_DIR / "_concat_list.txt"
        with open(list_file, "w") as f:
            for vf in video_files:
                # Need to normalize each video first
                f.write(f"file '{vf}'\n")

        # First normalize all videos to same format
        normalized = []
        for i, vf in enumerate(video_files):
            norm_file = RECORDINGS_DIR / f"_norm_{i}.mp4"
            self._normalize_video(vf, norm_file)
            normalized.append(norm_file)

        # Create new list with normalized files
        with open(list_file, "w") as f:
            for nf in normalized:
                f.write(f"file '{nf}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", VIDEO_CODEC,
            "-c:a", "copy",
            "-preset", "medium",
            "-crf", "23",
            str(output)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            success = output.exists()
        except Exception:
            success = False

        # Cleanup
        list_file.unlink(missing_ok=True)
        for nf in normalized:
            nf.unlink(missing_ok=True)

        return success

    def _normalize_video(self, input_file: Path, output_file: Path):
        """Normalize video to standard format for concatenation."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-vf", f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
                   f"fps={VIDEO_FPS},format=yuv420p",
            "-c:v", VIDEO_CODEC,
            "-preset", "fast",
            "-crf", "23",
            "-an",  # Remove audio for concat
            str(output_file)
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)

    def _convert_to_mp4(self, input_file: Path, output_file: Path):
        """Convert single video to MP4 format."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-vf", f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
                   f"fps={VIDEO_FPS}",
            "-c:v", VIDEO_CODEC,
            "-preset", "medium",
            "-crf", "23",
            "-an",
            str(output_file)
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)

    def _final_assembly(self, video_file: Path, audio_file: Optional[Path],
                        captions_file: Optional[Path], output_file: Path,
                        video_duration: float, audio_duration: float) -> bool:
        """
        Final assembly: video + audio + captions.
        CRITICAL: Extends video if audio is longer (for outro).
        """
        # Determine target duration - use the LONGER of video or audio
        # This ensures the outro doesn't get cut off
        target_duration = max(video_duration, audio_duration)

        # If audio is longer, we need to extend the video (freeze last frame)
        need_extension = audio_duration > video_duration + 0.5  # 0.5s buffer

        # Build filter chain
        vf_filters = []

        # Extend video if needed by looping/freezing last frame
        if need_extension:
            extension_time = audio_duration - video_duration + 1  # Extra second for safety
            # Use tpad to extend with last frame
            vf_filters.append(f"tpad=stop_mode=clone:stop_duration={extension_time}")

        # Add captions if provided
        if captions_file:
            caption_path = str(captions_file).replace("\\", "/").replace(":", "\\:")
            subtitle_filter = (
                f"subtitles='{caption_path}':"
                f"force_style='FontName=Arial Black,"
                f"FontSize=28,"
                f"PrimaryColour=&H00FFFFFF,"
                f"OutlineColour=&H00000000,"
                f"BackColour=&H80000000,"
                f"Bold=1,"
                f"Outline=3,"
                f"Shadow=2,"
                f"MarginV=60,"
                f"Alignment=2'"
            )
            vf_filters.append(subtitle_filter)

        # Add fade in/out
        vf_filters.insert(0, "fade=t=in:st=0:d=0.5")
        if target_duration > 1:
            vf_filters.append(f"fade=t=out:st={target_duration - 1}:d=1")

        # Build command
        cmd = ["ffmpeg", "-y", "-i", str(video_file)]

        if audio_file:
            cmd.extend(["-i", str(audio_file)])

        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])

        cmd.extend([
            "-c:v", VIDEO_CODEC,
            "-c:a", AUDIO_CODEC if audio_file else "copy",
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-preset", "medium",
            "-crf", "23",
        ])

        # Map streams
        cmd.extend(["-map", "0:v"])
        if audio_file:
            cmd.extend(["-map", "1:a"])

        # Set duration to match audio (ensures outro plays fully)
        if audio_file and audio_duration > 0:
            cmd.extend(["-t", str(audio_duration + 0.5)])  # Small buffer

        cmd.append(str(output_file))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return output_file.exists()
        except Exception as e:
            print(f"Assembly error: {e}")
            return False

    def _get_duration(self, file_path: Path) -> float:
        """Get media file duration in seconds."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception:
            return 0

    def _create_metadata(self, project_id: str, video_path: Path,
                         duration: float, scenes_data: dict):
        """Create metadata file for review/publishing."""
        manifest = scenes_data.get("manifest", {})
        title = manifest.get("title", f"Tutorial: {project_id}")
        topic = manifest.get("topic", "")

        # Load script if exists
        script_file = RECORDINGS_DIR / f"{project_id}_script.json"
        hook = ""
        outro = ""
        if script_file.exists():
            with open(script_file) as f:
                script_data = json.load(f)
                title = script_data.get("title", title)
                hook = script_data.get("hook", "")
                outro = script_data.get("outro", "")

        # Build description
        description_parts = [
            hook,
            "",
            f"Topic: {topic}" if topic else "",
            "",
            outro,
            "",
            "---",
            "#AI #Tutorial #Technology #Automation"
        ]
        description = "\n".join(filter(None, description_parts))

        meta = {
            "project_id": project_id,
            "video_file": str(video_path),
            "thumbnail_file": str(REVIEW_DIR / f"{project_id}_thumb.jpg"),
            "duration_seconds": duration,
            "duration_formatted": self._format_duration(duration),
            "title": title,
            "description": description,
            "scene_count": len(scenes_data.get("scenes", [])),
            "created_at": datetime.now().isoformat(),
            "status": "pending_review",
            "approved": False,
            "youtube_category": "28",
            "youtube_privacy": "private"
        }

        meta_file = REVIEW_DIR / f"{project_id}_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

    def _format_duration(self, seconds: float) -> str:
        """Format duration as MM:SS or HH:MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        if mins >= 60:
            hours = mins // 60
            mins = mins % 60
            return f"{hours}:{mins:02d}:{secs:02d}"
        return f"{mins}:{secs:02d}"


def assemble_scenes(project_id: str, with_captions: bool = True) -> dict:
    """
    Convenience function to assemble multi-scene video.

    Args:
        project_id: Project identifier
        with_captions: Include captions

    Returns:
        Assembly result as dict
    """
    from .captioner import CaptionGenerator
    from .thumbnail import ThumbnailGenerator

    assembler = SceneAssembler()

    # Generate captions
    if with_captions:
        try:
            captioner = CaptionGenerator(words_per_chunk=3)
            captioner.generate(project_id)
        except Exception:
            pass  # Continue without captions

    # Assemble
    result = assembler.assemble_from_manifest(project_id, include_captions=with_captions)

    # Generate thumbnail
    thumbnail_path = ""
    if result.success:
        try:
            thumb_gen = ThumbnailGenerator()
            thumb_result = thumb_gen.generate(project_id)
            if thumb_result.success:
                thumbnail_path = thumb_result.thumbnail_path
        except Exception:
            pass

    return {
        "project_id": result.project_id,
        "output_file": result.output_file,
        "thumbnail_file": thumbnail_path,
        "duration": result.duration,
        "scene_count": result.scene_count,
        "has_captions": result.has_captions,
        "has_transitions": result.has_transitions,
        "success": result.success,
        "message": result.message
    }
