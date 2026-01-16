"""
Script generation module using Claude.
Generates timed narration scripts that sync with recorded video.
"""
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
import anthropic

from config.settings import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_SCRIPT_TOKENS,
    TEMPLATES_DIR, RECORDINGS_DIR
)


@dataclass
class ScriptSegment:
    """A timed segment of narration."""
    start_time: float
    end_time: float
    text: str
    emphasis: str = "normal"  # normal, excited, thoughtful


@dataclass
class GeneratedScript:
    """Complete generated script with metadata."""
    project_id: str
    title: str
    hook: str
    segments: list[ScriptSegment]
    outro: str
    total_duration: float
    word_count: int


class ScriptGenerator:
    """Generates narration scripts from video timestamps."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.intro_template = self._load_template("script_intro.txt")
        self.body_template = self._load_template("script_body.txt")

    def _load_template(self, filename: str) -> str:
        """Load a prompt template."""
        template_path = TEMPLATES_DIR / filename
        if template_path.exists():
            return template_path.read_text()
        return ""

    def generate_from_timestamps(self, timestamps_file: str,
                                  topic: Optional[str] = None) -> GeneratedScript:
        """
        Generate a narration script from recording timestamps.

        Args:
            timestamps_file: Path to timestamps JSON from recording
            topic: Optional topic override

        Returns:
            GeneratedScript with timed segments
        """
        # Load timestamps
        with open(timestamps_file) as f:
            timestamps_data = json.load(f)

        project_id = timestamps_data["project_id"]
        url = timestamps_data["url"]
        duration = timestamps_data["duration_seconds"]
        actions = timestamps_data["actions"]

        # Build context for Claude
        actions_context = self._format_actions(actions)

        # Generate script via Claude
        prompt = self._build_script_prompt(
            url=url,
            duration=duration,
            actions=actions_context,
            topic=topic
        )

        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_SCRIPT_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse Claude's response into segments
        script_data = self._parse_script_response(
            response.content[0].text,
            project_id,
            duration
        )

        # Save script
        script_path = RECORDINGS_DIR / f"{project_id}_script.json"
        with open(script_path, "w") as f:
            json.dump(asdict(script_data), f, indent=2)

        return script_data

    def _format_actions(self, actions: list[dict]) -> str:
        """Format actions list for prompt context."""
        lines = []
        for action in actions:
            time_str = f"[{action['time_seconds']:.1f}s]"
            desc = action['description']
            if action.get('element_text'):
                desc += f" ({action['element_text'][:50]})"
            lines.append(f"{time_str} {action['action_type']}: {desc}")
        return "\n".join(lines)

    def _build_script_prompt(self, url: str, duration: float,
                              actions: str, topic: Optional[str]) -> str:
        """Build the prompt for script generation."""
        topic_context = f"Topic: {topic}\n" if topic else ""

        return f"""You are a YouTube script writer creating engaging, persuasive narration for a tutorial video.

{topic_context}URL being demonstrated: {url}
Total video duration: {duration:.1f} seconds

RECORDED ACTIONS (with timestamps):
{actions}

Generate a narration script that:
1. Opens with a compelling hook (first 5-10 seconds)
2. Matches the timing of each action shown
3. Uses conversational, authoritative tone
4. Includes natural pauses for visual comprehension
5. Ends with a clear call-to-action

OUTPUT FORMAT (JSON):
{{
    "title": "Video title for YouTube",
    "hook": "Opening hook text (5-10 seconds of speech)",
    "segments": [
        {{
            "start_time": 0.0,
            "end_time": 5.0,
            "text": "Narration text for this segment",
            "emphasis": "excited|normal|thoughtful"
        }}
    ],
    "outro": "Closing call-to-action text"
}}

IMPORTANT:
- Each segment should be 5-15 seconds of speech
- Leave brief pauses (1-2s) during complex visual actions
- Total speech time should be ~80% of video duration
- Be specific about what's shown, not generic

Generate the JSON script now:"""

    def _parse_script_response(self, response: str, project_id: str,
                                duration: float) -> GeneratedScript:
        """Parse Claude's response into a GeneratedScript."""
        # Extract JSON from response
        try:
            # Find JSON block in response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            # Fallback: create basic script
            return self._create_fallback_script(project_id, duration)

        # Convert to dataclass
        segments = [
            ScriptSegment(
                start_time=seg["start_time"],
                end_time=seg["end_time"],
                text=seg["text"],
                emphasis=seg.get("emphasis", "normal")
            )
            for seg in data.get("segments", [])
        ]

        # Calculate word count
        all_text = data.get("hook", "") + " ".join(s.text for s in segments) + data.get("outro", "")
        word_count = len(all_text.split())

        return GeneratedScript(
            project_id=project_id,
            title=data.get("title", f"Tutorial: {project_id}"),
            hook=data.get("hook", ""),
            segments=segments,
            outro=data.get("outro", ""),
            total_duration=duration,
            word_count=word_count
        )

    def _create_fallback_script(self, project_id: str, duration: float) -> GeneratedScript:
        """Create a basic fallback script if parsing fails."""
        segment_duration = min(15, duration / 3)

        return GeneratedScript(
            project_id=project_id,
            title=f"Tutorial: {project_id}",
            hook="Welcome to this quick tutorial. Let me show you something useful.",
            segments=[
                ScriptSegment(0, segment_duration,
                             "Here's what we're looking at today.", "normal"),
                ScriptSegment(segment_duration, segment_duration * 2,
                             "Watch how this works in practice.", "normal"),
                ScriptSegment(segment_duration * 2, duration,
                             "And that's the key takeaway.", "normal")
            ],
            outro="If this helped, subscribe for more tutorials like this.",
            total_duration=duration,
            word_count=50
        )


def generate_script(timestamps_file: str, topic: Optional[str] = None) -> dict:
    """
    Convenience function to generate a script.

    Args:
        timestamps_file: Path to timestamps JSON
        topic: Optional topic description

    Returns:
        Script data as dict
    """
    generator = ScriptGenerator()
    script = generator.generate_from_timestamps(timestamps_file, topic)
    return asdict(script)
