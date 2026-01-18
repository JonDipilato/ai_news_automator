"""
Script generation module using Claude.
Generates timed narration scripts that sync with recorded video.

Supports two modes:
- tutorial: Full educational content with learning objectives, concept explanations, and summaries
- howto: Quick practical content with concise step-by-step instructions
"""
import json
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, asdict
import anthropic

from config.settings import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_SCRIPT_TOKENS,
    TEMPLATES_DIR, RECORDINGS_DIR
)

# Script generation modes
ScriptMode = Literal["tutorial", "howto"]


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

    def __init__(self, mode: ScriptMode = "tutorial"):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.mode = mode
        self.intro_template = self._load_template("script_intro.txt")
        self.body_template = self._load_template("script_body.txt")

        # Load mode-specific templates
        self.tutorial_intro = self._load_template("tutorial_intro.txt")
        self.tutorial_segment = self._load_template("tutorial_segment.txt")
        self.tutorial_summary = self._load_template("tutorial_summary.txt")
        self.howto_intro = self._load_template("howto_intro.txt")
        self.howto_segment = self._load_template("howto_segment.txt")

    def _load_template(self, filename: str) -> str:
        """Load a prompt template."""
        template_path = TEMPLATES_DIR / filename
        if template_path.exists():
            return template_path.read_text()
        return ""

    def generate_from_timestamps(
        self,
        timestamps_file: str,
        topic: Optional[str] = None,
        mode: Optional[ScriptMode] = None,
        educational_context: Optional[dict] = None
    ) -> GeneratedScript:
        """
        Generate a narration script from recording timestamps.

        Args:
            timestamps_file: Path to timestamps JSON from recording
            topic: Optional topic override
            mode: Script generation mode ("tutorial" or "howto")
            educational_context: Optional dict with learning objectives, prerequisites, etc.

        Returns:
            GeneratedScript with timed segments
        """
        # Use provided mode or default
        script_mode = mode or self.mode

        # Load timestamps
        with open(timestamps_file) as f:
            timestamps_data = json.load(f)

        project_id = timestamps_data["project_id"]
        url = timestamps_data["url"]
        duration = timestamps_data["duration_seconds"]
        actions = timestamps_data["actions"]

        # Build context for Claude
        actions_context = self._format_actions(actions)

        # Generate script via Claude based on mode
        if script_mode == "tutorial":
            prompt = self._build_tutorial_prompt(
                url=url,
                duration=duration,
                actions=actions_context,
                topic=topic,
                educational_context=educational_context
            )
        else:  # howto
            prompt = self._build_howto_prompt(
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

        # Calculate timing for hook (first 5-8 seconds spoken during first action)
        hook_end_time = 8.0

        return f"""You're writing a YouTube tutorial script. Your job is to sound like a real person talking to a friend, not a robot or corporate presenter.

{topic_context}URL being demonstrated: {url}
Total video duration: {duration:.1f} seconds

RECORDED ACTIONS (with timestamps):
{actions}

VOICE & STYLE RULES:

1. SOUND HUMAN - Write like you talk:
   - Use contractions: "you're", "it's", "don't", "gonna", "wanna"
   - Start sentences with "So", "Now", "Okay", "Alright", "Look"
   - Use filler phrases naturally: "honestly", "basically", "pretty much", "real quick"
   - React genuinely: "this is sick", "super clean", "lowkey amazing", "no cap this works"

2. BANNED AI PHRASES - Never use these:
   - "Let's dive in" / "dive into"
   - "In this video"
   - "Welcome back"
   - "Without further ado"
   - "It's important to note"
   - "As you can see"
   - "Simply" / "Just simply"
   - "Leverage" / "Utilize"
   - "Robust" / "Seamless" / "Cutting-edge"
   - "Game-changer" / "Revolutionary"
   - "Take it to the next level"
   - "Here's the thing"
   - Any phrase that sounds like a LinkedIn post

3. HOOK FORMULA (first 5-8 seconds):
   - Start mid-thought like you're already talking
   - Create curiosity or call out a pain point
   - THE HOOK IS THE OPENING - segments come AFTER the hook ends
   - Examples:
     * "Okay so I just found something that's gonna save you hours..."
     * "Bro. Why did nobody tell me about this earlier?"
     * "Stop doing it the hard way. There's a better move."
     * "You know that annoying thing where [problem]? Fixed."

4. BODY STYLE (segments):
   - CRITICAL: First segment starts AFTER the hook (at {hook_end_time} seconds, not 0)
   - DO NOT repeat anything from the hook in the first segment
   - The hook ends, then the first segment continues the flow naturally
   - Short punchy sentences. Like this. Keep it moving.
   - When showing a step: "Now watch this" or "Check this out" or "Here's the move"
   - When something works: "Boom." or "There it is." or "See that?"
   - Explain WHY not just what: "This matters because..."
   - Add personality: "I actually use this daily" or "This one's clutch"

5. OUTRO (call-to-action) - MUST BE ENGAGING:
   - The outro is spoken at the END of the video, don't cut it short
   - Include a clear call-to-action (subscribe, like, comment)
   - Tease future content to keep them interested
   - REQUIRED ELEMENTS:
     * Wrap up what they learned (1 sentence)
     * Call to action (subscribe/like)
     * Future content tease
   - Examples:
     * "So now you know how to [thing]. If you want more stuff like this, hit subscribe - I drop new tutorials every week. Next up I'm covering [related topic]. Later."
     * "That's the whole setup. Took me way too long to figure this out, so hopefully I saved you some time. Sub if you want more. Got a video on [topic] coming next. Peace."
     * "And that's it - you're set up. Drop a comment if you have questions. Subscribe for more. I've got something even crazier coming next week. Stay tuned."

OUTPUT FORMAT (JSON):
{{
    "title": "Catchy YouTube title (use numbers, 'how to', or curiosity gaps)",
    "hook": "Opening hook - mid-conversation energy, spoken during seconds 0-{hook_end_time}",
    "segments": [
        {{
            "start_time": {hook_end_time},
            "end_time": 15.0,
            "text": "First segment - continues AFTER hook, NO repetition",
            "emphasis": "excited|normal|thoughtful"
        }},
        {{
            "start_time": 15.0,
            "end_time": 25.0,
            "text": "Next segment...",
            "emphasis": "normal"
        }}
    ],
    "outro": "Full engaging outro with CTA, subscribe mention, and future content tease (spoken during final 10-15 seconds)"
}}

TIMING RULES:
- Hook: 0 to {hook_end_time} seconds
- First segment: starts at {hook_end_time} seconds (NOT at 0!)
- Each segment: 5-15 seconds of speech
- Leave 1-2s pauses during complex visuals (let them breathe)
- Outro: final 10-15 seconds of the video
- Total speech: ~80% of video duration
- Match energy to what's happening on screen

CRITICAL CHECKLIST BEFORE GENERATING:
[ ] Hook starts at 0, ends around {hook_end_time}s
[ ] First segment starts at {hook_end_time}s, NOT at 0
[ ] First segment does NOT repeat anything from the hook
[ ] Outro includes: summary + subscribe CTA + future tease
[ ] No banned AI phrases used

Generate the JSON script. Sound like a real YouTuber, not a tutorial bot:"""

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
            title=f"You Need to See This | Quick Tutorial",
            hook="Okay so check this out real quick. Found something you're gonna wanna know about.",
            segments=[
                ScriptSegment(0, segment_duration,
                             "Alright so here's what we're working with.", "normal"),
                ScriptSegment(segment_duration, segment_duration * 2,
                             "Now watch this part. This is the move right here.", "normal"),
                ScriptSegment(segment_duration * 2, duration,
                             "And boom. That's pretty much it. Super clean.", "normal")
            ],
            outro="If this helped, sub's free. Got more stuff like this coming.",
            total_duration=duration,
            word_count=60
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
