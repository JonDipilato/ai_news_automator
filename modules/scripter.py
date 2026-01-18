"""
Script generation module using Claude.
Generates timed narration scripts that sync with recorded video.

Supports three modes:
- tutorial: Full educational content with learning objectives, concept explanations, and summaries
- howto: Quick practical content with concise step-by-step instructions
- news: Fast WorldofAI-style breaking news narration tied to demos
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
ScriptMode = Literal["tutorial", "howto", "news"]


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
        allowed_modes = {"tutorial", "howto", "news"}
        normalized_mode = (mode or "tutorial").lower()
        self.mode = normalized_mode if normalized_mode in allowed_modes else "tutorial"
        self.intro_template = self._load_template("script_intro.txt")
        self.body_template = self._load_template("script_body.txt")

        # Load mode-specific templates
        self.tutorial_intro = self._load_template("tutorial_intro.txt")
        self.tutorial_segment = self._load_template("tutorial_segment.txt")
        self.tutorial_summary = self._load_template("tutorial_summary.txt")
        self.howto_intro = self._load_template("howto_intro.txt")
        self.howto_segment = self._load_template("howto_segment.txt")
        self.news_intro = self._load_template("news_intro.txt")
        self.news_segment = self._load_template("news_segment.txt")
        self.news_outro = self._load_template("news_outro.txt")

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
        mode: Script generation mode ("tutorial", "howto", or "news")
            educational_context: Optional dict with learning objectives, prerequisites, etc.

        Returns:
            GeneratedScript with timed segments
        """
        # Use provided mode or default
        script_mode = (mode or self.mode or "tutorial").lower()
        if script_mode not in {"tutorial", "howto", "news"}:
            script_mode = "tutorial"

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
        elif script_mode == "howto":
            prompt = self._build_howto_prompt(
                url=url,
                duration=duration,
                actions=actions_context,
                topic=topic,
                educational_context=educational_context
            )
        else:  # news
            prompt = self._build_news_prompt(
                url=url,
                duration=duration,
                actions=actions_context,
                topic=topic,
                educational_context=educational_context
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
        if not actions:
            return "No actions recorded."

        lines = []
        for action in actions:
            time_str = f"[{action['time_seconds']:.1f}s]"
            desc = action['description']
            if action.get('element_text'):
                desc += f" ({action['element_text'][:50]})"

            scene_bits = []
            scene_index = action.get("scene_index")
            if scene_index is not None:
                scene_bits.append(f"scene {int(scene_index) + 1}")
            if action.get("scene_type"):
                scene_bits.append(action["scene_type"])

            scene_label = f" ({' | '.join(scene_bits)})" if scene_bits else ""
            lines.append(f"{time_str}{scene_label} {action['action_type']}: {desc}")
        return "\n".join(lines)

    def _format_educational_context(self, educational_context: Optional[dict]) -> str:
        """Format educational context for the prompt."""
        if not educational_context:
            return "Not provided."

        lines = []

        video = educational_context.get("video") or {}
        if video:
            if video.get("learning_objectives"):
                lines.append(f"Learning objectives: {', '.join(video['learning_objectives'])}")
            if video.get("target_audience"):
                lines.append(f"Target audience: {video['target_audience']}")
            if video.get("difficulty_level"):
                lines.append(f"Difficulty level: {video['difficulty_level']}")
            if video.get("prerequisites"):
                lines.append(f"Prerequisites: {', '.join(video['prerequisites'])}")
            if video.get("related_topics"):
                lines.append(f"Related topics: {', '.join(video['related_topics'])}")
            if video.get("outro_style"):
                lines.append(f"Outro style: {video['outro_style']}")
            if "include_subscribe_cta" in video:
                include_cta = "yes" if video["include_subscribe_cta"] else "no"
                lines.append(f"Include subscribe CTA: {include_cta}")
            if video.get("next_video_tease"):
                lines.append(f"Next video tease: {video['next_video_tease']}")

        scenes = educational_context.get("scenes") or []
        if scenes:
            lines.append("Scene notes:")
            for scene in scenes:
                scene_parts = []
                scene_index = scene.get("index")
                if scene_index is not None:
                    scene_parts.append(f"Scene {int(scene_index) + 1}")
                if scene.get("type"):
                    scene_parts.append(f"({scene['type']})")

                label = " ".join(scene_parts).strip()
                details = []
                if scene.get("description"):
                    details.append(f"description: {scene['description']}")
                if scene.get("learning_objective"):
                    details.append(f"objective: {scene['learning_objective']}")
                if scene.get("concept_explanation"):
                    details.append(f"concept: {scene['concept_explanation']}")
                if scene.get("prerequisites"):
                    details.append(f"prereq: {', '.join(scene['prerequisites'])}")
                if scene.get("common_mistakes"):
                    details.append(f"common mistakes: {', '.join(scene['common_mistakes'])}")
                if scene.get("key_takeaway"):
                    details.append(f"takeaway: {scene['key_takeaway']}")
                if scene.get("tips"):
                    details.append(f"tips: {', '.join(scene['tips'])}")
                if scene.get("narration_hint"):
                    details.append(f"narration hint: {scene['narration_hint']}")

                if label and details:
                    lines.append(f"- {label}: {'; '.join(details)}")
                elif details:
                    lines.append(f"- {'; '.join(details)}")

        if not lines:
            return "Not provided."

        return "\n".join(lines)

    def _build_tutorial_prompt(self, url: str, duration: float, actions: str,
                               topic: Optional[str], educational_context: Optional[dict]) -> str:
        """Build a full tutorial prompt with educational guidance."""
        topic_context = f"Topic: {topic}\n" if topic else ""
        context_block = self._format_educational_context(educational_context)

        guidance_parts = []
        if self.tutorial_intro.strip():
            guidance_parts.append(f"INTRO GUIDANCE:\n{self.tutorial_intro.strip()}")
        if self.tutorial_segment.strip():
            guidance_parts.append(f"SEGMENT GUIDANCE:\n{self.tutorial_segment.strip()}")
        if self.tutorial_summary.strip():
            guidance_parts.append(f"SUMMARY GUIDANCE:\n{self.tutorial_summary.strip()}")
        guidance_text = "\n\n".join(guidance_parts)

        hook_end_time = 8.0

        return f"""You are writing an educational tutorial narration.

{topic_context}URL being demonstrated: {url}
Total video duration: {duration:.1f} seconds

EDUCATIONAL CONTEXT:
{context_block}

RECORDED ACTIONS (with timestamps):
{actions}

{guidance_text}

STYLE RULES:
- Explain why before how.
- Tie narration to what happens on screen.
- Use the educational context when relevant.
- Mention prerequisites and common mistakes only if provided.
- Keep the tone clear, confident, and helpful.

TIMING RULES:
- Hook: 0 to {hook_end_time} seconds.
- First segment starts at {hook_end_time} seconds.
- Each segment: 5 to 15 seconds of speech.
- Outro: final 10 to 15 seconds of the video.
- Total speech: about 80 percent of the video duration.

OUTPUT REQUIREMENTS:
- Return a JSON object with keys: title, hook, segments, outro.
- Each segment has keys: start_time, end_time, text, emphasis.
- Emphasis values: excited, normal, thoughtful.
- Use timestamps that fit within the total duration.
"""

    def _build_howto_prompt(self, url: str, duration: float, actions: str,
                            topic: Optional[str], educational_context: Optional[dict]) -> str:
        """Build a quick how-to prompt with concise structure."""
        topic_context = f"Topic: {topic}\n" if topic else ""
        context_block = self._format_educational_context(educational_context)

        guidance_parts = []
        if self.howto_intro.strip():
            guidance_parts.append(f"INTRO GUIDANCE:\n{self.howto_intro.strip()}")
        if self.howto_segment.strip():
            guidance_parts.append(f"SEGMENT GUIDANCE:\n{self.howto_segment.strip()}")
        guidance_text = "\n\n".join(guidance_parts)

        hook_end_time = 6.0

        return f"""You are writing a fast, practical how-to narration.

{topic_context}URL being demonstrated: {url}
Total video duration: {duration:.1f} seconds

EDUCATIONAL CONTEXT:
{context_block}

RECORDED ACTIONS (with timestamps):
{actions}

{guidance_text}

STYLE RULES:
- Keep it concise and step-driven.
- Describe the action as it happens.
- Add tips only when they prevent errors.
- Use the educational context when it helps prevent errors.
- Keep the hook and outro short.

TIMING RULES:
- Hook: 0 to {hook_end_time} seconds.
- First segment starts at {hook_end_time} seconds.
- Each segment: 5 to 12 seconds of speech.
- Outro: final 8 to 12 seconds of the video.
- Total speech: about 75 percent of the video duration.

        OUTPUT REQUIREMENTS:
        - Return a JSON object with keys: title, hook, segments, outro.
        - Each segment has keys: start_time, end_time, text, emphasis.
        - Emphasis values: excited, normal, thoughtful.
        - Use timestamps that fit within the total duration.
        """

    def _build_news_prompt(self, url: str, duration: float, actions: str,
                           topic: Optional[str], educational_context: Optional[dict]) -> str:
        """Build a WorldofAI-style news prompt with hype pacing."""
        topic_context = f"Topic: {topic}\n" if topic else ""
        context_block = self._format_educational_context(educational_context)

        guidance_parts = []
        if self.news_intro.strip():
            guidance_parts.append(f"HOOK CADENCE:\n{self.news_intro.strip()}")
        if self.news_segment.strip():
            guidance_parts.append(f"SEGMENT CADENCE:\n{self.news_segment.strip()}")
        if self.news_outro.strip():
            guidance_parts.append(f"OUTRO CADENCE:\n{self.news_outro.strip()}")
        guidance_text = "\n\n".join(guidance_parts)

        hook_end_time = 5.0

        return f"""You are writing a WorldofAI-style AI news narration that reacts to a live screen recording.

{topic_context}URL being demonstrated: {url}
Total video duration: {duration:.1f} seconds

EDUCATIONAL CONTEXT (only use details that heighten urgency):
{context_block}

RECORDED ACTIONS (describe these exact beats like a play-by-play):
{actions}

{guidance_text}

STYLE RULES:
- Sound like you're mid-conversation and just saw this drop—no intros.
- Sentences stay under 10 words with energetic fragments and commands.
- Call out FREE/UPGRADED/INSANE speed moments the instant they appear.
- React to cursor moves, clicks, and results in real time ("watch", "look", "wait for it").
- Allowed slang: wild, ridiculous, insane, boom. Banned phrases: "in this video", "today we're".

TITLE RULES:
- Format inspiration: TOOL UPGRADE! BENEFIT + FREE HOOK (all caps for big words).
- Mention the benefit within the first six words and promise a payoff.
- Include "FREE", "UPGRADED", or "NEW" when the recording proves it.

TIMING RULES:
- Hook: 0 to {hook_end_time} seconds, feels like you hit record mid-sentence.
- Segments: 3-8 seconds each, always end by pointing to the next visual beat.
- Leave micro-pauses (1 second) only when the viewer needs to read on-screen text.
- Outro: final 7-10 seconds with recap, CTA, and tease of the next drop.

OUTPUT REQUIREMENTS:
- Return JSON with keys: title, hook, segments, outro.
- Each segment must include start_time, end_time, text, emphasis.
- Emphasis values: excited, normal, thoughtful (default to excited unless the visuals slow down).
- Keep timestamps within the total duration and never overlap segments.

Sound like a creator who just uncovered a wild AI update and needs the viewer to watch it NOW."""

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

        def _coerce_text(value) -> str:
            """Convert Claude responses to plain text."""
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                for key in ("text", "content", "value"):
                    if isinstance(value.get(key), str):
                        return value[key]
                # Fall back to stringified dict to retain info
                return " ".join(
                    _coerce_text(item) for item in value.values()
                    if isinstance(item, (str, dict, list))
                ).strip() or str(value)
            if isinstance(value, list):
                return " ".join(_coerce_text(item) for item in value).strip()
            return str(value)

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
        segments = []
        for seg in data.get("segments", []):
            if not isinstance(seg, dict):
                continue
            text_value = _coerce_text(seg.get("text", ""))
            emphasis_value = seg.get("emphasis", "normal")
            if isinstance(emphasis_value, dict):
                emphasis_value = emphasis_value.get("value", "normal")
            emphasis_text = str(emphasis_value or "normal").lower()
            segments.append(
                ScriptSegment(
                    start_time=seg.get("start_time", 0.0),
                    end_time=seg.get("end_time", 0.0),
                    text=text_value,
                    emphasis=emphasis_text or "normal"
                )
            )

        # Calculate word count
        hook_text = _coerce_text(data.get("hook", ""))
        outro_text = _coerce_text(data.get("outro", ""))
        all_text = f"{hook_text} {' '.join(s.text for s in segments)} {outro_text}".strip()
        word_count = len(all_text.split())

        return GeneratedScript(
            project_id=project_id,
            title=_coerce_text(data.get("title", f"Tutorial: {project_id}")),
            hook=hook_text,
            segments=segments,
            outro=outro_text,
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


def generate_script(
    timestamps_file: str,
    topic: Optional[str] = None,
    mode: Optional[ScriptMode] = None,
    educational_context: Optional[dict] = None
) -> dict:
    """
    Convenience function to generate a script.

    Args:
        timestamps_file: Path to timestamps JSON
        topic: Optional topic description
        mode: Script generation mode ("tutorial", "howto", or "news")
        educational_context: Optional dict with learning objectives, prerequisites, etc.

    Returns:
        Script data as dict
    """
    normalized_mode = mode.lower() if isinstance(mode, str) else mode
    generator = ScriptGenerator(mode=normalized_mode or "tutorial")
    script = generator.generate_from_timestamps(
        timestamps_file,
        topic,
        mode=normalized_mode,
        educational_context=educational_context
    )
    return asdict(script)
