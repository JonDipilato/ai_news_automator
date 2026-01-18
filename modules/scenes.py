"""
Scene-based video production system.
Supports multi-scene tutorials with terminal, browser, article, and transition scenes.
"""
import json
import yaml
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum

from config.settings import RECORDINGS_DIR, ASSETS_DIR


class SceneType(str, Enum):
    """Types of scenes supported."""
    BROWSER = "browser"       # Record a webpage demo
    TERMINAL = "terminal"     # Record terminal/CLI commands
    ARTICLE = "article"       # Static article with scroll/highlights
    TRANSITION = "transition" # Pure transition (no recording, just effect)
    VIDEO = "video"           # Pre-recorded video clip


class TransitionType(str, Enum):
    """FFmpeg transition effects between scenes."""
    CUT = "cut"               # Hard cut (no transition)
    FADE = "fade"             # Fade to black and back
    CROSSFADE = "crossfade"   # Dissolve between scenes
    WIPE_LEFT = "wipe_left"   # Wipe from right to left
    WIPE_RIGHT = "wipe_right" # Wipe from left to right
    WIPE_UP = "wipe_up"       # Wipe from bottom to top
    WIPE_DOWN = "wipe_down"   # Wipe from top to bottom
    ZOOM_IN = "zoom_in"       # Zoom into next scene
    ZOOM_OUT = "zoom_out"     # Zoom out to next scene
    SLIDE_LEFT = "slide_left" # Slide current scene left, new from right
    SLIDE_RIGHT = "slide_right"
    GLITCH = "glitch"         # Quick glitch effect (tech videos)


@dataclass
class Scene:
    """A single scene in the video."""
    type: SceneType
    duration: int = 30                              # Max duration in seconds
    url: Optional[str] = None                       # For browser/article scenes
    command: Optional[str] = None                   # For terminal scenes
    commands: Optional[list[str]] = None            # Multiple terminal commands
    video_path: Optional[str] = None                # For pre-recorded video scenes
    description: str = ""                           # What happens in this scene (for script)
    highlight_elements: list[str] = field(default_factory=list)  # CSS selectors to highlight
    scroll_to: Optional[str] = None                 # CSS selector to scroll to
    transition_in: TransitionType = TransitionType.FADE
    transition_duration: float = 0.5                # Transition duration in seconds
    demo_steps: list[dict] = field(default_factory=list)  # Detailed demo steps
    narration_hint: str = ""                        # Hint for script generation
    prompt: Optional[str] = None                    # For AI chat sites (ChatGPT, Claude, Gemini)

    # Recording backend selection
    recorder: Optional[str] = None                  # "screen_capture", "rendered", or "auto"

    # Educational content fields
    learning_objective: str = ""                    # What viewers will learn from this scene
    concept_explanation: str = ""                   # Background concept to explain
    prerequisites: list[str] = field(default_factory=list)  # What viewers should know first
    common_mistakes: list[str] = field(default_factory=list)  # Mistakes to warn about
    key_takeaway: str = ""                          # Main point to remember
    tips: list[str] = field(default_factory=list)   # Pro tips for this step

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = SceneType(self.type)
        if isinstance(self.transition_in, str):
            self.transition_in = TransitionType(self.transition_in)


@dataclass
class SceneManifest:
    """Complete manifest for a multi-scene video."""
    project_id: str
    title: str
    topic: str
    scenes: list[Scene]
    overall_theme: str = "tutorial"                 # tutorial, news, review, showcase
    target_duration: int = 180                      # Target total duration in seconds
    style: str = "energetic"                        # energetic, calm, professional

    # Script generation mode
    script_mode: str = "tutorial"                   # "tutorial" (full educational) or "howto" (quick practical)

    # Default recording backend
    default_recorder: str = "auto"                  # "screen_capture", "rendered", or "auto"

    # Educational content fields (video-level)
    video_learning_objectives: list[str] = field(default_factory=list)  # What viewers will learn overall
    target_audience: str = ""                       # Who this video is for
    difficulty_level: str = "beginner"              # beginner, intermediate, advanced
    prerequisites: list[str] = field(default_factory=list)  # What viewers need to know first
    related_topics: list[str] = field(default_factory=list)  # Topics to explore after

    # Outro settings
    outro_style: str = "casual"                     # casual, professional, hype
    include_subscribe_cta: bool = True
    next_video_tease: Optional[str] = None          # Tease for next video

    def __post_init__(self):
        self.scenes = [
            Scene(**s) if isinstance(s, dict) else s
            for s in self.scenes
        ]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['scenes'] = [
            {**asdict(s), 'type': s.type.value, 'transition_in': s.transition_in.value}
            for s in self.scenes
        ]
        return result

    def save(self, path: Optional[Path] = None) -> Path:
        """Save manifest to file."""
        if path is None:
            path = RECORDINGS_DIR / f"{self.project_id}_manifest.yaml"

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        return path


def load_manifest(path: str) -> SceneManifest:
    """Load a scene manifest from YAML or JSON file."""
    path = Path(path)

    with open(path) as f:
        if path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    return SceneManifest(**data)


def create_tutorial_manifest(
    project_id: str,
    title: str,
    topic: str,
    scenes: list[dict],
    target_duration: int = 180
) -> SceneManifest:
    """
    Create a manifest for an interactive tutorial.

    Example scenes:
    [
        {"type": "terminal", "commands": ["source venv/bin/activate", "claude"], "duration": 20},
        {"type": "browser", "url": "https://article.com", "duration": 30},
        {"type": "terminal", "command": "/plugins install xyz", "duration": 15},
    ]
    """
    return SceneManifest(
        project_id=project_id,
        title=title,
        topic=topic,
        scenes=[Scene(**s) for s in scenes],
        target_duration=target_duration,
        overall_theme="tutorial",
        style="energetic",
        include_subscribe_cta=True
    )


def create_news_compilation_manifest(
    project_id: str,
    title: str,
    topic: str,
    article_urls: list[str],
    target_duration: int = 300
) -> SceneManifest:
    """
    Create a manifest for a multi-article news compilation.
    Automatically adds transitions between articles.
    """
    scenes = []
    transitions = [
        TransitionType.WIPE_LEFT,
        TransitionType.ZOOM_IN,
        TransitionType.SLIDE_LEFT,
        TransitionType.CROSSFADE,
    ]

    for i, url in enumerate(article_urls):
        duration_per_article = target_duration // len(article_urls)
        transition = transitions[i % len(transitions)]

        scenes.append(Scene(
            type=SceneType.ARTICLE,
            url=url,
            duration=duration_per_article,
            transition_in=transition,
            transition_duration=0.7,
            description=f"Article {i+1}: {url}",
            narration_hint="Summarize key points, add insights, keep energy up"
        ))

    return SceneManifest(
        project_id=project_id,
        title=title,
        topic=topic,
        scenes=scenes,
        target_duration=target_duration,
        overall_theme="news",
        style="energetic",
        include_subscribe_cta=True,
        next_video_tease="More AI updates dropping soon"
    )


# Example manifest templates
EXAMPLE_CLAUDE_CODE_TUTORIAL = """
project_id: claude_code_plugins
title: "Install Claude Code Plugins in 60 Seconds"
topic: "How to install and use Claude Code plugins"
target_duration: 90
overall_theme: tutorial
style: energetic

# Script generation mode: "tutorial" for full educational, "howto" for quick practical
script_mode: tutorial

# Recording backend: "screen_capture", "rendered", or "auto"
default_recorder: auto

# Educational metadata
video_learning_objectives:
  - "Understand what Claude Code plugins are"
  - "Install a plugin from the community"
  - "Use plugin commands effectively"
target_audience: "Developers using Claude Code CLI"
difficulty_level: beginner
prerequisites:
  - "Claude Code installed and working"
  - "Basic terminal knowledge"
related_topics:
  - "Claude Code hooks"
  - "Building your own plugins"

scenes:
  - type: terminal
    commands:
      - "source venv/bin/activate"
      - "claude"
    duration: 15
    description: "Opening Claude Code in a virtual environment"
    transition_in: fade
    narration_hint: "Quick setup, show how easy it is"
    # Educational fields for this scene
    learning_objective: "Launch Claude Code from terminal"
    concept_explanation: "Claude Code is an AI coding assistant that runs in your terminal"
    tips:
      - "Use a virtual environment to keep dependencies isolated"

  - type: browser
    url: "https://github.com/anthropics/claude-code"
    duration: 20
    description: "Show the Claude Code repo and plugins section"
    highlight_elements:
      - ".markdown-body h2"
    transition_in: zoom_in
    narration_hint: "Highlight where to find plugins"
    learning_objective: "Find available plugins"
    key_takeaway: "The plugins directory lists all community plugins"

  - type: terminal
    command: "/plugins install superclaude"
    duration: 25
    description: "Installing a plugin live"
    transition_in: wipe_left
    narration_hint: "Show the magic happening"
    learning_objective: "Install a plugin with one command"
    common_mistakes:
      - "Forgetting to restart Claude Code after installing"
    tips:
      - "Plugins are installed to ~/.claude/plugins/"

  - type: terminal
    command: "/sc help"
    duration: 20
    description: "Demonstrating the plugin commands"
    transition_in: slide_left
    narration_hint: "Show off what you can do now"
    learning_objective: "Use the new plugin commands"
    key_takeaway: "Each plugin adds new slash commands"

outro_style: casual
include_subscribe_cta: true
next_video_tease: "Next up - my top 5 Claude Code workflows"
"""


def get_example_manifest() -> str:
    """Return example manifest YAML for reference."""
    return EXAMPLE_CLAUDE_CODE_TUTORIAL
