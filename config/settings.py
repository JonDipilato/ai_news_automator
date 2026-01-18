"""
Central configuration for AI News Video Automator.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
TEMPLATES_DIR = BASE_DIR / "templates"
ASSETS_DIR = BASE_DIR / "assets"

# Output subdirectories
RECORDINGS_DIR = OUTPUT_DIR / "recordings"
AUDIO_DIR = OUTPUT_DIR / "audio"
CAPTIONS_DIR = OUTPUT_DIR / "captions"
REVIEW_DIR = OUTPUT_DIR / "review"
PUBLISHED_DIR = OUTPUT_DIR / "published"

# Ensure directories exist
for dir_path in [RECORDINGS_DIR, AUDIO_DIR, CAPTIONS_DIR, REVIEW_DIR, PUBLISHED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Video settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"
AUDIO_SAMPLE_RATE = 44100

# TTS settings
TTS_MODEL = "tts-1-hd"
TTS_VOICE = "onyx"

# Script generation settings
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_SCRIPT_TOKENS = 4096

# Recording settings
RECORDING_FORMAT = "mp4"
DEFAULT_RECORDING_DURATION = 300  # 5 minutes max

# YouTube settings
YOUTUBE_CATEGORY_ID = "28"  # Science & Technology
YOUTUBE_PRIVACY = "private"  # Default to private for review
PUBLISH_DAYS = ["tuesday", "friday"]
PUBLISH_TIME = "14:00"  # 2 PM local

# Quality thresholds
MIN_AUDIO_VIDEO_SYNC_TOLERANCE = 0.05  # 5% tolerance
MIN_VIDEO_DURATION = 30  # seconds
MAX_VIDEO_DURATION = 900  # 15 minutes


def validate_api_keys():
    """Check that required API keys are configured."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    return missing


def get_project_path(project_id: str) -> dict:
    """Get all paths for a project."""
    return {
        "recording": RECORDINGS_DIR / f"{project_id}.{RECORDING_FORMAT}",
        "timestamps": RECORDINGS_DIR / f"{project_id}_timestamps.json",
        "script": RECORDINGS_DIR / f"{project_id}_script.json",
        "audio": AUDIO_DIR / f"{project_id}.mp3",
        "captions": CAPTIONS_DIR / f"{project_id}.srt",
        "review_video": REVIEW_DIR / f"{project_id}.mp4",
        "review_meta": REVIEW_DIR / f"{project_id}_meta.json",
        "published": PUBLISHED_DIR / f"{project_id}.mp4",
    }
