"""
AI News Video Automator Modules.

Core modules for single-URL recording and the new multi-scene system.
"""
from .recorder import BrowserRecorder
from .scripter import ScriptGenerator
from .voice import VoiceGenerator
from .assembler import VideoAssembler
from .discovery import NewsDiscovery
from .captioner import CaptionGenerator
from .thumbnail import ThumbnailGenerator

# Lazy import for YouTubePublisher (requires google auth libraries)
def get_youtube_publisher():
    """Get YouTubePublisher class (lazy import to avoid google dependency)."""
    from .publisher import YouTubePublisher
    return YouTubePublisher

# Multi-scene system (new)
from .scenes import (
    Scene, SceneType, TransitionType, SceneManifest,
    load_manifest, create_tutorial_manifest, create_news_compilation_manifest,
    get_example_manifest
)
from .scene_recorder import SceneRecorder, record_from_manifest
from .scene_assembler import SceneAssembler, assemble_scenes

__all__ = [
    # Original modules
    "BrowserRecorder",
    "ScriptGenerator",
    "VoiceGenerator",
    "VideoAssembler",
    "NewsDiscovery",
    "get_youtube_publisher",  # Lazy import function
    "CaptionGenerator",
    "ThumbnailGenerator",
    # Multi-scene system
    "Scene",
    "SceneType",
    "TransitionType",
    "SceneManifest",
    "load_manifest",
    "create_tutorial_manifest",
    "create_news_compilation_manifest",
    "get_example_manifest",
    "SceneRecorder",
    "record_from_manifest",
    "SceneAssembler",
    "assemble_scenes",
]
