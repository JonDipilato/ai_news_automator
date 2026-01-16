"""
AI News Video Automator Modules.
"""
from .recorder import BrowserRecorder
from .scripter import ScriptGenerator
from .voice import VoiceGenerator
from .assembler import VideoAssembler
from .discovery import NewsDiscovery
from .publisher import YouTubePublisher
from .captioner import CaptionGenerator
from .thumbnail import ThumbnailGenerator

__all__ = [
    "BrowserRecorder",
    "ScriptGenerator",
    "VoiceGenerator",
    "VideoAssembler",
    "NewsDiscovery",
    "YouTubePublisher",
    "CaptionGenerator",
    "ThumbnailGenerator",
]
