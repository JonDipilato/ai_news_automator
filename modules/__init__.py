"""
AI News Video Automator package.

Intentionally avoids importing heavy submodules at package load time so that
optional dependencies (Playwright, Google APIs, etc.) are only required when the
corresponding module is explicitly imported.
"""

__all__ = ["get_youtube_publisher"]


def get_youtube_publisher():
    """Return YouTubePublisher lazily to avoid importing google-auth dependencies."""
    from .publisher import YouTubePublisher

    return YouTubePublisher
