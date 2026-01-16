"""
Browser recording module using Playwright.
Records browser interactions with timestamp capture for narration sync.
"""
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
from playwright.async_api import async_playwright, Page, Browser

from config.settings import (
    VIDEO_WIDTH, VIDEO_HEIGHT, RECORDINGS_DIR,
    RECORDING_FORMAT, DEFAULT_RECORDING_DURATION
)


@dataclass
class ActionTimestamp:
    """Represents a timed action during recording."""
    time_seconds: float
    action_type: str
    description: str
    element_text: Optional[str] = None
    url: Optional[str] = None


class BrowserRecorder:
    """Records browser sessions with action timestamps."""

    def __init__(self):
        self.timestamps: list[ActionTimestamp] = []
        self.start_time: float = 0
        self.recording_path: Optional[Path] = None

    def _log_action(self, action_type: str, description: str,
                    element_text: str = None, url: str = None):
        """Log a timestamped action."""
        elapsed = time.time() - self.start_time
        self.timestamps.append(ActionTimestamp(
            time_seconds=round(elapsed, 2),
            action_type=action_type,
            description=description,
            element_text=element_text,
            url=url
        ))

    async def record_url(self, url: str, project_id: str,
                         duration: int = DEFAULT_RECORDING_DURATION,
                         demo_steps: Optional[list[dict]] = None) -> dict:
        """
        Record a browser session visiting a URL.

        Args:
            url: The URL to visit and record
            project_id: Unique identifier for this recording
            duration: Maximum recording duration in seconds
            demo_steps: Optional list of demo actions to perform

        Returns:
            dict with recording path, timestamps path, and duration
        """
        recording_file = RECORDINGS_DIR / f"{project_id}.{RECORDING_FORMAT}"
        timestamps_file = RECORDINGS_DIR / f"{project_id}_timestamps.json"

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": VIDEO_WIDTH, "height": VIDEO_HEIGHT},
                record_video_dir=str(RECORDINGS_DIR),
                record_video_size={"width": VIDEO_WIDTH, "height": VIDEO_HEIGHT}
            )

            page = await context.new_page()

            # Start timing
            self.start_time = time.time()
            self.timestamps = []

            # Navigate to URL
            self._log_action("navigate", f"Opening {url}", url=url)
            await page.goto(url, wait_until="networkidle")
            self._log_action("page_load", f"Page loaded: {await page.title()}")

            # Execute demo steps if provided
            if demo_steps:
                await self._execute_demo_steps(page, demo_steps)
            else:
                # Default: wait for manual interaction or duration
                await self._interactive_record(page, duration)

            # Get video path before closing
            video = page.video
            if video:
                video_path = await video.path()

            await context.close()
            await browser.close()

            # Move video to correct filename
            if video_path and Path(video_path).exists():
                Path(video_path).rename(recording_file)

            # Calculate actual duration
            actual_duration = time.time() - self.start_time

            # Save timestamps
            timestamps_data = {
                "project_id": project_id,
                "url": url,
                "recorded_at": datetime.now().isoformat(),
                "duration_seconds": round(actual_duration, 2),
                "actions": [asdict(ts) for ts in self.timestamps]
            }

            with open(timestamps_file, "w") as f:
                json.dump(timestamps_data, f, indent=2)

            return {
                "recording": str(recording_file),
                "timestamps": str(timestamps_file),
                "duration": actual_duration,
                "action_count": len(self.timestamps)
            }

    async def _execute_demo_steps(self, page: Page, steps: list[dict]):
        """Execute predefined demo steps with timing."""
        for step in steps:
            action = step.get("action")
            selector = step.get("selector")
            value = step.get("value")
            wait = step.get("wait", 1)
            description = step.get("description", action)

            try:
                if action == "click":
                    element = page.locator(selector)
                    text = await element.text_content() if await element.count() > 0 else None
                    self._log_action("click", description, element_text=text)
                    await element.click()

                elif action == "type":
                    self._log_action("type", description, element_text=value)
                    await page.locator(selector).fill(value)

                elif action == "scroll":
                    self._log_action("scroll", description)
                    await page.evaluate(f"window.scrollBy(0, {value or 500})")

                elif action == "wait":
                    self._log_action("wait", description)
                    await asyncio.sleep(float(value or wait))

                elif action == "screenshot":
                    self._log_action("highlight", description)
                    # Visual pause for emphasis
                    await asyncio.sleep(0.5)

                elif action == "navigate":
                    self._log_action("navigate", description, url=value)
                    await page.goto(value, wait_until="networkidle")

                # Wait between actions for natural pacing
                await asyncio.sleep(wait)

            except Exception as e:
                self._log_action("error", f"Failed: {description} - {str(e)}")

    async def _interactive_record(self, page: Page, max_duration: int):
        """Record with manual interaction support."""
        print(f"\n Recording started. Press Ctrl+C to stop (max {max_duration}s)")
        print(" Perform your demo actions in the browser...\n")

        # Set up action listeners
        page.on("load", lambda: self._log_action("page_load", f"Page loaded: {page.url}"))

        try:
            # Wait for duration or manual stop
            await asyncio.sleep(max_duration)
        except asyncio.CancelledError:
            self._log_action("end", "Recording stopped by user")

        self._log_action("end", "Recording completed")


async def record_demo(url: str, project_id: str, duration: int = 60,
                      steps_file: Optional[str] = None) -> dict:
    """
    Convenience function to record a demo.

    Args:
        url: URL to record
        project_id: Unique project identifier
        duration: Max duration in seconds
        steps_file: Optional JSON file with demo steps

    Returns:
        Recording result dict
    """
    recorder = BrowserRecorder()

    demo_steps = None
    if steps_file:
        with open(steps_file) as f:
            demo_steps = json.load(f)

    return await recorder.record_url(url, project_id, duration, demo_steps)
