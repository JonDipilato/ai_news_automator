"""
Multi-scene recorder with REAL interactions.
- Real command execution with actual output
- Real browser interactions using Playwright
- Screen recording of actual content

Supports multiple recording backends:
- screen_capture: Real screen capture (requires display)
- rendered: PIL-rendered frames (works headless)
- auto: Automatically selects best option
"""
import json
import time
import asyncio
import subprocess
import shutil
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
from dataclasses import dataclass, asdict

from playwright.async_api import async_playwright, Page, Browser

from config.settings import (
    VIDEO_WIDTH, VIDEO_HEIGHT, RECORDINGS_DIR,
    RECORDING_FORMAT, DEFAULT_RECORDING_DURATION
)
from .scenes import Scene, SceneType, SceneManifest, load_manifest

# Import recording backends
from .true_recorder import TrueTerminalRecorder, TrueRecordingResult
from .screen_recorder import ScreenCaptureRecorder, is_screen_capture_available

# Recorder type alias
RecorderType = Literal["screen_capture", "rendered", "auto"]


@dataclass
class SceneRecording:
    """Result of recording a single scene."""
    scene_index: int
    scene_type: str
    video_path: str
    duration: float
    timestamps: list[dict]
    success: bool
    error: Optional[str] = None


@dataclass
class ManifestRecordingResult:
    """Result of recording all scenes in a manifest."""
    project_id: str
    scene_recordings: list[SceneRecording]
    total_duration: float
    success: bool
    scenes_file: str


class RealTerminal:
    """Execute commands in a real PTY and capture output."""

    def __init__(self):
        self.master_fd = None
        self.slave_fd = None
        self.pid = None

    def execute_command(self, cmd: str, timeout: float = 30) -> tuple[str, int]:
        """Execute a command and return (output, exit_code)."""
        try:
            # For comments, just return empty
            if cmd.strip().startswith('#'):
                return ("", 0)

            # Run the actual command
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.expanduser("~")
            )

            output = result.stdout
            if result.stderr and result.returncode != 0:
                output += result.stderr

            return (output.strip(), result.returncode)

        except subprocess.TimeoutExpired:
            return ("Command timed out", 1)
        except Exception as e:
            return (str(e), 1)


class SceneRecorder:
    """Records multiple scenes with REAL interactions."""

    def __init__(self, default_recorder: RecorderType = "auto"):
        self.current_timestamps: list[dict] = []
        self.start_time: float = 0
        self.terminal = RealTerminal()
        self.default_recorder = default_recorder

        # Initialize recording backends
        self.true_recorder = TrueTerminalRecorder()
        self.screen_recorder = ScreenCaptureRecorder()

    def _select_recorder(self, scene_recorder: Optional[str] = None) -> str:
        """
        Select the appropriate recorder backend.

        Args:
            scene_recorder: Specific recorder requested for this scene

        Returns:
            Either "screen_capture" or "rendered"
        """
        recorder = scene_recorder or self.default_recorder

        if recorder == "auto":
            # Check if screen capture is available
            available, reason = is_screen_capture_available()
            if available:
                print(f"  Using screen capture recorder (display available)")
                return "screen_capture"
            else:
                print(f"  Using rendered recorder ({reason})")
                return "rendered"

        return recorder

    def _log_action(self, action_type: str, description: str,
                    element_text: str = None, url: str = None, scene_index: int = None):
        """Log a timestamped action."""
        elapsed = time.time() - self.start_time
        self.current_timestamps.append({
            "time_seconds": round(elapsed, 2),
            "action_type": action_type,
            "description": description,
            "element_text": element_text,
            "url": url,
            "scene_index": scene_index
        })

    async def record_manifest(self, manifest_path: str) -> ManifestRecordingResult:
        """Record all scenes from a manifest file."""
        manifest = load_manifest(manifest_path)
        return await self.record_scenes(manifest)

    async def record_scenes(self, manifest: SceneManifest) -> ManifestRecordingResult:
        """Record all scenes in a manifest."""
        recordings = []
        total_duration = 0

        for i, scene in enumerate(manifest.scenes):
            print(f"\n Recording scene {i+1}/{len(manifest.scenes)}: {scene.type.value}")

            if scene.type == SceneType.BROWSER:
                result = await self._record_browser_scene(manifest.project_id, i, scene)
            elif scene.type == SceneType.TERMINAL:
                result = await self._record_real_terminal_scene(manifest.project_id, i, scene)
            elif scene.type == SceneType.ARTICLE:
                result = await self._record_article_scene(manifest.project_id, i, scene)
            elif scene.type == SceneType.VIDEO:
                result = self._use_existing_video(manifest.project_id, i, scene)
            else:
                result = SceneRecording(
                    scene_index=i,
                    scene_type=scene.type.value,
                    video_path="",
                    duration=0,
                    timestamps=[],
                    success=True
                )

            recordings.append(result)
            if result.success:
                total_duration += result.duration

        # Save combined scene data
        scenes_file = RECORDINGS_DIR / f"{manifest.project_id}_scenes.json"
        scenes_data = {
            "project_id": manifest.project_id,
            "title": manifest.title,
            "topic": manifest.topic,
            "recorded_at": datetime.now().isoformat(),
            "total_duration": total_duration,
            "scene_count": len(recordings),
            "scenes": [asdict(r) for r in recordings],
            "manifest": manifest.to_dict()
        }

        with open(scenes_file, 'w') as f:
            json.dump(scenes_data, f, indent=2)

        successful_scenes = sum(1 for r in recordings if r.success)
        partial_success = successful_scenes >= len(recordings) // 2

        return ManifestRecordingResult(
            project_id=manifest.project_id,
            scene_recordings=recordings,
            total_duration=total_duration,
            success=partial_success,
            scenes_file=str(scenes_file)
        )

    async def _record_real_terminal_scene(self, project_id: str, index: int,
                                           scene: Scene) -> SceneRecording:
        """
        Record terminal with REAL command execution.

        Automatically selects the best recorder based on:
        - scene.recorder setting (if specified)
        - Default recorder (if set)
        - Auto-detection (screen_capture if display available, else rendered)
        """
        self.start_time = time.time()
        self.current_timestamps = []

        commands = scene.commands or ([scene.command] if scene.command else [])

        # Get recorder preference from scene (if available) or use default
        scene_recorder = getattr(scene, 'recorder', None)
        recorder_type = self._select_recorder(scene_recorder)

        try:
            if recorder_type == "screen_capture":
                # Use real screen capture
                result = self.screen_recorder.record_session(
                    project_id=project_id,
                    scene_index=index,
                    commands=commands,
                    duration=scene.duration
                )
            else:
                # Use PIL-rendered frames (works headless)
                result = self.true_recorder.record_session(
                    project_id=project_id,
                    scene_index=index,
                    commands=commands,
                    duration=scene.duration
                )

            # Log actions from timestamps
            for ts in result.timestamps:
                self._log_action(
                    ts.get("action_type", "command"),
                    ts.get("description", ""),
                    scene_index=index
                )

            if result.success:
                return SceneRecording(
                    scene_index=index,
                    scene_type="terminal",
                    video_path=str(result.video_path),
                    duration=result.duration,
                    timestamps=self.current_timestamps.copy(),
                    success=True
                )
            else:
                return SceneRecording(
                    scene_index=index,
                    scene_type="terminal",
                    video_path="",
                    duration=0,
                    timestamps=[],
                    success=False,
                    error=result.error or "Recording failed"
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return SceneRecording(
                scene_index=index,
                scene_type="terminal",
                video_path="",
                duration=0,
                timestamps=[],
                success=False,
                error=str(e)
            )

    def _generate_real_terminal_html(self, command_outputs: list[dict], duration: float) -> str:
        """Generate terminal HTML that shows REAL command outputs with typing animation."""

        # Escape for JSON
        data_json = json.dumps(command_outputs)

        # Calculate timing
        total_chars = sum(len(c["command"]) + len(c.get("output", "")) for c in command_outputs)
        char_delay = min(60, max(20, (duration * 600) / max(total_chars, 1)))

        return f'''<!DOCTYPE html>
<html>
<head>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{
    width: 100%;
    height: 100%;
    overflow: hidden;
}}
body {{
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', 'Consolas', monospace;
    font-size: 16px;
    color: #c9d1d9;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 40px;
}}
.terminal {{
    width: 100%;
    max-width: 1200px;
    height: 85vh;
    background: #0d1117;
    border-radius: 12px;
    box-shadow:
        0 0 0 1px rgba(48, 54, 61, 0.5),
        0 16px 70px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.05);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}}
.header {{
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    border-bottom: 1px solid #30363d;
    flex-shrink: 0;
}}
.dots {{ display: flex; gap: 8px; }}
.dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
    position: relative;
}}
.dot::after {{
    content: '';
    position: absolute;
    inset: 2px;
    border-radius: 50%;
    background: rgba(255,255,255,0.2);
}}
.dot.red {{ background: #ff5f56; }}
.dot.yellow {{ background: #ffbd2e; }}
.dot.green {{ background: #27c93f; }}
.title {{
    flex: 1;
    text-align: center;
    color: #8b949e;
    font-size: 13px;
    font-weight: 500;
}}
.content {{
    flex: 1;
    padding: 20px 24px;
    overflow-y: auto;
    line-height: 1.7;
}}
.line {{
    display: flex;
    align-items: flex-start;
    margin: 4px 0;
    min-height: 24px;
}}
.prompt {{
    color: #58a6ff;
    margin-right: 8px;
    font-weight: 600;
    white-space: nowrap;
    text-shadow: 0 0 10px rgba(88, 166, 255, 0.3);
}}
.prompt .dir {{ color: #7ee787; }}
.prompt .arrow {{ color: #f78166; margin: 0 4px; }}
.command {{
    color: #f0f6fc;
    white-space: pre-wrap;
    word-break: break-word;
}}
.command.comment {{
    color: #8b949e;
    font-style: italic;
}}
.output {{
    color: #8b949e;
    padding: 6px 0 6px 0;
    white-space: pre-wrap;
    font-size: 14px;
    line-height: 1.5;
    border-left: 2px solid #30363d;
    padding-left: 12px;
    margin: 8px 0 12px 0;
    opacity: 0;
    animation: fadeSlideIn 0.4s ease forwards;
}}
.output.error {{
    border-left-color: #f85149;
    color: #f85149;
}}
.output.success {{
    border-left-color: #3fb950;
}}
.cursor {{
    display: inline-block;
    width: 9px;
    height: 18px;
    background: #58a6ff;
    animation: blink 1s step-end infinite;
    margin-left: 1px;
    vertical-align: text-bottom;
    box-shadow: 0 0 10px rgba(88, 166, 255, 0.5);
}}
@keyframes blink {{
    0%, 50% {{ opacity: 1; }}
    51%, 100% {{ opacity: 0; }}
}}
@keyframes fadeSlideIn {{
    from {{
        opacity: 0;
        transform: translateY(-5px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}
.typing-indicator {{
    color: #6e7681;
    font-size: 12px;
    margin-top: 8px;
}}
</style>
</head>
<body>
<div class="terminal">
    <div class="header">
        <div class="dots">
            <div class="dot red"></div>
            <div class="dot yellow"></div>
            <div class="dot green"></div>
        </div>
        <div class="title">Terminal — bash</div>
    </div>
    <div class="content" id="content"></div>
</div>
<script>
const commandData = {data_json};
const content = document.getElementById('content');
const charDelay = {char_delay};

function sleep(ms) {{ return new Promise(r => setTimeout(r, ms)); }}

function createPrompt() {{
    const prompt = document.createElement('span');
    prompt.className = 'prompt';
    prompt.innerHTML = '<span class="dir">~</span><span class="arrow">❯</span>';
    return prompt;
}}

function escapeHtml(text) {{
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}}

async function typeText(element, text, delay) {{
    for (let i = 0; i < text.length; i++) {{
        element.textContent += text[i];
        // Vary typing speed for realism
        const variance = Math.random() * delay * 0.5;
        await sleep(delay + variance);

        // Occasional longer pause at spaces
        if (text[i] === ' ' && Math.random() > 0.7) {{
            await sleep(delay * 2);
        }}
    }}
}}

async function showOutput(text, isError) {{
    if (!text || text.trim() === '') return;

    const output = document.createElement('div');
    output.className = 'output' + (isError ? ' error' : '');
    output.textContent = text;
    content.appendChild(output);

    // Scroll to bottom
    content.scrollTop = content.scrollHeight;

    await sleep(400);
}}

async function runTerminal() {{
    await sleep(600);

    for (const item of commandData) {{
        const line = document.createElement('div');
        line.className = 'line';
        line.appendChild(createPrompt());

        const cmdSpan = document.createElement('span');
        cmdSpan.className = 'command';
        if (item.command.trim().startsWith('#')) {{
            cmdSpan.classList.add('comment');
        }}
        line.appendChild(cmdSpan);

        const cursor = document.createElement('span');
        cursor.className = 'cursor';
        line.appendChild(cursor);

        content.appendChild(line);
        content.scrollTop = content.scrollHeight;

        // Type the command
        await typeText(cmdSpan, item.command, charDelay);

        await sleep(200);
        cursor.remove();

        // Show "running" state briefly for non-comments
        if (!item.command.trim().startsWith('#') && item.output) {{
            await sleep(300);
        }}

        // Show real output
        if (item.output && item.output.trim()) {{
            await showOutput(item.output, item.exit_code !== 0);
        }}

        await sleep(600);
    }}

    // Final prompt with cursor
    const finalLine = document.createElement('div');
    finalLine.className = 'line';
    finalLine.appendChild(createPrompt());
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    finalLine.appendChild(cursor);
    content.appendChild(finalLine);
    content.scrollTop = content.scrollHeight;
}}

runTerminal();
</script>
</body>
</html>'''

    async def _record_browser_scene(self, project_id: str, index: int,
                                     scene: Scene) -> SceneRecording:
        """Record a browser scene with REAL interactions."""
        video_file = RECORDINGS_DIR / f"{project_id}_scene{index:03d}.{RECORDING_FORMAT}"

        self.current_timestamps = []
        self.start_time = time.time()

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--disable-blink-features=AutomationControlled']
                )
                context = await browser.new_context(
                    viewport={"width": VIDEO_WIDTH, "height": VIDEO_HEIGHT},
                    record_video_dir=str(RECORDINGS_DIR),
                    record_video_size={"width": VIDEO_WIDTH, "height": VIDEO_HEIGHT},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )

                page = await context.new_page()

                # Navigate
                self._log_action("navigate", f"Opening {scene.url}", url=scene.url, scene_index=index)
                await page.goto(scene.url, wait_until="networkidle", timeout=60000)
                await asyncio.sleep(2)
                self._log_action("page_load", "Page loaded", scene_index=index)

                # Check if this is an AI chat site
                if any(site in scene.url for site in ['claude.ai', 'chat.openai.com', 'gemini.google.com', 'chatgpt.com']):
                    await self._interact_with_ai_chat(page, scene, index)
                elif scene.demo_steps:
                    await self._execute_demo_steps(page, scene.demo_steps, index)
                else:
                    await self._interactive_browse(page, scene, index)

                # Ensure minimum duration
                elapsed = time.time() - self.start_time
                remaining = scene.duration - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)

                self._log_action("scene_end", "Scene completed", scene_index=index)

                video = page.video
                temp_path = await video.path() if video else None

                await context.close()
                await browser.close()

                if temp_path and Path(temp_path).exists():
                    Path(temp_path).rename(video_file)

                return SceneRecording(
                    scene_index=index,
                    scene_type="browser",
                    video_path=str(video_file),
                    duration=time.time() - self.start_time,
                    timestamps=self.current_timestamps.copy(),
                    success=True
                )

        except Exception as e:
            return SceneRecording(
                scene_index=index,
                scene_type="browser",
                video_path="",
                duration=0,
                timestamps=[],
                success=False,
                error=str(e)
            )

    async def _interact_with_ai_chat(self, page: Page, scene: Scene, index: int):
        """Type into AI chat and wait for response."""
        prompt = scene.prompt or "Write a Python function to calculate fibonacci numbers with memoization"

        self._log_action("ai_interaction", "Preparing to chat with AI", scene_index=index)
        await asyncio.sleep(2)

        # Common input selectors for AI chat sites
        input_selectors = [
            '#prompt-textarea',
            'textarea[placeholder*="Message"]',
            'textarea[placeholder*="message"]',
            'textarea[data-id="root"]',
            'div[contenteditable="true"][role="textbox"]',
            '.ProseMirror',
            'textarea',
        ]

        input_found = False
        for selector in input_selectors:
            try:
                input_el = await page.wait_for_selector(selector, timeout=3000)
                if input_el:
                    await input_el.click()
                    await asyncio.sleep(0.5)

                    # Type character by character for realistic effect
                    self._log_action("typing", f"Typing: {prompt[:40]}...", scene_index=index)

                    for i, char in enumerate(prompt):
                        await page.keyboard.type(char, delay=35 + (25 * (hash(char) % 10) / 10))

                        # Occasional thinking pause
                        if char in '.,' and i < len(prompt) - 1:
                            await asyncio.sleep(0.15)

                    input_found = True
                    break
            except Exception:
                continue

        if input_found:
            await asyncio.sleep(1)

            # Try to submit
            submit_methods = [
                lambda: page.keyboard.press('Enter'),
                lambda: page.click('button[data-testid="send-button"]'),
                lambda: page.click('button[aria-label*="Send"]'),
                lambda: page.click('button:has-text("Send")'),
            ]

            for method in submit_methods:
                try:
                    await method()
                    self._log_action("submit", "Sent message", scene_index=index)
                    break
                except Exception:
                    continue

            # Wait for response
            self._log_action("waiting", "Waiting for AI response...", scene_index=index)
            await asyncio.sleep(scene.duration * 0.5)
        else:
            # Fallback to browsing
            await self._interactive_browse(page, scene, index)

    async def _interactive_browse(self, page: Page, scene: Scene, index: int):
        """Browse a page with realistic mouse movement, scrolling, and interactions."""

        # Inject interactive cursor
        await page.evaluate('''
            const cursor = document.createElement('div');
            cursor.id = 'visual-cursor';
            cursor.style.cssText = `
                position: fixed;
                width: 20px;
                height: 20px;
                background: radial-gradient(circle, rgba(59, 130, 246, 0.8) 0%, rgba(59, 130, 246, 0.2) 60%, transparent 100%);
                border-radius: 50%;
                pointer-events: none;
                z-index: 999999;
                transition: left 0.3s ease-out, top 0.3s ease-out;
                box-shadow: 0 0 15px rgba(59, 130, 246, 0.6), 0 0 30px rgba(59, 130, 246, 0.3);
            `;
            cursor.style.left = '50%';
            cursor.style.top = '30%';
            document.body.appendChild(cursor);
        ''')

        # Find interesting elements
        interesting_selectors = [
            'h1', 'h2', '.hero', '[class*="hero"]',
            'button:not([disabled])', '.btn', '[class*="cta"]',
            'a[href]:not([href^="#"])', 'img[alt]',
            '[class*="feature"]', '[class*="card"]'
        ]

        elements_to_visit = []
        for selector in interesting_selectors:
            try:
                els = await page.query_selector_all(selector)
                for el in els[:3]:
                    box = await el.bounding_box()
                    if box and 0 < box['y'] < 2000 and box['width'] > 50:
                        elements_to_visit.append((el, box))
            except Exception:
                continue

        # Visit elements with cursor
        time_per_element = (scene.duration * 0.6) / max(len(elements_to_visit[:8]), 1)

        for i, (el, box) in enumerate(elements_to_visit[:8]):
            try:
                cx = box['x'] + box['width'] / 2
                cy = box['y'] + box['height'] / 2

                # Move cursor
                await page.evaluate(f'''
                    const c = document.getElementById('visual-cursor');
                    if (c) {{
                        c.style.left = '{cx}px';
                        c.style.top = '{cy}px';
                    }}
                ''')

                # Scroll element into view
                await el.scroll_into_view_if_needed()
                await asyncio.sleep(0.4)

                # Highlight
                await page.evaluate('''(el) => {
                    el.style.transition = 'all 0.3s ease';
                    el.style.boxShadow = '0 0 0 3px rgba(59, 130, 246, 0.5), 0 0 20px rgba(59, 130, 246, 0.3)';
                    el.style.transform = 'scale(1.02)';
                }''', el)

                self._log_action("highlight", f"Highlighting element {i+1}", scene_index=index)
                await asyncio.sleep(time_per_element * 0.7)

                # Remove highlight
                await page.evaluate('''(el) => {
                    el.style.boxShadow = '';
                    el.style.transform = '';
                }''', el)

                await asyncio.sleep(time_per_element * 0.3)

            except Exception:
                continue

        # Smooth scroll through page
        remaining_time = scene.duration - (time.time() - self.start_time)
        if remaining_time > 2:
            scroll_height = await page.evaluate('document.body.scrollHeight - window.innerHeight')
            if scroll_height > 100:
                steps = int(remaining_time * 2)
                for i in range(steps):
                    await page.evaluate(f'window.scrollBy(0, {scroll_height / steps})')

                    # Move cursor slightly
                    x_offset = 500 + (i % 5) * 60
                    y_offset = 350 + (i % 3) * 40
                    await page.evaluate(f'''
                        const c = document.getElementById('visual-cursor');
                        if (c) {{
                            c.style.left = '{x_offset}px';
                            c.style.top = '{y_offset}px';
                        }}
                    ''')

                    await asyncio.sleep(remaining_time / steps)

    async def _execute_demo_steps(self, page: Page, steps: list[dict], index: int):
        """Execute predefined demo steps."""
        for step in steps:
            action = step.get("action")
            selector = step.get("selector")
            value = step.get("value", "")
            wait = step.get("wait", 1)

            try:
                if action == "click":
                    el = await page.wait_for_selector(selector, timeout=5000)
                    if el:
                        await el.click()
                        self._log_action("click", step.get("description", "Clicked"), scene_index=index)

                elif action == "type":
                    el = await page.wait_for_selector(selector, timeout=5000)
                    if el:
                        await el.click()
                        for char in value:
                            await page.keyboard.type(char, delay=40)
                        self._log_action("type", step.get("description", f"Typed: {value[:30]}"), scene_index=index)

                elif action == "scroll":
                    await page.evaluate(f"window.scrollBy(0, {value or 500})")
                    self._log_action("scroll", "Scrolled", scene_index=index)

                elif action == "wait":
                    await asyncio.sleep(float(value or wait))

                await asyncio.sleep(wait)
            except Exception as e:
                self._log_action("error", f"Step failed: {str(e)}", scene_index=index)

    async def _record_article_scene(self, project_id: str, index: int,
                                     scene: Scene) -> SceneRecording:
        """Record an article with scrolling and highlighting."""
        # Use the same interactive browse logic
        video_file = RECORDINGS_DIR / f"{project_id}_scene{index:03d}.{RECORDING_FORMAT}"

        self.current_timestamps = []
        self.start_time = time.time()

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={"width": VIDEO_WIDTH, "height": VIDEO_HEIGHT},
                    record_video_dir=str(RECORDINGS_DIR),
                    record_video_size={"width": VIDEO_WIDTH, "height": VIDEO_HEIGHT}
                )

                page = await context.new_page()

                self._log_action("navigate", f"Opening: {scene.url}", scene_index=index)
                await page.goto(scene.url, wait_until="domcontentloaded", timeout=60000)
                await asyncio.sleep(2)

                # Use interactive browsing
                await self._interactive_browse(page, scene, index)

                # Ensure full duration
                elapsed = time.time() - self.start_time
                remaining = scene.duration - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)

                self._log_action("scene_end", "Article scene completed", scene_index=index)

                video = page.video
                temp_path = await video.path() if video else None

                await context.close()
                await browser.close()

                if temp_path and Path(temp_path).exists():
                    Path(temp_path).rename(video_file)

                return SceneRecording(
                    scene_index=index,
                    scene_type="article",
                    video_path=str(video_file),
                    duration=time.time() - self.start_time,
                    timestamps=self.current_timestamps.copy(),
                    success=True
                )

        except Exception as e:
            return SceneRecording(
                scene_index=index,
                scene_type="article",
                video_path="",
                duration=0,
                timestamps=[],
                success=False,
                error=str(e)
            )

    def _use_existing_video(self, project_id: str, index: int,
                            scene: Scene) -> SceneRecording:
        """Use a pre-recorded video file."""
        if not scene.video_path or not Path(scene.video_path).exists():
            return SceneRecording(
                scene_index=index,
                scene_type="video",
                video_path="",
                duration=0,
                timestamps=[],
                success=False,
                error=f"Video not found: {scene.video_path}"
            )

        try:
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                scene.video_path
            ], capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
        except Exception:
            duration = scene.duration

        return SceneRecording(
            scene_index=index,
            scene_type="video",
            video_path=scene.video_path,
            duration=duration,
            timestamps=[{"time_seconds": 0, "action_type": "video", "description": scene.description}],
            success=True
        )


async def record_from_manifest(
    manifest_path: str,
    recorder: RecorderType = "auto"
) -> dict:
    """
    Convenience function to record all scenes from a manifest.

    Args:
        manifest_path: Path to the scene manifest YAML/JSON
        recorder: Recording backend to use:
            - "screen_capture": Real screen capture (requires display)
            - "rendered": PIL-rendered frames (works headless)
            - "auto": Automatically selects best option
    """
    scene_recorder = SceneRecorder(default_recorder=recorder)
    result = await scene_recorder.record_manifest(manifest_path)

    return {
        "project_id": result.project_id,
        "scenes_file": result.scenes_file,
        "total_duration": result.total_duration,
        "scene_count": len(result.scene_recordings),
        "success": result.success,
        "scenes": [asdict(s) for s in result.scene_recordings]
    }
