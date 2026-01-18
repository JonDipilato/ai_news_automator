"""
Direct Terminal Recorder - Runs REAL commands and captures their output.

This module:
- Executes actual shell commands
- Captures real output (stdout/stderr)
- Renders authentic terminal sessions as video
- No mockups - everything is genuine
"""
import os
import json
import subprocess
import time
import asyncio
from pathlib import Path
from typing import List

from config.settings import VIDEO_WIDTH, VIDEO_HEIGHT, RECORDINGS_DIR, RECORDING_FORMAT


class DirectTerminalRecorder:
    """
    Records terminal by actually running commands and capturing their output.

    This is the most authentic approach - runs commands for real
    and renders the actual output as video.
    """

    def __init__(self, width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT):
        self.width = width
        self.height = height
        self.recordings_dir = Path(RECORDINGS_DIR)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

    def record_commands_real(
        self,
        project_id: str,
        scene_index: int,
        commands: List[str],
        duration: float
    ):
        """
        Run commands for real and capture their output.
        Then render as video.
        """
        video_path = self.recordings_dir / f"{project_id}_scene{scene_index:03d}.{RECORDING_FORMAT}"
        timestamps = []
        start_time = time.time()

        outputs = []

        for cmd in commands:
            if not cmd.strip() or cmd.strip().startswith('#'):
                outputs.append({"cmd": cmd, "output": "", "success": True})
                timestamps.append({
                    "time_seconds": round(time.time() - start_time, 2),
                    "action_type": "comment",
                    "description": f"Comment: {cmd[:50]}",
                })
                continue

            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd=os.path.expanduser("~")
                )

                output = result.stdout
                if result.stderr and not result.stdout:
                    output = result.stderr

                outputs.append({
                    "cmd": cmd,
                    "output": output,
                    "success": result.returncode == 0
                })

                timestamps.append({
                    "time_seconds": round(time.time() - start_time, 2),
                    "action_type": "command_executed",
                    "description": f"Ran: {cmd[:50]}",
                    "output": output[-100:] if output else ""
                })

            except subprocess.TimeoutExpired:
                outputs.append({"cmd": cmd, "output": "Command timed out", "success": False})
            except Exception as e:
                outputs.append({"cmd": cmd, "output": str(e), "success": False})

        # Generate HTML from real output
        html_path = Path(video_path).parent / f"{project_id}_scene{scene_index:03d}.html"
        html_content = self._generate_html_from_outputs(outputs, duration)
        html_path.write_text(html_content)

        # Convert to video
        self._html_to_video(str(html_path), str(video_path), duration, self.width, self.height)

        # Cleanup
        html_path.unlink(missing_ok=True)

        return {
            "scene_index": scene_index,
            "scene_type": "terminal",
            "video_path": str(video_path),
            "duration": time.time() - start_time,
            "timestamps": timestamps,
            "success": True
        }

    def _generate_html_from_outputs(self, outputs: list, duration: float) -> str:
        """Generate HTML that types out the real command outputs."""
        outputs_json = json.dumps(outputs)
        total_chars = sum(len(o.get("cmd", "")) + len(o.get("output", "")) for o in outputs)
        char_delay = max(15, min(60, (duration * 1000) / max(total_chars / 5, 1)))

        return f'''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #0d1117;
    font-family: 'Cascadia Code', 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 15px;
    line-height: 1.5;
    color: #c9d1d9;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    padding: 20px;
}}
.terminal {{
    width: 100%;
    max-width: 1200px;
    height: 85vh;
    background: #0d1117;
    border-radius: 12px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
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
}}
.dots {{ display: flex; gap: 8px; }}
.dot {{ width: 12px; height: 12px; border-radius: 50%; }}
.dot.red {{ background: #ff5f56; }}
.dot.yellow {{ background: #ffbd2e; }}
.dot.green {{ background: #27c93f; }}
.title {{ flex: 1; text-align: center; color: #8b949e; font-size: 13px; font-weight: 500; }}
.content {{ flex: 1; padding: 24px; overflow-y: auto; }}
.line {{ display: flex; align-items: flex-start; margin: 4px 0; min-height: 22px; }}
.prompt {{ color: #58a6ff; font-weight: 600; margin-right: 8px; white-space: nowrap; }}
.prompt span {{ color: #7ee787; }}
.command {{ color: #f0f6fc; white-space: pre-wrap; word-break: break-word; }}
.command.comment {{ color: #8b949e; font-style: italic; }}
.output {{ color: #8b949e; padding: 8px 0 8px 0; white-space: pre-wrap; font-size: 14px; line-height: 1.4; }}
.output.error {{ color: #f85149; }}
.output.success {{ color: #3fb950; }}
.cursor {{ display: inline-block; width: 8px; height: 16px; background: #58a6ff; animation: blink 1s step-end infinite; vertical-align: text-bottom; }}
@keyframes blink {{ 0%, 50% {{ opacity: 1; }} 51%, 100% {{ opacity: 0; }} }}
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
        <div class="title">Terminal — bash — 80×24</div>
    </div>
    <div class="content" id="content"></div>
</div>
<script>
const outputs = {outputs_json};
const content = document.getElementById('content');
const charDelay = {charDelay};

function sleep(ms) {{ return new Promise(r => setTimeout(r, ms)); }}

async function typeText(element, text) {{
    for (let i = 0; i < text.length; i++) {{
        element.textContent += text[i];
        await sleep(charDelay + Math.random() * 15);
        if (text[i] === ' ' && Math.random() > 0.7) await sleep(charDelay * 2);
    }}
}}

async function showOutput(text, isError) {{
    if (!text || !text.trim()) return;
    const output = document.createElement('div');
    output.className = 'output' + (isError ? ' error' : '');
    output.textContent = text;
    content.appendChild(output);
    content.scrollTop = content.scrollHeight;
    await sleep(300);
}}

function createPrompt() {{
    const prompt = document.createElement('span');
    prompt.className = 'prompt';
    prompt.innerHTML = '<span>~</span>❯';
    return prompt;
}}

async function runCommands() {{
    await sleep(600);

    for (const item of outputs) {{
        const line = document.createElement('div');
        line.className = 'line';
        line.appendChild(createPrompt());

        const cmdSpan = document.createElement('span');
        cmdSpan.className = 'command';
        if (item.cmd.trim().startsWith('#') || item.cmd.includes('echo')) {{
            cmdSpan.classList.add('comment');
        }}
        line.appendChild(cmdSpan);

        const cursor = document.createElement('span');
        cursor.className = 'cursor';
        line.appendChild(cursor);

        content.appendChild(line);

        await typeText(cmdSpan, item.cmd);
        await sleep(200);
        cursor.remove();

        if (!item.cmd.trim().startsWith('#')) {{
            await sleep(200);
            await showOutput(item.output, !item.success);
        }}

        await sleep(500);
    }}

    const finalLine = document.createElement('div');
    finalLine.className = 'line';
    finalLine.appendChild(createPrompt());
    const finalCursor = document.createElement('span');
    finalCursor.className = 'cursor';
    finalLine.appendChild(finalCursor);
    content.appendChild(finalLine);
    content.scrollTop = content.scrollHeight;
}}

runCommands();
</script>
</body>
</html>'''

    def _html_to_video(self, html_path: str, output_path: str, duration: float, width: int, height: int):
        """Convert HTML to video using Playwright."""
        from playwright.async_api import async_playwright

        async def record():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={"width": width, "height": height},
                    record_video_dir=str(Path(output_path).parent),
                    record_video_size={"width": width, "height": height}
                )
                page = await context.new_page()
                await page.goto(f"file://{html_path}")
                await asyncio.sleep(duration + 2)

                video = page.video
                if video:
                    temp_path = await video.path()
                    if temp_path and Path(temp_path).exists():
                        Path(temp_path).rename(output_path)

                await context.close()
                await browser.close()

        asyncio.run(record())


def record_live_terminal(project_id: str, scene_index: int, commands: list, duration: float) -> dict:
    """Record a terminal session with real command execution."""
    recorder = DirectTerminalRecorder()
    return recorder.record_commands_real(project_id, scene_index, commands, duration)
