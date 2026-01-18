#!/usr/bin/env python3
"""
Record a tutorial using the TRUE authentic recorder.

This script:
1. Reads the manifest
2. Records each scene using real command execution
3. Creates actual image frames for each scene
4. Combines frames into authentic videos
"""
import sys
import os
import json
import time
import subprocess
import asyncio
from pathlib import Path

# Add modules to path
sys.path.insert(0, os.path.dirname(__file__))

from modules.true_recorder import TrueTerminalRecorder, TrueRecordingResult
from modules.scenes import SceneManifest, load_manifest, Scene, SceneType
from playwright.async_api import async_playwright


def record_terminal_scene(
    project_id: str,
    scene_index: int,
    commands,
    duration: float
) -> TrueRecordingResult:
    """Record a terminal scene with TRUE authenticity."""
    recorder = TrueTerminalRecorder()
    return recorder.record_session(project_id, scene_index, commands, duration)


async def record_browser_scene(
    project_id: str,
    scene_index: int,
    url: str,
    duration: float
) -> TrueRecordingResult:
    """Record a browser scene with REAL interactions."""
    recordings_dir = Path("output/recordings")
    recordings_dir.mkdir(parents=True, exist_ok=True)
    video_path = recordings_dir / f"{project_id}_scene{scene_index:03d}.webm"

    timestamps = []
    start_time = time.time()

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                record_video_dir=str(recordings_dir),
                record_video_size={"width": 1920, "height": 1080}
            )
            page = await context.new_page()

            timestamps.append({
                "time_seconds": 0,
                "action_type": "navigate",
                "description": f"Opening {url}"
            })

            await page.goto(url, wait_until="domcontentloaded", timeout=30000)

            # Interactive browsing
            await page.evaluate('''
                const cursor = document.createElement('div');
                cursor.id = 'cursor';
                cursor.style.cssText = `
                    position: fixed;
                    width: 16px;
                    height: 16px;
                    background: white;
                    border-radius: 50%;
                    pointer-events: none;
                    z-index: 999999;
                    transition: left 0.2s, top 0.2s;
                    box-shadow: 0 0 10px rgba(255,255,255,0.8);
                `;
                document.body.appendChild(cursor);
            ''')

            # Find and highlight elements
            selectors = ['h1', 'h2', 'p', 'button', 'a[href]']
            time_per_element = (duration - 3) / 10

            for sel in selectors:
                try:
                    elements = await page.query_selector_all(sel)
                    for el in elements[:3]:
                        box = await el.bounding_box()
                        if box:
                            cx = box['x'] + box['width'] / 2
                            cy = box['y'] + box['height'] / 2
                            await page.evaluate(f'''
                                const c = document.getElementById('cursor');
                                c.style.left = '{cx}px';
                                c.style.top = '{cy}px';
                            ''')
                            await el.evaluate('''(el) => {
                                el.style.outline = '2px solid white';
                                el.style.outlineOffset = '4px';
                            }''')
                            await asyncio.sleep(time_per_element * 0.5)
                            await el.evaluate('''(el) => {
                                el.style.outline = '';
                            }''')
                            await asyncio.sleep(time_per_element * 0.3)
                except:
                    continue

                if time.time() - start_time > duration - 5:
                    break

            video = page.video
            if video:
                temp_path = await video.path()
                if temp_path and Path(temp_path).exists():
                    Path(temp_path).rename(video_path)

            await context.close()
            await browser.close()

        return TrueRecordingResult(
            scene_index=scene_index,
            scene_type="browser",
            video_path=str(video_path),
            duration=time.time() - start_time,
            timestamps=timestamps,
            success=True
        )

    except Exception as e:
        return TrueRecordingResult(
            scene_index=scene_index,
            scene_type="browser",
            video_path="",
            duration=0,
            timestamps=[],
            success=False,
            error=str(e)
        )


async def record_manifest_with_true_recorder(manifest_path: str):
    """Record all scenes from manifest using TRUE recorder."""
    manifest = load_manifest(manifest_path)

    recordings = []
    total_duration = 0

    print(f"\n Recording with TRUE Authentic Recorder")
    print(f" Project: {manifest.project_id}")
    print(f" Scenes: {len(manifest.scenes)}\n")

    for i, scene in enumerate(manifest.scenes):
        print(f" Scene {i+1}/{len(manifest.scenes)}: {scene.type.value}")

        if scene.type == SceneType.TERMINAL:
            commands = scene.commands or ([scene.command] if scene.command else [])
            result = record_terminal_scene(
                manifest.project_id,
                i,
                commands,
                scene.duration
            )

        elif scene.type in [SceneType.BROWSER, SceneType.ARTICLE]:
            result = await record_browser_scene(
                manifest.project_id,
                i,
                scene.url,
                scene.duration
            )
        else:
            result = TrueRecordingResult(
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
            print(f"   ✓ Success: {Path(result.video_path).name if result.video_path else 'No video'}")
        else:
            print(f"   ✗ Failed: {result.error}")

    # Save scene data
    scenes_file = Path("output/recordings") / f"{manifest.project_id}_scenes.json"
    scenes_data = {
        "project_id": manifest.project_id,
        "title": manifest.title,
        "topic": manifest.topic,
        "total_duration": total_duration,
        "scene_count": len(recordings),
        "scenes": [
            {
                "scene_index": r.scene_index,
                "scene_type": r.scene_type,
                "video_path": r.video_path,
                "duration": r.duration,
                "success": r.success,
                "error": r.error
            }
            for r in recordings
        ]
    }

    with open(scenes_file, 'w') as f:
        json.dump(scenes_data, f, indent=2)

    print(f"\n Recording Complete!")
    print(f" Total duration: {total_duration:.1f}s")
    print(f" Scenes file: {scenes_file}")
    print(f"\nNext: python main.py scene-script --project {manifest.project_id}")

    return recordings


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python record_true.py <manifest.yaml>")
        sys.exit(1)

    asyncio.run(record_manifest_with_true_recorder(sys.argv[1]))
