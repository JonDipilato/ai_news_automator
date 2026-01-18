#!/usr/bin/env python3
"""
AI News Video Automator - CLI Orchestrator

Fully automated pipeline to create AI/tech tutorial videos.
Demo-First workflow: Record browser demo -> Generate narration -> Assemble video
"""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    validate_api_keys, get_project_path,
    RECORDINGS_DIR, REVIEW_DIR, PUBLISHED_DIR
)

console = Console()


def generate_project_id() -> str:
    """Generate unique project ID based on timestamp."""
    return datetime.now().strftime("demo_%Y%m%d_%H%M%S")


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """AI News Video Automator - Create tutorial videos automatically."""
    pass


@cli.command()
@click.option("--url", required=True, help="URL to record")
@click.option("--project-id", default=None, help="Custom project ID (auto-generated if not provided)")
@click.option("--duration", default=60, help="Max recording duration in seconds")
@click.option("--demo-steps", default=None, help="JSON file with demo steps")
def record(url: str, project_id: str, duration: int, demo_steps: str):
    """Record a browser demo session."""
    from modules.recorder import record_demo

    if not project_id:
        project_id = generate_project_id()

    console.print(Panel(f"[bold blue]Recording Demo[/bold blue]\nURL: {url}\nProject: {project_id}"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Recording browser session...", total=None)

        result = asyncio.run(record_demo(
            url=url,
            project_id=project_id,
            duration=duration,
            steps_file=demo_steps
        ))

        progress.update(task, completed=True)

    # Display results
    table = Table(title="Recording Complete")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Project ID", project_id)
    table.add_row("Recording", result["recording"])
    table.add_row("Timestamps", result["timestamps"])
    table.add_row("Duration", f"{result['duration']:.1f}s")
    table.add_row("Actions Logged", str(result["action_count"]))

    console.print(table)
    console.print(f"\n[bold green]Next step:[/bold green] python main.py script --timestamps {result['timestamps']}")


@cli.command()
@click.option("--timestamps", required=True, help="Path to timestamps JSON from recording")
@click.option("--topic", default=None, help="Optional topic description for better narration")
def script(timestamps: str, topic: str):
    """Generate narration script from recording timestamps."""
    from modules.scripter import generate_script

    # Validate API keys
    missing = validate_api_keys()
    if "ANTHROPIC_API_KEY" in missing:
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY not configured in .env")
        return

    console.print(Panel("[bold blue]Generating Script[/bold blue]"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating narration with Claude...", total=None)

        result = generate_script(timestamps, topic)

        progress.update(task, completed=True)

    # Display results
    table = Table(title="Script Generated")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Title", result["title"])
    table.add_row("Segments", str(len(result["segments"])))
    table.add_row("Word Count", str(result["word_count"]))
    table.add_row("Duration", f"{result['total_duration']:.1f}s")

    console.print(table)

    # Show hook preview
    console.print(Panel(result["hook"], title="Hook Preview", border_style="yellow"))

    script_path = RECORDINGS_DIR / f"{result['project_id']}_script.json"
    console.print(f"\n[bold green]Next step:[/bold green] python main.py voice --script {script_path}")


@cli.command()
@click.option("--script", "script_file", required=True, help="Path to script JSON")
@click.option("--segmented", is_flag=True, help="Generate separate audio per segment")
def voice(script_file: str, segmented: bool):
    """Generate voice audio from script using OpenAI TTS."""
    from modules.voice import generate_voice

    # Validate API keys
    missing = validate_api_keys()
    if "OPENAI_API_KEY" in missing:
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not configured in .env")
        return

    console.print(Panel("[bold blue]Generating Voice Audio[/bold blue]"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating audio with OpenAI TTS...", total=None)

        result = generate_voice(script_file, segmented)

        progress.update(task, completed=True)

    if segmented:
        console.print(f"[green]Generated {len(result['segments'])} audio segments[/green]")
    else:
        table = Table(title="Audio Generated")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Audio File", result["audio_file"])
        table.add_row("Word Count", str(result["word_count"]))
        table.add_row("Est. Duration", f"{result['estimated_duration']:.1f}s")
        table.add_row("Voice", result["voice"])

        console.print(table)

    project_id = result.get("project_id", Path(script_file).stem.replace("_script", ""))
    console.print(f"\n[bold green]Next step:[/bold green] python main.py assemble --project {project_id}")


@cli.command()
@click.option("--project", required=True, help="Project ID to assemble")
@click.option("--no-captions", is_flag=True, help="Skip caption generation")
def assemble(project: str, no_captions: bool):
    """Assemble final video from recording, audio, and captions."""
    from modules.assembler import assemble_video

    console.print(Panel(f"[bold blue]Assembling Video[/bold blue]\nProject: {project}"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Assembling video with FFmpeg...", total=None)

        result = assemble_video(project, with_captions=not no_captions)

        progress.update(task, completed=True)

    if result["success"]:
        table = Table(title="Video Assembled")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Output", result["output_file"])
        table.add_row("Duration", f"{result['duration']:.1f}s")
        table.add_row("Captions", "Yes" if result["has_captions"] else "No")
        table.add_row("Audio/Video Sync", f"{result['audio_video_sync']:.1%}")

        console.print(table)
        console.print(f"\n[bold yellow]Review video at:[/bold yellow] {result['output_file']}")
        console.print(f"[bold green]When ready:[/bold green] python main.py approve --project {project}")
    else:
        console.print(f"[bold red]Assembly failed:[/bold red] {result['message']}")


@cli.command()
@click.option("--project", required=True, help="Project ID to approve")
def approve(project: str):
    """Approve a video for publishing after review."""
    meta_file = REVIEW_DIR / f"{project}_meta.json"

    if not meta_file.exists():
        console.print(f"[bold red]Error:[/bold red] Project {project} not found in review queue")
        return

    with open(meta_file) as f:
        meta = json.load(f)

    # Show current metadata
    table = Table(title=f"Approving: {project}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Title", meta["title"])
    table.add_row("Duration", f"{meta['duration_seconds']:.1f}s")
    table.add_row("Video", meta["video_file"])

    console.print(table)

    if click.confirm("Approve this video for publishing?"):
        meta["approved"] = True
        meta["approved_at"] = datetime.now().isoformat()

        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        console.print("[bold green]Video approved![/bold green]")
        console.print(f"[bold green]To publish:[/bold green] python main.py publish --project {project}")
    else:
        console.print("[yellow]Approval cancelled[/yellow]")


@cli.command()
def status():
    """Show status of all projects in the pipeline."""
    # Count files in each stage
    recordings = list(RECORDINGS_DIR.glob("*.webm"))
    review = list(REVIEW_DIR.glob("*.mp4"))
    published = list(PUBLISHED_DIR.glob("*.mp4"))

    table = Table(title="Pipeline Status")
    table.add_column("Stage", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Projects", style="yellow")

    table.add_row(
        "Recordings",
        str(len(recordings)),
        ", ".join(r.stem for r in recordings[:3]) + ("..." if len(recordings) > 3 else "")
    )
    table.add_row(
        "Review Queue",
        str(len(review)),
        ", ".join(r.stem for r in review[:3]) + ("..." if len(review) > 3 else "")
    )
    table.add_row(
        "Published",
        str(len(published)),
        ", ".join(r.stem for r in published[:3]) + ("..." if len(published) > 3 else "")
    )

    console.print(table)


@cli.command()
@click.option("--url", required=True, help="URL to create video about")
@click.option("--topic", default=None, help="Topic description")
@click.option("--duration", default=60, help="Max recording duration")
def create(url: str, topic: str, duration: int):
    """Full pipeline: Record -> Script -> Voice -> Assemble."""
    from modules.recorder import record_demo
    from modules.scripter import generate_script
    from modules.voice import generate_voice
    from modules.assembler import assemble_video

    # Validate API keys
    missing = validate_api_keys()
    if missing:
        console.print(f"[bold red]Missing API keys:[/bold red] {', '.join(missing)}")
        console.print("Configure these in your .env file")
        return

    project_id = generate_project_id()

    console.print(Panel(
        f"[bold blue]Full Pipeline[/bold blue]\n"
        f"URL: {url}\n"
        f"Project: {project_id}",
        title="AI News Video Automator"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Step 1: Record
        task1 = progress.add_task("[1/4] Recording browser demo...", total=None)
        record_result = asyncio.run(record_demo(url, project_id, duration))
        progress.update(task1, completed=True, description="[1/4] Recording complete")

        # Step 2: Script
        task2 = progress.add_task("[2/4] Generating script...", total=None)
        script_result = generate_script(record_result["timestamps"], topic)
        progress.update(task2, completed=True, description="[2/4] Script generated")

        # Step 3: Voice
        script_path = RECORDINGS_DIR / f"{project_id}_script.json"
        task3 = progress.add_task("[3/4] Generating voice...", total=None)
        voice_result = generate_voice(str(script_path))
        progress.update(task3, completed=True, description="[3/4] Voice generated")

        # Step 4: Assemble
        task4 = progress.add_task("[4/4] Assembling video...", total=None)
        assemble_result = assemble_video(project_id)
        progress.update(task4, completed=True, description="[4/4] Video assembled")

    if assemble_result["success"]:
        console.print(Panel(
            f"[bold green]Video Created Successfully![/bold green]\n\n"
            f"Title: {script_result['title']}\n"
            f"Duration: {assemble_result['duration']:.1f}s\n"
            f"Output: {assemble_result['output_file']}\n\n"
            f"[yellow]Review the video, then run:[/yellow]\n"
            f"python main.py approve --project {project_id}",
            title="Complete",
            border_style="green"
        ))
    else:
        console.print(f"[bold red]Pipeline failed:[/bold red] {assemble_result['message']}")


@cli.command()
def check():
    """Check system requirements and API keys."""
    import shutil

    table = Table(title="System Check")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")

    # Check FFmpeg
    ffmpeg = shutil.which("ffmpeg")
    table.add_row(
        "FFmpeg",
        "[green]OK[/green]" if ffmpeg else "[red]Missing[/red]",
        ffmpeg or "Install: apt install ffmpeg"
    )

    # Check FFprobe
    ffprobe = shutil.which("ffprobe")
    table.add_row(
        "FFprobe",
        "[green]OK[/green]" if ffprobe else "[red]Missing[/red]",
        ffprobe or "Included with FFmpeg"
    )

    # Check API keys
    from config.settings import OPENAI_API_KEY, ANTHROPIC_API_KEY, TAVILY_API_KEY

    table.add_row(
        "OpenAI API Key",
        "[green]Configured[/green]" if OPENAI_API_KEY else "[red]Missing[/red]",
        "For TTS and Whisper"
    )

    table.add_row(
        "Anthropic API Key",
        "[green]Configured[/green]" if ANTHROPIC_API_KEY else "[red]Missing[/red]",
        "For script generation"
    )

    table.add_row(
        "Tavily API Key",
        "[green]Configured[/green]" if TAVILY_API_KEY else "[yellow]Optional[/yellow]",
        "For news discovery (Phase 2)"
    )

    # Check Playwright
    try:
        from playwright.sync_api import sync_playwright
        table.add_row("Playwright", "[green]OK[/green]", "Browser automation ready")
    except ImportError:
        table.add_row(
            "Playwright",
            "[red]Missing[/red]",
            "Run: pip install playwright && playwright install"
        )

    console.print(table)


@cli.command()
@click.option("--category", default="ai_tools", help="Category to search (ai_tools, tutorials, news)")
@click.option("--count", default=5, help="Number of topics to discover")
def discover(category: str, count: int):
    """Discover trending AI/tech topics for video content."""
    from modules.discovery import discover_topics

    console.print(Panel(f"[bold blue]Discovering Topics[/bold blue]\nCategory: {category}"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Searching for trending topics...", total=None)

        topics = discover_topics(category, count)

        progress.update(task, completed=True)

    if not topics:
        console.print("[yellow]No topics found. Check your Tavily API key.[/yellow]")
        return

    table = Table(title=f"Discovered Topics ({len(topics)})")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Title", style="green", max_width=50)
    table.add_column("Source", style="yellow")
    table.add_column("Score", style="magenta")

    for i, topic in enumerate(topics, 1):
        table.add_row(
            str(i),
            topic["title"][:50],
            topic["source"],
            f"{topic['score']:.2f}"
        )

    console.print(table)

    # Show first topic details
    if topics:
        first = topics[0]
        console.print(Panel(
            f"[bold]{first['title']}[/bold]\n\n"
            f"URL: {first['url']}\n\n"
            f"{first['summary'][:300]}...",
            title="Top Result",
            border_style="green"
        ))
        console.print(f"\n[bold green]Create video:[/bold green] python main.py create --url \"{first['url']}\"")


@cli.command()
@click.option("--project", required=True, help="Project ID to publish")
@click.option("--schedule-day", default=None, help="Day to publish (tuesday, friday)")
@click.option("--schedule-time", default=None, help="Time to publish (e.g., 14:00)")
def publish(project: str, schedule_day: str, schedule_time: str):
    """Publish an approved video to YouTube."""
    from modules.publisher import publish_video

    console.print(Panel(f"[bold blue]Publishing to YouTube[/bold blue]\nProject: {project}"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Uploading to YouTube...", total=None)

        result = publish_video(project, schedule_day, schedule_time)

        progress.update(task, completed=True)

    if result["success"]:
        table = Table(title="Published Successfully")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Video ID", result["video_id"])
        table.add_row("URL", result["url"])
        if result["scheduled_time"]:
            table.add_row("Scheduled For", result["scheduled_time"])

        console.print(table)
        console.print(f"\n[bold green]Watch at:[/bold green] {result['url']}")
    else:
        console.print(f"[bold red]Publish failed:[/bold red] {result['message']}")


@cli.command("auth-youtube")
def auth_youtube():
    """Authenticate with YouTube API."""
    from modules.publisher import YouTubePublisher

    console.print(Panel("[bold blue]YouTube Authentication[/bold blue]"))

    publisher = YouTubePublisher()

    if publisher.authenticate():
        info = publisher.get_channel_info()
        if "error" not in info:
            table = Table(title="Authenticated Channel")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Channel", info["title"])
            table.add_row("Subscribers", info["subscribers"])
            table.add_row("Videos", info["videos"])

            console.print(table)
            console.print("[bold green]YouTube authentication successful![/bold green]")
        else:
            console.print(f"[yellow]Authenticated but: {info['error']}[/yellow]")
    else:
        console.print("[bold red]Authentication failed.[/bold red]")
        console.print("Make sure credentials.json exists in the project root.")
        console.print("Get it from: https://console.cloud.google.com/apis/credentials")


# ============================================================================
# MULTI-SCENE COMMANDS (New Interactive Tutorial System)
# ============================================================================

@cli.command("scene-example")
def scene_example():
    """Show an example scene manifest for multi-scene videos."""
    from modules.scenes import get_example_manifest

    console.print(Panel(
        "[bold blue]Example Scene Manifest[/bold blue]\n"
        "Use this as a template for creating multi-scene tutorial videos",
        title="Multi-Scene System"
    ))

    example = get_example_manifest()
    console.print(example)

    console.print("\n[bold green]Save this to a .yaml file and run:[/bold green]")
    console.print("python main.py scene-create --manifest your_manifest.yaml")


@cli.command("scene-init")
@click.option("--type", "manifest_type", default="tutorial",
              type=click.Choice(["tutorial", "news"]),
              help="Type of video (tutorial or news compilation)")
@click.option("--project-id", default=None, help="Custom project ID")
@click.option("--output", default=None, help="Output manifest file path")
def scene_init(manifest_type: str, project_id: str, output: str):
    """Create a new scene manifest interactively."""
    from modules.scenes import SceneManifest, Scene, SceneType, TransitionType

    if not project_id:
        project_id = generate_project_id()

    console.print(Panel(f"[bold blue]Creating Scene Manifest[/bold blue]\nType: {manifest_type}"))

    # Interactive input
    title = click.prompt("Video title", default=f"New {manifest_type.title()} Video")
    topic = click.prompt("Topic/description", default="")

    scenes = []
    scene_num = 1

    console.print("\n[yellow]Add scenes (enter 'done' to finish):[/yellow]")

    while True:
        console.print(f"\n[cyan]Scene {scene_num}:[/cyan]")

        scene_type = click.prompt(
            "Type",
            type=click.Choice(["terminal", "browser", "article", "done"]),
            default="browser" if manifest_type == "news" else "terminal"
        )

        if scene_type == "done":
            break

        if scene_type == "terminal":
            commands = click.prompt("Commands (comma-separated)", default="")
            command_list = [c.strip() for c in commands.split(",") if c.strip()]
            scene = Scene(
                type=SceneType.TERMINAL,
                commands=command_list,
                duration=click.prompt("Duration (seconds)", type=int, default=20),
                description=click.prompt("Description", default="Terminal commands"),
                transition_in=TransitionType.WIPE_LEFT
            )
        else:
            url = click.prompt("URL")
            scene = Scene(
                type=SceneType.BROWSER if scene_type == "browser" else SceneType.ARTICLE,
                url=url,
                duration=click.prompt("Duration (seconds)", type=int, default=30),
                description=click.prompt("Description", default=f"Showing {url}"),
                transition_in=TransitionType.ZOOM_IN if scene_num > 1 else TransitionType.FADE
            )

        scenes.append(scene)
        scene_num += 1

    if not scenes:
        console.print("[yellow]No scenes added. Exiting.[/yellow]")
        return

    # Create manifest
    manifest = SceneManifest(
        project_id=project_id,
        title=title,
        topic=topic,
        scenes=scenes,
        overall_theme=manifest_type,
        style="energetic",
        include_subscribe_cta=True
    )

    # Save manifest
    if output:
        output_path = Path(output)
    else:
        output_path = RECORDINGS_DIR / f"{project_id}_manifest.yaml"

    manifest.save(output_path)

    console.print(f"\n[bold green]Manifest saved to:[/bold green] {output_path}")
    console.print(f"[bold green]To record:[/bold green] python main.py scene-record --manifest {output_path}")


@cli.command("scene-record")
@click.option("--manifest", required=True, help="Path to scene manifest YAML/JSON")
def scene_record(manifest: str):
    """Record all scenes from a manifest file."""
    from modules.scene_recorder import record_from_manifest

    console.print(Panel(f"[bold blue]Recording Scenes[/bold blue]\nManifest: {manifest}"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Recording scenes...", total=None)

        result = asyncio.run(record_from_manifest(manifest))

        progress.update(task, completed=True)

    if result["success"]:
        table = Table(title="Recording Complete")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Project ID", result["project_id"])
        table.add_row("Scenes Recorded", str(result["scene_count"]))
        table.add_row("Total Duration", f"{result['total_duration']:.1f}s")
        table.add_row("Scenes File", result["scenes_file"])

        console.print(table)

        # Show scene breakdown
        scene_table = Table(title="Scene Breakdown")
        scene_table.add_column("#", style="cyan")
        scene_table.add_column("Type", style="yellow")
        scene_table.add_column("Duration", style="green")
        scene_table.add_column("Status", style="magenta")

        for i, scene in enumerate(result.get("scenes", []), 1):
            status = "[green]OK[/green]" if scene.get("success") else f"[red]{scene.get('error', 'Failed')}[/red]"
            scene_table.add_row(
                str(i),
                scene.get("scene_type", "unknown"),
                f"{scene.get('duration', 0):.1f}s",
                status
            )

        console.print(scene_table)

        console.print(f"\n[bold green]Next step:[/bold green] python main.py scene-script --project {result['project_id']}")
    else:
        console.print("[bold red]Recording failed. Check scene errors above.[/bold red]")


@cli.command("scene-script")
@click.option("--project", required=True, help="Project ID to generate script for")
@click.option("--topic", default=None, help="Optional topic override")
def scene_script(project: str, topic: str):
    """Generate script from recorded scenes."""
    from modules.scripter import ScriptGenerator

    scenes_file = RECORDINGS_DIR / f"{project}_scenes.json"

    if not scenes_file.exists():
        console.print(f"[bold red]Error:[/bold red] Scenes file not found: {scenes_file}")
        console.print("Run scene-record first.")
        return

    with open(scenes_file) as f:
        scenes_data = json.load(f)

    manifest = scenes_data.get("manifest", {})
    topic = topic or manifest.get("topic", "")

    # Combine all timestamps from all scenes
    all_actions = []
    for scene in scenes_data.get("scenes", []):
        for ts in scene.get("timestamps", []):
            ts["scene_index"] = scene.get("scene_index", 0)
            ts["scene_type"] = scene.get("scene_type", "unknown")
            all_actions.append(ts)

    # Create combined timestamps file
    combined_timestamps = {
        "project_id": project,
        "url": manifest.get("title", project),
        "duration_seconds": scenes_data.get("total_duration", 60),
        "actions": sorted(all_actions, key=lambda x: x.get("time_seconds", 0))
    }

    timestamps_file = RECORDINGS_DIR / f"{project}_timestamps.json"
    with open(timestamps_file, "w") as f:
        json.dump(combined_timestamps, f, indent=2)

    console.print(Panel("[bold blue]Generating Script for Multi-Scene Video[/bold blue]"))

    # Validate API keys
    missing = validate_api_keys()
    if "ANTHROPIC_API_KEY" in missing:
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY not configured in .env")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating narration with Claude...", total=None)

        from modules.scripter import generate_script
        result = generate_script(str(timestamps_file), topic)

        progress.update(task, completed=True)

    table = Table(title="Script Generated")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Title", result["title"])
    table.add_row("Segments", str(len(result["segments"])))
    table.add_row("Word Count", str(result["word_count"]))

    console.print(table)
    console.print(Panel(result["hook"], title="Hook Preview", border_style="yellow"))
    console.print(Panel(result["outro"], title="Outro Preview", border_style="green"))

    script_path = RECORDINGS_DIR / f"{project}_script.json"
    console.print(f"\n[bold green]Next step:[/bold green] python main.py voice --script {script_path}")


@cli.command("scene-assemble")
@click.option("--project", required=True, help="Project ID to assemble")
@click.option("--no-captions", is_flag=True, help="Skip caption generation")
def scene_assemble(project: str, no_captions: bool):
    """Assemble multi-scene video with transitions."""
    from modules.scene_assembler import assemble_scenes

    console.print(Panel(f"[bold blue]Assembling Multi-Scene Video[/bold blue]\nProject: {project}"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Assembling scenes with transitions...", total=None)

        result = assemble_scenes(project, with_captions=not no_captions)

        progress.update(task, completed=True)

    if result["success"]:
        table = Table(title="Video Assembled")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Output", result["output_file"])
        table.add_row("Duration", f"{result['duration']:.1f}s")
        table.add_row("Scenes", str(result["scene_count"]))
        table.add_row("Transitions", "Yes" if result["has_transitions"] else "No")
        table.add_row("Captions", "Yes" if result["has_captions"] else "No")

        console.print(table)
        console.print(f"\n[bold yellow]Review video at:[/bold yellow] {result['output_file']}")
        console.print(f"[bold green]When ready:[/bold green] python main.py approve --project {project}")
    else:
        console.print(f"[bold red]Assembly failed:[/bold red] {result['message']}")


@cli.command("scene-create")
@click.option("--manifest", required=True, help="Path to scene manifest YAML/JSON")
@click.option("--topic", default=None, help="Optional topic override for script generation")
def scene_create(manifest: str, topic: str):
    """Full multi-scene pipeline: Record scenes -> Script -> Voice -> Assemble with transitions."""
    from modules.scene_recorder import record_from_manifest
    from modules.scripter import generate_script
    from modules.voice import generate_voice
    from modules.scene_assembler import assemble_scenes
    from modules.scenes import load_manifest

    # Validate API keys
    missing = validate_api_keys()
    if missing:
        console.print(f"[bold red]Missing API keys:[/bold red] {', '.join(missing)}")
        console.print("Configure these in your .env file")
        return

    manifest_data = load_manifest(manifest)
    project_id = manifest_data.project_id

    console.print(Panel(
        f"[bold blue]Multi-Scene Video Pipeline[/bold blue]\n"
        f"Title: {manifest_data.title}\n"
        f"Scenes: {len(manifest_data.scenes)}\n"
        f"Project: {project_id}",
        title="AI News Video Automator"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Step 1: Record all scenes
        task1 = progress.add_task("[1/4] Recording scenes...", total=None)
        record_result = asyncio.run(record_from_manifest(manifest))

        if not record_result["success"]:
            progress.update(task1, description="[1/4] Recording failed")
            console.print("[bold red]Scene recording failed.[/bold red]")
            return

        progress.update(task1, completed=True, description=f"[1/4] Recorded {record_result['scene_count']} scenes")

        # Create combined timestamps for script generation
        scenes_file = Path(record_result["scenes_file"])
        with open(scenes_file) as f:
            scenes_data = json.load(f)

        all_actions = []
        for scene in scenes_data.get("scenes", []):
            for ts in scene.get("timestamps", []):
                ts["scene_index"] = scene.get("scene_index", 0)
                all_actions.append(ts)

        combined_timestamps = {
            "project_id": project_id,
            "url": manifest_data.title,
            "duration_seconds": record_result["total_duration"],
            "actions": sorted(all_actions, key=lambda x: x.get("time_seconds", 0))
        }

        timestamps_file = RECORDINGS_DIR / f"{project_id}_timestamps.json"
        with open(timestamps_file, "w") as f:
            json.dump(combined_timestamps, f, indent=2)

        # Step 2: Generate script
        task2 = progress.add_task("[2/4] Generating script...", total=None)
        script_result = generate_script(str(timestamps_file), topic or manifest_data.topic)
        progress.update(task2, completed=True, description="[2/4] Script generated")

        # Step 3: Generate voice
        script_path = RECORDINGS_DIR / f"{project_id}_script.json"
        task3 = progress.add_task("[3/4] Generating voice...", total=None)
        voice_result = generate_voice(str(script_path))
        progress.update(task3, completed=True, description="[3/4] Voice generated")

        # Step 4: Assemble with transitions
        task4 = progress.add_task("[4/4] Assembling with transitions...", total=None)
        assemble_result = assemble_scenes(project_id)
        progress.update(task4, completed=True, description="[4/4] Video assembled")

    if assemble_result["success"]:
        console.print(Panel(
            f"[bold green]Multi-Scene Video Created![/bold green]\n\n"
            f"Title: {script_result['title']}\n"
            f"Scenes: {assemble_result['scene_count']}\n"
            f"Duration: {assemble_result['duration']:.1f}s\n"
            f"Transitions: {'Yes' if assemble_result['has_transitions'] else 'No'}\n"
            f"Output: {assemble_result['output_file']}\n\n"
            f"[yellow]Review the video, then run:[/yellow]\n"
            f"python main.py approve --project {project_id}",
            title="Complete",
            border_style="green"
        ))
    else:
        console.print(f"[bold red]Pipeline failed:[/bold red] {assemble_result['message']}")


if __name__ == "__main__":
    cli()
