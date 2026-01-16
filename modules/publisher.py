"""
YouTube publishing module.
Handles OAuth authentication and video uploads with scheduling.
"""
import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from config.settings import (
    BASE_DIR, REVIEW_DIR, PUBLISHED_DIR,
    YOUTUBE_CATEGORY_ID, YOUTUBE_PRIVACY,
    PUBLISH_DAYS, PUBLISH_TIME
)


SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"


@dataclass
class PublishResult:
    """Result of video publishing."""
    project_id: str
    video_id: Optional[str]
    url: Optional[str]
    scheduled_time: Optional[str]
    success: bool
    message: str


class YouTubePublisher:
    """Publishes videos to YouTube with scheduling."""

    def __init__(self):
        self.youtube = None

    def authenticate(self) -> bool:
        """
        Authenticate with YouTube API.
        Requires credentials.json from Google Cloud Console.

        Returns:
            True if authentication successful
        """
        creds = None

        # Load existing token
        if TOKEN_FILE.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not CREDENTIALS_FILE.exists():
                    return False

                flow = InstalledAppFlow.from_client_secrets_file(
                    str(CREDENTIALS_FILE), SCOPES
                )
                creds = flow.run_local_server(port=8080)

            # Save token for future use
            with open(TOKEN_FILE, "w") as f:
                f.write(creds.to_json())

        self.youtube = build("youtube", "v3", credentials=creds)
        return True

    def publish(self, project_id: str,
                schedule_day: Optional[str] = None,
                schedule_time: Optional[str] = None) -> PublishResult:
        """
        Publish a video to YouTube.

        Args:
            project_id: Project ID to publish
            schedule_day: Day to publish (e.g., "tuesday")
            schedule_time: Time to publish (e.g., "14:00")

        Returns:
            PublishResult with upload details
        """
        if not self.youtube:
            if not self.authenticate():
                return PublishResult(
                    project_id=project_id,
                    video_id=None,
                    url=None,
                    scheduled_time=None,
                    success=False,
                    message="YouTube authentication failed. Check credentials.json"
                )

        # Load project metadata
        meta_file = REVIEW_DIR / f"{project_id}_meta.json"
        if not meta_file.exists():
            return PublishResult(
                project_id=project_id,
                video_id=None,
                url=None,
                scheduled_time=None,
                success=False,
                message=f"Project metadata not found: {meta_file}"
            )

        with open(meta_file) as f:
            meta = json.load(f)

        if not meta.get("approved"):
            return PublishResult(
                project_id=project_id,
                video_id=None,
                url=None,
                scheduled_time=None,
                success=False,
                message="Video not approved. Run: python main.py approve --project " + project_id
            )

        video_file = Path(meta["video_file"])
        if not video_file.exists():
            return PublishResult(
                project_id=project_id,
                video_id=None,
                url=None,
                scheduled_time=None,
                success=False,
                message=f"Video file not found: {video_file}"
            )

        # Calculate scheduled publish time
        scheduled_time = self._calculate_publish_time(schedule_day, schedule_time)

        # Prepare video metadata
        body = {
            "snippet": {
                "title": meta["title"],
                "description": meta["description"],
                "tags": meta.get("tags", ["tutorial", "ai"]),
                "categoryId": YOUTUBE_CATEGORY_ID
            },
            "status": {
                "privacyStatus": YOUTUBE_PRIVACY,
                "selfDeclaredMadeForKids": False
            }
        }

        # Add scheduled publish time if scheduling
        if scheduled_time and YOUTUBE_PRIVACY == "private":
            body["status"]["publishAt"] = scheduled_time.isoformat() + "Z"
            body["status"]["privacyStatus"] = "private"

        # Upload video
        try:
            media = MediaFileUpload(
                str(video_file),
                mimetype="video/mp4",
                resumable=True
            )

            request = self.youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media
            )

            response = None
            while response is None:
                status, response = request.next_chunk()

            video_id = response["id"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            # Move to published folder
            published_video = PUBLISHED_DIR / f"{project_id}.mp4"
            video_file.rename(published_video)

            # Update metadata
            meta["status"] = "published"
            meta["video_id"] = video_id
            meta["video_url"] = video_url
            meta["published_at"] = datetime.now().isoformat()
            meta["scheduled_time"] = scheduled_time.isoformat() if scheduled_time else None

            published_meta = PUBLISHED_DIR / f"{project_id}_meta.json"
            with open(published_meta, "w") as f:
                json.dump(meta, f, indent=2)

            # Remove from review queue
            meta_file.unlink()

            return PublishResult(
                project_id=project_id,
                video_id=video_id,
                url=video_url,
                scheduled_time=scheduled_time.isoformat() if scheduled_time else None,
                success=True,
                message="Video uploaded successfully"
            )

        except Exception as e:
            return PublishResult(
                project_id=project_id,
                video_id=None,
                url=None,
                scheduled_time=None,
                success=False,
                message=f"Upload failed: {str(e)}"
            )

    def _calculate_publish_time(self, day: Optional[str],
                                 time: Optional[str]) -> Optional[datetime]:
        """Calculate the next publish time based on schedule."""
        if not day:
            day = PUBLISH_DAYS[0]  # Default to first configured day
        if not time:
            time = PUBLISH_TIME

        # Parse target time
        hour, minute = map(int, time.split(":"))

        # Find next occurrence of target day
        today = datetime.now()
        days_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
        }

        target_day = days_map.get(day.lower(), 1)  # Default Tuesday
        days_ahead = target_day - today.weekday()

        if days_ahead <= 0:  # Target day already passed this week
            days_ahead += 7

        target_date = today + timedelta(days=days_ahead)
        return target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def get_channel_info(self) -> dict:
        """Get authenticated channel information."""
        if not self.youtube:
            if not self.authenticate():
                return {"error": "Not authenticated"}

        response = self.youtube.channels().list(
            part="snippet,statistics",
            mine=True
        ).execute()

        if response.get("items"):
            channel = response["items"][0]
            return {
                "id": channel["id"],
                "title": channel["snippet"]["title"],
                "subscribers": channel["statistics"].get("subscriberCount", "0"),
                "videos": channel["statistics"].get("videoCount", "0")
            }

        return {"error": "No channel found"}


def publish_video(project_id: str, schedule_day: str = None,
                  schedule_time: str = None) -> dict:
    """
    Convenience function to publish a video.

    Args:
        project_id: Project to publish
        schedule_day: Optional day to schedule
        schedule_time: Optional time to schedule

    Returns:
        Publish result as dict
    """
    publisher = YouTubePublisher()
    result = publisher.publish(project_id, schedule_day, schedule_time)

    return {
        "project_id": result.project_id,
        "video_id": result.video_id,
        "url": result.url,
        "scheduled_time": result.scheduled_time,
        "success": result.success,
        "message": result.message
    }
