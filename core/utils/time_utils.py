"""
Time and URL utility functions for YouTube RAG.
"""
import re
from typing import Optional


def extract_youtube_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats.
    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/v/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - Plain 11-character video ID
    """
    if not url:
        return None
    url = url.strip()
    
    # Direct 11-char ID check
    if re.fullmatch(r"^[a-zA-Z0-9_-]{11}$", url):
        return url

    patterns = [
        r"(?:v=|\/v\/|embed\/|shorts\/|youtu\.be\/)([\w-]{11})",
        r"(?:watch\?.*v=)([\w-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def seconds_to_timestamp(seconds: float) -> str:
    """
    Convert seconds into HH:MM:SS or MM:SS string.
    """
    if seconds is None or seconds < 0:
        return "00:00"
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert MM:SS or HH:MM:SS string to seconds.
    """
    if not timestamp:
        return 0.0
    parts = timestamp.strip().split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1:
            return float(parts[0])
    except ValueError:
        return 0.0
    return 0.0


def format_youtube_link(video_id: str, seconds: float) -> str:
    """
    Generate a direct YouTube link pointing to the specific timestamp.
    """
    sec = int(max(0, round(seconds)))
    return f"https://www.youtube.com/watch?v={video_id}&t={sec}s"

