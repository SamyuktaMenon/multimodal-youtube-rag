"""
Base Chunker abstractions and DocumentChunk data models.
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from core.ingestion.youtube_loader import TranscriptSegment, VideoMetadata
from core.utils.time_utils import format_youtube_link, seconds_to_timestamp


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    start_time: float
    end_time: float
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    frame_path: Optional[str] = None

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def timestamp_str(self) -> str:
        return f"[{seconds_to_timestamp(self.start_time)} - {seconds_to_timestamp(self.end_time)}]"

    @property
    def start_timestamp(self) -> str:
        return seconds_to_timestamp(self.start_time)

    def get_youtube_url(self, video_id: Optional[str] = None) -> str:
        vid = video_id or self.metadata.get("video_id", "")
        return format_youtube_link(vid, self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp_str"] = self.timestamp_str
        d["word_count"] = self.word_count
        return d


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def chunk(
        self,
        segments: List[TranscriptSegment],
        metadata: Optional[VideoMetadata] = None,
    ) -> List[DocumentChunk]:
        """Convert transcript segments into document chunks."""
        pass

