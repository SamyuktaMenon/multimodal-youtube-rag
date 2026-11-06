"""
Temporal and Chapter-based chunking implementations for Video RAG.
"""
from typing import List, Optional

from core.chunking.chunker_base import BaseChunker, DocumentChunk
from core.ingestion.youtube_loader import TranscriptSegment, VideoMetadata


class TemporalChunker(BaseChunker):
    """
    Chunks transcript segments into fixed time windows (e.g. 45 seconds or 60 seconds),
    guaranteeing strict temporal alignment with video timeline playback.
    """

    def __init__(self, time_window_seconds: int = 60, time_overlap_seconds: int = 10):
        super().__init__(name="temporal")
        self.time_window = time_window_seconds
        self.time_overlap = time_overlap_seconds

    def chunk(
        self,
        segments: List[TranscriptSegment],
        metadata: Optional[VideoMetadata] = None,
    ) -> List[DocumentChunk]:
        if not segments:
            return []

        total_duration = segments[-1].end
        step = max(10, self.time_window - self.time_overlap)
        chunks: List[DocumentChunk] = []

        vid_id = metadata.video_id if metadata else ""
        vid_title = metadata.title if metadata else ""

        current_window_start = 0.0
        chunk_idx = 0

        while current_window_start < total_duration:
            current_window_end = min(total_duration, current_window_start + self.time_window)

            matching_segs = [
                s
                for s in segments
                if (s.start >= current_window_start and s.start < current_window_end)
                or (s.end > current_window_start and s.end <= current_window_end)
            ]

            if matching_segs:
                text = " ".join(s.text for s in matching_segs)
                actual_start = min(s.start for s in matching_segs)
                actual_end = max(s.end for s in matching_segs)

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"chunk_temp_{chunk_idx}",
                        text=text,
                        start_time=actual_start,
                        end_time=actual_end,
                        strategy="temporal",
                        metadata={
                            "video_id": vid_id,
                            "title": vid_title,
                            "chunk_index": chunk_idx,
                            "window_span": f"{current_window_start:.0f}s-{current_window_end:.0f}s",
                        },
                    )
                )
                chunk_idx += 1

            current_window_start += step

        return chunks


class ChapterChunker(BaseChunker):
    """
    Chunks transcript along official video chapters or natural section markers.
    """

    def __init__(self):
        super().__init__(name="chapter")

    def chunk(
        self,
        segments: List[TranscriptSegment],
        metadata: Optional[VideoMetadata] = None,
    ) -> List[DocumentChunk]:
        if not segments:
            return []

        chunks: List[DocumentChunk] = []
        vid_id = metadata.video_id if metadata else ""
        vid_title = metadata.title if metadata else ""

        chapters = metadata.chapters if metadata and metadata.chapters else []

        if not chapters:
            # Fallback to temporal 90s chunks if no chapters present
            fallback = TemporalChunker(time_window_seconds=90, time_overlap_seconds=15)
            return fallback.chunk(segments, metadata)

        for idx, ch in enumerate(chapters):
            ch_segs = [
                s
                for s in segments
                if (s.start >= ch.start_time and s.start < ch.end_time)
                or (s.end > ch.start_time and s.end <= ch.end_time)
            ]
            if not ch_segs:
                continue

            text = f"Chapter: {ch.title}. " + " ".join(s.text for s in ch_segs)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chunk_chap_{idx}",
                    text=text,
                    start_time=ch.start_time,
                    end_time=ch.end_time,
                    strategy="chapter",
                    metadata={
                        "video_id": vid_id,
                        "title": vid_title,
                        "chapter_title": ch.title,
                        "chunk_index": idx,
                    },
                )
            )

        return chunks

