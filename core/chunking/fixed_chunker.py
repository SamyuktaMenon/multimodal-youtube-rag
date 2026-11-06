"""
Fixed-size chunker with token/word windowing and overlap.
"""
from typing import List, Optional

from core.chunking.chunker_base import BaseChunker, DocumentChunk
from core.ingestion.youtube_loader import TranscriptSegment, VideoMetadata


class FixedSizeChunker(BaseChunker):
    """
    Chunks text into fixed token/word chunks with specified overlap,
    interpolating start and end timestamps from underlying segments.
    """

    def __init__(self, chunk_size: int = 80, chunk_overlap: int = 15):
        super().__init__(name="fixed")
        self.chunk_size = chunk_size  # words
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        segments: List[TranscriptSegment],
        metadata: Optional[VideoMetadata] = None,
    ) -> List[DocumentChunk]:
        if not segments:
            return []

        # Flatten words with their time annotations
        word_time_pairs: List[tuple[str, float, float]] = []
        for s in segments:
            words = s.text.split()
            if not words:
                continue
            seg_duration = max(0.1, s.duration)
            time_per_word = seg_duration / len(words)
            for idx, w in enumerate(words):
                w_start = s.start + idx * time_per_word
                w_end = w_start + time_per_word
                word_time_pairs.append((w, w_start, w_end))

        if not word_time_pairs:
            return []

        chunks: List[DocumentChunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        vid_id = metadata.video_id if metadata else ""
        vid_title = metadata.title if metadata else ""

        chunk_idx = 0
        for i in range(0, len(word_time_pairs), step):
            slice_pairs = word_time_pairs[i : i + self.chunk_size]
            if not slice_pairs:
                break

            chunk_text = " ".join(p[0] for p in slice_pairs)
            start_t = slice_pairs[0][1]
            end_t = slice_pairs[-1][2]

            chunks.append(
                DocumentChunk(
                    chunk_id=f"chunk_fixed_{chunk_idx}",
                    text=chunk_text,
                    start_time=start_t,
                    end_time=end_t,
                    strategy="fixed",
                    metadata={
                        "video_id": vid_id,
                        "title": vid_title,
                        "chunk_index": chunk_idx,
                        "word_count": len(slice_pairs),
                    },
                )
            )
            chunk_idx += 1

            if i + self.chunk_size >= len(word_time_pairs):
                break

        return chunks

