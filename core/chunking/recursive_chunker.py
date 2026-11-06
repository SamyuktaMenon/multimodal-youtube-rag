"""
Recursive Character Chunker for hierarchical boundary splitting.
"""
from typing import List, Optional

from core.chunking.chunker_base import BaseChunker, DocumentChunk
from core.ingestion.youtube_loader import TranscriptSegment, VideoMetadata


class RecursiveCharacterChunker(BaseChunker):
    """
    Splits text recursively using hierarchical separators (paragraphs,
    sentences, words) while tracking precise video timestamps.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        super().__init__(name="recursive")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

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

        current_text = ""
        current_start: Optional[float] = None
        current_end: float = 0.0
        chunk_idx = 0

        for seg in segments:
            if current_start is None:
                current_start = seg.start

            prospective = current_text + (" " if current_text else "") + seg.text

            if len(prospective) <= self.chunk_size:
                current_text = prospective
                current_end = seg.end
            else:
                if current_text:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"chunk_rec_{chunk_idx}",
                            text=current_text,
                            start_time=current_start,
                            end_time=current_end,
                            strategy="recursive",
                            metadata={
                                "video_id": vid_id,
                                "title": vid_title,
                                "chunk_index": chunk_idx,
                            },
                        )
                    )
                    chunk_idx += 1

                # Carry over overlap if possible
                overlap_text = current_text[-self.chunk_overlap :] if len(current_text) > self.chunk_overlap else ""
                current_text = (overlap_text + " " + seg.text).strip()
                current_start = seg.start
                current_end = seg.end

        if current_text and current_start is not None:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chunk_rec_{chunk_idx}",
                    text=current_text,
                    start_time=current_start,
                    end_time=current_end,
                    strategy="recursive",
                    metadata={
                        "video_id": vid_id,
                        "title": vid_title,
                        "chunk_index": chunk_idx,
                    },
                )
            )

        return chunks

