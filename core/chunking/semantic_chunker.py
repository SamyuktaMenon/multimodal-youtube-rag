"""
Semantic Chunker: Detects topic shifts and breaks chunks when semantic similarity
between adjacent segments drops below a threshold.
"""
import math
from typing import List, Optional

from core.chunking.chunker_base import BaseChunker, DocumentChunk
from core.ingestion.youtube_loader import TranscriptSegment, VideoMetadata


class SemanticChunker(BaseChunker):
    """
    Groups segments based on semantic continuity. Splits when topic shift is detected.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.55,
        min_chunk_words: int = 40,
        max_chunk_words: int = 180,
    ):
        super().__init__(name="semantic")
        self.similarity_threshold = similarity_threshold
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words

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

        current_segments: List[TranscriptSegment] = [segments[0]]
        chunk_idx = 0

        for i in range(1, len(segments)):
            prev_seg = segments[i - 1]
            curr_seg = segments[i]

            # Calculate semantic similarity
            sim = self._calculate_similarity(prev_seg.text, curr_seg.text)
            curr_word_count = sum(len(s.text.split()) for s in current_segments)

            should_split = False
            # Check conditions
            if curr_word_count >= self.max_chunk_words:
                should_split = True
            elif curr_word_count >= self.min_chunk_words and sim < self.similarity_threshold:
                should_split = True

            if should_split:
                chunks.append(self._create_chunk(current_segments, chunk_idx, vid_id, vid_title))
                chunk_idx += 1
                current_segments = [curr_seg]
            else:
                current_segments.append(curr_seg)

        if current_segments:
            chunks.append(self._create_chunk(current_segments, chunk_idx, vid_id, vid_title))

        return chunks

    def _create_chunk(
        self,
        segs: List[TranscriptSegment],
        idx: int,
        vid_id: str,
        vid_title: str,
    ) -> DocumentChunk:
        text = " ".join(s.text for s in segs)
        return DocumentChunk(
            chunk_id=f"chunk_sem_{idx}",
            text=text,
            start_time=segs[0].start,
            end_time=segs[-1].end,
            strategy="semantic",
            metadata={
                "video_id": vid_id,
                "title": vid_title,
                "chunk_index": idx,
                "segment_count": len(segs),
            },
        )

    def _calculate_similarity(self, text_a: str, text_b: str) -> float:
        """
        Calculates lexical/semantic cosine similarity between two sentences.
        """
        words_a = set(w.lower() for w in text_a.split() if len(w) > 3)
        words_b = set(w.lower() for w in text_b.split() if len(w) > 3)
        if not words_a or not words_b:
            return 0.5  # Neutral continuity

        intersection = len(words_a.intersection(words_b))
        denom = math.sqrt(len(words_a)) * math.sqrt(len(words_b))
        return intersection / denom if denom > 0 else 0.0

