"""
Chunker Factory and Comparative Distribution Profiler.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from core.chunking.chunker_base import BaseChunker, DocumentChunk
from core.chunking.fixed_chunker import FixedSizeChunker
from core.chunking.recursive_chunker import RecursiveCharacterChunker
from core.chunking.semantic_chunker import SemanticChunker
from core.chunking.temporal_chunker import ChapterChunker, TemporalChunker
from core.ingestion.youtube_loader import TranscriptSegment, VideoMetadata


def get_chunker(strategy: str = "recursive", **kwargs) -> BaseChunker:
    """Instantiate a chunker by strategy name."""
    strategy = strategy.lower()
    if strategy == "fixed":
        return FixedSizeChunker(
            chunk_size=kwargs.get("chunk_size", 80),
            chunk_overlap=kwargs.get("chunk_overlap", 15),
        )
    elif strategy == "semantic":
        return SemanticChunker(
            similarity_threshold=kwargs.get("similarity_threshold", 0.55),
        )
    elif strategy == "temporal":
        return TemporalChunker(
            time_window_seconds=kwargs.get("time_window_seconds", 60),
            time_overlap_seconds=kwargs.get("time_overlap_seconds", 10),
        )
    elif strategy == "chapter":
        return ChapterChunker()
    else:
        return RecursiveCharacterChunker(
            chunk_size=kwargs.get("chunk_size", 500),
            chunk_overlap=kwargs.get("chunk_overlap", 100),
        )


def compare_chunking_strategies(
    segments: List[TranscriptSegment],
    metadata: Optional[VideoMetadata] = None,
) -> pd.DataFrame:
    """
    Runs all 5 chunking strategies on the segments and returns a comparative DataFrame
    with metrics like count, avg length, min/max words, and timestamp span.
    """
    strategies = ["recursive", "fixed", "semantic", "temporal", "chapter"]
    records = []

    for strat in strategies:
        chunker = get_chunker(strat)
        chunks = chunker.chunk(segments, metadata)
        if not chunks:
            continue

        word_counts = [c.word_count for c in chunks]
        durations = [c.duration for c in chunks]

        records.append(
            {
                "Strategy": strat.capitalize(),
                "Total Chunks": len(chunks),
                "Avg Words/Chunk": round(float(np.mean(word_counts)), 1),
                "Min Words": int(np.min(word_counts)),
                "Max Words": int(np.max(word_counts)),
                "Avg Duration (s)": round(float(np.mean(durations)), 1),
                "Max Duration (s)": round(float(np.max(durations)), 1),
            }
        )

    return pd.DataFrame(records)

