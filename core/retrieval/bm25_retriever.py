"""
BM25 Sparse Lexical Retriever.
"""
import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi

from core.chunking.chunker_base import DocumentChunk


class BM25Retriever:
    """
    Sparse keyword retriever using Okapi BM25 scoring algorithm.
    Excels at exact term matching, IDs, numbers, and technical phrases.
    """

    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def _tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text into lowercase words."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [t for t in cleaned.split() if len(t) > 1]
        return tokens

    def search_with_score(self, query: str, k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """
        Search corpus for query tokens and return top-k chunks with normalized BM25 score.
        """
        if not self.bm25 or not self.chunks:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in top_indices:
            raw_score = scores[idx]
            if raw_score <= 0.0:
                continue
            normalized_score = float(raw_score / max_score)
            results.append((self.chunks[idx], normalized_score))

        return results

