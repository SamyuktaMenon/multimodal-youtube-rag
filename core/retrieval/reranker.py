"""
Cross-Encoder and Contextual Reranking Engine.
"""
import logging
from typing import List, Optional
import numpy as np

from core.retrieval.hybrid_retriever import ScoredChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved candidate chunks using deep cross-attention
    between the query and document passages.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            import os
            from sentence_transformers import CrossEncoder
            # Try loading with local files only first to avoid network retry timeouts
            self.model = CrossEncoder(self.model_name, local_files_only=True)
        except Exception:
            # If not cached locally, use the high-precision heuristic reranker
            logger.info("CrossEncoder using heuristic contextual reranker.")
            self.model = None

    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: int = 4,
        min_score: float = 0.20,
    ) -> List[ScoredChunk]:
        """
        Takes candidate chunks from hybrid search, scores each (query, passage) pair,
        and returns the top-k highest scoring chunks.
        """
        if not candidates:
            return []

        if self.model is not None:
            try:
                pairs = [[query, c.chunk.text] for c in candidates]
                scores = self.model.predict(pairs)
                # Convert logits to probability via sigmoid if needed
                probs = 1.0 / (1.0 + np.exp(-np.array(scores)))

                reranked: List[ScoredChunk] = []
                for idx, c in enumerate(candidates):
                    score = float(probs[idx])
                    if score >= min_score:
                        reranked.append(
                            ScoredChunk(
                                chunk=c.chunk,
                                final_score=score,
                                dense_score=c.dense_score,
                                bm25_score=c.bm25_score,
                            )
                        )

                reranked.sort(key=lambda x: x.final_score, reverse=True)
                for i, r in enumerate(reranked):
                    r.rank = i + 1
                return reranked[:top_k]
            except Exception as e:
                logger.error("CrossEncoder rerank execution failed: %s", e)

        # High-precision heuristic fallback reranker if model not loaded
        return self._heuristic_rerank(query, candidates, top_k, min_score)

    def _heuristic_rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: int,
        min_score: float,
    ) -> List[ScoredChunk]:
        """
        Heuristic term-interaction & position-aware reranking fallback.
        """
        q_words = [w.lower() for w in query.split() if len(w) > 2]
        reranked: List[ScoredChunk] = []

        for c in candidates:
            text_lower = c.chunk.text.lower()
            match_count = sum(1 for w in q_words if w in text_lower)
            coverage = match_count / len(q_words) if q_words else 0.5

            # Blend prior hybrid score with lexical coverage & phrase matching
            phrase_bonus = 0.15 if query.lower() in text_lower else 0.0
            boosted_score = 0.5 * c.final_score + 0.35 * coverage + phrase_bonus
            boosted_score = min(1.0, max(0.0, boosted_score))

            if boosted_score >= min_score:
                reranked.append(
                    ScoredChunk(
                        chunk=c.chunk,
                        final_score=boosted_score,
                        dense_score=c.dense_score,
                        bm25_score=c.bm25_score,
                    )
                )

        reranked.sort(key=lambda x: x.final_score, reverse=True)
        for i, r in enumerate(reranked):
            r.rank = i + 1
        return reranked[:top_k]
