"""
Hybrid Retriever: Combines Dense Vector Search and Sparse BM25 Keyword Search
with Reciprocal Rank Fusion (RRF) and Weighted Alpha Blending.
"""
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from core.chunking.chunker_base import DocumentChunk
from core.embeddings.embedding_base import BaseEmbedding
from core.retrieval.bm25_retriever import BM25Retriever
from core.vectorstores.store_base import BaseVectorStore


@dataclass
class ScoredChunk:
    chunk: DocumentChunk
    final_score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rank: int = 0


class HybridRetriever:
    """
    Hybrid search engine merging semantic vectors and lexical BM25 matching.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbedding,
        chunks: List[DocumentChunk],
        alpha: float = 0.65,  # 1.0 = Pure Dense, 0.0 = Pure BM25
        fusion_method: Literal["weighted", "rrf"] = "weighted",
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.chunks = chunks
        self.alpha = alpha
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.bm25_retriever = BM25Retriever(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha_override: Optional[float] = None,
    ) -> List[ScoredChunk]:
        """
        Executes hybrid retrieval using dense + BM25 search.
        """
        alpha = self.alpha if alpha_override is None else alpha_override
        candidate_pool_k = max(top_k * 3, 15)

        # 1. Dense retrieval
        query_emb = self.embedding_model.embed_query(query)
        dense_results = self.vector_store.similarity_search_with_score(query_emb, k=candidate_pool_k)

        # 2. Sparse BM25 retrieval
        sparse_results = self.bm25_retriever.search_with_score(query, k=candidate_pool_k)

        # 3. Fuse scores
        if self.fusion_method == "rrf":
            fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        else:
            fused = self._weighted_alpha_fusion(dense_results, sparse_results, alpha=alpha)

        # Sort descending by final score
        fused.sort(key=lambda x: x.final_score, reverse=True)

        # Assign final ranks
        for i, item in enumerate(fused):
            item.rank = i + 1

        return fused[:top_k]

    def _weighted_alpha_fusion(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
        alpha: float,
    ) -> List[ScoredChunk]:
        """
        Calculates final score = alpha * dense_score + (1 - alpha) * bm25_score.
        """
        chunk_map: Dict[str, ScoredChunk] = {}

        # Add dense scores
        for chunk, d_score in dense_results:
            chunk_map[chunk.chunk_id] = ScoredChunk(
                chunk=chunk,
                final_score=alpha * d_score,
                dense_score=d_score,
                bm25_score=0.0,
            )

        # Add sparse BM25 scores
        for chunk, s_score in sparse_results:
            if chunk.chunk_id in chunk_map:
                existing = chunk_map[chunk.chunk_id]
                existing.bm25_score = s_score
                existing.final_score += (1.0 - alpha) * s_score
            else:
                chunk_map[chunk.chunk_id] = ScoredChunk(
                    chunk=chunk,
                    final_score=(1.0 - alpha) * s_score,
                    dense_score=0.0,
                    bm25_score=s_score,
                )

        return list(chunk_map.values())

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
    ) -> List[ScoredChunk]:
        """
        Calculates RRF score = sum(1 / (k + rank)).
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}
        dense_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}

        for rank, (chunk, score) in enumerate(dense_results, 1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            dense_scores[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, (chunk, score) in enumerate(sparse_results, 1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            bm25_scores[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        results = []
        for cid, rrf_score in rrf_scores.items():
            results.append(
                ScoredChunk(
                    chunk=chunk_map[cid],
                    final_score=rrf_score,
                    dense_score=dense_scores.get(cid, 0.0),
                    bm25_score=bm25_scores.get(cid, 0.0),
                )
            )

        return results

