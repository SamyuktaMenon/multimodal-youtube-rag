"""
Retrieval Evaluation: Precision@K, Recall@K, Hit Rate@K, MRR, and NDCG@K.
"""
from dataclasses import dataclass
import math
from typing import Dict, List, Set
import numpy as np
import pandas as pd

from core.chunking.chunker_base import DocumentChunk


@dataclass
class EvalQuery:
    query_id: str
    query_text: str
    relevant_chunk_ids: Set[str]


@dataclass
class RetrievalMetrics:
    hit_rate_at_k: float
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int

    def to_dict(self) -> Dict[str, float]:
        return {
            f"HitRate@{self.k}": round(self.hit_rate_at_k, 3),
            f"Precision@{self.k}": round(self.precision_at_k, 3),
            f"Recall@{self.k}": round(self.recall_at_k, 3),
            "MRR": round(self.mrr, 3),
            f"NDCG@{self.k}": round(self.ndcg_at_k, 3),
        }


class RetrievalEvaluator:
    """
    Computes standard Information Retrieval metrics for RAG retrieval engines.
    """

    def __init__(self, k: int = 4):
        self.k = k

    def evaluate(
        self,
        retriever_func,  # Callable[[str, int], List[DocumentChunk]]
        eval_queries: List[EvalQuery],
    ) -> RetrievalMetrics:
        """
        Runs retriever across evaluation queries and calculates metrics.
        """
        if not eval_queries:
            return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, self.k)

        hits = []
        precisions = []
        recalls = []
        reciprocal_ranks = []
        ndcgs = []

        for eq in eval_queries:
            retrieved = retriever_func(eq.query_text, self.k)
            retrieved_ids = [c.chunk_id for c in retrieved]

            # Relevant retrieved
            relevant_retrieved = [cid for cid in retrieved_ids if cid in eq.relevant_chunk_ids]

            # 1. Hit Rate
            hit = 1.0 if len(relevant_retrieved) > 0 else 0.0
            hits.append(hit)

            # 2. Precision@K
            prec = len(relevant_retrieved) / self.k if self.k > 0 else 0.0
            precisions.append(prec)

            # 3. Recall@K
            total_rel = len(eq.relevant_chunk_ids)
            rec = len(relevant_retrieved) / total_rel if total_rel > 0 else 0.0
            recalls.append(rec)

            # 4. MRR
            rr = 0.0
            for rank, cid in enumerate(retrieved_ids, 1):
                if cid in eq.relevant_chunk_ids:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            # 5. NDCG@K
            dcg = 0.0
            for rank, cid in enumerate(retrieved_ids, 1):
                rel = 1.0 if cid in eq.relevant_chunk_ids else 0.0
                dcg += rel / math.log2(rank + 1)

            idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(total_rel, self.k) + 1))
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            ndcgs.append(ndcg)

        return RetrievalMetrics(
            hit_rate_at_k=float(np.mean(hits)),
            precision_at_k=float(np.mean(precisions)),
            recall_at_k=float(np.mean(recalls)),
            mrr=float(np.mean(reciprocal_ranks)),
            ndcg_at_k=float(np.mean(ndcgs)),
            k=self.k,
        )

    @staticmethod
    def generate_synthetic_eval_dataset(chunks: List[DocumentChunk], num_queries: int = 5) -> List[EvalQuery]:
        """
        Automatically generates high-fidelity evaluation queries from actual chunks
        to enable instant evaluation benchmarking.
        """
        eval_queries: List[EvalQuery] = []
        selected_chunks = chunks[: min(num_queries, len(chunks))]

        for idx, c in enumerate(selected_chunks, 1):
            words = [w for w in c.text.split() if len(w) > 4 and w.isalpha()]
            query_phrase = " ".join(words[:5]) if len(words) >= 5 else c.text[:40]
            query_text = f"What does the video discuss regarding {query_phrase}?"
            eval_queries.append(
                EvalQuery(
                    query_id=f"q_{idx}",
                    query_text=query_text,
                    relevant_chunk_ids={c.chunk_id},
                )
            )

        return eval_queries

