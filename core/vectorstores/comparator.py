"""
FAISS vs ChromaDB Vector Database Benchmark and Comparative Evaluation.
"""
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from core.chunking.chunker_base import DocumentChunk
from core.vectorstores.chroma_store import ChromaDBVectorStore
from core.vectorstores.faiss_store import FAISSVectorStore


class VectorStoreComparator:
    """
    Empirical benchmark comparing FAISS and ChromaDB across:
    - Index creation time and throughput
    - Query search latency (mean, min, max, p95)
    - Retrieval result agreement (Jaccard similarity at Top-K)
    - Memory overhead & architectural capabilities
    """

    def __init__(self):
        self.faiss_store = FAISSVectorStore()
        self.chroma_store = ChromaDBVectorStore(collection_name="benchmark_coll")

    def run_benchmark(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        query_embeddings: Optional[List[List[float]]] = None,
        k: int = 4,
        num_query_repeats: int = 5,
    ) -> Dict[str, Any]:
        """
        Runs complete benchmark suite and returns comparative metrics.
        """
        if not chunks or not embeddings:
            return {"error": "No data provided for benchmark"}

        self.faiss_store.clear()
        self.chroma_store.clear()

        # 1. Benchmark FAISS Indexing
        t0 = time.perf_counter()
        self.faiss_store.add_documents(chunks, embeddings)
        faiss_index_time_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Benchmark ChromaDB Indexing
        t0 = time.perf_counter()
        self.chroma_store.add_documents(chunks, embeddings)
        chroma_index_time_ms = (time.perf_counter() - t0) * 1000.0

        # Prepare queries
        if not query_embeddings:
            # Use sample embeddings from indexed set as queries
            sample_indices = np.linspace(0, len(embeddings) - 1, min(5, len(embeddings))).astype(int)
            query_embeddings = [embeddings[i] for i in sample_indices]

        # 3. Benchmark Query Latency
        faiss_latencies: List[float] = []
        chroma_latencies: List[float] = []
        agreements: List[float] = []

        for q_emb in query_embeddings:
            for _ in range(num_query_repeats):
                t_f = time.perf_counter()
                f_res = self.faiss_store.similarity_search_with_score(q_emb, k=k)
                faiss_latencies.append((time.perf_counter() - t_f) * 1000.0)

                t_c = time.perf_counter()
                c_res = self.chroma_store.similarity_search_with_score(q_emb, k=k)
                chroma_latencies.append((time.perf_counter() - t_c) * 1000.0)

            # Measure Top-K Jaccard Agreement between FAISS and Chroma
            f_ids = set(c.chunk_id for c, _ in f_res)
            c_ids = set(c.chunk_id for c, _ in c_res)
            if f_ids or c_ids:
                jaccard = len(f_ids.intersection(c_ids)) / len(f_ids.union(c_ids))
                agreements.append(jaccard)

        faiss_mean_lat = float(np.mean(faiss_latencies))
        chroma_mean_lat = float(np.mean(chroma_latencies))
        avg_agreement = float(np.mean(agreements)) if agreements else 1.0

        # Comparative Summary Table
        summary_df = pd.DataFrame(
            [
                {
                    "Metric": "Indexing Time (ms)",
                    "FAISS": round(faiss_index_time_ms, 2),
                    "ChromaDB": round(chroma_index_time_ms, 2),
                    "Winner": "FAISS" if faiss_index_time_ms < chroma_index_time_ms else "ChromaDB",
                },
                {
                    "Metric": "Indexing Throughput (chunks/sec)",
                    "FAISS": round(len(chunks) / (faiss_index_time_ms / 1000.0 + 1e-6), 1),
                    "ChromaDB": round(len(chunks) / (chroma_index_time_ms / 1000.0 + 1e-6), 1),
                    "Winner": "FAISS" if faiss_index_time_ms < chroma_index_time_ms else "ChromaDB",
                },
                {
                    "Metric": "Avg Query Latency (ms)",
                    "FAISS": round(faiss_mean_lat, 3),
                    "ChromaDB": round(chroma_mean_lat, 3),
                    "Winner": "FAISS" if faiss_mean_lat < chroma_mean_lat else "ChromaDB",
                },
                {
                    "Metric": "P95 Query Latency (ms)",
                    "FAISS": round(float(np.percentile(faiss_latencies, 95)), 3),
                    "ChromaDB": round(float(np.percentile(chroma_latencies, 95)), 3),
                    "Winner": "FAISS" if np.percentile(faiss_latencies, 95) < np.percentile(chroma_latencies, 95) else "ChromaDB",
                },
                {
                    "Metric": "Result Agreement (Jaccard @ Top-K)",
                    "FAISS": f"{avg_agreement * 100:.1f}%",
                    "ChromaDB": f"{avg_agreement * 100:.1f}%",
                    "Winner": "Consensus",
                },
            ]
        )

        feature_matrix = pd.DataFrame(
            [
                {"Feature": "Underlying Core", "FAISS": "C++ Low-level Blazing Fast", "ChromaDB": "HNSWlib + SQLite / DuckDB"},
                {"Feature": "Metadata Filtering", "FAISS": "Requires external index / ID map", "ChromaDB": "Native rich where/contains queries"},
                {"Feature": "Persistence", "FAISS": "Manual read/write index binary", "ChromaDB": "Automatic SQLite persistence on disk"},
                {"Feature": "Horizontal Scaling", "FAISS": "Distributed GPU clusters via index sharding", "ChromaDB": "Single-node server / Cloud mode"},
                {"Feature": "Best Use Case", "FAISS": "Maximum speed, ultra-large scale vectors", "ChromaDB": "Full-text metadata apps, rapid prototyping"},
            ]
        )

        return {
            "summary_df": summary_df,
            "feature_matrix": feature_matrix,
            "metrics": {
                "faiss_index_ms": faiss_index_time_ms,
                "chroma_index_ms": chroma_index_time_ms,
                "faiss_latency_ms": faiss_mean_lat,
                "chroma_latency_ms": chroma_mean_lat,
                "top_k_agreement": avg_agreement,
            },
        }

