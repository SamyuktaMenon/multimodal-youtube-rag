"""
Empirical Benchmark Runner: Evaluates all 5 Chunking Strategies on Ground-Truth Video Retrieval.
"""
import os
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.chunking.chunker_factory import get_chunker
from core.ingestion.youtube_loader import TranscriptSegment, VideoMetadata
from core.embeddings.embedding_factory import get_embedding_model
from core.vectorstores.faiss_store import FAISSVectorStore
from core.retrieval.hybrid_retriever import HybridRetriever


def run_benchmark():
    print("=" * 75)
    print("🎬 Running Multimodal YouTube RAG: Chunking & Retrieval Benchmark")
    print("=" * 75)

    data_file = PROJECT_ROOT / "data" / "benchmark_dataset.json"
    with open(data_file) as f:
        bench_data = json.load(f)

    # Convert segments to TranscriptSegment objects
    segments = [
        TranscriptSegment(
            text=s["text"],
            start=float(s["start"]),
            duration=float(s["end"] - s["start"]),
            end=float(s["end"]),
        )
        for s in bench_data["segments"]
    ]
    queries = bench_data["benchmark_queries"]

    emb_model = get_embedding_model(provider="sentence-transformers", model_name="all-MiniLM-L6-v2")
    strategies = ["fixed", "recursive", "semantic", "temporal", "chapter"]
    results = []

    for strat in strategies:
        t0_chunk = time.perf_counter()
        chunker = get_chunker(strategy=strat)
        chunks = chunker.chunk(segments)
        chunk_latency_ms = (time.perf_counter() - t0_chunk) * 1000.0

        if not chunks:
            continue

        # Embed chunks and index into FAISS
        texts = [c.text for c in chunks]
        embeddings = emb_model.embed_documents(texts)
        store = FAISSVectorStore()
        store.add_documents(chunks, embeddings)

        # Build Hybrid Retriever
        hybrid = HybridRetriever(
            vector_store=store,
            embedding_model=emb_model,
            chunks=chunks,
            fusion_method="rrf",
        )

        hits_at_1 = 0
        hits_at_3 = 0
        reciprocal_ranks = []
        precision_at_3_scores = []
        recall_at_3_scores = []
        query_latencies = []

        k = 3
        for q_item in queries:
            q_text = q_item["query"]
            gold_kw = [kw.lower() for kw in q_item["gold_keywords"]]

            t0_q = time.perf_counter()
            retrieved = hybrid.retrieve(q_text, top_k=k)
            q_lat = (time.perf_counter() - t0_q) * 1000.0
            query_latencies.append(q_lat)

            found_rank = None
            relevant_count = 0

            for rank, item in enumerate(retrieved, 1):
                chunk = item.chunk
                c_text = chunk.text.lower()
                # Document is relevant if it contains key gold concepts
                matches = sum(1 for kw in gold_kw if kw in c_text)
                if matches >= 1:
                    relevant_count += 1
                    if found_rank is None:
                        found_rank = rank

            if found_rank == 1:
                hits_at_1 += 1
            if found_rank is not None and found_rank <= 3:
                hits_at_3 += 1
            if found_rank is not None:
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)

            precision_at_3_scores.append(relevant_count / max(1, len(retrieved)))
            recall_at_3_scores.append(1.0 if relevant_count > 0 else 0.0)

        n_q = len(queries)
        hit_1 = hits_at_1 / n_q
        hit_3 = hits_at_3 / n_q
        mrr = float(np.mean(reciprocal_ranks))
        prec_3 = float(np.mean(precision_at_3_scores))
        rec_3 = float(np.mean(recall_at_3_scores))
        avg_q_lat = float(np.mean(query_latencies))
        avg_tokens = int(np.mean([len(c.text.split()) for c in chunks]))

        results.append({
            "Strategy": strat.capitalize(),
            "Chunks Created": len(chunks),
            "Avg Words/Chunk": avg_tokens,
            "HitRate@1": f"{hit_1 * 100:.1f}%",
            "HitRate@3": f"{hit_3 * 100:.1f}%",
            "Recall@3": f"{rec_3 * 100:.1f}%",
            "Precision@3": f"{prec_3 * 100:.1f}%",
            "MRR": f"{mrr:.3f}",
            "Query Latency (ms)": f"{avg_q_lat:.1f}"
        })

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    print("=" * 75)

    # Save to JSON
    export_dir = PROJECT_ROOT / "data" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    out_json = export_dir / "chunking_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Benchmark results successfully exported to: {out_json}\n")


if __name__ == "__main__":
    run_benchmark()
