"""
Benchmarking Suite: Compares Embedding Models and LLM Generators.
"""
import time
from typing import Any, Dict, List, Optional
import pandas as pd

from core.chunking.chunker_base import DocumentChunk
from core.embeddings.embedding_base import BaseEmbedding
from core.embeddings.google_embed import TFIDFEmbedding
from core.embeddings.sentence_transformer_embed import SentenceTransformerEmbedding
from core.evaluation.faithfulness_eval import FaithfulnessEvaluator
from core.evaluation.retrieval_eval import EvalQuery, RetrievalEvaluator
from core.qa.contextual_qa import ContextualQA
from core.qa.llm_clients import UnifiedLLMClient
from core.retrieval.hybrid_retriever import HybridRetriever
from core.vectorstores.faiss_store import FAISSVectorStore


class BenchmarkSuite:
    """
    Runs systematic comparative evaluation across embedding models and LLMs.
    """

    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator(k=3)
        self.faithfulness_evaluator = FaithfulnessEvaluator()

    def compare_embedding_models(
        self,
        chunks: List[DocumentChunk],
        eval_queries: List[EvalQuery],
    ) -> pd.DataFrame:
        """
        Benchmarks multiple embedding models on the same corpus and query set.
        """
        models = [
            ("SentenceTransformers (MiniLM-L6)", SentenceTransformerEmbedding("all-MiniLM-L6-v2")),
            ("TF-IDF Dense Projection", TFIDFEmbedding(dimension=256)),
        ]

        results = []
        for name, emb_model in models:
            t0 = time.perf_counter()
            texts = [c.text for c in chunks]
            doc_embeddings = emb_model.embed_documents(texts)
            embed_time_ms = (time.perf_counter() - t0) * 1000.0

            store = FAISSVectorStore()
            store.add_documents(chunks, doc_embeddings)

            # Define retrieval func
            def retriever_fn(q_text: str, k: int):
                q_emb = emb_model.embed_query(q_text)
                res = store.similarity_search_with_score(q_emb, k=k)
                return [c for c, _ in res]

            t_q = time.perf_counter()
            metrics = self.retrieval_evaluator.evaluate(retriever_fn, eval_queries)
            query_time_ms = (time.perf_counter() - t_q) * 1000.0 / max(1, len(eval_queries))

            results.append(
                {
                    "Embedding Model": name,
                    "Dimension": emb_model.dimension,
                    "Embedding Latency (ms)": round(embed_time_ms, 1),
                    "Query Latency (ms)": round(query_time_ms, 2),
                    "HitRate@3": round(metrics.hit_rate_at_k, 2),
                    "Precision@3": round(metrics.precision_at_k, 2),
                    "Recall@3": round(metrics.recall_at_k, 2),
                    "MRR": round(metrics.mrr, 2),
                }
            )

        return pd.DataFrame(results)

    def compare_llm_generators(
        self,
        question: str,
        retrieved_chunks: List[Any],
        video_id: str,
    ) -> pd.DataFrame:
        """
        Compares different LLM configurations on generation speed and faithfulness.
        """
        generators = [
            ("Extractive Grounded RAG", UnifiedLLMClient(provider="mock")),
            ("Gemini 2.5 Flash", UnifiedLLMClient(provider="gemini", model_name="gemini-2.5-flash")),
            ("GPT-4o Mini", UnifiedLLMClient(provider="openai", model_name="gpt-4o-mini")),
        ]

        results = []
        chunks_only = [sc.chunk if hasattr(sc, "chunk") else sc for sc in retrieved_chunks]

        for name, client in generators:
            qa = ContextualQA(llm_client=client)
            t0 = time.perf_counter()
            resp = qa.answer_question(
                question=question,
                retrieved_chunks=retrieved_chunks,
                video_id=video_id,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0

            faith_rep = self.faithfulness_evaluator.evaluate(question, resp.answer, chunks_only)

            results.append(
                {
                    "LLM Generator": name,
                    "Response Latency (ms)": round(lat_ms, 1),
                    "Faithfulness Score": round(faith_rep.faithfulness_score, 2),
                    "Answer Relevancy": round(faith_rep.answer_relevancy_score, 2),
                    "Hallucination Rate": round(faith_rep.hallucination_rate, 2),
                    "Timestamp Citations": len(resp.citations),
                }
            )

        return pd.DataFrame(results)

