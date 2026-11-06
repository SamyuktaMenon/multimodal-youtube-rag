"""
Unit tests for BM25, Hybrid Search, and Reranking.
"""
import unittest
from core.chunking.chunker_base import DocumentChunk
from core.embeddings.google_embed import TFIDFEmbedding
from core.retrieval.bm25_retriever import BM25Retriever
from core.retrieval.hybrid_retriever import HybridRetriever
from core.retrieval.reranker import CrossEncoderReranker
from core.vectorstores.faiss_store import FAISSVectorStore


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            DocumentChunk("c1", "The Transformer architecture uses scaled dot-product attention.", 0.0, 15.0, "test"),
            DocumentChunk("c2", "Vector databases store dense embeddings for semantic search.", 15.0, 30.0, "test"),
            DocumentChunk("c3", "BM25 scoring relies on inverse document frequency and term frequency.", 30.0, 45.0, "test"),
            DocumentChunk("c4", "Cross-encoders compute deep cross-attention over query and passage tokens.", 45.0, 60.0, "test"),
        ]
        self.emb_model = TFIDFEmbedding(dimension=64)
        texts = [c.text for c in self.chunks]
        self.embeddings = self.emb_model.embed_documents(texts)
        self.store = FAISSVectorStore()
        self.store.add_documents(self.chunks, self.embeddings)

    def test_bm25_retriever(self):
        retriever = BM25Retriever(self.chunks)
        results = retriever.search_with_score("BM25 scoring frequency", k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0].chunk_id, "c3")

    def test_hybrid_retriever_weighted(self):
        hybrid = HybridRetriever(
            vector_store=self.store,
            embedding_model=self.emb_model,
            chunks=self.chunks,
            alpha=0.6,
            fusion_method="weighted",
        )
        results = hybrid.retrieve("transformer attention", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk.chunk_id, "c1")
        self.assertGreater(results[0].final_score, 0.0)

    def test_hybrid_retriever_rrf(self):
        hybrid = HybridRetriever(
            vector_store=self.store,
            embedding_model=self.emb_model,
            chunks=self.chunks,
            fusion_method="rrf",
            rrf_k=60,
        )
        results = hybrid.retrieve("dense embeddings vector", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk.chunk_id, "c2")

    def test_reranker(self):
        hybrid = HybridRetriever(
            vector_store=self.store,
            embedding_model=self.emb_model,
            chunks=self.chunks,
            alpha=0.5,
        )
        candidates = hybrid.retrieve("cross-encoder cross-attention", top_k=4)
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("cross-encoder cross-attention", candidates, top_k=2)
        self.assertGreater(len(reranked), 0)
        self.assertEqual(reranked[0].chunk.chunk_id, "c4")


if __name__ == "__main__":
    unittest.main()

