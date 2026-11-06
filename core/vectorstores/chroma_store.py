"""
ChromaDB Vector Store implementation.
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.chunking.chunker_base import DocumentChunk
from core.vectorstores.store_base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaDBVectorStore(BaseVectorStore):
    """
    Persistent/in-memory vector database powered by ChromaDB.
    Supports HNSW indexing and rich metadata filtering.
    """

    def __init__(self, collection_name: str = "youtube_chunks"):
        super().__init__(name="ChromaDB")
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.chunks_map: Dict[str, DocumentChunk] = {}
        self._init_client()

    def _init_client(self):
        try:
            import chromadb
            # Ephemeral in-memory client
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.warning("Failed to initialize ChromaDB client: %s", e)

    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        if not chunks or not embeddings:
            return

        if self.collection is None:
            self._init_client()

        if self.collection is None:
            logger.error("ChromaDB collection unavailable")
            return

        ids = []
        docs = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            # Unique ID
            cid = f"{chunk.chunk_id}_{i}"
            self.chunks_map[cid] = chunk
            ids.append(cid)
            docs.append(chunk.text)
            metadatas.append({
                "start_time": float(chunk.start_time),
                "end_time": float(chunk.end_time),
                "strategy": str(chunk.strategy),
                "video_id": str(chunk.metadata.get("video_id", "")),
            })

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=docs,
            metadatas=metadatas,
        )
        logger.info("Indexed %d chunks into ChromaDB (Total: %d)", len(chunks), self.count())

    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 4,
    ) -> List[Tuple[DocumentChunk, float]]:
        if self.collection is None or self.count() == 0:
            return []

        actual_k = min(k, self.count())
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
        )

        results: List[Tuple[DocumentChunk, float]] = []
        if res and res.get("ids") and res["ids"][0]:
            matched_ids = res["ids"][0]
            # Chroma returns cosine distance (0 is closest, 2 is opposite)
            distances = res.get("distances", [[0.0] * len(matched_ids)])[0]
            for cid, dist in zip(matched_ids, distances):
                chunk = self.chunks_map.get(cid)
                if chunk:
                    # Convert distance to similarity score
                    score = float(max(0.0, min(1.0, 1.0 - (dist / 2.0))))
                    results.append((chunk, score))

        return results

    def count(self) -> int:
        return self.collection.count() if self.collection else 0

    def clear(self) -> None:
        if self.client and self.collection:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                pass
        self.chunks_map.clear()

