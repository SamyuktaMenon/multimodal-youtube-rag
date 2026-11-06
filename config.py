"""
Configuration settings for Multimodal YouTube RAG System.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
FRAMES_DIR = DATA_DIR / "frames"
EXPORTS_DIR = DATA_DIR / "exports"

for d in [DATA_DIR, CACHE_DIR, FRAMES_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class EmbeddingConfig:
    provider: Literal["sentence-transformers", "google", "openai", "tfidf"] = "sentence-transformers"
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32


@dataclass
class VectorStoreConfig:
    store_type: Literal["faiss", "chroma", "memory"] = "faiss"
    metric: Literal["cosine", "l2", "ip"] = "cosine"
    persist_dir: Path = CACHE_DIR / "vectorstores"


@dataclass
class ChunkerConfig:
    strategy: Literal["recursive", "fixed", "semantic", "temporal", "chapter"] = "recursive"
    chunk_size: int = 500  # characters
    chunk_overlap: int = 100  # characters
    time_window_seconds: int = 60  # for temporal chunker
    semantic_similarity_threshold: float = 0.75  # for semantic chunker


@dataclass
class RetrievalConfig:
    top_k: int = 4
    enable_hybrid: bool = True
    hybrid_alpha: float = 0.65  # weight for dense (1 - alpha for sparse BM25)
    rrf_k: int = 60
    enable_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_confidence_score: float = 0.30


@dataclass
class LLMConfig:
    provider: Literal["gemini", "openai", "ollama", "mock"] = "gemini"
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.2
    max_tokens: int = 2048
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


@dataclass
class MultimodalConfig:
    extract_frames: bool = True
    frame_interval_seconds: int = 15
    max_frames: int = 30
    frame_width: int = 640
    enable_ocr: bool = True

