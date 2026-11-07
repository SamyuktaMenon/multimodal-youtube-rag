"""
Embedding Model Factory.
"""
from typing import Optional
from core.embeddings.embedding_base import BaseEmbedding
from core.embeddings.google_embed import GoogleEmbedding, OpenAIEmbedding, TFIDFEmbedding
from core.embeddings.sentence_transformer_embed import SentenceTransformerEmbedding
from core.embeddings.clip_embed import CLIPMultimodalEmbedding


def get_embedding_model(
    provider: str = "sentence-transformers",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseEmbedding:
    """Instantiate embedding model based on provider."""
    provider = provider.lower()
    if provider in ["clip", "multimodal", "vision"]:
        name = model_name or "clip-ViT-B-32"
        return CLIPMultimodalEmbedding(model_name=name)
    elif provider in ["sentence-transformers", "local", "hf"]:
        name = model_name or "all-MiniLM-L6-v2"
        return SentenceTransformerEmbedding(model_name=name)
    elif provider in ["google", "gemini"]:
        name = model_name or "text-embedding-004"
        return GoogleEmbedding(model_name=name, api_key=api_key)
    elif provider in ["openai"]:
        name = model_name or "text-embedding-3-small"
        return OpenAIEmbedding(model_name=name, api_key=api_key)
    elif provider in ["tfidf", "fast"]:
        return TFIDFEmbedding()
    else:
        return SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
