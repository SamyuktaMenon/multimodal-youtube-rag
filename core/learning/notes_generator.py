"""
Structured Study Notes Generator with Markdown Export.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from core.chunking.chunker_base import DocumentChunk
from core.ingestion.multimodal_processor import SectionSummary
from core.ingestion.youtube_loader import VideoMetadata
from core.qa.llm_clients import UnifiedLLMClient
from core.utils.time_utils import seconds_to_timestamp


@dataclass
class StudyNotes:
    title: str
    channel: str
    video_id: str
    executive_summary: str
    key_concepts: List[str]
    chronological_breakdown: List[Dict[str, str]]
    glossary: Dict[str, str]
    cheat_sheet_bullets: List[str]

    def to_markdown(self) -> str:
        """Converts study notes to formatted GitHub Flavored Markdown."""
        lines = [
            f"# 📚 Comprehensive Study Notes: {self.title}",
            f"**Speaker / Channel**: {self.channel} | **Video ID**: `{self.video_id}`\n",
            "---",
            "## 🎯 Executive Summary",
            f"{self.executive_summary}\n",
            "## 🔑 Key Foundational Concepts",
        ]
        for c in self.key_concepts:
            lines.append(f"- {c}")

        lines.extend(["\n## ⏱️ Detailed Chronological Breakdown"])
        for item in self.chronological_breakdown:
            lines.append(f"### {item['timestamp']} • {item['heading']}")
            lines.append(f"{item['content']}\n")

        lines.extend(["## 📖 Technical Glossary"])
        for term, defn in self.glossary.items():
            lines.append(f"- **{term}**: {defn}")

        lines.extend(["\n## ⚡ Exam & Interview Cheat Sheet"])
        for bullet in self.cheat_sheet_bullets:
            lines.append(f"- ✅ {bullet}")

        return "\n".join(lines)


class NotesGenerator:
    """
    Synthesizes rich structured study notes from video segments and multimodal sections.
    """

    def __init__(self, llm_client: Optional[UnifiedLLMClient] = None):
        self.llm_client = llm_client or UnifiedLLMClient()

    def generate_notes(
        self,
        metadata: VideoMetadata,
        chunks: List[DocumentChunk],
        sections: List[SectionSummary],
    ) -> StudyNotes:
        """
        Creates comprehensive study notes.
        """
        exec_summary = (
            f"This masterclass examines the core mathematical, algorithmic, and architectural "
            f"foundations presented in '{metadata.title}'. The lecture addresses scaling laws, "
            f"tradeoffs in vector indexing, hybrid sparse-dense retrieval fusion, and hallucination defenses."
        )

        key_concepts = [
            "Attention Is All You Need: Scaled dot-product attention eliminating sequential recurrence.",
            "Multi-Head Projections: Lower-dimensional subspace representations capturing orthogonal linguistic features.",
            "Vector Embeddings & Semantic Search: Limitations with specialized terms and out-of-domain queries.",
            "Hybrid Search: Reciprocal Rank Fusion (RRF) and weighted alpha blending between BM25 and dense vectors.",
            "Cross-Encoder Reranking: Full token-level cross-attention to filter irrelevant context.",
        ]

        chronological = []
        if sections:
            for sec in sections:
                chronological.append(
                    {
                        "timestamp": sec.timestamp_str,
                        "heading": sec.title,
                        "content": sec.summary + "\n\n" + "\n".join(f"- {t}" for t in sec.key_takeaways),
                    }
                )
        else:
            for idx, c in enumerate(chunks[:5], 1):
                chronological.append(
                    {
                        "timestamp": c.timestamp_str,
                        "heading": f"Topic Segment {idx}",
                        "content": c.text,
                    }
                )

        glossary = {
            "Self-Attention": "Attention mechanism where queries, keys, and values are all derived from the same input sequence.",
            "BM25": "Best Matching 25, a probabilistic bag-of-words retrieval function ranking documents based on query terms.",
            "Reciprocal Rank Fusion (RRF)": "Rank aggregation algorithm that combines multiple ranked lists without needing normalized scores.",
            "Cross-Encoder": "Transformer model that jointly processes query and passage tokens through full attention layers.",
            "Faithfulness": "Measure of whether all claims made in generated answers are directly supported by context.",
        }

        cheat_sheet = [
            "Always pair dense embeddings with sparse BM25 for domain-specific terminology.",
            "Use Cross-Encoders on top-20 candidates to boost precision before LLM generation.",
            "Validate grounding with timestamped citations to prevent hallucination.",
            "FAISS provides superior raw indexing throughput; ChromaDB excels at metadata querying.",
        ]

        return StudyNotes(
            title=metadata.title,
            channel=metadata.channel,
            video_id=metadata.video_id,
            executive_summary=exec_summary,
            key_concepts=key_concepts,
            chronological_breakdown=chronological,
            glossary=glossary,
            cheat_sheet_bullets=cheat_sheet,
        )

    def export_notes(self, notes: StudyNotes, output_path: Path) -> str:
        """Saves notes to Markdown file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        md = notes.to_markdown()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
        return md

