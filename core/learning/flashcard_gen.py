"""
Flashcard Generator with Anki TSV and JSON Export Capabilities.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.chunking.chunker_base import DocumentChunk
from core.qa.llm_clients import UnifiedLLMClient
from core.utils.time_utils import format_youtube_link, seconds_to_timestamp


@dataclass
class Flashcard:
    card_id: int
    front: str
    back: str
    timestamp_seconds: float
    topic_tag: str
    video_id: str = ""

    @property
    def timestamp_str(self) -> str:
        return seconds_to_timestamp(self.timestamp_seconds)

    @property
    def youtube_url(self) -> str:
        return format_youtube_link(self.video_id, self.timestamp_seconds)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp_str"] = self.timestamp_str
        d["youtube_url"] = self.youtube_url
        return d


class FlashcardGenerator:
    """
    Synthesizes active-recall flashcards from video chunks.
    """

    def __init__(self, llm_client: Optional[UnifiedLLMClient] = None):
        self.llm_client = llm_client or UnifiedLLMClient()

    def generate_flashcards(
        self,
        chunks: List[DocumentChunk],
        video_id: str,
        num_cards: int = 6,
    ) -> List[Flashcard]:
        """
        Creates flashcards distributed across the video timeline.
        """
        if not chunks:
            return []

        step = max(1, len(chunks) // num_cards)
        sample_chunks = [chunks[i] for i in range(0, len(chunks), step)][:num_cards]

        cards: List[Flashcard] = []
        for idx, chunk in enumerate(sample_chunks, 1):
            sentences = [s.strip() for s in chunk.text.split(". ") if len(s.strip()) > 15]
            concept = sentences[0] if sentences else "Core Concept"
            details = ". ".join(sentences[1:3]) if len(sentences) > 1 else chunk.text

            front = f"What is the significance of the concept discussed at [{chunk.start_timestamp}]?"
            back = (
                f"**Key Idea**: {concept}.\n\n"
                f"**Details**: {details}\n\n"
                f"*(Video Reference: {chunk.start_timestamp})*"
            )

            cards.append(
                Flashcard(
                    card_id=idx,
                    front=front,
                    back=back,
                    timestamp_seconds=chunk.start_time,
                    topic_tag="VideoLecture",
                    video_id=video_id,
                )
            )

        return cards

    def export_to_anki_tsv(self, cards: List[Flashcard], output_path: Path) -> str:
        """
        Exports cards to standard Anki TSV format:
        Front [TAB] Back [TAB] Tags
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for c in cards:
            clean_front = c.front.replace("\t", " ").replace("\n", "<br>")
            clean_back = c.back.replace("\t", " ").replace("\n", "<br>")
            tag = c.topic_tag.replace(" ", "_")
            lines.append(f"{clean_front}\t{clean_back}\t{tag}")

        tsv_content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(tsv_content)

        return tsv_content

    def export_to_json(self, cards: List[Flashcard], output_path: Path) -> str:
        """
        Exports cards to JSON format.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [c.to_dict() for c in cards]
        content = json.dumps(data, indent=2, ensure_ascii=False)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return content

