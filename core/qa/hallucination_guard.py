"""
Hallucination Guard and Out-of-Domain Question Detector.
"""
from dataclasses import dataclass
import re
from typing import List, Optional

from core.retrieval.hybrid_retriever import ScoredChunk


@dataclass
class GuardResult:
    is_answerable: bool
    confidence: float
    reason: str
    suggested_response: Optional[str] = None


class HallucinationGuard:
    """
    Guards against hallucination by verifying whether the retrieved video transcript
    contains sufficient factual evidence to answer the user's question.
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.25,
        min_keyword_overlap_ratio: float = 0.25,
    ):
        self.min_confidence = min_confidence_threshold
        self.min_keyword_overlap = min_keyword_overlap_ratio

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[ScoredChunk],
        video_title: str = "",
    ) -> GuardResult:
        """
        Evaluate if query is answerable from retrieved chunks.
        """
        if not retrieved_chunks:
            return GuardResult(
                is_answerable=False,
                confidence=0.0,
                reason="No relevant transcript chunks could be retrieved.",
                suggested_response=(
                    f"⚠️ **Not covered in this video**: I searched the entire transcript of *'{video_title}'*, "
                    f"but could not find any information regarding **\"{query}\"**.\n\n"
                    "This video does not discuss this topic. Please ask a question related to the video's covered subjects."
                ),
            )

        top_chunk = retrieved_chunks[0]
        top_score = top_chunk.final_score

        # Check keyword overlap between query and all retrieved chunks
        combined_text = " ".join(c.chunk.text.lower() for c in retrieved_chunks)
        query_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))

        # Remove common query stop words
        common_stops = {"what", "when", "where", "which", "who", "whom", "this", "that", "these", "those", "does", "explain", "describe", "tell", "about", "video", "mention"}
        meaningful_words = query_words - common_stops

        if meaningful_words:
            matched_words = {w for w in meaningful_words if w in combined_text}
            overlap_ratio = len(matched_words) / len(meaningful_words)
        else:
            overlap_ratio = 1.0

        # Assess answerability
        if top_score < self.min_confidence and overlap_ratio < self.min_keyword_overlap:
            missing_terms = list(meaningful_words - matched_words)[:3]
            missing_str = ", ".join(f"'{m}'" for m in missing_terms)
            return GuardResult(
                is_answerable=False,
                confidence=top_score,
                reason=f"Low retrieval score ({top_score:.2f}) and low keyword overlap ({overlap_ratio:.1%}).",
                suggested_response=(
                    f"⚠️ **Out of Context / Not in Video**: The query refers to topics ({missing_str}) that are not "
                    f"covered or explained in this video (*'{video_title}'*).\n\n"
                    "To prevent hallucination, the model abstains from answering ungrounded questions. "
                    "Feel free to ask about the core concepts presented in this lecture."
                ),
            )

        return GuardResult(
            is_answerable=True,
            confidence=top_score,
            reason=f"Sufficient grounding found (Score: {top_score:.2f}, Overlap: {overlap_ratio:.1%}).",
        )

