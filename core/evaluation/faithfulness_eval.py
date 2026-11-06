"""
Answer Faithfulness and Hallucination Evaluation Suite.
"""
from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional
import numpy as np

from core.chunking.chunker_base import DocumentChunk


@dataclass
class ClaimVerification:
    claim: str
    is_supported: bool
    supporting_quote: Optional[str] = None
    confidence: float = 0.0


@dataclass
class FaithfulnessReport:
    faithfulness_score: float  # [0.0, 1.0]
    answer_relevancy_score: float  # [0.0, 1.0]
    hallucination_rate: float  # [0.0, 1.0]
    total_claims: int
    supported_claims: int
    claim_verifications: List[ClaimVerification] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        return {
            "Faithfulness Score": round(self.faithfulness_score, 3),
            "Answer Relevancy": round(self.answer_relevancy_score, 3),
            "Hallucination Rate": round(self.hallucination_rate, 3),
            "Supported Claims": self.supported_claims,
            "Total Claims": self.total_claims,
        }


class FaithfulnessEvaluator:
    """
    Evaluates whether generated answers are strictly faithful to retrieved context
    and directly relevant to the user query.
    """

    def evaluate(
        self,
        question: str,
        answer: str,
        context_chunks: List[DocumentChunk],
        llm_client: Optional[object] = None,
    ) -> FaithfulnessReport:
        """
        Decomposes answer into atomic sentences, verifies each against context,
        and computes faithfulness and relevancy metrics.
        """
        if not answer or not context_chunks:
            return FaithfulnessReport(
                faithfulness_score=0.0,
                answer_relevancy_score=0.0,
                hallucination_rate=1.0,
                total_claims=0,
                supported_claims=0,
            )

        # Decompose answer into claim sentences
        raw_claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 15]
        if not raw_claims:
            raw_claims = [answer]

        context_full_text = " ".join(c.text.lower() for c in context_chunks)
        verifications: List[ClaimVerification] = []
        supported_count = 0

        for claim in raw_claims:
            # Strip timestamps [MM:SS] for clean token analysis
            clean_claim = re.sub(r"\[\d{2}:\d{2}(?::\d{2})?\]", "", claim).strip()
            claim_words = [w for w in re.findall(r"\b\w{3,}\b", clean_claim.lower())]

            if not claim_words:
                continue

            # Check overlap against context
            matched = sum(1 for w in claim_words if w in context_full_text)
            overlap = matched / len(claim_words)

            is_supported = overlap >= 0.50
            if is_supported:
                supported_count += 1

            verifications.append(
                ClaimVerification(
                    claim=clean_claim,
                    is_supported=is_supported,
                    confidence=float(overlap),
                    supporting_quote=self._find_best_match(clean_claim, context_chunks) if is_supported else None,
                )
            )

        total_claims = len(verifications)
        faith_score = (supported_count / total_claims) if total_claims > 0 else 1.0
        hallucination_rate = 1.0 - faith_score

        # Answer Relevancy
        relevancy_score = self._compute_relevancy(question, answer)

        return FaithfulnessReport(
            faithfulness_score=float(faith_score),
            answer_relevancy_score=float(relevancy_score),
            hallucination_rate=float(hallucination_rate),
            total_claims=total_claims,
            supported_claims=supported_count,
            claim_verifications=verifications,
        )

    def _find_best_match(self, claim: str, chunks: List[DocumentChunk]) -> Optional[str]:
        words = set(claim.lower().split())
        best_chunk = None
        best_overlap = 0
        for c in chunks:
            c_words = set(c.text.lower().split())
            ov = len(words.intersection(c_words))
            if ov > best_overlap:
                best_overlap = ov
                best_chunk = c
        return best_chunk.text[:120] + "..." if best_chunk else None

    def _compute_relevancy(self, question: str, answer: str) -> float:
        q_words = set(re.findall(r"\b\w{3,}\b", question.lower()))
        a_words = set(re.findall(r"\b\w{3,}\b", answer.lower()))
        if not q_words:
            return 0.8
        intersection = len(q_words.intersection(a_words))
        return float(min(1.0, max(0.2, (intersection / len(q_words)) * 1.5)))

