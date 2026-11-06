"""
Contextual Question Answering with Timestamp References,
Visual Frame Evidence, and Hallucination Prevention.
"""
from dataclasses import dataclass, field
import logging
import re
from typing import List, Optional

from core.ingestion.frame_extractor import VideoFrame
from core.qa.hallucination_guard import GuardResult, HallucinationGuard
from core.qa.llm_clients import UnifiedLLMClient
from core.retrieval.hybrid_retriever import ScoredChunk
from core.utils.time_utils import format_youtube_link, seconds_to_timestamp

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    timestamp_str: str
    seconds: float
    youtube_url: str
    quote: str
    score: float
    frame_path: Optional[str] = None


@dataclass
class QAResponse:
    question: str
    answer: str
    is_answerable: bool
    confidence_score: float
    citations: List[Citation] = field(default_factory=list)
    retrieved_chunks: List[ScoredChunk] = field(default_factory=list)
    guard_result: Optional[GuardResult] = None


class ContextualQA:
    """
    Contextual Q&A orchestrator that binds hybrid retrieval, hallucination guard,
    citation verification, and multimodal frame evidence.
    """

    def __init__(
        self,
        llm_client: Optional[UnifiedLLMClient] = None,
        guard: Optional[HallucinationGuard] = None,
    ):
        self.llm_client = llm_client or UnifiedLLMClient()
        self.guard = guard or HallucinationGuard()

    def answer_question(
        self,
        question: str,
        retrieved_chunks: List[ScoredChunk],
        video_id: str,
        video_title: str = "",
        frames: Optional[List[VideoFrame]] = None,
    ) -> QAResponse:
        """
        Processes question over retrieved chunks and formats timestamp-grounded answer.
        """
        # Step 1: Hallucination Guard Check
        guard_res = self.guard.evaluate(question, retrieved_chunks, video_title=video_title)
        if not guard_res.is_answerable:
            return QAResponse(
                question=question,
                answer=guard_res.suggested_response or "This topic is not discussed in the video.",
                is_answerable=False,
                confidence_score=guard_res.confidence,
                retrieved_chunks=retrieved_chunks,
                guard_result=guard_res,
            )

        # Step 2: Build Context with Timestamps
        context_blocks = []
        for i, sc in enumerate(retrieved_chunks, 1):
            ts = sc.chunk.timestamp_str
            context_blocks.append(f"Passage {i} {ts}:\n\"{sc.chunk.text}\"")

        context_str = "\n\n".join(context_blocks)

        system_instruction = (
            "You are an expert AI teaching assistant for video lectures.\n"
            "Answer the student's question accurately and concisely, relying SOLELY on the provided video passages.\n"
            "MANDATORY RULES:\n"
            "1. Whenever you mention a fact, concept, or quote, you MUST cite the exact timestamp in brackets like [MM:SS] or [HH:MM:SS].\n"
            "2. If the context does not explicitly provide the answer, say: 'The video does not provide information to fully answer this.'\n"
            "3. Do not invent timestamps or facts not found in the context."
        )

        prompt = (
            f"VIDEO TITLE: {video_title}\n\n"
            f"CONTEXT PASSAGES:\n{context_str}\n\n"
            f"USER QUESTION: {question}\n\n"
            "ANSWER (remember to include [MM:SS] timestamp citations):"
        )

        raw_answer = self.llm_client.generate(prompt=prompt, system_instruction=system_instruction)

        # Step 3: Extract Citations & Link Visual Frames
        citations: List[Citation] = []
        for sc in retrieved_chunks:
            chunk = sc.chunk
            # Find matching frame close to chunk start
            frame_path = None
            if frames:
                matching = [f for f in frames if abs(f.timestamp - chunk.start_time) <= 30]
                if matching:
                    frame_path = matching[0].image_path

            citations.append(
                Citation(
                    timestamp_str=chunk.start_timestamp,
                    seconds=chunk.start_time,
                    youtube_url=format_youtube_link(video_id, chunk.start_time),
                    quote=chunk.text[:140] + ("..." if len(chunk.text) > 140 else ""),
                    score=sc.final_score,
                    frame_path=frame_path,
                )
            )

        return QAResponse(
            question=question,
            answer=raw_answer,
            is_answerable=True,
            confidence_score=guard_res.confidence,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            guard_result=guard_res,
        )

