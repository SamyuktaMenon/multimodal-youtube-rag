"""
Multimodal Processor: Combines text transcripts, audio metadata, and visual frames
into unified multimodal sections, and generates section summaries.
"""
from dataclasses import dataclass, field
import logging
from typing import List, Optional

from core.ingestion.frame_extractor import FrameExtractor, VideoFrame
from core.ingestion.youtube_loader import TranscriptSegment, VideoChapter, VideoMetadata
from core.utils.time_utils import seconds_to_timestamp

logger = logging.getLogger(__name__)


@dataclass
class SectionSummary:
    section_id: int
    title: str
    start_time: float
    end_time: float
    summary: str
    key_takeaways: List[str]
    representative_frame: Optional[VideoFrame] = None

    @property
    def timestamp_str(self) -> str:
        return f"{seconds_to_timestamp(self.start_time)} - {seconds_to_timestamp(self.end_time)}"


@dataclass
class MultimodalDocument:
    metadata: VideoMetadata
    segments: List[TranscriptSegment]
    frames: List[VideoFrame]
    sections: List[SectionSummary] = field(default_factory=list)


class MultimodalProcessor:
    """
    Coordinates transcript ingestion, frame extraction, multimodal alignment,
    and section-by-section summary generation.
    """

    def __init__(self, frame_extractor: Optional[FrameExtractor] = None):
        self.frame_extractor = frame_extractor or FrameExtractor()

    def process(
        self,
        metadata: VideoMetadata,
        segments: List[TranscriptSegment],
        extract_visual_frames: bool = True,
        llm_client: Optional[object] = None,
    ) -> MultimodalDocument:
        """
        Process the raw segments, extract frames, align multimodal signals,
        and generate structured section summaries.
        """
        frames: List[VideoFrame] = []
        if extract_visual_frames and segments:
            frames = self.frame_extractor.generate_multimodal_frames_from_segments(
                video_id=metadata.video_id,
                segments=segments,
                interval_seconds=30,
                max_frames=20,
            )

        # Build sections from chapters or automatic segmentation
        sections = self._build_section_summaries(metadata, segments, frames, llm_client)

        return MultimodalDocument(
            metadata=metadata,
            segments=segments,
            frames=frames,
            sections=sections,
        )

    def _build_section_summaries(
        self,
        metadata: VideoMetadata,
        segments: List[TranscriptSegment],
        frames: List[VideoFrame],
        llm_client: Optional[object] = None,
    ) -> List[SectionSummary]:
        """
        Divide the transcript into coherent sections and generate summaries.
        """
        if not segments:
            return []

        total_duration = metadata.duration if metadata.duration > 0 else segments[-1].end

        # If chapters exist, use them; otherwise auto-divide into 3-5 sections
        chapter_bounds: List[tuple[str, float, float]] = []
        if metadata.chapters:
            for ch in metadata.chapters:
                chapter_bounds.append((ch.title, ch.start_time, ch.end_time))
        else:
            # Auto-divide into roughly 3 to 5 logical sections
            num_sections = max(2, min(5, len(segments) // 5))
            step = total_duration / num_sections
            for i in range(num_sections):
                start = i * step
                end = (i + 1) * step if i < num_sections - 1 else total_duration
                chapter_bounds.append((f"Part {i+1}: Educational Overview", start, end))

        summaries: List[SectionSummary] = []
        for idx, (title, start, end) in enumerate(chapter_bounds, 1):
            sec_segments = [s for s in segments if s.start >= start and s.start < end]
            if not sec_segments:
                # Catch boundary segments
                sec_segments = [s for s in segments if s.end > start and s.start < end]

            sec_text = " ".join(s.text for s in sec_segments)
            if not sec_text:
                continue

            # Find matching frame
            matching_frames = [f for f in frames if start <= f.timestamp <= end]
            rep_frame = matching_frames[0] if matching_frames else None

            # Generate summary text
            sec_summary, takeaways = self._summarize_section_text(title, sec_text, llm_client)

            summaries.append(
                SectionSummary(
                    section_id=idx,
                    title=title,
                    start_time=start,
                    end_time=end,
                    summary=sec_summary,
                    key_takeaways=takeaways,
                    representative_frame=rep_frame,
                )
            )

        return summaries

    def _summarize_section_text(
        self, title: str, text: str, llm_client: Optional[object] = None
    ) -> tuple[str, List[str]]:
        """
        Summarizes section text using LLM if available, or structured extractive summarizer.
        """
        if llm_client and hasattr(llm_client, "generate_section_summary"):
            try:
                res = llm_client.generate_section_summary(title, text)
                if res and isinstance(res, tuple):
                    return res
            except Exception as e:
                logger.warning("LLM section summarization failed, falling back: %s", e)

        # High-quality extractive summary fallback
        sentences = [s.strip() for s in text.split(". ") if len(s.strip()) > 15]
        if not sentences:
            return f"Overview covering {title}.", ["Key discussion point covering the module."]

        summary = ". ".join(sentences[:2]) + "."
        takeaways = []
        for s in sentences[1:4]:
            clean = s.rstrip(".")
            if len(clean) > 20:
                takeaways.append(f"{clean}.")
        if not takeaways:
            takeaways = [f"Foundational concepts of {title}.", "Core methodologies and operational workflows."]

        return summary, takeaways

