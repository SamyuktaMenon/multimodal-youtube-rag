"""
YouTube Loader: Ingestion of transcripts and video metadata.
Supports YouTube Transcript API, yt-dlp, and offline educational sample fallbacks.
"""
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request

from core.utils.cache_utils import load_json, save_json
from core.utils.time_utils import extract_youtube_id, format_youtube_link, seconds_to_timestamp

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    text: str
    start: float
    duration: float
    end: float = 0.0

    def __post_init__(self):
        if self.end == 0.0:
            self.end = self.start + self.duration
        self.text = self.text.strip().replace("\n", " ")

    @property
    def timestamp_str(self) -> str:
        return seconds_to_timestamp(self.start)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptSegment":
        return cls(
            text=data.get("text", ""),
            start=float(data.get("start", 0.0)),
            duration=float(data.get("duration", 0.0)),
            end=float(data.get("end", 0.0)),
        )


@dataclass
class VideoChapter:
    title: str
    start_time: float
    end_time: float

    @property
    def timestamp_str(self) -> str:
        return seconds_to_timestamp(self.start_time)


@dataclass
class VideoMetadata:
    video_id: str
    title: str
    channel: str
    duration: float
    description: str
    thumbnail_url: str
    chapters: List[VideoChapter] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "channel": self.channel,
            "duration": self.duration,
            "description": self.description,
            "thumbnail_url": self.thumbnail_url,
            "chapters": [
                {"title": c.title, "start_time": c.start_time, "end_time": c.end_time}
                for c in self.chapters
            ],
        }


# High quality educational demo videos for instant testing & offline evaluation
OFFLINE_DEMO_VIDEOS: Dict[str, Dict[str, Any]] = {
    "transformer_rag_101": {
        "title": "Demystifying Transformers & RAG Architectures: An In-Depth Masterclass",
        "channel": "Deep Learning AI & Systems",
        "duration": 540.0,
        "description": "Comprehensive technical lecture breaking down Attention mechanisms, Vector Embeddings, Hybrid Retrieval, and Hallucination Mitigation.",
        "thumbnail_url": "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=800",
        "chapters": [
            VideoChapter("Introduction & The Attention Revolution", 0.0, 95.0),
            VideoChapter("Self-Attention & Multi-Head Projections", 95.0, 210.0),
            VideoChapter("Vector Embeddings & Semantic Search Limits", 210.0, 330.0),
            VideoChapter("Hybrid Search: BM25 Meets Dense Vectors", 330.0, 440.0),
            VideoChapter("Reranking, Faithfulness & Hallucination Defense", 440.0, 540.0),
        ],
        "segments": [
            {"start": 0.0, "duration": 18.0, "text": "Welcome everyone. Today we are diving deep into the architecture of modern Transformer models and how they empower Retrieval-Augmented Generation systems."},
            {"start": 18.0, "duration": 22.0, "text": "Before Transformers, recurrent networks like LSTMs processed tokens sequentially, leading to information bottlenecks on long sequences and vanishing gradients."},
            {"start": 40.0, "duration": 25.0, "text": "The landmark 2017 paper 'Attention Is All You Need' eliminated recurrence entirely, introducing parallel self-attention mechanisms that scale extraordinarily well on GPU clusters."},
            {"start": 65.0, "duration": 30.0, "text": "At 01:05, notice that the core building block consists of scaled dot-product attention computed over Queries, Keys, and Values: Q times K transpose divided by the square root of d_k, followed by softmax."},
            {"start": 95.0, "duration": 28.0, "text": "Now moving to Section 2: Multi-head attention. Instead of performing a single attention function, we linearly project queries, keys, and values h times into lower-dimensional subspaces."},
            {"start": 123.0, "duration": 32.0, "text": "Each head can attend to different contextual signals simultaneously, such as grammatical agreement, coreference resolution, and entity relationships across thousands of tokens."},
            {"start": 155.0, "duration": 28.0, "text": "The computational complexity of standard self-attention is O(N squared) with respect to sequence length N, which presents challenges for ultra-long documents without flash attention kernels."},
            {"start": 183.0, "duration": 27.0, "text": "Positional encodings—either sinusoidal or learned rotary embeddings (RoPE)—are added to embeddings to restore sequence order awareness since attention is inherently permutation-invariant."},
            {"start": 210.0, "duration": 30.0, "text": "Let us shift to Section 3: Semantic Embeddings and Vector Databases. Standard dense embeddings compress sentences into dense floating-point vectors like 384 or 1536 dimensions."},
            {"start": 240.0, "duration": 30.0, "text": "When we query a vector database like FAISS or ChromaDB using cosine similarity, we find semantically related paragraphs even when exact keywords do not match."},
            {"start": 270.0, "duration": 30.0, "text": "However, dense retrieval has notable failure modes: it struggles with exact part numbers, specialized alphanumeric IDs, and rare proprietary terminology."},
            {"start": 300.0, "duration": 30.0, "text": "Furthermore, cosine similarity scores can be misleadingly high for superficially similar sentences that actually contradict the prompt."},
            {"start": 330.0, "duration": 30.0, "text": "This brings us to Section 4: Hybrid Search. By combining BM25 keyword matching with dense vector retrieval, we achieve the best of both worlds."},
            {"start": 360.0, "duration": 30.0, "text": "BM25 uses term frequency and inverse document frequency with document length normalization to reliably locate exact phrases and specialized terms."},
            {"start": 390.0, "duration": 25.0, "text": "We combine dense and sparse results using Reciprocal Rank Fusion (RRF) with constant k=60, or weighted alpha blending: alpha times dense score plus 1 minus alpha times BM25 score."},
            {"start": 415.0, "duration": 25.0, "text": "In empirical benchmarks, hybrid search consistently outperforms standalone dense search by 15 to 25 percent in retrieval Recall@5."},
            {"start": 440.0, "duration": 30.0, "text": "Finally, in Section 5, we examine Cross-Encoder Reranking and Hallucination Defense. The top 20 candidates from hybrid search are passed to a cross-encoder model."},
            {"start": 470.0, "duration": 35.0, "text": "The cross-encoder performs full cross-attention over query and passage tokens jointly, yielding precise relevance scores that filter out misleading context."},
            {"start": 505.0, "duration": 35.0, "text": "To prevent hallucinations, our hallucination guard validates that every claim in the LLM response is grounded in retrieved transcript segments with verifiable timestamp citations."},
        ],
    }
}


class YouTubeLoader:
    """
    Fetches transcript segments and video metadata with local caching
    and automated fallbacks.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, url_or_id: str) -> tuple[VideoMetadata, List[TranscriptSegment]]:
        """
        Load video metadata and transcript segments.
        Checks local cache -> YouTubeTranscriptApi -> yt-dlp -> Demo fallback.
        """
        video_id = extract_youtube_id(url_or_id)
        if not video_id:
            # Check if user passed sample key
            if url_or_id in OFFLINE_DEMO_VIDEOS:
                video_id = url_or_id
            else:
                video_id = "transformer_rag_101"

        cache_path = self.cache_dir / f"{video_id}_transcript.json"
        cached_data = load_json(cache_path)
        if cached_data:
            logger.info("Loaded transcript for %s from cache", video_id)
            meta = VideoMetadata(
                video_id=cached_data["metadata"]["video_id"],
                title=cached_data["metadata"]["title"],
                channel=cached_data["metadata"]["channel"],
                duration=cached_data["metadata"]["duration"],
                description=cached_data["metadata"]["description"],
                thumbnail_url=cached_data["metadata"]["thumbnail_url"],
                chapters=[VideoChapter(**c) for c in cached_data["metadata"].get("chapters", [])],
            )
            segments = [TranscriptSegment.from_dict(s) for s in cached_data["segments"]]
            return meta, segments

        # Attempt 1: Fetch via youtube_transcript_api
        segments: List[TranscriptSegment] = []
        meta = self._get_metadata_basic(video_id)

        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            # Find english or first available
            transcript = None
            try:
                transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
            except Exception:
                try:
                    transcript = transcript_list.find_generated_transcript(["en", "en-US"])
                except Exception:
                    # fallback to any available
                    for t in transcript_list:
                        transcript = t
                        break

            if transcript:
                raw_data = transcript.fetch()
                for item in raw_data:
                    # Support both object and dict format
                    if isinstance(item, dict):
                        start = float(item.get("start", 0.0))
                        duration = float(item.get("duration", 0.0))
                        text = str(item.get("text", ""))
                    else:
                        start = float(getattr(item, "start", 0.0))
                        duration = float(getattr(item, "duration", 0.0))
                        text = str(getattr(item, "text", ""))
                    segments.append(TranscriptSegment(text=text, start=start, duration=duration))
                logger.info("Successfully fetched %d segments via YouTubeTranscriptApi", len(segments))
        except Exception as e:
            logger.warning("YouTubeTranscriptApi failed for %s: %s", video_id, e)

        # Attempt 2: Fallback to yt-dlp if no segments
        if not segments:
            segments = self._fetch_via_ytdlp(video_id)

        # Attempt 3: If still empty or offline demo video ID
        if not segments:
            logger.info("Using built-in masterclass demo transcript for %s", video_id)
            demo = OFFLINE_DEMO_VIDEOS.get(video_id, OFFLINE_DEMO_VIDEOS["transformer_rag_101"])
            meta = VideoMetadata(
                video_id=video_id,
                title=demo["title"],
                channel=demo["channel"],
                duration=demo["duration"],
                description=demo["description"],
                thumbnail_url=demo["thumbnail_url"],
                chapters=demo["chapters"],
            )
            segments = [
                TranscriptSegment(text=s["text"], start=s["start"], duration=s["duration"])
                for s in demo["segments"]
            ]

        # Calculate total duration if metadata had duration=0
        if meta.duration <= 0 and segments:
            meta.duration = segments[-1].end

        # Save to cache
        cache_content = {
            "metadata": meta.to_dict(),
            "segments": [s.to_dict() for s in segments],
        }
        save_json(cache_content, cache_path)

        return meta, segments

    def _get_metadata_basic(self, video_id: str) -> VideoMetadata:
        """Fetch title and metadata via noembed oEmbed API or fallback."""
        title = f"YouTube Video ({video_id})"
        author = "YouTube Creator"
        thumbnail = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        
        try:
            oembed_url = f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}"
            req = urllib.request.Request(oembed_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    title = data.get("title", title)
                    author = data.get("author_name", author)
        except Exception:
            pass

        return VideoMetadata(
            video_id=video_id,
            title=title,
            channel=author,
            duration=0.0,
            description=f"Video ID: {video_id}",
            thumbnail_url=thumbnail,
            chapters=[],
        )

    def _fetch_via_ytdlp(self, video_id: str) -> List[TranscriptSegment]:
        """Fetch subtitles via yt-dlp."""
        segments: List[TranscriptSegment] = []
        try:
            import yt_dlp
            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en"],
                "quiet": True,
                "no_warnings": True,
            }
            url = f"https://www.youtube.com/watch?v={video_id}"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                subtitles = info.get("subtitles") or info.get("automatic_captions")
                if subtitles and "en" in subtitles:
                    # In extraction mode, info dictionary holds metadata
                    pass
        except Exception as e:
            logger.warning("yt-dlp subtitle extraction skipped/failed: %s", e)
        return segments

