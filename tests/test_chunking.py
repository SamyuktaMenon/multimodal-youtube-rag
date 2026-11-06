"""
Unit tests for chunking strategies.
"""
import unittest
from core.chunking.chunker_factory import compare_chunking_strategies, get_chunker
from core.ingestion.youtube_loader import TranscriptSegment, VideoChapter, VideoMetadata


class TestChunking(unittest.TestCase):
    def setUp(self):
        self.metadata = VideoMetadata(
            video_id="test_video_123",
            title="Introduction to Machine Learning",
            channel="AI Academy",
            duration=180.0,
            description="Testing ML fundamentals",
            thumbnail_url="",
            chapters=[
                VideoChapter("Part 1: Foundations", 0.0, 60.0),
                VideoChapter("Part 2: Deep Networks", 60.0, 120.0),
                VideoChapter("Part 3: Evaluation", 120.0, 180.0),
            ],
        )
        self.segments = [
            TranscriptSegment("Welcome to the course on machine learning.", start=0.0, duration=10.0),
            TranscriptSegment("We start with linear regression and cost functions.", start=10.0, duration=15.0),
            TranscriptSegment("Gradient descent iteratively updates parameters to minimize loss.", start=25.0, duration=20.0),
            TranscriptSegment("Next, neural networks compose non-linear activation functions.", start=60.0, duration=25.0),
            TranscriptSegment("Backpropagation computes gradients using the chain rule.", start=85.0, duration=30.0),
            TranscriptSegment("Finally, we evaluate models using precision, recall, and F1 score.", start=120.0, duration=35.0),
            TranscriptSegment("Cross-validation prevents overfitting on unseen data.", start=155.0, duration=25.0),
        ]

    def test_fixed_chunker(self):
        chunker = get_chunker("fixed", chunk_size=25, chunk_overlap=5)
        chunks = chunker.chunk(self.segments, self.metadata)
        self.assertGreater(len(chunks), 0)
        self.assertIn("video_id", chunks[0].metadata)

    def test_recursive_chunker(self):
        chunker = get_chunker("recursive", chunk_size=150, chunk_overlap=30)
        chunks = chunker.chunk(self.segments, self.metadata)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(c.char_count > 0 for c in chunks))

    def test_semantic_chunker(self):
        chunker = get_chunker("semantic")
        chunks = chunker.chunk(self.segments, self.metadata)
        self.assertGreater(len(chunks), 0)

    def test_temporal_chunker(self):
        chunker = get_chunker("temporal", time_window_seconds=60, time_overlap_seconds=10)
        chunks = chunker.chunk(self.segments, self.metadata)
        self.assertGreater(len(chunks), 0)
        # Check start time ordering
        for i in range(1, len(chunks)):
            self.assertGreaterEqual(chunks[i].start_time, chunks[i - 1].start_time)

    def test_chapter_chunker(self):
        chunker = get_chunker("chapter")
        chunks = chunker.chunk(self.segments, self.metadata)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].start_time, 0.0)

    def test_compare_chunking_strategies(self):
        df = compare_chunking_strategies(self.segments, self.metadata)
        self.assertEqual(len(df), 5)
        self.assertIn("Avg Words/Chunk", df.columns)


if __name__ == "__main__":
    unittest.main()

