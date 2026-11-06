"""
Unit tests for QA, Hallucination Guard, Evaluation, and Learning features.
"""
import unittest

from core.chunking.chunker_base import DocumentChunk
from core.evaluation.faithfulness_eval import FaithfulnessEvaluator
from core.evaluation.retrieval_eval import EvalQuery, RetrievalEvaluator
from core.learning.flashcard_gen import FlashcardGenerator
from core.learning.notes_generator import NotesGenerator
from core.learning.quiz_generator import QuizGenerator
from core.qa.contextual_qa import ContextualQA
from core.qa.hallucination_guard import HallucinationGuard
from core.retrieval.hybrid_retriever import ScoredChunk


class TestQAAndEval(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            DocumentChunk("c1", "Attention is all you need. Queries, Keys, and Values compute attention weights.", 0.0, 30.0, "test"),
            DocumentChunk("c2", "Multi-head attention projects vectors into lower-dimensional subspaces.", 30.0, 60.0, "test"),
            DocumentChunk("c3", "Positional encodings restore sequence order awareness in Transformers.", 60.0, 90.0, "test"),
        ]
        self.scored_chunks = [
            ScoredChunk(chunk=self.chunks[0], final_score=0.88, dense_score=0.85, bm25_score=0.90),
            ScoredChunk(chunk=self.chunks[1], final_score=0.72, dense_score=0.70, bm25_score=0.75),
        ]

    def test_hallucination_guard_answerable(self):
        guard = HallucinationGuard()
        res = guard.evaluate("What are queries, keys, and values?", self.scored_chunks, video_title="Transformers")
        self.assertTrue(res.is_answerable)

    def test_hallucination_guard_unanswerable(self):
        guard = HallucinationGuard()
        unrelated_chunks = [
            ScoredChunk(chunk=self.chunks[0], final_score=0.10, dense_score=0.10, bm25_score=0.10)
        ]
        res = guard.evaluate("How do you bake sourdough bread with yeast?", unrelated_chunks, video_title="Transformers")
        self.assertFalse(res.is_answerable)
        self.assertIn("Not in Video", res.suggested_response)

    def test_contextual_qa(self):
        qa = ContextualQA()
        response = qa.answer_question(
            question="Explain self attention",
            retrieved_chunks=self.scored_chunks,
            video_id="demo123",
            video_title="Transformer Masterclass",
        )
        self.assertTrue(response.is_answerable)
        self.assertGreater(len(response.citations), 0)
        self.assertIn("https://www.youtube.com/watch?v=demo123", response.citations[0].youtube_url)

    def test_retrieval_evaluator(self):
        evaluator = RetrievalEvaluator(k=2)
        eval_queries = [
            EvalQuery("q1", "What is attention?", {"c1"}),
            EvalQuery("q2", "Explain positional encoding", {"c3"}),
        ]
        # Mock retriever
        def mock_retriever(q: str, k: int):
            if "attention" in q:
                return self.chunks[:k]
            return [self.chunks[2], self.chunks[1]][:k]

        metrics = evaluator.evaluate(mock_retriever, eval_queries)
        self.assertEqual(metrics.k, 2)
        self.assertGreaterEqual(metrics.hit_rate_at_k, 0.5)
        self.assertGreaterEqual(metrics.mrr, 0.5)

    def test_faithfulness_evaluator(self):
        evaluator = FaithfulnessEvaluator()
        question = "What compute attention weights?"
        answer = "Queries, Keys, and Values compute attention weights at [00:10]."
        rep = evaluator.evaluate(question, answer, self.chunks)
        self.assertGreaterEqual(rep.faithfulness_score, 0.5)
        self.assertLessEqual(rep.hallucination_rate, 0.5)

    def test_quiz_generator(self):
        quiz_gen = QuizGenerator()
        questions = quiz_gen.generate_quiz(self.chunks, video_id="demo123", num_questions=2)
        self.assertEqual(len(questions), 2)
        self.assertEqual(len(questions[0].options), 4)
        self.assertIn(questions[0].correct_idx, [0, 1, 2, 3])

    def test_flashcard_generator(self):
        card_gen = FlashcardGenerator()
        cards = card_gen.generate_flashcards(self.chunks, video_id="demo123", num_cards=2)
        self.assertEqual(len(cards), 2)
        self.assertTrue(len(cards[0].front) > 0)
        self.assertTrue(len(cards[0].back) > 0)

    def test_notes_generator(self):
        from core.ingestion.youtube_loader import VideoMetadata
        meta = VideoMetadata(
            video_id="demo123",
            title="Transformer Masterclass",
            channel="AI",
            duration=90.0,
            description="Deep dive",
            thumbnail_url="",
        )
        notes_gen = NotesGenerator()
        notes = notes_gen.generate_notes(meta, self.chunks, sections=[])
        md = notes.to_markdown()
        self.assertIn("# 📚 Comprehensive Study Notes", md)
        self.assertIn("Technical Glossary", md)


if __name__ == "__main__":
    unittest.main()

