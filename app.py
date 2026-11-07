"""
Multimodal YouTube RAG Chatbot - Streamlit Application
Advanced RAG, Multimodal Ingestion, Contextual QA, Evaluation & Learning Suite.
"""
from pathlib import Path
import time
import pandas as pd
import streamlit as st

from config import CACHE_DIR, EXPORTS_DIR, FRAMES_DIR
from core.chunking.chunker_factory import compare_chunking_strategies, get_chunker
from core.embeddings.embedding_factory import get_embedding_model
from core.evaluation.benchmark_suite import BenchmarkSuite
from core.evaluation.faithfulness_eval import FaithfulnessEvaluator
from core.evaluation.retrieval_eval import RetrievalEvaluator
from core.ingestion.frame_extractor import FrameExtractor
from core.ingestion.multimodal_processor import MultimodalProcessor
from core.ingestion.youtube_loader import OFFLINE_DEMO_VIDEOS, YouTubeLoader
from core.learning.flashcard_gen import FlashcardGenerator
from core.learning.notes_generator import NotesGenerator
from core.learning.quiz_generator import QuizGenerator
from core.qa.contextual_qa import ContextualQA
from core.qa.hallucination_guard import HallucinationGuard
from core.qa.llm_clients import UnifiedLLMClient
from core.retrieval.hybrid_retriever import HybridRetriever
from core.retrieval.reranker import CrossEncoderReranker
from core.utils.time_utils import extract_youtube_id, format_youtube_link, seconds_to_timestamp
from core.vectorstores.chroma_store import ChromaDBVectorStore
from core.vectorstores.comparator import VectorStoreComparator
from core.vectorstores.faiss_store import FAISSVectorStore

# Page Config
st.set_page_config(
    page_title="Multimodal YouTube RAG & Learning Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        background-color: #1e293b;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #38bdf8;
        margin-bottom: 12px;
    }
    .citation-badge {
        display: inline-block;
        background-color: #1e1b4b;
        color: #a5b4fc;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        margin-right: 6px;
        border: 1px solid #4338ca;
        text-decoration: none;
    }
    .citation-badge:hover { color: #ffffff; background-color: #312e81; }
    .card-box {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state():
    defaults = {
        "metadata": None,
        "segments": [],
        "multimodal_doc": None,
        "chunks": [],
        "vector_store": None,
        "embedding_model": None,
        "hybrid_retriever": None,
        "reranker": None,
        "qa_history": [],
        "active_chunk_strategy": "recursive",
        "active_vstore_type": "faiss",
        "quiz_questions": [],
        "flashcards": [],
        "study_notes": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


initialize_session_state()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("🎬 Multimodal Video RAG")
    st.caption("Advanced Multi-Vector Search & Learning Studio")

    st.subheader("1. Video Source")
    demo_options = ["transformer_rag_101 (Demo Masterclass)", "Custom YouTube URL"]
    source_choice = st.selectbox("Choose Input", demo_options)

    if source_choice == "Custom YouTube URL":
        video_url = st.text_input(
            "Enter YouTube URL",
            value="https://www.youtube.com/watch?v=kCc8FmEb1nY",
            placeholder="https://www.youtube.com/watch?v=...",
        )
    else:
        video_url = "transformer_rag_101"

    st.subheader("2. RAG Pipeline Configuration")
    chunk_strategy = st.selectbox(
        "Chunking Strategy",
        ["recursive", "fixed", "semantic", "temporal", "chapter"],
        index=0,
        help="Recursive, Fixed Word, Semantic break detection, Temporal interval, or Chapter based.",
    )
    vstore_type = st.selectbox(
        "Vector Database",
        ["faiss", "chroma"],
        index=0,
        help="FAISS (in-memory C++ similarity) or ChromaDB (HNSW collection).",
    )
    embed_provider = st.selectbox(
        "Embedding Model",
        ["sentence-transformers", "clip", "tfidf", "google", "openai"],
        index=0,
        help="Local SentenceTransformers, CLIP (Multimodal ViT), TF-IDF, or Cloud APIs.",
    )
    llm_provider = st.selectbox(
        "LLM Provider",
        ["gemini", "openai", "mock"],
        index=0,
        help="Choose active LLM or Mock/Extractive fallback.",
    )

    api_key_input = None
    if llm_provider in ["gemini", "openai"]:
        api_key_input = st.text_input(f"{llm_provider.capitalize()} API Key", type="password")

    enable_reranking = st.checkbox("Enable Cross-Encoder Reranker", value=True)
    extract_frames = st.checkbox("Extract Visual Frames (Multimodal)", value=True)

    load_btn = st.button("🚀 Process & Ingest Video", use_container_width=True, type="primary")


# ----------------- PROCESSING PIPELINE -----------------
def process_video_pipeline():
    with st.spinner("⏳ Extracting video transcript, metadata, and visual frames..."):
        loader = YouTubeLoader(cache_dir=CACHE_DIR)
        metadata, segments = loader.load(video_url)

        # Multimodal processing
        processor = MultimodalProcessor(frame_extractor=FrameExtractor(output_dir=FRAMES_DIR))
        mm_doc = processor.process(metadata, segments, extract_visual_frames=extract_frames)

        # Chunking
        chunker = get_chunker(chunk_strategy)
        chunks = chunker.chunk(segments, metadata)

        # Embeddings & Vector Store
        emb_model = get_embedding_model(provider=embed_provider, api_key=api_key_input)
        texts = [c.text for c in chunks]

        # Vector Store
        if vstore_type == "chroma":
            store = ChromaDBVectorStore(collection_name=f"coll_{metadata.video_id[:8]}")
            store.clear()
        else:
            store = FAISSVectorStore(metric="cosine")

        if embed_provider in ["clip", "multimodal"] and mm_doc.frames:
            # Genuine multimodal vector fusion: fuse text passage vectors with nearest keyframe image vectors
            frame_map = {f.timestamp: f.image_path for f in mm_doc.frames}
            doc_embeddings = []
            for c in chunks:
                nearest_img = None
                for ts, ipath in frame_map.items():
                    if abs(ts - c.start_time) <= 35 and os.path.exists(ipath):
                        nearest_img = ipath
                        break
                if nearest_img:
                    t_vec = np.array(emb_model.embed_query(c.text))
                    i_vec = np.array(emb_model.embed_image(nearest_img))
                    composite = (t_vec + i_vec) / 2.0
                    composite = composite / (np.linalg.norm(composite) + 1e-9)
                    doc_embeddings.append(composite.tolist())
                else:
                    doc_embeddings.append(emb_model.embed_query(c.text))
        else:
            doc_embeddings = emb_model.embed_documents(texts)

        store.add_documents(chunks, doc_embeddings)

        # Hybrid Retriever
        hybrid = HybridRetriever(
            vector_store=store,
            embedding_model=emb_model,
            chunks=chunks,
            alpha=0.65,
            fusion_method="weighted",
        )

        reranker = CrossEncoderReranker() if enable_reranking else None

        # Store in session state
        st.session_state["metadata"] = metadata
        st.session_state["segments"] = segments
        st.session_state["multimodal_doc"] = mm_doc
        st.session_state["chunks"] = chunks
        st.session_state["vector_store"] = store
        st.session_state["embedding_model"] = emb_model
        st.session_state["hybrid_retriever"] = hybrid
        st.session_state["reranker"] = reranker
        st.session_state["active_chunk_strategy"] = chunk_strategy
        st.session_state["active_vstore_type"] = vstore_type

        # Generate pre-cached learning materials
        quiz_gen = QuizGenerator(UnifiedLLMClient(provider=llm_provider, api_key=api_key_input))
        st.session_state["quiz_questions"] = quiz_gen.generate_quiz(chunks, metadata.video_id, num_questions=4)

        card_gen = FlashcardGenerator(UnifiedLLMClient(provider=llm_provider, api_key=api_key_input))
        st.session_state["flashcards"] = card_gen.generate_flashcards(chunks, metadata.video_id, num_cards=5)

        notes_gen = NotesGenerator(UnifiedLLMClient(provider=llm_provider, api_key=api_key_input))
        st.session_state["study_notes"] = notes_gen.generate_notes(metadata, chunks, mm_doc.sections)


if load_btn or st.session_state["metadata"] is None:
    process_video_pipeline()

metadata = st.session_state["metadata"]
chunks = st.session_state["chunks"]
mm_doc = st.session_state["multimodal_doc"]
hybrid_retriever = st.session_state["hybrid_retriever"]
reranker = st.session_state["reranker"]

# ----------------- MAIN TABS -----------------
st.title(f"📺 {metadata.title if metadata else 'Multimodal Video RAG'}")
if metadata:
    st.caption(f"**Channel**: {metadata.channel}  |  **Duration**: {seconds_to_timestamp(metadata.duration)}  |  **Indexed Chunks**: {len(chunks)} ({chunk_strategy.capitalize()})")

tab_qa, tab_mm, tab_rag, tab_eval, tab_learning = st.tabs(
    [
        "💬 Contextual Chat & Citations",
        "🎥 Multimodal Frames & Chapters",
        "🔬 Advanced RAG Lab (FAISS vs Chroma)",
        "📊 Evaluation & Benchmarking",
        "🎓 Learning Suite (Quiz, Cards, Notes)",
    ]
)

# ----------------- TAB 1: CONTEXTUAL QA -----------------
with tab_qa:
    st.subheader("Interactive Video Chatbot with Timestamp Citations & Grounding")
    st.write("Ask any conceptual or technical question. The system verifies grounding, retrieves supporting excerpts, and highlights timestamp references.")

    # Search bar & controls
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        user_question = st.text_input(
            "Your Question",
            value="How does self-attention compute query, key, and value vectors?",
            placeholder="Ask about equations, chapters, or concepts discussed...",
        )
    with col_btn:
        st.write("")
        st.write("")
        ask_btn = st.button("Search & Answer", type="primary", use_container_width=True)

    if ask_btn and user_question and hybrid_retriever:
        with st.spinner("🔍 Retrieving multimodal context & verifying grounding..."):
            t_start = time.perf_counter()

            # Hybrid Retrieval
            candidates = hybrid_retriever.retrieve(user_question, top_k=6)

            # Reranking
            if reranker:
                final_chunks = reranker.rerank(user_question, candidates, top_k=4)
            else:
                final_chunks = candidates[:4]

            # Contextual QA with Hallucination Guard
            llm_client = UnifiedLLMClient(provider=llm_provider, api_key=api_key_input)
            qa_engine = ContextualQA(llm_client=llm_client, guard=HallucinationGuard())

            frames = mm_doc.frames if mm_doc else []
            qa_res = qa_engine.answer_question(
                question=user_question,
                retrieved_chunks=final_chunks,
                video_id=metadata.video_id,
                video_title=metadata.title,
                frames=frames,
            )
            lat_ms = (time.perf_counter() - t_start) * 1000.0

            # Store in history
            st.session_state["qa_history"].insert(0, (qa_res, lat_ms))

    # Display latest response or history
    if st.session_state["qa_history"]:
        latest_qa, latency = st.session_state["qa_history"][0]

        # Status badge
        col_status1, col_status2, col_status3 = st.columns([2, 2, 2])
        with col_status1:
            if latest_qa.is_answerable:
                st.success(f"✅ Grounded Response ({latency:.0f}ms)")
            else:
                st.warning("⚠️ Out-of-Domain Question Detected")
        with col_status2:
            st.metric("Retrieval Confidence", f"{latest_qa.confidence_score * 100:.1f}%")
        with col_status3:
            st.metric("Timestamp Citations", len(latest_qa.citations))

        # Main Answer Box
        st.markdown("### Answer")
        st.markdown(latest_qa.answer)

        # Citations & Timestamps
        if latest_qa.citations:
            st.markdown("#### ⏱️ Verified Timestamp References")
            badge_cols = st.columns(min(4, len(latest_qa.citations)))
            for idx, cit in enumerate(latest_qa.citations[:4]):
                with badge_cols[idx]:
                    st.markdown(
                        f"""
                        <div class="card-box">
                            <b><a href="{cit.youtube_url}" target="_blank" style="color: #38bdf8; text-decoration: none;">▶ Jump to {cit.timestamp_str}</a></b><br>
                            <small style="color: #94a3b8;">Score: {cit.score:.2f}</small><br>
                            <span style="font-size: 0.85rem; color: #cbd5e1;">"{cit.quote}"</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Supporting Transcript & Multimodal Evidence
        with st.expander("🔍 Supporting Transcript Excerpts & Visual Evidence", expanded=False):
            for i, cit in enumerate(latest_qa.citations, 1):
                col_txt, col_img = st.columns([3, 2])
                with col_txt:
                    st.markdown(f"**Passage {i} [{cit.timestamp_str}]** — Similarity: `{cit.score:.2f}`")
                    st.info(cit.quote)
                    st.markdown(f"[🔗 Open at exact YouTube timecode]({cit.youtube_url})")
                with col_img:
                    if cit.frame_path and Path(cit.frame_path).exists():
                        st.image(cit.frame_path, caption=f"Keyframe at {cit.timestamp_str}", use_container_width=True)
                    else:
                        st.caption("No keyframe extracted at this timecode.")

    else:
        st.info("💡 Try clicking **'Search & Answer'** above to test grounding and timestamp citations!")


# ----------------- TAB 2: MULTIMODAL FRAMES & CHAPTERS -----------------
with tab_mm:
    st.subheader("Multimodal Understanding: Visual Keyframes & Chapter Summaries")
    st.write("Inspect how the video is parsed into visual slides, optical recognition cues, and chapter summaries.")

    if mm_doc and mm_doc.sections:
        st.markdown("### 📑 Section Summaries")
        for sec in mm_doc.sections:
            with st.expander(f"📍 {sec.title} ({sec.timestamp_str})", expanded=True):
                c_sum, c_frame = st.columns([3, 2])
                with c_sum:
                    st.markdown(f"**Summary**: {sec.summary}")
                    st.markdown("**Key Takeaways**:")
                    for t in sec.key_takeaways:
                        st.markdown(f"- {t}")
                    jump_link = format_youtube_link(metadata.video_id, sec.start_time)
                    st.markdown(f"[▶ Jump to Chapter in YouTube]({jump_link})")
                with c_frame:
                    if sec.representative_frame and Path(sec.representative_frame.image_path).exists():
                        st.image(
                            sec.representative_frame.image_path,
                            caption=f"Visual Concept at {sec.representative_frame.timestamp_str}",
                            use_container_width=True,
                        )

    if mm_doc and mm_doc.frames:
        st.markdown("### 🖼️ Keyframe Gallery (Extracted Timeline Frames)")
        cols = st.columns(3)
        for i, frame in enumerate(mm_doc.frames[:6]):
            with cols[i % 3]:
                if Path(frame.image_path).exists():
                    st.image(frame.image_path, caption=f"Frame at {frame.timestamp_str}", use_container_width=True)
                st.caption(f"OCR / Visual Cue: {frame.ocr_text[:60]}...")


# ----------------- TAB 3: ADVANCED RAG LAB (FAISS vs ChromaDB) -----------------
with tab_rag:
    st.subheader("🔬 Advanced RAG Engineering Lab")

    # Section 1: Chunking Strategy Profiler
    st.markdown("### 1. Chunking Strategy Comparison")
    st.caption("Compare how different splitting algorithms segment the video timeline.")
    if st.button("Run Chunking Strategy Profiler"):
        with st.spinner("Analyzing chunk distributions..."):
            chunk_df = compare_chunking_strategies(st.session_state["segments"], metadata)
            st.dataframe(chunk_df, use_container_width=True)

    # Section 2: FAISS vs ChromaDB Benchmark
    st.markdown("### 2. FAISS vs ChromaDB Performance Benchmark")
    st.caption("Head-to-head empirical comparison of indexing speed, search latency, and result consensus.")

    if st.button("⚡ Run Live FAISS vs ChromaDB Benchmark", type="primary"):
        with st.spinner("Benchmarking vector stores..."):
            comparator = VectorStoreComparator()
            texts = [c.text for c in chunks]
            embeddings = st.session_state["embedding_model"].embed_documents(texts)
            report = comparator.run_benchmark(chunks=chunks, embeddings=embeddings, k=4)

            col_sum, col_feat = st.columns(2)
            with col_sum:
                st.markdown("#### Performance Metrics")
                st.dataframe(report["summary_df"], use_container_width=True)
            with col_feat:
                st.markdown("#### Architecture Comparison")
                st.dataframe(report["feature_matrix"], use_container_width=True)

    # Section 3: Hybrid Search Alpha Tuner
    st.markdown("### 3. Hybrid Search & Fusion Simulator")
    alpha_slider = st.slider(
        "Dense vs BM25 Weight (Alpha)",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        help="1.0 = Pure Semantic Dense Vectors, 0.0 = Pure Keyword BM25.",
    )
    test_q = st.text_input("Test Hybrid Query", value="multi-head attention projection")
    if st.button("Simulate Hybrid Retrieval"):
        if hybrid_retriever:
            sim_results = hybrid_retriever.retrieve(test_q, top_k=4, alpha_override=alpha_slider)
            sim_data = [
                {
                    "Rank": r.rank,
                    "Timestamp": r.chunk.timestamp_str,
                    "Final Fused Score": round(r.final_score, 3),
                    "Dense Score": round(r.dense_score, 3),
                    "BM25 Score": round(r.bm25_score, 3),
                    "Excerpt": r.chunk.text[:90] + "...",
                }
                for r in sim_results
            ]
            st.dataframe(pd.DataFrame(sim_data), use_container_width=True)


# ----------------- TAB 4: EVALUATION & BENCHMARKING -----------------
with tab_eval:
    st.subheader("📊 Evaluation Framework & Faithfulness Verification")
    st.write("Measure information retrieval accuracy, claim-level faithfulness, and model performance matrices.")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("### 1. Retrieval Metrics (Precision & Recall)")
        if st.button("Run Retrieval Evaluation Suite"):
            with st.spinner("Computing Precision@K, Recall@K, MRR, and NDCG..."):
                evaluator = RetrievalEvaluator(k=3)
                eval_queries = evaluator.generate_synthetic_eval_dataset(chunks, num_queries=5)

                def retriever_fn(q_text, k):
                    res = hybrid_retriever.retrieve(q_text, top_k=k)
                    return [r.chunk for r in res]

                metrics = evaluator.evaluate(retriever_fn, eval_queries)
                m_df = pd.DataFrame([metrics.to_dict()])
                st.dataframe(m_df, use_container_width=True)

    with col_e2:
        st.markdown("### 2. Answer Faithfulness & Hallucination")
        if st.session_state["qa_history"]:
            latest_qa, _ = st.session_state["qa_history"][0]
            if st.button("Evaluate Faithfulness of Last Answer"):
                faith_eval = FaithfulnessEvaluator()
                context_chunks = [sc.chunk for sc in latest_qa.retrieved_chunks]
                faith_rep = faith_eval.evaluate(latest_qa.question, latest_qa.answer, context_chunks)

                st.metric("Faithfulness Score", f"{faith_rep.faithfulness_score * 100:.1f}%")
                st.metric("Hallucination Rate", f"{faith_rep.hallucination_rate * 100:.1f}%")

                st.markdown("#### Claim Grounding Verification")
                for c in faith_rep.claim_verifications:
                    status = "✅ Supported" if c.is_supported else "❌ Hallucination Risk"
                    st.write(f"- {status}: *\"{c.claim}\"* (Overlap: {c.confidence:.1%})")
        else:
            st.info("Submit a question in Tab 1 to run faithfulness evaluation.")

    st.markdown("### 3. Embedding Model Comparison Matrix")
    if st.button("Benchmark Embedding Models"):
        with st.spinner("Comparing SentenceTransformers vs TF-IDF projection..."):
            suite = BenchmarkSuite()
            eval_queries = RetrievalEvaluator.generate_synthetic_eval_dataset(chunks, num_queries=4)
            comp_df = suite.compare_embedding_models(chunks, eval_queries)
            st.dataframe(comp_df, use_container_width=True)


# ----------------- TAB 5: LEARNING SUITE -----------------
with tab_learning:
    st.subheader("🎓 Automated Learning Suite")
    st.write("Supercharge retention with automatically synthesized quizzes, active-recall flashcards, and study notes.")

    sub_quiz, sub_cards, sub_notes = st.tabs(["📝 Interactive Quiz", "🗂️ Flashcards (Anki)", "📚 Structured Study Notes"])

    # 1. Quizzes
    with sub_quiz:
        st.markdown("### 📝 Video Comprehension Quiz")
        questions = st.session_state.get("quiz_questions", [])
        if not questions:
            if st.button("Generate Quiz Questions"):
                quiz_gen = QuizGenerator(UnifiedLLMClient(provider=llm_provider, api_key=api_key_input))
                st.session_state["quiz_questions"] = quiz_gen.generate_quiz(chunks, metadata.video_id, num_questions=4)
                st.rerun()

        for q in questions:
            st.markdown(f"**Q{q.question_id}: {q.question}** `[{q.difficulty}]`")
            user_choice = st.radio(
                f"Select answer for Q{q.question_id}:",
                options=q.options,
                key=f"quiz_opt_{q.question_id}",
                index=None,
            )
            if user_choice is not None:
                chosen_idx = q.options.index(user_choice)
                if chosen_idx == q.correct_idx:
                    st.success(f"🎉 Correct! {q.explanation}")
                else:
                    st.error(f"❌ Incorrect. {q.explanation}")
                st.markdown(f"[▶ Review concept in video at {q.timestamp_str}]({q.youtube_url})")
            st.markdown("---")

    # 2. Flashcards
    with sub_cards:
        st.markdown("### 🗂️ Active-Recall Flashcards")
        cards = st.session_state.get("flashcards", [])

        # Download buttons
        if cards:
            c_d1, c_d2 = st.columns(2)
            card_gen = FlashcardGenerator()
            anki_tsv = card_gen.export_to_anki_tsv(cards, EXPORTS_DIR / "anki_deck.tsv")
            json_exp = card_gen.export_to_json(cards, EXPORTS_DIR / "flashcards.json")

            with c_d1:
                st.download_button(
                    "📥 Export to Anki Deck (.tsv)",
                    data=anki_tsv,
                    file_name="anki_cards.tsv",
                    mime="text/tab-separated-values",
                )
            with c_d2:
                st.download_button(
                    "📥 Export to JSON",
                    data=json_exp,
                    file_name="flashcards.json",
                    mime="application/json",
                )

        # Card viewer
        for card in cards:
            with st.expander(f"🎴 Card #{card.card_id}: {card.front}", expanded=False):
                st.markdown(card.back)
                st.markdown(f"[▶ Video Timestamp Link ({card.timestamp_str})]({card.youtube_url})")

    # 3. Study Notes
    with sub_notes:
        st.markdown("### 📚 Comprehensive Study Notes")
        notes = st.session_state.get("study_notes")
        if notes:
            st.download_button(
                "📥 Download Notes as Markdown (.md)",
                data=notes.to_markdown(),
                file_name="video_study_notes.md",
                mime="text/markdown",
            )
            st.markdown(notes.to_markdown())
        else:
            st.info("Study notes are generated upon video ingestion.")

