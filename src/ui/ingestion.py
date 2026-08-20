"""
BioGuard Ingestion Tab — document upload, processing queue, and index status.
"""
import os
import streamlit as st


def render_ingestion_tab(settings: dict, emb_model, vector_store, bm25) -> None:
    """Render the Document Knowledge Base ingestion page."""
    from src.ingestion.document_loader import UniversalDocumentLoader
    from src.ingestion.chunker import MedicalAwareChunker

    st.markdown("""
    <div style="max-width:860px; margin:0 auto;">
        <div style="font-size:22px; font-weight:800; color:var(--text-primary); margin-bottom:4px;">
            📚 Knowledge Base
        </div>
        <div style="font-size:13px; color:var(--text-secondary); margin-bottom:24px; line-height:1.6;">
            Upload clinical guidelines, pharmacopeia, or research PDFs.
            Documents are chunked, embedded, and indexed into FAISS and BM25 for grounded retrieval.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Index Status ──
    idx_dir = settings["paths"]["index_dir"]
    faiss_exists = os.path.exists(os.path.join(idx_dir, "faiss.index"))
    bm25_exists = os.path.exists(os.path.join(idx_dir, "bm25.pkl"))

    st.markdown("#### Index Status")
    si1, si2, si3, si4 = st.columns(4)
    si1.metric("FAISS Index", "✅ Loaded" if faiss_exists else "⬜ Empty")
    si2.metric("BM25 Index", "✅ Loaded" if bm25_exists else "⬜ Empty")
    chunks_count = len(getattr(vector_store, "chunks_metadata", [])) if vector_store else 0
    si3.metric("Indexed Chunks", str(chunks_count))
    si4.metric("Retrieval Mode", "Hybrid (Dense + BM25)")

    st.divider()

    # ── Upload Area ──
    st.markdown("#### Upload Clinical Documents")
    st.markdown("""
    <div style="font-size:12px; color:var(--text-muted); margin-bottom:10px;">
        Supported formats: PDF, DOCX, TXT, CSV, JSON — maximum 50 MB per file.
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drag & Drop or Browse",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv", "json"],
        label_visibility="collapsed"
    )

    # File queue display
    if uploaded_files:
        st.markdown("#### Document Queue")
        for uf in uploaded_files:
            size_kb = uf.size / 1024
            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.2f} MB"
            ok = uf.size <= 50 * 1024 * 1024
            status_icon = "✅" if ok else "❌ >50MB"
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:14px; padding:8px 12px;
                        background:var(--bg-subtle); border:1px solid var(--border);
                        border-radius:var(--radius-md); margin-bottom:6px; font-size:13px;">
                <span>📄</span>
                <span style="flex:1; color:var(--text-primary); font-weight:500;">{uf.name}</span>
                <span style="color:var(--text-muted);">{size_str}</span>
                <span>{status_icon}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Build Index ──
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    if st.button("🚀 Process & Build Knowledge Base", type="primary", key="btn_ingest") and uploaded_files:
        raw_dir = settings["paths"]["raw_data_dir"]
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(idx_dir, exist_ok=True)
        all_docs = []

        progress_bar = st.progress(0, text="Starting…")
        total = len(uploaded_files)

        with st.spinner("Extracting text from documents…"):
            for i, up_file in enumerate(uploaded_files):
                if up_file.size > 50 * 1024 * 1024:
                    st.error(f"Skipped {up_file.name} — exceeds 50 MB limit.")
                    continue
                save_path = os.path.join(raw_dir, up_file.name)
                with open(save_path, "wb") as f:
                    f.write(up_file.getbuffer())
                try:
                    docs = UniversalDocumentLoader.load_file(save_path)
                    all_docs.extend(docs)
                    progress_bar.progress(
                        int((i + 1) / total * 40),
                        text=f"Parsed {up_file.name}"
                    )
                except Exception as e:
                    st.error(f"Failed to parse {up_file.name}: {e}")

        if not all_docs:
            st.warning("No text content could be extracted from the uploaded files.")
            progress_bar.empty()
            return

        chunker = MedicalAwareChunker()
        chunks = chunker.chunk_documents(all_docs)
        progress_bar.progress(55, text=f"Generated {len(chunks)} structure-aware chunks…")

        with st.spinner("Generating embeddings (this may take a few minutes)…"):
            texts = [c["text"] for c in chunks]
            embeddings = emb_model.embed_documents(texts)
            progress_bar.progress(80, text="Building FAISS index…")
            vector_store.add_chunks(chunks, embeddings)
            vector_store.save(idx_dir)
            progress_bar.progress(90, text="Building BM25 index…")
            bm25.index_chunks(vector_store.chunks_metadata)
            bm25.save(idx_dir)
            progress_bar.progress(100, text="Complete!")

        st.success(
            f"✅ Indexing complete — {len(chunks)} chunks from {len(uploaded_files)} files "
            f"are now available in the retrieval pipeline."
        )
        st.cache_resource.clear()
        progress_bar.empty()
