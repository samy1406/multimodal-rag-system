# 🔍 Multimodal RAG System

A Retrieval-Augmented Generation system that understands both **text and images** — built with CLIP, FAISS, BLIP, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![CLIP](https://img.shields.io/badge/Embeddings-CLIP%20ViT--B%2F32-orange)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## 📌 What It Does

Most RAG systems only handle text. Real-world documents — research papers, medical reports, technical manuals — contain both text and images. This system handles both.

Given a PDF with mixed text and images:
- Extracts and chunks all text
- Extracts all embedded images, generates captions via BLIP
- Embeds everything into a **unified CLIP vector space**
- Stores in a single FAISS index for cross-modal retrieval
- Lets users query via **text or image** and retrieves relevant text chunks and images

---

## 🏗️ Architecture

```
INGESTION PIPELINE
──────────────────
PDF Input
  │
  ├── Text Pages ──► Chunker ──► CLIP Text Encoder ──► vectors
  │
  └── Images ──► BLIP Captioner + CLIP Image Encoder ──► vectors
                          │
                    Save image to disk
                          │
                   FAISS Index (unified, 512-dim)
                   + metadata.json


RETRIEVAL PIPELINE
──────────────────
User Query (text or image)
  │
  CLIP Encoder (text or image)
  │
  FAISS similarity search (top-K)
  │
  Lookup metadata → type: text or image
  │
  Display results in Streamlit UI
```

---

## 📁 Project Structure

```
multimodal-rag-system/
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py       # Extract text + images from PDF using fitz
│   │   ├── chunker.py          # RecursiveCharacterTextSplitter
│   │   ├── embedder.py         # CLIP embeddings for text + images
│   │   ├── captioner.py        # BLIP image captioning
│   │   ├── metadata_store.py   # Save/load metadata as JSON
│   │   └── pipeline.py         # End-to-end ingestion glue
│   ├── retrieval/
│   │   ├── index_builder.py    # Build + save FAISS IndexFlatIP
│   │   └── searcher.py         # Load index + search
│   └── app/
│       └── app.py              # Streamlit UI
├── notebooks/                  # Colab experiments (GPU work)
├── data/
│   ├── sample_docs/            # Input PDFs
│   ├── extracted_images/       # Images extracted from PDFs
│   └── index/                  # Saved FAISS index
├── config.py                   # Central config (TOP_K, paths, chunk size)
├── metadata.json               # Vector ID → metadata mapping
├── requirements.txt
├── LEARNING.md                 # Concept notes built while coding
└── README.md
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- GitHub Codespace (CPU work) or Google Colab (GPU/captioning)

### Installation

```bash
# Clone repo
git clone https://github.com/samy1406/multimodal-rag-system.git
cd multimodal-rag-system

# Install dependencies
pip install -r requirements.txt

# Install OpenAI CLIP (not on PyPI)
pip install git+https://github.com/openai/CLIP.git
```

---

## 🚀 Usage

### Step 1 — Ingest a PDF

```bash
python -m src.ingestion.pipeline
```

Edit the path in `pipeline.py` `__main__` block to point to your PDF. This will:
- Extract text chunks + images from the PDF
- Embed everything with CLIP
- Generate captions for images with BLIP
- Save `data/index/faiss.index` and `metadata.json`

### Step 2 — Run the App

```bash
python -m streamlit run src/app/app.py
```

Open `http://localhost:8501` in your browser.

- **Text query** → type a question → retrieves relevant text chunks + images
- **Image query** → upload an image → finds visually similar content

---

## 🧠 Key Design Decisions

### Why CLIP for both text and images?
CLIP was trained on (image, caption) pairs and projects both modalities into the same 512-dimensional vector space. This enables a single text query to retrieve both text chunks and images in one FAISS search — no separate pipelines or result merging needed.

### Why a single FAISS index?
Since CLIP produces same-dimensional vectors for both modalities, a single `IndexFlatIP` index handles everything. This simplifies retrieval to one query with one result set.

### Why BLIP captions alongside CLIP embeddings?
CLIP embeddings capture visual semantics well but lose fine-grained details — text inside images, exact numbers, spatial labels. BLIP captions capture these details and are stored in metadata for LLM context injection.

### Chunking strategy
`RecursiveCharacterTextSplitter` with `chunk_size=512`, `chunk_overlap=50`. Recursive splitting preserves sentence and paragraph boundaries before falling back to hard character splits. Overlap prevents information loss at chunk boundaries.

---

## ⚠️ Known Limitations

- **Score imbalance:** Text-to-text cosine similarity scores are naturally higher than text-to-image scores in CLIP space. Current fix: higher `TOP_K=20`. Production fix would be per-modality score normalization or a cross-encoder re-ranker.
- **CLIP token limit:** CLIP tokenizer has a 77-token context limit. Long text chunks are truncated via `truncate=True`. For production, use a dedicated text embedding model (e.g. `sentence-transformers`) with a separate index.
- **No LLM answer generation:** Current version retrieves and displays results. LLM answer synthesis (e.g. via Ollama) is the next milestone.
- **Local only:** FAISS index stored on disk. Production would use a vector database (Pinecone, Weaviate, Qdrant).

---

## 🗺️ Roadmap

- [x] PDF ingestion (text + image extraction)
- [x] CLIP unified embeddings
- [x] BLIP image captioning
- [x] FAISS single index retrieval
- [x] Streamlit UI (text + image query)
- [ ] LLM answer generation via Ollama
- [ ] Multi-PDF support with source filtering
- [ ] Score normalization / re-ranking
- [ ] Evaluation with RAGAS metrics
- [ ] Docker deployment

---

## 📚 What I Learned

See [LEARNING.md](./LEARNING.md) for detailed concept notes on CLIP, FAISS index types, chunking strategy, and interview Q&A built while coding this project.

---

## 🔗 Related Projects

- [RAG Project 1 — Text RAG](https://github.com/samy1406) — text-only RAG pipeline (predecessor to this project)

---

## 📄 License

MIT