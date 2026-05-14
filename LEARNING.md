# LEARNING.md — Multimodal RAG System

> Personal concept notes built while building this project.
> Written to reinforce understanding — not copied from docs.

---

## Core Concept: Why Multimodal RAG?

RAG Project 1 handled text-only retrieval. Real-world knowledge bases contain **both images and text** — product manuals with diagrams, research papers with figures, medical reports with scans. Multimodal RAG bridges this gap.

---

## Key Concept 1: How CLIP Enables Unified Retrieval

**CLIP (Contrastive Language-Image Pretraining)** was trained on millions of (image, caption) pairs using contrastive loss — it learned to pull matching pairs close and push non-matching pairs apart in a **shared embedding space**.

Result:
- `CLIP.encode_image(dog_photo)` → vector A
- `CLIP.encode_text("a dog running in a park")` → vector B
- cosine_similarity(A, B) ≈ **high**

Both modalities land in the **same 768-dimensional vector space**. This means a single text query can retrieve both images and text chunks — no separate pipelines needed.

**Interview answer:** "CLIP projects images and text into a shared semantic space via contrastive pretraining, enabling cross-modal similarity search."

---

## Key Concept 2: Single Index vs Separate Indexes

### Option A — Single FAISS Index (our approach)
- All vectors (image + text) in one index
- One query → one search → results can be image or text
- Works because CLIP puts both in same space
- ✅ Simpler, faster, unified ranking

### Option B — Separate Indexes
- Image index + text index → 2 queries → merge + re-rank
- Useful when: filtering by modality, different embedding models, very large scale
- ✅ More control, easier per-type optimization
- ❌ More complexity, needs a fusion/re-ranking layer

**When to use separate:** When image embeddings and text embeddings come from **different models** (e.g., CLIP for images, sentence-transformers for text) — they live in different spaces, so you cannot mix them in one index.

**Our design decision:** Single index with CLIP for both → simpler unified retrieval.

---

## Key Concept 3: Direct Embedding vs Captioning

### CLIP Direct Embedding
- Image → CLIP encoder → vector
- Fast, no extra model needed
- Captures visual semantics well (objects, scenes, style)
- ❌ Loses: text in images, exact numbers, fine-grained attributes

### Caption-then-Embed
- Image → captioning model (e.g., BLIP, LLaVA) → text caption → embed
- Slower, requires extra model
- ✅ Captures what CLIP misses: OCR text, specific counts, relationships

### Hybrid (Best Practice)
- Store both: CLIP vector + generated caption
- CLIP vector → retrieval
- Caption → injected into LLM context alongside the image
- **Why:** CLIP finds the right image; caption gives LLM the fine details it needs to answer accurately

**Interview question you will get:** *"What does CLIP embedding lose compared to a caption?"*
**Answer:** Fine-grained textual details embedded in the image — numbers, labels, OCR content, and precise spatial relationships.

---

## Architecture Overview

```
INGESTION PIPELINE
─────────────────
PDF/Images → Extract pages/images
                    │
          ┌─────────┴──────────┐
          │                    │
     Text chunks           Images
          │                    │
  Sentence-transformer    CLIP encoder
     embeddings           embeddings
          │                    │
          └─────────┬──────────┘
                    │
             FAISS Index (unified)
             + metadata store
             (type: image/text, path, caption)


RETRIEVAL PIPELINE
──────────────────
User query (text)
      │
CLIP text encoder
      │
  Query vector
      │
FAISS similarity search → Top-K results (mix of images + text)
      │
Re-rank / filter
      │
Build LLM context:
  - Text chunks → inject directly
  - Images → inject caption + image path
      │
Ollama / LLM → Final Answer
```

---

## Chunking Strategy for This Project

| Data Type | Strategy | Why |
|-----------|----------|-----|
| Text (PDF) | Recursive character splitter, 512 tokens, 50 overlap | Preserves sentence boundaries |
| Images | One embedding per image | Images are atomic — no chunking |
| Captions | Stored as metadata, not indexed separately | Avoid duplicate retrieval |

---

## Vector DB Design

**Tool:** FAISS (local, no server needed)

**Each entry stores:**
```python
{
  "id": "unique_id",
  "vector": [...],          # 512-dim CLIP or text embedding
  "type": "image" | "text",
  "content": "text chunk or caption string",
  "source_path": "path/to/file.pdf or image.jpg",
  "page": 3                 # for PDFs
}
```

**Index type:** `IndexFlatIP` (inner product = cosine on normalized vectors) — good for < 100K vectors. Switch to `IndexIVFFlat` for production scale.

---

## Interview Questions to Know Cold

| Question | Key Answer Points |
|----------|-------------------|
| RAG vs Fine-tuning? | RAG = external memory, no retraining, updatable. Fine-tune = baked-in knowledge, costly, stale. Use RAG when knowledge changes. |
| Why CLIP for multimodal RAG? | Shared embedding space → cross-modal similarity in one index |
| What fails in CLIP retrieval? | Fine-grained text, numbers, OCR content inside images |
| HNSW vs IVF index? | HNSW = graph-based, high recall, high memory. IVF = inverted file, faster at scale, slight recall drop |
| Faithfulness in RAG eval? | Answer grounded in retrieved context, not hallucinated |
| What is context recall? | How much of the ground truth answer is coverable by retrieved chunks |

---

## Mistakes Made & Fixed

*(Fill this in as you build — interviewers love this section)*

---

## Resources

- CLIP paper: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- FAISS docs: https://faiss.ai
- LangChain multimodal: https://python.langchain.com
- BLIP for captioning: https://huggingface.co/Salesforce/blip-image-captioning-base