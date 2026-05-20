#pdf_loader → chunker (text) → embedder → captioner (images) → metadata_store → index_builder
from src.ingestion import pdf_loader, chunker, embedder, captioner, metadata_store
from src.retrieval import index_builder
import numpy as np
from PIL import Image


def run_pipeline(pdf_path):
    print("run pipeline function called")
    pages = pdf_loader.load_pdf(pdf_path)
    
    all_vectors = []
    all_metadata = {}
    current_id = 0
    
    for page in pages:
        # TEXT
        if page["text"].strip():
            chunks = chunker.chunk_text(page["text"])
            vectors = embedder.embed_text(chunks)
            for i, chunk in enumerate(chunks):
                all_vectors.append(vectors[i])
                all_metadata[current_id] = {
                    "type": "text",
                    "source": pdf_path,
                    "page": page["page_number"],
                    "content": chunk,
                    "image_path": None
                }
                current_id += 1
        
        # IMAGES
        for img in page["images"]:
            caption = captioner.generate_caption(img)
            vector = embedder.embed_image([img])
            all_vectors.append(vector[0])
            all_metadata[current_id] = {
                "type": "image",
                "source": pdf_path,
                "page": page["page_number"],
                "content": caption,
                "image_path": pdf_path + "_page_" + str(page["page_number"])
            }
            current_id += 1
    
    # build + save index
    all_vectors = np.vstack(all_vectors)
    index = index_builder.build_index(all_vectors)
    index_builder.save_index(index, "data/index/faiss.index")
    metadata_store.save_metadata(all_metadata, "metadata.json")

if __name__ == "__main__":
    print("main function called")
    run_pipeline("data/{sample_docs}/sample-10-page-pdf-a4-size.pdf")