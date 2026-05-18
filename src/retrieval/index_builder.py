# build and save FAISS INDEX
import faiss
import numpy as np

def build_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def save_index(index: faiss.Index, path: str) -> None:
    faiss.write_index(index, path)
    return None
