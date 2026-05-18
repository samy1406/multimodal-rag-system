# LOAD INDEX and SEARCH
import faiss
import numpy as np

def load_index(path: str) -> faiss.Index:
    index = faiss.read_index(path)
    return index

def search(index, query_vector: np.ndarray, top_k: int = 5) -> list:
    distance, indices = index.search(query_vector, top_k)
    result = []
    for i in range(len(indices[0])):
        result.append({
            "index": indices[0][i],
            "score": distance[0][i]
        })
    return result