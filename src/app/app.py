import streamlit as st
# import ingestion.embedder as emd
# import retrieval.searcher as search
from src.ingestion import metadata_store as metadataLoad
from PIL import Image
from src.ingestion import embedder as emd
from src.retrieval import searcher as search

metadata = metadataLoad.load_metadata("metadata.json")

st.title("Multimodal RAG System")

# 1. Input section
query_text = st.text_input("Enter your query")
query_image = st.file_uploader("Upload an image", type=["jpg","png"])
index = search.load_index(path="data/index/faiss.index")
results = []
# 2. On button click
if st.button("Search"):
    if query_text:
        embedding = emd.embed_text([query_text])
        results = search.search(index, embedding)
        # embed_text → search → display
    elif query_image:
        img = Image.open(query_image)
        embedding = emd.embed_image([img])
        results = search.search(index, embedding)
        # embed_image → search → display

# 3. Display results loop
for result in results:
    idx = result["index"]
    meta = metadata[idx]        # get metadata for this vector
    if meta["type"] == "text":
        st.write(meta["content"])
    elif meta["type"] == "image":
        st.image(meta["image_path"])