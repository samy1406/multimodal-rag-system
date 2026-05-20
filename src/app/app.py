import streamlit as st
from src.ingestion import metadata_store as metadataLoad
from src.ingestion import embedder as emd
from src.retrieval import searcher as search
from PIL import Image

@st.cache_resource
def load_index():
    return search.load_index(path="data/index/faiss.index")

@st.cache_resource
def load_metadata():
    return metadataLoad.load_metadata("metadata.json")


st.title("Multimodal RAG System")

# 1. Input section
query_text = st.text_input("Enter your query")
query_image = st.file_uploader("Upload an image", type=["jpg","png"])
results = []




# replace direct calls with cached functions
index = load_index()
metadata = load_metadata()
# 2. On button click
if st.button("Search"):
    if query_text:
        embedding = emd.embed_text([query_text])
        results = search.search(index, embedding, 20)
        # embed_text → search → display
    elif query_image:
        img = Image.open(query_image)
        embedding = emd.embed_image([img])
        results = search.search(index, embedding)
        # embed_image → search → display

# 3. Display results loop
for result in results:
    idx = result["index"]
    if idx == -1:        # skip invalid
        continue
    meta = metadata[idx]        # get metadata for this vector
    if meta["type"] == "text":
        st.write(meta["content"])
    elif meta["type"] == "image":
        st.image(meta["image_path"])