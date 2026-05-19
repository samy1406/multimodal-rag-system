import streamlit as st
import ingestion.embedder as emd
import retrieval.searcher as search
import ingestion.metadata_store as metadataLoad
from PIL import Image


metadata = metadataLoad.load_metadata("")

st.title("Multimodal RAG System")

# 1. Input section
query_text = st.text_input("Enter your query")
query_image = st.file_uploader("Upload an image", type=["jpg","png"])
index = search.load_index(path="path")

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
    # check type → show accordingly
    if result["type"] == "text":
        st.write(metadata[result["index"]]["content"])
    elif result["type"] == "image":
        st.image(metadata[result["index"]]["image_path"])