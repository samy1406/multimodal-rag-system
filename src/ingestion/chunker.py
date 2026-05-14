from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text, chunk_size=512, chunk_overlap=50):

    # check if the text is not empty
    if not text or not text.strip():
        return []

    # creating the splitter to split the chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # split using spliter created
    return splitter.split_text(text)