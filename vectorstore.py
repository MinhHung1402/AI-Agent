#vectorstore.py

from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def load():

    # Load Chroma vectorstore WITHOUT embedding again
    persist_dir = "content/vectordb"
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    return vectorstore

