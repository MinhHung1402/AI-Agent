import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_documents(folder_path: str):
    documents = []
    for root, dirs, files in os.walk(folder_path):   # RECURSIVE!
        for filename in files:
            if filename.endswith(".txt"):
                filepath = os.path.join(root, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    documents.append(Document(page_content=f.read(), metadata={"source": filepath}))
    return documents

if __name__ == "__main__":
    load_dotenv()
    doc_dir = "docs"
    persist_dir = "content/vectordb"

    if not os.path.exists(doc_dir):
        raise FileNotFoundError("Docs directory not found.")

    docs = load_documents(doc_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    vectorstore.persist()
    print("✅ Vector DB built and saved.")
